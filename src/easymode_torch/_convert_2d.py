"""Convert Ais 2D .scnm model weights to PyTorch format.

.scnm files are TAR archives containing:
  - *_weights.h5  (Keras saved model with architecture + weights)
  - *_metadata.json (model metadata: apix, box_size, overlap, etc.)

This module extracts the h5, reads weights with h5py, and maps them
to the PyTorch UNet2D state dict. No TensorFlow required.
"""

import glob
import json
import os
import tarfile
import tempfile
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch


def _load_keras_model_weights(h5_path):
    """Extract weight arrays from a Keras saved-model h5 file.

    Keras save_model stores weights under 'model_weights/<layer_name>/<layer_name>/<var>'
    Returns a flat dict of layer_name -> var_name -> numpy array.
    """
    layers = {}
    with h5py.File(h5_path, 'r') as f:
        # Keras saved models store weights under 'model_weights'
        root = f['model_weights'] if 'model_weights' in f else f

        for layer_name in root:
            group = root[layer_name]
            if not isinstance(group, h5py.Group):
                continue
            # Keras nests: layer_name/layer_name/var:0
            inner = group[layer_name] if layer_name in group else group
            if not isinstance(inner, h5py.Group):
                continue
            layer_vars = {}
            for var_name in inner:
                ds = inner[var_name]
                if isinstance(ds, h5py.Dataset):
                    # Strip ':0' suffix
                    clean_name = var_name.replace(':0', '')
                    layer_vars[clean_name] = np.array(ds)
            if layer_vars:
                layers[layer_name] = layer_vars
    return layers


def _conv2d_weight(tf_weight):
    """TF Conv2D kernel: (H, W, Cin, Cout) -> PyTorch: (Cout, Cin, H, W)"""
    return torch.from_numpy(np.transpose(tf_weight, (3, 2, 0, 1))).float()


def _conv2d_transpose_weight(tf_weight):
    """TF Conv2DTranspose kernel: (H, W, Cout, Cin) -> PyTorch: (Cin, Cout, H, W)"""
    return torch.from_numpy(np.transpose(tf_weight, (3, 2, 0, 1))).float()


def _bn_params(layer_vars):
    """Extract BatchNorm parameters from a layer's variables."""
    return {
        'weight': torch.from_numpy(layer_vars['gamma']).float(),
        'bias': torch.from_numpy(layer_vars['beta']).float(),
        'running_mean': torch.from_numpy(layer_vars['moving_mean']).float(),
        'running_var': torch.from_numpy(layer_vars['moving_variance']).float(),
    }


def _map_weights_to_state_dict(layers):
    """Map Keras layer weights to PyTorch UNet2D state dict.

    The Keras model has layers named like:
      conv2d, conv2d_1, conv2d_2, ...  (Conv2D layers in order)
      batch_normalization, batch_normalization_1, ...  (BN layers in order)
      conv2d_transpose, conv2d_transpose_1, ...  (upsampling layers)

    The PyTorch model structure:
      enc1.conv1, enc1.bn1, enc1.conv2, enc1.bn2
      enc2.conv1, enc2.bn1, enc2.conv2, enc2.bn2
      enc3.conv1, enc3.bn1, enc3.conv2, enc3.bn2
      enc4.conv1, enc4.bn1, enc4.conv2, enc4.bn2
      bottleneck_conv1, bottleneck_bn1, bottleneck_conv2, bottleneck_bn2
      dec1.up, dec1.conv1, dec1.bn1, dec1.conv2, dec1.bn2
      dec2.up, dec2.conv1, dec2.bn1, dec2.conv2, dec2.bn2
      dec3.up, dec3.conv1, dec3.bn1, dec3.conv2, dec3.bn2
      dec4.up, dec4.conv1, dec4.bn1, dec4.conv2, dec4.bn2
      out_conv
    """
    state = OrderedDict()

    # Sort conv2d layers by index
    conv_layers = sorted(
        [(k, v) for k, v in layers.items() if k.startswith('conv2d') and 'transpose' not in k],
        key=lambda x: int(x[0].split('_')[-1]) if '_' in x[0] and x[0].split('_')[-1].isdigit() else 0,
    )

    # Sort batch_normalization layers by index
    bn_layers = sorted(
        [(k, v) for k, v in layers.items() if k.startswith('batch_normalization')],
        key=lambda x: int(x[0].split('_')[-1]) if x[0].split('_')[-1].isdigit() else 0,
    )

    # Sort conv2d_transpose layers by index
    up_layers = sorted(
        [(k, v) for k, v in layers.items() if 'conv2d_transpose' in k],
        key=lambda x: int(x[0].split('_')[-1]) if x[0].split('_')[-1].isdigit() else 0,
    )

    # Encoder: 4 blocks x 2 convs = 8 conv layers, 8 BN layers
    enc_map = [
        ('enc1.conv1', 'enc1.bn1', 'enc1.conv2', 'enc1.bn2'),
        ('enc2.conv1', 'enc2.bn1', 'enc2.conv2', 'enc2.bn2'),
        ('enc3.conv1', 'enc3.bn1', 'enc3.conv2', 'enc3.bn2'),
        ('enc4.conv1', 'enc4.bn1', 'enc4.conv2', 'enc4.bn2'),
    ]

    conv_idx = 0
    bn_idx = 0

    for block in enc_map:
        c1, b1, c2, b2 = block
        # Conv 1
        state[f'{c1}.weight'] = _conv2d_weight(conv_layers[conv_idx][1]['kernel'])
        state[f'{c1}.bias'] = torch.from_numpy(conv_layers[conv_idx][1]['bias']).float()
        conv_idx += 1
        # BN 1
        for k, v in _bn_params(bn_layers[bn_idx][1]).items():
            state[f'{b1}.{k}'] = v
        bn_idx += 1
        # Conv 2
        state[f'{c2}.weight'] = _conv2d_weight(conv_layers[conv_idx][1]['kernel'])
        state[f'{c2}.bias'] = torch.from_numpy(conv_layers[conv_idx][1]['bias']).float()
        conv_idx += 1
        # BN 2
        for k, v in _bn_params(bn_layers[bn_idx][1]).items():
            state[f'{b2}.{k}'] = v
        bn_idx += 1

    # Bottleneck: 2 convs, 2 BNs
    state['bottleneck_conv1.weight'] = _conv2d_weight(conv_layers[conv_idx][1]['kernel'])
    state['bottleneck_conv1.bias'] = torch.from_numpy(conv_layers[conv_idx][1]['bias']).float()
    conv_idx += 1
    for k, v in _bn_params(bn_layers[bn_idx][1]).items():
        state[f'bottleneck_bn1.{k}'] = v
    bn_idx += 1

    state['bottleneck_conv2.weight'] = _conv2d_weight(conv_layers[conv_idx][1]['kernel'])
    state['bottleneck_conv2.bias'] = torch.from_numpy(conv_layers[conv_idx][1]['bias']).float()
    conv_idx += 1
    for k, v in _bn_params(bn_layers[bn_idx][1]).items():
        state[f'bottleneck_bn2.{k}'] = v
    bn_idx += 1

    # Decoder: 4 blocks, each with 1 transpose conv + 2 convs + 2 BNs
    dec_map = [
        ('dec1.up', 'dec1.conv1', 'dec1.bn1', 'dec1.conv2', 'dec1.bn2'),
        ('dec2.up', 'dec2.conv1', 'dec2.bn1', 'dec2.conv2', 'dec2.bn2'),
        ('dec3.up', 'dec3.conv1', 'dec3.bn1', 'dec3.conv2', 'dec3.bn2'),
        ('dec4.up', 'dec4.conv1', 'dec4.bn1', 'dec4.conv2', 'dec4.bn2'),
    ]

    up_idx = 0
    for block in dec_map:
        up, c1, b1, c2, b2 = block
        # Transpose conv
        state[f'{up}.weight'] = _conv2d_transpose_weight(up_layers[up_idx][1]['kernel'])
        state[f'{up}.bias'] = torch.from_numpy(up_layers[up_idx][1]['bias']).float()
        up_idx += 1
        # Conv 1
        state[f'{c1}.weight'] = _conv2d_weight(conv_layers[conv_idx][1]['kernel'])
        state[f'{c1}.bias'] = torch.from_numpy(conv_layers[conv_idx][1]['bias']).float()
        conv_idx += 1
        # BN 1
        for k, v in _bn_params(bn_layers[bn_idx][1]).items():
            state[f'{b1}.{k}'] = v
        bn_idx += 1
        # Conv 2
        state[f'{c2}.weight'] = _conv2d_weight(conv_layers[conv_idx][1]['kernel'])
        state[f'{c2}.bias'] = torch.from_numpy(conv_layers[conv_idx][1]['bias']).float()
        conv_idx += 1
        # BN 2
        for k, v in _bn_params(bn_layers[bn_idx][1]).items():
            state[f'{b2}.{k}'] = v
        bn_idx += 1

    # Output conv (last conv2d layer)
    state['out_conv.weight'] = _conv2d_weight(conv_layers[conv_idx][1]['kernel'])
    state['out_conv.bias'] = torch.from_numpy(conv_layers[conv_idx][1]['bias']).float()

    return state


def extract_scnm(scnm_path):
    """Extract .scnm (tar) and return (h5_path, metadata_dict) in a temp dir.

    Caller is responsible for cleaning up the returned temp directory.
    """
    temp_dir = tempfile.mkdtemp()
    with tarfile.open(str(scnm_path), 'r') as archive:
        archive.extractall(path=temp_dir)

    h5_files = glob.glob(os.path.join(temp_dir, "*_weights.h5"))
    meta_files = glob.glob(os.path.join(temp_dir, "*_metadata.json"))

    if not h5_files:
        raise RuntimeError(f"No *_weights.h5 found in {scnm_path}")

    metadata = {}
    if meta_files:
        with open(meta_files[0]) as f:
            metadata = json.load(f)

    return h5_files[0], metadata, temp_dir


def convert_scnm_to_pth(scnm_path, pth_path):
    """Convert Ais .scnm model to PyTorch .pth weights.

    Parameters
    ----------
    scnm_path : str or Path
    pth_path : str or Path

    Returns
    -------
    tuple[OrderedDict, dict]
        (state_dict, metadata)
    """
    import shutil

    scnm_path = Path(scnm_path)
    pth_path = Path(pth_path)

    h5_path, metadata, temp_dir = extract_scnm(scnm_path)
    try:
        layers = _load_keras_model_weights(h5_path)
        state = _map_weights_to_state_dict(layers)

        pth_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, pth_path)
        return state, metadata
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
