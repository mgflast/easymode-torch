"""Convert TensorFlow/Keras HDF5 weights to PyTorch format.

Uses h5py to read the .h5 file directly — no TensorFlow required.

Based on work by alisterburt: https://github.com/mgflast/easymode/pull/4
"""

import re
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch


def _load_tf_weights_from_h5(h5_path):
    weights = {}
    with h5py.File(h5_path, 'r') as f:
        def extract(name, obj):
            if isinstance(obj, h5py.Dataset):
                weights[name] = np.array(obj)
        f.visititems(extract)
    return weights


def _conv3d_weight(tf_weight):
    # TF: (D, H, W, Cin, Cout) -> PyTorch: (Cout, Cin, D, H, W)
    return torch.from_numpy(np.transpose(tf_weight, (4, 3, 0, 1, 2))).float()


def _assign_bn(bn_params, prefix, state):
    state[f'{prefix}.weight'] = torch.from_numpy(bn_params['gamma']).float()
    state[f'{prefix}.bias'] = torch.from_numpy(bn_params['beta']).float()
    state[f'{prefix}.running_mean'] = torch.from_numpy(bn_params['moving_mean']).float()
    state[f'{prefix}.running_var'] = torch.from_numpy(bn_params['moving_variance']).float()


def _layer_number(name):
    match = re.search(r'_(\d+)$', name.split('/')[-2])
    return int(match.group(1)) if match else 0


def _bn_number(path):
    match = re.search(r'_(\d+)$', path.split('/')[-1])
    return int(match.group(1)) if match else 0


def _convert_resblock(block_weights, prefix, state):
    conv_kernels = []
    bn_paths = {}

    for name, weight in block_weights.items():
        if 'kernel' in name:
            conv_kernels.append((name, weight))
        elif any(k in name for k in ('gamma', 'beta', 'moving_mean', 'moving_variance')):
            bn_path = '/'.join(name.split('/')[:-1])
            bn_paths.setdefault(bn_path, {})[name.split('/')[-1].replace(':0', '')] = weight

    conv_kernels.sort(key=lambda x: _layer_number(x[0]))

    conv1 = conv2 = skip = None
    for name, kernel in conv_kernels:
        if kernel.shape[0] == 1:
            skip = kernel
        elif conv1 is None:
            conv1 = kernel
        else:
            conv2 = kernel

    if conv1 is not None:
        state[f'{prefix}.conv1.weight'] = _conv3d_weight(conv1)
    if conv2 is not None:
        state[f'{prefix}.conv2.weight'] = _conv3d_weight(conv2)
    if skip is not None:
        state[f'{prefix}.skip_conv.weight'] = _conv3d_weight(skip)

    bn_list = sorted(
        [(p, params) for p, params in bn_paths.items() if len(params) == 4],
        key=lambda x: _bn_number(x[0])
    )
    bn_names = ['bn1', 'bn2', 'skip_bn']
    for i, (_, params) in enumerate(bn_list):
        _assign_bn(params, f'{prefix}.{bn_names[i]}', state)


def _convert_encoder(block_weights, idx, state):
    prefix = f'encoders.{idx}'
    res_weights = {k: v for k, v in block_weights.items() if 'res_block' in k}
    other_weights = {k: v for k, v in block_weights.items() if 'res_block' not in k}

    if idx > 0:
        for name, weight in other_weights.items():
            if 'kernel' in name:
                state[f'{prefix}.downsample.conv.weight'] = _conv3d_weight(weight)
                break

        bn_paths = {}
        for name, weight in other_weights.items():
            if any(k in name for k in ('gamma', 'beta', 'moving_mean', 'moving_variance')):
                bn_path = '/'.join(name.split('/')[:-1])
                bn_paths.setdefault(bn_path, {})[name.split('/')[-1].replace(':0', '')] = weight

        for _, params in bn_paths.items():
            if len(params) == 4:
                _assign_bn(params, f'{prefix}.downsample_bn', state)
                break

    _convert_resblock(res_weights, f'{prefix}.res_block', state)


def _convert_decoder(block_weights, idx, state):
    prefix = f'decoders.{idx}'
    res_weights = {k: v for k, v in block_weights.items() if 'res_block' in k}
    other_weights = {k: v for k, v in block_weights.items() if 'res_block' not in k}

    for name, weight in other_weights.items():
        if 'conv3d_transpose' in name and 'kernel' in name:
            state[f'{prefix}.upsample.weight'] = _conv3d_weight(weight)
            break

    bn_paths = {}
    for name, weight in other_weights.items():
        if any(k in name for k in ('gamma', 'beta', 'moving_mean', 'moving_variance')):
            bn_path = '/'.join(name.split('/')[:-1])
            bn_paths.setdefault(bn_path, {})[name.split('/')[-1].replace(':0', '')] = weight

    for _, params in bn_paths.items():
        if len(params) == 4:
            _assign_bn(params, f'{prefix}.upsample_bn', state)
            break

    _convert_resblock(res_weights, f'{prefix}.res_block', state)


def convert_h5_to_pth(h5_path, pth_path):
    """Convert easymode TensorFlow .h5 weights to PyTorch .pth.

    No TensorFlow required — reads the HDF5 file directly with h5py.

    Parameters
    ----------
    h5_path : str or Path
    pth_path : str or Path

    Returns
    -------
    OrderedDict
        PyTorch state_dict
    """
    h5_path = Path(h5_path)
    pth_path = Path(pth_path)

    tf_weights = _load_tf_weights_from_h5(h5_path)

    blocks = {}
    for name, weight in tf_weights.items():
        if 'optimizer' in name.lower() or 'iteration' in name.lower():
            continue
        block_name = name.split('/')[0]
        blocks.setdefault(block_name, {})[name] = weight

    state = OrderedDict()
    for block_name, block_weights in sorted(blocks.items()):
        if block_name.startswith('encoder_'):
            _convert_encoder(block_weights, int(block_name.split('_')[1]), state)
        elif block_name.startswith('decoder_'):
            _convert_decoder(block_weights, int(block_name.split('_')[1]), state)
        elif block_name == 'output':
            for name, weight in block_weights.items():
                if 'kernel' in name:
                    state['final_conv.weight'] = _conv3d_weight(weight)
                elif 'bias' in name:
                    state['final_conv.bias'] = torch.from_numpy(weight).float()

    pth_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, pth_path)
    return state
