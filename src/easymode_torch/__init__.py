"""easymode-torch: PyTorch inference for easymode pretrained segmentation networks.

No TensorFlow required.

Basic usage
-----------
>>> import easymode_torch
>>> easymode_torch.segment('ribosome', 'path/to/tomograms/', 'segmented/')
"""

from pathlib import Path
import glob
import os
import time

import mrcfile
import numpy as np
import torch

from ._distribution import get_model_weights, get_model_weights_2d, list_models
from ._model import UNet
from ._model_2d import UNet2D
from ._utils import get_device


def _collect_tomograms(data_directory):
    if isinstance(data_directory, (list, tuple)):
        patterns = list(data_directory)
    else:
        patterns = [str(data_directory)]

    tomograms = []
    for p in patterns:
        if os.path.isdir(p):
            tomograms.extend(glob.glob(os.path.join(p, '*.mrc')))
        else:
            tomograms.extend(glob.glob(p))
    tomograms = [f for f in sorted(set(tomograms)) if os.path.splitext(f)[-1] == '.mrc']
    return tomograms, patterns


def _save_mrc(data, path, voxel_size, data_format='int8'):
    if data_format == 'float32':
        data = data.astype(np.float32)
    elif data_format == 'uint16':
        data = (data * 255).astype(np.uint16)
    elif data_format == 'int8':
        data = (data * 127).astype(np.int8)
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(data)
        m.voxel_size = voxel_size


def segment(
    feature,
    data_directory,
    output_directory='segmented',
    *,
    tta=4,
    batch_size=1,
    input_apix=None,
    gpu=None,
    overwrite=False,
    silent=True,
    data_format='int8',
    use_depth=1.0,
    xy_margin=0,
):
    """Segment tomograms using a pretrained easymode 3D model.

    Parameters
    ----------
    feature : str
        Name of the feature to segment, e.g. 'ribosome', 'microtubule'.
        Use ``list_models()`` to see what's available.
    data_directory : str or Path or list
        Directory containing .mrc tomograms, or a glob pattern, or a list of paths.
    output_directory : str or Path
        Where to write output segmentation .mrc files.
    tta : int
        Test-time augmentation passes (1 = none, up to 16).
    batch_size : int
        Number of tiles to process in parallel.
    input_apix : float or None
        Override the pixel size from the MRC header (Å/px).
    gpu : int or None
        GPU device ID. None = auto-select.
    overwrite : bool
        Re-segment tomograms that already have output files.
    silent : bool
        Suppress all output. Default True (for use as library).
        The CLI sets this to False.
    data_format : str
        Output format: 'int8' (default, 0-127), 'uint16' (0-255), or 'float32' (0.0-1.0).
    use_depth : float
        Fraction of the Z range to segment (0.0-1.0).
    xy_margin : int
        Pixels to crop from XY edges before segmenting.
    """
    from ._inference import segment_tomogram

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    tomograms, patterns = _collect_tomograms(data_directory)

    if not tomograms:
        if not silent:
            print(f"No .mrc files found in {patterns}")
        return

    device = get_device(gpu)

    # Download/convert weights
    pth_path, metadata = get_model_weights(feature, silent=silent)
    if pth_path is None:
        if not silent:
            print(f"Could not find model for {feature}! Exiting.")
        return
    model_apix = metadata["apix"] if metadata else 10.0
    model_apix_z = metadata.get("apix_z") if metadata else None

    if not silent:
        print(
            f"easymode-torch segment\n"
            f"feature: {feature}\n"
            f"data_patterns: {patterns}\n"
            f"output_directory: {output_directory}\n"
            f"output_format: {data_format}\n"
            f"device: {device}\n"
            f"tta: {tta}\n"
            f"overwrite: {overwrite}\n"
            f"batch_size: {batch_size}\n"
        )

        if model_apix_z is not None:
            print(f"Using model: {pth_path}, inference at {model_apix} Å/px (XY) / {model_apix_z} Å/px (Z). \n")
        else:
            print(f"Using model: {pth_path}, inference at {model_apix} Å/px. \n")

        print(f"Found {len(tomograms)} tomograms to segment. \n")

    model = UNet()
    state = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)

    start_time = time.time()

    for i, tomo_path in enumerate(tomograms, 1):
        tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]
        out_path = output_directory / f"{tomo_name}__{feature}.mrc"

        if out_path.exists() and not overwrite:
            continue

        seg, apix = segment_tomogram(
            model, tomo_path, device,
            tta=tta, batch_size=batch_size,
            model_apix=model_apix, input_apix=input_apix,
            model_apix_z=model_apix_z,
            use_depth=use_depth, xy_margin=xy_margin,
        )
        _save_mrc(seg, out_path, apix, data_format)

        if not silent:
            elapsed = time.time() - start_time
            eta = time.strftime('%H:%M:%S', time.gmtime(elapsed / i * (len(tomograms) - i)))
            avg_secs = int(elapsed / i)
            per_tomo = f"{avg_secs // 60:02d}:{avg_secs % 60:02d}"
            print(f"{i}/{len(tomograms)} (on {device}) - {feature} - {os.path.basename(tomo_path)} - eta {eta} ({per_tomo} per tomo)")

    if not silent:
        print()
        print(f"\033[92mSegmentation finished!\033[0m")
        print()


def segment_2d(
    feature,
    data_directory,
    output_directory='segmented',
    *,
    tta=4,
    gpu=None,
    overwrite=False,
    silent=True,
    data_format='int8',
    use_depth=1.0,
    stride=1,
):
    """Segment tomograms slice-by-slice using a pretrained 2D model.

    Parameters
    ----------
    feature : str
        Name of the feature to segment, e.g. 'ribosome'.
    data_directory : str or Path or list
        Directory containing .mrc tomograms, or a glob pattern, or a list of paths.
    output_directory : str or Path
        Where to write output segmentation .mrc files.
    tta : int
        Test-time augmentation passes (1-8).
    gpu : int or None
        GPU device ID. None = auto-select.
    overwrite : bool
        Re-segment tomograms that already have output files.
    silent : bool
        Suppress all output. Default True (for use as library).
        The CLI sets this to False.
    data_format : str
        Output format: 'int8' (default, 0-127), 'uint16' (0-255), or 'float32' (0.0-1.0).
    use_depth : float
        Fraction of the Z range to segment (0.0-1.0).
    stride : int
        Process every Nth slice. Default 1 (all slices).
    """
    from ._inference_2d import segment_tomogram_2d

    tta = max(tta, 4)

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    tomograms, patterns = _collect_tomograms(data_directory)

    if not tomograms:
        if not silent:
            print(f"No .mrc files found in {patterns}")
        return

    device = get_device(gpu)

    # Download/convert weights
    pth_path, metadata, scnm_metadata = get_model_weights_2d(feature, silent=silent)
    if pth_path is None:
        if not silent:
            print(f"Could not find model for {feature}! Exiting.")
        return

    if not silent:
        print(
            f"easymode-torch segment\n"
            f"feature: {feature}\n"
            f"data_patterns: {patterns}\n"
            f"output_directory: {output_directory}\n"
            f"output_format: {data_format}\n"
            f"device: {device}\n"
            f"tta: {tta}\n"
            f"overwrite: {overwrite}\n"
            f"2d_model: True\n"
        )
        print(f"Found {len(tomograms)} tomograms to segment.\n")

    model = UNet2D()
    state = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model = model.to(device)

    start_time = time.time()

    for i, tomo_path in enumerate(tomograms, 1):
        tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]
        out_path = output_directory / f"{tomo_name}__{feature}.mrc"

        if out_path.exists() and not overwrite:
            continue

        seg, apix = segment_tomogram_2d(
            model, tomo_path, device,
            tta=tta, use_depth=use_depth, stride=stride,
        )
        _save_mrc(seg, out_path, apix, data_format)

        if not silent:
            elapsed = time.time() - start_time
            eta = time.strftime('%H:%M:%S', time.gmtime(elapsed / i * (len(tomograms) - i)))
            avg_secs = int(elapsed / i)
            per_tomo = f"{avg_secs // 60:02d}:{avg_secs % 60:02d}"
            print(f"{i}/{len(tomograms)} (on {device}) - {feature} - {os.path.basename(tomo_path)} - eta {eta} ({per_tomo} per tomo)")

    if not silent:
        print()
        print(f"\033[92mSegmentation finished!\033[0m")
        print()


__all__ = ['segment', 'segment_2d', 'list_models']
