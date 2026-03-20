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

import mrcfile
import numpy as np
import torch

from ._distribution import get_model_weights, list_models
from ._model import UNet
from ._utils import get_device


def _load_model(pth_path, device):
    model = UNet()
    state = torch.load(pth_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def _save_mrc(data, path, voxel_size):
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(data.astype(np.float32))
        m.voxel_size = voxel_size


def segment(
    feature,
    data_directory,
    output_directory='segmented',
    *,
    tta=1,
    batch_size=2,
    input_apix=None,
    gpu=None,
    overwrite=False,
    silent=False,
    use_depth=1.0,
    xy_margin=0,
):
    """Segment tomograms using a pretrained easymode model.

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
        More passes = better quality, slower.
    batch_size : int
        Number of tiles to process in parallel. Reduce if you run out of GPU memory.
    input_apix : float or None
        Override the pixel size from the MRC header (Å/px).
    gpu : int or None
        GPU device ID. None = auto-select (GPU if available, else CPU).
    overwrite : bool
        Re-segment tomograms that already have output files.
    silent : bool
        Suppress progress output.
    use_depth : float
        Fraction of the Z range to segment (0.0–1.0). Default 1.0 = full volume.
    xy_margin : int
        Pixels to crop from XY edges before segmenting (in original coords).
    """
    from ._inference import segment_tomogram

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Collect tomograms
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
    tomograms = sorted(set(f for f in tomograms if f.endswith('.mrc')))

    if not tomograms:
        print(f"No .mrc files found in {patterns}")
        return

    if not silent:
        print(f"easymode-torch segment\n  feature: {feature}\n  tomograms: {len(tomograms)}")

    # Download/convert weights
    pth_path, metadata = get_model_weights(feature, silent=silent)
    model_apix = metadata["apix"] if metadata else 10.0
    model_apix_z = metadata.get("apix_z") if metadata else None

    device = get_device(gpu)
    if not silent:
        print(f"  device: {device}  model_apix: {model_apix} Å/px")

    model = _load_model(pth_path, device)

    for i, tomo_path in enumerate(tomograms, 1):
        tomo_name = os.path.splitext(os.path.basename(tomo_path))[0]
        out_path = output_directory / f"{tomo_name}__{feature}.mrc"

        if out_path.exists() and not overwrite:
            if not silent:
                print(f"  [{i}/{len(tomograms)}] skipping {tomo_name} (already exists)")
            continue

        if not silent:
            print(f"  [{i}/{len(tomograms)}] {tomo_name}")

        seg, apix = segment_tomogram(
            model, tomo_path, device,
            tta=tta, batch_size=batch_size,
            model_apix=model_apix, input_apix=input_apix,
            model_apix_z=model_apix_z,
            use_depth=use_depth, xy_margin=xy_margin,
        )
        _save_mrc(seg, out_path, apix)

    if not silent:
        print(f"\nDone. Results in {output_directory}/")


__all__ = ['segment', 'list_models']
