"""2D slice-by-slice segmentation inference.

Processes each Z-slice of a tomogram independently through a 2D UNet.
Matches the Ais CLI inference logic from cli_fn._segment_tomo.
"""

import gc

import mrcfile
import numpy as np
import torch


@torch.no_grad()
def _segment_slices(volume, model, device):
    """Run 2D model on each Z-slice.

    Parameters
    ----------
    volume : np.ndarray
        (D, H, W) normalized volume, H and W padded to multiple of 32.
    model : torch.nn.Module
    device : torch.device

    Returns
    -------
    np.ndarray
        (D, H, W) segmented volume
    """
    d, h, w = volume.shape
    segmented = np.zeros_like(volume)

    for j in range(d):
        slc = torch.from_numpy(volume[j:j+1][np.newaxis]).float().to(device)  # (1, 1, H, W)
        out = model(slc)
        segmented[j] = out.squeeze().cpu().numpy()
        del slc, out

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return segmented


def segment_tomogram_2d(model, tomogram_path, device, tta=1,
                        use_depth=1.0, stride=1):
    """Segment a tomogram slice-by-slice using a 2D model.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded, eval-mode PyTorch UNet2D
    tomogram_path : str or Path
    device : torch.device
    tta : int
        Test-time augmentation passes (1-8).
        Augmentations are [0, 90, 180, 270] degree rotations,
        optionally with horizontal flip (indices 4-7).
    use_depth : float
        Fraction of Z range to process (0.0-1.0). Default 1.0.
    stride : int
        Process every Nth slice. Default 1 (all slices).

    Returns
    -------
    tuple[np.ndarray, float]
        (segmentation_volume, voxel_size_angstroms)
    """
    with mrcfile.open(str(tomogram_path)) as m:
        volume = m.data.astype(np.float32)
        volume_apix = float(m.voxel_size.x)

    # Normalize
    volume -= np.mean(volume)
    std = np.std(volume)
    if std > 0:
        volume /= std

    d, h, w = volume.shape

    # Compute active Z range
    z_margin = int(d * (1.0 - use_depth) / 2) if d * use_depth >= 32 else 0
    z_start = z_margin
    z_end = d - z_margin

    # Pad H, W to multiple of 32
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        volume = np.pad(volume, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')

    segmented_volume = np.zeros((d, h, w), dtype=np.float32)

    # TTA augmentations: rotation + flip
    rotations = [0, 1, 2, 3, 0, 1, 2, 3]
    flips =     [0, 0, 0, 0, 1, 1, 1, 1]

    tta = min(tta, 8)

    for k in range(tta):
        aug = volume.copy()

        # Apply augmentation
        aug = np.rot90(aug, k=rotations[k], axes=(1, 2))
        if flips[k]:
            aug = np.flip(aug, axis=2)

        # Re-pad after rotation (dims may have changed)
        _, ah, aw = aug.shape
        rpad_h = (32 - (ah % 32)) % 32
        rpad_w = (32 - (aw % 32)) % 32
        if rpad_h > 0 or rpad_w > 0:
            aug = np.pad(aug, ((0, 0), (0, rpad_h), (0, rpad_w)), mode='reflect')

        # Segment slices
        seg_aug = np.zeros_like(aug)
        for j in range(z_start, z_end, stride):
            slc = torch.from_numpy(aug[j:j+1][np.newaxis]).float().to(device)  # (1, 1, H, W)
            out = model(slc)
            seg_aug[j] = out.squeeze().cpu().numpy()
            del slc, out

        # Remove rotation padding
        if rpad_h > 0:
            seg_aug = seg_aug[:, :ah, :]
        if rpad_w > 0:
            seg_aug = seg_aug[:, :, :aw]

        # Undo augmentation
        if flips[k]:
            seg_aug = np.flip(seg_aug, axis=2)
        seg_aug = np.rot90(seg_aug, k=-rotations[k], axes=(1, 2))

        # Crop back to original size
        seg_aug = seg_aug[:, :h, :w]

        segmented_volume += seg_aug

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    segmented_volume /= tta
    segmented_volume = np.clip(segmented_volume, 0.0, 1.0)
    return segmented_volume, volume_apix
