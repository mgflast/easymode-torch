"""PyTorch segmentation inference with tiled processing.

Adapted from easymode inference.py — same tiling/overlap logic, same
test-time augmentation, but running in PyTorch instead of TensorFlow.
"""

import gc
from pathlib import Path

import mrcfile
import numpy as np
import torch

TILE_SIZE = (128, 256, 256)
OVERLAP = (32, 32, 32)


# ---------------------------------------------------------------------------
# Tiling helpers (framework-agnostic numpy)
# ---------------------------------------------------------------------------

def _tile_volume(volume, patch_size, overlap):
    pz, py, px = patch_size
    oz, oy, ox = overlap
    d, h, w = volume.shape
    sz, sy, sx = pz - 2 * oz, py - 2 * oy, px - 2 * ox

    z_boxes = max(1, (d + sz - 1) // sz)
    y_boxes = max(1, (h + sy - 1) // sy)
    x_boxes = max(1, (w + sx - 1) // sx)

    tiles, positions = [], []
    for zi in range(z_boxes):
        for yi in range(y_boxes):
            for xi in range(x_boxes):
                z0 = max(0, zi * sz - oz)
                y0 = max(0, yi * sy - oy)
                x0 = max(0, xi * sx - ox)
                z1 = min(d, zi * sz - oz + pz)
                y1 = min(h, yi * sy - oy + py)
                x1 = min(w, xi * sx - ox + px)

                extracted = volume[z0:z1, y0:y1, x0:x1]
                tile = np.zeros((pz, py, px), dtype=volume.dtype)
                tz, ty, tx = z0 - (zi * sz - oz), y0 - (yi * sy - oy), x0 - (xi * sx - ox)
                tile[tz:tz + extracted.shape[0], ty:ty + extracted.shape[1], tx:tx + extracted.shape[2]] = extracted

                tiles.append(tile)
                positions.append((zi * sz, yi * sy, xi * sx))

    return np.array(tiles), positions, volume.shape


def _detile_volume(segmented_tiles, positions, original_shape, patch_size, overlap):
    pz, py, px = patch_size
    oz, oy, ox = overlap
    d, h, w = original_shape
    sz, sy, sx = pz - 2 * oz, py - 2 * oy, px - 2 * ox

    out = np.zeros((d, h, w), dtype=np.float32)
    wgt = np.zeros((d, h, w), dtype=np.float32)

    for tile, (zp, yp, xp) in zip(segmented_tiles, positions):
        center = tile[oz:oz + sz, oy:oy + sy, ox:ox + sx]
        ze = min(zp + sz, d)
        ye = min(yp + sy, h)
        xe = min(xp + sx, w)
        az, ay, ax = ze - zp, ye - yp, xe - xp
        out[zp:ze, yp:ye, xp:xe] += center[:az, :ay, :ax]
        wgt[zp:ze, yp:ye, xp:xe] += 1.0

    wgt[wgt == 0] = 1.0
    return out / wgt


def _pad_volume(volume, min_pad=16, div=32):
    pads = []
    for n in volume.shape:
        total = max(2 * min_pad, ((n + 2 * min_pad + div - 1) // div) * div - n)
        before = total // 2
        pads.append((before, total - before))
    return np.pad(volume, pads, mode='reflect'), tuple(pads)


# ---------------------------------------------------------------------------
# Torch inference on tiles
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_tiles(tiles, model, device, batch_size=2):
    """Run model on a batch of numpy tiles, return numpy results."""
    # tiles: (N, D, H, W) -> add channel dim -> (N, 1, D, H, W)
    results = []
    n = len(tiles)
    for i in range(0, n, batch_size):
        chunk = tiles[i:i + batch_size]
        t = torch.from_numpy(chunk[:, np.newaxis]).float().to(device)
        out = model(t)                          # (B, 1, D, H, W)
        results.append(out.squeeze(1).cpu().numpy())
        del t, out
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def segment_tomogram(model, tomogram_path, device, tta=1, batch_size=2,
                     model_apix=10.0, input_apix=None, model_apix_z=None):
    """Segment a single tomogram using a loaded PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded, eval-mode PyTorch UNet
    tomogram_path : str or Path
    device : torch.device
    tta : int
        Number of test-time augmentation passes (1 = no TTA, max 16)
    batch_size : int
    model_apix : float
        Pixel size the model was trained at (Å/px, XY)
    input_apix : float or None
        Override pixel size from MRC header
    model_apix_z : float or None
        Z pixel size if model is anisotropic; None = same as model_apix

    Returns
    -------
    tuple[np.ndarray, float]
        (segmentation_volume, voxel_size_angstroms)
    """
    if model_apix_z is None:
        model_apix_z = model_apix

    with mrcfile.open(str(tomogram_path)) as m:
        volume = m.data.astype(np.float32)
        volume_apix = float(m.voxel_size.x)
        oj, ok, ol = volume.shape
        if volume_apix <= 1.0 and input_apix is None:
            print(f"warning: {tomogram_path} header lists voxel size as 1.0 Å/px — assuming 10.0 Å/px")
            volume_apix = 10.0

    apix_xy = input_apix if input_apix else volume_apix
    scale_xy = apix_xy / model_apix
    scale_z = apix_xy / model_apix_z

    rescaled = False
    if abs(scale_xy - 1.0) > 0.05 or abs(scale_z - 1.0) > 0.05:
        from scipy.ndimage import zoom
        volume = zoom(volume, (scale_z, scale_xy, scale_xy), order=1)
        rescaled = True

    # Normalize
    _j, _k, _l = volume.shape
    km = min(int(0.2 * _k), 64)
    lm = min(int(0.2 * _l), 64)
    volume -= np.mean(volume[:, km:-km, lm:-lm])
    volume /= np.std(volume[:, km:-km, lm:-lm]) + 1e-7

    volume, padding = _pad_volume(volume)

    tile_size = (
        min(256, volume.shape[0]),
        min(256, volume.shape[1]),
        min(256, volume.shape[2]),
    )
    overlap = tuple(0 if tile_size[i] == volume.shape[i] else 48 for i in range(3))

    # 16 TTA augmentations (same as easymode)
    k_xy = [0, 2, 2, 0, 1, 3, 0, 1, 2, 3, 0, 1, 2, 3, 1, 3]
    k_fx = [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    k_yz = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1]

    seg_accum = np.zeros((oj, ok, ol), dtype=np.float32)

    for j in range(tta):
        aug = volume.copy()
        aug = np.rot90(aug, k=k_xy[j], axes=(1, 2))
        if k_fx[j]:
            aug = np.flip(aug, axis=1)
        aug = np.rot90(aug, k=2 * k_yz[j], axes=(0, 1))

        tiles, positions, orig_shape = _tile_volume(aug, tile_size, overlap)
        seg_tiles = _run_tiles(tiles, model, device, batch_size)
        seg_aug = _detile_volume(seg_tiles, positions, orig_shape, tile_size, overlap)

        # Undo augmentation
        seg_aug = np.rot90(seg_aug, k=-2 * k_yz[j], axes=(0, 1))
        if k_fx[j]:
            seg_aug = np.flip(seg_aug, axis=1)
        seg_aug = np.rot90(seg_aug, k=-k_xy[j], axes=(1, 2))

        # Remove padding
        (j0, j1), (k0, k1), (l0, l1) = padding
        seg_aug = seg_aug[j0:seg_aug.shape[0] - j1, k0:seg_aug.shape[1] - k1, l0:seg_aug.shape[2] - l1]

        if rescaled:
            from scipy.ndimage import zoom as zoom_
            sj, sk, sl = seg_aug.shape
            seg_aug = zoom_(seg_aug, (oj / sj, ok / sk, ol / sl), order=1)
            seg_aug = seg_aug[:oj, :ok, :ol]

        seg_accum += seg_aug

    return seg_accum / tta, volume_apix
