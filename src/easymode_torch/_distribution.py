"""Download and cache easymode model weights from HuggingFace.

Downloads the .h5 / .scnm TF weights, converts to .pth on first use,
and caches the result so subsequent loads are TF-free.
"""

import json
import os
from pathlib import Path

import urllib.request
from huggingface_hub import hf_hub_download

HF_REPO_ID = "mgflast/easymode"
_DEFAULT_CACHE = Path.home() / ".cache" / "easymode_torch"


def _cache_dir():
    d = Path(os.environ.get("EASYMODE_TORCH_CACHE", _DEFAULT_CACHE))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_online():
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


def _download_file(filename, dest_dir, silent=False):
    if not silent:
        print(f"Downloading {filename} from {HF_REPO_ID}...")
    return hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        local_dir=str(dest_dir),
    )


def _get_metadata(feature, cache, online, _2d=False):
    """Fetch remote and/or local metadata for a feature."""
    meta_filename = f"{feature}_2d.json" if _2d else f"{feature}.json"
    meta_path = cache / meta_filename

    remote_meta = None
    if online:
        try:
            downloaded = _download_file(meta_filename, cache, silent=True)
            with open(downloaded) as f:
                remote_meta = json.load(f)
        except Exception:
            pass

    local_meta = None
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                local_meta = json.load(f)
        except Exception:
            pass

    return remote_meta, local_meta


def _needs_update(remote_meta, local_meta):
    if not remote_meta or not local_meta:
        return False
    remote_ts = remote_meta.get("timestamp")
    local_ts = local_meta.get("timestamp")
    return bool(remote_ts and local_ts and remote_ts > local_ts)


def get_model_weights(feature, force_download=False, silent=False):
    """Return path to PyTorch .pth weights for a 3D model.

    Downloads .h5 from HuggingFace if needed, converts to .pth on first use.

    Returns
    -------
    tuple[Path, dict]
        (path_to_pth, metadata_dict)
    """
    cache = _cache_dir()
    h5_path = cache / f"{feature}.h5"
    pth_path = cache / f"{feature}.pth"

    online = _is_online()
    remote_meta, local_meta = _get_metadata(feature, cache, online, _2d=False)
    metadata = remote_meta or local_meta

    if metadata is None and not h5_path.exists():
        raise RuntimeError(
            f"Model '{feature}' not found on HuggingFace and not in local cache. "
            f"Run 'easymode-torch list' to see available models."
        )

    needs_download = force_download or not h5_path.exists()
    if not needs_download and _needs_update(remote_meta, local_meta):
        needs_download = True
        if not silent:
            print(f"New version of {feature} available, updating...")

    if needs_download:
        if not online:
            raise RuntimeError(
                f"Weights for '{feature}' not cached and no internet connection available."
            )
        _download_file(f"{feature}.h5", cache, silent=silent)
        if pth_path.exists():
            pth_path.unlink()

    if not pth_path.exists():
        if not silent:
            print(f"Converting {feature} weights to PyTorch format...")
        from ._convert import convert_h5_to_pth
        convert_h5_to_pth(h5_path, pth_path)
        if not silent:
            print(f"Cached PyTorch weights at {pth_path}")

    return pth_path, metadata


def get_model_weights_2d(feature, force_download=False, silent=False):
    """Return path to PyTorch .pth weights for a 2D model.

    Downloads .scnm from HuggingFace if needed, converts to .pth on first use.

    Returns
    -------
    tuple[Path, dict, dict]
        (path_to_pth, hf_metadata, scnm_metadata)
        hf_metadata: from {feature}_2d.json (apix, timestamp)
        scnm_metadata: from the .scnm archive (box_size, overlap, threshold, etc.)
    """
    cache = _cache_dir()
    scnm_path = cache / f"{feature}.scnm"
    pth_path = cache / f"{feature}_2d.pth"
    scnm_meta_path = cache / f"{feature}_2d_scnm.json"

    online = _is_online()
    remote_meta, local_meta = _get_metadata(feature, cache, online, _2d=True)
    metadata = remote_meta or local_meta

    if metadata is None and not scnm_path.exists():
        raise RuntimeError(
            f"2D model '{feature}' not found on HuggingFace and not in local cache. "
            f"Run 'easymode-torch list' to see available models."
        )

    needs_download = force_download or not scnm_path.exists()
    if not needs_download and _needs_update(remote_meta, local_meta):
        needs_download = True
        if not silent:
            print(f"New version of {feature} (2D) available, updating...")

    if needs_download:
        if not online:
            raise RuntimeError(
                f"2D weights for '{feature}' not cached and no internet connection available."
            )
        _download_file(f"{feature}.scnm", cache, silent=silent)
        if pth_path.exists():
            pth_path.unlink()
        if scnm_meta_path.exists():
            scnm_meta_path.unlink()

    # Convert .scnm -> .pth if needed
    scnm_metadata = None
    if not pth_path.exists():
        if not silent:
            print(f"Converting {feature} 2D weights to PyTorch format...")
        from ._convert_2d import convert_scnm_to_pth
        _, scnm_metadata = convert_scnm_to_pth(scnm_path, pth_path)
        # Cache the scnm metadata separately
        if scnm_metadata:
            with open(scnm_meta_path, 'w') as f:
                json.dump(scnm_metadata, f)
        if not silent:
            print(f"Cached PyTorch 2D weights at {pth_path}")
    else:
        if scnm_meta_path.exists():
            with open(scnm_meta_path) as f:
                scnm_metadata = json.load(f)

    return pth_path, metadata, scnm_metadata


def list_models(silent=False):
    """List available segmentation models on HuggingFace.

    Returns
    -------
    list[dict]
        List of dicts with 'title', 'has_3d', 'has_2d' keys
    """
    from huggingface_hub import HfApi
    api = HfApi()
    files = list(api.list_repo_files(HF_REPO_ID))

    h5_bases = {
        os.path.splitext(os.path.basename(f))[0]
        for f in files
        if f.endswith('.h5') and not any(x in f for x in ('ddw', 'n2n', 'tilt'))
    }

    scnm_bases = {
        os.path.splitext(os.path.basename(f))[0]
        for f in files
        if f.endswith('.scnm')
    }

    all_bases = sorted(h5_bases | scnm_bases)

    models = []
    if not silent:
        print("\nAvailable segmentation models:")
    for base in all_bases:
        has_3d = base in h5_bases
        has_2d = base in scnm_bases
        if has_3d and has_2d:
            dim = "3D / 2D"
        elif has_3d:
            dim = "3D"
        else:
            dim = "2D (--2d)"

        if not silent:
            print(f"  {base.ljust(30)} {dim}")
        models.append({"title": base, "has_3d": has_3d, "has_2d": has_2d})

    if not silent:
        print()
    return models
