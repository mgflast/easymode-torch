"""Download and cache easymode model weights from HuggingFace.

Downloads the .h5 TF weights, converts to .pth on first use, and caches
the result so subsequent loads are TF-free.
"""

import json
import os
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

HF_REPO_ID = "mgflast/easymode"
_DEFAULT_CACHE = Path.home() / ".cache" / "easymode_torch"


def _cache_dir():
    d = Path(os.environ.get("EASYMODE_TORCH_CACHE", _DEFAULT_CACHE))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _is_online():
    try:
        return requests.get("https://huggingface.co", timeout=5).status_code == 200
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


def get_model_weights(feature, force_download=False, silent=False):
    """Return path to PyTorch .pth weights for the given feature.

    Downloads the .h5 weights from HuggingFace if needed, converts them
    to .pth on first use, and caches the result.

    Parameters
    ----------
    feature : str
        Model name, e.g. 'ribosome', 'microtubule'
    force_download : bool
        Re-download even if cached
    silent : bool
        Suppress progress output

    Returns
    -------
    tuple[Path, dict]
        (path_to_pth, metadata_dict)
    """
    cache = _cache_dir()
    h5_path = cache / f"{feature}.h5"
    pth_path = cache / f"{feature}.pth"
    meta_path = cache / f"{feature}.json"

    online = _is_online()

    # Try to fetch fresh metadata
    remote_meta = None
    if online:
        try:
            downloaded_meta = _download_file(f"{feature}.json", cache, silent=True)
            with open(downloaded_meta) as f:
                remote_meta = json.load(f)
        except Exception:
            pass

    # Load local metadata fallback
    local_meta = None
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                local_meta = json.load(f)
        except Exception:
            pass

    metadata = remote_meta or local_meta

    if metadata is None and not h5_path.exists():
        raise RuntimeError(
            f"Model '{feature}' not found on HuggingFace and not in local cache. "
            f"Run 'python -m easymode_torch list' to see available models."
        )

    # Check if weights need (re)downloading
    needs_download = force_download or not h5_path.exists()
    if not needs_download and remote_meta and local_meta:
        remote_ts = remote_meta.get("timestamp")
        local_ts = local_meta.get("timestamp")
        if remote_ts and local_ts and remote_ts > local_ts:
            needs_download = True
            if not silent:
                print(f"New version of {feature} available, updating...")

    if needs_download:
        if not online:
            raise RuntimeError(
                f"Weights for '{feature}' not cached and no internet connection available."
            )
        _download_file(f"{feature}.h5", cache, silent=silent)
        # Invalidate cached .pth so it gets re-converted
        if pth_path.exists():
            pth_path.unlink()

    # Convert .h5 -> .pth if needed
    if not pth_path.exists():
        if not silent:
            print(f"Converting {feature} weights to PyTorch format...")
        from ._convert import convert_h5_to_pth
        convert_h5_to_pth(h5_path, pth_path)
        if not silent:
            print(f"Cached PyTorch weights at {pth_path}")

    return pth_path, metadata


def list_models(silent=False):
    """List available segmentation models on HuggingFace.

    Returns
    -------
    list[str]
        Feature names with 3D models available
    """
    from huggingface_hub import HfApi
    api = HfApi()
    files = list(api.list_repo_files(HF_REPO_ID))
    models = sorted(
        os.path.splitext(os.path.basename(f))[0]
        for f in files
        if f.endswith('.h5') and not any(x in f for x in ('ddw', 'n2n', 'tilt'))
    )
    if not silent:
        print("\nAvailable 3D segmentation models:")
        for m in models:
            print(f"  {m}")
        print()
    return models
