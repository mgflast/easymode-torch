# easymode-torch

PyTorch inference for [easymode](https://github.com/mgflast/easymode) pretrained segmentation networks. No TensorFlow required.

Uses the same pretrained weights from HuggingFace as easymode, automatically converted to PyTorch on first use.

Based on torch inference work by [@alisterburt](https://github.com/mgflast/easymode/pull/4).

## Install

```bash
pip install git+https://github.com/mgflast/easymode-torch.git
```

## CLI usage

```bash
# List available models
easymode-torch list

# Segment tomograms
easymode-torch segment ribosome --data /path/to/tomograms/

# Multiple features at once
easymode-torch segment ribosome membrane microtubule --data volumes/

# All options
easymode-torch segment ribosome \
    --data volumes/ \
    --output segmented/ \
    --tta 4 \
    --gpu 0 \
    --overwrite
```

## Python API

```python
import easymode_torch

# List available models
easymode_torch.list_models()

# Segment tomograms
easymode_torch.segment(
    'ribosome',
    'path/to/tomograms/',
    'segmented/',
    tta=4,
)
```

## How it works

1. Downloads `.h5` weights from HuggingFace (`mgflast/easymode`)
2. Converts to `.pth` on first use via `h5py` (no TensorFlow needed)
3. Caches the `.pth` at `~/.cache/easymode_torch/`
4. Runs tiled inference with test-time augmentation in PyTorch

## CLI options

| Option | Default | Description |
|---|---|---|
| `--data` | required | Directories, file paths, or glob patterns |
| `--output` | `segmented/` | Output directory |
| `--tta` | `4` | Test-time augmentation passes (1-16) |
| `--gpu` | auto | GPU device ID |
| `--batch` | `1` | Tiles per batch |
| `--apix` | from header | Override pixel size (Å/px) |
| `--overwrite` | off | Re-segment existing outputs |
| `--use_depth` | `1.0` | Fraction of Z range to process |
| `--xy_margin` | `0` | Pixels to crop from XY edges |
