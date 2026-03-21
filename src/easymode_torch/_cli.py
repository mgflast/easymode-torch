"""CLI for easymode-torch."""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="easymode-torch: PyTorch inference for easymode pretrained segmentation networks."
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    subparsers.add_parser('list', help='List available pretrained segmentation models.')

    seg = subparsers.add_parser('segment', help='Segment tomograms using pretrained easymode networks.')
    seg.add_argument("features", metavar='FEATURE', nargs="+", type=str,
                     help="One or more features to segment (e.g. 'ribosome membrane microtubule'). Use 'easymode-torch list' to see available features.")
    seg.add_argument("--data", nargs="+", type=str, required=True,
                     help="One or more directories, file paths, or glob patterns.")
    seg.add_argument("--tta", type=int, default=4,
                     help="Test-time augmentation factor, 1-16 for 3D, 1-8 for 2D. Higher = better but slower. (default: 4)")
    seg.add_argument("--output", type=str, default="segmented",
                     help="Output directory (default: ./segmented/)")
    seg.add_argument("--overwrite", action='store_true',
                     help="Overwrite existing segmentations.")
    seg.add_argument("--batch", type=int, default=1,
                     help="Batch size for tile processing (default: 1). 3D only.")
    seg.add_argument("--gpu", type=int, default=None,
                     help="GPU device ID (default: auto-select).")
    seg.add_argument("--apix", type=float, default=None,
                     help="Override the pixel size from the .mrc header (Å/px).")
    seg.add_argument("--use_depth", type=float, default=1.0,
                     help="Fraction of Z range to process, 0.0-1.0 (default: 1.0).")
    seg.add_argument("--xy_margin", type=int, default=0,
                     help="Pixels to crop from XY edges (default: 0). 3D only.")
    seg.add_argument("--format", type=str, choices=['float32', 'uint16', 'int8'], default='int8',
                     help="Output format for segmented volumes (default: int8).")
    seg.add_argument("--2d", dest='use_2d', action='store_true',
                     help="Use 2D slice-by-slice model instead of 3D volumetric model.")
    seg.add_argument("--stride", type=int, default=1,
                     help="Process every Nth slice (default: 1). 2D only.")

    args = parser.parse_args()

    if args.command == 'list':
        from ._distribution import list_models
        list_models()

    elif args.command == 'segment':
        features = [f.lower() for f in args.features]
        if args.use_2d:
            from . import segment_2d
            for feature in features:
                segment_2d(
                    feature=feature,
                    data_directory=args.data,
                    output_directory=args.output,
                    tta=args.tta,
                    gpu=args.gpu,
                    overwrite=args.overwrite,
                    silent=False,
                    data_format=args.format,
                    use_depth=args.use_depth,
                    stride=args.stride,
                )
        else:
            from . import segment
            for feature in features:
                segment(
                    feature=feature,
                    data_directory=args.data,
                    output_directory=args.output,
                    tta=args.tta,
                    batch_size=args.batch,
                    input_apix=args.apix,
                    gpu=args.gpu,
                    overwrite=args.overwrite,
                    silent=False,
                    data_format=args.format,
                    use_depth=args.use_depth,
                    xy_margin=args.xy_margin,
                )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
