"""Shared helpers for pykct command line scripts."""

import os
import sys


def check_output(path, force):
    """Exit with an error if ``path`` exists and ``force`` is not set."""
    if os.path.exists(path) and not force:
        print(f"Error: File '{path}' already exists.")
        print("Use --force to overwrite.")
        sys.exit(1)


def add_projection_arguments(parser):
    """Projection options shared by the *_sinogram scripts."""
    group = parser.add_argument_group("Projection parameters")
    group.add_argument("--n-angles", type=int, default=512,
                       help="Number of projection angles (default: 512)")
    group.add_argument("--start-angle", type=float, default=0.0,
                       help="Starting angle in degrees (default: 0.0)")
    group.add_argument("--angle-range", type=float, default=360.0,
                       help="Angular range in degrees (default: 360.0)")
    group.add_argument("--detector-size", type=int, default=256,
                       help="Detector size (default: 256)")
    cor = group.add_mutually_exclusive_group()
    cor.add_argument("--cor-offset", type=float, default=None,
                     help="CoR offset in pixels (default: 0.0): rotation axis (principal ray) "
                          "is imaged at detector pixel (N-1)/2 + COR_OFFSET, as "
                          "createProjectionMatrices.py --rotation-center-offset-pix.")
    cor.add_argument("--detector-center-offsetvx", type=float, default=None,
                     help="Detector center offset along VX in pixels as in "
                          "createCameraMatricesForCircularScanTrajectoryParallelRay3D.py, "
                          "equivalent to --cor-offset=-OFFSET.")
    group.add_argument("--pixel-size", type=float, default=1.0,
                       help="Pixel size (default: 1.0)")
    parser.add_argument("--platform", type=str, default=None,
                        help="OpenCL platform:device, e.g. 1:2. Omit it (recommended) to let the "
                             "C++ runtime pick a well performing platform and device by its "
                             "own heuristic; do not pass a dummy 0:0.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress information")
    return group


def projection_kwargs(args):
    """Translate parsed projection options to create_*_sinogram kwargs."""
    if args.cor_offset is not None:
        cor_offset = args.cor_offset
    elif args.detector_center_offsetvx is not None:
        cor_offset = -args.detector_center_offsetvx
    else:
        cor_offset = 0.0
    return dict(
        n_angles=args.n_angles,
        start_angle=args.start_angle,
        angle_range=args.angle_range,
        detector_size=args.detector_size,
        cor_offset=cor_offset,
        pixel_size=args.pixel_size,
        platform_id=args.platform,
        verbose=args.verbose,
    )
