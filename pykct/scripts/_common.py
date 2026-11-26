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
    parser.add_argument("--platform", type=str, default=None,
                        help="OpenCL platform:device, e.g. 1:2. Omit it (recommended) to let the "
                             "C++ runtime pick a well performing platform and device by its "
                             "own heuristic; do not pass a dummy 0:0.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress information")
    return group


def add_discretization_arguments(parser):
    """Antialiasing and supersampling options of create_phantom2d."""
    group = parser.add_argument_group("Discretization")
    group.add_argument("--antialias", action="store_true",
                       help="Average ANTIALIAS_FACTOR^2 samples per pixel")
    group.add_argument("--antialias-factor", type=int, default=4,
                       help="Samples per pixel and axis with --antialias (default: 4)")
    group.add_argument("--supersampling-factor", type=int, default=1,
                       help="Draw on a grid SUPERSAMPLING_FACTOR times finer, i.e. "
                            "SIZE*FACTOR pixels per side (default: 1, off); sinograms keep "
                            "the geometry and units of the SIZE grid")
    return group


def discretization_kwargs(args):
    """Translate parsed discretization options to create_phantom2d kwargs."""
    if args.antialias_factor < 1 or args.supersampling_factor < 1:
        raise SystemExit("Error: --antialias-factor and --supersampling-factor must be >= 1")
    antialias = bool(args.antialias) and args.antialias_factor > 1
    return dict(
        antialias=antialias,
        antialias_factor=args.antialias_factor if antialias else 1,
        supersampling=args.supersampling_factor > 1,
        supersampling_factor=args.supersampling_factor,
    )


def projection_kwargs(args):
    """Translate parsed projection options to create_sinogram2d kwargs."""
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
        platform_id=args.platform,
        verbose=args.verbose,
    )
