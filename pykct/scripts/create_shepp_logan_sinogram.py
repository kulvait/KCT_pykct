#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create sinogram from Shepp-Logan phantom.

Usage:
    python -m pykct.scripts.create_shepp_logan_sinogram sl_sino.den --detector-center-offsetvx 5.0
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from denpy import DEN
from denpy import PHANTOM
from pykct.core import project_image


def main():
    parser = argparse.ArgumentParser(
        description="Create sinogram from Shepp-Logan phantom."
    )
    
    parser.add_argument("sinogram_file",
                        help="Output sinogram DEN file (required)")
    
    # Phantom parameters
    phantom_group = parser.add_argument_group("Phantom parameters")
    phantom_group.add_argument("-n", "--size", type=int, default=256,
                               help="Phantom size (default: 256)")
    
    # Projection parameters
    proj_group = parser.add_argument_group("Projection parameters")
    proj_group.add_argument("--n-angles", type=int, default=512,
                            help="Number of projection angles (default: 512)")
    proj_group.add_argument("--start-angle", type=float, default=0.0,
                            help="Starting angle in degrees (default: 0.0)")
    proj_group.add_argument("--angle-range", type=float, default=360.0,
                            help="Angular range in degrees (default: 360.0)")
    proj_group.add_argument("--detector-size", type=int, default=256,
                            help="Detector size (default: 256)")
    proj_group.add_argument("--detector-center-offsetvx", type=float, default=0.0,
                            help="CoR offset in pixels (default: 0.0). "
                                 "Positive = CoR to the right of detector center.")
    proj_group.add_argument("--cor-offset", type=float, default=None,
                            help="Alias for --detector-center-offsetvx")
    proj_group.add_argument("--pixel-size", type=float, default=1.0,
                            help="Pixel size (default: 1.0)")
    
    parser.add_argument("--platform", type=str, default=None,
                        help="OpenCL platform ID (let C++ runtime auto-detect if not specified)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing output file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress information")
    
    args = parser.parse_args()
    
    # Resolve CoR offset
    cor_offset = args.cor_offset if args.cor_offset is not None else args.detector_center_offsetvx
    
    # Create phantom using denpy's specification
    phantom_2d = PHANTOM.phantom2d(n=args.size).astype(np.float32)
    
    # Create sinogram
    sinogram = project_image(
        image=phantom_2d,
        n_angles=args.n_angles,
        start_angle=args.start_angle,
        angle_range=args.angle_range,
        detector_sizex=args.detector_size,
        detector_sizey=1,
        cor_offset=cor_offset,
        pixel_size=args.pixel_size,
        platform_id=args.platform,
        verbose=args.verbose
    )
    
    # Check if output file exists
    if os.path.exists(args.sinogram_file) and not args.force:
        print(f"Error: File '{args.sinogram_file}' already exists.")
        print(f"Use --force to overwrite.")
        sys.exit(1)
    
    # Save
    DEN.storeNdarrayAsDEN(args.sinogram_file, sinogram, force=args.force)
    
    print(f"Created sinogram: {args.sinogram_file}")
    print(f"  Phantom: Shepp-Logan, size={args.size}")
    print(f"  Projection: {args.n_angles} angles, detector={args.detector_size}")
    print(f"  CoR offset: {cor_offset:+.2f} px")
    print(f"  Sinogram shape: {sinogram.shape}")


if __name__ == "__main__":
    main()
