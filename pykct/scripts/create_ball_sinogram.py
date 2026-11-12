#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create sinogram from ball phantom.

Usage:
    python -m pykct.scripts.create_ball_sinogram ball_sino.den --radius 10 --detector-center-offsetvx 5.0
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from denpy import DEN
from pykct.core import project_image


def create_circle(
    size: int,
    radius: float,
    x_center: float = None,
    y_center: float = None,
    antialias: int = 1
) -> np.ndarray:
    """
    Create a 2D circular phantom with supersampling antialiasing.
    """
    if antialias < 1:
        raise ValueError("Antialias factor must be >= 1")
    
    # Geometric center
    if x_center is None:
        x_center = (size - 1) / 2.0
    if y_center is None:
        y_center = (size - 1) / 2.0
    
    # Pixel coordinates
    yy, xx = np.ogrid[:size, :size]
    
    if antialias == 1:
        # Binary circle
        distance_squared = (xx - x_center)**2 + (yy - y_center)**2
        return (distance_squared <= radius**2).astype(np.float32)
    
    # Supersampling antialiasing
    offsets = (np.arange(antialias) + 0.5) / antialias - 0.5
    
    image = np.zeros((size, size), dtype=np.float32)
    
    for dy in offsets:
        for dx in offsets:
            distance_squared = (xx + dx - x_center)**2 + (yy + dy - y_center)**2
            image += (distance_squared <= radius**2)
    
    image /= antialias**2
    
    return image


def main():
    parser = argparse.ArgumentParser(
        description="Create sinogram from ball phantom."
    )
    
    parser.add_argument("sinogram_file",
                        help="Output sinogram DEN file (required)")
    
    # Phantom parameters
    phantom_group = parser.add_argument_group("Phantom parameters")
    phantom_group.add_argument("-n", "--size", type=int, default=256,
                               help="Phantom size (default: 256)")
    phantom_group.add_argument("-r", "--radius", type=float, default=10.0,
                               help="Ball radius in pixels (default: 10.0)")
    phantom_group.add_argument("--x-offset", type=float, default=0.0,
                               help="X offset from center (default: 0.0)")
    phantom_group.add_argument("--y-offset", type=float, default=0.0,
                               help="Y offset from center (default: 0.0)")
    phantom_group.add_argument("--antialias", action="store_true",
                               help="Enable supersampling antialiasing")
    phantom_group.add_argument("--antialias-factor", type=int, default=4,
                               help="Antialiasing factor (default: 4)")
    
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
    
    # Create phantom
    center = (args.size - 1) / 2.0
    x_center = center + args.x_offset
    y_center = center + args.y_offset
    aa_factor = args.antialias_factor if args.antialias else 1
    
    phantom_2d = create_circle(
        size=args.size,
        radius=args.radius,
        x_center=x_center,
        y_center=y_center,
        antialias=aa_factor
    )
    
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
    print(f"  Phantom: size={args.size}, radius={args.radius}")
    print(f"  Projection: {args.n_angles} angles, detector={args.detector_size}")
    print(f"  CoR offset: {cor_offset:+.2f} px")
    print(f"  Sinogram shape: {sinogram.shape}")


if __name__ == "__main__":
    main()
