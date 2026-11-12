#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a 2D circular phantom as a single-slice 3D DEN.

Usage:
    python -m pykct.scripts.create_ball_phantom --radius 10 --antialias phantom_dot.den
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from denpy import DEN


def create_circle(
    size: int,
    radius: float,
    x_center: float = None,
    y_center: float = None,
    antialias: int = 1
) -> np.ndarray:
    """
    Create a 2D circular phantom with supersampling antialiasing.
    
    This is the actual implementation from create_ball_phantom.py
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
        description="Create a 2D circular phantom as a single-slice 3D DEN."
    )
    
    parser.add_argument("output",
                        help="Output DEN file (required)")
    
    parser.add_argument("-n", "--size", type=int, default=256,
                        help="Image width and height (default: 256)")
    
    parser.add_argument("-r", "--radius", type=float, default=10.0,
                        help="Circle radius in pixels (default: 10.0)")
    
    parser.add_argument("--x", type=float, default=None,
                        help="Absolute X coordinate (default: image center)")
    parser.add_argument("--y", type=float, default=None,
                        help="Absolute Y coordinate (default: image center)")
    
    parser.add_argument("--x-offset", type=float, default=0.0,
                        help="X offset from the selected/default X coordinate")
    parser.add_argument("--y-offset", type=float, default=0.0,
                        help="Y offset from the selected/default Y coordinate")
    
    parser.add_argument("--antialias", action="store_true",
                        help="Enable supersampling antialiasing")
    
    parser.add_argument("--antialias-factor", type=int, default=4,
                        help="Number of subpixel samples per dimension (default: 4)")
    
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output file")
    
    args = parser.parse_args()
    
    if args.size < 1:
        parser.error("--size must be positive")
    
    if args.radius <= 0:
        parser.error("--radius must be positive")
    
    if args.antialias_factor < 1:
        parser.error("--antialias-factor must be positive")
    
    # Geometric center
    center = (args.size - 1) / 2.0
    
    # Absolute coordinates
    x_center = (args.x if args.x is not None else center) + args.x_offset
    y_center = (args.y if args.y is not None else center) + args.y_offset
    
    # Select antialiasing factor
    aa_factor = args.antialias_factor if args.antialias else 1
    
    # Create 2D circle
    image_2d = create_circle(
        size=args.size,
        radius=args.radius,
        x_center=x_center,
        y_center=y_center,
        antialias=aa_factor
    )
    
    # Convert to 3D volume (z, y, x)
    volume = image_2d[np.newaxis, :, :].astype(np.float32)
    
    # Check if output file exists
    if os.path.exists(args.output) and not args.force:
        print(f"Error: File '{args.output}' already exists.")
        print(f"Use --force to overwrite.")
        sys.exit(1)
    
    # Store as DEN
    DEN.storeNdarrayAsDEN(args.output, volume, force=args.force)
    
    print(f"Created phantom: {args.output}")
    print(f"  Size: {args.size}x{args.size}")
    print(f"  Radius: {args.radius}")
    print(f"  Center: ({x_center:.2f}, {y_center:.2f})")
    print(f"  Antialias: {aa_factor}x" if args.antialias else "  Antialias: off")


if __name__ == "__main__":
    main()
