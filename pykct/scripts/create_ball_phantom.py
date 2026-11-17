#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a 2D circular phantom as a single-slice 3D DEN.

Usage:
    python -m pykct.scripts.create_ball_phantom --radius 10 --antialias phantom_dot.den
"""

import argparse

from denpy import DEN

from pykct.phantoms import create_ball_phantom
from pykct.scripts._common import check_output


def main():
    parser = argparse.ArgumentParser(
        description="Create a 2D circular phantom as a single-slice 3D DEN."
    )
    parser.add_argument("output", help="Output DEN file (required)")
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
    check_output(args.output, args.force)

    volume = create_ball_phantom(
        size=args.size,
        radius=args.radius,
        x_center=args.x,
        y_center=args.y,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        antialias=args.antialias,
        antialias_factor=args.antialias_factor,
    )
    DEN.storeNdarrayAsDEN(args.output, volume, force=True)

    center = (args.size - 1) / 2.0
    x_center = (args.x if args.x is not None else center) + args.x_offset
    y_center = (args.y if args.y is not None else center) + args.y_offset
    print(f"Created phantom: {args.output}")
    print(f"  Size: {args.size}x{args.size}")
    print(f"  Radius: {args.radius}")
    print(f"  Center: ({x_center:.2f}, {y_center:.2f})")
    print(f"  Antialias: {args.antialias_factor}x" if args.antialias else "  Antialias: off")


if __name__ == "__main__":
    main()
