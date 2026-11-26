#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a 2D disc phantom (pykct.Circle) as a single-slice DEN (1, n, n).

Usage:
    python -m pykct.scripts.create_ball_phantom --radius 10 --antialias phantom_dot.den
"""

import argparse

from denpy import DEN

from pykct.phantoms import Circle, create_phantom2d
from pykct.scripts._common import (add_discretization_arguments, check_output,
                                   discretization_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Create a 2D disc phantom as a single-slice DEN (1, n, n)."
    )
    parser.add_argument("output", help="Output DEN file (required)")
    parser.add_argument("-n", "--size", type=int, default=256,
                        help="Image width and height in pixels (default: 256)")
    parser.add_argument("-r", "--radius", type=float, default=10.0,
                        help="Disc radius in pixels (default: 10.0)")
    parser.add_argument("--x", type=float, default=None,
                        help="Absolute X coordinate (default: image center (n-1)/2)")
    parser.add_argument("--y", type=float, default=None,
                        help="Absolute Y coordinate (default: image center (n-1)/2)")
    parser.add_argument("--x-offset", type=float, default=0.0,
                        help="X offset from the selected/default X coordinate")
    parser.add_argument("--y-offset", type=float, default=0.0,
                        help="Y offset from the selected/default Y coordinate")
    add_discretization_arguments(parser)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output file")
    args = parser.parse_args()

    if args.size < 1:
        parser.error("--size must be positive")
    if args.radius <= 0:
        parser.error("--radius must be positive")
    check_output(args.output, args.force)

    disc = Circle(radius=args.radius, x_center=args.x, y_center=args.y,
                  x_offset=args.x_offset, y_offset=args.y_offset)
    image = create_phantom2d(args.size, disc, **discretization_kwargs(args))
    DEN.storeNdarrayAsDEN(args.output, image[None, :, :], force=True)

    center = (args.size - 1) / 2.0
    x_center = (args.x if args.x is not None else center) + args.x_offset
    y_center = (args.y if args.y is not None else center) + args.y_offset
    print(f"Created phantom: {args.output}")
    print(f"  Grid: {args.size}x{args.size}, image {image.shape[1]}x{image.shape[0]}")
    print(f"  Radius: {args.radius}")
    print(f"  Center: ({x_center:.2f}, {y_center:.2f})")
    print(f"  Antialias: {args.antialias_factor}x" if args.antialias else "  Antialias: off")


if __name__ == "__main__":
    main()
