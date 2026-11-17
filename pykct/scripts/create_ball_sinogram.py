#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create sinogram from ball phantom.

Usage:
    python -m pykct.scripts.create_ball_sinogram ball_sino.den --radius 10 --cor-offset 5.0
"""

import argparse

from pykct.phantoms import create_ball_sinogram
from pykct.scripts._common import (add_projection_arguments, check_output,
                                   projection_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Create sinogram from ball phantom.")
    parser.add_argument("sinogram_file", help="Output sinogram DEN file (required)")

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
    add_projection_arguments(parser)
    args = parser.parse_args()

    check_output(args.sinogram_file, args.force)
    kwargs = projection_kwargs(args)

    sinogram = create_ball_sinogram(
        size=args.size,
        radius=args.radius,
        x_offset=args.x_offset,
        y_offset=args.y_offset,
        antialias=args.antialias,
        antialias_factor=args.antialias_factor,
        output_file=args.sinogram_file,
        **kwargs,
    )

    print(f"Created sinogram: {args.sinogram_file}")
    print(f"  Phantom: size={args.size}, radius={args.radius}")
    print(f"  Projection: {args.n_angles} angles, detector={args.detector_size}")
    print(f"  CoR offset: {kwargs['cor_offset']:+.2f} px")
    print(f"  Sinogram shape: {sinogram.shape}")


if __name__ == "__main__":
    main()
