#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create sinogram from Shepp-Logan phantom.

Usage:
    python -m pykct.scripts.create_shepp_logan_sinogram sl_sino.den --cor-offset 5.0
"""

import argparse

from pykct.phantoms import SHEPP_LOGAN_VARIANTS, create_shepp_logan_sinogram
from pykct.scripts._common import (add_projection_arguments, check_output,
                                   projection_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Create sinogram from Shepp-Logan phantom.")
    parser.add_argument("sinogram_file", help="Output sinogram DEN file (required)")

    phantom_group = parser.add_argument_group("Phantom parameters")
    phantom_group.add_argument("-n", "--size", type=int, default=256,
                               help="Phantom size (default: 256)")
    phantom_group.add_argument("--phantom", choices=SHEPP_LOGAN_VARIANTS,
                               default="SheppLogan2D",
                               help="Phantom variant (default: SheppLogan2D)")
    add_projection_arguments(parser)
    args = parser.parse_args()

    check_output(args.sinogram_file, args.force)
    kwargs = projection_kwargs(args)

    sinogram = create_shepp_logan_sinogram(
        size=args.size,
        phantom=args.phantom,
        output_file=args.sinogram_file,
        **kwargs,
    )

    print(f"Created sinogram: {args.sinogram_file}")
    print(f"  Phantom: {args.phantom}, size={args.size}")
    print(f"  Projection: {args.n_angles} angles, detector={args.detector_size}")
    print(f"  CoR offset: {kwargs['cor_offset']:+.2f} px")
    print(f"  Sinogram shape: {sinogram.shape}")


if __name__ == "__main__":
    main()
