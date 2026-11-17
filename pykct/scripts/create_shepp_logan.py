#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a Shepp-Logan phantom.

Usage:
    python -m pykct.scripts.create_shepp_logan shepp_logan.den
"""

import argparse

from denpy import DEN

from pykct.phantoms import SHEPP_LOGAN_VARIANTS, create_shepp_logan_phantom
from pykct.scripts._common import check_output


def main():
    parser = argparse.ArgumentParser(
        description="Create a Shepp-Logan phantom as a single-slice 3D DEN."
    )
    parser.add_argument("output", help="Output DEN file (required)")
    parser.add_argument("-n", "--size", type=int, default=256,
                        help="Image width and height (default: 256)")
    parser.add_argument("--phantom", choices=SHEPP_LOGAN_VARIANTS, default="SheppLogan2D",
                        help="Phantom variant (default: SheppLogan2D)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output file")
    args = parser.parse_args()

    if args.size < 1:
        parser.error("--size must be positive")
    check_output(args.output, args.force)

    image = create_shepp_logan_phantom(size=args.size, phantom=args.phantom)
    volume = image[None, :, :]  # NumPy order: (z, y, x)
    DEN.storeNdarrayAsDEN(args.output, volume, force=True)

    print(f"Created Shepp-Logan phantom: {args.output}")
    print(f"  Variant: {args.phantom}")
    print(f"  Size: {args.size}x{args.size}")
    print(f"  Range: [{image.min():.2f}, {image.max():.2f}]")


if __name__ == "__main__":
    main()
