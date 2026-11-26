#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a Shepp-Logan phantom as a single-slice DEN (1, n, n).

Usage:
    python -m pykct.scripts.create_shepp_logan shepp_logan.den
"""

import argparse

from denpy import DEN

from pykct.phantoms import (SHEPP_LOGAN_VARIANTS, create_legacy_shepp_logan_phantom2d,
                            create_shepp_logan_phantom2d)
from pykct.scripts._common import (add_discretization_arguments, check_output,
                                   discretization_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Create a Shepp-Logan phantom as a single-slice DEN (1, n, n)."
    )
    parser.add_argument("output", help="Output DEN file (required)")
    parser.add_argument("-n", "--size", type=int, default=256,
                        help="Image width and height (default: 256)")
    parser.add_argument("--phantom", choices=SHEPP_LOGAN_VARIANTS + ("legacy",),
                        default="SheppLogan2D",
                        help="Phantom variant (default: SheppLogan2D); legacy is the "
                             "deprecated pykct <= 0.1.0 phantom")
    add_discretization_arguments(parser)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output file")
    args = parser.parse_args()

    if args.size < 1:
        parser.error("--size must be positive")
    check_output(args.output, args.force)

    disc = discretization_kwargs(args)
    if args.phantom == "legacy":
        if disc["antialias"] or disc["supersampling"]:
            parser.error("the legacy phantom supports neither antialiasing nor supersampling")
        image = create_legacy_shepp_logan_phantom2d(size=args.size)
    else:
        image = create_shepp_logan_phantom2d(size=args.size, phantom=args.phantom, **disc)
    DEN.storeNdarrayAsDEN(args.output, image[None, :, :], force=True)

    print(f"Created Shepp-Logan phantom: {args.output}")
    print(f"  Variant: {args.phantom}")
    print(f"  Grid: {args.size}x{args.size}, image {image.shape[1]}x{image.shape[0]}")
    print(f"  Range: [{image.min():.2f}, {image.max():.2f}]")


if __name__ == "__main__":
    main()
