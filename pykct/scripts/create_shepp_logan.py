#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a Shepp-Logan phantom.

Usage:
    python -m pykct.scripts.create_shepp_logan shepp_logan.den
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from denpy import DEN
from denpy import PHANTOM


def main():
    parser = argparse.ArgumentParser(
        description="Create a Shepp-Logan phantom as a single-slice 3D DEN."
    )
    
    parser.add_argument("output",
                        help="Output DEN file (required)")
    
    parser.add_argument("-n", "--size", type=int, default=256,
                        help="Image width and height (default: 256)")
    
    parser.add_argument("--force", action="store_true",
                        help="Overwrite an existing output file")
    
    args = parser.parse_args()
    
    if args.size < 1:
        parser.error("--size must be positive")
    
    # Use denpy's phantom specification
    image = PHANTOM.phantom2d(n=args.size).astype(np.float32)
    volume = image[np.newaxis, :, :]  # NumPy order: (z, y, x)
    
    # Check if output file exists
    if os.path.exists(args.output) and not args.force:
        print(f"Error: File '{args.output}' already exists.")
        print(f"Use --force to overwrite.")
        sys.exit(1)
    
    DEN.storeNdarrayAsDEN(args.output, volume, force=args.force)
    
    print(f"Created Shepp-Logan phantom: {args.output}")
    print(f"  Size: {args.size}x{args.size}")
    print(f"  Range: [{image.min():.2f}, {image.max():.2f}]")


if __name__ == "__main__":
    main()
