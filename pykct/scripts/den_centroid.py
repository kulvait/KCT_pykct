#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate intensity-weighted centroid displacement from the geometrical
center of each DEN frame.

For a 360 degree sinogram frame (angles x detector) Offset X equals
cor_offset in the pykct convention (rotation axis at (N-1)/2 + cor_offset).

Usage:
    python -m pykct.scripts.den_centroid data/sinogram.den
"""

import argparse
import os

import numpy as np
from denpy import DEN

from pykct.phantoms import calculate_centroid


def main():
    parser = argparse.ArgumentParser(
        description="Calculate centroid displacement from the geometrical center."
    )
    parser.add_argument("input", help="Input DEN file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Ignore pixels with intensity below this threshold")
    parser.add_argument("--csv", default=None, help="Optional CSV output file")
    args = parser.parse_args()

    info = DEN.readHeader(args.input)
    shape = tuple(info["shape"])
    if len(shape) == 2:
        n_frames, (height, width) = 1, shape
    elif len(shape) == 3:
        n_frames, height, width = shape
    else:
        parser.error("Expected a 2D or 3D DEN dataset.")

    # Geometrical center of pixel grid
    x_center = (width - 1) / 2.0
    y_center = (height - 1) / 2.0

    filename = os.path.basename(args.input)
    print(f"{filename} : Dimensions: {info['dimspec']} | "
          f"NumPy shape: {info['shape']} | "
          f"Dimension count: {info['dimcount']}")
    print(f"Offsets relative to geometric center: "
          f"X = {x_center:.6f}, Y = {y_center:.6f}")
    print(f"{'Frame':>8} {'Centroid X':>14} {'Centroid Y':>14} "
          f"{'Offset X':>14} {'Offset Y':>14} {'Distance':>14}")

    results = []
    for i in range(n_frames):
        centroid = calculate_centroid(DEN.getFrame(args.input, i), threshold=args.threshold)
        if centroid is None:
            print(f"{i:8d} {'Empty frame':>14}")
            continue
        cx, cy = centroid
        dx = cx - x_center
        dy = cy - y_center
        distance = np.hypot(dx, dy)
        print(f"{i:8d} {cx:14.6f} {cy:14.6f} {dx:14.6f} {dy:14.6f} {distance:14.6f}")
        results.append((i, cx, cy, dx, dy, distance))

    if args.csv:
        np.savetxt(
            args.csv,
            np.asarray(results).reshape(-1, 6),
            delimiter=",",
            header="frame,centroid_x,centroid_y,offset_x,offset_y,distance",
            comments="",
            fmt=["%d"] + ["%.8f"] * 5,
        )
        print(f"Results saved to: {args.csv}")
    print("")


if __name__ == "__main__":
    main()
