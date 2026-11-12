
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Calculate intensity-weighted centroid displacement from the geometrical
center of each DEN frame.

Usage:
    python -m pykct.scripts.den_centroid data/sinogram.den
"""

import argparse
import os
import numpy as np
from denpy import DEN


def calculate_centroid(frame):
    """Calculate intensity-weighted centroid of a 2D frame."""
    frame = np.asarray(frame, dtype=np.float64)

    total_mass = np.sum(frame)

    if np.isclose(total_mass, 0):
        return None

    yy, xx = np.indices(frame.shape)

    x_centroid = np.sum(xx * frame) / total_mass
    y_centroid = np.sum(yy * frame) / total_mass

    return x_centroid, y_centroid


def main():
    parser = argparse.ArgumentParser(
        description="Calculate centroid displacement from the geometrical center."
    )

    parser.add_argument("input", help="Input DEN file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Ignore pixels with intensity below this threshold"
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output file"
    )

    args = parser.parse_args()

    # Read DEN header and data
    info = DEN.readHeader(args.input)
    data = DEN.getNumpyArray(args.input)

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    if data.ndim != 3:
        parser.error("Expected a 2D or 3D DEN dataset.")

    n_frames, height, width = data.shape

    # Geometrical center of pixel grid
    x_center = (width - 1) / 2.0
    y_center = (height - 1) / 2.0

    # First output line
    filename = os.path.basename(args.input)
    print(f"{filename} : Dimensions: {info['dimspec']} | "
        f"NumPy shape: {info['shape']} | "
        f"Dimension count: {info['dimcount']}"
    )

    print(
        f"Offsets relative to geometric center: "
        f"X = {x_center:.6f}, Y = {y_center:.6f}"
    )

    print(
        f"{'Frame':>8} {'Centroid X':>14} {'Centroid Y':>14} "
        f"{'Offset X':>14} {'Offset Y':>14} {'Distance':>14}"
    )

    results = []

    for i in range(n_frames):
        frame = DEN.getFrame(args.input, i)

        if args.threshold is not None:
            frame = np.where(frame >= args.threshold, frame, 0.0)

        centroid = calculate_centroid(frame)

        if centroid is None:
            print(f"{i:8d} {'Empty frame':>14}")
            continue

        cx, cy = centroid

        dx = cx - x_center
        dy = cy - y_center

        distance = np.sqrt(dx**2 + dy**2)

        print(
            f"{i:8d} {cx:14.6f} {cy:14.6f} "
            f"{dx:14.6f} {dy:14.6f} {distance:14.6f}"
        )

        results.append((i, cx, cy, dx, dy, distance))

    if args.csv:
        header = "frame,centroid_x,centroid_y,offset_x,offset_y,distance"

        np.savetxt(
            args.csv,
            np.asarray(results).reshape(-1, 6),
            delimiter=",",
            header=header,
            comments="",
            fmt=["%d"] + ["%.8f"] * 5
        )

        print(f"Results saved to: {args.csv}")
    print("")

    # Ensure exactly one final newline, without extra blank lines.
    # All preceding print statements produce single-line output.


if __name__ == "__main__":
    main()
