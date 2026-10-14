#!/usr/bin/env python3
"""Compute reproducible pixel metrics for two same-camera viewer screenshots."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--crop", help="x0,y0,x1,y1")
    parser.add_argument("--threshold", type=int, default=8)
    parser.add_argument("--black-threshold", type=int, default=12)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = np.asarray(Image.open(args.left).convert("RGB"), dtype=np.int16)
    right = np.asarray(Image.open(args.right).convert("RGB"), dtype=np.int16)
    if left.shape != right.shape:
        raise ValueError(f"screenshot dimensions differ: {left.shape} vs {right.shape}")
    height, width, _ = left.shape
    crop = [0, 0, width, height]
    if args.crop:
        crop = [int(value) for value in args.crop.split(",")]
        if len(crop) != 4:
            raise ValueError("--crop must contain x0,y0,x1,y1")
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"crop outside image: {crop}, image={width}x{height}")
    a = left[y0:y1, x0:x1]
    b = right[y0:y1, x0:x1]
    absolute = np.abs(a - b)
    rmse = float(np.sqrt(np.square(a.astype(np.float64) - b).mean()))
    changed = absolute.max(axis=2) > args.threshold
    left_black = left.max(axis=2) <= args.black_threshold
    right_black = right.max(axis=2) <= args.black_threshold
    result = {
        "left": str(args.left),
        "right": str(args.right),
        "imageSize": [width, height],
        "crop": crop,
        "threshold": args.threshold,
        "meanAbsoluteError": float(absolute.mean()),
        "rmse": rmse,
        "psnrDb": None if rmse == 0 else 20 * math.log10(255 / rmse),
        "changedPixels": int(changed.sum()),
        "changedPixelFraction": float(changed.mean()),
        "fullImageBlackPixelFraction": {
            "left": float(left_black.mean()),
            "right": float(right_black.mean()),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
