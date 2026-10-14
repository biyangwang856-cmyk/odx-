"""Read-only old-mosaic vehicle observations with overlapping real-pixel cuts.

Observations are conservative boundary evidence, not unique identity or complete
shadow labels. The detector and local model weights are exactly the saved OGS2
ones. No reconstruction, reprojection, compositing or generated image is used.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from hashlib import sha256 as new_hash
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from model_engine.ogs2 import Config, Detector, save_json, sha256
from model_engine.ogs2_deghost import appearance_descriptor
from PIL import Image, ImageDraw, ImageFont
from rasterio.windows import Window


def integer_offset(reference, target):
    if reference.crs != target.crs or not np.allclose(
        [reference.transform.a, reference.transform.b, reference.transform.d, reference.transform.e],
        [target.transform.a, target.transform.b, target.transform.d, target.transform.e],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Cached grid and old mosaic have different map pixels")
    x, y = ~reference.transform * (target.transform.c, target.transform.f)
    if max(abs(x - round(x)), abs(y - round(y))) > 1e-6:
        raise ValueError("Cached grid has fractional old-mosaic pixel offset")
    return round(x), round(y)


def positions(length, maximum):
    if length <= maximum:
        return [0]
    stride = maximum - 480
    if stride <= 0:
        raise ValueError("Detection cuts require at least 480px overlap")
    return sorted(set([*range(0, length - maximum + 1, stride), length - maximum]))


def gpu_guard(physical_gpu):
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={physical_gpu}",
            "--query-gpu=memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    free, usage = [int(value.strip()) for value in result.stdout.strip().split(",")]
    if free < 22 * 1024 or usage != 0:
        raise RuntimeError(f"Requested GPU is no longer idle with 22GiB free: free={free}MiB, util={usage}%")
    return {"physicalGpu": physical_gpu, "freeMiB": free, "utilizationPercent": usage}


def pixels_hash(array):
    value = new_hash()
    value.update(json.dumps({"shape": array.shape, "dtype": str(array.dtype)}).encode())
    value.update(array.tobytes(order="C"))
    return value.hexdigest()


def show_masks(rgba, records, path):
    canvas = rgba[..., :3].copy()
    from scipy import ndimage

    for record in records:
        mask = record["mask"]
        left, top = record["expandedOrigin"]
        target = canvas[top : top + mask.shape[0], left : left + mask.shape[1]]
        edge = mask & ~ndimage.binary_erosion(mask)
        target[edge] = [255, 255, 0] if record["complete"] else [255, 0, 0]
    view = Image.fromarray(canvas)
    draw = ImageDraw.Draw(view)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    for index, record in enumerate(records):
        left, top = record["expandedOrigin"]
        draw.text((left, top), str(index), fill="white", stroke_width=1, stroke_fill="black", font=font)
    view.save(path, quality=95)


def collect_window(detector, old, cached_path, output, context, cut_size, baseline_hash, provenance):
    output.mkdir(parents=True, exist_ok=False)
    with rio.open(cached_path) as cached:
        origin = integer_offset(old, cached)
        core = Window(*origin, cached.width, cached.height)
        expanded = Window(
            core.col_off - context,
            core.row_off - context,
            core.width + 2 * context,
            core.height + 2 * context,
        ).intersection(Window(0, 0, old.width, old.height))
        expanded = expanded.round_offsets().round_lengths()
        rgba = old.read([1, 2, 3, 4], window=expanded).transpose(1, 2, 0)
        transform = old.window_transform(expanded)
        grid = {
            "crs": str(old.crs),
            "transform": list(transform),
            "width": int(expanded.width),
            "height": int(expanded.height),
            "expandedWindowOnOldMosaic": list(expanded.flatten()),
            "cachedWindowOnOldMosaic": list(core.flatten()),
            "cachedResult": str(cached_path),
            "cachedResultSha256": sha256(cached_path),
            "originConvention": (
                "origin is in cached result pixels; expandedOrigin is in expanded crop pixels"
            ),
        }
    records, cuts = [], []
    for row in positions(rgba.shape[0], cut_size):
        for col in positions(rgba.shape[1], cut_size):
            cut = rgba[row : row + cut_size, col : col + cut_size]
            cut_id = len(cuts)
            found = detector.masks(cut[..., :3])
            cuts.append(
                {
                    "id": cut_id,
                    "windowOnExpandedCrop": [col, row, cut.shape[1], cut.shape[0]],
                    "rgbaPixelSha256": pixels_hash(cut),
                    "detections": len(found),
                }
            )
            for item in found:
                ys, xs = np.where(item["mask"])
                x0, y0 = int(xs.min()), int(ys.min())
                x1, y1 = int(xs.max() + 1), int(ys.max() + 1)
                left, top = col + x0, row + y0
                global_left, global_top = int(expanded.col_off) + left, int(expanded.row_off) + top
                visible = bool((cut[..., 3][item["mask"]] > 0).all())
                records.append(
                    {
                        "kind": "mosaic",
                        "view": 0,
                        "sourceImage": "__old_ogs2_mosaic__",
                        "capture_time": None,
                        "clock": "unknown-mosaic-multiple-exposures",
                        "origin": [global_left - origin[0], global_top - origin[1]],
                        "expandedOrigin": [left, top],
                        "globalPixelOrigin": [global_left, global_top],
                        "mask": item["mask"][y0:y1, x0:x1].copy(),
                        "complete": bool(item["complete"] and visible),
                        "confidence": item["samConfidence"],
                        "samConfidence": item["samConfidence"],
                        "detectionConfidence": item["confidence"],
                        "alphaFullyValid": visible,
                        "appearance": appearance_descriptor(cut[..., :3], item["mask"]),
                        "cutInputId": cut_id,
                        "bboxOnCut": item["box"].tolist(),
                    }
                )
            save_json(
                output / "status.json",
                {
                    "state": "collecting",
                    "cuts": len(cuts),
                    "observations": len(records),
                },
            )
            print(output.name, "cut", cut_id, "observations", len(records), flush=True)
    # Keep overlap duplicates: dropping one mask could discard boundary support.
    # Their cut provenance makes independent manual reconciliation possible.
    packed = {f"mask_{index}": record["mask"].astype(np.uint8) for index, record in enumerate(records)}
    np.savez_compressed(output / "observation-masks.npz", **packed)
    metadata = [
        {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in record.items()
            if key != "mask"
        }
        for record in records
    ]
    save_json(
        output / "observations.json",
        {
            "schemaVersion": "old-ogs2-mosaic-observations/v1",
            "baselinePath": str(Path(old.name)),
            "baselineSha256": baseline_hash,
            "grid": grid,
            "detector": provenance,
            "cutInputs": cuts,
            "observations": metadata,
            "maskArchive": str(output / "observation-masks.npz"),
            "limitations": [
                "SAM completeness is only mask/crop visibility evidence, not semantic vehicle completeness.",
                "Overlapping cuts may repeat the same vehicle; every observed support is preserved.",
                "The mixed-time mosaic observations cannot identify a physical track.",
                "These masks do not include guaranteed complete vehicle shadows.",
            ],
        },
    )
    Image.fromarray(rgba[..., :3]).save(output / "old-ogs2-context.jpg", quality=95)
    show_masks(rgba, records, output / "mosaic-mask-evidence.jpg")
    save_json(
        output / "status.json",
        {
            "state": "success",
            "cuts": len(cuts),
            "observations": len(records),
            "completeObservations": sum(record["complete"] for record in records),
            "incompleteObservations": sum(not record["complete"] for record in records),
        },
    )
    return {"window": output.name, "observations": len(records), "grid": grid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "models", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path, action="append", required=True,
        help="New-scene bounded deghost cache directory; repeat for multiple windows",
    )
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--context-pixels", type=int, default=240)
    parser.add_argument("--cut-size", type=int, default=1024)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh evidence directory; existing inputs/results are never overwritten")
    if args.context_pixels < 240:
        raise ValueError("This evidence collection requires at least 240px surrounding context")
    caches = [path.resolve() for path in args.cache]
    names = [path.name for path in caches]
    if len(set(names)) != len(names):
        raise ValueError("Cache directory basenames must be distinct for separate evidence outputs")
    for cache in caches:
        if not (cache / "ogs2_orthophoto.tif").is_file() or not (cache / "report.json").is_file():
            raise ValueError("Provide completed new-scene deghost caches, not scan-only directories")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise ValueError("CUDA_VISIBLE_DEVICES must expose only the explicitly guarded physical GPU")
    config = Config(cpu_threads=8)
    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(False)
    baseline_hash = sha256(args.baseline)
    model_hashes = {
        str(path.relative_to(args.models)): sha256(path)
        for path in sorted(args.models.rglob("*"))
        if path.is_file() and path.suffix in {".json", ".pt", ".pth", ".safetensors", ".bin"}
    }
    import model_engine.ogs2 as detector_module

    provenance = {
        "detectorEngineSha256": sha256(Path(detector_module.__file__)),
        "scriptSha256": sha256(Path(__file__)),
        "modelFiles": model_hashes,
        "config": asdict(config),
        "gpuGuard": gpu_guard(args.physical_gpu),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    detector = Detector(args.models, "cuda:0", config)
    started = time.monotonic()
    results = []
    with rio.open(args.baseline) as old:
        for cache in caches:
            name = cache.name
            cached = cache / "ogs2_orthophoto.tif"
            results.append(
                collect_window(
                    detector,
                    old,
                    cached,
                    args.output / name,
                    args.context_pixels,
                    args.cut_size,
                    baseline_hash,
                    provenance,
                )
            )
    if sha256(args.baseline) != baseline_hash:
        raise RuntimeError("Old OGS2 input changed during read-only evidence collection")
    save_json(
        args.output / "collection-report.json",
        {
            "schemaVersion": "old-ogs2-context-collection/v1",
            "baselineSha256": baseline_hash,
            "inputUnchanged": True,
            "elapsedSeconds": time.monotonic() - started,
            "windows": results,
            "detector": provenance,
        },
    )
    print(json.dumps({"output": str(args.output), "windows": results}), flush=True)


if __name__ == "__main__":
    main()
