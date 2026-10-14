"""Collect real projected photo, road and old-mosaic evidence; no image editing.

The photo names are run inputs, not scene-specific algorithm constants. Road
masks identify candidate surface only; their confidence is not vehicle truth.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from model_engine.ogs2 import Config, Detector, save_json, sha256, surface_points
from PIL import Image
from rasterio.windows import Window

from experiments.tools.collect_mosaic_vehicles import collect_window
from experiments.tools.inspect_joint_static import cached_projection, load_saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "cache", "surface", "base-orthophoto", "models", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--context-pixels", type=int, default=480)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.context_pixels < 240 or len(set(args.images)) < 2:
        raise ValueError("Require surrounding context and at least two independent exposures")
    if args.output.exists():
        raise FileExistsError("Use a fresh evidence directory")
    args.output.mkdir(parents=True)
    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(False)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    os.nice(10)
    _, surface, st, transform, _, profile, factory, data, report = load_saved(
        args.project, args.cache, args.surface, args.context_pixels
    )
    for name in args.images:
        if name not in data["shots"] or Path(name).name != name:
            raise ValueError("Photo must be a calibrated local checkpoint shot: " + name)
    source_key = {
        "inputHashes": report["inputHashes"],
        "projectionSha256": report["projectionSha256"],
        "surfaceSha256": sha256(args.surface),
        "grid": report["grid"],
    }
    points = surface_points(surface, st, Window(0, 0, profile["width"], profile["height"]), transform)
    provenance = {
        "scriptSha256": sha256(Path(__file__)),
        "detectorEngineSha256": sha256(Path(__file__).parents[2] / "model_engine/ogs2.py"),
        "modelFiles": {
            str(p.relative_to(args.models)): sha256(p)
            for p in sorted(args.models.rglob("*"))
            if p.is_file() and p.suffix in {".json", ".pt", ".safetensors", ".bin"}
        },
        "device": args.device,
        "gpuSnapshot": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,utilization.gpu", "--format=csv,noheader"],
            text=True,
        ),
    }
    detector = Detector(args.models, args.device, Config(cpu_threads=8))
    records = []
    for name in args.images:
        result = cached_projection(
            args.project, name, factory, points, surface, st, args.output / "sources", source_key
        )
        if result is None:
            raise ValueError("Projection has no valid source pixels: " + name)
        image, valid, _, path = result
        inputs = detector.processor(
            images=Image.fromarray(image), text="road. paved road.", return_tensors="pt"
        ).to(detector.device)
        with detector.torch.inference_mode():
            predictions = detector.model(**inputs)
            detections = detector.processor.post_process_grounded_object_detection(
                predictions,
                inputs.input_ids,
                box_threshold=0.25,
                text_threshold=0.2,
                target_sizes=[image.shape[:2]],
            )[0]
            boxes = detections["boxes"].cpu().numpy()
            detector.sam.set_image(image)
            if len(boxes):
                masks, scores, _ = detector.sam.predict(box=boxes, multimask_output=False)
                masks = masks.reshape((-1, *image.shape[:2]))
            else:
                masks, scores = [], []
        road = np.zeros(valid.shape, bool)
        roads = []
        for mask, score, box in zip(masks, np.ravel(scores), boxes):
            if score >= 0.8 and mask.sum() >= 1000:
                road |= mask.astype(bool) & valid
                roads.append({"box": box.tolist(), "samConfidence": float(score)})
        # Overlapping original-resolution cuts avoid losing small cars when
        # SAM resizes a wide map. Keep duplicate supports, not identity counts.
        vehicles = []
        from experiments.tools.collect_mosaic_vehicles import positions

        for y in positions(image.shape[0], 1024):
            for x in positions(image.shape[1], 1024):
                for vehicle in detector.masks(image[y : y + 1024, x : x + 1024]):
                    mask = np.zeros(valid.shape, bool)
                    mask[y : y + 1024, x : x + 1024] = vehicle.pop("mask")
                    vehicles.append(
                        {
                            **vehicle,
                            "mask": mask,
                            "validComplete": bool(vehicle["complete"] and valid[mask].all()),
                        }
                    )
        np.savez_compressed(
            args.output / (name + ".npz"),
            road=road,
            **{f"vehicle_{i}": v["mask"] for i, v in enumerate(vehicles)},
        )
        canvas = image.copy()
        for mask, color in [(road, (255, 0, 255)), *[(v["mask"], (255, 255, 0)) for v in vehicles]]:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, contours, -1, color, 1)
        Image.fromarray(canvas).save(args.output / (name + ".jpg"), quality=95)
        archive = args.output / (name + ".npz")
        records.append(
            {
                "sourceImage": name,
                "projectionCache": str(path.resolve()),
                "projectionCacheSha256": sha256(path),
                "maskArchiveSha256": sha256(archive),
                "grid": report["grid"],
                "roadPixels": int(road.sum()),
                "roadObservations": roads,
                "vehicles": [
                    {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in o.items() if k != "mask"}
                    for o in vehicles
                ],
            }
        )
        save_json(args.output / "report.json", records)
        print(name, "road pixels", int(road.sum()), "vehicle observations", len(vehicles), flush=True)
    base_hash = sha256(args.base_orthophoto)
    with rio.open(args.base_orthophoto) as old:
        collect_window(
            detector,
            old,
            args.cache / "ogs2_orthophoto.tif",
            args.output / "base-observations",
            args.context_pixels,
            1024,
            base_hash,
            provenance,
        )
    if sha256(args.base_orthophoto) != base_hash:
        raise RuntimeError("Retained mosaic changed")
    save_json(
        args.output / "collection-report.json",
        {
            "provenance": provenance,
            "grid": report["grid"],
            "contextPixels": args.context_pixels,
            "baselineSha256": base_hash,
            "inputUnchanged": True,
        },
    )
    save_json(args.output / "status.json", {"state": "success"})


if __name__ == "__main__":
    main()
