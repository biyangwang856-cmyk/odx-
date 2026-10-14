"""Read-only broad static evidence on a saved generic observation window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from model_engine.ogs2 import Config, cameras_from_checkpoint, project_view, save_json, sha256, surface_points
from model_engine.ogs2_deghost import mask_in_window
from model_engine.ogs2_joint_geometry import independent_static_evidence
from model_engine.ogs2_temporal import TemporalConfig, associate_observations
from rasterio.windows import Window
from scipy import ndimage


def load_saved(project, cache, surface_path, context_pixels=0):
    report = json.loads((cache / "report.json").read_text())
    metadata = json.loads((cache / "observations.json").read_text())
    for path, value in report["inputHashes"].items():
        if sha256(Path(path)) != value:
            raise ValueError("Preserved input no longer matches observation cache")
    if sha256(Path(__file__).parents[2] / "model_engine/ogs2.py") != report["projectionSha256"]:
        raise ValueError("Projection version changed from observation cache")
    if metadata["cacheKey"]["inputHashes"] != report["inputHashes"]:
        raise ValueError("Observation input provenance differs from result")
    with np.load(cache / "observation-masks.npz", allow_pickle=False) as masks:
        observations = [
            {
                **record,
                "mask": masks[f"mask_{index}"].astype(bool),
                "appearance": np.asarray(record["appearance"], np.float32),
            }
            for index, record in enumerate(metadata["observations"])
        ]
    surface_info = json.loads(surface_path.with_suffix(".json").read_text())
    if surface_info != report["surface"]:
        raise ValueError("Surface provenance differs")
    with rio.open(surface_path) as ref:
        surface, surface_transform = ref.read(1), ref.transform
    with rio.open(cache / "ogs2_orthophoto.tif") as ref:
        transform, width, height, crs = ref.transform, ref.width, ref.height, ref.crs
        grid = report["grid"]
        if str(crs) != grid["crs"] or (width, height) != (grid["width"], grid["height"]):
            raise ValueError("Cached result CRS/size differs from saved observation grid")
        if not np.allclose(list(transform), grid["transform"], rtol=0, atol=1e-10):
            raise ValueError("Cached result transform differs from saved observation grid")
    with rio.open(project / "odm_orthophoto/odm_orthophoto.tif") as ref:
        if ref.crs != crs or not np.allclose(
            [ref.transform.a, ref.transform.b, ref.transform.d, ref.transform.e],
            [transform.a, transform.b, transform.d, transform.e],
            rtol=0,
            atol=1e-10,
        ):
            raise ValueError("Observation cache uses a different baseline CRS/pixel frame")
        col, row = ~ref.transform * (transform.c, transform.f)
        if max(abs(col - round(col)), abs(row - round(row))) > 1e-6:
            raise ValueError("Cached result is not on baseline grid")
        old_col, old_row = round(col), round(row)
        left, top = max(0, old_col - context_pixels), max(0, old_row - context_pixels)
        right, bottom = (
            min(ref.width, old_col + width + context_pixels),
            min(ref.height, old_row + height + context_pixels),
        )
        window = Window(left, top, right - left, bottom - top)
        transform = ref.window_transform(window)
        width, height = int(window.width), int(window.height)
        for record in observations:
            record["origin"] = [record["origin"][0] + old_col - left, record["origin"][1] + old_row - top]
        baseline = ref.read(window=window).transpose(1, 2, 0)
        profile = {**ref.profile, "width": width, "height": height, "transform": transform}
    factory, data = cameras_from_checkpoint(project, crs)
    report["grid"] = {"crs": str(crs), "transform": list(transform), "width": width, "height": height}
    return observations, surface, surface_transform, transform, baseline, profile, factory, data, report


def cached_projection(project, name, factory, points, surface, surface_transform, directory, input_key):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + ".npz")
    metadata = path.with_suffix(".json")
    key = {**input_key, "sourceImageSha256": sha256(project / "opensfm/undistorted/images" / name)}
    if path.exists():
        if json.loads(metadata.read_text()) != key:
            raise ValueError("Projection cache provenance differs")
        with np.load(path, allow_pickle=False) as saved:
            return saved["image"], saved["valid"].astype(bool), saved["cost"], path
    result = project_view(
        project, name, factory.get(name), points, surface, surface_transform, None, Config(cpu_threads=8)
    )
    if result is None:
        return None
    image, valid, cost, _ = result
    np.savez_compressed(path, image=image, valid=valid, cost=cost)
    save_json(metadata, key)
    return image, valid, cost, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-cache", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(False)
    obs, surface, st, transform, baseline, profile, factory, data, old_report = load_saved(
        args.project, args.cache, args.surface
    )
    groups, association = associate_observations(
        obs, abs(transform.a), TemporalConfig(require_explicit_clock=True)
    )
    points = surface_points(surface, st, Window(0, 0, profile["width"], profile["height"]), transform)
    all_vehicle = np.zeros(points.shape[1:], bool)
    for record in obs:
        all_vehicle |= mask_in_window(record, (0, 0), all_vehicle.shape)
    z = points[2]
    projection_key = {
        "inputHashes": old_report["inputHashes"],
        "projectionSha256": old_report["projectionSha256"],
        "surfaceSha256": sha256(args.surface),
        "grid": old_report["grid"],
    }
    records = []
    for number, group in enumerate(groups, 1):
        bodies = np.zeros(z.shape, bool)
        for index in group:
            bodies |= mask_in_window(obs[index], (0, 0), z.shape)
        height = float(np.nanmedian(z[bodies]))
        ground = np.isfinite(z) & (np.abs(z - height) < 0.75)
        static = ground & (baseline[..., 3] > 0) & ~ndimage.binary_dilation(all_vehicle, iterations=40)
        names = sorted({obs[index]["sourceImage"] for index in group})
        frames = {}
        evidence = {
            "group": number,
            "observations": group,
            "names": names,
            "pairEvidence": [],
            "baselineEvidence": [],
        }
        for name in names:
            result = cached_projection(
                args.project,
                name,
                factory,
                points,
                surface,
                st,
                args.projection_cache or args.output / "sources",
                projection_key,
            )
            if result:
                frames[name] = result
                image, valid, _, _ = result
                comparison = independent_static_evidence(baseline[..., :3], image, static & valid)
                evidence["baselineEvidence"].append({"sourceImage": name, **comparison})
        for index, name in enumerate(names):
            if name not in frames:
                continue
            for other in names[index + 1 :]:
                if other not in frames:
                    continue
                a, va, _, _ = frames[name]
                b, vb, _, _ = frames[other]
                comparison = independent_static_evidence(a, b, static & va & vb)
                evidence["pairEvidence"].append({"first": name, "second": other, **comparison})
        records.append(evidence)
        save_json(args.output / "static-evidence.json", records)
        print(
            json.dumps(
                {
                    "group": number,
                    "sources": len(frames),
                    "pairs": [
                        {k: v for k, v in e.items() if k != "correspondences"}
                        for e in evidence["pairEvidence"]
                    ],
                    "baseline": [
                        {k: v for k, v in e.items() if k != "correspondences"}
                        for e in evidence["baselineEvidence"]
                    ],
                }
            ),
            flush=True,
        )
    save_json(args.output / "association.json", association)


if __name__ == "__main__":
    main()
