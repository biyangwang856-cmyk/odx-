"""Opt-in temporal conflict patches on an unchanged 2.5D orthophoto.

This module chooses a real photo for a whole short-time conflict region. It
does not synthesize background, move a vehicle, or overwrite the ODX checkpoint.
"""

from __future__ import annotations

import argparse
import calendar
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy import ndimage

from model_engine.ogs2 import (
    Config,
    Detector,
    cameras_from_checkpoint,
    candidate_views,
    export_surface,
    project_view,
    save_json,
    sha256,
    surface_points,
)
from model_engine.ogs2_temporal import TemporalConfig, associate_observations, render_conflict


@dataclass(frozen=True)
class PatchConfig:
    candidates: int = 12
    tile_size: int = 768
    halo: int = 160
    padding_m: float = 0.8
    max_patch_pixels: int = 1_000_000
    max_surface_spread_m: float = 2.0
    static_p90_limit_px: float = 2.0
    min_static_matches: int = 12
    max_static_relative_bad_fraction: float = 0.15
    cpu_threads: int = 8


def capture_time(name: str, shot: dict) -> tuple[float | None, str]:
    """Keep real time intervals; filename fallback is a consistent wall clock."""
    value = shot.get("capture_time")
    if isinstance(value, (float, int)) and np.isfinite(value) and value > 0:
        return float(value), "opensfm-capture-time"
    match = re.search(r"(?:^|_)DJI_?(\d{14})_|^DJI_(\d{14})_", name)
    if match:
        try:
            stamp = next(part for part in match.groups() if part)
            # Encode one common wall clock for intervals without assuming
            # that a DJI filename records its timezone.
            return float(calendar.timegm(datetime.strptime(stamp, "%Y%m%d%H%M%S").timetuple())), (
                "dji-filename-wall-clock"
            )
        except ValueError:
            pass
    return None, "unknown"


def appearance_descriptor(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Small observable descriptor; never an independent identity assertion."""
    pixels = rgb[mask]
    if len(pixels) == 0:
        return np.zeros(36, np.float32)
    pixels = pixels[:: max(1, len(pixels) // 2000)]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hist, _, _ = np.histogram2d(hsv[:, 0], hsv[:, 1], bins=(12, 3), range=((0, 180), (0, 256)))
    flat = hist.ravel().astype(np.float32)
    return flat / max(float(np.linalg.norm(flat)), 1e-8)


def observation_cache(output: Path, expected: dict, build):
    metadata = output / "observations.json"
    masks_file = output / "observation-masks.npz"
    if metadata.exists() and masks_file.exists():
        saved = json.loads(metadata.read_text())
        if saved.get("cacheKey") != expected:
            raise ValueError("Observation cache provenance differs; use a new output directory")
        with np.load(masks_file, allow_pickle=False) as masks:
            records = [
                {
                    **item,
                    "mask": masks[f"mask_{i}"].astype(bool),
                    "appearance": np.asarray(item["appearance"], np.float32),
                }
                for i, item in enumerate(saved["observations"])
            ]
        return records, True
    records = build()
    packed = {f"mask_{i}": o["mask"].astype(np.uint8) for i, o in enumerate(records)}
    temporary = output / "observation-masks.partial.npz"
    np.savez_compressed(temporary, **packed)
    temporary.replace(masks_file)
    serial = [
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in o.items() if k != "mask"}
        for o in records
    ]
    save_json(metadata, {"cacheKey": expected, "observations": serial})
    return records, False


def collect_observations(
    project,
    factory,
    data,
    names,
    surface,
    surface_transform,
    transform,
    width,
    height,
    detector,
    config,
    output,
    baseline,
    baseline_window,
):
    records = []
    source_ids = {name: i + 1 for i, name in enumerate(names)}
    started = time.monotonic()
    tile = 0
    for row in range(0, height, config.tile_size):
        for col in range(0, width, config.tile_size):
            left, top = max(0, col - config.halo), max(0, row - config.halo)
            right, bottom = (
                min(width, col + config.tile_size + config.halo),
                min(height, row + config.tile_size + config.halo),
            )
            ext = Window(left, top, right - left, bottom - top)
            points = surface_points(surface, surface_transform, ext, transform)
            with rio.open(baseline) as ref:
                old = ref.read(
                    [1, 2, 3],
                    window=Window(
                        baseline_window.col_off + left,
                        baseline_window.row_off + top,
                        ext.width,
                        ext.height,
                    ),
                ).transpose(1, 2, 0)
            for item in detector.masks(old):
                ys, xs = np.where(item["mask"])
                x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
                records.append(
                    {
                        "kind": "baseline",
                        "view": 0,
                        "sourceImage": "__baseline__",
                        "capture_time": None,
                        "clock": "unknown",
                        "origin": (left + x0, top + y0),
                        "mask": item["mask"][y0:y1, x0:x1].copy(),
                        "complete": item["complete"],
                        "confidence": item["samConfidence"],
                        "samConfidence": item["samConfidence"],
                        "appearance": appearance_descriptor(old, item["mask"]),
                    }
                )
            center = np.nanmedian(points.reshape(3, -1), axis=1)
            chosen = (
                candidate_views(factory, data, center, config.candidates) if np.isfinite(center).all() else []
            )
            for name in chosen:
                stamp, clock = capture_time(name, data["shots"][name])
                if stamp is None:
                    continue
                projected = project_view(
                    project,
                    name,
                    factory.get(name),
                    points,
                    surface,
                    surface_transform,
                    detector,
                    Config(cpu_threads=config.cpu_threads),
                )
                if projected is None:
                    continue
                rgb, valid, _, observations = projected
                for item in observations:
                    mask = item["mask"]
                    ys, xs = np.where(mask)
                    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
                    records.append(
                        {
                            "kind": "photo",
                            "view": source_ids[name],
                            "sourceImage": name,
                            "capture_time": stamp,
                            "clock": clock,
                            "origin": (left + x0, top + y0),
                            "mask": mask[y0:y1, x0:x1].copy(),
                            "complete": item["complete"],
                            "confidence": item["samConfidence"],
                            "detectionConfidence": item["confidence"],
                            "samConfidence": item["samConfidence"],
                            "appearance": appearance_descriptor(rgb, mask),
                            "observedVisibleFraction": float(valid[mask].mean()),
                        }
                    )
            tile += 1
            save_json(
                output / "status.json",
                {
                    "state": "collecting-observations",
                    "tiles": tile,
                    "observations": len(records),
                    "elapsedSeconds": time.monotonic() - started,
                },
            )
            print("scan", tile, "observations", len(records), flush=True)
    return records


def mask_in_window(observation, origin, shape):
    out = np.zeros(shape, bool)
    x0, y0 = observation["origin"]
    x1, y1 = x0 + observation["mask"].shape[1], y0 + observation["mask"].shape[0]
    left, top = max(x0, origin[0]), max(y0, origin[1])
    right, bottom = min(x1, origin[0] + shape[1]), min(y1, origin[1] + shape[0])
    if right > left and bottom > top:
        out[top - origin[1] : bottom - origin[1], left - origin[0] : right - origin[0]] = observation["mask"][
            top - y0 : bottom - y0, left - x0 : right - x0
        ]
    return out


def cuts_observed_vehicle(region, origin, observations):
    """Do not leave part of a reliable vehicle outside a replacement patch."""
    for index, observation in enumerate(observations):
        if not observation["complete"] or observation.get("confidence", 0.0) < 0.8:
            continue
        overlap = int((mask_in_window(observation, origin, region.shape) & region).sum())
        if 0 < overlap < int(observation["mask"].sum()):
            return index
    return None


def static_agreement(baseline, image, static, config):
    """Measure displacement without warping any candidate or output."""
    if static.sum() < 256:
        return {"accepted": False, "reason": "insufficient-static-area"}
    sift = cv2.SIFT_create(nfeatures=2500)
    a, da = sift.detectAndCompute(cv2.cvtColor(baseline, cv2.COLOR_RGB2GRAY), static.astype(np.uint8) * 255)
    b, db = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), static.astype(np.uint8) * 255)
    if da is None or db is None:
        return {"accepted": False, "reason": "insufficient-static-features"}
    pairs = cv2.BFMatcher().knnMatch(da, db, k=2)
    matches = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.65 * n.distance]
    if len(matches) < config.min_static_matches:
        return {"accepted": False, "reason": "insufficient-static-matches", "matches": len(matches)}
    shifts = np.asarray([np.asarray(b[m.trainIdx].pt) - np.asarray(a[m.queryIdx].pt) for m in matches])
    displacement = np.linalg.norm(shifts, axis=1)
    p90 = float(np.percentile(displacement, 90))
    bad = float(np.mean(displacement > config.static_p90_limit_px))
    accepted = p90 <= config.static_p90_limit_px and bad <= config.max_static_relative_bad_fraction
    return {
        "accepted": accepted,
        "reason": "verified" if accepted else "static-displacement",
        "matches": len(matches),
        "p90DisplacementPixels": p90,
        "badFraction": bad,
    }


def select_patch(
    old, region, images, valids, costs, observations, group, source_names, config, all_vehicle, origin=(0, 0)
):
    """Atomically select one whole conflict patch or leave every pixel alone."""
    candidates = []
    decisions = []
    for index, name in enumerate(source_names):
        present = [i for i in group if observations[i]["sourceImage"] == name and observations[i]["complete"]]
        if not present:
            decisions.append({"sourceImage": name, "reason": "no-complete-associated-vehicle"})
            continue
        cut_vehicle = cuts_observed_vehicle(
            region, origin, [item for item in observations if item["sourceImage"] == name]
        )
        if cut_vehicle is not None:
            decisions.append({"sourceImage": name, "reason": "conflict-cuts-source-vehicle"})
            continue
        if not valids[index][region].all():
            decisions.append({"sourceImage": name, "reason": "incomplete-conflict-coverage"})
            continue
        if np.max(costs[index][region]) >= 1e5:
            continue
        exclusion = ndimage.binary_dilation(all_vehicle, iterations=5)
        static = region & ~exclusion & (old[3] > 0) & valids[index]
        agreement = static_agreement(old[:3].transpose(1, 2, 0), images[index], static, config)
        if not agreement["accepted"]:
            decisions.append(
                {"sourceImage": name, "reason": agreement["reason"], "staticAgreement": agreement}
            )
            continue
        candidates.append((float(np.mean(costs[index][region])), name, index, agreement))
    if not candidates:
        return None, {"state": "baseline-fallback", "rejections": decisions}
    cost, name, index, agreement = min(candidates)
    return index, {
        "state": "accepted",
        "sourceImage": name,
        "meanProjectionCost": cost,
        "staticAgreement": agreement,
        "rejections": decisions,
    }


def run(
    project: Path, output: Path, models: Path, config: PatchConfig, device: str, bounds=None, scan_only=False
):
    started = time.monotonic()
    project, output = project.resolve(), output.resolve()
    if output.is_relative_to(project) or project.is_relative_to(output):
        raise ValueError("Output must be separate from preserved checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "ogs2_orthophoto.tif"
    partial = output / "ogs2_orthophoto.partial.tif"
    if completed.exists() or partial.exists():
        raise FileExistsError("Refuse to overwrite completed/partial experiment; use a new directory")
    baseline = project / "odm_orthophoto/odm_orthophoto.tif"
    critical = [
        baseline,
        project / "opensfm/undistorted/reconstruction.json",
        project / "odm_meshing/odm_25dmesh.ply",
    ]
    code_paths = {
        "engine": Path(__file__),
        "policy": Path(__file__).with_name("ogs2_temporal.py"),
        "projection": Path(__file__).with_name("ogs2.py"),
    }
    code_hashes = {key: sha256(path) for key, path in code_paths.items()}
    before = {str(p): sha256(p) for p in critical}
    save_json(output / "status.json", {"state": "preparing", "config": asdict(config)})
    cv2.setNumThreads(config.cpu_threads)
    cv2.ocl.setUseOpenCL(False)
    surface_path = output.parent / "ogs2-surface.tif"
    surface_info = export_surface(project, baseline, surface_path, 2)
    with rio.open(surface_path) as src:
        surface, surface_transform = src.read(1), src.transform
    with rio.open(baseline) as ref:
        if ref.count != 4:
            raise ValueError("Require baseline RGB+alpha")
        factory, data = cameras_from_checkpoint(project, ref.crs)
        if bounds:
            raw = rio.windows.from_bounds(*bounds, transform=ref.transform)
            window = raw.round_offsets().round_lengths().intersection(Window(0, 0, ref.width, ref.height))
        else:
            window = Window(0, 0, ref.width, ref.height)
        width, height = int(window.width), int(window.height)
        transform = ref.window_transform(window)
        names = sorted(data["shots"])
        source_ids = {name: i + 1 for i, name in enumerate(names)}
        key = {
            "inputHashes": before,
            "window": list(window.flatten()),
            "detectorEngine": code_hashes["projection"],
            "collectionEngine": code_hashes["engine"],
            "modelFiles": {
                str(path.relative_to(models)): sha256(path)
                for path in sorted(models.rglob("*"))
                if path.is_file() and path.suffix in {".json", ".pt", ".pth", ".safetensors", ".bin"}
            },
            "config": asdict(config),
            "descriptor": "hsv-12x3/v1",
        }

        def collect():
            detector = Detector(models, device, Config(cpu_threads=config.cpu_threads))
            return collect_observations(
                project,
                factory,
                data,
                names,
                surface,
                surface_transform,
                transform,
                width,
                height,
                detector,
                config,
                output,
                baseline,
                window,
            )

        observations, reused = observation_cache(output, key, collect)
        temporal_config = TemporalConfig(require_explicit_clock=True)
        groups, association = associate_observations(observations, abs(transform.a), temporal_config)
        save_json(output / "association.json", association)
        save_json(
            output / "dynamic-groups.json",
            [
                {
                    "group": i + 1,
                    "observations": g,
                    "sourceImages": [observations[j]["sourceImage"] for j in g],
                }
                for i, g in enumerate(groups)
            ],
        )
        if scan_only:
            save_json(
                output / "status.json",
                {"state": "scan-complete", "observations": len(observations), "dynamicGroups": len(groups)},
            )
            return
        profile = {
            **ref.profile,
            "width": width,
            "height": height,
            "transform": transform,
            "count": 4,
            "dtype": "uint8",
            "compress": "DEFLATE",
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "nodata": None,
        }
        source_profile = {**profile, "count": 1, "dtype": "uint16", "nodata": 0}
        patch_records = []
        accepted_mask = np.zeros((height, width), bool)
        claimed_mask = np.zeros((height, width), bool)
        with (
            rio.open(partial, "w", **profile) as dst,
            rio.open(output / "source-index.tif", "w", **source_profile) as source_dst,
        ):
            dst.colorinterp = ref.colorinterp
            for _, block in dst.block_windows(1):
                old = ref.read(
                    window=Window(
                        window.col_off + block.col_off,
                        window.row_off + block.row_off,
                        block.width,
                        block.height,
                    )
                )
                dst.write(old, window=block)
                source_dst.write(np.zeros((int(block.height), int(block.width)), np.uint16), 1, window=block)
        with (
            rio.open(partial, "r+") as dst,
            rio.open(output / "source-index.tif", "r+") as source_dst,
        ):
            # Temporal grouping happens once globally; tile traversal cannot
            # assign separate times to the same connected short track.
            plans = []
            padding = max(1, math.ceil(config.padding_m / abs(transform.a)))
            for group_id, group in enumerate(groups, 1):
                try:
                    region, origin = render_conflict(
                        group, observations, padding_pixels=padding, max_pixels=config.max_patch_pixels
                    )
                except ValueError as error:
                    patch_records.append(
                        {"group": group_id, "state": "baseline-fallback", "reason": str(error)}
                    )
                    continue
                x0, y0 = origin
                x1, y1 = x0 + region.shape[1], y0 + region.shape[0]
                if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
                    patch_records.append(
                        {
                            "group": group_id,
                            "state": "baseline-fallback",
                            "reason": "conflict-crosses-output-boundary",
                        }
                    )
                    continue
                plans.append((group_id, group, region, origin))
                claimed_mask[y0:y1, x0:x1] |= region
            # Reject overlapping plans before any write; order must not change
            # whether a previously accepted vehicle is overwritten.
            overlap_count = np.zeros((height, width), np.uint16)
            for _, _, region, (x0, y0) in plans:
                overlap_count[y0 : y0 + region.shape[0], x0 : x0 + region.shape[1]] += region
            for group_id, group, region, origin in plans:
                x0, y0 = origin
                h, w = region.shape
                record = {
                    "group": group_id,
                    "window": [x0, y0, w, h],
                    "pixels": int(region.sum()),
                    "observations": group,
                }
                if (overlap_count[y0 : y0 + h, x0 : x0 + w][region] > 1).any():
                    patch_records.append(
                        {**record, "state": "baseline-fallback", "reason": "overlapping-conflict-plans"}
                    )
                    continue
                local_window = Window(x0, y0, w, h)
                old = dst.read(window=local_window)
                if not (old[3][region] > 0).all():
                    patch_records.append(
                        {
                            **record,
                            "state": "baseline-fallback",
                            "reason": "conflict-crosses-invalid-baseline",
                        }
                    )
                    continue
                cut_vehicle = cuts_observed_vehicle(
                    region, origin, [item for item in observations if item.get("kind") == "baseline"]
                )
                if cut_vehicle is not None:
                    patch_records.append(
                        {
                            **record,
                            "state": "baseline-fallback",
                            "reason": "conflict-cuts-observed-vehicle",
                            "cutObservation": cut_vehicle,
                        }
                    )
                    continue
                points = surface_points(surface, surface_transform, local_window, transform)
                zs = points[2][region]
                if not len(zs) or not np.isfinite(zs).all() or np.ptp(zs) > config.max_surface_spread_m:
                    patch_records.append(
                        {**record, "state": "baseline-fallback", "reason": "unreliable-or-multilayer-surface"}
                    )
                    continue
                source_names = sorted({observations[i]["sourceImage"] for i in group})
                images, valids, costs, used = [], [], [], []
                for name in source_names:
                    value = project_view(
                        project,
                        name,
                        factory.get(name),
                        points,
                        surface,
                        surface_transform,
                        None,
                        Config(cpu_threads=config.cpu_threads),
                    )
                    if value is not None:
                        image, valid, cost, _ = value
                        images.append(image)
                        valids.append(valid)
                        costs.append(cost)
                        used.append(name)
                all_vehicle = np.zeros(region.shape, bool)
                for observation in observations:
                    ox, oy = observation["origin"]
                    oh, ow = observation["mask"].shape
                    if ox < x0 + w and oy < y0 + h and ox + ow > x0 and oy + oh > y0:
                        all_vehicle |= mask_in_window(observation, origin, region.shape)
                selected, decision = select_patch(
                    old, region, images, valids, costs, observations, group, used, config, all_vehicle, origin
                )
                if selected is not None:
                    old[:3, region] = images[selected][region].T
                    dst.write(old, window=local_window)
                    ids = source_dst.read(1, window=local_window)
                    ids[region] = source_ids[used[selected]]
                    source_dst.write(ids, 1, window=local_window)
                    accepted_mask[y0 : y0 + h, x0 : x0 + w] |= region
                    decision["captureTime"], decision["clock"] = capture_time(
                        used[selected], data["shots"][used[selected]]
                    )
                patch_records.append({**record, **decision})
                save_json(output / "objects.json", patch_records)
                print("patch", group_id, decision["state"], flush=True)
            dst.build_overviews([2, 4, 8, 16], Resampling.average)
        save_json(output / "objects.json", patch_records)
        with rio.open(output / "protected-objects.tif", "w", **{**source_profile, "dtype": "uint8"}) as dst:
            dst.write(accepted_mask.astype(np.uint8), 1)
        with rio.open(output / "conflict-regions.tif", "w", **{**source_profile, "dtype": "uint8"}) as dst:
            dst.write(claimed_mask.astype(np.uint8), 1)
        after = {str(p): sha256(p) for p in critical}
        if before != after:
            raise RuntimeError("Preserved checkpoint changed")
        if code_hashes != {key: sha256(path) for key, path in code_paths.items()}:
            raise RuntimeError("Algorithm source changed during run; preserve this attempt and retry")
        partial.replace(completed)
        report = {
            "schemaVersion": "ogs2-temporal-patches/v1",
            "engineSha256": code_hashes["engine"],
            "policySha256": code_hashes["policy"],
            "projectionSha256": key["detectorEngine"],
            "inputHashes": before,
            "surface": surface_info,
            "config": asdict(config),
            "associationConfig": asdict(temporal_config),
            "observations": len(observations),
            "reusedObservationCache": reused,
            "dynamicGroups": len(groups),
            "acceptedPatches": sum(p["state"] == "accepted" for p in patch_records),
            "acceptedPixels": int(accepted_mask.sum()),
            "sourceIndex": {str(v): k for k, v in source_ids.items()},
            "grid": {"crs": str(ref.crs), "transform": list(transform), "width": width, "height": height},
            "elapsedSeconds": time.monotonic() - started,
            "output": str(completed),
            "limitations": [
                "Short-time association is not a global vehicle identity guarantee.",
                "Only complete single-photo conflict patches are replaced; others retain baseline.",
                "No unseen background synthesis, no vehicle repositioning, no static image warp.",
                "Existing surface remains unchanged; crane overhangs are not solved.",
            ],
        }
        save_json(output / "report.json", report)
        save_json(output / "status.json", {"state": "success", "elapsedSeconds": report["elapsedSeconds"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--bounds", nargs=4, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args()
    if args.candidates < 2:
        parser.error("At least two candidates are required")
    try:
        run(
            args.project,
            args.output,
            args.models,
            PatchConfig(candidates=args.candidates),
            args.device,
            args.bounds,
            args.scan_only,
        )
    except Exception as error:
        if args.output.exists() and not (args.output / "ogs2_orthophoto.tif").exists():
            save_json(
                args.output / "status.json",
                {"state": "failed", "error": type(error).__name__ + ": " + str(error)},
            )
        raise


if __name__ == "__main__":
    main()
