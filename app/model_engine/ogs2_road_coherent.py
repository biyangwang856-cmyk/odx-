"""Optional coherent-exposure road segments around verified vehicle anchors.

This path uses complete observed road and vehicle pixels from one real photo.
It does not infer a simultaneous timestamp for the entire survey. A pure hard
boundary is used if background geometry cannot certify safe feathering.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from experiments.tools.inspect_joint_static import load_saved
from rasterio.windows import Window
from scipy import ndimage

from model_engine.ogs2 import save_json, sha256, surface_points
from model_engine.ogs2_deghost import capture_time, mask_in_window
from model_engine.ogs2_joint import read_base
from model_engine.ogs2_joint_closure import close_whole_foregrounds
from model_engine.ogs2_joint_geometry import independent_static_evidence
from model_engine.ogs2_joint_seam import SeamConfig, static_seam_weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project", "cache", "surface", "base-orthophoto", "base-masks", "road-masks", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--context-pixels", type=int, default=480)
    args = parser.parse_args()
    project, cache, surface_path, base, base_masks, road_masks, output = [
        getattr(args, name.replace("-", "_")).resolve()
        for name in ["project", "cache", "surface", "base-orthophoto", "base-masks", "road-masks", "output"]
    ]
    if any(
        output.is_relative_to(path) or path.is_relative_to(output)
        for path in (project, cache, base, base_masks, road_masks)
    ):
        raise ValueError("Output overlaps retained inputs")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(False)
    obs, surface, st, transform, original, profile, factory, data, old_report = load_saved(
        project, cache, surface_path, args.context_pixels
    )
    old = read_base(base, profile)
    base_hash = sha256(base)
    shape = old.shape[:2]
    pixel = abs(transform.a)
    points = surface_points(surface, st, Window(0, 0, shape[1], shape[0]), transform)
    mosaic_metadata = json.loads((base_masks / "observations.json").read_text())
    assert mosaic_metadata["baselineSha256"] == base_hash
    for key in ("crs", "transform", "width", "height"):
        if mosaic_metadata["grid"][key] != old_report["grid"][key]:
            raise ValueError("Old-mosaic masks use a different projection grid")
    with np.load(base_masks / "observation-masks.npz", allow_pickle=False) as masks:
        mosaic = [
            {**record, "origin": record["expandedOrigin"], "mask": masks[f"mask_{i}"].astype(bool)}
            for i, record in enumerate(mosaic_metadata["observations"])
        ]
    with rio.open(cache / "ogs2_orthophoto.tif") as ref:
        cx, cy = ~transform * (ref.transform.c, ref.transform.f)
        original_width, original_height = ref.width, ref.height
    evidence_crop = np.s_[round(cy) : round(cy) + original_height, round(cx) : round(cx) + original_width]
    records = json.loads((road_masks / "report.json").read_text())
    sources = {}
    for record in records:
        name = record["sourceImage"]
        path = Path(record["projectionCache"]).resolve()
        if sha256(path) != record["projectionCacheSha256"]:
            raise ValueError("Source cache differs from road segmentation input")
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["grid"] != old_report["grid"]:
            raise ValueError("Source evidence uses a different map pixel frame")
        if sha256(project / "opensfm/undistorted/images" / name) != metadata["sourceImageSha256"]:
            raise ValueError("Source photo differs from preserved projection")
        if (
            "maskArchiveSha256" in record
            and sha256(road_masks / (name + ".npz")) != record["maskArchiveSha256"]
        ):
            raise ValueError("Road mask archive has changed")
        with np.load(path, allow_pickle=False) as frame:
            image, valid, cost = frame["image"], frame["valid"].astype(bool), frame["cost"]
        with np.load(road_masks / (name + ".npz"), allow_pickle=False) as masks:
            road = masks["road"].astype(bool)
            labels, n = ndimage.label(road)
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            if sizes.max() < 1000:
                raise ValueError("No supported road component in source evidence")
            road = labels == int(np.argmax(sizes))
            # Semantic pavement masks exclude cars. Filling mask holes reserves
            # their road footprints; all output colors still come from photos.
            road = ndimage.binary_fill_holes(road)
            vehicles = [
                {**item, "mask": masks[f"vehicle_{i}"].astype(bool)}
                for i, item in enumerate(record["vehicles"])
            ]
        sources[name] = {
            "image": image,
            "valid": valid,
            "cost": cost,
            "road": road,
            "vehicles": vehicles,
            "path": path,
        }
    if len(sources) < 2:
        raise ValueError("Road coherence requires independently verified real exposures")
    names = sorted(sources)
    all_vehicle = np.zeros(shape, bool)
    for record in obs:
        all_vehicle |= mask_in_window(record, (0, 0), shape)
    # The fixed road anchor is measured from the original observation window;
    # the additional context is used only to find complete patch boundaries.
    ground = np.isfinite(points[2]) & (
        np.abs(points[2] - np.nanmedian(points[2][sources[names[0]]["road"]])) < 0.75
    )
    static = ground & (original[..., 3] > 0) & ~ndimage.binary_dilation(all_vehicle, iterations=40)
    first, second = sources[names[0]], sources[names[1]]
    proof = independent_static_evidence(
        first["image"][evidence_crop],
        second["image"][evidence_crop],
        (static & first["valid"] & second["valid"])[evidence_crop],
    )
    save_json(output / "geometry-proof.json", proof)
    if not proof["accepted"]:
        raise ValueError("Independent road source geometry failed: " + proof["reason"])
    # Reserve the complete visible road corridor, not just the old 1024px tile:
    # historical instances can lie outside that tile. Keep a 2m map boundary
    # collar; foreground closure must never be allowed to truncate an object.
    domain = np.zeros(shape, bool)
    collar = math.ceil(2 / pixel)
    domain[collar:-collar, collar:-collar] = True
    common_road = np.logical_and.reduce([source["road"] for source in sources.values()])
    anchor = ndimage.binary_erosion(common_road, iterations=12) & domain & (old[..., 3] > 0)
    if anchor.sum() < 1000:
        raise ValueError("No sufficiently supported common road region")
    proposals = []
    rejected = []
    for name, source in sources.items():
        # Pair overlap certifies calibrated ground geometry. The chosen real
        # exposure may cover a longer corridor than that overlap; otherwise a
        # historical vehicle just beyond the other photo remains as a ghost.
        seed = ndimage.binary_erosion(source["road"], iterations=12) & domain
        seed &= source["valid"] & (old[..., 3] > 0)
        labels, _ = ndimage.label(seed)
        touching = np.unique(labels[anchor])
        seed &= np.isin(labels, touching[touching > 0])
        catalog = {}
        retained = []
        for i, record in enumerate(source["vehicles"]):
            mask = record["mask"]
            area = float(mask.sum() * pixel * pixel)
            road_fraction = float(source["road"][mask].mean())
            if (
                record["complete"]
                and 0.5 <= area <= 70
                and road_fraction >= 0.7
                and source["valid"][mask].all()
            ):
                catalog[f"source-{i}"] = mask
                if (mask & seed).any():
                    retained.append(i)
        if not retained:
            rejected.append({"sourceImage": name, "reason": "no-complete-road-vehicle"})
            continue
        for i, record in enumerate(mosaic):
            mask = mask_in_window(record, (0, 0), shape)
            if 0.5 <= mask.sum() * pixel * pixel <= 70 and source["road"][mask].mean() >= 0.7:
                if not record["complete"] and (mask & seed).any():
                    # Another overlapping cut may prove completeness; don't
                    # confuse a crop-edge observation with a cropped vehicle.
                    covered = np.zeros(shape, bool)
                    for other in mosaic:
                        if other["complete"]:
                            covered |= mask_in_window(other, (0, 0), shape)
                    if (mask & ~ndimage.binary_dilation(covered, iterations=3)).sum() > max(
                        5, 0.02 * mask.sum()
                    ):
                        rejected.append({"sourceImage": name, "reason": "incomplete-old-road-vehicle"})
                        catalog = None
                        break
                catalog[f"mosaic-{i}"] = mask
        if catalog is None:
            continue
        try:
            hard = close_whole_foregrounds(
                seed, catalog, protection_steps=math.ceil(0.5 / pixel), max_support_pixels=2_000_000
            ).support
            if not source["valid"][hard].all() or not (old[..., 3][hard] > 0).all():
                raise ValueError("Incomplete road segment coverage")
            yy, xx = np.where(hard)
            x0, y0 = max(0, int(xx.min()) - 40), max(0, int(yy.min()) - 40)
            x1, y1 = min(shape[1], int(xx.max()) + 41), min(shape[0], int(yy.max()) + 41)
            crop = np.s_[y0:y1, x0:x1]
            # No fractional blending. The verified road projection is retained
            # intact even where the older texture has a local baking offset.
            config = SeamConfig(max_pixels=2_000_000, min_hard_margin_px=0, feather_radius_px=0)
            pair_static = (
                np.max(np.abs(first["image"].astype(np.int16) - second["image"].astype(np.int16)), axis=2)
                < 15
            )
            weights, _, _, seam = static_seam_weights(
                old[crop],
                source["image"][crop],
                hard[crop],
                verified_static=pair_static[crop],
                valid_source=source["valid"][crop],
                protected_foregrounds=[mask[crop] for mask in catalog.values()],
                config=config,
            )
            full_weight = np.zeros(shape, np.float32)
            full_weight[crop] = weights
            # Every actually selected vehicle is preserved whole or wholly left
            # outside; no composite may cut through an observed silhouette.
            for key, mask in catalog.items():
                count = int((mask & (full_weight > 0)).sum())
                if count and count < int(mask.sum()):
                    raise ValueError("Road seam cuts an observed complete vehicle")
            proposals.append(
                (float(np.mean(source["cost"][hard])), name, hard, full_weight, seam, retained, catalog)
            )
        except (ValueError, RuntimeError) as error:
            rejected.append({"sourceImage": name, "reason": str(error)})
    save_json(output / "rejections.json", rejected)
    if not proposals:
        raise ValueError("No whole observed road exposure accepted")
    _, name, hard, weight, seam, retained, catalog = min(proposals, key=lambda proposal: proposal[:2])
    source = sources[name]
    result = old.copy()
    write = weight > 0
    result[write, :3] = source["image"][write]
    mode = write.astype(np.uint8)
    source_id = sorted(data["shots"]).index(name) + 1
    ids = write.astype(np.uint16) * source_id
    actual_profile = {
        **profile,
        "compress": "DEFLATE",
        "nodata": None,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    partial = output / "ogs2_orthophoto.partial.tif"
    with rio.open(partial, "w", **actual_profile) as ref:
        ref.write(result.transpose(2, 0, 1))
    for filename, array, dtype in [
        ("source-index", ids, "uint16"),
        ("source-weight", weight, "float32"),
        ("composition-mode", mode, "uint8"),
        ("protected-objects", hard, "uint8"),
        ("write-support", write, "uint8"),
    ]:
        with rio.open(
            output / (filename + ".tif"), "w", **{**actual_profile, "count": 1, "dtype": dtype}
        ) as ref:
            ref.write(array.astype(dtype), 1)
    np.savez_compressed(
        output / "component-1-masks.npz",
        hard=hard,
        write=write,
        weight=weight,
        measuredChange=np.zeros(shape, bool),
        **catalog,
    )
    with rio.open(partial) as ref:
        actual = ref.read().transpose(1, 2, 0)
    if not np.array_equal(actual[..., 3], old[..., 3]) or not np.array_equal(actual[~write], old[~write]):
        raise RuntimeError("Road output changes alpha or pixels outside write")
    if not np.array_equal(actual[write, :3], source["image"][write]):
        raise RuntimeError("Road source pixels differ from original observed source")
    if sha256(base) != base_hash:
        raise RuntimeError("Old scene changed during experiment")
    partial.replace(output / "ogs2_orthophoto.tif")
    stamp, clock = capture_time(name, data["shots"][name])
    object_record = {
        "component": 1,
        "state": "accepted",
        "sourceImage": name,
        "sourceIndex": source_id,
        "sourceCache": str(source["path"]),
        "captureTime": stamp,
        "clock": clock,
        "retainedVehicleObservations": retained,
        "trackMasks": str(output / "component-1-masks.npz"),
        "window": [0, 0, shape[1], shape[0]],
        "requiredTracks": [f"source-{i}" for i in retained],
        "hardPixels": int(hard.sum()),
        "writePixels": int(write.sum()),
        "seam": {k: v for k, v in seam.items() if k != "verifiedStaticMask"},
        "timeChoice": "One actual exposure for this road segment",
    }
    save_json(output / "objects.json", [object_record])
    save_json(
        output / "report.json",
        {
            "schemaVersion": "ogs2-road-coherent/v1",
            "baseOrthophoto": str(base),
            "baseSha256": base_hash,
            "project": str(project),
            "cache": str(cache),
            "surface": str(surface_path),
            "grid": old_report["grid"],
            "inputHashes": old_report["inputHashes"],
            "sourceIndex": {str(i + 1): n for i, n in enumerate(sorted(data["shots"]))},
            "sourceRoadEvidence": str(road_masks),
            "geometryProof": proof,
            "acceptedComponents": 1,
            "elapsedSeconds": time.monotonic() - start,
            "output": str(output / "ogs2_orthophoto.tif"),
            "limitations": [
                "Local reference time; not simultaneous across the whole survey.",
                "Native surface and absolute survey accuracy are not improved.",
                "Pure boundaries can expose original texture displacement; inspect road lines and curbs.",
                "Semantic road and vehicle masks can be wrong; only actual observed RGB is written.",
            ],
        },
    )
    save_json(output / "status.json", {"state": "success"})
    print(
        json.dumps(
            {
                "sourceImage": name,
                "sourceId": source_id,
                "hardPixels": int(hard.sum()),
                "retainedObservedVehicles": len(retained),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
