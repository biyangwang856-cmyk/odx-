"""Optional whole-exposure vehicle and shadow patches on preserved OGS2.

Static correspondences verify source-to-source physical projection. Real
inter-exposure changes expand protection beyond SAM bodies. Static seams never
warp a vehicle or synthesize a background. Default ODX paths are unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import rasterio as rio
from experiments.tools.inspect_joint_static import cached_projection, load_saved
from rasterio.windows import Window
from scipy import ndimage

from model_engine.ogs2 import save_json, sha256, surface_points
from model_engine.ogs2_deghost import capture_time, mask_in_window
from model_engine.ogs2_joint_closure import close_whole_foregrounds, reserve_joint_components
from model_engine.ogs2_joint_geometry import (
    GeometryConfig,
    close_unknown_boundary,
    independent_static_evidence,
    observed_dynamic_support,
)
from model_engine.ogs2_joint_policy import CandidateEvidence, choose_whole_source, compose_static_collar
from model_engine.ogs2_temporal import TemporalConfig, associate_observations, render_conflict


def read_base(path, profile):
    with rio.open(path) as ref:
        if ref.crs != profile["crs"] or ref.count != 4:
            raise ValueError("Output base must be RGB+alpha in the same CRS")
        transform = profile["transform"]
        if not np.allclose(
            [ref.transform.a, ref.transform.b, ref.transform.d, ref.transform.e],
            [transform.a, transform.b, transform.d, transform.e],
            atol=1e-10,
            rtol=0,
        ):
            raise ValueError("Output base uses a different pixel grid")
        col, row = ~ref.transform * (transform.c, transform.f)
        if max(abs(col - round(col)), abs(row - round(row))) > 1e-6:
            raise ValueError("Output base grid origin differs")
        window = Window(round(col), round(row), profile["width"], profile["height"])
        if col < 0 or row < 0 or col + window.width > ref.width or row + window.height > ref.height:
            raise ValueError("Observation window is outside output base")
        return ref.read(window=window).transpose(1, 2, 0)


def component_indices(masks):
    parents = list(range(len(masks)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if (masks[i] & masks[j]).any():
                parents[find(j)] = find(i)
    components = {}
    for i in range(len(masks)):
        components.setdefault(find(i), []).append(i)
    return list(components.values())


def run(project, cache, surface_path, output, base_path, projection_cache=None, base_masks=None):
    started = time.monotonic()
    for retained in (project, cache, surface_path, base_path):
        if output.is_relative_to(retained) or retained.is_relative_to(output):
            raise ValueError("Output overlaps preserved input/cache")
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(8)
    cv2.ocl.setUseOpenCL(False)
    geometry = GeometryConfig()
    code_files = [
        Path(__file__),
        Path(__file__).with_name("ogs2_joint_geometry.py"),
        Path(__file__).with_name("ogs2_joint_policy.py"),
        Path(__file__).with_name("ogs2_joint_seam.py"),
    ]
    before_code = {p.name: sha256(p) for p in code_files}
    obs, surface, st, transform, baseline, profile, factory, data, old_report = load_saved(
        project, cache, surface_path, context_pixels=240
    )
    base_hash = sha256(base_path)
    old = read_base(base_path, profile)
    with rio.open(cache / "ogs2_orthophoto.tif") as original_window:
        cx, cy = ~transform * (original_window.transform.c, original_window.transform.f)
        evidence_crop = np.s_[
            round(cy) : round(cy) + original_window.height, round(cx) : round(cx) + original_window.width
        ]
    mosaic_obs = []
    if base_masks is not None:
        metadata = json.loads((base_masks / "observations.json").read_text())
        if metadata["baselineSha256"] != base_hash or any(
            metadata["grid"][key] != old_report["grid"][key]
            for key in ("crs", "transform", "width", "height")
        ):
            raise ValueError("Mosaic mask cache uses different input/grid")
        with np.load(base_masks / "observation-masks.npz", allow_pickle=False) as saved:
            mosaic_obs = [
                {**record, "origin": record["expandedOrigin"], "mask": saved[f"mask_{i}"].astype(bool)}
                for i, record in enumerate(metadata["observations"])
            ]
    points = surface_points(surface, st, Window(0, 0, profile["width"], profile["height"]), transform)
    pixel = abs(transform.a)
    groups, association = associate_observations(obs, pixel, TemporalConfig(require_explicit_clock=True))
    all_vehicle = np.zeros(old.shape[:2], bool)
    for record in obs:
        all_vehicle |= mask_in_window(record, (0, 0), all_vehicle.shape)
    projection_key = {
        "inputHashes": old_report["inputHashes"],
        "projectionSha256": old_report["projectionSha256"],
        "surfaceSha256": sha256(surface_path),
        "grid": old_report["grid"],
    }
    source_directory = projection_cache or output / "sources"
    frames = {}
    tracks = []
    rejections = []
    for number, group in enumerate(groups, 1):
        names = sorted({obs[i]["sourceImage"] for i in group})
        body = np.zeros(old.shape[:2], bool)
        for i in group:
            body |= mask_in_window(obs[i], (0, 0), body.shape)
        finite = body & np.isfinite(points[2])
        if not finite.any():
            rejections.append({"group": number, "reason": "no-finite-body-surface"})
            continue
        z = float(np.median(points[2][finite]))
        ground = np.isfinite(points[2]) & (np.abs(points[2] - z) < 0.75)
        static = ground & (baseline[..., 3] > 0) & ~ndimage.binary_dilation(all_vehicle, iterations=40)
        present = []
        for name in names:
            if name not in frames:
                value = cached_projection(
                    project, name, factory, points, surface, st, source_directory, projection_key
                )
                if value is not None:
                    frames[name] = value
            if name in frames:
                present.append(name)
        proven = []
        for i, name in enumerate(present):
            for other in present[i + 1 :]:
                a, va, _, _ = frames[name]
                b, vb, _, _ = frames[other]
                # Added sampling context must not change the predefined spatial
                # train/holdout partition in the original observation window.
                proof = independent_static_evidence(
                    a[evidence_crop], b[evidence_crop], (static & va & vb)[evidence_crop], geometry
                )
                proof["evidenceWindow"] = [
                    round(cx),
                    round(cy),
                    original_window.width,
                    original_window.height,
                ]
                if proof["accepted"]:
                    proven.append((name, other, proof))
        if not proven:
            rejections.append({"group": number, "reason": "unverified-source-pair-geometry"})
            continue
        # Use an actual verified pair; longer tracks are not silently extrapolated.
        first, second, proof = min(proven, key=lambda pair: pair[2]["p90ActualPixels"])
        indexes = [i for i in group if obs[i]["sourceImage"] in {first, second}]
        region, origin = render_conflict(indexes, obs, padding_pixels=math.ceil(0.8 / pixel))
        swept = np.zeros(body.shape, bool)
        x, y = origin
        h, w = region.shape
        if x < 0 or y < 0 or x + w > body.shape[1] or y + h > body.shape[0]:
            rejections.append({"group": number, "reason": "body-sweep-crosses-window"})
            continue
        swept[y : y + h, x : x + w] = region
        a, va, _, _ = frames[first]
        b, vb, _, _ = frames[second]
        # A vehicle shadow lies on the neighbouring road surface. Tree-crown
        # parallax cannot become road motion merely because its RGB changed.
        road_surface = np.isfinite(points[2]) & (np.abs(points[2] - z) <= 1.0)
        hard, measured, changes = observed_dynamic_support(
            [a, b], [va & road_surface, vb & road_surface], body, pixel, geometry
        )
        # Include actual old mosaic remnants, using its observed pixels rather
        # than assuming the new photo pair contains every historical ghost.
        old_hard, old_measured, old_changes = observed_dynamic_support(
            [old[..., :3], a, b],
            [(old[..., 3] > 0) & road_surface, va & road_surface, vb & road_surface],
            body,
            pixel,
            geometry,
        )
        # Protection dilation is bounded by the observed neighbouring surface;
        # it must not spill into a tall crown that is outside road evidence.
        hard = (hard | old_hard) & road_surface
        hard |= swept | body
        measured |= old_measured
        if not (old[..., 3][hard] > 0).all():
            rejections.append({"group": number, "reason": "hard-support-crosses-invalid-base"})
            continue
        height_spread = float(np.ptp(points[2][hard]))
        if not np.isfinite(height_spread) or height_spread > 2:
            rejections.append(
                {"group": number, "reason": "multilayer-dynamic-support", "spread": height_spread}
            )
            continue
        candidates = {}
        for name in {first, second}:
            entries = [i for i in group if obs[i]["sourceImage"] == name and obs[i]["complete"]]
            if len(entries) == 1:
                candidates[name] = entries
        tracks.append(
            {
                "identity": f"track-{number}",
                "group": number,
                "indices": indexes,
                "hard": hard,
                "measured": measured,
                "sources": candidates,
                "pairProof": proof,
                "changes": changes,
                "oldChanges": old_changes,
            }
        )
        print("verified track", number, hard.sum(), flush=True)
    from model_engine.ogs2_joint_seam import SeamConfig, static_seam_weights

    mosaic_catalog = {
        f"mosaic-{i}": mask_in_window(record, (0, 0), old.shape[:2])
        for i, record in enumerate(mosaic_obs)
        if record["complete"] and record.get("confidence", 0) >= 0.8
    }
    catalogs = []
    for track in tracks:
        catalogs.append(
            {
                **mosaic_catalog,
                **{
                    f"photo-{i}": mask_in_window(record, (0, 0), old.shape[:2])
                    for i, record in enumerate(obs)
                    if record["sourceImage"] in track["sources"]
                    and record["complete"]
                    and record.get("confidence", 0) >= 0.8
                },
            }
        )
    reserves, components = reserve_joint_components(
        [track["hard"] for track in tracks],
        catalogs,
        collar_steps=math.ceil(3 / pixel),
        protection_steps=math.ceil(0.8 / pixel),
        max_support_pixels=1_000_000,
    )
    result = old.copy()
    mode = np.zeros(old.shape[:2], np.uint8)
    weight = np.zeros(old.shape[:2], np.float32)
    source_index = np.zeros(old.shape[:2], np.uint16)
    protected = np.zeros(old.shape[:2], bool)
    measured_all = np.zeros(old.shape[:2], bool)
    names = sorted(data["shots"])
    source_ids = {name: i + 1 for i, name in enumerate(names)}
    records = []
    for number, component in enumerate(components, 1):
        chosen_tracks = [tracks[i] for i in component]
        hard = np.logical_or.reduce([track["hard"] for track in chosen_tracks])
        measured = np.logical_or.reduce([track["measured"] for track in chosen_tracks])
        candidates = []
        candidate_data = {}
        for name in sorted(set(name for track in chosen_tracks for name in track["sources"])):
            image, valid, cost, path = frames[name]
            active = [track for track in chosen_tracks if name in track["sources"]]
            required = frozenset(track["identity"] for track in active)
            masks = {
                track["identity"]: mask_in_window(obs[track["sources"][name][0]], (0, 0), hard.shape)
                for track in active
            }
            # A real reference exposure selects its actual co-occurring tracks;
            # spatially overlapping vehicles from other times are not fabricated.
            current_hard = hard.copy()
            source_catalog = {
                f"photo-{i}": mask_in_window(record, (0, 0), hard.shape)
                for i, record in enumerate(obs)
                if record["sourceImage"] == name and record["complete"] and record.get("confidence", 0) >= 0.8
            }
            actual_catalog = {**mosaic_catalog, **source_catalog}
            current_hard = close_whole_foregrounds(
                current_hard,
                actual_catalog,
                protection_steps=math.ceil(0.8 / pixel),
                max_support_pixels=1_000_000,
            ).support
            # All observed vehicle/time footprints are excluded from any blend.
            active_bodies = np.zeros(hard.shape, bool)
            for track in active:
                for index in track["indices"]:
                    active_bodies |= mask_in_window(obs[index], (0, 0), hard.shape)
            forbidden = ndimage.binary_dilation(active_bodies | measured, iterations=5)
            pair_static = np.ones(hard.shape, bool)
            for track in active:
                for other in track["sources"]:
                    if other == name:
                        continue
                    other_rgb, other_valid, _, _ = frames[other]
                    # Smooth only the classification evidence, never output
                    # RGB. Isolated sensor/compression noise is not motion.
                    first_smooth = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
                    other_smooth = cv2.GaussianBlur(other_rgb.astype(np.float32), (0, 0), 1.0)
                    pair_static &= other_valid & (np.max(np.abs(first_smooth - other_smooth), axis=2) < 15)
            static_allowed = pair_static & ~forbidden
            # A 15..25 RGB transition near a measured shadow is unknown, not
            # certified static. Protect that near-boundary uncertainty as hard
            # source content instead of allowing it into a fractional seam.
            try:
                current_hard, uncertainty = close_unknown_boundary(
                    current_hard, static_allowed, valid, old[..., 3] > 0
                )
            except ValueError as error:
                candidate_data[name] = {"rejection": str(error)}
                continue
            current_hard = close_whole_foregrounds(
                current_hard,
                actual_catalog,
                protection_steps=math.ceil(0.8 / pixel),
                max_support_pixels=1_000_000,
            ).support
            np.savez_compressed(
                output / (name + "-attempt.npz"),
                hard=current_hard,
                static=static_allowed,
                valid=valid,
                measured=measured,
            )
            if not np.isfinite(points[2][current_hard]).all() or np.ptp(points[2][current_hard]) > 2:
                candidate_data[name] = {"rejection": "expanded-support-multilayer-surface"}
                continue
            foreground_masks = list(actual_catalog.values())
            yy, xx = np.where(current_hard)
            pad = math.ceil(3 / pixel)
            x0, y0 = max(0, int(xx.min()) - pad), max(0, int(yy.min()) - pad)
            x1, y1 = min(old.shape[1], int(xx.max()) + pad + 1), min(old.shape[0], int(yy.max()) + pad + 1)
            crop = np.s_[y0:y1, x0:x1]
            try:
                local_weights, _, _, seam = static_seam_weights(
                    old[crop],
                    image[crop],
                    current_hard[crop],
                    verified_static=static_allowed[crop],
                    valid_source=valid[crop],
                    protected_foregrounds=[mask[crop] for mask in foreground_masks],
                    config=SeamConfig(max_pixels=1_000_000),
                )
                weights = np.zeros(hard.shape, np.float32)
                weights[crop] = local_weights
            except ValueError as error:
                candidate_data[name] = {"rejection": str(error)}
                continue
            evidence = CandidateEvidence(
                name,
                valid,
                masks,
                required,
                True,
                True,
                float(np.mean(cost[current_hard])) - len(required) * 100,
                {track["identity"]: (str(track["sources"][name][0]),) for track in active},
            )
            selected, decisions = choose_whole_source(current_hard, required, [evidence])
            if selected is None:
                candidate_data[name] = {"rejection": decisions}
                continue
            candidates.append((-len(required), float(np.mean(cost[current_hard])), name))
            candidate_data[name] = {
                "weights": weights,
                "seam": seam,
                "hard": current_hard,
                "static": static_allowed,
                "required": required,
                "masks": masks,
                "path": path,
                "valid": valid,
            }
        if not candidates:
            records.append(
                {
                    "component": number,
                    "state": "baseline-fallback",
                    "tracks": [t["identity"] for t in chosen_tracks],
                    "rejections": candidate_data,
                }
            )
            continue
        _, _, name = min(candidates)
        selected = candidate_data[name]
        image = frames[name][0]
        write = selected["weights"] > 0
        if (write & (mode > 0)).any():
            raise ValueError("Joint seam writers overlap after support expansion")
        # Only the seam routine's explicitly certified background is mixable.
        static = selected["static"]
        result, local_mode, local_weight = compose_static_collar(
            result, image, selected["hard"], selected["weights"], static, selected["valid"]
        )
        mode[write] = local_mode[write]
        weight[write] = local_weight[write]
        source_index[write] = source_ids[name]
        protected |= selected["hard"]
        measured_all |= measured
        mask_path = output / f"component-{number}-masks.npz"
        np.savez_compressed(
            mask_path,
            hard=selected["hard"],
            write=write,
            weight=selected["weights"],
            measuredChange=measured,
            **{key: value for key, value in selected["masks"].items()},
        )
        stamp, clock = capture_time(name, data["shots"][name])
        record = {
            "component": number,
            "state": "accepted",
            "sourceImage": name,
            "sourceCache": str(selected["path"]),
            "sourceIndex": source_ids[name],
            "captureTime": stamp,
            "clock": clock,
            "requiredTracks": sorted(selected["required"]),
            "tracks": [t["identity"] for t in chosen_tracks],
            "trackMasks": str(mask_path),
            "window": [0, 0, profile["width"], profile["height"]],
            "hardPixels": int(selected["hard"].sum()),
            "writePixels": int(write.sum()),
            "mixedPixels": int((local_mode == 2).sum()),
            "seam": {key: value for key, value in selected["seam"].items() if key != "verifiedStaticMask"},
            "pairProofs": [t["pairProof"] for t in chosen_tracks],
            "timeChoice": (
                "Actual complete co-occurring tracks in the chosen exposure; "
                "other times replaced by its observed scene"
            ),
        }
        records.append(record)
        print("accepted component", number, name, record["hardPixels"], flush=True)
    output_profile = {
        **profile,
        "compress": "DEFLATE",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "nodata": None,
    }
    partial = output / "ogs2_orthophoto.partial.tif"
    with rio.open(partial, "w", **output_profile) as ref:
        ref.write(result.transpose(2, 0, 1))
    for filename, array, dtype in [
        ("source-index", source_index, "uint16"),
        ("source-weight", weight, "float32"),
        ("composition-mode", mode, "uint8"),
        ("protected-objects", protected, "uint8"),
        ("write-support", mode > 0, "uint8"),
        ("measured-change", measured_all, "uint8"),
    ]:
        with rio.open(
            output / (filename + ".tif"), "w", **{**output_profile, "count": 1, "dtype": dtype}
        ) as ref:
            ref.write(array.astype(dtype), 1)
    if sha256(base_path) != base_hash or before_code != {p.name: sha256(p) for p in code_files}:
        raise RuntimeError("Input/source changed while experiment ran")
    if any(sha256(Path(path)) != value for path, value in old_report["inputHashes"].items()):
        raise RuntimeError("Preserved reconstruction changed during experiment")
    with rio.open(partial) as ref:
        actual = ref.read().transpose(1, 2, 0)
        if not np.array_equal(actual[..., 3], old[..., 3]) or not np.array_equal(
            actual[mode == 0], old[mode == 0]
        ):
            raise RuntimeError("Alpha or outside-write invariant failed")
        if not (weight[protected] == 1).all() or not (mode[protected] == 1).all():
            raise RuntimeError("Dynamic support was blended")
    partial.replace(output / "ogs2_orthophoto.tif")
    save_json(output / "objects.json", records)
    save_json(output / "rejections.json", rejections)
    save_json(output / "association.json", association)
    report = {
        "schemaVersion": "ogs2-joint-exposure/v1",
        "baseOrthophoto": str(base_path),
        "baseSha256": base_hash,
        "project": str(project),
        "cache": str(cache),
        "surface": str(surface_path),
        "inputHashes": old_report["inputHashes"],
        "codeHashes": before_code,
        "geometryConfig": asdict(geometry),
        "observations": len(obs),
        "dynamicGroups": len(groups),
        "verifiedTracks": len(tracks),
        "acceptedComponents": sum(r["state"] == "accepted" for r in records),
        "grid": old_report["grid"],
        "sourceIndex": {str(value): name for name, value in source_ids.items()},
        "elapsedSeconds": time.monotonic() - started,
        "output": str(output / "ogs2_orthophoto.tif"),
        "limitations": [
            "No unseen background synthesis or image warp.",
            "Reference exposure is local, not simultaneous across the full scene.",
            "Dynamic support is observed appearance evidence, not perfect semantic shadow segmentation.",
            "Unverified tracks retain old mosaic; global vehicle identity remains unverified.",
        ],
    }
    save_json(output / "report.json", report)
    save_json(
        output / "status.json", {"state": "success", "acceptedComponents": report["acceptedComponents"]}
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["project", "cache", "surface", "output", "base-orthophoto"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--projection-cache", type=Path)
    parser.add_argument("--base-masks", type=Path)
    args = parser.parse_args()
    run(
        args.project.resolve(),
        args.cache.resolve(),
        args.surface.resolve(),
        args.output.resolve(),
        args.base_orthophoto.resolve(),
        args.projection_cache.resolve() if args.projection_cache else None,
        args.base_masks.resolve() if args.base_masks else None,
    )


if __name__ == "__main__":
    main()
