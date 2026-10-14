"""Apply hash-bound, explicitly reviewed corridor evidence without inventing it.

Projection manifests alone do not authorize writing. Caller certificates must
provide observed road, verified empty road, independent baseline geometry,
whole foregrounds and unknown protection. Rejected regions remain baseline.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Callable

import numpy as np

from .ogs2_corridor_policy import (
    CorridorCandidate, array_sha256, compose_corridor, select_corridor_source,
)


CERTIFICATE_SCHEMA = "ogs2.corridor-review-certificates/v1"
REPORT_SCHEMA = "ogs2.corridor-reviewed-application/v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def _save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _path(value, relative_to):
    if not isinstance(value, str) or not value:
        raise ValueError("Input paths must be nonempty strings")
    path = Path(value)
    return (path if path.is_absolute() else relative_to / path).resolve()


def _pin(path, expected, pins):
    path = Path(path).resolve()
    if not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
        raise ValueError("Require an exact lowercase SHA256 input pin")
    if str(path) in pins and pins[str(path)] != expected:
        raise ValueError("Contradictory input hash binding: " + str(path))
    if not path.is_file() or file_sha256(path) != expected:
        raise ValueError("Pinned input changed: " + str(path))
    pins[str(path)] = expected
    return path


def _verify_pins(pins):
    for path, expected in pins.items():
        if not Path(path).is_file() or file_sha256(path) != expected:
            raise ValueError("Input changed during corridor review/export: " + path)


def load_certificate_arrays(binding, relative_to, pins):
    """Reject undeclared, object, nonboolean, changed or differently sized masks."""
    path = _pin(_path(binding["path"], relative_to), binding["sha256"], pins)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    expected = binding.get("arraySha256")
    if not isinstance(expected, dict) or set(expected) != set(arrays):
        raise ValueError("Every certificate mask array requires its own exact hash")
    for key, value in arrays.items():
        if value.dtype != np.bool_ or value.ndim != 2:
            raise ValueError("Certificate arrays must be boolean two-dimensional masks")
        if expected[key] != array_sha256(value):
            raise ValueError("Certificate array hash mismatch: " + key)
    return arrays


def _mask(arrays, key, shape):
    if not isinstance(key, str) or key not in arrays or arrays[key].shape != shape:
        raise ValueError("Missing or differently framed certificate mask: " + str(key))
    return arrays[key]


def _catalog(arrays, mapping, shape):
    if not isinstance(mapping, dict) or any(not isinstance(k, str) or not k for k in mapping):
        raise ValueError("Require explicit foreground observation IDs")
    return {name: _mask(arrays, key, shape) for name, key in mapping.items()}


def candidate_from_certificate(certificate, source_arrays, source_record, masks, baseline_rgba, conflict):
    """Use actual cached RGB, validity, cost and capture time, never replacements."""
    shape = baseline_rgba.shape[:2]
    required = ("image", "valid", "cost", "body_mask")
    bindings = certificate.get("sourceArraysSha256")
    if not isinstance(bindings, dict) or set(bindings) != set(required):
        raise ValueError("Require all four actual source array hash bindings")
    for name in required:
        if name not in source_arrays or bindings[name] != array_sha256(source_arrays[name]):
            raise ValueError("Source array hash mismatch: " + name)
    if certificate.get("sourceImage") != source_record.get("sourceImage"):
        raise ValueError("Certificate source differs from actual projection record")
    foregrounds = _catalog(masks, certificate.get("foregrounds"), shape)
    complete = certificate.get("complete_foregrounds")
    if not isinstance(complete, list) or any(not isinstance(name, str) or not name for name in complete) or len(set(complete)) != len(complete):
        raise ValueError("Complete foreground IDs must be explicit and unique")
    keys = certificate.get("masks")
    if not isinstance(keys, dict) or set(keys) != {"observed_road", "verified_empty_road", "static_support", "unknown_foreground"}:
        raise ValueError("Require road, empty-road, static and unknown masks")
    values = {name: _mask(masks, key, shape) for name, key in keys.items()}
    body = source_arrays["body_mask"]
    full_union = np.logical_or.reduce(list(foregrounds.values())) if foregrounds else np.zeros(shape, bool)
    if body.shape != shape or body.dtype != np.bool_ or (body & ~full_union).any():
        raise ValueError("Whole foreground catalogue omits actually observed source body pixels")
    for observation in source_record.get("vehicles", []):
        actual = source_arrays.get(observation.get("maskKey"))
        if actual is None or actual.shape != shape or actual.dtype != np.bool_:
            raise ValueError("Actual source observation has no bound mask")
        if observation.get("complete") is not True and (actual & ~values["unknown_foreground"]).any():
            raise ValueError("Certificate omits incomplete source-body uncertainty")
    stamp = source_record.get("captureTime")
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        raise ValueError("Actual source has no finite capture time")
    rgb, valid, cost = (source_arrays[name] for name in ("image", "valid", "cost"))
    if rgb.dtype != np.uint8 or rgb.shape != (*shape, 3) or valid.dtype != np.bool_ or valid.shape != shape:
        raise ValueError("Actual RGB/validity use a different pixel frame")
    if cost.shape != shape or not np.isfinite(cost).all():
        raise ValueError("Actual source cost is not finite in the common grid")
    score = float(np.mean(cost[conflict], dtype=np.float64))
    return CorridorCandidate(
        source_record["sourceImage"], rgb, valid, values["observed_road"], values["verified_empty_road"],
        values["static_support"], foregrounds, frozenset(complete), score, stamp,
        certificate["static_verification"], values["unknown_foreground"],
        certificate.get("foreground_catalog_complete") is True,
    )


@dataclass
class ReviewedPatch:
    identity: str
    window: tuple[int, int, int, int]
    grid: dict
    baseline_rgba: np.ndarray
    output_rgba: np.ndarray
    hard_support: np.ndarray
    source_rgb: np.ndarray
    source_id: str
    report: dict
    static_collar: np.ndarray
    array_bindings: dict[str, str] = field(init=False)
    frame_binding: str = field(init=False)

    def __post_init__(self):
        self.array_bindings = _patch_array_bindings(self)
        self.frame_binding = _patch_frame_binding(self)
        self.report["reviewedArraySha256"] = dict(self.array_bindings)
        self.report["reviewedFrameSha256"] = self.frame_binding


def _patch_array_bindings(patch):
    return {name: array_sha256(getattr(patch, name)) for name in
            ("baseline_rgba", "output_rgba", "hard_support", "source_rgb", "static_collar")}


def _patch_frame_binding(patch):
    return hashlib.sha256(json.dumps({"id": patch.identity, "sourceId": patch.source_id,
                                      "window": patch.window, "grid": patch.grid},
                                     sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def _verify_patch_snapshot(patch):
    if patch.array_bindings != _patch_array_bindings(patch) or patch.frame_binding != _patch_frame_binding(patch):
        raise ValueError("Reviewed patch pixels/support/reservation/frame changed before export")
    if patch.static_collar.dtype != np.bool_ or patch.static_collar.shape != patch.hard_support.shape or not patch.static_collar.any():
        raise ValueError("Every reviewed patch requires its actual nonempty static reservation")
    if (patch.static_collar & patch.hard_support).any():
        raise ValueError("Static reservation must be outside the hard writer")


def reject_overlapping_patches(patches):
    """Reject writing into another writer or its proved baseline static collar."""
    bad, conflicts = set(), []
    for patch in patches:
        _verify_patch_snapshot(patch)
    for first, a in enumerate(patches):
        ax, ay, aw, ah = a.window
        for second in range(first + 1, len(patches)):
            b = patches[second]
            bx, by, bw, bh = b.window
            left, right, top, bottom = max(ax, bx), min(ax + aw, bx + bw), max(ay, by), min(ay + ah, by + bh)
            if left >= right or top >= bottom:
                continue
            mask_a = a.hard_support[top - ay:bottom - ay, left - ax:right - ax]
            mask_b = b.hard_support[top - by:bottom - by, left - bx:right - bx]
            collar_a = a.static_collar[top - ay:bottom - ay, left - ax:right - ax]
            collar_b = b.static_collar[top - by:bottom - by, left - bx:right - bx]
            hard_overlap = mask_a & mask_b
            reservation_overlap = (mask_a & collar_b) | (mask_b & collar_a)
            overlap = int((hard_overlap | reservation_overlap).sum())
            if overlap:
                bad.update((first, second))
                conflicts.append({"first": a.identity, "second": b.identity, "overlapPixels": overlap,
                                  "hardOverlapPixels": int(hard_overlap.sum()),
                                  "staticReservationConflictPixels": int(reservation_overlap.sum())})
    return [patch for index, patch in enumerate(patches) if index not in bad], conflicts


def _load_bound_source(path, name):
    from .ogs2_corridor_projection import load_bound_source
    return load_bound_source(path, name, recheck_inputs=True)


def _baseline_roi(path, grid):
    import rasterio as rio
    from rasterio.windows import Window
    values = grid.get("referenceWindow")
    if not isinstance(values, list) or len(values) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or int(x) != x for x in values):
        raise ValueError("Evidence window must contain four finite integer pixel coordinates")
    x, y, width, height = map(int, values)
    with rio.open(path) as src:
        if src.count != 4 or src.dtypes != ("uint8",) * 4 or not src.crs or not src.crs.is_projected or src.transform.b or src.transform.d:
            raise ValueError("Require a north-up projected uint8 RGBA baseline")
        if x < 0 or y < 0 or width < 3 or height < 3 or x + width > src.width or y + height > src.height:
            raise ValueError("Evidence ROI is outside the actual baseline")
        win = Window(x, y, width, height)
        if grid.get("crs") != src.crs.to_string() or grid.get("referenceWidth") != src.width or grid.get("referenceHeight") != src.height:
            raise ValueError("Evidence baseline CRS or extent differs")
        for key, expected in (("transform", list(src.window_transform(win))[:6]),
                              ("referenceTransform", list(src.transform)[:6])):
            actual = np.asarray(grid.get(key, []))
            if actual.shape != (6,) or not np.issubdtype(actual.dtype, np.number) or not np.isfinite(actual).all() or not np.allclose(actual, expected, rtol=0, atol=1e-10):
                raise ValueError("Evidence uses different baseline map pixels")
        if grid.get("width") != width or grid.get("height") != height:
            raise ValueError("Evidence ROI dimensions differ")
        return src.read(window=win).transpose(1, 2, 0), (x, y, width, height)


def review_certificates(certificate_path, *, source_loader: Callable | None = None, baseline_loader: Callable | None = None):
    """Strict read-only review; optional loaders only support deterministic tests."""
    path = Path(certificate_path).resolve()
    pins = {str(path): file_sha256(path)}
    certificate = _json(path)
    if certificate.get("schema") != CERTIFICATE_SCHEMA or certificate.get("state") != "complete":
        raise ValueError("Caller certificate manifest is not complete or supported")
    if "policyConfig" in certificate:
        raise ValueError("Certificates may not override or relax policy guards")
    base_binding = certificate.get("baseline")
    if not isinstance(base_binding, dict):
        raise ValueError("Require an explicitly pinned retained baseline")
    baseline = _pin(_path(base_binding["path"], path.parent), base_binding["sha256"], pins)
    entries = certificate.get("patches")
    if not isinstance(entries, list) or any(not isinstance(p, dict) for p in entries):
        raise ValueError("Require an explicit patch certificate list")
    ids = [p.get("id") for p in entries]
    if any(not isinstance(identity, str) or not re.fullmatch("[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", identity) for identity in ids) or len(set(ids)) != len(ids):
        raise ValueError("Patch IDs must be safe and unique")
    source_loader, baseline_loader = source_loader or _load_bound_source, baseline_loader or _baseline_roi
    accepted, records = [], []
    for entry in entries:
        binding = entry["evidenceManifest"]
        evidence_path = _pin(_path(binding["path"], path.parent), binding["sha256"], pins)
        evidence = _json(evidence_path)
        if evidence.get("schema") != "ogs2.corridor-projection/v1" or evidence.get("state") != "complete":
            raise ValueError("Only completed actual projection manifests are accepted")
        for input_path, expected in {**evidence["criticalInputs"], **evidence["sourceInputHashes"]}.items():
            _pin(_path(input_path, evidence_path.parent), expected, pins)
        if pins.get(str(baseline)) != evidence["criticalInputs"].get(str(baseline)):
            raise ValueError("Evidence is not bound to the actual retained baseline")
        rgba, window = baseline_loader(baseline, evidence["grid"])
        snapshot = _pin(evidence_path.parent / "baseline-roi.npz", evidence["snapshotSha256"], pins)
        with np.load(snapshot, allow_pickle=False) as archive:
            original = archive["rgba"]
            requested_domain = archive["corridor"]
        if not np.array_equal(original, rgba) or original.dtype != np.uint8 or requested_domain.dtype != np.bool_ or requested_domain.shape != rgba.shape[:2]:
            raise ValueError("Actual baseline ROI differs from preserved projection evidence")
        masks = load_certificate_arrays(entry["certificateArrays"], path.parent, pins)
        shape = rgba.shape[:2]
        roles = entry["masks"]
        conflict = _mask(masks, roles["conflict_union"], shape)
        road = _mask(masks, roles["road_domain"], shape)
        unknown = _mask(masks, roles["protected_unknown"], shape)
        if not conflict.any() or (conflict & ~requested_domain).any():
            raise ValueError("Conflict seed exceeds the evidence corridor domain")
        foregrounds = _catalog(masks, roles.get("baseline_foregrounds", {}), shape)
        candidates, clocks = [], set()
        source_entries = entry.get("candidates")
        if not isinstance(source_entries, list):
            raise ValueError("Require explicit actual-source certificates")
        for source_entry in source_entries:
            arrays, record, actual_manifest = source_loader(evidence_path, source_entry["sourceImage"])
            if actual_manifest != evidence:
                raise ValueError("Bound source loader returned another evidence manifest")
            for candidate_path, expected in ((record["cachePath"], record["cacheSha256"]),
                                              (str(Path(record["cachePath"]).with_suffix(".json")), record["sourceMetadataSha256"])):
                _pin(_path(candidate_path, evidence_path.parent), expected, pins)
            clocks.add(record.get("captureTimeDomain"))
            candidates.append(candidate_from_certificate(source_entry, arrays, record, masks, rgba, conflict))
        if len(clocks) > 1 or (clocks and (None in clocks or "unknown" in clocks)):
            raise ValueError("Candidate exposure times do not share one verified clock domain")
        plan, decisions = select_corridor_source(rgba, conflict, road, candidates,
                                                 baseline_foregrounds=foregrounds, protected_unknown=unknown)
        record = {"patchId": entry["id"], "evidenceManifest": str(evidence_path),
                  "accepted": plan is not None, "candidateDecisions": decisions, "window": list(window)}
        if plan is None:
            record["reason"] = "no-physically-supported-real-exposure"
        else:
            output, mode, application = compose_corridor(rgba, plan)
            record.update(reason="verified-one-exposure-road-region", sourceId=plan.source.source_id,
                          selectedSourceReport=application)
            accepted.append(ReviewedPatch(entry["id"], window, evidence["grid"], rgba, output,
                                           mode.astype(bool), plan.source.rgb, plan.source.source_id, record,
                                           plan.static_collar.copy()))
        records.append(record)
    accepted, conflicts = reject_overlapping_patches(accepted)
    conflicting_ids = {p[key] for p in conflicts for key in ("first", "second")}
    for record in records:
        if record["patchId"] in conflicting_ids:
            record.update(accepted=False, reason="conflicting-write-or-static-reservation-supports")
    _verify_pins(pins)
    return baseline, accepted, {"schema": REPORT_SCHEMA, "certificateManifest": str(path),
                               "certificateManifestSha256": pins[str(path)], "baseline": str(baseline),
                               "baselineSha256": pins[str(baseline)], "patches": records,
                               "overlapConflicts": conflicts, "acceptedPatches": len(accepted),
                               "inputs": pins, "allInputHashesVerifiedAfterReview": True,
                               "globalInstantClaimed": False, "identityAssociationAssumed": False}


def _intersection(block, patch):
    bx, by, bw, bh = map(int, block.flatten())
    px, py, pw, ph = patch.window
    left, top, right, bottom = max(bx, px), max(by, py), min(bx + bw, px + pw), min(by + bh, py + ph)
    if left >= right or top >= bottom:
        return None
    return ((slice(top - by, bottom - by), slice(left - bx, right - bx)),
            (slice(top - py, bottom - py), slice(left - px, right - px)))


def export_reviewed_full(baseline_path, patches, output_dir, pins, source_ids):
    """Losslessly copy every baseline block; overwrite only accepted real support."""
    import rasterio as rio
    _, conflicts = reject_overlapping_patches(patches)
    if conflicts:
        raise RuntimeError("Accepted writers overlap a write/static reservation")
    output_dir = Path(output_dir)
    paths = {name: output_dir / filename for name, filename in (
        ("orthophoto", "corridor_orthophoto.partial.tif"),
        ("mode", "corridor-mode.partial.tif"), ("source", "corridor-sources.partial.tif"))}
    changed = outside = alpha = source_errors = written = 0
    with rio.open(baseline_path) as base:
        if base.count != 4 or base.dtypes != ("uint8",) * 4:
            raise ValueError("Full export requires retained uint8 RGBA")
        profile = {**base.profile, "tiled": True, "blockxsize": 512, "blockysize": 512,
                   "compress": "DEFLATE", "BIGTIFF": "IF_SAFER"}
        with rio.open(paths["orthophoto"], "w", **profile) as dst, \
                rio.open(paths["mode"], "w", **{**profile, "count": 1, "dtype": "uint8", "nodata": 0}) as mode_dst, \
                rio.open(paths["source"], "w", **{**profile, "count": 1, "dtype": "uint16", "nodata": 0}) as source_dst:
            dst.colorinterp = base.colorinterp
            dst.update_tags(**base.tags())
            for band in range(1, 5):
                dst.update_tags(band, **base.tags(band))
                dst.set_band_description(band, base.descriptions[band - 1] or "")
            for _, block in dst.block_windows(1):
                data = base.read(window=block)
                block_shape = data.shape[1:]
                modes, sources = np.zeros(block_shape, np.uint8), np.zeros(block_shape, np.uint16)
                for patch in patches:
                    overlap = _intersection(block, patch)
                    if overlap is None:
                        continue
                    local, region = overlap
                    support = patch.hard_support[region]
                    if (modes[local][support] != 0).any():
                        raise RuntimeError("Two accepted writers overlap during export")
                    target = data[:3, local[0], local[1]]
                    source = patch.source_rgb[region].transpose(2, 0, 1)
                    target[:, support] = source[:, support]
                    modes[local][support] = 1
                    sources[local][support] = source_ids[patch.source_id]
                dst.write(data, window=block)
                mode_dst.write(modes, 1, window=block)
                source_dst.write(sources, 1, window=block)
        with rio.open(paths["orthophoto"]) as actual, rio.open(paths["mode"]) as mode_src, rio.open(paths["source"]) as source_src:
            if actual.crs != base.crs or actual.transform != base.transform or actual.shape != base.shape:
                raise RuntimeError("Full output changes the retained map grid")
            for _, block in actual.block_windows(1):
                output, original = actual.read(window=block), base.read(window=block)
                modes, sources = mode_src.read(1, window=block), source_src.read(1, window=block)
                allowed = np.zeros(modes.shape, bool)
                expected_sources = np.zeros(sources.shape, np.uint16)
                for patch in patches:
                    overlap = _intersection(block, patch)
                    if overlap is None:
                        continue
                    local, region = overlap
                    support = patch.hard_support[region]
                    allowed[local] |= support
                    expected_sources[local][support] = source_ids[patch.source_id]
                    rgb = output[:3, local[0], local[1]]
                    real = patch.source_rgb[region].transpose(2, 0, 1)
                    source_errors += int(np.any(rgb[:, support] != real[:, support], axis=0).sum())
                different = np.any(output != original, axis=0)
                changed += int(different.sum())
                outside += int((different & ~allowed).sum())
                alpha += int((output[3] != original[3]).sum())
                written += int(allowed.sum())
                if not np.array_equal(modes, allowed.astype(np.uint8)) or not np.array_equal(sources, expected_sources):
                    raise RuntimeError("Source/mode raster differs from actual accepted support")
        if outside or alpha or source_errors:
            raise RuntimeError("Full export violated outside/alpha/real-source invariants")
    _verify_pins(pins)
    final = {}
    for name, partial in paths.items():
        target = partial.with_name(partial.name.replace(".partial.tif", ".tif"))
        partial.replace(target)
        final[name] = {"path": str(target), "sha256": file_sha256(target)}
    return {"files": final, "changedPixels": changed, "writtenPixels": written,
            "outsideSupportChangedPixels": outside, "alphaChangedPixels": alpha,
            "realSourceRgbMismatchPixels": source_errors, "allInputHashesVerifiedAfterExport": True}


def apply_reviewed_certificates(certificate_path, output, *, full_export=False):
    started = time.monotonic()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Fresh output required; do not overwrite a prior result")
    # Review completes before any candidate output is created.
    base, patches, report = review_certificates(certificate_path)
    for path in map(Path, report["inputs"]):
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Output overlaps retained inputs")
    output.mkdir(parents=True, exist_ok=False)
    try:
        source_ids = {name: index + 1 for index, name in enumerate(sorted({p.source_id for p in patches}))}
        if len(source_ids) > 65535:
            raise ValueError("Source index exceeds uint16 provenance capacity")
        report["sourceIds"] = source_ids
        report["fullExportRequested"] = full_export
        report["candidateOutputs"] = []
        if patches:
            import rasterio as rio
            from rasterio.transform import Affine
            for patch in patches:
                _verify_patch_snapshot(patch)
                folder = output / patch.identity
                folder.mkdir()
                rgba_path, mode_path, sources_path = [folder / name for name in (
                    "candidate-orthophoto.tif", "write-mode.tif", "source-index.tif")]
                height, width = patch.hard_support.shape
                profile = {"driver": "GTiff", "height": height, "width": width,
                           "crs": patch.grid["crs"], "transform": Affine(*patch.grid["transform"]),
                           "tiled": True, "blockxsize": 256, "blockysize": 256,
                           "compress": "DEFLATE", "BIGTIFF": "IF_SAFER"}
                with rio.open(rgba_path, "w", **profile, count=4, dtype="uint8") as dst:
                    dst.write(patch.output_rgba.transpose(2, 0, 1))
                    with rio.open(base) as retained:
                        dst.colorinterp = retained.colorinterp
                with rio.open(mode_path, "w", **profile, count=1, dtype="uint8", nodata=0) as dst:
                    dst.write(patch.hard_support.astype(np.uint8), 1)
                with rio.open(sources_path, "w", **profile, count=1, dtype="uint16", nodata=0) as dst:
                    dst.write(patch.hard_support.astype(np.uint16) * source_ids[patch.source_id], 1)
                with rio.open(rgba_path) as dst:
                    actual = dst.read().transpose(1, 2, 0)
                    if not np.array_equal(actual, patch.output_rgba):
                        raise RuntimeError("Candidate TIFF changed accepted pixels")
                with rio.open(mode_path) as dst:
                    if not np.array_equal(dst.read(1), patch.hard_support.astype(np.uint8)):
                        raise RuntimeError("Candidate mode raster changed accepted write support")
                with rio.open(sources_path) as dst:
                    if not np.array_equal(dst.read(1), patch.hard_support.astype(np.uint16) * source_ids[patch.source_id]):
                        raise RuntimeError("Candidate source raster changed source provenance")
                source_mismatch = int(np.any(actual[..., :3][patch.hard_support] != patch.source_rgb[patch.hard_support], axis=1).sum())
                outside = int(np.any(actual[~patch.hard_support] != patch.baseline_rgba[~patch.hard_support], axis=1).sum())
                alpha = int((actual[..., 3] != patch.baseline_rgba[..., 3]).sum())
                if source_mismatch or outside or alpha:
                    raise RuntimeError("Candidate readback violates actual source/outside/alpha invariants")
                _save_json(folder / "decision.json", patch.report)
                report["candidateOutputs"].append({"patchId": patch.identity,
                                                    "orthophoto": str(rgba_path), "mode": str(mode_path),
                                                    "sources": str(sources_path), "sourceId": patch.source_id,
                                                    "writtenPixels": int(patch.hard_support.sum()),
                                                    "changedPixels": int(np.any(actual != patch.baseline_rgba, axis=2).sum()),
                                                    "alphaChangedPixels": alpha, "outsideSupportChangedPixels": outside,
                                                    "realSourceRgbMismatchPixels": source_mismatch})
            _verify_pins(report["inputs"])
            if full_export:
                report["fullExport"] = export_reviewed_full(base, patches, output, report["inputs"], source_ids)
        else:
            report["fullExport"] = None
            report["reason"] = "no-accepted-patches-no-new-full-mosaic-created"
        _verify_pins(report["inputs"])
        report.update(state="complete", elapsedSeconds=time.monotonic() - started,
                      allInputHashesVerifiedAtEnd=True)
        _save_json(output / "review.json", report)
        return report
    except Exception as error:
        _save_json(output / "status.json", {"schema": REPORT_SCHEMA, "state": "failed",
                                            "error": str(error), "elapsedSeconds": time.monotonic() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-evidence-certificates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-export", action="store_true")
    args = parser.parse_args()
    report = apply_reviewed_certificates(args.review_evidence_certificates, args.output, full_export=args.full_export)
    print(json.dumps({"state": report["state"], "acceptedPatches": report["acceptedPatches"],
                      "report": str(args.output / "review.json"), "fullExport": report.get("fullExport")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
