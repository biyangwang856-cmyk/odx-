"""One-time validation of actual joint orthophoto composition and photo evidence.

The manifest supplies explicit evaluation masks and cache locations. It never
changes production parameters. Mask preservation proves observed pixels were
retained, not a universal vehicle/shadow segmentation guarantee.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from html import escape
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import rasterio as rio
from rasterio.enums import ColorInterp
from rasterio.windows import Window
from scipy import ndimage


RASTERS = {
    "new": "ogs2_orthophoto.tif", "source": "source-index.tif", "weight": "source-weight.tif",
    "mode": "composition-mode.tif", "hard": "protected-objects.tif", "write": "write-support.tif",
}


def required(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing actual validation input: {path}")
    return path


def file_hash(path):
    value = sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def same_grid(first, second):
    return first.crs == second.crs and first.width == second.width and first.height == second.height \
        and first.transform.almost_equals(second.transform, precision=1e-8)


def offset(reference, target):
    if reference.crs != target.crs or not np.allclose(
        [reference.transform.a, reference.transform.b, reference.transform.d, reference.transform.e],
        [target.transform.a, target.transform.b, target.transform.d, target.transform.e],
        rtol=0, atol=1e-10,
    ):
        raise ValueError("Actual outputs must use the original map CRS and pixel scale/rotation")
    x, y = ~reference.transform * (target.transform.c, target.transform.f)
    if abs(x - round(x)) > 1e-6 or abs(y - round(y)) > 1e-6:
        raise ValueError("Output does not use original map pixels")
    x, y = round(x), round(y)
    if x < 0 or y < 0 or x + target.width > reference.width or y + target.height > reference.height:
        raise ValueError("Output lies outside the reference grid")
    return x, y


def resolve(value, manifest_parent):
    path = Path(value)
    return required(path if path.is_absolute() else manifest_parent / path)


def mask_from(description, shape, window, result, manifest_parent, inputs):
    """Explicit NPZ/Numpy local masks or map-aligned TIFF evaluation masks."""
    description = {"path": description} if isinstance(description, str) else description
    path = resolve(description["path"], manifest_parent)
    inputs.add(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with rio.open(path) as src:
            target_transform = result.window_transform(window)
            if src.crs != result.crs:
                raise ValueError("Evaluation mask CRS differs")
            if src.width == shape[1] and src.height == shape[0] and src.transform.almost_equals(
                target_transform, precision=1e-8
            ):
                data = src.read(1)
            elif same_grid(src, result):
                data = src.read(1, window=window)
            else:
                raise ValueError("Evaluation mask must match the explicit patch or full output grid")
    elif path.suffix.lower() == ".npz":
        if "key" not in description:
            raise ValueError("NPZ evaluation mask requires an explicit key")
        with np.load(path, allow_pickle=False) as archive:
            data = archive[description["key"]]
    elif path.suffix.lower() == ".npy":
        data = np.load(path, allow_pickle=False)
    else:
        raise ValueError("Evaluation masks must be TIFF, NPZ or NPY")
    if path.suffix.lower() in {".npz", ".npy"} and description.get("scope") == "full":
        if data.shape != (result.height, result.width):
            raise ValueError("Declared full evaluation mask does not match output dimensions")
        x, y, width, height = map(int, window.flatten())
        data = data[y:y + height, x:x + width]
    if data.shape != shape or not np.isfinite(data).all():
        raise ValueError("Evaluation mask has a different shape or nonfinite pixels")
    return data.astype(bool)


def cache_from(patch, shape, window, result, manifest_parent, inputs):
    path = resolve(patch["sourceCache"], manifest_parent)
    inputs.add(path)
    with np.load(path, allow_pickle=False) as archive:
        image, valid, cost = archive["image"], archive["valid"], archive["cost"]
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Source cache must contain uint8 RGB image pixels")
    if valid.dtype != bool or cost.dtype != np.float32:
        raise ValueError("Source cache valid/cost must be bool/float32")
    cache_window = patch.get("cacheWindow")
    if cache_window is None:
        if image.shape[:2] == shape:
            cache_window = list(window.flatten())
        elif image.shape[:2] == (result.height, result.width):
            cache_window = [0, 0, result.width, result.height]
        else:
            raise ValueError("Cache pixel location is ambiguous; provide cacheWindow")
    x, y, width, height = map(int, cache_window)
    if image.shape[:2] != (height, width) or valid.shape != (height, width) or cost.shape != (height, width):
        raise ValueError("Source cache dimensions differ from its declared map window")
    left, top = int(window.col_off) - x, int(window.row_off) - y
    if left < 0 or top < 0 or left + shape[1] > width or top + shape[0] > height:
        raise ValueError("Source cache does not cover the exact evaluation window")
    section = np.s_[top:top + shape[0], left:left + shape[1]]
    return image[section], valid[section], cost[section], path


def boundary(mask):
    return mask & ~ndimage.binary_erosion(mask)


def save_panel(images, labels, path):
    height, width = images[0].shape[:2]
    canvas = Image.new("RGB", (width * len(images), height + 64), "white")
    draw = ImageDraw.Draw(canvas)
    face = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        canvas.paste(Image.fromarray(image), (index * width, 64))
        draw.text((index * width + 8, 20), label, fill="black", font=face)
    canvas.save(path, quality=95)
    canvas.thumbnail((2304, 2304))
    canvas.save(path.with_name(path.stem + "-preview.jpg"), quality=95)


def global_checks(baseline_path, result_paths, source_map):
    totals = {name: 0 for name in (
        "pixels", "alphaDifferentPixels", "outsideWriteDifferentPixels", "baselineModeDifferentPixels",
        "sourceZeroDifferentPixels", "invalidWeightPixels", "unknownModePixels", "baseModeSemanticFailures",
        "pureModeSemanticFailures", "mixModeSemanticFailures", "hardSemanticFailures",
        "writeSemanticFailures",
        "changedPixels", "writePixels", "hardPixels",
    )}
    counts = {}
    with rio.open(baseline_path) as baseline, rio.open(result_paths["new"]) as result:
        if baseline.count != 4 or result.count != 4:
            raise ValueError("Baseline and actual new result require RGB+alpha")
        if baseline.colorinterp[3] != ColorInterp.alpha or result.colorinterp[3] != ColorInterp.alpha:
            raise ValueError("Band 4 must be explicit alpha")
        x, y = offset(baseline, result)
        readers = {name: rio.open(path) for name, path in result_paths.items() if name != "new"}
        try:
            if not all(same_grid(src, result) for src in readers.values()):
                raise ValueError("Mode, source, weight, hard and write rasters must share output pixels")
            if readers["source"].dtypes[0] != "uint16" or readers["weight"].dtypes[0] != "float32":
                raise ValueError("Source index and weight must use uint16 and float32")
            for _, block in result.block_windows(1):
                old = baseline.read(window=Window(
                    x + block.col_off, y + block.row_off, block.width, block.height
                ))
                new = result.read(window=block)
                data = {name: src.read(1, window=block) for name, src in readers.items()}
                ids, weight, mode = data["source"], data["weight"], data["mode"]
                hard, write = data["hard"] > 0, data["write"] > 0
                changed = np.any(old != new, axis=0)
                base, pure, mix = mode == 0, mode == 1, mode == 2
                totals["pixels"] += ids.size
                totals["alphaDifferentPixels"] += int((old[3] != new[3]).sum())
                totals["outsideWriteDifferentPixels"] += int((changed & ~write).sum())
                totals["baselineModeDifferentPixels"] += int((changed & base).sum())
                totals["sourceZeroDifferentPixels"] += int((changed & (ids == 0)).sum())
                totals["invalidWeightPixels"] += int(
                    (~np.isfinite(weight) | (weight < 0) | (weight > 1)).sum()
                )
                totals["unknownModePixels"] += int((~(base | pure | mix)).sum())
                totals["baseModeSemanticFailures"] += int(
                    (base & ((ids != 0) | (weight != 0) | hard | write)).sum()
                )
                totals["pureModeSemanticFailures"] += int(
                    (pure & ((ids == 0) | (weight != 1) | ~write)).sum()
                )
                totals["mixModeSemanticFailures"] += int(
                    (mix & ((ids == 0) | (weight <= 0) | (weight >= 1) | hard | ~write)).sum()
                )
                totals["hardSemanticFailures"] += int((hard & (~pure | (weight != 1) | (ids == 0))).sum())
                totals["writeSemanticFailures"] += int((write != (pure | mix)).sum())
                totals["changedPixels"] += int(changed.sum())
                totals["writePixels"] += int(write.sum())
                totals["hardPixels"] += int(hard.sum())
                unique, numbers = np.unique(ids, return_counts=True)
                for identity, number in zip(unique.tolist(), numbers.tolist(), strict=True):
                    counts[identity] = counts.get(identity, 0) + number
        finally:
            for reader in readers.values():
                reader.close()
    unknown = [identity for identity in counts if identity != 0 and identity not in source_map]
    diagnostics = {"pixels", "changedPixels", "writePixels", "hardPixels"}
    return {
        **totals, "unknownSourceIds": unknown,
        "sourcePixels": [{"id": identity, "image": "baseline" if identity == 0 else source_map.get(identity),
                          "pixels": count} for identity, count in sorted(counts.items())],
        "checksPassed": not unknown and all(
            value == 0 for name, value in totals.items() if name not in diagnostics
        ),
    }


def validate_patch(patch, dataset_paths, baseline_path, old_ogs_path, manifest, parent, output, inputs):
    window = Window(*patch["window"])
    if any(value != int(value) for value in window.flatten()) or window.width <= 0 or window.height <= 0:
        raise ValueError("Evaluation window must have positive integer dimensions")
    shape = int(window.height), int(window.width)
    with (
        rio.open(dataset_paths["new"]) as result,
        rio.open(baseline_path) as baseline,
        rio.open(old_ogs_path) as old,
    ):
        x, y = offset(baseline, result)
        ox, oy = offset(old, result)
        if window.col_off < 0 or window.row_off < 0 or window.col_off + window.width > result.width \
                or window.row_off + window.height > result.height:
            raise ValueError("Evaluation window is outside actual output")
        baseline_rgb = baseline.read([1, 2, 3], window=Window(
            x + window.col_off, y + window.row_off, window.width, window.height
        )).transpose(1, 2, 0)
        old_rgb = old.read([1, 2, 3], window=Window(
            ox + window.col_off, oy + window.row_off, window.width, window.height
        )).transpose(1, 2, 0)
        new_rgb = result.read([1, 2, 3], window=window).transpose(1, 2, 0)
        image, valid, cost, cache_path = cache_from(patch, shape, window, result, parent, inputs)
        mask_names = ("hardMask", "writeMask", "retainedTrackMask", "measuredChange")
        optional_mask_names = (
            "discardedTrackMask", "selectedShadowMask", "discardedShadowMask", "backgroundEvidenceMask",
        )
        masks = {
            name: mask_from(patch[name], shape, window, result, parent, inputs) for name in mask_names
        }
        missing_evidence = [name for name in optional_mask_names if not patch.get(name)]
        masks.update({
            name: mask_from(patch[name], shape, window, result, parent, inputs)
            if patch.get(name) else np.zeros(shape, bool)
            for name in optional_mask_names
        })
        independent_vehicles = [
            (entry["id"], mask_from(entry["mask"], shape, window, result, parent, inputs))
            for entry in patch.get("independentVehicleMasks", [])
        ]
    arrays = {}
    for name in ("source", "weight", "mode", "hard", "write"):
        with rio.open(dataset_paths[name]) as src:
            arrays[name] = src.read(1, window=window)
    chosen = int(patch["sourceIndex"])
    source_map = {int(identity): name for identity, name in manifest["sourceIndex"].items()}
    if chosen <= 0 or source_map.get(chosen) != patch["sourceImage"]:
        raise ValueError("Chosen source index does not map to the actual source image")
    photo_path = resolve(patch["sourcePhoto"], parent)
    if photo_path.name != patch["sourceImage"]:
        raise ValueError("Original source photo filename differs from chosen image")
    inputs.add(photo_path)
    photo_hash = file_hash(photo_path)
    metadata_path = cache_path.with_suffix(".json")
    cache_provenance_verified = False
    if metadata_path.exists():
        inputs.add(metadata_path)
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("sourceImageSha256") != photo_hash:
            raise ValueError("Projection cache provenance differs from the actual chosen source photograph")
        cache_provenance_verified = True
    ids, weight, mode = arrays["source"], arrays["weight"], arrays["mode"]
    actual_hard, actual_write = arrays["hard"] > 0, arrays["write"] > 0
    hard, write = masks["hardMask"], masks["writeMask"]
    retained, discarded = masks["retainedTrackMask"], masks["discardedTrackMask"]
    dynamic = retained | discarded | masks["measuredChange"] | masks["selectedShadowMask"] \
        | masks["discardedShadowMask"]
    background = (discarded & ~retained) | (
        masks["discardedShadowMask"] & ~masks["selectedShadowMask"] & ~retained
    )
    # The actual new experiment inherits the old OGS2 result. The original
    # 2.5D remains an independent display/reference column, not the blend base.
    expected = old_rgb.astype(np.float32)
    if manifest["blendFormula"] == "base-plus-difference":
        expected += weight[..., None] * (image.astype(np.float32) - expected)
    elif manifest["blendFormula"] == "weighted-sum":
        expected = expected * (1 - weight[..., None]) + image.astype(np.float32) * weight[..., None]
    else:
        raise ValueError("Unknown declared production blend formula")
    if manifest["blendRounding"] == "nearest":
        expected = np.rint(expected)
    elif manifest["blendRounding"] != "floor":
        raise ValueError("Unknown declared production blend rounding")
    expected = np.clip(expected, 0, 255).astype(np.uint8)
    different_source = np.any(new_rgb != image, axis=2)
    different_expected = np.any(new_rgb != expected, axis=2)
    checks = {
        "declaredHardMissingFromActualPixels": int((hard & ~actual_hard).sum()),
        "declaredWriteMissingFromActualPixels": int((write & ~actual_write).sum()),
        "hardNotSingleChosenSourcePixels": int((hard & (ids != chosen)).sum()),
        "hardNotPurePixels": int((hard & ((mode != 1) | (weight != 1))).sum()),
        "hardDiffersFromRealCachedSourcePixels": int((hard & different_source).sum()),
        "hardWithoutValidSourcePixels": int((hard & ~valid).sum()),
        "writeWithoutValidSourcePixels": int((write & ~valid).sum()),
        "writeWithNonfiniteProjectionCostPixels": int((write & ~np.isfinite(cost)).sum()),
        "writeNotChosenSourcePixels": int((write & (ids != chosen)).sum()),
        "writeBlendFormulaDifferentPixels": int((write & different_expected).sum()),
        "retainedTrackOutsideHardPixels": int((retained & ~hard).sum()),
        "dynamicOrShadowOutsideHardPixels": int((dynamic & ~hard).sum()),
        "discardedPositionWithoutBackgroundEvidencePixels": int(
            (background & ~masks["backgroundEvidenceMask"]).sum()
        ),
        "discardedPositionDiffersFromChosenBackgroundPixels": int((background & different_source).sum()),
        "discardedPositionWithoutValidSourcePixels": int((background & ~valid).sum()),
    }
    vehicle_boundaries = []
    for identity, mask in independent_vehicles:
        count, written = int(mask.sum()), int((mask & actual_write).sum())
        vehicle_boundaries.append({
            "id": identity, "observedPixels": count, "writtenPixels": written,
            "partlyCutByWriteBoundary": 0 < written < count,
            "writtenVehicleNotPurePixels": int((mask & actual_write & (mode != 1)).sum()),
            "writtenVehicleOutsideHardPixels": int((mask & actual_write & ~actual_hard).sum()),
        })
    checks["independentVehicleBoundaryFailures"] = sum(
        row["partlyCutByWriteBoundary"] or row["writtenVehicleNotPurePixels"] > 0
        or row["writtenVehicleOutsideHardPixels"] > 0 for row in vehicle_boundaries
    )
    _, retained_components = ndimage.label(retained)
    identifier = str(patch["id"])
    if Path(identifier).name != identifier or identifier in ("", ".", ".."):
        raise ValueError("Patch IDs must be simple artifact names")
    save_panel(
        [baseline_rgb, old_rgb, new_rgb, image],
        ["Original 2.5D", "Old OGS2 (actual)", "New actual output", "Chosen real photo cache"],
        output / (identifier + "-four-sources.jpg"),
    )
    shown = new_rgb.copy()
    shown[boundary(dynamic)] = [255, 0, 255]
    shown[boundary(retained)] = [255, 255, 0]
    shown[boundary(background)] = [0, 255, 255]
    Image.fromarray(shown).save(output / (identifier + "-mask-evidence.jpg"), quality=95)
    ys, xs = np.where(dynamic)
    if len(xs):
        left, top, right, bottom = max(0, xs.min() - 12), max(0, ys.min() - 12), \
            min(shape[1], xs.max() + 13), min(shape[0], ys.max() + 13)
        save_panel(
            [view[top:bottom, left:right] for view in (baseline_rgb, old_rgb, new_rgb, image)],
            ["2.5D", "Old OGS2", "New actual", "Chosen source"], output / (identifier + "-bus-shadow.jpg"),
        )
    return {
        "id": identifier, "window": list(window.flatten()), "sourceImage": patch["sourceImage"],
        "sourceIndex": chosen, "sourceCache": str(cache_path), "sourceCacheSha256": file_hash(cache_path),
        "sourcePhoto": str(photo_path), "sourcePhotoSha256": photo_hash,
        "sourceCachePhotoHashVerified": cache_provenance_verified,
        "retainedObservedTrackPixels": int(retained.sum()), "retainedObservedComponents": retained_components,
        "discardedObservedTrackPixels": int(discarded.sum()),
        "backgroundEvidencePixels": int(background.sum()),
        "selectedShadowEvidencePixels": int(masks["selectedShadowMask"].sum()),
        "discardedShadowEvidencePixels": int(masks["discardedShadowMask"].sum()),
        "checks": checks,
        "checksPassed": bool(hard.any() and write.any() and retained.any())
        and all(value == 0 for value in checks.values()),
        "shadowAndRemovedPositionEvidenceComplete": not missing_evidence and bool(
            discarded.any() and masks["selectedShadowMask"].any() and masks["backgroundEvidenceMask"].any()
        ),
        "missingIndependentEvidence": missing_evidence,
        "independentVehicleBoundaries": vehicle_boundaries,
        "manualAcceptance": "pending real-photo review; mask preservation is not physical completeness proof",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "old-ogs", "result", "manifest", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError("Use a new validation directory; completed evidence is never overwritten")
    baseline, old_ogs = required(args.baseline), required(args.old_ogs)
    manifest_path = required(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    if not manifest.get("patches"):
        raise ValueError("Validation requires explicit actual accepted patches and evaluation evidence")
    if json.loads(required(args.result / "status.json").read_text()).get("state") != "success":
        raise ValueError("Only completed actual production outputs can be validated")
    paths = {name: required(args.result / filename) for name, filename in RASTERS.items()}
    if "sourceIndex" not in manifest:
        manifest["sourceIndex"] = json.loads(required(args.result / "report.json").read_text())["sourceIndex"]
    source_map = {int(identity): name for identity, name in manifest["sourceIndex"].items()}
    inputs = {baseline, old_ogs, manifest_path, *paths.values()}
    global_result = global_checks(old_ogs, paths, source_map)
    output.mkdir(parents=True, exist_ok=False)
    rows = [validate_patch(
        patch, paths, baseline, old_ogs, manifest, manifest_path.parent, output, inputs
    ) for patch in manifest["patches"]]
    passed = global_result["checksPassed"] and all(row["checksPassed"] for row in rows)
    report = {
        "schemaVersion": "joint-composition-validation/v1", "checksPassed": passed,
        "compositionBase": str(old_ogs), "originalReference2_5D": str(baseline),
        "globalIntegrity": global_result, "patches": rows,
        "photoPixelVerificationScope": "The explicit patches and write masks supplied in this manifest.",
        "allCachesLinkedToActualPhotoHashes": all(row["sourceCachePhotoHashVerified"] for row in rows),
        "allShadowAndRemovedPositionEvidenceComplete": all(
            row["shadowAndRemovedPositionEvidenceComplete"] for row in rows
        ),
        "manualAcceptance": "pending independent real-photo and actual-output inspection",
        "inputs": [{"path": str(path), "sha256": file_hash(path)} for path in sorted(inputs)],
        "scriptSha256": file_hash(__file__),
        "limitations": [
            "Evaluation masks and selected-photo provenance need independent real-photo review.",
            "Numerical checks prove composition and observed-mask preservation, "
            "not universal vehicle identity.",
            "Shadow-support completeness is limited by the supplied measured/evaluation masks.",
            "No generated background, fitted image warp, production threshold change or fewer-vehicle score.",
        ],
    }
    (output / "validation-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    articles = "".join(
        f'<article><h2>{escape(row["id"])}</h2>'
        f'<a href="{escape(row["id"])}-four-sources.jpg">'
        f'<img src="{escape(row["id"])}-four-sources-preview.jpg"></a>'
        f'<p>组合与观测掩码核查：{row["checksPassed"]}。仍需人工核查完整车身、唯一身份、背景与车影。</p>'
        f'<p>独立车影及旧位置证据齐全：{row["shadowAndRemovedPositionEvidenceComplete"]}。</p>'
        f'<a href="{escape(row["id"])}-bus-shadow.jpg">公交与车影放大</a> · '
        f'<a href="{escape(row["id"])}-mask-evidence.jpg">区域证据</a></article>' for row in rows
    )
    html = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>真实车辆组合验收</title>'
        '<style>body{font-family:system-ui;max-width:1500px;margin:30px auto;background:#f5f6f8}'
        'article{background:white;padding:20px;margin:20px 0}img{width:100%;height:auto}</style>'
        '<h1>真实车辆与车影组合核查</h1><p>原2.5D／旧OGS2真实裁剪／新实际结果／所选原照片缓存。'
        '同一原始地理像元；没有拟合对齐、生成式背景或少车质量评分。</p>'
        f'<p>数值约束核查：{passed}；不能据此声称通用车辆身份和物理阴影已完全解决。</p>'
        + articles + '</html>'
    )
    (output / "index.html").write_text(html)
    print(json.dumps({"output": str(output), "checksPassed": passed, "patches": len(rows)}))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
