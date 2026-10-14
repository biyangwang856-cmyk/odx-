"""Composite the native road replacement into the unchanged original grid.

Run with the vehicle Python environment. Original input rasters are read-only.
Every output block is reopened and checked against the intended replacement.
"""

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio as rio
from PIL import Image, ImageDraw
from rasterio.windows import Window

p = argparse.ArgumentParser()
p.add_argument("--job", type=Path, required=True)
p.add_argument("--candidate", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(exist_ok=False)
started = time.time()
quality = json.loads((a.candidate / "report.json").read_text())
if quality["state"] != "completed" or quality["scale"] != 1:
    raise ValueError("A completed native-resolution candidate is required")
scope = np.load(a.candidate / "replacement-scope.npz")["replacement"]
original = a.job / "vehicle-ogs2/ogs2_orthophoto.tif"
candidate = a.candidate / "CANDIDATE-full-highway-corridor.tif"
output = a.output / "CANDIDATE-main-highway-orthophoto.tif"
changed = 0
replacement_count = 0
blocks = 0
with rio.open(original) as src, rio.open(candidate) as road:
    extent = Window(0, 17000, 24945, 4500)
    if (
        (road.width, road.height) != scope.shape[::-1]
        or road.transform != src.window_transform(extent)
        or road.crs != src.crs
    ):
        raise ValueError("The road candidate does not share the exact original world grid")
    profile = src.profile.copy()
    profile.update(compress="deflate", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    with rio.open(output, "w", **profile) as dst:
        dst.colorinterp = src.colorinterp
        dst.update_tags(**src.tags())
        dst.update_tags(
            ACCEPTED="false",
            STATUS="whole main highway candidate; see quality-report.json",
            METHOD="registered real-photo background selection and foreground-aware hard joins",
            ORIGINAL_REFERENCE=str(original),
        )
        for _, window in dst.block_windows(1):
            raw = src.read(window=window)
            new = raw.copy()
            left = max(int(window.col_off), 0)
            right = min(int(window.col_off + window.width), 24945)
            top = max(int(window.row_off), 17000)
            bottom = min(int(window.row_off + window.height), 21500)
            if left < right and top < bottom:
                region = Window(left, top - 17000, right - left, bottom - top)
                patch = road.read(window=region)
                replace = scope[top - 17000 : bottom - 17000, left:right]
                local = new[
                    :,
                    top - int(window.row_off) : bottom - int(window.row_off),
                    left - int(window.col_off) : right - int(window.col_off),
                ]
                local[:, replace] = patch[:, replace]
                replacement_count += int(replace.sum())
            changed += int(np.any(raw != new, axis=0).sum())
            dst.write(new, window=window)
            blocks += 1
            if blocks % 300 == 0:
                print("WRITE", blocks, round(time.time() - started), flush=True)
    reference = dict(
        crs=src.crs.to_string(),
        transform=list(src.transform)[:6],
        bounds=list(src.bounds),
        width=src.width,
        height=src.height,
        count=src.count,
        dtype=src.dtypes,
        resolution=list(src.res),
    )

# Verify actual saved pixels rather than only trusting the writer's arrays.
outside_changed = 0
inside_mismatch = 0
checked_blocks = 0
with rio.open(original) as src, rio.open(output) as dst, rio.open(candidate) as road:
    metadata_equal = (
        src.crs == dst.crs
        and src.transform == dst.transform
        and src.bounds == dst.bounds
        and src.shape == dst.shape
        and src.count == dst.count
        and src.dtypes == dst.dtypes
        and src.res == dst.res
    )
    if not metadata_equal:
        raise AssertionError("Full orthophoto georeferencing changed")
    for _, window in dst.block_windows(1):
        before = src.read(window=window)
        after = dst.read(window=window)
        allowed = np.zeros(before.shape[1:], bool)
        left = max(int(window.col_off), 0)
        right = min(int(window.col_off + window.width), 24945)
        top = max(int(window.row_off), 17000)
        bottom = min(int(window.row_off + window.height), 21500)
        if left < right and top < bottom:
            replace = scope[top - 17000 : bottom - 17000, left:right]
            ys = slice(top - int(window.row_off), bottom - int(window.row_off))
            xs = slice(left - int(window.col_off), right - int(window.col_off))
            allowed[ys, xs] = replace
            patch = road.read(window=Window(left, top - 17000, right - left, bottom - top))
            inside_mismatch += int((np.any(after[:, ys, xs] != patch, axis=0) & replace).sum())
        outside_changed += int((np.any(before != after, axis=0) & ~allowed).sum())
        checked_blocks += 1
    if outside_changed or inside_mismatch:
        raise AssertionError("Saved raster differs from the authorised replacement")
    full = dst.read([1, 2, 3], out_shape=(3, 2000, 1998), resampling=rio.enums.Resampling.average).transpose(
        1, 2, 0
    )
    Image.fromarray(full).save(a.output / "full-scene-after.jpg", quality=97)


for name in [
    "before-highway-overview.jpg",
    "after-highway-overview.jpg",
    "before-after-highway.jpg",
    "source-photo-provenance.tif",
    "retained-foreground.tif",
    "replacement-scope.npz",
]:
    shutil.copy2(a.candidate / name, a.output / name)
experiment = a.candidate.parent
for source, name in [
    (experiment / "geometry-04/report.json", "road-geometry-measurements.json"),
    (experiment / "camera-corridor-plan.json", "camera-corridor-plan.json"),
    (experiment / "road-mask-08/report.json", "road-scope-seeds.json"),
    (experiment / "road-mask-08/road-mask.npz", "road-search-scope.npz"),
]:
    shutil.copy2(source, a.output / name)
quality["export"] = dict(
    file=str(output),
    reference=reference,
    georeferencingUnchanged=metadata_equal,
    replacementPixels=replacement_count,
    changedPixels=changed,
    outsideReplacementChangedPixels=outside_changed,
    insideReplacementMismatchPixels=inside_mismatch,
    blocksVerified=checked_blocks,
)
quality["acceptanceExplanation"] = (
    "GeoTIFF format and the original world grid are verified. This is "
    "not an independent measurement-accuracy certificate. Geometry and"
    " residual foreground issues are retained in this report."
)
(a.output / "quality-report.json").write_text(json.dumps(quality, indent=2))

# Six same-grid local comparisons; display native detail with no artificial
# sharpening, generation, or removal of remaining objects.
locations = [
    ("west-join", 3000),
    ("west-middle", 5000),
    ("centre-west", 10500),
    ("centre-east", 14500),
    ("east-middle", 18800),
    ("east-end", 22600),
]
road_scope = np.load(experiment / "road-mask-08/road-mask.npz")
locals = []
with rio.open(original) as src, rio.open(output) as dst:
    for label, cx in locations:
        cy = int(road_scope["median"][cx // 4] * 4 + 80)
        w, h = 2400, 1400
        x = max(0, min(24945 - w, cx - w // 2))
        y = max(0, min(4500 - h, cy - h // 2))
        win = Window(x, 17000 + y, w, h)
        before = src.read(
            [1, 2, 3], window=win, out_shape=(3, 700, 1200), resampling=rio.enums.Resampling.average
        ).transpose(1, 2, 0)
        after = dst.read(
            [1, 2, 3], window=win, out_shape=(3, 700, 1200), resampling=rio.enums.Resampling.average
        ).transpose(1, 2, 0)
        Image.fromarray(before).save(a.output / f"{label}-before.jpg", quality=98)
        Image.fromarray(after).save(a.output / f"{label}-after.jpg", quality=98)
        cmp = Image.new("RGB", (1200, 1460), "white")
        draw = ImageDraw.Draw(cmp)
        draw.text((12, 8), label + " - BEFORE", fill="black")
        cmp.paste(Image.fromarray(before), (0, 30))
        draw.text((12, 738), "AFTER", fill="black")
        cmp.paste(Image.fromarray(after), (0, 760))
        cmp.save(a.output / f"{label}-comparison.jpg", quality=98)
        locals.append(
            dict(
                label=label, window=[x, 17000 + y, w, h], bounds=list(rio.windows.bounds(win, src.transform))
            )
        )
(a.output / "local-comparison-windows.json").write_text(json.dumps(locals, indent=2))


def sha(path):
    d = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            d.update(block)
    return d.hexdigest()


expected_original = "b84804934c9d64675a95bde7358f69de2911ed447804803e922a086e45e80a3c"
original_hash = sha(original)
if original_hash != expected_original:
    raise AssertionError("Original OGS2 raster changed since the saved baseline")
manifest = dict(
    state="completed",
    accepted=False,
    originalSHA256=original_hash,
    originalUnchanged=True,
    files=[
        dict(name=f.name, bytes=f.stat().st_size, sha256=sha(f))
        for f in sorted(a.output.iterdir())
        if f.is_file()
    ],
    elapsedSeconds=time.time() - started,
)
(a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("COMPLETED", str(a.output), json.dumps(quality["export"]), flush=True)
