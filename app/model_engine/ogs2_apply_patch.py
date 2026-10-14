"""Export a numerically verified local correction on the full retained mosaic.

This stage is a lossless spatial composition, not a new reconstruction. It
requires hashed validation of the actual patch and preserves all other pixels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import rasterio as rio
from rasterio.windows import Window

from model_engine.ogs2 import save_json, sha256


def aligned_window(base, patch):
    if base.crs != patch.crs or base.count != 4 or patch.count != 4 or base.dtypes != patch.dtypes:
        raise ValueError("Full and patch rasters must share RGBA dtype and CRS")
    if not np.allclose(
        [base.transform.a, base.transform.b, base.transform.d, base.transform.e],
        [patch.transform.a, patch.transform.b, patch.transform.d, patch.transform.e],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError("Patch resolution or rotation differs")
    x, y = ~base.transform * (patch.transform.c, patch.transform.f)
    if max(abs(x - round(x)), abs(y - round(y))) > 1e-6:
        raise ValueError("Patch does not share original map pixel centers")
    window = Window(round(x), round(y), patch.width, patch.height)
    if window.col_off < 0 or window.row_off < 0 or window.col_off+patch.width > base.width or window.row_off+patch.height > base.height:
        raise ValueError("Patch exceeds retained scene extent")
    return window


def export_patch(base_path, patch_dir, validation_path, output):
    base_path, patch_dir, validation_path, output = [
        Path(p).resolve() for p in (base_path, patch_dir, validation_path, output)
    ]
    for path in (base_path, patch_dir, validation_path):
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("Fresh output may not overlap retained inputs")
    if output.exists():
        raise FileExistsError("Do not overwrite an earlier full export")
    validation = json.loads(validation_path.read_text())
    if not validation.get("checksPassed") or not validation.get("allCachesLinkedToActualPhotoHashes"):
        raise ValueError("Patch validation has not passed")
    if Path(validation["compositionBase"]).resolve() != base_path:
        raise ValueError("Validation used a different retained scene")
    pins = {Path(entry["path"]).resolve(): entry["sha256"] for entry in validation["inputs"]}
    patch_path = patch_dir / "ogs2_orthophoto.tif"
    write_path = patch_dir / "write-support.tif"
    for path in (base_path, patch_path, write_path):
        if path not in pins:
            raise ValueError("Actual raster missing from validated input hashes")
    for path, value in pins.items():
        if sha256(path) != value:
            raise ValueError("A validated input has changed: " + str(path))
    started = time.monotonic()
    with rio.open(base_path) as base, rio.open(patch_path) as patch, rio.open(write_path) as support:
        window = aligned_window(base, patch)
        if support.crs != patch.crs or support.shape != patch.shape or support.transform != patch.transform:
            raise ValueError("Write mask uses a different pixel frame")
        old_patch = base.read(window=window)
        replacement = patch.read()
        write = support.read(1)
        if not np.isin(write, [0, 1]).all():
            raise ValueError("Write mask is not binary")
        write = write.astype(bool)
        if not np.array_equal(replacement[3], old_patch[3]):
            raise ValueError("Patch changes original coverage or alpha")
        if not np.array_equal(replacement[:, ~write], old_patch[:, ~write]):
            raise ValueError("Patch alters pixels outside its declared correction")
        output.mkdir(parents=True, exist_ok=False)
        partial = output / "ogs2_orthophoto.partial.tif"
        profile = {
            **base.profile,
            "compress": "DEFLATE",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
            "BIGTIFF": "IF_SAFER",
        }
        with rio.open(partial, "w", **profile) as full:
            full.colorinterp = base.colorinterp
            full.update_tags(**base.tags())
            for band in range(1, 5):
                full.update_tags(band, **base.tags(band))
                full.set_band_description(band, base.descriptions[band - 1] or "")
            for _, block in full.block_windows(1):
                array = base.read(window=block)
                bx, by, bw, bh = map(int, block.flatten())
                px, py, pw, ph = map(int, window.flatten())
                left, top = max(bx, px), max(by, py)
                right, bottom = min(bx + bw, px + pw), min(by + bh, py + ph)
                if left < right and top < bottom:
                    array[:, top - by : bottom - by, left - bx : right - bx] = replacement[
                        :, top - py : bottom - py, left - px : right - px
                    ]
                full.write(array, window=block)
        changed = outside = alpha = 0
        with rio.open(partial) as full:
            if full.transform != base.transform or full.shape != base.shape or full.crs != base.crs:
                raise RuntimeError("Full export changes scene map frame")
            for _, block in full.block_windows(1):
                actual = full.read(window=block)
                original = base.read(window=block)
                different = np.any(actual != original, axis=0)
                alpha += int((actual[3] != original[3]).sum())
                changed += int(different.sum())
                bx, by, bw, bh = map(int, block.flatten())
                permitted = np.zeros((bh, bw), bool)
                px, py, pw, ph = map(int, window.flatten())
                left, top = max(bx, px), max(by, py)
                right, bottom = min(bx + bw, px + pw), min(by + bh, py + ph)
                if left < right and top < bottom:
                    permitted[top - by : bottom - by, left - bx : right - bx] = write[
                        top - py : bottom - py, left - px : right - px
                    ]
                outside += int((different & ~permitted).sum())
            if not np.array_equal(full.read(window=window), replacement):
                raise RuntimeError("Full export does not contain the actual accepted patch")
        if alpha or outside:
            raise RuntimeError("Full export violates retained pixel invariants")
    for path in (base_path, patch_path, write_path):
        if sha256(path) != pins[path]:
            raise RuntimeError("Retained raster changed during export")
    partial.replace(output / "ogs2_orthophoto.tif")
    report = {
        "schemaVersion": "ogs2-verified-full-export/v1",
        "base": str(base_path),
        "baseSha256": pins[base_path],
        "patch": str(patch_dir),
        "patchSha256": pins[patch_path],
        "validation": str(validation_path),
        "validationSha256": sha256(validation_path),
        "patchWindowOnFull": list(window.flatten()),
        "changedPixels": changed,
        "outsideWriteDifferentPixels": outside,
        "alphaDifferentPixels": alpha,
        "fullOutput": str(output / "ogs2_orthophoto.tif"),
        "outputSha256": sha256(output / "ogs2_orthophoto.tif"),
        "scriptSha256": sha256(Path(__file__)),
        "elapsedSeconds": time.monotonic() - started,
        "scope": "Only the accepted road correction; remaining scene retains original OGS2.",
        "allSceneVehicleIdentityVerified": False,
    }
    save_json(output / "report.json", report)
    save_json(output / "status.json", {"state": "success"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-orthophoto", "patch", "validation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(export_patch(args.base_orthophoto, args.patch, args.validation, args.output)), flush=True
    )


if __name__ == "__main__":
    main()
