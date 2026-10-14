"""CPU export regression with a correction crossing four 512px TIFF blocks."""

from __future__ import annotations

import json

import numpy as np
import rasterio as rio
from affine import Affine
from rasterio.enums import ColorInterp
from model_engine.ogs2 import sha256
from model_engine.ogs2_apply_patch import export_patch


def write(path, array, transform):
    with rio.open(
        path, "w", driver="GTiff", count=array.shape[0], width=array.shape[2],
        height=array.shape[1], dtype="uint8", crs="EPSG:32650", transform=transform,
        tiled=True, blockxsize=512, blockysize=512,
    ) as dataset:
        dataset.write(array)
        if array.shape[0] == 4:
            dataset.colorinterp = (
                ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha,
            )


def test_patch_crossing_horizontal_and_vertical_blocks_preserves_exact_pixels(tmp_path):
    rng = np.random.default_rng(1000)
    old = rng.integers(0, 256, (4, 1031, 1037), dtype=np.uint8)
    old[3] = 255
    transform = Affine(0.05, 0, 191000, 0, -0.05, 2494000)
    base = tmp_path / "base.tif"
    write(base, old, transform)
    patch = tmp_path / "patch"
    patch.mkdir()
    # 509:522 by 510:525 overlaps both sides of the 512px block boundaries.
    part = old[:, 509:522, 510:525].copy()
    support = np.zeros((1, 13, 15), np.uint8)
    support[0, 1:12, 1:14] = 1
    support[0, 4:7, 5:8] = 0  # undeclared interior island must stay byte-identical
    part[:3, support[0] > 0] ^= 137
    patch_transform = transform * Affine.translation(510, 509)
    write(patch / "ogs2_orthophoto.tif", part, patch_transform)
    write(patch / "write-support.tif", support, patch_transform)
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({
        "checksPassed": True,
        "allCachesLinkedToActualPhotoHashes": True,
        "compositionBase": str(base),
        "inputs": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (base, patch / "ogs2_orthophoto.tif", patch / "write-support.tif")
        ],
    }))
    output = tmp_path / "full"
    report = export_patch(base, patch, validation, output)
    expected = old.copy()
    expected[:, 509:522, 510:525] = part
    with rio.open(output / "ogs2_orthophoto.tif") as actual:
        np.testing.assert_array_equal(actual.read(), expected)
        assert actual.transform == transform
    assert report["changedPixels"] == int(support.sum())
    assert report["outsideWriteDifferentPixels"] == 0
    assert report["alphaDifferentPixels"] == 0
    assert report["patchWindowOnFull"] == [510, 509, 15, 13]
