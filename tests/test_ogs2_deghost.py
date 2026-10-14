# Optional projection dependencies are isolated from the default ODX environment.
# ruff: noqa: E402
import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("rasterio")
pytest.importorskip("scipy")
from model_engine.ogs2_deghost import PatchConfig, capture_time, mask_in_window, select_patch, static_agreement


def test_real_capture_interval_not_filename_order_or_frame_rate():
    first, _ = capture_time("DJI_20260925164213_0064_V.JPG.tif", {})
    second, _ = capture_time("DJI_20260925164214_0065_V.JPG.tif", {})
    assert second - first == 1
    assert capture_time("unknown.tif", {}) == (None, "unknown")
    assert capture_time("unknown.tif", {"capture_time": 111.5})[0] == 111.5


def test_sparse_mask_crop_respects_global_pixel_origin():
    observed = {"origin": (10, 12), "mask": np.ones((3, 4), bool)}
    out = mask_in_window(observed, (12, 11), (4, 4))
    assert out.sum() == 6
    assert out[1:4, :2].all()


def test_partial_conflict_coverage_cannot_mix_new_vehicle_with_old_ghost():
    old = np.full((4, 32, 32), 100, np.uint8)
    region = np.ones((32, 32), bool)
    images = [np.full((32, 32, 3), 200, np.uint8)]
    valid = np.ones((1, 32, 32), bool)
    valid[0, 0, 0] = False
    selected, record = select_patch(old, region, images, valid, np.zeros((1, 32, 32)),
                                    [{"sourceImage": "real", "complete": True}], [0], ["real"], PatchConfig(), region)
    assert selected is None
    assert record["state"] == "baseline-fallback"
    assert record["rejections"][0]["reason"] == "incomplete-conflict-coverage"
    assert np.all(old == 100)


def test_empty_photo_cannot_count_as_preserving_an_associated_vehicle():
    old = np.full((4, 32, 32), 100, np.uint8)
    region = np.ones((32, 32), bool)
    selected, record = select_patch(old, region, [old[:3].transpose(1, 2, 0)],
                                    np.ones((1, 32, 32), bool), np.zeros((1, 32, 32)),
                                    [{"sourceImage": "vehicle-photo", "complete": True}], [0],
                                    ["empty-photo"], PatchConfig(), region)
    assert selected is None
    assert record["rejections"][0]["reason"] == "no-complete-associated-vehicle"


def test_static_displacement_is_measured_without_warping():
    import cv2

    rng = np.random.default_rng(23)
    image = rng.integers(0, 256, (192, 192, 3), dtype=np.uint8)
    shifted = cv2.warpAffine(image, np.array([[1., 0., 6.], [0., 1., 0.]]), (192, 192))
    mask = np.zeros((192, 192), bool)
    mask[15:-15, 15:-15] = True
    same = static_agreement(image, image, mask, PatchConfig())
    moved = static_agreement(image, shifted, mask, PatchConfig())
    assert same["accepted"]
    assert not moved["accepted"]
    assert moved["p90DisplacementPixels"] > 5
