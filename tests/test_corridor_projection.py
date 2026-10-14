"""Counterexamples for corridor evidence; requires the optional vehicle runtime."""

import json
from dataclasses import replace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
rio = pytest.importorskip("rasterio")
pytest.importorskip("scipy")

from model_engine.ogs2_corridor_projection import (
    ProjectionConfig,
    _project_stripe,
    atomic_json,
    bilinear_support,
    canonical_digest,
    detect_projected_bodies,
    native_projection_agreement,
    rank_corridor_sources,
    validate_window,
    verify_hash_binding,
)
from model_engine.ogs2 import sha256


def test_bilinear_rejects_one_invalid_neighbour_even_at_integer_pixel():
    mask = np.ones((4, 4), bool)
    mask[1, 1] = False
    xy = np.array([[0.0, 0.2, 2.0], [0.0, 0.2, 2.0]])
    np.testing.assert_array_equal(bilinear_support(mask, xy), [False, False, True])


def test_bilinear_rejects_last_row_column_nan_and_infinity():
    mask = np.ones((4, 4), bool)
    xy = np.array([[3, 1, np.nan, np.inf, -0.001, 2.999], [1, 3, 1, 1, 1, 2.999]])
    np.testing.assert_array_equal(bilinear_support(mask, xy), [False, False, False, False, False, True])
    assert not bilinear_support(np.ones((1, 4), bool), np.array([[0.0], [0.0]])).any()


def test_valid_black_source_is_not_treated_as_nodata():
    assert bilinear_support(np.ones((4, 4), bool), np.array([[0.5], [0.5]])).all()


def test_no_objects_is_distinct_from_no_detector_and_never_proves_empty_road():
    class NoDetections:
        def masks(self, image):
            return []
    image = np.zeros((40, 40, 3), np.uint8)
    valid = np.ones((40, 40), bool)
    assert detect_projected_bodies(image, valid, None, ProjectionConfig()) == ([], "not-run")
    assert detect_projected_bodies(image, valid, NoDetections(), ProjectionConfig()) == ([], "executed-no-accepted-objects")


def test_partial_photo_detection_never_becomes_complete_by_roi_crop():
    class Partial:
        def masks(self, image):
            mask = np.zeros(image.shape[:2], bool)
            mask[10:20, 10:20] = True
            return [{"mask": mask, "box": [10, 10, 20, 20], "complete": False, "confidence": 0.9, "samConfidence": 0.95}]
    records, _ = detect_projected_bodies(np.zeros((40, 40, 3), np.uint8), np.ones((40, 40), bool), Partial(), ProjectionConfig())
    assert len(records) == 1 and not records[0]["complete"]


def test_input_bytes_mutation_rejects_bound_cache(tmp_path):
    critical = tmp_path / "surface.bin"
    critical.write_bytes(b"retained 2.5D")
    binding = {str(critical): sha256(critical)}
    verify_hash_binding(binding)
    critical.write_bytes(b"modified height")
    with pytest.raises(ValueError, match="Evidence input changed"):
        verify_hash_binding(binding)


def test_camera_identity_changes_when_focal_or_pose_changes():
    original = {"camera": {"focal": 0.7}, "shot": {"translation": [1, 2, 3]}}
    changed = {"camera": {"focal": 0.701}, "shot": {"translation": [1, 2, 3]}}
    assert canonical_digest(original) != canonical_digest(changed)
    assert canonical_digest(original) == canonical_digest(json.loads(json.dumps(original)))


def test_full_corridor_candidate_outranks_centre_only_camera():
    class Camera:
        def __init__(self, scale):
            self.scale = scale
            self.im_size = (100, 100)
            self.pos = [50.0, 0.0, 100.0]
        def world_to_pixel(self, xyz):
            return np.stack([(xyz[0] - 50) * self.scale + 50, np.full(xyz.shape[1], 50)])
    class Factory:
        def get(self, name):
            return Camera(3 if name == "centre-only" else 0.8)
    y, x = np.mgrid[0:10, 0:100]
    points = np.stack([x, y, np.full_like(x, 10)]).astype(float)
    data = {"shots": {name: {"rotation": [0, 0, 0], "translation": [0, 0, 0]} for name in ("centre-only", "entire-corridor")}}
    ranked = rank_corridor_sources(Factory(), data, points, np.ones(x.shape, bool), [0, 0, 0], cells=10)
    assert ranked[0]["sourceImage"] == "entire-corridor"
    assert ranked[0]["estimatedCorridorCoverage"] == 1
    assert ranked[1]["estimatedCorridorCoverage"] < 0.5


def test_native_perspective_half_pixel_convention_is_recorded_not_adjusted():
    class Camera:
        im_size = (400, 300)
        def __init__(self, bias=0):
            self.bias = bias
        def world_to_pixel(self, xyz):
            return xyz[:2] / xyz[2] * 280 + np.array([199.5, 149.5])[:, None] + self.bias
    shot = {"rotation": [0, 0, 0], "translation": [0, 0, 0]}
    parameters = {"projection_type": "perspective", "focal": 0.7, "k1": 0, "k2": 0}
    points = np.array([[0, 1, 2, -2], [0, 1, -1, 2], [10, 10, 12, 20]], float)
    proof = native_projection_agreement(Camera(), shot, parameters, [0, 0, 0], points)
    assert proof["accepted"]
    np.testing.assert_allclose(proof["orthorityMinusWidthHalfConventionMedianXY"], [-0.5, -0.5])
    wrong = native_projection_agreement(Camera(0.5), shot, parameters, [0, 0, 0], points)
    assert not wrong["accepted"]


def test_real_rgba_photo_sampling_rejects_invalid_bilinear_support(tmp_path):
    source = tmp_path / "photo.tif"
    rgba = np.zeros((4, 4, 4), np.uint8)
    rgba[:3] = 60
    rgba[3] = 255
    rgba[3, 1, 1] = 0
    with rio.open(source, "w", driver="GTiff", width=4, height=4, count=4, dtype="uint8") as dst:
        dst.colorinterp = (rio.enums.ColorInterp.red, rio.enums.ColorInterp.green, rio.enums.ColorInterp.blue, rio.enums.ColorInterp.alpha)
        dst.write(rgba)
    class Camera:
        im_size = (4, 4)
        pos = [0.0, 0.0, 10.0]
        def world_to_pixel(self, xyz):
            return np.stack([xyz[0], -xyz[1]])
    y, x = np.mgrid[0:3, 0:3]
    points = np.stack([x + 0.5, -y - 0.5, np.zeros_like(x)])
    shot = {"rotation": [np.pi, 0, 0], "translation": [0, 0, 10]}
    surface = np.zeros((100, 100), np.float32)
    transform = rio.transform.from_origin(-50, 50, 1, 1)
    image, valid, cost = _project_stripe(source, Camera(), shot, [0, 0, 0], points, surface, transform, 0.35)
    expected = np.array([[False, False, True], [False, False, True], [True, True, True]])
    np.testing.assert_array_equal(valid, expected)
    assert np.all(image[~valid] == 0) and np.all(cost[~valid] == 1e6)
    assert np.all(image[valid] == 60)


def test_fractional_or_unbounded_window_rejected():
    with pytest.raises(ValueError, match="integer"):
        validate_window([0.5, 0, 5, 5], 10, 10, 100)
    with pytest.raises(ValueError, match="bounded-memory"):
        validate_window([0, 0, 10, 10], 10, 10, 90)
