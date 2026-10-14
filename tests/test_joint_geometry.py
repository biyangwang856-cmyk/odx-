"""Counterexamples for independent geometry and measured dynamic support."""

# Optional algorithm dependencies are separate from the base environment.
# ruff: noqa: E402
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
from model_engine import ogs2_joint_geometry as geometry
from model_engine.ogs2_joint_geometry import GeometryConfig
from model_engine.ogs2_joint_policy import CandidateEvidence, choose_whole_source


def spatial_points():
    training, heldout = [], []
    for col, row in [(0, 0), (1, 1), (2, 0), (3, 1), (1, 0), (0, 1), (3, 0), (2, 1)]:
        destination = training if (col + row) % 2 == 0 else heldout
        for dx, dy in [(4, 4), (9, 4), (4, 9), (9, 9)]:
            destination.append([col * 25 + dx, row * 25 + dy])
    return np.asarray(training + heldout, float), len(training)


def evaluate_points(monkeypatch, a, shifts, config=GeometryConfig()):
    b = a + shifts
    monkeypatch.setattr(
        geometry,
        "unique_correspondences",
        lambda *args: (a.copy(), b.copy(), {"rawForward": len(a), "rawMutual": len(a), "unique": len(a)}),
    )
    image = np.zeros((100, 100, 3), np.uint8)
    return geometry.independent_static_evidence(image, image, np.ones((100, 100), bool), config)


def test_spatially_separated_small_actual_shifts_pass(monkeypatch):
    points, _ = spatial_points()
    result = evaluate_points(monkeypatch, points, np.tile([0.4, 0.3], (len(points), 1)))
    assert result["accepted"]
    assert result["trainingPoints"] == result["heldoutPoints"] == 16
    assert result["p90ActualPixels"] == pytest.approx(0.5)
    assert result["fittedCorrectionApplied"] is False


def test_bad_heldout_matches_are_not_screened_by_full_data_consensus(monkeypatch):
    points, split = spatial_points()
    shifts = np.zeros_like(points)
    shifts[split : split + 4] = [3.0, 0.0]
    result = evaluate_points(monkeypatch, points, shifts)
    assert result["accepted"] is False
    assert result["reason"] == "independent-holdout-failed"
    assert result["trainingPoints"] == result["heldoutPoints"] == 16
    assert result["holdoutResidualP90Pixels"] == pytest.approx(3.0)
    assert result["medianVectorPixels"] == [0.0, 0.0]
    assert sum(not match["inlier"] for match in result["correspondences"]) == 4


def test_changing_only_heldout_cannot_change_training_translation(monkeypatch):
    points, split = spatial_points()
    shifts = np.zeros_like(points)
    shifts[:split] = [0.3, -0.2]
    shifts[split:] = [0.4, -0.1]
    first = evaluate_points(monkeypatch, points, shifts)
    shifts[split:] = [0.6, 0.2]
    second = evaluate_points(monkeypatch, points, shifts)
    assert first["medianVectorPixels"] == pytest.approx([0.3, -0.2])
    assert second["medianVectorPixels"] == pytest.approx(first["medianVectorPixels"])


def test_one_small_feature_cluster_is_rejected_even_when_it_crosses_four_cells(monkeypatch):
    # Straddling cell boundaries must not turn one small road feature into
    # independent coverage of the image.
    points = np.asarray(
        [[x, y] for x in [16, 20, 28, 32] for y in [16, 20, 28, 32]], float
    )
    result = evaluate_points(monkeypatch, points, np.tile([0.2, 0.1], (len(points), 1)))
    assert result["accepted"] is False
    assert result["reason"] == "spatially-concentrated-features"
    assert len(result["occupiedCells"]) == 4


def test_consistent_large_shift_is_not_accepted_by_small_fitted_residual(monkeypatch):
    points, _ = spatial_points()
    result = evaluate_points(monkeypatch, points, np.tile([3.0, 0.0], (len(points), 1)))
    assert result["holdoutResidualP90Pixels"] == pytest.approx(0.0)
    assert result["p90ActualPixels"] == pytest.approx(3.0)
    assert result["accepted"] is False
    assert result["reason"] == "actual-static-displacement"


def fake_sift(monkeypatch, first_points, second_points, first_descriptors, second_descriptors):
    batches = iter(
        [
            ([cv2.KeyPoint(float(x), float(y), 1) for x, y in first_points], first_descriptors),
            ([cv2.KeyPoint(float(x), float(y), 1) for x, y in second_points], second_descriptors),
        ]
    )

    class Extractor:
        def detectAndCompute(self, *_args):
            return next(batches)

    monkeypatch.setattr(geometry.cv2, "SIFT_create", lambda **kwargs: Extractor())


def test_duplicate_sift_orientations_count_as_one_physical_corner(monkeypatch):
    points = [(10, 10), (10, 10), (30, 30), (45, 45)]
    descriptors = np.eye(4, 128, dtype=np.float32)
    fake_sift(monkeypatch, points, np.asarray(points) + [0.2, 0.2], descriptors, descriptors.copy())
    image = np.zeros((64, 64, 3), np.uint8)
    first, second, counts = geometry.unique_correspondences(image, image, np.ones((64, 64), bool))
    assert counts["rawMutual"] == 4
    assert counts["unique"] == len(first) == len(second) == 3


def test_many_to_one_descriptor_matches_do_not_count_as_independent_support(monkeypatch):
    descriptors = np.eye(3, 128, dtype=np.float32)
    first_descriptors = np.stack([descriptors[0], descriptors[0], descriptors[1]])
    fake_sift(
        monkeypatch,
        [(10, 10), (30, 10), (45, 45)],
        [(20, 10), (45, 45), (50, 20)],
        first_descriptors,
        descriptors,
    )
    image = np.zeros((64, 64, 3), np.uint8)
    first, second, counts = geometry.unique_correspondences(image, image, np.ones((64, 64), bool))
    assert counts["rawForward"] == 3
    assert counts["unique"] == len(first) == len(second) == 1
    assert first[0].tolist() == [45.0, 45.0]


def test_no_texture_or_unknown_static_region_never_verifies_geometry():
    image = np.zeros((64, 64, 3), np.uint8)
    result = geometry.independent_static_evidence(image, image, np.zeros((64, 64), bool))
    assert result["accepted"] is False
    assert result["reason"] == "insufficient-independent-matches"


def test_different_pixel_frames_fail_before_feature_matching():
    with pytest.raises(ValueError, match="pixel frame"):
        geometry.unique_correspondences(
            np.zeros((32, 32, 3), np.uint8), np.zeros((31, 32, 3), np.uint8), np.ones((32, 32), bool)
        )


def dynamic_example():
    first = np.full((64, 64, 3), 100, np.uint8)
    second = first.copy()
    body = np.zeros((64, 64), bool)
    body[24:34, 24:34] = True
    # A real changed shadow with one unchanged-pixel gap to the car seed.
    second[24:34, 35:44] = 0
    return first, second, body


def test_shadow_closing_bridge_is_protection_but_not_measured_evidence():
    first, second, body = dynamic_example()
    valid = np.ones(body.shape, bool)
    hard, measured, report = geometry.observed_dynamic_support([first, second], [valid, valid], body, 0.1)
    assert measured[26:32, 35:43].all()
    assert not measured[:, 34].any()  # Closing may bridge, but that column did not change.
    assert hard[28, 34]
    raw_change = np.max(np.abs(first.astype(int) - second.astype(int)), axis=2) >= 25
    assert not (measured & ~raw_change).any()
    assert report["outsideBodyMeasuredPixels"] > 0


def test_unknown_source_pixels_are_never_claimed_as_measured_shadow():
    first, second, body = dynamic_example()
    va, vb = np.ones(body.shape, bool), np.ones(body.shape, bool)
    vb[24:34, 35:44] = False
    _, measured, report = geometry.observed_dynamic_support([first, second], [va, vb], body, 0.1)
    assert not measured.any()
    assert report["measuredChangePixels"] == 0


def test_far_unrelated_change_does_not_expand_vehicle_support():
    first, second, body = dynamic_example()
    second[:] = first
    second[48:56, 48:56] = 0
    valid = np.ones(body.shape, bool)
    _, measured, _ = geometry.observed_dynamic_support([first, second], [valid, valid], body, 1.0)
    assert not measured.any()


def test_support_touching_unknown_source_boundary_is_rejected_by_whole_source_policy():
    first, second, body = dynamic_example()
    valid = np.ones(body.shape, bool)
    hard, _, _ = geometry.observed_dynamic_support([first, second], [valid, valid], body, 0.1)
    valid[28, 43] = False  # Inside a required observed shadow footprint.
    candidate = CandidateEvidence(
        source_image="real-frame",
        valid=valid,
        observed_track_masks={"bus": body},
        complete_tracks=frozenset({"bus"}),
        static_verified=True,
        boundary_clear=True,
        cost=1.0,
        observation_ids_by_track={"bus": ("real-observation",)},
    )
    selected, decisions = choose_whole_source(hard, frozenset({"bus"}), [candidate])
    assert selected is None
    assert decisions[0]["reason"] == "incomplete-dynamic-support-coverage"


@pytest.mark.parametrize("scale", [0.0, -0.1, np.nan, np.inf])
def test_invalid_physical_scale_cannot_produce_a_dynamic_support(scale):
    first, second, body = dynamic_example()
    valid = np.ones(body.shape, bool)
    with pytest.raises(ValueError):
        geometry.observed_dynamic_support([first, second], [valid, valid], body, scale)
