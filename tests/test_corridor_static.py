"""Real-image counterexamples for unwarped local road-marking measurements."""

from dataclasses import replace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")

from model_engine.ogs2_corridor_static import (
    NCCConfig, compare_static_providers, evidence_from_correspondences,
    gftt_ncc_static_evidence, unique_ncc_correspondences,
)


def paint_corners():
    """Low-texture pavement with many complete, separated painted corners."""
    gray = np.full((256, 256), 80, np.uint8)
    for y in range(30, 227, 40):
        for x in range(30, 227, 40):
            # Identical L shapes are separated farther than the measurement
            # search window, so they are locally unambiguous, not rich texture.
            cv2.line(gray, (x, y), (x+12, y), 190, 3)
            cv2.line(gray, (x, y), (x, y+12), 190, 3)
    return np.repeat(gray[..., None], 3, axis=2)


def shift(image, dx, dy=0):
    return cv2.warpAffine(image, np.float32([[1, 0, dx], [0, 1, dy]]),
                          (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(80, 80, 80))


def test_identical_low_texture_paint_corners_are_independently_verified():
    image = paint_corners()
    proof = gftt_ncc_static_evidence(image, image.copy(), np.ones(image.shape[:2], bool))
    assert proof["accepted"], proof["reason"]
    assert proof["unique"] >= 12 and proof["heldoutPoints"] >= 4
    assert proof["p90ActualPixels"] < 0.1 and proof["holdoutResidualP90Pixels"] < 0.1
    assert proof["fittedCorrectionApplied"] is False


def test_three_pixel_real_texture_shift_is_measured_and_rejected_not_prefiltered():
    image = paint_corners()
    proof = gftt_ncc_static_evidence(image, shift(image, 3), np.ones(image.shape[:2], bool))
    assert proof["unique"] >= 12, proof
    assert not proof["accepted"] and proof["reason"] == "actual-static-displacement"
    assert proof["p90ActualPixels"] > 2.9
    assert proof["displacementUsedForMatchSelection"] is False
    assert all(abs(c["delta"][0]-3) < 0.1 for c in proof["correspondences"])


def test_fractional_real_motion_keeps_noninteger_measurements():
    image = paint_corners()
    proof = gftt_ncc_static_evidence(image, shift(image, 1.5), np.ones(image.shape[:2], bool))
    assert proof["unique"] >= 12, proof
    measured = np.asarray([c["delta"][0] for c in proof["correspondences"]])
    # Bilinear creation/resampling is not exact continuous radiance ground
    # truth, but it must retain the known half-pixel motion instead of rounding.
    assert abs(float(np.median(measured))-1.5) < 0.2
    assert np.mean(np.abs(measured-np.rint(measured)) > 0.1) > 0.8


def test_fractional_motion_above_tolerance_remains_measured_and_rejected():
    image = paint_corners()
    proof = gftt_ncc_static_evidence(image, shift(image, 2.5), np.ones(image.shape[:2], bool))
    assert proof["unique"] >= 12, proof
    measured = np.asarray([c["delta"][0] for c in proof["correspondences"]])
    assert abs(float(np.median(measured))-2.5) < 0.2
    assert not proof["accepted"] and proof["p90ActualPixels"] > 2.3
    assert proof["reason"] == "actual-static-displacement"


def test_two_pixel_tolerance_is_not_used_as_a_ncc_search_limit():
    with pytest.raises(ValueError, match="search must exceed"):
        unique_ncc_correspondences(paint_corners(), paint_corners(), np.ones((256, 256), bool),
                                   replace(NCCConfig(), search_radius=2))


def test_no_texture_is_rejected():
    image = np.full((256, 256, 3), 90, np.uint8)
    proof = gftt_ncc_static_evidence(image, image, np.ones((256, 256), bool))
    assert not proof["accepted"] and proof["unique"] == 0


def test_repeated_road_bar_pattern_has_no_unique_local_peak():
    rows, cols = np.indices((256, 256))
    pattern = ((rows//3+cols//3) % 2 * 100+80).astype(np.uint8)
    image = np.repeat(pattern[..., None], 3, axis=2)
    proof = gftt_ncc_static_evidence(image, shift(image, 3), np.ones((256, 256), bool))
    assert not proof["accepted"] and proof["unique"] < 12
    assert any(r["reason"] == "ambiguous-repeated-patch" for r in proof["qualityRejections"])


def test_static_support_must_cover_complete_search_not_just_corner_centres():
    image = paint_corners()
    support = np.zeros((256, 256), bool)
    support[::4, ::4] = True
    proof = gftt_ncc_static_evidence(image, image, support)
    assert not proof["accepted"] and proof["unique"] == 0


def test_linear_exposure_difference_does_not_become_a_measured_motion():
    image = paint_corners()
    brighter = np.rint(image.astype(np.float32)*1.1+7).astype(np.uint8)
    proof = gftt_ncc_static_evidence(image, brighter, np.ones((256, 256), bool))
    assert proof["accepted"] and proof["p90ActualPixels"] < 0.1
    assert all(m["forward"]["integerPatchCorrelationFloat64"] > 1-1e-10
               for m in proof["qualityAcceptedMeasurements"])


def test_search_boundary_motion_is_explicitly_rejected():
    image = paint_corners()
    proof = gftt_ncc_static_evidence(image, shift(image, 8), np.ones((256, 256), bool))
    assert not proof["accepted"]
    assert any(r["reason"] == "motion-reaches-search-boundary" for r in proof["qualityRejections"])


def test_every_quality_accepted_heldout_outlier_is_reported_and_evaluated():
    # These already measured coordinates test the holdout contract independently
    # of matcher failures. Checkerboard holdout points move 3 px; training points
    # remain still. Their actual displacement must not be filtered by consensus.
    a = np.array([(x, y) for y in (24, 88, 152, 216) for x in (24, 88, 152, 216)], np.float64)
    cells = (a/256*4).astype(int)
    heldout = (cells[:, 0]+cells[:, 1]) % 2 == 1
    b = a.copy()
    b[heldout, 0] += 3
    proof = evidence_from_correspondences(a, b, (256, 256))
    assert not proof["accepted"]
    assert proof["heldoutPoints"] == int(heldout.sum())
    assert proof["p90ActualPixels"] == 3 and proof["holdoutResidualP90Pixels"] == 3
    assert len(proof["correspondences"]) == len(a)
    assert sum(c["heldout"] for c in proof["correspondences"]) == heldout.sum()


def test_provider_report_keeps_sift_reason_and_never_auto_promotes_ncc():
    image = paint_corners()
    report = compare_static_providers(image, image, np.ones((256, 256), bool))
    assert set(report["providers"]) == {"sift", "gfttNcc"}
    assert report["automaticProviderPromotion"] is False
    assert report["reasons"]["sift"] == report["providers"]["sift"]["reason"]
    assert report["fittedCorrectionApplied"] is False


def test_wrong_frame_or_nonboolean_static_never_silently_converts():
    image = paint_corners()
    with pytest.raises(ValueError, match="boolean"):
        unique_ncc_correspondences(image, image, np.ones((256, 256), np.uint8))
    with pytest.raises(ValueError, match="uint8"):
        unique_ncc_correspondences(image.astype(np.float32), image, np.ones((256, 256), bool))
