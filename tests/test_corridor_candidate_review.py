"""Counterexamples for semantic review and actual clear-road witness binding."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/lab/review_corridor_candidates.py"
spec = importlib.util.spec_from_file_location("corridor_candidate_review", SCRIPT)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def source_entry():
    return {"sourceImage": "actual-A", "foregrounds": {"A:body0": "fore0", "A:body1": "fore1"},
            "projectionCompleteForegroundIds": ["A:body0"]}


def approval():
    return {"schema": review.REVIEW_SCHEMA, "proposalManifestSha256": "a" * 64,
            "reviewer": "root", "reviewedAt": "2026-10-13T18:00:00+08:00", "sources": [{
                "sourceImage": "actual-A", "roadDomainReviewed": True,
                "historicConflictDomainReviewed": True, "protectedUnknownReviewed": True,
                "foregroundCatalogReviewed": True, "completeBodyAndShadowBoundariesReviewed": True,
                "completeForegroundIds": ["A:body0"], "appearanceNotes": "Source body/shadow edges and road domain inspected."}]}


def test_no_model_confidence_or_absent_review_authorizes_truth():
    assert review.approval_for_source(None, "a" * 64, source_entry()) is None
    record = approval(); record["sources"][0]["foregroundCatalogReviewed"] = False
    assert review.approval_for_source(record, "a" * 64, source_entry()) is None


def test_review_is_bound_to_exact_proposal_not_source_name_only():
    with pytest.raises(ValueError, match="exact proposal"):
        review.approval_for_source(approval(), "b" * 64, source_entry())


def test_appearance_review_cannot_override_incomplete_projected_body():
    record = approval(); record["sources"][0]["completeForegroundIds"].append("A:body1")
    with pytest.raises(ValueError, match="incomplete projected"):
        review.approval_for_source(record, "a" * 64, source_entry())


def test_catalog_and_shadow_review_are_both_required():
    record = approval(); record["sources"][0]["completeBodyAndShadowBoundariesReviewed"] = False
    assert review.approval_for_source(record, "a" * 64, source_entry()) is None
    record = approval(); record["sources"][0]["appearanceNotes"] = ""
    with pytest.raises(ValueError, match="inspection notes"):
        review.approval_for_source(record, "a" * 64, source_entry())


def test_successful_review_only_certifies_the_named_explicit_foregrounds():
    accepted = review.approval_for_source(approval(), "a" * 64, source_entry())
    assert accepted["completeForegroundIds"] == ["A:body0"]


def witness_fixture():
    empty = np.zeros((16, 20), bool); empty[4:8, 6:10] = True
    obs = {"candidate_empty_road": empty, "clear_reference_0": empty.copy()}
    record = {"sourceImage": "actual-A", "clearReferences": [{"sourceImage": "actual-C", "maskKey": "clear_reference_0", "cacheSha256": "c" * 64}]}
    sources = {"actual-A": {}, "actual-C": {}}
    projection = {"sources": [{"sourceImage": "actual-A", "cacheSha256": "a" * 64}, {"sourceImage": "actual-C", "cacheSha256": "c" * 64}]}
    pairs = [{"firstSource": "actual-A", "secondSource": "actual-C", "accepted": True, "colorComparisonAccepted": True}]
    return record, obs, sources, projection, pairs


def test_empty_road_requires_positive_independent_real_exposure_mask():
    record, obs, sources, projection, pairs = witness_fixture()
    accepted, audit = review.validate_clear_witnesses(record, obs, sources, projection, pairs)
    np.testing.assert_array_equal(accepted, obs["candidate_empty_road"])
    assert audit[0]["sourceImage"] == "actual-C"
    obs["candidate_empty_road"][11, 11] = True
    with pytest.raises(ValueError, match="explicit independent clear witness"):
        review.validate_clear_witnesses(record, obs, sources, projection, pairs)


def test_same_exposure_and_failed_geometry_are_not_empty_road_witnesses():
    record, obs, sources, projection, pairs = witness_fixture()
    record["clearReferences"][0]["sourceImage"] = "actual-A"
    with pytest.raises(ValueError, match="independent retained"):
        review.validate_clear_witnesses(record, obs, sources, projection, pairs)
    record, obs, sources, projection, pairs = witness_fixture()
    pairs[0]["accepted"] = False
    with pytest.raises(ValueError, match="accepted measured"):
        review.validate_clear_witnesses(record, obs, sources, projection, pairs)


def test_clear_witness_cache_cannot_be_replaced_by_another_frame():
    record, obs, sources, projection, pairs = witness_fixture()
    record["clearReferences"][0]["cacheSha256"] = "e" * 64
    with pytest.raises(ValueError, match="cache binding"):
        review.validate_clear_witnesses(record, obs, sources, projection, pairs)


def test_context_is_explicit_geometric_margin_does_not_modify_rgb():
    body = np.zeros((21, 21), bool); body[10, 10] = True
    expanded = review.context_mask(body, 0.05, 0.25)
    assert expanded[10, 15] and not expanded[10, 16]
    assert expanded[10, 10] and body.sum() == 1
    assert review.union_masks({}, body.shape).sum() == 0


def test_all_failure_pixels_are_counted_while_semantics_remain_unapproved():
    from model_engine.ogs2_corridor_policy import CorridorCandidate
    shape = (28, 42); base = np.full((*shape, 4), 120, np.uint8); base[..., 3] = 255
    mask = np.zeros(shape, bool); mask[8:16, 10:30] = True
    empty = np.ones(shape, bool); empty[11, 17] = False
    unknown = np.zeros(shape, bool); unknown[12, 18] = True
    candidate = CorridorCandidate("actual-A", base[..., :3].copy(), np.ones(shape, bool), np.ones(shape, bool), empty,
        np.ones(shape, bool), {}, frozenset(), 1.0, 0.0, {"accepted": False, "p90ActualPixels": 2.1, "holdoutResidualP90Pixels": 1.2}, unknown, False)
    counts, hard, collar = review.blocker_counts(base, mask, np.ones(shape, bool), candidate, {}, np.zeros(shape, bool))
    assert counts["unsupportedOldPositions"] == 1 and counts["protectedUnknown"] == 1
    assert counts["notAppearanceReviewed"] and not counts["baselineGeometryAccepted"]
    assert counts["actualDisplacementP90"] == 2.1 and counts["holdoutResidualP90"] == 1.2
    assert counts["componentCount"] == 1 and counts["wholeSupportComponents"][0]["failPixels"]["unsupportedOldPositions"] == 1
