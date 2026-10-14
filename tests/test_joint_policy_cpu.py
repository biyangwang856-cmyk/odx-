"""Standard-library unittest draft, runnable without GPU or pytest."""

import unittest

import numpy as np

from model_engine.ogs2_joint_policy import (
    CandidateEvidence,
    PatchPlan,
    SparseRegion,
    choose_whole_source,
    compose_static_collar,
    merge_conflicts,
)


def plan(identity, origin, track, coverage, first=100, last=101, clock="dji-filename-wall-clock"):
    return PatchPlan(
        identity, SparseRegion(origin, np.ones((10, 10), bool)), frozenset([track]),
        {name: frozenset(tracks) for name, tracks in coverage.items()}, clock, first, last,
    )


class JointPolicyTests(unittest.TestCase):
    def test_overlapping_windows_merge_tracks_but_require_one_common_photo(self):
        first = plan("first", (0, 0), "bus", {"photo-a": ["bus"], "photo-b": ["bus"]})
        second = plan("second", (8, 0), "car", {"photo-b": ["car"]})
        merged = merge_conflicts([first, second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].required_tracks, frozenset(["bus", "car"]))
        self.assertEqual(merged[0].source_candidates, ("photo-b",))
        self.assertEqual(merged[0].support.mask.shape, (10, 18))
        self.assertIsNone(merged[0].rejection_reason)

    def test_joint_merge_does_not_depend_on_writer_order(self):
        first = plan("first", (0, 0), "bus", {"photo": ["bus"]})
        second = plan("second", (8, 0), "car", {"photo": ["car"]})
        a, b = merge_conflicts([first, second])[0], merge_conflicts([second, first])[0]
        self.assertEqual(a.members, b.members)
        np.testing.assert_array_equal(a.support.mask, b.support.mask)

    def test_no_common_source_does_not_select_empty_background(self):
        merged = merge_conflicts([
            plan("a", (0, 0), "bus", {"photo-a": ["bus"]}),
            plan("b", (8, 0), "car", {"photo-b": ["car"]}),
        ])[0]
        self.assertEqual(merged.rejection_reason, "no-common-whole-track-source")
        self.assertIsNone(merged.support)

    def test_future_vehicle_is_a_different_cohort_not_silently_preserved_in_the_past(self):
        merged = merge_conflicts([
            plan("a", (0, 0), "bus", {"photo": ["bus"]}),
            plan("b", (8, 0), "car", {"photo": ["car"]}, first=153, last=154),
        ])[0]
        self.assertEqual(merged.rejection_reason, "incompatible-temporal-cohorts")

    def test_bbox_overlap_without_mask_overlap_does_not_merge(self):
        diagonal_a, diagonal_b = np.eye(10, dtype=bool), np.fliplr(np.eye(10, dtype=bool))
        first = plan("a", (0, 0), "bus", {"photo": ["bus"]})
        second = plan("b", (0, 0), "car", {"photo": ["car"]})
        first = PatchPlan(**{**first.__dict__, "support": SparseRegion((0, 0), diagonal_a)})
        second = PatchPlan(**{**second.__dict__, "support": SparseRegion((0, 0), diagonal_b)})
        self.assertEqual(len(merge_conflicts([first, second])), 2)

    def test_empty_photo_loses_even_if_its_projection_cost_is_better(self):
        hard = np.ones((10, 10), bool)
        common = dict(valid=hard, static_verified=True, boundary_clear=True, observation_ids_by_track={"bus": ("detected-bus",)})
        candidates = [
            CandidateEvidence("empty", observed_track_masks={}, complete_tracks=frozenset(), cost=0, **common),
            CandidateEvidence("real", observed_track_masks={"bus": hard}, complete_tracks=frozenset(["bus"]), cost=1, **common),
        ]
        selected, decisions = choose_whole_source(hard, frozenset(["bus"]), candidates)
        self.assertEqual(selected, "real")
        self.assertEqual(decisions[0]["reason"], "missing-complete-protected-vehicle")

    def test_vehicle_claim_without_actual_mask_is_not_evidence(self):
        hard = np.ones((10, 10), bool)
        candidate = CandidateEvidence("empty", hard, {"bus": np.zeros_like(hard)}, frozenset(["bus"]), True, True, 0, {"bus": ("detected-bus",)})
        self.assertIsNone(choose_whole_source(hard, frozenset(["bus"]), [candidate])[0])

    def test_one_unknown_shadow_pixel_rejects_entire_candidate(self):
        hard = np.ones((10, 10), bool)
        valid = hard.copy()
        valid[0, 0] = False
        candidate = CandidateEvidence("real", valid, {"bus": hard}, frozenset(["bus"]), True, True, 0, {"bus": ("detected-bus",)})
        self.assertIsNone(choose_whole_source(hard, frozenset(["bus"]), [candidate])[0])

    def test_same_protected_identity_cannot_have_two_source_instances(self):
        hard = np.ones((10, 10), bool)
        candidate = CandidateEvidence("real", hard, {"bus": hard}, frozenset(["bus"]), True, True, 0, {"bus": ("bus-a", "bus-b")})
        selected, decisions = choose_whole_source(hard, frozenset(["bus"]), [candidate])
        self.assertIsNone(selected)
        self.assertEqual(decisions[0]["reason"], "protected-vehicle-not-observed-exactly-once")

    def test_one_observation_cannot_satisfy_two_distinct_vehicle_identities(self):
        hard = np.ones((10, 10), bool)
        required = frozenset(["bus", "car"])
        candidate = CandidateEvidence("real", hard, {"bus": hard, "car": hard}, required, True, True, 0, {"bus": ("same-object",), "car": ("same-object",)})
        selected, decisions = choose_whole_source(hard, required, [candidate])
        self.assertIsNone(selected)
        self.assertEqual(decisions[0]["reason"], "different-identities-share-one-observation")

    def collar_inputs(self):
        baseline = np.full((8, 8, 4), 20, np.uint8)
        baseline[..., 3] = 255
        source = np.full((8, 8, 3), 220, np.uint8)
        hard = np.zeros((8, 8), bool)
        hard[3:5, 3:5] = True
        weight = np.zeros((8, 8), np.float32)
        weight[2:6, 2:6] = 0.5
        weight[hard] = 1
        static = ~hard
        return baseline, source, hard, weight, static, np.ones((8, 8), bool)

    def test_only_static_collar_blends_and_every_hard_pixel_keeps_true_source(self):
        baseline, source, hard, weight, static, valid = self.collar_inputs()
        image, mode, saved_weight = compose_static_collar(baseline, source, hard, weight, static, valid)
        np.testing.assert_array_equal(image[hard, :3], source[hard])
        np.testing.assert_array_equal(image[weight == 0], baseline[weight == 0])
        np.testing.assert_array_equal(image[..., 3], baseline[..., 3])
        self.assertTrue(np.all(mode[hard] == 1))
        self.assertTrue(np.all(mode[(weight > 0) & ~hard] == 2))
        np.testing.assert_array_equal(saved_weight, weight)

    def test_a_shadow_left_in_blend_collar_is_rejected(self):
        inputs = list(self.collar_inputs())
        inputs[4][2, 3] = False
        with self.assertRaisesRegex(ValueError, "static"):
            compose_static_collar(*inputs)

    def test_shadow_hard_support_cannot_be_faded_into_the_old_ghost(self):
        inputs = list(self.collar_inputs())
        inputs[3][3, 3] = 0.5
        with self.assertRaisesRegex(ValueError, "cannot be blended"):
            compose_static_collar(*inputs)

    def test_invalid_baseline_coverage_cannot_be_filled(self):
        inputs = list(self.collar_inputs())
        inputs[0][3, 3, 3] = 0
        with self.assertRaisesRegex(ValueError, "coverage"):
            compose_static_collar(*inputs)


if __name__ == "__main__":
    unittest.main()
