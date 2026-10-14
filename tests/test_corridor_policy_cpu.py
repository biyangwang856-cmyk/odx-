"""Counterexamples for real-exposure road replacement without invented identities."""

from dataclasses import replace
import unittest

import numpy as np

from model_engine.ogs2_corridor_policy import (
    CorridorCandidate, CorridorConfig, array_sha256, compose_corridor, select_corridor_source,
)


SHAPE = (28, 42)


def region(rows, columns):
    mask = np.zeros(SHAPE, bool)
    mask[slice(*rows), slice(*columns)] = True
    return mask


class CorridorPolicyTests(unittest.TestCase):
    def setUp(self):
        self.baseline = np.full((*SHAPE, 4), 120, np.uint8)
        self.baseline[..., 3] = 255
        self.conflict = region((9, 18), (9, 32))
        self.road = np.ones(SHAPE, bool)
        self.unknown = np.zeros(SHAPE, bool)
        self.config = CorridorConfig(protection_steps=1, static_collar_steps=2, min_static_pixels=48)

    def candidate(self, identity="photo", cost=1.0, foregrounds=None):
        masks = foregrounds if foregrounds is not None else {
            "simultaneous-car-a+observed-shadow": region((11, 14), (13, 17)),
            "simultaneous-car-b+observed-shadow": region((13, 16), (24, 28)),
        }
        rgb = np.full((*SHAPE, 3), 90, np.uint8)
        for index, mask in enumerate(masks.values()):
            rgb[mask] = [20 + index * 100, 80, 200]
        candidate = CorridorCandidate(
            identity, rgb, np.ones(SHAPE, bool), np.ones(SHAPE, bool), np.ones(SHAPE, bool),
            np.ones(SHAPE, bool), masks, frozenset(masks), cost, 12.0, {}, self.unknown.copy(), True,
        )
        return self.bind(candidate)

    def bind(self, candidate, **extra):
        proof = {
            "accepted": True, "referenceKind": "orthophoto-baseline", "sourceId": candidate.source_id,
            "sourceImageSha256": array_sha256(candidate.rgb),
            "baselineImageSha256": array_sha256(self.baseline[..., :3]),
            "baselineRgbaSha256": array_sha256(self.baseline),
            "staticSupportSha256": array_sha256(candidate.static_support),
            "trainingPoints": 8, "heldoutPoints": 8, "inliers": 16,
            "p90ActualPixels": 0.8, "holdoutResidualP90Pixels": 0.5,
            "occupiedCells": [0, 3, 12, 15], "spanFractions": [0.6, 0.3],
            "consensusFraction": 0.8, "fittedCorrectionApplied": False,
        }
        proof.update(extra)
        return replace(candidate, static_verification=proof)

    def select(self, *candidates, **extra):
        return select_corridor_source(self.baseline, self.conflict, self.road, candidates,
                                      protected_unknown=self.unknown, config=self.config, **extra)

    def test_two_real_cars_in_one_exposure_are_preserved_without_identity_merging(self):
        candidate = self.candidate()
        plan, decisions = self.select(candidate)
        self.assertIsNotNone(plan)
        output, mode, report = compose_corridor(self.baseline, plan)
        for mask in candidate.foregrounds.values():
            np.testing.assert_array_equal(output[..., :3][mask], candidate.rgb[mask])
        self.assertEqual(report["selectedForegroundObservations"], 2)
        self.assertFalse(report["identityAssociationAssumed"])
        self.assertFalse(report["globalInstantClaimed"])
        self.assertTrue(decisions[0]["accepted"])
        self.assertEqual(set(np.unique(mode)), {0, 1})

    def test_verified_real_empty_road_can_replace_historical_vehicle_positions(self):
        old = region((10, 15), (20, 25))
        self.baseline[..., :3][old] = [220, 30, 30]
        empty = self.candidate(foregrounds={})
        plan, _ = self.select(empty, baseline_foregrounds={"past-car+shadow": old})
        output, _, report = compose_corridor(self.baseline, plan)
        np.testing.assert_array_equal(output[..., :3][old], empty.rgb[old])
        self.assertEqual(report["selectedForegroundObservations"], 0)
        self.assertGreater(report["emptyRoadPixels"], 0)

    def test_fewer_cars_do_not_win_source_ranking(self):
        empty = self.candidate("empty", cost=3, foregrounds={})
        occupied = self.candidate("occupied", cost=1)
        plan, decisions = self.select(empty, occupied)
        self.assertEqual(plan.source.source_id, "occupied")
        self.assertTrue(all(d["accepted"] for d in decisions))
        self.assertFalse(plan.report["vehicleCountUsedForRanking"])

    def test_zero_detections_without_complete_catalog_is_not_empty_road(self):
        candidate = replace(self.candidate(foregrounds={}), foreground_catalog_complete=False)
        plan, decisions = self.select(candidate)
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "source-foreground-catalog-is-incomplete")

    def test_one_unverified_old_position_rejects_entire_replacement(self):
        candidate = self.candidate(foregrounds={})
        candidate.verified_empty_road[12, 20] = False
        plan, decisions = self.select(candidate)
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "old-position-has-no-verified-observed-empty-road")

    def test_parking_or_unknown_object_cannot_be_deleted(self):
        self.unknown[12, 20] = True
        plan, decisions = self.select(self.candidate(foregrounds={}))
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "protected-unknown-or-parked-foreground")

    def test_one_invalid_source_pixel_does_not_make_a_half_vehicle(self):
        candidate = self.candidate()
        candidate.valid[12, 15] = False
        plan, decisions = self.select(candidate)
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "incomplete-real-source-or-baseline-coverage")

    def test_whole_vehicle_and_shadow_closure_crosses_initial_region_atomically(self):
        whole = region((11, 15), (29, 36))
        candidate = self.candidate(foregrounds={"bus+its-observed-shadow": whole})
        plan, _ = self.select(candidate)
        self.assertTrue(plan.hard_support[whole].all())
        output, _, _ = compose_corridor(self.baseline, plan)
        np.testing.assert_array_equal(output[..., :3][whole], candidate.rgb[whole])

    def test_clipped_vehicle_or_shadow_at_context_edge_is_refused(self):
        candidate = self.candidate(foregrounds={"clipped-bus": region((10, 14), (29, 42))})
        plan, decisions = self.select(candidate)
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "expand-context-boundary-is-unknown")

    def test_incomplete_object_claim_rejects_source(self):
        candidate = replace(self.candidate(), complete_foregrounds=frozenset())
        plan, decisions = self.select(candidate)
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "incomplete-source-foreground")

    def test_nonroad_support_is_not_flattened_or_replaced(self):
        self.road[12, 20] = False
        plan, decisions = self.select(self.candidate())
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "whole-support-outside-approved-road")

    def test_one_unknown_collar_pixel_refuses_a_visible_seam(self):
        candidate = self.candidate()
        candidate.static_support[8, 10] = False
        candidate = self.bind(candidate)
        plan, decisions = self.select(candidate)
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "unverified-static-collar")

    def test_source_to_source_proof_cannot_substitute_for_actual_baseline_seam(self):
        candidate = self.bind(self.candidate(), referenceKind="other-source")
        plan, decisions = self.select(candidate)
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "baseline-static-reference-required")

    def test_bad_geometry_is_not_accepted_despite_accepted_boolean(self):
        for changes in ({"p90ActualPixels": 2.01}, {"holdoutResidualP90Pixels": np.nan},
                        {"consensusFraction": 0.5}, {"fittedCorrectionApplied": True},
                        {"spanFractions": [0.01, 0.8]}, {"heldoutPoints": 3}):
            with self.subTest(changes=changes):
                candidate = self.bind(self.candidate(), **changes)
                self.assertIsNone(self.select(candidate)[0])

    def test_proof_for_another_source_or_image_is_not_reused(self):
        candidate = self.bind(self.candidate(), sourceId="different-photo")
        plan, decisions = self.select(candidate)
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "baseline-static-proof-binding-mismatch")

    def test_nan_mask_or_nan_source_score_is_not_silently_cast_true(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.select(replace(self.candidate(), valid=np.full(SHAPE, np.nan)))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.select(replace(self.candidate(), cost=np.nan))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.select(replace(self.candidate(), capture_time="bad"))

    def test_no_alpha_or_outside_region_change_and_no_source_rgb_generation(self):
        candidate = self.candidate()
        before = self.baseline.copy()
        plan, _ = self.select(candidate)
        output, _, report = compose_corridor(self.baseline, plan)
        np.testing.assert_array_equal(output[..., :3][plan.hard_support], candidate.rgb[plan.hard_support])
        np.testing.assert_array_equal(output[~plan.hard_support], before[~plan.hard_support])
        np.testing.assert_array_equal(output[..., 3], before[..., 3])
        np.testing.assert_array_equal(self.baseline, before)
        self.assertEqual(report["outsideSupportChangedPixels"], 0)

    def test_mutated_plan_source_and_alpha_are_rejected(self):
        for mutation in ("support", "source", "alpha", "evidence"):
            with self.subTest(mutation=mutation):
                self.setUp()
                candidate = self.candidate()
                plan, _ = self.select(candidate)
                if mutation == "support":
                    plan.hard_support[2, 2] = True
                elif mutation == "source":
                    candidate.rgb[12, 20] = 0
                elif mutation == "alpha":
                    self.baseline[12, 20, 3] = 1
                else:
                    candidate.static_verification["p90ActualPixels"] = 9.0
                with self.assertRaisesRegex(ValueError, "changed"):
                    compose_corridor(self.baseline, plan)

    def test_unknown_baseline_foreground_in_static_collar_is_not_cut(self):
        candidate = self.candidate()
        adjacent = region((7, 9), (20, 24))
        plan, decisions = self.select(candidate, baseline_foregrounds={"other-real-car": adjacent})
        self.assertIsNone(plan)
        self.assertEqual(decisions[0]["reason"], "static-collar-touches-unselected-foreground")


if __name__ == "__main__":
    unittest.main()
