"""Closed-patch counterexamples; RGB similarity is not scene certification."""

from dataclasses import replace
import unittest

import numpy as np

from model_engine.ogs2_road_patch import RoadPatchConfig, select_road_patch, compose_road_patch


class RoadPatchTests(unittest.TestCase):
    def setUp(self):
        self.shape = (25, 31)
        self.base = np.full((*self.shape, 3), 100, np.uint8)
        self.candidate = np.full((*self.shape, 3), 120, np.uint8)
        self.va = np.ones(self.shape, bool)
        self.vb = self.va.copy()
        self.hard = np.zeros(self.shape, bool)
        self.hard[10:15, 12:18] = True
        self.allowed = np.zeros(self.shape, bool)
        self.allowed[3:22, 3:28] = True
        self.forbidden = np.zeros(self.shape, bool)

    def select(self, **kwargs):
        return select_road_patch(self.base, self.candidate, self.va, self.vb,
                                 self.hard, self.allowed, self.forbidden, **kwargs)

    def test_real_source_values_and_no_blending(self):
        result = self.select()
        self.assertTrue(result.accepted, result.reason)
        rgb, labels, qc = compose_road_patch(self.base, self.candidate, result)
        np.testing.assert_array_equal(rgb[labels == 0], self.base[labels == 0])
        np.testing.assert_array_equal(rgb[labels == 1], self.candidate[labels == 1])
        self.assertTrue((labels[self.hard] == 1).all())
        self.assertTrue((labels[~self.allowed] == 0).all())
        self.assertEqual(qc["mixedPixels"], 0)
        self.assertFalse(qc["geometryCertified"])
        self.assertFalse(qc["semanticsCertified"])
        self.assertFalse(qc["wholeOrthophotoCertified"])
        self.assertGreater(qc["infiniteCapacity"], qc["finiteCapacitySum"])
        self.assertEqual(qc["flowValue"], qc["cutCapacity"])

    def test_whole_vehicle_and_one_pixel_endpoint_guard_take_one_source(self):
        self.forbidden[8:18, 10:21] = True
        result = self.select()
        self.assertTrue(result.accepted, result.reason)
        self.assertFalse((result.seam_mask & self.forbidden).any())
        self.assertTrue(result.take_candidate[self.forbidden].all())

    def test_vehicle_crossing_allowed_boundary_and_core_is_refused(self):
        self.forbidden[11:14, 1:19] = True
        result = self.select()
        self.assertFalse(result.accepted)
        self.assertIn("forbidden_cut_component", result.reason)
        self.assertIsNone(result.source_index)

    def test_hard_core_outside_allowed_refused(self):
        self.hard[1, 1] = True
        self.assertEqual(self.select().reason, "hard_candidate_outside_allowed_region")

    def test_no_invalid_source_fallback_at_core(self):
        self.vb[12, 14] = False
        self.assertEqual(self.select().reason, "hard_candidate_has_invalid_source")

    def test_invalid_base_inside_allowed_forces_real_candidate(self):
        self.va[8:18, 8:23] = False
        result = self.select()
        self.assertTrue(result.accepted, result.reason)
        self.assertTrue(result.take_candidate[~self.va].all())

    def test_invalid_base_outside_allowed_refused(self):
        self.va[1, 1] = False
        self.assertEqual(self.select().reason, "base_invalid_outside_allowed_region")

    def test_uncovered_pixel_refused(self):
        self.va[7, 7] = self.vb[7, 7] = False
        self.assertEqual(self.select().reason, "both_sources_invalid")

    def test_patch_must_close_before_image_boundary(self):
        self.allowed[:] = True
        self.hard[0, 12] = True
        self.assertIn("nonclosed_patch", self.select().reason)

    def test_no_unrequested_empty_patch(self):
        self.hard[:] = False
        self.assertEqual(self.select().reason, "hard_candidate_core_is_empty")

    def test_prefers_low_difference_ring_over_shorter_bad_seam(self):
        self.candidate[:] = 255
        self.candidate[6:19, 7:24] = 100
        self.candidate[9:16, 11:19] = 255
        result = self.select()
        self.assertTrue(result.accepted, result.reason)
        self.assertEqual(result.qc["seamRgbDifferenceMax"], 0.0)

    def test_saturated_pixels_and_huge_finite_weights_do_not_overflow(self):
        self.base[:] = 0
        self.candidate[:] = 255
        result = self.select(config=RoadPatchConfig(rgb_weight=1e308, gradient_weight=1e308))
        self.assertTrue(result.accepted, result.reason)
        self.assertGreater(result.qc["flowValue"], 0)
        self.assertLessEqual(result.qc["infiniteCapacity"], np.iinfo(np.int32).max)

    def test_bad_capacity_config_rejected(self):
        for config in (RoadPatchConfig(capacity_levels=257), RoadPatchConfig(max_pixels=2_000_001),
                       RoadPatchConfig(rgb_weight=float("inf")), RoadPatchConfig(rgb_weight=-1),
                       RoadPatchConfig(rgb_weight=0, gradient_weight=0)):
            with self.assertRaises(ValueError):
                self.select(config=config)

    def test_budget_uses_cropped_graph(self):
        result = self.select(config=RoadPatchConfig(max_pixels=600))
        self.assertTrue(result.accepted, result.reason)
        self.assertLess(result.qc["graphPixels"], np.prod(self.shape))
        self.assertEqual(self.select(config=RoadPatchConfig(max_pixels=100)).reason,
                         "graph_pixel_budget_exceeded")

    def test_compose_refuses_mutated_image_plan_or_failed_result(self):
        result = self.select()
        self.candidate[0, 0, 0] += 1
        with self.assertRaises(ValueError):
            compose_road_patch(self.base, self.candidate, result)
        self.candidate[0, 0, 0] -= 1
        result.source_index[0, 0] = 1
        with self.assertRaises(ValueError):
            compose_road_patch(self.base, self.candidate, result)
        with self.assertRaises(ValueError):
            compose_road_patch(self.base, self.candidate, replace(result, accepted=False))

    def test_input_type_and_shape_validation(self):
        self.allowed = self.allowed.astype(np.uint8)
        with self.assertRaises(ValueError):
            self.select()


if __name__ == "__main__":
    unittest.main()
