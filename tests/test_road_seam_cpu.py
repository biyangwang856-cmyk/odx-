"""Counterexamples for monotone hard seams; no learned/static truth assumed."""

from dataclasses import replace
import unittest

import numpy as np

from model_engine.ogs2_road_seam import RoadSeamConfig, select_road_seam, compose_road_seam


class RoadSeamTests(unittest.TestCase):
    def setUp(self):
        self.shape = (14, 18)
        self.a = np.full((*self.shape, 3), 110, np.uint8)
        self.b = np.full((*self.shape, 3), 113, np.uint8)
        self.va = np.ones(self.shape, bool)
        self.vb = self.va.copy()
        self.static = self.va.copy()
        self.geo = self.va.copy()
        self.fa = np.zeros(self.shape, bool)
        self.fb = self.fa.copy()
        self.ha = self.fa.copy(); self.ha[:, :3] = True
        self.hb = self.fa.copy(); self.hb[:, -3:] = True

    def select(self, **kwargs):
        return select_road_seam(self.a, self.b, self.va, self.vb, self.static, self.geo,
                                self.fa, self.fb, self.ha, self.hb, **kwargs)

    def test_real_source_equality_and_explicit_source_indices(self):
        result = self.select()
        self.assertTrue(result.accepted)
        image, ids, qc = compose_road_seam(self.a, self.b, result)
        np.testing.assert_array_equal(image[ids == 1], self.a[ids == 1])
        np.testing.assert_array_equal(image[ids == 2], self.b[ids == 2])
        self.assertEqual(qc["mixedPixels"], 0)
        self.assertFalse(qc["wholeOrthophotoCertified"])
        self.assertFalse(qc["markingMaskProvided"])

    def test_identical_rgb_is_not_static_or_geometry_proof(self):
        self.b[:] = self.a
        self.static[:] = False
        self.assertFalse(self.select().accepted)
        self.static[:] = True; self.geo[:] = False
        self.assertFalse(self.select().accepted)

    def test_whole_object_and_shadow_exclusion_both_sources(self):
        self.fa[3:8, 5:10] = True
        self.fb[5:12, 9:14] = True
        result = self.select()
        self.assertTrue(result.accepted)
        self.assertFalse((result.seam_mask & (self.fa | self.fb)).any())
        self.assertEqual(np.unique(result.source_index[self.fa]).size, 1)
        self.assertEqual(np.unique(result.source_index[self.fb]).size, 1)

    def test_full_width_foreground_blocks_every_possible_seam(self):
        self.fb[7, :] = True
        result = self.select()
        self.assertFalse(result.accepted)
        self.assertIsNone(result.take_b)
        with self.assertRaises(ValueError):
            compose_road_seam(self.a, self.b, result)

    def test_no_hidden_fallback_when_a_locked_pixel_is_invalid(self):
        self.va[7, 1] = False
        result = self.select()
        self.assertFalse(result.accepted)
        self.assertIsNone(result.source_index)

    def test_partial_source_coverage_can_be_valid_without_blending(self):
        self.va[:, 11:] = False
        self.vb[:, :7] = False
        result = self.select()
        self.assertTrue(result.accepted)
        self.assertTrue(self.va[~result.take_b].all())
        self.assertTrue(self.vb[result.take_b].all())

    def test_rgb_difference_and_gradient_difference_are_gates(self):
        self.b[:] = 155
        self.assertFalse(self.select().accepted)
        self.b[:] = self.a
        self.b[::2] += 15
        self.assertFalse(self.select(config=RoadSeamConfig(max_gradient_difference=2)).accepted)

    def test_disconnected_static_islands_not_a_complete_path(self):
        self.static[:] = False
        self.static[:7, 3:5] = True
        self.static[7:, 12:14] = True
        result = self.select()
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "no-connected-verified-static-seam")

    def test_diagonal_turn_cannot_cross_an_unverified_stair_edge(self):
        self.static[:] = False
        # Every row has a legal A/B edge, but the abrupt staircase has an
        # intermediate pixel not verified in both rows. Centres alone pass.
        self.static[:7, 5:7] = True
        self.static[7:, 7:9] = True
        self.assertFalse(self.select().accepted)

    def test_continuous_diagonal_corridor_is_supported(self):
        self.static[:] = False
        for y in range(self.shape[0]):
            center = 5 + y // 3
            self.static[y, center - 2:center + 3] = True
        result = self.select()
        self.assertTrue(result.accepted)
        self.assertFalse((result.seam_mask & ~self.static).any())

    def test_marking_crossing_needs_explicit_continuity_evidence(self):
        markings = np.zeros(self.shape, bool); markings[6:8] = True
        self.assertFalse(self.select(markings=markings).accepted)
        self.assertTrue(self.select(markings=markings, marking_continuity_verified=markings).accepted)
        with self.assertRaises(ValueError):
            self.select(marking_continuity_verified=markings)

    def test_horizontal_orientation_uses_top_bottom_cores(self):
        self.ha[:] = False; self.ha[:3] = True
        self.hb[:] = False; self.hb[-3:] = True
        result = self.select(orientation="horizontal")
        self.assertTrue(result.accepted)
        self.assertFalse(result.take_b[:3].any())
        self.assertTrue(result.take_b[-3:].all())

    def test_contradictory_or_missing_source_cores_rejected(self):
        self.hb[2, 1] = True
        self.assertEqual(self.select().reason, "conflicting-hard-source-cores")
        self.hb[:] = False
        self.assertEqual(self.select().reason, "both-explicit-source-cores-required")

    def test_modified_real_rgb_or_assignment_rejected_at_composition(self):
        result = self.select()
        altered = self.b.copy(); altered[1, 1, 0] += 1
        with self.assertRaises(ValueError):
            compose_road_seam(self.a, altered, result)
        take_b = result.take_b.copy(); take_b[1, 1] ^= True
        with self.assertRaises(ValueError):
            compose_road_seam(self.a, self.b, replace(result, take_b=take_b))

    def test_nonboolean_mask_and_invalid_limits_rejected(self):
        self.static = self.static.astype(np.uint8)
        with self.assertRaises(ValueError):
            self.select()
        self.static = self.static.astype(bool)
        with self.assertRaises(ValueError):
            self.select(config=RoadSeamConfig(max_step=5))
        with self.assertRaises(ValueError):
            self.select(config=RoadSeamConfig(max_rgb_difference=float("nan")))


if __name__ == "__main__":
    unittest.main()
