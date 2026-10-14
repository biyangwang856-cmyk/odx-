"""Tiny CPU contracts for the real minimum-cut reference solver."""

import unittest

import numpy as np

from model_engine.ogs2_joint_policy import compose_static_collar
from model_engine.ogs2_joint_seam import SeamConfig, _gradient, static_seam_weights


class SeamTests(unittest.TestCase):
    def inputs(self):
        baseline = np.full((20, 24, 4), 60, np.uint8)
        baseline[..., 3] = 255
        source = np.full((20, 24, 3), 67, np.uint8)
        hard = np.zeros((20, 24), bool)
        hard[8:12, 10:14] = True
        return baseline, source, hard, ~hard, np.ones(hard.shape, bool)

    def run_seam(self, inputs, **options):
        return static_seam_weights(*inputs, backend="python", **options)

    def test_real_cut_keeps_full_dynamic_support_and_records_static_mixing(self):
        inputs = self.inputs()
        weight, mode, selected, report = self.run_seam(inputs)
        image, composition, saved_weight = compose_static_collar(*inputs[:3], weight, *inputs[3:])
        hard = inputs[2]
        self.assertTrue(np.all(weight[hard] == 1))
        np.testing.assert_array_equal(image[hard, :3], inputs[1][hard])
        np.testing.assert_array_equal(image[weight == 0], inputs[0][weight == 0])
        np.testing.assert_array_equal(image[..., 3], inputs[0][..., 3])
        np.testing.assert_array_equal(mode, composition)
        np.testing.assert_array_equal(weight, saved_weight)
        self.assertTrue(report["noWarp"])
        self.assertLessEqual(report["cutEnergy"], report["feasibleCutEnergy"])
        self.assertGreater(report["mixedStaticPixels"], 0)
        self.assertTrue(selected[hard].all())

    def test_unknown_outer_area_is_exactly_baseline(self):
        inputs = list(self.inputs())
        inputs[3][:, :4] = False
        weight, _, _, _ = self.run_seam(inputs)
        self.assertTrue(np.all(weight[:, :4] == 0))

    def test_unknown_or_invalid_hard_shadow_rejects_without_partial_composition(self):
        inputs = list(self.inputs())
        inputs[4][8, 10] = False
        original = inputs[0].copy()
        with self.assertRaisesRegex(ValueError, "unknown"):
            self.run_seam(inputs)
        np.testing.assert_array_equal(inputs[0], original)

    def test_unknown_neighbor_cannot_form_a_safe_collar(self):
        inputs = list(self.inputs())
        inputs[3][7, 10] = False
        with self.assertRaisesRegex(ValueError, "static collar"):
            self.run_seam(inputs)

    def test_cutting_an_adjacent_car_is_rejected(self):
        inputs = self.inputs()
        other = np.zeros(inputs[2].shape, bool)
        other[8:12, 8:12] = True
        with self.assertRaisesRegex(ValueError, "protected foreground"):
            self.run_seam(inputs, protected_foregrounds=[other])

    def test_whole_adjacent_foreground_remains_baseline_and_is_not_feathered(self):
        inputs = self.inputs()
        other = np.zeros(inputs[2].shape, bool)
        other[8:12, 3:6] = True
        weight, _, _, _ = self.run_seam(inputs, protected_foregrounds=[other])
        self.assertTrue(np.all(weight[other] == 0))

    def test_high_gradient_misaligned_road_marks_are_never_fractionally_blended(self):
        inputs = list(self.inputs())
        inputs[0][3:17, 6:8, :3] = 240
        inputs[1][3:17, 3:5] = 240  # Three-pixel road-marker disagreement.
        weight, _, _, _ = self.run_seam(inputs)
        high_gradient = np.maximum(_gradient(inputs[0][..., :3]), _gradient(inputs[1])) > 10
        self.assertFalse(np.any((weight > 0) & (weight < 1) & high_gradient))

    def test_large_exposure_difference_is_a_pure_seam_not_a_blurred_vehicle(self):
        inputs = list(self.inputs())
        inputs[1][:] = 110
        weight, _, _, report = self.run_seam(inputs)
        self.assertFalse(np.any((weight > 0) & (weight < 1)))
        self.assertGreater(report["cutRgbDifferenceP90"], 12)

    def test_hard_support_touching_window_edge_requires_more_context(self):
        inputs = list(self.inputs())
        inputs[2][0, 0] = True
        with self.assertRaisesRegex(ValueError, "context"):
            self.run_seam(inputs)

    def test_reference_solver_cannot_accidentally_run_full_scene(self):
        inputs = self.inputs()
        with self.assertRaisesRegex(ValueError, "pixel budget"):
            self.run_seam(inputs, config=SeamConfig(python_max_pixels=100))


if __name__ == "__main__":
    unittest.main()
