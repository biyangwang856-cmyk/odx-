"""Counterexamples for bounded map-grid source sampling corrections."""
import unittest
from dataclasses import replace

import numpy as np

from model_engine.ogs2_road_registration import RoadRegistrationConfig, fit_road_registration


class RoadRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.shape = (640, 768)
        yy, xx = np.mgrid[32:608:32, 32:736:32]
        self.a = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        self.c = RoadRegistrationConfig()

    def fit(self, b, **kwargs):
        return fit_road_registration(self.a, b, self.shape, config=kwargs.get('config', self.c))

    def test_translation_direction_is_output_to_source(self):
        result = self.fit(self.a+[3, -2])
        self.assertTrue(result.accepted, result.report)
        self.assertEqual(result.report['selectedModel'], 'translation')
        np.testing.assert_allclose(result.source_coordinates([[100, 100]]), [[103, 98]])
        self.assertLess(result.report['afterHoldoutP90Px'], 1e-8)
        self.assertFalse(result.report['wholeOrthophotoCertified'])

    def test_coordinate_origin_shift_preserves_physical_holdout(self):
        b=self.a+[3,-2]
        original=self.fit(b)
        offset=np.array([250.,200.])
        expanded=fit_road_registration(self.a+offset,b+offset,(940,1268),
            config=replace(self.c,cell_origin_xy=tuple(offset)))
        self.assertTrue(expanded.accepted,expanded.report)
        self.assertEqual(original.report['outerHoldoutIndices'],expanded.report['outerHoldoutIndices'])
        self.assertEqual(original.report['trainingIndices'],expanded.report['trainingIndices'])
        self.assertAlmostEqual(original.report['afterHoldoutP90Px'],expanded.report['afterHoldoutP90Px'],places=8)
        np.testing.assert_allclose(original.matrix,expanded.matrix)

    def test_affine_selected_using_inner_validation(self):
        b = self.a @ np.array([[1.004, .002], [-.001, .997]]).T + [1, 2]
        result = self.fit(b)
        self.assertTrue(result.accepted, result.report)
        self.assertEqual(result.report['selectedModel'], 'affine')
        self.assertFalse(result.report['outerHoldoutUsedForModelSelection'])
        self.assertLess(result.report['afterHoldoutP90Px'], 1e-8)

    def test_outer_holdout_outlier_is_not_removed(self):
        b = self.a+[2, 1]
        clean = self.fit(b)
        b[clean.report['outerHoldoutIndices'][0]] += [4, 0]
        result = self.fit(b)
        self.assertFalse(result.accepted)
        self.assertGreater(result.report['afterHoldoutMaxPx'], 3.9)
        self.assertEqual(result.report['selectedModel'], clean.report['selectedModel'])
        np.testing.assert_allclose(result.matrix, clean.matrix)
        self.assertFalse(result.report['outerHoldoutOutliersRemoved'])

    def test_outer_spoof_cannot_select_affine(self):
        b = self.a+[2, 1]
        clean = self.fit(b)
        idx = clean.report['outerHoldoutIndices']
        b[idx] += .009*(self.a[idx]-[384, 320])
        result = self.fit(b)
        self.assertFalse(result.accepted)
        self.assertEqual(result.report['selectedModel'], 'translation')
        np.testing.assert_allclose(result.matrix, clean.matrix)

    def test_displacement_checked_on_domain_not_just_matches(self):
        # Small central point displacement cannot authorize excessive extrapolation.
        b = self.a+[9, 0]
        result = self.fit(b)
        self.assertFalse(result.accepted)
        self.assertIn('no-bounded-inner-model', result.report['reasons'])

    def test_domain_corners_reject_affine_extrapolation(self):
        a = self.a[(self.a[:, 0] >= 160) & (self.a[:, 0] <= 608)
                   & (self.a[:, 1] >= 128) & (self.a[:, 1] <= 480)]
        b = a + .008*(a-[384, 320]) + [5, 0]
        self.assertLess(np.linalg.norm(b-a, axis=1).max(), 8)
        result = fit_road_registration(a, b, self.shape)
        affine = next(x for x in result.report['modelCandidates'] if x['model'] == 'affine')
        self.assertGreater(affine['maxDomainDisplacementPx'], 8)
        self.assertFalse(affine['eligible'])

    def test_correction_cannot_be_evaluated_outside_domain(self):
        result = self.fit(self.a+[1, 2])
        with self.assertRaises(ValueError):
            result.source_coordinates([[2000, 2000]])

    def test_jacobian_scale_bound(self):
        b = (self.a-[384, 320])*1.02+[384, 320]
        result = self.fit(b)
        self.assertFalse(result.accepted)
        affine = next(x for x in result.report['modelCandidates'] if x['model'] == 'affine')
        self.assertFalse(affine['eligible'])

    def test_no_caller_holdout_flags_supported(self):
        with self.assertRaises(TypeError):
            fit_road_registration(self.a, self.a, self.shape, heldout=np.ones(len(self.a), bool))

    def test_collinear_evidence_rejected(self):
        a = self.a.copy()
        a[:, 1] = a[:, 0]*.3
        a = np.unique(a, axis=0)
        result = fit_road_registration(a, a+[1, 1], self.shape)
        self.assertFalse(result.accepted)

    def test_duplicate_measurements_rejected(self):
        a = np.vstack((self.a, self.a[0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            fit_road_registration(a, a+[1, 1], self.shape)

    def test_out_of_grid_and_nonfinite_rejected(self):
        for invalid in ([np.nan, 2], [-1, 2], [900, 2]):
            b = self.a.copy()
            b[0] = invalid
            with self.assertRaises(ValueError):
                self.fit(b)

    def test_input_order_does_not_change_fit(self):
        b = self.a+[2, -1]
        r1 = self.fit(b)
        indices = np.random.default_rng(4).permutation(len(b))
        r2 = fit_road_registration(self.a[indices], b[indices], self.shape)
        np.testing.assert_allclose(r1.matrix, r2.matrix)
        self.assertEqual(r1.accepted, r2.accepted)

    def test_rejected_matrix_requires_explicit_diagnostic(self):
        b = self.a+[2, 1]
        clean = self.fit(b)
        b[clean.report['outerHoldoutIndices'][0]] += [4, 0]
        r = self.fit(b)
        with self.assertRaises(ValueError):
            r.source_coordinates([[100, 100]])
        np.testing.assert_allclose(r.source_coordinates([[100, 100]], allow_diagnostic=True), [[102, 101]])

    def test_config_rejects_invalid_jacobian(self):
        with self.assertRaises(ValueError):
            self.fit(self.a, config=replace(self.c, min_jacobian_singular_value=1.1))

    def test_identity_not_needlessly_promoted_to_affine(self):
        r = self.fit(self.a.copy())
        self.assertTrue(r.accepted, r.report)
        self.assertEqual(r.report['selectedModel'], 'translation')


if __name__ == '__main__':
    unittest.main()
