"""Production SciPy flow contracts; skipped when SciPy is not installed.

The tiny Python solver is independent of SciPy's sparse residual traversal.
Equal minimum-cut energy is required; labels may differ on tied minimum cuts.
"""

import importlib.util
import unittest

import numpy as np

from model_engine.ogs2_joint_seam import static_seam_weights


@unittest.skipUnless(importlib.util.find_spec("scipy"), "SciPy unavailable in this runtime")
class ScipyMinimumCutContract(unittest.TestCase):
    def fixture(self):
        base = np.full((20, 24, 4), 65, np.uint8)
        base[..., 3] = 255
        source = np.full((20, 24, 3), 68, np.uint8)
        # Some structure and illumination variation make this a genuine
        # weighted cut, rather than a single all-constant graph fixture.
        base[4:16, 4:6, :3] = 130
        source[4:16, 5:7] = 125
        source[2:6, 12:18] = 90
        hard = np.zeros(base.shape[:2], bool)
        hard[8:12, 10:14] = True
        static = ~hard
        static[:, :2] = False
        valid = np.ones_like(hard)
        return base, source, hard, static, valid

    def test_sparse_backend_and_independent_reference_have_equal_cut_energy(self):
        inputs = self.fixture()
        _, _, _, reference = static_seam_weights(*inputs, backend="python")
        weight, mode, selected, production = static_seam_weights(*inputs, backend="scipy")
        self.assertEqual(production["cutEnergy"], reference["cutEnergy"])
        self.assertLessEqual(production["cutEnergy"], production["feasibleCutEnergy"])
        self.assertTrue(selected[inputs[2]].all())
        self.assertTrue((weight[inputs[2]] == 1).all())
        self.assertTrue((mode[inputs[2]] == 1).all())
        self.assertTrue((weight[:, :2] == 0).all())

    def test_sparse_residual_selection_never_moves_the_hard_source_pin(self):
        inputs = list(self.fixture())
        inputs[0][..., :3] = 0
        inputs[1][...] = 255
        weight, _, selected, report = static_seam_weights(*inputs, backend="scipy")
        self.assertTrue(selected[inputs[2]].all())
        self.assertTrue((weight[inputs[2]] == 1).all())
        self.assertFalse(np.any((weight > 0) & (weight < 1)))
        self.assertLessEqual(report["cutEnergy"], report["feasibleCutEnergy"])


if __name__ == "__main__":
    unittest.main()
