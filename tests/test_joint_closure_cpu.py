import unittest

import numpy as np

from model_engine.ogs2_joint_closure import close_whole_foregrounds, reserve_joint_components


def region(start, stop, shape=(12, 30)):
    mask = np.zeros(shape, bool)
    mask[4:8, start:stop] = True
    return mask


class WholeForegroundClosureContract(unittest.TestCase):
    def test_three_foreground_chain_closes_even_when_catalog_order_is_reversed(self):
        seed = region(2, 4)
        masks = {"far": region(9, 15), "middle": region(5, 10), "near": region(3, 6)}
        result = close_whole_foregrounds(seed, masks)
        reverse = close_whole_foregrounds(seed, dict(reversed(list(masks.items()))))
        self.assertEqual(result.included_foregrounds, ("far", "middle", "near"))
        np.testing.assert_array_equal(result.support, reverse.support)
        self.assertTrue(result.support[4:8, 2:15].all())
        self.assertFalse(result.support[:, 16:].any())

    def test_protection_margin_can_reach_another_whole_vehicle(self):
        result = close_whole_foregrounds(region(2, 4), {"a": region(3, 5), "b": region(6, 10)}, protection_steps=2)
        self.assertEqual(result.included_foregrounds, ("a", "b"))
        self.assertTrue(result.support[4:8, 6:10].all())

    def test_whole_expansion_replans_components_before_writing(self):
        initial = [region(2, 4), region(18, 20)]
        _, components = reserve_joint_components(initial, [{"long": region(3, 18)}, {}], collar_steps=1)
        self.assertEqual(components, [(0, 1)])

    def test_disjoint_boxes_alone_do_not_merge_writers(self):
        _, components = reserve_joint_components([region(2, 4), region(18, 20)], [{}, {}], collar_steps=1)
        self.assertEqual(components, [(0,), (1,)])

    def test_joint_catalog_closure_can_reach_a_third_previously_disjoint_writer(self):
        seeds = [region(2, 4), region(5, 7), region(19, 21)]
        catalogs = [{}, {"neighbor-seen-only-in-b-source": region(3, 19)}, {}]
        closures, components = reserve_joint_components(seeds, catalogs, collar_steps=1)
        self.assertEqual(components, [(0, 1, 2)])
        self.assertTrue(closures[0].support[4:8, 2:21].all())

    def test_same_observation_id_cannot_hide_different_masks(self):
        with self.assertRaisesRegex(ValueError, "contradictory"):
            reserve_joint_components([region(2, 4), region(5, 7)],
                                     [{"obs-1": region(3, 6)}, {"obs-1": region(3, 10)}],
                                     collar_steps=1)

    def test_budget_failure_never_returns_partial_vehicle_or_mutates_input(self):
        seed = region(2, 4)
        original = seed.copy()
        with self.assertRaisesRegex(ValueError, "budget"):
            close_whole_foregrounds(seed, {"vehicle": region(3, 20)}, max_support_pixels=20)
        np.testing.assert_array_equal(seed, original)

    def test_touching_foreground_is_preserved_complete_without_becoming_identity(self):
        result = close_whole_foregrounds(region(2, 4), {"actual-neighbor-car": region(3, 6), "unrelated": region(20, 25)})
        self.assertEqual(result.included_foregrounds, ("actual-neighbor-car",))
        self.assertFalse(result.support[:, 20:25].any())


if __name__ == "__main__":
    unittest.main()
