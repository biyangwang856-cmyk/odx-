"""Independent explicit-reference contracts, no detector or image runtime."""
import unittest

import numpy as np

from model_engine.ogs2_joint_cohort import ExposureReference, merge_with_explicit_reference
from model_engine.ogs2_joint_policy import PatchPlan, SparseRegion, merge_conflicts


def plan(identity, time, source, origin=(0, 0), mask=None, clock="dji-exif-utc"):
    return PatchPlan(
        identity, SparseRegion(origin, np.ones((5, 5), bool) if mask is None else mask),
        frozenset({identity}), {source: frozenset({identity})}, clock, time, time + 1,
    )


class ReferenceCohortContract(unittest.TestCase):
    def test_long_time_component_has_explicit_real_reference_and_all_time_support(self):
        plans = [plan("bus", 0, "a"), plan("future-white-car", 53, "b", (3, 0))]
        refs = {"a": ExposureReference(0.5, "dji-exif-utc"), "b": ExposureReference(53.5, "dji-exif-utc")}
        combined = merge_with_explicit_reference(plans, refs)[0]
        self.assertIsNone(combined.rejection_reason)
        self.assertEqual(len(combined.reference_cohorts), 2)
        self.assertEqual(combined.support.mask.sum(), 40)
        first = combined.reference_cohorts[0]
        self.assertEqual(first.source_image, "a")
        self.assertEqual(first.required_tracks, frozenset({"bus"}))
        self.assertEqual(first.excluded_temporal_tracks, frozenset({"future-white-car"}))
        self.assertEqual(first.capture_time, 0.5)
        # The conservative mode remains unchanged and continues withholding it.
        self.assertEqual(merge_conflicts(plans)[0].rejection_reason, "incompatible-temporal-cohorts")

    def test_actual_cooccurring_identities_are_all_required(self):
        plans = [plan("bus", 1, "a"), plan("car", 1, "a", (2, 0))]
        combined = merge_with_explicit_reference(plans, {"a": ExposureReference(1.5, "dji-exif-utc")})[0]
        self.assertEqual(combined.reference_cohorts[0].required_tracks, frozenset({"bus", "car"}))

    def test_spatially_disjoint_supports_keep_independent_exposures(self):
        combined = merge_with_explicit_reference(
            [plan("bus", 0, "a"), plan("car", 53, "b", (100, 0))],
            {"a": ExposureReference(.5, "dji-exif-utc"), "b": ExposureReference(53.5, "dji-exif-utc")},
        )
        self.assertEqual(len(combined), 2)
        self.assertEqual([len(p.reference_cohorts) for p in combined], [1, 1])

    def test_cross_clock_component_is_not_a_shared_time(self):
        plans = [plan("bus", 0, "a"), plan("car", 0, "b", (3, 0), clock="filename-clock")]
        combined = merge_with_explicit_reference(plans, {})[0]
        self.assertEqual(combined.rejection_reason, "incompatible-clocks")
        self.assertIsNone(combined.support)

    def test_missing_reference_or_wrong_clock_cannot_appear_as_valid_photo(self):
        combined = merge_with_explicit_reference([plan("bus", 1, "a")], {})[0]
        self.assertEqual(combined.rejected_sources, (("a", "missing-reference-time"),))
        combined = merge_with_explicit_reference([plan("bus", 1, "a")], {"a": ExposureReference(1.5, "unknown")})[0]
        self.assertEqual(combined.rejected_sources, (("a", "incompatible-reference-clock"),))

    def test_foreign_timestamp_and_empty_vehicle_source_cannot_win(self):
        base = plan("bus", 1, "a")
        plans = [PatchPlan(base.identity, base.support, base.required_tracks,
                           {"a": frozenset({"bus"}), "empty": frozenset()}, base.clock, 1, 2)]
        refs = {"a": ExposureReference(53, base.clock), "empty": ExposureReference(1.5, base.clock)}
        combined = merge_with_explicit_reference(plans, refs)[0]
        self.assertEqual(dict(combined.rejected_sources)["a"], "reference-time-outside-observed-track")
        self.assertEqual(dict(combined.rejected_sources)["empty"], "no-real-vehicle-anchor")
        self.assertEqual(combined.rejection_reason, "no-valid-explicit-reference")

    def test_union_budget_rejected_before_large_allocation(self):
        combined = merge_with_explicit_reference([plan("bus", 1, "a")], {}, max_pixels=4)[0]
        self.assertEqual(combined.rejection_reason, "joint-support-exceeds-budget")
        self.assertIsNone(combined.support)

    def test_component_and_catalog_order_does_not_change_cohort(self):
        plans = [plan("bus", 0, "a"), plan("car", 53, "b", (3, 0))]
        refs = {"b": ExposureReference(53.5, "dji-exif-utc"), "a": ExposureReference(.5, "dji-exif-utc")}
        forward = merge_with_explicit_reference(plans, refs)[0]
        reverse = merge_with_explicit_reference(plans[::-1], refs)[0]
        self.assertEqual(forward.reference_cohorts, reverse.reference_cohorts)
        np.testing.assert_array_equal(forward.support.mask, reverse.support.mask)


if __name__ == "__main__":
    unittest.main()
