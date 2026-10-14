"""File-bound certificate and conflicting-writer counterexamples, without models."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from model_engine.ogs2_corridor_apply import (
    CERTIFICATE_SCHEMA, ReviewedPatch, candidate_from_certificate, file_sha256,
    export_reviewed_full, load_certificate_arrays, reject_overlapping_patches, review_certificates,
)
from model_engine.ogs2_corridor_policy import array_sha256


def collar(mask, steps=2):
    expanded = mask.copy()
    for _ in range(steps):
        old = expanded.copy()
        expanded[1:] |= old[:-1]
        expanded[:-1] |= old[1:]
        expanded[:, 1:] |= old[:, :-1]
        expanded[:, :-1] |= old[:, 1:]
    return expanded & ~mask


class CertificateTests(unittest.TestCase):
    def setUp(self):
        # Keep temporary test files within the shared workspace on Windows.
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)
        self.shape = (28, 42)
        self.base = np.full((*self.shape, 4), 120, np.uint8)
        self.base[..., 3] = 255
        self.body = np.zeros(self.shape, bool)
        self.body[11:14, 13:17] = True
        self.conflict = np.zeros(self.shape, bool)
        self.conflict[9:18, 9:32] = True
        self.masks = {
            "conflict": self.conflict, "road": np.ones(self.shape, bool),
            "empty": np.ones(self.shape, bool), "static": np.ones(self.shape, bool),
            "unknown": np.zeros(self.shape, bool), "whole": self.body.copy(),
        }
        self.arrays = {"image": np.full((*self.shape, 3), 90, np.uint8),
                       "valid": np.ones(self.shape, bool), "cost": np.full(self.shape, 3.0, np.float32),
                       "body_mask": self.body, "vehicle_0": self.body}
        self.base_path = self.root / "baseline.tif"
        self.base_path.write_bytes(b"fixture-baseline-file")
        self.cache = self.root / "source-000.npz"
        np.savez_compressed(self.cache, **self.arrays)
        self.metadata = self.cache.with_suffix(".json")
        self.metadata.write_text("{}", encoding="utf-8")
        self.source_record = {
            "sourceImage": "actual-photo.jpg", "captureTime": 42.0, "captureTimeDomain": "dji-wall-clock",
            "cachePath": str(self.cache), "cacheSha256": file_sha256(self.cache),
            "sourceMetadataSha256": file_sha256(self.metadata),
            "vehicles": [{"maskKey": "vehicle_0", "complete": True}],
        }
        proof = {
            "accepted": True, "referenceKind": "orthophoto-baseline", "sourceId": "actual-photo.jpg",
            "sourceImageSha256": array_sha256(self.arrays["image"]),
            "baselineImageSha256": array_sha256(self.base[..., :3]),
            "baselineRgbaSha256": array_sha256(self.base), "staticSupportSha256": array_sha256(self.masks["static"]),
            "trainingPoints": 8, "heldoutPoints": 8, "inliers": 16,
            "p90ActualPixels": 0.8, "holdoutResidualP90Pixels": 0.5,
            "occupiedCells": [0, 3, 12, 15], "spanFractions": [0.6, 0.3],
            "consensusFraction": 0.8, "fittedCorrectionApplied": False,
        }
        self.source_certificate = {
            "sourceImage": "actual-photo.jpg",
            "sourceArraysSha256": {k: array_sha256(self.arrays[k]) for k in ("image", "valid", "cost", "body_mask")},
            "masks": {"observed_road": "road", "verified_empty_road": "empty", "static_support": "static", "unknown_foreground": "unknown"},
            "foregrounds": {"car-and-observed-shadow": "whole"},
            "complete_foregrounds": ["car-and-observed-shadow"],
            "foreground_catalog_complete": True, "static_verification": proof,
        }
        np.savez_compressed(self.root / "baseline-roi.npz", rgba=self.base, corridor=np.ones(self.shape, bool))
        self.evidence = {
            "schema": "ogs2.corridor-projection/v1", "state": "complete",
            "criticalInputs": {str(self.base_path): file_sha256(self.base_path)}, "sourceInputHashes": {},
            "snapshotSha256": file_sha256(self.root / "baseline-roi.npz"),
            "grid": {"referenceWindow": [0, 0, self.shape[1], self.shape[0]]},
        }
        self.evidence_path = self.root / "manifest.json"
        self.write_evidence()
        self.mask_path = self.root / "certificate-masks.npz"
        np.savez_compressed(self.mask_path, **self.masks)
        self.mask_binding = {"path": str(self.mask_path), "sha256": file_sha256(self.mask_path),
                             "arraySha256": {k: array_sha256(v) for k, v in self.masks.items()}}
        self.manifest = {
            "schema": CERTIFICATE_SCHEMA, "state": "complete",
            "baseline": {"path": str(self.base_path), "sha256": file_sha256(self.base_path)},
            "patches": [{"id": "west", "evidenceManifest": {"path": str(self.evidence_path), "sha256": file_sha256(self.evidence_path)},
                         "certificateArrays": self.mask_binding,
                         "masks": {"conflict_union": "conflict", "road_domain": "road", "protected_unknown": "unknown", "baseline_foregrounds": {}},
                         "candidates": [self.source_certificate]}],
        }
        self.cert_path = self.root / "certificates.json"
        self.write_certificate()

    def tearDown(self):
        self.temp.cleanup()

    def write_evidence(self):
        self.evidence_path.write_text(json.dumps(self.evidence), encoding="utf-8")

    def write_certificate(self):
        self.cert_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def source_loader(self, path, source):
        self.assertEqual(path, self.evidence_path)
        self.assertEqual(source, "actual-photo.jpg")
        return self.arrays, self.source_record, self.evidence

    def baseline_loader(self, path, grid):
        self.assertEqual(path, self.base_path)
        return self.base.copy(), (0, 0, self.shape[1], self.shape[0])

    def review(self):
        return review_certificates(self.cert_path, source_loader=self.source_loader, baseline_loader=self.baseline_loader)

    def test_complete_hash_bound_certificate_uses_actual_source_pixels(self):
        base, patches, report = self.review()
        self.assertEqual(base, self.base_path)
        self.assertEqual(len(patches), 1)
        np.testing.assert_array_equal(patches[0].output_rgba[..., :3][patches[0].hard_support],
                                      self.arrays["image"][patches[0].hard_support])
        self.assertTrue(report["allInputHashesVerifiedAfterReview"])
        self.assertFalse(report["identityAssociationAssumed"])

    def test_changed_source_rgb_invalidates_unchanged_old_certificate(self):
        self.arrays["image"][12, 20] = 0
        with self.assertRaisesRegex(ValueError, "Source array hash"):
            self.review()

    def test_changed_mask_archive_invalidates_file_binding(self):
        with self.mask_path.open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "Pinned input changed"):
            self.review()

    def test_declared_archive_hash_does_not_excuse_wrong_array_hash(self):
        self.mask_binding["arraySha256"]["empty"] = "0" * 64
        self.write_certificate()
        with self.assertRaisesRegex(ValueError, "array hash mismatch"):
            self.review()

    def test_undeclared_masks_cannot_enter_certificate(self):
        del self.mask_binding["arraySha256"]["whole"]
        self.write_certificate()
        with self.assertRaisesRegex(ValueError, "Every certificate mask"):
            self.review()

    def test_nan_masks_are_not_treated_as_true(self):
        self.masks["empty"] = np.full(self.shape, np.nan)
        np.savez_compressed(self.mask_path, **self.masks)
        binding = {"path": str(self.mask_path), "sha256": file_sha256(self.mask_path),
                   "arraySha256": {k: array_sha256(v) for k, v in self.masks.items()}}
        with self.assertRaisesRegex(ValueError, "boolean"):
            load_certificate_arrays(binding, self.root, {})

    def test_omitted_real_source_body_cannot_be_called_empty_road(self):
        certificate = {**self.source_certificate, "foregrounds": {}, "complete_foregrounds": []}
        with self.assertRaisesRegex(ValueError, "omits actually observed"):
            candidate_from_certificate(certificate, self.arrays, self.source_record, self.masks, self.base, self.conflict)

    def test_truncated_source_body_must_be_unknown_instead_of_claimed_complete(self):
        record = {**self.source_record, "vehicles": [{"maskKey": "vehicle_0", "complete": False}]}
        with self.assertRaisesRegex(ValueError, "incomplete source-body uncertainty"):
            candidate_from_certificate(self.source_certificate, self.arrays, record, self.masks, self.base, self.conflict)

    def test_source_to_source_proof_fails_actual_baseline_acceptance(self):
        self.source_certificate["static_verification"]["referenceKind"] = "another-source"
        self.write_certificate()
        _, patches, report = self.review()
        self.assertEqual(patches, [])
        self.assertEqual(report["patches"][0]["candidateDecisions"][0]["reason"], "baseline-static-reference-required")

    def test_one_unverified_empty_road_pixel_rejects_region_not_just_that_pixel(self):
        self.masks["empty"][12, 20] = False
        np.savez_compressed(self.mask_path, **self.masks)
        self.mask_binding.update(sha256=file_sha256(self.mask_path), arraySha256={k: array_sha256(v) for k, v in self.masks.items()})
        self.write_certificate()
        _, patches, report = self.review()
        self.assertFalse(patches)
        self.assertEqual(report["acceptedPatches"], 0)

    def test_changed_actual_baseline_snapshot_is_rejected(self):
        self.base[10, 20, 3] = 1
        with self.assertRaisesRegex(ValueError, "Actual baseline ROI differs"):
            self.review()

    def test_certificate_may_not_relax_policy_thresholds(self):
        self.manifest["policyConfig"] = {"max_static_displacement_px": 100}
        self.write_certificate()
        with self.assertRaisesRegex(ValueError, "relax policy guards"):
            self.review()

    def test_inputs_modified_during_review_are_detected_at_end(self):
        loader = self.source_loader
        def modify(*args):
            result = loader(*args)
            self.cert_path.write_text("{}", encoding="utf-8")
            return result
        with self.assertRaisesRegex(ValueError, "Input changed during"):
            review_certificates(self.cert_path, source_loader=modify, baseline_loader=self.baseline_loader)

    def test_context_overlap_without_write_overlap_is_allowed_but_shared_writer_is_rejected(self):
        def patch(identity, x, support):
            return ReviewedPatch(identity, (x, 0, self.shape[1], self.shape[0]), {}, self.base, self.base,
                                 support, self.arrays["image"], identity, {}, collar(support))
        left = self.conflict.copy()
        right = np.zeros(self.shape, bool)
        right[9:18, 35:39] = True
        accepted, conflicts = reject_overlapping_patches([patch("a", 0, left), patch("b", 0, right)])
        self.assertEqual(len(accepted), 2)
        self.assertFalse(conflicts)
        accepted, conflicts = reject_overlapping_patches([patch("a", 0, left), patch("b", 0, left)])
        self.assertFalse(accepted)
        self.assertEqual(conflicts[0]["overlapPixels"], int(left.sum()))

    def test_disjoint_hard_regions_must_not_write_into_other_certified_static_collar(self):
        left, right = self.conflict.copy(), np.zeros(self.shape, bool)
        right[9:18, 33:38] = True  # One empty pixel column separates the hard regions.
        self.assertFalse((left & right).any())
        def patch(identity, hard):
            return ReviewedPatch(identity, (0, 0, self.shape[1], self.shape[0]), {}, self.base, self.base,
                                 hard, self.arrays["image"], identity, {}, collar(hard))
        accepted, conflicts = reject_overlapping_patches([patch("a", left), patch("b", right)])
        self.assertFalse(accepted)
        self.assertEqual(conflicts[0]["hardOverlapPixels"], 0)
        self.assertGreater(conflicts[0]["staticReservationConflictPixels"], 0)

    def test_after_review_array_or_reservation_mutation_is_detected(self):
        for field in ("source_rgb", "hard_support", "static_collar", "baseline_rgba"):
            with self.subTest(field=field):
                patch = ReviewedPatch("a", (0, 0, self.shape[1], self.shape[0]), {}, self.base.copy(), self.base.copy(),
                                      self.conflict.copy(), self.arrays["image"].copy(), "actual-photo.jpg", {}, collar(self.conflict))
                array = getattr(patch, field)
                array.flat[0] = 1 if array.dtype == bool else 13
                with self.assertRaisesRegex(ValueError, "changed before export"):
                    reject_overlapping_patches([patch])


try:
    import rasterio as rio
    from rasterio.enums import ColorInterp
    from rasterio.transform import from_origin
except ImportError:
    rio = None


@unittest.skipIf(rio is None, "Server rasterio required for actual GeoTIFF readback fixture")
class FullRasterExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.root = Path(self.temp.name)
        self.shape = (55, 70)
        self.base = np.random.default_rng(24).integers(0, 255, (*self.shape, 4), dtype=np.uint8)
        self.base[..., 3] = 255
        self.base[:5, :5, 3] = 0
        self.path = self.root / "base.tif"
        with rio.open(self.path, "w", driver="GTiff", height=55, width=70, count=4,
                      dtype="uint8", crs="EPSG:32650", transform=from_origin(191706, 2494789, 0.03, 0.03),
                      tiled=True, blockxsize=16, blockysize=16) as dst:
            dst.write(self.base.transpose(2, 0, 1))
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
            dst.update_tags(fixture="retained-metadata")
            dst.set_band_description(1, "red retained")
        self.output = self.root / "output"
        self.output.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def patch(self, identity, x, y, rgb_value):
        window = (x, y, 35, 30)
        original = self.base[y:y + 30, x:x + 35].copy()
        source = np.full((30, 35, 3), rgb_value, np.uint8)
        hard = np.zeros((30, 35), bool)
        hard[6:13, 7:17] = True
        replacement = original.copy()
        replacement[..., :3][hard] = source[hard]
        return ReviewedPatch(identity, window, {}, original, replacement, hard, source, identity, {}, collar(hard))

    def test_real_full_tiff_preserves_alpha_outside_and_exact_selected_source(self):
        patches = [self.patch("first", 5, 6, 13), self.patch("second", 25, 20, 221)]
        report = export_reviewed_full(self.path, patches, self.output,
                                     {str(self.path): file_sha256(self.path)}, {"first": 1, "second": 2})
        with rio.open(report["files"]["orthophoto"]["path"]) as dst, rio.open(self.path) as base:
            actual = dst.read().transpose(1, 2, 0)
            self.assertEqual(dst.transform, base.transform)
            self.assertEqual(dst.crs, base.crs)
            self.assertEqual(dst.colorinterp, base.colorinterp)
            self.assertEqual(dst.tags()["fixture"], "retained-metadata")
            self.assertEqual(dst.descriptions[0], "red retained")
        permitted = np.zeros(self.shape, bool)
        for patch in patches:
            x, y, width, height = patch.window
            permitted[y:y + height, x:x + width] |= patch.hard_support
            np.testing.assert_array_equal(actual[y:y + height, x:x + width, :3][patch.hard_support],
                                          patch.source_rgb[patch.hard_support])
        np.testing.assert_array_equal(actual[~permitted], self.base[~permitted])
        np.testing.assert_array_equal(actual[..., 3], self.base[..., 3])
        self.assertEqual(report["writtenPixels"], int(permitted.sum()))
        self.assertEqual(report["outsideSupportChangedPixels"], 0)
        self.assertEqual(report["alphaChangedPixels"], 0)
        self.assertEqual(report["realSourceRgbMismatchPixels"], 0)

    def test_overlapping_real_writers_do_not_publish_a_full_tiff(self):
        a, b = self.patch("a", 5, 6, 13), self.patch("b", 5, 6, 221)
        with self.assertRaisesRegex(RuntimeError, "writers overlap"):
            export_reviewed_full(self.path, [a, b], self.output,
                                 {str(self.path): file_sha256(self.path)}, {"a": 1, "b": 2})
        self.assertFalse((self.output / "corridor_orthophoto.tif").exists())

    def test_disjoint_real_writers_entering_certified_collar_do_not_publish_full_tiff(self):
        a, b = self.patch("a", 5, 6, 13), self.patch("b", 16, 6, 221)
        self.assertFalse((a.hard_support[:, 11:] & b.hard_support[:, :-11]).any())
        with self.assertRaisesRegex(RuntimeError, "static reservation"):
            export_reviewed_full(self.path, [a, b], self.output,
                                 {str(self.path): file_sha256(self.path)}, {"a": 1, "b": 2})
        self.assertFalse((self.output / "corridor_orthophoto.tif").exists())

    def test_after_review_rgb_mutation_cannot_become_new_expected_source(self):
        patch = self.patch("a", 5, 6, 13)
        patch.source_rgb[7, 8] = 222
        with self.assertRaisesRegex(ValueError, "changed before export"):
            export_reviewed_full(self.path, [patch], self.output,
                                 {str(self.path): file_sha256(self.path)}, {"a": 1})
        self.assertFalse((self.output / "corridor_orthophoto.tif").exists())


if __name__ == "__main__":
    unittest.main()
