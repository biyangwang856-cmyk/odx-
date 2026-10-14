"""Small decision-contract checks for the projected-SfM controller."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("controller", Path(__file__).with_name("validate_projected_sfm.py"))
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def healthy_audit():
    row = {"count": 100, "finiteCount": 100, "nonFiniteCount": 0,
           "meanFullPixels": 1.5, "p50FullPixels": 1.0, "p90FullPixels": 3.0,
           "p99FullPixels": 6.0, "maxFullPixels": 9.0,
           "fractionAbove4Pixels": 0.05, "fractionAbove8Pixels": 0.01,
           "fractionAbove20Pixels": 0.0}
    return {"hardFailures": [], "inputsUnchanged": True, "registeredShots": 2,
            "reconstructedPoints": 60, "untruncatedReprojection": {**row, "count": 200, "finiteCount": 200},
            "perShot": {"a.JPG": row.copy(), "b.JPG": row.copy()}}


class DecisionContractTests(unittest.TestCase):
    def test_coordinates_require_explicit_valid_zone_and_finite_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coords.txt"
            for hemisphere, expected in [("N", "EPSG:32650"), ("S", "EPSG:32750")]:
                path.write_text(f"WGS84 UTM 50{hemisphere}\n191706 2494789\n-15.57 15.26 102.825\n")
                actual = controller.parse_coords(path)
                self.assertEqual(actual["projection"], expected)
                self.assertEqual(actual["photoCoordinateCount"], 1)
                self.assertEqual(actual["offsetX"], 191706)
            for contents in ["WGS84 UTM 61N\n0 0\n1 2 3\n", "WGS84 UTM 50N\n0\n1 2 3\n",
                             "WGS84 UTM 50N\n0 0\n1 nan 3\n", "EPSG:32650\n0 0\n1 2 3\n"]:
                path.write_text(contents)
                with self.assertRaises(RuntimeError):
                    controller.parse_coords(path)

    def test_good_global_statistics_cannot_hide_one_bad_photo(self):
        good = healthy_audit()
        self.assertEqual(controller.screen_audit(good, 2), [])
        bad = copy.deepcopy(good)
        bad["perShot"]["b.JPG"]["p50FullPixels"] = 3.01
        self.assertTrue(controller.screen_audit(bad, 2))
        bad = copy.deepcopy(good)
        bad["perShot"]["b.JPG"]["finiteCount"] = 49
        self.assertTrue(controller.screen_audit(bad, 2))
        self.assertTrue(controller.screen_audit(good, 3))

    def test_complete_pairing_rejects_missing_support_even_if_means_match(self):
        reference = healthy_audit()
        compared = copy.deepcopy(reference)
        compared["perShot"]["a.JPG"]["meanFullPixels"] += 1e-11
        self.assertLess(controller.paired_audits(reference, compared), 1e-7)
        compared["perShot"]["a.JPG"]["count"] = 99
        with self.assertRaises(RuntimeError):
            controller.paired_audits(reference, compared)
        compared = copy.deepcopy(reference)
        compared["reconstructedPoints"] -= 1
        with self.assertRaises(RuntimeError):
            controller.paired_audits(reference, compared)

    def test_export_must_include_all_registered_photos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reconstruction.nvm"
            cameras = "images/a.JPG.tif 700 1 0 0 0 1 2 3 0 0\nimages/b.JPG.tif 700 1 0 0 0 2 2 3 0 0\n"
            points = "1 2 3 100 150 200 2 0 1 100 200 1 1 150 220\n" * 50
            path.write_text("NVM_V3\n\n2\n" + cameras + "\n50\n" + points)
            self.assertEqual(controller.verify_nvm(path, 2)["minimumCameraLandmarkSupport"], 50)
            with self.assertRaises(RuntimeError):
                controller.verify_nvm(path, 3)
            path.write_text("NVM_V3\n\n")
            with self.assertRaises(RuntimeError):
                controller.verify_nvm(path, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
