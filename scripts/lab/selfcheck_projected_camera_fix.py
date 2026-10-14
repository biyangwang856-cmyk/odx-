"""Native geometric self-checks for projected-camera candidate construction."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("pose_fix", Path(__file__).with_name("fix_projected_camera_poses.py"))
fix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fix)
try:
    from opensfm import io, pygeometry
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False


def transform_matrix():
    angle = 0.02
    Q = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    return Q @ np.diag([1.000774447, 1.000774447, 1.0])


class FrameTests(unittest.TestCase):
    def test_similarity_xy_preserves_xy_and_is_SO3(self):
        A = transform_matrix()
        B, Q, scale = fix.compute_frame(A, "similarity-xy")
        np.testing.assert_allclose(B[:2], A[:2], atol=1e-14)
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-14)
        self.assertAlmostEqual(np.linalg.det(Q), 1.0)
        self.assertAlmostEqual(scale, 1.000774447)

    def test_reflection_and_nonconformal_are_rejected(self):
        for A in [np.diag([-1., 1., 1.]), np.diag([1.02, 1., 1.])]:
            with self.assertRaises(RuntimeError):
                fix.compute_frame(A, "similarity-xy")

    def test_atomic_artifact_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            fix.atomic_new_json(path, {"value": 1})
            before = fix.sha(path)
            with self.assertRaises(RuntimeError):
                fix.atomic_new_json(path, {"value": 2})
            self.assertEqual(before, fix.sha(path))


@unittest.skipUnless(HAS_NATIVE, "Run with recovered native OpenSfM for camera geometry checks")
class NativeTests(unittest.TestCase):
    def setup_scene(self):
        camera = pygeometry.Camera.create_perspective(0.7, 0.001, 0.0001)
        camera.id = "test-camera"
        camera.width, camera.height = 1000, 800
        pose = pygeometry.Pose(np.array([np.pi - 1e-8, 0., 0.]))
        pose.set_origin(np.array([-120., 430., 100.]))
        source = {"rotation": list(pose.rotation), "translation": list(pose.translation)}
        points = np.array([[-110., 440., 20.], [-140., 415., 35.], [-90., 450., 45.]])
        return camera, pose, source, points

    def test_near_pi_similarity_roundtrip_preserves_real_rays(self):
        camera, pose, source, points = self.setup_scene()
        A = transform_matrix()
        b = np.array([0.1, -0.2, 0.3])
        B, Q, _ = fix.compute_frame(A, "similarity-xy")
        new, error, _, _ = fix.repaired_pose(source, B, Q, b)
        restored = io.pose_from_json(json.loads(json.dumps(new)))
        before = camera.project_many(points @ pose.get_rotation_matrix().T + pose.translation)
        mapped = points @ B.T + b
        after = camera.project_many(mapped @ restored.get_rotation_matrix().T + restored.translation)
        np.testing.assert_allclose(after, before, atol=1e-11)
        self.assertLess(error, 1e-9)

    def test_polar_keeps_affine_centres_after_json_roundtrip(self):
        _, _, source, _ = self.setup_scene()
        A = transform_matrix()
        B, Q, _ = fix.compute_frame(A, "polar")
        _, error, _, _ = fix.repaired_pose(source, B, Q, np.zeros(3))
        self.assertLess(error, 1e-9)

    def test_candidate_does_not_mutate_original_and_preserves_undistorted_K(self):
        camera, pose, source, points = self.setup_scene()
        A = transform_matrix()
        b = np.array([0.1, -0.2, 0.3])
        original_shot = {**source, "camera": camera.id, "capture_time": 123., "gps_accuracy": [0.02, 0.02, 0.05]}
        topo = {"cameras": {camera.id: io.camera_to_json(camera)}, "shots": {"image.JPG": original_shot},
                "points": {str(index): {"coordinates": row.tolist(), "color": [10., 20., 30.]}
                           for index, row in enumerate(points)},
                "rig_cameras": {camera.id: {"rotation": [0., 0., 0.], "translation": [0., 0., 0.]}},
                "rig_instances": {"image.JPG": {**source, "rig_camera_ids": {"image.JPG": camera.id}}}}
        current = copy.deepcopy(topo)
        bad_pose = pygeometry.Pose()
        bad_pose.set_rotation_matrix(pose.get_rotation_matrix() @ np.linalg.inv(A))
        bad_pose.set_origin(A @ pose.get_origin() + b)
        bad = {"rotation": list(bad_pose.rotation), "translation": list(bad_pose.translation)}
        current["shots"]["image.JPG"].update(bad)
        current["rig_instances"]["image.JPG"].update(bad)
        for point in current["points"].values():
            point["coordinates"] = (A @ np.asarray(point["coordinates"]) + b).tolist()
        und = copy.deepcopy(current)
        und["shots"]["image.JPG.tif"] = und["shots"].pop("image.JPG")
        und["rig_instances"]["image.JPG"]["rig_camera_ids"] = {"image.JPG.tif": camera.id}
        und["cameras"][camera.id]["focal"] = 0.72
        # One preexisting extreme triangulation point must remain visible in
        # the report. The strict policy rejects it; camera/scale policy keeps
        # the point and exposes its Z change rather than silently filtering it.
        extreme = np.array([-120., 430., -1469.])
        topo["points"]["preexisting-extreme"] = {"coordinates": extreme.tolist(), "color": [10., 20., 30.]}
        mapped_extreme = (A @ extreme + b).tolist()
        current["points"]["preexisting-extreme"] = {"coordinates": mapped_extreme, "color": [10., 20., 30.]}
        und["points"]["preexisting-extreme"] = copy.deepcopy(current["points"]["preexisting-extreme"])
        original_hash = fix.json_identity([topo, current, und])
        with self.assertRaises(fix.GeometryPolicyFailure) as rejected:
            fix.repair_documents([topo], [current], [und], A, b, "similarity-xy", 0.005, 0.25)
        self.assertEqual(rejected.exception.metrics["sparsePointZChangesAboveAbsoluteScreenCount"], 1)
        root, corrected_und, metrics = fix.repair_documents([topo], [current], [und], A, b,
                                                            "similarity-xy", 0.005, 0.25,
                                                            z_policy="camera-and-scale")
        self.assertEqual(original_hash, fix.json_identity([topo, current, und]))
        self.assertEqual(corrected_und[0]["cameras"], und["cameras"])
        self.assertEqual(root[0]["shots"]["image.JPG"]["capture_time"], 123.)
        self.assertGreater(metrics["pointZDeviationFromActualUTMMeters"]["absoluteMax"], 0)
        self.assertLess(metrics["pointXYDeviationFromActualUTMMeters"]["absoluteMax"], 1e-9)
        self.assertEqual(metrics["sparsePointZChangesAboveAbsoluteScreenCount"], 1)
        self.assertIn("preexisting-extreme", root[0]["points"])
        self.assertEqual(metrics["counts"]["rootPoints"], 4)
        with self.assertRaises(fix.GeometryPolicyFailure):
            fix.repair_documents([topo], [current], [und], A, b, "similarity-xy", 0.005, 0.25,
                                 z_policy="camera-and-scale", max_scale_correction=0.0001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
