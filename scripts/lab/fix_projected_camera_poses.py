"""Build audited camera-coordinate candidates; apply only after explicit QA.

Repairs a pinned OpenSfM export that serialized nonorthogonal matrices as
axis-angle. Default writes new candidates only. Original SfM, TIFF images,
tracks, features, matches and native algorithm code are never changed by build.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np


class GeometryPolicyFailure(RuntimeError):
    def __init__(self, message, metrics):
        super().__init__(message)
        self.metrics = metrics


def sha(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def json_identity(value):
    return "sha256:" + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_new_json(path, value):
    path = Path(path)
    if path.exists():
        raise RuntimeError("Refusing to overwrite existing artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    with os.fdopen(fd, "w") as stream:
        json.dump(value, stream, indent=4, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    # Atomic no-overwrite publication. OCI/shared filesystems here support
    # hardlinks; refusal is preferable to replacing an existing candidate.
    os.link(temporary, path)
    os.unlink(temporary)


def summary(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"count": 0}
    if not np.isfinite(a).all():
        raise RuntimeError("Non-finite geometric differences")
    return {"count": len(a), "signedMean": float(a.mean()), "signedMin": float(a.min()),
            "signedMax": float(a.max()), "absoluteP50": float(np.percentile(np.abs(a), 50)),
            "absoluteP90": float(np.percentile(np.abs(a), 90)),
            "absoluteP99": float(np.percentile(np.abs(a), 99)), "absoluteMax": float(np.max(np.abs(a)))}


def compute_frame(A, method, max_nonconformal=1e-6):
    A = np.asarray(A, dtype=float)
    if A.shape != (3, 3) or not np.isfinite(A).all() or np.linalg.det(A) <= 0:
        raise RuntimeError("Expected finite orientation-preserving 3x3 transform")
    if method == "polar":
        U, _, Vt = np.linalg.svd(A)
        Q = U @ Vt
        B = A
        scale = None
    else:
        U, singular, Vt = np.linalg.svd(A[:2, :2])
        if np.linalg.det(A[:2, :2]) <= 0:
            raise RuntimeError("XY transform reverses orientation")
        scale = float(np.mean(singular))
        nonconformal = float(abs(singular[0] - singular[1]) / scale)
        if nonconformal > max_nonconformal:
            raise RuntimeError("XY transform is not sufficiently conformal")
        Q = np.eye(3)
        Q[:2, :2] = U @ Vt
        B = scale * Q
    if np.linalg.det(Q) <= 0 or np.linalg.norm(Q.T @ Q - np.eye(3)) > 1e-10:
        raise RuntimeError("Candidate rotation is not SO(3)")
    return B, Q, scale


def repaired_pose(source, B, Q, b):
    from opensfm import io, pygeometry
    original = io.pose_from_json(source)
    target_centre = B @ original.get_origin() + b
    R = original.get_rotation_matrix() @ Q.T
    pose = pygeometry.Pose()
    pose.set_rotation_matrix(R)
    pose.set_origin(target_centre)
    result = {"rotation": list(pose.rotation), "translation": list(pose.translation)}
    reloaded = io.pose_from_json(json.loads(json.dumps(result)))
    error = float(np.linalg.norm(reloaded.get_origin() - target_centre))
    defect = float(np.linalg.norm(reloaded.get_rotation_matrix().T @ reloaded.get_rotation_matrix() - np.eye(3)))
    if error > 1e-8 or defect > 1e-10:
        raise RuntimeError("Candidate pose fails JSON roundtrip geometry validation")
    return result, error, original.get_origin(), target_centre


def update_single_shot_rigs(document):
    from opensfm import io
    rigs = copy.deepcopy(document.get("rig_instances", {}))
    for key, instance in rigs.items():
        mappings = instance["rig_camera_ids"]
        if len(mappings) != 1:
            raise RuntimeError("This repair is scoped to verified single-shot rig instances")
        name, camera_id = next(iter(mappings.items()))
        if name not in document["shots"]:
            raise RuntimeError("Rig references a missing shot")
        rig_camera = document["rig_cameras"][camera_id]
        rig_pose = io.pose_from_json(rig_camera)
        if np.linalg.norm(rig_pose.get_rotation_matrix() - np.eye(3)) > 1e-12 or np.linalg.norm(rig_pose.get_origin()) > 1e-12:
            raise RuntimeError("Nonidentity rig camera needs a separate generalized repair")
        instance["rotation"] = document["shots"][name]["rotation"]
        instance["translation"] = document["shots"][name]["translation"]
    document["rig_instances"] = rigs


def repair_documents(topo, actual, undistorted, A, b, method, max_xy_error, max_z_error,
                     z_policy="strict-all-points", max_scale_correction=0.001):
    from opensfm import io
    if not (len(topo) == len(actual) == len(undistorted)):
        raise RuntimeError("Reconstruction container counts differ")
    B, Q, scale = compute_frame(A, method)
    root_candidates, undist_candidates = [], []
    point_xy, point_z, camera_xy, camera_z, centre_roundtrip = [], [], [], [], []
    input_point_error = []
    point_z_outliers = []
    source_z_values = []
    counts = {"rootShots": 0, "rootPoints": 0, "undistortedShots": 0, "undistortedPoints": 0}
    for rec_index, (original, current, und) in enumerate(zip(topo, actual, undistorted)):
        if original["cameras"] != current["cameras"]:
            raise RuntimeError("Source and actual camera calibration dictionaries differ")
        if set(original["shots"]) != set(current["shots"]) or set(original["points"]) != set(current["points"]):
            raise RuntimeError("Source/current shot or point identities differ")
        corrected = current.copy()
        corrected["shots"] = {}
        original_names = set(original["shots"])
        for name, shot in current["shots"].items():
            rebuilt, error, C, target = repaired_pose(original["shots"][name], B, Q, b)
            row = shot.copy()
            row.update(rebuilt)
            corrected["shots"][name] = row
            affine_target = A @ C + b
            camera_xy.append(float(np.linalg.norm(target[:2] - affine_target[:2])))
            camera_z.append(float(target[2] - affine_target[2]))
            centre_roundtrip.append(error)
        corrected_points = {}
        for name, point in current["points"].items():
            p = np.asarray(original["points"][name]["coordinates"], dtype=float)
            expected = A @ p + b
            actual_point = np.asarray(point["coordinates"], dtype=float)
            error = float(np.linalg.norm(actual_point - expected))
            input_point_error.append(error)
            if error > 1e-7:
                raise RuntimeError("Actual UTM point is not the expected affine source point: " + name)
            new_point = B @ p + b
            point_xy.append(float(np.linalg.norm(new_point[:2] - actual_point[:2])))
            point_z.append(float(new_point[2] - actual_point[2]))
            source_z_values.append(float(p[2]))
            if abs(new_point[2] - actual_point[2]) > max_z_error:
                point_z_outliers.append({"reconstructionIndex": rec_index, "pointId": name,
                                         "sourceTopocentricZ": float(p[2]), "actualUTMZ": float(actual_point[2]),
                                         "candidateZ": float(new_point[2]),
                                         "signedZChangeMeters": float(new_point[2] - actual_point[2])})
            new = point.copy()
            # Polar preserves the actual point values bit-for-bit; Sim3 changes
            # coordinates consistently with cameras to preserve every ray.
            new["coordinates"] = point["coordinates"] if method == "polar" else new_point.tolist()
            corrected_points[name] = new
        corrected["points"] = corrected_points
        update_single_shot_rigs(corrected)
        und_new = und.copy()
        und_new["shots"] = {}
        mapping = {}
        for name, shot in und["shots"].items():
            possible = [root_name for root_name in original_names
                        if name in {root_name, root_name + ".tif", root_name + ".tiff"}]
            if len(possible) != 1:
                raise RuntimeError("Undistorted shot mapping is not unique: " + name)
            root_name = possible[0]
            if root_name in mapping:
                raise RuntimeError("Multiple undistorted views require a separate repair")
            mapping[root_name] = name
            row = shot.copy()
            row["rotation"] = corrected["shots"][root_name]["rotation"]
            row["translation"] = corrected["shots"][root_name]["translation"]
            und_new["shots"][name] = row
        if set(mapping) != original_names:
            raise RuntimeError("Undistorted view coverage differs from root reconstruction")
        und_new["points"] = {}
        for name, point in und["points"].items():
            if name not in corrected_points or name not in current["points"]:
                raise RuntimeError("Undistorted landmark identity cannot be reconciled")
            if not np.allclose(point["coordinates"], current["points"][name]["coordinates"], rtol=0, atol=1e-8):
                raise RuntimeError("Undistorted landmark differs from actual root coordinates")
            new = point.copy()
            new["coordinates"] = point["coordinates"] if method == "polar" else corrected_points[name]["coordinates"]
            und_new["points"][name] = new
        update_single_shot_rigs(und_new)
        # Verify every preserved field independently of the pose/point edits.
        for before, after in [(current, corrected), (und, und_new)]:
            if before["cameras"] != after["cameras"]:
                raise RuntimeError("Camera calibration changed")
            for name in before["shots"]:
                unchanged = {key: value for key, value in before["shots"][name].items() if key not in {"rotation", "translation"}}
                after_unchanged = {key: value for key, value in after["shots"][name].items() if key not in {"rotation", "translation"}}
                if unchanged != after_unchanged:
                    raise RuntimeError("Shot metadata changed")
            for name in before["points"]:
                if {k: v for k, v in before["points"][name].items() if k != "coordinates"} != {k: v for k, v in after["points"][name].items() if k != "coordinates"}:
                    raise RuntimeError("Point attributes changed")
        counts["rootShots"] += len(corrected["shots"])
        counts["rootPoints"] += len(corrected["points"])
        counts["undistortedShots"] += len(und_new["shots"])
        counts["undistortedPoints"] += len(und_new["points"])
        root_candidates.append(corrected)
        undist_candidates.append(und_new)
    metrics = {"counts": counts, "method": method, "candidateMatrix": B.tolist(), "properRotation": Q.tolist(),
               "similarityScale": scale, "pointXYDeviationFromActualUTMMeters": summary(point_xy),
               "pointZDeviationFromActualUTMMeters": summary(point_z),
               "cameraXYDeviationFromAffineUTMMeters": summary(camera_xy),
               "cameraZDeviationFromAffineUTMMeters": summary(camera_z),
               "poseRoundtripCentreErrorMeters": summary(centre_roundtrip),
               "actualUTMPointDeviationFromAffineSourceMeters": summary(input_point_error),
               "sourceTopocentricPointZMeters": summary(source_z_values),
               "sparsePointZChangesAboveAbsoluteScreenCount": len(point_z_outliers),
               "sparsePointZChangesAboveAbsoluteScreen": sorted(point_z_outliers, key=lambda row: -abs(row["signedZChangeMeters"])),
               "zPolicy": z_policy, "relativeScaleCorrection": abs(scale - 1) if scale is not None else None,
               "heightSemantics": "Uniform projected-grid scale also affects Z; raw/topocentric geometry remains preserved as the height reference." if method == "similarity-xy" else "Actual affine UTM XYZ retained; residual anisotropy remains measurable."}
    if max(point_xy + camera_xy, default=0) > max_xy_error:
        raise GeometryPolicyFailure("Candidate changes map XY beyond the allowed tolerance", metrics)
    if z_policy == "strict-all-points":
        if max((abs(v) for v in point_z + camera_z), default=0) > max_z_error:
            raise GeometryPolicyFailure("Candidate Z changes exceed the declared all-points tolerance", metrics)
    elif z_policy == "camera-and-scale":
        if max((abs(v) for v in camera_z), default=0) > max_z_error:
            raise GeometryPolicyFailure("Candidate camera Z changes exceed the declared tolerance", metrics)
        if scale is not None and abs(scale - 1) > max_scale_correction:
            raise GeometryPolicyFailure("Uniform grid scale exceeds the declared relative-scale correction", metrics)
    else:
        raise RuntimeError("Unknown height acceptance policy")
    return root_candidates, undist_candidates, metrics


def relevant_sfm_processes(project):
    matches = []
    commands = {"detect_features", "match_features", "create_tracks", "reconstruct", "undistort",
                "compute_statistics", "export_geocoords", "export_visualsfm", "export_openmvs"}
    for entry in Path("/proc").glob("[0-9]*"):
        if int(entry.name) == os.getpid():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().decode(errors="replace").split("\x00")
        except (OSError, PermissionError):
            continue
        if any(command in arguments for command in commands) and any(str(project) in arg for arg in arguments):
            matches.append({"pid": int(entry.name), "arguments": arguments})
    return matches


def replace_from_copy(source, target, expected):
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".replace", dir=target.parent)
    os.close(fd)
    shutil.copy2(source, temporary)
    if sha(temporary) != expected:
        raise RuntimeError("Staged replacement SHA does not match")
    os.replace(temporary, target)


def apply_manifest(args):
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    project = Path(manifest["opensfmPath"]).resolve()
    workspace = Path("/volume/wzhang/mesh-research-workspace")
    if not project.is_relative_to(workspace) or not manifest_path.is_relative_to(workspace):
        raise RuntimeError("Apply inputs must stay in the assigned persistent workspace")
    acceptance = json.loads(args.acceptance.read_text())
    if acceptance.get("schemaVersion") != "mesh.projected-camera-fix-acceptance/v1" or acceptance.get("state") != "passed":
        raise RuntimeError("An explicit passed acceptance record is required")
    if acceptance.get("candidateManifestSha256") != sha(manifest_path):
        raise RuntimeError("Acceptance refers to a different candidate manifest")
    audit = acceptance["fullCandidateAudit"]
    if sha(audit["path"]) != audit["sha256"]:
        raise RuntimeError("Full candidate audit was altered")
    audit_content = json.loads(Path(audit["path"]).read_text())
    root_candidate_hash = manifest["candidates"]["root"]["sha256"]
    if root_candidate_hash not in audit_content.get("inputHashes", {}).values() or audit_content.get("hardFailures"):
        raise RuntimeError("Full audit does not prove this root candidate was evaluated without hard failures")
    if audit_content.get("registeredShots") != manifest["metrics"]["counts"]["rootShots"]:
        raise RuntimeError("Full audit shot count differs")
    if not (project / "undistorted/nominal_done.txt").is_file():
        raise RuntimeError("Undistortion has not completed")
    active = relevant_sfm_processes(project)
    if active:
        raise RuntimeError("SfM export/mutation is still active: " + json.dumps(active))
    for path, expected in manifest["inputHashes"].items():
        if sha(path) != expected:
            raise RuntimeError("Input changed since candidate construction: " + path)
    for candidate in manifest["candidates"].values():
        if not Path(candidate["path"]).resolve().is_relative_to(project):
            raise RuntimeError("Candidate path escapes the project")
        if sha(candidate["path"]) != candidate["sha256"]:
            raise RuntimeError("Candidate changed after validation")
    targets = [(project / "reconstruction.json", manifest["candidates"]["root"]),
               (project / "undistorted/reconstruction.json", manifest["candidates"]["undistorted"])]
    checkpoint = args.checkpoint_dir.resolve()
    if not checkpoint.is_relative_to(Path("/volume/wzhang/mesh-research-workspace")) or checkpoint.exists():
        raise RuntimeError("Use a new checkpoint directory in the persistent workspace")
    checkpoint.mkdir(parents=True)
    backups = []
    for index, (target, candidate) in enumerate(targets):
        backup = checkpoint / ("root-reconstruction.json" if index == 0 else "undistorted-reconstruction.json")
        shutil.copy2(target, backup)
        expected = manifest["inputHashes"][str(target)]
        if sha(backup) != expected:
            raise RuntimeError("Backup verification failed")
        backups.append({"target": str(target), "backup": str(backup), "sha256": expected})
    shutil.copy2(manifest_path, checkpoint / "candidate-manifest.json")
    shutil.copy2(args.acceptance, checkpoint / "acceptance.json")
    transaction = {"state": "prepared", "backups": backups, "method": manifest["method"],
                   "derivedArtifactsRequireRegeneration": manifest["derivedArtifactsRequireRegeneration"]}
    atomic_new_json(checkpoint / "transaction.json", transaction)
    try:
        for target, candidate in targets:
            replace_from_copy(Path(candidate["path"]), target, candidate["sha256"])
        for target, candidate in targets:
            if sha(target) != candidate["sha256"]:
                raise RuntimeError("Applied candidate hash differs")
    except BaseException:
        for backup in backups:
            replace_from_copy(Path(backup["backup"]), Path(backup["target"]), backup["sha256"])
        raise
    transaction["state"] = "applied"
    transaction["appliedAt"] = dt.datetime.now(dt.timezone.utc).isoformat()
    atomic_new_json(checkpoint / "applied.json", transaction)
    print(json.dumps({"state": "applied", "checkpoint": str(checkpoint), "method": manifest["method"]}), flush=True)


def build(args):
    from opensfm import geo
    from opensfm.dataset import DataSet
    from pyproj import CRS, Transformer
    project = args.opensfm.resolve()
    workspace = Path("/volume/wzhang/mesh-research-workspace")
    if not project.is_relative_to(workspace):
        raise RuntimeError("Project must be in the assigned persistent workspace")
    paths = {"topocentric": project / "reconstruction.topocentric.json",
             "actual": project / "reconstruction.json", "undistorted": project / "undistorted/reconstruction.json"}
    watched = [*paths.values(), project / "tracks.csv", project / "undistorted/tracks.csv",
               project / "reference_lla.json", args.diagnostic.resolve()]
    before = {str(path): sha(path) for path in watched if path.is_file()}
    diagnostic = json.loads(args.diagnostic.read_text())
    if not args.manifest.resolve().is_relative_to(workspace):
        raise RuntimeError("Candidate manifest must stay in the assigned persistent workspace")
    for path in [paths["topocentric"], paths["actual"], project / "tracks.csv", project / "reference_lla.json"]:
        if diagnostic.get("inputHashes", {}).get(str(path)) != before[str(path)]:
            raise RuntimeError("Diagnostic geometry SHA differs: " + str(path))
    documents = {key: json.loads(path.read_text()) for key, path in paths.items()}
    data = DataSet(str(project))
    projection = Transformer.from_crs(CRS.from_epsg(4979), CRS.from_user_input(args.proj), always_xy=True).transform
    matrix = geo.get_proj_transform_matrix(data.load_reference(), projection, (args.offset_x, args.offset_y))
    if not np.allclose(matrix, np.asarray(diagnostic["transform"]), rtol=0, atol=1e-12):
        raise RuntimeError("Projection transform differs from the reviewed diagnostic")
    try:
        root, und, metrics = repair_documents(documents["topocentric"], documents["actual"], documents["undistorted"],
                                              matrix[:3, :3], matrix[:3, 3], args.method, args.max_xy_error_m, args.max_z_error_m,
                                              args.z_policy, args.max_scale_correction)
    except GeometryPolicyFailure as error:
        after = {str(path): sha(path) for path in watched if path.is_file()}
        failure_path = args.manifest.with_name(args.manifest.stem + ".failed-metrics.json")
        atomic_new_json(failure_path, {"schemaVersion": "mesh.projected-camera-candidate-rejection/v1",
                                      "state": "rejected", "error": str(error), "metrics": error.metrics,
                                      "inputHashes": before, "inputsUnchanged": before == after,
                                      "candidatesWritten": False})
        print(json.dumps({"state": "rejected", "failureMetrics": str(failure_path)}), flush=True)
        raise
    filename = "reconstruction." + args.method + "-camera-candidate.json"
    root_path = project / filename
    und_path = project / "undistorted" / filename
    atomic_new_json(root_path, root)
    atomic_new_json(und_path, und)
    after = {str(path): sha(path) for path in watched if path.is_file()}
    if before != after:
        raise RuntimeError("Original inputs changed during candidate build; candidates are not accepted")
    derived = [project / "undistorted/reconstruction.nvm", project / "undistorted/openmvs/scene.mvs",
               project / "undistorted/openmvs/depthmaps", project / "undistorted/openmvs/scene_dense.mvs",
               project / "undistorted/openmvs/scene_dense_dense_filtered.ply",
               project.parent / "odm_filterpoints", project.parent / "odm_meshing",
               project.parent / "odm_texturing", project.parent / "odm_texturing_25d", project.parent / "odm_orthophoto"]
    manifest = {"schemaVersion": "mesh.projected-camera-candidate/v1", "state": "candidate-only",
                "method": args.method, "opensfmPath": str(project), "inputHashes": before,
                "inputsUnchanged": True, "transform": matrix.tolist(), "metrics": metrics,
                "policies": {"maxXYErrorMeters": args.max_xy_error_m, "maxZErrorMeters": args.max_z_error_m,
                             "maxXYNonconformality": 1e-6, "zPolicy": args.z_policy,
                             "maxRelativeScaleCorrection": args.max_scale_correction,
                             "rationale": "For orthophoto-only use, camera-and-scale bounds camera elevation and uniform coordinate scale while reporting every sparse-point Z change, including preexisting extreme triangulation points. It does not claim absolute Z change below the camera bound for all points." if args.z_policy == "camera-and-scale" else "All sparse points and cameras must obey the same absolute Z bound."},
                "candidates": {"root": {"path": str(root_path), "sha256": sha(root_path)},
                               "undistorted": {"path": str(und_path), "sha256": sha(und_path)}},
                "sourceCameraModelsSha256": json_identity([doc["cameras"] for doc in documents["actual"]]),
                "undistortedCameraModelsSha256": json_identity([doc["cameras"] for doc in documents["undistorted"]]),
                "derivedArtifactsRequireRegeneration": [{"path": str(path), "exists": path.exists()} for path in derived],
                "reuse": "Existing undistorted TIFF pixels, intrinsics, features, matches and track observations retained. Geometry-dependent NVM and MVS scene must be re-exported after apply.",
                "scriptSha256": sha(Path(__file__))}
    atomic_new_json(args.manifest, manifest)
    print(json.dumps({"state": "candidate-only", "manifest": str(args.manifest), "metrics": metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensfm", type=Path)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--method", choices=["polar", "similarity-xy"], default="polar")
    parser.add_argument("--proj", default="EPSG:32650")
    parser.add_argument("--offset-x", type=float)
    parser.add_argument("--offset-y", type=float)
    parser.add_argument("--max-xy-error-m", type=float, default=0.005)
    parser.add_argument("--max-z-error-m", type=float, default=0.25)
    parser.add_argument("--z-policy", choices=["strict-all-points", "camera-and-scale"], default="strict-all-points")
    parser.add_argument("--max-scale-correction", type=float, default=0.001)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    if args.apply:
        if not args.acceptance or not args.checkpoint_dir:
            parser.error("--apply requires --acceptance and --checkpoint-dir")
        apply_manifest(args)
    else:
        if not args.opensfm or not args.diagnostic or args.offset_x is None or args.offset_y is None:
            parser.error("Build requires --opensfm, --diagnostic and both offsets")
        build(args)


if __name__ == "__main__":
    main()
