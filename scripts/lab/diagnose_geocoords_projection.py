"""Read-only comparison of ENU, existing UTM and proposed proper rotations.

Never saves a reconstruction or writes a pipeline gate. Candidate pose changes
exist only in independent in-memory Pose objects. Uses original camera models
and tracks; retains actual UTM points/origins for the polar-rotation candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
from opensfm import geo, io, pygeometry
from opensfm.dataset import DataSet


def sha(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def summary(values):
    a = np.asarray(values, dtype=float)
    finite = a[np.isfinite(a)]
    if not len(finite):
        return {"count": len(a), "nonFinite": len(a)}
    return {"count": len(a), "nonFinite": len(a) - len(finite), "mean": float(finite.mean()),
            "p50": float(np.percentile(finite, 50)), "p90": float(np.percentile(finite, 90)),
            "p99": float(np.percentile(finite, 99)), "max": float(finite.max()),
            "fractionAbove4": float(np.mean(finite > 4)), "fractionAbove8": float(np.mean(finite > 8))}


def new_pose(rotation, origin):
    pose = pygeometry.Pose()
    pose.set_rotation_matrix(rotation)
    pose.set_origin(origin)
    return pose


def roundtrip(pose):
    # The actual JSON serialization stores axis-angle, not an arbitrary 3x3.
    document = {"rotation": list(pose.rotation), "translation": list(pose.translation)}
    return io.pose_from_json(json.loads(json.dumps(document)))


def project_with_pose(camera, pose, points):
    coordinates = points @ pose.get_rotation_matrix().T + np.asarray(pose.translation)
    return camera.project_many(coordinates)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensfm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--proj", default="EPSG:32650")
    parser.add_argument("--offset-x", type=float, required=True)
    parser.add_argument("--offset-y", type=float, required=True)
    parser.add_argument("--observations-per-shot", type=int, default=128)
    args = parser.parse_args()
    project = args.opensfm.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("Choose a new diagnostic output path")
    protected = [project / "reconstruction.topocentric.json", project / "reconstruction.json",
                 project / "tracks.csv", project / "reference_lla.json"]
    before = {str(path): sha(path) for path in protected}
    data = DataSet(str(project))
    original = data.load_reconstruction("reconstruction.topocentric.json")
    exported = data.load_reconstruction("reconstruction.json")
    tracks = data.load_tracks_manager()
    transformer = Transformer.from_crs(CRS.from_epsg(4979), CRS.from_user_input(args.proj), always_xy=True)
    transform = geo.get_proj_transform_matrix(data.load_reference(), transformer.transform,
                                               (args.offset_x, args.offset_y))
    A, b = transform[:3, :3], transform[:3, 3]
    invA = np.linalg.inv(A)
    U, singular, Vt = np.linalg.svd(A)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt
    # XY-conformal similarity is an additional diagnostic, not the proposed
    # fix. It changes Z scale and therefore requires separate elevation QA.
    U2, s2, V2 = np.linalg.svd(A[:2, :2])
    Q2 = U2 @ V2
    Qxy = np.eye(3)
    Qxy[:2, :2] = Q2
    scale_xy = float(np.mean(s2))
    residuals = {name: [] for name in ["topocentric", "currentExport", "affineMemory",
                                      "affineJsonReload", "polarJsonReload", "similarityXY"]}
    deltas = {name: [] for name in residuals if name != "topocentric"}
    per_shot = {}
    centre_errors = {name: [] for name in ["currentExport", "affineJsonReload", "polarJsonReload"]}
    rotation_differences = []
    exported_point_transform_errors = []
    similarity_xy_errors = []
    rig_counts = []
    camera_mismatches = []
    for rec_index, topo in enumerate(original):
        current = exported[rec_index]
        rig_counts.append({"reconstructionIndex": rec_index, "shots": len(topo.shots),
                           "rigInstances": len(topo.rig_instances),
                           "instanceSizes": [len(instance.shots) for instance in topo.rig_instances.values()]})
        for name in sorted(topo.shots):
            shot = topo.shots[name]
            current_shot = current.shots[name]
            if io.camera_to_json(shot.camera) != io.camera_to_json(current_shot.camera):
                camera_mismatches.append(name)
            R = shot.pose.get_rotation_matrix()
            C = np.asarray(shot.pose.get_origin())
            target_C = A @ C + b
            affine_pose = new_pose(R @ invA, target_C)
            affine_reload = roundtrip(affine_pose)
            polar_pose = roundtrip(new_pose(R @ Q.T, target_C))
            similarity_pose = roundtrip(new_pose(R @ Qxy.T, scale_xy * (Qxy @ C) + b))
            centre_errors["currentExport"].append(float(np.linalg.norm(current_shot.pose.get_origin() - target_C)))
            centre_errors["affineJsonReload"].append(float(np.linalg.norm(affine_reload.get_origin() - target_C)))
            centre_errors["polarJsonReload"].append(float(np.linalg.norm(polar_pose.get_origin() - target_C)))
            relative_R = current_shot.pose.get_rotation_matrix() @ polar_pose.get_rotation_matrix().T
            angle = np.arccos(np.clip((np.trace(relative_R) - 1) / 2, -1, 1))
            rotation_differences.append(float(np.degrees(angle)))
            observations = tracks.get_shot_observations(name)
            ids = sorted(point for point in observations if point in topo.points and point in current.points)
            if len(ids) > args.observations_per_shot:
                indices = np.linspace(0, len(ids) - 1, args.observations_per_shot, dtype=int)
                ids = [ids[index] for index in indices]
            if not ids:
                per_shot[name] = {"state": "no-common-landmarks"}
                continue
            points = np.asarray([topo.points[point].coordinates for point in ids])
            geo_points = points @ A.T + b
            actual_geo_points = np.asarray([current.points[point].coordinates for point in ids])
            sim_points = scale_xy * (points @ Qxy.T) + b
            observed = np.asarray([observations[point].point for point in ids])
            normalized = {
                "topocentric": shot.project_many(points),
                "currentExport": current_shot.project_many(actual_geo_points),
                "affineMemory": project_with_pose(shot.camera, affine_pose, geo_points),
                "affineJsonReload": project_with_pose(shot.camera, affine_reload, geo_points),
                "polarJsonReload": project_with_pose(shot.camera, polar_pose, geo_points),
                "similarityXY": project_with_pose(shot.camera, similarity_pose, sim_points),
            }
            factor = max(shot.camera.width, shot.camera.height)
            rows = {}
            for candidate, predicted in normalized.items():
                values = np.linalg.norm(predicted - observed, axis=1) * factor
                residuals[candidate].append(values)
                rows[candidate] = summary(values)
                if candidate != "topocentric":
                    delta = np.linalg.norm(predicted - normalized["topocentric"], axis=1) * factor
                    deltas[candidate].append(delta)
            per_shot[name] = rows
            exported_point_transform_errors.extend(np.linalg.norm(actual_geo_points - geo_points, axis=1).tolist())
            similarity_xy_errors.extend(np.linalg.norm(sim_points[:, :2] - geo_points[:, :2], axis=1).tolist())
        print(json.dumps({"auditedReconstruction": rec_index, "shots": len(topo.shots)}), flush=True)
    after = {str(path): sha(path) for path in protected}
    if before != after:
        raise RuntimeError("Input changed while auditing; discard this snapshot")
    result = {
        "schemaVersion": "mesh.geocoords-projection-diagnostic/v1", "readOnly": True,
        "inputHashes": before, "inputsUnchanged": True, "transform": transform.tolist(),
        "A_SingularValues": singular.tolist(), "A_Determinant": float(np.linalg.det(A)),
        "A_OrthogonalityDefectFrobenius": float(np.linalg.norm(A.T @ A - np.eye(3))),
        "polarRotation": Q.tolist(), "polarOrthogonalityDefectFrobenius": float(np.linalg.norm(Q.T @ Q - np.eye(3))),
        "xySimilarityScale": scale_xy,
        "residualFullPixels": {name: summary(np.concatenate(arrays)) if arrays else {} for name, arrays in residuals.items()},
        "projectionChangeFromTopocentricFullPixels": {name: summary(np.concatenate(arrays)) if arrays else {} for name, arrays in deltas.items()},
        "cameraCentreDeviationFromAffineTargetMeters": {name: summary(values) for name, values in centre_errors.items()},
        "currentRotationDeviationFromPolarDegrees": summary(rotation_differences),
        "exportedPointsDeviationFromAffineMeters": summary(exported_point_transform_errors),
        "xySimilarityVersusAffineXYMeters": summary(similarity_xy_errors),
        "rigStructure": rig_counts, "changedCameraModelShots": camera_mismatches,
        "perShot": per_shot,
        "limitation": "Deterministic 128-observation per-shot diagnostic, not a full acceptance run. Similarity changes Z scale; polar keeps actual affine XYZ but retains measurable anisotropy error.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["A_SingularValues", "A_OrthogonalityDefectFrobenius",
                       "residualFullPixels", "projectionChangeFromTopocentricFullPixels",
                       "cameraCentreDeviationFromAffineTargetMeters", "xySimilarityVersusAffineXYMeters"]}), flush=True)


if __name__ == "__main__":
    main()
