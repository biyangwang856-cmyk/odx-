"""Verify projected SfM between completed SfM and dense reconstruction.

Host-Python controller; all camera calculations use the recovered native
OpenSfM runtime. Artifacts are fresh, originals are checkpointed before repair,
and the hash-bound dense gate is published only after export succeeds.
This checks geometry integrity, not surveying accuracy or vehicle appearance.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


WORKSPACE = Path("/volume/wzhang/mesh-research-workspace")
REVISION = "eb9fd56ac430fc3735401d88aaef9945ebd00e14"
SCREEN = {"globalP50MaxPixels": 2.0, "globalP90MaxPixels": 6.0,
          "globalFractionAbove8Max": 0.05, "perShotP50MaxPixels": 3.0,
          "perShotFiniteSupportMin": 50, "registrationFractionMin": 0.95,
          "pairedMetricMaxDelta": 1e-7, "xyMaxMeters": 0.005,
          "cameraZMaxMeters": 0.25, "scaleCorrectionMax": 0.001}
METRICS = ["meanFullPixels", "p50FullPixels", "p90FullPixels", "p99FullPixels",
           "maxFullPixels", "fractionAbove4Pixels", "fractionAbove8Pixels", "fractionAbove20Pixels"]


def sha(path):
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def fresh_json(path, value):
    """Publish a complete JSON artifact atomically without replacing history."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def parse_coords(path):
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise RuntimeError("coords.txt needs a projection, offset and photo coordinates")
    match = re.fullmatch(r"WGS84\s+UTM\s+(\d{1,2})([NS])", lines[0].strip())
    if not match or not 1 <= int(match[1]) <= 60:
        raise RuntimeError("Only explicit WGS84 UTM zone/hemisphere is supported")
    offset = [float(value) for value in lines[1].split()]
    if len(offset) != 2 or not all(math.isfinite(value) for value in offset):
        raise RuntimeError("Expected exactly two finite UTM offsets")
    photo_rows = []
    for number, line in enumerate(lines[2:], 3):
        values = [float(value) for value in line.split()]
        if len(values) != 3 or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"Invalid photo XYZ at coords.txt line {number}")
        photo_rows.append(values)
    epsg = (32600 if match[2] == "N" else 32700) + int(match[1])
    return {"projection": "EPSG:" + str(epsg), "declaredProjection": lines[0].strip(),
            "offsetX": offset[0], "offsetY": offset[1], "photoCoordinateCount": len(photo_rows)}


def screen_audit(audit, photo_count):
    errors = []
    if audit.get("hardFailures") or not audit.get("inputsUnchanged"):
        errors.append("Native audit is not a finite, stable snapshot")
    stats = audit["untruncatedReprojection"]
    if stats.get("count", 0) <= 0 or stats.get("finiteCount") != stats.get("count"):
        errors.append("Full reprojection observations are empty or nonfinite")
    if audit["registeredShots"] < photo_count * SCREEN["registrationFractionMin"]:
        errors.append("Registered photos below 95% of metadata-audited input")
    for key, limit in [("p50FullPixels", "globalP50MaxPixels"),
                       ("p90FullPixels", "globalP90MaxPixels"),
                       ("fractionAbove8Pixels", "globalFractionAbove8Max")]:
        if not math.isfinite(stats.get(key, float("inf"))) or stats.get(key, float("inf")) > SCREEN[limit]:
            errors.append("Full residual distribution exceeds " + limit)
    if len(audit["perShot"]) != audit["registeredShots"]:
        errors.append("Per-shot audit coverage differs from registered photo count")
    bad = []
    for name, row in audit["perShot"].items():
        if (row.get("finiteCount", 0) < SCREEN["perShotFiniteSupportMin"]
                or row.get("nonFiniteCount", 0)
                or row.get("p50FullPixels", float("inf")) > SCREEN["perShotP50MaxPixels"]):
            bad.append(name)
    if bad:
        errors.append("Unsupported/high-residual photos: " + ", ".join(bad[:20]))
    return errors


def paired_audits(reference, compared):
    if set(reference["perShot"]) != set(compared["perShot"]):
        raise RuntimeError("Paired full audits have different photo identities")
    if reference["reconstructedPoints"] != compared["reconstructedPoints"]:
        raise RuntimeError("Paired full audits have different landmark counts")
    if reference["untruncatedReprojection"]["count"] != compared["untruncatedReprojection"]["count"]:
        raise RuntimeError("Paired full audits have different observation counts")
    maximum = 0.0
    for name, before in reference["perShot"].items():
        after = compared["perShot"][name]
        if before["count"] != after["count"] or before["finiteCount"] != after["finiteCount"]:
            raise RuntimeError("Paired observation/support count differs: " + name)
        for key in METRICS:
            delta = abs(before[key] - after[key])
            if not math.isfinite(delta):
                raise RuntimeError("Nonfinite paired metric delta")
            maximum = max(maximum, delta)
    return maximum


def rotation_matrix(vector):
    angle = math.sqrt(sum(value * value for value in vector))
    if angle < 1e-16:
        return [[float(row == column) for column in range(3)] for row in range(3)]
    axis = [value / angle for value in vector]
    skew = [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    c, s = math.cos(angle), math.sin(angle)
    return [[c * float(row == column) + (1 - c) * axis[row] * axis[column] + s * skew[row][column]
             for column in range(3)] for row in range(3)]


def same_pose(first, second):
    r1, r2 = rotation_matrix(first["rotation"]), rotation_matrix(second["rotation"])
    return (max(abs(r1[i][j] - r2[i][j]) for i in range(3) for j in range(3)) <= 1e-9
            and max(abs(a - b) for a, b in zip(first["translation"], second["translation"])) <= 1e-8)


def single_identity_rigs(path):
    """Explicitly reject generalized rigs and multiple reconstruction containers."""
    document = load(path)
    if len(document) != 1:
        raise RuntimeError("Only one merged reconstruction container is supported")
    reconstruction = document[0]
    for name, shot in reconstruction["shots"].items():
        for key in ["rotation", "translation"]:
            if len(shot[key]) != 3 or not all(math.isfinite(value) for value in shot[key]):
                raise RuntimeError("Nonfinite/malformed photo pose: " + name)
    for name, point in reconstruction["points"].items():
        if len(point["coordinates"]) != 3 or not all(math.isfinite(value) for value in point["coordinates"]):
            raise RuntimeError("Nonfinite/malformed landmark: " + name)
    seen = set()
    for instance in reconstruction.get("rig_instances", {}).values():
        mapping = instance.get("rig_camera_ids", {})
        if len(mapping) != 1:
            raise RuntimeError("Only single-shot rig instances are supported")
        name, camera_id = next(iter(mapping.items()))
        if name in seen or name not in reconstruction["shots"]:
            raise RuntimeError("Invalid or repeated rig shot")
        seen.add(name)
        for key in ["rotation", "translation"]:
            if len(instance[key]) != 3 or not all(math.isfinite(value) for value in instance[key]):
                raise RuntimeError("Malformed rig instance pose")
        if not same_pose(instance, reconstruction["shots"][name]):
            raise RuntimeError("Identity-rig instance pose differs from its photo pose")
        camera = reconstruction["rig_cameras"][camera_id]
        for key in ["rotation", "translation"]:
            if (len(camera[key]) != 3 or not all(math.isfinite(value) for value in camera[key])
                    or max(abs(value) for value in camera[key]) > 1e-12):
                raise RuntimeError("Nonidentity rig cameras are unsupported")
    if seen != set(reconstruction["shots"]):
        raise RuntimeError("Rig coverage differs from all registered shots")
    model_hash = hashlib.sha256(json.dumps(reconstruction["cameras"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    return set(reconstruction["shots"]), set(reconstruction["points"]), model_hash


def validate_undistorted(project, root_shots, root_points):
    path = project / "undistorted/reconstruction.json"
    names, points, _ = single_identity_rigs(path)
    root, undistorted = load(project / "reconstruction.json")[0], load(path)[0]
    if any(camera.get("projection_type") != "perspective" for camera in undistorted["cameras"].values()):
        raise RuntimeError("Only undistorted perspective cameras are supported by this export validator")
    mapping = {}
    for name in names:
        candidates = [root_name for root_name in root_shots if name in {root_name, root_name + ".tif", root_name + ".tiff"}]
        if len(candidates) != 1 or candidates[0] in mapping:
            raise RuntimeError("Unsupported undistorted view mapping")
        mapping[candidates[0]] = name
        if not same_pose(root["shots"][candidates[0]], undistorted["shots"][name]):
            raise RuntimeError("Undistorted pose differs from its original photo pose: " + name)
        if not (project / "undistorted/images" / name).is_file():
            raise RuntimeError("Undistorted RGB is missing: " + name)
    if set(mapping) != root_shots or not points.issubset(root_points):
        raise RuntimeError("Undistorted shot/landmark identities differ")
    for name in points:
        if max(abs(a - b) for a, b in zip(root["points"][name]["coordinates"], undistorted["points"][name]["coordinates"])) > 1e-8:
            raise RuntimeError("Undistorted landmark differs from the root geometry: " + name)
    if not (project / "undistorted/nominal_done.txt").is_file():
        raise RuntimeError("Undistortion completion marker is absent")


def verify_nvm(path, expected_shots):
    with Path(path).open() as stream:
        if stream.readline().strip() != "NVM_V3":
            raise RuntimeError("Native NVM export header is invalid")

        def nonblank():
            for row in stream:
                if row.strip():
                    return row.strip()
            raise RuntimeError("Truncated native NVM export")

        if int(nonblank()) != expected_shots:
            raise RuntimeError("NVM export photo count differs")
        for _ in range(expected_shots):
            camera = nonblank().split()
            if len(camera) != 11 or not all(math.isfinite(float(value)) for value in camera[1:]):
                raise RuntimeError("Malformed/nonfinite NVM camera record")
        point_count = int(nonblank())
        if point_count < SCREEN["perShotFiniteSupportMin"]:
            raise RuntimeError("NVM export has too little sparse support")
        support = [0] * expected_shots
        for _ in range(point_count):
            words = nonblank().split()
            if len(words) < 7 or not all(math.isfinite(float(value)) for value in words[:3]):
                raise RuntimeError("Malformed/nonfinite NVM point")
            views = int(words[6])
            if views < 2 or len(words) != 7 + 4 * views:
                raise RuntimeError("Malformed/unsupported NVM track")
            seen = set()
            for index in range(7, len(words), 4):
                camera_index = int(words[index])
                if not 0 <= camera_index < expected_shots or camera_index in seen:
                    raise RuntimeError("NVM track contains an invalid/repeated camera")
                if not all(math.isfinite(float(value)) for value in words[index + 2:index + 4]):
                    raise RuntimeError("NVM track has nonfinite image observations")
                seen.add(camera_index)
                support[camera_index] += 1
        if min(support) < SCREEN["perShotFiniteSupportMin"]:
            raise RuntimeError("NVM contains a registered camera with insufficient landmark support")
    if Path(path).stat().st_size < 1000:
        raise RuntimeError("Native NVM export is unexpectedly empty")
    return {"registeredCameras": expected_shots, "exportedSparsePoints": point_count,
            "minimumCameraLandmarkSupport": min(support)}


def has_dense_artifacts(project):
    paths = [project / "opensfm/undistorted/openmvs/depthmaps", project / "opensfm/undistorted/openmvs/scene_dense.mvs",
             project / "opensfm/undistorted/openmvs/scene_dense_dense_filtered.ply"]
    paths += [project / name for name in ["odm_filterpoints", "odm_meshing", "odm_texturing", "odm_texturing_25d", "odm_orthophoto"]]
    return [str(path) for path in paths if path.exists() and (path.is_file() or any(path.iterdir()))]


def require_idle_sfm(project):
    commands = {"detect_features", "match_features", "create_tracks", "reconstruct", "undistort",
                "compute_statistics", "export_geocoords", "export_visualsfm", "export_openmvs"}
    for entry in Path("/proc").glob("[0-9]*"):
        if int(entry.name) == os.getpid():
            continue
        try:
            arguments = (entry / "cmdline").read_bytes().decode(errors="replace").split("\x00")
        except (OSError, PermissionError):
            continue
        mutator = any(item in commands for item in arguments)
        dense = any(Path(item).name in {"DensifyPointCloud", "ReconstructMesh", "TextureMesh", "odm_orthophoto"} for item in arguments)
        if (mutator or dense) and any(str(project) in item for item in arguments):
            raise RuntimeError("Project SfM/dense mutation is still active: PID " + entry.name)


def run_controller(args):
    job, bundle = args.job.resolve(), args.runtime.resolve()
    if (bundle / "odx-runtime-gpu").is_dir():
        bundle = bundle / "odx-runtime-gpu"
    if not job.is_relative_to(WORKSPACE / "runs") or not bundle.is_relative_to(WORKSPACE / "runtime"):
        raise RuntimeError("Use assigned persistent job/runtime directories")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", args.validation_name):
        raise RuntimeError("Use a simple fresh validation directory name")
    artifacts = job / args.validation_name
    gate = job / "sfm-merge-quality-gate.json"
    if artifacts.exists() or gate.exists():
        raise RuntimeError("Validation history or gate already exists; preserve it and use a new job/attempt")
    artifacts.mkdir()
    result = {"schemaVersion": "mesh.projected-sfm-validation/v1", "state": "running",
              "job": str(job), "runtimeBundle": str(bundle), "screening": SCREEN,
              "screeningMeaning": "Engineering screening; not a surveying accuracy standard or vehicle-quality measurement.",
              "startedAt": dt.datetime.now(dt.timezone.utc).isoformat(), "scriptSha256": sha(__file__), "commands": []}
    stage = "preflight"
    try:
        project = job / "datasets/project"
        sfm = project / "opensfm"
        coords = project / "odm_georeferencing/coords.txt"
        metadata = job / "metadata-audit.json"
        preflight_path = bundle / "native-loader-python-preflight.json"
        pre = load(preflight_path)
        identity = load(bundle / "source-overlay-validation.json")
        if load(bundle / "smoke-native/result.json").get("state") != "passed":
            raise RuntimeError("Native preflight did not pass")
        if identity.get("state") != "pinned" or identity.get("sourceRevision") != REVISION or identity.get("version") != "3.8.3":
            raise RuntimeError("Runtime is not the verified pinned ODX 3.8.3")
        code = bundle / "rootfs/code"
        for relative, expected in identity["pinnedSourceFiles"].items():
            if sha(code / relative) != expected:
                raise RuntimeError("Pinned runtime source changed: " + relative)
        for key in ["loader", "python"]:
            if not Path(pre[key]).resolve().is_relative_to(bundle / "rootfs") or not Path(pre[key]).is_file():
                raise RuntimeError("Preflight executable escapes or is missing from its bundle")
        if pre["environment"].get("ODX_NATIVE_ROOT") != str(bundle / "rootfs"):
            raise RuntimeError("Native preflight belongs to another runtime")
        prefix = [pre["loader"], "--library-path", pre["libraryPath"], pre["python"]]
        environment = {**os.environ, **pre["environment"], "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "HOME": str(job)}
        entry = code / "SuperBuild/install/bin/opensfm/bin/opensfm_main.py"
        if not entry.is_file():
            raise RuntimeError("Native Python OpenSfM entrypoint is missing")
        helpers = Path(__file__).resolve().parent
        scripts = {name: helpers / filename for name, filename in {
            "audit": "audit_merged_sfm.py", "diagnostic": "diagnose_geocoords_projection.py",
            "fix": "fix_projected_camera_poses.py", "selfcheck": "selfcheck_projected_camera_fix.py"}.items()}
        for helper in scripts.values():
            if not helper.is_file():
                raise RuntimeError("Required reviewed helper is missing: " + str(helper))
        meta = load(metadata)
        if meta.get("errors") or not meta.get("photos"):
            raise RuntimeError("Input metadata audit is incomplete or failed")
        photo_count = len(meta["photos"])
        coordinate_info = parse_coords(coords)
        if coordinate_info["photoCoordinateCount"] != photo_count:
            raise RuntimeError("coords.txt coverage differs from metadata-audited photos")
        protected = [coords, metadata, preflight_path, bundle / "source-overlay-validation.json",
                     bundle / "smoke-native/result.json", sfm / "config.yaml",
                     sfm / "reconstruction.topocentric.json", sfm / "reconstruction.json",
                     sfm / "undistorted/reconstruction.json", sfm / "tracks.csv", sfm / "undistorted/tracks.csv", sfm / "reference_lla.json"]
        originals = {str(path): sha(path) for path in protected}
        result.update(inputHashes=originals, coordinateMetadata=coordinate_info, inputPhotoCount=photo_count,
                      helperHashes={str(path): sha(path) for path in scripts.values()})
        root_shots, root_points, root_K = single_identity_rigs(sfm / "reconstruction.json")
        topo_shots, topo_points, topo_K = single_identity_rigs(sfm / "reconstruction.topocentric.json")
        if (root_shots, root_points, root_K) != (topo_shots, topo_points, topo_K):
            raise RuntimeError("Topocentric/exported shot, landmark or calibration identities differ")
        input_names = {Path(photo["file"]).name for photo in meta["photos"]}
        if len(input_names) != photo_count or not root_shots.issubset(input_names):
            raise RuntimeError("Registered photo identities do not match this metadata-audited input")
        validate_undistorted(sfm, root_shots, root_points)
        require_idle_sfm(sfm)
        dense = has_dense_artifacts(project)
        if dense:
            raise RuntimeError("Run validation before dense-derived files exist: " + json.dumps(dense))

        def command(label, arguments, timeout=1800):
            log = artifacts / (label + ".log")
            argv = [*prefix, *map(str, arguments)]
            started = time.monotonic()
            with log.open("xb") as stream:
                completed = subprocess.run(argv, env=environment, cwd=helpers.parent.parent, stdout=stream,
                                           stderr=subprocess.STDOUT, timeout=timeout)
            result["commands"].append({"label": label, "argv": argv, "exitCode": completed.returncode,
                                       "elapsedSeconds": time.monotonic() - started, "log": str(log), "logSha256": sha(log)})
            if completed.returncode:
                raise RuntimeError(f"{label} failed; inspect {log}")

        def audit(label, filename):
            output = artifacts / (label + ".json")
            command(label, [scripts["audit"], "--opensfm", sfm, "--output", output, "--reconstruction-file", filename])
            return output, load(output)

        stage = "native-selfcheck"
        command("native-selfcheck", [scripts["selfcheck"]])
        if "skipped=" in (artifacts / "native-selfcheck.log").read_text():
            raise RuntimeError("Native camera selfchecks were skipped")
        stage = "topocentric-screening"
        topo_path, topo = audit("topocentric-full-audit", "reconstruction.topocentric.json")
        failures = screen_audit(topo, photo_count)
        if failures:
            result["screeningFailures"] = failures
            raise RuntimeError("Original SfM failed engineering screening: " + "; ".join(failures))
        stage = "projected-full-audit"
        geo_path, exported = audit("projected-full-audit", "reconstruction.json")
        if exported.get("hardFailures") or not exported.get("inputsUnchanged"):
            raise RuntimeError("Projected SfM has unsupported/nonfinite geometry")
        before_delta = paired_audits(topo, exported)
        result["beforeMaxPairedMetricDelta"] = before_delta
        graph = next(row for row in topo["tracksConnectivity"] if row["minimumSharedTracks"] == 1)
        limits = ["No independent GCP/checkpoint accuracy is established; GPS residuals fit priors.",
                  "Geometry integrity and residual screening do not measure vehicle duplication or orthophoto appearance."]
        if graph["componentCount"] > 1:
            limits.append("Visual track graph has disconnected components " + str(graph["componentSizes"]) + "; geolocation alignment does not provide cross-component visual constraints.")
        accepted_path, accepted_audit = geo_path, exported
        acceptance_path = artifacts / "geometry-acceptance.json"
        checkpoint = artifacts / "pre-apply-checkpoint"
        expected_geometry = {str(sfm / "reconstruction.json"): originals[str(sfm / "reconstruction.json")],
                             str(sfm / "undistorted/reconstruction.json"): originals[str(sfm / "undistorted/reconstruction.json")]}
        if before_delta <= SCREEN["pairedMetricMaxDelta"]:
            failures = screen_audit(exported, photo_count)
            if failures:
                raise RuntimeError("Exported SfM failed screening: " + "; ".join(failures))
            decision = "No coordinate repair needed: complete paired photo statistics preserve the source reconstruction."
            fresh_json(acceptance_path, {"schemaVersion": "mesh.projected-sfm-no-change-acceptance/v1", "state": "passed",
                                       "decision": decision, "fullAudit": {"path": str(geo_path), "sha256": sha(geo_path)},
                                       "topocentricAudit": {"path": str(topo_path), "sha256": sha(topo_path)},
                                       "maxPairedPerShotMetricDeltaPixels": before_delta, "limits": limits, "screening": SCREEN})
        else:
            stage = "serialization-diagnosis"
            diagnostic_path = artifacts / "projection-diagnostic.json"
            command("projection-diagnostic", [scripts["diagnostic"], "--opensfm", sfm, "--output", diagnostic_path,
                    "--proj", coordinate_info["projection"], "--offset-x", coordinate_info["offsetX"], "--offset-y", coordinate_info["offsetY"]])
            diagnostic = load(diagnostic_path)
            if diagnostic["changedCameraModelShots"] or not diagnostic["inputsUnchanged"]:
                raise RuntimeError("The discrepancy is not an unchanged-camera export")
            if diagnostic["exportedPointsDeviationFromAffineMeters"]["max"] > 1e-7:
                raise RuntimeError("Existing points do not match the original affine coordinate export")
            sampled = diagnostic["residualFullPixels"]
            for key in ["mean", "p50", "p90", "p99", "max"]:
                if abs(sampled["currentExport"][key] - sampled["affineJsonReload"][key]) > 1e-7:
                    raise RuntimeError("Current geometry does not reproduce the identified serialization defect")
            for key in ["meanFullPixels", "p50FullPixels", "p90FullPixels"]:
                if exported["untruncatedReprojection"][key] <= topo["untruncatedReprojection"][key] + 1e-3:
                    raise RuntimeError("Coordinate serialization is not consistently degrading full residual statistics")
            if diagnostic["projectionChangeFromTopocentricFullPixels"]["similarityXY"]["max"] > 1e-7:
                raise RuntimeError("XY similarity does not preserve sampled source projections")
            stage = "candidate-build"
            manifest_path = artifacts / "candidate-manifest.json"
            command("candidate-build", [scripts["fix"], "--opensfm", sfm, "--diagnostic", diagnostic_path,
                    "--manifest", manifest_path, "--method", "similarity-xy", "--proj", coordinate_info["projection"],
                    "--offset-x", coordinate_info["offsetX"], "--offset-y", coordinate_info["offsetY"],
                    "--z-policy", "camera-and-scale", "--max-xy-error-m", SCREEN["xyMaxMeters"],
                    "--max-z-error-m", SCREEN["cameraZMaxMeters"], "--max-scale-correction", SCREEN["scaleCorrectionMax"]])
            manifest = load(manifest_path)
            stage = "candidate-full-audit"
            candidate_path, candidate = audit("candidate-full-audit", Path(manifest["candidates"]["root"]["path"]).name)
            failures = screen_audit(candidate, photo_count)
            candidate_delta = paired_audits(topo, candidate)
            if failures or candidate_delta > SCREEN["pairedMetricMaxDelta"]:
                raise RuntimeError("Candidate failed full preservation/screening: " + json.dumps(failures))
            if manifest["metrics"]["counts"]["rootShots"] != candidate["registeredShots"]:
                raise RuntimeError("Candidate native count differs from manifest")
            decision = "Repair confirmed nonorthogonal-matrix serialization with XY-conformal similarity, preserving all audited source photo residuals."
            m = manifest["metrics"]
            limits.append("Uniform grid scale changes Z: camera max " + str(m["cameraZDeviationFromAffineUTMMeters"]["absoluteMax"])
                          + " m; sparse point max " + str(m["pointZDeviationFromActualUTMMeters"]["absoluteMax"])
                          + " m; " + str(m["sparsePointZChangesAboveAbsoluteScreenCount"]) + " original sparse points exceed the camera absolute screen. No strict DEM height accuracy is claimed.")
            fresh_json(acceptance_path, {"schemaVersion": "mesh.projected-camera-fix-acceptance/v1", "state": "passed",
                      "candidateManifestSha256": sha(manifest_path), "fullCandidateAudit": {"path": str(candidate_path), "sha256": sha(candidate_path)},
                      "topocentricAudit": {"path": str(topo_path), "sha256": sha(topo_path)}, "decision": decision,
                      "maxPairedPerShotMetricDeltaPixels": candidate_delta, "limits": limits, "screening": SCREEN})
            stage = "checkpointed-apply"
            command("apply", [scripts["fix"], "--apply", "--manifest", manifest_path,
                               "--acceptance", acceptance_path, "--checkpoint-dir", checkpoint])
            result["repairApplied"] = True
            expected_geometry = {str(sfm / "reconstruction.json"): manifest["candidates"]["root"]["sha256"],
                                 str(sfm / "undistorted/reconstruction.json"): manifest["candidates"]["undistorted"]["sha256"]}
            stage = "applied-full-audit"
            accepted_path, accepted_audit = audit("applied-full-audit", "reconstruction.json")
            if screen_audit(accepted_audit, photo_count) or paired_audits(topo, accepted_audit) > SCREEN["pairedMetricMaxDelta"]:
                raise RuntimeError("Actual applied geometry does not pass the full native audit")

        # Back up generated exports before replacement even in the no-change case.
        stage = "geometry-derived-export"
        require_idle_sfm(sfm)
        validate_undistorted(sfm, root_shots, root_points)
        export_backups = artifacts / "pre-export-backups"
        export_backups.mkdir()
        nvm, scene = sfm / "undistorted/reconstruction.nvm", sfm / "undistorted/openmvs/scene.mvs"
        had_scene = scene.exists()
        backup_rows = []
        for path in [nvm, scene]:
            if path.exists():
                backup = export_backups / path.name
                shutil.copy2(path, backup)
                if sha(path) != sha(backup):
                    raise RuntimeError("Geometry export backup SHA differs")
                backup_rows.append({"original": str(path), "backup": str(backup), "sha256": sha(backup)})
        fresh_json(export_backups / "manifest.json", {"backups": backup_rows})
        command("export-visualsfm", [entry, "export_visualsfm", "--points", sfm])
        nvm_summary = verify_nvm(nvm, accepted_audit["registeredShots"])
        if had_scene:
            command("export-openmvs", [entry, "export_openmvs", sfm])
            if not scene.is_file() or scene.stat().st_size < 1000:
                raise RuntimeError("Native scene.mvs re-export is missing/empty")
        for path, expected in originals.items():
            required = expected_geometry.get(path, expected)
            if sha(path) != required:
                raise RuntimeError("Protected geometry/input changed unexpectedly: " + path)
        for path, expected in result["helperHashes"].items():
            if sha(path) != expected:
                raise RuntimeError("Reviewed helper changed during validation: " + path)
        if sha(__file__) != result["scriptSha256"]:
            raise RuntimeError("Controller changed during validation")
        for relative, expected in identity["pinnedSourceFiles"].items():
            if sha(code / relative) != expected:
                raise RuntimeError("Pinned runtime source changed during validation: " + relative)
        stage = "publish-hash-bound-gate"
        gate_value = {"schemaVersion": "mesh.projected-sfm-dense-gate/v1", "state": "passed",
                      "reconstructionSha256": sha(sfm / "reconstruction.json").removeprefix("sha256:"),
                      "undistortedReconstructionSha256": sha(sfm / "undistorted/reconstruction.json").removeprefix("sha256:"),
                      "nvmSha256": sha(nvm).removeprefix("sha256:"), "coordsSha256": sha(coords).removeprefix("sha256:"),
                      "acceptanceSha256": sha(acceptance_path), "acceptancePath": str(acceptance_path),
                      "fullAuditSha256": sha(accepted_path), "fullAuditPath": str(accepted_path),
                      "fullUntruncatedResidual": accepted_audit["untruncatedReprojection"], "decision": decision,
                      "limits": limits, "screening": SCREEN, "coordinateMetadata": coordinate_info,
                      "sourceRevision": REVISION, "runtimePreflightSha256": sha(preflight_path),
                      "sourceOverlayValidationSha256": sha(bundle / "source-overlay-validation.json"),
                      "sfmConfigSha256": sha(sfm / "config.yaml"), "controllerSha256": result["scriptSha256"],
                      "helperHashes": result["helperHashes"], "nativeNVMExport": nvm_summary}
        if had_scene:
            gate_value["openmvsSceneSha256"] = sha(scene).removeprefix("sha256:")
        result.update(state="passed", gate=str(gate), decision=decision, limits=limits,
                      acceptedFullAudit={"path": str(accepted_path), "sha256": sha(accepted_path)},
                      finalGeometryHashes=expected_geometry, finishedAt=dt.datetime.now(dt.timezone.utc).isoformat())
        fresh_json(artifacts / "result.json", result)
        gate_value["validationResultSha256"] = sha(artifacts / "result.json")
        gate_value["validationResultPath"] = str(artifacts / "result.json")
        fresh_json(gate, gate_value)
        print(json.dumps({"state": "passed", "gate": str(gate), "artifacts": str(artifacts), "decision": decision}), flush=True)
    except BaseException as error:
        result.update(state="failed", failedStage=stage, error=type(error).__name__ + ": " + str(error),
                      finishedAt=dt.datetime.now(dt.timezone.utc).isoformat(), gateWritten=gate.exists())
        failure = artifacts / "failure.json"
        if not failure.exists():
            fresh_json(failure, result)
        print(json.dumps({"state": "failed", "stage": stage, "artifacts": str(artifacts), "error": str(error)}), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True, help="Runtime root or recovered GPU bundle")
    parser.add_argument("--validation-name", default="sfm-projection-validation", help="Fresh artifact directory; existing history is never overwritten")
    run_controller(parser.parse_args())


if __name__ == "__main__":
    main()
