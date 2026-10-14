"""Read-only audit of pinned OpenSfM's merged reconstruction before dense MVS.

Uses native OpenSfM camera/error implementations. In contrast to stats.json,
reprojection summaries retain residuals above the upstream 4-pixel cutoff.
No feature extraction, matching, reconstruction, BA, or input mutation occurs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from opensfm import io, pygeometry, pymap, types
from opensfm.dataset import DataSet


def digest(path):
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def summarize(values):
    a = np.asarray(values, dtype=float)
    valid = a[np.isfinite(a)]
    if not len(valid):
        return {"count": len(a), "finiteCount": 0}
    return {"count": len(a), "finiteCount": len(valid),
            "meanFullPixels": float(valid.mean()),
            "p50FullPixels": float(np.percentile(valid, 50)),
            "p90FullPixels": float(np.percentile(valid, 90)),
            "p99FullPixels": float(np.percentile(valid, 99)),
            "maxFullPixels": float(valid.max()),
            "fractionAbove4Pixels": float(np.mean(valid > 4)),
            "fractionAbove8Pixels": float(np.mean(valid > 8)),
            "fractionAbove20Pixels": float(np.mean(valid > 20)),
            "nonFiniteCount": int(len(a) - len(valid))}


def graph_components(shots, pair_counts, threshold):
    parent = {s: s for s in shots}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for (a, b), count in pair_counts.items():
        if a in parent and b in parent and count >= threshold:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    groups = {}
    for a in shots:
        groups.setdefault(find(a), []).append(a)
    ordered = sorted((sorted(group) for group in groups.values()), key=lambda x: (-len(x), x))
    return {"minimumSharedTracks": threshold, "componentCount": len(ordered),
            "componentSizes": [len(group) for group in ordered], "components": ordered}


def report_memberships(report_path, shot_ids):
    if not report_path.exists():
        return {}, {"state": "report-missing", "unassignedShots": sorted(shot_ids)}
    report = json.loads(report_path.read_text())
    groups = {}
    summaries = []
    for index, rec_report in enumerate(report.get("reconstructions", [])):
        known = set(rec_report.get("bootstrap", {}).get("image_pair", []))
        rows = rec_report.get("grow", {}).get("grow", {}).get("resected_shots", [])
        for row in rows:
            if isinstance(row, dict) and row.get("shot_id"):
                known.add(row["shot_id"])
        known &= shot_ids
        for shot in known:
            if shot in groups and groups[shot] != index:
                raise RuntimeError("Report assigns one shot to multiple partial reconstructions")
            groups[shot] = index
        summaries.append({"partialIndex": index, "knownShotCount": len(known),
                          "knownShotIds": sorted(known)})
    # The bootstrap growth to five photos is not retained in the final grow
    # report. Such unassigned photos stay unknown; do not fabricate membership.
    return groups, {"state": "partial-report-exact-members-only", "partials": summaries,
                    "unassignedShots": sorted(shot_ids - set(groups)),
                    "limitation": "Bootstrap pre-growth photos may be missing; cross-group counts are lower bounds."}


def verify_same_camera_behavior():
    r = types.Reconstruction()
    first = pygeometry.Camera.create_perspective(0.7, 0.0, 0.0)
    second = pygeometry.Camera.create_perspective(0.8, 0.0, 0.0)
    first.id = second.id = "read-only-audit-camera"
    r.add_camera(first)
    retained = r.add_camera(second)
    shot = r.create_shot("read-only-audit-shot", first.id, pygeometry.Pose())
    return {"firstFocal": 0.7, "secondFocal": 0.8,
            "retainedFocal": float(retained.focal), "newShotFocal": float(shot.camera.focal),
            "behavior": "first-camera-retained" if abs(retained.focal - 0.7) < 1e-12 else "unexpected"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--opensfm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reconstruction-file", default="reconstruction.json")
    parser.add_argument("--batch-shots", type=int, default=16)
    args = parser.parse_args()
    project = args.opensfm.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("Audit output exists; choose a new output filename")
    watched = [project / args.reconstruction_file, project / "tracks.csv",
               project / "config.yaml", project / "reports/reconstruction.json"]
    before = {str(path): digest(path) for path in watched if path.is_file()}
    started = time.monotonic()
    data = DataSet(str(project))
    reconstructions = data.load_reconstruction(args.reconstruction_file)
    tracks = data.load_tracks_manager()
    shot_ids = {shot for rec in reconstructions for shot in rec.shots}
    memberships, membership_report = report_memberships(project / "reports/reconstruction.json", shot_ids)
    pair_counts = tracks.get_all_pairs_connectivity()
    graphs = [graph_components(shot_ids, pair_counts, threshold) for threshold in [1, 10, 30, 100]]
    cross_pairs = []
    for (a, b), count in pair_counts.items():
        if a in memberships and b in memberships and memberships[a] != memberships[b]:
            cross_pairs.append({"a": a, "b": b, "sharedTracks": int(count),
                                "partialA": memberships[a], "partialB": memberships[b]})
    cross_pairs.sort(key=lambda row: -row["sharedTracks"])
    per_shot = {}
    all_values = []
    group_values = {}
    point_bits = {}
    for rec_index, rec in enumerate(reconstructions):
        names = sorted(rec.shots)
        for offset in range(0, len(names), args.batch_shots):
            subset = names[offset:offset + args.batch_shots]
            errors = rec.map.compute_reprojection_errors(tracks, pymap.ErrorType.Pixel, subset)
            for shot_name in subset:
                shot = rec.shots[shot_name]
                rows = errors.get(shot_name, {})
                if rows:
                    array = np.stack(list(rows.values()))
                    values = np.linalg.norm(array, axis=1) * max(shot.camera.width, shot.camera.height)
                else:
                    values = np.empty(0)
                summary = summarize(values)
                summary.update(reconstructionIndex=rec_index,
                               originalPartialIndex=memberships.get(shot_name),
                               cameraId=shot.camera.id)
                per_shot[shot_name] = summary
                all_values.append(values)
                group = memberships.get(shot_name, "unknown")
                group_values.setdefault(str(group), []).append(values)
                if group != "unknown":
                    for point_id in rows:
                        key = (rec_index, point_id)
                        point_bits[key] = point_bits.get(key, 0) | (1 << group)
            print(json.dumps({"auditedShots": min(offset + args.batch_shots, len(names)),
                              "reconstructionIndex": rec_index}), flush=True)
    values = np.concatenate(all_values) if all_values else np.empty(0)
    group_summary = {group: summarize(np.concatenate(arrays)) for group, arrays in group_values.items()}
    crossing_ids = [point for point, bits in point_bits.items() if bits.bit_count() > 1]
    worst = sorted(per_shot.items(), key=lambda row: row[1].get("p90FullPixels", float("inf")), reverse=True)[:30]
    after = {str(path): digest(path) for path in watched if path.is_file()}
    if before != after:
        raise RuntimeError("Input SfM files changed during audit; report is not a stable snapshot")
    nonfinite = sum(row.get("nonFiniteCount", 0) for row in per_shot.values())
    empty = [name for name, row in per_shot.items() if row.get("finiteCount", 0) == 0]
    hard_failures = []
    if nonfinite:
        hard_failures.append("Non-finite reprojection residuals detected")
    if empty:
        hard_failures.append("Registered shots with no finite reconstructed-point support")
    # These are diagnostic screening policies, not universal accuracy standards.
    # Disconnected graphs or >4px outliers alone do not declare a GPS-aligned
    # scene invalid. Per-photo/per-partial evidence must guide the operator.
    result = {
        "schemaVersion": "mesh.sfm-merge-readonly-audit/v1", "state": "blocked" if hard_failures else "review",
        "hardFailures": hard_failures, "inputHashes": before, "inputsUnchanged": True,
        "registeredShots": len(shot_ids), "reconstructionContainers": len(reconstructions),
        "reconstructedPoints": sum(len(rec.points) for rec in reconstructions),
        "upstreamStatsLimitation": "stats.reprojection_error_pixels excludes residuals over 4 full pixels",
        "untruncatedReprojection": summarize(values), "perOriginalPartial": group_summary,
        "perShot": per_shot, "worst30ShotsByP90": worst,
        "tracksConnectivity": graphs, "originalPartialMembership": membership_report,
        "knownCrossPartialTrackPairs": len(cross_pairs), "strongestCrossPartialTrackPairs": cross_pairs[:50],
        "crossPartialTriangulatedPointsLowerBound": len(crossing_ids),
        "sampleCrossPartialPointIds": [point[1] for point in crossing_ids[:30]],
        "sameCameraIdRuntimeBehavior": verify_same_camera_behavior(),
        "mergedCameras": [{key: io.camera_to_json(camera) for key, camera in rec.cameras.items()}
                          for rec in reconstructions],
        "missingOriginalCalibrations": "An already merged JSON cannot recover each partial's original optimized intrinsics.",
        "interpretation": "Visual disconnection is a relative-alignment limitation, not proof of distortion. RTK residual is fit to a prior, not an independent checkpoint.",
        "elapsedSeconds": round(time.monotonic() - started, 3),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["state", "registeredShots", "untruncatedReprojection",
                                                   "perOriginalPartial", "knownCrossPartialTrackPairs",
                                                   "crossPartialTriangulatedPointsLowerBound", "elapsedSeconds"]}), flush=True)
    if hard_failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
