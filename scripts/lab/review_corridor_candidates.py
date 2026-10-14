"""Review actual one-exposure road proposals before publishing apply certificates.

This script never exports an orthophoto. In proposal mode SAM road, shadow and
detector masks remain candidates. A separate hash-bound appearance review is
required to certify their semantic domain/catalogue; measured geometry and
two-exposure clear-road evidence are still required and cannot be waived.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "app"))
from model_engine.ogs2_corridor_apply import CERTIFICATE_SCHEMA, file_sha256
from model_engine.ogs2_corridor_policy import (
    CorridorCandidate, CorridorConfig, array_sha256, select_corridor_source,
)
from model_engine.ogs2_joint_closure import close_whole_foregrounds

PROPOSAL_SCHEMA = "ogs2.corridor-review-proposals/v1"
REVIEW_SCHEMA = "ogs2.corridor-appearance-review/v1"


def read_json(path):
    def invalid(value):
        raise ValueError("Nonfinite JSON value " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def save_json(path, record):
    Path(path).write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def pinned_path(binding, parent):
    path = Path(binding["path"])
    path = (path if path.is_absolute() else parent / path).resolve()
    if file_sha256(path) != binding["sha256"]:
        raise ValueError("Pinned file changed: " + str(path))
    return path


def read_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def require_mask(value, shape, name):
    if value.dtype != np.bool_ or value.shape != shape:
        raise ValueError("Mask dtype/frame mismatch: " + name)
    return value


def context_mask(mask, pixel_size_m, margin_m=0.25):
    """A declared no-cut margin, NOT a measured shadow or invented RGB."""
    if pixel_size_m <= 0 or margin_m < 0:
        raise ValueError("Invalid metric mask margin")
    if not mask.any():
        return mask.copy()
    return ndimage.distance_transform_edt(~mask) * pixel_size_m <= margin_m


def union_masks(masks, shape):
    masks = list(masks)
    return np.logical_or.reduce(masks) if masks else np.zeros(shape, bool)


def validate_clear_witnesses(observation, obs, all_records, projection, pair_evidence=None):
    """A body-free detector result alone is never enough to certify empty road."""
    shape = obs["candidate_empty_road"].shape
    supported = np.zeros(shape, bool)
    audits = []
    sources = {r["sourceImage"]: r for r in projection["sources"]}
    for ref in observation["clearReferences"]:
        name = ref["sourceImage"]
        if name == observation["sourceImage"] or name not in all_records or name not in sources:
            raise ValueError("Clear-road witness is not an independent retained source")
        if ref["cacheSha256"] != sources[name]["cacheSha256"]:
            raise ValueError("Clear-road witness cache binding changed")
        if pair_evidence is not None:
            pairs = [p for p in pair_evidence if {p.get("firstSource"), p.get("secondSource")} == {name, observation["sourceImage"]}]
            if len(pairs) != 1 or pairs[0].get("accepted") is not True or pairs[0].get("colorComparisonAccepted") is not True:
                raise ValueError("Clear-road witness lacks an accepted measured static/exposure pair")
        key = ref.get("maskKey")
        # Earlier manifests used clearMaskKey; accept an explicitly named mask,
        # never infer an undeclared witness from a nonzero detector count.
        key = key or ref.get("clearMaskKey")
        if key is None:
            raise ValueError("Clear-road witness has no explicit boolean mask")
        mask = require_mask(obs[key], shape, key)
        supported |= mask
        audits.append({"sourceImage": name, "maskKey": key, "pixels": int(mask.sum()),
                       "referenceCacheSha256": ref["cacheSha256"]})
    empty = require_mask(obs["candidate_empty_road"], shape, "candidate_empty_road")
    if (empty & ~supported).any():
        raise ValueError("Empty-road candidate lacks an explicit independent clear witness")
    return empty.copy(), audits


def blocker_counts(base, conflict, road, candidate, baseline_foregrounds, unknown):
    """Diagnose all mask failures without temporarily asserting semantic truth."""
    cfg = CorridorConfig()
    catalog = {"history:" + k: v for k, v in baseline_foregrounds.items()}
    catalog.update({"selected:" + k: v for k, v in candidate.foregrounds.items()})
    closure = close_whole_foregrounds(conflict, catalog, cfg.protection_steps, cfg.max_support_pixels)
    hard = closure.support
    collar = ndimage.binary_dilation(hard, iterations=cfg.static_collar_steps) & ~hard
    touching = {k: m for k, m in candidate.foregrounds.items() if (m & hard).any()}
    foreground = union_masks(touching.values(), hard.shape)
    failures = {
        "outsideApprovedRoad": hard & ~road,
        "protectedUnknown": hard & (unknown | candidate.unknown_foreground),
        "missingRealCoverage": hard & (~candidate.valid | (base[..., 3] == 0)),
        "unsupportedOldPositions": hard & ~foreground & ~(candidate.observed_road & candidate.verified_empty_road),
        "unverifiedStaticCollar": collar & (~candidate.static_support | ~candidate.valid | (base[..., 3] == 0) | unknown | candidate.unknown_foreground),
    }
    labels, count = ndimage.label(hard)
    components = []
    for index, slices in enumerate(ndimage.find_objects(labels), 1):
        component = labels == index
        reservation = ndimage.binary_dilation(component, iterations=cfg.static_collar_steps)
        components.append({"component": index, "pixels": int(component.sum()),
            "boundingBoxXYWH": [slices[1].start, slices[0].start, slices[1].stop-slices[1].start, slices[0].stop-slices[0].start],
            "failPixels": {key: int((mask & (reservation if key == "unverifiedStaticCollar" else component)).sum()) for key, mask in failures.items()}})
    return {
        "hardSupportPixels": int(hard.sum()), "staticCollarPixels": int(collar.sum()),
        **{key: int(mask.sum()) for key, mask in failures.items()},
        "wholeSupportComponents": components, "componentCount": count,
        "touchingForegroundObservations": list(touching),
        "notAppearanceReviewed": not candidate.foreground_catalog_complete,
        "baselineGeometryAccepted": candidate.static_verification.get("accepted") is True,
        "actualDisplacementP90": candidate.static_verification.get("p90ActualPixels"),
        "holdoutResidualP90": candidate.static_verification.get("holdoutResidualP90Pixels"),
    }, hard, collar


def approval_for_source(review, proposal_sha, source_entry):
    """No model score or CLI flag can replace the separately recorded review."""
    if review is None:
        return None
    if review.get("schema") != REVIEW_SCHEMA or review.get("proposalManifestSha256") != proposal_sha:
        raise ValueError("Appearance review is not bound to this exact proposal")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip() or not review.get("reviewedAt"):
        raise ValueError("Require an identified, timestamped appearance review")
    records = [r for r in review.get("sources", []) if r.get("sourceImage") == source_entry["sourceImage"]]
    if len(records) > 1:
        raise ValueError("Duplicate source appearance reviews")
    if not records:
        return None
    approved = records[0]
    fields = ("roadDomainReviewed", "historicConflictDomainReviewed", "protectedUnknownReviewed",
              "foregroundCatalogReviewed", "completeBodyAndShadowBoundariesReviewed")
    if any(approved.get(key) is not True for key in fields):
        return None
    expected = set(source_entry["foregrounds"])
    complete = approved.get("completeForegroundIds")
    if not isinstance(complete, list) or len(set(complete)) != len(complete) or not set(complete).issubset(expected):
        raise ValueError("Reviewed foreground IDs differ from the actual proposal catalogue")
    # Reviews may not promote an incomplete cache observation to a whole car.
    possible = set(source_entry["projectionCompleteForegroundIds"])
    if not set(complete).issubset(possible):
        raise ValueError("Appearance review cannot override an incomplete projected vehicle")
    if not approved.get("appearanceNotes"):
        raise ValueError("Record actual source/body/shadow/domain inspection notes")
    return approved


def overlay(rgb, mask, color):
    result = rgb.copy()
    result[mask] = np.rint(result[mask] * 0.45 + np.asarray(color) * 0.55).astype(np.uint8)
    return result


def preview(path, base, image, arrays, foreground, hard, collar, title):
    from PIL import Image, ImageDraw
    rows = [
        (base[..., :3], "Retained OGS2 baseline"), (image, "Actual single exposure " + title),
        (overlay(image, arrays["road_domain"], (0, 180, 255)), "CANDIDATE approved road domain; review required"),
        (overlay(image, arrays["verified_empty_road"], (0, 240, 40)), "Positive two-source empty-road evidence"),
        (overlay(image, foreground, (255, 140, 0)), "Body + measured shadow candidate + declared 0.25m no-cut context"),
        (overlay(image, arrays["unknown_foreground"] | arrays["protected_unknown"], (255, 0, 190)), "Protected unknown/incomplete/parked"),
        (overlay(image, hard, (0, 230, 230)), "Whole historic road-conflict proposal, not accepted output"),
        (overlay(image, collar, (255, 20, 20)), "Required unchanged-baseline static collar"),
    ]
    width = min(1000, image.shape[1]); height = round(image.shape[0] * width / image.shape[1])
    canvas = Image.new("RGB", (width * 2, (height + 36) * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (rgb, label) in enumerate(rows):
        x, y = i % 2 * width, i // 2 * (height + 36)
        canvas.paste(Image.fromarray(rgb).resize((width, height)), (x, y + 36))
        draw.text((x + 6, y + 8), label, fill="black")
    canvas.save(path, quality=94)


def build_proposals(observation_path, baseline_review_path, output, source_names=None, region_masks=None):
    from model_engine.ogs2_corridor_projection import load_bound_source
    observation_path, baseline_review_path = Path(observation_path).resolve(), Path(baseline_review_path).resolve()
    report, baseline_review = read_json(observation_path), read_json(baseline_review_path)
    if report.get("state") != "candidate-evidence-complete":
        raise ValueError("Observation evidence is incomplete")
    if baseline_review.get("observationManifestSha256") != file_sha256(observation_path):
        raise ValueError("Baseline proofs were measured for different observation evidence")
    pm = pinned_path({"path": report["projectionManifest"], "sha256": report["projectionManifestSha256"]}, observation_path.parent)
    projection = read_json(pm)
    if baseline_review.get("projectionManifestSha256") != file_sha256(pm):
        raise ValueError("Baseline review refers to another projection")
    snapshot = pm.parent / "baseline-roi.npz"
    if file_sha256(snapshot) != projection["snapshotSha256"]:
        raise ValueError("Baseline ROI snapshot changed")
    base = read_npz(snapshot)["rgba"]; shape = base.shape[:2]
    grid = projection["grid"]
    pixel_size = abs(float(grid["transform"][0]))
    names = source_names or [r["sourceImage"] for r in report["records"]]
    records = {r["sourceImage"]: r for r in report["records"]}
    if len(records) != len(report["records"]) or len(set(names)) != len(names):
        raise ValueError("Duplicate observation/source name")
    measured = {r["sourceImage"]: r for r in baseline_review["records"]}
    data, historical, road_union = {}, {}, np.zeros(shape, bool)
    for name, obs_record in records.items():
        source, source_record, checked = load_bound_source(pm, name)
        if checked != projection or source_record["cacheSha256"] != obs_record["sourceCacheSha256"]:
            raise ValueError("Observation source cache differs from completed projection")
        op = pinned_path({"path": obs_record["observationCachePath"], "sha256": obs_record["observationCacheSha256"]}, observation_path.parent)
        obs = read_npz(op)
        road_union |= require_mask(obs["road_candidate"], shape, "road_candidate")
        foregrounds, possible = {}, []
        for i, foreground in enumerate(obs_record["foregrounds"]):
            key = foreground["maskKey"]
            whole = context_mask(require_mask(obs[key], shape, key), pixel_size)
            if not whole.any():
                continue
            identity = name + ":" + key
            foregrounds[identity] = whole
            historical[identity] = whole
            if foreground["bodyCompleteInProjectionCache"] is True:
                possible.append(identity)
        data[name] = (source, source_record, obs, obs_record, foregrounds, possible)
    # These are candidates for explicit appearance review, not semantic truth.
    history_union = union_masks(historical.values(), shape)
    road = road_union | history_union
    conflict = history_union.copy()
    protected = np.zeros(shape, bool)
    input_region_binding = None
    if region_masks is not None:
        region_masks = Path(region_masks).resolve()
        supplied = read_npz(region_masks)
        if set(supplied) != {"road_domain", "conflict_union", "protected_unknown"}:
            raise ValueError("Region masks require exactly road_domain/conflict_union/protected_unknown")
        road, conflict, protected = (require_mask(supplied[k], shape, k) for k in ("road_domain", "conflict_union", "protected_unknown"))
        input_region_binding = {"path": str(region_masks), "sha256": file_sha256(region_masks)}
    if not conflict.any():
        raise ValueError("No historical whole-foreground conflict; supply a reviewed conflict mask")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Fresh proposal directory required")
    output.mkdir(parents=True)
    entries = []
    for index, name in enumerate(names):
        source, source_record, obs, observation, foregrounds, possible = data[name]
        empty, witness_audit = validate_clear_witnesses(observation, obs, records, projection, report["sourcePairGeometryEvidence"])
        summary = measured[name]["baselineGeometry"]
        proof_path = pinned_path({"path": summary["fullProofPath"], "sha256": summary["fullProofSha256"]}, baseline_review_path.parent)
        proof = read_json(proof_path)
        static = obs["static_support_candidate"] & source["valid"] & (base[..., 3] > 0)
        if proof["staticSupportSha256"] != array_sha256(static):
            raise ValueError("Reconstructed static proof mask differs; cannot reuse measurement")
        masks = {"road_domain": road, "conflict_union": conflict, "protected_unknown": protected,
                 "observed_road": obs["road_candidate"], "verified_empty_road": empty,
                 "static_support": static, "unknown_foreground": obs["unknown_foreground_candidate"]}
        source_mapping, historical_mapping = {}, {}
        for i, (key, mask) in enumerate(foregrounds.items()):
            role = "selected_foreground_" + str(i); masks[role] = mask; source_mapping[key] = role
        for i, (key, mask) in enumerate(historical.items()):
            role = "historical_foreground_" + str(i); masks[role] = mask; historical_mapping[key] = role
        candidate = CorridorCandidate(name, source["image"], source["valid"], masks["observed_road"], empty,
            static, foregrounds, frozenset(), float(np.mean(source["cost"][conflict])), source_record["captureTime"], proof,
            masks["unknown_foreground"], False)
        blockers, hard, collar = blocker_counts(base, conflict, road, candidate, historical, protected)
        _, decisions = select_corridor_source(base, conflict, road, [candidate], baseline_foregrounds=historical, protected_unknown=protected)
        mask_path = output / f"proposal-{index:03d}-masks.npz"
        np.savez_compressed(mask_path, **masks)
        preview_path = output / f"proposal-{index:03d}-review.jpg"
        preview(preview_path, base, source["image"], masks, union_masks(foregrounds.values(), shape), hard, collar, name)
        entry = {"id": f"corridor-{index:03d}", "sourceImage": name,
            "certificateArrays": {"path": str(mask_path), "sha256": file_sha256(mask_path), "arraySha256": {k: array_sha256(v) for k, v in masks.items()}},
            "masks": {"conflict_union": "conflict_union", "road_domain": "road_domain", "protected_unknown": "protected_unknown", "baseline_foregrounds": historical_mapping},
            "candidateMasks": {k: k for k in ("observed_road", "verified_empty_road", "static_support", "unknown_foreground")},
            "foregrounds": source_mapping, "projectionCompleteForegroundIds": possible,
            "sourceArraysSha256": {k: array_sha256(source[k]) for k in ("image", "valid", "cost", "body_mask")},
            "static_verification": proof, "sourceCacheSha256": source_record["cacheSha256"],
            "sourceImageSha256": source_record["sourceImageSha256"], "clearWitnessAudit": witness_audit,
            "preview": str(preview_path), "previewSha256": file_sha256(preview_path),
            "blockerCounts": blockers, "policyDecisions": decisions,
            "semanticTruthCertified": False, "foregroundCatalogueComplete": False}
        entries.append(entry)
        save_json(output / "progress.json", entries)
    baseline_path = next((p for p in projection["criticalInputs"] if Path(p).name == "ogs2_orthophoto.tif"), None)
    if baseline_path is None:
        raise ValueError("Could not identify explicitly pinned retained OGS2 baseline")
    manifest = {"schema": PROPOSAL_SCHEMA, "state": "appearance-review-pending", "entries": entries,
        "baseline": {"path": baseline_path, "sha256": projection["criticalInputs"][baseline_path]},
        "projectionManifest": {"path": str(pm), "sha256": file_sha256(pm)},
        "observationManifest": {"path": str(observation_path), "sha256": file_sha256(observation_path)},
        "baselineReviewManifest": {"path": str(baseline_review_path), "sha256": file_sha256(baseline_review_path)},
        "regionMasks": input_region_binding, "declaredNoCutContextMetres": 0.25,
        "contextMarginIsMeasuredShadow": False, "automaticMosaicAuthorized": False,
        "vehicleCountUsedForRanking": False, "outputRgbGenerated": False}
    result = output / "proposal-manifest.json"; save_json(result, manifest)
    save_json(output / "appearance-review-template.json", {"schema": REVIEW_SCHEMA,
        "proposalManifestSha256": file_sha256(result), "reviewer": "", "reviewedAt": "", "sources": [
        {"sourceImage": e["sourceImage"], "roadDomainReviewed": False, "historicConflictDomainReviewed": False,
         "protectedUnknownReviewed": False, "foregroundCatalogReviewed": False,
         "completeBodyAndShadowBoundariesReviewed": False, "completeForegroundIds": [], "appearanceNotes": ""} for e in entries]})
    return result


def certify_proposals(proposal_path, review_path, output):
    proposal_path, review_path = Path(proposal_path).resolve(), Path(review_path).resolve()
    proposal, review = read_json(proposal_path), read_json(review_path)
    if proposal.get("schema") != PROPOSAL_SCHEMA or proposal.get("state") != "appearance-review-pending":
        raise ValueError("Require a completed unapproved proposal manifest")
    proposal_sha = file_sha256(proposal_path)
    for key in ("baseline", "projectionManifest", "observationManifest", "baselineReviewManifest"):
        pinned_path(proposal[key], proposal_path.parent)
    patches = []
    for entry in proposal["entries"]:
        approved = approval_for_source(review, proposal_sha, entry)
        if approved is None:
            continue
        pinned_path(entry["certificateArrays"], proposal_path.parent)
        pinned_path({"path": entry["preview"], "sha256": entry["previewSha256"]}, proposal_path.parent)
        patches.append({"id": entry["id"], "evidenceManifest": proposal["projectionManifest"],
            "certificateArrays": entry["certificateArrays"], "masks": entry["masks"], "candidates": [{
                "sourceImage": entry["sourceImage"], "sourceArraysSha256": entry["sourceArraysSha256"],
                "masks": entry["candidateMasks"], "foregrounds": entry["foregrounds"],
                "complete_foregrounds": approved["completeForegroundIds"], "foreground_catalog_complete": True,
                "static_verification": entry["static_verification"]}]})
    if len(patches) > 1:
        raise ValueError("This bounded-region tool permits exactly one reviewed exposure; choose one source")
    if not patches:
        raise ValueError("No exact proposal has a complete explicit appearance review")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Do not overwrite an existing certificate")
    save_json(output, {"schema": CERTIFICATE_SCHEMA, "state": "complete", "baseline": proposal["baseline"],
        "patches": patches, "appearanceReview": {"path": str(review_path), "sha256": file_sha256(review_path)},
        "proposalManifest": {"path": str(proposal_path), "sha256": proposal_sha},
        "measuredGeometryAndEmptyRoadGuardsStillRequired": True, "outputRgbGenerated": False})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("propose")
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--baseline-review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", action="append")
    p.add_argument("--region-masks", type=Path)
    p = sub.add_parser("certify")
    p.add_argument("--proposal", type=Path, required=True)
    p.add_argument("--appearance-review", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "propose":
        result = build_proposals(args.observations, args.baseline_review, args.output, args.source, args.region_masks)
        record = read_json(result)
        print(json.dumps({"proposal": str(result), "sources": len(record["entries"]), "refusals": dict(Counter(
            d["reason"] for e in record["entries"] for d in e["policyDecisions"]))}, ensure_ascii=False))
    else:
        print(str(certify_proposals(args.proposal, args.appearance_review, args.output)))


if __name__ == "__main__":
    main()
