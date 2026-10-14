"""Candidate road, real empty-road and shadow observations for corridor QA.

SAM masks and color consistency are evidence, never semantic ground truth.
This module does not certify a foreground catalogue or authorize a mosaic
change. Every output names its real projection cache and rejection reasons.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from model_engine.ogs2 import cameras_from_checkpoint, sha256
from model_engine.ogs2_corridor_projection import (
    array_digest, atomic_json, atomic_npz, canonical_digest, load_bound_source,
)
from model_engine.ogs2_joint_geometry import GeometryConfig, independent_static_evidence


SCHEMA = "ogs2.corridor-observation-candidates/v1"


@dataclass(frozen=True)
class ObservationConfig:
    max_sources: int = 10
    max_seed_prompts: int = 8
    max_negative_prompts: int = 0
    source_prompt_radius_px: int = 420
    min_sam_score: float = 0.65
    min_component_pixels: int = 100
    max_component_fraction: float = 0.9
    seed_body_clearance_m: float = 0.8
    static_body_clearance_m: float = 0.7
    geometry_possible_shadow_clearance_m: float = 4.0
    geometry_shadow_guard_mode: str = 'circular'
    geometry_dark_shadow_ratio: float = 0.85
    geometry_dark_shadow_difference: float = 15.0
    geometry_dark_shadow_margin_m: float = 0.35
    empty_body_clearance_m: float = 0.2
    neutral_chroma_range: float = 35.0
    paint_min_rgb: float = 145.0
    paint_neighbourhood_m: float = 0.35
    agreement_rgb_threshold: float = 18.0
    min_clear_reference_sources: int = 1
    min_color_fit_pixels: int = 512
    shadow_difference: float = 18.0
    shadow_chroma_difference: float = 0.08
    shadow_max_body_distance_m: float = 4.0
    shadow_body_connection_m: float = 0.65
    min_change_component_pixels: int = 8


def neutral_road_seed_pixels(image, valid, body, pixel_size_m, config=ObservationConfig()):
    rgb = np.asarray(image, np.float32)
    neutral = rgb.max(axis=2) - rgb.min(axis=2) <= config.neutral_chroma_range
    neutral &= (rgb.min(axis=2) > 30) & (rgb.max(axis=2) < 235)
    body = np.asarray(body, bool)
    distance = ndimage.distance_transform_edt(~body) * pixel_size_m if body.any() else np.full(body.shape, np.inf)
    return neutral & valid & (distance > config.seed_body_clearance_m)


def _spaced_seed_points(mask, maximum):
    """Distribute prompts; each prompt independently selects one component."""
    yy, xx = np.where(mask)
    if not len(xx):
        return []
    points = np.column_stack([xx, yy])
    seeds = [points[int(np.argmin(np.sum((points - np.median(points, axis=0)) ** 2, axis=1)))]]
    distance = np.sum((points - seeds[0]) ** 2, axis=1)
    for _ in range(maximum - 1):
        index = int(np.argmax(distance))
        if distance[index] < 60 ** 2:
            break
        seeds.append(points[index])
        distance = np.minimum(distance, np.sum((points - points[index]) ** 2, axis=1))
    return [(int(x), int(y)) for x, y in seeds]


def _component_at_seed(mask, x, y):
    labels, _ = ndimage.label(mask)
    value = labels[y, x]
    return labels == value if value else np.zeros_like(mask, bool)


def candidate_road_from_original_context(arrays, points, camera, road_hint, predictor, pixel_size_m, config):
    """Individually prompt original-photo crops and project actual mask pixels.

    A hint controls search only. It does not certify road, empty background,
    bridges, or paint. SAM holes are preserved and never blindly filled.
    """
    shape = arrays["valid"].shape
    output = np.zeros(shape, bool)
    report = {"acceptedSemanticTruth": False, "promptRecords": [], "hintIsSearchOnly": True,
              "holesFilled": False, "samScoreIsProbabilityOfCorrectRoad": False}
    if predictor is None:
        return output, {**report, "reason": "no-road-predictor"}
    if any(key not in arrays for key in ("source_crop_image", "source_crop_valid", "source_crop_body", "source_crop_window")):
        return output, {**report, "reason": "original-photo-context-not-cached"}
    raw = arrays["source_crop_image"]
    raw_valid = arrays["source_crop_valid"]
    raw_body = arrays["source_crop_body"]
    x0, y0, _, _ = arrays["source_crop_window"]
    uv = camera.world_to_pixel(points.reshape(3, -1)).reshape(2, *shape).astype(np.float32)
    maps = uv - np.asarray([x0, y0], np.float32)[:, None, None]
    safe_maps = np.nan_to_num(maps, nan=-1e6, posinf=-1e6, neginf=-1e6)
    possible = neutral_road_seed_pixels(arrays["image"], arrays["valid"], arrays["body_mask"], pixel_size_m, config) & road_hint
    for seed_x, seed_y in _spaced_seed_points(possible, config.max_seed_prompts):
        source_x, source_y = maps[:, seed_y, seed_x]
        sx, sy = int(round(float(source_x))), int(round(float(source_y)))
        if not (0 <= sx < raw.shape[1] and 0 <= sy < raw.shape[0]) or not raw_valid[sy, sx] or raw_body[sy, sx]:
            continue
        radius = config.source_prompt_radius_px
        cx0, cy0 = max(0, sx - radius), max(0, sy - radius)
        cx1, cy1 = min(raw.shape[1], sx + radius + 1), min(raw.shape[0], sy + radius + 1)
        context = raw[cy0:cy1, cx0:cx1]
        color = context.astype(np.float32)
        green = (color[..., 1] > color[..., 0] + 12) & (color[..., 1] > color[..., 2] + 8)
        negative_mask = (green | raw_body[cy0:cy1, cx0:cx1]) & raw_valid[cy0:cy1, cx0:cx1]
        negatives = _spaced_seed_points(negative_mask, config.max_negative_prompts) if config.max_negative_prompts else []
        prompt = np.asarray([(sx - cx0, sy - cy0), *negatives], np.float32)
        labels = np.r_[1, np.zeros(len(negatives), np.int32)].astype(np.int32)
        predictor.set_image(context)
        multimask = len(prompt) <= 3
        masks, scores, _ = predictor.predict(point_coords=prompt, point_labels=labels,
                                            multimask_output=multimask, return_logits=False)
        options = []
        option_records = []
        for index, (mask, score) in enumerate(zip(masks, np.ravel(scores))):
            mask = np.asarray(mask)
            if mask.shape != context.shape[:2]:
                option_records.append({'index': index, 'score': float(score), 'reason': 'wrong-mask-shape'})
                continue
            if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
                raise ValueError('SAM return_logits=False requires binary mask values, not arbitrary float logits')
            mask = mask.astype(bool)
            component = _component_at_seed(mask, sx - cx0, sy - cy0)
            fraction = float(component.mean())
            reasons = []
            if not np.isfinite(score) or float(score) < config.min_sam_score: reasons.append('low-model-mask-quality')
            if component.sum() < config.min_component_pixels: reasons.append('insufficient-seed-component')
            if fraction > config.max_component_fraction: reasons.append('excessive-component-fraction')
            option_records.append({'index': index, 'score': float(score), 'maskPixels': int(mask.sum()),
                                   'seedComponentPixels': int(component.sum()), 'componentFraction': fraction,
                                   'rejectionReasons': reasons})
            if reasons:
                continue
            component &= raw_valid[cy0:cy1, cx0:cx1]
            component &= ~raw_body[cy0:cy1, cx0:cx1]
            options.append((float(score), index, component, fraction))
        if not options:
            report["promptRecords"].append({"roiSeed": [seed_x, seed_y], 'sourceSeed': [sx+int(x0), sy+int(y0)],
                'positivePrompts': 1, 'negativePrompts': len(negatives), 'multimaskOutput': multimask,
                'options': option_records, "reason": "no-supported-single-component-mask"})
            continue
        score, _, chosen, fraction = max(options, key=lambda v: (v[0], -v[3]))
        source_mask = np.zeros(raw.shape[:2], np.uint8)
        source_mask[cy0:cy1, cx0:cx1] = chosen
        projected = cv2.remap(source_mask, safe_maps[0], safe_maps[1], cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).astype(bool)
        projected &= arrays["valid"] & road_hint & ~arrays["body_mask"]
        output |= projected
        report["promptRecords"].append({"roiSeed": [seed_x, seed_y], "sourceSeed": [sx + int(x0), sy + int(y0)],
            "contextWindowInCachedPhoto": [cx0, cy0, cx1 - cx0, cy1 - cy0], "samScore": score,
            'positivePrompts': 1, 'negativePrompts': len(negatives), 'multimaskOutput': multimask, 'options': option_records,
            "componentFraction": fraction, "projectedCandidatePixels": int(projected.sum()), "reason": "candidate-only"})
    report.update(reason="candidate-only" if output.any() else "no-road-candidate", candidatePixels=int(output.sum()))
    return output, report


def observed_paint_candidate(image, valid, body, road, pixel_size_m, config=ObservationConfig()):
    """Keep only actual neutral bright pixels neighbouring a road candidate."""
    if not road.any():
        return np.zeros(road.shape, bool)
    rgb = np.asarray(image, np.float32)
    bright = (rgb.min(axis=2) >= config.paint_min_rgb) & (rgb.max(axis=2) - rgb.min(axis=2) < 30)
    radius = max(1, math.ceil(config.paint_neighbourhood_m / pixel_size_m))
    # This does not fill a SAM hole: only photograph pixels satisfying the
    # independent paint appearance condition enter the candidate set.
    nearby = ndimage.distance_transform_edt(~road) <= radius
    return bright & valid & ~body & nearby


def static_road_appearance(image):
    """A measurement-domain guard; neutral asphalt and actual yellow markings.

    Color is not semantic road truth. Apply only inside a separately declared
    road search domain; red flags and other saturated objects are excluded.
    """
    rgb = np.asarray(image, np.float32)
    r, g, b = rgb.transpose(2, 0, 1)
    yellow = (r >= 140) & (g >= 120) & (b <= 160) & (r >= g - 15) & (r - b >= 40) & (g - b >= 30)
    return (np.ptp(rgb, axis=2) <= 35) | yellow


def pair_geometry_support(first, second, first_road, second_road, pixel_size_m, config=ObservationConfig()):
    """Exclude ONLY this pair's possible dynamic shadow before measuring.

    The conservative radius is a measurement exclusion, not observed shadow,
    empty-road evidence or a permanent prohibition on static output context.
    """
    body = first['body_mask'] | second['body_mask']
    distance = ndimage.distance_transform_edt(~body) * pixel_size_m if body.any() else np.full(body.shape, np.inf)
    if config.geometry_shadow_guard_mode == 'circular':
        protected = distance <= config.geometry_possible_shadow_clearance_m
    elif config.geometry_shadow_guard_mode == 'observed-dark-near-body':
        protected = (distance <= config.static_body_clearance_m)
        protected |= possible_dark_shadow_guard(first, first_road, pixel_size_m, config)
        protected |= possible_dark_shadow_guard(second, second_road, pixel_size_m, config)
    else:
        raise ValueError('Unknown geometry shadow guard mode')
    return (first_road & second_road & first['valid'] & second['valid'] & ~protected &
            static_road_appearance(first['image']) & static_road_appearance(second['image']))


def possible_dark_shadow_guard(source, road, pixel_size_m, config=ObservationConfig()):
    """Exclude observed near-body darkening without needing pair registration.

    This is a conservative measurement guard, never a semantic shadow label
    or authority to replace a pixel. It also excludes static dark road/tree
    shadows when they fall nearby. Unknown/no background support stays guarded.
    """
    body = source['body_mask']
    result = np.zeros(body.shape, bool)
    if not body.any(): return result
    gray = source['image'].astype(np.float32).mean(axis=2)
    distance = ndimage.distance_transform_edt(~body) * pixel_size_m
    near = distance <= config.geometry_possible_shadow_clearance_m
    background = near & (distance > config.static_body_clearance_m) & road & source['valid'] & ~body
    values = gray[background]
    if len(values) < 128: return near
    # A bright comparison level in this exposure's bounded road domain
    # deliberately over-excludes darker asphalt and weak surrounding shade.
    level = float(np.percentile(values,75))
    dark = near & road & source['valid'] & ~body
    dark &= (gray <= level*config.geometry_dark_shadow_ratio) | (gray <= level-config.geometry_dark_shadow_difference)
    if dark.any():
        result |= ndimage.distance_transform_edt(~dark)*pixel_size_m <= config.geometry_dark_shadow_margin_m
    return result


def fit_exposure_relation(selected, reference, support, config=ObservationConfig()):
    """Estimate comparison-only RGB gains/offsets from retained static pixels."""
    yy, xx = np.where(support)
    if len(xx) < config.min_color_fit_pixels:
        return None, {"accepted": False, "reason": "insufficient-static-color-support", "pixels": int(len(xx))}
    stride = max(1, len(xx) // 20000)
    a = selected[yy[::stride], xx[::stride]].astype(np.float32)
    b = reference[yy[::stride], xx[::stride]].astype(np.float32)
    gain, offset = [], []
    for channel in range(3):
        qa, qb = np.percentile(a[:, channel], [25, 50, 75]), np.percentile(b[:, channel], [25, 50, 75])
        slope = 1.0 if qb[2] - qb[0] < 8 else (qa[2] - qa[0]) / (qb[2] - qb[0])
        if not 0.7 <= slope <= 1.4:
            return None, {"accepted": False, "reason": "implausible-exposure-gain", "channel": channel}
        bias = float(qa[1] - slope * qb[1])
        if abs(bias) > 45:
            return None, {"accepted": False, "reason": "implausible-exposure-offset", "channel": channel}
        gain.append(float(slope))
        offset.append(bias)
    adjusted = reference.astype(np.float32) * np.asarray(gain, np.float32) + np.asarray(offset, np.float32)
    residual = np.max(np.abs(selected.astype(np.float32) - adjusted), axis=2)
    return (adjusted, residual), {"accepted": True, "gain": gain, "offset": offset, "pixels": int(len(xx)),
        "staticResidualP90": float(np.percentile(residual[support], 90)), "comparisonOnly": True, "outputRgbChanged": False}


def measured_shadow_change(selected, adjusted_reference, own_body, reference_body, common_valid, pixel_size_m, config=ObservationConfig()):
    """Return measured near-body darkening and unexplained near-body changes.

    A shadow candidate must attach to this exposure's own body vicinity, and
    may not be another exposure's vehicle. It is still not a semantic proof.
    """
    if not own_body.any():
        return np.zeros(own_body.shape, bool), np.zeros(own_body.shape, bool)
    selected_f = selected.astype(np.float32)
    gray_a = selected_f.mean(axis=2)
    gray_b = adjusted_reference.mean(axis=2)
    chroma_a = selected_f / np.maximum(selected_f.sum(axis=2, keepdims=True), 1)
    chroma_b = adjusted_reference / np.maximum(adjusted_reference.sum(axis=2, keepdims=True), 1)
    near = ndimage.distance_transform_edt(~own_body) * pixel_size_m <= config.shadow_max_body_distance_m
    changed = (np.max(np.abs(selected_f - adjusted_reference), axis=2) > config.agreement_rgb_threshold) & common_valid & near
    dark = (gray_b - gray_a >= config.shadow_difference) & (gray_a < 0.85 * np.maximum(gray_b, 1))
    dark &= np.max(np.abs(chroma_a - chroma_b), axis=2) <= config.shadow_chroma_difference
    dark &= common_valid & near & ~own_body & ~reference_body
    labels, n = ndimage.label(dark)
    sizes = np.bincount(labels.ravel(), minlength=n + 1)
    attached = ndimage.distance_transform_edt(~own_body) * pixel_size_m <= config.shadow_body_connection_m
    ids = np.unique(labels[attached])
    ids = ids[(ids > 0) & (sizes[ids] >= config.min_change_component_pixels)]
    measured = dark & np.isin(labels, ids)
    # A known vehicle in the OTHER real exposure explains a difference; it is
    # not evidence of an unknown foreground in this selected exposure.
    # That pair still cannot certify empty road there: reference_body blocks it.
    unknown = changed & ~own_body & ~reference_body & ~measured
    return measured, unknown


def clear_road_pair_candidate(selected_road, selected_body, selected_shadow, selected_unknown,
                              reference_road, reference_body, reference_shadow, reference_unknown,
                              valid, residual, pixel_size_m, config=ObservationConfig()):
    """Positive visible road in BOTH clear exposures, even if a third has a car."""
    own_forbidden = selected_body | selected_shadow | selected_unknown
    other_forbidden = reference_body | reference_shadow | reference_unknown
    radius = config.empty_body_clearance_m / pixel_size_m
    own_clear = ndimage.distance_transform_edt(~own_forbidden) > radius if own_forbidden.any() else np.ones(valid.shape, bool)
    other_clear = ndimage.distance_transform_edt(~other_forbidden) > radius if other_forbidden.any() else np.ones(valid.shape, bool)
    return selected_road & reference_road & own_clear & other_clear & valid & (residual <= config.agreement_rgb_threshold)


def evaluate_static_pair(first, second, support, *, static_provider=None, static_provider_name=None,
                         geometry_config=GeometryConfig()):
    """Evaluate a predeclared primary measurement and retain independent SIFT audit.

    ``static_provider`` receives (first_rgb, second_rgb, static_mask). Its
    result is primary whether it passes or fails. There is no automatic
    passing-provider selection and no threshold adjustment after evaluation.
    """
    if static_provider is not None and not static_provider_name:
        raise ValueError("An alternative static measurement must be named before evaluation")
    sift = independent_static_evidence(first, second, support, geometry_config)
    if static_provider is None:
        primary = dict(sift)
        name = "sift-mutual-heldout"
    else:
        primary = dict(static_provider(first, second, support))
        name = static_provider_name
    primary.update(primaryStaticProvider=name, providerSelectionPredeclared=True,
                   automaticPassingProviderSelection=False, siftAudit=sift)
    return primary


def _annotate_holdout(proof, shape):
    """Keep every match and make the fixed checkerboard split explicit."""
    height, width = shape
    result = dict(proof)
    if "correspondences" in result:
        records = []
        for value in result["correspondences"]:
            item = dict(value)
            if "heldout" not in item and "first" in item:
                point = np.asarray(item["first"], np.float64)
                cells = np.clip((point / [width, height] * 4).astype(int), 0, 3)
                item["heldout"] = bool((int(cells[0]) + int(cells[1])) % 2)
                item["checkerboardCell"] = int(cells[1] * 4 + cells[0])
            records.append(item)
        result["correspondences"] = records
    if "siftAudit" in result:
        result["siftAudit"] = _annotate_holdout(result["siftAudit"], shape)
    return result


def publish_pair_proof(output_dir, index, proof, shape):
    """Preserve full primary and SIFT matches, including rejected holdouts."""
    full = _annotate_holdout(proof, shape)
    path = Path(output_dir) / f"source-pair-proof-{index:03d}.json"
    atomic_json(path, full)
    summary = {k: v for k, v in full.items() if k not in ("correspondences", "siftAudit")}
    summary["primaryCorrespondenceCount"] = len(full.get("correspondences", []))
    sift = full.get("siftAudit", {})
    summary["siftAudit"] = {k: v for k, v in sift.items() if k != "correspondences"}
    summary["siftAudit"]["correspondenceCount"] = len(sift.get("correspondences", []))
    summary.update(fullProofPath=str(path.resolve()), fullProofSha256=sha256(path),
                   allCorrespondencesPreserved=True, actualHeldoutOutliersPreserved=True)
    return summary


def directional_clear_support(name, sources, pairs, pixel_size_m, config=ObservationConfig()):
    """Aggregate clear witnesses without contaminating them with other times.

    Unknown evidence is directional and reference-specific. A/C positive
    calibrated clear-road agreement remains usable when B has a historical
    vehicle or shadow. Unexplained changes remain unknown unless an actual
    clear pair provides positive evidence at exactly those pixels.
    """
    source = sources[name]
    arrays = source["arrays"]
    shape = arrays["valid"].shape
    counts = np.zeros(shape, np.uint16)
    static = np.zeros(shape, bool)
    unresolved_union = np.zeros(shape, bool)
    refs, archive = [], {}
    for other, reference in sources.items():
        if (name, other) not in pairs or (other, name) not in pairs:
            continue
        forward, reverse = pairs[(name, other)], pairs[(other, name)]
        if any(pair.get("geometryAccepted") is not True or pair.get("colorComparisonAccepted") is not True for pair in (forward, reverse)):
            continue
        # Counter-shadow is measured on the reference exposure in the SAME
        # accepted pair. Do not use an unverified all-time shadow difference.
        own_unknown = forward["unknown"] & ~reverse["shadow"]
        reference_unknown = reverse["unknown"] & ~forward["shadow"]
        unresolved_union |= own_unknown
        clear = clear_road_pair_candidate(source["road"], arrays["body_mask"], source["shadow"], own_unknown,
            reference["road"], reference["arrays"]["body_mask"], reference["shadow"], reference_unknown,
            arrays["valid"] & reference["arrays"]["valid"], forward["residual"], pixel_size_m, config)
        counts += clear.astype(np.uint16)
        key = f"clear_reference_{len(refs)}"
        archive[key] = clear
        archive[f"unknown_vs_reference_{len(refs)}"] = own_unknown
        # After accepted geometry, positive clear road in this actual pair
        # can support an output boundary. A 4m measurement exclusion must not
        # permanently prohibit a true no-cut object context or historical spot.
        static |= clear
        refs.append({"sourceImage": other, "maskKey": key,
            "unknownMaskKey": f"unknown_vs_reference_{len(refs)}",
            "cacheSha256": reference["record"]["cacheSha256"], "supportedEmptyRoadPixels": int(clear.sum())})
    positive_clear = counts >= config.min_clear_reference_sources
    resolved = unresolved_union & positive_clear
    archive["cross_reference_conflict_resolved_by_clear_pair"] = resolved
    archive["unexplained_pairwise_difference_union"] = unresolved_union
    return counts, static, unresolved_union & ~positive_clear, refs, archive


def build_observation_candidates(manifest_path: Path, output_dir: Path, *, predictor,
                                 predictor_identity: dict, source_names=None, road_hint=None,
                                 config=ObservationConfig(), static_provider=None,
                                 static_provider_name=None, static_provider_identity=None,
                                 reuse_road_manifest=None):
    """Publish candidate masks and complete rejection evidence, never a patch.

    Returns observation-manifest.json. The root acceptance stage must verify
    source vehicle appearance, semantic domain and the actual baseline seam.
    The output's ``candidate_empty_road`` cannot be relabelled certified solely
    because a detector found no object.
    """
    started = time.monotonic()
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    if output_dir == manifest_path.parent or manifest_path.is_relative_to(output_dir):
        raise ValueError("Observation output overlaps retained projection evidence")
    if not predictor_identity:
        raise ValueError("Road predictor/config identity is required")
    if static_provider is not None and (not static_provider_name or not static_provider_identity):
        raise ValueError("Alternative static provider name and implementation identity must be predeclared")
    if config.min_clear_reference_sources < 1 or config.max_sources < 2 or config.max_seed_prompts < 1:
        raise ValueError("At least two real exposures and positive prompt/support limits are required")
    if isinstance(config.max_negative_prompts, bool) or not isinstance(config.max_negative_prompts, int) or config.max_negative_prompts < 0:
        raise ValueError('Negative prompt count must be a nonnegative integer')
    if not math.isfinite(config.geometry_possible_shadow_clearance_m) or config.geometry_possible_shadow_clearance_m < config.shadow_max_body_distance_m:
        raise ValueError('Geometry measurement must protect the declared possible near-body shadow radius')
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "status.json", {"state": "preparing", "schema": SCHEMA})
    input_manifest_sha = sha256(manifest_path)
    import json
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reused = None
    if reuse_road_manifest is not None:
        reuse_road_manifest = Path(reuse_road_manifest).resolve()
        reused = json.loads(reuse_road_manifest.read_text(encoding='utf-8'))
        if reused.get('state') != 'candidate-evidence-complete' or reused.get('projectionManifestSha256') != input_manifest_sha:
            raise ValueError('Road candidate reuse is not bound to this completed projection')
        reused_records = {r['sourceImage']: r for r in reused['records']}
        reuse_identity = {'path': str(reuse_road_manifest), 'sha256': sha256(reuse_road_manifest),
                          'originalPredictorIdentity': reused['predictorIdentity'], 'semanticTruthCertified': False}
    else:
        reuse_identity = None
    names = source_names or [r["sourceImage"] for r in manifest["sources"][:config.max_sources]]
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Require at least two distinct real source exposures")
    sources = {}
    for name in names:
        arrays, record, checked = load_bound_source(manifest_path, name)
        sources[name] = {"arrays": arrays, "record": record}
    grid = manifest["grid"]
    transform = np.asarray(grid["transform"], np.float64)
    pixel_size = float(math.sqrt(abs(transform[0] * transform[4])))
    with np.load(manifest_path.parent / "baseline-roi.npz", allow_pickle=False) as old:
        hint = old["corridor"].astype(bool) if road_hint is None else np.asarray(road_hint, bool)
        z = old["surface_z"].astype(np.float64)
    if hint.shape != z.shape:
        raise ValueError("Road search hint uses a different pixel frame")
    yy, xx = np.mgrid[:z.shape[0], :z.shape[1]]
    points = np.stack([transform[0] * (xx + 0.5) + transform[2], transform[4] * (yy + 0.5) + transform[5], z])
    reconstruction_paths = [Path(p) for p in manifest["criticalInputs"] if Path(p).name == "reconstruction.json"]
    project = reconstruction_paths[0].parents[2]
    import rasterio as rio
    factory, _ = cameras_from_checkpoint(project, rio.CRS.from_string(grid["crs"]))
    geometry_config = GeometryConfig()
    for index, name in enumerate(names):
        source = sources[name]
        arrays = source["arrays"]
        if reused is None:
            road, road_report = candidate_road_from_original_context(arrays, points, factory.get(name), hint, predictor, pixel_size, config)
        else:
            parent = reused_records[name]
            if parent['sourceCacheSha256'] != source['record']['cacheSha256']:
                raise ValueError('Reused road candidates refer to a different source image cache')
            cache = Path(parent['observationCachePath'])
            if sha256(cache) != parent['observationCacheSha256']:
                raise ValueError('Reused candidate cache changed')
            with np.load(cache, allow_pickle=False) as archive:
                road = archive['sam_road_candidate']
            if road.dtype != bool or road.shape != hint.shape:
                raise ValueError('Reused road mask has a different shape or dtype')
            road = road & hint & arrays['valid'] & ~arrays['body_mask']
            road_report = {**parent['roadCandidateReport'], 'candidatePixelsInNewSearchDomain': int(road.sum()),
                           'reuse': reuse_identity, 'acceptedSemanticTruth': False}
        paint = observed_paint_candidate(arrays["image"], arrays["valid"], arrays["body_mask"], road, pixel_size, config) & hint
        source.update(road=road | paint, road_mask=road, paint=paint, road_report=road_report,
                      shadow=np.zeros(z.shape, bool), pairs=[])
        atomic_json(output_dir / "status.json", {"schema": SCHEMA, "state": "road-candidates", "completedSources": index + 1, "totalSources": len(names)})
    pair_records = []
    pairs = {}
    for index, name in enumerate(names):
        first = sources[name]
        for other in names[index + 1:]:
            second = sources[other]
            a, b = first["arrays"], second["arrays"]
            support = pair_geometry_support(a, b, first['road'], second['road'], pixel_size, config)
            proof = evaluate_static_pair(a["image"], b["image"], support, static_provider=static_provider,
                static_provider_name=static_provider_name, geometry_config=geometry_config)
            proof.update(firstSource=name, secondSource=other, firstCacheSha256=first["record"]["cacheSha256"],
                         secondCacheSha256=second["record"]["cacheSha256"], staticMaskSha256=array_digest(support),
                         firstRgbSha256=array_digest(a["image"]), secondRgbSha256=array_digest(b["image"]),
                         projectionIdentitySha256=manifest["identitySha256"], grid=grid, staticPixels=int(support.sum()))
            pair_records.append(proof)
            if not proof["accepted"]:
                continue
            corrected, fit = fit_exposure_relation(a["image"], b["image"], support, config)
            if corrected is None:
                proof.update(colorComparisonAccepted=False, colorComparison=fit)
                continue
            adjusted_b, residual_ab = corrected
            reverse, fit_reverse = fit_exposure_relation(b["image"], a["image"], support, config)
            if reverse is None:
                proof.update(colorComparisonAccepted=False, colorComparison=fit_reverse)
                continue
            adjusted_a, residual_ba = reverse
            common = a["valid"] & b["valid"]
            shadow_a, unknown_a = measured_shadow_change(a["image"], adjusted_b, a["body_mask"], b["body_mask"], common, pixel_size, config)
            shadow_b, unknown_b = measured_shadow_change(b["image"], adjusted_a, b["body_mask"], a["body_mask"], common, pixel_size, config)
            first["shadow"] |= shadow_a
            second["shadow"] |= shadow_b
            pairs[(name, other)] = {"residual": residual_ab, "support": support, "shadow": shadow_a, "unknown": unknown_a,
                                   "geometryAccepted": True, "colorComparisonAccepted": True}
            pairs[(other, name)] = {"residual": residual_ba, "support": support, "shadow": shadow_b, "unknown": unknown_b,
                                   "geometryAccepted": True, "colorComparisonAccepted": True}
            proof.update(colorComparisonAccepted=True, colorComparison=fit, reverseColorComparison=fit_reverse,
                         measuredSelectedShadowPixels=int(shadow_a.sum()), measuredReferenceShadowPixels=int(shadow_b.sum()))
    pair_summaries = [publish_pair_proof(output_dir, i, p, z.shape) for i, p in enumerate(pair_records)]
    records = []
    for index, name in enumerate(names):
        source, refs = sources[name], []
        arrays = source["arrays"]
        counts, static, pairwise_unknown, refs, reference_arrays = directional_clear_support(name, sources, pairs, pixel_size, config)
        incomplete = np.zeros(z.shape, bool)
        foregrounds = []
        output_arrays = {"road_candidate": source["road"], "sam_road_candidate": source["road_mask"],
                         "observed_paint_candidate": source["paint"], "measured_shadow_candidate": source["shadow"],
                         "candidate_empty_road": counts >= config.min_clear_reference_sources,
                         "clear_reference_count": counts, "static_support_candidate": static, **reference_arrays}
        for vehicle_index, vehicle in enumerate(source["record"]["vehicles"]):
            body = arrays[vehicle["maskKey"]]
            nearby = ndimage.distance_transform_edt(~body) * pixel_size <= config.shadow_max_body_distance_m
            measured = source["shadow"] & nearby
            mask_key = f"foreground_{vehicle_index}"
            output_arrays[mask_key] = body | measured
            if not vehicle["complete"]:
                incomplete |= body
            foregrounds.append({"maskKey": mask_key, "vehicleMaskKey": vehicle["maskKey"], "sourceVehicleIndex": vehicle_index,
                "bodyCompleteInProjectionCache": bool(vehicle["complete"]), "shadowCompleteCertified": False,
                "measuredShadowPixels": int(measured.sum()), "requiresAppearanceAndShadowBoundaryReview": True})
        unknown = (pairwise_unknown & ~source["shadow"]) | incomplete
        unknown |= hint & ~arrays["valid"]
        output_arrays["unknown_foreground_candidate"] = unknown
        output_arrays["candidate_empty_road"] &= ~unknown
        reasons = []
        if not source["road"].any(): reasons.append("no-positive-road-candidate")
        if not refs: reasons.append("no-independent-calibrated-clear-reference")
        if incomplete.any(): reasons.append("incomplete-source-vehicle-observations")
        if unknown.any(): reasons.append("unexplained-near-body-change-or-incomplete-foreground")
        path = output_dir / f"observations-{index:03d}.npz"
        atomic_npz(path, **output_arrays)
        record = {"schema": SCHEMA, "sourceImage": name, "sourceCacheSha256": source["record"]["cacheSha256"],
            "projectionManifestSha256": input_manifest_sha, "projectionIdentitySha256": manifest["identitySha256"],
            "grid": grid, "observationCachePath": str(path), "observationCacheSha256": sha256(path),
            "roadCandidateReport": source["road_report"], "clearReferences": refs, "foregrounds": foregrounds,
            "candidatePixels": {key: int(value.sum()) for key, value in output_arrays.items() if value.dtype == bool},
            "candidateForegroundCatalogComplete": False, "semanticTruthCertified": False,
            "sourceToBaselineStaticProofProvided": False, "automaticMosaicPermission": False,
            "zeroDetectionsUsedAsEmptyRoadProof": False, "samScoreUsedAsSemanticTruth": False,
            "emptyRoadSupportInterpretation": "Selected real exposure plus independent clear real references; other frames may contain historical vehicles",
            "rejectionReasons": reasons, "requiresRootAcceptance": True}
        atomic_json(path.with_suffix(".json"), record)
        records.append(record)
    if sha256(manifest_path) != input_manifest_sha:
        raise ValueError("Retained projection manifest changed during observation extraction")
    # Recheck source/critical input provenance at publication, without trusting
    # the fact that an earlier cache load once succeeded.
    for name in names:
        load_bound_source(manifest_path, name)
    if reused is not None and sha256(reuse_road_manifest) != reuse_identity['sha256']:
        raise ValueError('Reused road candidate manifest changed during extraction')
    report = {"schema": SCHEMA, "state": "candidate-evidence-complete", "projectionManifest": str(manifest_path),
        "projectionManifestSha256": input_manifest_sha, "config": asdict(config), "predictorIdentity": predictor_identity,
        "primaryStaticProvider": static_provider_name if static_provider is not None else "sift-mutual-heldout",
        'roadCandidateReuse': reuse_identity, 'geometryProtectionScope': 'this source pair only; possible shadow exclusion, not semantic shadow',
        "staticProviderIdentity": static_provider_identity,
        "roadHintSha256": array_digest(hint), "roadHintIsSearchOnly": True, "records": records,
        "sourcePairGeometryEvidence": pair_summaries, "outputRgbGenerated": False,
        "automaticMosaicPermission": False, "elapsedSeconds": time.monotonic() - started}
    report["candidateEvidenceIdentitySha256"] = canonical_digest({"projectionManifestSha256": input_manifest_sha,
        "config": asdict(config), "predictorIdentity": predictor_identity, "roadHintSha256": array_digest(hint),
        "primaryStaticProvider": report["primaryStaticProvider"], "staticProviderIdentity": static_provider_identity})
    output = output_dir / "observation-manifest.json"
    atomic_json(output, report)
    atomic_json(output_dir / "status.json", {"state": "candidate-evidence-complete", "schema": SCHEMA, "manifestSha256": sha256(output)})
    return output
