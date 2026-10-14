"""Physical counterexamples for candidate road observations, not model accuracy."""

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("rasterio")
pytest.importorskip("scipy")

import model_engine.ogs2_corridor_observations as observations
from model_engine.ogs2_corridor_observations import (
    ObservationConfig,
    clear_road_pair_candidate,
    directional_clear_support,
    evaluate_static_pair,
    fit_exposure_relation,
    measured_shadow_change,
    neutral_road_seed_pixels,
    observed_paint_candidate,
    publish_pair_proof,
)


def masks(shape=(80, 80)):
    return np.ones(shape, bool), np.zeros(shape, bool)


def test_third_exposure_historical_vehicle_does_not_block_two_clear_real_exposures():
    road, blank = masks()
    historical_vehicle_in_third_source = blank.copy()
    historical_vehicle_in_third_source[30:45, 30:50] = True
    result = clear_road_pair_candidate(road, blank, blank, blank, road, blank, blank, blank,
                                      road, np.zeros(road.shape, np.float32), 0.03)
    assert result[historical_vehicle_in_third_source].all()
    # Excluding this third exposure is deliberate: it is the old location to
    # replace with independently observed empty road in the selected exposure.


def test_either_actual_supporting_exposure_vehicle_blocks_empty_claim():
    road, blank = masks()
    body = blank.copy()
    body[30:45, 30:50] = True
    for selected, reference in ((body, blank), (blank, body)):
        result = clear_road_pair_candidate(road, selected, blank, blank, road, reference, blank, blank,
                                          road, np.zeros(road.shape, np.float32), 0.03)
        assert not result[body].any()


def test_zero_detection_cannot_create_positive_road_or_empty_claim():
    valid, blank = masks()
    result = clear_road_pair_candidate(blank, blank, blank, blank, blank, blank, blank, blank,
                                      valid, np.zeros(valid.shape, np.float32), 0.03)
    assert not result.any()
    result = clear_road_pair_candidate(valid, blank, blank, blank, blank, blank, blank, blank,
                                      valid, np.zeros(valid.shape, np.float32), 0.03)
    assert not result.any()


def test_unknown_foreground_shadow_or_invalid_support_prevents_empty_claim():
    road, blank = masks()
    suspect = blank.copy()
    suspect[30:45, 30:50] = True
    for shadow, unknown, valid in ((suspect, blank, road), (blank, suspect, road), (blank, blank, ~suspect)):
        result = clear_road_pair_candidate(road, blank, shadow, unknown, road, blank, blank, blank,
                                          valid, np.zeros(road.shape, np.float32), 0.03)
        assert not result[suspect].any()


def test_actual_bright_photo_pixels_can_support_paint_but_holes_are_not_filled():
    road, body = masks((40, 40))
    road[:] = False
    road[10:30, 10:30] = True
    road[15:20, 15:20] = False
    image = np.full((40, 40, 3), 100, np.uint8)
    image[16:18, 16:18] = 200
    paint = observed_paint_candidate(image, np.ones(road.shape, bool), body, road, 0.03)
    assert paint[16:18, 16:18].all()
    assert not paint[15, 15]
    assert not observed_paint_candidate(image, np.ones(road.shape, bool), body, np.zeros_like(road), 0.03).any()


def test_global_exposure_change_is_removed_for_comparison_not_reported_as_motion():
    rng = np.random.default_rng(10)
    reference = rng.integers(60, 160, size=(80, 80, 3), dtype=np.uint8)
    selected = np.round(reference.astype(float) * 1.1 + 9).astype(np.uint8)
    body = np.zeros((80, 80), bool)
    body[30:40, 30:45] = True
    support = ~body
    fitted, report = fit_exposure_relation(selected, reference, support)
    assert report["accepted"] and report["comparisonOnly"] and not report["outputRgbChanged"]
    adjusted, residual = fitted
    assert np.percentile(residual[support], 90) < 2
    shadow, unknown = measured_shadow_change(selected, adjusted, body, np.zeros_like(body), np.ones_like(body), 0.03)
    assert not shadow.any() and not unknown.any()


def test_local_observed_dark_shadow_retains_measured_pixels_without_dilated_semantic_claim():
    reference = np.full((80, 80, 3), 150, np.float32)
    selected = reference.astype(np.uint8)
    body = np.zeros((80, 80), bool)
    body[30:40, 30:40] = True
    selected[31:39, 40:49] = 60
    shadow, _ = measured_shadow_change(selected, reference, body, np.zeros_like(body), np.ones_like(body), 0.03)
    assert shadow[31:39, 40:49].all()
    assert not shadow[30, 49] and not shadow[body].any()


def test_no_body_cannot_make_virtual_edge_shadow_or_paint_from_absent_road():
    road, blank = masks()
    selected = np.full((80, 80, 3), 50, np.uint8)
    reference = np.full((80, 80, 3), 150, np.float32)
    shadow, unknown = measured_shadow_change(selected, reference, blank, blank, road, 0.03)
    assert not shadow.any() and not unknown.any()
    seeds = neutral_road_seed_pixels(selected, road, blank, 0.03)
    assert seeds.all()


def test_primary_provider_failure_cannot_be_replaced_by_passing_sift_audit(monkeypatch):
    monkeypatch.setattr(observations, "independent_static_evidence", lambda *a, **k: {"accepted": True, "reason": "verified"})
    provider = lambda *a: {"accepted": False, "reason": "actual-static-displacement", "p90ActualPixels": 3.0}
    image = np.zeros((80, 80, 3), np.uint8)
    result = evaluate_static_pair(image, image, np.ones((80, 80), bool),
                                  static_provider=provider, static_provider_name="predeclared-road-paint-ncc")
    assert result["siftAudit"]["accepted"]
    assert not result["accepted"] and not result["automaticPassingProviderSelection"]
    assert result["primaryStaticProvider"] == "predeclared-road-paint-ncc"


def test_default_sift_failure_stays_rejected_and_unnamed_alternative_refused(monkeypatch):
    monkeypatch.setattr(observations, "independent_static_evidence", lambda *a, **k: {"accepted": False, "reason": "insufficient-independent-matches"})
    image = np.zeros((80, 80, 3), np.uint8)
    support = np.ones((80, 80), bool)
    result = evaluate_static_pair(image, image, support)
    assert not result["accepted"]
    assert result["primaryStaticProvider"] == "sift-mutual-heldout"
    with pytest.raises(ValueError, match="named before"):
        evaluate_static_pair(image, image, support, static_provider=lambda *a: {"accepted": True})


def _historical_vehicle_scene():
    shape = (100, 100)
    blank, valid = np.zeros(shape, bool), np.ones(shape, bool)
    current_body = blank.copy()
    current_body[20:30, 20:30] = True
    historic_body = blank.copy()
    historic_body[20:30, 42:52] = True
    historic_shadow = blank.copy()
    historic_shadow[21:29, 52:61] = True
    image_a = np.full((*shape, 3), 150, np.uint8)
    image_a[current_body] = 40
    image_b = image_a.copy()
    image_b[historic_body] = 60
    image_b[historic_shadow] = 50
    sources = {}
    for name, image, body in (("A", image_a, current_body), ("B", image_b, current_body | historic_body), ("C", image_a.copy(), current_body)):
        sources[name] = {"arrays": {"image": image, "valid": valid.copy(), "body_mask": body.copy()},
                         "record": {"cacheSha256": name * 64}, "road": valid & ~body,
                         "shadow": blank.copy()}
    pairs = {}
    for first, second in (("A", "B"), ("A", "C"), ("B", "C")):
        a, b = sources[first]["arrays"], sources[second]["arrays"]
        shadow_a, unknown_a = measured_shadow_change(a["image"], b["image"].astype(float), a["body_mask"], b["body_mask"], valid, 0.03)
        shadow_b, unknown_b = measured_shadow_change(b["image"], a["image"].astype(float), b["body_mask"], a["body_mask"], valid, 0.03)
        sources[first]["shadow"] |= shadow_a
        sources[second]["shadow"] |= shadow_b
        residual = np.max(np.abs(a["image"].astype(float) - b["image"].astype(float)), axis=2)
        pairs[(first, second)] = {"residual": residual, "support": valid.copy(), "shadow": shadow_a,
                                  "unknown": unknown_a, "geometryAccepted": True, "colorComparisonAccepted": True}
        pairs[(second, first)] = {"residual": residual, "support": valid.copy(), "shadow": shadow_b,
                                  "unknown": unknown_b, "geometryAccepted": True, "colorComparisonAccepted": True}
    return sources, pairs, historic_body, historic_shadow


def test_integration_clear_ac_survives_b_historical_car_and_shadow_near_a_own_car():
    sources, pairs, historical_body, historical_shadow = _historical_vehicle_scene()
    # The actual AB measurement sees B's shadow as an unexplained difference
    # in A before assigning that shadow to the real B exposure.
    assert pairs[("A", "B")]["unknown"][historical_shadow].any()
    counts, _, unknown, refs, archive = directional_clear_support("A", sources, pairs, 0.03)
    assert counts[historical_body | historical_shadow].min() >= 1
    assert not unknown[historical_body | historical_shadow].any()
    ac = next(ref for ref in refs if ref["sourceImage"] == "C")
    ab = next(ref for ref in refs if ref["sourceImage"] == "B")
    assert archive[ac["maskKey"]][historical_body | historical_shadow].all()
    assert not archive[ab["maskKey"]][historical_body | historical_shadow].any()


def test_integration_ab_alone_cannot_authorize_the_same_historical_car_or_shadow():
    sources, pairs, historical_body, historical_shadow = _historical_vehicle_scene()
    sources = {name: sources[name] for name in ("A", "B")}
    pairs = {key: value for key, value in pairs.items() if "C" not in key}
    counts, _, _, _, _ = directional_clear_support("A", sources, pairs, 0.03)
    assert not counts[historical_body | historical_shadow].any()


def test_pair_without_calibrated_geometry_cannot_supply_clear_road():
    sources, pairs, historical_body, historical_shadow = _historical_vehicle_scene()
    pairs[("A", "C")]["geometryAccepted"] = False
    pairs[("C", "A")]["geometryAccepted"] = False
    counts, _, _, refs, _ = directional_clear_support("A", sources, pairs, 0.03)
    assert not counts[historical_body | historical_shadow].any()
    assert all(r["sourceImage"] != "C" for r in refs)


def test_unexplained_difference_is_retained_without_a_real_positive_clear_witness():
    sources, pairs, _, _ = _historical_vehicle_scene()
    sources = {name: sources[name] for name in ("A", "B")}
    pairs = {key: value for key, value in pairs.items() if "C" not in key}
    patch = np.zeros((100, 100), bool)
    patch[60:68, 60:68] = True
    pairs[("A", "B")]["unknown"] |= patch
    pairs[("A", "B")]["residual"][patch] = 70
    pairs[("B", "A")]["residual"][patch] = 70
    counts, _, unknown, _, _ = directional_clear_support("A", sources, pairs, 0.03)
    assert not counts[patch].any()
    assert unknown[patch].all()


def test_full_primary_and_sift_correspondences_retain_outlier_holdout_and_hash(tmp_path):
    import json
    from model_engine.ogs2 import sha256
    proof = {"accepted": False, "reason": "actual-static-displacement", "primaryStaticProvider": "predeclared-ncc",
             "correspondences": [{"first": [10, 10], "second": [10, 10], "inlier": True},
                                 {"first": [30, 10], "second": [33, 10], "inlier": False}],
             "siftAudit": {"accepted": False, "correspondences": [{"first": [60, 10], "second": [64, 10], "inlier": False}]}}
    summary = publish_pair_proof(tmp_path, 0, proof, (100, 100))
    from pathlib import Path
    path = Path(summary["fullProofPath"])
    assert sha256(path) == summary["fullProofSha256"]
    full = json.loads(path.read_text())
    assert len(full["correspondences"]) == 2 and len(full["siftAudit"]["correspondences"]) == 1
    assert full["correspondences"][1]["heldout"] is True
    assert full["correspondences"][1]["inlier"] is False
    assert full["siftAudit"]["correspondences"][0]["inlier"] is False
    assert "correspondences" not in summary and "correspondences" not in summary["siftAudit"]


def _prompt_fixture():
    shape=(80,80); yy,xx=np.mgrid[:80,:80]
    image=np.full((*shape,3),100,np.uint8)
    arrays={'image':image,'valid':np.ones(shape,bool),'body_mask':np.zeros(shape,bool),
            'source_crop_image':image.copy(),'source_crop_valid':np.ones(shape,bool),
            'source_crop_body':np.zeros(shape,bool),'source_crop_window':np.array([0,0,80,80])}
    class Camera:
        def world_to_pixel(self,points): return points[:2]
    class Predictor:
        def set_image(self,context): self.shape=context.shape[:2]
        def predict(self,**kwargs):
            self.kwargs=kwargs
            mask=np.zeros(self.shape,np.float32);mask[15:65,15:65]=1
            return np.array([mask]),np.array([.9]),np.zeros((1,10,10))
    return arrays,np.stack([xx,yy,np.zeros(shape)]),Camera(),Predictor()


def test_actual_sam_float_binary_contract_keeps_single_seed_without_dense_negative_prompts():
    arrays,points,camera,predictor=_prompt_fixture()
    result,report=observations.candidate_road_from_original_context(arrays,points,camera,
        np.ones((80,80),bool),predictor,.03,replace(ObservationConfig(),max_seed_prompts=1))
    assert result.sum()==2500
    assert predictor.kwargs['point_labels'].tolist()==[1]
    assert predictor.kwargs['multimask_output'] is True
    assert predictor.kwargs['return_logits'] is False
    assert report['promptRecords'][0]['negativePrompts']==0
    assert not report['acceptedSemanticTruth']


def test_sam_low_scores_are_preserved_and_cannot_be_accepted_as_good_road():
    arrays,points,camera,predictor=_prompt_fixture()
    original=predictor.predict
    predictor.predict=lambda **kw: (lambda value:(value[0],np.array([1e-8]),value[2]))(original(**kw))
    result,report=observations.candidate_road_from_original_context(arrays,points,camera,
        np.ones((80,80),bool),predictor,.03,replace(ObservationConfig(),max_seed_prompts=1))
    assert not result.any()
    option=report['promptRecords'][0]['options'][0]
    assert option['score']==1e-8 and 'low-model-mask-quality' in option['rejectionReasons']


def test_unthresholded_sam_logits_cannot_be_cast_to_all_true_road():
    arrays,points,camera,predictor=_prompt_fixture()
    predictor.predict=lambda **kw:(np.full((1,80,80),-.25),np.array([.9]),np.zeros((1,10,10)))
    with pytest.raises(ValueError,match='binary mask values'):
        observations.candidate_road_from_original_context(arrays,points,camera,
            np.ones((80,80),bool),predictor,.03,replace(ObservationConfig(),max_seed_prompts=1))


def test_static_appearance_preserves_yellow_paint_but_excludes_red_flag():
    rgb=np.array([[[100,102,101],[220,200,90],[200,30,30],[210,210,210]]],np.uint8)
    assert observations.static_road_appearance(rgb).tolist()==[[True,True,False,True]]


def test_pair_geometry_protects_possible_shadow_without_all_time_body_union():
    shape=(400,400);road=np.ones(shape,bool);blank=np.zeros(shape,bool);image=np.full((*shape,3),100,np.uint8)
    a={'image':image,'valid':road,'body_mask':blank.copy()};b={'image':image,'valid':road,'body_mask':blank.copy()}
    a['body_mask'][10:20,10:20]=True
    support=observations.pair_geometry_support(a,b,road,road,.03)
    assert not support[10,100] # 2.4m near the actual current vehicle
    historical_other_frame=blank.copy();historical_other_frame[300:320,300:320]=True
    assert support[historical_other_frame].all() # unrelated past position is not a static exclusion


def test_measurement_shadow_exclusion_is_not_a_permanent_output_collar_ban():
    shape=(400,400);road=np.ones(shape,bool);blank=np.zeros(shape,bool);image=np.full((*shape,3),100,np.uint8)
    body=blank.copy();body[180:190,180:190]=True
    arrays={'image':image,'valid':road,'body_mask':body}
    measured=observations.pair_geometry_support(arrays,arrays,road&~body,road&~body,.03)
    assert not measured[180,201]
    sources={name:{'arrays':arrays,'record':{'cacheSha256':name*64},'road':road&~body,'shadow':blank} for name in ['A','C']}
    pair={'residual':np.zeros(shape),'support':measured,'shadow':blank,'unknown':blank,'geometryAccepted':True,'colorComparisonAccepted':True}
    counts,static,unknown,refs,archive=observations.directional_clear_support('A',sources,{('A','C'):pair,('C','A'):pair},.03)
    assert counts[180,201]>0 and static[180,201] and not unknown[180,201]


def test_observed_dark_shadow_guard_keeps_bright_road_and_excludes_actual_dark_pixels():
    shape=(400,400);road=np.ones(shape,bool);body=np.zeros(shape,bool);body[180:190,180:190]=True
    image=np.full((*shape,3),100,np.uint8);image[170:185,190:235]=30
    source={'image':image,'valid':road,'body_mask':body}
    config=replace(ObservationConfig(),geometry_shadow_guard_mode='observed-dark-near-body')
    support=observations.pair_geometry_support(source,source,road&~body,road&~body,.03,config)
    assert not support[175,225] # photographically dark suspected shadow
    assert not support[175,240] # conservative margin of the actual dark region
    assert support[230,200] # bright road need not be forbidden by a 4m circle


def test_empty_body_has_no_invented_dark_shadow_guard():
    source={'image':np.full((100,100,3),10,np.uint8),'valid':np.ones((100,100),bool),'body_mask':np.zeros((100,100),bool)}
    assert not observations.possible_dark_shadow_guard(source,np.ones((100,100),bool),.03).any()
