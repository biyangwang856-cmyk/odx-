"""Counterexamples for independent, bounded road-plane reconstruction."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from model_engine.ogs2_road_surface import RoadSurfaceConfig, fit_road_surface


PROVENANCE = {'reconstruction_sha256': 'a'*64,
              'image_coordinate_frame': 'undistorted-photo', 'source': 'opensfm'}
BASE = np.array([500000., 2600000.])


def scene(*, candidates=False, curve=0.0):
    def camera(center):
        def project(xyz):
            xyz = np.asarray(xyz)
            return (xyz[:, :2]-center[:2])*1000/(center[2]-xyz[:, 2, None])+[2000, 1500]
        return project
    projectors = {'photo-a': camera(np.r_[BASE+[0, 0], 140]),
                  'photo-b': camera(np.r_[BASE+[25, 10], 140])}
    points = []
    for iy in range(12):
        for ix in range(16):
            x, y = 1.+ix*2, 1.+iy*2
            xyz = np.r_[BASE+[x, y], 40.+x*.01+y*.005+curve*(x-16)**2]
            obs = [{'image_id': name, 'pixel_xy': projector(xyz[None])[0].tolist(),
                    'road_support': 'candidate' if candidates else 'reviewed',
                    'dynamic': False, 'evidence_id': name+'-raw-road-review-v1'}
                   for name, projector in projectors.items()]
            points.append({'track_id': f'{iy:02}-{ix:02}', 'xyz': xyz.tolist(),
                           'origin': 'sfm-multiview-triangulation', 'observations': obs})
    return points, projectors


def fit(points, projectors, config=RoadSurfaceConfig()):
    return fit_road_surface(points, projectors, provenance=PROVENANCE, config=config)


def heldout_indices(points, config=RoadSurfaceConfig()):
    xy = np.array([p['xyz'][:2] for p in points])
    cells = np.floor((xy-config.split_origin_xy)/config.split_cell_size_m).astype(int)
    return (cells[:, 0]+2*cells[:, 1]) % 4 == 0


def test_calibrated_independent_plane_preserves_metre_coordinates_and_raw_measurements():
    points, projectors = scene()
    result = fit(points, projectors)
    assert result.report['accepted'], result.report
    assert result.coefficients[:2] == pytest.approx([.01, .005], abs=1e-10)
    heights = result.evaluate(BASE+[[10, 10], [20, 15]])
    assert heights == pytest.approx([40.15, 40.275], abs=1e-9)
    assert result.report['holdoutHeightMaxM'] < 1e-9
    assert result.report['holdoutReprojectionMaxPx'] < 1e-8
    assert len(result.report['holdout']) >= 6
    assert all(len(p['observations']) == 2 for p in result.report['holdout'])


def test_automatic_masks_do_not_become_semantic_ground_truth():
    points, projectors = scene(candidates=True)
    result = fit(points, projectors)
    assert result.report['geometryAccepted']
    assert not result.report['accepted']
    assert 'road-semantics-remain-candidate' in result.report['reasons']
    with pytest.raises(ValueError, match='candidate use must be explicit'):
        result.evaluate(BASE+[[10, 10]])
    assert np.isfinite(result.evaluate(BASE+[[10, 10]], allow_candidate=True)).all()


def test_dsm_samples_cannot_masquerade_as_independent_geometry():
    points, projectors = scene()
    points[0]['origin'] = 'dsm-sampled'
    with pytest.raises(ValueError, match='not independent'):
        fit(points, projectors)


def test_duplicate_photograph_cannot_supply_two_independent_views():
    points, projectors = scene()
    points[0]['observations'][1] = deepcopy(points[0]['observations'][0])
    with pytest.raises(ValueError, match='one observation per photograph'):
        fit(points, projectors)


def test_dynamic_or_unknown_road_observations_are_not_ground_support():
    points, projectors = scene()
    for p in points:
        p['observations'][0]['dynamic'] = True
        p['observations'][1]['road_support'] = 'unknown'
    result = fit(points, projectors)
    assert not result.report['accepted']
    assert result.report['reasons'] == ['no-qualified-independent-road-points']


def test_holdout_outlier_is_not_trimmed_or_used_to_rescue_plane():
    points, projectors = scene()
    original = fit(points, projectors)
    indices = np.flatnonzero(heldout_indices(points))
    points[indices[0]]['xyz'][2] += 3.
    changed = fit(points, projectors)
    assert changed.coefficients == pytest.approx(original.coefficients, abs=1e-12)
    assert not changed.report['accepted']
    assert changed.report['holdoutHeightMaxM'] > 2.9
    assert changed.report['holdoutOutliersRemoved'] is False
    assert changed.report['holdoutUsedForModelSelection'] is False
    assert len(changed.report['holdout']) == len(original.report['holdout'])
    with pytest.raises(ValueError):
        changed.evaluate(BASE+[[10, 10]], allow_candidate=True)


def test_training_outliers_can_be_robustly_fit_without_hiding_holdout():
    points, projectors = scene()
    train = np.flatnonzero(~heldout_indices(points))
    for index in train[::5]:
        points[index]['xyz'][2] += 1.7
    result = fit(points, projectors)
    assert result.report['accepted'], result.report['reasons']
    assert .75 < result.report['fitInlierFraction'] < .9
    assert result.report['holdoutHeightMaxM'] < 1e-8
    assert sum(not p['inlier'] for p in result.report['fit']) > 0


def test_curved_road_is_not_silently_flattened():
    points, projectors = scene(curve=.015)
    result = fit(points, projectors)
    assert not result.report['geometryAccepted']
    assert any(r in result.report['reasons'] for r in
               ('insufficient-training-plane-consensus', 'independent-height-holdout-failed'))


def test_correct_heights_with_wrong_raw_camera_observations_fail():
    points, projectors = scene()
    for p in points:
        p['observations'][1]['pixel_xy'][0] += 4.
    result = fit(points, projectors)
    assert not result.report['geometryAccepted']
    assert result.report['holdoutHeightMaxM'] < 1e-8
    assert 'independent-photo-reprojection-holdout-failed' in result.report['reasons']
    assert 'original-sfm-photo-reprojection-failed' in result.report['reasons']


def test_no_extrapolation_outside_training_hull_or_across_unobserved_gap():
    points, projectors = scene()
    result = fit(points, projectors, replace(RoadSurfaceConfig(), max_support_distance_m=.25))
    assert result.report['accepted']
    support_point = result.support_xy[len(result.support_xy)//2]
    values = result.evaluate(np.array([support_point, BASE+[-1, 10], BASE+[10, 10]]))
    assert np.isfinite(values[0])
    assert np.isnan(values[1:]).all()


def test_collinear_lane_marks_cannot_constrain_a_two_dimensional_surface():
    points, projectors = scene()
    points = [p for p in points if p['xyz'][1] == BASE[1]+1]
    result = fit(points, projectors, replace(RoadSurfaceConfig(), min_fit_points=6, min_holdout_points=3,
                                            min_fit_cells=2, min_holdout_cells=1))
    assert not result.report['accepted']
    assert any(reason in result.report['reasons'] for reason in
               ('insufficient-two-dimensional-road-support', 'insufficient-preassigned-fit-or-holdout-points'))


def test_duplicate_xyz_tracks_do_not_inflate_independent_support():
    points, projectors = scene()
    duplicate = deepcopy(points[0])
    duplicate['track_id'] = 'zz-duplicate'
    result = fit(points+[duplicate], projectors)
    assert result.report['accepted']
    assert result.report['excluded'] == [{'trackId': 'zz-duplicate', 'reason': 'spatially-duplicate-track'}]


def test_reordered_points_have_identical_split_and_fit():
    points, projectors = scene()
    first, second = fit(points, projectors), fit(points[::-1], projectors)
    assert first.coefficients == pytest.approx(second.coefficients, abs=1e-12)
    assert first.report['holdout'] == second.report['holdout']


def test_original_reconstruction_and_coordinate_frame_are_required():
    points, projectors = scene()
    with pytest.raises(ValueError, match='SHA256'):
        fit_road_surface(points, projectors, provenance={})
    with pytest.raises(ValueError, match='coordinate frame'):
        fit_road_surface(points, projectors, provenance={'reconstruction_sha256': 'a'*64})


def test_nonfinite_coordinates_and_duplicate_track_ids_are_rejected():
    points, projectors = scene()
    with pytest.raises(ValueError, match='unique'):
        fit(points+[deepcopy(points[0])], projectors)
    points[0]['xyz'][2] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        fit(points, projectors)
