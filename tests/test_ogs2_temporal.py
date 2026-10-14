"""Counterexamples for local temporal association, not identity guarantees."""

# Optional CPU algorithm dependencies are separate from the base environment.
# ruff: noqa: E402

import numpy as np
import pytest

pytest.importorskip("scipy")
from model_engine.ogs2_temporal import TemporalConfig, associate_observations, render_conflict


def observation(view, capture_time, origin, appearance=(1.0, 0.0, 0.0), **changes):
    result = {
        "view": view,
        "capture_time": capture_time,
        "mask": np.ones((10, 20), dtype=bool),
        "origin": origin,
        "complete": True,
        "confidence": 0.95,
        "appearance": np.asarray(appearance, dtype=float),
        "category": "bus",
    }
    result.update(changes)
    return result


def test_two_nonoverlapping_observations_of_moving_vehicle_form_one_group():
    obs = [observation(1, 0, (100, 200)), observation(2, 1, (130, 200))]
    groups, diagnostic = associate_observations(obs, pixel_size_m=0.1)
    assert groups == [[0, 1]]
    assert diagnostic["groups"][0]["maximum_step_m"] == pytest.approx(3.0)
    assert diagnostic["groups"][0]["dynamic_confirmed"] is True


def test_same_view_similar_neighbouring_vehicles_do_not_merge():
    obs = [observation(1, 0, (0, 0)), observation(1, 0, (22, 0))]
    groups, diagnostic = associate_observations(obs, 0.1)
    assert groups == []
    assert diagnostic["retained_observations"] == 2


def test_long_time_gap_is_never_linked():
    groups, _ = associate_observations(
        [observation(1, 0, (0, 0)), observation(2, 3.01, (25, 0))], 0.1
    )
    assert groups == []


def test_parked_vehicle_is_not_a_dynamic_conflict():
    groups, diagnostic = associate_observations(
        [observation(1, 0, (0, 0)), observation(2, 1, (2, 0))], 0.1
    )
    assert groups == []
    assert diagnostic["rejected"]["static_group"] == 1


def test_ambiguous_same_colour_neighbours_are_withheld():
    obs = [
        observation(1, 0, (0, 0)),
        observation(2, 1, (25, 0)),
        observation(2, 1, (25, 15)),
    ]
    groups, diagnostic = associate_observations(obs, 0.1)
    assert groups == []
    assert diagnostic["rejected"]["ambiguous_direction"] > 0


def test_missing_or_nonfinite_time_never_proposes_motion():
    obs = [observation(1, None, (0, 0)), observation(2, np.nan, (25, 0))]
    groups, diagnostic = associate_observations(obs, 0.1)
    assert groups == []
    assert diagnostic["rejected"]["missing_time"] == 2


def test_motion_gate_uses_real_elapsed_seconds_and_metres():
    obs = [observation(1, 0, (0, 0)), observation(2, 0.1, (50, 0))]
    config = TemporalConfig(max_speed_mps=5)
    assert associate_observations(obs, 0.1, config)[0] == []  # 5 m in 0.1 s.
    obs[1]["capture_time"] = 1.0
    assert associate_observations(obs, 0.1, config)[0] == [[0, 1]]
    assert associate_observations(obs, 1.0, config)[0] == []  # 50 m in 1 s.


def test_duplicate_same_view_across_tiles_keeps_original_best_index():
    first = observation(1, 0, (100, 100))
    # Same pixels, expressed in a differently padded local window.
    padded = np.pad(first["mask"], ((3, 2), (4, 1)))
    obs = [
        first,
        observation(1, 0, (96, 97), mask=padded, confidence=0.9),
        observation(2, 1, (130, 100)),
    ]
    groups, diagnostic = associate_observations(obs, 0.1)
    assert groups == [[0, 2]]
    assert diagnostic["duplicates"] == [{"discarded": 1, "retained": 0}]


def test_static_candidate_blocks_association_to_moving_lookalike():
    obs = [
        observation(1, 0, (0, 0)),
        observation(2, 1, (0, 0)),
        observation(2, 1, (30, 0)),
    ]
    assert associate_observations(obs, 0.1)[0] == []


def test_one_observation_cannot_associate_with_two_targets():
    obs = [
        observation(1, 0, (0, 0)),
        observation(2, 1, (30, 0)),
        observation(2, 1, (35, 0), appearance=(0, 1, 0)),
    ]
    groups, _ = associate_observations(obs, 0.1)
    assert groups == [[0, 1]]


@pytest.mark.parametrize("change", [{"confidence": 0.5}, {"complete": False}, {"appearance": None}])
def test_untrusted_observation_is_not_used(change):
    obs = [observation(1, 0, (0, 0)), observation(2, 1, (30, 0), **change)]
    assert associate_observations(obs, 0.1)[0] == []


def test_conflict_mask_stays_local_and_connects_short_trajectory():
    obs = [observation(1, 0, (14000, 13000)), observation(2, 1, (14030, 13000))]
    mask, origin = render_conflict([0, 1], obs, padding_pixels=3)
    assert origin == (13997, 12997)
    assert mask.shape == (16, 56)
    assert mask[8, 28]  # Swept region between disjoint object masks.
    assert mask.size < 1000


def test_huge_conflict_envelope_fails_instead_of_allocating_global_mask():
    obs = [observation(1, 0, (0, 0)), observation(2, 1, (14000, 13000))]
    with pytest.raises(ValueError, match="pixel budget"):
        render_conflict([0, 1], obs)


def test_group_limit_withholds_overlarge_association():
    obs = [observation(1, 0, (0, 0)), observation(2, 1, (30, 0))]
    groups, diagnostic = associate_observations(obs, 0.1, TemporalConfig(max_group_diameter_m=2))
    assert groups == []
    assert diagnostic["rejected"]["group_diameter"] == 1


def test_invalid_scale_is_rejected():
    with pytest.raises(ValueError, match="pixel_size_m"):
        associate_observations([], 0)


def test_three_views_of_same_vehicle_are_not_false_identity_ambiguity():
    obs = [observation(view, view, (view * 30, 0)) for view in range(3)]
    groups, _ = associate_observations(obs, 0.1)
    assert groups == [[0, 1, 2]]


def test_transitive_edges_cannot_create_group_over_three_second_span():
    obs = [observation(view, view * 2, (view * 30, 0)) for view in range(3)]
    groups, diagnostic = associate_observations(obs, 0.1)
    assert groups == [[0, 1]]
    assert diagnostic["rejected"]["group_span"] == 1


def test_nonfinite_confidence_duplicate_does_not_discard_good_observation():
    obs = [
        observation(1, 0, (0, 0), confidence=np.inf),
        observation(1, 0, (0, 0)),
        observation(2, 1, (30, 0)),
    ]
    groups, diagnostic = associate_observations(obs, 0.1)
    assert groups == [[1, 2]]
    assert diagnostic["duplicates"] == [{"discarded": 0, "retained": 1}]
