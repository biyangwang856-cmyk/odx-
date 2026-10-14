"""Clock-domain counterexamples for vehicle association."""

# Optional CPU algorithm dependencies are separate from the base environment.
# ruff: noqa: E402
import numpy as np
import pytest

pytest.importorskip("scipy")
from model_engine.ogs2_temporal import TemporalConfig, associate_observations


def observation(view, stamp, x, **changes):
    item = {
        "view": view,
        "capture_time": stamp,
        "mask": np.ones((10, 20), dtype=bool),
        "origin": (x, 0),
        "complete": True,
        "confidence": 0.95,
        "appearance": np.array([1.0, 0.0, 0.0]),
    }
    item.update(changes)
    return item


def test_epoch_looking_opensfm_and_dji_clocks_never_mix():
    # Identical numeric bases can occur despite different timezone semantics.
    items = [
        observation(1, 1_790_000_000, 0, clock="opensfm-capture-time"),
        observation(2, 1_790_000_001, 30, clock="dji-filename-wall-clock"),
    ]
    groups, diagnostic = associate_observations(items, 0.1)
    assert groups == []
    assert diagnostic["candidate_pairs"] == 0
    assert diagnostic["clock_domains"] == {
        "dji-filename-wall-clock": 1,
        "opensfm-capture-time": 1,
    }


@pytest.mark.parametrize("clock", [None, "", "unknown", " UNKNOWN ", 123])
def test_explicit_unknown_clock_is_never_legacy_compatible(clock):
    items = [observation(1, 0, 0, clock=clock), observation(2, 1, 30, clock=clock)]
    groups, diagnostic = associate_observations(items, 0.1)
    assert groups == []
    assert diagnostic["rejected"]["unknown_clock"] == 2


def test_old_synthetic_inputs_share_an_explicit_legacy_domain_by_default():
    groups, diagnostic = associate_observations([observation(1, 0, 0), observation(2, 1, 30)], 0.1)
    assert groups == [[0, 1]]
    assert diagnostic["groups"][0]["clock"] == "legacy-test-clock"


def test_strict_production_configuration_rejects_missing_domain():
    groups, diagnostic = associate_observations(
        [observation(1, 0, 0), observation(2, 1, 30)],
        0.1,
        TemporalConfig(require_explicit_clock=True),
    )
    assert groups == []
    assert diagnostic["rejected"]["missing_clock"] == 2


def test_source_image_records_cannot_silently_fall_back_to_legacy_clock():
    items = [
        observation(1, 0, 0, sourceImage="DJI_A.JPG"),
        observation(2, 1, 30, sourceImage="DJI_B.JPG"),
    ]
    groups, diagnostic = associate_observations(items, 0.1)
    assert groups == []
    assert diagnostic["rejected"]["missing_clock"] == 2


def test_cross_domain_nearer_candidate_cannot_hide_valid_same_domain_match():
    items = [
        observation(1, 0, 0, clock="opensfm-capture-time"),
        observation(2, 0.5, 15, clock="dji-filename-wall-clock"),
        observation(3, 1, 30, clock="opensfm-capture-time"),
    ]
    groups, diagnostic = associate_observations(items, 0.1, TemporalConfig(require_explicit_clock=True))
    assert groups == [[0, 2]]
    assert diagnostic["groups"][0]["clock"] == "opensfm-capture-time"


def test_independent_domains_can_associate_without_numeric_time_interference():
    items = [
        observation(1, 0, 0, clock="opensfm-capture-time"),
        observation(2, 1, 30, clock="opensfm-capture-time"),
        observation(3, 0, 300, clock="dji-filename-wall-clock"),
        observation(4, 1, 330, clock="dji-filename-wall-clock"),
    ]
    groups, _ = associate_observations(items, 0.1)
    assert sorted(groups) == [[0, 1], [2, 3]]
