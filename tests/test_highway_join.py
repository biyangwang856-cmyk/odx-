import numpy as np
import pytest
from model_engine.ogs2_highway_join import (
    cut_edges,
    preserve_boundary_foreground,
    sample_cached_body,
    select_highway_join,
)


def image(h, w):
    return np.full((h, w, 3), 100, np.uint8)


def test_large_horizontal_shift_through_clear_divider():
    rgb = image(8, 30)
    protected = np.ones((8, 30), bool)
    protected[:3, 2:5] = False
    protected[3:5] = False
    protected[5:, 23:26] = False
    labels = select_highway_join(rgb, rgb, np.ones_like(protected), protected)
    assert labels is not None
    assert not np.any(cut_edges(labels) & protected)
    assert labels[0].sum() > labels[-1].sum() + 15


def test_complete_protected_barrier_refuses_join():
    rgb = image(8, 30)
    protected = np.zeros((8, 30), bool)
    protected[4] = True
    assert select_highway_join(rgb, rgb, ~np.zeros_like(protected), protected) is None


def test_both_cut_endpoints_protected():
    rgb = image(12, 20)
    protected = np.zeros((12, 20), bool)
    protected[2:9, 8:12] = True
    labels = select_highway_join(rgb, rgb, np.ones_like(protected), protected)
    assert labels is not None
    assert not np.any(cut_edges(labels) & protected)
    assert np.all(labels[:, 0] == 0) and np.all(labels[:, -1] == 1)


def test_invalid_full_row_refuses_join():
    rgb = image(8, 30)
    valid = np.ones((8, 30), bool)
    valid[4] = False
    assert select_highway_join(rgb, rgb, valid, np.zeros_like(valid)) is None


def test_minimum_two_column_partition():
    rgb = image(8, 2)
    labels = select_highway_join(rgb, rgb, np.ones((8, 2), bool), np.zeros((8, 2), bool))
    assert np.all(labels[:, 0] == 0) and np.all(labels[:, 1] == 1)


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        select_highway_join(image(4, 8), image(4, 9), np.ones((4, 8)), np.zeros((4, 8)))


def test_soft_temporal_penalty_does_not_split_a_detected_vehicle():
    rgb = image(12, 20)
    protected = np.zeros((12, 20), bool)
    protected[2:9, 8:12] = True
    penalty = np.zeros_like(protected, dtype=np.float32)
    penalty[:, :5] = 100
    labels = select_highway_join(rgb, rgb, np.ones_like(protected), protected, penalty=penalty)
    assert labels is not None and not np.any(cut_edges(labels) & protected)
    assert not np.any(cut_edges(labels)[:, :5])


def test_exterior_boundary_includes_a_whole_observed_road_vehicle():
    replacement = np.zeros((30, 30), bool)
    replacement[:15] = True
    body = np.zeros_like(replacement)
    body[12:18, 10:20] = True
    result, report = preserve_boundary_foreground(
        replacement, body, np.ones_like(body), np.ones_like(body), np.ones_like(body), padding_pixels=2
    )
    assert np.all(result[body]) and report["detectedBodyCutEndpoints"] == 0


def test_unobserved_or_out_of_scope_vehicle_is_kept_whole_original():
    replacement = np.zeros((30, 30), bool)
    replacement[:15] = True
    body = np.zeros_like(replacement)
    body[12:18, 10:20] = True
    valid = np.ones_like(body)
    valid[17, 12] = False
    result, report = preserve_boundary_foreground(
        replacement, body, valid, np.ones_like(body), np.ones_like(body), padding_pixels=2
    )
    assert not np.any(result[body]) and report["detectedBodyCutEndpoints"] == 0


def test_missing_mask_context_is_not_a_detected_vehicle():
    guard = np.zeros((5, 5), bool)
    guard[2, 2] = True
    uv = np.array([[[-1.0, 1.5, 0.0]], [[2.0, 1.5, 0.0]]], np.float32)
    body, unknown = sample_cached_body(guard, uv)
    assert unknown.tolist() == [[True, False, False]]
    assert body.tolist() == [[False, True, False]]
