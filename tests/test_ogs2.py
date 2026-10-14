# Optional algorithm dependencies are absent from the base CI environment.
# ruff: noqa: E402
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
rio = pytest.importorskip("rasterio")
pytest.importorskip("scipy")
from model_engine.ogs2 import (
    Config,
    _rasterize,
    cameras_from_checkpoint,
    group_masks,
    lock_objects,
    reliable_mask,
    run,
    surface_points,
    visibility_mask,
)
from rasterio.transform import from_origin
from rasterio.windows import Window


def test_completed_output_cannot_be_overwritten(tmp_path):
    project = tmp_path / "project"
    output = tmp_path / "result"
    project.mkdir()
    output.mkdir()
    result = output / "ogs2_orthophoto.tif"
    result.write_bytes(b"preserved experiment")
    with pytest.raises(FileExistsError):
        run(project, output, tmp_path / "models", Config(), "cpu")
    assert result.read_bytes() == b"preserved experiment"


def test_weak_or_leaked_segment_is_rejected_without_clipping():
    vehicle = np.zeros((20, 20), bool)
    vehicle[5:10, 5:10] = True
    box = [4, 4, 11, 11]
    assert reliable_mask(vehicle, box, 0.95, Config())
    assert not reliable_mask(vehicle, box, 0.5, Config())
    leaked = np.ones((20, 20), bool)
    assert not reliable_mask(leaked, box, 0.95, Config())
    assert leaked.all()


def test_tiny_object_cannot_lock_a_large_ambiguous_background_group():
    region = np.ones((20, 20), bool)
    core = np.zeros_like(region)
    core[5:10, 5:10] = True
    labels = np.zeros_like(region, dtype=np.int16)
    records = lock_objects(
        labels,
        [region],
        np.zeros((1, 20, 20)),
        np.ones((1, 20, 20), bool),
        [{"view": 0, "complete": True, "mask": core}],
        Config(),
    )
    assert records[0]["state"] == "baseline-fallback"
    assert records[0]["rejections"]["ambiguousGroup"] == 1
    assert np.all(labels == -1)


def test_whole_object_uses_one_observed_photo_not_cheaper_background():
    labels = np.zeros((10, 10), dtype=np.int16)
    region = np.zeros_like(labels, dtype=bool)
    region[2:8, 2:8] = True
    costs = np.zeros((2, 10, 10), np.float32)
    costs[1] = 1
    valid = np.ones_like(costs, dtype=bool)
    obs = [{"view": 1, "complete": True, "mask": region}]
    record = lock_objects(labels, [region], costs, valid, obs, Config())
    assert np.all(labels[region] == 1)
    assert record[0]["state"] == "locked"


@pytest.mark.parametrize("invalid,complete", [(True, True), (False, False)])
def test_incomplete_or_occluded_source_reverts_whole_object(invalid, complete):
    region = np.ones((8, 8), bool)
    labels = np.zeros((8, 8), np.int16)
    valid = np.ones((1, 8, 8), bool)
    if invalid:
        valid[0, 3, 3] = False
    records = lock_objects(
        labels,
        [region],
        np.zeros((1, 8, 8)),
        valid,
        [{"view": 0, "complete": complete, "mask": region}],
        Config(),
    )
    assert np.all(labels == -1)
    assert records[0]["state"] == "baseline-fallback"


def test_unobserved_object_never_counts_as_restored():
    labels = np.zeros((8, 8), np.int16)
    region = np.ones((8, 8), bool)
    records = lock_objects(labels, [region], np.zeros((1, 8, 8)), np.ones((1, 8, 8), bool), [], Config())
    assert records[0]["state"] == "baseline-fallback"
    assert np.all(labels == -1)


def test_unknown_padding_does_not_reject_a_fully_visible_vehicle():
    region = np.ones((10, 10), bool)
    core = np.zeros((10, 10), bool)
    core[2:8, 2:8] = True
    valid = np.ones((1, 10, 10), bool)
    valid[0, 0, 0] = False
    labels = np.zeros((10, 10), np.int16)
    records = lock_objects(
        labels,
        [region],
        np.zeros((1, 10, 10)),
        valid,
        [{"view": 0, "complete": True, "mask": core}],
        Config(),
    )
    assert records[0]["state"] == "locked"
    assert records[0]["objectCorePixels"] == 36
    assert labels[0, 0] == -1
    assert np.all(labels[core] == 0)


def test_surface_keeps_holes_and_topmost_overlap():
    vertices = np.array(
        [[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 4.0], [3.0, 0.0, 4.0], [0.0, 3.0, 4.0]]
    )
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    surface = _rasterize(vertices, faces, 5, 5)
    assert surface[1, 1] == 4
    assert not np.isfinite(surface[4, 4])


def test_visibility_rejects_intervening_wall_and_unknown_surface():
    surface = np.zeros((20, 20), np.float32)
    transform = from_origin(0, 20, 1, 1)
    target = np.array([[[2.5]], [[10.5]], [[0.0]]])
    camera = (15.5, 10.5, 5.0)
    assert visibility_mask(target, camera, surface, transform, 0.1).all()
    surface[:, 8:10] = 8
    assert not visibility_mask(target, camera, surface, transform, 0.1).any()
    surface[:, 8:10] = np.nan
    assert not visibility_mask(target, camera, surface, transform, 0.1).any()


def test_grid_sampling_uses_pixel_centers():
    surface = np.arange(25, dtype=np.float32).reshape(5, 5)
    transform = from_origin(100, 200, 1, 1)
    points = surface_points(surface, transform, Window(1, 1, 2, 2), transform)
    np.testing.assert_allclose(points[2], surface[1:3, 1:3])
    assert points[0, 0, 0] == 101.5
    assert points[1, 0, 0] == 198.5


def test_overlapping_observations_form_indivisible_region():
    a, b = np.zeros((20, 20), bool), np.zeros((20, 20), bool)
    a[4:10, 4:10] = True
    b[5:11, 8:14] = True
    regions = group_masks([a, b], 1)
    assert len(regions) == 1
    assert np.all(regions[0][a | b])


def test_opensfm_projection_and_offset_match_actual_camera(tmp_path: Path):
    for directory in ["opensfm/undistorted/images", "odm_georeferencing"]:
        (tmp_path / directory).mkdir(parents=True)
    offset = np.array([191718.0, 2494779.0, 0.0])
    (tmp_path / "odm_georeferencing/odm_georeferencing_model_geo.txt").write_text(
        "WGS84 UTM 50N\n191718 2494779\n"
    )
    rotation = np.array([np.pi, 0.0, 0.0])
    translation = [0.0, 0.0, 100.0]
    camera = {
        "projection_type": "perspective",
        "width": 100,
        "height": 80,
        "focal": 0.8,
        "k1": 0.0,
        "k2": 0.0,
    }
    data = [
        {
            "cameras": {"cam": camera},
            "shots": {
                "image.tif": {"rotation": rotation.tolist(), "translation": translation, "camera": "cam"}
            },
            "reference_lla": {
                "latitude": 22.53176228711584,
                "longitude": 114.00304142889469,
                "altitude": 0.0,
            },
        }
    ]
    (tmp_path / "opensfm/undistorted/reconstruction.json").write_text(json.dumps(data))
    with rio.open(
        tmp_path / "opensfm/undistorted/images/image.tif",
        "w",
        driver="GTiff",
        width=100,
        height=80,
        count=3,
        dtype="uint8",
    ) as dst:
        dst.write(np.zeros((3, 80, 100), np.uint8))
    factory, _ = cameras_from_checkpoint(tmp_path, rio.CRS.from_epsg(32650))
    points = np.array([[0.0, 1.0, -2.0], [0.0, 3.0, -2.0], [0.0, 2.0, 4.0]])
    r = cv2.Rodrigues(rotation)[0]
    camera_coordinates = r @ points + np.array(translation)[:, None]
    expected = camera_coordinates[:2] / camera_coordinates[2] * 80 + np.array([49.5, 39.5])[:, None]
    np.testing.assert_allclose(
        factory.get("image.tif").world_to_pixel(points + offset[:, None]), expected, atol=1e-7
    )
