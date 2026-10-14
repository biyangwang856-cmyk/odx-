"""CPU-only raster contracts for opt-in temporal patches.

Inference and association are injected to exercise the real output writer; no
GPU, downloaded weights, or laboratory input data are required.
"""

# Optional algorithm dependencies are absent from the base project environment.
# ruff: noqa: E402
import json

import numpy as np
import pytest

pytest.importorskip("cv2")
rio = pytest.importorskip("rasterio")
pytest.importorskip("scipy")
from rasterio.transform import from_origin

from model_engine import ogs2_deghost as engine
from model_engine.ogs2_temporal import associate_observations


@pytest.fixture
def synthetic_checkpoint(tmp_path, monkeypatch):
    project = tmp_path / "preserved-project"
    for relative in ("odm_orthophoto", "odm_meshing", "opensfm/undistorted/images"):
        (project / relative).mkdir(parents=True, exist_ok=True)
    names = ["DJI_20260925164213_0064_V.JPG.tif", "DJI_20260925164214_0065_V.JPG.tif"]
    (project / "odm_meshing/odm_25dmesh.ply").write_bytes(b"preserved mesh")
    (project / "opensfm/undistorted/reconstruction.json").write_text("{}")
    for name in names:
        (project / "opensfm/undistorted/images" / name).write_bytes(b"injected photo")
    baseline = np.empty((4, 64, 64), np.uint8)
    rows, columns = np.indices((64, 64))
    baseline[0], baseline[1], baseline[2], baseline[3] = rows, columns, 17, 255
    baseline[:, 0, 0] = 0  # Alpha and uncovered RGB must remain exact.
    profile = dict(
        driver="GTiff", width=64, height=64, count=4, dtype="uint8",
        crs="EPSG:32650", transform=from_origin(100, 200, 0.1, 0.1),
    )
    baseline_file = project / "odm_orthophoto/odm_orthophoto.tif"
    with rio.open(baseline_file, "w", **profile) as dst:
        dst.write(baseline)
        dst.colorinterp = (
            rio.enums.ColorInterp.red, rio.enums.ColorInterp.green,
            rio.enums.ColorInterp.blue, rio.enums.ColorInterp.alpha,
        )
    input_hash = engine.sha256(baseline_file)

    def export_surface(project, grid, output, scale):
        del project, grid, scale
        with rio.open(output, "w", **{**profile, "count": 1, "dtype": "float32"}) as dst:
            dst.write(np.zeros((64, 64), np.float32), 1)
        return {"synthetic": True}

    class Cameras:
        def get(self, name):
            return name

    data = {"shots": {name: {"capture_time": 100 + index} for index, name in enumerate(names)}}
    monkeypatch.setattr(engine, "export_surface", export_surface)
    monkeypatch.setattr(engine, "cameras_from_checkpoint", lambda *_: (Cameras(), data))
    observations = [
        {
            "view": index + 1, "sourceImage": name, "capture_time": 100 + index,
            "kind": "photo",
            "clock": "opensfm-capture-time", "origin": (20 + index * 12, 24),
            "mask": np.ones((6, 10), bool), "complete": True, "confidence": 0.95,
            "samConfidence": 0.95,
            "appearance": np.asarray([1.0, 0.0]),
        }
        for index, name in enumerate(names)
    ]
    monkeypatch.setattr(engine, "observation_cache", lambda *_: (observations, False))
    monkeypatch.setattr(engine, "associate_observations", lambda *_: ([[0, 1]], {}))
    monkeypatch.setattr(
        engine, "static_agreement", lambda *_: {"accepted": True, "reason": "injected-static-pass"}
    )
    source_colors = {names[0]: np.asarray([73, 109, 151], np.uint8), names[1]: np.asarray([81, 117, 159], np.uint8)}

    def project_view(project, name, camera, points, *_):
        del project, camera
        shape = points.shape[1:]
        image = np.broadcast_to(source_colors[name], (*shape, 3)).copy()
        valid = np.ones(shape, bool)
        cost = np.full(shape, names.index(name), np.float32)
        return image, valid, cost, []

    monkeypatch.setattr(engine, "project_view", project_view)
    models = tmp_path / "models"
    models.mkdir()
    (models / "provenance.json").write_text("{}")
    return {
        "project": project, "baseline": baseline, "baseline_file": baseline_file,
        "baseline_hash": input_hash, "models": models, "profile": profile,
        "source_colors": source_colors, "names": names, "observations": observations,
        "project_view": project_view,
    }


def execute(checkpoint, output, **options):
    engine.run(
        checkpoint["project"], output, checkpoint["models"],
        engine.PatchConfig(padding_m=0.2), "cpu", **options,
    )
    with rio.open(output / "ogs2_orthophoto.tif") as src:
        image, profile = src.read(), src.profile
    with rio.open(output / "source-index.tif") as src:
        source = src.read(1)
    with rio.open(output / "protected-objects.tif") as src:
        accepted = src.read(1).astype(bool)
    return image, source, accepted, profile


def test_accepted_patch_is_one_real_source_and_every_other_pixel_is_exact_baseline(
    synthetic_checkpoint, tmp_path,
):
    checkpoint = synthetic_checkpoint
    output = tmp_path / "accepted"
    image, source, accepted, profile = execute(checkpoint, output)
    assert accepted.any()
    np.testing.assert_array_equal(image[:, ~accepted], checkpoint["baseline"][:, ~accepted])
    np.testing.assert_array_equal(image[3], checkpoint["baseline"][3])
    np.testing.assert_array_equal(
        image[:3, accepted].T,
        np.broadcast_to(checkpoint["source_colors"][checkpoint["names"][0]], (accepted.sum(), 3)),
    )
    assert np.all(source[accepted] == 1)
    assert np.all(source[~accepted] == 0)
    assert profile["transform"] == checkpoint["profile"]["transform"]
    assert profile["crs"] == rio.CRS.from_epsg(32650)
    assert engine.sha256(checkpoint["baseline_file"]) == checkpoint["baseline_hash"]
    report = json.loads((output / "report.json").read_text())
    assert report["acceptedPatches"] == 1
    assert report["sourceIndex"]["1"] == checkpoint["names"][0]


def test_one_invalid_pixel_rejects_whole_patch_without_reintroducing_partial_baseline(
    synthetic_checkpoint, tmp_path, monkeypatch,
):
    checkpoint = synthetic_checkpoint

    def partial_projection(*args):
        image, valid, cost, obs = checkpoint["project_view"](*args)
        valid[valid.shape[0] // 2, valid.shape[1] // 2] = False
        return image, valid, cost, obs

    monkeypatch.setattr(engine, "project_view", partial_projection)
    image, source, accepted, _ = execute(checkpoint, tmp_path / "rejected")
    np.testing.assert_array_equal(image, checkpoint["baseline"])
    assert not source.any()
    assert not accepted.any()
    objects = json.loads((tmp_path / "rejected/objects.json").read_text())
    assert objects[0]["state"] == "baseline-fallback"


def test_zero_conflicts_preserves_full_baseline_and_zero_sources(
    synthetic_checkpoint, tmp_path, monkeypatch,
):
    monkeypatch.setattr(engine, "associate_observations", lambda *_: ([], {}))
    image, source, accepted, _ = execute(synthetic_checkpoint, tmp_path / "no-conflicts")
    np.testing.assert_array_equal(image, synthetic_checkpoint["baseline"])
    assert not source.any()
    assert not accepted.any()


def test_conflict_boundary_cannot_split_a_vehicle_visible_in_the_baseline(
    synthetic_checkpoint, tmp_path, monkeypatch,
):
    checkpoint = synthetic_checkpoint
    # The moving track begins at x=20 with a 2-pixel patch buffer. A vehicle
    # actually visible in the baseline crosses that left boundary; replacing
    # only its right half would leave a fresh source seam through the vehicle.
    checkpoint["observations"].append(
        {
            **checkpoint["observations"][0],
            "kind": "baseline", "view": 0, "capture_time": None,
            "sourceImage": "__baseline__", "clock": "baseline",
            "origin": (15, 24), "mask": np.ones((6, 10), bool),
            "appearance": np.asarray([0.0, 1.0]),
        }
    )

    def forbidden_projection(*_):
        pytest.fail("A patch cutting another observed vehicle must fail before projection")

    monkeypatch.setattr(engine, "project_view", forbidden_projection)
    output = tmp_path / "unrelated-vehicle"
    image, source, accepted, _ = execute(checkpoint, output)
    np.testing.assert_array_equal(image, checkpoint["baseline"])
    assert not source.any()
    assert not accepted.any()
    objects = json.loads((output / "objects.json").read_text())
    assert objects[0]["state"] == "baseline-fallback"
    assert objects[0]["reason"] == "conflict-cuts-observed-vehicle"


def test_a_cut_vehicle_in_a_candidate_photo_rejects_only_that_candidate(
    synthetic_checkpoint, tmp_path,
):
    checkpoint = synthetic_checkpoint
    checkpoint["observations"].append(
        {
            **checkpoint["observations"][0],
            "origin": (15, 24), "mask": np.ones((6, 10), bool),
            "appearance": np.asarray([0.0, 1.0]),
        }
    )
    output = tmp_path / "candidate-cut"
    image, source, accepted, _ = execute(checkpoint, output)
    # Photo 1 is the cheaper projection, but its boundary-crossing vehicle
    # cannot be split. Photo 2 safely covers the whole track and is selected.
    assert accepted.any()
    assert np.all(source[accepted] == 2)
    assert np.all(source[~accepted] == 0)
    np.testing.assert_array_equal(image[:, ~accepted], checkpoint["baseline"][:, ~accepted])
    np.testing.assert_array_equal(
        image[:3, accepted].T,
        np.broadcast_to(checkpoint["source_colors"][checkpoint["names"][1]], (accepted.sum(), 3)),
    )
    objects = json.loads((output / "objects.json").read_text())
    assert objects[0]["sourceImage"] == checkpoint["names"][1]
    assert any(
        decision["sourceImage"] == checkpoint["names"][0]
        and decision["reason"] == "conflict-cuts-source-vehicle"
        for decision in objects[0]["rejections"]
    )


def test_a_vehicle_seen_only_at_an_unrelated_time_does_not_block_the_current_patch(
    synthetic_checkpoint, tmp_path,
):
    checkpoint = synthetic_checkpoint
    checkpoint["observations"].append(
        {
            **checkpoint["observations"][0],
            "view": 99, "capture_time": 153,
            "sourceImage": "DJI_20260925164306_0102_V.JPG.tif",
            "origin": (15, 24), "mask": np.ones((6, 10), bool),
            "appearance": np.asarray([0.0, 1.0]),
        }
    )
    image, source, accepted, _ = execute(checkpoint, tmp_path / "future-vehicle")
    assert accepted.any()
    assert np.all(source[accepted] == 1)
    assert np.all(source[~accepted] == 0)
    np.testing.assert_array_equal(image[:, ~accepted], checkpoint["baseline"][:, ~accepted])


def test_overlapping_conflict_plans_are_rejected_before_any_photo_is_written(
    synthetic_checkpoint, tmp_path, monkeypatch,
):
    monkeypatch.setattr(engine, "associate_observations", lambda *_: ([[0, 1], [0, 1]], {}))

    def forbidden_projection(*_):
        pytest.fail("Overlapping plans must be rejected before projection or mutation")

    monkeypatch.setattr(engine, "project_view", forbidden_projection)
    output = tmp_path / "overlap"
    image, source, accepted, _ = execute(synthetic_checkpoint, output)
    np.testing.assert_array_equal(image, synthetic_checkpoint["baseline"])
    assert not source.any()
    assert not accepted.any()
    objects = json.loads((output / "objects.json").read_text())
    assert [item["reason"] for item in objects] == ["overlapping-conflict-plans"] * 2


def test_crop_preserves_geographic_window_and_unchanged_pixels(synthetic_checkpoint, tmp_path):
    checkpoint = synthetic_checkpoint
    image, source, accepted, profile = execute(
        checkpoint, tmp_path / "crop", bounds=[101, 194, 106, 199],
    )
    assert image.shape == (4, 50, 50)
    assert profile["transform"] == from_origin(101, 199, 0.1, 0.1)
    np.testing.assert_array_equal(image[:, ~accepted], checkpoint["baseline"][:, 10:60, 10:60][:, ~accepted])
    assert np.all(source[accepted] == 1)


@pytest.mark.parametrize("artifact", ["ogs2_orthophoto.tif", "ogs2_orthophoto.partial.tif"])
def test_completed_or_partial_output_is_never_overwritten(tmp_path, artifact):
    project, output = tmp_path / "project", tmp_path / "output"
    project.mkdir()
    output.mkdir()
    original = output / artifact
    original.write_bytes(b"preserved experiment")
    with pytest.raises(FileExistsError):
        engine.run(project, output, tmp_path / "models", engine.PatchConfig(), "cpu")
    assert original.read_bytes() == b"preserved experiment"


def test_output_cannot_overlap_the_preserved_checkpoint_in_either_direction(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for output in (project, project / "result", tmp_path):
        with pytest.raises(ValueError, match="separate"):
            engine.run(project, output, tmp_path / "models", engine.PatchConfig(), "cpu")


def test_different_capture_clock_domains_cannot_be_associated_without_calibration():
    observations = [
        {
            "view": index + 1, "capture_time": 100 + index, "origin": (index * 30, 0),
            "mask": np.ones((10, 20), bool), "complete": True, "confidence": 0.95,
            "appearance": np.asarray([1.0, 0.0]), "clock": clock,
        }
        for index, clock in enumerate(("opensfm-capture-time", "dji-filename-wall-clock"))
    ]
    groups, _ = associate_observations(observations, 0.1)
    assert groups == []
