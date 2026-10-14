import numpy as np
import pytest
from model_engine.ogs2_highway_background import HighwayBackgroundConfig, select_background


def inputs():
    images = np.full((4, 90, 110, 3), 110, np.uint8)
    images[:, 44:47, 12:98] = 205
    valid = np.ones(images.shape[:3], bool)
    guards = np.zeros_like(valid)
    road = np.ones(valid.shape[1:], bool)
    return images, valid, guards, road


def config():
    return HighwayBackgroundConfig(shadow_padding_pixels=0, patch_collar_pixels=15)


def test_whole_body_removed_with_exact_observed_background_and_no_blend():
    im, v, g, r = inputs()
    im[0, 30:40, 45:60] = [30, 45, 70]
    g[0, 26:43, 41:64] = True
    result = select_background(im, v, g, r, config=config())
    assert result["report"]["unresolvedPixels"] == 0
    assert result["report"]["applied"]
    assert np.array_equal(result["image"][30:40, 45:60], im[1, 30:40, 45:60])
    assert result["report"]["mixedPixels"] == 0
    assert result["report"]["selectedDonorGuardPixels"] == 0


def test_no_unobserved_background_is_invented():
    im, v, g, r = inputs()
    im[:, 30:40, 45:60] = [30, 45, 70]
    g[:, 26:43, 41:64] = True
    result = select_background(im, v, g, r, config=config())
    assert np.array_equal(result["image"], im[0])
    assert result["report"]["unresolvedPixels"] > 0
    assert result["report"]["inventedPixels"] == 0


def test_guarded_donor_cannot_be_selected_even_if_rgb_matches():
    im, v, g, r = inputs()
    im[0, 30:40, 45:60] = [30, 45, 70]
    g[0, 26:43, 41:64] = True
    g[1, 26:43, 41:64] = True
    result = select_background(im, v, g, r, config=config())
    assert not (result["source"][26:43, 41:64] == 1).any()
    assert result["report"]["selectedDonorGuardPixels"] == 0


def test_masked_coverage_hole_has_explicit_real_source():
    im, v, g, r = inputs()
    v[0, 30:40, 45:60] = False
    im[0, 30:40, 45:60] = 0
    result = select_background(im, v, g, r, config=config())
    assert result["valid"][30:40, 45:60].all()
    assert (result["source"][30:40, 45:60] > 0).all()
    assert np.array_equal(result["image"][30:40, 45:60], im[1, 30:40, 45:60])


def test_corrections_are_bounded_and_shapes_are_checked():
    im, v, g, r = inputs()
    im[1] += 30
    result = select_background(im, v, g, r, config=config())
    assert max(abs(x) for delta in result["report"]["colourOffsets"] for x in delta) <= 12
    with pytest.raises(ValueError):
        select_background(im, v.astype(np.uint8), g, r)
    with pytest.raises(ValueError):
        select_background(im, v, g, r, config=HighwayBackgroundConfig(colour_offset_limit=13))


def test_several_clear_donors_jointly_cover_one_foreground_component():
    im, v, g, r = inputs()
    im[0, 25:42, 35:75] = [30, 45, 70]
    g[0, 22:45, 32:78] = True
    g[1, 22:45, 55:78] = True
    g[2, 22:45, 32:55] = True
    g[3, 22:45, 32:78] = True
    result = select_background(im, v, g, r, config=config())
    assert result["report"]["unresolvedPixels"] == 0
    assert result["report"]["selectedDonorGuardPixels"] == 0
    assert np.array_equal(result["image"][25:42, 35:75], im[1, 25:42, 35:75])
