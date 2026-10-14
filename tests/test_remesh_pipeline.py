from __future__ import annotations

from experiments.remesh import PROFILES


def test_runtime_profile_encodes_v55_selective_etc1s_policy() -> None:
    assert set(PROFILES) == {"runtime"}
    profile = PROFILES["runtime"]
    assert profile.simplify_ratio is None
    assert profile.meshopt_level == "high"
    assert profile.gpu_texture_codec == "etc1s"
    assert profile.gpu_texture_quality == 255
    assert profile.gpu_texture_compression_level == 1
    assert profile.gpu_texture_minimum_psnr == 32.0
    assert profile.gpu_texture_maximum_byte_ratio == 1.15
    assert "v55" in profile.output
