from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mesh_release_uses_broker_and_two_business_workers() -> None:
    pipeline = (ROOT / ".gitlab-ci.yml").read_text(encoding="utf-8")
    policy = (ROOT / "release/policy.yaml").read_text(encoding="utf-8")
    manifest = (ROOT / "release/image.yaml").read_text(encoding="utf-8")
    release_script = (ROOT / "scripts/ci/release-images.sh").read_text(encoding="utf-8")
    runtime_fetch_script = (ROOT / "scripts/ci/fetch-ai-worker-runtime.sh").read_text(encoding="utf-8")
    cache_warm_script = (ROOT / "scripts/ci/warm-worker-dependency-cache.sh").read_text(encoding="utf-8")

    assert "remesh-toolchain:" in pipeline
    assert (
        "npm --prefix docker/remesh audit --package-lock-only --omit=dev --audit-level=high "
        "--registry=https://registry.npmjs.org" in pipeline
    )
    assert "npm --prefix services/viewer test" in pipeline
    assert "release-contract:" in pipeline
    assert "warm-worker-dependency-cache:" in pipeline
    assert "release-request:" in pipeline
    assert "prepare-runtime:" in pipeline
    assert "release-images:" in pipeline
    release_images = pipeline.split("release-images:", 1)[1]
    assert "tags: [rc1-mesh-image-publisher]" in release_images
    assert pipeline.count('resource_group: "release-control/${CI_PROJECT_PATH_SLUG}"') == 3
    assert "surveying.release-request/v1" in pipeline
    assert "release-broker.release-broker.svc:8080/v1/requests" in pipeline
    assert "notify-platform-release" not in pipeline
    assert "projectPath: surveying/ai-models/mesh-research" in policy
    assert "manifest: release/image.yaml" in policy
    assert re.search(r"(?m)^releaseBaseline: (?:true|false)$", manifest)
    assert "deployableImages: [photogrammetryWorker, remeshWorker]" in manifest
    assert '"photogrammetryWorker"' in release_script
    assert '"remeshWorker"' in release_script
    assert '"odxBase"' in release_script
    assert '"pixiBase"' in release_script
    assert '"runtime"' in release_script
    assert '"basisUniversal"' in release_script
    assert "docker/remesh/basisu.json" in release_script
    assert "MESH_REMESH_TOOLCHAIN_IMAGE" in release_script
    assert re.search(r'MESH_REMESH_TOOLCHAIN_IMAGE: "harbor\.ai\.internal/.+@sha256:[0-9a-f]{64}"', pipeline)
    assert "build-arg:REMESH_TOOLCHAIN_IMAGE=$MESH_REMESH_TOOLCHAIN_IMAGE" in release_script
    assert '--import-cache "type=registry,ref=$cache_ref"' in release_script
    assert "--export-cache" in release_script
    assert "mode=max" in release_script
    assert "buildcache-pixi-$pixi_lock_sha256" in release_script
    assert '--opt "target=$target"' in cache_warm_script
    assert '"$build_root/photogrammetry" runtime-abi' in cache_warm_script
    assert '"$build_root/remesh" production-environment' in cache_warm_script
    assert "--export-cache" in cache_warm_script
    assert "--output" not in cache_warm_script
    assert "DEPLOY-TOKEN:${AI_WORKER_RUNTIME_PACKAGE_TOKEN}" in runtime_fetch_script
    assert "JOB-TOKEN" not in runtime_fetch_script

    tag_patterns = re.findall(r"\$CI_COMMIT_TAG =~ /(.+?)/", pipeline)
    assert len(tag_patterns) == 2
    for pattern in tag_patterns:
        assert re.fullmatch(pattern, "v1.2.3")
        assert re.fullmatch(pattern, "v1.2.3-rc.1")
        assert not re.fullmatch(pattern, "v1.2.3+build.7")


def test_release_commands_are_maintained_entrypoints() -> None:
    justfile = (ROOT / "justfile").read_text(encoding="utf-8")
    prepare = (ROOT / "scripts/prepare_release.sh").read_text(encoding="utf-8")
    rootless_test = ROOT / "scripts/ci/test-release-images-rootless.sh"

    assert "prepare-release version:" in justfile
    assert "release-contract:" in justfile
    assert "pixi workspace version set" in prepare
    assert 'branch="release/image/${version}"' in prepare
    assert "release version must be stable or prerelease SemVer without build metadata" in prepare
    assert rootless_test.stat().st_mode & 0o111
