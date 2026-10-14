from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGER = ROOT / "scripts/deploy/stage_worker_image_context.sh"
REMESH_TOOLCHAIN_STAGER = ROOT / "scripts/deploy/stage_remesh_toolchain_context.sh"
DEPENDENCY_STAGER = ROOT / "scripts/deploy/stage_worker_dependency_context.sh"


def staged_context(tmp_path: Path, worker: str) -> Path:
    runtime = tmp_path / "ai-worker-runtime"
    runtime.write_bytes(b"runtime-v0.2.3")
    runtime.chmod(0o755)
    runtime_sha256 = hashlib.sha256(runtime.read_bytes()).hexdigest()
    context = tmp_path / f"{worker}-context"
    subprocess.run(
        [str(STAGER), str(ROOT), worker, str(runtime), str(context), runtime_sha256],
        check=True,
        env={**os.environ, "LC_ALL": "C"},
    )
    return context


def test_worker_context_rejects_a_drifted_runtime_artifact(tmp_path: Path) -> None:
    runtime = tmp_path / "ai-worker-runtime"
    runtime.write_bytes(b"not-the-released-runtime")
    context = tmp_path / "context"

    completed = subprocess.run(
        [str(STAGER), str(ROOT), "remesh", str(runtime), str(context), "0" * 64],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "ai-worker-runtime checksum mismatch" in completed.stderr
    assert not context.exists()


def test_remesh_toolchain_descriptors_bind_the_locked_inputs() -> None:
    basisu = json.loads((ROOT / "docker/remesh/basisu.json").read_text(encoding="utf-8"))
    gltf_transform = json.loads((ROOT / "docker/remesh/gltf-transform.json").read_text(encoding="utf-8"))

    assert basisu["binaryPlatform"] == "linux-x86_64"
    assert (
        gltf_transform["packageJsonSha256"]
        == hashlib.sha256((ROOT / "docker/remesh/package.json").read_bytes()).hexdigest()
    )
    assert (
        gltf_transform["packageLockSha256"]
        == hashlib.sha256((ROOT / "docker/remesh/package-lock.json").read_bytes()).hexdigest()
    )
    assert gltf_transform["audit"]["high"] == 0
    assert gltf_transform["audit"]["critical"] == 0


def test_remesh_toolchain_context_rejects_an_unlocked_binary(tmp_path: Path) -> None:
    binary = tmp_path / "basisu"
    binary.write_bytes(b"not-the-locked-linux-binary")
    binary.chmod(0o755)
    license_file = tmp_path / "LICENSE"
    license_file.write_text("Apache-2.0", encoding="utf-8")
    gltf_root = tmp_path / "gltf-transform"
    (gltf_root / "node_modules").mkdir(parents=True)
    (gltf_root / "package.json").write_text("{}\n", encoding="utf-8")
    (gltf_root / "package-lock.json").write_text("{}\n", encoding="utf-8")
    context = tmp_path / "remesh-toolchain-context"

    completed = subprocess.run(
        [
            str(REMESH_TOOLCHAIN_STAGER),
            str(ROOT),
            str(binary),
            str(license_file),
            str(gltf_root),
            str(context),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "BasisU binary checksum mismatch" in completed.stderr
    assert not context.exists()
    assert REMESH_TOOLCHAIN_STAGER.stat().st_mode & 0o111


def assert_production_only(context: Path) -> None:
    forbidden = {
        ".git",
        ".pixi",
        ".zvec-grep",
        "browser",
        "data",
        "docs",
        "graft",
        "services",
        "tests",
        "work_dirs",
        "weights",
    }
    relative = {path.relative_to(context).as_posix() for path in context.rglob("*")}
    assert not {path.split("/", 1)[0] for path in relative} & forbidden
    assert not any(
        path.endswith((".env", ".env.local", ".pyc", ".pyo", ".tar")) or "/__pycache__/" in f"/{path}/"
        for path in relative
    )


def test_photogrammetry_worker_context_is_complete_and_minimal(tmp_path: Path) -> None:
    context = staged_context(tmp_path, "photogrammetry")

    assert_production_only(context)
    assert (context / "worker-runtime/ai-worker-runtime").stat().st_mode & 0o111
    assert (context / "environment/pixi-config.toml").is_file()
    assert (context / "runtime-root/app/core/contracts/photogrammetry_odx.py").is_file()
    assert (context / "runtime-root/app/model_engine/odx.py").is_file()
    runtime = tomllib.loads(
        (context / "runtime-root/configs/server/odx.runtime.toml").read_text(encoding="utf-8")
    )
    assert runtime["engine"]["args"][0] == "PATH=/usr/local/bin:/usr/bin:/bin"
    task = tomllib.loads((context / "runtime-root/configs/server/odx.task.toml").read_text(encoding="utf-8"))
    assert task["task_name"] == "orthomosaic_production"
    assert task["task_revision"] == "photogrammetry.odx.v1"
    assert task["runtime_name"] == "ai-worker-runtime"
    assert task["runtime_version"] == "0.2.3"
    assert task["adapter_mode"] == "oneshot"
    assert task["resource_kind"] == "gpu"

    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")
    assert "# syntax=" not in dockerfile
    assert "ARG ODX_BASE_IMAGE" in dockerfile
    assert "FROM ${ODX_BASE_IMAGE}" in dockerfile
    assert "COPY environment/pixi-config.toml /etc/pixi/config.toml" in dockerfile
    assert "--mount=type=cache,id=mesh-pixi-rattler" in dockerfile
    assert "COPY --chmod=0755 worker-runtime/ai-worker-runtime" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/ai-worker-runtime"]' in dockerfile
    assert "apt-get" not in dockerfile


def test_remesh_worker_context_is_complete_and_minimal(tmp_path: Path) -> None:
    context = staged_context(tmp_path, "remesh")

    assert_production_only(context)
    assert (context / "worker-runtime/ai-worker-runtime").stat().st_mode & 0o111
    assert (context / "runtime-root/app/core/pipelines/remesh.py").is_file()
    assert (context / "runtime-root/app/model_engine/remesh.py").is_file()
    assert (context / "runtime-root/configs/mesh-candidates.json").is_file()
    assert (context / "runtime-root/configs/server/remesh-profile.json").is_file()
    assert (context / "environment/pixi.toml").is_file()
    assert (context / "environment/pixi.lock").is_file()
    assert (context / "environment/pixi-config.toml").is_file()
    assert not (context / "toolchain/package.json").exists()
    assert not (context / "toolchain/package-lock.json").exists()
    gltf_transform = json.loads((context / "toolchain/gltf-transform.json").read_text(encoding="utf-8"))
    assert gltf_transform["version"] == "4.5.0"
    assert gltf_transform["sharpVersion"] == "0.35.4"
    assert gltf_transform["treeSha256"] == (
        "01b4b34350737f03de9e95a9be68c9a9a7a055f261f2d691cd28d38f14e92aad"
    )
    assert gltf_transform["audit"] == {
        "command": "npm audit --omit=dev --audit-level=high",
        "high": 0,
        "critical": 0,
    }
    basisu = json.loads((context / "toolchain/basisu.json").read_text(encoding="utf-8"))
    assert basisu["version"] == "1.16.3"
    assert basisu["gitCommit"] == "1531cfaf9ed5232248a0a45736686a849ca3befc"
    assert basisu["sourceSha256"] == ("b89563aa5879eed20f56b9cfa03b52848e759531fd5a1d51a8f63c846f96c2ac")
    assert basisu["binaryPlatform"] == "linux-x86_64"
    assert basisu["binarySha256"] == "2c3c041c2b3c33cacfee8b7afd5012513531f71009c4dfd83e1c5131463d4087"
    assert basisu["binaryBytes"] == 2642168
    task = tomllib.loads(
        (context / "runtime-root/configs/server/remesh.task.toml").read_text(encoding="utf-8")
    )
    assert task["task_name"] == "reality_model_production"
    assert task["task_revision"] == "mesh.remesh-3d-tiles.v2"
    assert task["runtime_name"] == "ai-worker-runtime"
    assert task["runtime_version"] == "0.2.3"
    assert task["adapter_mode"] == "oneshot"
    assert task["resource_kind"] == "cpu"

    dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")
    assert "# syntax=" not in dockerfile
    assert "ARG PIXI_IMAGE" in dockerfile
    assert "ARG REMESH_TOOLCHAIN_IMAGE" in dockerfile
    assert "FROM ${REMESH_TOOLCHAIN_IMAGE} AS remesh-toolchain" in dockerfile
    assert "COPY environment/pixi-config.toml /etc/pixi/config.toml" in dockerfile
    assert "pixi install --locked --environment production" in dockerfile
    assert "--mount=type=cache,id=mesh-pixi-rattler" in dockerfile
    assert "PIXI_CACHE_DIR=/root/.cache/rattler timeout 900s" in dockerfile
    assert "pixi install --locked --environment basisu-build" not in dockerfile
    assert "cmake --build" not in dockerfile
    assert "urllib.request" not in dockerfile
    assert "npm ci" not in dockerfile
    assert "COPY --from=remesh-toolchain /usr/local/bin/basisu" in dockerfile
    assert "COPY --from=remesh-toolchain /opt/gltf-transform" in dockerfile
    assert "BasisU binary checksum mismatch" in dockerfile
    assert "COPY --chmod=0755 worker-runtime/ai-worker-runtime" in dockerfile
    assert 'ENTRYPOINT ["/usr/local/bin/ai-worker-runtime"]' in dockerfile


def test_dependency_contexts_are_lock_scoped_and_exclude_runtime_payloads(tmp_path: Path) -> None:
    config = (ROOT / "docker/pixi/config.toml").read_text(encoding="utf-8")
    assert "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge" in config
    assert "https://mirrors.nju.edu.cn/anaconda/cloud/conda-forge" in config

    for worker, target in (("photogrammetry", "runtime-abi"), ("remesh", "production-environment")):
        context = tmp_path / worker
        subprocess.run([str(DEPENDENCY_STAGER), str(ROOT), worker, str(context)], check=True)
        assert (context / "environment/pixi.toml").is_file()
        assert (context / "environment/pixi.lock").read_bytes() == (ROOT / "pixi.lock").read_bytes()
        assert (context / "environment/pixi-config.toml").read_text(encoding="utf-8") == config
        assert target in (context / "Dockerfile").read_text(encoding="utf-8")
        assert not (context / "runtime-root").exists()
        assert not (context / "worker-runtime").exists()

    assert DEPENDENCY_STAGER.stat().st_mode & 0o111


def test_locked_npm_path_resolves_node_from_the_same_environment(tmp_path: Path) -> None:
    environment = tmp_path / "environment"
    bin_dir = environment / "bin"
    npm_cli = environment / "lib/node_modules/npm/bin/npm-cli.js"
    bin_dir.mkdir(parents=True)
    npm_cli.parent.mkdir(parents=True)
    node = bin_dir / "node"
    node.write_text("#!/bin/sh\nprintf 'locked-node:%s\\n' \"$*\"\n", encoding="utf-8")
    node.chmod(0o755)
    npm_cli.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    npm_cli.chmod(0o755)
    (bin_dir / "npm").symlink_to(Path("../lib/node_modules/npm/bin/npm-cli.js"))

    completed = subprocess.run(
        [str(bin_dir / "npm"), "--version"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin"},
    )

    assert completed.stdout == f"locked-node:{bin_dir / 'npm'} --version\n"
