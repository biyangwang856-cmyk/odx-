from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "task",
    [
        "quality.audit-glb-textures",
        "quality.validate-adaptive-hlod",
        "quality.candidate-acceptance",
        "quality.staging-gate",
        "quality.publish-staging",
    ],
)
def test_quality_tasks_dispatch_from_app(task: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "experiments.cli", "run", task, "--", "--help"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "app")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_experiment_cli_runs_from_the_app_package_root() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "experiments.cli", "list", "--json"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "app")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert '"build.source-leaf-tiles"' in result.stdout
    assert '"analysis.compare-screenshots"' in result.stdout


def test_shared_core_does_not_import_experiments_or_process_protocol() -> None:
    sources = list((ROOT / "app/core").rglob("*.py"))
    assert sources, "shared core is missing"
    for path in sources:
        for node in ast.walk(ast.parse(path.read_text())):
            imports = []
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imports = [node.module or ""]
            assert not any(name.split(".")[0] in {"experiments", "model_engine"} for name in imports), path
    assert not (ROOT / "src").exists()
    assert not (ROOT / "protocols").exists()
