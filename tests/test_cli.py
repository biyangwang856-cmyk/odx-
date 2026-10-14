from __future__ import annotations

import json

import pytest
from experiments.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_list_json_reports_registered_metadata() -> None:
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0, result.output
    rows = json.loads(result.stdout)
    assert any(row["name"] == "build.remesh-runtime" for row in rows)
    assert all(set(row) == {"name", "category", "description"} for row in rows)


def test_module_task_dry_run_does_not_execute() -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "quality.validate-adaptive-hlod",
            "--dry-run",
            "--",
            "--help",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "quality.adaptive_hlod --help" in result.stdout


@pytest.mark.parametrize(
    ("task", "entrypoint"),
    [
        ("analysis.profile-interactive-path", "browser/playwright/profile_interactive_path.cjs"),
        ("quality.audit-visible-tiles", "browser/playwright/audit_visible_tiles.cjs"),
    ],
)
def test_external_task_dry_run_uses_owned_component(task: str, entrypoint: str) -> None:
    result = runner.invoke(app, ["run", task, "--dry-run"])
    assert result.exit_code == 0, result.output
    assert entrypoint in result.stdout
