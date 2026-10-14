"""Shared execution context for registered research tasks."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DATA_ROOT = Path("/mnt/juicefs/shared/mesh-research-runs/20260814-lglglg-adaptive-hlod-v1")


def repository_root() -> Path:
    """Return the source checkout root for this Pixi workspace."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TaskContext:
    """Paths and process policy shared by every registered task."""

    repo_root: Path
    data_root: Path
    dry_run: bool = False

    @classmethod
    def create(cls, *, data_root: Path | None = None, dry_run: bool = False) -> TaskContext:
        configured = data_root or Path(os.environ.get("MESH_RESEARCH_DATA_ROOT", str(DEFAULT_DATA_ROOT)))
        return cls(
            repo_root=repository_root(),
            data_root=configured.expanduser().resolve(),
            dry_run=dry_run,
        )

    def run(
        self,
        command: Sequence[str],
        *,
        extra_env: Mapping[str, str] | None = None,
    ) -> int:
        """Run one task command with a consistent repository and data-root environment."""
        rendered = shlex.join(str(part) for part in command)
        if self.dry_run:
            print(rendered)
            return 0

        environment = os.environ.copy()
        environment["MESH_RESEARCH_DATA_ROOT"] = str(self.data_root)
        environment["MESH_RESEARCH_REPO_ROOT"] = str(self.repo_root)
        if extra_env:
            environment.update(extra_env)
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=self.repo_root,
            env=environment,
            check=False,
        )
        return completed.returncode

    def run_python_module(self, module: str, arguments: Sequence[str]) -> int:
        """Run a packaged Python module in an isolated child process."""
        return self.run([sys.executable, "-m", module, *arguments])
