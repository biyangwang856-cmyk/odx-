"""Registered Worker adapter entrypoints."""

from __future__ import annotations

from collections.abc import Sequence

from experiments.context import TaskContext
from experiments.registry import register_task
from experiments.tasks._helpers import module_task


@register_task(
    "adapter.photogrammetry-odx",
    category="adapter",
    description="Run the digest-pinned ODX one-shot Worker adapter.",
)
def photogrammetry_odx(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "model_engine.odx", arguments)


@register_task(
    "adapter.remesh-3d-tiles",
    category="adapter",
    description="Build staged Mesh 3D Tiles from a validated ODX bundle.",
)
def remesh_3d_tiles(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "model_engine.remesh", arguments)
