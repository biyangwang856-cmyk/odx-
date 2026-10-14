"""Registered mesh build tasks."""

from __future__ import annotations

from collections.abc import Sequence

from experiments.context import TaskContext
from experiments.registry import register_task
from experiments.remesh import run_profile
from experiments.tasks._helpers import module_task


@register_task(
    "build.remesh-runtime",
    category="build",
    description="Build the v55 ETC1S/KTX2 and Meshopt production publication.",
)
def remesh_runtime(context: TaskContext, arguments: Sequence[str]) -> int:
    return run_profile("runtime", context, arguments)


@register_task(
    "build.adaptive-hlod",
    category="build",
    description="Build measured adaptive HLOD proxies from a tiled source.",
)
def adaptive_hlod(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "core.pipelines.adaptive_hlod", arguments)


@register_task(
    "build.source-leaf-tiles",
    category="build",
    description="Partition source meshes into material-aware spatial leaf tiles.",
)
def source_leaf_tiles(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "core.pipelines.source_leaf_tiler", arguments)


@register_task(
    "build.textured-proxy",
    category="build",
    description="Build the textured proxy proof-of-concept candidate.",
)
def textured_proxy(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "core.pipelines.textured_proxy", arguments)


@register_task(
    "build.meshopt-proxy",
    category="build",
    description="Build a Meshopt-compressed textured proxy candidate.",
)
def meshopt_proxy(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "core.pipelines.meshopt_proxy", arguments)


@register_task(
    "build.texture-budget",
    category="build",
    description="Build a texture-budget-constrained proxy candidate.",
)
def texture_budget(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "core.pipelines.texture_budget", arguments)


@register_task(
    "build.targeted-texture",
    category="build",
    description="Restore targeted texture coverage for selected proxy tiles.",
)
def targeted_texture(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "core.pipelines.targeted_texture", arguments)
