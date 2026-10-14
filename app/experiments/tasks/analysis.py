"""Registered comparison and profiling tasks."""

from __future__ import annotations

from collections.abc import Sequence

from experiments.context import TaskContext
from experiments.registry import register_task
from experiments.tasks._helpers import module_task, repository_script


@register_task(
    "analysis.compare-screenshots",
    category="analysis",
    description="Compare fixed-camera screenshots and emit foreground metrics.",
)
def compare_screenshots(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "experiments.analysis.compare_screenshots", arguments)


@register_task(
    "analysis.map-texture-lineage",
    category="analysis",
    description="Map source textures to proxy tiles and summarize lineage.",
)
def map_texture_lineage(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "experiments.analysis.texture_lineage", arguments)


@register_task(
    "analysis.validate-candidate-protocol",
    category="analysis",
    description="Validate the frozen candidate protocol and source-image identity.",
)
def validate_candidate_protocol(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "experiments.analysis.candidate_protocol", arguments)


@register_task(
    "analysis.inventory-candidates",
    category="analysis",
    description="Inventory frozen candidate publications and unresolved evidence gaps.",
)
def inventory_candidates(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "experiments.analysis.candidate_inventory", arguments)


@register_task(
    "analysis.profile-interactive-path",
    category="analysis",
    description="Profile browser loading across the fixed interactive camera path.",
)
def profile_interactive_path(context: TaskContext, arguments: Sequence[str]) -> int:
    return repository_script(
        context,
        "node",
        "browser/playwright/profile_interactive_path.cjs",
        arguments,
    )


@register_task(
    "analysis.profile-pair-visual",
    category="analysis",
    description="Capture and profile a labeled two-publication viewer.",
)
def profile_pair_visual(context: TaskContext, arguments: Sequence[str]) -> int:
    return repository_script(
        context,
        "node",
        "browser/playwright/profile_pair_visual.cjs",
        arguments,
    )
