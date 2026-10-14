"""Registered quality and validation tasks."""

from __future__ import annotations

from collections.abc import Sequence

from experiments.context import TaskContext
from experiments.registry import register_task
from experiments.tasks._helpers import module_task, repository_script


@register_task(
    "quality.audit-glb-textures",
    category="quality",
    description="Audit embedded GLB texture payloads and duplication.",
)
def audit_glb_textures(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "core.validation.glb_textures", arguments)


@register_task(
    "quality.validate-adaptive-hlod",
    category="quality",
    description="Validate adaptive HLOD structure and measured error metadata.",
)
def validate_adaptive_hlod(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "experiments.quality.adaptive_hlod", arguments)


@register_task(
    "quality.candidate-acceptance",
    category="quality",
    description="Audit candidate 3D Tiles and select the budgeted production default.",
)
def candidate_acceptance(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "experiments.quality.candidate_acceptance", arguments)


@register_task(
    "quality.staging-gate",
    category="quality",
    description="Validate a Mesh staging directory before publication.",
)
def staging_gate(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "core.validation.staging_gate", arguments)


@register_task(
    "quality.publish-staging",
    category="quality",
    description="Atomically promote and register validated Mesh staging output.",
)
def publish_staging(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "core.validation.staging_publication", arguments)


@register_task(
    "quality.audit-visible-tiles",
    category="quality",
    description="Audit visible browser tiles and request failures with Playwright.",
)
def audit_visible_tiles(context: TaskContext, arguments: Sequence[str]) -> int:
    return repository_script(
        context,
        "node",
        "browser/playwright/audit_visible_tiles.cjs",
        arguments,
    )
