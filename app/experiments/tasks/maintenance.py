"""Registered historical schedule-maintenance tasks."""

from __future__ import annotations

from collections.abc import Sequence

from experiments.context import TaskContext
from experiments.registry import register_task
from experiments.tasks._helpers import module_task


@register_task(
    "maintenance.reschedule-hlod-errors",
    category="maintenance",
    description="Recompute HLOD geometric-error scheduling for a publication.",
)
def reschedule_hlod_errors(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "experiments.tools.reschedule_hlod_errors", arguments)


@register_task(
    "maintenance.restore-v6-schedule",
    category="maintenance",
    description="Restore the recorded v6 p99 geometric-error schedule.",
)
def restore_v6_schedule(context: TaskContext, arguments: Sequence[str]) -> int:
    return module_task(context, "experiments.tools.restore_v6_schedule", arguments)
