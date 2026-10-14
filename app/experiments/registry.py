"""Decorator-based registry for independently runnable research tasks."""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from experiments.context import TaskContext

TaskCallable = Callable[[TaskContext, Sequence[str]], int]
TASK_NAME = re.compile(r"^[a-z][a-z0-9-]*(?:\.[a-z][a-z0-9-]*)+$")


class TaskRegistryError(RuntimeError):
    """Raised when task registration or lookup violates the registry contract."""


@dataclass(frozen=True)
class TaskDefinition:
    """Stable metadata and implementation for one runnable task."""

    name: str
    category: str
    description: str
    callback: TaskCallable


_TASKS: dict[str, TaskDefinition] = {}
_DISCOVERED = False


def register_task(
    name: str,
    *,
    category: str,
    description: str,
) -> Callable[[TaskCallable], TaskCallable]:
    """Register a task function and return it unchanged."""
    if not TASK_NAME.fullmatch(name):
        raise TaskRegistryError(f"invalid task name {name!r}; use a stable dotted kebab-case name")
    if not category.strip() or not description.strip():
        raise TaskRegistryError(f"task {name!r} requires category and description")

    def decorator(callback: TaskCallable) -> TaskCallable:
        if name in _TASKS:
            owner = _TASKS[name].callback.__module__
            raise TaskRegistryError(f"duplicate task {name!r}; already registered by {owner}")
        _TASKS[name] = TaskDefinition(
            name=name,
            category=category.strip(),
            description=description.strip(),
            callback=callback,
        )
        return callback

    return decorator


def discover_tasks() -> None:
    """Import every public module in the tasks package exactly once."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    package = importlib.import_module("experiments.tasks")
    modules = sorted(
        module.name for module in pkgutil.iter_modules(package.__path__) if not module.name.startswith("_")
    )
    for module in modules:
        importlib.import_module(f"{package.__name__}.{module}")
    _DISCOVERED = True


def list_tasks() -> tuple[TaskDefinition, ...]:
    """Return all registered tasks in deterministic name order."""
    discover_tasks()
    return tuple(_TASKS[name] for name in sorted(_TASKS))


def get_task(name: str) -> TaskDefinition:
    """Resolve one task by exact stable name."""
    discover_tasks()
    try:
        return _TASKS[name]
    except KeyError as error:
        available = ", ".join(sorted(_TASKS))
        raise TaskRegistryError(f"unknown task {name!r}; available: {available}") from error
