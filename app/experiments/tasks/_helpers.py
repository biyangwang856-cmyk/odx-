"""Small adapters shared by task registration modules."""

from __future__ import annotations

from collections.abc import Sequence

from experiments.context import TaskContext


def module_task(context: TaskContext, module: str, arguments: Sequence[str]) -> int:
    """Execute an existing argparse module through the shared context."""
    return context.run_python_module(module, arguments)


def repository_script(
    context: TaskContext,
    interpreter: str,
    relative_path: str,
    arguments: Sequence[str],
) -> int:
    """Execute a repository script without embedding an absolute code path."""
    return context.run([interpreter, str(context.repo_root / relative_path), *arguments])
