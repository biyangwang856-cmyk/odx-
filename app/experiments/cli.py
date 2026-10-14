"""Typer command-line interface for the registered research task catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from experiments.context import TaskContext
from experiments.registry import TaskRegistryError, get_task, list_tasks

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
    help="Run registered Remesh, HLOD, analysis, and quality research tasks.",
)


@app.command("list")
def list_registered_tasks(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable task metadata."),
    ] = False,
) -> None:
    """List registered tasks in stable name order."""
    tasks = list_tasks()
    if as_json:
        typer.echo(
            json.dumps(
                [
                    {
                        "name": task.name,
                        "category": task.category,
                        "description": task.description,
                    }
                    for task in tasks
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    for task in tasks:
        typer.echo(f"{task.name:<38} {task.description}")


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run(
    context: typer.Context,
    task_name: Annotated[str, typer.Argument(metavar="TASK", help="Exact task name from 'list'.")],
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Research input/output root."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Render the task without executing it."),
    ] = False,
) -> None:
    """Run one task; arguments after `--` are passed to that task unchanged."""
    task_arguments = list(context.args)
    if task_arguments[:1] == ["--"]:
        task_arguments.pop(0)
    try:
        task = get_task(task_name)
        task_context = TaskContext.create(data_root=data_root, dry_run=dry_run)
        exit_code = task.callback(task_context, task_arguments)
    except TaskRegistryError as error:
        raise typer.BadParameter(str(error), param_hint="TASK") from error
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1) from error
    if exit_code:
        raise typer.Exit(code=exit_code)


def main() -> None:
    """Invoke the Typer application."""
    app()


if __name__ == "__main__":
    main()
