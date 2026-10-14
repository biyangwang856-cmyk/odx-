from __future__ import annotations

import pytest
from experiments.registry import TaskRegistryError, get_task, list_tasks, register_task


def test_expected_tasks_are_auto_discovered() -> None:
    names = [task.name for task in list_tasks()]
    assert names == sorted(names)
    assert "build.remesh-runtime" in names
    assert "quality.validate-adaptive-hlod" in names
    assert all(not name.startswith("viewer.") for name in names)


def test_unknown_task_fails_with_available_names() -> None:
    with pytest.raises(TaskRegistryError, match="unknown task"):
        get_task("build.does-not-exist")


def test_duplicate_registration_is_rejected() -> None:
    list_tasks()
    with pytest.raises(TaskRegistryError, match="duplicate task"):

        @register_task("build.remesh-runtime", category="test", description="duplicate registration")
        def second(context, arguments):  # type: ignore[no-untyped-def]
            return 0
