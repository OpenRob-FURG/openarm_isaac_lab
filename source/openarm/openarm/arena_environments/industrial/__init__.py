"""The 50 staged industrial tasks for the OpenArm bimanual embodiment.

Each task is a :class:`~openarm.arena_environments.industrial._base.IndustrialTask`:
a declarative stage list plus the scene it needs. Because a task is pure data,
all 50 share one environment class, one builder and one oracle policy; adding a
51st task means adding a :class:`TaskSpec` to a family module, not a new module.
"""

from __future__ import annotations

from openarm.arena_environments.industrial import beverage, containers, fastening, inspection, packing, tools
from openarm.arena_environments.industrial._base import (
    IndustrialEnvironment,
    IndustrialTask,
    build_environment,
    make_environment_class,
)

#: Every industrial task, keyed by the task key used on the command line and by
#: the oracle policy's ``--task`` flag.
INDUSTRIAL_TASKS: dict[str, IndustrialTask] = {
    task.spec.name: task
    for family in (fastening, packing, beverage, containers, tools, inspection)
    for task in family.TASKS
}

#: Environment classes for every task, keyed by environment name. The CLI turns
#: this into one subcommand per task.
INDUSTRIAL_ENVIRONMENTS: dict[str, type[IndustrialEnvironment]] = {
    task.name: make_environment_class(key) for key, task in INDUSTRIAL_TASKS.items()
}


def _validate() -> None:
    """Check that the task table is internally consistent.

    Every problem is collected before raising, so a single import reports all of
    them rather than one per run.

    Raises:
        AssertionError: Listing every inconsistency found.
    """
    problems: list[str] = []
    if len(INDUSTRIAL_TASKS) != 50:
        problems.append(f"expected 50 industrial tasks, found {len(INDUSTRIAL_TASKS)}")
    for key, task in INDUSTRIAL_TASKS.items():
        spec = task.spec
        if key != spec.name:
            problems.append(f"{key}: task key does not match spec name '{spec.name}'")
        if not spec.stages:
            problems.append(f"{key}: no stages")
        if not spec.goals():
            problems.append(f"{key}: no stage goals, so success would always hold")
        if not spec.instruction:
            problems.append(f"{key}: no instruction")
        for stage in spec.stages:
            for role, name in (("target", stage.target), ("secondary", stage.secondary)):
                if name and name not in task.objects:
                    problems.append(f"{key}: stage {role} '{name}' is not in the scene")
        for goal in spec.goals():
            for name in goal.object_names():
                if name not in task.objects:
                    problems.append(f"{key}: goal refers to unknown object '{name}'")
        for name in task.pinned_objects:
            if name not in task.objects:
                problems.append(f"{key}: pinned object '{name}' is not in the scene")
    assert not problems, "invalid industrial task table:\n  " + "\n  ".join(problems)


_validate()

__all__ = [
    "INDUSTRIAL_ENVIRONMENTS",
    "INDUSTRIAL_TASKS",
    "IndustrialEnvironment",
    "IndustrialTask",
    "build_environment",
    "make_environment_class",
]
