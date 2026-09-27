"""Staged task framework for OpenArm bimanual industrial tasks.

A staged task is described once, as data, in a
:class:`~openarm.arena_tasks.staged.stages.TaskSpec`. The same description
builds both the Arena task (success predicates, observations, metrics) and the
scripted oracle policy (demonstrations), so the two cannot drift apart.

The pure-data half of the package (:mod:`~openarm.arena_tasks.staged.goals` and
:mod:`~openarm.arena_tasks.staged.stages`) imports nothing from Isaac Sim, so it
is safe to import before the simulation app has been launched.
:class:`~openarm.arena_tasks.staged.task.StagedTask` is *not* safe to import at
that point -- it subclasses an Arena task base, which transitively pulls in
Omniverse modules that must only be touched after ``SimulationApp`` exists. It is
therefore re-exported lazily through :pep:`562`, so that the CLI parser, which
needs the task data to build its subparsers, does not crash the process.
"""

from typing import TYPE_CHECKING

from openarm.arena_tasks.staged.goals import (
    All,
    AtLeast,
    Goal,
    JointNormalizedAbove,
    JointNormalizedBelow,
    ObjectAbovePlane,
    ObjectAtRest,
    ObjectInRegion,
    ObjectRelativeTo,
    ObjectsWithinDistance,
    ObjectUpright,
)
from openarm.arena_tasks.staged.stages import OBJECT_SKILLS, Skill, Stage, TaskSpec

if TYPE_CHECKING:
    from openarm.arena_tasks.staged.task import StagedTask

__all__ = [
    "All",
    "AtLeast",
    "Goal",
    "JointNormalizedAbove",
    "JointNormalizedBelow",
    "OBJECT_SKILLS",
    "ObjectAbovePlane",
    "ObjectAtRest",
    "ObjectInRegion",
    "ObjectRelativeTo",
    "ObjectsWithinDistance",
    "ObjectUpright",
    "Skill",
    "Stage",
    "StagedTask",
    "TaskSpec",
]

#: Names resolved from :mod:`openarm.arena_tasks.staged.task` on first access.
_LAZY = {"StagedTask"}


def __getattr__(name: str):
    """Import the simulation-app-dependent part of the package on first use."""
    if name in _LAZY:
        from openarm.arena_tasks.staged import task as _task

        return getattr(_task, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
