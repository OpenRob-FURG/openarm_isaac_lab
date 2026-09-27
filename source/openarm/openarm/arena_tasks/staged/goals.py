"""Declarative success predicates ("goals") for staged OpenArm tasks.

A :class:`Goal` is a declarative description of a condition that must hold at
the end of an episode. Each goal knows how to build the manager parameters for
the matching termination function in
:mod:`openarm.arena_tasks.staged.terminations`, so a task can be specified
entirely as data.

Goals compose: :class:`All` and :class:`AtLeast` build conjunctions, which is
how a multi-stage task states "the nut is on the bolt *and* the microwave door
is closed".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union


@dataclass(frozen=True)
class Goal:
    """Base class for a success predicate.

    ``label`` is keyword-only on purpose. It is declared here, on the base
    class, so it would otherwise be the first positional parameter of every
    subclass, and ``ObjectInRegion("mug", center, half_extents)`` would silently
    bind ``"mug"`` to ``label`` instead of ``object_name``. Every goal in the
    package is written positionally, so making the label keyword-only removes
    that whole class of silent misconfiguration.
    """

    #: Human-readable name, used in logging and in the language instruction.
    label: str = field(default="", kw_only=True)

    def params(self) -> dict[str, Any]:
        """Build the keyword parameters for the matching termination function."""
        raise NotImplementedError

    def object_names(self) -> list[str]:
        """Return every scene object name this goal refers to, recursively.

        Used to check that a task's goals only mention objects that its scene
        actually contains, which is the most common way a declarative task spec
        goes wrong.

        Returns:
            Deduplicated scene names, in the order they appear.
        """
        names: list[str] = []
        for key in ("object_name", "reference_name", "other_name", "articulation_name"):
            value = getattr(self, key, None)
            name = getattr(value, "name", None) or (value if isinstance(value, str) else None)
            if name and name not in names:
                names.append(name)
        for nested in self.nested_goals():
            for name in nested.object_names():
                if name not in names:
                    names.append(name)
        return names

    def nested_goals(self) -> tuple["Goal", ...]:
        """Return the goals composed inside this one, if any.

        Returns:
            The child goals, empty for a leaf goal.
        """
        return ()

    def describe(self) -> str:
        """Return a short human-readable description of the goal."""
        return self.label or type(self).__name__


@dataclass(frozen=True)
class ObjectInRegion(Goal):
    """The object's centre must lie inside an axis-aligned world-frame box.

    This is the workhorse goal for "place the object in the bin": bins and
    target zones are specified as a centre plus half-extents rather than as a
    second object, so a task can be defined without requiring a mesh for the
    destination.

    Attributes:
        object_name: Scene name of the object.
        center: Centre of the target box [m].
        half_extents: Half-size of the target box along each axis [m].
    """

    object_name: str = ""
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_extents: tuple[float, float, float] = (0.1, 0.1, 0.1)

    def params(self) -> dict[str, Any]:
        return {
            "kind": "in_region",
            "object_cfg": self.object_name,
            "center": self.center,
            "half_extents": self.half_extents,
        }

    def describe(self) -> str:
        return self.label or f"{self.object_name} inside region at {tuple(round(v, 3) for v in self.center)}"


@dataclass(frozen=True)
class ObjectRelativeTo(Goal):
    """The object must lie within ``tolerance`` of a fixed offset from a reference object.

    This captures assembly relationships that are defined by geometry rather
    than by an absolute zone: a nut threaded onto a bolt, a gear seated on a
    base, a lid on a container. Because the reference moves with the assembly,
    the predicate keeps working after the whole assembly is translated.

    Attributes:
        object_name: Scene name of the moving object.
        reference_name: Scene name of the reference object.
        offset: Required position offset of the object from the reference [m].
        tolerance: Allowed Euclidean deviation from the offset [m].
    """

    object_name: str = ""
    reference_name: str = ""
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tolerance: float = 0.02

    def params(self) -> dict[str, Any]:
        return {
            "kind": "relative_to",
            "object_cfg": self.object_name,
            "reference_cfg": self.reference_name,
            "offset": self.offset,
            "tolerance": self.tolerance,
        }

    def describe(self) -> str:
        return self.label or f"{self.object_name} seated on {self.reference_name}"


@dataclass(frozen=True)
class ObjectUpright(Goal):
    """The object's local +Z axis must be within ``tolerance`` of world +Z.

    Used for "place the can upright" and "set the bottle the right way up",
    where the position alone is not enough to distinguish success.

    Attributes:
        object_name: Scene name of the object.
        tolerance: Maximum angle between the axes [rad].
    """

    object_name: str = ""
    tolerance: float = 0.35

    def params(self) -> dict[str, Any]:
        return {
            "kind": "upright",
            "object_cfg": self.object_name,
            "tolerance": self.tolerance,
        }

    def describe(self) -> str:
        return self.label or f"{self.object_name} upright"


@dataclass(frozen=True)
class ObjectsWithinDistance(Goal):
    """Two objects must be closer than ``max_distance``.

    Attributes:
        object_name: Scene name of the first object.
        other_name: Scene name of the second object.
        max_distance: Maximum centre-to-centre distance [m].
    """

    object_name: str = ""
    other_name: str = ""
    max_distance: float = 0.05

    def params(self) -> dict[str, Any]:
        return {
            "kind": "within_distance",
            "object_cfg": self.object_name,
            "other_cfg": self.other_name,
            "max_distance": self.max_distance,
        }

    def describe(self) -> str:
        return self.label or f"{self.object_name} within {self.max_distance} m of {self.other_name}"


@dataclass(frozen=True)
class JointNormalizedAbove(Goal):
    """A normalized joint position must exceed a threshold.

    This is how the articulated fixtures are checked: an openable door's
    openness, a pressable button's depression, and a turnable dial's angle all
    reduce to a normalized joint coordinate in [0, 1].

    Attributes:
        articulation_name: Scene name of the articulation.
        joint_name: Name of the joint to read.
        threshold: Normalized threshold to exceed.
    """

    articulation_name: str = ""
    joint_name: str = ""
    threshold: float = 0.5

    def params(self) -> dict[str, Any]:
        return {
            "kind": "joint_above",
            "asset_cfg": self.articulation_name,
            "joint_name": self.joint_name,
            "threshold": self.threshold,
        }

    def describe(self) -> str:
        return self.label or f"{self.articulation_name}:{self.joint_name} above {self.threshold}"


@dataclass(frozen=True)
class JointNormalizedBelow(Goal):
    """A normalized joint position must fall below a threshold.

    Attributes:
        articulation_name: Scene name of the articulation.
        joint_name: Name of the joint to read.
        threshold: Normalized threshold to fall below.
    """

    articulation_name: str = ""
    joint_name: str = ""
    threshold: float = 0.5

    def params(self) -> dict[str, Any]:
        return {
            "kind": "joint_below",
            "asset_cfg": self.articulation_name,
            "joint_name": self.joint_name,
            "threshold": self.threshold,
        }

    def describe(self) -> str:
        return self.label or f"{self.articulation_name}:{self.joint_name} below {self.threshold}"


@dataclass(frozen=True)
class ObjectAtRest(Goal):
    """The object must be nearly stationary.

    Used as a settling condition so that "released the object" is not scored
    while the object is still sliding or tumbling.

    Attributes:
        object_name: Scene name of the object.
        max_speed: Maximum linear speed [m/s].
        max_angular_speed: Maximum angular speed [rad/s].
    """

    object_name: str = ""
    max_speed: float = 0.05
    max_angular_speed: float = 0.5

    def params(self) -> dict[str, Any]:
        return {
            "kind": "at_rest",
            "object_cfg": self.object_name,
            "max_speed": self.max_speed,
            "max_angular_speed": self.max_angular_speed,
        }

    def describe(self) -> str:
        return self.label or f"{self.object_name} at rest"


@dataclass(frozen=True)
class ObjectAbovePlane(Goal):
    """The object must be above a world-frame height, and optionally upright.

    Attributes:
        object_name: Scene name of the object.
        min_height: Minimum world-frame height [m].
        max_height: Maximum world-frame height [m].
    """

    object_name: str = ""
    min_height: float = 0.0
    max_height: float = 1e6

    def params(self) -> dict[str, Any]:
        return {
            "kind": "above_plane",
            "object_cfg": self.object_name,
            "min_height": self.min_height,
            "max_height": self.max_height,
        }

    def describe(self) -> str:
        return self.label or f"{self.object_name} between {self.min_height} and {self.max_height} m"


@dataclass(frozen=True)
class All(Goal):
    """Every contained goal must hold.

    Attributes:
        goals: The goals that must all be satisfied.
    """

    goals: tuple[Goal, ...] = field(default_factory=tuple)

    def nested_goals(self) -> tuple[Goal, ...]:
        """Return the composed child goals."""
        return tuple(self.goals)

    def params(self) -> dict[str, Any]:
        return {
            "kind": "all",
            "goals": [goal.params() for goal in self.goals],
        }

    def describe(self) -> str:
        return " and ".join(goal.describe() for goal in self.goals) or "all conditions"


@dataclass(frozen=True)
class AtLeast(Goal):
    """At least ``count`` of the contained goals must hold.

    This is what makes partially-successful industrial tasks scoreable. A
    palletizing task that places five of six cans correctly is meaningfully
    better than one that places none, and a binary success term cannot express
    that. It also gives a learning signal during post-training.

    Attributes:
        count: Minimum number of contained goals that must hold.
        goals: The candidate goals.
    """

    count: int = 1
    goals: tuple[Goal, ...] = field(default_factory=tuple)

    def nested_goals(self) -> tuple[Goal, ...]:
        """Return the composed child goals."""
        return tuple(self.goals)

    def params(self) -> dict[str, Any]:
        return {
            "kind": "at_least",
            "count": self.count,
            "goals": [goal.params() for goal in self.goals],
        }

    def describe(self) -> str:
        return f"at least {self.count} of: " + ", ".join(goal.describe() for goal in self.goals)


AnyGoal = Union[
    ObjectInRegion,
    ObjectRelativeTo,
    ObjectUpright,
    ObjectsWithinDistance,
    JointNormalizedAbove,
    JointNormalizedBelow,
    ObjectAtRest,
    ObjectAbovePlane,
    All,
    AtLeast,
]
