"""Which arm handles a staged task's object, derived from where the object is.

A stage never declares an arm. The OpenArm torso sits on the environment
centreline, so an object placed on the robot's left is a left-arm object and one
on its right is a right-arm object, and the only question worth asking is which
side of the centreline the object ended up on. That makes the arm a *consequence
of the layout* rather than a fact about the task, which is what lets one
declarative :class:`~openarm.arena_tasks.staged.stages.TaskSpec` drive the
scene, the oracle, the reward terms and the Mimic config without any of them
disagreeing about who does what.

Three ideas make that work.

**Nominal positions.** :func:`nominal_positions` walks a
:class:`~openarm.arena_tasks.staged.layout.SceneLayout` and returns an estimated
planar position per object, honouring anchored poses, ``AtPosition`` pins and
``NextTo`` chains. It is an estimate, not the solver's answer: it uses nominal
footprints where the real solver uses measured bounding boxes. It only has to be
good enough to tell which side of the centreline an object is on, and it has to
be computable *before* the scene exists, because the layout and the arm choice
feed each other.

**Seeding the side.** The estimate cannot invent a side out of nothing, so
:func:`stage_sides` seeds each stage from the geometry the task already states:
a goal region's centre, an anchored pose, the region a receptacle serves, or the
previous stage. What it never does is consult a hand-written ``arm`` field, which
is the thing this module exists to delete.

**Deadbands.** Heavy pose randomization moves objects by roughly 0.035 m in y,
so an object sitting exactly on the centreline could be claimed by either arm
depending on the reset. :func:`arm_for_y` resolves that with a deadband around
the torso, and :func:`find_arm_conflicts` reports the stages where nothing can
be decided cleanly instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Literal

from openarm.arena_tasks.staged.goals import Goal, ObjectInRegion
from openarm.arena_tasks.staged.layout import (
    AtPosition,
    IsAnchor,
    LayoutRelation,
    NextTo,
    Placement,
    PositionLimits,
    SceneLayout,
    Side,
)
from openarm.arena_tasks.staged.stages import Skill, Stage, TaskSpec

#: A side of the robot's centreline, named the way the rest of the package names
#: an arm.
Arm = Literal["left", "right"]

#: Both arms, in the order used when a task needs to name either one.
ARMS: tuple[Arm, Arm] = ("left", "right")

#: World y of the torso centreline [m]. The OpenArm humanoid torso is spawned on
#: the environment origin, so this is the frame the two arms are symmetric about.
TORSO_Y = 0.0

#: World y of each shoulder [m]. Not used to decide a side -- both shoulders are
#: within a centimetre of the centreline, so they carry no side information -- but
#: exported because :func:`stage_sides` reports reachability against them.
SHOULDER_Y = 0.031

#: Half-width of the ambiguous band around the centreline [m].
#:
#: Wider than the y jitter the domain-randomization presets apply, so a layout
#: that resolves to the right side does not flip sides on a later reset.
ARM_DEADBAND_M = 0.05

#: Arm joints commanded per arm, excluding the finger [rad].
#:
#: The OpenArm is a 7-DoF arm per side, which is also the width of one
#: differential-IK pose command. That coincidence is why the action space has to
#: be identified by its terms rather than by its width: 7 joints plus a finger is
#: 8 values per arm and a 7-wide pose command plus a gripper is also 8.
ARM_JOINTS_PER_ARM = 7

#: End-effector body per arm, in the OpenArm USD.
ARM_EE_BODIES = {"left": "openarm_left_ee_tcp", "right": "openarm_right_ee_tcp"}

#: Arm joints per arm, as the asset names them.
ARM_JOINT_NAMES = {arm: [f"openarm_{arm}_joint{index}" for index in range(1, ARM_JOINTS_PER_ARM + 1)] for arm in ARMS}

#: Finger joint per arm, as the asset names it.
ARM_FINGER_JOINTS = {
    "left": "openarm_left_finger_joint1",
    "right": "openarm_right_finger_joint1",
}

#: Nominal object footprint used when estimating a ``NextTo`` offset [m].
#:
#: The real solver measures each asset's bounding box; this is the fallback for
#: the before-the-scene-exists estimate, and it only shifts an object by its own
#: width, which cannot move it across a side boundary for any object that fits in
#: a gripper.
DEFAULT_NOMINAL_HALF_EXTENT_M = 0.05

#: Skills whose controller drives more than one arm.
#:
#: These are not conflicts. :attr:`Skill.INSERT` uses ``stage.arm`` for the arm
#: doing the inserting and its opposite to steady the receptacle, and
#: :attr:`Skill.DOCK` returns both end effectors home; either way ``stage.arm``
#: names the acting arm and the rest follows, so the stage still resolves to a
#: single side like any other. The set exists because a caller validating a spec
#: needs to know that "this stage names one arm" does not imply "this stage only
#: moves one arm".
MULTI_ARM_SKILLS = frozenset({Skill.INSERT, Skill.DOCK})


def sign(value: float) -> int:
    """Return the sign of *value* as -1, 0 or 1."""
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def arm_for_offset(offset_y: float, deadband_m: float = ARM_DEADBAND_M) -> Arm | None:
    """Return the arm whose side of the torso *offset_y* falls on.

    Args:
        offset_y: Object y relative to the torso centreline [m].
        deadband_m: Half-width of the ambiguous band around the centreline [m].

    Returns:
        ``"left"`` for a positive offset, ``"right"`` for a negative one, and
        ``None`` inside the deadband, where neither arm is a defensible answer
        and the caller has to look at the stage instead.
    """
    magnitude = abs(float(offset_y))
    if magnitude <= float(deadband_m):
        return None
    return "left" if offset_y > 0.0 else "right"


def arm_for_y(y: float, torso_y: float = TORSO_Y, deadband_m: float = ARM_DEADBAND_M) -> Arm | None:
    """Return the arm that handles an object at world y, or ``None`` if unclear.

    Args:
        y: Object world y [m].
        torso_y: World y of the torso centreline [m].
        deadband_m: Half-width of the ambiguous band around the centreline [m].

    Returns:
        The arm, or ``None`` when the object is within the deadband.
    """
    return arm_for_offset(float(y) - float(torso_y), deadband_m)


def nominal_positions(
    layout: SceneLayout,
    half_extents: dict[str, float] | None = None,
    default_half_extent_m: float = DEFAULT_NOMINAL_HALF_EXTENT_M,
) -> dict[str, tuple[float, float]]:
    """Estimate each placed object's planar world position from the layout alone.

    The relation solver has not run when this is called, and it cannot have: the
    side an object is placed on is an input to the layout. So the position is
    reconstructed from the declarations, in placement order, which the layout
    guarantees is parent-before-child.

    Relations are applied as follows. An anchored ``pose`` or an ``AtPosition``
    sets the axes it names and leaves the rest. A ``PositionLimits`` clamps an
    axis that nothing else pinned. A ``NextTo`` offsets from the parent's
    estimate by the nominal footprint of both objects along the chosen axis and
    carries the other axis across at the relation's ``cross_position_ratio``.
    Anything unconstrained defaults to the work surface centre, which is the same
    anchor the solver starts from.

    Args:
        layout: The relational layout to walk.
        half_extents: Nominal half-extent per object name [m], overriding the
            default. Supply this when a scene's real footprints are known.
        default_half_extent_m: Half-extent for objects with no entry [m].

    Returns:
        Object name to estimated ``(x, y)`` in world coordinates, in placement
        order.
    """
    extents = half_extents or {}

    def half(object_name: str) -> float:
        return float(extents.get(object_name, default_half_extent_m))

    positions: dict[str, tuple[float, float]] = {}
    origin = (layout.surface.center[0], layout.surface.center[1])

    for entry in layout.placements:
        x, y = positions.get(entry.object_name, origin)
        for relation in entry.relations:
            x, y = _apply_relation(relation, entry, x, y, positions, half)
        if entry.pose is not None:
            position, _rotation = entry.pose
            x, y = float(position[0]), float(position[1])
        positions[entry.object_name] = (x, y)
    return positions


def _apply_relation(
    relation: LayoutRelation,
    entry: Placement,
    x: float,
    y: float,
    positions: dict[str, tuple[float, float]],
    half,
) -> tuple[float, float]:
    """Fold one declared relation into an estimated position."""
    if isinstance(relation, AtPosition):
        if relation.x is not None:
            x = float(relation.x)
        if relation.y is not None:
            y = float(relation.y)
    elif isinstance(relation, PositionLimits):
        if relation.x_min is not None:
            x = max(x, float(relation.x_min))
        if relation.x_max is not None:
            x = min(x, float(relation.x_max))
        if relation.y_min is not None:
            y = max(y, float(relation.y_min))
        if relation.y_max is not None:
            y = min(y, float(relation.y_max))
    elif isinstance(relation, NextTo):
        parent = positions.get(relation.object_name)
        if parent is not None:
            gap = half(relation.object_name) + half(entry.object_name) + float(relation.distance_m)
            ratio = float(relation.cross_position_ratio)
            if relation.side is Side.POSITIVE_X:
                x, y = parent[0] + gap, parent[1] + ratio * gap
            elif relation.side is Side.NEGATIVE_X:
                x, y = parent[0] - gap, parent[1] - ratio * gap
            elif relation.side is Side.POSITIVE_Y:
                x, y = parent[0] - ratio * gap, parent[1] + gap
            else:
                x, y = parent[0] + ratio * gap, parent[1] - gap
    elif isinstance(relation, IsAnchor):
        pass
    return x, y


def region_centre_y(goal: Goal | None, ambiguous_y: float | None = None) -> float | None:
    """Return the y an ``ObjectInRegion`` goal is centred on, or ``None``.

    A goal region is the one place a staged task states an absolute world y for
    something, which makes it the natural seed for a stage's side.

    A region sitting within the deadband of the torso centreline carries no side
    information, so it reads as ``None`` unless the caller supplies the y the
    scene builder actually resolved it to. The environment builder remaps those
    regions onto the goal band, and passing that resolved y back in here keeps
    the side a stage derives and the side its object ends up on in agreement.

    Args:
        goal: The goal to inspect, or ``None``.
        ambiguous_y: The resolved y to use in place of a deadband-ambiguous
            region [m]. ``None`` reports such a region as having no side.

    Returns:
        The region's centre y [m], or ``None`` when the goal is not a region or
        names no nested one.
    """
    if isinstance(goal, ObjectInRegion):
        centre = float(goal.center[1])
        return ambiguous_y if abs(centre) <= ARM_DEADBAND_M else centre
    for nested in goal.nested_goals() if goal is not None else ():
        centre = region_centre_y(nested, ambiguous_y)
        if centre is not None:
            return centre
    return None


def _nested_region_goals(goal: Goal | None) -> Iterable[Goal]:
    """Yield a goal and every goal nested inside it, depth first."""
    if goal is None:
        return
    yield goal
    for nested in goal.nested_goals():
        yield from _nested_region_goals(nested)


def goal_region_sides(
    spec: TaskSpec,
    torso_y: float = TORSO_Y,
    resolved_region_y: dict[int, float] | None = None,
) -> list[str | None]:
    """Seed each stage's side from the goal regions the task itself declares.

    This is the part of :func:`stage_sides` that needs no layout, which is what
    makes it usable from :func:`~openarm.arena_environments.industrial._base.derive_layout`:
    the layout is built *from* these sides, so consulting it here would be
    circular.

    The seeds are tried in order:

    1. the y of the stage's own goal region, when it has one -- the one place a
       staged task states an absolute world y for something it cares about;
    2. the previous stage's side, so a stage that manipulates something already
       on a side stays on it;
    3. the right arm, matching the side the suite has always started on.

    Args:
        spec: The task's stage sequence.
        torso_y: World y of the torso centreline [m].
        resolved_region_y: The y the scene builder resolved each deadband-ambiguous
            goal region to, keyed by the region's index in the flattened
            declaration order that
            :func:`~openarm.openarm.arena_environments.industrial._base._region_centres`
            produces. Without it, an ambiguous region seeds no side.

    Returns:
        One side per stage, in stage order.
    """
    resolved_region_y = resolved_region_y or {}
    sides: list[str | None] = []
    region_index = 0

    for stage in spec.stages:
        side: Arm | None = None
        for nested in _nested_region_goals(stage.goal):
            if not isinstance(nested, ObjectInRegion):
                continue
            ambiguous = abs(float(nested.center[1])) <= ARM_DEADBAND_M
            centre = resolved_region_y.get(region_index) if ambiguous else None
            region_index += 1
            if ambiguous and centre is None:
                continue
            resolved = centre if ambiguous else float(nested.center[1])
            side = arm_for_y(resolved, torso_y)
            if side is not None:
                break

        if side is None and sides:
            side = sides[-1]
        sides.append(side or "right")
    return sides


def stage_sides(
    spec: TaskSpec,
    layout: SceneLayout,
    torso_y: float = TORSO_Y,
    resolved_region_y: dict[int, float] | None = None,
) -> list[str | None]:
    """Seed each stage's side of the torso from the geometry the task states.

    Starts from :func:`goal_region_sides` and adds the two seeds that need the
    layout:

    1. the y of an anchored pose for the stage's target object;
    2. the y of the goal region of whichever stage moves that same object, for a
       stage that only reads an object's position rather than placing it.

    Args:
        spec: The task's stage sequence.
        layout: The task's layout, used for anchored poses.
        torso_y: World y of the torso centreline [m].

    Returns:
        One side per stage, in stage order.
    """
    positions = nominal_positions(layout)
    sides = goal_region_sides(spec, torso_y, resolved_region_y)
    anchored = set(layout.anchored_names())

    for index, stage in enumerate(spec.stages):
        side = sides[index]

        if side is None and stage.target in anchored:
            target_y = positions.get(stage.target, (0.0, torso_y))[1]
            side = arm_for_y(target_y, torso_y)

        if side is None:
            for other, other_stage in enumerate(spec.stages):
                if other >= len(sides) or other_stage.target != stage.target:
                    continue
                for nested in _nested_region_goals(other_stage.goal):
                    centre = region_centre_y(nested)
                    if centre is not None:
                        side = arm_for_y(centre, torso_y)
                        break
                if side is not None:
                    break

        if side is not None:
            sides[index] = side
    return sides


def handoff_sides(spec: TaskSpec, torso_y: float = TORSO_Y) -> list[Arm]:
    """Return the side the receiving arm takes for each handoff in the spec.

    A :attr:`Skill.HANDOFF` stage exists to move a held object across the torso,
    so both of its stages describe the same object and the stage's target Y can
    only speak to the side that object starts on. The receiving arm is the other
    side, and which one that is has to come from somewhere: the hand-written
    ``arm`` on the stage is the only statement in the spec that distinguishes
    "this arm receives" from "this arm gives".

    Unlike :func:`resolve_stage_arms`, this *reads* ``Stage.arm`` rather than
    deriving it. That is not a contradiction of the rule that arms come from
    geometry: a handoff has no target position of its own to read, and treating
    the one authored bit as the receiving arm is what keeps the two tasks that
    exist to demonstrate a side change actually crossing sides. The spec is
    expected to declare the receiving arm, which is the opposite of the arm that
    picked the object up; :func:`find_arm_conflicts` reports a handoff whose
    sides do not actually differ.

    Args:
        spec: The task's stage sequence.
        torso_y: World y of the torso centreline [m].

    Returns:
        The receiving side per handoff stage, in stage order.

    Raises:
        ValueError: If a handoff stage declares no arm, since either way the
            receiving side is undetermined.
    """
    resolved: list[Arm] = []
    for index, stage in enumerate(spec.stages):
        if stage.skill is not Skill.HANDOFF:
            continue
        declared = stage.arm
        if declared not in ARMS:
            raise ValueError(
                f"stage {index} of '{spec.name}' is a handoff but declares arm={declared!r}; "
                "a handoff has to say which arm receives the object"
            )
        resolved.append(declared)
    return resolved


def resolve_stage_arms(
    spec: TaskSpec,
    layout: SceneLayout,
    torso_y: float = TORSO_Y,
    deadband_m: float = ARM_DEADBAND_M,
    resolved_region_y: dict[int, float] | None = None,
) -> TaskSpec:
    """Return *spec* with every stage's arm derived from its target's position.

    The arm written on each stage is the one :func:`arm_for_y` gives for the
    stage's target object's nominal position, so the arm follows the layout rather
    than being authored beside it. Stages whose target lands in the deadband keep
    the arm their seed side implies, because an explicit choice is better than a
    ``None`` in a field the rest of the package reads as a string.

    This intentionally discards whatever ``Stage.arm`` already said. A hand-written
    arm is exactly the field this module exists to derive, and a stage whose two
    arms would otherwise disagree is reported by :func:`find_arm_conflicts` rather
    than silently keeping the authored value. The one exception is
    :attr:`Skill.HANDOFF`, which takes its arm from :func:`handoff_sides` because a
    side change has no target position to read.

    Args:
        spec: The task spec to resolve.
        layout: The task's layout.
        torso_y: World y of the torso centreline [m].
        deadband_m: Half-width of the ambiguous band around the centreline [m].
        resolved_region_y: The y the scene builder resolved each deadband-ambiguous
            goal region to, keyed by declaration order.

    Returns:
        A new :class:`TaskSpec` with the same goals, names and targets, and an
        arm set on every stage.

    Raises:
        ValueError: Propagated from :func:`handoff_sides` for a handoff whose
            receiving side is undetermined.
    """
    positions = nominal_positions(layout)
    sides = stage_sides(spec, layout, torso_y, resolved_region_y)

    receivers = iter(handoff_sides(spec, torso_y))
    resolved: list[Stage] = []
    for stage, side in zip(spec.stages, sides):
        if stage.skill is Skill.HANDOFF:
            arm = next(receivers)
        else:
            target_y = positions.get(stage.target, (0.0, torso_y))[1]
            arm = arm_for_y(target_y, torso_y, deadband_m) or side or "right"
        resolved.append(stage if stage.arm == arm else replace(stage, arm=arm))
    return replace(spec, stages=tuple(resolved))


@dataclass(frozen=True)
class ArmConflict:
    """A stage whose arm cannot be derived from geometry alone.

    Attributes:
        stage_index: Index of the stage in the spec.
        stage: The stage itself.
        offset_y: The stage target's nominal y relative to the torso [m].
        reason: Why no arm could be chosen.
    """

    stage_index: int
    stage: Stage
    offset_y: float
    reason: str


def find_arm_conflicts(
    spec: TaskSpec,
    layout: SceneLayout,
    torso_y: float = TORSO_Y,
    deadband_m: float = ARM_DEADBAND_M,
    resolved_region_y: dict[int, float] | None = None,
) -> list[ArmConflict]:
    """Report the stages that geometry alone cannot assign to an arm.

    A stage conflicts when its target sits inside the deadband and nothing else
    in the task says which arm owns it, when the stage names both arms at once,
    or when a handoff would not actually change sides. Reporting these rather
    than raising is deliberate: the caller is usually generating configuration
    for fifty tasks at once and wants the whole list, not the first failure.

    A stage in :data:`MULTI_ARM_SKILLS` is *not* a conflict. Those controllers
    drive a second arm themselves, deriving it as the opposite of ``stage.arm``,
    so a resolved single arm is exactly what they need.

    The check runs against the *resolved* arms rather than the authored ones,
    because a handoff whose receiving arm agrees with the authored arm may still
    end up on the same side as the arm that resolved to give the object away.
    Judging the authored values instead would report the two handoff tasks as
    consistent when the derived layout puts both of their stages on one side.

    Args:
        spec: The task spec to check.
        layout: The task's layout.
        torso_y: World y of the torso centreline [m].
        deadband_m: Half-width of the ambiguous band around the centreline [m].
        resolved_region_y: The y the scene builder resolved each deadband-ambiguous
            goal region to, keyed by declaration order.

    Returns:
        One conflict per unassignable stage, in stage order.
    """
    positions = nominal_positions(layout)
    sides = stage_sides(spec, layout, torso_y, resolved_region_y)
    resolved = resolve_stage_arms(spec, layout, torso_y, deadband_m, resolved_region_y)
    conflicts: list[ArmConflict] = []

    for index, stage in enumerate(resolved.stages):
        target_y = positions.get(stage.target, (0.0, torso_y))[1]
        offset_y = target_y - torso_y

        if stage.arm == "both":
            conflicts.append(ArmConflict(index, stage, offset_y, "stage names both arms, so neither owns the target"))
            continue

        if stage.skill is Skill.HANDOFF:
            giver = resolved.stages[index - 1].arm if index else ""
            if giver in ARMS and giver == stage.arm:
                conflicts.append(
                    ArmConflict(
                        index,
                        stage,
                        offset_y,
                        f"handoff to the {stage.arm} arm from a stage that already uses the {giver} arm",
                    )
                )
            continue

        if abs(offset_y) <= float(deadband_m) and sides[index] is None:
            conflicts.append(
                ArmConflict(index, stage, offset_y, f"target '{stage.target}' is on the torso centreline")
            )
    return conflicts