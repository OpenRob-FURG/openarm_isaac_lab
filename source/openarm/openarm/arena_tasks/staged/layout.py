"""Semantic spatial layout for staged OpenArm tasks.

A :class:`SceneLayout` states *where things are relative to each other* -- on the
work surface, next to the previous item, inside the robot's reachable band --
rather than a pile of absolute world coordinates. That vocabulary deliberately
mirrors ``isaaclab_arena.relations.relations`` one class at a time, so the two
can be read side by side, and the actual placement is solved by Arena's own
:class:`~isaaclab_arena.relations.object_placer.ObjectPlacer`.

Why go through Arena instead of computing poses here:

* the solver works from real asset bounding boxes, so an object's resting height
  comes from its own mesh instead of from a guessed constant;
* overlaps between neighbours, and the "does the gripper have room" question, are
  answered by validation and retries rather than by hoping the arithmetic was
  right;
* per-reset jitter becomes an Arena ``RandomAroundSolution``, so randomization is
  expressed as part of the layout instead of being bolted on afterwards.

The module deliberately holds no Isaac Sim imports. :class:`SceneLayout` and the
relation dataclasses are pure data, so a task table stays importable and
testable without a simulator; :func:`build_relations` and :func:`solve_layout`
import Arena lazily, exactly as the environment builder does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Side(str, Enum):
    """Axis direction for a :class:`NextTo` relation.

    Mirrors ``isaaclab_arena.relations.relations.Side``; the resolver maps each
    member onto the Arena enum of the same value.
    """

    POSITIVE_X = "positive_x"
    NEGATIVE_X = "negative_x"
    POSITIVE_Y = "positive_y"
    NEGATIVE_Y = "negative_y"


# -- relations ---------------------------------------------------------------
#
# Each class below mirrors the Arena relation it stands for. They exist so a task
# can be written as data: Arena's relations reference a live ``Object`` as their
# parent, which cannot be resolved until the scene is instantiated.


@dataclass(frozen=True)
class On:
    """The object rests on the work surface (or another object).

    Attributes:
        surface: Name of the surface to rest on. Resolved against the layout's
            own :attr:`SceneLayout.surface` unless it names another object.
        clearance_m: Gap left between the object's underside and the surface
            top. Slightly positive so the object settles onto the surface
            instead of starting interpenetrating it.
        relation_loss_weight: Weight of this relation in the solver's loss.
    """

    surface: str = ""
    clearance_m: float = 0.005
    relation_loss_weight: float = 1.0


@dataclass(frozen=True)
class NextTo:
    """The object sits beside another object, on a named side.

    This is what expresses a row: each item references the previous one, so the
    row keeps its structure and spacing without anybody computing coordinates.

    Attributes:
        object_name: Scene name of the object to sit beside.
        side: Which side of it the object goes.
        distance_m: Gap between the two bounding boxes, measured from the
            neighbour's boundary [m].
        cross_position_ratio: Where along the neighbour's perpendicular axis the
            object lands, from -1.0 (one edge) through 0.0 (centred) to 1.0.
        relation_loss_weight: Weight of this relation in the solver's loss.
    """

    object_name: str = ""
    side: Side = Side.POSITIVE_X
    distance_m: float = 0.12
    cross_position_ratio: float = 0.0
    relation_loss_weight: float = 1.0


@dataclass(frozen=True)
class AtPosition:
    """The object is pinned to world coordinates on the given axes.

    Used where a task genuinely needs an absolute location, such as putting a
    receptacle where its goal region says it is. Omitted axes are left to the
    object's other relations, which is what lets ``x``/``y`` be stated while
    ``z`` follows from :class:`On`.

    Attributes:
        x: World x [m], or ``None`` to leave unconstrained.
        y: World y [m], or ``None`` to leave unconstrained.
        z: World z [m], or ``None`` to leave unconstrained.
        relation_loss_weight: Weight of this relation in the solver's loss.
    """

    x: float | None = None
    y: float | None = None
    z: float | None = None
    relation_loss_weight: float = 1.0


@dataclass(frozen=True)
class PositionLimits:
    """The object must stay inside a world-frame axis-aligned box.

    The reachability band. Every industrial object has to end up somewhere the
    OpenArm arms can actually get to, and stating that as a box means the
    constraint is enforced by the solver rather than hoped for.

    Attributes:
        x_min: Minimum world x [m], or ``None``.
        x_max: Maximum world x [m], or ``None``.
        y_min: Minimum world y [m], or ``None``.
        y_max: Maximum world y [m], or ``None``.
        z_min: Minimum world z [m], or ``None``.
        z_max: Maximum world z [m], or ``None``.
        relation_loss_weight: Weight of this relation in the solver's loss.
    """

    x_min: float | None = None
    x_max: float | None = None
    y_min: float | None = None
    y_max: float | None = None
    z_min: float | None = None
    z_max: float | None = None
    relation_loss_weight: float = 1.0


@dataclass(frozen=True)
class NoCollision:
    """The object's bounding box must not overlap another object's.

    Attributes:
        object_name: Scene name of the object to keep clear of.
        clearance_m: Extra separation to leave between the boxes [m].
        relation_loss_weight: Weight of this relation in the solver's loss.
    """

    object_name: str = ""
    clearance_m: float = 0.01
    relation_loss_weight: float = 1.0


@dataclass(frozen=True)
class IsAnchor:
    """The object is a fixed reference and must not be moved.

    An anchor is how a layout states "this pose is given, not solved": the
    work surface, and any object the task pins to an explicit pose.
    """


@dataclass(frozen=True)
class RandomAroundSolution:
    """Resample the solved position on every reset, within a box around it.

    Arena turns this marker into a ``PoseRange``, which is what makes the
    placement a per-reset randomization term. It is the native way to do the
    pose jitter that used to be applied as a second pass over already-solved
    poses.

    Attributes:
        x_half_m: Half-extent of the sampled box along x [m].
        y_half_m: Half-extent of the sampled box along y [m].
        z_half_m: Half-extent of the sampled box along z [m].
        yaw_half_rad: Symmetric yaw jitter [rad].
    """

    x_half_m: float = 0.0
    y_half_m: float = 0.0
    z_half_m: float = 0.0
    yaw_half_rad: float = 0.0

    @property
    def is_active(self) -> bool:
        """Whether this marker would actually randomize anything."""
        return any((self.x_half_m, self.y_half_m, self.z_half_m, self.yaw_half_rad))


@dataclass(frozen=True)
class RotateAroundSolution:
    """Apply a fixed rotation on top of the solved position.

    Attributes:
        roll_rad: Roll about local x [rad].
        pitch_rad: Pitch about local y [rad].
        yaw_rad: Yaw about local z [rad].
    """

    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0

    @property
    def is_active(self) -> bool:
        """Whether this marker would actually rotate anything."""
        return any((self.roll_rad, self.pitch_rad, self.yaw_rad))


#: Every relation a :class:`Placement` may carry. Named ``LayoutRelation`` rather
#: than ``Relation`` because Arena already uses that name for the base class of
#: its binary relations, and the two appearing in one scope would be a trap.
LayoutRelation = (
    On | NextTo | AtPosition | PositionLimits | NoCollision | IsAnchor | RandomAroundSolution | RotateAroundSolution
)


@dataclass(frozen=True)
class Surface:
    """The work surface an industrial scene is laid out on.

    Arena's ``ObjectPlacer`` needs an anchor with a world bounding box to solve
    against. The industrial background is a procedurally spawned cuboid, so there
    is no USD to read a box out of and no prim to point an ``ObjectReference``
    at; describing the slab here lets the same native machinery drive it.

    Attributes:
        name: Name of the surface, used to resolve :class:`On` relations.
        center: Centre of the slab in the world frame [m].
        size: Full size of the slab along each axis [m].
    """

    name: str = "work_surface"
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: tuple[float, float, float] = (0.8, 1.5, 0.04)

    @property
    def top_z(self) -> float:
        """Height of the surface's top face in the world frame [m]."""
        return self.center[2] + self.size[2] / 2.0

    def extents(self) -> tuple[float, float, float, float]:
        """Return the ``(x_min, x_max, y_min, y_max)`` footprint of the slab [m]."""
        half_x, half_y = self.size[0] / 2.0, self.size[1] / 2.0
        return (
            self.center[0] - half_x,
            self.center[0] + half_x,
            self.center[1] - half_y,
            self.center[1] + half_y,
        )


@dataclass(frozen=True)
class Placement:
    """Where one scene object goes, and what it is constrained by.

    Attributes:
        object_name: Scene instance name this placement applies to.
        relations: The spatial relations and markers to attach.
        pose: Explicit ``(position, rotation_xyzw)`` for an anchored object.
            Required when the placement carries :class:`IsAnchor` for something
            other than the work surface.
    """

    object_name: str = ""
    relations: tuple[LayoutRelation, ...] = ()
    pose: tuple | None = None

    def with_relations(self, relations: tuple[LayoutRelation, ...]) -> "Placement":
        """Return a copy carrying ``relations`` in addition to its own.

        Args:
            relations: Relations to append.

        Returns:
            A new :class:`Placement`.
        """
        return Placement(object_name=self.object_name, relations=(*self.relations, *relations), pose=self.pose)

    @property
    def is_anchor(self) -> bool:
        """Whether this placement is fixed rather than solved."""
        return any(isinstance(relation, IsAnchor) for relation in self.relations)

    def parent_names(self) -> list[str]:
        """Return every scene object name this placement references.

        An :class:`On` without a ``surface`` refers to the layout's own work
        surface, which is not a scene object and so is excluded. An ``On`` that
        names an object refers to that object, and is included like any other
        parent.

        Returns:
            The referenced names, in declaration order and deduplicated.
        """
        names: list[str] = []
        for relation in self.relations:
            candidates = [getattr(relation, "object_name", "")]
            if isinstance(relation, On) and relation.surface:
                candidates.append(relation.surface)
            for name in candidates:
                if name and name not in names:
                    names.append(name)
        return names


@dataclass(frozen=True)
class SceneLayout:
    """A complete relational description of a scene's arrangement.

    Attributes:
        surface: The work surface everything is laid out on.
        placements: One :class:`Placement` per scene object, in declaration
            order. Order matters: it is the order the solver receives objects
            in, and it is what makes a derived layout reproducible.
    """

    surface: Surface = field(default_factory=Surface)
    placements: tuple[Placement, ...] = ()

    def placement(self, object_name: str) -> Placement:
        """Return the placement for a scene object.

        Args:
            object_name: Scene instance name.

        Returns:
            The matching :class:`Placement`.

        Raises:
            KeyError: If the object has no placement.
        """
        for entry in self.placements:
            if entry.object_name == object_name:
                return entry
        raise KeyError(f"'{object_name}' has no placement in this layout")

    def object_names(self) -> list[str]:
        """Return the scene object names this layout places, in order."""
        return [entry.object_name for entry in self.placements]

    def anchored_names(self) -> list[str]:
        """Return the names of the placements that are fixed rather than solved."""
        return [entry.object_name for entry in self.placements if entry.is_anchor]


# -- resolution to native Arena relations ------------------------------------


def _arena_side(side: Side):
    """Translate a :class:`Side` into Arena's enum of the same value.

    Args:
        side: The declared side.

    Returns:
        The matching ``isaaclab_arena.relations.relations.Side``.
    """
    from isaaclab_arena.relations import relations as arena_relations

    return arena_relations.Side(side.value)


def build_relation(relation: LayoutRelation, parents: dict[str, Any], surface_object: Any):
    """Build the native Arena relation for one declared relation.

    Args:
        relation: The declared relation.
        parents: Scene objects by name, used to resolve relation parents.
        surface_object: The Arena object standing in for the work surface.

    Returns:
        An instance of the corresponding Arena relation class.

    Raises:
        KeyError: If the relation names an object the scene does not contain.
    """
    from isaaclab_arena.relations import relations as arena_relations

    if isinstance(relation, IsAnchor):
        return arena_relations.IsAnchor()
    if isinstance(relation, On):
        parent = surface_object if not relation.surface else parents[relation.surface]
        return arena_relations.On(parent, relation.relation_loss_weight, relation.clearance_m)
    if isinstance(relation, NextTo):
        return arena_relations.NextTo(
            parents[relation.object_name],
            relation.relation_loss_weight,
            relation.distance_m,
            _arena_side(relation.side),
            relation.cross_position_ratio,
        )
    if isinstance(relation, AtPosition):
        return arena_relations.AtPosition(relation.x, relation.y, relation.z, relation.relation_loss_weight)
    if isinstance(relation, PositionLimits):
        return arena_relations.PositionLimits(
            relation.x_min,
            relation.x_max,
            relation.y_min,
            relation.y_max,
            relation.z_min,
            relation.z_max,
            relation.relation_loss_weight,
        )
    if isinstance(relation, NoCollision):
        return arena_relations.NoCollision(
            parents[relation.object_name],
            relation.relation_loss_weight,
            relation.clearance_m,
        )
    if isinstance(relation, RandomAroundSolution):
        return arena_relations.RandomAroundSolution(
            relation.x_half_m,
            relation.y_half_m,
            relation.z_half_m,
            0.0,
            0.0,
            relation.yaw_half_rad,
        )
    if isinstance(relation, RotateAroundSolution):
        return arena_relations.RotateAroundSolution(relation.roll_rad, relation.pitch_rad, relation.yaw_rad)
    raise TypeError(f"{type(relation).__name__} is not an Arena relation")


def build_surface(surface: Surface):
    """Build the Arena anchor object standing in for the work surface.

    The industrial backgrounds are procedurally spawned cuboids, so Arena's
    ``Object`` cannot derive a bounding box for them and an ``ObjectReference``
    has no USD prim to point at. ``DummyObject`` is the native way to hand the
    solver an explicit box instead, and because the surface is always an anchor
    the placer never writes a position back to it, so it never has to be part of
    the scene.

    Args:
        surface: The declared surface.

    Returns:
        A ``DummyObject`` marked with ``IsAnchor`` and an ``AxisAlignedBoundingBox``
        matching the declared slab.
    """
    from isaaclab_arena.assets.dummy_object import DummyObject
    from isaaclab_arena.relations.relations import IsAnchor as ArenaIsAnchor
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
    from isaaclab_arena.utils.pose import Pose

    # The box is local to the object's own origin, which sits at the surface
    # centre: ``get_world_bounding_box`` translates it by the initial pose, so
    # bounds already in world coordinates would be offset by the centre twice.
    half = tuple(extent / 2.0 for extent in surface.size)
    anchor = DummyObject(
        name=surface.name,
        bounding_box=AxisAlignedBoundingBox(
            min_point=tuple(-extent for extent in half),
            max_point=tuple(half),
        ),
        initial_pose=Pose(position_xyz=surface.center, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)),
    )
    anchor.add_relation(ArenaIsAnchor())
    return anchor


def build_relations(
    layout: SceneLayout,
    objects: dict[str, Any],
    jitter: PoseJitter | None = None,
    exclude_from_jitter: frozenset[str] = frozenset(),
) -> list[Any]:
    """Attach the declared relations to the instantiated scene objects.

    Args:
        layout: The relational description of the scene.
        objects: Scene objects by name, as already instantiated from the task's
            asset table.
        jitter: Pose-jitter half-extents to declare as ``RandomAroundSolution``
            on any placement that is neither anchored nor already carries its
            own marker. Pass ``None`` for a deterministic layout.
        exclude_from_jitter: Scene object names that must keep their solved
            position exactly, typically the domain-randomization preset's pinned
            objects. Excluded names are not given a jitter marker at all, rather
            than being given one and zeroed out, so a mistake here cannot leave a
            fixture quietly randomized.

    Returns:
        The list of objects to hand to
        :func:`~isaaclab_arena.relations.object_placer.ObjectPlacer.place`: the
        work-surface anchor followed by every placed object.

    Raises:
        KeyError: If a placement names an object the scene does not contain, or
            if a scene object has no placement.
    """
    from isaaclab_arena.utils.pose import Pose

    placed = layout.object_names()
    missing = [name for name in placed if name not in objects]
    if missing:
        raise KeyError(f"layout places objects that are not in the scene: {missing}")
    unplaced = [name for name in objects if name not in placed]
    if unplaced:
        raise KeyError(f"scene objects have no placement: {unplaced}")

    surface = build_surface(layout.surface)
    parents = {**objects, layout.surface.name: surface}

    placer_objects: list[Any] = [surface]
    for entry in layout.placements:
        obj = objects[entry.object_name]
        relations = entry.relations
        if entry.pose is not None:
            position, rotation = entry.pose
            obj.set_initial_pose(Pose(position_xyz=tuple(position), rotation_xyzw=tuple(rotation)))
        if (
            jitter is not None
            and not entry.is_anchor
            and entry.object_name not in exclude_from_jitter
            and not _has_jitter_marker(relations)
        ):
            relations = (*relations, jitter.as_marker())
        for relation in relations:
            obj.add_relation(build_relation(relation, parents, surface))
        placer_objects.append(obj)
    return placer_objects


def solve_layout(
    placer_objects: list[Any],
    seed: int | None = None,
    min_separation_m: float = 0.0,
    max_placement_attempts: int = 5,
) -> Any:
    """Run Arena's relation solver over a built layout.

    The solver's own defaults are tuned for debugging -- it narrates every
    hundred iterations and keeps a full position history for plots. Neither is
    wanted when placing a scene as part of building an environment, so both are
    turned off here; the loss is still returned in the result, which is what a
    caller needs to decide whether the layout held.

    Args:
        placer_objects: The object list returned by :func:`build_relations`.
        seed: Seed for the placement RNG, so a rollout is reproducible. ``None``
            leaves the result up to the global RNG.
        min_separation_m: Extra separation required between object bounding boxes
            [m]. A small positive value gives neighbouring props clearance so they
            do not start the episode interpenetrating.
        max_placement_attempts: Placement attempts before the best-so-far result
            is accepted. Retrying a layout that cannot be satisfied only costs
            build time, and the result is applied either way.

    Returns:
        The ``PlacementResult``. ``result.success`` is ``False`` when no attempt
        satisfied every constraint; the best-effort positions are applied either
        way, so the scene is still usable and the caller can report the failure.
    """
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams

    params = ObjectPlacerParams(
        solver_params=RelationSolverParams(verbose=False, save_position_history=False),
        placement_seed=seed,
        min_separation_m=min_separation_m,
        max_placement_attempts=max_placement_attempts,
    )
    return ObjectPlacer(params=params).place(placer_objects)


def _has_jitter_marker(relations: tuple[LayoutRelation, ...]) -> bool:
    """Whether the placement already declares its own reset-time jitter."""
    return any(isinstance(relation, RandomAroundSolution) for relation in relations)


@dataclass(frozen=True)
class PoseJitter:
    """Per-reset pose variation, as half-extents around the solved position.

    Carrying the domain-randomization magnitudes into the layout as an Arena
    ``RandomAroundSolution`` is what makes the jitter part of the placement
    instead of a second pass that could drift away from it.

    Attributes:
        x_half_m: Symmetric position jitter along x [m].
        y_half_m: Symmetric position jitter along y [m].
        z_half_m: Symmetric height jitter [m].
        yaw_half_rad: Symmetric yaw jitter [rad].
    """

    x_half_m: float = 0.0
    y_half_m: float = 0.0
    z_half_m: float = 0.0
    yaw_half_rad: float = 0.0

    @classmethod
    def from_dr_config(cls, cfg) -> "PoseJitter":
        """Build the jitter a domain-randomization config asks for.

        Args:
            cfg: An ``openarm.arena_dr.config.DRConfig``.

        Returns:
            The equivalent :class:`PoseJitter`, or the zero marker when the
            preset disables pose jitter.
        """
        if not getattr(cfg, "enable_pose_jitter", False):
            return cls()
        jitter_x, jitter_y, _ = cfg.position_jitter
        return cls(
            x_half_m=jitter_x,
            y_half_m=jitter_y,
            z_half_m=cfg.height_jitter,
            yaw_half_rad=cfg.yaw_jitter,
        )

    def as_marker(self) -> RandomAroundSolution:
        """Return the Arena marker that declares this jitter."""
        return RandomAroundSolution(self.x_half_m, self.y_half_m, self.z_half_m, self.yaw_half_rad)
