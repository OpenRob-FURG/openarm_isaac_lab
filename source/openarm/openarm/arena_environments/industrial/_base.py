"""Shared machinery for the 50 industrial tasks.

Every industrial task is a :class:`TaskSpec` plus a scene layout. Both are
static data, so all 50 environments share a single environment class and a
single builder; individual environments differ only by which spec they hold.
That keeps the per-task code to the task definition itself, and means a fix to
the builder or the oracle applies to all 50 tasks at once.

The layout is semantic rather than numeric. A task says what is *on the work
surface*, what is *next to* what, and which receptacle sits where its goal
region says it is; the coordinates themselves come out of Arena's own
:class:`~isaaclab_arena.relations.object_placer.ObjectPlacer`, driven by the
relation vocabulary in :mod:`openarm.arena_tasks.staged.layout`. See
:func:`derive_layout` for how a task that does not spell its layout out gets
one, and :func:`build_environment` for the solve itself.

Passing ``--rl-training`` reconfigures the same tasks for reinforcement
learning: it attaches the dense reward manager and privileged critic group that
:class:`~openarm.arena_tasks.staged.task.StagedTask` derives from the same task
spec, and the RSL-RL runner config in :mod:`rsl_rl_cfg` that splits the actor
and critic observations. See that module for why the split is asymmetric.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field, replace

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from openarm.arena_dr.config import DRConfig
from openarm.arena_tasks.staged.layout import (
    AtPosition,
    IsAnchor,
    NextTo,
    On,
    Placement,
    PositionLimits,
    PoseJitter,
    SceneLayout,
    Side,
    Surface,
    build_relations,
    solve_layout,
)
from openarm.arena_tasks.staged.stages import TaskSpec

#: Name of the OpenArm bimanual embodiment asset.
EMBODIMENT = "openarm_bimanual"

#: Background used when a task does not ask for a different one.
DEFAULT_BACKGROUND = "procedural_table"

# -- robot workspace -------------------------------------------------------
#
# The OpenArm bimanual USD puts the shoulder mounts at z=0.698 m and z=+-0.031 m
# in y, with a 0.8167 m link chain from each mount to its ``ee_tcp``. The robot
# root sits at the world origin, so the reachable sphere around either shoulder
# is fixed by the asset, not by the environment. A work surface therefore has to
# be placed inside that sphere: at a 0.05 m table the arms cannot even reach
# the objects, let alone place them.
#
# At TABLE_HEIGHT below, the furthest goal region in the whole task family sits
# 0.625 m from its nearest shoulder, which is 0.19 m inside the true reach and
# leaves room for a top-down grasp.
SHOULDER_HEIGHT = 0.698
SHOULDER_REACH = 0.8167

#: Height of the work surface in the world frame. Also the default release
#: height for pick-and-place stages, and the base every task's goal-region
#: heights are derived from.
TABLE_HEIGHT = 0.35

#: X of the row of manipulated objects, in front of the robot and inside the
#: arm's reach. The reset wrists sit near x=0.41 m; placing props at x=0.30 m
#: puts them in the open fingers before a policy starts and makes even a tiny
#: reset settling motion knock the first target away. This offset leaves a
#: clear approach corridor while remaining within the arm's stable workspace.
WORK_X = 0.41

#: X and Y of the row of objects the robot never moves but that no goal region
#: accounts for. These sit in the back-right corner, clear of both the goal
#: regions (the receptacle band on the robot's right) and the manipulated row
#: down the middle of the table. The corner is bounded by the table edge at
#: x=0.88: the widest prop that lands here is a 0.31 m bin, so x has to stay
#: under 0.72 or the prop overhangs the slab and Arena's ``On`` validation
#: rejects the whole layout. Likewise y=0.60 keeps a parked bin clear of the
#: topmost receptacle at y=0.34.
PARK_X = 0.70
PARK_Y = 0.60

#: Centre of the table. The plate is 0.8 x 1.5 x 0.04, so its top ends up
#: TABLE_HEIGHT above the floor. Its near edge starts at x=0.38, clear of the
#: reset wrists while still supporting the can tower region at x=0.42.
TABLE_CENTER = (0.48, 0.0, TABLE_HEIGHT - 0.02)

#: Size of the work surface slab, matching Arena's ``ProceduralTable``.
TABLE_SIZE = (0.8, 1.5, 0.04)

#: Name of the work surface anchor, which relations refer to as "the surface".
SURFACE_NAME = "work_surface"

#: Substrings marking an object as something a goal region can be "inside", so
#: the layout anchors the receptacle rather than the distractors that happen to
#: share its task.
RECEPTACLE_HINTS = (
    "bin",
    "box",
    "crate",
    "bag",
    "bowl",
    "tray",
    "shelf",
    "rack",
    "pallet",
    "carton",
    "machine",
    "freezer",
    "source",
    "mixing",
    "base",
    "plate",
    "mug",
    "cup",
)

#: Default episode length [s] for a short assembly task.
SHORT_EPISODE_S = 30.0

#: Default episode length [s] for a task with several sequential stages.
LONG_EPISODE_S = 60.0

#: Spacing between neighbouring items in a row of props [m]. Wide enough that two
#: neighbouring bounding boxes cannot touch, which is what lets the solver treat
#: a row as a chain of ``NextTo`` relations and still validate it.
ROW_SPACING = 0.24

#: Gap left between a prop's underside and the work surface when it is placed
#: with an ``On`` relation [m]. Small but positive, so the prop starts clear of
#: the slab rather than inside it and drops the last fraction of a millimetre.
SURFACE_CLEARANCE = 0.005

#: Placement attempts before the solver gives up on a fully valid layout. A
#: layout that cannot be satisfied is applied best-effort and reported, not
#: retried forever, because a usable-but-imperfect scene beats no scene.
MAX_PLACEMENT_ATTEMPTS = 5


@dataclass(frozen=True)
class IndustrialTask:
    """A task specification together with the scene it needs.

    Attributes:
        spec: The declarative task description.
        objects: Mapping of scene instance name to registered asset name.
        background: Registered background asset name.
        object_poses: Optional initial world poses, keyed by scene instance
            name. An object listed here is anchored at that pose rather than
            solved, which is the way to state a position the task insists on.
            Anything omitted is placed by the layout.
        layout: Explicit relational layout, taking precedence over the spec's own
            layout. ``None`` uses ``spec.layout`` if it declares one, and
            otherwise derives one; see :func:`derive_layout`.
        pinned_objects: Objects excluded from randomization. Fixtures and
            receptacles belong here, otherwise domain randomization slides the
            target out from under the goal region.
        dr_preset: Default domain-randomization preset for the task.
        primary_object: Scene name of the main manipulated object, used for the
        contact sensor and the object-pose observation. Defaults to the
        first manipulated object in the spec.
    """

    spec: TaskSpec
    objects: dict[str, str] = field(default_factory=dict)
    background: str = DEFAULT_BACKGROUND
    object_poses: dict[str, tuple] = field(default_factory=dict)
    layout: SceneLayout | None = None
    pinned_objects: tuple[str, ...] = field(default_factory=tuple)
    dr_preset: str = "light"
    primary_object: str = ""

    @property
    def name(self) -> str:
        """The environment name, prefixed to identify the task family."""
        return f"openarm_bimanual_industrial_{self.spec.name}"

    def resolve_layout(self) -> SceneLayout:
        """Return the relational layout for this task.

        A task that spells its layout out gets exactly that, whether the layout
        is attached to the task or declared on its spec; every other task gets
        one derived from its stages and goals, so the two paths cannot drift
        apart in the vocabulary they use.

        Returns:
            The :class:`SceneLayout` describing where every scene object goes.
        """
        if self.layout is not None:
            return self.layout
        if self.spec.layout is not None:
            return self.spec.layout
        return derive_layout(self)


    def resolve_primary(self) -> str:
        """Return the primary object name, defaulting to the first target."""
        if self.primary_object:
            return self.primary_object
        for stage in self.spec.stages:
            if stage.target and stage.target in self.objects:
                return stage.target
        return next(iter(self.objects), "")


def work_surface() -> Surface:
    """Return the work surface every industrial scene is laid out on.

    The robot's reach is what fixes this: at a 0.05 m table the arms cannot even
    reach the objects, so the surface has to sit inside the sphere the OpenArm
    asset defines, and its size has to match the Arena background that provides
    the actual collision surface.

    Returns:
        The :class:`Surface` describing the slab.
    """
    return Surface(name=SURFACE_NAME, center=TABLE_CENTER, size=TABLE_SIZE)


def _region_centres(task: IndustrialTask) -> list[tuple[float, float, float]]:
    """Return every absolute target-box centre the task declares.

    Goal regions are the only goals written in absolute world coordinates; the
    joint goals are expressed in normalized fixture coordinates instead. They
    therefore double as the authoritative description of where a receptacle is
    meant to be, which :func:`derive_layout` relies on to keep the scene and
    the goals describing the same world.

    Args:
        task: The task to inspect.

    Returns:
        The centres, in declaration order, flattened across nested composites.
    """
    found: list[tuple[float, float, float]] = []

    def walk(goal) -> None:
        if goal is None:
            return
        centre = getattr(goal, "center", None)
        if centre is not None:
            found.append(tuple(float(value) for value in centre))
        for child in goal.nested_goals():
            walk(child)

    for stage in task.spec.stages:
        walk(stage.goal)
    return found


def _on_surface(
    object_name: str,
    surface: Surface,
    x: float | None = None,
    y: float | None = None,
) -> Placement:
    """Build the placement for a prop resting on the work surface.

    Every derived placement is the same three relations: rest on the surface so
    the prop's height comes from its own bounding box, stay within the surface
    footprint so it cannot end up off the edge or out of the arms' reach, and
    optionally sit at a stated ``(x, y)`` where the task needs an absolute
    location. The surface is the only parent, so nothing here can form a cycle
    the solver would have to untangle.

    Args:
        object_name: Scene instance name.
        surface: The work surface to place on.
        x: World x [m], or ``None`` to let the solver choose.
        y: World y [m], or ``None`` to let the solver choose.

    Returns:
        The :class:`Placement`.
    """
    x_min, x_max, y_min, y_max = surface.extents()
    relations: list = [
        On(clearance_m=SURFACE_CLEARANCE),
        PositionLimits(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max),
    ]
    # Arena's AtPosition asserts that at least one axis is given, so a placement
    # that is only positioned relative to its neighbours must not carry one.
    if x is not None or y is not None:
        relations.append(AtPosition(x=x, y=y))
    return Placement(object_name=object_name, relations=tuple(relations))


def derive_layout(task: IndustrialTask) -> SceneLayout:
    """Build a relational layout from the task's stages and goals.

    No industrial task states its layout by hand, so this derives one from the
    task instead. The derivation is a statement about *roles*, and every role
    becomes a relation rather than a coordinate:

    * a prop with an explicit :attr:`IndustrialTask.object_poses` entry is an
      ``IsAnchor``: the task insists on that pose, so the solver leaves it be;
    * a receptacle is ``On`` the surface and ``AtPosition`` the goal region it
      serves, matched in declaration order, so the goal describes a receptacle
      the scene actually has. Receptacles are recognised by name because a
      distractor can share a task with the bin the goal is about;
    * the objects the robot manipulates form a front row, ordered by the stages
      that consume them, so the first stage is closest to the robot;
    * whatever is left over is parked to the robot's right as a ``NextTo``
      chain, so the row's shape and spacing are stated once and the solver
      places the members.

    The result is ordinary declarative data; resolving it into Arena relations
    and solving it happens in :func:`build_environment`.

    Args:
        task: The task to lay out.

    Returns:
        The derived :class:`SceneLayout`.
    """
    surface = work_surface()
    centres = _region_centres(task)
    manipulated = set(task.spec.object_names)
    placements: list[Placement] = []
    placed: set[str] = set()

    for name, pose in task.object_poses.items():
        placements.append(Placement(object_name=name, relations=(IsAnchor(),), pose=pose))
        placed.add(name)

    remaining = [name for name in task.objects if name not in placed]

    # Objects the robot never moves are the receptacles and fixtures. Anchor
    # them to the regions they serve, matched in declaration order, so the goal
    # describes a box the scene actually has. Only the region's x and y are
    # pinned: the height follows from the surface, so a bin rests on the table
    # instead of floating at the height its contents have to reach.
    spare = [name for name in remaining if name not in manipulated]
    receptacles = [name for name in spare if any(hint in name.lower() for hint in RECEPTACLE_HINTS)]
    for name, centre in zip(receptacles, centres):
        placements.append(_on_surface(name, surface, x=centre[0], y=centre[1]))
        placed.add(name)

    # The manipulated row is centred on the arm's line. The first object's arm
    # decides which side of the robot it starts on, and each subsequent object
    # is stated relative to the previous one, so the row keeps its shape even if
    # the table moves.
    front = [name for name in remaining if name in manipulated and name not in placed]
    first_stage = next((stage for stage in task.spec.stages if stage.target in front), None)
    first_arm = getattr(first_stage, "arm", "right")
    first_y = -0.153 if first_arm == "right" else 0.153 if first_arm == "left" else -0.06 * (len(front) - 1)
    previous = ""
    for index, name in enumerate(front):
        if index == 0:
            placements.append(_on_surface(name, surface, x=WORK_X, y=first_y))
        else:
            placements.append(
                _on_surface(name, surface).with_relations(
                    (NextTo(previous, side=Side.POSITIVE_Y, distance_m=ROW_SPACING),)
                )
            )
        previous = name
        placed.add(name)

    # Whatever is left is scenery: a row marching along the back-right corner,
    # far enough out that the front row and the goal regions stay clear.
    back = [name for name in remaining if name not in placed]
    previous = ""
    for index, name in enumerate(back):
        if index == 0:
            placements.append(_on_surface(name, surface, x=PARK_X, y=PARK_Y))
        else:
            placements.append(
                _on_surface(name, surface).with_relations(
                    (NextTo(previous, side=Side.NEGATIVE_Y, distance_m=ROW_SPACING),)
                )
            )
        previous = name

    return SceneLayout(surface=surface, placements=tuple(placements))


def _install_analytic_bounding_box(obj) -> None:
    """Give a spawner-backed asset the local bounding box Arena will not.

    ``Object.get_bounding_box`` measures the asset's USD prim and so asserts that
    the asset has a USD path. A ``SPAWNER`` asset is built from a simulator cfg
    instead and has no path, which makes it unmeasurable -- but Arena's relation
    solver needs a local box for every object it places, so a task using one dies
    inside the placer with an ``AssertionError``. For the primitive spawners the
    suite uses, the box follows from the cfg, so derive it and install it on the
    instance. USD-backed assets are left alone: measuring the real prim is always
    more accurate than a guess.

    Args:
        obj: A freshly instantiated scene asset.

    Raises:
        ValueError: If a spawner-backed asset has a shape that cannot be
            measured, so the failure names the asset instead of surfacing as an
            assertion deep inside the placer.
    """
    if getattr(obj, "usd_path", None) is not None:
        return
    cfg = getattr(obj, "spawner_cfg", None)
    radius = getattr(cfg, "radius", None)
    if radius is None:
        raise ValueError(
            f"asset '{getattr(obj, 'name', obj)}' has no USD path and its spawner cfg exposes no "
            "measurable extent, so the relation solver cannot place it"
        )
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox

    scale = getattr(obj, "scale", None) or (1.0, 1.0, 1.0)
    # A sphere's AABB is exactly the cube of its diameter, so this is tight.
    half = tuple(float(radius) * float(axis) for axis in scale)
    box = AxisAlignedBoundingBox(min_point=tuple(-h for h in half), max_point=half)
    obj.bounding_box = box
    # The assert lives inside the method, so the cached box alone would not be
    # reached; shadow it on the instance for the solver's calls to find.
    obj.get_bounding_box = lambda: box


def _instantiate(asset_cls: type, instance_name: str):
    """Construct an asset, naming it ``instance_name`` where that is possible.

    ``LibraryObject`` accepts ``instance_name``, but a number of Arena's own
    subclasses (``RedCube``, ``RedContainer``, ...) override ``__init__`` and
    drop the parameter, so passing it unconditionally raises ``TypeError``.
    Those classes fall back to their registered name, so the name is rewritten
    afterwards -- together with ``prim_path``, which is derived from it. That
    also covers a task needing two distinct prims for the same asset, which is
    what a rewritten name gives for free.

    Args:
        asset_cls: The registered asset class.
        instance_name: The desired scene instance name.

    Returns:
        The constructed asset.
    """
    import inspect

    try:
        params = inspect.signature(asset_cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        params = {}
    if "instance_name" in params:
        obj = asset_cls(instance_name=instance_name)
    else:
        obj = asset_cls()
        # The class dropped the kwarg, so it carries the registered asset name.
        # Force the task's scene name anyway: the spec is the source of truth for
        # scene keys, and a task that wants a distinct prim for two instances of
        # the same asset depends on it. prim_path is derived from the name, so it
        # has to be rewritten too or the two would disagree.
        if getattr(obj, "name", None) != instance_name:
            obj.name = instance_name
            obj.prim_path = "{ENV_REGEX_NS}/" + instance_name
    _install_analytic_bounding_box(obj)
    return obj


def _layout_jitter(dr: DRConfig) -> PoseJitter | None:
    """Return the per-reset jitter to declare as part of the layout.

    The magnitude is the domain-randomization preset's, so the preset still
    decides how much variation a task gets; what changes is that the jitter is
    declared around the *solved* position as an Arena ``RandomAroundSolution``
    rather than applied to it afterwards.

    Args:
        dr: The domain-randomization preset in force.

    Returns:
        The jitter to declare, or ``None`` when the preset disables pose jitter.
    """
    jitter = PoseJitter.from_dr_config(dr)
    return jitter if jitter.as_marker().is_active else None


def _solve(placer_objects: list, task: IndustrialTask, seed: int | None) -> None:
    """Run Arena's relation solver and report a layout it could not satisfy.

    A failed validation is not fatal: the placer applies its best attempt either
    way, so the scene is still playable. It is worth saying out loud though,
    because the usual cause is two props whose bounding boxes genuinely cannot
    both fit where the task wants them, and that is a task bug rather than a
    transient one.

    Args:
        placer_objects: Object list from
            :func:`~openarm.arena_tasks.staged.layout.build_relations`.
        task: The task being laid out.
        seed: Seed for the placement RNG, for reproducible rollouts.
    """
    result = solve_layout(
        placer_objects,
        seed=seed,
        max_placement_attempts=MAX_PLACEMENT_ATTEMPTS,
    )
    if not result.success:
        print(
            f"[{task.spec.name}] relational layout did not satisfy every constraint after "
            f"{result.attempts} attempts (loss {result.final_loss:.4g}); using the best attempt."
        )


def build_environment(
    task: IndustrialTask,
    args_cli: argparse.Namespace,
    registry,
    hdr_registry=None,
    dr: DRConfig | None = None,
):
    """Instantiate the Arena environment for one industrial task.

    All heavyweight imports happen here rather than at module scope, because
    this module is imported by the CLI parser before the Isaac Sim
    application has been launched.

    Args:
        task: The task to build.
        args_cli: Parsed command line arguments. Read for ``enable_cameras``,
            ``no_cameras``, ``dr_preset``, ``single_arm``, ``rl_training`` and
            ``seed``.
        registry: The Arena ``AssetRegistry``.
        hdr_registry: The Arena ``HDRImageRegistry``, used to sample an
            environment map for the dome light.
        dr: Domain-randomization preset to apply. Defaults to the task's own
            preset.

    Returns:
        A configured ``IsaacLabArenaEnvironment`` with an oracle policy
        attached, and an RSL-RL runner config when ``--rl-training`` is set.
    """
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    # Importing these modules is what *registers* the OpenArm assets and the
    # embodiment with the Arena registries. Without them the registry only
    # knows about the stock Arena components, and every OpenArm-specific name
    # (bolt_m16, nut_m16, peg, hole, openarm_bimanual, ...) fails to resolve.
    # The `as _registered` alias states the side-effect-only intent in a way
    # linters accept without a blanket noqa.
    import openarm.assets as _registered_assets
    import openarm.embodiments as _registered_embodiments

    del _registered_assets, _registered_embodiments
    from openarm.arena_tasks.staged import StagedTask

    from openarm.arena_dr.applier import (
        build_light,
        jitter_object_poses,
        make_env_cfg_callback,
        randomize_collider_offsets,
    )
    from openarm.policies.oracle.staged_oracle import StagedOraclePolicy, StagedOraclePolicyCfg

    preset = getattr(args_cli, "dr_preset", None) or task.dr_preset
    dr = dr or DRConfig.named(preset)
    # Pinned fixtures and receptacles are excluded from randomization: sliding
    # the target out from under a goal region would make a task unsolvable
    # rather than merely harder.
    dr = replace(dr, pinned_objects=tuple({*dr.pinned_objects, *task.pinned_objects}))

    # One flag turns the whole RL path on or off, because the three things it
    # changes belong together: a reward manager and a privileged critic group
    # are both pure overhead for the scripted oracle, and the runner config is
    # useless without them.
    rl_training = bool(getattr(args_cli, "rl_training", True))

    seed = getattr(args_cli, "seed", None)
    background = registry.get_asset_by_name(task.background)()
    # The table is what puts the task inside the robot's workspace, so it is
    # placed from the asset's own geometry rather than left at the origin. This
    # is the same slab ``work_surface`` describes, which is what lets the layout
    # solve against it.
    background.set_initial_pose(Pose(position_xyz=TABLE_CENTER, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    objects: dict[str, object] = {}
    object_list = []
    for name, asset_name in task.objects.items():
        asset = _instantiate(registry.get_asset_by_name(asset_name), name)
        objects[name] = asset
        object_list.append(asset)

    # -- the semantic layout: relations go on, then Arena's own solver places
    # everything from the assets' real bounding boxes.
    placer_objects = build_relations(
        task.resolve_layout(),
        objects,
        _layout_jitter(dr),
        exclude_from_jitter=frozenset(dr.pinned_objects),
    )
    _solve(placer_objects, task, seed)
    # Placement for an industrial task is solved just above, against a
    # synthetic ``DummyObject`` work-surface anchor that has no scene prim.
    # Arena's builder solves relations a second time when it compiles the
    # scene, but it can only see real scene prims, so that pass would abort
    # with "No anchor object found" now that every object carries relations.
    # The solved poses are authoritative, so switch the redundant pass off.
    # ``args_cli`` is a plain Namespace shared with the builder that is built
    # from it, which is how the flag reaches the builder.
    args_cli.solve_relations = False

    # Rollout-level randomization: one draw per rollout, applied to every
    # sub-environment. Per-environment variation comes from the event terms.
    # ``jitter_object_poses`` only has anything left to do for an object the
    # layout anchored, because a solved placement already carries its own
    # ``PoseRange``.
    rng = random.Random(seed)
    jitter_object_poses(object_list, dr)
    randomize_collider_offsets(object_list, dr, rng)

    assets = [background, *object_list]
    dome_light, hdr_name = build_light(dr, hdr_registry, rng)
    if dome_light is not None:
        assets.insert(0, dome_light)

    # --single-arm is the flag that exists; the dual-arm case is the default
    # when it is absent, so the embodiment stays bimanual for existing callers
    # that never pass arm-mode arguments at all.
    arm_mode = ArmMode.SINGLE_ARM if getattr(args_cli, "single_arm", False) else ArmMode.DUAL_ARM
    # --no-cameras wins over --enable-cameras: it is the documented way to run
    # on a machine without a render device. Training from privileged state
    # ignores the camera entirely, so a training run has no reason to pay for
    # one.
    enable_cameras = getattr(args_cli, "enable_cameras", True) and not getattr(args_cli, "no_cameras", False)
    if rl_training:
        enable_cameras = False
    embodiment = registry.get_asset_by_name(EMBODIMENT)(
        enable_cameras=enable_cameras,
        arm_mode=arm_mode,
    )

    scene = Scene(assets=assets)

    primary = objects.get(task.resolve_primary())

    task_obj = StagedTask(
        spec=task.spec,
        background=background,
        primary_object=primary,
        episode_length_s=task.spec.episode_length_s,
        enable_cameras=enable_cameras,
        enable_rewards=rl_training,
        # The full object list, not the spec's manipulated subset: the critic
        # needs the position of every prop, including the ones a policy is
        # supposed to leave alone.
        scene_objects=list(task.objects),
    )

    environment = IsaacLabArenaEnvironment(
        name=task.name,
        embodiment=embodiment,
        scene=scene,
        task=task_obj,
        env_cfg_callback=make_env_cfg_callback(dr),
    )
    if rl_training:
        _attach_rl_config(environment, task)
    environment.oracle_policy = StagedOraclePolicy(StagedOraclePolicyCfg(task=task.spec.name))
    return environment


def _attach_rl_config(environment, task: IndustrialTask) -> None:
    """Give an Arena environment the RSL-RL runner config the train script reads.

    The runner config is attached two ways because two readers want it.
    ``scripts/reinforcement_learning/rsl_rl/train.py`` reads
    ``arena_env.rl_config`` directly, while anything that goes through the Gym
    registry resolves ``rsl_rl_cfg_entry_point`` from the registration kwargs.
    Setting both means the same environment trains whichever path reaches it,
    and the two cannot end up describing different runs.

    Args:
        environment: The Arena environment to configure.
        task: The task it was built for, used to name the run.
    """
    from openarm.arena_environments.industrial.rsl_rl_cfg import IndustrialStagedRslRlRunnerCfg

    agent_cfg = IndustrialStagedRslRlRunnerCfg(run_name=task.spec.name)
    environment.rl_config = agent_cfg
    environment.rl_framework_entry_point = "rsl_rl_cfg_entry_point"
    environment.rl_policy_cfg = f"{IndustrialStagedRslRlRunnerCfg.__module__}:IndustrialStagedRslRlRunnerCfg"


class IndustrialEnvironment(ExampleEnvironmentBase):
    """Environment class for a single named industrial task.

    Subclasses only set :attr:`task_key`; the scene, task, oracle and domain
    randomization are all derived from the registry entry.
    """

    #: Key into the :data:`INDUSTRIAL_TASKS` mapping.
    task_key: str = ""

    def get_env(self, args_cli: argparse.Namespace):
        """Build the environment for this task.

        Args:
            args_cli: Parsed command line arguments.

        Returns:
            The configured ``IsaacLabArenaEnvironment``.

        Raises:
            KeyError: If :attr:`task_key` is not registered.
        """
        from openarm.arena_environments.industrial import INDUSTRIAL_TASKS

        task = INDUSTRIAL_TASKS.get(self.task_key)
        if task is None:
            raise KeyError(f"Unknown industrial task '{self.task_key}'. Known: {sorted(INDUSTRIAL_TASKS)}")
        return build_environment(task, args_cli, self.asset_registry, self.hdr_registry)

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Register the domain-randomization options shared by all tasks.

        Args:
            parser: The subparser for this environment.

        Returns:
            The same parser, for chaining.
        """
        group = parser.add_argument_group("domain randomization")
        group.add_argument(
            "--dr-preset",
            type=str,
            default=None,
            choices=["none", "light", "default", "heavy"],
            help="Domain-randomization preset. Defaults to the task's own preset.",
        )
        group.add_argument(
            "--no-cameras",
            action="store_true",
            help="Disable the top-down camera. Required on machines without a GPU.",
        )
        group.add_argument(
            "--single-arm",
            action="store_true",
            help="Run the task with a single arm instead of the bimanual default.",
        )
        group.add_argument(
            "--rl-training",
            action="store_true",
            help="Configure the task for RSL-RL PPO training on privileged state. Adds a dense "
            "reward manager and a privileged critic observation group, attaches the runner config, "
            "and disables the camera. Omit it to run the scripted oracle, where both would be "
            "computed and never read.",
        )
        return parser


def make_environment_class(task_key: str) -> type[IndustrialEnvironment]:
    """Build an environment class bound to one task key.

    Args:
        task_key: The key in the :data:`INDUSTRIAL_TASKS` mapping.

    Returns:
        A new :class:`IndustrialEnvironment` subclass.
    """
    return type(
        f"Industrial{''.join(part.capitalize() for part in task_key.split('_'))}Environment",
        (IndustrialEnvironment,),
        {"task_key": task_key, "name": f"openarm_bimanual_industrial_{task_key}", "__doc__": f"Industrial task '{task_key}'."},
    )
