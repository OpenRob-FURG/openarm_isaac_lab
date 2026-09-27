"""Shared machinery for the 50 industrial tasks.

Every industrial task is a :class:`TaskSpec` plus a scene layout. Both are
static data, so all 50 environments share a single environment class and a
single builder; individual environments differ only by which spec they hold.
That keeps the per-task code to the task definition itself, and means a fix to
the builder or the oracle applies to all 50 tasks at once.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field, replace

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from openarm.arena_dr.config import DRConfig
from openarm.arena_tasks.staged.stages import TaskSpec

#: Name of the OpenArm bimanual embodiment asset.
EMBODIMENT = "openarm_bimanual"

#: Background used when a task does not ask for a different one.
DEFAULT_BACKGROUND = "procedural_table"

#: Height of a typical work surface in the world frame, used as the default
#: release height for pick-and-place stages.
TABLE_HEIGHT = 0.05

#: Default episode length [s] for a short assembly task.
SHORT_EPISODE_S = 30.0

#: Default episode length [s] for a task with several sequential stages.
LONG_EPISODE_S = 60.0


@dataclass(frozen=True)
class IndustrialTask:
    """A task specification together with the scene it needs.

    Attributes:
        spec: The declarative task description.
        objects: Mapping of scene instance name to registered asset name.
        background: Registered background asset name.
        object_poses: Optional initial world poses, keyed by scene instance
            name. Anything omitted is placed by the scene's own default.
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
    pinned_objects: tuple[str, ...] = field(default_factory=tuple)
    dr_preset: str = "default"
    primary_object: str = ""

    @property
    def name(self) -> str:
        """The environment name, prefixed to identify the task family."""
        return f"openarm_bimanual_industrial_{self.spec.name}"

    def resolve_primary(self) -> str:
        """Return the primary object name, defaulting to the first target."""
        if self.primary_object:
            return self.primary_object
        for stage in self.spec.stages:
            if stage.target and stage.target in self.objects:
                return stage.target
        return next(iter(self.objects), "")


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
        return asset_cls(instance_name=instance_name)

    obj = asset_cls()
    # The class dropped the kwarg, so it carries the registered asset name.
    # Force the task's scene name anyway: the spec is the source of truth for
    # scene keys, and a task that wants a distinct prim for two instances of
    # the same asset depends on it. prim_path is derived from the name, so it
    # has to be rewritten too or the two would disagree.
    if getattr(obj, "name", None) != instance_name:
        obj.name = instance_name
        obj.prim_path = "{ENV_REGEX_NS}/" + instance_name
    return obj


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
            ``no_cameras``, ``dr_preset``, ``single_arm`` and ``seed``.
        registry: The Arena ``AssetRegistry``.
        hdr_registry: The Arena ``HDRImageRegistry``, used to sample an
            environment map for the dome light.
        dr: Domain-randomization preset to apply. Defaults to the task's own
            preset.

    Returns:
        A configured ``IsaacLabArenaEnvironment`` with an oracle policy
        attached.
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

    background = registry.get_asset_by_name(task.background)()
    objects = []
    for name, asset_name in task.objects.items():
        asset = _instantiate(registry.get_asset_by_name(asset_name), name)
        pose = task.object_poses.get(name)
        if pose is not None:
            asset.set_initial_pose(Pose(position_xyz=pose[0], rotation_xyzw=pose[1]))
        objects.append(asset)

    # Rollout-level randomization: one draw per rollout, applied to every
    # sub-environment. Per-environment variation comes from the event terms.
    rng = random.Random(getattr(args_cli, "seed", None))
    jitter_object_poses(objects, dr)
    randomize_collider_offsets(objects, dr, rng)

    assets = [background, *objects]
    dome_light, hdr_name = build_light(dr, hdr_registry, rng)
    if dome_light is not None:
        assets.insert(0, dome_light)

    # --single-arm is the flag that exists; the dual-arm case is the default
    # when it is absent, so the embodiment stays bimanual for existing callers
    # that never pass arm-mode arguments at all.
    arm_mode = ArmMode.SINGLE_ARM if getattr(args_cli, "single_arm", False) else ArmMode.DUAL_ARM
    # --no_cameras wins over --enable_cameras: it is the documented way to run
    # on a machine without a render device.
    enable_cameras = getattr(args_cli, "enable_cameras", True) and not getattr(args_cli, "no_cameras", False)
    embodiment = registry.get_asset_by_name(EMBODIMENT)(
        enable_cameras=enable_cameras,
        arm_mode=arm_mode,
    )

    scene = Scene(assets=assets)

    primary = next((obj for obj in objects if obj.name == task.resolve_primary()), None)
    task_obj = StagedTask(
        spec=task.spec,
        background=background,
        primary_object=primary,
        episode_length_s=task.spec.episode_length_s,
        enable_cameras=enable_cameras,
    )

    environment = IsaacLabArenaEnvironment(
        name=task.name,
        embodiment=embodiment,
        scene=scene,
        task=task_obj,
        env_cfg_callback=make_env_cfg_callback(dr),
    )
    environment.oracle_policy = StagedOraclePolicy(StagedOraclePolicyCfg(task=task.spec.name))
    return environment


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
