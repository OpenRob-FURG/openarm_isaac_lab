"""Wiring for OpenArm domain randomization.

This module turns a :class:`~openarm.arena_dr.config.DRConfig` into concrete
Isaac Lab configuration. Application happens at three points in the environment
lifecycle:

*Scene build time* (inside the environment's ``get_env``): object pose ranges,
collider offsets, and the dome light with its environment map.
See :func:`jitter_object_poses`, :func:`randomize_collider_offsets` and
:func:`build_light`. Object scale is deliberately not sampled here; it is
handled by a per-environment ``usd``-mode event term in :func:`build_dr_events`,
which is the only way to give each sub-environment a different size.

*Config finalization* (via the ``env_cfg_callback`` hook on
:class:`~isaaclab_arena.environments.isaaclab_arena_environment.IsaacLabArenaEnvironment`):
camera extrinsics and intrinsics, and the global physics timestep and solver
budget. See :func:`make_env_cfg_callback`.

*Runtime* (event terms, see :mod:`openarm.arena_dr.events`): per-env mass,
friction and restitution, robot joint initialization, action latency and noise,
and image photometric augmentation. See :func:`build_dr_events`.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

import torch
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from openarm.arena_dr import events as dr_events
from openarm.arena_dr.config import DRConfig

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnvCfg

DOME_LIGHT_PRIM_PATH = "/World/Light"


def jitter_object_poses(objects: list["ObjectBase"], cfg: DRConfig) -> None:
    """Convert fixed object initial poses into randomized ``PoseRange`` values.

    Only yaw and small position offsets are randomized. Randomizing the full
    orientation would change which face of an object the gripper can reach, so
    it is deliberately excluded: the goal is placement realism, not a different
    task.

    Objects named in :attr:`DRConfig.pinned_objects` keep their fixed pose, as
    do objects whose current pose is already a ``PoseRange`` (for example
    objects positioned by the arena's placement solver) and objects with no
    pose set at all.

    Args:
        objects: Scene objects to randomize.
        cfg: The domain-randomization configuration.
    """
    if not cfg.enable_pose_jitter:
        return

    from isaaclab_arena.utils.pose import PoseRange

    for obj in objects:
        if obj.name in cfg.pinned_objects:
            continue
        current = obj.get_initial_pose()
        if current is None or isinstance(current, PoseRange):
            continue

        position = torch.tensor(current.position_xyz, dtype=torch.float32)
        euler = euler_xyz_from_quat(torch.tensor(current.rotation_xyzw, dtype=torch.float32).unsqueeze(0))

        jitter_x, jitter_y, _ = cfg.position_jitter
        height_jitter = cfg.height_jitter
        yaw_jitter = cfg.yaw_jitter

        position_range = PoseRange(
            position_xyz_min=(
                float(position[0] - jitter_x),
                float(position[1] - jitter_y),
                float(position[2] - height_jitter),
            ),
            position_xyz_max=(
                float(position[0] + jitter_x),
                float(position[1] + jitter_y),
                float(position[2] + height_jitter),
            ),
            rpy_min=(float(euler[0]), float(euler[1]), float(euler[2] - yaw_jitter)),
            rpy_max=(float(euler[0]), float(euler[1]), float(euler[2] + yaw_jitter)),
        )
        # ``set_initial_pose`` rebuilds the object's reset event, which is what
        # turns the PoseRange into a per-reset randomization term.
        obj.set_initial_pose(position_range)


def randomize_collider_offsets(objects: list["ObjectBase"], cfg: DRConfig, rng: random.Random) -> None:
    """Randomize each object's PhysX contact and rest offsets at spawn time.

    Contact and rest offsets are per-collider properties with no global
    equivalent, so they can only be set through the asset's spawn config, and
    only before the simulation starts. One value is drawn per rollout per
    object.

    Args:
        objects: Scene objects to randomize.
        cfg: The domain-randomization configuration.
        rng: Random number generator used for the draws.
    """
    wants_contact = cfg.is_range_active(cfg.contact_offset_scale_range)
    wants_rest = cfg.is_range_active(cfg.rest_offset_scale_range)
    if not (wants_contact or wants_rest):
        return

    for obj in objects:
        if obj.name in cfg.pinned_objects:
            continue
        spawn = getattr(getattr(obj, "object_cfg", None), "spawn", None)
        # contact_offset / rest_offset are CollisionPropertiesCfg fields, not
        # RigidBodyPropertiesCfg fields. Reading them off rigid_props silently
        # found nothing, and assigning there would have raised AttributeError.
        props = getattr(spawn, "collision_props", None)
        if props is None:
            continue
        if wants_contact and props.contact_offset is not None:
            props.contact_offset = float(props.contact_offset) * rng.uniform(*cfg.contact_offset_scale_range)
        if wants_rest and props.rest_offset is not None:
            props.rest_offset = float(props.rest_offset) * rng.uniform(*cfg.rest_offset_scale_range)


def build_light(cfg: DRConfig, hdr_registry: Any, rng: random.Random) -> tuple[Any, str | None]:
    """Create the dome light used for illumination and environment reflections.

    When :attr:`DRConfig.use_hdr` is set, an HDR environment map is sampled
    from the arena's HDR registry, restricted to :attr:`DRConfig.hdr_tags`. The
    ``industrial`` and ``workshop`` tags match an industrial deployment domain;
    ``indoor`` is kept last as a fallback so sampling still succeeds against a
    registry that has been slimmed down.

    Args:
        cfg: The domain-randomization configuration.
        hdr_registry: The arena ``HDRImageRegistry``.
        rng: Random number generator used for the draw.

    Returns:
        Tuple of ``(light, hdr_name)``. ``light`` is a ``light`` asset
        instance, and ``hdr_name`` is the attached environment map, or ``None``
        when no HDR was attached.
    """
    from isaaclab_arena.assets.object_library import DomeLight

    light = DomeLight()
    if not cfg.use_hdr:
        return light, None

    for tag in cfg.hdr_tags:
        try:
            candidates = hdr_registry.get_hdrs_by_tag(tag)
        except Exception:  # pragma: no cover - registry not initialised yet
            candidates = []
        if candidates:
            # The registry hands back HDRImage *classes*; add_hdr needs an
            # instance, so instantiate the one we picked.
            hdr_cls = rng.choice(candidates)
            hdr = hdr_cls()
            light.add_hdr(hdr)
            return light, hdr.name
    return light, None


def build_dr_events(
    cfg: DRConfig,
    object_names: list[str],
    robot_name: str = "robot",
    light_prim_path: str = DOME_LIGHT_PRIM_PATH,
) -> dict[str, EventTermCfg]:
    """Build the runtime (event term) half of the domain randomization.

    Args:
        cfg: The domain-randomization configuration.
        object_names: Names of the rigid objects that participate in physics
            randomization. The background scene and the robot are excluded,
            because randomizing the material of a large static background is
            both expensive and not physically meaningful.
        robot_name: Scene name of the robot articulation.
        light_prim_path: USD path of the dome light prim.

    Returns:
        Mapping of event term name to :class:`EventTermCfg`, to be merged into
        ``env_cfg.events``.
    """
    from isaaclab.envs import mdp as mdp_isaac_lab

    terms: dict[str, EventTermCfg] = {}
    randomize_material = cfg.enable_material and (
        cfg.is_range_active(cfg.static_friction_range)
        or cfg.is_range_active(cfg.dynamic_friction_range)
        or cfg.is_range_active(cfg.restitution_range)
    )

    for index, name in enumerate(object_names):
        if cfg.enable_scale_jitter and cfg.is_range_active(cfg.scale_range):
            # Scale has to be randomized in "usd" mode: it rewrites the
            # xformOp:scale attribute, which the physics engine parses once at
            # startup, so doing it later has no effect. The Arena builder
            # already sets replicate_physics=False, which this requires in order
            # to give each sub-environment its own scale.
            terms[f"dr_scale_{index}"] = EventTermCfg(
                func=mdp_isaac_lab.events.randomize_rigid_body_scale,
                mode="usd",
                params={
                    "asset_cfg": SceneEntityCfg(name),
                    "scale_range": cfg.scale_range,
                },
            )
        if randomize_material:
            terms[f"dr_material_{index}"] = EventTermCfg(
                func=mdp_isaac_lab.events.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(name),
                    "static_friction_range": cfg.static_friction_range,
                    "dynamic_friction_range": cfg.dynamic_friction_range,
                    "restitution_range": cfg.restitution_range,
                    "num_buckets": cfg.num_material_buckets,
                    "make_consistent": True,
                },
            )
        if cfg.enable_mass and cfg.is_range_active(cfg.mass_scale_range):
            terms[f"dr_mass_{index}"] = EventTermCfg(
                func=mdp_isaac_lab.events.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(name),
                    "operation": "scale",
                    "mass_distribution_params": cfg.mass_scale_range,
                    "recompute_inertia": True,
                },
            )

    if cfg.enable_robot_jitter:
        terms["dr_joint_reset"] = EventTermCfg(
            func=mdp_isaac_lab.events.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": cfg.joint_position_offset,
                "velocity_range": cfg.joint_velocity_range,
                "asset_cfg": SceneEntityCfg(robot_name),
            },
        )

    max_delay = int(cfg.action_delay_steps[1])
    # Gate on the maximum, not the minimum. A range of (0, 2) means "sometimes
    # delay, sometimes not", which is exactly the behaviour worth simulating;
    # gating on the lower bound would silently disable action delay for every
    # preset, since every default range starts at zero.
    if cfg.enable_action_delay and max_delay > 0:
        terms["dr_resample_action_delay"] = EventTermCfg(
            func=dr_events.resample_action_delay,
            mode="reset",
            params={"max_delay_steps": max_delay},
        )
        terms["dr_delay_action"] = EventTermCfg(
            func=dr_events.delay_action,
            mode="apply",
            params={"max_delay_steps": max_delay},
        )

    if cfg.enable_action_noise and cfg.action_noise_std > 0.0:
        terms["dr_action_noise"] = EventTermCfg(
            func=dr_events.perturb_action,
            mode="apply",
            params={"action_noise_std": cfg.action_noise_std},
        )

    if cfg.enable_ee_obs_noise and cfg.ee_obs_noise_std > 0.0:
        for arm in ("left", "right"):
            terms[f"dr_obs_noise_{arm}"] = EventTermCfg(
                func=dr_events.randomize_ee_observation,
                mode="apply",
                params={"obs_name": f"policy.{arm}_eef_pos", "noise_std": cfg.ee_obs_noise_std},
            )

    if cfg.enable_photometric:
        terms["dr_sample_photometric"] = EventTermCfg(
            func=dr_events.sample_photometric_params,
            mode="reset",
            params={
                "brightness_range": cfg.brightness_range,
                "contrast_range": cfg.contrast_range,
                "gamma_range": cfg.gamma_range,
                "color_cast_range": cfg.color_cast_range,
                "pixel_noise_std": cfg.pixel_noise_std,
            },
        )

    if cfg.enable_light:
        terms["dr_light"] = EventTermCfg(
            func=dr_events.randomize_light_appearance,
            mode="startup",
            params={
                "prim_path": light_prim_path,
                "intensity_range": cfg.light_intensity_range,
                "color_temp_range": cfg.light_color_temp_range,
            },
        )

    return terms


def _sample_physics(env_cfg: "IsaacLabArenaManagerBasedRLEnvCfg", cfg: DRConfig, rng: random.Random) -> None:
    """Randomize the global timestep and solver budget, holding control rate fixed."""
    control_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    new_dt = rng.uniform(*cfg.physics_dt_range)
    # Holding the control rate fixed is essential: changing dt alone would
    # silently change the task's action rate and invalidate the oracle policies.
    new_decimation = max(1, int(round(control_dt / new_dt)))
    env_cfg.decimation = new_decimation
    env_cfg.sim.dt = control_dt / new_decimation

    iterations = rng.randint(*cfg.solver_iteration_range)
    physics = env_cfg.sim.physics
    if physics is None:
        return
    for field in ("min_position_iteration_count", "min_velocity_iteration_count"):
        if hasattr(physics, field):
            setattr(physics, field, int(iterations))
    for field in ("max_position_iteration_count", "max_velocity_iteration_count"):
        if hasattr(physics, field):
            # The max is the per-actor upper bound; keep it at or above the min.
            setattr(physics, field, max(int(iterations), int(getattr(physics, field))))


def make_env_cfg_callback(
    cfg: DRConfig,
    camera_name: str = "top_camera",
    rng: random.Random | None = None,
) -> Any:
    """Build the ``env_cfg_callback`` that applies global (per-rollout) randomization.

    Camera extrinsics and intrinsics and the physics timestep cannot vary per
    environment, so they are drawn once here.

    Args:
        cfg: The domain-randomization configuration.
        camera_name: Name of the camera field in the scene config. The camera
            is only present when the embodiment was built with
            ``enable_cameras=True``, so this lookup is allowed to miss.
        rng: Random number generator. One is created if omitted.

    Returns:
        A callable suitable for the ``env_cfg_callback`` argument of
        :class:`~isaaclab_arena.environments.isaaclab_arena_environment.IsaacLabArenaEnvironment`.
    """
    generator = rng or random.Random()

    def _callback(env_cfg: "IsaacLabArenaManagerBasedRLEnvCfg") -> "IsaacLabArenaManagerBasedRLEnvCfg":
        camera = getattr(env_cfg.scene, camera_name, None)

        if camera is not None and cfg.enable_camera_extrinsics:
            offset = camera.offset
            jx, jy, jz = cfg.camera_position_jitter
            offset.pos = (
                float(offset.pos[0]) + generator.uniform(-jx, jx),
                float(offset.pos[1]) + generator.uniform(-jy, jy),
                float(offset.pos[2]) + generator.uniform(-jz, jz),
            )
            if cfg.camera_rotation_jitter > 0.0:
                rot = offset.rot
                offset.rot = (
                    float(rot[0]) + generator.uniform(-cfg.camera_rotation_jitter, cfg.camera_rotation_jitter),
                    float(rot[1]) + generator.uniform(-cfg.camera_rotation_jitter, cfg.camera_rotation_jitter),
                    float(rot[2]) + generator.uniform(-cfg.camera_rotation_jitter, cfg.camera_rotation_jitter),
                    float(rot[3]),
                )

        if camera is not None and cfg.enable_camera_intrinsics:
            spawn = camera.spawn
            if hasattr(spawn, "focal_length"):
                spawn.focal_length = generator.uniform(*cfg.focal_length_range)
            if hasattr(spawn, "horizontal_aperture"):
                spawn.horizontal_aperture = generator.uniform(*cfg.horizontal_aperture_range)
            if hasattr(spawn, "focus_distance"):
                spawn.focus_distance = generator.uniform(*cfg.focus_distance_range)

        if cfg.enable_physics:
            _sample_physics(env_cfg, cfg, generator)

        return env_cfg

    return _callback
