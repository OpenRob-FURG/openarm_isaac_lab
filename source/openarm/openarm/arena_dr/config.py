"""Domain-randomization configuration for OpenArm bimanual sim-to-real training.

The configuration is split into two tiers based on what Isaac Sim actually
supports in this arena setup:

*Global (one sample per rollout).* Isaac Sim creates a single dome light on a
global prim (``/World/Light``, see
:class:`isaaclab_arena.assets.object_library.DomeLight`) and a single set of
camera intrinsics per environment config. These quantities are shared by every
parallel environment, so they cannot vary per-env. They are resampled on every
rollout, which is the correct semantics for "which warehouse is the robot in
today" domain variation.

*Per-env (resampled on every episode reset).* Object geometry and placement,
physics material parameters, robot joint initialisation, actuator/action
disturbances and photometric image augmentation. These use per-env event terms
and therefore vary across the parallel environments within a single rollout.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# Number of photometric parameters sampled per environment. Kept in sync with
# :data:`openarm.arena_dr.events.PHOTOMETRIC_PARAM_COUNT`.
PHOTOMETRIC_PARAM_COUNT = 5


@dataclass(frozen=True)
class DRConfig:
    """Domain-randomization ranges.

    All ranges are ``(min, max)`` tuples. A range whose two entries are equal
    disables that particular randomization, so a single config object can span
    everything from fully deterministic (``DRConfig.none()``) to heavily
    randomized (``DRConfig.heavy()``) without any ``if`` flags.

    Attributes:
        enable_pose_jitter: Randomize object initial position and yaw inside a
            ``PoseRange`` instead of using a fixed ``Pose``.
        position_jitter: Symmetric position jitter [m] applied per axis.
        yaw_jitter: Symmetric yaw (rotation about world Z) jitter [rad].
        height_jitter: Symmetric height jitter [m]. Kept small by default since
            large height offsets change reachability, not just appearance.
        enable_scale_jitter: Randomize object scale.
        scale_range: Uniform object scale range (multiplicative).
        enable_material: Randomize friction and restitution.
        static_friction_range: Static friction coefficient range.
        dynamic_friction_range: Dynamic friction coefficient range.
        restitution_range: Restitution (bounciness) range.
        num_material_buckets: Number of discretized material presets. PhysX
            allows only a limited number of unique materials, so this is kept
            small.
        enable_mass: Randomize object mass.
        mass_scale_range: Mass multiplier range applied on top of the authored
            mass.
        enable_robot_jitter: Randomize the robot joint positions on reset.
        joint_position_offset: Joint position offset range [rad].
        joint_velocity_range: Joint velocity range [rad/s] applied on reset.
        enable_action_delay: Simulate sensing/actuation latency by delaying the
            action buffer.
        action_delay_steps: Inclusive range of delayed control steps.
        enable_action_noise: Additive Gaussian noise on the commanded action.
        action_noise_std: Action noise standard deviation. Interpreted in the
            same units as the action term (m/rad for the delta-EE command).
        enable_ee_obs_noise: Additive Gaussian noise on proprioceptive
            end-effector observations.
        ee_obs_noise_std: End-effector observation noise standard deviation [m].
        enable_light: Randomize the dome light intensity and colour temperature.
        light_intensity_range: Dome light intensity multiplier range.
        light_color_temp_range: Blackbody colour temperature range [K]. The
            lower bound is warm/tungsten, the upper bound is cool/daylight.
        enable_camera_extrinsics: Randomize the camera mount offset.
        camera_position_jitter: Camera mount position jitter [m] per axis.
        camera_rotation_jitter: Camera mount rotation jitter [rad].
        enable_camera_intrinsics: Randomize focal length and aperture.
        focal_length_range: Pinhole focal length range [mm].
        horizontal_aperture_range: Horizontal aperture range [mm].
        focus_distance_range: Focus distance range [m].
        enable_photometric: Per-env image augmentation (brightness, contrast,
            gamma, colour cast and sensor noise). This is the primary source of
            visual diversity because the dome light is global.
        brightness_range: Multiplicative brightness gain range.
        contrast_range: Multiplicative contrast gain range around mid-grey.
        gamma_range: Gamma correction exponent range.
        color_cast_range: Per-channel colour cast gain range.
        pixel_noise_std: Additive Gaussian sensor noise standard deviation, in
            normalized [0, 1] image units.
        enable_physics: Randomize the global physics timestep and solver budget.
            Resampled once per rollout, because the timestep and solver budget
            are properties of the shared simulation context.
        physics_dt_range: Physics timestep range [s]. ``decimation`` is adjusted
            to hold the control rate fixed.
        solver_iteration_range: Position and velocity solver iteration count
            range, mapped onto the PhysX ``min_*``/``max_*_iteration_count``
            fields.
        contact_offset_scale_range: Contact offset multiplier range, applied to
            each object's rigid body properties at spawn time. Models
            mesh/collision tolerance differences.
        rest_offset_scale_range: Rest offset multiplier range, applied to each
            object's rigid body properties at spawn time.
        hdr_tags: HDR environment-map tags sampled for the dome light. The
            industrial and workshop tags are the ones that match the target
            deployment domain.
        use_hdr: Sample an HDR environment map instead of a uniform dome colour.
    """

    enable_pose_jitter: bool = True
    position_jitter: tuple[float, float, float] = (0.02, 0.02, 0.005)
    yaw_jitter: float = 0.26
    height_jitter: float = 0.005

    enable_scale_jitter: bool = True
    scale_range: tuple[float, float] = (0.95, 1.05)

    enable_material: bool = True
    static_friction_range: tuple[float, float] = (0.6, 1.4)
    dynamic_friction_range: tuple[float, float] = (0.5, 1.2)
    restitution_range: tuple[float, float] = (0.0, 0.15)
    num_material_buckets: int = 8

    enable_mass: bool = True
    mass_scale_range: tuple[float, float] = (0.8, 1.25)

    enable_robot_jitter: bool = True
    joint_position_offset: tuple[float, float] = (-0.02, 0.02)
    joint_velocity_range: tuple[float, float] = (-0.05, 0.05)

    enable_action_delay: bool = True
    action_delay_steps: tuple[int, int] = (0, 2)
    enable_action_noise: bool = True
    action_noise_std: float = 0.002
    enable_ee_obs_noise: bool = True
    ee_obs_noise_std: float = 0.003

    enable_light: bool = True
    light_intensity_range: tuple[float, float] = (0.6, 1.6)
    light_color_temp_range: tuple[float, float] = (4000.0, 7500.0)

    enable_camera_extrinsics: bool = True
    camera_position_jitter: tuple[float, float, float] = (0.005, 0.005, 0.005)
    camera_rotation_jitter: float = 0.01
    enable_camera_intrinsics: bool = True
    focal_length_range: tuple[float, float] = (22.0, 26.0)
    horizontal_aperture_range: tuple[float, float] = (78.0, 82.0)
    focus_distance_range: tuple[float, float] = (0.3, 1.2)

    enable_photometric: bool = True
    brightness_range: tuple[float, float] = (0.8, 1.25)
    contrast_range: tuple[float, float] = (0.8, 1.25)
    gamma_range: tuple[float, float] = (0.85, 1.2)
    color_cast_range: tuple[float, float] = (0.95, 1.05)
    pixel_noise_std: float = 0.01

    enable_physics: bool = True
    physics_dt_range: tuple[float, float] = (0.004, 0.008)
    solver_iteration_range: tuple[int, int] = (4, 16)
    contact_offset_scale_range: tuple[float, float] = (0.9, 1.1)
    rest_offset_scale_range: tuple[float, float] = (0.85, 1.2)

    use_hdr: bool = True
    hdr_tags: tuple[str, ...] = ("industrial", "workshop", "indoor")

    # Object roles whose poses are pinned even when pose jitter is enabled.
    # Fixtures such as a microwave or a bolt to be threaded must not move.
    pinned_objects: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def none(cls) -> "DRConfig":
        """Return a fully deterministic configuration (all ranges collapsed)."""
        return cls(
            enable_pose_jitter=False,
            position_jitter=(0.0, 0.0, 0.0),
            yaw_jitter=0.0,
            height_jitter=0.0,
            enable_scale_jitter=False,
            scale_range=(1.0, 1.0),
            enable_material=False,
            static_friction_range=(1.0, 1.0),
            dynamic_friction_range=(1.0, 1.0),
            restitution_range=(0.0, 0.0),
            enable_mass=False,
            mass_scale_range=(1.0, 1.0),
            enable_robot_jitter=False,
            joint_position_offset=(0.0, 0.0),
            joint_velocity_range=(0.0, 0.0),
            enable_action_delay=False,
            action_delay_steps=(0, 0),
            enable_action_noise=False,
            action_noise_std=0.0,
            enable_ee_obs_noise=False,
            ee_obs_noise_std=0.0,
            enable_light=False,
            light_intensity_range=(1.0, 1.0),
            light_color_temp_range=(6500.0, 6500.0),
            enable_camera_extrinsics=False,
            camera_position_jitter=(0.0, 0.0, 0.0),
            camera_rotation_jitter=0.0,
            enable_camera_intrinsics=False,
            focal_length_range=(24.0, 24.0),
            horizontal_aperture_range=(80.0, 80.0),
            focus_distance_range=(0.4, 0.4),
            enable_photometric=False,
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            gamma_range=(1.0, 1.0),
            color_cast_range=(1.0, 1.0),
            pixel_noise_std=0.0,
            enable_physics=False,
            physics_dt_range=(0.005, 0.005),
            solver_iteration_range=(8, 8),
            contact_offset_scale_range=(1.0, 1.0),
            rest_offset_scale_range=(1.0, 1.0),
            use_hdr=False,
        )

    @classmethod
    def light(cls) -> "DRConfig":
        """Visual-only randomization.

        Keeps geometry, physics and robot state deterministic so that a policy
        can be validated on manipulation skill before visual noise is added.
        """
        base = cls()
        return replace(
            base,
            enable_pose_jitter=False,
            position_jitter=(0.0, 0.0, 0.0),
            yaw_jitter=0.0,
            enable_scale_jitter=False,
            scale_range=(1.0, 1.0),
            enable_material=False,
            static_friction_range=(1.0, 1.0),
            dynamic_friction_range=(1.0, 1.0),
            restitution_range=(0.0, 0.0),
            enable_mass=False,
            mass_scale_range=(1.0, 1.0),
            enable_robot_jitter=False,
            joint_position_offset=(0.0, 0.0),
            joint_velocity_range=(0.0, 0.0),
            enable_action_delay=False,
            action_delay_steps=(0, 0),
            enable_action_noise=False,
            action_noise_std=0.0,
            enable_ee_obs_noise=False,
            ee_obs_noise_std=0.0,
            enable_physics=False,
            physics_dt_range=(0.005, 0.005),
            solver_iteration_range=(8, 8),
            contact_offset_scale_range=(1.0, 1.0),
            rest_offset_scale_range=(1.0, 1.0),
        )

    @classmethod
    def default(cls) -> "DRConfig":
        """The default sim-to-real configuration used by the industrial tasks."""
        return cls()

    @classmethod
    def heavy(cls) -> "DRConfig":
        """Aggressive randomization for robustness training and evaluation."""
        base = cls()
        return replace(
            base,
            position_jitter=(0.035, 0.035, 0.01),
            yaw_jitter=0.52,
            height_jitter=0.01,
            scale_range=(0.9, 1.1),
            static_friction_range=(0.4, 1.8),
            dynamic_friction_range=(0.35, 1.5),
            mass_scale_range=(0.6, 1.6),
            joint_position_offset=(-0.04, 0.04),
            action_delay_steps=(0, 4),
            action_noise_std=0.005,
            ee_obs_noise_std=0.006,
            camera_position_jitter=(0.01, 0.01, 0.01),
            camera_rotation_jitter=0.02,
            pixel_noise_std=0.02,
            solver_iteration_range=(2, 24),
            contact_offset_scale_range=(0.8, 1.25),
            rest_offset_scale_range=(0.7, 1.4),
        )

    @classmethod
    def named(cls, preset: str) -> "DRConfig":
        """Look up a preset by name.

        Args:
            preset: One of ``"none"``, ``"light"``, ``"default"`` or ``"heavy"``.

        Returns:
            The matching :class:`DRConfig`.

        Raises:
            ValueError: If ``preset`` is not a known name.
        """
        presets = {
            "none": cls.none,
            "light": cls.light,
            "default": cls.default,
            "heavy": cls.heavy,
        }
        if preset not in presets:
            raise ValueError(f"Unknown domain randomization preset '{preset}'. Choose from {sorted(presets)}.")
        return presets[preset]()

    def is_range_active(self, value_range: tuple[float, float]) -> bool:
        """Return True if a ``(min, max)`` range actually varies."""
        return abs(value_range[0] - value_range[1]) > 1e-9
