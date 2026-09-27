"""Batched success predicates for staged OpenArm tasks.

Every function here returns a boolean tensor of shape ``(num_envs,)`` and is
safe to use directly as an Isaac Lab termination term. They are written to be
vectorized across parallel environments: there is no Python loop over
environments anywhere in this module.

All object positions are taken relative to the environment origin so that the
same world-frame constants can be used whether the scene contains one
environment or thousands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _object_position(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return the environment-frame position of a scene object, shape ``(N, 3)``."""
    asset = env.unwrapped.scene[name]
    if hasattr(asset.data, "root_pos_w"):
        position = wp.to_torch(asset.data.root_pos_w)
    else:
        position = asset.get_world_poses()[0]
    return position - env.unwrapped.scene.env_origins


def _object_quat(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return the environment-frame orientation of a scene object as ``(w, x, y, z)``."""
    asset = env.unwrapped.scene[name]
    if hasattr(asset.data, "root_quat_w"):
        return wp.to_torch(asset.data.root_quat_w)
    quat = asset.get_world_poses()[1]
    # Isaac Lab world poses are returned as (w, x, y, z) for quaternions.
    return quat


def _normalized_joint_position(env: "ManagerBasedEnv", articulation_name: str, joint_name: str) -> torch.Tensor:
    """Return a joint position normalized to [0, 1] from its authored limits.

    Mirrors :func:`isaaclab_arena.utils.joint_utils.get_normalized_joint_position`
    but resolves the joint by name per environment, so a task can be declared
    with a joint name rather than an index.
    """
    articulation = env.unwrapped.scene[articulation_name]
    joint_names: list[str] = articulation.data.joint_names
    if joint_name not in joint_names:
        raise KeyError(f"Joint '{joint_name}' not found on '{articulation_name}'. Available: {joint_names}")
    joint_index = joint_names.index(joint_name)
    joint_pos = wp.to_torch(articulation.data.joint_pos)[:, joint_index]
    limits = wp.to_torch(articulation.data.joint_pos_limits)[:, joint_index, :]
    joint_min, joint_max = limits[:, 0], limits[:, 1]
    span = (joint_max - joint_min).clamp_min(1e-6)
    normalized = (joint_pos - joint_min) / span
    # A joint whose range is entirely negative is "open" at its minimum.
    return torch.where(joint_min < 0.0, 1.0 - normalized, normalized)


def object_in_region(
    env: "ManagerBasedEnv",
    object_cfg: str,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> torch.Tensor:
    """Check that an object's centre lies inside an axis-aligned box.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        center: Box centre in the environment frame [m].
        half_extents: Box half-size along each axis [m].

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    position = _object_position(env, object_cfg)
    center_t = torch.tensor(center, device=position.device, dtype=position.dtype).view(1, 3)
    half = torch.tensor(half_extents, device=position.device, dtype=position.dtype).view(1, 3)
    return ((position - center_t).abs() <= half).all(dim=1)


def object_relative_to(
    env: "ManagerBasedEnv",
    object_cfg: str,
    reference_cfg: str,
    offset: tuple[float, float, float],
    tolerance: float,
) -> torch.Tensor:
    """Check that an object sits at a fixed offset from a reference object.

    Args:
        env: The environment.
        object_cfg: Scene name of the moving object.
        reference_cfg: Scene name of the reference object.
        offset: Required position offset from the reference [m].
        tolerance: Allowed Euclidean deviation [m].

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    position = _object_position(env, object_cfg)
    reference = _object_position(env, reference_cfg)
    offset_t = torch.tensor(offset, device=position.device, dtype=position.dtype).view(1, 3)
    return torch.norm((position - reference) - offset_t, dim=1) <= float(tolerance)


def object_upright(env: "ManagerBasedEnv", object_cfg: str, tolerance: float) -> torch.Tensor:
    """Check that an object's local +Z axis is within ``tolerance`` of world +Z.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        tolerance: Maximum angle between the axes [rad].

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    quat = _object_quat(env, object_cfg)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Third column of the rotation matrix, i.e. the body +Z axis in world frame.
    body_z = torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=1)
    cos_angle = body_z[:, 2].clamp(-1.0, 1.0)
    return torch.acos(cos_angle) <= float(tolerance)


def objects_within_distance(
    env: "ManagerBasedEnv",
    object_cfg: str,
    other_cfg: str,
    max_distance: float,
) -> torch.Tensor:
    """Check that two objects are closer than a threshold.

    Args:
        env: The environment.
        object_cfg: Scene name of the first object.
        other_cfg: Scene name of the second object.
        max_distance: Maximum centre-to-centre distance [m].

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    position = _object_position(env, object_cfg)
    other = _object_position(env, other_cfg)
    return torch.norm(position - other, dim=1) <= float(max_distance)


def joint_normalized_above(
    env: "ManagerBasedEnv",
    asset_cfg: str,
    joint_name: str,
    threshold: float,
) -> torch.Tensor:
    """Check that a normalized joint coordinate exceeds a threshold.

    Args:
        env: The environment.
        asset_cfg: Scene name of the articulation.
        joint_name: Name of the joint.
        threshold: Normalized threshold.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    return _normalized_joint_position(env, asset_cfg, joint_name) > float(threshold)


def joint_normalized_below(
    env: "ManagerBasedEnv",
    asset_cfg: str,
    joint_name: str,
    threshold: float,
) -> torch.Tensor:
    """Check that a normalized joint coordinate falls below a threshold.

    Args:
        env: The environment.
        asset_cfg: Scene name of the articulation.
        joint_name: Name of the joint.
        threshold: Normalized threshold.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    return _normalized_joint_position(env, asset_cfg, joint_name) < float(threshold)


def object_at_rest(
    env: "ManagerBasedEnv",
    object_cfg: str,
    max_speed: float,
    max_angular_speed: float,
) -> torch.Tensor:
    """Check that an object has stopped moving.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        max_speed: Maximum linear speed [m/s].
        max_angular_speed: Maximum angular speed [rad/s].

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    asset = env.unwrapped.scene[object_cfg]
    linear = torch.norm(wp.to_torch(asset.data.root_lin_vel_w), dim=1)
    angular = torch.norm(wp.to_torch(asset.data.root_ang_vel_w), dim=1)
    return (linear <= float(max_speed)) & (angular <= float(max_angular_speed))


def object_above_plane(
    env: "ManagerBasedEnv",
    object_cfg: str,
    min_height: float,
    max_height: float,
) -> torch.Tensor:
    """Check that an object's height lies within a band.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        min_height: Lower bound on world-frame height [m].
        max_height: Upper bound on world-frame height [m].

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    position = _object_position(env, object_cfg)
    height = position[:, 2]
    return (height >= float(min_height)) & (height <= float(max_height))


def all_goals(env: "ManagerBasedEnv", goals: list[Any]) -> torch.Tensor:
    """Conjunction over a list of goal parameter dictionaries.

    Args:
        env: The environment.
        goals: Parameter dictionaries produced by
            :meth:`openarm.arena_tasks.staged.goals.Goal.params`.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.

    Raises:
        ValueError: If the list is empty, which would silently evaluate to
            vacuously true.
    """
    if not goals:
        raise ValueError("all_goals() requires at least one goal; an empty list would always report success.")
    result: torch.Tensor | None = None
    for params in goals:
        value = evaluate_goal(env, params)
        result = value if result is None else (result & value)
    assert result is not None
    return result


def at_least_goals(env: "ManagerBasedEnv", count: int, goals: list[Any]) -> torch.Tensor:
    """Count how many goals hold and compare against a threshold.

    Args:
        env: The environment.
        count: Minimum number of goals that must hold.
        goals: Parameter dictionaries produced by
            :meth:`openarm.arena_tasks.staged.goals.Goal.params`.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.

    Raises:
        ValueError: If the list is empty or ``count`` exceeds its length.
    """
    if not goals:
        raise ValueError("at_least_goals() requires at least one goal.")
    if count > len(goals):
        raise ValueError(f"at_least_goals() count={count} exceeds the number of goals ({len(goals)}).")
    stacked = torch.stack([evaluate_goal(env, params) for params in goals], dim=0)
    return stacked.sum(dim=0) >= int(count)


def object_below_height(env: "ManagerBasedEnv", object_names: list[str], minimum_height: float) -> torch.Tensor:
    """Check whether any tracked object has fallen below a height.

    Used as the failure termination for staged tasks: once an object is on the
    floor there is nothing useful left to do with the episode, so it should end
    rather than burn the remaining time.

    Args:
        env: The environment.
        object_names: Scene names of the tracked objects.
        minimum_height: Threshold on world-frame height [m].

    Returns:
        Boolean tensor of shape ``(num_envs,)``, true if *any* object is below
        the threshold.
    """
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    result = torch.zeros(num_envs, device=device, dtype=torch.bool)
    for name in object_names:
        result |= _object_position(env, name)[:, 2] < float(minimum_height)
    return result


_DISPATCH = {
    "in_region": lambda env, p: object_in_region(env, p["object_cfg"], p["center"], p["half_extents"]),
    "relative_to": lambda env, p: object_relative_to(
        env, p["object_cfg"], p["reference_cfg"], p["offset"], p["tolerance"]
    ),
    "upright": lambda env, p: object_upright(env, p["object_cfg"], p["tolerance"]),
    "within_distance": lambda env, p: objects_within_distance(env, p["object_cfg"], p["other_cfg"], p["max_distance"]),
    "joint_above": lambda env, p: joint_normalized_above(env, p["asset_cfg"], p["joint_name"], p["threshold"]),
    "joint_below": lambda env, p: joint_normalized_below(env, p["asset_cfg"], p["joint_name"], p["threshold"]),
    "at_rest": lambda env, p: object_at_rest(env, p["object_cfg"], p["max_speed"], p["max_angular_speed"]),
    "above_plane": lambda env, p: object_above_plane(env, p["object_cfg"], p["min_height"], p["max_height"]),
    "all": lambda env, p: all_goals(env, p["goals"]),
    "at_least": lambda env, p: at_least_goals(env, p["count"], p["goals"]),
}


def evaluate_goal(env: "ManagerBasedEnv", params: dict[str, Any]) -> torch.Tensor:
    """Dispatch a goal parameter dictionary to its predicate function.

    Dispatch is on the explicit ``kind`` key that every
    :class:`~openarm.arena_tasks.staged.goals.Goal` emits, rather than on the
    shape of the parameter dictionary, so adding a new goal can never silently
    change how an existing one is routed.

    Args:
        env: The environment.
        params: Parameters produced by
            :meth:`openarm.arena_tasks.staged.goals.Goal.params`.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.

    Raises:
        KeyError: If the goal ``kind`` is missing or unknown.
    """
    kind = params.get("kind")
    if kind is None:
        raise KeyError(f"Goal parameters are missing the 'kind' key: {sorted(params)}")
    handler = _DISPATCH.get(kind)
    if handler is None:
        raise KeyError(f"Unknown goal kind '{kind}'. Known kinds: {sorted(_DISPATCH)}")
    return handler(env, params)
