"""Batched success predicates for staged OpenArm tasks.

Every function here returns a boolean tensor of shape ``(num_envs,)`` and is
safe to use directly as an Isaac Lab termination term. They are written to be
vectorized across parallel environments: there is no Python loop over
environments anywhere in this module.

All object positions are taken relative to the environment origin so that the
same world-frame constants can be used whether the scene contains one
environment or thousands. The readers themselves live in
:mod:`openarm.arena_tasks.staged.scene_state`, shared with the dense reward
terms so the two cannot disagree about what a quantity means.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from openarm.arena_tasks.staged.scene_state import (
    normalized_joint_position,
    object_angular_speed,
    object_position,
    object_speed,
    tilt_from_vertical,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

#: Local aliases, so the predicate bodies below read as one word.
_object_position = object_position
_normalized_joint_position = normalized_joint_position


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
    return tilt_from_vertical(env, object_cfg) <= float(tolerance)


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
    """Check that a normalized joint coordinate reaches a threshold.

    Inclusive, like every other predicate in this module: reaching the declared
    threshold is the goal being met. The strict comparison this used to write
    made the joint goals the only ones in the package where a state sitting
    exactly on the threshold was a failure, which put the boolean predicate and
    the dense shaping term in
    :mod:`openarm.arena_tasks.staged.rewards` in direct disagreement at that one
    point.

    Args:
        env: The environment.
        asset_cfg: Scene name of the articulation.
        joint_name: Name of the joint.
        threshold: Normalized threshold.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    return _normalized_joint_position(env, asset_cfg, joint_name) >= float(threshold)


def joint_normalized_below(
    env: "ManagerBasedEnv",
    asset_cfg: str,
    joint_name: str,
    threshold: float,
) -> torch.Tensor:
    """Check that a normalized joint coordinate falls to a threshold.

    Inclusive, for the same reason as :func:`joint_normalized_above`.

    Args:
        env: The environment.
        asset_cfg: Scene name of the articulation.
        joint_name: Name of the joint.
        threshold: Normalized threshold.

    Returns:
        Boolean tensor of shape ``(num_envs,)``.
    """
    return _normalized_joint_position(env, asset_cfg, joint_name) <= float(threshold)


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
    linear = object_speed(env, object_cfg)
    angular = object_angular_speed(env, object_cfg)
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
