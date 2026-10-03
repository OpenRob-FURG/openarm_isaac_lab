"""Observation terms for the privileged critic of a staged task.

An industrial task's success condition is a statement about where objects are,
and that is precisely what a policy cannot see. The OpenArm bimanual
embodiment's ``policy`` group is proprioception only -- joint states, end
effectors, grippers, and the robot's own links -- so an actor trained on it is
blind to whether the can is already in the bin.

Asymmetric actor-critic training is the standard answer: the actor keeps the
partial observation it could actually get on the real robot, and the critic is
handed the full simulator state so that its value estimates are not handicapped
by the same blindness. These terms are that full state. They are read-only, so
they can be served to the critic without changing what the actor sees, and they
are marked so that no deployment path can pick them up by accident.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from openarm.arena_tasks.staged.scene_state import (
    ee_orientation,
    ee_position,
    gripper_opening,
    object_angular_velocity,
    object_linear_velocity,
    object_pose,
    object_position,
    to_torch,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

#: End-effector frame suffixes, in the order their observations are stacked.
ARMS = ("left", "right")


def object_poses(env: "ManagerBasedEnv", object_names: list[str]) -> torch.Tensor:
    """Stack the environment-frame pose of every named object.

    The core of the privileged state: where the props actually are, including
    the ones the policy is not currently touching. A task can only be scored on
    this, so a critic without it is estimating a value function from a state
    that does not determine the reward.

    Args:
        env: The environment.
        object_names: Scene names of the objects, in a fixed order.

    Returns:
        Tensor of shape ``(N, 7 * len(object_names))``, each block
        ``(x, y, z, qw, qx, qy, qz)``.

    Raises:
        ValueError: If no object is named, which would produce a zero-width
            observation that silently tells the critic nothing.
    """
    if not object_names:
        raise ValueError("object_poses() requires at least one object name.")
    return torch.cat([object_pose(env, name) for name in object_names], dim=1)


def object_positions(env: "ManagerBasedEnv", object_names: list[str]) -> torch.Tensor:
    """Stack the environment-frame position of every named object.

    Args:
        env: The environment.
        object_names: Scene names of the objects, in a fixed order.

    Returns:
        Tensor of shape ``(N, 3 * len(object_names))``.

    Raises:
        ValueError: If no object is named.
    """
    if not object_names:
        raise ValueError("object_positions() requires at least one object name.")
    return torch.cat([object_position(env, name) for name in object_names], dim=1)


def object_velocities(env: "ManagerBasedEnv", object_names: list[str]) -> torch.Tensor:
    """Stack the world-frame linear and angular velocity of every named object.

    The privileged twin of the ``at_rest`` goal's inputs. Without it a critic
    cannot tell a can that is seated in the bin from one that is still tumbling
    through it, and it will learn to value both as done.

    Args:
        env: The environment.
        object_names: Scene names of the objects, in a fixed order.

    Returns:
        Tensor of shape ``(N, 6 * len(object_names))``.

    Raises:
        ValueError: If no object is named.
    """
    if not object_names:
        raise ValueError("object_velocities() requires at least one object name.")
    blocks = []
    for name in object_names:
        blocks.append(object_linear_velocity(env, name))
        blocks.append(object_angular_velocity(env, name))
    return torch.cat(blocks, dim=1)


def object_relative_to_surface(env: "ManagerBasedEnv", object_names: list[str]) -> torch.Tensor:
    """Stack each object's height above the ground plane.

    A compact stand-in for the full pose when the critic mainly needs to know
    whether something is still on the table. The ``dropped`` termination and
    the lift goals are both height questions, and height is the one pose
    component that changes discretely during a task.

    Args:
        env: The environment.
        object_names: Scene names of the objects, in a fixed order.

    Returns:
        Tensor of shape ``(N, len(object_names))``.

    Raises:
        ValueError: If no object is named.
    """
    if not object_names:
        raise ValueError("object_relative_to_surface() requires at least one object name.")
    return torch.stack([object_position(env, name)[:, 2] for name in object_names], dim=1)


def goal_targets(env: "ManagerBasedEnv", targets: list[tuple[float, float, float]]) -> torch.Tensor:
    """Broadcast a task's world-frame goal points to every environment.

    These are constants of the task rather than measurements of the world, but
    they belong in the critic's input: the value of a state is defined relative
    to a target, and handing the critic the target explicitly lets it learn one
    shared value function across a vectorized batch whose tasks differ.

    Args:
        env: The environment.
        targets: Goal points in the environment frame [m].

    Returns:
        Tensor of shape ``(N, 3 * len(targets))``, constant along the batch.

    Raises:
        ValueError: If no target is given.
    """
    if not targets:
        raise ValueError("goal_targets() requires at least one target.")
    device = env.unwrapped.device
    row = torch.tensor(targets, device=device, dtype=torch.float32).reshape(1, -1)
    return row.expand(env.unwrapped.num_envs, -1).contiguous()


def ee_world_poses(env: "ManagerBasedEnv", arms: tuple[str, ...] = ARMS) -> torch.Tensor:
    """Stack the world-frame pose of each arm's end effector.

    The embodiment's own ``policy`` group reports end effectors relative to the
    robot, which is the right thing for an actor that only ever sees the robot.
    The critic needs the world-frame pose as well, because a goal is written in
    world coordinates and the critic has to relate the two.

    Args:
        env: The environment.
        arms: Arms to include, in order.

    Returns:
        Tensor of shape ``(N, 7 * len(arms))``.

    Raises:
        ValueError: If no arm is given.
    """
    if not arms:
        raise ValueError("ee_world_poses() requires at least one arm.")
    blocks = []
    for arm in arms:
        blocks.append(ee_position(env, arm))
        blocks.append(ee_orientation(env, arm))
    return torch.cat(blocks, dim=1)


def gripper_openings(env: "ManagerBasedEnv", arms: tuple[str, ...] = ARMS) -> torch.Tensor:
    """Stack the finger joint position of each arm.

    Args:
        env: The environment.
        arms: Arms to include, in order.

    Returns:
        Tensor of shape ``(N, len(arms))``.

    Raises:
        ValueError: If no arm is given.
    """
    if not arms:
        raise ValueError("gripper_openings() requires at least one arm.")
    return torch.stack([gripper_opening(env, arm) for arm in arms], dim=1)


def joint_efforts(env: "ManagerBasedEnv", asset_name: str = "robot") -> torch.Tensor:
    """Return the applied joint torques of an articulation.

    Applied joint torque is the clearest available signal that the policy is
    fighting the robot rather than moving it -- a stall against a jammed object
    looks almost identical to good progress in position terms alone -- so it is
    worth the critic having.

    Args:
        env: The environment.
        asset_name: Scene name of the articulation.

    Returns:
        Tensor of shape ``(N, num_joints)``.
    """
    return to_torch(env.unwrapped.scene[asset_name].data.applied_torque)


def goal_relative_object_offsets(
    env: "ManagerBasedEnv",
    pairs: list[tuple[str, tuple[float, float, float]]],
) -> torch.Tensor:
    """Stack each named object's offset from its own goal point.

    This is the one representation of task progress that is independent of where
    the objects started, so it is the term that makes the critic's value
    function transferable rather than tied to one resolved layout. It also
    halves the width of the privileged state: one block per goal instead of one
    per object plus one per target.

    Args:
        env: The environment.
        pairs: ``(object scene name, goal point in the environment frame)``
            pairs, in a fixed order.

    Returns:
        Tensor of shape ``(N, 3 * len(pairs))``.

    Raises:
        ValueError: If no pair is given.
    """
    if not pairs:
        raise ValueError("goal_relative_object_offsets() requires at least one pair.")
    blocks = []
    for name, target in pairs:
        position = object_position(env, name)
        blocks.append(position - _row(target, position))
    return torch.cat(blocks, dim=1)


def _row(values: tuple[float, float, float], reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a 3-vector to a ``(1, 3)`` tensor on ``reference``'s device."""
    return torch.tensor(values, device=reference.device, dtype=reference.dtype).view(1, 3)
