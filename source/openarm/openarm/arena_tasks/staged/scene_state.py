"""Readers for the scene state a staged task's predicates and rewards are built on.

A staged task is declared as data, so both the success predicates in
:mod:`openarm.arena_tasks.staged.terminations` and the dense shaping terms in
:mod:`openarm.arena_tasks.staged.rewards` need to read the same quantities from
the same places: an object's environment-frame pose, a joint's normalized
coordinate, an end-effector's world pose. Those readers live here so the two
modules cannot disagree about what "the object's position" means -- in
particular about the environment-origin subtraction that makes one set of
world-frame constants valid for both a single environment and thousands.

Everything here is vectorized across parallel environments and returns tensors
on the environment's own device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

#: Name of the frame transformer the OpenArm bimanual embodiment publishes.
EE_FRAME_NAME = "ee_frame"


def to_torch(value, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert a simulator buffer to a torch tensor.

    The physics backend does not guarantee torch: ``isaaclab_physx`` hands back
    warp arrays and the legacy backend hands back torch. Normalising here keeps
    every reader below agnostic to which one is active.

    Args:
        value: Simulator buffer, or an existing torch tensor.
        dtype: Requested floating-point dtype.

    Returns:
        Tensor of the requested dtype. Never a copy of an already-matching
        torch tensor, so callers that only read may avoid a clone.
    """
    if isinstance(value, torch.Tensor):
        return value if value.dtype == dtype else value.to(dtype)
    return wp.to_torch(value).to(dtype)


def object_position(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return the environment-frame position of a scene object.

    Args:
        env: The environment.
        name: Scene name of the object.

    Returns:
        Tensor of shape ``(N, 3)``.
    """
    asset = env.unwrapped.scene[name]
    if hasattr(asset.data, "root_pos_w"):
        position = to_torch(asset.data.root_pos_w)
    else:
        position = to_torch(asset.get_world_poses()[0])
    return position - to_torch(env.unwrapped.scene.env_origins)


def object_orientation(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return the world-frame orientation of a scene object as ``(w, x, y, z)``.

    Orientation is not offset-corrected: a relative quaternion would need a
    per-environment inverse rotation and nothing in this package consumes one.

    Args:
        env: The environment.
        name: Scene name of the object.

    Returns:
        Tensor of shape ``(N, 4)``.
    """
    asset = env.unwrapped.scene[name]
    if hasattr(asset.data, "root_quat_w"):
        return to_torch(asset.data.root_quat_w)
    # Isaac Lab world poses are returned as (w, x, y, z) for quaternions.
    return to_torch(asset.get_world_poses()[1])


def object_pose(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return the environment-frame pose of a scene object as ``(x, y, z, qw, qx, qy, qz)``.

    Args:
        env: The environment.
        name: Scene name of the object.

    Returns:
        Tensor of shape ``(N, 7)``.
    """
    return torch.cat([object_position(env, name), object_orientation(env, name)], dim=1)


def tilt_from_vertical(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return how far an object's local +Z axis leans away from world +Z.

    This is the continuous quantity behind the ``upright`` goal: zero means
    upright, :math:`\\pi/2` means lying on its side. Kept here rather than in the
    predicate so the dense reward can shape the same angle the termination
    thresholds.

    Args:
        env: The environment.
        name: Scene name of the object.

    Returns:
        Tensor of shape ``(N,)`` in ``[0, pi]`` [rad].
    """
    quat = object_orientation(env, name)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Third column of the rotation matrix, i.e. the body +Z axis in world frame.
    body_z = torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=1)
    cos_angle = body_z[:, 2].clamp(-1.0, 1.0)
    return torch.acos(cos_angle)


def object_linear_velocity(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return the world-frame linear velocity of a scene object.

    Args:
        env: The environment.
        name: Scene name of the object.

    Returns:
        Tensor of shape ``(N, 3)``.
    """
    return to_torch(env.unwrapped.scene[name].data.root_lin_vel_w)


def object_angular_velocity(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return the world-frame angular velocity of a scene object.

    Args:
        env: The environment.
        name: Scene name of the object.

    Returns:
        Tensor of shape ``(N, 3)``.
    """
    return to_torch(env.unwrapped.scene[name].data.root_ang_vel_w)


def object_speed(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return the linear speed of a scene object.

    Args:
        env: The environment.
        name: Scene name of the object.

    Returns:
        Tensor of shape ``(N,)``.
    """
    return torch.norm(object_linear_velocity(env, name), dim=1)


def object_angular_speed(env: "ManagerBasedEnv", name: str) -> torch.Tensor:
    """Return the angular speed of a scene object.

    Args:
        env: The environment.
        name: Scene name of the object.

    Returns:
        Tensor of shape ``(N,)``.
    """
    return torch.norm(object_angular_velocity(env, name), dim=1)


def normalized_joint_position(env: "ManagerBasedEnv", articulation_name: str, joint_name: str) -> torch.Tensor:
    """Return a joint position normalized to ``[0, 1]`` from its authored limits.

    Mirrors :func:`isaaclab_arena.utils.joint_utils.get_normalized_joint_position`
    but resolves the joint by name per environment, so a task can be declared
    with a joint name rather than an index.

    Args:
        env: The environment.
        articulation_name: Scene name of the articulation.
        joint_name: Name of the joint.

    Returns:
        Tensor of shape ``(N,)``.

    Raises:
        KeyError: If the articulation does not have the named joint.
    """
    articulation = env.unwrapped.scene[articulation_name]
    joint_names: list[str] = articulation.data.joint_names
    if joint_name not in joint_names:
        raise KeyError(f"Joint '{joint_name}' not found on '{articulation_name}'. Available: {joint_names}")
    joint_index = joint_names.index(joint_name)
    joint_pos = to_torch(articulation.data.joint_pos)[:, joint_index]
    limits = to_torch(articulation.data.joint_pos_limits)[:, joint_index, :]
    joint_min, joint_max = limits[:, 0], limits[:, 1]
    span = (joint_max - joint_min).clamp_min(1e-6)
    normalized = (joint_pos - joint_min) / span
    # A joint whose range is entirely negative is "open" at its minimum.
    return torch.where(joint_min < 0.0, 1.0 - normalized, normalized)


def ee_frame(env: "ManagerBasedEnv", frame_name: str = EE_FRAME_NAME):
    """Return the embodiment's frame transformer.

    Args:
        env: The environment.
        frame_name: Scene name of the transformer.

    Returns:
        The ``FrameTransformer`` asset.

    Raises:
        KeyError: If the scene has no such transformer, which happens when an
            embodiment was built without one.
    """
    return env.unwrapped.scene[frame_name]


def ee_position(env: "ManagerBasedEnv", arm: str, frame_name: str = EE_FRAME_NAME) -> torch.Tensor:
    """Return the world-frame position of one arm's end-effector frame.

    Args:
        env: The environment.
        arm: Either ``"left"`` or ``"right"``.
        frame_name: Scene name of the frame transformer.

    Returns:
        Tensor of shape ``(N, 3)``.

    Raises:
        KeyError: If the transformer publishes no frame for that arm.
    """
    transformer = ee_frame(env, frame_name)
    index = transformer.data.target_frame_names.index(f"{arm}_ee_tcp")
    return to_torch(transformer.data.target_pos_w)[:, index, :]  - to_torch(env.unwrapped.scene.env_origins)


def ee_orientation(env: "ManagerBasedEnv", arm: str, frame_name: str = EE_FRAME_NAME) -> torch.Tensor:
    """Return the world-frame orientation of one arm's end-effector frame.

    Args:
        env: The environment.
        arm: Either ``"left"`` or ``"right"``.
        frame_name: Scene name of the frame transformer.

    Returns:
        Tensor of shape ``(N, 4)`` in ``(w, x, y, z)`` order.
    """
    transformer = ee_frame(env, frame_name)
    index = transformer.data.target_frame_names.index(f"{arm}_ee_tcp")
    return to_torch(transformer.data.target_quat_w)[:, index, :]


def gripper_opening(env: "ManagerBasedEnv", arm: str, joint_name: str | None = None) -> torch.Tensor:
    """Return the joint position of one arm's finger, shape ``(N,)``.

    Used as a grasp signal: the finger is fully open at the joint's upper limit
    and fully closed at zero, so a low value means the gripper is shut.

    Args:
        env: The environment.
        arm: Either ``"left"`` or ``"right"``.
        joint_name: Finger joint to read. Defaults to the OpenArm bimanual
            naming convention.

    Returns:
        Tensor of shape ``(N,)``.

    Raises:
        KeyError: If the robot has no such joint.
    """
    if joint_name is None:
        joint_name = f"openarm_{arm}_finger_joint1"
    robot = env.unwrapped.scene["robot"]
    if joint_name not in list(robot.data.joint_names):
        raise KeyError(f"Joint '{joint_name}' not found on 'robot'. Available: {list(robot.data.joint_names)}")
    index = list(robot.data.joint_names).index(joint_name)
    return to_torch(robot.data.joint_pos)[:, index]
