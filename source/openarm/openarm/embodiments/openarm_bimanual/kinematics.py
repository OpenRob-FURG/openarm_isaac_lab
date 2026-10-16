"""Pose/action conversion for the bimanual OpenArm, which is joint-space.

The OpenArm is commanded in absolute joint position, so every Cartesian quantity
in the Mimic API has to be converted at the boundary: a target pose has to be
solved for, and a commanded action has to be read back as the pose it is heading
towards. Both conversions live here rather than in either caller, because the
scripted oracle and the Mimic environment need them to agree exactly -- a target
pose the oracle reached and the same pose the datagen replays have to produce the
same joint targets, or the generated demonstrations drift away from the scripted
ones they were seeded from.

Three facts about this action space drive the shape of the code.

**The action layout is not fixed.** ``OpenArmBimanualActionsCfg`` declares a term
per arm and per finger, and the embodiment drops the terms of the arm an arm mode
does not use. Under ``ArmMode.RIGHT`` the right arm's channels start at offset
zero. Every index here is therefore read from the live action manager.

**The arm count is 7 and a pose is 7.** A flat joint action is 8 values per arm
(7 joints and a finger), and a Cartesian pose command plus a gripper is also 8.
Widths cannot tell the two action spaces apart; only the action terms can.

**A Jacobian step is enough.** These conversions run once per control step, so the
arm is already tracking the target and the residual error per step is small.
Iterating inside one call would fight the low-level PD controller, which acts on
whatever this returns one step later.
"""

from __future__ import annotations

from typing import Mapping

import torch

from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_matrix

from openarm.arena_tasks.staged.arms import ARM_EE_BODIES, ARM_FINGER_JOINTS, ARM_JOINT_NAMES

#: Damping for the damped-least-squares solve [rad].
#:
#: Matches the "dls" method's default in
#: :class:`~isaaclab.controllers.DifferentialIKControllerCfg`. The command lands on
#: the PD loop a step later, so a stiffer solve would oscillate rather than
#: converge.
IK_DAMPING = 0.05


def _unwrapped(env):
    return getattr(env, "unwrapped", env)


def _scene(env):
    """The scene, however the env was handed over.

    An Arena mimic env exposes ``.scene``; the oracle reaches the same object
    through the task's cfg. Both are tried because both callers exist.
    """
    if hasattr(env, "assets") and hasattr(env, "env_origins"):
        return env
    base = _unwrapped(env)
    scene = getattr(base, "scene", None)
    if scene is not None:
        return scene
    arena_env = getattr(getattr(base, "cfg", None), "isaaclab_arena_env", None)
    scene = getattr(arena_env, "scene", None)
    if scene is not None:
        return scene
    raise AttributeError(f"Could not reach the scene from {type(env).__name__}")


def _robot(env):
    """The robot articulation."""
    scene = _scene(env)
    if "robot" not in scene.keys():
        raise ValueError("The joint-space pose/action conversion needs the 'robot' articulation in the scene.")
    return scene["robot"]


def _to_torch(array, device: torch.device) -> torch.Tensor:
    """Convert a warp array from the physics backend into a torch tensor."""
    import warp as wp

    return wp.to_torch(array).to(device=device, dtype=torch.float32)


def active_arms(env) -> list[str]:
    """Arms the action space actually commands, in declaration order.

    Args:
        env: The environment.

    Returns:
        ``["left"]``, ``["right"]`` or ``["left", "right"]``.

    Raises:
        ValueError: If the environment exposes no action terms.
    """
    action_manager = getattr(_unwrapped(env), "action_manager", None)
    terms = list(getattr(action_manager, "active_terms", []) or [])
    if not terms:
        raise ValueError(
            "The environment exposes no action terms, so the joint action layout cannot be resolved. "
            "The conversion needs a running environment whose action manager is built."
        )
    arms = [name.split("_", 1)[0] for name in terms if "_arm_action" in name]
    if not arms:
        raise ValueError(f"No arm action terms in the action space. Terms: {terms}")
    return arms


def action_slices(env) -> dict[str, dict[str, slice]]:
    """Resolve each arm's channel ranges in the flat action vector.

    Args:
        env: The environment.

    Returns:
        Arm name to ``{"arm": slice, "finger": slice}``.

    Raises:
        ValueError: If an arm has no arm term or no hand term.
    """
    action_manager = getattr(_unwrapped(env), "action_manager", None)
    terms = list(getattr(action_manager, "active_terms", []) or [])
    widths = list(getattr(action_manager, "action_term_dim", []) or [])

    slices: dict[str, dict[str, slice]] = {}
    offset = 0
    for name, width in zip(terms, widths):
        arm = name.split("_", 1)[0]
        if "_arm_action" in name:
            slices.setdefault(arm, {})["arm"] = slice(offset, offset + width)
        elif "_hand_action" in name:
            # The finger is the only joint in the hand term.
            slices.setdefault(arm, {})["finger"] = slice(offset, offset + width)
        offset += width

    missing = [arm for arm in active_arms(env) if not {"arm", "finger"} <= set(slices.get(arm, {}))]
    if missing:
        raise ValueError(f"No joint action terms for arm(s) {missing}. Terms: {terms}, slices: {slices}")
    return slices


def _device(env) -> torch.device:
    device = getattr(_unwrapped(env), "device", None)
    return torch.device("cpu") if device is None else torch.device(device)


def arm_jacobian(env, arm: str) -> torch.Tensor:
    """Read one arm's geometric Jacobian, in the robot base frame.

    The indexing mirrors ``DifferentialInverseKinematicsAction``: a fixed-base
    articulation has no Jacobian rows for the base and its body indices are shifted
    down by one, while a floating-base one has six base DoFs ahead of the arm
    joints. The OpenArm torso floats, but the shift is derived from the asset
    rather than assumed, since a fixed-base USD of the same arm would otherwise
    silently return the wrong rows.

    Args:
        env: The environment.
        arm: ``"left"`` or ``"right"``.

    Returns:
        Jacobian of shape ``(1, 6, 7)``, ordered to match
        :data:`~openarm.arena_tasks.staged.arms.ARM_JOINT_NAMES`.
    """
    robot = _robot(env)
    joint_ids, _names = robot.find_joints(ARM_JOINT_NAMES[arm], preserve_order=True)
    body_ids, _bodies = robot.find_bodies(ARM_EE_BODIES[arm])
    if robot.is_fixed_base:
        row = body_ids[0] - 1
        columns = list(joint_ids)
    else:
        row = body_ids[0]
        columns = [index + 6 for index in joint_ids]
    raw = _to_torch(robot.root_view.get_jacobians(), _device(env))
    return raw[:, row, :, columns]


def eef_pose(env, arm: str, env_id: int = 0) -> torch.Tensor:
    """Read one sub-environment's world pose for an end effector.

    Args:
        env: The environment.
        arm: ``"left"`` or ``"right"``.
        env_id: Index of the sub-environment.

    Returns:
        The pose as a ``(4, 4)`` matrix.

    Raises:
        ValueError: If the scene has no end-effector frame for the arm.
    """
    scene = _scene(env)
    frames = [key for key in scene.keys() if key.endswith("ee_frame")]
    if not frames:
        raise KeyError(f"No 'ee_frame' sensor in the scene. Available: {sorted(scene.keys())}")
    frame = scene[frames[0]]

    index = None
    for i, name in enumerate(frame.data.target_frame_names):
        if arm in name.replace("/", "_").split("_"):
            index = i
            break
    if index is None:
        raise ValueError(f"No end-effector frame for arm '{arm}'. Frames: {list(frame.data.target_frame_names)}")

    device = _device(env)
    pos = convert_to_torch(frame.data.target_pos_w, dtype=torch.float32, device=device)
    quat = convert_to_torch(frame.data.target_quat_w, dtype=torch.float32, device=device)
    rotation = matrix_from_quat(quat[env_id, index])
    pose = torch.eye(4, device=device, dtype=torch.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = pos[env_id, index]
    return pose


def _pose_error(target: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    """Pose error as ``[dx, dy, dz, droll, dpitch, dyaw]``, world-frame rotation."""
    position = (target[:3, 3] - current[:3, 3]).reshape(1, 3)
    rotation = axis_angle_from_quat(quat_from_matrix(target[:3, :3] @ current[:3, :3].T))
    return torch.cat([position, rotation.reshape(1, 3)], dim=1)


def _dls(jacobian: torch.Tensor, error: torch.Tensor) -> torch.Tensor:
    """One damped least-squares step, as ``DifferentialIKController``'s "dls" method."""
    device = error.device
    jacobian_t = jacobian.transpose(1, 2)
    damping = IK_DAMPING**2 * torch.eye(jacobian.shape[1], device=device)
    return jacobian_t @ torch.linalg.solve(jacobian @ jacobian_t + damping, error.unsqueeze(-1))


def joint_targets_from_poses(
    env,
    target_poses: Mapping[str, torch.Tensor],
    gripper_actions: Mapping[str, torch.Tensor],
    env_id: int = 0,
) -> torch.Tensor:
    """Convert target end-effector poses into absolute joint position targets.

    An arm absent from ``target_poses`` is commanded to hold its current joint
    positions, which is how a stage keeps one arm out of the way while the other
    works.

    Args:
        env: The environment.
        target_poses: Arm name to a ``(1, 4, 4)`` or ``(4, 4)`` target pose.
        gripper_actions: Arm name to a finger target [rad].
        env_id: Index of the sub-environment.

    Returns:
        A 1-D action tensor of the environment's action dimension.

    Raises:
        ValueError: If the robot articulation is not reachable from the scene.
    """
    device = _device(env)
    slices = action_slices(env)
    robot = _robot(env)
    action_manager = getattr(_unwrapped(env), "action_manager", None)
    action = torch.zeros(action_manager.total_action_dim, device=device, dtype=torch.float32)

    held = _to_torch(robot.data.joint_pos, device)[env_id]
    for arm in slices:
        action[slices[arm]["arm"]] = held[slices[arm]["arm"]]

    for arm, pose in target_poses.items():
        if arm not in slices:
            continue
        joint_ids, _names = robot.find_joints(ARM_JOINT_NAMES[arm], preserve_order=True)
        finger_ids, _fingers = robot.find_joints(ARM_FINGER_JOINTS[arm], preserve_order=True)
        target = pose.reshape(4, 4)
        delta = _dls(arm_jacobian(env, arm), _pose_error(target, eef_pose(env, arm, env_id)))
        # The Jacobian's columns are articulation joint indices while the action
        # vector is indexed by action slice, so the arm's indices are resolved
        # rather than sliced: under ArmMode.RIGHT the right arm's action channels
        # start at zero and no longer line up with its joint indices.
        joints = _to_torch(robot.data.joint_pos, device)[env_id, joint_ids] + delta.squeeze(-1).reshape(-1)
        action[slices[arm]["arm"]] = joints

        finger = gripper_actions.get(arm)
        if finger is None:
            finger = _to_torch(robot.data.joint_pos, device)[env_id, finger_ids]
        action[slices[arm]["finger"]] = finger.reshape(-1).to(device=device, dtype=torch.float32)
    return action


def target_poses_from_actions(env, action: torch.Tensor) -> dict[str, torch.Tensor]:
    """Read a commanded joint action back as the end-effector poses it heads for.

    There is no forward-kinematics entry point on the articulation that can be
    evaluated for a hypothetical joint configuration, so the pose is propagated
    one control step along the Jacobian from the measured one. That is what this
    is actually asked for: ``annotate_demos`` passes the action the PD loop is
    currently tracking, so the quantity wanted is where the arm is going, not
    where a fully-converged solve would put it.

    Args:
        env: The environment.
        action: The commanded action, ``(num_envs, action_dim)``.

    Returns:
        Arm name to a ``(num_envs, 4, 4)`` pose tensor.
    """
    device = _device(env)
    slices = action_slices(env)
    robot = _robot(env)
    measured = _to_torch(robot.data.joint_pos, device)
    poses: dict[str, torch.Tensor] = {}

    for arm, ranges in slices.items():
        joint_ids, _names = robot.find_joints(ARM_JOINT_NAMES[arm], preserve_order=True)
        delta = action[:, ranges["arm"]] - measured[:, joint_ids]
        jacobian = arm_jacobian(env, arm).expand(action.shape[0], -1, -1)
        step = (jacobian @ delta.unsqueeze(-1)).squeeze(-1)

        current = eef_pose(env, arm, 0)
        pose = torch.eye(4, device=device, dtype=torch.float32).expand(action.shape[0], 4, 4).clone()
        pose[:, :3, :3] = current[:3, :3]
        pose[:, :3, 3] = current[:3, 3] + step[:, :3]
        # A small-angle rotation is the right first-order model for one step.
        angle = torch.linalg.norm(step[:, 3:], dim=-1)
        poses[arm] = pose @ _small_angle_rotation(step[:, 3:], angle)
    return poses


def _small_angle_rotation(axis_angle: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation for a batch of axis-angle vectors."""
    safe = torch.where(angle > 1e-8, angle, torch.ones_like(angle))
    axis = axis_angle / safe.unsqueeze(-1)
    zeros = torch.zeros_like(angle)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    outer = torch.stack(
        [
            torch.stack([zeros, -z, y], dim=-1),
            torch.stack([z, zeros, -x], dim=-1),
            torch.stack([-y, x, zeros], dim=-1),
        ],
        dim=-2,
    )
    identity = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).expand(axis_angle.shape[0], 3, 3)
    sin = torch.sin(angle).reshape(-1, 1, 1)
    cos = torch.cos(angle).reshape(-1, 1, 1)
    return identity + sin * outer + (1.0 - cos) * (outer @ outer)