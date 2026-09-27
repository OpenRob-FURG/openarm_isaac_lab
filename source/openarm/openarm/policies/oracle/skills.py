"""Scripted controllers for every :class:`~openarm.arena_tasks.staged.stages.Skill`.

Each controller is a small state machine that drives one stage. The controllers
share the same interface, so
:class:`~openarm.policies.oracle.staged_oracle.StagedOraclePolicy` can run any
task by walking its stage list without knowing what the stages actually do.

The controllers deliberately do not depend on any learned policy. They are
demonstration generators: their job is to produce correct trajectories and
recover from their own mistakes, not to be optimal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.array import convert_to_torch
from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_from_matrix
# Aliased: this module defines its own make_pose(position, euler) helper.
from isaaclab.utils.math import make_pose as make_pose_from_rot

from openarm.arena_tasks.staged.stages import Skill, Stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.scene import InteractiveScene

#: Action meaning for the scalar gripper channel of each arm.
GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

#: Maximum magnitude of the Cartesian position command per step, in action
#: units. Keeps the oracle from demanding an unreachable jump on the first step
#: of a stage, which is the main cause of scripted policies flailing at start.
MAX_POSITION_COMMAND = 0.08

#: Maximum magnitude of the rotation command per step, in axis-angle units.
MAX_ROTATION_COMMAND = 0.35

#: Peak speed above which an object is considered to be in flight rather than
#: held, used to decide whether a grasp actually took.
MOTION_SPEED_THRESHOLD = 0.6


def _zeros() -> torch.Tensor:
    return torch.zeros(7, dtype=torch.float32)


def clamp_action(action: torch.Tensor) -> torch.Tensor:
    """Clamp the position and rotation channels of a 7-DoF arm action.

    Only the first six channels are clamped. The gripper channel is left alone
    because it is a sign convention rather than a magnitude.

    Args:
        action: A single-arm action of shape ``(7,)``.

    Returns:
        The clamped action, shape ``(7,)``.
    """
    clamped = action.clone()
    clamped[:3] = clamped[:3].clamp(-MAX_POSITION_COMMAND, MAX_POSITION_COMMAND)
    clamped[3:6] = clamped[3:6].clamp(-MAX_ROTATION_COMMAND, MAX_ROTATION_COMMAND)
    return clamped


def pose_error(current: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the position and orientation error between two poses.

    Args:
        current: Current pose, shape ``(4, 4)``.
        target: Desired pose, shape ``(4, 4)``.

    Returns:
        A tuple of the position error norm ``(,)`` and the quaternion distance
        ``(,)``, both scalars.
    """
    position_error = torch.norm(current[:3, 3] - target[:3, 3])
    quat_current = quat_from_matrix(current[:3, :3])
    quat_target = quat_from_matrix(target[:3, :3])
    orientation_error = torch.norm(quat_current - quat_target, dim=-1).sum()
    return position_error, orientation_error


def make_pose(
    position: torch.Tensor,
    euler_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build a homogeneous transform from a position and an XYZ euler triple.

    Args:
        position: World position ``(3,)``.
        euler_xyz: Orientation as roll, pitch, yaw in radians.
        device: Device to build the result on.

    Returns:
        A ``(4, 4)`` homogeneous transform.
    """
    from isaaclab.utils.math import matrix_from_euler

    pose = torch.eye(4, device=device, dtype=torch.float32)
    pose[:3, 3] = position
    pose[:3, :3] = matrix_from_euler(
        torch.tensor([list(euler_xyz)], device=device, dtype=torch.float32), "XYZ"
    )[0]
    return pose


def local_point(object_pose: torch.Tensor, local_offset: tuple[float, float, float]) -> torch.Tensor:
    """Transform a point from an object's local frame into the world frame.

    Args:
        object_pose: Object pose ``(4, 4)``.
        local_offset: Point in the object's local frame ``(3,)``.

    Returns:
        The world-frame position ``(3,)``.
    """
    # The offset has to live on the pose's device: the physics backend hands
    # back CUDA tensors, and a CPU operand makes the matmul fail.
    offset = torch.as_tensor(local_offset, dtype=torch.float32, device=object_pose.device)
    return object_pose[:3, :3] @ offset + object_pose[:3, 3]


def grasp_pose(object_pose: torch.Tensor, stage: Stage) -> torch.Tensor:
    """Compute where the end effector must be to grasp an object.

    Args:
        object_pose: Pose of the object to grasp, shape ``(4, 4)``.
        stage: The stage describing the grasp.

    Returns:
        The target end-effector pose, shape ``(4, 4)``.
    """
    device = object_pose.device
    from isaaclab.utils.math import matrix_from_euler

    local_rotation = matrix_from_euler(
        torch.tensor([list(stage.grasp_euler)], device=device, dtype=torch.float32), "XYZ"
    )[0]
    if stage.use_world_orientation:
        rotation = local_rotation
    else:
        rotation = object_pose[:3, :3] @ local_rotation
    pose = torch.eye(4, device=device, dtype=torch.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = local_point(object_pose, stage.grasp_offset)
    return pose


def approach_pose(grasp: torch.Tensor, stage: Stage, distance: float | None = None) -> torch.Tensor:
    """Back the grasp pose off along the object's approach axis.

    Args:
        grasp: The grasp pose, shape ``(4, 4)``.
        stage: The stage describing the grasp.
        distance: Overrides ``stage.approach_distance`` [m].

    Returns:
        The pre-grasp pose, shape ``(4, 4)``.
    """
    offset = torch.zeros(3, device=grasp.device, dtype=torch.float32)
    axis = stage.approach_axis
    offset[axis] = float(stage.approach_distance if distance is None else distance)
    pose = grasp.clone()
    pose[:3, 3] = grasp[:3, 3] - grasp[:3, :3] @ offset
    return pose


def lift_pose(grasp: torch.Tensor, height: float) -> torch.Tensor:
    """Return a pose directly above the grasp pose at a given height.

    Args:
        grasp: The grasp pose, shape ``(4, 4)``.
        height: Height above the grasp point [m].

    Returns:
        The lifted pose, shape ``(4, 4)``.
    """
    pose = grasp.clone()
    pose[2, 3] = float(height)
    return pose


def embodiment_of(env) -> object:
    """Resolve the Arena embodiment attached to the running environment.

    Args:
        env: A Gym env or a manager-based env.

    Returns:
        The embodiment instance.

    Raises:
        AttributeError: If the environment has no Arena embodiment.
    """
    base = getattr(env, "unwrapped", env)
    cfg = getattr(base, "cfg", None)
    arena_env = getattr(cfg, "isaaclab_arena_env", None)
    embodiment = getattr(arena_env, "embodiment", None)
    if embodiment is None:
        raise AttributeError(
            f"No Arena embodiment reachable from {type(base).__name__}; "
            "expected cfg.isaaclab_arena_env.embodiment."
        )
    return embodiment


def device_of(env) -> torch.device:
    """Torch device the running environment uses.

    The physics backend hands back warp arrays that convert to the GPU while
    helper math defaults to the CPU, so the oracle has to pin one device for
    every tensor it builds or the pose maths mixes devices.

    Args:
        env: A Gym env or a manager-based env.

    Returns:
        The environment's torch device, falling back to CPU.
    """
    base = getattr(env, "unwrapped", env)
    device = getattr(base, "device", None)
    if device is None:
        return torch.device("cpu")
    return torch.device(device)


def scene_of(env) -> "InteractiveScene":
    """Resolve the live Arena scene from whatever the policy was handed.

    Arena registers the plain ``isaaclab.envs:ManagerBasedRLEnv`` for
    non-mimic runs, so ``get_object_poses`` / ``get_robot_eef_pose`` are *not*
    available on it: those live on the embodiment and on the mimic env only.
    Reading the scene directly works in every mode and with either physics
    backend, which is what the oracle needs.

    Args:
        env: A Gym env, a manager-based env, or an Arena scene.

    Returns:
        The ``InteractiveScene`` for the running environment.

    Raises:
        AttributeError: If no scene can be reached from ``env``.
    """
    if hasattr(env, "assets") and hasattr(env, "env_origins"):
        return env
    base = getattr(env, "unwrapped", env)
    scene = getattr(base, "scene", None)
    if scene is not None:
        return scene
    cfg = getattr(base, "cfg", None)
    arena_env = getattr(cfg, "isaaclab_arena_env", None)
    scene = getattr(arena_env, "scene", None)
    if scene is not None:
        return scene
    raise AttributeError(
        f"Could not reach the Arena scene from {type(env).__name__}; the oracle needs "
        "a manager-based env whose cfg carries isaaclab_arena_env.scene."
    )


def object_pose(env: "ManagerBasedEnv", name: str, env_id: int = 0) -> torch.Tensor:
    """Read one sub-environment's world pose for an object.

    Args:
        env: The environment.
        name: Scene name of the object.
        env_id: Index of the sub-environment.

    Returns:
        The pose ``(4, 4)``.

    Raises:
        KeyError: If the scene name is unknown.
    """
    scene = scene_of(env)
    if name not in scene.keys():
        raise KeyError(f"Object '{name}' not in the scene. Available: {sorted(scene.keys())}")
    asset = scene[name]
    if hasattr(asset.data, "root_pose_w"):
        raw = asset.data.root_pose_w
    else:
        raw = torch.cat(asset.get_world_poses(), dim=-1)
    device = device_of(env)
    pose = convert_to_torch(raw, dtype=torch.float32, device=device).clone()
    # Make the pose relative to the sub-environment origin, matching the frame
    # the observations and goals are expressed in.
    pose[:, :3] -= convert_to_torch(scene.env_origins, dtype=torch.float32, device=device)
    return make_pose_from_rot(pose[env_id, :3], matrix_from_quat(pose[env_id, 3:7]))


def eef_pose(env: "ManagerBasedEnv", arm: str, env_id: int = 0) -> torch.Tensor:
    """Read one sub-environment's world pose for an end effector.

    Args:
        env: The environment.
        arm: ``"left"`` or ``"right"``.
        env_id: Index of the sub-environment.

    Returns:
        The pose ``(4, 4)``.

    Raises:
        ValueError: If the scene has no end-effector frame for the given arm.
    """
    frame = eef_frame(env)
    names = list(frame.data.target_frame_names)
    index = _eef_index(names, arm)
    device = device_of(env)
    pos = convert_to_torch(frame.data.target_pos_w, dtype=torch.float32, device=device)
    quat = convert_to_torch(frame.data.target_quat_w, dtype=torch.float32, device=device)
    return make_pose_from_rot(pos[env_id, index], matrix_from_quat(quat[env_id, index]))


def eef_frame(env: "ManagerBasedEnv"):
    """Return the scene's ``ee_frame`` frame transformer.

    Raises:
        KeyError: If the scene has no end-effector frame transformer.
    """
    scene = scene_of(env)
    for key in scene.keys():
        if key.endswith("ee_frame"):
            return scene[key]
    raise KeyError(f"No 'ee_frame' sensor in the scene. Available: {sorted(scene.keys())}")


def _eef_index(names: "list[str]", arm: str) -> int:
    """Index of ``arm``'s target frame in the frame transformer's frame list.

    The frames are full prim names (``openarm_left_ee_tcp``) while ``arm`` is a
    short name (``left``), so the arm is matched as a whole token to avoid
    substring collisions.
    """
    for i, name in enumerate(names):
        tokens = name.replace("/", "_").split("_")
        if arm in tokens:
            return i
    raise ValueError(f"No end-effector frame for arm '{arm}'. Frames: {list(names)}")


def _eef_names(env: "ManagerBasedEnv") -> list[str]:
    """Ordered names of the embodiment's active end effectors.

    Two things have to line up here, and they do not agree on their own:

    * Stages address arms as ``"left"``/``"right"``, while the scene's frame
      transformer reports full USD frame names such as ``openarm_left_ee_tcp``.
    * The action layout is the embodiment's action-config declaration order
      (left then right), which need not match the frame transformer's order.

    So the active arms come from the action config -- it is the same thing the
    action manager uses -- and the frame transformer is only used as a fallback
    when no embodiment is reachable. Arms whose action term is ``None`` are
    inactive, which is what collapses a single-arm embodiment to one entry.
    """
    try:
        embodiment = embodiment_of(env)
    except AttributeError:
        embodiment = None

    if embodiment is not None:
        action_config = getattr(embodiment, "action_config", None)
        if action_config is not None:
            arms = [
                arm
                for arm in ("left", "right")
                if getattr(action_config, f"{arm}_arm_action", None) is not None
            ]
            if arms:
                return arms

    # Fallback: recover the arm token from the scene's frame names.
    names: list[str] = []
    for frame in eef_frame(env).data.target_frame_names:
        for arm in ("left", "right"):
            if arm in frame and arm not in names:
                names.append(arm)
                break
    if not names:
        raise ValueError("Could not determine any end-effector arms from the scene frames.")
    return names


def build_action(
    env: "ManagerBasedEnv",
    targets: dict[str, torch.Tensor],
    grippers: dict[str, float],
    env_id: int = 0,
) -> torch.Tensor:
    """Convert target poses and gripper commands into an environment action.

    Arms absent from ``targets`` are commanded to hold their current pose, which
    is how a stage keeps one arm out of the way while the other works.

    Args:
        env: The environment.
        targets: Mapping of arm name to the desired end-effector pose.
        grippers: Mapping of arm name to :data:`GRIPPER_OPEN` or
            :data:`GRIPPER_CLOSE`. Missing arms hold their current gripper.
        env_id: Index of the sub-environment.

    Returns:
        A 1-D action tensor of the environment's action dimension.
    """
    device = device_of(env)
    target_dict: dict[str, torch.Tensor] = {}
    gripper_dict: dict[str, torch.Tensor] = {}
    for arm in _eef_names(env):
        target = targets.get(arm)
        current = eef_pose(env, arm, env_id)
        # Every entry keeps the (1, 4, 4) shape _poses_to_action expects, so an
        # arm that is only being held still produces a valid zero-delta entry.
        pose = current if target is None else target.to(device)
        target_dict[arm] = pose.reshape(1, 4, 4)
        gripper = grippers.get(arm)
        if gripper is not None:
            gripper_dict[arm] = torch.as_tensor(gripper, dtype=torch.float32, device=device).reshape(1)
    return _poses_to_action(env, target_dict, gripper_dict, env_id=env_id)


def _poses_to_action(
    env: "ManagerBasedEnv",
    target_poses: Mapping[str, torch.Tensor],
    gripper_actions: Mapping[str, torch.Tensor],
    env_id: int = 0,
) -> torch.Tensor:
    """Convert target EEF poses into the environment's delta-pose action.

    The OpenArm bimanual action space is seven values per end effector -- three
    positional deltas, three rotational deltas as axis-angle, and one gripper
    value -- concatenated in the order the frame transformer reports its target
    frames. That is the same convention the differential IK controller expects,
    so the deltas are computed here rather than delegated: Arena only exposes
    ``target_eef_pose_to_action`` on the mimic env, and the non-mimic run uses
    the plain ``ManagerBasedRLEnv``.

    Args:
        env: The environment.
        target_poses: Mapping of arm name to a ``(1, 4, 4)`` target pose.
        gripper_actions: Mapping of arm name to a ``(1,)`` gripper command.
        env_id: Index of the sub-environment.

    Returns:
        A 1-D action tensor of the environment's action dimension.
    """
    device = device_of(env)
    actions: list[torch.Tensor] = []
    for arm in _eef_names(env):
        target = target_poses[arm]
        current = eef_pose(env, arm, env_id)

        delta_position = target[0, :3, 3] - current[:3, 3]

        # Delta rotation as a rotation matrix, then axis-angle, which is the
        # representation the action space uses.
        delta_rot_mat = target[0, :3, :3] @ current[:3, :3].transpose(-1, -2)
        delta_quat = quat_from_matrix(delta_rot_mat)
        delta_rotation = axis_angle_from_quat(delta_quat)

        pose_action = torch.cat([delta_position, delta_rotation]).to(device)
        gripper = gripper_actions.get(arm)
        if gripper is None:
            # Arms that are not being commanded hold their current gripper.
            gripper = _current_gripper(env, arm, env_id)
        actions.append(torch.cat([pose_action, torch.as_tensor(gripper, dtype=torch.float32, device=device).reshape(-1)]))
    return torch.cat(actions)


def _current_gripper(env: "ManagerBasedEnv", arm: str, env_id: int = 0) -> torch.Tensor:
    """Read an arm's current gripper command from the action manager.

    Falling back to fully open keeps an untouched arm from snapping shut.
    """
    action_manager = getattr(getattr(env, "unwrapped", env), "action_manager", None)
    last_action = getattr(action_manager, "action", None)
    names = list(_eef_names(env))
    if last_action is None or arm not in names:
        return torch.full((1,), float(GRIPPER_OPEN), dtype=torch.float32)
    index = names.index(arm) * 7 + 6
    return last_action[env_id, index].reshape(1).to(torch.float32)


class SkillController:
    """Base class for stage controllers.

    Subclasses override :meth:`step` and, when the stage manipulates something
    that can be checked, :meth:`verify`.
    """

    def __init__(self, stage: Stage, env_id: int = 0):
        """Store the stage and reset the controller.

        Args:
            stage: The stage this controller runs.
            env_id: Index of the sub-environment it drives. Each sub-environment
                gets its own controller so that a vectorized rollout produces
                independent, randomized demonstrations.
        """
        self.stage = stage
        self.env_id = env_id
        self.elapsed = 0
        self.phase = "approach"
        self.start_object_pose: torch.Tensor | None = None

    # -- lifecycle ---------------------------------------------------

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        """Reset controller state at the start of a stage.

        Args:
            env: The environment.
        """
        self.elapsed = 0
        self.phase = "approach"
        if self.stage.target:
            self.start_object_pose = object_pose(env, self.stage.target, self.env_id).clone()

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        """Advance the controller by one environment step.

        Args:
            env: The environment.

        Returns:
            The action for this step.
        """
        raise NotImplementedError

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        """Whether the stage has completed.

        Args:
            env: The environment.

        Returns:
            True when the stage is finished.
        """
        raise NotImplementedError

    def failed(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        """Whether the stage has failed unrecoverably.

        Args:
            env: The environment.

        Returns:
            True when the controller should give up on this stage.
        """
        return self.elapsed > self.stage.max_steps

    def verify(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        """Whether the stage actually achieved its effect.

        The default is optimistic. Controllers for stages that move something
        override this with a physical check so that a slip or a missed grasp
        triggers a retry instead of a silent advance.

        Args:
            env: The environment.

        Returns:
            True if the effect is confirmed.
        """
        return True

    # -- helpers -----------------------------------------------------

    def settled(self, env: "ManagerBasedEnv", target: torch.Tensor, rotation: bool = True) -> bool:
        """Check whether the end effector has reached a target pose.

        Args:
            env: The environment.
            target: The desired pose.
            rotation: Whether to also require orientation convergence.

        Returns:
            True if the pose is within the stage's tolerances.
        """
        position_error, orientation_error = pose_error(eef_pose(env, self.stage.arm, self.env_id), target)
        reached = position_error <= self.stage.tolerance_pos
        if rotation and self.stage.tolerance_rot > 0.0:
            reached = reached and orientation_error <= self.stage.tolerance_rot
        return bool(reached)

    def object_moved(self, env: "ManagerBasedEnv", minimum: float = 0.02) -> bool:
        """Check whether the target object has moved away from its initial pose.

        Args:
            env: The environment.
            minimum: Minimum displacement to count as movement [m].

        Returns:
            True if the object moved by at least ``minimum``.
        """
        if self.start_object_pose is None:
            return False
        current = object_pose(env, self.stage.target, self.env_id)
        return bool(torch.norm(current[:3, 3] - self.start_object_pose[:3, 3]) > minimum)

    def object_held(self, env: "ManagerBasedEnv", maximum: float = 0.12) -> bool:
        """Check whether the target object is close enough to be grasped.

        Args:
            env: The environment.
            maximum: Maximum distance from the end effector [m].

        Returns:
            True if the object is within ``maximum`` of the end effector.
        """
        distance = torch.norm(object_pose(env, self.stage.target, self.env_id)[:3, 3] - eef_pose(env, self.stage.arm, self.env_id)[:3, 3])
        return bool(distance < maximum)

    def wait(self) -> None:
        """Count one step of elapsed time."""
        self.elapsed += 1


class MoveController(SkillController):
    """Move an end effector to a fixed world anchor."""

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        target = make_pose(torch.as_tensor(self.stage.anchor, dtype=torch.float32), device=eef_pose(env, self.stage.arm, self.env_id).device)
        action = build_action(env, {self.stage.arm: target}, {}, self.env_id)
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        target = make_pose(torch.as_tensor(self.stage.anchor, dtype=torch.float32), device=eef_pose(env, self.stage.arm, self.env_id).device)
        return self.settled(env, target)


class PickController(SkillController):
    """Approach an object, close the gripper, and lift it."""

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        super().reset(env, env_id)
        self.pregrasp = approach_pose(grasp_pose(object_pose(env, self.stage.target, self.env_id), self.stage), self.stage)
        self.height = None

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        device = eef_pose(env, self.stage.arm, self.env_id).device
        if self.phase == "approach":
            action = build_action(env, {self.stage.arm: self.pregrasp}, {self.stage.arm: GRIPPER_OPEN}, self.env_id)
            if self.settled(env, self.pregrasp):
                self.phase = "grasp"
        elif self.phase == "grasp":
            target = grasp_pose(object_pose(env, self.stage.target, self.env_id), self.stage)
            action = build_action(env, {self.stage.arm: target}, {self.stage.arm: GRIPPER_CLOSE}, self.env_id)
            if self.settled(env, target) or self.elapsed > self.stage.max_steps // 3:
                self.phase = "lift"
                self.height = float(target[2, 3]) + self.stage.lift_height
        else:
            target = torch.eye(4, device=device, dtype=torch.float32)
            target[:3, :3] = eef_pose(env, self.stage.arm, self.env_id)[:3, :3]
            target[2, 3] = self.height if self.height is not None else target[2, 3]
            action = build_action(env, {self.stage.arm: target}, {self.stage.arm: GRIPPER_CLOSE}, self.env_id)
            if self.settled(env, target):
                self.phase = "done"
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "done"

    def verify(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        if not self.object_moved(env):
            return False
        _, speed = object_velocity(env, self.stage.target, self.env_id)
        return bool(speed < MOTION_SPEED_THRESHOLD) and self.object_held(env)


class PlaceController(SkillController):
    """Carry a held object to a target pose, release, and retreat."""

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        super().reset(env, env_id)
        self.target = torch.eye(4, device=eef_pose(env, self.stage.arm, self.env_id).device, dtype=torch.float32)
        self.target[:3, :3] = eef_pose(env, self.stage.arm, self.env_id)[:3, :3]
        self.target[2, 3] = self.stage.release_height
        self.retreat = self.target.clone()
        self.retreat[2, 3] = self.stage.release_height + self.stage.lift_height

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        if self.phase == "approach":
            action = build_action(env, {self.stage.arm: self.target}, {self.stage.arm: GRIPPER_CLOSE}, self.env_id)
            if self.settled(env, self.target):
                self.phase = "release"
        elif self.phase == "release":
            action = build_action(env, {self.stage.arm: self.target}, {self.stage.arm: GRIPPER_OPEN}, self.env_id)
            if self.settled(env, self.target):
                self.phase = "retreat"
        else:
            action = build_action(env, {self.stage.arm: self.retreat}, {self.stage.arm: GRIPPER_OPEN}, self.env_id)
            if self.settled(env, self.retreat):
                self.phase = "done"
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "done"

    def verify(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return True


class PickPlaceController(PickController):
    """A full pick followed by a place at a target height."""

    def __init__(self, stage: Stage, env_id: int = 0):
        super().__init__(stage, env_id)
        self.place_target: torch.Tensor | None = None
        self.released = False

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        super().reset(env, env_id)
        device = eef_pose(env, self.stage.arm, self.env_id).device
        self.place_target = torch.eye(4, device=device, dtype=torch.float32)
        self.place_target[:3, :3] = eef_pose(env, self.stage.arm, self.env_id)[:3, :3]
        self.place_target[2, 3] = self.stage.release_height
        self.released = False

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        if not self.released:
            action = super().step(env, env_id)
            if self.phase == "done":
                self.released = True
                self.phase = "transfer"
            return action
        if self.phase == "transfer":
            action = build_action(env, {self.stage.arm: self.place_target}, {self.stage.arm: GRIPPER_CLOSE}, self.env_id)
            if self.settled(env, self.place_target):
                self.phase = "release"
        elif self.phase == "release":
            action = build_action(env, {self.stage.arm: self.place_target}, {self.stage.arm: GRIPPER_OPEN}, self.env_id)
            if self.settled(env, self.place_target):
                self.phase = "retreat"
        else:
            retreat = self.place_target.clone()
            retreat[2, 3] = self.stage.release_height + self.stage.lift_height
            action = build_action(env, {self.stage.arm: retreat}, {self.stage.arm: GRIPPER_OPEN}, self.env_id)
            if self.settled(env, retreat):
                self.phase = "done"
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "done" and self.released

    def verify(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        if self.phase != "done":
            return False
        if self.stage.target:
            height = object_pose(env, self.stage.target, self.env_id)[2, 3]
            return bool(height > self.stage.release_height - 0.06)
        return True


class PressController(SkillController):
    """Push a pressable control and hold it."""

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        super().reset(env, env_id)
        self.target = torch.eye(4, device=eef_pose(env, self.stage.arm, self.env_id).device, dtype=torch.float32)
        self.target[:3, :3] = eef_pose(env, self.stage.arm, self.env_id)[:3, :3]
        self.target[2, 3] = float(self.stage.anchor[2])
        self.pressed = False

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        if self.phase == "press":
            target = self.target.clone()
            target[self.stage.push_axis] += self.stage.push_distance
            action = build_action(env, {self.stage.arm: target}, {}, self.env_id)
            if self.settled(env, target):
                self.phase = "hold"
                self.pressed = True
        else:
            action = build_action(env, {self.stage.arm: self.target}, {}, self.env_id)
            if self.pressed and self.elapsed > self.stage.max_steps // 2:
                self.phase = "done"
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "done"


def _fixture_joint(stage: Stage) -> tuple[str | None, str | None]:
    """Pull the articulation and joint name out of a stage's success goal.

    The joint-based goals are the ones that name a fixture joint; anything else
    leaves the controller without one, which it treats as "no joint to watch".
    """
    goal = getattr(stage, "goal", None)
    for candidate in (goal, *(getattr(goal, "goals", ()) or ())):
        if candidate is None:
            continue
        articulation = getattr(candidate, "articulation_name", None)
        joint = getattr(candidate, "joint_name", None)
        if articulation and joint:
            return articulation, joint
    return None, None


class ArticulationController(SkillController):
    """Drive an openable or turnable articulation to a target configuration.

    Args:
        stage: The stage describing the manipulation.
        controller: Which kind of articulation this is.
    """

    def __init__(self, stage: Stage, controller: str, env_id: int = 0):
        super().__init__(stage, env_id)
        self.controller = controller
        self.start_position = 0.0
        self.reached = False
        # The fixture joint comes from the stage's own success goal, which is
        # the only place the articulation and joint names are declared. Reading
        # the robot's joint_pos instead would silently track an arm joint and
        # report the fixture as never having moved.
        self.articulation_name, self.joint_name = _fixture_joint(stage)

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        super().reset(env, env_id)
        self.start_position = self._joint_position(env)
        self.reached = False
        self.phase = "drive"

    def _joint_position(self, env: "ManagerBasedEnv", env_id: int = 0) -> float:
        """Normalized position of the fixture's joint, or 0.0 when unavailable."""
        if not self.articulation_name or not self.joint_name:
            return 0.0
        scene = scene_of(env)
        # InteractiveScene has __getitem__/keys but no .get.
        if self.articulation_name not in scene.keys():
            return 0.0
        articulation = scene[self.articulation_name]
        if articulation is None or not hasattr(articulation.data, "joint_pos"):
            return 0.0
        names = list(articulation.data.joint_names)
        if self.joint_name not in names:
            return 0.0
        index = names.index(self.joint_name)
        pos = convert_to_torch(articulation.data.joint_pos, dtype=torch.float32, device=device_of(env))
        limits = convert_to_torch(articulation.data.joint_pos_limits, dtype=torch.float32, device=device_of(env))
        low, high = limits[env_id, index, 0], limits[env_id, index, 1]
        span = max(float(high - low), 1e-6)
        return float((pos[env_id, index] - low) / span)

    def _joint_delta(self, env: "ManagerBasedEnv", env_id: int = 0) -> float:
        return self._joint_position(env, env_id) - self.start_position

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        if self.controller == "turn":
            target = torch.eye(4, device=eef_pose(env, self.stage.arm, self.env_id).device, dtype=torch.float32)
            current = eef_pose(env, self.stage.arm, self.env_id)
            target[:3, :3] = current[:3, :3]
            target[:3, 3] = current[:3, 3]
            delta = float(self.stage.turn_angle) / max(1, self.stage.max_steps // 2)
            target[2, 3] = current[2, 3] + delta * 0.0
            action = build_action(env, {self.stage.arm: target}, {}, self.env_id)
        else:
            target = torch.eye(4, device=eef_pose(env, self.stage.arm, self.env_id).device, dtype=torch.float32)
            current = eef_pose(env, self.stage.arm, self.env_id)
            target[:3, :3] = current[:3, :3]
            target[:3, 3] = current[:3, 3]
            action = build_action(env, {self.stage.arm: target}, {}, self.env_id)
        if self._joint_delta(env) > 0.02:
            self.reached = True
        if self.reached and self.elapsed > self.stage.settle_steps:
            self.phase = "done"
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "done"

    def verify(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.reached


class OpenController(ArticulationController):
    """Open an openable articulation."""

    def __init__(self, stage: Stage, env_id: int = 0):
        super().__init__(stage, "open", env_id)


class CloseController(ArticulationController):
    """Close an openable articulation."""

    def __init__(self, stage: Stage, env_id: int = 0):
        super().__init__(stage, "close", env_id)


class TurnController(ArticulationController):
    """Turn a turnable articulation."""

    def __init__(self, stage: Stage, env_id: int = 0):
        super().__init__(stage, "turn", env_id)


class StrikeController(SkillController):
    """Swing a held tool into a target."""

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        super().reset(env, env_id)
        self.tool_pose = eef_pose(env, self.stage.arm, self.env_id).clone()
        self.cycle = 0
        self.progress = 0.0

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        amplitude = self.stage.strike_amplitude
        period = max(2, self.stage.max_steps // max(1, self.stage.strike_cycles))
        self.progress = float(amplitude * (self.elapsed % period) / period)
        target = self.tool_pose.clone()
        target[self.stage.strike_axis] -= self.progress
        action = build_action(env, {self.stage.arm: target}, {}, self.env_id)
        self.cycle = self.elapsed // period
        if self.elapsed >= period * self.stage.strike_cycles:
            self.phase = "done"
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "done"

    def verify(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.cycle > 0


class InsertController(SkillController):
    """Insert a held object into a receptacle held steady by the other arm."""

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        super().reset(env, env_id)
        device = eef_pose(env, self.stage.arm, self.env_id).device
        secondary = object_pose(env, self.stage.secondary, self.env_id) if self.stage.secondary else None
        self.insert = torch.eye(4, device=device, dtype=torch.float32)
        self.insert[:3, :3] = eef_pose(env, self.stage.arm, self.env_id)[:3, :3]
        if secondary is not None:
            self.insert[:3, 3] = secondary[:3, 3] + torch.as_tensor(self.stage.anchor, device=device)
        else:
            self.insert[:3, 3] = eef_pose(env, self.stage.arm, self.env_id)[:3, 3] + torch.as_tensor(self.stage.anchor, device=device)

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        if self.phase == "align":
            action = build_action(env, {self.stage.arm: self.insert}, {self.stage.arm: GRIPPER_CLOSE}, self.env_id)
            if self.settled(env, self.insert):
                self.phase = "insert"
        elif self.phase == "insert":
            depth = self.insert.clone()
            depth[2, 3] -= self.stage.push_distance
            action = build_action(env, {self.stage.arm: depth}, {self.stage.arm: GRIPPER_CLOSE}, self.env_id)
            if self.settled(env, depth):
                self.phase = "release"
        else:
            action = build_action(env, {self.stage.arm: self.insert}, {self.stage.arm: GRIPPER_OPEN}, self.env_id)
            if self.elapsed > self.stage.settle_steps:
                self.phase = "done"
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "done"

    def verify(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        if not self.stage.secondary:
            return True
        distance = torch.norm(
            object_pose(env, self.stage.target, self.env_id)[:3, 3] - object_pose(env, self.stage.secondary, self.env_id)[:3, 3]
        )
        return bool(distance < 0.08)


class HandoffController(SkillController):
    """Transfer a held object from one arm to the other."""

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        super().reset(env, env_id)
        self.midpoint = (eef_pose(env, "left", self.env_id)[:3, 3] + eef_pose(env, "right", self.env_id)[:3, 3]) / 2.0

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        giver = self.stage.arm
        taker = "right" if giver == "left" else "left"
        midpoint = torch.eye(4, device=eef_pose(env, giver, self.env_id).device, dtype=torch.float32)
        midpoint[:3, 3] = self.midpoint
        if self.phase == "present":
            action = build_action(env, {giver: midpoint}, {giver: GRIPPER_CLOSE}, self.env_id)
            if self.settled(env, midpoint):
                self.phase = "grasp"
        elif self.phase == "grasp":
            action = build_action(env, {taker: midpoint}, {taker: GRIPPER_CLOSE}, self.env_id)
            if self.settled(env, midpoint):
                self.phase = "release"
        else:
            action = build_action(env, {taker: midpoint}, {giver: GRIPPER_OPEN, taker: GRIPPER_CLOSE}, self.env_id)
            if self.settled(env, midpoint):
                self.phase = "retreat"
                self.stage = Stage(**{**self.stage.__dict__, "arm": taker})
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "retreat" and self.elapsed > self.stage.settle_steps


class StabilizeController(SkillController):
    """Hold the current pose and grip for a fixed number of steps."""

    def reset(self, env: "ManagerBasedEnv", env_id: int = 0) -> None:
        super().reset(env, env_id)
        self.hold = eef_pose(env, self.stage.arm, self.env_id).clone()

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        action = build_action(env, {self.stage.arm: self.hold}, {self.stage.arm: GRIPPER_CLOSE if self.stage.hold_gripper else GRIPPER_OPEN}, self.env_id)
        if self.elapsed >= self.stage.settle_steps:
            self.phase = "done"
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "done"


class DockController(SkillController):
    """Return both end effectors to a neutral pose."""

    def step(self, env: "ManagerBasedEnv", env_id: int = 0) -> torch.Tensor:
        device = eef_pose(env, "left", self.env_id).device
        targets = {}
        for arm in ("left", "right"):
            targets[arm] = make_pose(torch.as_tensor(self.stage.anchor, dtype=torch.float32, device=device), device=device)
        action = build_action(env, targets, {"left": GRIPPER_OPEN, "right": GRIPPER_OPEN}, self.env_id)
        if all(
            self.settled(env, targets[arm])
            for arm in _eef_names(env)
            if arm in targets
        ):
            self.phase = "done"
        self.wait()
        return action

    def done(self, env: "ManagerBasedEnv", env_id: int = 0) -> bool:
        return self.phase == "done"


#: Controller class for each skill.
CONTROLLERS: dict[Skill, type[SkillController]] = {
    Skill.MOVE: MoveController,
    Skill.PICK: PickController,
    Skill.PLACE: PlaceController,
    Skill.PICK_PLACE: PickPlaceController,
    Skill.PRESS: PressController,
    Skill.OPEN: OpenController,
    Skill.CLOSE: CloseController,
    Skill.TURN: TurnController,
    Skill.STRIKE: StrikeController,
    Skill.INSERT: InsertController,
    Skill.HANDOFF: HandoffController,
    Skill.STABILIZE: StabilizeController,
    Skill.DOCK: DockController,
}


def build_controller(stage: Stage, env_id: int = 0) -> SkillController:
    """Instantiate the controller for a stage.

    Args:
        stage: The stage to run.
        env_id: Index of the sub-environment the controller drives.

    Returns:
        A fresh controller.

    Raises:
        KeyError: If the stage's skill has no registered controller.
    """
    controller = CONTROLLERS.get(stage.skill)
    if controller is None:
        raise KeyError(f"No controller for skill '{stage.skill}'. Known: {sorted(s.value for s in CONTROLLERS)}")
    return controller(stage, env_id)


def object_velocity(env: "ManagerBasedEnv", name: str, env_id: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one sub-environment's linear and angular speed for an object.

    Args:
        env: The environment.
        name: Scene name of the object.
        env_id: Index of the sub-environment.

    Returns:
        A tuple of linear and angular speed scalars. Both are zero when the
        object has no rigid-body data, which is the case for purely visual or
        fixed assets.
    """
    try:
        data = env.unwrapped.physics_env.get_asset(name).data
        return data.root_lin_vel_w[env_id], data.root_ang_vel_w[env_id]
    except (AttributeError, KeyError, IndexError, TypeError):
        zero = torch.zeros((), device=env.unwrapped.device)
        return zero, zero
