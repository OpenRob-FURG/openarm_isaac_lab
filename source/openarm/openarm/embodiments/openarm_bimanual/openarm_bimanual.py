from dataclasses import MISSING
from typing import Sequence

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp, ManagerBasedRLMimicEnv
from isaaclab.managers import ActionTermCfg, ObservationTermCfg, ObservationGroupCfg, SceneEntityCfg, EventTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg, CameraCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab_teleop.xr_cfg import XrCfg

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import OPEN_ARM_HIGH_PD_CFG, OPEN_ARM_HIGH_PD_FIXED_BASE_CFG
#from isaaclab_tasks.manager_based.manipulation.reach.config.openarm.bimanual.joint_pos_env_cfg import OPENARM_BI_HIGH_PD_CFG

from .mdp import rel_ee_pos, rel_ee_quat, gripper_opening, all_poses_w


@register_asset
class OpenArmBimanualEmbodiment(EmbodimentBase):
    """OpenArm bimanual embodiment.

    The robot is controlled through differential IK (relative pose commands) for both arms
    together with binary gripper actions. In single-arm modes the unused arm is disabled.
    """

    name = "openarm_bimanual"
    default_arm_mode = ArmMode.DUAL_ARM

    def __init__(
        self,
        enable_cameras: bool = False,
        initial_pose=None,
        concatenate_observation_terms: bool = True,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = OpenArmBimanualSceneCfg(num_envs=1, env_spacing=30.0, replicate_physics=True)
        self.action_config = OpenArmBimanualActionsCfg()
        self.observation_config = OpenArmBimanualObservationsCfg()
        self.event_config = OpenArmBimanualEventsCfg()
        self.camera_config = OpenArmBimanualCameraCfg()
        self.reward_config = None
        self.xr = XrCfg()
        self.mimic_env = OpenArmBimanualMimicEnv

        if self.arm_mode in [ArmMode.SINGLE_ARM, ArmMode.LEFT]:
            self.action_config.right_arm_action = None
            self.action_config.right_hand_action = None
            self.scene_config.robot.init_state.joint_pos["openarm_right_joint4"] = 0.0
        elif self.arm_mode == ArmMode.RIGHT:
            self.action_config.left_arm_action = None
            self.action_config.left_hand_action = None
            self.scene_config.robot.init_state.joint_pos["openarm_left_joint4"] = 0.0

        self.observation_config.policy.concatenate_terms = self.concatenate_observation_terms

    def get_teleop_target_frame_prim_path(self):
        return "/World/envs/env_0/Robot/openarm_body_link"

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        if arm_mode == ArmMode.RIGHT:
            return "right_ee_tcp"
        return "left_ee_tcp"

    def get_command_body_name(self) -> str:
        if self.action_config.left_arm_action is not None:
            return self.action_config.left_arm_action.body_name
        return self.action_config.right_arm_action.body_name


@configclass
class OpenArmBimanualSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = OPEN_ARM_HIGH_PD_FIXED_BASE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=dict(
                openarm_left_joint4=1.65,
                openarm_right_joint4=1.65,
                openarm_left_finger_joint1=0.043,
                openarm_right_finger_joint1=0.043,
            )
        ),
    )

    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_left_ee_tcp",
                name="left_ee_tcp",
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/openarm_right_ee_tcp",
                name="right_ee_tcp",
            ),
        ],
    )


@configclass
class OpenArmBimanualCameraCfg:
    """Configuration for cameras."""

    top_camera: CameraCfg | TiledCameraCfg = MISSING

    def __post_init__(self):
        self.top_camera = TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/openarm_body_link/top_camera",
            offset=TiledCameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.65),
            rot=(0.0, 0.383, 0.0, 0.924),
                convention="world",
            ),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=80.0,
                clipping_range=(0.001, 100.0),
            ),
            width=640,
            height=480,
            data_types=["rgb"]
        )


@configclass
class OpenArmBimanualActionsCfg:
    left_arm_action: ActionTermCfg = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["openarm_left_joint.*"],
        body_name="openarm_left_ee_tcp",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            # The reset arm pose is close to a wrist singularity. A modest
            # damping term prevents millimetre Cartesian steps from turning
            # into large joint targets that kick the TCP away from the goal.
            #ik_params={"lambda_val": 0.2},
        ),
    )

    '''left_arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_left_joint.*"],
        # Absolute joint targets, not a delta on the reset pose. Every consumer of
        # this action space already treats it as absolute: the IK solve in
        # :mod:`~openarm.embodiments.openarm_bimanual.kinematics` returns joint
        # positions, the recorded demonstrations are joint positions, and Mimic
        # regenerates them as joint positions. Leaving the default offset in place
        # would silently subtract the reset pose from every one of those.
        use_default_offset=False,
    )'''

    left_hand_action: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_left_finger_joint1"],
        open_command_expr=dict(openarm_left_finger_joint1=0.043),
        close_command_expr=dict(openarm_left_finger_joint1=0.0),
    )

    right_arm_action: ActionTermCfg = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["openarm_right_joint.*"],
        body_name="openarm_right_ee_tcp",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            #ik_params={"lambda_val": 0.2},
        ),
    )

    '''right_arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_right_joint.*"],
        # Absolute, for the same reason as the left arm.
        use_default_offset=False,
    )'''

    right_hand_action: ActionTermCfg = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["openarm_right_finger_joint1"],
        open_command_expr=dict(openarm_right_finger_joint1=0.043),
        close_command_expr=dict(openarm_right_finger_joint1=0.0),
    )


@configclass
class OpenArmBimanualObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        left_eef_pos = ObservationTermCfg(
            func=rel_ee_pos,
            params=dict(arm="left"),
        )
        left_eef_quat = ObservationTermCfg(
            func=rel_ee_quat,
            params=dict(arm="left"),
        )
        left_gripper_pos = ObservationTermCfg(
            func=gripper_opening,
            params=dict(arm="left"),
        )

        right_eef_pos = ObservationTermCfg(
            func=rel_ee_pos,
            params=dict(arm="right"),
        )
        right_eef_quat = ObservationTermCfg(
            func=rel_ee_quat,
            params=dict(arm="right"),
        )
        right_gripper_pos = ObservationTermCfg(
            func=gripper_opening,
            params=dict(arm="right"),
        )
        #body_poses = ObservationTermCfg(
        #    func=all_poses_w,
        #)
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel,
        )
        last_action = ObservationTermCfg(
            func=mdp.last_action,
        )

    policy = PolicyCfg(concatenate_terms=True)


@configclass
class OpenArmBimanualEventsCfg:
    reset_scene_to_initial_state = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
    )


class OpenArmBimanualMimicEnv(ManagerBasedRLMimicEnv):
    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get current robot end effector pose.

        Args:
            eef_name: Name of the end effector ("left" or "right").
            env_ids: Environment indices to get the pose for. If None, all envs are considered.

        Returns:
            A torch.Tensor eef pose matrix. Shape is (len(env_ids), 4, 4)
        """
        if eef_name == "robot":
            eef_name = "left"
        if env_ids is None:
            env_ids = slice(None)

        eef_pos = rel_ee_pos(env=self, arm=eef_name)[env_ids]
        eef_quat = rel_ee_quat(env=self, arm=eef_name)[env_ids]
        return PoseUtils.make_pose(eef_pos, PoseUtils.matrix_from_quat(eef_quat))

    def _get_eef_names(self) -> list[str]:
        subtask_configs = getattr(self.cfg, "subtask_configs", None)
        if subtask_configs:
            return list(subtask_configs.keys())
        if self.single_action_space.shape[0] == 7:
            return ["left"]
        return ["left", "right"]

    def _joint_action_slices(self) -> dict[str, dict[str, slice]]:
        """Return each arm's channel ranges in the flat action vector.

        Offsets come from the live action manager rather than being hardcoded.
        The OpenArm action layout is declared in :class:`OpenArmBimanualActionsCfg`
        and its terms are dropped per arm mode, so dropping the *left* arm's terms
        for ``ArmMode.RIGHT`` moves the right arm's slice down to offset zero. A
        hardcoded offset would write right-arm joints into left-arm slots for half
        the arm modes, and matching terms by prefix would let the one-wide finger
        term overwrite the arm's own range.

        Returns:
            Arm name to its ``arm`` and ``finger`` slices.
        """
        from openarm.embodiments.openarm_bimanual import kinematics

        return kinematics.action_slices(self)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Turn a target pose per end effector into an absolute joint position action.

        The OpenArm is commanded in absolute joint position, so a Cartesian target
        is solved for by one damped least-squares step along the arm's Jacobian.
        The conversion lives in
        :mod:`openarm.embodiments.openarm_bimanual.kinematics` because the scripted
        oracle needs the identical solve: a target pose the oracle reached and the
        same pose the datagen replays have to produce the same joint targets.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            action_noise_dict: Unused in joint space. Cartesian noise has no
                meaning as a joint target, so it is ignored rather than applied
                to the solved joints, where it would show up as tracking error.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().

        Raises:
            ValueError: If the robot articulation is not reachable from the scene.
        """
        from openarm.embodiments.openarm_bimanual import kinematics

        return kinematics.joint_targets_from_poses(
            self, target_eef_pose_dict, gripper_action_dict, env_id
        )

    def _delta_pose_target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Return a normalized delta pose action, for a delta-pose action space.

        Kept as the reference implementation of the interface. It is not wired up:
        the OpenArm action space is absolute joint position, so this path is
        reachable only if an action space of delta poses is configured, which the
        shipped config does not do.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            action_noise_dict: Noise to add to the action for each end-effector.
            env_id: Environment index to get the action for.

        Returns:
            An action torch.Tensor that's compatible with env.step().
        """
        eef_actions = []
        for eef_name in self._get_eef_names():
            # target position and rotation
            target_eef_pose = target_eef_pose_dict[eef_name]
            target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

            # current position and rotation
            curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            # normalized delta position action
            delta_position = target_pos - curr_pos

            # normalized delta rotation action
            delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
            delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
            delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

            # get gripper action for single eef
            gripper_action = gripper_action_dict[eef_name]

            # add noise to action
            pose_action = torch.cat([delta_position, delta_rotation], dim=0)
            if action_noise_dict is not None:
                noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
                pose_action += noise
                pose_action = torch.clamp(pose_action, -1.0, 1.0)
            eef_actions.append(torch.cat([pose_action, gripper_action], dim=0))

        return torch.cat(eef_actions, dim=0)

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        """Converts an action to a target pose for the end effector controller.

        The OpenArm's action space is absolute joint position, so this is the
        inverse of :meth:`target_eef_pose_to_action`: the pose a commanded set of
        joints is heading for. It is read by the annotation recorder, which asks
        for the pose the PD loop is currently tracking, so the commanded pose is
        propagated one control step along the arm's Jacobian rather than
        integrated to convergence.

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """
        from openarm.embodiments.openarm_bimanual import kinematics

        return kinematics.target_poses_from_actions(self, action)

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Evaluate the task's stage goals into Mimic's subtask segmentation signals.

        The signals are the goals themselves rather than a second, hand-written set
        of predicates, so the split the datagen makes cannot disagree with the
        success condition the environment terminates on. Names come from the same
        function that built the ``SubTaskConfig`` list, which is the only thing
        keeping the two in step.

        Args:
            env_ids: Environment indices to evaluate. Defaults to all.

        Returns:
            Signal name to a boolean tensor, matching the signals the configured
            subtasks terminate on.
        """
        from openarm.arena_tasks.staged.mimic import staged_signals

        task = getattr(self.cfg, "isaaclab_arena_env", None)
        spec = getattr(getattr(task, "task", None), "spec", None)
        if spec is None:
            raise AttributeError(
                "The staged task is not reachable from this environment's config, so its subtask "
                "signals cannot be evaluated. Arena attaches it as cfg.isaaclab_arena_env.task."
            )

        arm_mode = getattr(self.cfg, "arm_mode", None)
        if arm_mode is None:
            raise AttributeError("The environment config carries no arm_mode, so the active arms are unknown.")

        signals = staged_signals(self, spec, arm_mode)
        if env_ids is None:
            return signals
        return {name: flags[env_ids] for name, flags in signals.items()}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract the finger-joint channel from a sequence of env actions.

        The OpenArm gripper is a position-controlled finger joint, not a binary
        actuator, so the "gripper action" Mimic carries is the finger target in
        radians and the value passes through untouched. Reading it out of a joint
        action needs the finger's real offset, which depends on the arm mode.

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        slices = self._joint_action_slices()
        return {arm: actions[:, ranges["finger"]] for arm, ranges in slices.items()}

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict:
        """Gets the pose of each object (rigid and articulated) in the current scene."""
        if env_ids is None:
            env_ids = slice(None)

        state = self.scene.get_state(is_relative=True)
        return get_rigid_and_articulated_object_poses(state, env_ids)
