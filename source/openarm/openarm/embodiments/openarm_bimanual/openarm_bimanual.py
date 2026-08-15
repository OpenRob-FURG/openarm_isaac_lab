from dataclasses import MISSING
from typing import Sequence

import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import mdp, ManagerBasedRLMimicEnv
from isaaclab.managers import ActionTermCfg, ObservationTermCfg, ObservationGroupCfg, SceneEntityCfg, EventTermCfg
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg, CameraCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.embodiments.common.mimic_utils import get_rigid_and_articulated_object_poses
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

from openarm.tasks.manager_based.openarm_manipulation.assets.openarm_bimanual import OPEN_ARM_HIGH_PD_CFG

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
        concatenate_observation_terms: bool = False,
        arm_mode: ArmMode | None = None,
    ):
        super().__init__(enable_cameras, initial_pose, concatenate_observation_terms, arm_mode)
        self.scene_config = OpenArmBimanualSceneCfg()
        self.action_config = OpenArmBimanualActionsCfg()
        self.observation_config = OpenArmBimanualObservationsCfg()
        self.event_config = OpenArmBimanualEventsCfg()
        self.camera_config = OpenArmBimanualCameraCfg()
        self.reward_config = None
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

    def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
        if arm_mode == ArmMode.RIGHT:
            return "right_ee_tcp"
        return "left_ee_tcp"

    def get_command_body_name(self) -> str:
        if self.action_config.left_arm_action is not None:
            return self.action_config.left_arm_action.body_name
        return self.action_config.right_arm_action.body_name


@configclass
class OpenArmBimanualSceneCfg:
    robot: ArticulationCfg = OPEN_ARM_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=dict(
                openarm_left_joint4=torch.pi / 2,
                openarm_right_joint4=torch.pi / 2,
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
            width=256,
            height=256,
        )


@configclass
class OpenArmBimanualActionsCfg:
    left_arm_action: ActionTermCfg = mdp.DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["openarm_left_joint.*"],
        body_name="openarm_left_ee_tcp",
        controller=mdp.DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
    )

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
        controller=mdp.DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
        ),
    )

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
        body_poses = ObservationTermCfg(
            func=all_poses_w,
        )
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos,
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel,
        )
        last_action = ObservationTermCfg(
            func=mdp.last_action,
        )

    policy = PolicyCfg(concatenate_terms=False)


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

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Takes a target pose and gripper action and returns a normalized delta pose action.

        Args:
            target_eef_pose_dict: Dictionary of 4x4 target eef pose for each end-effector.
            gripper_action_dict: Dictionary of gripper actions for each end-effector.
            action_noise_dict: Noise to add to the action for each end-effector. If None, no noise is added.
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

        Args:
            action: Environment action. Shape is (num_envs, action_dim)

        Returns:
            A dictionary of eef pose torch.Tensor that @action corresponds to
        """
        pose_dict = dict()
        for eef_index, eef_name in enumerate(self._get_eef_names()):
            delta_position = action[:, (eef_index * 7) : (eef_index * 7) + 3]
            delta_rotation = action[:, (eef_index * 7) + 3 : (eef_index * 7) + 6]

            # current position and rotation
            curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
            curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

            # get pose target
            target_pos = curr_pos + delta_position

            # Convert delta_rotation to axis angle form
            delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
            delta_rotation_axis = delta_rotation / delta_rotation_angle

            # Handle invalid division for the case when delta_rotation_angle is close to zero
            is_close_to_zero_angle = torch.isclose(
                delta_rotation_angle, torch.zeros_like(delta_rotation_angle)
            ).squeeze(1)
            delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[
                is_close_to_zero_angle
            ]

            delta_quat = PoseUtils.quat_from_angle_axis(
                delta_rotation_angle.squeeze(1), delta_rotation_axis
            ).squeeze(0)
            delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
            target_rot = torch.matmul(delta_rot_mat, curr_rot)

            target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()
            pose_dict.update({eef_name: target_poses})
        return pose_dict

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extracts the gripper actuation part from a sequence of env actions.

        Args:
            actions: environment actions. The shape is (num_envs, num steps in a demo, action_dim).

        Returns:
            A dictionary of torch.Tensor gripper actions. Key to each dict is an eef_name.
        """
        eef_actions = dict()
        for eef_index, eef_name in enumerate(self._get_eef_names()):
            eef_actions[eef_name] = actions[:, [eef_index * 7 + 6]]
        return eef_actions

    def get_object_poses(self, env_ids: Sequence[int] | None = None) -> dict:
        """Gets the pose of each object (rigid and articulated) in the current scene."""
        if env_ids is None:
            env_ids = slice(None)

        state = self.scene.get_state(is_relative=True)
        return get_rigid_and_articulated_object_poses(state, env_ids)
