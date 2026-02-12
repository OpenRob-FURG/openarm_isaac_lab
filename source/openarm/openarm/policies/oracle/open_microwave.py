from isaaclab_arena.policy.policy_base import PolicyBase
from openarm.embodiments.openarm_bimanual.openarm_bimanual import OpenArmBimanualMimicEnv
import isaaclab.utils.math as PoseUtils
import torch

class OracleOpenMicrowavePolicyStates:
    APPROACH = 0
    GRASP = 1
    PULL_DOOR = 2

class OracleOpenMicrowavePolicy(PolicyBase):
    def __init__(self, microwave_handle_offset):
        super().__init__()
        self.microwave_handle_offset = microwave_handle_offset
        self.reset()

    def get_action(self, env: OpenArmBimanualMimicEnv, observation):
        action = torch.zeros(size=env.action_space.shape, device=env.device)
        
        if self.state == OracleOpenMicrowavePolicyStates.APPROACH:
            if self.state_step_counter == 0:
                microwave_pose = env.get_object_poses()["microwave"]
                right_eef_pose = env.get_robot_eef_pose(eef_name="right")
                grasp_pose = right_eef_pose.clone()
                grasp_pose[..., 0, 3] = microwave_pose[..., 0, 3] + self.microwave_handle_offset[0]
                grasp_pose[..., 1, 3] = microwave_pose[..., 1, 3] + self.microwave_handle_offset[1]
                grasp_pose[..., 2, 3] = microwave_pose[..., 2, 3] + self.microwave_handle_offset[2]
                self.right_target_pose = grasp_pose
                self.left_target_pose = env.get_robot_eef_pose(eef_name="left")
            action[..., :] = env.target_eef_pose_to_action(
                target_eef_pose_dict=dict(
                    left=self.left_target_pose[0],
                    right=self.right_target_pose[0]
                ),
                gripper_action_dict=dict(
                    left=torch.as_tensor([1.0,], device=env.device),
                    right=torch.as_tensor([1.0,], device=env.device)
                ),
                action_noise_dict=dict(
                    left=0.0,
                    right=0.03
                )
            )    
            right_eef_pose = env.get_robot_eef_pose(eef_name="right")
            right_quat = PoseUtils.quat_from_matrix(right_eef_pose[..., :3, :3])
            right_pos = right_eef_pose[..., :3, 3].reshape((1, 3))
            target_quat = PoseUtils.quat_from_matrix(self.right_target_pose[..., :3, :3])
            target_pos = self.right_target_pose[..., :3, 3].reshape((1, 3))
            if (torch.norm(right_pos[0] - target_pos[0], p=2) <= 0.05 and torch.norm(right_quat[0] - target_quat[0], p=2) <= 0.1) or self.state_step_counter >= 30:
                self.state = OracleOpenMicrowavePolicyStates.GRASP
                self.state_step_counter = 0

        if self.state == OracleOpenMicrowavePolicyStates.GRASP:
            action[..., 6] = 1.0
            action[..., 13] = -1.0
            if self.state_step_counter >= 30:
                self.state = OracleOpenMicrowavePolicyStates.PULL_DOOR
                self.state_step_counter = 0

        if self.state == OracleOpenMicrowavePolicyStates.PULL_DOOR:
            action[..., 6] = 1.0
            action[..., 13] = -1.0
            action[..., 7] = -0.025

        '''if self.state == OraclePourPolicyStates.PREPARE_ARMS:
            target_left_ee_pose = torch.zeros(size=(4, 4), device=env.device)
            target_right_ee_pose = torch.zeros(size=(4, 4), device=env.device)
            
            target_left_ee_pose[1, 3] = 0.3
            target_left_ee_pose[2, 3] = 0.45
            target_left_ee_pose[0, 0] = 1.0
            target_left_ee_pose[1, 1] = 1.0
            target_left_ee_pose[2, 2] = 1.0

            target_right_ee_pose[1, 3] = -0.3
            target_right_ee_pose[2, 3] = 0.45
            target_right_ee_pose[0, 0] = 1.0
            target_right_ee_pose[1, 1] = 1.0
            target_right_ee_pose[2, 2] = 1.0

            action[0, :] = env.target_eef_pose_to_action(
                target_eef_pose_dict=dict(
                    left=target_left_ee_pose,
                    right=target_right_ee_pose
                ),
                gripper_action_dict=dict(
                    left=torch.as_tensor([1.0,], device=env.device),
                    right=torch.as_tensor([1.0,], device=env.device)
                ),
                action_noise_dict=dict(
                    left=0.003,
                    right=0.003
                )
            )

            if self.state_step_counter >= 120:
                self.state_step_counter = 0
                self.state= OraclePourPolicyStates.APPROACH
        
        if self.state == OraclePourPolicyStates.APPROACH:
            source_container_pose = env.get_object_poses()[self.source_recipient.name]
            source_grasp_pose = source_container_pose.clone()
            source_grasp_pose[..., 0, 3] += self.source_recipient.grasp_pose.position_xyz[0]
            source_grasp_pose[..., 1, 3] += self.source_recipient.grasp_pose.position_xyz[1]
            source_grasp_pose[..., 2, 3] += self.source_recipient.grasp_pose.position_xyz[2]
            source_grasp_pose[..., :3, :3] = PoseUtils.matrix_from_quat(self.source_recipient.grasp_pose.to_tensor(env.device)[..., 3:7])
            source_pre_grasp_pose = source_grasp_pose.clone()
            source_pre_grasp_pose[..., self.source_recipient.grasp_approach_axis] -= self.source_recipient.grasp_approach_direction

            destination_container_pose = env.get_object_poses()[self.destination_recipient.name]
            destination_grasp_pose = destination_container_pose.clone()
            destination_grasp_pose[..., 0, 3] += self.destination_recipient.grasp_pose.position_xyz[0]
            destination_grasp_pose[..., 1, 3] += self.destination_recipient.grasp_pose.position_xyz[1]
            destination_grasp_pose[..., 2, 3] += self.destination_recipient.grasp_pose.position_xyz[2]
            destination_grasp_pose[..., :3, :3] = PoseUtils.matrix_from_quat(self.destination_recipient.grasp_pose.to_tensor(env.device)[..., 3:7])
            destination_pre_grasp_pose = destination_grasp_pose.clone()
            destination_pre_grasp_pose[..., self.destination_recipient.grasp_approach_axis] -= self.destination_recipient.grasp_approach_direction

            action[0, :] = env.target_eef_pose_to_action(
                target_eef_pose_dict=dict(
                    left=destination_pre_grasp_pose[0],
                    right=source_pre_grasp_pose[0]
                ),
                gripper_action_dict=dict(
                    left=torch.as_tensor([1.0,], device=env.device),
                    right=torch.as_tensor([1.0,], device=env.device)
                ),
                action_noise_dict=dict(
                    left=0.003,
                    right=0.003
                )
            )

            if self.state_step_counter >= 240:
                self.state_step_counter = 0
                self.state = OraclePourPolicyStates.GRASP
        
        if self.state == OraclePourPolicyStates.GRASP:
            if self.state_step_counter <= 10:
                action[..., 0+self.destination_recipient.grasp_approach_axis] = self.destination_recipient.grasp_approach_direction/10.0
                action[..., 6] = 1.0
                action[..., 7+self.source_recipient.grasp_approach_axis] = self.source_recipient.grasp_approach_direction/10.0
                action[..., 13] = 1.0
            elif self.state_step_counter <= 20:
                action[..., 6] = -1.0
                action[..., 13] = -1.0
            elif self.state_step_counter <= 30:
                action[..., 2] = 0.01
                action[..., 6] = -1.0
                action[..., 9] = 0.01
                action[..., 13] = -1.0
            else:
                self.state_step_counter = 0
                left_eef_pose = env.get_robot_eef_pose(eef_name="left")
                right_eef_pose = env.get_robot_eef_pose(eef_name="right")

                left_eef_pose[..., 0, 3] = 0.4
                left_eef_pose[..., 1, 3] = 0.1
                left_eef_pose[..., 2, 3] = 0.3

                right_eef_pose[..., 0, 3] = 0.4
                right_eef_pose[..., 1, 3] = -0.1
                right_eef_pose[..., 2, 3] = 0.4
                self.target_left_pose = left_eef_pose
                self.target_right_pose = right_eef_pose
                self.state = OraclePourPolicyStates.APPROACH_BOTH_CONTAINERS

        if self.state == OraclePourPolicyStates.APPROACH_BOTH_CONTAINERS:
            action[0, :] = env.target_eef_pose_to_action(
                target_eef_pose_dict=dict(
                    left=self.target_left_pose[0],
                    right=self.target_right_pose[0]
                ),
                gripper_action_dict=dict(
                    left=torch.as_tensor([-1.0,], device=env.device),
                    right=torch.as_tensor([-1.0,], device=env.device)
                ),
                action_noise_dict=dict(
                    left=0.003,
                    right=0.003
                )
            )

            if self.state_step_counter >= 60:
                self.state_step_counter = 0
                self.state = OraclePourPolicyStates.POUR_SOURCE_DESTINATION

        if self.state == OraclePourPolicyStates.POUR_SOURCE_DESTINATION:
            action[..., 6] = -1.0
            action[..., 13] = -1.0
            if self.state_step_counter <= 60:
                action[..., 7+4] = 1.570/60.0'''

        self.state_step_counter += 1
        return action
    
    def reset(self, env_ids = None):
        self.state = OracleOpenMicrowavePolicyStates.APPROACH
        self.state_step_counter = 0