from isaaclab_arena.policy.policy_base import PolicyBase
from openarm.embodiments.openarm_bimanual.openarm_bimanual import OpenArmBimanualMimicEnv
import isaaclab.utils.math as PoseUtils
import argparse
import torch

class OraclePickPlaceExhaustPipePolicyStates:
    APPROACH_PIPE = 0
    GRASP_PIPE = 1
    APPROACH_TRANSFER = 2
    TRANSFER_PIPE = 3
    APPROACH_CRATE = 4
    DROP_PIPE = 5

class OraclePickPlaceExhaustPipePolicy(PolicyBase):
    def __init__(self, pipe, destination):
        super().__init__(config=None)
        self.pipe = pipe
        self.destination = destination
        self.reset()

    def get_action(self, env: OpenArmBimanualMimicEnv, observation):
        action = torch.zeros(size=env.action_space.shape, device=env.device)
        if self.state == OraclePickPlaceExhaustPipePolicyStates.APPROACH_PIPE:
            if self.state_step_counter == 0:
                pipe_pose = env.get_object_poses()[self.pipe.name]
                left_eef_pose = env.get_robot_eef_pose(eef_name="left")
                grasp_pose = left_eef_pose.clone()
                grasp_pose[..., 0, 3] = pipe_pose[..., 0, 3] - 0.1
                grasp_pose[..., 1, 3] = pipe_pose[..., 1, 3]
                grasp_pose[..., 2, 3] = pipe_pose[..., 2, 3] + 0.1
                grasp_pose[..., :3, :3] = PoseUtils.matrix_from_euler(euler_angles=torch.as_tensor([[0.0, torch.pi/2, 0.0]], device=env.device), convention="XYZ")
                self.right_target_pose = env.get_robot_eef_pose(eef_name="right")
                self.left_target_pose = grasp_pose
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
                    right=0.0
                )
            )    
            left_eef_pose = env.get_robot_eef_pose(eef_name="left")
            left_quat = PoseUtils.quat_from_matrix(left_eef_pose[..., :3, :3])
            left_pos = left_eef_pose[..., :3, 3].reshape((1, 3))
            target_quat = PoseUtils.quat_from_matrix(self.left_target_pose[..., :3, :3])
            target_pos = self.left_target_pose[..., :3, 3].reshape((1, 3))
            if (torch.norm(left_pos[0] - target_pos[0], p=2) <= 0.08 and torch.norm(left_quat[0] - target_quat[0], p=2) <= 0.1):
                self.state = OraclePickPlaceExhaustPipePolicyStates.GRASP_PIPE
                self.state_step_counter = 0
            
        if self.state == OraclePickPlaceExhaustPipePolicyStates.GRASP_PIPE:
            if self.state_step_counter < 20:
                self.left_target_pose[..., 0, 3] += 0.01
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
                        right=0.0
                    )
                )
            elif self.state_step_counter < 30:
                action[..., :] = env.target_eef_pose_to_action(
                    target_eef_pose_dict=dict(
                        left=self.left_target_pose[0],
                        right=self.right_target_pose[0]
                    ),
                    gripper_action_dict=dict(
                        left=torch.as_tensor([-1.0,], device=env.device),
                        right=torch.as_tensor([1.0,], device=env.device)
                    ),
                    action_noise_dict=dict(
                        left=0.0,
                        right=0.0
                    )
                )
            elif self.state_step_counter < 40:
                self.left_target_pose[..., 2, 3] += 0.005
                action[..., :] = env.target_eef_pose_to_action(
                    target_eef_pose_dict=dict(
                        left=self.left_target_pose[0],
                        right=self.right_target_pose[0]
                    ),
                    gripper_action_dict=dict(
                        left=torch.as_tensor([-1.0,], device=env.device),
                        right=torch.as_tensor([1.0,], device=env.device)
                    ),
                    action_noise_dict=dict(
                        left=0.0,
                        right=0.0
                    )
                )
            else:
                self.state = OraclePickPlaceExhaustPipePolicyStates.APPROACH_CRATE
                self.state_step_counter = 0

        if self.state == OraclePickPlaceExhaustPipePolicyStates.APPROACH_TRANSFER:
            if self.state_step_counter == 0:
                right_eef_pose = env.get_robot_eef_pose(eef_name="right")
                left_eef_pose = env.get_robot_eef_pose(eef_name="left")
                target_left_pose = left_eef_pose.clone()
                target_left_pose[..., 1, 3] = 0.0
                target_left_pose[..., 0, 3] += 0.0
                target_left_pose[..., 2, 3] += 0.1
                #target_left_pose[..., :3, :3] = PoseUtils.matrix_from_euler(euler_angles=torch.as_tensor([[0.0, -torch.pi/2, 0.0]], device=env.device), convention="XYZ")
                #target_left_pose[..., 2, 3] = 0.5
                #target_right_pose = target_left_pose.clone()
                #target_right_pose[..., 2, 3] -= 0.1
                #target_right_pose[..., 1, 3] = -0.15
                #target_right_pose[..., 0, 3] -= 0.15
                target_right_pose = right_eef_pose.clone()
                self.right_target_pose = target_right_pose
                self.left_target_pose = target_left_pose
            action[..., :] = env.target_eef_pose_to_action(
                target_eef_pose_dict=dict(
                    left=self.left_target_pose[0],
                    right=self.right_target_pose[0]
                ),
                gripper_action_dict=dict(
                    left=torch.as_tensor([-1.0,], device=env.device),
                    right=torch.as_tensor([1.0,], device=env.device)
                ),
                action_noise_dict=dict(
                    left=0.0,
                    right=0.0
                )
            )    
            left_eef_pose = env.get_robot_eef_pose(eef_name="left")
            left_quat = PoseUtils.quat_from_matrix(left_eef_pose[..., :3, :3])
            left_pos = left_eef_pose[..., :3, 3].reshape((1, 3))
            target_left_quat = PoseUtils.quat_from_matrix(self.left_target_pose[..., :3, :3])
            target_left_pos = self.left_target_pose[..., :3, 3].reshape((1, 3))

            right_eef_pose = env.get_robot_eef_pose(eef_name="right")
            right_quat = PoseUtils.quat_from_matrix(right_eef_pose[..., :3, :3])
            right_pos = right_eef_pose[..., :3, 3].reshape((1, 3))
            target_right_quat = PoseUtils.quat_from_matrix(self.right_target_pose[..., :3, :3])
            target_right_pos = self.right_target_pose[..., :3, 3].reshape((1, 3))

            if (
                (torch.norm(left_pos[0] - target_left_pos[0], p=2) <= 0.1) and 
                (torch.norm(left_quat[0] - target_left_quat[0], p=2) <= 0.5) and
                (torch.norm(right_pos[0] - target_right_pos[0], p=2) <= 0.1) and
                (torch.norm(right_quat[0] - target_right_quat[0], p=2) <= 0.5)
            ) or self.state_step_counter > 60:
                self.state = OraclePickPlaceExhaustPipePolicyStates.TRANSFER_PIPE
                self.state_step_counter = 0

        if self.state == OraclePickPlaceExhaustPipePolicyStates.TRANSFER_PIPE:
            if self.state_step_counter < 100:
                #self.left_target_pose[..., 1, 3] = 0.0
                pipe_pose = env.get_object_poses()[self.pipe.name]
                #self.right_target_pose[..., 0, 3] = pipe_pose[..., 0, 3]
                self.right_target_pose[..., :3, 3] = self.left_target_pose[..., :3, 3].clone() #pipe_pose[..., 1, 3] + 0.2
                #self.right_target_pose[..., 0, 3] += 0.05
                self.right_target_pose[..., :3, :3] = self.left_target_pose[..., :3, :3].clone()
                
                #self.right_target_pose[..., 2, 3] = pipe_pose[..., 2, 3]
                #self.left_target_pose[..., 1, 3] -= 0.01
                action[..., :] = env.target_eef_pose_to_action(
                    target_eef_pose_dict=dict(
                        left=self.left_target_pose[0],
                        right=self.right_target_pose[0]
                    ),
                    gripper_action_dict=dict(
                        left=torch.as_tensor([-1.0,], device=env.device),
                        right=torch.as_tensor([1.0,], device=env.device)
                    ),
                    action_noise_dict=dict(
                        left=0.0,
                        right=0.0
                    )
                )
            elif self.state_step_counter < 110:
                action[..., :] = env.target_eef_pose_to_action(
                    target_eef_pose_dict=dict(
                        left=self.left_target_pose[0],
                        right=self.right_target_pose[0]
                    ),
                    gripper_action_dict=dict(
                        left=torch.as_tensor([-1.0,], device=env.device),
                        right=torch.as_tensor([-1.0,], device=env.device)
                    ),
                    action_noise_dict=dict(
                        left=0.0,
                        right=0.0
                    )
                )
            elif self.state_step_counter < 120:
                self.left_target_pose[..., 0, 3] -= 0.005
                action[..., :] = env.target_eef_pose_to_action(
                    target_eef_pose_dict=dict(
                        left=self.left_target_pose[0],
                        right=self.right_target_pose[0]
                    ),
                    gripper_action_dict=dict(
                        left=torch.as_tensor([1.0,], device=env.device),
                        right=torch.as_tensor([-1.0,], device=env.device)
                    ),
                    action_noise_dict=dict(
                        left=0.0,
                        right=0.0
                    )
                )
            else:
                self.state = OraclePickPlaceExhaustPipePolicyStates.APPROACH_CRATE
                self.state_step_counter = 0

        if self.state == OraclePickPlaceExhaustPipePolicyStates.APPROACH_CRATE:
            if self.state_step_counter == 0:
                right_eef_pose = env.get_robot_eef_pose(eef_name="right")
                left_eef_pose = env.get_robot_eef_pose(eef_name="left")
                target_left_pose = left_eef_pose.clone()
                target_right_pose = right_eef_pose.clone()
                bin_position = self.destination.get_initial_pose().position_xyz
                target_left_pose[..., 0, 3] = bin_position[0] + 0.2
                target_left_pose[..., 1, 3] = bin_position[1] - 1.0
                target_left_pose[..., 2, 3] = bin_position[2] + 0.2
                self.right_target_pose = target_right_pose
                self.left_target_pose = target_left_pose
            action[..., :] = env.target_eef_pose_to_action(
                target_eef_pose_dict=dict(
                    left=self.left_target_pose[0],
                    right=self.right_target_pose[0]
                ),
                gripper_action_dict=dict(
                    left=torch.as_tensor([-1.0,], device=env.device),
                    right=torch.as_tensor([1.0,], device=env.device)
                ),
                action_noise_dict=dict(
                    left=0.0,
                    right=0.0
                )
            )    
            left_eef_pose = env.get_robot_eef_pose(eef_name="left")
            left_quat = PoseUtils.quat_from_matrix(left_eef_pose[..., :3, :3])
            left_pos = left_eef_pose[..., :3, 3].reshape((1, 3))
            target_left_quat = PoseUtils.quat_from_matrix(self.left_target_pose[..., :3, :3])
            target_left_pos = self.left_target_pose[..., :3, 3].reshape((1, 3))

            right_eef_pose = env.get_robot_eef_pose(eef_name="right")
            right_quat = PoseUtils.quat_from_matrix(right_eef_pose[..., :3, :3])
            right_pos = right_eef_pose[..., :3, 3].reshape((1, 3))
            target_right_quat = PoseUtils.quat_from_matrix(self.right_target_pose[..., :3, :3])
            target_right_pos = self.right_target_pose[..., :3, 3].reshape((1, 3))

            if (
                (torch.norm(left_pos[0] - target_left_pos[0], p=2) <= 0.1) and 
                #(torch.norm(left_quat[0] - target_left_quat[0], p=2) <= 0.2) and
                (torch.norm(right_pos[0] - target_right_pos[0], p=2) <= 0.1)
                #(torch.norm(right_quat[0] - target_right_quat[0], p=2) <= 0.2)
            ):
                self.state = OraclePickPlaceExhaustPipePolicyStates.DROP_PIPE
                self.state_step_counter = 0
        
        if self.state == OraclePickPlaceExhaustPipePolicyStates.DROP_PIPE:
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
                    right=0.0
                )
            )

        print(self.state)
        self.state_step_counter += 1
        return action
    
    def reset(self, env_ids = None):
        self.state = OraclePickPlaceExhaustPipePolicyStates.APPROACH_PIPE
        self.state_step_counter = 0

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "OraclePickPlaceExhaustPipePolicy":
        raise NotImplementedError(
            "OraclePickPlaceExhaustPipePolicy requires pipe and destination objects. "
            "Construct it directly with access to the environment assets."
        )