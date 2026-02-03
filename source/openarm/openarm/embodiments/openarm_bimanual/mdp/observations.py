from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import FrameTransformer, ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def rel_ee_pos(env: ManagerBasedRLEnv, arm: str)->torch.Tensor:
    transformer: FrameTransformer = env.scene['ee_frame']
    return transformer.data.target_pos_source[..., transformer.data.target_frame_names.index(f'{arm}_ee_tcp'), :]

def rel_ee_quat(env: ManagerBasedRLEnv, arm: str)->torch.Tensor:
    transformer: FrameTransformer = env.scene['ee_frame']
    return transformer.data.target_quat_source[..., transformer.data.target_frame_names.index(f'{arm}_ee_tcp'), :]

def gripper_opening(env: ManagerBasedRLEnv, arm: str)->torch.Tensor:
    robot = env.scene.articulations["robot"]
    return robot.data.joint_pos[..., [robot.joint_names.index(f'openarm_{arm}_finger_joint1'),]]

def all_poses_w(env: ManagerBasedRLEnv)->torch.Tensor:
    state = env.scene.get_state(is_relative=True)
    all_states = []
    for asset_type in state.keys():
        for state_tensor_key in state[asset_type].keys():
            for key2 in state[asset_type][state_tensor_key].keys():
                all_states.append(state[asset_type][state_tensor_key][key2].flatten(start_dim=1))
    return torch.cat(all_states, dim=1)