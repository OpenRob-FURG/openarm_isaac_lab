# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import (
    object_position_in_robot_root_frame,
    ee_left_position_in_robot_root_frame,
    ee_right_position_in_robot_root_frame,
    object_position_world_frame,
)

def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


# def object_ee_distance(
#     env: ManagerBasedRLEnv,
#     std: float,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
#     ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
# ) -> torch.Tensor:
#     """Reward the agent for reaching the object using tanh-kernel."""
#     # extract the used quantities (to enable type-hinting)
#     object: RigidObject = env.scene[object_cfg.name]
#     ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
#     # Target object position: (num_envs, 3)
#     cube_pos_w = object.data.root_pos_w
#     # End-effector position: (num_envs, 3)
#     ee_w = ee_frame.data.target_pos_w[..., 0, :]
#     # Distance of the end-effector to the object: (num_envs,)
#     object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

#     return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b
    )
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (
        1 - torch.tanh(distance / std)
    )


def bimanual_line_midpoint_alignment(
    env: ManagerBasedRLEnv,
    distance_std: float,
    arms_proximity_threshold: float,
    arms_distance_std: float,
    object_size: float,
):
    """
    Reward for aligning the midpoint between the two grippers with the object
    and positioning the grippers around the object for a grasp.
    """

    # posições no frame do robô
    obj_pos = object_position_in_robot_root_frame(env)
    ee_left = ee_left_position_in_robot_root_frame(env)
    ee_right = ee_right_position_in_robot_root_frame(env)

    # midpoint entre os grippers
    midpoint_pos = (ee_left + ee_right) / 2.0

    # distância midpoint → objeto
    distance_to_center = torch.norm(midpoint_pos - obj_pos, dim=1)

    alignment_reward = 1 - torch.tanh(distance_to_center / distance_std)

    # distância entre os dois braços
    arms_distance = torch.norm(ee_left - ee_right, dim=1)

    arms_distance_error = torch.relu(arms_distance - object_size)

    # só ativa reward de grasp quando perto do objeto
    is_close_to_center = (distance_to_center < arms_proximity_threshold).float()

    arm_proximity_reward = is_close_to_center * (
        1 - torch.tanh(arms_distance_error / arms_distance_std)
    )

    # bonus para levantar o objeto
    obj_pos_w = object_position_world_frame(env)
    height = obj_pos_w[:, 2]

    lift_bonus = torch.clamp(height - 0.04, min=0.0)

    total_reward = alignment_reward + arm_proximity_reward + 4.0 * lift_bonus

    return total_reward