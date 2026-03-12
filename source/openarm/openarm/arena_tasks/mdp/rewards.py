import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_arena.tasks.terminations import object_on_destination
from isaaclab.managers import SceneEntityCfg


def arm_near_object_goal(env: ManagerBasedRLEnv, arm: str=None, goal_object: str=None):
    ee_frame: FrameTransformer = env.scene['ee_frame']
    ee_pos = ee_frame.data.target_pos_w[..., ee_frame.data.target_frame_names.index(f"{arm}_ee_tcp"), :]
    goal_object_pos = env.scene.rigid_objects[goal_object].data.root_pos_w
    distance = torch.norm(ee_pos - goal_object_pos, p=2, dim=-1)
    if arm == "left":
        reward = (goal_object_pos[..., 1] > 0.0)* -distance*0.1
        reward += (distance <= 0.05)*(env.action_manager.prev_action[..., 6] < 0.0)*1000.0
    else:
        reward = (goal_object_pos[..., 1] <= 0.0)* -distance*0.1
        reward += (distance <= 0.05)*(env.action_manager.prev_action[..., 13] < 0.0)*1000.0
    return reward

def object_near_pos_goal(env: ManagerBasedRLEnv, object_name: str=None, target_pos: tuple[float|float|float]=None):
    object_pos = env.scene.rigid_objects[object_name].data.root_pos_w - env.scene.env_origins
    target_pos_tensor = torch.zeros_like(object_pos)
    target_pos_tensor[..., 0] += target_pos[0]
    target_pos_tensor[..., 1] += target_pos[1]
    target_pos_tensor[..., 2] += target_pos[2]
    distances = torch.norm(target_pos_tensor - object_pos, p=2, dim=-1)
    on_destination = object_on_destination(env=env, object_cfg=SceneEntityCfg(name=object_name), velocity_threshold=0.1)
    
    return (on_destination * 1000.0) - (distances*0.1)