import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def root_height_above_maximum(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, height: float):
    return torch.as_tensor(env.scene[object_cfg.name].get_world_poses()[0])[..., 2] > height

def obj_near_obj(env: ManagerBasedRLEnv, object_cfg_1: SceneEntityCfg, object_cfg_2: SceneEntityCfg, distance: float=0.1):
    print(env.scene[object_cfg_1.name].get_world_poses(indices=[0,]))
    pos1 = torch.as_tensor(env.scene[object_cfg_1.name].get_world_poses()[0])
    pos2 = torch.as_tensor(env.scene[object_cfg_2.name].get_world_poses()[0])
    norm = torch.norm(pos1 - pos2, p=2, dim=1)
    print(pos1, pos2, norm, norm < distance)
    return norm < distance