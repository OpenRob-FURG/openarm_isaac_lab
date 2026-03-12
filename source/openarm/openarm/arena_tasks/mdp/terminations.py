import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def root_height_above_maximum(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg, height: float):
    return env.scene.rigid_objects[object_cfg.name].data.root_pos_w[..., 2] > height