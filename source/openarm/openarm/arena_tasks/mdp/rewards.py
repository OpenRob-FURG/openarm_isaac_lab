import torch
from typing import TYPE_CHECKING
from isaaclab.sensors import FrameTransformer
from isaaclab.envs import ManagerBasedRLEnv

def arm_near_object_goal(env: ManagerBasedRLEnv, arm: str=None, goal_object=None):
    ee_frame: FrameTransformer = env.scene['ee_frame']
    
