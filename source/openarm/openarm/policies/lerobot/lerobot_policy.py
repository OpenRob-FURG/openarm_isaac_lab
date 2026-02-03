from isaaclab_arena.policy.policy_base import PolicyBase
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
import torch

class LeRobotPolicy(PolicyBase):
    def __init__(self, repo_id, task: str, policy_type="smolvla") -> None:
        super().__init__()
        self.policy_config = make_policy_config(policy_type=policy_type)
        self.policy_config.pretrained_path = repo_id
        self.pre_processor, self.post_processor = make_pre_post_processors(self.policy_config, pretrained_path=repo_id)
        self.policy = make_policy(self.policy_config)
        self.task = task

    def get_action(self, env, observation):
        return self.post_processor(
            self.policy.select_action(
                self.pre_processor(
                    batch={
                        "observation.images.top": observation["camera_top"],
                        "observation.state": torch.cat([
                            observation["policy"]["left_eef_pos"],
                            observation["policy"]["left_eef_quat"],
                            observation["policy"]["left_gripper_pos"],
                            observation["policy"]["right_eef_pos"],
                            observation["policy"]["right_eef_quat"],
                            observation["policy"]["right_gripper_pos"],
                        ], dim=-1),
                        "task": [self.task,]*env.num_envs,
                    }
                )
            )
        )
    
    def reset(self, env_ids=None):
        self.policy.reset()