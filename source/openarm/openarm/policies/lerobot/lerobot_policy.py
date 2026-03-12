from isaaclab_arena.policy.policy_base import PolicyBase
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, SmolVLAConfig
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy, XVLAConfig
import torch

class LeRobotPolicy(PolicyBase):
    def __init__(self, repo_id, task: str, policy_type="smolvla") -> None:
        super().__init__()
        #self.policy_config = make_policy_config(policy_type=policy_type)
        #self.policy_config.pretrained_path = repo_id
        #assert policy_type == "smolvla"
        if policy_type == "smolvla":
            self.policy = SmolVLAPolicy.from_pretrained(pretrained_name_or_path=repo_id)
            #self.policy.config.n_action_steps = 50
            #self.policy.config.num_steps = 100
        elif policy_type == "groot":
            self.policy = GrootPolicy.from_pretrained(pretrained_name_or_path=repo_id).bfloat16()
            #self.policy.config.chunk_size = 5
            #print(self.policy.config.max_steps)
        elif policy_type == "xvla":
            self.policy = XVLAPolicy.from_pretrained(pretrained_name_or_path=repo_id)
            #self.policy.config.n_action_steps = 1
            print(self.policy.config.action_mode)
            exit()
        else:
            raise ValueError()
        torch.cuda.empty_cache()
        #self.policy.config.n_action_steps = 1
        self.policy_config = self.policy.config
        self.pre_processor, self.post_processor = make_pre_post_processors(self.policy_config, pretrained_path=repo_id)
        #self.policy = make_policy(self.policy_config)
        self.task = task

    def get_action(self, env, observation):
        batch = self.pre_processor(
                    {
                        "observation.images.top": observation["camera_obs"].float().reshape((env.num_envs, 3, 256, 256)).to(self.policy.config.device)/255.0,
                        "observation.state": torch.cat([
                            observation["policy"]["left_eef_pos"],
                            observation["policy"]["left_eef_quat"],
                            observation["policy"]["left_gripper_pos"],
                            observation["policy"]["right_eef_pos"],
                            observation["policy"]["right_eef_quat"],
                            observation["policy"]["right_gripper_pos"],
                        ], dim=-1),
                        #"observation.state": observation["policy"].to(self.policy.config.device),
                        "task": [self.task,]*env.num_envs,
                    }
                )
        #print(batch)
        #exit()
        return self.post_processor(
            self.policy.select_action(
                batch
            )
        ).to(env.device)
    
    def reset(self, env_ids=None):
        self.policy.reset()