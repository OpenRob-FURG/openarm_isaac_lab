from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab.envs import MimicEnvCfg
from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.utils.configclass import MISSING

@configclass
class PourMimicEnvCfg(MimicEnvCfg):
    subtask_configs = dict(
        left=[],
        right=[]
    )

@configclass
class PourEventsCfg:
    pass

@configclass
class PourSceneCfg(InteractiveSceneCfg):
    pass

@configclass
class PourTerminationCfg:
    pass

@configclass
class PourRewardsCfg:
    pass

class PourTask(TaskBase):
    def __init__(self, episode_length_s = None, source_recipient=None, destination_recipient=None, table_limit_lower: tuple[float|float|float]=None, table_limit_upper: tuple[float|float|float]=None):
        super().__init__(episode_length_s)
        self.source_recipient = source_recipient
        self.destination_recipient = destination_recipient
        self.table_limit_lower = table_limit_lower
        self.table_limit_upper = table_limit_upper

    def get_mimic_env_cfg(self, embodiment_name):
        if embodiment_name == "openarm_bimanual":
            return PourMimicEnvCfg()
        
    def get_prompt(self):
        return f"pour the {self.source_recipient.name} into the {self.destination_recipient.name}"
    
    def get_events_cfg(self):
        return PourEventsCfg()
    
    def get_metrics(self):
        return []
    
    def get_scene_cfg(self):
        pass
    
    def get_termination_cfg(self):
        return PourTerminationCfg()
    
    def get_viewer_cfg(self):
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))
    
    def get_rewards_cfg(self):
        cfg = PourRewardsCfg()
        return cfg