from isaaclab.devices.retargeter_base import RetargeterCfg

from isaaclab_arena.assets.retargeter_library import register_retargeter, RetargetterBase

@register_retargeter
class BimanualOpenArmKeyboardRertargetter(RetargetterBase):
    device = "keyboard"
    embodiment = "openarm_bimanual"

    def get_retargeter_cfg(self, embodiment, sim_device, enable_visualization = False):
        return None