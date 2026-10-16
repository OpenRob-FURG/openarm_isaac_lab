from isaaclab_arena.assets.device_library import OpenXRCfg
from isaaclab_arena.assets.register import register_device


@register_device
class OpenArmOpenXRTeleopDevice(OpenXRCfg):
    name = "openxr__openarm_bimanual"
