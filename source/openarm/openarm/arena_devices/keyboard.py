from isaaclab_arena.assets.device_library import KeyboardCfg
from isaaclab_arena.assets.register import register_device


@register_device
class OpenArmKeyboardTeleopDevice(KeyboardCfg):
    name = "keyboard__openarm_bimanual"
