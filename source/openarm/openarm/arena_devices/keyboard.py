from isaaclab_arena.teleop_devices import register_device, KeyboardTeleopDevice
import torch

@register_device
class OpenArmKeyboardTeleopDevice(KeyboardTeleopDevice):
    name = "keyboard__openarm_bimanual"