from collections.abc import Callable

from isaaclab_arena.assets.register import register_retargeter
from isaaclab_arena.assets.retargeter_library import RetargetterBase


@register_retargeter
class BimanualOpenArmKeyboardRertargetter(RetargetterBase):
    device = "keyboard"
    embodiment = "openarm_bimanual"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class BimanualOpenArmKeyboardBimanualRetargetter(RetargetterBase):
    device = "keyboard_bimanual"
    embodiment = "openarm_bimanual"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class BimanualOpenArmKeyboardOpenArmRetargetter(RetargetterBase):
    device = "keyboard__openarm_bimanual"
    embodiment = "openarm_bimanual"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class BimanualOpenArmMediaPipeRetargetter(RetargetterBase):
    device = "mediapipe"
    embodiment = "openarm_bimanual"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None
