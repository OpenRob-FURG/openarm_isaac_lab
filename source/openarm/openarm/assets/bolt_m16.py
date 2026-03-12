from isaaclab_arena.assets.object_library import LibraryObject, ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"

@register_asset
class BoltM16(LibraryObject):
    name = "bolt_m16"
    tags = ["object"]
    object_type = ObjectType.ARTICULATION
    usd_path = f"{ASSET_DIR}/factory_bolt_m16.usd"

    def _generate_articulation_cfg(self):
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {}
        cfg.init_state.joint_pos = {}
        cfg.init_state.joint_vel = {}
        return cfg