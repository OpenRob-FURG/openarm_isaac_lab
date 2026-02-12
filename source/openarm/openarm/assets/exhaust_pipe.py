from isaaclab_arena.assets.object_library import LibraryObject, ObjectType, BlueExhaustPipe
from isaaclab_arena.assets.register import register_asset
import openarm

@register_asset
class CustomExhaustPipe(BlueExhaustPipe):
    name = "custom_exhaust_pipe"
    usd_path = f"{openarm.__path__[0]}/assets/meshes/blue_exhaust_pipe.usd"

    def _generate_rigid_cfg(self):
        cfg = super()._generate_rigid_cfg()
        cfg.spawn.activate_contact_sensors = True
        return cfg
    
    def get_contact_sensor_cfg(self, contact_against_prim_paths = None):
        cfg = super().get_contact_sensor_cfg(contact_against_prim_paths)
        cfg.prim_path += "/Geometry/sm_gtc_sorting_exhaust_pipe_a01_01"
        return cfg