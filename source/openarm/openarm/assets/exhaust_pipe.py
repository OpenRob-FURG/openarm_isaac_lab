from isaaclab_arena.assets.object_library import LibraryObject, ObjectType, BlueExhaustPipe
from isaaclab_arena.assets.register import register_asset
import openarm

@register_asset
class CustomExhaustPipe(BlueExhaustPipe):
    name = "custom_exhaust_pipe"
    usd_path = f"{openarm.__path__[0]}/assets/meshes/blue_exhaust_pipe.usd"
    scale = (0.5, 0.5, 2.0)

    def _generate_rigid_cfg(self):
        cfg = super()._generate_rigid_cfg()
        cfg.spawn.activate_contact_sensors = True
        return cfg
    
    def get_contact_sensor_cfg(self, contact_against_prim_paths = None):
        cfg = super().get_contact_sensor_cfg(contact_against_prim_paths)
        cfg.prim_path += "/Geometry/sm_gtc_sorting_exhaust_pipe_a01_01"
        return cfg
    
    def get_interaction_poses(self):
        return dict(
            left_arm_grasp_pose=dict(
                pos=[0.0, 0.0, 0.5],
                quat=[ 6.7656e-04, -7.0710e-01,  1.0669e-03, -7.0712e-01]
                #quat=[0.707, 0.0, 0.707, 0.0]
                #quat=[ 0.0825, -0.6985,  0.1375, -0.6975]
            ),
            right_arm_grasp_pose=dict(
                pos=[0.0, 0.0, 0.0],
                quat=[0.0, 0.707, 0.0, 0.707]
            )
        )