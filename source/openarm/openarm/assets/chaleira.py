from isaaclab_arena.assets.object_library import LibraryObject, ObjectType
from isaaclab_arena.assets.register import register_asset
import openarm
from isaaclab_arena.utils.pose import Pose

@register_asset
class Chaleira(LibraryObject):
    name = "chaleira"
    tags = ["object"]
    usd_path = f"{openarm.__path__[0]}/assets/meshes/chaleira.usd"
    object_type = ObjectType.RIGID
    scale = (0.2, 0.2, 0.2)
    grasp_pose = Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.707, 0.707))
    grasp_approach_axis = 2
    grasp_approach_direction = -0.1