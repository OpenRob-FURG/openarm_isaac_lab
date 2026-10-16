from isaaclab_arena.assets.object_library import LibraryObject, Openable, ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose
from isaaclab.sensors import ContactSensorCfg
import openarm

@register_asset
class DesktopComputer(LibraryObject, Openable):
    """A desktop computer."""

    name = "desktop_computer"
    tags = ["object", "openable"]
    usd_path = f"{openarm.__path__[0]}/assets/meshes/computer.usd"
    object_type = ObjectType.ARTICULATION

    # Openable affordance parameters
    openable_joint_name = "RevoluteJoint"
    
    openable_threshold = 0.5  # Bistate threshold (open > threshold, closed <= threshold)

    def __init__(
        self, instance_name: str | None = None, prim_path: str | None = None, initial_pose: Pose | None = None
    ):
        super().__init__(
            instance_name=instance_name,
            prim_path=prim_path,
            initial_pose=initial_pose,
            openable_joint_name=self.openable_joint_name,
            openable_threshold=self.openable_threshold,
        )

    def get_link_contact_sensor_cfg(self, link_path: str, contact_against_prim_paths: list[str]):
        if contact_against_prim_paths is None:
            contact_against_prim_paths = []
        return ContactSensorCfg(
            prim_path=self.prim_path + link_path,
            filter_prim_paths_expr=contact_against_prim_paths,
        )