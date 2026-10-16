
from isaaclab_arena.assets.background_library import LibraryBackground
from isaaclab_arena.assets.register import register_asset
import openarm


@register_asset
class ChemicalWorkshop(LibraryBackground):
    """A chemical workshop background for industrial tasks"""

    name = "chemical_background"
    usd_path = f"{openarm.__path__[0]}/assets/meshes/Chemical workshop.usd"
    scale = (1.0, 1.0, 1.0)
