"""A right-sized sorting bin for the industrial tasks.

Arena ships ``blue_sorting_bin`` at ``scale=(4, 2, 1)``, which makes it a
0.80 x 0.50 m catch tray for the Mimic exhaust-pipe task. At that size it
collides with every other receptacle and prop on the 0.8 x 1.5 m table, so the
industrial sorting tasks that reuse it cannot be laid out. The industrial suite
gets its own bin at the USD's natural scale instead, which is the same blue
tray at a size that can share the table with a front row and other bins.
"""

from isaaclab_arena.assets.object_library import LibraryObject
from isaaclab_arena.assets.register import register_asset
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@register_asset
class OpenArmSortingBin(LibraryObject):
    """A blue plastic sorting bin sized for the industrial work surface."""

    name = "openarm_sorting_bin"
    tags = ["object"]
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Mimic/exhaust_pipe_task/exhaust_pipe_assets/blue_sorting_bin.usd"
    scale = (1.0, 1.0, 1.0)
