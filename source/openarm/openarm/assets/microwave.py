"""A countertop-sized microwave for the industrial tasks.

Arena ships ``microwave`` at its full 0.59 x 0.52 x 0.34 m size. On the 0.8 x
1.5 m work table that depth collides with the front row of manipulated props
(the row already spans ~0.48 m of the table's 1.5 m, leaving no room for a
0.52 m appliance beside it) and the appliance cannot be parked in the back
corner without running off the table. The industrial suite gets a scaled
variant at 0.7x -- a realistic countertop oven -- that shares the table with
a front row and a bin, and keeps the same openable door joint so the
open/close affordance and its normalized-joint goals are unchanged.
"""

from isaaclab_arena.assets.object_library import Microwave
from isaaclab_arena.assets.register import register_asset


@register_asset
class OpenArmMicrowave(Microwave):
    """A countertop microwave sized for the industrial work surface."""

    name = "openarm_microwave"
    tags = ["object", "openable"]
    scale = (0.7, 0.7, 0.7)
