import os
from isaaclab_arena.assets.hdr_image_library import LibraryHDR
from isaaclab_arena.assets.register import register_hdr
import openarm


for filename in os.listdir(f"{openarm.__path__[0]}/assets/hdris"):
    if filename.endswith(".hdr"):
        @register_hdr
        class PolyHavenHDR(LibraryHDR):
            name = filename.strip(".hdr")
            tags = ["polyhaven"]
            texture_file = f"{openarm.__path__[0]}/assets/hdris/{filename}"