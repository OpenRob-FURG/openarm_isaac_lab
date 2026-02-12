import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

class BimanualOpenArmPickAndPlaceExhaustPipeEnvironment(ExampleEnvironmentBase):
    name: str = "openarm_bimanual_pick_and_place_exhaust_pipe"
    
    def get_env(self, args_cli):
        import torch
        import tqdm

        import pinocchio  # noqa: F401
        from isaaclab.app import AppLauncher

        import openarm.embodiments
        import openarm.assets
        from isaaclab_arena.assets.object_reference import ObjectReference
        from openarm.arena_tasks.pick_and_place import PickAndPlaceTask
        from openarm.arena_tasks.pour import PourTask
        from openarm.policies.oracle.pour import OraclePourPolicy
        from isaaclab_arena.assets.asset_registry import AssetRegistry
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        import isaaclab_arena.teleop_devices
        import openarm.arena_devices.keyboard
        import openarm.arena_devices.mediapipe_teleop_device
        import openarm.assets.exhaust_pipe
        from isaaclab_arena.tasks.dummy_task import DummyTask

        teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        # Step 1: Initialize and get the assets from the registry

        background = self.asset_registry.get_asset_by_name("packing_table")()
        embodiment = self.asset_registry.get_asset_by_name("openarm_bimanual")(enable_cameras=args_cli.enable_cameras)
        sorting_bin = ObjectReference(
            name="sorting_bin",
            prim_path="{ENV_REGEX_NS}/packing_table/container_h20",#/container_h20_inst/Container_H20_01
            parent_asset=background,
        )
        exhaust_pipe = self.asset_registry.get_asset_by_name("custom_exhaust_pipe")()
        
        #sorting_bin.set_initial_pose(
        #    Pose(position_xyz=(0.4, -0.2, 0.3), rotation_wxyz=(0.707, 0.0, 0.0, -0.707))
        #)
        exhaust_pipe.set_initial_pose(
            Pose(position_xyz=(0.5, 0.2, 0.3), rotation_wxyz=(0.707, 0.0, 0.0, -0.707))
        )
        embodiment.set_initial_pose(
            Pose(position_xyz=(0.2, 0.0, -0.2), rotation_wxyz=(1, 0, 0, 0))
        )

        # Step 2: Create a scene with the assets
        scene = Scene(assets=[background, exhaust_pipe])

        # Step 3: Create a task
        task = PickAndPlaceTask(pick_up_object=exhaust_pipe, destination_location=sorting_bin, background_scene=background, episode_length_s=10.0)
        #task = DummyTask()

        # Step 4: Create the IsaacLab Arena environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name="my_first_arena_env",
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )

        return isaaclab_arena_environment
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--teleop_device", type=str, default="keyboard")