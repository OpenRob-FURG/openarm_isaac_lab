import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

class BimanualOpenArmOpenMicrowaveEnvironment(ExampleEnvironmentBase):
    name: str = "openarm_bimanual_open_microwave"
    
    def get_env(self, args_cli):
        import torch
        import tqdm

        import pinocchio  # noqa: F401
        from isaaclab.app import AppLauncher

        import openarm.embodiments
        import openarm.assets
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from openarm.arena_tasks.pour import PourTask
        from openarm.policies.oracle.pour import OraclePourPolicy
        from isaaclab_arena.assets.asset_registry import AssetRegistry
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        import openarm.arena_devices.keyboard
        import openarm.arena_retargeters.openarm_bimanual

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Step 1: Initialize and get the assets from the registry

        background = self.asset_registry.get_asset_by_name("kitchen")()
        embodiment = self.asset_registry.get_asset_by_name("openarm_bimanual")(enable_cameras=args_cli.enable_cameras)
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        
        microwave.set_initial_pose(
            Pose(position_xyz=(0.4, 0.0, 0.2), rotation_xyzw=(0.0, 0.0, -0.707, 0.707))
        )
        embodiment.set_initial_pose(
            Pose(position_xyz=(-0.3, 0.0, -0.2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )

        # Step 2: Create a scene with the assets
        scene = Scene(assets=[background, microwave])

        # Step 3: Create a task
        task = OpenDoorTask(openable_object=microwave, episode_length_s=10.0, openness_threshold=0.25)

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