import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

class BimanualOpenArmPickExhaustPipeEnvironment(ExampleEnvironmentBase):
    name: str = "openarm_bimanual_pick_exhaust_pipe"
    
    def get_env(self, args_cli):
        import torch
        import tqdm

        import pinocchio  # noqa: F401
        from isaaclab.app import AppLauncher
        from scene_synthesizer.assets import USDAsset


        import openarm.embodiments
        import openarm.assets
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.assets.object_reference import ObjectReference
        from openarm.arena_tasks.pick_and_place import PickAndPlaceTask
        from openarm.arena_tasks.pick import PickTask
        from openarm.arena_tasks.pour import PourTask
        from openarm.policies.oracle.pour import OraclePourPolicy
        from openarm.policies.llm.llm_policy import LLMPolicy
        from isaaclab_arena.assets.asset_registry import AssetRegistry
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        import openarm.arena_devices.keyboard
        import openarm.arena_devices.keyboard_bimanual
        import openarm.arena_devices.mediapipe_teleop_device
        import openarm.arena_retargeters.openarm_bimanual
        import openarm.assets.exhaust_pipe
        from openarm.policies.oracle.pick_place_exhaust_pipe import OraclePickPlaceExhaustPipePolicy
        from isaaclab_arena.tasks.dummy_task import DummyTask
        from isaaclab_rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg, configclass, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
        import omni.client
        

        teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        # Step 1: Initialize and get the assets from the registry

        background = self.asset_registry.get_asset_by_name("packing_table")()
        background.set_initial_pose(
            Pose(position_xyz=(0.2, 0.0, -1.0), rotation_xyzw=(0.0, 0.0, -0.707, 0.707))
        )
        embodiment = self.asset_registry.get_asset_by_name("openarm_bimanual")(
            enable_cameras=args_cli.enable_cameras, arm_mode=ArmMode.LEFT
        )
        exhaust_pipe = self.asset_registry.get_asset_by_name("custom_exhaust_pipe")()
        
        #sorting_bin.set_initial_pose(
        #    Pose(position_xyz=(0.4, -0.2, 0.3), rotation_xyzw=(0.0, 0.0, 0.707, -0.707))
        #)
        exhaust_pipe.set_initial_pose(
            Pose(position_xyz=(0.3, 0.1, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        embodiment.set_initial_pose(
            #Pose(position_xyz=(0.2, 0.0, -0.2), rotation_xyzw=(0, 0, 0, 1))
            Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0, 0, 0, 1))
        )

        # Step 2: Create a scene with the assets
        scene = Scene(assets=[background, exhaust_pipe])

        #print(scene_description)
        #exit()


        # Step 3: Create a task
        task = PickTask(pick_up_object=exhaust_pipe, background_scene=background, episode_length_s=20.0)
        #task = DummyTask()

        # Step 4: Create the IsaacLab Arena environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name="my_first_arena_env",
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )

        # Step 5: Create oracle policy
        #oracle = OraclePickPlaceExhaustPipePolicy(pipe=exhaust_pipe, destination=sorting_bin)
        #isaaclab_arena_environment.oracle_policy = oracle

        return isaaclab_arena_environment
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--teleop_device", type=str, default="keyboard")