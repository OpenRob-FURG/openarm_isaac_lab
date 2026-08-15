import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

class BimanualOpenArmPickAndPlaceExhaustPipeEnvironment(ExampleEnvironmentBase):
    name: str = "openarm_bimanual_pick_and_place_exhaust_pipe"
    
    def get_env(self, args_cli):
        import torch
        import tqdm

        import pinocchio  # noqa: F401
        from isaaclab.app import AppLauncher
        from scene_synthesizer.assets import USDAsset


        import openarm.embodiments
        import openarm.assets
        from isaaclab_arena.assets.object_reference import ObjectReference
        from openarm.arena_tasks.pick_and_place import PickAndPlaceTask
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
        embodiment = self.asset_registry.get_asset_by_name("openarm_bimanual")(enable_cameras=args_cli.enable_cameras)
        sorting_bin = ObjectReference(
            name="sorting_bin",
            prim_path="{ENV_REGEX_NS}/packing_table/container_h20",#/container_h20_inst/Container_H20_01
            parent_asset=background,
        )
        exhaust_pipe = self.asset_registry.get_asset_by_name("custom_exhaust_pipe")()
        
        #sorting_bin.set_initial_pose(
        #    Pose(position_xyz=(0.4, -0.2, 0.3), rotation_xyzw=(0.0, 0.0, -0.707, 0.707))
        #)
        exhaust_pipe.set_initial_pose(
            Pose(position_xyz=(0.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        )
        embodiment.set_initial_pose(
            #Pose(position_xyz=(0.2, 0.0, -0.2), rotation_xyzw=(0, 0, 0, 1))
            Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0, 0, 0, 1))
        )

        # Step 2: Create a scene with the assets
        scene = Scene(assets=[background, exhaust_pipe])

        scene_description = {}
        for asset_name in scene.assets.keys():
            scene_description[asset_name] = {}
            print(asset_name)
            if scene.assets[asset_name].usd_path.startswith('https://'):
                omni.client.copy(scene.assets[asset_name].usd_path, "/tmp/downloaded_usd.usd")
                usd_asset = USDAsset("/tmp/downloaded_usd.usd")
            else:
                usd_asset = USDAsset(fname=scene.assets[asset_name].usd_path)
            usd_asset = usd_asset.scene()
            for geometry_name in usd_asset.get_geometry_names():
                print(geometry_name)
                geometry_transform = usd_asset.get_transform(node=geometry_name).tolist()
                geometry_bounds = usd_asset.get_bounds(query=geometry_name).tolist()
                scene_description[asset_name][geometry_name] = dict(
                    geometry_transform=geometry_transform,
                    geometry_bounds=geometry_bounds
                )
        #print(scene_description)
        #exit()


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

        isaaclab_arena_environment.rl_config = RslRlOnPolicyRunnerCfg(
            num_steps_per_env=16,
            max_iterations=1500,
            obs_groups=dict(
                policy=["policy"],
            ),
            save_interval=50,
            experiment_name=f"openarm_bimanual_pick_and_place_exhaust_pipe",
            policy = RslRlPpoActorCriticCfg(
                init_noise_std=1.0,
                actor_obs_normalization=True,
                critic_obs_normalization=True,
                actor_hidden_dims=[256, 128, 64],
                critic_hidden_dims=[256, 128, 64],
                activation="elu",
            ),
            algorithm = RslRlPpoAlgorithmCfg(
                value_loss_coef=1.0,
                use_clipped_value_loss=True,
                clip_param=0.2,
                entropy_coef=0.0,
                num_learning_epochs=8,
                num_mini_batches=8,
                learning_rate=5.0e-4,
                schedule="adaptive",
                gamma=0.99,
                lam=0.95,
                desired_kl=0.008,
                max_grad_norm=1.0,
            )
        )

        llm_policy = LLMPolicy(scene=scene, task_description=task.get_prompt())
        isaaclab_arena_environment.llm_policy = llm_policy
        # Step 5: Create oracle policy
        #oracle = OraclePickPlaceExhaustPipePolicy(pipe=exhaust_pipe, destination=sorting_bin)
        #isaaclab_arena_environment.oracle_policy = oracle

        return isaaclab_arena_environment
    
    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--teleop_device", type=str, default="keyboard")