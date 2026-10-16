import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

class BimanualOpenArmComputerDisassemblyEnvironment(ExampleEnvironmentBase):
    name: str = "openarm_bimanual_computer_disassembly"
    
    def get_env(self, args_cli):
        import torch
        import tqdm

        import pinocchio  # noqa: F401
        from isaaclab.app import AppLauncher


        import openarm.embodiments
        import openarm.assets
        import openarm.arena_devices.openxr
        import openarm.arena_retargeters.openarm_bimanual

        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.assets.asset_registry import AssetRegistry
        from isaaclab_arena.assets.object_library import ObjectType
        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.no_task import NoTask
        from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from openarm.arena_tasks.pick_and_place_component import PickAndPlaceComponentTask
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena.relations.relations import IsAnchor, NextTo, AtPosition, On, RotateAroundSolution
        import openarm.assets.desktop_computer
        from isaaclab_rl.rsl_rl.rl_cfg import RslRlOnPolicyRunnerCfg, configclass, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
        import omni.client
        

        #teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        # Step 1: Initialize and get the assets from the registry

        #background = self.asset_registry.get_asset_by_name("packing_table")()
        #background.set_initial_pose(
        #    Pose(position_xyz=(0.5, 0.0, -0.8), rotation_xyzw=(0.0, 0.0, -0.707, 0.707))
        #)
        hdr = self.hdr_registry.get_random_hdr_by_tag(tag="polyhaven")()
        dome_light = self.asset_registry.get_asset_by_name("light")()
        dome_light.add_hdr(hdr)
        embodiment = self.asset_registry.get_asset_by_name("openarm_bimanual")(
            enable_cameras=args_cli.enable_cameras, arm_mode=ArmMode.DUAL_ARM
        )
        computer = self.asset_registry.get_asset_by_name("desktop_computer")()
        background = self.asset_registry.get_asset_by_name("maple_table_robolab")()

        background.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        table_reference = ObjectReference(
            name="table",
            prim_path="{ENV_REGEX_NS}/maple_table_robolab/table",
            parent_asset=background,
            object_type=ObjectType.RIGID,
        )
        sorting_bin = self.asset_registry.get_asset_by_name("openarm_sorting_bin")()
        '''gpu_reference = ObjectReference(
            name="gpu",
            prim_path="{ENV_REGEX_NS}/desktop_computer/computer/gpu",
            parent_asset=computer,
            object_type=ObjectType.BASE,
        )
        pc_case_reference = ObjectReference(
            name="pc_case",
            prim_path="{ENV_REGEX_NS}/desktop_computer/computer/pc_case",
            parent_asset=computer,
            object_type=ObjectType.BASE,
        )
        msi_motherboard_reference = ObjectReference(
            name="msi_motherboard",
            prim_path="{ENV_REGEX_NS}/desktop_computer/computer/msi_motherboard",
            parent_asset=computer,
            object_type=ObjectType.BASE,
        )
        trident_z_rgb_01_reference = ObjectReference(
            name="trident_z_rgb_01",
            prim_path="{ENV_REGEX_NS}/desktop_computer/computer/trident_z_rgb_01",
            parent_asset=computer,
            object_type=ObjectType.BASE,
        )
        trident_z_rgb_02_reference = ObjectReference(
            name="trident_z_rgb_02",
            prim_path="{ENV_REGEX_NS}/desktop_computer/computer/trident_z_rgb_02",
            parent_asset=computer,
            object_type=ObjectType.BASE,
        )
        power_supply_reference = ObjectReference(
            name="power_supply",
            prim_path="{ENV_REGEX_NS}/desktop_computer/computer/power_supply",
            parent_asset=computer,
            object_type=ObjectType.BASE,
        )
        storage_rack_reference = ObjectReference(
            name="storage_rack",
            prim_path="{ENV_REGEX_NS}/desktop_computer/computer/storage_rack",
            parent_asset=computer,
            object_type=ObjectType.BASE,
        )
        ssd_reference = ObjectReference(
            name="ssd",
            prim_path="{ENV_REGEX_NS}/desktop_computer/computer/ssd",
            parent_asset=computer,
            object_type=ObjectType.BASE,
        )'''
        table_reference.add_relation(IsAnchor())
        computer.add_relation(On(table_reference))
        computer.add_relation(AtPosition(z=0.0, y=-0.3, x=1.0))
        computer.add_relation(RotateAroundSolution(roll_rad=3.1416/2, yaw_rad=-3.1416/2))
        #computer.add_relation(AtPosition(z=0.1))
        sorting_bin.add_relation(On(table_reference))
        sorting_bin.add_relation(AtPosition(x=0.6, y=0.4))
        sorting_bin.add_relation(RotateAroundSolution(yaw_rad=3.1416/2))
        #exhaust_pipe = self.asset_registry.get_asset_by_name("custom_exhaust_pipe")()
        
        #sorting_bin.set_initial_pose(
        #    Pose(position_xyz=(0.4, -0.2, 0.3), rotation_xyzw=(0.0, 0.0, 0.707, -0.707))
        #)
        #exhaust_pipe.set_initial_pose(
        #    Pose(position_xyz=(0.50, 0.2, 0.2), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
        #)
        #embodiment.set_initial_pose(
        #    #Pose(position_xyz=(0.2, 0.0, -0.2), rotation_xyzw=(0, 0, 0, 1))
        #    Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0, 0, 0, 1))
        #)

        # Step 2: Create a scene with the assets
        #scene = Scene(assets=[background, exhaust_pipe])
        scene = Scene(assets=[background, table_reference, computer, dome_light, sorting_bin])
        #print(scene_description)
        #exit()


        # Step 3: Create a task
        #task = PickTask(pick_up_object=exhaust_pipe, background_scene=background, episode_length_s=20.0)
        #task = NoTask()
        task = SequentialTaskBase(
            subtasks=[
                OpenDoorTask(openable_object=computer, reset_openness=0.1),
                PickAndPlaceComponentTask(composite_object=computer, pick_up_component="/computer/gpu", destination_location=sorting_bin, background_scene=background),
                PickAndPlaceComponentTask(composite_object=computer, pick_up_component="/computer/trident_z_rgb_01", destination_location=sorting_bin, background_scene=background),
                PickAndPlaceComponentTask(composite_object=computer, pick_up_component="/computer/trident_z_rgb_02", destination_location=sorting_bin, background_scene=background),
                PickAndPlaceComponentTask(composite_object=computer, pick_up_component="/computer/msi_motherboard", destination_location=sorting_bin, background_scene=background),
                #PickAndPlaceTask(pick_up_object=gpu_reference, destination_location=sorting_bin, background_scene=background)
                #GoalPoseTask(object=gpu_reference, target_x_range=(0.1, 0.3), target_y_range=(-0.1, -0.3), target_z_range=(0.0, 0.3))
            ]
        )

        teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        # Step 4: Create the IsaacLab Arena environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name="computer_disassembly",
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
        parser.add_argument("--teleop_device", type=str, default="openxr")