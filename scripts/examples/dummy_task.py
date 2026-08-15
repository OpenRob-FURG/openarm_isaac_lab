import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

# Launch the Isaac Sim application
print("Launching simulation app")
simulation_app = AppLauncher(launcher_args=dict(
    enable_cameras=True
))

import openarm.embodiments
import openarm.assets
from openarm.arena_tasks.pour import PourTask
from openarm.policies.oracle.pour import OraclePourPolicy
from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.open_door_task import OpenDoorTask
from isaaclab_arena.utils.pose import Pose
from openarm.arena_devices.keyboard import OpenArmKeyboardTeleopDevice

keyboard = OpenArmKeyboardTeleopDevice()

# Step 1: Initialize and get the assets from the registry
asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("kitchen")()
embodiment = asset_registry.get_asset_by_name("openarm_bimanual")()
microwave = asset_registry.get_asset_by_name("microwave")()
#chaleira = asset_registry.get_asset_by_name("chaleira")()
#chimarrao = asset_registry.get_asset_by_name("chimarrao")()
#chaleira.set_initial_pose(
#    Pose(position_xyz=(0.3, -0.2, 0.1), rotation_xyzw=(0.707, 0.0, 0.707, 0.0))
#)
#chimarrao.set_initial_pose(
#    Pose(position_xyz=(0.3, 0.1, 0.1), rotation_xyzw=(0.707, 0.0, 0.707, 0.0))
#)
microwave.set_initial_pose(
    Pose(position_xyz=(0.3, 0.0, 0.1), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
)
embodiment.set_initial_pose(
    Pose(position_xyz=(0.0, 0.0, -0.2), rotation_xyzw=(0, 0, 0, 1))
)

# Step 2: Create a scene with the assets
#scene = Scene(assets=[background, chaleira, chimarrao])
scene = Scene(assets=[background, microwave])

# Step 3: Create a task
#task = PourTask(source_recipient=chaleira, destination_recipient=chimarrao)
task = OpenDoorTask(openable_object=microwave)

# Step 4: Create the IsaacLab Arena environment
isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="my_first_arena_env",
    embodiment=embodiment,
    scene=scene,
    task=task,
    teleop_device=keyboard,
)

#policy = OraclePourPolicy(source_recipient=chaleira, destination_recipient=chimarrao)

# Step 5: Build and compile the environment
args_cli = get_isaaclab_arena_cli_parser().parse_args([])
args_cli.mimic = True
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
obs, info = env.reset()

# Step 6: Run the simulation with zero actions
NUM_STEPS = 1000
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        #actions = policy.get_action(env=env, observation=obs)
        obs, reward, terminated, truncated, info = env.step(actions)