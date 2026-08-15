# OpenArm Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

This repository provides simulation and learning environments for the **OpenArm robotic platform**, built on **NVIDIA Isaac Sim** and **Isaac Lab**. It supports research and development in **reinforcement learning (RL)**, **imitation learning (IL)**, **teleoperation**, and **sim-to-real transfer** for both **unimanual (single-arm)** and **bimanual (dual-arm)** robotic systems.

The project integrates with [**IsaacLab-Arena**](https://github.com/isaac-sim/IsaacLab-Arena), enabling composable task definitions, asset registries, and advanced manipulation environments beyond the built-in Isaac Lab tasks.

### What this repo offers

- **Isaac Sim USD models** for OpenArm unimanual and bimanual robots.
- **Isaac Lab RL environments** for manipulation tasks (reach, lift, open drawer, and more).
- **IsaacLab-Arena integration** for composable scene building, task authoring, and multi-policy evaluation.
- **Arena tasks** for pick, pick-and-place, pour, and open door manipulation.
- **Multiple policy types**: RL (RSL-RL, RL-Games, SKRL), LeRobot VLA, LLM-based, and oracle policies.
- **Teleoperation interfaces** including keyboard, bimanual keyboard, and MediaPipe hand tracking.
- **Demonstration recording and dataset generation** for imitation learning pipelines.
- **Custom object assets** (exhaust pipe, chaleira, chimarrao, M16 nut and bolt) with USD meshes and interaction poses.

### Tested with

- **Ubuntu 22.04**
- **Isaac Sim v5.1.0**
- **Isaac Lab v2.3.0**
- **Python 3.11**

---

## Table of Contents

- [Installation Guide](#installation-guide)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
  - [Isaac Lab RL Tasks](#isaac-lab-rl-tasks)
  - [IsaacLab-Arena Tasks](#isaaclab-arena-tasks)
  - [Arena Environments](#arena-environments)
- [Reinforcement Learning (RL)](#reinforcement-learning-rl)
- [IsaacLab-Arena Integration](#isaaclab-arena-integration)
- [Policies](#policies)
- [Teleoperation and Demonstration Recording](#teleoperation-and-demonstration-recording)
- [How to Create New Tasks](#how-to-create-new-tasks)
  - [Creating a New Isaac Lab RL Task](#creating-a-new-isaac-lab-rl-task)
  - [Creating a New Arena Task](#creating-a-new-arena-task)
  - [Creating a New Arena Environment](#creating-a-new-arena-environment)
- [Related Links](#related-links)
- [License](#license)
- [Code of Conduct](#code-of-conduct)

---

## Installation Guide

### (Option 1) Docker installation (Linux only)

1. Pull the minimal Isaac Lab container:
```bash
docker pull nvcr.io/nvidia/isaac-lab:2.3.0
```

2. Create the container:
```bash
xhost +
docker run --name isaac-lab --entrypoint bash -it --gpus all --rm -e "ACCEPT_EULA=Y" --network=host \
   -e "PRIVACY_CONSENT=Y" \
   -e DISPLAY \
   -v $HOME/.Xauthority:/root/.Xauthority \
   -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
   -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
   -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
   -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
   -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
   -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
   -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
   -v ~/docker/isaac-sim/documents:/root/Documents:rw \
   nvcr.io/nvidia/isaac-lab:2.3.0
```

3. Clone the repository:
```bash
cd /workspace
git clone git@github.com:enactic/openarm_isaac_lab.git
```

4. Install the Python package:
```bash
cd openarm_isaac_lab
python -m pip install -e source/openarm
```

5. Verify the installation:
```bash
python ./scripts/tools/list_envs.py
```

### (Option 2) Local installation

It is assumed that you have created a virtual environment named `env_isaaclab` using miniconda or anaconda and will be working within that environment.

1. Clone the repository:
```bash
cd ~
git clone git@github.com:enactic/openarm_isaac_lab.git
```

2. Activate your virtual environment that contains the Isaac Lab package:
```bash
conda activate env_isaaclab
```

3. Install the Python package:
```bash
cd openarm_isaac_lab
python -m pip install -e source/openarm
```

4. Verify the installation:
```bash
python ./scripts/tools/list_envs.py
```

---

## Project Structure

```
openarm_isaac_lab/
├── source/openarm/openarm/
│   ├── tasks/                      # Isaac Lab RL task definitions
│   │   └── manager_based/openarm_manipulation/
│   │       ├── assets/             # Robot articulation configs (unimanual, bimanual)
│   │       ├── usds/               # USD robot model files
│   │       ├── unimanual/          # Unimanual tasks (reach, lift, cabinet)
│   │       └── bimanual/           # Bimanual tasks (reach)
│   ├── arena_tasks/                # IsaacLab-Arena task definitions
│   │   ├── pick.py                 # Pick task
│   │   ├── pick_and_place.py       # Pick and place task
│   │   ├── pour.py                 # Pouring task
│   │   ├── open_door.py            # Door/microwave opening task
│   │   └── mdp/                    # Custom rewards and terminations
│   ├── arena_environments/         # Composable Arena environment definitions
│   │   ├── pick_and_place_exhaust_pipe.py
│   │   ├── pick_exhaust_pipe.py
│   │   ├── open_microwave.py
│   │   └── assembly_nut_and_bolt.py
│   ├── embodiments/                # Robot embodiment configs for Arena
│   │   └── openarm_bimanual/       # Bimanual embodiment with diff-IK control
│   ├── policies/                   # Policy implementations
│   │   ├── lerobot/                # LeRobot VLA policy wrapper
│   │   ├── llm/                    # LLM-based policy
│   │   └── oracle/                 # Hand-crafted oracle policies
│   ├── arena_devices/              # Custom teleoperation devices
│   │   ├── keyboard.py             # Keyboard teleop
│   │   ├── keyboard_bimanual.py    # Bimanual keyboard teleop
│   │   └── mediapipe_teleop_device.py  # MediaPipe hand tracking teleop
│   └── assets/                     # Custom object assets with USD meshes
│       ├── exhaust_pipe.py
│       ├── chaleira.py
│       ├── chimarrao.py
│       ├── nut_m16.py
│       └── bolt_m16.py
├── scripts/
│   ├── reinforcement_learning/     # RL train/play scripts (rsl_rl, rl_games, skrl)
│   ├── policy_runner.py            # Run LeRobot/Oracle/LLM policies
│   ├── record_demos.py             # Record human demonstrations
│   ├── teleop.py                   # Teleoperation interface
│   ├── generate_dataset.py         # Generate training datasets
│   ├── convert_hdf5_to_lerobot_dataset.py
│   └── examples/dummy_task.py      # Example Arena task setup
└── video/                          # Demo GIFs
```

---

## Tasks

### Isaac Lab RL Tasks

These are standard Isaac Lab `ManagerBasedRLEnv` tasks registered as Gymnasium environments. Each task includes full MDP configuration (observations, actions, rewards, terminations, curriculum) and agent configs for multiple RL frameworks.

| Task | Environment ID | Description | Demo |
|------|---------------|-------------|------|
| **Reach** | `Isaac-Reach-OpenArm-v0` | Move end-effector to a randomly sampled target pose in the workspace. Rewards track position and orientation error. | <img src="video/openarm-reach-demo.gif" width="300"/> |
| **Lift Cube** | `Isaac-Lift-Cube-OpenArm-v0` | Grasp a cube from the table and lift it to a target height. Multi-stage rewards for approaching, grasping, and lifting. | <img src="video/openarm-lift-demo.gif" width="300"/> |
| **Open Drawer** | `Isaac-Open-Drawer-OpenArm-v0` | Approach a cabinet drawer handle, grasp it, and pull the drawer open. Complex multi-stage reward shaping (approach, align, grasp, open). | <img src="video/openarm-drawer-demo.gif" width="300"/> |
| **Bimanual Reach** | `Isaac-Reach-OpenArm-Bi-v0` | Dual-arm reach task where both arms move to independent target poses simultaneously. | <img src="video/openarm-bi-reach-demo.gif" width="300"/> |

Each task also has a `-Play` variant (e.g., `Isaac-Reach-OpenArm-Play-v0`) with viewer enabled and fewer environments for visualization.

**Supported RL frameworks:** `rsl_rl`, `rl_games`, `skrl`

### IsaacLab-Arena Tasks

Arena tasks are composable task definitions that can be combined with any embodiment, scene, and set of objects. They inherit from `TaskBase` and define rewards, terminations, events, and metrics independently of the robot.

| Task Class | File | Description |
|-----------|------|-------------|
| **PickTask** | `arena_tasks/pick.py` | Pick up an object from a surface. Success when the object is lifted above a height threshold. Includes object drop detection. |
| **PickAndPlaceTask** | `arena_tasks/pick_and_place.py` | Pick an object and place it at a destination location. Rewards for arm-object proximity and object-goal proximity. Includes contact-based success detection and Isaac Lab Mimic support. |
| **PourTask** | `arena_tasks/pour.py` | Pour from a source container into a destination container. Configurable table limits for containment. |
| **OpenDoorTask** | `arena_tasks/open_door.py` | Open a door or openable object (e.g., microwave). Configurable openness threshold and reset openness. Uses the `Openable` affordance interface. |

### Arena Environments

Arena environments combine an embodiment, a scene (with objects), a task, and optionally a teleop device into a complete simulation. These are the ready-to-run configurations.

| Environment | Class Name | Description |
|------------|-----------|-------------|
| **Pick and Place Exhaust Pipe** | `BimanualOpenArmPickAndPlaceExhaustPipeEnvironment` | Bimanual pick-and-place of an exhaust pipe into a sorting bin on a packing table. Includes RSL-RL training config and LLM policy integration. |
| **Pick Exhaust Pipe** | `BimanualOpenArmPickExhaustPipeEnvironment` | Single-arm pick of an exhaust pipe from a packing table using the bimanual embodiment. |
| **Open Microwave** | `BimanualOpenArmOpenMicrowaveEnvironment` | Open a microwave door in a kitchen scene using the bimanual robot. |
| **Assembly Nut and Bolt** | `BimanualOpenArmAssemblyNutAndBoltEnvironment` | Bimanual assembly task: thread an M16 nut onto a bolt. |

---

## Reinforcement Learning (RL)

### Training a Model

Replace `<TASK_NAME>` with an environment ID from the table above, and `<POLICY_NAME>` with `rsl_rl`, `rl_games`, or `skrl`:

```bash
python ./scripts/reinforcement_learning/<POLICY_NAME>/train.py --task <TASK_NAME> --headless
```

Example:
```bash
python ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-OpenArm-v0 --headless
```

### Replaying a Trained Model

```bash
python ./scripts/reinforcement_learning/<POLICY_NAME>/play.py --task <TASK_NAME> --num_envs 64
```

### Analyzing Training Logs

```bash
python -m tensorboard.main --logdir=logs
```

Then open your browser at `http://localhost:6006/`.

---

## IsaacLab-Arena Integration

The [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) framework provides a composable way to build manipulation environments by combining independent components:

- **Embodiments** define the robot (articulation, actuators, controllers, observations).
- **Assets** define objects in the scene (USD models, interaction poses, contact sensors).
- **Scenes** group assets into a spatial arrangement.
- **Tasks** define the MDP (rewards, terminations, events, metrics) independently of the robot.
- **Environments** wire everything together into a runnable Isaac Lab environment.

The OpenArm project registers:
- **Embodiment:** `openarm_bimanual` -- bimanual robot with differential inverse kinematics control, binary grippers, and optional RGB cameras.
- **Custom assets:** `custom_exhaust_pipe`, `chaleira`, `chimarrao`, `nut_m16`, `bolt_m16`.
- **Custom teleop devices:** keyboard, bimanual keyboard, MediaPipe hand tracking.

### Running Arena Environments

Arena environments are run through the IsaacLab-Arena CLI or scripts:

```bash
# Run the policy runner with an Arena environment
python ./scripts/policy_runner.py \
  --environment-name openarm_bimanual_pick_and_place_exhaust_pipe \
  --policy-type smolvla \
  --policy-repo-id <HuggingFace-model-id> \
  --policy-task "pick and place exhaust pipe"
```

```bash
# Record demonstrations with teleoperation
python ./scripts/record_demos.py \
  --environment-name openarm_bimanual_pick_exhaust_pipe \
  --dataset_file ./datasets/dataset.hdf5 \
  --num_demos 100
```

---

## Policies

| Policy Type | Description | Location |
|------------|-------------|----------|
| **RL Policies** | Standard RL policies trained with RSL-RL, RL-Games, or SKRL. | `scripts/reinforcement_learning/` |
| **LeRobot VLA** | Vision-Language-Action models (SmolVLA, Groot, XVLA) from LeRobot. Accepts camera RGB + robot state, outputs actions. | `policies/lerobot/lerobot_policy.py` |
| **LLM Policy** | LLM-based policy that receives a scene description and task prompt to generate actions. | `policies/llm/llm_policy.py` |
| **Oracle Policies** | Hand-crafted scripted policies for pick-and-place, pouring, and microwave opening. Used for baselines and demonstration generation. | `policies/oracle/` |

---

## Teleoperation and Demonstration Recording

### Teleop Devices

- **Keyboard:** Standard keyboard control for single-arm operation.
- **Bimanual Keyboard:** Keyboard control with key mappings for both left and right arms.
- **MediaPipe:** Hand tracking via webcam using MediaPipe for natural teleoperation.

### Recording Demonstrations

```bash
python ./scripts/record_demos.py \
  --environment-name <ENVIRONMENT_NAME> \
  --dataset_file ./datasets/my_demos.hdf5 \
  --num_demos 50
```

### Converting to LeRobot Format

```bash
python ./scripts/policy/convert_hdf5_to_lerobot_dataset.py
```

---

## How to Create New Tasks

### Creating a New Isaac Lab RL Task

Isaac Lab RL tasks use the `ManagerBasedRLEnv` API with a declarative configuration pattern. Follow these steps:

**1. Create the task directory structure:**

```
source/openarm/openarm/tasks/manager_based/openarm_manipulation/unimanual/my_task/
├── __init__.py
├── my_task_env_cfg.py
├── config/
│   ├── __init__.py              # Gymnasium registration
│   ├── joint_pos_env_cfg.py     # Concrete config with action space
│   └── agents/
│       ├── rsl_rl_ppo_cfg.py    # RSL-RL hyperparameters
│       ├── rl_games_ppo_cfg.yaml
│       └── skrl_ppo_cfg.yaml
└── mdp/
    ├── __init__.py
    └── rewards.py               # Custom reward functions
```

**2. Define the environment config** (`my_task_env_cfg.py`):

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.managers import RewardTermCfg, TerminationTermCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

@configclass
class MyTaskSceneCfg(InteractiveSceneCfg):
    """Define the scene: ground plane, robot, objects, lights."""
    # Add robot articulation, ground plane, objects, lighting
    pass

@configclass
class MyTaskEnvCfg(ManagerBasedRLEnvCfg):
    scene: MyTaskSceneCfg = MyTaskSceneCfg(num_envs=4096, env_spacing=2.5)
    # Define: observations, actions, rewards, terminations, events, curriculum
    # See reach_env_cfg.py for a complete example
```

Key MDP components to define:
- **Observations:** Joint positions, velocities, target poses, object states.
- **Actions:** Joint position commands (arm) + binary gripper action.
- **Rewards:** Task-specific reward terms with weights (use `RewardTermCfg`).
- **Terminations:** Time-out, success conditions, failure conditions.
- **Events:** Reset behaviors (randomize robot pose, object pose).
- **Curriculum:** Gradually adjust difficulty (e.g., increase penalties over training).

**3. Register the environment** (`config/__init__.py`):

```python
import gymnasium as gym

gym.register(
    id="Isaac-My-Task-OpenArm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:MyTaskJointPosEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MyTaskPPORunnerCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

**4. Import the task** in `tasks/__init__.py` so it gets registered on package import.

**5. Train and evaluate:**

```bash
python ./scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-My-Task-OpenArm-v0 --headless
```

Reference implementations:
- Simple: `unimanual/reach/` -- basic end-effector tracking with position/orientation rewards.
- Intermediate: `unimanual/lift/` -- multi-stage pick and lift with custom observations.
- Advanced: `unimanual/cabinet/` -- multi-stage grasp and pull with physics randomization.

### Creating a New Arena Task

Arena tasks are more modular. They define the MDP logic independently of the robot or scene layout.

**1. Create a task class** in `arena_tasks/`:

```python
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.assets.asset import Asset
from isaaclab.managers import TerminationTermCfg, RewardTermCfg, EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_isaac_lab

class MyArenaTask(TaskBase):
    def __init__(self, target_object: Asset, episode_length_s: float = None):
        super().__init__(episode_length_s=episode_length_s)
        self.target_object = target_object
        # Initialize configs...

    def get_rewards_cfg(self):
        """Return a configclass with RewardTermCfg fields."""
        return MyRewardsCfg()

    def get_termination_cfg(self):
        """Return a configclass with TerminationTermCfg fields."""
        return MyTerminationsCfg()

    def get_events_cfg(self):
        """Return a configclass with EventTermCfg fields (e.g., reset poses)."""
        return MyEventsCfg()

    def get_scene_cfg(self):
        """Return additional scene elements (sensors, frame transformers)."""
        return None

    def get_prompt(self):
        """Natural language description for LLM policies."""
        return f"Do something with the {self.target_object.name}."

    def get_metrics(self):
        """Return list of MetricBase instances for evaluation."""
        return [SuccessRateMetric()]

    def get_viewer_cfg(self):
        """Camera viewpoint for visualization."""
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))

    def get_mimic_env_cfg(self, embodiment_name: str):
        """Return MimicEnvCfg for Isaac Lab Mimic data generation."""
        return MyMimicEnvCfg(embodiment_name=embodiment_name)
```

**2. Add custom reward functions** in `arena_tasks/mdp/rewards.py`:

```python
def my_custom_reward(env, asset_cfg, target_pos):
    """Compute reward based on distance to target."""
    asset = env.scene[asset_cfg.name]
    distance = torch.norm(asset.data.root_pos_w - target_pos, dim=-1)
    return torch.exp(-distance)
```

### Creating a New Arena Environment

Arena environments wire together an embodiment, scene, task, and optional policies.

**1. Create an environment class** in `arena_environments/`:

```python
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

class MyEnvironment(ExampleEnvironmentBase):
    name: str = "openarm_bimanual_my_task"

    def get_env(self, args_cli):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        import openarm.embodiments
        import openarm.assets

        # 1. Get assets
        teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        background = self.asset_registry.get_asset_by_name("packing_table")()
        embodiment = self.asset_registry.get_asset_by_name("openarm_bimanual")(
            enable_cameras=args_cli.enable_cameras
        )
        my_object = self.asset_registry.get_asset_by_name("my_custom_object")()

        # 2. Set poses
        background.set_initial_pose(Pose(position_xyz=(0.2, 0.0, -1.0), rotation_xyzw=(0.0, 0.0, -0.707, 0.707)))
        my_object.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0, 0, 0, 1)))

        # 3. Create scene and task
        scene = Scene(assets=[background, my_object])
        task = MyArenaTask(target_object=my_object, episode_length_s=10.0)

        # 4. Build environment
        return IsaacLabArenaEnvironment(
            name="my_arena_env",
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )

    @staticmethod
    def add_cli_args(parser):
        parser.add_argument("--teleop_device", type=str, default="keyboard")
```

**2. Register the environment** by importing the module so the `ExampleEnvironmentBase` metaclass picks it up.

**3. Run it:**

```bash
python ./scripts/policy_runner.py --environment-name openarm_bimanual_my_task --policy-type oracle
```

---

## Related Links

* Read the [documentation](https://docs.openarm.dev/)
* Join the community on [Discord](https://discord.gg/FsZaZ4z3We)
* Contact us through <openarm@enactic.ai>

## License

[Apache License 2.0](LICENSE.txt)

Copyright 2025 Enactic, Inc.

## Code of Conduct

All participation in the OpenArm project is governed by our [Code of Conduct](CODE_OF_CONDUCT.md).
