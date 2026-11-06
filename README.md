# OpenArm Isaac Lab

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.0.0--dev2-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/index.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-3.0.0--beta-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-Apache2.0-yellow.svg)](https://opensource.org/license/apache-2-0)

## Overview

This repository provides simulation and learning environments for the **OpenArm robotic platform**, built on **NVIDIA Isaac Sim** and **Isaac Lab**. It supports research and development in **reinforcement learning (RL)**, **imitation learning (IL)**, **teleoperation**, and **sim-to-real transfer** for both **unimanual (single-arm)** and **bimanual (dual-arm)** robotic systems.

The project integrates with [**IsaacLab-Arena**](https://github.com/isaac-sim/IsaacLab-Arena), enabling composable task definitions, asset registries, and advanced manipulation environments beyond the built-in Isaac Lab tasks.

### What this repo offers

- **Isaac Sim USD models** for OpenArm unimanual and bimanual robots, plus the OpenRob mobile manipulator.
- **Isaac Lab RL environments** for reach, lift, open drawer, and bimanual reach, with `rsl_rl`, `rl_games`, and `skrl` configs.
- **IsaacLab-Arena integration** for composable scene building, task authoring, Mimic data generation, and multi-policy evaluation.
- **50 staged industrial tasks** across fastening, packing, beverage, containers, tools, and inspection families, all generated from declarative task specs.
- **A staged task framework** that drives task goals, dense rewards, observations, Mimic segmentation, and the scripted oracle policy from one description.
- **Domain randomization** covering object pose, scale, material, mass, robot state, action latency/noise, camera parameters, photometric augmentation, physics, and HDR lighting, with `none`, `light`, `default`, and `heavy` presets.
- **Two Arena embodiments:** `openarm_bimanual` (fixed-base dual/single arm) and `openrob` (mecanum mobile base with two arms).
- **Multiple policy types**: RL (RSL-RL, RL-Games, SKRL), LeRobot VLA (SmolVLA, Groot, XVLA), LLM-based, scripted oracle, and zero-action policies.
- **Teleoperation interfaces** including keyboard, bimanual keyboard, MediaPipe hand tracking, and OpenXR.
- **Demonstration recording, Mimic annotation/generation, and dataset conversion** to HDF5 and LeRobot formats, including OpenRob locomanip dataset generation.
- **Custom object assets** (exhaust pipe, sorting bin, microwave, desktop computer, chaleira, chimarrao, M16 nut and bolt) and a 300+ image PolyHaven HDR library.
- **Static and runtime tooling**: environment listing, industrial task smoke tests, oracle evaluation, and five pytest suites.

### Tested with

- **Ubuntu 22.04**
- **Isaac Sim v6.0.0-dev2**
- **Isaac Lab v3.0.0-beta** (pinned submodule commit `e57379c63`)
- **IsaacLab-Arena release/0.2.1** (pinned submodule commit `8b4a3a47f`)
- **Python 3.11**

---

## Table of Contents

- [Installation Guide](#installation-guide)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
  - [Isaac Lab RL Tasks](#isaac-lab-rl-tasks)
  - [IsaacLab-Arena Task Library](#isaaclab-arena-task-library)
  - [Arena Environments](#arena-environments)
  - [Industrial Staged Tasks](#industrial-staged-tasks)(WIP)
  - [Staged Task Framework](#staged-task-framework)(WIP)
  - [Embodiments](#embodiments)
- [Reinforcement Learning (RL)](#reinforcement-learning-rl)
- [IsaacLab-Arena Integration](#isaaclab-arena-integration)
  - [Running Arena Environments](#running-arena-environments)
  - [External Environment Modules](#external-environment-modules)
  - [Assets and HDR Library](#assets-and-hdr-library)
- [Domain Randomization](#domain-randomization)(WIP)
- [Policies](#policies)
- [Teleoperation and Demonstration Recording](#teleoperation-and-demonstration-recording)
- [Data Generation and Dataset Conversion](#data-generation-and-dataset-conversion)
- [Tools and Tests](#tools-and-tests)
- [How to Create New Tasks](#how-to-create-new-tasks)
  - [Creating a New Industrial Staged Task](#creating-a-new-industrial-staged-task)
  - [Creating a New Isaac Lab RL Task](#creating-a-new-isaac-lab-rl-task)
  - [Creating a New Arena Task](#creating-a-new-arena-task)
  - [Creating a New Arena Environment](#creating-a-new-arena-environment)
- [Related Links](#related-links)
- [License](#license)
- [Code of Conduct](#code-of-conduct)

---

## Installation Guide

Isaac Lab and IsaacLab-Arena are included as **pinned git submodules** under `submodules/`, so the
first step is to clone them after (or while) cloning this repository.

1. Clone the repository **with submodules**:
```bash
git clone --recurse-submodules git@github.com:OpenRob-FURG/openarm_isaac_lab.git
```

Or, if the repository was already cloned, initialize them:
```bash
git submodule update --init --recursive
```

The submodules are pinned to the tested versions:
- `submodules/IsaacLab` → `e57379c63` (Isaac Lab 3.0.0-beta)
- `submodules/IsaacLab-Arena` → `8b4a3a47f` (IsaacLab-Arena release/0.2.1)

### (Option 1) Docker installation (Linux only)

1. Build the image and start a container (the build pulls Isaac Sim's `nvcr.io/nvidia/isaac-sim:6.0.0-dev2` base image):
```bash
./docker/run_docker.sh -r
```

This builds `openarm_isaac_lab:latest` with Isaac Lab, IsaacLab-Arena, LeRobot, and the `openarm` package
installed, then drops you into a shell inside the container with the repository mounted at
`/workspaces/openarm_isaac_lab`.

Additional run options:
```bash
./docker/run_docker.sh -n my_image     # use a different image name
./docker/run_docker.sh -m ~/models     # mount a host models directory at /models
./docker/run_docker.sh -d ~/datasets   # mount a host datasets directory at /datasets
./docker/run_docker.sh -e ~/eval       # mount a host evaluation directory at /eval
./docker/run_docker.sh -r              # force an image rebuild
./docker/run_docker.sh -h              # print the full help
```

Inside the container, `python` is aliased to `/isaac-sim/python.sh` and `pytest` to
`/isaac-sim/python.sh -m pytest`, so the commands in this README can be run directly.

2. Verify the installation inside the container:
```bash
python ./scripts/tools/list_envs.py
```

To build the image manually:
```bash
docker build -t openarm_isaac_lab:latest -f docker/Dockerfile .
```

### (Option 2) Local installation

It is assumed that you have created a virtual environment named `env_isaaclab` using miniconda or anaconda and will be working within that environment.

1. Clone the repository (with submodules, see above) and activate your virtual environment that contains the Isaac Lab package:
```bash
git clone --recurse-submodules git@github.com:OpenRob-FURG/openarm_isaac_lab.git
cd openarm_isaac_lab
conda activate env_isaaclab
```

2. Install IsaacLab-Arena (from its submodule):
```bash
pip install -e submodules/IsaacLab-Arena
```

3. Install the Python package:
```bash
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
├── submodules/
│   ├── IsaacLab-Arena/             # Pinned git submodule (release/0.2.1)
│   └── IsaacLab/                   # Pinned git submodule (v3.0.0-beta)
├── docker/
│   ├── Dockerfile                  # Self-contained build (Isaac Sim → Lab → Arena → openarm)
│   ├── run_docker.sh               # Build/run helper
│   └── setup/entrypoint.sh         # Container entrypoint
├── source/openarm/openarm/
│   ├── tasks/                      # Isaac Lab RL task definitions
│   │   └── manager_based/openarm_manipulation/
│   │       ├── assets/             # Robot articulation configs (unimanual, bimanual, openrob)
│   │       ├── usds/               # USD robot model files
│   │       ├── unimanual/          # Unimanual tasks (reach, lift, cabinet)
│   │       └── bimanual/           # Bimanual tasks (reach)
│   ├── arena_tasks/                # IsaacLab-Arena task definitions
│   │   ├── pick.py                 # Pick task
│   │   ├── pick_and_place.py       # Pick and place task
│   │   ├── pour.py                 # Pouring task
│   │   ├── pick_and_place_component.py  # Component-level pick and place
│   │   ├── mdp/                    # Custom rewards and terminations
│   │   └── staged/                 # Declarative staged task framework
│   │       ├── stages.py           # TaskSpec, Stage, Skill
│   │       ├── task.py             # StagedTask
│   │       ├── goals.py            # Composable success predicates
│   │       ├── layout.py           # Semantic scene layout
│   │       ├── arms.py             # Arm assignment/kinematics helpers
│   │       ├── scene_state.py      # Scene snapshots for goals and rewards
│   │       ├── rewards.py          # Dense reward terms
│   │       ├── observations.py     # Privileged observations
│   │       ├── terminations.py     # Goal evaluation
│   │       └── mimic.py            # Mimic segmentation from stages
│   ├── arena_environments/         # Ready-to-run Arena environments
│   │   ├── pick_and_place_exhaust_pipe.py
│   │   ├── pick_exhaust_pipe.py
│   │   ├── open_microwave.py
│   │   ├── assembly_nut_and_bolt.py
│   │   ├── computer_disassembly.py # Additional module (loaded externally)
│   │   ├── galileo_openrob_locomanip_pick_place.py  # Additional module (loaded externally)
│   │   ├── cli.py                  # Environment registry and CLI
│   │   └── industrial/             # 50 staged industrial tasks
│   │       ├── fastening.py
│   │       ├── packing.py
│   │       ├── beverage.py
│   │       ├── containers.py
│   │       ├── tools.py
│   │       ├── inspection.py
│   │       ├── _base.py            # Shared builder and CLI flags
│   │       └── rsl_rl_cfg.py       # Asymmetric actor/critic runner config
│   ├── arena_dr/                   # Domain randomization
│   │   ├── config.py               # DRConfig and presets
│   │   ├── events.py               # Runtime randomization events
│   │   ├── terms.py                # Observations and augmentation terms
│   │   └── applier.py              # Config/asset/light application
│   ├── arena_devices/              # Custom teleoperation devices
│   │   ├── keyboard.py             # OpenArm keyboard teleop
│   │   ├── keyboard_bimanual.py    # Dual-arm keyboard teleop
│   │   ├── mediapipe_teleop_device.py  # MediaPipe hand tracking teleop
│   │   └── openxr.py               # OpenXR device config
│   ├── arena_retargeters/          # Teleop retargeters
│   │   ├── openarm_bimanual.py     # Keyboard/MediaPipe/OpenXR → OpenArm
│   │   ├── openrob.py              # OpenXR → OpenRob
│   │   └── mecanum_joystick_retargeter.py
│   ├── embodiments/                # Robot embodiment configs for Arena
│   │   └── openarm_bimanual/       # openarm_bimanual and openrob embodiments
│   ├── policies/                   # Policy implementations
│   │   ├── lerobot/                # LeRobot VLA policy wrapper
│   │   ├── llm/                    # Ollama-backed LLM policy
│   │   └── oracle/                 # Scripted and staged oracle policies
│   ├── assets/                     # Custom object assets with USD meshes
│   │   ├── exhaust_pipe.py
│   │   ├── sorting_bin.py
│   │   ├── microwave.py
│   │   ├── desktop_computer.py
│   │   ├── chaleira.py
│   │   ├── chimarrao.py
│   │   ├── nut_m16.py
│   │   ├── bolt_m16.py
│   │   ├── chemical_workshop.py
│   │   ├── polyhaven_hdr_library.py
│   │   ├── meshes/                 # USD/GLB meshes
│   │   └── hdris/                  # PolyHaven HDR backgrounds
│   ├── tests/                      # Pytest suites for the staged framework
│   └── ui_extension_example.py     # Isaac Sim UI extension example
├── scripts/
│   ├── reinforcement_learning/     # RL train/play scripts (rsl_rl, rl_games, skrl)
│   ├── policy_runner.py            # Run oracle, LLM, zero-action, or LeRobot policies
│   ├── record_demos.py             # Record human demonstrations
│   ├── annotate_demos.py           # Add Mimic subtask annotations
│   ├── generate_dataset.py         # Generate Mimic datasets
│   ├── teleop.py                   # Standalone teleoperation interface
│   ├── tools/
│   │   ├── list_envs.py            # List registered Isaac Lab tasks
│   │   ├── smoke_industrial.py     # Build/step smoke test for industrial tasks
│   │   └── eval_oracle.py          # Measure oracle success and explain failures
│   ├── policy/
│   │   ├── convert_hdf5_to_lerobot_dataset.py
│   │   ├── eval_policy.py
│   │   ├── replay_papare_video_for_lerobot_dataset.py
│   │   └── config/                 # Conversion/evaluation YAML configs
│   ├── env_data_generation/
│   │   └── galileo_openrob_locomanip_pick_and_place.py
│   ├── train_lerobot_on_modal.py   # Modal-hosted LeRobot training
│   └── examples/dummy_task.py      # Example Arena task setup
├── datasets/                       # Recorded and generated datasets
├── outputs/                        # Training outputs
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

### IsaacLab-Arena Task Library

Arena tasks are composable task definitions that can be combined with any embodiment, scene, and set of objects. They inherit from `TaskBase` and define rewards, terminations, events, and metrics independently of the robot.

| Task Class | Location | Description |
|-----------|----------|-------------|
| **PickTask** | `arena_tasks/pick.py` | Pick up an object from a surface. Success when the object is lifted above a height threshold. Includes object drop detection. |
| **PickAndPlaceTask** | `arena_tasks/pick_and_place.py` | Pick an object and place it at a destination location. Rewards for arm-object proximity and object-goal proximity, with Arena's `object_on_destination` success check and Isaac Lab Mimic support. |
| **PourTask** | `arena_tasks/pour.py` | Pour from a source container into a destination container. Configurable table limits for containment. |
| **PickAndPlaceComponentTask** | `arena_tasks/pick_and_place_component.py` | Pick a named sub-component out of a composite/openable asset (e.g. a GPU or motherboard from a desktop computer) and place it at a destination. |
| **OpenDoorTask** | IsaacLab-Arena: `isaaclab_arena.tasks.open_door_task` | Open an openable object such as a microwave or computer case. Configurable openness threshold and reset openness. Used by `open_microwave.py` and `computer_disassembly.py`. |

The staged industrial tasks use an additional task class, `StagedTask`, described in
[Staged Task Framework](#staged-task-framework). Custom reward and termination terms shared by
Arena tasks live in `arena_tasks/mdp/rewards.py` and `arena_tasks/mdp/terminations.py`.

### Arena Environments

Arena environments combine an embodiment, a scene (with objects), a task, and optionally a teleop device into a complete simulation. These are the ready-to-run configurations registered by `openarm/arena_environments/cli.py`.

| Environment | Class Name | Description |
|------------|-----------|-------------|
| **Pick and Place Exhaust Pipe** | `BimanualOpenArmPickAndPlaceExhaustPipeEnvironment` | Bimanual pick-and-place of an exhaust pipe into a sorting bin on a packing table. Includes RSL-RL training config and LLM policy integration. |
| **Pick Exhaust Pipe** | `BimanualOpenArmPickExhaustPipeEnvironment` | Pick an exhaust pipe from a packing table using the bimanual embodiment; no placement stage. |
| **Open Microwave** | `BimanualOpenArmOpenMicrowaveEnvironment` | Open a microwave door in a kitchen scene using the bimanual robot. |
| **Assembly Nut and Bolt** | `BimanualOpenArmAssemblyNutAndBoltEnvironment` | Bimanual assembly task: thread an M16 nut onto a bolt. |
| **50 Industrial Staged Tasks** | `Industrial<TaskKeyCamel>Environment` (e.g. `IndustrialThreadNutOnBoltEnvironment`) | Generated one-per-task from the staged task registry. See [Industrial Staged Tasks](#industrial-staged-tasks). |

### Industrial Staged Tasks

The `industrial/` package defines **exactly 50** staged tasks. Every task is pure data: a
`TaskSpec` plus the scene it needs. All 50 share a single environment implementation, one builder,
and one oracle policy, and each is exposed as its own CLI subcommand named
`openarm_bimanual_industrial_<task_key>`.

Run any of them with the environment name as the positional subcommand:

```bash
# Scripted oracle rollout
python scripts/policy_runner.py --policy-type oracle openarm_bimanual_industrial_thread_nut_on_bolt

# Headless build/step smoke test
python scripts/tools/smoke_industrial.py --steps 3 openarm_bimanual_industrial_thread_nut_on_bolt --no-cameras

# Measure the oracle's success rate and explain failures
python scripts/tools/eval_oracle.py --episodes 3 openarm_bimanual_industrial_thread_nut_on_bolt
```

Per-task CLI flags (place them **after** the environment name):

| Flag | Description |
|------|-------------|
| `--dr-preset {none,light,default,heavy}` | Override the task's domain-randomization preset. |
| `--no-cameras` | Disable the task's RGB camera. Required on machines without a render device. |
| `--single-arm` | Run the task with one arm instead of the bimanual default. |
| `--rl-training` | Attach the dense reward manager and privileged critic observation group, attach the RSL-RL runner config, and disable the camera. |

#### Fastening (10)

| Task key | Instruction |
|----------|-------------|
| `thread_nut_on_bolt` | Thread the nut onto the bolt. |
| `peg_into_hole` | Insert the peg into the hole. |
| `gear_pair_meshing` | Mount both gears on the base so that they mesh. |
| `gear_train_build` | Mount all three gears on the base in order. |
| `hold_and_hammer` | Hold the block steady with one arm and hammer it with the other. |
| `nail_driving` | Hold the plank and drive the nail with the hammer. |
| `clamp_bracing` | Clamp the bracket to the table and tighten it. |
| `drill_fastening` | Hold the plate and drill the fastener through it. |
| `bolt_bracing_insertion` | Brace the gear base with one arm and insert the bolt with the other. |
| `sort_fastener_sizes` | Sort the fasteners by size into the matching bins. |

#### Packing (10)

| Task key | Instruction |
|----------|-------------|
| `lunchbag_packing` | Pack the three food items into the lunchbag. |
| `can_sorting_by_size` | Sort the cans into the bins by size. |
| `snack_box_kitting` | Kit the snacks into the storage box. |
| `bagel_snack_assembly` | Put the bagel on the plate and add the spread. |
| `bottle_destuffing` | Move the bottles into the crate. |
| `case_palletizing` | Stack the two cases into a stable tower. |
| `visual_qc_reject` | Inspect the parts and reject the damaged one into the reject bin. |
| `shelf_restock` | Restock the shelf with both items. |
| `meal_kit_portioning` | Portion the ingredients into the bowl. |
| `frozen_goods_restock` | Restock the frozen goods into the freezer bin. |

#### Beverage (8)

| Task key | Instruction |
|----------|-------------|
| `microwave_heat_cycle` | Open the microwave, put the mug inside, and close the door. |
| `coffee_dispense` | Press the coffee machine and place the mug under the spout. |
| `mug_handoff_dispense` | Pour the coffee and hand the filled mug to the other arm. |
| `kettle_pour` | Pour from the jug into the mug. |
| `stand_mixer_setup` | Set the bowl on the mixer and turn the control on. |
| `dual_mug_serve` | Serve two mugs at the same time, one per arm. |
| `reheat_then_pour` | Reheat the drink in the microwave, then pour it into a clean mug. |
| `measured_dosing` | Measure the correct dose with the measuring cup and pour it in. |

#### Containers (6)

| Task key | Instruction |
|----------|-------------|
| `container_transfer` | Transfer the item from the red container into the green one. |
| `can_tower_build` | Stack all three cans into a tower. |
| `unstack_restack` | Unstack the tower and restack the cans in reverse order. |
| `bin_dump_and_clear` | Empty the container's contents into the bin. |
| `carton_handling` | Move the carton into the bin and keep it upright. |
| `double_container_load` | Load both containers with their matching items. |

#### Tools (10)

| Task key | Instruction |
|----------|-------------|
| `ladle_serving` | Hold the bowl steady and serve from it with the ladle. |
| `tongs_serving` | Use the tongs to move the item into the dish. |
| `measuring_cup_transfer` | Measure a cupful and transfer it to the mixing bowl. |
| `tool_racking` | Return both tools to the rack. |
| `cross_hand_tool_handoff` | Pass the tool from one hand to the other. |
| `stir_while_holding` | Hold the bowl in one hand and stir it with the other. |
| `spreading_task` | Spread the mixture across the plate. |
| `scissor_cutting` | Cut the material with the scissors. |
| `scrubbing_plates` | Hold the plate and scrub it with the sponge. |
| `mash_while_holding` | Hold the bowl and mash its contents with the masher. |

#### Inspection (6)

| Task key | Instruction |
|----------|-------------|
| `colour_sorting` | Sort the four blocks into the bins by colour. |
| `shape_sorting` | Sort the objects by shape into the bins. |
| `multi_attribute_sorting` | Sort the fruit by both colour and type. |
| `odd_one_out_reject` | Find the item that does not belong and reject it. |
| `count_and_stack` | Count the items, then stack all three of them. |
| `ripeness_grading` | Grade the fruit into the fresh and the reject bins. |

### Staged Task Framework

A staged task is described once, as data, in a
`TaskSpec` (`openarm/arena_tasks/staged/stages.py`):

- **`Stage`** — one step of a task, with a target object, an optional secondary object, a `Skill`,
  and an optional `Goal`. The acting arm is resolved from the layout's left/right sides rather
  than trusted from the authored field.
- **`Skill`** — the kind of manipulation a stage performs: `MOVE`, `PICK`, `PLACE`,
  `PICK_PLACE`, `PRESS`, `OPEN`, `CLOSE`, `TURN`, `STRIKE`, `INSERT`, `HANDOFF`, `STABILIZE`,
  or `DOCK`.
- **`Goal`** — a composable success predicate: `ObjectInRegion`, `ObjectRelativeTo`,
  `ObjectUpright`, `ObjectsWithinDistance`, `JointNormalizedAbove`, `JointNormalizedBelow`,
  `ObjectAtRest`, `ObjectAbovePlane`, plus `All` and `AtLeast` combinators.
- **`SceneLayout`** — a semantic description of where things are (`On`, `NextTo`, `AtPosition`,
  `PositionLimits`, `NoCollision`, `IsAnchor`, `RandomAroundSolution`, `RotateAroundSolution`)
  solved by Arena's own `ObjectPlacer` from real asset bounding boxes.

The same stage list drives four consumers, so they cannot drift apart:

1. **Task goals and terminations** — every stage's goal becomes a success predicate and a
   language instruction.
2. **Dense rewards** — `arena_tasks/staged/rewards.py` mirrors each goal with a continuous score,
   giving a learning signal before the full task is completed.
3. **Privileged observations** — object poses, joint states, and end-effector state for the
   asymmetric critic used by RL training.
4. **Mimic segmentation and the oracle** — `arena_tasks/staged/mimic.py` reads stage boundaries
   off the stage list, and `policies/oracle/staged_oracle.py` runs one skill controller per stage
   with retries and a shared retry budget.

The pure-data half of the package (`goals`, `layout`, `stages`) imports nothing from Isaac Sim, so
it can be used by the CLI parser and by tests before the simulation app starts.

### Embodiments

Two Arena embodiments are registered:

| Embodiment | Description |
|-----------|-------------|
| **`openarm_bimanual`** | Fixed-base OpenArm with two 7-DoF arms. Differential-IK arm actions with binary grippers; `ArmMode` supports dual-arm, single-arm, left, and right configurations. Provides top, left wrist, and right wrist cameras, a `policy` proprioception group, Mimic support, and an XR anchor configuration. |
| **`openrob`** | Mecanum-wheeled mobile manipulator with two 7-DoF arms and parallel grippers. Exposes a base velocity command `[vx, vy, wz]` alongside the arm/gripper actions, XR thumbstick retargeting, A*/occupancy-grid navigation, scripted pick/place/open/close/press/bimanual-lift/bimanual-drop skills, and LeRobot episode recording through `OpenRobInterface`. |

Both embodiments live under `openarm/embodiments/openarm_bimanual/`. The `openrob` embodiment is
used by the Galileo OpenRob locomanip environment.

---

## Reinforcement Learning (RL)

### Isaac Lab RL Tasks

`rl_games` and `skrl` use the standard Isaac Lab task IDs:

```bash
python ./scripts/reinforcement_learning/rl_games/train.py --task Isaac-Reach-OpenArm-v0 --headless
python ./scripts/reinforcement_learning/skrl/train.py --task Isaac-Reach-OpenArm-v0 --headless
```

The `rsl_rl` training entry point in this repository is wired to the Arena builder rather than to
Hydra task IDs, so it takes an Arena environment subcommand instead of `--task` (see the next
section). The standard RSL-RL configs are still registered for the Isaac Lab tasks, and
`rsl_rl/play.py` uses `--task` normally.

### Industrial Staged Tasks with RSL-RL

The industrial tasks have a dedicated RSL-RL path. `--rl-training` attaches the dense reward
manager, a privileged critic observation group, and the runner config from
`arena_environments/industrial/rsl_rl_cfg.py`. The actor observes proprioception only; the critic
additionally observes the scene objects, which is what makes the value estimate depend on the
state that determines the reward.

```bash
python ./scripts/reinforcement_learning/rsl_rl/train.py \
  --headless --num_envs 8 \
  openarm_bimanual_industrial_thread_nut_on_bolt --rl-training
```

### Replaying a Trained Isaac Lab Model

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

### Running Arena Environments

> **Argument order matters.** Arena scripts take the environment name as a required positional
> subcommand (`example_environment`). Global arguments such as `--headless`, `--num_envs`,
> `--enable_cameras`, `--seed`, `--mimic`, and script-level flags such as `--policy-type`,
> `--policy-repo-id`, `--dataset_file`, or `--episodes` must be passed **before** the environment
> name. Per-environment flags such as `--teleop_device`, `--dr-preset`, `--no-cameras`,
> `--single-arm`, and `--rl-training` must be passed **after** it.

```bash
# Run the policy runner with a staged industrial task and its scripted oracle policy
python ./scripts/policy_runner.py \
  --policy-type oracle \
  openarm_bimanual_industrial_thread_nut_on_bolt

# Run a LeRobot VLA policy (SmolVLA, Groot, or XVLA)
python ./scripts/policy_runner.py \
  --policy-type smolvla \
  --policy-repo-id <HuggingFace-model-id> \
  --policy-task "pick and place exhaust pipe" \
  openarm_bimanual_pick_and_place_exhaust_pipe

# Run an LLM policy served by Ollama's OpenAI-compatible endpoint
python ./scripts/policy_runner.py \
  --policy-type llm \
  --policy-task "pick and place exhaust pipe" \
  openarm_bimanual_pick_and_place_exhaust_pipe
```

The scripted oracle is only attached to the industrial staged tasks; the other environments do not
set an `oracle_policy`. The LeRobot policies read the `camera_obs` observation group, while
`get_arena_builder_from_cli` currently forces cameras off in the builder, so using those policies
through this script requires enabling camera sensors in the builder first.

The policy runner records successful episodes to
`./datasets/policy_generated_dataset.hdf5` and prints the success rate over 100 episodes.
`--policy-type` accepts `scripted`/`oracle`, `llm`, `zero_action`, `smolvla`, `groot`, and `xvla`.

### External Environment Modules

Two additional environment modules are implemented but are not part of the default
CLI registry. Load them with `--external_environment_class_path`:

```bash
# Bimanual computer disassembly: open the case, then remove the GPU, two Trident Z memory modules, and motherboard
python ./scripts/policy_runner.py \
  --policy-type zero_action \
  --external_environment_class_path \
  openarm.arena_environments.computer_disassembly:BimanualOpenArmComputerDisassemblyEnvironment \
  openarm_bimanual_computer_disassembly

# OpenRob mobile-manipulator pick-and-place
python ./scripts/policy_runner.py \
  --policy-type zero_action \
  --external_environment_class_path \
  openarm.arena_environments.galileo_openrob_locomanip_pick_place:GalileoOpenRobLocomanipPickAndPlaceEnvironment \
  galileo_openrob_locomanip_pick_and_place
```

- **`openarm_bimanual_computer_disassembly`** uses Arena's `SequentialTaskBase` to chain
  `OpenDoorTask` with four `PickAndPlaceComponentTask` steps (GPU, two Trident Z memory modules,
  motherboard) into a sorting bin.
- **`galileo_openrob_locomanip_pick_and_place`** uses the `openrob` embodiment and Arena's
  `G1LocomanipPickAndPlaceTask`, including a Mimic patch for mobile manipulation.

### Assets and HDR Library

The `openarm/assets/` package provides:

- **Custom objects:** `custom_exhaust_pipe`, `openarm_sorting_bin`, `openarm_microwave`,
  `chaleira`, `chimarrao`, `nut_m16`, `bolt_m16`.
- **Openable/component assets:** `desktop_computer` (openable, with a GPU, two Trident Z memory
  modules, and an MSI motherboard). It is registered when its module is imported, which the
  computer-disassembly environment does.
- **Custom background:** `chemical_background` (`chemical_workshop.py`), available for scenes that
  import it.
- **PolyHaven HDR library:** 301 HDR environment maps under `assets/hdris/`, registered with the
  `polyhaven` tag and sampled by the domain-randomization light builder.

Robot USD models live in `tasks/manager_based/openarm_manipulation/usds/` for the unimanual,
bimanual, and OpenRob robots. `procedural_table` and other backgrounds/objects come from Arena's
own asset library.

---

## Domain Randomization

`openarm/arena_dr/` provides sim-to-real randomization for the industrial tasks. All ranges are
`(min, max)` tuples; a collapsed range disables that randomization. The `DRConfig` presets are:

| Preset | Purpose |
|--------|---------|
| `none` | Fully deterministic. |
| `light` | Visual-only randomization. Geometry, physics, and robot state stay deterministic. This preset is the industrial suite's default, though individual tasks may choose another preset (for example, `thread_nut_on_bolt` uses `default`). |
| `default` | Sim-to-real randomization: pose/scale/material/mass, robot joints, action delay/noise, end-effector observation noise, light, camera extrinsics/intrinsics, photometric augmentation, physics, and HDR sampling. |
| `heavy` | Aggressive ranges for robustness training and evaluation. |

The randomization covers:

- **Objects:** initial pose/yaw/height, scale, friction/restitution, mass, contact and rest offsets.
- **Robot:** joint position offsets and velocities on reset.
- **Control:** action delay, additive action noise, end-effector observation noise.
- **Vision:** light intensity/colour temperature, camera mount jitter, focal length/aperture/focus,
  brightness/contrast/gamma/colour cast/sensor noise, and HDR environment-map sampling.
- **Physics:** timestep, solver iteration budget, and collider offsets.

Pinned objects (fixtures such as a microwave or a bolt to be threaded) are excluded from pose
jitter. Use `--dr-preset` on an industrial task subcommand to override the task's default.

---

## Policies

| Policy Type | Description | Location |
|------------|-------------|----------|
| **RL Policies** | Policies trained with RSL-RL, RL-Games, or SKRL. | `scripts/reinforcement_learning/` |
| **LeRobot VLA** | Vision-Language-Action models (SmolVLA, Groot, XVLA) from LeRobot. Uses camera RGB and robot state and outputs actions. | `policies/lerobot/lerobot_policy.py` |
| **LLM Policy** | Queries a vision-language model through Ollama's OpenAI-compatible API, parses an action chunk, and supports action/observation history buffers. Default model: `qwen3.5` at `http://localhost:11434/v1`. | `policies/llm/llm_policy.py` |
| **Oracle Policies** | Hand-crafted scripted policies for pick-and-place, pouring, and microwave opening, plus a retry-capable `StagedOraclePolicy` that walks any staged `TaskSpec`. | `policies/oracle/` |
| **Zero-Action Policy** | Emits zero actions. Useful for smoke tests and environment inspection. | `scripts/policy_runner.py` |

The staged oracle is deterministic given the environment, which makes generated demonstrations
reproducible. When a stage times out or fails to produce its intended effect, the policy retreats,
opens the gripper, and retries within the stage's `max_attempts` and a global retry budget.

---

## Teleoperation and Demonstration Recording

### Teleop Devices

| Device | Registered Name | Description |
|--------|-----------------|-------------|
| **OpenArm Keyboard** | `keyboard__openarm_bimanual` | Keyboard teleoperation for the OpenArm bimanual embodiment. |
| **Bimanual Keyboard** | `keyboard_bimanual` | Two-arm 6-DoF delta-pose and binary-gripper keyboard control, with active-arm switching. |
| **MediaPipe** | `mediapipe` | Webcam hand tracking via MediaPipe for natural teleoperation. |
| **OpenXR** | `openxr__openarm_bimanual` | OpenXR hand-controller teleoperation. |

Retargeters in `arena_retargeters/` map these devices onto the embodiments: keyboard,
bimanual keyboard, MediaPipe, and OpenXR for `openarm_bimanual`; OpenXR for `openrob`; and a
mecanum joystick retargeter for mobile-base commands.

### Recording Demonstrations

```bash
python ./scripts/record_demos.py \
  --dataset_file ./datasets/my_demos.hdf5 \
  --num_demos 50 \
  openarm_bimanual_pick_and_place_exhaust_pipe \
  --teleop_device keyboard_bimanual
```

`--num_demos 0` records indefinitely, and `--num_success_steps` controls how many consecutive
success steps are required before an episode is saved.

---

## Data Generation and Dataset Conversion

The imitation-learning pipeline supports human demonstrations, Mimic generation from a handful of
source demonstrations, and conversion to LeRobot datasets.

### 1. Record Demonstrations

Use `scripts/record_demos.py` as shown above to produce an HDF5 dataset.

### 2. Annotate Demonstrations for Mimic

```bash
python ./scripts/annotate_demos.py \
  --mimic \
  --input_file ./datasets/my_demos.hdf5 \
  --output_file ./datasets/my_demos_annotated.hdf5 \
  openarm_bimanual_pick_and_place_exhaust_pipe
```

`--mimic` is required: the Arena builder only constructs a `ManagerBasedRLMimicEnv` in Mimic mode.
Add `--auto` to annotate subtasks automatically. The staged industrial tasks derive their Mimic
subtask boundaries from the stage list, so no hand-written `MimicEnvCfg` subclasses are needed.

### 3. Generate Mimic Data

```bash
python ./scripts/generate_dataset.py \
  --mimic \
  --generation_num_trials 100 \
  --input_file ./datasets/my_demos_annotated.hdf5 \
  --output_file ./datasets/generated.hdf5 \
  openarm_bimanual_pick_and_place_exhaust_pipe
```

### 4. Convert HDF5 to a LeRobot Dataset

Conversion configs for several robots are in `scripts/policy/config/` (for example
`OpenArmBimanual.yaml`). The config declares the dataset features, state/action layout, camera
names, and task description.

```bash
python ./scripts/policy/convert_hdf5_to_lerobot_dataset.py \
  --root_path ./datasets/my_demos \
  --config_yaml ./scripts/policy/config/OpenArmBimanual.yaml \
  --tgt_repo_id my-openarm-dataset
```

`scripts/policy/replay_papare_video_for_lerobot_dataset.py` replays dataset episodes and records
videos, and `scripts/policy/eval_policy.py` evaluates a policy from a YAML configuration.

### 5. Generate an OpenRob Locomanip LeRobot Dataset

The OpenRob script records every physics step through `OpenRobInterface` into a LeRobot dataset
(requiring `lerobot>=0.6.1`) with the three Arena cameras and bimanual state:

```bash
python ./scripts/env_data_generation/galileo_openrob_locomanip_pick_and_place.py \
  --num_episodes 50 \
  --repo_id my-openrob-locomanip
```

It rolls out a navigation → pick → navigation → bimanual drop episode and supports
configurable approach poses, goal-yaw randomization, retries, and optional video recording.

### 6. Train a LeRobot Policy on Modal

`scripts/train_lerobot_on_modal.py` defines a Modal app that installs LeRobot and runs
`lerobot-train` on an A100 for Groot training.

---

## Tools and Tests

### Tools

```bash
# List registered Isaac Lab Gym environments
python ./scripts/tools/list_envs.py

# List the 50 industrial tasks and check that each is registered
python ./scripts/tools/smoke_industrial.py --list

# Build, reset, and step one industrial task under Isaac Sim
python ./scripts/tools/smoke_industrial.py --steps 3 \
  openarm_bimanual_industrial_thread_nut_on_bolt

# Probe every industrial task in one Isaac Sim session (the positional is still required)
python ./scripts/tools/smoke_industrial.py --all \
  openarm_bimanual_industrial_thread_nut_on_bolt

# Evaluate the scripted oracle, with optional per-step tracing and frame capture
python ./scripts/tools/eval_oracle.py --episodes 5 --trace \
  openarm_bimanual_industrial_thread_nut_on_bolt
```

`smoke_industrial.py --arena` resolves the environment through Arena's own registry instead of the
OpenArm registry, which helps distinguish a repository regression from an upstream container issue.
`eval_oracle.py` re-evaluates the task's declared goals instead of trusting the environment's
`terminated` flag, so it can report which goal failed.

### Tests

Static tests can be run with pytest. The staged suites validate task data, kinematics, Mimic
segmentation, and reward/success agreement without launching Isaac Sim:

```bash
pytest source/openarm/openarm/tests/test_industrial_tasks.py
pytest source/openarm/openarm/tests/test_staged_kinematics.py
pytest source/openarm/openarm/tests/test_staged_mimic.py
pytest source/openarm/openarm/tests/test_staged_rewards.py
pytest source/openarm/openarm/tests/test_industrial_rl_cfg.py
```

In the Docker container, the `pytest` alias maps to `/isaac-sim/python.sh -m pytest`.

---

## How to Create New Tasks

### Creating a New Industrial Staged Task

Industrial tasks are data, not new modules. Add a `TaskSpec` and an `IndustrialTask` entry to the
appropriate family module in `arena_environments/industrial/` (`fastening.py`, `packing.py`,
`beverage.py`, `containers.py`, `tools.py`, or `inspection.py`), then add it to that module's
`TASKS` list. Registration, the CLI subcommand, the environment class, the oracle, the rewards,
and the Mimic segmentation are generated automatically. The registry validates the whole table on
import and reports every inconsistency at once.

Minimal shape:

```python
from openarm.arena_environments.industrial._base import IndustrialTask
from openarm.arena_tasks.staged import ObjectInRegion, Stage, Skill, TaskSpec

IndustrialTask(
    spec=TaskSpec(
        name="my_industrial_task",
        instruction="Pick the widget and place it in the bin.",
        stages=(
            Stage(skill=Skill.PICK, arm="right", target="widget", label="pick up the widget"),
            Stage(
                skill=Skill.PLACE,
                arm="right",
                target="widget",
                secondary="bin",
                label="place the widget in the bin",
            ).with_goal(
                ObjectInRegion(
                    "widget",
                    center=(0.5, 0.0, 0.4),
                    half_extents=(0.1, 0.1, 0.1),
                )
            ),
        ),
        object_names=("widget",),
        episode_length_s=30.0,
        success_label="widget in bin",
    ),
    objects={"widget": "my_widget", "bin": "openarm_sorting_bin"},
    primary_object="widget",
)
```

Then run it as `openarm_bimanual_industrial_my_industrial_task`.

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

from . import agents

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
python ./scripts/reinforcement_learning/rl_games/train.py --task Isaac-My-Task-OpenArm-v0 --headless
```

Or with SKRL:
```bash
python ./scripts/reinforcement_learning/skrl/train.py --task Isaac-My-Task-OpenArm-v0 --headless
```

The RSL-RL config is registered for the task, but this repository's `rsl_rl/train.py` targets Arena
environment subcommands rather than `--task` IDs.

Reference implementations:
- Simple: `unimanual/reach/` -- basic end-effector tracking with position/orientation rewards.
- Intermediate: `unimanual/lift/` -- multi-stage pick and lift with custom observations.
- Advanced: `unimanual/cabinet/` -- multi-stage grasp and pull with physics randomization.

### Creating a New Arena Task

Arena tasks are more modular. They define the MDP logic independently of the robot or scene layout.

**1. Create a task class** in `arena_tasks/`:

```python
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import TerminationTermCfg, RewardTermCfg, EventTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp_isaac_lab

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

class MyArenaTask(TaskBase):
    def __init__(self, target_object: Asset, episode_length_s: float | None = None):
        super().__init__(
            episode_length_s=episode_length_s,
            task_description=f"Do something with the {target_object.name}.",
        )
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

    def get_metrics(self):
        """Return list of MetricBase instances for evaluation."""
        return [SuccessRateMetric()]

    def get_viewer_cfg(self):
        """Camera viewpoint for visualization."""
        return ViewerCfg(eye=(-1.5, -1.5, 1.5), lookat=(0.0, 0.0, 0.5))

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        """Return MimicEnvCfg for Isaac Lab Mimic data generation."""
        return MyMimicEnvCfg(arm_mode=arm_mode)
```

`get_task_description()` returns the description LLM policies use; it is set through the
`task_description` constructor argument rather than a separate `get_prompt()` method.

**2. Add custom reward functions** in `arena_tasks/mdp/rewards.py`:

```python
def my_custom_reward(env, asset_cfg, target_pos):
    """Compute reward based on distance to target."""
    asset = env.scene[asset_cfg.name]
    distance = torch.norm(asset.data.root_pos_w - target_pos, dim=-1)
    return torch.exp(-distance)
```

To use Arena's built-in door/openable behaviour, import `OpenDoorTask` from
`isaaclab_arena.tasks.open_door_task`; there is no local `arena_tasks/open_door.py`.

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

**2. Register the environment** by importing the module in `arena_environments/cli.py` and adding
the class to `ExampleEnvironments`, or load it without editing the registry by passing
`--external_environment_class_path module_path:ClassName`.

**3. Run it:**

```bash
python ./scripts/policy_runner.py --policy-type zero_action openarm_bimanual_my_task
```

To use `--policy-type oracle`, the environment must also set
`isaaclab_arena_environment.oracle_policy`; the zero-action policy works without any policy wiring.

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
