# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers
# SPDX-License-Identifier: Apache-2.0

"""
Dataset generation for galileo_openrob_locomanip_pick_and_place.

Rollout structure per episode:
    1. navigate    -> approach pose in front of the shelf
    2. pick        -> grasp the brown box
    3. bimanual_lift -> raise the box with both arms
    4. navigate    -> drive to the blue sorting bin
    5. bimanual_drop -> release the box over the bin

Every physics step is recorded via OpenRobInterface._step() into a
LeRobot 0.6.1 dataset with the three Arena cameras + bimanual state.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

# ---- Arena / IsaacLab bootstrap ----
# Keep these imports at the top; Arena requires app_launcher before
# importing Sim-dependent modules.
from isaaclab.app import AppLauncher
from openarm.arena_environments.cli import (
    add_example_environments_cli_args,
    get_arena_builder_from_cli,
)

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a LeRobot dataset from the OpenRob "
                    "locomanip pick-and-place task.")

    parser.add_argument("--task", type=str,
                        default="galileo_openrob_locomanip_pick_and_place")
    parser.add_argument("--object", type=str, default="brown_box")
    parser.add_argument("--embodiment", type=str, default="openrob")
    parser.add_argument("--teleop_device", type=str, default=None,
                        help="Leave None for scripted rollouts.")

    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--env_spacing", type=float, default=50.0)
    parser.add_argument("--solve_relations", type=bool, default=True)
    parser.add_argument("--mimic", type=bool, default=False)
    parser.add_argument("--disable_fabric", type=bool, default=False)

    parser.add_argument("--max_skill_retries", type=int, default=2,
                        help="Retry the whole episode if a skill returns False.")

    # Dataset
    parser.add_argument("--repo_id", type=str,
                        default="you/openrob_locomanip_pick_and_place")
    parser.add_argument("--dataset_root", type=str, default="datasets/openrob")
    parser.add_argument("--use_videos", action="store_true", default=True)
    parser.add_argument("--no_videos", dest="use_videos",
                        action="store_false")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete an existing dataset dir before writing.")

    # Domain randomization
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize_goal_yaw", action="store_true",
                        default=True)
    parser.add_argument("--yaw_range_deg", type=float, default=15.0)

    # Goal poses (world frame)
    parser.add_argument("--pick_approach_xy", type=float, nargs=2,
                        default=[0.1, 0.0],
                        help="Base xy at the shelf, in front of the box.")
    parser.add_argument("--bin_approach_xy", type=float, nargs=2,
                        default=[-0.10, -1.10],
                        help="Base xy in front of the blue bin.")
    parser.add_argument("--bin_drop_offset_xy", type=float, nargs=2,
                        default=[0.0, -0.15],
                        help="Offset from base to drop point over the bin.")
    parser.add_argument("--settle_steps", type=int, default=30,
                        help="Extra idle steps after bimanual_drop before "
                         "checking the success termination term.")
    # Pass-through so AppLauncher sees --headless, --device, etc.
    AppLauncher.add_app_launcher_args(parser)
    #add_example_environments_cli_args(parser)
    return parser.parse_args()


ARGS_CLI = _parse_args()

# Launch Isaac Sim *before* any isaaclab imports
app_launcher = AppLauncher(ARGS_CLI)
simulation_app = app_launcher.app
from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.gen.omap")  # Substitua ou adicione o sub-módulo necessário

# ---- Post-launch imports ----
import gymnasium as gym
import torch

import isaaclab_arena.environments.arena_env_builder as isaaclab_arena_gym
from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.environments.isaaclab_arena_environment import (
    IsaacLabArenaEnvironment,
)
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import (
    G1LocomanipPickAndPlaceTask,
)
from isaaclab_arena.utils.pose import Pose, PoseRange
from isaaclab_arena_environments.example_environment_base import (
    ExampleEnvironmentBase,
)

from openarm.embodiments.openarm_bimanual.openrob_interface import OpenRobInterface



# ----------------------------------------------------------------------
# 1. Environment builder (copy of your ExampleEnvironmentBase subclass)
# ----------------------------------------------------------------------
def build_env_cfg(args_cli: argparse.Namespace):
    """
    Replicate GalileoOpenRobLocomanipPickAndPlaceEnvironment.get_env
    without going through the registry, so we can add domain
    randomization hooks around it.
    """
    # Register your embodiment and retargeters
    import openarm.embodiments.openarm_bimanual.openrob  # noqa: F401
    import openarm.arena_retargeters.openrob              # noqa: F401

    registry = AssetRegistry()

    background = registry.get_asset_by_name("galileo_locomanip")()
    pick_up_object = registry.get_asset_by_name(args_cli.object)(scale=(0.4, 0.4, 0.4))
    blue_sorting_bin = registry.get_asset_by_name("blue_sorting_bin")()
    embodiment = registry.get_asset_by_name(args_cli.embodiment)(
        enable_cameras=args_cli.use_videos)

    teleop_device = None
    if args_cli.teleop_device is not None:
        from isaaclab_arena.assets.register import DeviceRegistry
        teleop_device = DeviceRegistry().get_device_by_name(
            args_cli.teleop_device)()

    # --- object pose randomization ---
    XY_RANGE_M = 0.025
    pick_up_object.set_initial_pose(
        PoseRange(
            position_xyz_min=(0.5785 - XY_RANGE_M, 0.18 - XY_RANGE_M, 0.0707),
            position_xyz_max=(0.5785 + XY_RANGE_M, 0.18 + XY_RANGE_M, 0.0707),
            rpy_min=(math.pi, 0.0, math.pi),
            rpy_max=(math.pi, 0.0, math.pi),
        )
    )

    blue_sorting_bin.set_initial_pose(
        Pose(
            position_xyz=(-0.2450, -1.6272, -0.2641),
            rotation_xyzw=(0.0, 0.0, 1.0, 0.0),
        )
    )

    # --- robot spawn pose randomization ---
    spawn_yaw = 0.0
    if args_cli.randomize_goal_yaw:
        spawn_yaw = math.radians(
            random.uniform(-args_cli.yaw_range_deg,
                           args_cli.yaw_range_deg))
    embodiment.set_initial_pose(
        Pose(
            position_xyz=(0.0, 0.18, -0.4),
            rotation_xyzw=(
                0.0, 0.0, math.sin(spawn_yaw / 2), math.cos(spawn_yaw / 2)
            ),
        )
    )

    scene = Scene(assets=[background, pick_up_object, blue_sorting_bin])
    env_cfg = IsaacLabArenaEnvironment(
        name=args_cli.task,
        embodiment=embodiment,
        scene=scene,
        task=G1LocomanipPickAndPlaceTask(
            pick_up_object,
            blue_sorting_bin,
            background,
            episode_length_s=300.0,
            task_description=(
                "Pick up the brown box from the shelf, and place it into the "
                "blue bin on the table located at the right of the shelf."
            ),
        ),
        teleop_device=teleop_device,
    )
    return env_cfg


SUCCESS_TERM_CANDIDATES = (
    "success",
    "task_success",
    "object_in_bin",
    "object_placed",
    "goal_reached",
)

# Add these helpers
def _env_id(iface: OpenRobInterface) -> int:
    """The interface drives env 0 unless told otherwise."""
    return 0


def _get_termination_flags(iface: OpenRobInterface) -> dict:
    """
    Snapshot the env's termination state immediately after the last step.

    Returns
    -------
    dict with:
        'any'      : bool   -- any termination term fired
        'terms'    : dict[str, bool] -- per-term flags, if available
        'success'  : bool   -- heuristic success flag
    """
    env = iface.env
    env_id = _env_id(iface)
    mgr = getattr(env, "termination_manager", None)
    if mgr is None:
        return {"any": False, "terms": {}, "success": False}

    any_flag = bool(mgr.terminated[env_id].item()) if hasattr(
        mgr, "terminated") else False

    # Per-term flags — the manager keeps active term names
    terms: dict[str, bool] = {}
    try:
        for name in mgr.active_terms:
            val = mgr.get_term(name)
            terms[name] = bool(val[env_id].item())
    except Exception:
        # Older Arena/IsaacLab builds may not expose active_terms;
        # treat the aggregate flag as the only signal.
        pass

    # Success heuristic
    success = False
    for candidate in SUCCESS_TERM_CANDIDATES:
        if candidate in terms:
            success = terms[candidate]
            break
    else:
        # Fallback: terminated for any reason other than timeout
        timeout = bool(
            env.episode_length_buf[env_id].item()
            >= env.max_episode_length
        ) if hasattr(env, "episode_length_buf") else False
        success = any_flag and not timeout

    return {"any": any_flag, "terms": terms, "success": success}

# ----------------------------------------------------------------------
# 2. Rollout: scripted skill sequence
# ----------------------------------------------------------------------
def run_episode(iface: OpenRobInterface, args_cli, episode_idx: int) -> dict:
    """
    Execute one scripted rollout, recording every step.

    Returns a dict with:
        'ok'         : bool -- every skill completed without raising
        'success'    : bool -- the env's success termination term fired
        'termination': dict -- the raw termination snapshot (for logging)
    """
    result = {"ok": False, "success": False, "termination": {}}

    # ---- 1. Drive to the shelf ----
    iface._current_task = "navigate_to_shelf"
    reached = iface.navigate(
        goal_xy=args_cli.pick_approach_xy,
        goal_yaw=0.0,
        pos_tol=0.05,
        yaw_tol=0.05,
        max_iters=400,
    )
    if not reached:
        print(f"[ep {episode_idx}] Failed to reach shelf.")
        return result

    # ---- 2. Pick box ----
    iface._current_task = "pick_object"
    try:
        iface.pick(
            obj_name="brown_box",
            arm="right",
            gripper_quat=(0.0, 0.0, 0.0, 1.0)
        )
    except Exception as exc:
        print(f"[ep {episode_idx}] pick_object failed: {exc}")
        return result

    '''# ---- 2. Bimanual lift ----
    iface._current_task = "bimanual_lift"
    try:
        iface.bimanual_lift(
            obj_name="brown_box",
            lift_height=0.15,
            approach_height=0.20,
            lateral_offset=0.14,
            steps=30,
        )
    except Exception as exc:
        print(f"[ep {episode_idx}] bimanual_lift failed: {exc}")
        return result'''

    # ---- 3. Drive to the bin ----
    iface._current_task = "navigate_to_bin"
    reached = iface.navigate(
        goal_xy=args_cli.bin_approach_xy,
        goal_yaw=math.pi / 2,
        pos_tol=0.06,
        yaw_tol=0.08,
        max_iters=600,
    )
    if not reached:
        print(f"[ep {episode_idx}] Failed to reach bin.")
        return result

    # ---- 4. Move both arms over the bin and drop ----
    iface._current_task = "bimanual_drop"
    try:
        iface._move_arms_to(
            iface._left_target.pos
            + torch.tensor([0.05, 0.15, 0.05], device=iface.device),
            iface._right_target.pos
            + torch.tensor([0.05, -0.15, 0.05], device=iface.device),
            iface._left_target.quat,
            left_grip=1.0, right_grip=1.0, steps=40,
        )
        iface.bimanual_drop(
            release_height=0.06,
            release_steps=18,
            retreat_height=0.20,
            retreat_steps=25,
        )
    except Exception as exc:
        print(f"[ep {episode_idx}] bimanual_drop failed: {exc}")
        return result

    # ---- 5. Allow physics to settle, then check termination ----
    # Some success terms fire only after the object has come to rest in
    # the bin, so we give the sim a few extra ticks beyond `bimanual_drop`.
    iface._current_task = "settle"
    iface.hold(steps=args_cli.settle_steps)

    result["ok"] = True
    result["termination"] = _get_termination_flags(iface)
    result["success"] = result["termination"]["success"]
    return result


# ----------------------------------------------------------------------
# 3. Main loop
# ----------------------------------------------------------------------
def main():
    random.seed(ARGS_CLI.seed)
    torch.manual_seed(ARGS_CLI.seed)

    # ---- dataset directory ----
    dataset_root = Path(ARGS_CLI.dataset_root)
    repo_path = dataset_root / ARGS_CLI.repo_id
    if ARGS_CLI.overwrite and repo_path.exists():
        import shutil
        shutil.rmtree(repo_path)
    #dataset_root.mkdir(parents=True, exist_ok=True)

    # ---- create the environment ----
    env_cfg = build_env_cfg(ARGS_CLI)
    env = isaaclab_arena_gym.ArenaEnvBuilder(env_cfg, args=ARGS_CLI).make_registered()

    iface = OpenRobInterface(env.unwrapped, device=ARGS_CLI.device or "cuda:0")

    # ---- dataset ----
    iface.start_recording(
        repo_id=ARGS_CLI.repo_id,
        root=str(dataset_root),
        use_videos=ARGS_CLI.use_videos,
        robot_type="openrob",
    )

    episodes_saved = 0
    attempts = 0
    max_attempts = ARGS_CLI.num_episodes * (1 + ARGS_CLI.max_skill_retries)

    try:
        while episodes_saved < ARGS_CLI.num_episodes and attempts < max_attempts:
            attempts += 1
            print(f"\n=== Attempt {attempts} | "
                f"episode {episodes_saved} / {ARGS_CLI.num_episodes} ===")

            iface.reset()
            iface.reset_arm_targets_to_current()

            l_ee_pos, l_ee_quat = iface._ee_pose("left")
            r_ee_pos, r_ee_quat = iface._ee_pose("right")
            from openarm.embodiments.openarm_bimanual.openrob_interface import ArmTarget
            iface._left_target = ArmTarget(
                l_ee_pos[0].clone(), l_ee_quat[0].clone(), grip=0.0)
            iface._right_target = ArmTarget(
                r_ee_pos[0].clone(), r_ee_quat[0].clone(), grip=0.0)

            result = run_episode(iface, ARGS_CLI, episodes_saved)

            # ---- success gate ----
            term = result["termination"]
            term_str = ", ".join(f"{k}={v}" for k, v in term.get("terms", {}).items()
                                if v) or "none"
            print(f"  termination: any={term.get('any')} success={result['success']} "
                f"fired=[{term_str}]")

            if result["ok"] and result["success"]:
                iface.save_episode(
                    task="locomanip_pick_and_place_bimanual_drop")
                episodes_saved += 1
                print(f"[OK] Saved episode {episodes_saved - 1}.")
            else:
                reason = (
                    "skill_failure" if not result["ok"]
                    else "no_success_signal"
                )
                print(f"[SKIP] Attempt {attempts} discarded ({reason}).")
                iface.clear_episode_buffer()

    except KeyboardInterrupt:
        print("\nInterrupted. Finalizing dataset with what we have.")

    except BaseException as e:
        raise e

    #finally:
    #    iface.stop_recording()
    #    env.close()
    #    simulation_app.close()

    print(f"\nDone. {episodes_saved} episode(s) saved to "
          f"{dataset_root / ARGS_CLI.repo_id}")


if __name__ == "__main__":
    main()