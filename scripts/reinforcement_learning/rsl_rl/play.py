# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import openarm.tasks  # noqa: F401

from openarm.tasks.manager_based.openarm_manipulation.unimanual.lift.mdp.terminations import object_is_lifted

# PLACEHOLDER: Extension template (do not remove this comment)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    
    def get_success_function(task_name):
        if "Lift" in task_name:
            return lambda env: object_is_lifted(env, minimal_height=0.05)

        elif "Reach" in task_name and ("Bi" in task_name or "Bimanual" in task_name):

            episode_success = {}

            def success_fn(env):
                env_id = id(env)
                
                if env_id not in episode_success:
                    episode_success[env_id] = None

                robot = env.unwrapped.scene["robot"]
                body_names = robot.data.body_names

                left_ee_idx = body_names.index("openarm_left_ee_tcp")
                right_ee_idx = body_names.index("openarm_right_ee_tcp")

                left_ee_pos = robot.data.body_pos_w[:, left_ee_idx]
                right_ee_pos = robot.data.body_pos_w[:, right_ee_idx]
                
                env_origins = env.scene.env_origins

                left_ee_pos_local = left_ee_pos - env_origins
                right_ee_pos_local = right_ee_pos - env_origins

                left_target = env.unwrapped.command_manager.get_command("left_ee_pose")[:, :3]
                right_target = env.unwrapped.command_manager.get_command("right_ee_pose")[:, :3]

                left_error = torch.norm(left_ee_pos_local - left_target, dim=1)
                right_error = torch.norm(right_ee_pos_local - right_target, dim=1)

                in_range = (left_error < 0.15) & (right_error < 0.15)

                if episode_success[env_id] is None:
                    episode_success[env_id] = torch.zeros_like(in_range, dtype=torch.bool)

                episode_success[env_id] = episode_success[env_id] | in_range
                
                return episode_success[env_id]

            return success_fn

        elif "Reach" in task_name and "Bi" not in task_name:
            episode_success = {}
            
            def reach_unimanual_success_fn(env):
                env_id = id(env)
                
                if env_id not in episode_success:
                    episode_success[env_id] = None
                
                robot = env.unwrapped.scene["robot"]
                body_names = robot.data.body_names

                ee_idx = body_names.index("openarm_ee_tcp")

                ee_pos = robot.data.body_pos_w[:, ee_idx]
                env_origins = env.scene.env_origins
                ee_pos_local = ee_pos - env_origins

                target = env.unwrapped.command_manager.get_command("ee_pose")[:, :3]

                error = torch.norm(ee_pos_local - target, dim=1)
                success_now = error < 0.1

                if episode_success[env_id] is None:
                    episode_success[env_id] = torch.zeros_like(success_now, dtype=torch.bool)

                episode_success[env_id] |= success_now

                return episode_success[env_id]
            return reach_unimanual_success_fn

        elif "Drawer" in task_name:
            episode_success = {}
            
            def drawer_success_fn(env):
                env_id = id(env)
                
                if env_id not in episode_success:
                    episode_success[env_id] = None
                
                cabinet = env.unwrapped.scene["cabinet"]

                joint_idx = cabinet.data.joint_names.index("drawer_bottom_joint")
                joint_pos = cabinet.data.joint_pos[:, joint_idx]

                success_now = joint_pos > 0.35

                if episode_success[env_id] is None:
                    episode_success[env_id] = torch.zeros_like(success_now, dtype=torch.bool)

                episode_success[env_id] |= success_now

                return episode_success[env_id]          
            return drawer_success_fn
        else:
            raise ValueError(f"Unknown task: {task_name}")


    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    success_fn = get_success_function(args_cli.task)

    # reset environment
    obs = env.get_observations()
    timestep = 0
    successful_episodes = 0
    total_episodes = 0
    max_episodes = 250  # pode ajustar
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)

        success = success_fn(env.unwrapped)

        if not torch.is_tensor(success):
            success = torch.tensor(success, device=env.unwrapped.device)

        done_envs = dones.nonzero(as_tuple=True)[0]

        if len(done_envs) > 0:
            successful_episodes += success[done_envs].sum().item()
            total_episodes += len(done_envs)

            print(f"Partial Success Rate: {successful_episodes}/{total_episodes}")

        if total_episodes >= max_episodes:
            break

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if total_episodes > 0:
        success_rate = successful_episodes / total_episodes
        print("\n==============================")
        print(f"Evaluation over {total_episodes} episodes")
        print(f"Success Rate: {success_rate * 100:.2f}%")
        print("==============================\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
