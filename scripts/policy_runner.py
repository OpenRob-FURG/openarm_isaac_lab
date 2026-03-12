# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import random
import torch
import tqdm

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.example_environments.cli import get_arena_builder_from_cli
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena.examples.example_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli


def main():
    """Script to run an IsaacLab Arena environment with a zero-action agent."""
    args_parser = get_isaaclab_arena_cli_parser()
    args_parser.add_argument(
        "--policy-repo-id",
        type=str,
        required=True,
        help="The repository ID or path to the pretrained LeRobot policy.",
    )
    args_parser.add_argument(
        "--policy-type",
        type=str,
        default="smolvla",
        help="The type of LeRobot policy to use.",
    )
    args_parser.add_argument(
        "--policy-task",
        type=str,
        required=True,
        help="The task to run the LeRobot policy on.",
    )
    args_parser.add_argument(
        "--policy-num-steps",
        type=int,
        default=10000,
        help="The number of steps to run the LeRobot policy.",
    )
    add_example_environments_cli_args(args_parser)

    # We do this as the parser is shared between the example environment and policy runner
    args_cli, unknown = args_parser.parse_known_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):
        from openarm.policies.lerobot.lerobot_policy import LeRobotPolicy
        from openarm.policies.oracle.open_microwave import OracleOpenMicrowavePolicy
        from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.managers import DatasetExportMode
        from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
        from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
        from isaaclab.utils import configclass

        class PreStepFlatCameraObservationsRecorder(RecorderTerm):
            """Recorder term that records the camera observations in each step."""

            def record_pre_step(self):
                return "camera_obs", self._env.obs_buf["camera_obs"]


        @configclass
        class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
            """Configuration for the camera observation recorder term."""

            class_type = PreStepFlatCameraObservationsRecorder


        @configclass
        class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
            """Add the camera observation recorder term."""

            record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()

        
        # Add policy-related arguments to the parser
        #args_parser.add_argument_group(title="LeRobot Policy Arguments")
        args_cli = args_parser.parse_args()
        # Build scene
        arena_builder = get_arena_builder_from_cli(args_cli)
        if args_cli.policy_type == "scripted":
            policy = arena_builder.arena_env.oracle_policy
        elif args_cli.policy_type == "llm":
            policy = arena_builder.arena_env.llm_policy
        elif args_cli.policy_type == "zero_action":
            policy = ZeroActionPolicy()
        else:
            policy = LeRobotPolicy(
                repo_id=args_cli.policy_repo_id,
                policy_type=args_cli.policy_type,
                task=args_cli.policy_task,
            )
        env_name, env_cfg = arena_builder.build_registered()
        env_cfg.episode_length_s = 10.0
        env_cfg.recorders = ArenaEnvRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = "./datasets"
        env_cfg.recorders.dataset_filename = "policy_generated_dataset"
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        env = arena_builder.make_registered(env_cfg=env_cfg)
        max_episodes = 100
        episode_counter = 0
        success_counter = 0

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()
        for _ in range(50):
            obs, _, terminated, truncated, _ = env.step(torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device))
        #for _ in range(10):
        #    obs, _, terminated, truncated, _ = env.step(torch.as_tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], device=env.device))

        # NOTE(xinjieyao, 2025-09-29): General rule of thumb is to have as many non-standard python
        # library imports after app launcher as possible, otherwise they will likely stall the sim
        # app. Given current SimulationAppContext setup, use lazy import to handle policy-related
        # deps inside create_policy() function to bringup sim app.
        num_steps = args_cli.policy_num_steps
        success = False
        success_term = env_cfg.terminations.success
        success_vector = [0.0,]*max_episodes
        # NOTE(xinjieyao, 2025-10-07): lazy import to prevent app stalling caused by omni.kit
        from isaaclab_arena.metrics.metrics import compute_metrics
        #counter = success_counter if args_cli.policy_type == "scripted" else episode_counter
        while episode_counter < max_episodes:
            with torch.inference_mode(mode=(args_cli.policy_type not in ('llm', 'oracle'))):
                actions = policy.get_action(env, obs)
                #actions[..., 6] *= -1.0
                #print(actions)
                #actions[..., 6] = -1.0 if actions[0, 6] < -0.9 else 1.0
                #print(success_term.func(env, **success_term.params))
                
                obs, _, terminated, truncated, info = env.step(actions)
                
                if terminated.any() or truncated.any():
                    print(info)
                    success_vector[episode_counter] = info["log"]["Episode_Termination/success"]
                    env.recorder_manager.record_pre_reset([0], force_export_or_skip=False)
                    env.recorder_manager.set_success_to_episodes(
                        [0], torch.tensor([[success]], dtype=torch.bool, device=env.device)
                    )
                    env.recorder_manager.export_episodes([0])
                    # only reset policy for those envs that are terminated or truncated
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
                    episode_counter += 1
                    obs, info = env.reset()
                    for _ in range(50):
                        obs, _, terminated, truncated, _ = env.step(torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device))
                    success = False

        #metrics = compute_metrics(env)
        #print(f"Metrics: {metrics}")
        print(f"Success rate: {np.mean(success_vector)}")
        print(success_vector)

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()