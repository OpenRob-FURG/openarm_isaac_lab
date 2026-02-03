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

def main():
    """Script to run an IsaacLab Arena environment with a zero-action agent."""
    args_parser = get_isaaclab_arena_cli_parser()
    # We do this as the parser is shared between the example environment and policy runner
    args_cli, unknown = args_parser.parse_known_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):
        # Add policy-related arguments to the parser
        lerobot_policy_argument_group = args_parser.add_argument_group(title="LeRobot Policy Arguments")
        lerobot_policy_argument_group.add_argument(
            "--policy-repo-id",
            type=str,
            required=True,
            help="The repository ID or path to the pretrained LeRobot policy.",
        )
        lerobot_policy_argument_group.add_argument(
            "--policy-type",
            type=str,
            default="smolvla",
            help="The type of LeRobot policy to use.",
        )
        lerobot_policy_argument_group.add_argument(
            "--policy-task",
            type=str,
            required=True,
            help="The task to run the LeRobot policy on.",
        )
        lerobot_policy_argument_group.add_argument(
            "--policy-num-steps",
            type=int,
            default=1000,
            help="The number of steps to run the LeRobot policy.",
        )
        args_cli = args_parser.parse_args()
        # Build scene
        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()

        # NOTE(xinjieyao, 2025-09-29): General rule of thumb is to have as many non-standard python
        # library imports after app launcher as possible, otherwise they will likely stall the sim
        # app. Given current SimulationAppContext setup, use lazy import to handle policy-related
        # deps inside create_policy() function to bringup sim app.
        from openarm.policies.lerobot.lerobot_policy import LeRobotPolicy
        policy = LeRobotPolicy(
            repo_id=args_cli.policy_repo_id,
            policy_type=args_cli.policy_type,
            task=args_cli.policy_task,
        )
        num_steps = args_cli.policy_num_steps
        # NOTE(xinjieyao, 2025-10-07): lazy import to prevent app stalling caused by omni.kit
        from isaaclab_arena.metrics.metrics import compute_metrics

        for _ in tqdm.tqdm(range(num_steps)):
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

                if terminated.any() or truncated.any():
                    # only reset policy for those envs that are terminated or truncated
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)

        metrics = compute_metrics(env)
        print(f"Metrics: {metrics}")

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()