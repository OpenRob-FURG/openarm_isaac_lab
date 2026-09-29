"""Retry-capable scripted oracle for staged tasks.

The policy walks a :class:`~openarm.arena_tasks.staged.stages.TaskSpec` stage
list, running one
:class:`~openarm.policies.oracle.skills.SkillController` per stage per
sub-environment. When a controller times out, or finishes without producing its
intended effect, the policy does not just move on: it retreats, opens the
gripper, and retries the same stage up to the stage's ``max_attempts``. A global
attempt budget stops a pathological task from spending the whole episode in
retries.

The policy is deterministic given the environment, which is what makes the
generated demonstrations reproducible.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import torch

from isaaclab_arena.policy.policy_base import PolicyBase

from openarm.arena_tasks.staged.stages import Skill, Stage, TaskSpec
from openarm.policies.oracle.skills import (
    GRIPPER_OPEN,
    SkillController,
    build_action,
    build_controller,
    object_pose,
)

#: Total number of stage retries allowed across one episode. Individual stages
#: may retry more than this on their own, but not all of them at once.
DEFAULT_RETRY_BUDGET = 8

#: Number of recovery steps inserted between a failure and a retry.
RECOVERY_STEPS = 12


@dataclass
class OracleState:
    """Bookkeeping for one sub-environment's currently running stage.

    Attributes:
        index: Index of the running stage.
        attempt: Zero-based attempt number for the running stage.
        recovering: Whether recovery motion is currently running.
        recovery_elapsed: Steps spent in recovery.
        complete: Whether the whole task has finished.
        stage_results: Per-stage outcome, one of ``"ok"``, ``"retried"`` or
            ``"abandoned"``.
        retries_used: Total retries consumed so far.
    """

    index: int = 0
    attempt: int = 0
    recovering: bool = False
    recovery_elapsed: int = 0
    complete: bool = False
    stage_results: list[str] = field(default_factory=list)
    retries_used: int = 0
    settle: int = 0
    controller_needs_reset: bool = False

    def summary(self) -> str:
        """Return a compact human-readable progress string."""
        return f"stage {self.index} attempt {self.attempt} results={self.stage_results}"


@dataclass
class StagedOraclePolicyCfg:
    """Configuration for :class:`StagedOraclePolicy`.

    Attributes:
        task: Key of the task in
            :data:`openarm.arena_environments.industrial.INDUSTRIAL_TASKS` to
            demonstrate. Required.
        retry_budget: Maximum number of retries across one episode.
    """

    task: str
    retry_budget: int = DEFAULT_RETRY_BUDGET


class StagedOraclePolicy(PolicyBase):
    """Scripted, retry-capable demonstration generator for staged tasks."""

    #: Enables ``PolicyBase.from_dict``.
    config_class = StagedOraclePolicyCfg

    def __init__(self, config: StagedOraclePolicyCfg):
        """Create the policy for a task specification.

        Args:
            config: The policy configuration.

        Raises:
            ValueError: If the named task is unknown or has no stages.
        """
        super().__init__(config)
        from openarm.arena_environments.industrial import INDUSTRIAL_TASKS

        if config.task not in INDUSTRIAL_TASKS:
            raise ValueError(f"Unknown industrial task '{config.task}'. Known: {sorted(INDUSTRIAL_TASKS)}")
        self.task = INDUSTRIAL_TASKS[config.task]
        self.spec: TaskSpec = self.task.spec
        self.retry_budget = config.retry_budget
        self.states: list[OracleState] = []
        self.controllers: list[SkillController] = []
        self.task_description: str | None = None

    # ------------------------------------------------------------------
    # PolicyBase interface
    # ------------------------------------------------------------------

    def get_action(self, env, observation) -> torch.Tensor:
        """Return the action for every sub-environment.

        Args:
            env: The environment being stepped.
            observation: The observation dictionary, unused because the oracle
                reads the environment's own pose buffers directly.

        Returns:
            An action tensor of shape ``(num_envs, action_dim)``.
        """
        del observation
        num_envs = int(env.unwrapped.num_envs)
        self._ensure_size(num_envs, env)
        device = env.unwrapped.device
        actions = []
        for env_id in range(num_envs):
            if self.states[env_id].complete:
                actions.append(self._hold(env, env_id))
                continue
            action = self._recover(env, env_id) if self.states[env_id].recovering else self._advance(env, env_id)
            actions.append(action)
        return torch.stack(actions).to(device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Restart the demonstration for the given sub-environments.

        Args:
            env_ids: Indices to reset. ``None`` resets all of them.
        """
        indices = range(len(self.states)) if env_ids is None else [int(i) for i in env_ids]
        for env_id in indices:
            if env_id >= len(self.states):
                continue
            self.states[env_id] = OracleState()
            self.controllers[env_id] = build_controller(self.spec.stages[0], env_id)
            self.states[env_id].controller_needs_reset = True

    def set_task_description(self, task_description: str | None) -> str:
        """Record the language instruction the environment sampled.

        The oracle ignores the instruction, since it is scripted, but the
        runner sets it on every policy and logging it is how a demonstration is
        traced back to the instruction the policy was evaluated under.

        Args:
            task_description: The sampled instruction.

        Returns:
            The instruction, or an empty string if none was given.
        """
        self.task_description = task_description
        return task_description or ""

    def has_length(self) -> bool:
        """The oracle is not dataset-driven, so it has no fixed length."""
        return False

    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Register the oracle's command line options.

        Args:
            parser: The policy parser.

        Returns:
            The same parser, for chaining.
        """
        parser.add_argument("--task", type=str, required=True, help="Key of the industrial task to demonstrate.")
        parser.add_argument(
            "--retry-budget",
            type=int,
            default=DEFAULT_RETRY_BUDGET,
            help="Maximum number of stage retries across an episode.",
        )
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "StagedOraclePolicy":
        """Build the policy from parsed command line arguments.

        Args:
            args: Parsed arguments, including ``task`` and ``retry_budget``.

        Returns:
            The configured policy.
        """
        return StagedOraclePolicy(
            StagedOraclePolicyCfg(task=args.task, retry_budget=getattr(args, "retry_budget", DEFAULT_RETRY_BUDGET))
        )

    # ------------------------------------------------------------------
    # Stage machine
    # ------------------------------------------------------------------

    def _ensure_size(self, num_envs: int, env) -> None:
        """Allocate controller state for any sub-environments not yet covered."""
        while len(self.states) < num_envs:
            env_id = len(self.states)
            controller = build_controller(self.spec.stages[0], env_id)
            controller.reset(env, env_id)
            self.controllers.append(controller)
            self.states.append(OracleState())
        del self.controllers[num_envs:]

    def _advance(self, env, env_id: int) -> torch.Tensor:
        """Run one step of the current stage for one sub-environment."""
        state = self.states[env_id]
        controller = self.controllers[env_id]
        if state.controller_needs_reset:
            # PolicyBase.reset does not receive the environment. Capture the
            # new episode's object pose on its first action so verification
            # compares against this reset's initial state.
            controller.reset(env, env_id)
            state.controller_needs_reset = False
        stage = self.spec.stages[state.index]
        action = controller.step(env, env_id)

        if controller.done(env, env_id):
            # The controller's motion has finished, but the world has not. Hold
            # still for settle_steps so the goal is checked against a settled
            # simulation rather than mid-flight.
            if state.settle < stage.settle_steps:
                state.settle += 1
                return action
            if controller.verify(env, env_id):
                self._finish_stage(env, env_id, "ok")
            else:
                self._fail_stage(env, env_id, "verify failed")
            return action

        if controller.failed(env, env_id):
            self._fail_stage(env, env_id, "step budget exhausted")
        return action

    def _finish_stage(self, env, env_id: int, outcome: str) -> None:
        """Advance to the next stage, or dock when the task is finished."""
        state = self.states[env_id]
        state.stage_results.append(outcome)
        state.index += 1
        state.attempt = 0
        state.settle = 0
        if state.index >= len(self.spec.stages):
            state.complete = True
            dock = build_controller(Stage(skill=Skill.DOCK, anchor=(0.0, 0.0, 0.35)), env_id)
            dock.reset(env, env_id)
            self.controllers[env_id] = dock
            return
        controller = build_controller(self.spec.stages[state.index], env_id)
        controller.reset(env, env_id)
        self.controllers[env_id] = controller

    def _fail_stage(self, env, env_id: int, reason: str) -> None:
        """Either schedule a retry or abandon the stage."""
        state = self.states[env_id]
        stage = self.spec.stages[state.index]
        state.retries_used += 1
        if state.attempt + 1 < stage.max_attempts and state.retries_used < self.retry_budget:
            state.attempt += 1
            state.stage_results.append("retried")
            state.recovering = True
            state.recovery_elapsed = 0
            state.settle = 0
            self.last_failure = reason
            return
        self.last_failure = reason
        self._finish_stage(env, env_id, "abandoned")

    def _recover(self, env, env_id: int) -> torch.Tensor:
        """Retreat and open the gripper before retrying a failed stage."""
        state = self.states[env_id]
        controller = self.controllers[env_id]
        arm = controller.stage.arm
        current = _eef(env, arm, env_id)
        safe = current.clone()
        safe[2, 3] = max(float(current[2, 3]) + 0.15, 0.25)
        action = build_action(env, {arm: safe}, {arm: GRIPPER_OPEN}, env_id)
        state.recovery_elapsed += 1
        if state.recovery_elapsed >= RECOVERY_STEPS:
            state.recovering = False
            retry = build_controller(self.spec.stages[state.index], env_id)
            retry.reset(env, env_id)
            self.controllers[env_id] = retry
        return action

    def _hold(self, env, env_id: int) -> torch.Tensor:
        """Keep the end effectors still once the task is done."""
        return build_action(env, {}, {"left": GRIPPER_OPEN, "right": GRIPPER_OPEN}, env_id)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def progress(self, env_id: int = 0) -> float:
        """Return the fraction of stages completed for a sub-environment."""
        return self.states[env_id].index / max(1, len(self.spec.stages))

    def describe(self, env_id: int = 0) -> str:
        """Return a human-readable description of the policy state."""
        return f"{self.spec.name}: {self.states[env_id].summary()}"

    def objects_touched(self) -> list[str]:
        """Return the scene names of every object the policy manipulates."""
        names: list[str] = []
        for stage in self.spec.stages:
            for name in (stage.target, stage.secondary):
                if name and name not in names:
                    names.append(name)
        return names

    def final_object_positions(self, env, env_id: int = 0) -> dict[str, torch.Tensor]:
        """Return the world position of every object the policy touched."""
        return {name: object_pose(env, name, env_id)[:3, 3].clone() for name in self.objects_touched()}


def _eef(env, arm: str, env_id: int) -> torch.Tensor:
    from openarm.policies.oracle.skills import eef_pose

    return eef_pose(env, arm, env_id)
