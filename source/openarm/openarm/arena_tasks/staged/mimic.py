"""Mimic datagen configuration derived from a :class:`TaskSpec`.

Fifty hand-written ``MimicEnvCfg`` subclasses would be fifty places for a stage
list and its ``SubTaskConfig`` list to drift apart, and the drift would only
show up as a datagen run that splits an episode at the wrong step. The stages are
already declarative and already ordered, so the segmentation is read off them.

Two properties of the staged tasks make that more than a loop over stages.

**An arm stops working before the task does.** Most of these tasks are bimanual
for two stages and single-arm for the rest, so one arm's segments end while the
other arm still has stages to go. Every end effector needs a non-empty subtask
list, so the arm that stops early gets a final segment with
``subtask_term_signal=None``, which is exactly the semantics wanted: it runs from
its last signal to the end of the episode while that arm holds still.

**Only stages with a goal mark a boundary.** A ``MOVE`` or ``DOCK`` stage is a
means, not an end, and carries ``goal=None``. Emitting a signal for one would
produce a signal that the annotation tool can never observe go high, and
``annotate_demos.py`` treats a signal that never fires as a failed annotation.

The signals themselves are the goals, evaluated live. That is why the signal name
is derived from the stage index rather than chosen freely: the config and
:func:`staged_signals` in the mimic environment have to agree, and a shared
naming function is what makes them agree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.utils import configclass

from openarm.arena_tasks.staged.arms import ARMS, Arm
from openarm.arena_tasks.staged.stages import TaskSpec

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.common.arm_mode import ArmMode


def signal_name(index: int, arm: Arm) -> str:
    """Name of the annotation signal that closes stage *index* for *arm*.

    Shared by the config generator here and by the mimic environment's
    :meth:`~openarm.embodiments.openarm_bimanual.openarm_bimanual.OpenArmBimanualMimicEnv.get_subtask_term_signals`,
    which is the pair that has to stay in step.

    Args:
        index: Position of the stage in :attr:`TaskSpec.stages`.
        arm: The arm the stage resolved to.

    Returns:
        The signal name, unique across the task because the stage index is.
    """
    return f"stage_{index:02d}_{arm}_done"


def stage_object_ref(stage: Stage, fallback: str) -> str:
    """Name of the object a stage's subtask is expressed relative to.

    Mimic needs one frame per subtask to anchor on, and a stage's own target is
    that frame. Stages whose target is a fixture or is absent from the scene fall
    back to the task's primary object, which keeps ``object_ref`` pointing at
    something the datagen can actually read a pose for.

    Args:
        stage: The stage to find a reference for.
        fallback: Used when the stage names no usable object.

    Returns:
        A scene object name.
    """
    return stage.target or fallback


def active_arms(spec: TaskSpec, arm_mode: "ArmMode") -> tuple[Arm, ...]:
    """Arms that get subtask configs under *arm_mode*.

    Only the arms the embodiment is actually commanding can appear: the action
    manager drops the terms of an absent arm, so a config naming it would have
    the datagen try to command end effectors that do not exist.

    Args:
        spec: The resolved task spec.
        arm_mode: The embodiment's arm mode.

    Returns:
        The arms to generate configs for.
    """
    used = {stage.arm for stage in spec.stages}
    if arm_mode.value == "dual_arm":
        return ARMS
    if arm_mode.value == "left":
        return ("left",)
    if arm_mode.value == "right":
        return ("right",)
    # SINGLE_ARM keeps whichever arm the task's first stage resolved to; a task
    # that never names an arm falls back to the right, as the rest of the
    # OpenArm stack does.
    return (next(iter(used - {"right"}), "right"),)


def _segment(object_ref: str, signal: str | None, *, arm: Arm) -> SubTaskConfig:
    """Build one :class:`SubTaskConfig`, with the offsets a segment boundary needs.

    The final segment of an arm carries no signal and no offset: the episode ends
    there, so there is nothing to shift the boundary by and a random offset would
    only risk running past the end of the recording.

    Args:
        object_ref: Scene object the segment is anchored to.
        signal: The signal closing the segment, or ``None`` for the last one.
        arm: The arm the segment belongs to, for the description.

    Returns:
        The subtask config.
    """
    final = signal is None
    return SubTaskConfig(
        object_ref=object_ref,
        subtask_term_signal=signal,
        subtask_term_offset_range=(0, 0) if final else (10, 20),
        # The nearest-neighbour strategy picks the source segment whose object is
        # closest to the one being generated, which is what makes a generated
        # episode track the source instead of its own previous position.
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        # Noise is applied to the solved joint targets. Joint-space commands are
        # absolute, so a fixed amplitude in radians is a real, bounded
        # perturbation of the arm's configuration.
        action_noise=0.005,
        # A staged task's segments start from a known object state, so the
        # generator has less distance to bridge than a free-form pick and place.
        num_interpolation_steps=5,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
        description=f"{arm} arm segment",
    )


def build_subtask_configs(
    spec: TaskSpec, arm_mode: "ArmMode", fallback_object: str = ""
) -> dict[str, list[SubTaskConfig]]:
    """Segment every active arm's timeline into Mimic subtasks.

    Args:
        spec: A spec whose stage arms have already been resolved from the layout.
        arm_mode: The embodiment's arm mode.
        fallback_object: Object used when a stage names no usable target.

    Returns:
        Arm name to its ordered list of subtask configs. Every arm present has a
        non-empty list.

    Raises:
        ValueError: If a stage names an arm that *arm_mode* does not command.
    """
    configs: dict[str, list[SubTaskConfig]] = {arm: [] for arm in active_arms(spec, arm_mode)}
    known = set(configs)

    for index, stage in enumerate(spec.stages):
        if stage.arm not in known:
            raise ValueError(
                f"Stage {index} ({stage.skill.value}) resolved to arm '{stage.arm}', which arm mode "
                f"{arm_mode.value} does not command. Known arms: {sorted(known)}"
            )
        if stage.goal is None:
            # A means rather than an end: it belongs inside whichever segment the
            # next real stage closes.
            continue
        configs[stage.arm].append(
            _segment(
                stage_object_ref(stage, fallback_object),
                signal_name(index, stage.arm),
                arm=stage.arm,
            )
        )

    # Close each arm's timeline. The arm that works to the end of the task already
    # ends on its last goal; the one that stopped early gets its trailing hold,
    # which is what keeps its list non-empty and lets the other arm's later
    # segments run independently.
    for arm, segments in configs.items():
        object_ref = segments[-1].object_ref if segments else fallback_object
        segments.append(_segment(object_ref, None, arm=arm))
    return configs


def staged_signals(env, spec: TaskSpec, arm_mode: "ArmMode") -> dict[str, torch.Tensor]:
    """Evaluate every stage goal into a per-environment signal.

    Args:
        env: The running environment, used only for the device and the scene.
        spec: A spec whose stage arms have already been resolved.
        arm_mode: The embodiment's arm mode.

    Returns:
        Signal name to a boolean tensor of shape ``(num_envs,)``. Every signal
        named by :func:`build_subtask_configs` appears here, and nothing else.
    """
    from openarm.arena_tasks.staged.terminations import evaluate_goal

    active = active_arms(spec, arm_mode)
    signals: dict[str, torch.Tensor] = {}
    for index, stage in enumerate(spec.stages):
        if stage.goal is None or stage.arm not in active:
            continue
        satisfied = evaluate_goal(env, stage.goal.params())
        signals[signal_name(index, stage.arm)] = satisfied.to(
            device=env.device, dtype=torch.bool
        ).reshape(env.num_envs)
    return signals


@configclass
class StagedMimicEnvCfg(MimicEnvCfg):
    """Mimic datagen config for any staged industrial task.

    The segmentation is not stored here. It is rebuilt from the spec in
    ``__post_init__`` so that the config cannot describe a subtask split the task
    does not have.
    """

    task_spec: TaskSpec | None = None
    fallback_object: str = ""
    arm_mode: "ArmMode" = None

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "demo_src_staged_isaac_lab_task_D0"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = False
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_select_src_per_arm = False
        self.datagen_config.generation_transform_first_robot_pose = False
        # Interpolating from the last target pose is what carries an arm that is
        # holding still into the next segment without a jump.
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 1

        if self.task_spec is None:
            raise ValueError("StagedMimicEnvCfg needs the task spec to build its subtask configs.")
        if self.arm_mode is None:
            raise ValueError("StagedMimicEnvCfg needs the embodiment's arm mode.")
        self.subtask_configs = build_subtask_configs(
            self.task_spec, self.arm_mode, self.fallback_object
        )