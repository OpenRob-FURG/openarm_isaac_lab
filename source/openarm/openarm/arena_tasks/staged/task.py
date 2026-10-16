"""A generic Arena task driven by a declarative :class:`TaskSpec`.

Every industrial task in this package is a :class:`TaskSpec`. This class turns
that spec into the six things the Arena environment builder needs: a success
termination, a failure termination, a dense reward manager, an observation group
that includes the camera, a privileged critic observation group, and a set of
metrics.
"""

from __future__ import annotations

import random
from typing import Any

from isaaclab.envs import mdp as mdp_rewards
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from openarm.arena_dr import terms as dr_terms
from openarm.arena_tasks.staged import observations as obs_terms
from openarm.arena_tasks.staged import rewards as rew
from openarm.arena_tasks.staged import terminations as pred
from openarm.arena_tasks.staged.stages import TaskSpec

CAMERA_NAME = "top_camera"

#: Distance at which the approach reward is roughly halved [m]. Sized to the
#: OpenArm's 0.82 m reach rather than to a goal tolerance: this term is about
#: covering ground towards the object, not about the last few centimetres.
APPROACH_STD_M = 0.15


@configclass
class StagedTerminationsCfg:
    """Termination terms for a staged task.

    Attributes:
        success: Every stage goal holds. Ends the episode with success.
        dropped: A tracked object fell below the ground. Ends the episode as a
            failure so that a dropped object cannot be recovered from and the
            remaining episode length is not wasted.
    """

    success: TerminationTermCfg = None
    dropped: TerminationTermCfg = None
    done: TerminationTermCfg = None


@configclass
class StagedRewardsCfg:
    """Reward terms for a staged task.

    Built from the task's own goals rather than hand-written per task, which is
    what lets one config serve all 50 industrial tasks: the shaping surface and
    the success condition are read off the same :class:`TaskSpec`, so they
    cannot drift apart the way a hand-tuned reward does.

    The terms fall into four groups, deliberately weighted so that no single one
    can dominate:

    * ``goal_progress`` and its ``mean``/``min``/``product`` siblings are the
      task signal itself, all in ``[0, 1]``. The mean answers "how far along
      am I", the minimum answers "what is still missing", and the product
      rewards getting every part right at once.
    * ``goal_fraction`` and ``success_bonus`` are the discrete objective,
      straight from the boolean predicates. Shaping on its own is gameable by
      creeping toward every target in parallel; these two keep the policy
      pointed at the thing the episode is actually scored on.
    * ``ee_reach`` and ``grasp`` shape the *means* -- closing on an object and
      shutting the gripper on it -- which the goals cannot express but without
      which the first grasp of an episode is unguided.
    * The remaining terms are penalties: smoothness, not flinging the props, and
      not sweeping them off the table.

    Attributes:
        goal_progress: Mean continuous score over every goal of the task.
        goal_progress_bottleneck: Lowest-scoring goal, which puts the weight on
            the single part of the task still undone.
        goal_progress_conjunctive: Product over goals, approximating "all of
            them at once".
        goal_fraction: Fraction of goals currently satisfied, in ``[0, 1]``.
        success_bonus: One on the step every goal holds, zero otherwise.
        ee_reach: Chosen arm closing on its stage target, the dense approach
            signal a pick stage needs.
        grasp: Gripper shut on an object that is within reach.
        action_rate: Penalizes jerky action changes.
        joint_velocity: Penalizes fast joint motion.
        joint_acceleration: Penalizes abrupt joint acceleration.
        object_motion: Penalizes the scene props moving, which keeps a policy
            from bulldozing the distractors to get somewhere.
        dropped: Penalizes an object sinking below the ground line, so the
            ``dropped`` termination has a gradient leading into it rather than
            only a terminal.
    """

    goal_progress: RewardTermCfg = None
    goal_progress_bottleneck: RewardTermCfg = None
    goal_progress_conjunctive: RewardTermCfg = None
    goal_fraction: RewardTermCfg = None
    success_bonus: RewardTermCfg = None
    ee_reach: RewardTermCfg = None
    grasp: RewardTermCfg = None
    action_rate: RewardTermCfg = None
    joint_velocity: RewardTermCfg = None
    joint_acceleration: RewardTermCfg = None
    object_motion: RewardTermCfg = None
    dropped: RewardTermCfg = None


@configclass
class StagedTaskObsCfg(ObservationGroupCfg):
    """Task-specific observations for a staged task.

    The OpenArm bimanual embodiment ships a top-down RGB camera but leaves it
    out of its own observation group, so the view is added here. Both a raw and
    a photometrically augmented copy are exposed: a policy should train on the
    augmented view, while evaluation and the scripted oracle use the raw one.

    The group is deliberately *not* called ``policy``. Arena merges the
    embodiment's and the task's observation configclasses field by field and
    rejects the merge outright when a shared field has two different types, so
    redeclaring ``policy`` here would make every one of the 50 tasks fail to
    build. The embodiment owns ``policy``; the task contributes ``task_obs``.

    The group is only built when the camera is enabled, because it is the one
    group that cannot be concatenated, and RSL-RL requires flat tensors. See
    :meth:`StagedTask._make_observation_cfg`.

    Attributes:
        rgb: Raw camera image, shape ``(N, H, W, 3)``, float in [0, 1].
        rgb_augmented: Photometrically randomized view of the same image.
        object_pose: Pose of the primary manipulated object.
    """

    rgb: ObservationTermCfg = None
    rgb_augmented: ObservationTermCfg = None
    object_pose: ObservationTermCfg = None
    # RGB frames and the numeric object pose have different trailing shapes;
    # keep them as named dictionary entries instead of stacking them.
    concatenate_terms: bool = False


@configclass
class StagedPrivilegedObsCfg(ObservationGroupCfg):
    """Full simulator state for the critic of an asymmetric actor-critic.

    The embodiment's ``policy`` group is proprioception only, so a critic
    trained on it would be estimating the value of a state it cannot see the
    reward in. This group supplies what is missing: where every object actually
    is, how fast it is moving, where the end effectors are in the world frame
    the goals are written in, and what each goal object is doing relative to its
    own target.

    The last of those is the reason to include the goals at all.
    ``goal_relative_object_offsets`` expresses progress as a displacement from a
    target rather than as an absolute position, so one value function serves
    every environment in a vectorized batch even though the object placer gave
    each of them a different layout.

    Nothing here is observable on a real robot, which is exactly why the group
    is named ``critic`` and never ``policy``. The RSL-RL runner config in
    :mod:`openarm.arena_environments.industrial.rsl_rl_cfg` is what binds the
    two sides together, and it lists this group under ``critic`` only.

    Attributes:
        object_poses: Environment-frame pose of every scene object.
        object_velocities: World-frame linear and angular velocity of every
            scene object, so "resting in the bin" is distinguishable from
            "tumbling through it".
        object_heights: Height of every scene object above the ground, the
            quantity the ``dropped`` termination and the lift goals turn on.
        goal_offsets: Displacement of each goal object from its own target.
        ee_world_poses: World-frame end-effector poses, since the goals are
            written in world coordinates and the embodiment reports poses
            relative to the robot.
        gripper_openings: Finger joint position of each arm.
        joint_efforts: Applied joint torques, which distinguish a stall against
            a jammed object from genuine progress.
    """

    object_poses: ObservationTermCfg = None
    object_velocities: ObservationTermCfg = None
    object_heights: ObservationTermCfg = None
    goal_offsets: ObservationTermCfg = None
    ee_world_poses: ObservationTermCfg = None
    gripper_openings: ObservationTermCfg = None
    joint_efforts: ObservationTermCfg = None
    concatenate_terms: bool = True


@configclass
class StagedObservationsCfg:
    """Observation group wrapper for staged tasks.

    Attributes:
        task_obs: The task-specific observation group, including the camera.
            ``None`` when the camera is disabled, which is how the camera-less
            RL-training path avoids a non-concatenated group.
        critic: The privileged state group for asymmetric actor-critic
            training. ``None`` when the scene holds no objects.
    """

    task_obs: StagedTaskObsCfg = None
    critic: StagedPrivilegedObsCfg = None


@configclass
class StagedSceneCfg:
    """Per-task scene additions.

    Intentionally empty. An earlier revision attached a contact sensor to the
    primary object here to tell "resting on the table" apart from "resting on
    another object", but nothing ever read it: the settling predicates are
    pose- and velocity-based (:func:`object_below_height`, ``at_rest``). Keeping
    it would have made every one of the 50 tasks depend on the
    ``omni.physics.tensors`` extension being enabled just to construct, for no
    behavioural gain.
    """

@configclass
class StagedTask(TaskBase):
    """An Arena task whose success condition is a conjunction of stage goals."""

    def __init__(
        self,
        spec: TaskSpec,
        background: Any,
        primary_object: Any = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        seed: int | None = None,
        enable_cameras: bool = True,
        enable_rewards: bool = True,
        scene_objects: list[str] | None = None,
    ):
        """Build the task.

        Args:
            spec: The declarative task description.
            background: The background scene asset, used for the ground height
                in the drop termination.
            primary_object: The main manipulated object, used for the contact
                sensor and the object-pose observation.
            episode_length_s: Overrides ``spec.episode_length_s`` when given.
            task_description: Overrides ``spec.instruction`` when given.
            seed: Seed for choosing among the instruction variants. ``None``
                samples a variant on every construction, which is what gives
                language diversity across vectorized environments.
            enable_cameras: Whether the embodiment was built with a camera.
                When it was not (``--no-cameras`` or ``--rl-training``) the whole
                ``task_obs`` group is omitted, not just its RGB terms: those
                terms resolve ``top_camera`` at manager construction time and
                would otherwise fail, and the group they live in cannot be
                concatenated. See :meth:`_make_observation_cfg`.
            enable_rewards: Whether to build a reward manager. Left on for RL
                training and off for scripted evaluation, where the reward is
                computed every step and never read. Note that a task without
            ``enable_rewards`` still ends on the same success condition, so the
                oracle's behaviour is unaffected either way.
            scene_objects: Every object in the scene, not only the ones the
                spec manipulates. The privileged critic group reads all of them,
                because a prop the robot must not disturb is as much a part of
                the state as one it must move. Defaults to the spec's own
                ``object_names``.

        Raises:
            ValueError: If the spec declares no stage goals, which would make
                the success condition vacuously true.
        """
        super().__init__(
            episode_length_s=episode_length_s if episode_length_s is not None else spec.episode_length_s,
            task_description=task_description,
        )
        if not spec.goals():
            raise ValueError(
                f"Task '{spec.name}' declares no stage goals, so its success condition would always hold. "
                "Attach at least one goal with Stage.with_goal()."
            )
        self.spec = spec
        self.background = background
        self.primary_object = primary_object
        self.enable_cameras = enable_cameras
        self.enable_rewards = enable_rewards
        self._rng = random.Random(seed)

        descriptions = spec.all_instructions()
        if task_description is None and descriptions:
            self.task_description = self._rng.choice(list(descriptions))
        elif self.task_description is None:
            self.task_description = spec.name.replace("_", " ")

        # Every object the scene actually holds. Resolved once, here, because the
        # reward and critic terms all need the same list and each would otherwise
        # re-derive it -- and derive it slightly differently.
        self.scene_objects = list(scene_objects) if scene_objects is not None else list(spec.object_names)

        self.scene_config = self._make_scene_cfg()
        self.termination_cfg = self._make_termination_cfg()
        self.reward_config = self._make_reward_cfg() if enable_rewards else None
        self.observation_cfg = self._make_observation_cfg()
        self.events_cfg = None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _make_scene_cfg(self) -> StagedSceneCfg | None:
        return StagedSceneCfg()

    def _make_termination_cfg(self) -> StagedTerminationsCfg:
        success = TerminationTermCfg(
            func=pred.all_goals,
            params={"goals": [goal.params() for goal in self.spec.goals()]},
        )
        dropped = None
        if self.spec.ground_height is not None and self.spec.object_names:
            dropped = TerminationTermCfg(
                func=pred.object_below_height,
                params={
                    "object_names": list(self.spec.object_names),
                    "minimum_height": self.spec.ground_height,
                },
            )
        done = TerminationTermCfg(
            func=mdp_rewards.time_out
        )
        return StagedTerminationsCfg(success=success, dropped=dropped, done=done)

    def _make_reward_cfg(self) -> StagedRewardsCfg | None:
        """Build the reward manager from the task's own goals.

        Every task-shaped term is derived from ``spec.goals()`` rather than
        written out per task, which is what makes one config serve all 50
        industrial tasks and keeps the shaping surface tied to the success
        condition by construction.
        """
        goals = [goal.params() for goal in self.spec.goals()]
        tracked = list(self.spec.object_names)

        goal_progress = RewardTermCfg(func=rew.goal_progress, weight=10.0, params={"goals": goals})
        goal_fraction = RewardTermCfg(func=rew.goal_fraction, weight=5.0, params={"goals": goals})
        success = RewardTermCfg(func=rew.success_bonus, weight=5000.0, params={"goals": goals})
        action_rate = RewardTermCfg(func=mdp_rewards.action_rate_l2, weight=-1e-4)
        joint_velocity = RewardTermCfg(
            func=mdp_rewards.joint_vel_l2,
            weight=-1e-4,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        joint_acceleration = RewardTermCfg(
            func=mdp_rewards.joint_acc_l2,
            weight=-1e-6,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        bottleneck = RewardTermCfg(
            func=rew.goal_progress,
            weight=5.0,
            params={"goals": goals, "aggregation": "min"},
        )
        conjunctive = RewardTermCfg(
            func=rew.goal_progress,
            weight=5.0,
            params={"goals": goals, "aggregation": "product"},
        )

        # Objects the task never mentions are the props the robot must not
        # disturb, so they are the ones the motion penalty watches.
        untouched = [name for name in self.scene_objects if name not in set(tracked)]
        object_motion = None
        if untouched:
            object_motion = RewardTermCfg(
                func=rew.object_motion_l2,
                weight=-1.0,
                params={"object_names": untouched, "lin_scale": 0.5, "ang_scale": 1.0},
            )

        dropped = None
        if self.spec.ground_height is not None and tracked:
            dropped = RewardTermCfg(
                func=rew.object_below_height_l2,
                weight=-1.0,
                params={"object_names": tracked, "minimum_height": self.spec.ground_height},
            )

        ee_reach, grasp = self._make_manipulation_rewards()

        return StagedRewardsCfg(
            goal_progress=goal_progress,
            goal_progress_bottleneck=bottleneck,
            goal_progress_conjunctive=conjunctive,
            goal_fraction=goal_fraction,
            success_bonus=success,
            ee_reach=ee_reach,
            grasp=grasp,
            action_rate=action_rate,
            joint_velocity=joint_velocity,
            joint_acceleration=joint_acceleration,
            object_motion=object_motion,
            dropped=dropped,
        )

    def _make_manipulation_rewards(self) -> tuple[RewardTermCfg | None, RewardTermCfg | None]:
        """Build the approach and grasp terms for the task's first pick stage.

        The goals say where an object has to end up but say nothing about how it
        gets there, so the first grasp of an episode would otherwise be
        unguided. Both terms are anchored to the first stage that names a
        target, which is the stage that introduces an object to move; tasks
        whose stages are all means rather than ends get neither.
        """
        stage = next(
            (stage for stage in self.spec.stages if stage.target and stage.target in self.scene_objects),
            None,
        )
        if stage is None:
            return None, None
        return (
            RewardTermCfg(
                func=rew.ee_object_distance,
                weight=2.0,
                params={
                    "arm": "closest",
                    "object_cfg": stage.target,
                    "std": APPROACH_STD_M,
                },
            ),
            RewardTermCfg(
                func=rew.grasp_attempt,
                weight=1.0,
                params={"arm": "closest", "object_cfg": stage.target},
            ),
        )

    def _make_observation_cfg(self) -> StagedObservationsCfg | None:
        # ``task_obs`` exists to carry the camera, and it is the one group that
        # cannot be concatenated: RGB frames and the numeric object pose have
        # different trailing shapes, so the group stays a nested dict. RSL-RL
        # expects every group to be a flat tensor, and its NaN check rejects the
        # nesting outright, so the group is dropped whenever the camera is off --
        # which is what ``--rl-training`` does. With the camera off there is
        # nothing in it the actor or critic read anyway.
        if not self.enable_cameras:
            return StagedObservationsCfg(task_obs=None, critic=self._make_critic_obs_cfg())
        obs = StagedTaskObsCfg()
        obs.rgb = ObservationTermCfg(
            func=dr_terms.rgb_image,
            params={"sensor_cfg": SceneEntityCfg(CAMERA_NAME)},
        )
        obs.rgb_augmented = ObservationTermCfg(
            func=dr_terms.augmented_rgb,
            params={"sensor_cfg": SceneEntityCfg(CAMERA_NAME)},
        )
        if self.primary_object is not None:
            obs.object_pose = ObservationTermCfg(
                func=dr_terms.object_pose,
                params={"asset_cfg": SceneEntityCfg(self.primary_object.name)},
            )
        return StagedObservationsCfg(task_obs=obs, critic=self._make_critic_obs_cfg())

    def _make_critic_obs_cfg(self) -> StagedPrivilegedObsCfg | None:
        """Build the privileged critic group, or ``None`` when there is no state.

        Every term here needs at least one scene object, so a task whose spec
        names none gets no critic group rather than a set of zero-width
        observations.
        """
        if not self.scene_objects:
            return None
        critic = StagedPrivilegedObsCfg()
        critic.object_poses = ObservationTermCfg(
            func=obs_terms.object_poses,
            params={"object_names": self.scene_objects},
        )
        critic.object_velocities = ObservationTermCfg(
            func=obs_terms.object_velocities,
            params={"object_names": self.scene_objects},
        )
        critic.object_heights = ObservationTermCfg(
            func=obs_terms.object_relative_to_surface,
            params={"object_names": self.scene_objects},
        )
        critic.ee_world_poses = ObservationTermCfg(func=obs_terms.ee_world_poses, params={})
        critic.gripper_openings = ObservationTermCfg(func=obs_terms.gripper_openings, params={})
        critic.joint_efforts = ObservationTermCfg(func=obs_terms.joint_efforts, params={})

        offsets = self._goal_object_pairs()
        if offsets:
            critic.goal_offsets = ObservationTermCfg(
                func=obs_terms.goal_relative_object_offsets,
                params={"pairs": offsets},
            )
        return critic

    def _goal_object_pairs(self) -> list[tuple[str, tuple[float, float, float]]]:
        """Pair every goal that names both an object and a point with that point.

        Only the ``in_region`` goal states a target as a point, but it is the
        one the industrial tasks use for "place it in the bin", and an object
        offset from its own target is the quantity that survives a change of
        layout. Goals of every other kind are skipped rather than approximated:
        a wrong target would be worse than no target, because the critic would
        be handed a confidently wrong notion of progress.
        """
        pairs: list[tuple[str, tuple[float, float, float]]] = []
        for goal in self.spec.goals():
            object_name = getattr(goal, "object_name", "")
            center = getattr(goal, "center", None)
            if not object_name or center is None or object_name not in self.scene_objects:
                continue
            if any(existing == object_name for existing, _ in pairs):
                continue
            pairs.append((object_name, tuple(float(value) for value in center)))
        return pairs

    # ------------------------------------------------------------------
    # TaskBase interface
    # ------------------------------------------------------------------

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_rewards_cfg(self) -> Any:
        return self.reward_config

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_observation_cfg(self) -> Any:
        return self.observation_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Any:
        """Build the Mimic datagen config for this task.

        The subtask split is derived from the stages rather than written out, so
        it cannot describe a segmentation the task does not have.

        Args:
            arm_mode: The embodiment's arm mode, which decides which arms get
                subtask configs: the embodiment only commands the arms its action
                terms include.

        Returns:
            A :class:`~openarm.arena_tasks.staged.mimic.StagedMimicEnvCfg`.
        """
        from openarm.arena_tasks.staged.mimic import StagedMimicEnvCfg

        primary = next(iter(self.scene_objects), "")
        return StagedMimicEnvCfg(
            task_spec=self.spec,
            fallback_object=primary,
            arm_mode=arm_mode,
        )

    def get_metrics(self) -> list[MetricBase]:
        metrics: list[MetricBase] = [SuccessRateMetric()]
        if self.primary_object is not None:
            metrics.append(ObjectMovedRateMetric(self.primary_object))
        return metrics

    # ------------------------------------------------------------------
    # Introspection helpers used by the oracle policies
    # ------------------------------------------------------------------

    def stage_names(self) -> list[str]:
        """Return a readable name for every stage, for logging."""
        names = []
        for index, stage in enumerate(self.spec.stages):
            label = stage.label or stage.skill.value
            names.append(f"{index}:{label}")
        return names

    def describe_goal(self) -> str:
        """Return a readable description of the full success condition."""
        return " AND ".join(goal.describe() for goal in self.spec.goals())

    def describe_reward(self) -> str:
        """Return a readable description of the reward terms, for logging."""
        if self.reward_config is None:
            return "no reward terms"
        active = [
            f"{name} (w={term.weight:+g})"
            for name, term in vars(self.reward_config).items()
            if isinstance(term, RewardTermCfg)
        ]
        return ", ".join(active)
