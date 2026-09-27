"""A generic Arena task driven by a declarative :class:`TaskSpec`.

Every industrial task in this package is a :class:`TaskSpec`. This class turns
that spec into the four things the Arena environment builder needs: a success
termination, a failure termination, an observation group that includes the
camera, and a set of metrics.
"""

from __future__ import annotations

import random
from typing import Any

from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.object_moved import ObjectMovedRateMetric
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase

from openarm.arena_dr import terms as dr_terms
from openarm.arena_tasks.staged import terminations as pred
from openarm.arena_tasks.staged.stages import TaskSpec

CAMERA_NAME = "top_camera"


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

    Attributes:
        rgb: Raw camera image, shape ``(N, H, W, 3)``, float in [0, 1].
        rgb_augmented: Photometrically randomized view of the same image.
        object_pose: Pose of the primary manipulated object.
    """

    rgb: ObservationTermCfg = None
    rgb_augmented: ObservationTermCfg = None
    object_pose: ObservationTermCfg = None


@configclass
class StagedObservationsCfg:
    """Observation group wrapper for staged tasks.

    Attributes:
        task_obs: The task-specific observation group.
    """

    task_obs: StagedTaskObsCfg = None


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
                When it was not (``--no_cameras``) the RGB observation terms are
                omitted, because the terms resolve ``top_camera`` at manager
                construction time and would otherwise fail.

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
        self._rng = random.Random(seed)

        descriptions = spec.all_instructions()
        if task_description is None and descriptions:
            self.task_description = self._rng.choice(list(descriptions))
        elif self.task_description is None:
            self.task_description = spec.name.replace("_", " ")

        self.scene_config = self._make_scene_cfg()
        self.termination_cfg = self._make_termination_cfg()
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
        return StagedTerminationsCfg(success=success, dropped=dropped)

    def _make_observation_cfg(self) -> StagedObservationsCfg | None:
        obs = StagedTaskObsCfg()
        if self.enable_cameras:
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
        return StagedObservationsCfg(task_obs=obs)

    # ------------------------------------------------------------------
    # TaskBase interface
    # ------------------------------------------------------------------

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_observation_cfg(self) -> Any:
        return self.observation_cfg

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> Any:
        raise NotImplementedError(
            f"Staged task '{self.spec.name}' does not provide a Mimic datagen config. "
            "Use the scripted oracle policy to collect demonstrations instead."
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
