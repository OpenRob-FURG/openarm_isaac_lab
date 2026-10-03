"""Declarative manipulation stages shared by staged tasks and oracle policies.

A :class:`Stage` is the single source of truth for one step of a task. The same
stage list drives two consumers:

* the task, which turns each stage's :class:`~openarm.arena_tasks.staged.goals.Goal`
  into a success predicate and a language instruction, and
* the scripted oracle policy, which turns each stage's
  :class:`Skill` into a controller.

Keeping one description for both is what stops the demonstrations and the
success metric from drifting apart, which is the usual failure mode when a
task's success condition and its scripted policy are written separately.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # imported lazily so the pure-data modules stay simulator-free
    from openarm.arena_tasks.staged.layout import SceneLayout


class Skill(str, Enum):
    """The kind of manipulation a stage performs.

    The oracle policy implements one controller per skill. Adding a new stage
    behaviour means adding a skill here and a controller in
    :mod:`openarm.policies.oracle.skills`.
    """

    #: Move an end effector to a named anchor without touching anything.
    MOVE = "move"
    #: Approach, close the gripper, and lift.
    PICK = "pick"
    #: Carry an object to a target pose, open the gripper, and retreat.
    PLACE = "place"
    #: A full pick followed by a place.
    PICK_PLACE = "pick_place"
    #: Push a pressable control.
    PRESS = "press"
    #: Open an openable articulation.
    OPEN = "open"
    #: Close an openable articulation.
    CLOSE = "close"
    #: Turn a turnable articulation.
    TURN = "turn"
    #: Strike a target with a held tool.
    STRIKE = "strike"
    #: Insert one object into another while the other arm stabilises it.
    INSERT = "insert"
    #: Transfer a held object from one arm to the other.
    HANDOFF = "handoff"
    #: Hold the current pose and grip for a number of steps.
    STABILIZE = "stabilize"
    #: Return both end effectors to the home pose.
    DOCK = "dock"


#: Skills whose success is captured by an explicit goal in the task spec.
OBJECT_SKILLS = frozenset(
    {
        Skill.PICK_PLACE,
        Skill.PLACE,
        Skill.PRESS,
        Skill.OPEN,
        Skill.CLOSE,
        Skill.TURN,
        Skill.STRIKE,
        Skill.INSERT,
    }
)


@dataclass
class Stage:
    """One step of a staged task.

    Attributes:
        skill: Which manipulation to perform.
        arm: Which arm performs it: ``"left"``, ``"right"`` or ``"both"``.
        target: Scene name of the primary object acted on. For
            :attr:`Skill.MOVE` this is ignored unless ``anchor`` is set.
        secondary: Scene name of a second object, used by
            :attr:`Skill.INSERT` (the receptacle) and :attr:`Skill.PICK`
            (the object to stabilise).
        anchor: World-frame target position for :attr:`Skill.MOVE`, as
            ``(x, y, z)``.
        grasp_offset: Grasp point in the target object's local frame [m].
        grasp_euler: Gripper orientation. Expressed as an XYZ euler triple
            applied on top of the object's own orientation, unless
            ``use_world_orientation`` is set. The end-effector convention makes
            a zero offset a sensible object-aligned grasp, which is the right
            default for cans, boxes and bottles.
        use_world_orientation: If true, ``grasp_euler`` is an absolute world
            frame orientation instead of a local offset. Needed for tasks where
            the object can be spawned at arbitrary yaw but the gripper must
            still approach from a fixed world direction.
        approach_axis: Local axis along which the gripper approaches, 0/1/2 for
            x/y/z.
        approach_distance: Distance of the pre-grasp pose from the grasp point
            along ``approach_axis`` [m].
        lift_height: Height added after a grasp or before a release [m].
        release_height: Height at which the gripper opens during a place [m].
        settle_steps: Number of steps to hold still after the stage completes,
            used to let physics settle before the next stage reads object poses.
        max_steps: Step budget for the stage before it is declared failed.
            The default of 600 covers the collision-free route out from under
            the work surface, a reach, grasp, lift and placement at the OpenArm
            bimanual control rate. The episode timeout remains the outer bound
            for tasks with several stages.
        max_attempts: Number of times the oracle will retry the stage.
        tolerance_pos: Position convergence tolerance [m].
        tolerance_rot: Orientation convergence tolerance, as a quaternion
            distance [unitless].
        label: Text used to build the language instruction for this stage.
        goal: Success predicate for the stage, or ``None`` for stages that are
            means rather than ends (``MOVE``, ``STABILIZE``, ``DOCK``).
        push_axis: Local axis along which :attr:`Skill.PRESS` pushes.
        push_distance: How far :attr:`Skill.PRESS` travels [m].
        turn_axis: Local axis about which :attr:`Skill.TURN` rotates.
        turn_angle: Rotation applied by :attr:`Skill.TURN` [rad].
        strike_axis: Direction of the :attr:`Skill.STRIKE` swing, expressed in
            the tool's local frame.
        strike_amplitude: Peak displacement of the strike [m].
        strike_cycles: Number of strike repetitions.
        hold_gripper: Whether the arm keeps its gripper closed while another arm
            acts, which is what makes genuinely bimanual stages possible.
    """

    skill: Skill = Skill.MOVE
    arm: str = "right"
    target: str = ""
    secondary: str = ""
    anchor: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    grasp_euler: tuple[float, float, float] = (0.0, 0.0, 0.0)
    use_world_orientation: bool = False
    approach_axis: int = 2
    approach_distance: float = 0.08
    lift_height: float = 0.12
    release_height: float = 0.10
    settle_steps: int = 5
    max_steps: int = 600
    max_attempts: int = 3
    tolerance_pos: float = 0.03
    tolerance_rot: float = 0.15
    label: str = ""
    goal: object | None = None
    push_axis: int = 2
    push_distance: float = 0.03
    turn_axis: int = 2
    turn_angle: float = 1.5708
    strike_axis: int = 0
    strike_amplitude: float = 0.12
    strike_cycles: int = 3
    hold_gripper: bool = False

    def with_goal(self, goal: object) -> "Stage":
        """Return a copy of this stage with its success goal attached.

        Args:
            goal: The goal to attach.

        Returns:
            A new :class:`Stage` instance.
        """
        return Stage(**{**self.__dict__, "goal": goal})


@dataclass
class TaskSpec:
    """Complete declarative description of a staged task.

    Attributes:
        name: Environment name, also used as the CLI subcommand.
        background: Name of the background scene asset.
        instruction: Canonical language instruction for the task.
        instruction_variants: Additional paraphrases. A generalist policy needs
            many surface forms per task, and this is the cheapest source of
            them.
        stages: Ordered manipulation stages.
        object_names: Scene names of the rigid objects that participate in
            domain randomization. Excludes the background and the robot.
        episode_length_s: Episode duration [s].
        ground_height: Height below which a dropped object ends the episode
            [m]. ``None`` disables the drop termination.
        pinned_objects: Objects excluded from pose and scale randomization.
        success_label: Short description of the success condition, used in
            metrics logging.
        layout: Explicit relational arrangement of the scene, solved by Arena's
            relation solver. ``None`` derives one from the stages' goal regions,
            which is what every task in the industrial suite does today; setting
            it is the escape hatch for a scene the derivation cannot express.
    """

    name: str
    background: str = "packing_table"
    instruction: str = ""
    instruction_variants: tuple[str, ...] = field(default_factory=tuple)
    stages: tuple[Stage, ...] = field(default_factory=tuple)
    object_names: tuple[str, ...] = field(default_factory=tuple)
    episode_length_s: float = 40.0
    ground_height: float | None = -0.25
    pinned_objects: tuple[str, ...] = field(default_factory=tuple)
    success_label: str = ""
    layout: "SceneLayout | None" = None

    def all_instructions(self) -> tuple[str, ...]:
        """Return the canonical instruction followed by its paraphrases."""
        return (self.instruction, *self.instruction_variants) if self.instruction else self.instruction_variants

    def goals(self) -> list[object]:
        """Collect the goals of every stage that carries one."""
        return [stage.goal for stage in self.stages if stage.goal is not None]
