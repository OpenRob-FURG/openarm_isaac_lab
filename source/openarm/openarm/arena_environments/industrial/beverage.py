"""Task 21-28: beverage and appliance operation.

This family is the only place the three articulated fixtures are used:
``microwave`` (openable), ``coffee_machine`` (pressable) and ``stand_mixer``
(turnable). The articulated goals are expressed through
:class:`~openarm.arena_tasks.staged.JointNormalizedAbove` and its ``below``
counterpart, so a task is successful only when the joint actually moved, not
merely when the robot's arm passed near the control.
"""

from __future__ import annotations

from openarm.arena_environments.industrial._base import TABLE_HEIGHT, IndustrialTask
from openarm.arena_tasks.staged import (
    All,
    JointNormalizedAbove,
    JointNormalizedBelow,
    ObjectInRegion,
    ObjectRelativeTo,
    Skill,
    Stage,
    TaskSpec,
)

Z = TABLE_HEIGHT + 0.08

#: Joint names driving the openable/pressable/turnable fixtures. These are the
#: real names inside the Arena USD assets, not generic ones: the microwave door
#: is ``Disc001_joint``, the coffee machine's front button is
#: ``CoffeeMachine108_Button001_joint``, and the mixer knob is
#: ``knob_speed_joint``. A wrong name fails at reset with a KeyError listing
#: what the articulation actually exposes.
MICROWAVE_DOOR_JOINT = "Disc001_joint"
COFFEE_BUTTON_JOINT = "CoffeeMachine108_Button001_joint"
MIXER_JOINT = "knob_speed_joint"

TASKS: list[IndustrialTask] = [
    IndustrialTask(
        spec=TaskSpec(
            name="microwave_heat_cycle",
            instruction="Open the microwave, put the mug inside, and close the door.",
            instruction_variants=(
                "Load the mug into the microwave and shut the door.",
                "Place the mug in the microwave and close it.",
                "Open the microwave, insert the mug, and close the door again.",
            ),
            stages=(
                Stage(
                    skill=Skill.OPEN,
                    arm="right",
                    target="microwave",
                    max_steps=200,
                    label="open the microwave door",
                ).with_goal(JointNormalizedAbove("microwave", MICROWAVE_DOOR_JOINT, threshold=0.5)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="mug",
                    release_height=TABLE_HEIGHT + 0.10,
                    label="put the mug in the microwave",
                ).with_goal(ObjectInRegion("mug", (0.30, 0.02, TABLE_HEIGHT + 0.10), (0.09, 0.09, 0.08))),
                Stage(
                    skill=Skill.CLOSE,
                    arm="right",
                    target="microwave",
                    max_steps=200,
                    label="close the microwave door",
                ).with_goal(JointNormalizedBelow("microwave", MICROWAVE_DOOR_JOINT, threshold=0.3)),
            ),
            object_names=("mug",),
            episode_length_s=70.0,
            success_label="mug inside a closed microwave",
        ),
        objects={"microwave": "microwave", "mug": "mug"},
        primary_object="mug",
        pinned_objects=("microwave",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="coffee_dispense",
            instruction="Press the coffee machine and place the mug under the spout.",
            instruction_variants=(
                "Start a coffee and hold the mug under the dispenser.",
                "Press the brew button, then position the mug.",
                "Dispense coffee into the mug.",
            ),
            stages=(
                Stage(
                    skill=Skill.PRESS,
                    arm="right",
                    target="coffee_machine",
                    anchor=(0.0, 0.0, TABLE_HEIGHT + 0.22),
                    push_axis=2,
                    push_distance=0.02,
                    max_steps=140,
                    label="press the brew button",
                ).with_goal(JointNormalizedAbove("coffee_machine", COFFEE_BUTTON_JOINT, threshold=0.5)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="mug",
                    release_height=TABLE_HEIGHT + 0.10,
                    label="place the mug under the spout",
                ).with_goal(ObjectInRegion("mug", (0.30, 0.06, TABLE_HEIGHT + 0.10), (0.08, 0.08, 0.07))),
            ),
            object_names=("mug",),
            episode_length_s=55.0,
            success_label="coffee dispensed into mug",
        ),
        objects={"coffee_machine": "coffee_machine", "mug": "mug"},
        primary_object="mug",
        pinned_objects=("coffee_machine",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="mug_handoff_dispense",
            instruction="Pour the coffee and hand the filled mug to the other arm.",
            instruction_variants=(
                "Serve the drink, then pass the mug across to the left gripper.",
                "Hand the filled mug from one hand to the other.",
                "Pour and then transfer the mug between arms.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="mug",
                    release_height=TABLE_HEIGHT + 0.12,
                    label="hold the mug to fill",
                ).with_goal(ObjectInRegion("mug", (0.34, 0.10, TABLE_HEIGHT + 0.14), (0.08, 0.08, 0.08))),
                Stage(
                    skill=Skill.HANDOFF,
                    arm="right",
                    target="mug",
                    label="hand the mug to the left arm",
                ).with_goal(ObjectInRegion("mug", (0.42, 0.06, TABLE_HEIGHT + 0.14), (0.10, 0.10, 0.10))),
            ),
            object_names=("mug",),
            episode_length_s=60.0,
            success_label="mug transferred between arms",
        ),
        objects={"mug": "mug", "coffee_machine": "coffee_machine"},
        primary_object="mug",
        pinned_objects=("coffee_machine",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="kettle_pour",
            instruction="Pour from the jug into the mug.",
            instruction_variants=(
                "Fill the mug from the jug.",
                "Tilt the jug over the mug and pour.",
                "Pour a drink from the pitcher into the cup.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="jug",
                    label="pick up the jug",
                ),
                Stage(
                    skill=Skill.PLACE,
                    arm="right",
                    target="mug",
                    release_height=TABLE_HEIGHT + 0.16,
                    label="pour into the mug",
                ).with_goal(ObjectInRegion("mug", (0.44, 0.14, TABLE_HEIGHT + 0.08), (0.08, 0.08, 0.07))),
            ),
            object_names=("jug", "mug"),
            episode_length_s=55.0,
            success_label="liquid poured into mug",
        ),
        objects={"jug": "jug", "mug": "mug_ycb_robolab"},
        primary_object="jug",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="stand_mixer_setup",
            instruction="Set the bowl on the mixer and turn the control on.",
            instruction_variants=(
                "Mount the bowl on the stand mixer and switch it on.",
                "Put the bowl in place, then turn the mixer control.",
                "Prepare the stand mixer for use.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="bowl",
                    release_height=TABLE_HEIGHT + 0.14,
                    label="seat the bowl on the mixer",
                ).with_goal(ObjectRelativeTo("bowl", "stand_mixer", (0.0, -0.06, 0.10), tolerance=0.07)),
                Stage(
                    skill=Skill.TURN,
                    arm="left",
                    target="stand_mixer",
                    turn_axis=2,
                    turn_angle=1.5708,
                    max_steps=160,
                    label="turn the mixer control",
                ).with_goal(JointNormalizedAbove("stand_mixer", MIXER_JOINT, threshold=0.5)),
            ),
            object_names=("bowl",),
            episode_length_s=60.0,
            success_label="mixer bowl mounted and control turned",
        ),
        objects={"stand_mixer": "stand_mixer", "bowl": "bowl_ycb_robolab"},
        primary_object="bowl",
        pinned_objects=("stand_mixer",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="dual_mug_serve",
            instruction="Serve two mugs at the same time, one per arm.",
            instruction_variants=(
                "Place both mugs on the counter, using both hands.",
                "Set down two cups simultaneously.",
                "Serve the pair of mugs with both grippers.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="left_mug",
                    release_height=Z,
                    label="place the left mug",
                ).with_goal(ObjectInRegion("left_mug", (0.32, 0.18, Z), (0.08, 0.08, 0.07))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="right_mug",
                    release_height=Z,
                    label="place the right mug",
                ).with_goal(ObjectInRegion("right_mug", (0.56, 0.18, Z), (0.08, 0.08, 0.07))),
            ),
            object_names=("left_mug", "right_mug"),
            episode_length_s=60.0,
            success_label="two mugs served",
        ),
        objects={"left_mug": "mug", "right_mug": "mug_hot3d_robolab"},
        primary_object="left_mug",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="reheat_then_pour",
            instruction="Reheat the drink in the microwave, then pour it into a clean mug.",
            instruction_variants=(
                "Warm the jug in the microwave and pour it into the mug.",
                "Reheat first, then serve into the mug.",
                "Heat the jug and pour a fresh drink.",
            ),
            stages=(
                Stage(
                    skill=Skill.OPEN,
                    arm="right",
                    target="microwave",
                    max_steps=200,
                    label="open the microwave",
                ).with_goal(JointNormalizedAbove("microwave", MICROWAVE_DOOR_JOINT, threshold=0.5)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="jug",
                    release_height=TABLE_HEIGHT + 0.10,
                    label="put the jug inside",
                ).with_goal(ObjectInRegion("jug", (0.30, 0.02, TABLE_HEIGHT + 0.10), (0.09, 0.09, 0.08))),
                Stage(
                    skill=Skill.CLOSE,
                    arm="right",
                    target="microwave",
                    max_steps=200,
                    label="close the door",
                ).with_goal(JointNormalizedBelow("microwave", MICROWAVE_DOOR_JOINT, threshold=0.3)),
                Stage(
                    skill=Skill.OPEN,
                    arm="right",
                    target="microwave",
                    max_steps=200,
                    label="take the jug back out",
                ).with_goal(JointNormalizedAbove("microwave", MICROWAVE_DOOR_JOINT, threshold=0.5)),
                Stage(
                    skill=Skill.PLACE,
                    arm="right",
                    target="mug",
                    release_height=Z,
                    label="pour into the mug",
                ).with_goal(ObjectInRegion("mug", (0.46, 0.16, Z), (0.08, 0.08, 0.07))),
            ),
            object_names=("jug", "mug"),
            episode_length_s=90.0,
            success_label="reheated drink poured into mug",
        ),
        objects={"microwave": "microwave", "jug": "jug", "mug": "mug"},
        primary_object="jug",
        pinned_objects=("microwave",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="measured_dosing",
            instruction="Measure the correct dose with the measuring cup and pour it in.",
            instruction_variants=(
                "Use the measuring cup to dose the bottle's contents.",
                "Measure a dose and transfer it to the mug.",
                "Dose the drink with the measuring cup first.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="cup",
                    release_height=Z,
                    label="dip the measuring cup into the bottle",
                ).with_goal(ObjectInRegion("cup", (0.34, 0.16, Z), (0.07, 0.07, 0.06))),
                Stage(
                    skill=Skill.PLACE,
                    arm="right",
                    target="bottle",
                    release_height=TABLE_HEIGHT + 0.16,
                    label="pour the measured dose",
                ).with_goal(
                    All(
                        goals=(
                            ObjectRelativeTo("cup", "bottle", (0.0, 0.0, 0.05), tolerance=0.10),
                            ObjectInRegion("bottle", (0.50, 0.12, TABLE_HEIGHT + 0.10), (0.09, 0.09, 0.08)),
                        )
                    )
                ),
            ),
            object_names=("cup", "bottle"),
            episode_length_s=60.0,
            success_label="measured dose poured",
        ),
        objects={
            "cup": "measuring_cups_handal_robolab",
            "bottle": "bbq_sauce_bottle_hope_robolab",
        },
        primary_object="cup",
    ),
]
