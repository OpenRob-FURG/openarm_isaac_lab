"""Task 1-10: mechanical assembly and fastening.

These tasks lean on the small mechanical parts in the registry (``peg``/``hole``,
the gear set, clamps, bolts) and on the hammer and drill tools, and are the
family where the two arms genuinely have to cooperate: one arm holds a part in
place while the other drives something into it.
"""

from __future__ import annotations

from openarm.arena_environments.industrial._base import TABLE_HEIGHT, IndustrialTask
from openarm.arena_tasks.staged import (
    ObjectAtRest,
    ObjectInRegion,
    ObjectRelativeTo,
    ObjectUpright,
    ObjectsWithinDistance,
    Skill,
    Stage,
    TaskSpec,
)

_HAMMER = "hammer_handal_robolab"
_DIAMETER = 0.05

TASKS: list[IndustrialTask] = [
    IndustrialTask(
        spec=TaskSpec(
            name="thread_nut_on_bolt",
            instruction="Thread the nut onto the bolt.",
            instruction_variants=(
                "Screw the nut onto the bolt.",
                "Pick up the nut and thread it onto the bolt.",
                "Hold the bolt steady and turn the nut onto it.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="bolt",
                    grasp_offset=(0.0, 0.0, 0.0),
                    label="hold the bolt upright",
                ).with_goal(ObjectAtRest("bolt", max_speed=0.02, max_angular_speed=0.2)),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="nut",
                    grasp_offset=(0.0, 0.0, 0.0),
                    label="pick up the nut",
                ),
                Stage(
                    skill=Skill.INSERT,
                    arm="right",
                    target="nut",
                    secondary="bolt",
                    anchor=(0.0, 0.0, 0.03),
                    push_distance=0.02,
                    label="thread the nut onto the bolt",
                ).with_goal(ObjectRelativeTo("nut", "bolt", (0.0, 0.0, 0.005), tolerance=0.012)),
            ),
            object_names=("bolt", "nut"),
            episode_length_s=45.0,
            success_label="nut threaded onto bolt",
        ),
        objects={"bolt": "bolt_m16", "nut": "nut_m16"},
        primary_object="nut",
        dr_preset="default",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="peg_into_hole",
            instruction="Insert the peg into the hole.",
            instruction_variants=(
                "Put the peg into the hole.",
                "Pick up the peg and place it in the hole block.",
                "Fit the peg into the hole.",
            ),
            stages=(
                Stage(skill=Skill.PICK, arm="right", target="peg", label="pick up the peg").with_goal(
                    ObjectAtRest("peg", max_speed=0.02, max_angular_speed=0.2)
                ),
                Stage(
                    skill=Skill.INSERT,
                    arm="right",
                    target="peg",
                    secondary="hole_block",
                    anchor=(0.0, 0.0, 0.02),
                    push_distance=0.03,
                    label="insert the peg into the hole",
                ).with_goal(ObjectRelativeTo("peg", "hole_block", (0.0, 0.0, 0.06), tolerance=0.025)),
            ),
            object_names=("peg",),
            episode_length_s=35.0,
            success_label="peg seated in hole",
        ),
        objects={"peg": "peg", "hole_block": "hole"},
        primary_object="peg",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="gear_pair_meshing",
            instruction="Mount both gears on the base so that they mesh.",
            instruction_variants=(
                "Place the two gears onto the gear base and mesh them.",
                "Fit the large and small gear onto the base so their teeth engage.",
                "Assemble the gear pair on the base.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="large_gear",
                    release_height=TABLE_HEIGHT + 0.02,
                    label="place the large gear",
                ).with_goal(ObjectInRegion("large_gear", (0.30, -0.12, TABLE_HEIGHT + 0.02), (0.07, 0.07, 0.05))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="small_gear",
                    release_height=TABLE_HEIGHT + 0.02,
                    label="place the small gear to mesh",
                ).with_goal(
                    ObjectInRegion("small_gear", (0.42, -0.12, TABLE_HEIGHT + 0.02), (0.06, 0.06, 0.05))
                ),
                Stage(
                    skill=Skill.STABILIZE,
                    arm="left",
                    anchor=(0.0, 0.0, 0.0),
                    settle_steps=8,
                    label="hold the base steady",
                ),
            ),
            object_names=("large_gear", "small_gear"),
            episode_length_s=55.0,
            success_label="both gears mounted and meshing",
        ),
        objects={
            "large_gear": "large_gear",
            "small_gear": "small_gear",
            "gear_base": "gear_base",
        },
        primary_object="small_gear",
        pinned_objects=("gear_base",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="gear_train_build",
            instruction="Mount all three gears on the base in order.",
            instruction_variants=(
                "Build the three-gear train on the base.",
                "Place the small, medium and large gears onto the gear base.",
                "Assemble the full gear train.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="small_gear",
                    release_height=TABLE_HEIGHT + 0.02,
                    label="place the small gear",
                ).with_goal(ObjectInRegion("small_gear", (0.28, -0.10, TABLE_HEIGHT + 0.02), (0.06, 0.06, 0.05))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="medium_gear",
                    release_height=TABLE_HEIGHT + 0.02,
                    label="place the medium gear",
                ).with_goal(ObjectInRegion("medium_gear", (0.40, -0.10, TABLE_HEIGHT + 0.02), (0.07, 0.07, 0.05))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="large_gear",
                    release_height=TABLE_HEIGHT + 0.02,
                    label="place the large gear",
                ).with_goal(ObjectInRegion("large_gear", (0.52, -0.10, TABLE_HEIGHT + 0.02), (0.08, 0.08, 0.05))),
            ),
            object_names=("small_gear", "medium_gear", "large_gear"),
            episode_length_s=70.0,
            success_label="three-gear train assembled",
        ),
        objects={
            "small_gear": "small_gear",
            "medium_gear": "medium_gear",
            "large_gear": "large_gear",
            "gear_base": "gear_base",
        },
        primary_object="medium_gear",
        pinned_objects=("gear_base",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="hold_and_hammer",
            instruction="Hold the block steady with one arm and hammer it with the other.",
            instruction_variants=(
                "Brace the block with your left gripper and strike it with the hammer in your right.",
                "Use both arms: one holds the block, the other hammers it.",
                "Stabilize the block and drive the hammer into it.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="block",
                    label="brace the block",
                ).with_goal(ObjectAtRest("block", max_speed=0.02, max_angular_speed=0.2)),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target=_HAMMER,
                    label="pick up the hammer",
                ),
                Stage(
                    skill=Skill.STRIKE,
                    arm="right",
                    target=_HAMMER,
                    secondary="block",
                    strike_amplitude=0.14,
                    strike_cycles=3,
                    label="hammer the block",
                ),
            ),
            object_names=("block", _HAMMER),
            episode_length_s=55.0,
            success_label="block hammered while braced",
        ),
        objects={"block": "wood_block_ycb_robolab", _HAMMER: _HAMMER},
        primary_object="block",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="nail_driving",
            instruction="Hold the plank and drive the nail with the hammer.",
            instruction_variants=(
                "Keep the plank still and hammer the nail into it.",
                "Hold the plank with one arm and strike the nail with the other.",
                "Drive the nail into the plank.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="plank",
                    label="hold the plank flat",
                ).with_goal(ObjectUpright("plank", tolerance=0.5)),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target=_HAMMER,
                    label="pick up the hammer",
                ),
                Stage(
                    skill=Skill.STRIKE,
                    arm="right",
                    target=_HAMMER,
                    secondary="plank",
                    strike_amplitude=0.12,
                    strike_cycles=4,
                    label="drive the nail",
                ),
            ),
            object_names=("plank", _HAMMER),
            episode_length_s=50.0,
            success_label="nail driven into plank",
        ),
        objects={"plank": "brick_ycb_robolab", _HAMMER: _HAMMER},
        primary_object="plank",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="clamp_bracing",
            instruction="Clamp the bracket to the table and tighten it.",
            instruction_variants=(
                "Secure the bracket to the table edge with the clamp.",
                "Attach the clamp to the table so the bracket is held.",
                "Mount and secure the bracket.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="bracket",
                    release_height=TABLE_HEIGHT + 0.03,
                    label="position the bracket on the table edge",
                ).with_goal(ObjectInRegion("bracket", (0.40, 0.22, TABLE_HEIGHT + 0.02), (0.08, 0.08, 0.06))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="clamp",
                    release_height=TABLE_HEIGHT + 0.06,
                    label="attach the clamp",
                ).with_goal(ObjectsWithinDistance("clamp", "bracket", max_distance=0.10)),
            ),
            object_names=("bracket", "clamp"),
            episode_length_s=50.0,
            success_label="bracket clamped to table",
        ),
        objects={"bracket": "clamp_ycb_robolab", "clamp": "spring_clamp_ycb_robolab"},
        primary_object="clamp",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="drill_fastening",
            instruction="Hold the plate and drill the fastener through it.",
            instruction_variants=(
                "Keep the plate steady and drill into it.",
                "Hold the plate with one arm and drill with the other.",
                "Drill a hole through the plate.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="plate",
                    label="hold the plate flat",
                ).with_goal(ObjectUpright("plate", tolerance=0.5)),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="drill",
                    label="pick up the drill",
                ),
                Stage(
                    skill=Skill.PRESS,
                    arm="right",
                    target="drill",
                    secondary="plate",
                    anchor=(0.0, 0.0, TABLE_HEIGHT + 0.08),
                    push_axis=2,
                    push_distance=0.05,
                    label="drill into the plate",
                ).with_goal(ObjectAtRest("plate", max_speed=0.03, max_angular_speed=0.3)),
            ),
            object_names=("plate", "drill"),
            episode_length_s=50.0,
            success_label="fastener drilled through plate",
        ),
        objects={"plate": "wood_block_ycb_robolab", "drill": "cordless_drill_ycb_robolab"},
        primary_object="plate",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="bolt_bracing_insertion",
            instruction="Brace the gear base with one arm and insert the bolt with the other.",
            instruction_variants=(
                "Hold the base still and push the bolt through it.",
                "Stabilize the base and drive the bolt in.",
                "Insert the bolt while bracing the base.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="gear_base",
                    label="brace the base",
                ).with_goal(ObjectAtRest("gear_base", max_speed=0.03, max_angular_speed=0.3)),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="bolt",
                    label="pick up the bolt",
                ),
                Stage(
                    skill=Skill.INSERT,
                    arm="right",
                    target="bolt",
                    secondary="gear_base",
                    anchor=(0.0, 0.0, 0.02),
                    push_distance=0.03,
                    label="insert the bolt through the base",
                ).with_goal(ObjectRelativeTo("bolt", "gear_base", (0.0, 0.0, 0.04), tolerance=0.02)),
            ),
            object_names=("bolt",),
            episode_length_s=45.0,
            success_label="bolt inserted through braced base",
        ),
        objects={"bolt": "bolt_m16", "gear_base": "gear_base"},
        primary_object="bolt",
        pinned_objects=("gear_base",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="sort_fastener_sizes",
            instruction="Sort the fasteners by size into the matching bins.",
            instruction_variants=(
                "Put each fastener in the bin that matches its size.",
                "Sort the bolts and nuts into the two bins.",
                "Separate the fasteners by size.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="big_fastener",
                    release_height=TABLE_HEIGHT + 0.05,
                    label="place the large fastener in its bin",
                ).with_goal(ObjectInRegion("big_fastener", (0.65, 0.34, TABLE_HEIGHT + 0.05), (0.09, 0.09, 0.08))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="small_fastener",
                    release_height=TABLE_HEIGHT + 0.05,
                    label="place the small fastener in its bin",
                ).with_goal(ObjectInRegion("small_fastener", (0.65, 0.02, TABLE_HEIGHT + 0.05), (0.08, 0.08, 0.08))),
            ),
            object_names=("big_fastener", "small_fastener"),
            episode_length_s=50.0,
            success_label="fasteners sorted by size",
        ),
        objects={
            "big_fastener": "bolt_m16",
            "small_fastener": "nut_m16",
            "big_bin": "openarm_sorting_bin",
            "small_bin": "red_container",
        },
        primary_object="big_fastener",
        pinned_objects=("big_bin", "small_bin"),
    ),
]
