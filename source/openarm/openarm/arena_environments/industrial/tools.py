"""Task 36-44: tool use.

The registry carries a full HANDAL tool set (ladle, tongs, spoons, measuring
cups, spatula, potato masher) plus the YCB scissors and sponge. The interesting
difficulty here is not reaching the object but coordinating a grasped tool with
a second hand that holds the receptacle steady, so several of these tasks are
genuinely bimanual rather than single-arm with extra steps.
"""

from __future__ import annotations

from openarm.arena_environments.industrial._base import TABLE_HEIGHT, IndustrialTask
from openarm.arena_tasks.staged import (
    ObjectInRegion,
    ObjectsWithinDistance,
    Skill,
    Stage,
    TaskSpec,
)

Z = TABLE_HEIGHT + 0.07

TASKS: list[IndustrialTask] = [
    IndustrialTask(
        spec=TaskSpec(
            name="ladle_serving",
            instruction="Hold the bowl steady and serve from it with the ladle.",
            instruction_variants=(
                "Brace the bowl with one arm and ladle the soup out.",
                "Steady the bowl, then scoop with the ladle.",
                "Serve a ladleful while holding the bowl still.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="bowl",
                    label="hold the bowl steady",
                ),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="ladle",
                    label="pick up the ladle",
                ),
                Stage(
                    skill=Skill.PLACE,
                    arm="right",
                    target="ladle",
                    release_height=Z,
                    label="ladle into the serving bowl",
                ).with_goal(ObjectInRegion("serving_bowl", (0.65, 0.02, Z), (0.10, 0.10, 0.08))),
            ),
            object_names=("bowl", "ladle"),
            episode_length_s=65.0,
            success_label="ladled while holding the bowl",
        ),
        objects={
            "bowl": "bowl_ycb_robolab",
            "ladle": "ladle_handal_robolab",
            "serving_bowl": "wooden_bowl_hot3d_robolab",
        },
        primary_object="ladle",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="tongs_serving",
            instruction="Use the tongs to move the item into the dish.",
            instruction_variants=(
                "Grip the item with the tongs and place it in the dish.",
                "Tong the item across to the dish.",
                "Serve the item with the tongs.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="tongs",
                    label="pick up the tongs",
                ),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="item",
                    release_height=Z,
                    label="tong the item into the dish",
                ).with_goal(ObjectInRegion("item", (0.70, 0.55, Z), (0.10, 0.10, 0.08))),
                Stage(
                    skill=Skill.PLACE,
                    arm="right",
                    target="tongs",
                    release_height=Z,
                    label="set the tongs down",
                ),
            ),
            object_names=("item", "tongs"),
            episode_length_s=60.0,
            success_label="item served with tongs",
        ),
        objects={
            "tongs": "salad_tongs_handal_robolab",
            "item": "red_bell_pepper_objaverse_robolab",
            "serving_plate": "clay_plates_hot3d_robolab",
        },
        primary_object="item",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="measuring_cup_transfer",
            instruction="Measure a cupful and transfer it to the mixing bowl.",
            instruction_variants=(
                "Fill the measuring cup and empty it into the bowl.",
                "Transfer a measured amount into the bowl with the cup.",
                "Use the measuring cup to portion into the bowl.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="cup",
                    release_height=Z,
                    label="fill the measuring cup",
                ).with_goal(ObjectInRegion("cup", (0.32, 0.16, Z), (0.08, 0.08, 0.07))),
                Stage(
                    skill=Skill.PLACE,
                    arm="right",
                    target="cup",
                    release_height=Z,
                    label="empty the cup into the bowl",
                ).with_goal(ObjectInRegion("mixing_bowl", (0.52, 0.10, Z), (0.10, 0.10, 0.08))),
            ),
            object_names=("cup",),
            episode_length_s=60.0,
            success_label="measured amount transferred",
        ),
        objects={
            "cup": "measuring_cups_1_handal_robolab",
            "source": "bbq_sauce_bottle_hot3d_robolab",
            "mixing_bowl": "wooden_bowl_hot3d_robolab",
        },
        primary_object="cup",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="tool_racking",
            instruction="Return both tools to the rack.",
            instruction_variants=(
                "Put the spoon and the spatula back on the rack.",
                "Rack the tools you took out.",
                "Store the utensils back in their holder.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="spoon",
                    release_height=Z,
                    label="return the spoon to the rack",
                ).with_goal(ObjectInRegion("spoon", (0.65, 0.02, Z), (0.10, 0.10, 0.08))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="spatula",
                    release_height=Z,
                    label="return the spatula to the rack",
                ).with_goal(ObjectInRegion("spatula", (0.65, 0.02, Z), (0.10, 0.10, 0.08))),
            ),
            object_names=("spoon", "spatula"),
            episode_length_s=55.0,
            success_label="tools racked",
        ),
        objects={
            "rack": "storage_box_hot3d_robolab",
            "spoon": "serving_spoon_handal_robolab",
            "spatula": "spatula_hot3d_robolab",
        },
        primary_object="spoon",
        pinned_objects=("rack",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="cross_hand_tool_handoff",
            instruction="Pass the tool from one hand to the other.",
            instruction_variants=(
                "Hand the utensil across from the right gripper to the left.",
                "Transfer the tool between the arms.",
                "Pass the implement to your other hand.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="tool",
                    label="pick up the tool",
                ),
                Stage(
                    skill=Skill.HANDOFF,
                    arm="right",
                    target="tool",
                    label="pass the tool to the left arm",
                ).with_goal(ObjectInRegion("tool", (0.42, 0.02, TABLE_HEIGHT + 0.16), (0.12, 0.12, 0.12))),
            ),
            object_names=("tool",),
            episode_length_s=50.0,
            success_label="tool handed between arms",
        ),
        objects={"tool": "wooden_spoons_hot3d_robolab"},
        primary_object="tool",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="stir_while_holding",
            instruction="Hold the bowl in one hand and stir it with the other.",
            instruction_variants=(
                "Brace the bowl and stir the mixture.",
                "Stir the bowl while keeping it from sliding.",
                "Use both arms: one holds the bowl, one stirs.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="bowl",
                    label="grip the bowl",
                ),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="spoon",
                    label="pick up the spoon",
                ),
                Stage(
                    skill=Skill.STRIKE,
                    arm="right",
                    target="spoon",
                    secondary="bowl",
                    strike_amplitude=0.09,
                    strike_cycles=4,
                    label="stir the bowl",
                ).with_goal(ObjectsWithinDistance("spoon", "bowl", max_distance=0.14)),
            ),
            object_names=("bowl", "spoon"),
            episode_length_s=65.0,
            success_label="bowl stirred while held",
        ),
        objects={
            "bowl": "bowl_ycb_robolab",
            "spoon": "spoon_handal_robolab",
        },
        primary_object="bowl",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="spreading_task",
            instruction="Spread the mixture across the plate.",
            instruction_variants=(
                "Use the spatula to spread the topping evenly.",
                "Spread the paste over the surface.",
                "Smooth the mixture with the spatula.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="plate",
                    label="steady the plate",
                ),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="spatula",
                    label="pick up the spatula",
                ),
                Stage(
                    skill=Skill.STRIKE,
                    arm="right",
                    target="spatula",
                    secondary="plate",
                    strike_amplitude=0.10,
                    strike_cycles=3,
                    label="spread across the plate",
                ).with_goal(ObjectsWithinDistance("spatula", "plate", max_distance=0.14)),
            ),
            object_names=("plate", "spatula"),
            episode_length_s=60.0,
            success_label="mixture spread on plate",
        ),
        objects={
            "plate": "clay_plates_hot3d_robolab",
            "spatula": "spatula_hot3d_robolab",
        },
        primary_object="plate",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="scissor_cutting",
            instruction="Cut the material with the scissors.",
            instruction_variants=(
                "Snip the sponge with the scissors.",
                "Use the scissors to cut the material.",
                "Trim the material with the scissors.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="material",
                    label="hold the material",
                ),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="scissors",
                    label="pick up the scissors",
                ),
                Stage(
                    skill=Skill.STRIKE,
                    arm="right",
                    target="scissors",
                    secondary="material",
                    strike_amplitude=0.06,
                    strike_cycles=5,
                    label="cut the material",
                ).with_goal(ObjectsWithinDistance("scissors", "material", max_distance=0.16)),
            ),
            object_names=("material", "scissors"),
            episode_length_s=60.0,
            success_label="material cut with scissors",
        ),
        objects={
            "material": "soft_scrub_ycb_robolab",
            "scissors": "scissors_ycb_robolab",
        },
        primary_object="material",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="scrubbing_plates",
            instruction="Hold the plate and scrub it with the sponge.",
            instruction_variants=(
                "Steady the plate and wipe it clean.",
                "Scrub the plate while holding it in place.",
                "Clean the plate with the sponge.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="plate",
                    label="hold the plate",
                ),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="sponge",
                    label="pick up the sponge",
                ),
                Stage(
                    skill=Skill.STRIKE,
                    arm="right",
                    target="sponge",
                    secondary="plate",
                    strike_amplitude=0.08,
                    strike_cycles=4,
                    label="scrub the plate",
                ).with_goal(ObjectsWithinDistance("sponge", "plate", max_distance=0.12)),
            ),
            object_names=("plate", "sponge"),
            episode_length_s=60.0,
            success_label="plate scrubbed",
        ),
        objects={
            "plate": "clay_plates_hot3d_robolab",
            "sponge": "soft_scrub_ycb_robolab",
        },
        primary_object="plate",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="mash_while_holding",
            instruction="Hold the bowl and mash its contents with the masher.",
            instruction_variants=(
                "Stabilise the bowl and mash with the masher.",
                "Mash the contents while keeping the bowl still.",
                "Use both arms to mash the bowl.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK,
                    arm="left",
                    target="bowl",
                    label="grip the bowl",
                ),
                Stage(
                    skill=Skill.PICK,
                    arm="right",
                    target="masher",
                    label="pick up the masher",
                ),
                Stage(
                    skill=Skill.STRIKE,
                    arm="right",
                    target="masher",
                    secondary="bowl",
                    strike_amplitude=0.10,
                    strike_cycles=5,
                    label="mash the contents",
                ).with_goal(ObjectsWithinDistance("masher", "bowl", max_distance=0.14)),
            ),
            object_names=("bowl", "masher"),
            episode_length_s=65.0,
            success_label="contents mashed while holding bowl",
        ),
        objects={
            "bowl": "wooden_bowl_hot3d_robolab",
            "masher": "potato_masher_hot3d_robolab",
        },
        primary_object="bowl",
    ),
]
