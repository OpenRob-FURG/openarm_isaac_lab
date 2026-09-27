"""Task 45-50: inspection, grading and attribute sorting.

These are the visual tasks. The registry has four robolab blocks in four
distinct colours, plus a range of fruit and vegetables, which is enough to
build colour, shape, count and multi-attribute sorters. The sorting bins are
pinned, so domain randomization moves the items but not the destinations.
"""

from __future__ import annotations

from openarm.arena_environments.industrial._base import TABLE_HEIGHT, IndustrialTask
from openarm.arena_tasks.staged import (
    ObjectInRegion,
    Skill,
    Stage,
    TaskSpec,
)

Z = TABLE_HEIGHT + 0.06
BIN_HALF = (0.10, 0.10, 0.09)


def _dest(name: str, x: float, y: float) -> ObjectInRegion:
    return ObjectInRegion(name, (x, y, Z), BIN_HALF)


TASKS: list[IndustrialTask] = [
    IndustrialTask(
        spec=TaskSpec(
            name="colour_sorting",
            instruction="Sort the four blocks into the bins by colour.",
            instruction_variants=(
                "Put each block in the bin that matches its colour.",
                "Sort the blocks by colour into the four bins.",
                "Match every block to its colour bin.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="red_block",
                    release_height=Z,
                    label="bin the red block",
                ).with_goal(_dest("red_block", 0.28, 0.20)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="green_block",
                    release_height=Z,
                    label="bin the green block",
                ).with_goal(_dest("green_block", 0.44, 0.20)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="yellow_block",
                    release_height=Z,
                    label="bin the yellow block",
                ).with_goal(_dest("yellow_block", 0.28, 0.04)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="blue_block",
                    release_height=Z,
                    label="bin the blue block",
                ).with_goal(_dest("blue_block", 0.44, 0.04)),
            ),
            object_names=("red_block", "green_block", "yellow_block", "blue_block"),
            episode_length_s=80.0,
            success_label="four blocks sorted by colour",
        ),
        objects={
            "red_block": "red_block_basic_robolab",
            "green_block": "green_block_basic_robolab",
            "yellow_block": "yellow_block_basic_robolab",
            "blue_block": "blue_block_basic_robolab",
            "red_bin": "red_container",
            "green_bin": "green_container",
            "yellow_bin": "blue_sorting_bin",
            "blue_bin": "wooden_bowl_hot3d_robolab",
        },
        primary_object="red_block",
        pinned_objects=("red_bin", "green_bin", "yellow_bin", "blue_bin"),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="shape_sorting",
            instruction="Sort the objects by shape into the bins.",
            instruction_variants=(
                "Put the round object in one bin and the boxy objects in the other.",
                "Separate the items by shape.",
                "Sort the sphere and the blocks by shape.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="sphere",
                    release_height=Z,
                    label="bin the sphere",
                ).with_goal(_dest("sphere", 0.30, 0.16)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="cube",
                    release_height=Z,
                    label="bin the cube",
                ).with_goal(_dest("cube", 0.50, 0.16)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="dex_cube",
                    release_height=Z,
                    label="bin the second cube",
                ).with_goal(_dest("dex_cube", 0.30, -0.02)),
            ),
            object_names=("sphere", "cube", "dex_cube"),
            episode_length_s=65.0,
            success_label="objects sorted by shape",
        ),
        objects={
            "sphere": "sphere",
            "cube": "red_cube",
            "dex_cube": "dex_cube",
            "round_bin": "wooden_bowl_hot3d_robolab",
            "box_bin": "blue_sorting_bin",
        },
        primary_object="sphere",
        pinned_objects=("round_bin", "box_bin"),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="multi_attribute_sorting",
            instruction="Sort the fruit by both colour and type.",
            instruction_variants=(
                "Match each fruit to the bin for its type.",
                "Sort the citrus and the non-citrus separately.",
                "Group the fruit by kind into the bins.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="lemon",
                    release_height=Z,
                    label="bin the lemon",
                ).with_goal(_dest("lemon", 0.32, 0.18)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="lime",
                    release_height=Z,
                    label="bin the lime",
                ).with_goal(_dest("lime", 0.32, 0.02)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="apple",
                    release_height=Z,
                    label="bin the apple",
                ).with_goal(_dest("apple", 0.54, 0.18)),
            ),
            object_names=("lemon", "lime", "apple"),
            episode_length_s=70.0,
            success_label="fruit sorted by type",
        ),
        objects={
            "lemon": "lemon_01_fruits_veggies_robolab",
            "lime": "lime01_fruits_veggies_robolab",
            "apple": "apple_02_objaverse_robolab",
            "citrus_bin": "green_container",
            "sweet_bin": "red_container",
        },
        primary_object="lemon",
        pinned_objects=("citrus_bin", "sweet_bin"),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="odd_one_out_reject",
            instruction="Find the item that does not belong and reject it.",
            instruction_variants=(
                "Remove the odd item from the group and bin it.",
                "Reject the one piece that is not like the others.",
                "Pick out the different object and put it in the reject bin.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="odd_item",
                    release_height=Z,
                    label="reject the odd item",
                ).with_goal(_dest("odd_item", 0.56, -0.18)),
            ),
            object_names=("odd_item",),
            episode_length_s=45.0,
            success_label="odd item rejected",
        ),
        objects={
            "group_a": "tomato_soup_can",
            "group_b": "corn_can_hope_robolab",
            "group_c": "green_beans_can_hope_robolab",
            "odd_item": "canned_peaches_hope_robolab",
            "reject_bin": "red_container",
        },
        primary_object="odd_item",
        pinned_objects=("group_a", "group_b", "group_c", "reject_bin"),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="count_and_stack",
            instruction="Count the items, then stack all three of them.",
            instruction_variants=(
                "Stack the three pieces into a tower.",
                "Count the blocks and build a stack of all of them.",
                "Pile the three items on top of each other.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="block_1",
                    release_height=TABLE_HEIGHT + 0.02,
                    label="place the first block",
                ).with_goal(ObjectInRegion("block_1", (0.40, 0.0, TABLE_HEIGHT + 0.03), (0.08, 0.08, 0.06))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="block_2",
                    release_height=TABLE_HEIGHT + 0.12,
                    label="place the second block",
                ).with_goal(ObjectInRegion("block_2", (0.40, 0.0, TABLE_HEIGHT + 0.13), (0.08, 0.08, 0.06))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="block_3",
                    release_height=TABLE_HEIGHT + 0.22,
                    label="place the third block",
                ).with_goal(ObjectInRegion("block_3", (0.40, 0.0, TABLE_HEIGHT + 0.23), (0.08, 0.08, 0.06))),
            ),
            object_names=("block_1", "block_2", "block_3"),
            episode_length_s=70.0,
            success_label="all three items stacked",
        ),
        objects={
            "block_1": "wood_block_ycb_robolab",
            "block_2": "red_cube",
            "block_3": "green_cube",
        },
        primary_object="block_1",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="ripeness_grading",
            instruction="Grade the fruit into the fresh and the reject bins.",
            instruction_variants=(
                "Sort the fruit by ripeness: fresh items to one bin, the rest to the other.",
                "Grade the produce and separate the under-ripe items.",
                "Sort the fruit into good and reject piles.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="ripe",
                    release_height=Z,
                    label="grade the ripe fruit as fresh",
                ).with_goal(_dest("ripe", 0.30, 0.16)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="unripe",
                    release_height=Z,
                    label="grade the unripe fruit as reject",
                ).with_goal(_dest("unripe", 0.54, 0.16)),
            ),
            object_names=("ripe", "unripe"),
            episode_length_s=55.0,
            success_label="fruit graded by ripeness",
        ),
        objects={
            "ripe": "orange_01_fruits_veggies_robolab",
            "unripe": "lime01_fruits_veggies_robolab",
            "fresh_bin": "green_container",
            "reject_bin": "red_container",
        },
        primary_object="ripe",
        pinned_objects=("fresh_bin", "reject_bin"),
    ),
]
