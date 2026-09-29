"""Task 29-35: containers, packaging and stacking.

Reuses the ``red_container``/``green_container``/``blue_sorting_bin`` family and
the cans and cartons, and covers the structural packing skills: lid cycles,
transfer between containers, and building or dismantling stacks.
"""

from __future__ import annotations

from openarm.arena_environments.industrial._base import TABLE_HEIGHT, IndustrialTask
from openarm.arena_tasks.staged import (
    All,
    ObjectInRegion,
    ObjectRelativeTo,
    ObjectUpright,
    Skill,
    Stage,
    TaskSpec,
)

Z = TABLE_HEIGHT + 0.06
BIN_HALF = (0.12, 0.12, 0.10)

TASKS: list[IndustrialTask] = [
    IndustrialTask(
        spec=TaskSpec(
            name="container_transfer",
            instruction="Transfer the item from the red container into the green one.",
            instruction_variants=(
                "Move the object out of the red bin and into the green bin.",
                "Take the part from the red container and drop it in the green container.",
                "Relocate the item between the two containers.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="item",
                    grasp_offset=(0.0, 0.0, 0.06),
                    release_height=Z,
                    max_steps=1500,
                    label="move the item to the green container",
                ).with_goal(ObjectInRegion("item", (0.55, 0.02, Z), BIN_HALF)),
            ),
            object_names=("item",),
            episode_length_s=45.0,
            success_label="item transferred between containers",
        ),
        objects={
            "red_bin": "red_container",
            "green_bin": "green_container",
            "item": "sweet_potato",
        },
        primary_object="item",
        pinned_objects=("red_bin", "green_bin"),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="can_tower_build",
            instruction="Stack all three cans into a tower.",
            instruction_variants=(
                "Build a three-can stack.",
                "Pile the cans on top of each other in a tower.",
                "Stack the cans one on top of another.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="can_1",
                    grasp_offset=(0.1, 0.0, 0.0),
                    release_height=TABLE_HEIGHT + 0.02,
                    label="place the first can",
                ).with_goal(ObjectInRegion("can_1", (0.32, 0.0, TABLE_HEIGHT + 0.03), (0.08, 0.08, 0.06))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="can_2",
                    grasp_offset=(0.1, 0.0, 0.0),
                    release_height=TABLE_HEIGHT + 0.16,
                    label="place the second can",
                ).with_goal(ObjectRelativeTo("can_2", "can_1", (0.0, 0.0, 0.12), tolerance=0.1)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="can_3",
                    grasp_offset=(0.1, 0.0, 0.0),
                    release_height=TABLE_HEIGHT + 0.28,
                    label="place the third can",
                ).with_goal(ObjectRelativeTo("can_3", "can_2", (0.0, 0.0, 0.12), tolerance=0.1)),
            ),
            object_names=("can_1", "can_2", "can_3"),
            episode_length_s=75.0,
            success_label="three-can tower built",
        ),
        objects={
            "can_1": "corn_can_hope_robolab",
            "can_2": "green_beans_can_hope_robolab",
            "can_3": "canned_tuna_hope_robolab",
        },
        primary_object="can_3",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="unstack_restack",
            instruction="Unstack the tower and restack the cans in reverse order.",
            instruction_variants=(
                "Take the stack apart, then rebuild it back to front.",
                "Reverse the stack: top can goes to the bottom.",
                "Unstack the cans and rebuild the pile the other way up.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="can_1",
                    release_height=TABLE_HEIGHT + 0.02,
                    label="move the bottom can aside",
                ).with_goal(ObjectInRegion("can_1", (0.24, -0.14, TABLE_HEIGHT + 0.03), (0.08, 0.08, 0.06))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="can_3",
                    release_height=TABLE_HEIGHT + 0.02,
                    label="set the old top can down first",
                ).with_goal(ObjectInRegion("can_3", (0.44, -0.02, TABLE_HEIGHT + 0.03), (0.08, 0.08, 0.06))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="can_2",
                    release_height=TABLE_HEIGHT + 0.16,
                    label="stack the middle can on top",
                ).with_goal(ObjectRelativeTo("can_2", "can_3", (0.0, 0.0, 0.12), tolerance=0.05)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="can_1",
                    release_height=TABLE_HEIGHT + 0.28,
                    label="cap the reversed stack",
                ).with_goal(ObjectRelativeTo("can_1", "can_2", (0.0, 0.0, 0.12), tolerance=0.05)),
            ),
            object_names=("can_1", "can_2", "can_3"),
            episode_length_s=90.0,
            success_label="stack reversed",
        ),
        objects={
            "can_1": "corn_can_hope_robolab",
            "can_2": "green_beans_can_hope_robolab",
            "can_3": "canned_tuna_hope_robolab",
        },
        primary_object="can_1",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="bin_dump_and_clear",
            instruction="Empty the container's contents into the bin.",
            instruction_variants=(
                "Dump everything from the container into the sorting bin.",
                "Empty the container into the bin.",
                "Clear the container into the big bin.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="item_1",
                    release_height=Z,
                    label="dump the first item",
                ).with_goal(ObjectInRegion("item_1", (0.65, 0.02, Z), (0.13, 0.13, 0.10))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="item_2",
                    release_height=Z,
                    label="dump the second item",
                ).with_goal(ObjectInRegion("item_2", (0.65, 0.02, Z), (0.13, 0.13, 0.10))),
            ),
            object_names=("item_1", "item_2"),
            episode_length_s=55.0,
            success_label="container emptied into bin",
        ),
        objects={
            "origin_tote": "storage_box_hot3d_robolab",
            "sorting_bin": "openarm_sorting_bin",
            "item_1": "broccoli",
            "item_2": "lemon_01_fruits_veggies_robolab",
        },
        primary_object="item_1",
        pinned_objects=("origin_tote", "sorting_bin"),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="carton_handling",
            instruction="Move the carton into the bin and keep it upright.",
            instruction_variants=(
                "Place the carton in the bin without tipping it over.",
                "Bin the carton, keeping it vertical.",
                "Set the carton upright in the container.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="carton",
                    release_height=Z,
                    label="place the carton in the bin",
                ).with_goal(
                    All(
                        goals=(
                            ObjectInRegion("carton", (0.55, 0.10, Z), BIN_HALF),
                            ObjectUpright("carton", tolerance=0.4),
                        )
                    )
                ),
            ),
            object_names=("carton",),
            episode_length_s=45.0,
            success_label="carton binned upright",
        ),
        objects={
            "carton": "milk_carton_hope_robolab",
            "bin": "red_container",
        },
        primary_object="carton",
        pinned_objects=("bin",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="double_container_load",
            instruction="Load both containers with their matching items.",
            instruction_variants=(
                "Put the red item in the red container and the green item in the green one.",
                "Sort the two items into the two containers by color.",
                "Load each container with its own item.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="red_item",
                    release_height=Z,
                    label="load the red container",
                ).with_goal(ObjectInRegion("red_item", (0.65, 0.34, Z), BIN_HALF)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="green_item",
                    release_height=Z,
                    label="load the green container",
                ).with_goal(ObjectInRegion("green_item", (0.65, 0.02, Z), BIN_HALF)),
            ),
            object_names=("red_item", "green_item"),
            episode_length_s=60.0,
            success_label="both containers loaded",
        ),
        objects={
            "red_bin": "red_container",
            "green_bin": "green_container",
            "red_item": "red_block_basic_robolab",
            "green_item": "green_block_basic_robolab",
        },
        primary_object="red_item",
        pinned_objects=("red_bin", "green_bin"),
    ),
]
