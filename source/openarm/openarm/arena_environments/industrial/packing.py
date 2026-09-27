"""Task 11-20: packing, kitting and restocking.

The registry is rich in groceries, so this family leans on it heavily: canned
goods, bottles, boxes, produce and the ``lunchbag``/``storage_box`` receptacles
that the other tasks reuse as containers.
"""

from __future__ import annotations

from openarm.arena_environments.industrial._base import TABLE_HEIGHT, IndustrialTask
from openarm.arena_tasks.staged import (
    ObjectInRegion,
    ObjectRelativeTo,
    Skill,
    Stage,
    TaskSpec,
)

Z = TABLE_HEIGHT + 0.06


def _bin(name: str, x: float, y: float) -> ObjectInRegion:
    return ObjectInRegion(name, (x, y, Z), (0.11, 0.11, 0.10))


TASKS: list[IndustrialTask] = [
    IndustrialTask(
        spec=TaskSpec(
            name="lunchbag_packing",
            instruction="Pack the three food items into the lunchbag.",
            instruction_variants=(
                "Put all three items in the lunchbag.",
                "Fill the lunchbag with the fruit, the bar and the can.",
                "Pack a lunch: bag, then all three items inside.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="apple",
                    release_height=Z,
                    label="put the apple in the bag",
                ).with_goal(_bin("apple", 0.52, 0.20)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="snickers",
                    release_height=Z,
                    label="put the bar in the bag",
                ).with_goal(_bin("snickers", 0.52, 0.20)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="soup_can",
                    release_height=Z,
                    label="put the can in the bag",
                ).with_goal(_bin("soup_can", 0.52, 0.20)),
            ),
            object_names=("apple", "snickers", "soup_can"),
            episode_length_s=70.0,
            success_label="three items packed in lunchbag",
        ),
        objects={
            "lunchbag": "lunchbag_objaverse_robolab",
            "apple": "apple_01_objaverse_robolab",
            "snickers": "snickers_bar_objaverse_robolab",
            "soup_can": "alphabet_soup_can_hope_robolab",
        },
        primary_object="apple",
        pinned_objects=("lunchbag",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="can_sorting_by_size",
            instruction="Sort the cans into the bins by size.",
            instruction_variants=(
                "Put the large can in the large bin and the small can in the small bin.",
                "Sort these two cans by size.",
                "Place each can in the matching size bin.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="large_can",
                    release_height=Z,
                    label="bin the large can",
                ).with_goal(_bin("large_can", 0.55, -0.15)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="small_can",
                    release_height=Z,
                    label="bin the small can",
                ).with_goal(_bin("small_can", 0.28, -0.15)),
            ),
            object_names=("large_can", "small_can"),
            episode_length_s=55.0,
            success_label="cans sorted by size",
        ),
        objects={
            "large_can": "soup_can_hot3d_robolab",
            "small_can": "corn_can_hope_robolab",
            "large_bin": "blue_sorting_bin",
            "small_bin": "green_container",
        },
        primary_object="large_can",
        pinned_objects=("large_bin", "small_bin"),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="snack_box_kitting",
            instruction="Kit the snacks into the storage box.",
            instruction_variants=(
                "Fill the storage box with the snack items.",
                "Load the popcorn, the bars and the cookies into the box.",
                "Assemble the snack kit in the box.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="popcorn",
                    release_height=Z,
                    label="load the popcorn",
                ).with_goal(_bin("popcorn", 0.55, 0.18)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="granola",
                    release_height=Z,
                    label="load the granola bars",
                ).with_goal(_bin("granola", 0.55, 0.18)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="cookies",
                    release_height=Z,
                    label="load the cookies",
                ).with_goal(_bin("cookies", 0.55, 0.18)),
            ),
            object_names=("popcorn", "granola", "cookies"),
            episode_length_s=70.0,
            success_label="snack kit assembled in box",
        ),
        objects={
            "snack_box": "storage_box_hot3d_robolab",
            "popcorn": "popcorn_box_hope_robolab",
            "granola": "granola_bars_hope_robolab",
            "cookies": "oatmeal_raisin_cookies_hope_robolab",
        },
        primary_object="popcorn",
        pinned_objects=("snack_box",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="bagel_snack_assembly",
            instruction="Put the bagel on the plate and add the spread.",
            instruction_variants=(
                "Plate the bagel and set the cream cheese beside it.",
                "Place the bagel on the plate, then the spread.",
                "Assemble the bagel snack on the plate.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="bagel",
                    release_height=Z,
                    label="plate the bagel",
                ).with_goal(_bin("bagel", 0.45, 0.10)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="cream_cheese",
                    release_height=Z,
                    label="add the cream cheese",
                ).with_goal(ObjectRelativeTo("cream_cheese", "bagel", (0.06, 0.0, 0.02), tolerance=0.09)),
            ),
            object_names=("bagel", "cream_cheese"),
            episode_length_s=50.0,
            success_label="bagel plated with spread",
        ),
        objects={
            "plate": "clay_plates_hot3d_robolab",
            "bagel": "bagel_00_objaverse_robolab",
            "cream_cheese": "cream_cheese_hope_robolab",
        },
        primary_object="bagel",
        pinned_objects=("plate",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="bottle_destuffing",
            instruction="Move the bottles into the crate.",
            instruction_variants=(
                "Put both bottles in the crate.",
                "Destuff the bottles into the box.",
                "Load the two bottles into the container.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="ketchup",
                    release_height=Z,
                    label="load the ketchup bottle",
                ).with_goal(_bin("ketchup", 0.55, 0.05)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="mayo",
                    release_height=Z,
                    label="load the mayonnaise bottle",
                ).with_goal(_bin("mayo", 0.55, 0.05)),
            ),
            object_names=("ketchup", "mayo"),
            episode_length_s=55.0,
            success_label="bottles loaded into crate",
        ),
        objects={
            "crate": "red_container",
            "ketchup": "ketchup_bottle_hope_robolab",
            "mayo": "mayonnaise_bottle_hope_robolab",
        },
        primary_object="ketchup",
        pinned_objects=("crate",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="case_palletizing",
            instruction="Stack the two cases into a stable tower.",
            instruction_variants=(
                "Palletize the cases: one on top of the other.",
                "Stack both cases so the pile stands upright.",
                "Build a two-case stack.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="bottom_case",
                    release_height=TABLE_HEIGHT + 0.02,
                    label="set the bottom case",
                ).with_goal(ObjectInRegion("bottom_case", (0.42, -0.05, TABLE_HEIGHT + 0.03), (0.12, 0.12, 0.06))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="top_case",
                    release_height=TABLE_HEIGHT + 0.16,
                    label="stack the top case",
                ).with_goal(
                    ObjectRelativeTo("top_case", "bottom_case", (0.0, 0.0, 0.13), tolerance=0.06)
                ),
            ),
            object_names=("bottom_case", "top_case"),
            episode_length_s=60.0,
            success_label="cases stacked into a tower",
        ),
        objects={
            "bottom_case": "sugar_box_ycb_robolab",
            "top_case": "cracker_box",
        },
        primary_object="top_case",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="visual_qc_reject",
            instruction="Inspect the parts and reject the damaged one into the reject bin.",
            instruction_variants=(
                "Find the broken part and put it in the reject bin.",
                "Quality check: send the defective item to the reject bin.",
                "Move the faulty part off the line into the reject container.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="defective",
                    release_height=Z,
                    label="reject the defective part",
                ).with_goal(_bin("defective", 0.55, -0.20)),
                Stage(
                    skill=Skill.STABILIZE,
                    arm="left",
                    settle_steps=6,
                    label="hold the line clear",
                ),
            ),
            object_names=("defective",),
            episode_length_s=45.0,
            success_label="defective part rejected",
        ),
        objects={
            "defective": "wood_block_ycb_robolab",
            "good_part": "clamp_ycb_robolab",
            "reject_bin": "red_container",
        },
        primary_object="defective",
        pinned_objects=("reject_bin", "good_part"),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="shelf_restock",
            instruction="Restock the shelf with both items.",
            instruction_variants=(
                "Place both products on the shelf.",
                "Refill the shelf with the can and the box.",
                "Stock the two items on the shelf.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="shelf_can",
                    release_height=0.32,
                    label="restock the can",
                ).with_goal(ObjectInRegion("shelf_can", (0.40, 0.26, 0.34), (0.10, 0.10, 0.08))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="shelf_box",
                    release_height=0.32,
                    label="restock the box",
                ).with_goal(ObjectInRegion("shelf_box", (0.54, 0.26, 0.34), (0.10, 0.10, 0.08))),
            ),
            object_names=("shelf_can", "shelf_box"),
            episode_length_s=55.0,
            success_label="shelf restocked",
        ),
        objects={
            "shelf_can": "coffee_can_ycb_robolab",
            "shelf_box": "sugar_box_ycb_robolab",
        },
        primary_object="shelf_can",
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="meal_kit_portioning",
            instruction="Portion the ingredients into the bowl.",
            instruction_variants=(
                "Put both vegetables in the bowl.",
                "Fill the bowl with the broccoli and the potato.",
                "Assemble the meal kit in the bowl.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="broccoli",
                    release_height=Z,
                    label="add the broccoli",
                ).with_goal(_bin("broccoli", 0.48, 0.16)),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="potato",
                    release_height=Z,
                    label="add the sweet potato",
                ).with_goal(_bin("potato", 0.48, 0.16)),
            ),
            object_names=("broccoli", "potato"),
            episode_length_s=55.0,
            success_label="meal kit portioned into bowl",
        ),
        objects={
            "bowl": "wooden_bowl_hot3d_robolab",
            "broccoli": "broccoli",
            "potato": "sweet_potato",
        },
        primary_object="broccoli",
        pinned_objects=("bowl",),
    ),
    IndustrialTask(
        spec=TaskSpec(
            name="frozen_goods_restock",
            instruction="Restock the frozen goods into the freezer bin.",
            instruction_variants=(
                "Load the frozen items into the bin.",
                "Put both frozen packages in the bin.",
                "Restock the freezer with the waffles and the vegetables.",
            ),
            stages=(
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="right",
                    target="waffles",
                    release_height=TABLE_HEIGHT + 0.04,
                    label="load the waffles",
                ).with_goal(ObjectInRegion("waffles", (0.55, 0.14, TABLE_HEIGHT + 0.06), (0.12, 0.12, 0.10))),
                Stage(
                    skill=Skill.PICK_PLACE,
                    arm="left",
                    target="frozen_veg",
                    release_height=TABLE_HEIGHT + 0.04,
                    label="load the frozen vegetables",
                ).with_goal(
                    ObjectInRegion("frozen_veg", (0.55, 0.14, TABLE_HEIGHT + 0.06), (0.12, 0.12, 0.10))
                ),
            ),
            object_names=("waffles", "frozen_veg"),
            episode_length_s=55.0,
            success_label="frozen goods restocked",
        ),
        objects={
            "freezer_bin": "blue_sorting_bin",
            "waffles": "frozen_waffles_hot3d_robolab",
            "frozen_veg": "frozen_vegetable_block_hot3d_robolab",
        },
        primary_object="waffles",
        pinned_objects=("freezer_bin",),
    ),
]
