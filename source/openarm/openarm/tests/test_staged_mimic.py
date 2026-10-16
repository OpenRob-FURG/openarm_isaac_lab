"""Static checks of the generated Mimic segmentation for all 50 staged tasks.

``openarm.arena_tasks.staged.mimic`` turns a :class:`TaskSpec` into a Mimic
datagen config, which means the subtask split a datagen run executes is derived
rather than written out. A derived split is only safe if the properties the
Mimic datagen relies on are actually true of all 50 of them, so those properties
are asserted here over every task instead of over a hand-built example:

* every end effector the embodiment commands has a **non-empty** subtask list,
  which the datagen requires and which an arm that never works would otherwise
  violate;
* each list's **last** segment terminates on ``None`` with zero offsets, because
  there is no signal after the end of the episode to shift;
* the signals a list terminates on are exactly the stages that **have a goal**,
  named by :func:`signal_name` -- the same function the mimic environment's
  :meth:`get_subtask_term_signals` uses, so the config and the recorder agree;
* the arms named are the ones the **resolved** spec assigns, so the split cannot
  be built from the authored arms the layout overrides.

These run without Isaac Sim: the segmentation is pure data over the task table.

Run with ``pytest source/openarm/openarm/tests/test_staged_mimic.py``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the simulator stubs from the sibling task-table tests so both files agree
# on how ``isaaclab`` and ``isaaclab_arena`` are faked.
from test_industrial_tasks import _install_stubs  # noqa: E402

_install_stubs()

from openarm.arena_tasks.staged.arms import ARMS  # noqa: E402
from openarm.arena_tasks.staged.mimic import (  # noqa: E402
    active_arms,
    build_subtask_configs,
    signal_name,
)


class _ArmMode:
    """Stand-in for Arena's ``ArmMode``.

    The real enum is stubbed out for these tests, and
    :func:`active_arms` reads ``arm_mode.value`` rather than comparing against
    enum members, so this carries everything it needs.
    """

    def __init__(self, value: str) -> None:
        self.value = value


@pytest.fixture(scope="module")
def tasks():
    return importlib.import_module("openarm.arena_environments.industrial").INDUSTRIAL_TASKS


@pytest.fixture(scope="module")
def resolved(tasks):
    return {key: task.resolve_arms() for key, task in tasks.items()}


@pytest.fixture(scope="module")
def segmented(resolved):
    return {
        key: build_subtask_configs(task.spec, _ArmMode("dual_arm"), next(iter(task.objects)))
        for key, task in resolved.items()
    }


def test_every_task_generates_a_config(resolved):
    for key, task in resolved.items():
        configs = build_subtask_configs(task.spec, _ArmMode("dual_arm"), next(iter(task.objects)))
        assert set(configs) == set(ARMS), f"{key}: expected configs for both arms, got {sorted(configs)}"


def test_every_arm_has_at_least_one_segment(segmented):
    """The datagen rejects an empty list, so an idle arm needs a static segment.

    ``dual_mug_serve`` is the case that matters: the right arm is commanded but
    never works, and its list has to be non-empty all the same.
    """
    for key, configs in segmented.items():
        for arm, segments in configs.items():
            assert segments, f"{key}/{arm}: no subtask segments for a commanded arm"
            assert all(segment.object_ref for segment in segments), f"{key}/{arm}: a segment has no object_ref"


def test_the_last_segment_of_each_arm_terminates_on_nothing(segmented):
    for key, configs in segmented.items():
        for arm, segments in configs.items():
            last = segments[-1]
            assert last.subtask_term_signal is None, f"{key}/{arm}: last segment signals {last.subtask_term_signal!r}"
            assert tuple(last.subtask_term_offset_range) == (0, 0), f"{key}/{arm}: last segment has an offset"


def test_no_segment_terminates_twice_on_the_same_signal(segmented):
    """A signal closes exactly one segment, so a name cannot appear twice.

    Names are keyed by stage index, which makes a collision impossible unless a
    stage were counted twice -- but a repeated signal would silently split an
    episode at two steps and only show up as a datagen run that does not
    reproduce.
    """
    for key, configs in segmented.items():
        names = [segment.subtask_term_signal for segments in configs.values() for segment in segments]
        named = [name for name in names if name is not None]
        assert len(named) == len(set(named)), f"{key}: repeated signal in {named}"


def test_signals_are_exactly_the_stages_that_have_a_goal(segmented, resolved):
    """The split is defined by goals, not by stage count.

    A ``MOVE`` or ``DOCK`` stage carries ``goal=None``: it is a means rather than
    an end. Emitting a signal for one would produce a signal the annotation
    recorder can never see go high, and ``annotate_demos.py`` rejects an episode
    with a signal that never fires.
    """
    for key, configs in segmented.items():
        spec = resolved[key].spec
        expected = {
            signal_name(index, stage.arm)
            for index, stage in enumerate(spec.stages)
            if stage.goal is not None
        }
        produced = {
            segment.subtask_term_signal
            for segments in configs.values()
            for segment in segments
            if segment.subtask_term_signal is not None
        }
        assert produced == expected, f"{key}: signals {sorted(produced)} != goal stages {sorted(expected)}"


def test_each_segment_anchors_to_its_own_stage_target(segmented, resolved):
    for key, configs in segmented.items():
        spec = resolved[key].spec
        by_signal = {
            segment.subtask_term_signal: segment
            for segments in configs.values()
            for segment in segments
            if segment.subtask_term_signal is not None
        }
        for index, stage in enumerate(spec.stages):
            if stage.goal is None:
                continue
            segment = by_signal[signal_name(index, stage.arm)]
            assert segment.object_ref == (stage.target or next(iter(resolved[key].objects))), (
                f"{key}: stage {index} is anchored to {segment.object_ref!r}"
            )


def test_segments_are_ordered_by_stage_index(segmented):
    """A segment's position has to match the order its stages run in.

    The datagen walks an arm's list in order, so an out-of-order signal would
    invert the task for that arm even though every name is individually correct.
    """
    for key, configs in segmented.items():
        for arm, segments in configs.items():
            indices = [
                int(name.split("_")[1])
                for name in (segment.subtask_term_signal for segment in segments)
                if name is not None
            ]
            assert indices == sorted(indices), f"{key}/{arm}: signals out of order: {indices}"


def test_an_arm_that_stops_early_ends_in_a_hold_on_its_own_last_object(segmented):
    """An arm whose stages end before the task does gets a trailing hold.

    ``colour_sorting`` is the case: two stages per arm, interleaved, so the left
    arm is idle for the whole second half. Its final ``None`` segment is what
    carries that arm through the other arm's stages.
    """
    task = segmented["colour_sorting"]
    assert len(task["left"]) == 3, "colour_sorting: left arm should have two stages and a hold"
    assert len(task["right"]) == 3
    assert task["left"][-1].object_ref == "green_block"
    assert task["right"][-1].object_ref == "blue_block"


def test_an_entirely_idle_arm_gets_one_static_segment(segmented):
    """A commanded arm that never works still needs a non-empty list."""
    task = segmented["dual_mug_serve"]
    assert len(task["right"]) == 1, "dual_mug_serve: an idle arm should get a single segment"
    idle = task["right"][0]
    assert idle.subtask_term_signal is None
    assert tuple(idle.subtask_term_offset_range) == (0, 0)


def test_a_handoff_splits_across_arms(segmented, resolved):
    """The handoff tasks are the only ones whose two stages share an object.

    Both stages name the same object, so its side cannot name both arms: the
    receiver comes from the spec. The generated split has to reflect that, or the
    datagen would try to solve one arm to a pose on the other arm's side of the
    table.
    """
    for key in ("mug_handoff_dispense", "cross_hand_tool_handoff"):
        arms = [stage.arm for stage in resolved[key].spec.stages]
        assert arms[0] != arms[1], f"{key}: a handoff must cross arms, got {arms}"

        giver, receiver = arms[0], arms[1]
        # The receiving arm always has the handoff's goal, so it always has a
        # segment that terminates on a signal. The giving arm only has one if its
        # own stage is an end rather than a means: ``cross_hand_tool_handoff``
        # picks the tool up without a goal, because handing it over is the point.
        assert segmented[key][receiver][0].subtask_term_signal is not None, f"{key}/{receiver}: no handoff segment"
        assert segmented[key][receiver][-1].subtask_term_signal is None
        assert segmented[key][giver], f"{key}/{giver}: empty list"

    # The mug task's pick does have a goal, so both arms work and both lists split.
    assert len(segmented["mug_handoff_dispense"]["left"]) == 2


def test_arm_mode_selects_which_arms_get_configs(resolved):
    """Only the arms the embodiment commands may appear.

    The action manager drops an absent arm's terms, so a config naming it would
    have the datagen command end effectors that do not exist.
    """
    task = resolved["dual_mug_serve"]
    assert active_arms(task.spec, _ArmMode("dual_arm")) == ARMS
    assert active_arms(task.spec, _ArmMode("left")) == ("left",)
    assert active_arms(task.spec, _ArmMode("right")) == ("right",)


def test_an_arm_mode_that_cannot_reach_a_stage_is_rejected(resolved):
    """A single-arm run of a task that needs both arms fails loudly.

    Silently dropping the other arm's stages would generate demonstrations of a
    task the robot cannot perform.
    """
    task = resolved["colour_sorting"]
    with pytest.raises(ValueError, match="does not command"):
        build_subtask_configs(task.spec, _ArmMode("left"), next(iter(task.objects)))


def test_single_arm_mode_follows_the_task_rather_than_defaulting(resolved):
    """``SINGLE_ARM`` keeps the arm the task actually uses.

    Defaulting to the right would silently pick the wrong arm for the left-handed
    tasks, which is most of them.
    """
    left_led = resolved["dual_mug_serve"]
    assert active_arms(left_led.spec, _ArmMode("single_arm")) == ("left",)
    right_led = resolved["thread_nut_on_bolt"]
    assert active_arms(right_led.spec, _ArmMode("single_arm")) == ("right",)