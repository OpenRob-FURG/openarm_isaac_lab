"""Static checks of the joint-space pose/action conversion the OpenArm needs.

``openarm.embodiments.openarm_bimanual.kinematics`` converts between the
Cartesian quantities the Mimic API speaks and the absolute joint positions the
OpenArm is commanded in. Both conversions run against a live articulation, so what
is checked here is the part that can be decided without one: that the action
layout is read from the action manager rather than assumed, and that it is read
correctly for each arm mode.

That last part is the point of the file. The action layout is not a constant --
dropping the left arm's terms for ``ArmMode.RIGHT`` moves the right arm's
channels from offset 8 to offset 0 -- so an index computed once and reused would
silently write right-arm joints into left-arm slots in half the arm modes. And the
two action spaces over this robot are the same width, so a discriminator based on
the number of action values cannot work at all.

Run with ``pytest source/openarm/openarm/tests/test_staged_kinematics.py``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_industrial_tasks import _install_stubs  # noqa: E402

_install_stubs()

from openarm.policies.oracle.skills import _is_joint_action_space  # noqa: E402


def _load_kinematics():
    """Import ``kinematics`` without running the embodiment package's ``__init__``.

    ``openarm.embodiments.openarm_bimanual.__init__`` eagerly imports the
    embodiment, which needs a simulator. The module under test does not -- it
    only reaches for ``isaaclab.utils``, which is stubbed here -- and both of its
    callers import it lazily for the same reason. Loading it by path is what lets
    the pure-python half of it be tested at all.
    """
    path = Path(__file__).resolve().parents[1] / "embodiments" / "openarm_bimanual" / "kinematics.py"
    spec = importlib.util.spec_from_file_location("openarm_bimanual_kinematics_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


kinematics = _load_kinematics()


class _FakeJointPositionAction:
    """An action term class, named as :class:`JointPositionAction` is."""

    pass


class _FakeDifferentialInverseKinematicsAction:
    """Named as :class:`DifferentialInverseKinematicsAction` is."""

    pass


def _action_manager(terms: dict[str, tuple[type, int]]) -> SimpleNamespace:
    """A stand-in carrying only what the two readers touch."""
    return SimpleNamespace(
        _terms={
            name: SimpleNamespace(_cfg=SimpleNamespace(class_type=cfg)) for name, (cfg, _w) in terms.items()
        },
        active_terms=list(terms),
        action_term_dim=[width for _cfg, width in terms.values()],
        total_action_dim=sum(width for _cfg, width in terms.values()),
    )


DUAL = {
    "left_arm_action": (_FakeJointPositionAction, 7),
    "left_hand_action": (_FakeJointPositionAction, 1),
    "right_arm_action": (_FakeJointPositionAction, 7),
    "right_hand_action": (_FakeJointPositionAction, 1),
}
RIGHT_ONLY = {
    "right_arm_action": (_FakeJointPositionAction, 7),
    "right_hand_action": (_FakeJointPositionAction, 1),
}
LEFT_ONLY = {
    "left_arm_action": (_FakeJointPositionAction, 7),
    "left_hand_action": (_FakeJointPositionAction, 1),
}
IK_DUAL = {
    "left_arm_action": (_FakeDifferentialInverseKinematicsAction, 7),
    "right_arm_action": (_FakeDifferentialInverseKinematicsAction, 7),
}


def _env(terms: dict) -> SimpleNamespace:
    return SimpleNamespace(action_manager=_action_manager(terms), device="cpu", scene=None)


def test_dual_arm_layout_reads_eight_channels_per_arm():
    slices = kinematics.action_slices(_env(DUAL))
    assert slices == {
        "left": {"arm": slice(0, 7), "finger": slice(7, 8)},
        "right": {"arm": slice(8, 15), "finger": slice(15, 16)},
    }


def test_dropping_the_left_arm_moves_the_right_arm_to_offset_zero():
    """The layout is per arm mode, not a constant.

    A hardcoded offset-8 for the right arm would write right-arm joints into the
    right arm's channels correctly under DUAL_ARM and completely wrongly under
    ArmMode.RIGHT.
    """
    slices = kinematics.action_slices(_env(RIGHT_ONLY))
    assert slices == {"right": {"arm": slice(0, 7), "finger": slice(7, 8)}}
    assert "left" not in slices


def test_the_finger_term_does_not_overwrite_the_arm_range():
    """The one-wide finger term follows the seven-wide arm term.

    Both are ``JointPositionActionCfg`` and both are named ``<arm>_...``, so a
    prefix match that assigned either would let the finger's single channel
    replace the arm's seven.
    """
    slices = kinematics.action_slices(_env(DUAL))
    for arm, base in (("left", 0), ("right", 8)):
        assert slices[arm]["arm"].stop - slices[arm]["arm"].start == 7
        assert slices[arm]["finger"].stop - slices[arm]["finger"].start == 1
        assert slices[arm]["arm"] == slice(base, base + 7)
        assert slices[arm]["finger"] == slice(base + 7, base + 8)


def test_active_arms_follows_the_declared_terms():
    assert kinematics.active_arms(_env(DUAL)) == ["left", "right"]
    assert kinematics.active_arms(_env(RIGHT_ONLY)) == ["right"]
    assert kinematics.active_arms(_env(LEFT_ONLY)) == ["left"]


def test_an_action_space_with_no_terms_is_an_error():
    """Reading the layout needs the action manager, and saying so beats guessing."""
    with pytest.raises(ValueError, match="no action terms"):
        kinematics.action_slices(_env({}))
    with pytest.raises(ValueError, match="no action terms"):
        kinematics.active_arms(_env({}))


def test_an_arm_missing_its_finger_term_is_an_error():
    """Half an arm's terms would otherwise silently produce a short action."""
    with pytest.raises(ValueError, match="No joint action terms"):
        kinematics.action_slices(_env({"left_arm_action": (_FakeJointPositionAction, 7)}))


def test_joint_space_is_told_apart_from_pose_space_by_the_term_class():
    """Both spaces are 16 wide here, so width cannot decide it.

    A pose action is also seven values per arm, and with a gripper channel the two
    layouts coincide exactly. Only the declared term class distinguishes them, and
    the term *name* deliberately does not: ``left_arm_action`` reads like a pose
    command and is actually the joint one.
    """
    assert _is_joint_action_space(_env(DUAL)) is True
    assert _is_joint_action_space(_env(IK_DUAL)) is False


def test_the_discriminator_reads_the_class_not_the_name():
    """A joint term named for a pose must still read as joint space."""
    misnamed = {
        "left_inverse_kinematics_action": (_FakeJointPositionAction, 7),
        "right_pose_action": (_FakeJointPositionAction, 7),
    }
    assert _is_joint_action_space(_env(misnamed)) is True


def test_an_unrecognizable_action_space_defaults_to_joint_space():
    """The shipped OpenArm space is joint, so an unknown one is treated as joint.

    Falling the other way would have the oracle read absolute joint targets as
    Cartesian deltas, and the resulting poses would be nonsense rather than
    obviously wrong.
    """
    unknown = {"thing": (SimpleNamespace, 3)}
    assert _is_joint_action_space(_env(unknown)) is True