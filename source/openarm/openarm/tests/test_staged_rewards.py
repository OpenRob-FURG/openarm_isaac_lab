"""Static checks of the staged dense reward terms and their agreement with success.

``openarm.arena_tasks.staged.rewards`` is a continuous restatement of the
boolean vocabulary in ``terminations``: one task declaration has to drive both
what ends an episode and what the policy is paid for on the way there. These
tests pin down the property that makes that safe --

    dense score == 1.0   if and only if   the boolean predicate holds

-- and walk all 50 industrial tasks to prove no goal escapes it. They run
without Isaac Sim: the reward functions only ever reach for a scene object, the
environment origins and the device, so a stand-in environment exercises the real
code paths.

Run with ``pytest source/openarm/openarm/tests/test_staged_rewards.py``.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

# Reuse the simulator stubs from the sibling task-table tests so both files
# agree on how ``isaaclab`` and ``isaaclab_arena`` are faked.
from test_industrial_tasks import _install_stubs  # noqa: E402

_install_stubs()

NUM_ENVS = 4

#: can at (0.41, 0, 0.36), tilted 0.2 rad about X, moving at 0.02 m/s and
#: 0.1 rad/s; bin at (0.70, 0.30, 0.35); lid at (0.55, 0.15, 0.40); the robot's
#: single normalized joint sits at 0.8. Every goal kind has at least one
#: satisfying and one failing configuration built from these.
CAN = (0.41, 0.0, 0.36)
BIN = (0.70, 0.30, 0.35)
LID = (0.55, 0.15, 0.40)
JOINT_VALUE = 0.8
JOINT_RANGE = (0.0, 1.0)


def _quat_from_tilt(tilt: float) -> list[float]:
    """A ``(w, x, y, z)`` rotation of ``tilt`` rad about X.

    About X specifically: a rotation about Z leaves the object's own +Z pointing
    straight up, so a Z-rotation cannot express a tilt at all and would make
    every uprightness check vacuously pass.
    """
    import math

    return [math.cos(tilt / 2.0), math.sin(tilt / 2.0), 0.0, 0.0]


class _RigidData:
    """``RigidObjectData`` stand-in: world pose and velocity for one object."""

    def __init__(self, position, speed=0.0, angular_speed=0.0, tilt=0.0):
        self.root_pos_w = torch.tensor(position, dtype=torch.float32).repeat(NUM_ENVS, 1)
        self.root_quat_w = torch.tensor(_quat_from_tilt(tilt), dtype=torch.float32).repeat(NUM_ENVS, 1)
        self.root_lin_vel_w = torch.tensor([speed, 0.0, 0.0]).repeat(NUM_ENVS, 1)
        self.root_ang_vel_w = torch.tensor([0.0, 0.0, angular_speed]).repeat(NUM_ENVS, 1)


class _RigidObject:
    def __init__(self, name, position, speed=0.0, angular_speed=0.0, tilt=0.0):
        self.name = name
        self.data = _RigidData(position, speed, angular_speed, tilt)


class _Articulation:
    """``Articulation`` stand-in.

    Deliberately has no ``data.root_pos_w`` and does provide ``get_world_poses``,
    which is exactly how Isaac Lab distinguishes an articulation from a rigid
    object, and the branch ``scene_state.object_position`` takes for the
    stand mixer.
    """

    def __init__(self, joint_names=("hinge",), joint_value=JOINT_VALUE, limits=JOINT_RANGE):
        count = max(len(joint_names), 1)
        self.data = types.SimpleNamespace(
            joint_names=list(joint_names),
            joint_pos=torch.full((NUM_ENVS, count), float(joint_value)),
            joint_pos_limits=torch.tensor([list(limits)] * count).repeat(NUM_ENVS, 1, 1),
            applied_torque=torch.zeros(NUM_ENVS, count),
        )
        self.body_names = [f"openarm_{arm}_ee_tcp" for arm in ("left", "right")]
        self.data.body_names = self.body_names
        self.data.body_link_lin_vel_w = torch.zeros(NUM_ENVS, len(self.body_names), 3)
        position = torch.tensor([0.62, 0.0, 0.55]).repeat(NUM_ENVS, 1)
        rotation = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(NUM_ENVS, 1)
        self.get_world_poses = lambda: (position, rotation)


class _FrameTransformer:
    def __init__(self):
        self.data = types.SimpleNamespace(
            target_frame_names=["left_ee_tcp", "right_ee_tcp"],
            target_pos_w=torch.tensor([[0.45, 0.0, 0.52], [0.45, -0.20, 0.52]]).repeat(NUM_ENVS, 1, 1),
            target_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(NUM_ENVS, 2, 1),
        )


class _Scene(dict):
    env_origins: torch.Tensor


def make_env(objects=None, joint_names=("hinge",), joint_value=JOINT_VALUE, arms=None):
    """Build a stand-in environment with the can/bin/lid scene by default."""
    objects = {
        "can": _RigidObject("can", CAN, speed=0.02, angular_speed=0.1, tilt=0.2),
        "bin": _RigidObject("bin", BIN),
        "lid": _RigidObject("lid", LID),
        **(objects or {}),
    }
    scene = _Scene(objects)
    scene.env_origins = torch.zeros(NUM_ENVS, 3)
    scene["ee_frame"] = _FrameTransformer()
    scene["robot"] = _Articulation(
        tuple(joint_names) + ("openarm_left_finger_joint1", "openarm_right_finger_joint1"),
        joint_value,
    )
    if arms is not None:
        scene["ee_frame"].data.target_pos_w = torch.tensor(list(arms)).repeat(NUM_ENVS, 1, 1)
    return types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(scene=scene, device="cpu", num_envs=NUM_ENVS)
    )


@pytest.fixture(scope="module")
def rewards():
    return importlib.import_module("openarm.arena_tasks.staged.rewards")


@pytest.fixture(scope="module")
def terminations():
    return importlib.import_module("openarm.arena_tasks.staged.terminations")


@pytest.fixture(scope="module")
def observations():
    return importlib.import_module("openarm.arena_tasks.staged.observations")


@pytest.fixture(scope="module")
def scene_state():
    return importlib.import_module("openarm.arena_tasks.staged.scene_state")


@pytest.fixture(scope="module")
def goals():
    return importlib.import_module("openarm.arena_tasks.staged.goals")


# ----------------------------------------------------------------------
# The two kernels
# ----------------------------------------------------------------------


def test_tolerance_score_is_exactly_one_across_the_success_set(rewards):
    """The whole point of the gate: flat 1.0, not merely near it."""
    for error in (0.0, 0.25, 0.99, 1.0):
        got = rewards.tolerance_score(torch.full((1,), error), 1.0)
        assert float(got[0]) == 1.0, f"error {error} inside tolerance scored {float(got[0])!r}"


def test_tolerance_score_decays_outside_the_success_set(rewards):
    assert float(rewards.tolerance_score(torch.full((1,), 1.0), 1.0)[0]) == 1.0
    assert 0.3 < float(rewards.tolerance_score(torch.full((1,), 2.0), 1.0)[0]) < 0.4
    assert float(rewards.tolerance_score(torch.full((1,), 4.0), 1.0)[0]) < 0.05


def test_tolerance_score_is_monotonic_and_never_negative(rewards):
    previous = float("inf")
    for step in range(0, 40):
        value = float(rewards.tolerance_score(torch.full((1,), step * 0.25), 1.0)[0])
        assert value <= previous + 1e-7, f"score rose with error at step {step}"
        assert 0.0 < value <= 1.0
        previous = value


def test_ramp_score_pins_both_bounds_exactly(rewards):
    """``(high - low) / (high - low)`` is not reliably 1.0 in float32."""
    for value, expected in ((0.3, 0.0), (0.4, 0.0), (0.5, 1.0), (0.6, 1.0), (1.0, 1.0)):
        got = float(rewards.ramp_score(torch.tensor([value]), 0.4, 0.5)[0])
        assert got == expected, f"ramp_score({value}, 0.4, 0.5) == {got!r}, want {expected}"


def test_kernels_reject_degenerate_scales(rewards):
    with pytest.raises(ValueError):
        rewards.tanh_score(torch.zeros(1), 0.0)
    with pytest.raises(ValueError):
        rewards.tolerance_score(torch.zeros(1), 1.0, scale=0.0)
    with pytest.raises(ValueError):
        rewards.tolerance_score(torch.zeros(1), -1.0)
    with pytest.raises(ValueError):
        rewards.ramp_score(torch.zeros(1), 0.5, 0.5)


# ----------------------------------------------------------------------
# Dense and boolean must agree
# ----------------------------------------------------------------------


def test_both_dispatches_know_the_same_goal_kinds(rewards, terminations):
    assert set(rewards._DENSE_DISPATCH) == set(terminations._DISPATCH)


@pytest.mark.parametrize(
    ("goal_name", "args", "kwargs", "satisfied"),
    [
        # in_region
        ("ObjectInRegion", ("can", CAN, (0.02, 0.02, 0.02)), {}, True),
        ("ObjectInRegion", ("can", (0.90, 0.0, 0.36), (0.02, 0.02, 0.02)), {}, False),
        # relative_to
        ("ObjectRelativeTo", ("can", "bin", (0.0, 0.0, 0.0), 0.02), {}, False),
        ("ObjectRelativeTo", ("can", "bin", (0.0, 0.0, 0.0), 0.50), {}, True),
        # upright
        ("ObjectUpright", ("can", 0.35), {}, True),
        ("ObjectUpright", ("can", 0.05), {}, False),
        # within_distance
        ("ObjectsWithinDistance", ("can", "lid", 0.05), {}, False),
        ("ObjectsWithinDistance", ("can", "lid", 0.50), {}, True),
        # joint_above
        ("JointNormalizedAbove", ("robot", "hinge", 0.5), {}, True),
        ("JointNormalizedAbove", ("robot", "hinge", 0.9), {}, False),
        # joint_below
        ("JointNormalizedBelow", ("robot", "hinge", 0.5), {}, False),
        ("JointNormalizedBelow", ("robot", "hinge", 0.9), {}, True),
        # at_rest
        ("ObjectAtRest", ("can", 0.05, 0.5), {}, True),
        ("ObjectAtRest", ("can", 0.001, 0.5), {}, False),
        ("ObjectAtRest", ("can", 0.05, 0.001), {}, False),
        # above_plane
        ("ObjectAbovePlane", ("can", 0.30, 0.40), {}, True),
        ("ObjectAbovePlane", ("can", 0.90, 1.10), {}, False),
    ],
)
def test_dense_score_agrees_with_the_boolean_predicate(
    rewards, terminations, goals, goal_name, args, kwargs, satisfied
):
    goal = getattr(goals, goal_name)(*args, **kwargs)
    env = make_env()
    dense = rewards.evaluate_goal_score(env, goal.params())
    boolean = terminations.evaluate_goal(env, goal.params())

    assert boolean.dtype == torch.bool
    assert (float(dense[0]) == 1.0) == satisfied, (
        f"{goal_name}{args}: dense {float(dense[0])!r} vs predicate {bool(boolean[0])}"
    )
    assert 0.0 <= float(dense[0]) <= 1.0


def test_joint_below_rewards_going_the_right_way(rewards):
    """Regression: a ramp from ``threshold`` to ``threshold + std`` paid 1.0 for
    a joint far *above* the threshold, i.e. in full for the state furthest from
    a "close it" goal."""
    env = make_env()
    satisfied = float(rewards.joint_below_score(env, "robot", "hinge", 0.8)[0])
    violated = float(rewards.joint_below_score(env, "robot", "hinge", 0.2)[0])
    assert satisfied == 1.0
    assert violated < 0.05, f"joint_below scored {violated} far above its threshold"


def test_joint_thresholds_are_inclusive(terminations, goals):
    """Regression: the joint predicates used strict comparisons, making a joint
    resting exactly on its threshold a failure -- the only predicates in the
    package that behaved that way."""
    env = make_env()
    for goal in (
        goals.JointNormalizedAbove("robot", "hinge", JOINT_VALUE),
        goals.JointNormalizedBelow("robot", "hinge", JOINT_VALUE),
    ):
        assert bool(terminations.evaluate_goal(env, goal.params())[0]), goal.describe()


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def test_aggregations_agree_with_the_boolean_on_conjunctions(rewards, terminations, goals):
    """``All`` succeeds exactly when every child does, and the mean of the gated
    child scores is then exactly 1.0."""
    satisfied = goals.ObjectUpright("can", 0.35)
    failed = goals.ObjectUpright("can", 0.05)
    env = make_env()

    for children, expected in (
        ((satisfied,), True),
        ((satisfied, failed), False),
        ((satisfied, satisfied), True),
    ):
        params = goals.All(tuple(goal for goal in children)).params()
        dense = float(rewards.evaluate_goal_score(env, params)[0])
        boolean = bool(terminations.evaluate_goal(env, params)[0])
        assert boolean == expected, f"All{children}: predicate {boolean}"
        assert (dense == 1.0) == expected, f"All{children}: dense {dense!r} vs {expected}"


def test_at_least_averages_the_best_candidates(rewards, goals):
    """k-of-n is a conjunction over the best k, so the mean of the top k gated
    scores is 1.0 exactly when the goal holds, and below 1.0 otherwise."""
    good = goals.ObjectUpright("can", 0.35)
    bad = goals.ObjectUpright("can", 0.05)
    env = make_env()

    assert float(rewards.evaluate_goal_score(env, goals.AtLeast(1, (good, bad)).params())[0]) == 1.0
    assert float(rewards.evaluate_goal_score(env, goals.AtLeast(2, (good, bad)).params())[0]) < 1.0
    assert float(rewards.evaluate_goal_score(env, goals.AtLeast(1, (bad,)).params())[0]) < 1.0


def test_aggregators_reject_impossible_inputs(rewards, goals):
    env = make_env()
    params = [goals.ObjectUpright("can", 0.35).params()]
    with pytest.raises(ValueError):
        rewards.goal_scores(env, [])
    with pytest.raises(ValueError):
        rewards.goal_progress(env, params, "median")
    with pytest.raises(ValueError):
        rewards.at_least_score(env, params, 5)


def test_unknown_goal_kind_is_reported_not_guessed(rewards):
    with pytest.raises(KeyError):
        rewards.evaluate_goal_score(make_env(), {"kind": "teleport"})
    with pytest.raises(KeyError):
        rewards.evaluate_goal_score(make_env(), {})


# ----------------------------------------------------------------------
# Observations and penalties
# ----------------------------------------------------------------------


def test_observations_have_the_widths_their_docs_claim(observations):
    env = make_env()
    names = ["can", "bin", "lid"]
    assert observations.object_poses(env, names).shape == (NUM_ENVS, 3 * 7)
    assert observations.object_positions(env, names).shape == (NUM_ENVS, 3 * 3)
    assert observations.object_velocities(env, names).shape == (NUM_ENVS, 3 * 6)
    assert observations.object_relative_to_surface(env, names).shape == (NUM_ENVS, 3)
    assert observations.ee_world_poses(env).shape == (NUM_ENVS, 2 * 7)
    assert observations.gripper_openings(env).shape == (NUM_ENVS, 2)
    assert observations.goal_targets(env, [BIN, CAN]).shape == (NUM_ENVS, 6)
    assert observations.goal_relative_object_offsets(env, [("can", BIN)]).shape == (NUM_ENVS, 3)


def test_observations_are_in_the_environment_frame(scene_state):
    """Every reward and observation reads the same environment-frame position, so
    subtracting the origins is what makes one work out of the box the others do."""
    import types as _types

    env = make_env()
    env.unwrapped.scene.env_origins = torch.tensor([5.0, 0.0, 0.0]).repeat(NUM_ENVS, 1)
    # Move the object to the matching world position; environment-frame x is 0.41.
    env.unwrapped.scene["can"].data.root_pos_w[:, 0] += 5.0
    position = scene_state.object_position(env, "can")
    assert torch.allclose(position[:, 0], torch.full((NUM_ENVS,), CAN[0]), atol=1e-5)
    assert position.shape == (NUM_ENVS, 3)


def test_empty_object_lists_are_rejected(observations):
    """A zero-width observation term would silently change the critic's input
    size, so an empty list is an error rather than an empty tensor."""
    env = make_env()
    for call in (
        lambda: observations.object_poses(env, []),
        lambda: observations.object_velocities(env, []),
        lambda: observations.object_relative_to_surface(env, []),
        lambda: observations.goal_targets(env, []),
    ):
        with pytest.raises(ValueError):
            call()


def test_missing_joints_name_what_is_available(scene_state):
    """A wrong joint name otherwise fails at reset with an opaque KeyError."""
    env = make_env()
    with pytest.raises(KeyError, match="hinge"):
        scene_state.gripper_opening(env, "right", "no_such_joint")
    with pytest.raises(KeyError, match="hinge"):
        scene_state.normalized_joint_position(env, "robot", "no_such_joint")


def test_penalties_charge_nothing_when_clear(rewards):
    env = make_env()
    # bin and lid are at rest, so policing them is free.
    assert float(rewards.object_motion_l2(env, ["bin", "lid"])[0]) == 0.0
    assert float(rewards.object_motion_l2(env, ["can"])[0]) > 0.0
    # The can is above both of these lines, so it is not "below" either.
    assert float(rewards.object_below_height_l2(env, ["can"], CAN[2] - 0.06)[0]) == 0.0
    assert float(rewards.object_below_height_l2(env, ["can"], CAN[2] + 0.14)[0]) > 0.0


def test_penalties_are_non_negative_over_the_batch(rewards):
    env = make_env()
    for value in rewards.object_motion_l2(env, ["can", "bin", "lid"]).tolist():
        assert value >= 0.0
    for value in rewards.object_below_height_l2(env, ["can", "bin", "lid"], 0.9).tolist():
        assert value >= 0.0


def test_penalties_reject_degenerate_scales(rewards):
    env = make_env()
    with pytest.raises(ValueError):
        rewards.object_motion_l2(env, ["can"], lin_scale=0.0)
    with pytest.raises(ValueError):
        rewards.object_motion_l2(env, ["can"], ang_scale=0.0)
    with pytest.raises(ValueError):
        rewards.ee_speed_l2(env, "right", scale=0.0)


def test_ee_speed_reads_body_velocities_not_a_cached_difference(rewards, scene_state):
    """The frame transformer publishes pose only, so a differencing penalty would
    have to cache the previous step and could desynchronize at an episode reset."""
    env = make_env()
    assert float(rewards.ee_speed_l2(env, "right")[0]) == 0.0
    env.unwrapped.scene["robot"].data.body_link_lin_vel_w[:, 1, 0] = 0.4
    assert float(rewards.ee_speed_l2(env, "right")[0]) == pytest.approx(0.16)
    with pytest.raises(KeyError, match="openarm_left_ee_tcp"):
        rewards.ee_speed_l2(env, "left_wrong_name")


# ----------------------------------------------------------------------
# Every real task
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def tasks():
    return importlib.import_module("openarm.arena_environments.industrial").INDUSTRIAL_TASKS


def _task_env(task, rewards, terminations, goals_module):
    """A stand-in scene holding every object and articulation a task names."""
    spec = task.spec
    names = sorted(set(task.objects) | set(spec.object_names))
    articulations: list[str] = []
    joint_names: list[str] = []
    for goal in spec.goals():
        params = goal.params()
        if params.get("asset_cfg") and params["asset_cfg"] not in articulations:
            articulations.append(params["asset_cfg"])
        if params.get("joint_name") and params["joint_name"] not in joint_names:
            joint_names.append(params["joint_name"])

    scene = _Scene()
    scene.env_origins = torch.zeros(NUM_ENVS, 3)
    for index, name in enumerate(names):
        # Distinct, deterministic poses and velocities per object, so a dispatch
        # that reads the wrong object or a stale index is visibly wrong.
        scene[name] = _RigidObject(
            name,
            (0.40 + 0.03 * index, -0.05 * index, 0.36 + 0.01 * (index % 3)),
            speed=0.01 * (index + 1),
            angular_speed=0.05 * (index + 1),
            tilt=0.10 * (index + 1),
        )
    scene["ee_frame"] = _FrameTransformer()
    scene["robot"] = _Articulation(tuple(joint_names) or ("hinge",))
    for name in articulations:
        scene[name] = _Articulation(tuple(joint_names) or ("hinge",))
    return types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(scene=scene, device="cpu", num_envs=NUM_ENVS)
    )


def test_every_industrial_goal_resolves_through_both_dispatches(tasks, rewards, terminations):
    checked = 0
    for key, task in tasks.items():
        env = _task_env(task, rewards, terminations, None)
        for goal in task.spec.goals():
            params = goal.params()
            dense = rewards.evaluate_goal_score(env, params)
            boolean = terminations.evaluate_goal(env, params)
            assert dense.shape == (NUM_ENVS,), f"{key}/{goal.describe()}"
            assert boolean.shape == (NUM_ENVS,), f"{key}/{goal.describe()}"
            assert torch.isfinite(dense).all(), f"{key}/{goal.describe()}"
            assert float(dense.max()) <= 1.0, f"{key}/{goal.describe()}"
            # Equality, not a tolerance: both sides read the same tolerance off
            # the same floats, so any disagreement is a real one.
            assert torch.equal(dense == 1.0, boolean), (
                f"{key}/{goal.describe()}: gate {[round(float(d), 4) for d in dense]} "
                f"disagrees with predicate {[bool(b) for b in boolean]}"
            )
            checked += 1
    assert checked >= 100, f"only {checked} goals checked; the table should have at least 100"


def test_every_industrial_task_aggregates(tasks, rewards):
    for key, task in tasks.items():
        env = _task_env(task, rewards, None, None)
        params = [goal.params() for goal in task.spec.goals()]
        for aggregation in ("mean", "min", "product"):
            value = rewards.goal_progress(env, params, aggregation)
            assert value.shape == (NUM_ENVS,) and torch.isfinite(value).all(), f"{key}/{aggregation}"
        assert float(rewards.goal_fraction(env, params).max()) <= 1.0
        assert float(rewards.success_bonus(env, params).max()) <= 1.0
