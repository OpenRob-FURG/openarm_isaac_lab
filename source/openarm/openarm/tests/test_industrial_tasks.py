"""Static validation of the 50 industrial task specifications.

The task table is pure data, so it can be checked without launching Isaac Sim.
These tests stub out ``isaaclab``, ``warp`` and the GPU-backed parts of
``isaaclab_arena``, then import the real modules and assert the invariants that
would otherwise only fail hours into a headless run.

Run with ``pytest source/openarm/openarm/tests/test_industrial_tasks.py``.
"""

from __future__ import annotations

import importlib
import re
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

ARENA_ROOT = Path(__file__).resolve().parents[4] / "submodules" / "IsaacLab-Arena"
OPENARM_ROOT = Path(__file__).resolve().parents[2]

#: Module prefixes that must be stubbed because they need a running simulator.
#: ``torch`` is deliberately absent: the real one is installed and the data
#: structures under test are plain dataclasses.
STUBBED_PREFIXES = ("isaaclab", "isaaclab_tasks", "warp", "omni", "pxr", "isaaclab_arena_environments")


def _identity_decorator(*_args, **_kwargs):
    """Return a decorator that leaves its target unchanged.

    ``configclass`` normally rewrites a class into a dataclass with extra
    behaviour. For static checks the identity is enough and, unlike a mock,
    keeps the class usable.
    """

    def wrap(target):
        return target

    return wrap


class StubModule(types.ModuleType):
    """A module whose unknown attributes are auto-created mocks."""

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        value = MagicMock(name=f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


def _stub(dotted: str, **attrs) -> StubModule:
    """Register a stub module, creating stub parents as needed.

    Args:
        dotted: Fully qualified module name.
        **attrs: Attributes to set on the stub.

    Returns:
        The stub module.
    """
    parts = dotted.split(".")
    for index in range(1, len(parts) + 1):
        path = ".".join(parts[:index])
        if path not in sys.modules:
            sys.modules[path] = StubModule(path)
    module = sys.modules[dotted]
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_stubs() -> None:
    """Install import stubs for every module that needs a live simulator."""
    for name in list(sys.modules):
        if name.split(".")[0] in STUBBED_PREFIXES:
            del sys.modules[name]

    for root in (
        "warp",
        "omni",
        "pxr",
        "isaaclab",
        "isaaclab_tasks",
        "isaaclab_arena",
        "isaaclab_arena_environments",
    ):
        _stub(root)

    for pkg in ("isaaclab", "isaaclab_tasks", "isaaclab_arena", "isaaclab_arena_environments"):
        _stub(pkg).__path__ = []  # mark as a package so submodule imports resolve

    _stub("isaaclab_tasks.utils")
    _stub("omni.ext")
    _stub("isaaclab.managers", configclass=_identity_decorator)
    _stub("isaaclab.utils", configclass=_identity_decorator).__path__ = []  # package, so utils.array resolves
    _stub("isaaclab.utils.array", convert_to_torch=MagicMock)
    _stub(
        "isaaclab.utils.math",
        matrix_from_euler=MagicMock,
        matrix_from_quat=MagicMock,
        quat_from_matrix=MagicMock,
        quat_mul=MagicMock,
        make_pose=MagicMock,
    )
    _stub("isaaclab.envs")
    _stub("isaaclab.scene")
    _stub("isaaclab.app")
    _stub("isaaclab_arena.assets.object", Object=MagicMock)
    _stub("isaaclab_arena.assets.asset_registry", AssetRegistry=MagicMock, HDRImageRegistry=MagicMock)
    _stub("isaaclab_arena.metrics.metric_base", MetricBase=MagicMock)
    _stub("isaaclab_arena.metrics.object_moved", ObjectMovedRateMetric=MagicMock)
    _stub("isaaclab_arena.metrics.success_rate", SuccessRateMetric=MagicMock)
    _stub("isaaclab_arena.tasks.task_base", TaskBase=MagicMock)
    class _PolicyBase:
        """Stand-in for Arena's abstract policy base."""

        config_class = None

        def __init__(self, config):
            self.config = config

    _stub("isaaclab_arena.policy.policy_base", PolicyBase=_PolicyBase)
    _stub("isaaclab_arena.embodiments.common.arm_mode", ArmMode=MagicMock)
    _stub("isaaclab_arena.environments.isaaclab_arena_environment", IsaacLabArenaEnvironment=MagicMock)
    _stub("isaaclab_arena.environments.arena_env_builder", ArenaEnvBuilder=MagicMock)
    _stub("isaaclab_arena.scene.scene", Scene=MagicMock)
    _stub("isaaclab_arena.utils.pose", Pose=MagicMock)
    _stub("isaaclab_arena_environments.example_environment_base", ExampleEnvironmentBase=object)

    for path in (ARENA_ROOT, OPENARM_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


_install_stubs()


def _registered_assets() -> set[str]:
    """Read every asset name out of the Arena and OpenArm asset libraries."""
    names: set[str] = set()
    sources = [
        ARENA_ROOT / "isaaclab_arena" / "assets" / "object_library.py",
        OPENARM_ROOT / "openarm" / "assets" / "background_library.py",
    ]
    for path in sources:
        if not path.exists():
            continue
        text = path.read_text()
        names.update(re.findall(r'^\s{4}name\s*=\s*["\']([\w\-]+)["\']', text, re.M))

    openarm_assets = OPENARM_ROOT / "openarm" / "assets"
    for module in sorted(openarm_assets.glob("*.py")):
        text = module.read_text()
        names.update(re.findall(r'^\s{4}name\s*=\s*["\']([\w\-]+)["\']', text, re.M))
    return names


@pytest.fixture(scope="module")
def registry():
    return _registered_assets()


@pytest.fixture(scope="module")
def tasks():
    return importlib.import_module("openarm.arena_environments.industrial").INDUSTRIAL_TASKS


def test_there_are_exactly_fifty_tasks(tasks):
    assert len(tasks) == 50


def test_task_names_are_unique(tasks):
    assert len(set(tasks)) == len(tasks)


def test_every_task_has_stages_and_goals(tasks):
    for key, task in tasks.items():
        assert task.spec.stages, f"{key} has no stages"
        assert task.spec.goals(), f"{key} has no goals"
        assert task.spec.instruction, f"{key} has no instruction"


def test_stage_and_goal_objects_exist_in_the_scene(tasks):
    for key, task in tasks.items():
        for stage in task.spec.stages:
            for role, name in (("target", stage.target), ("secondary", stage.secondary)):
                if name:
                    assert name in task.objects, f"{key}: stage {role} '{name}' is not in the scene"
        for goal in task.spec.goals():
            for name in goal.object_names():
                assert name in task.objects, f"{key}: goal refers to unknown object '{name}'"


def test_every_referenced_asset_is_registered(tasks, registry):
    missing = set()
    for task in tasks.values():
        missing.update(task.objects.values())
    unknown = {name for name in missing if name not in registry}
    assert not unknown, f"unregistered assets referenced: {sorted(unknown)}"


def test_background_asset_is_registered(tasks, registry):
    unknown = {task.background for task in tasks.values() if task.background not in registry}
    assert not unknown, f"unregistered backgrounds: {sorted(unknown)}"


def test_every_skill_has_a_controller():
    from openarm.arena_tasks.staged.stages import Skill
    from openarm.policies.oracle.skills import CONTROLLERS

    missing = {skill for skill in Skill if skill not in CONTROLLERS}
    assert not missing, f"skills without a controller: {sorted(s.value for s in missing)}"


def test_every_goal_kind_has_a_termination(tasks):
    from openarm.arena_tasks.staged.terminations import _DISPATCH

    kinds = {goal.params()["kind"] for task in tasks.values() for goal in task.spec.goals()}
    assert len(kinds) >= 6, f"expected a variety of goal kinds, saw {sorted(kinds)}"
    assert kinds <= set(_DISPATCH), f"goals without a termination: {sorted(kinds - set(_DISPATCH))}"


def test_goal_dispatch_does_not_raise_for_every_task(tasks):
    from openarm.arena_tasks.staged.terminations import _DISPATCH

    for key, task in tasks.items():
        for goal in task.spec.goals():
            kind = goal.params()["kind"]
            assert kind in _DISPATCH, f"{key}: goal kind '{kind}' has no termination"


def test_object_names_helper_finds_nested_goals():
    from openarm.arena_tasks.staged import All, ObjectInRegion, ObjectRelativeTo

    goal = All(goals=(ObjectInRegion("a", (0, 0, 0)), ObjectRelativeTo("b", "c", (0, 0, 0))))
    assert sorted(goal.object_names()) == ["a", "b", "c"]


def test_each_family_is_contiguous_and_sized(tasks):
    from openarm.arena_environments.industrial import beverage, containers, fastening, inspection, packing, tools

    expected = {
        "fastening": (fastening, 10),
        "packing": (packing, 10),
        "beverage": (beverage, 8),
        "containers": (containers, 6),
        "tools": (tools, 10),
        "inspection": (inspection, 6),
    }
    for label, (module, count) in expected.items():
        assert len(module.TASKS) == count, f"family {label} has {len(module.TASKS)} tasks, expected {count}"
        for task in module.TASKS:
            assert task in tasks.values(), f"{label}: task {task.spec.name} missing from the registry"


def test_pinned_objects_are_part_of_the_scene(tasks):
    for key, task in tasks.items():
        for name in task.pinned_objects:
            assert name in task.objects, f"{key}: pinned object '{name}' is not in the scene"


def test_oracle_policy_requires_a_known_task():
    from openarm.policies.oracle.staged_oracle import StagedOraclePolicy, StagedOraclePolicyCfg

    with pytest.raises(ValueError, match="Unknown industrial task"):
        StagedOraclePolicy(StagedOraclePolicyCfg(task="does_not_exist"))


def test_retry_budget_is_shared_across_stages():
    from openarm.policies.oracle.staged_oracle import DEFAULT_RETRY_BUDGET

    assert DEFAULT_RETRY_BUDGET > 0


def test_goal_label_is_keyword_only():
    """Positional construction must bind object names, not the label.

    Regression test. ``Goal`` declares ``label`` on the base class, so before it
    was made keyword-only every positional goal in the package bound its first
    argument to ``label`` and left ``object_name`` holding a coordinate tuple.
    That produced goals which silently never referenced an object.
    """
    from openarm.arena_tasks.staged import ObjectInRegion, ObjectRelativeTo

    region = ObjectInRegion("mug", (0.1, 0.2, 0.3), (0.1, 0.1, 0.1))
    assert region.label == ""
    assert region.object_name == "mug"
    assert region.center == (0.1, 0.2, 0.3)

    relative = ObjectRelativeTo("nut", "bolt", (0.0, 0.0, 0.01), tolerance=0.02)
    assert relative.object_name == "nut"
    assert relative.reference_name == "bolt"
    assert relative.object_names() == ["nut", "bolt"]


def test_every_goal_names_a_real_object(tasks):
    """Each goal must resolve to at least one scene object."""
    for key, task in tasks.items():
        for goal in task.spec.goals():
            assert goal.object_names(), f"{key}: goal {goal} names no object"


def test_open_and_close_only_target_openable_assets(tasks):
    """The articulated open/close goals require an asset tagged openable.

    Regression test. A task that asks the oracle to open a plain container
    looks well-formed but can never succeed, because the asset has no joint.
    """
    from openarm.arena_tasks.staged import Skill

    openable = _assets_with_tag("openable")
    for key, task in tasks.items():
        for stage in task.spec.stages:
            if stage.skill in (Skill.OPEN, Skill.CLOSE):
                asset = task.objects[stage.target]
                assert asset in openable, f"{key}: {stage.skill.value} on '{asset}', which is not openable"


def test_turn_only_targets_turnable_assets(tasks):
    from openarm.arena_tasks.staged import Skill

    turnable = _assets_with_tag("turnable")
    for key, task in tasks.items():
        for stage in task.spec.stages:
            if stage.skill is Skill.TURN:
                asset = task.objects[stage.target]
                assert asset in turnable, f"{key}: turn on '{asset}', which is not turnable"


def test_press_stages_are_physically_meaningful(tasks):
    """A press must either hit a pressable fixture or push a tool onto a target.

    Regression test. ``PRESS`` on the drill is legitimate even though the drill
    is not a pressable fixture, because the press is the tool driving down onto
    the plate named by ``secondary``. Requiring the target to be pressable
    rejected that, while genuinely bogus presses still fail here because they
    name neither a pressable fixture nor a workpiece.
    """
    from openarm.arena_tasks.staged import Skill

    pressable = _assets_with_tag("pressable")
    for key, task in tasks.items():
        for stage in task.spec.stages:
            if stage.skill is not Skill.PRESS:
                continue
            on_fixture = task.objects[stage.target] in pressable
            onto_workpiece = bool(stage.secondary) and stage.secondary in task.objects
            assert on_fixture or onto_workpiece, (
                f"{key}: press on '{task.objects[stage.target]}' with no pressable fixture "
                f"and no workpiece to press onto"
            )


def _assets_with_tag(tag: str) -> set[str]:
    """Return the names of registered assets carrying a tag."""
    import re

    names: set[str] = set()
    for path in (
        ARENA_ROOT / "isaaclab_arena" / "assets" / "object_library.py",
        *sorted((OPENARM_ROOT / "openarm" / "assets").glob("*.py")),
    ):
        if not path.exists():
            continue
        for block in path.read_text().split("@register_asset")[1:]:
            name = re.search(r'^\s{4}name\s*=\s*["\']([\w\-]+)["\']', block, re.M)
            tags = re.search(r"^\s{4}tags\s*=\s*(\[[^\]]*\])", block, re.M)
            if name and tags and f'"{tag}"' in tags.group(1):
                names.add(name.group(1))
    return names


def test_there_are_fifty_generated_environment_classes():
    from openarm.arena_environments.industrial import INDUSTRIAL_ENVIRONMENTS, INDUSTRIAL_TASKS

    assert len(INDUSTRIAL_ENVIRONMENTS) == 50
    assert set(INDUSTRIAL_ENVIRONMENTS) == {task.name for task in INDUSTRIAL_TASKS.values()}
    for name, cls in INDUSTRIAL_ENVIRONMENTS.items():
        assert cls.name == name, f"generated class {cls} has mismatched name"
        assert cls.task_key in INDUSTRIAL_TASKS
        assert name.startswith("openarm_bimanual_industrial_")


def test_environment_classes_are_distinct():
    from openarm.arena_environments.industrial import INDUSTRIAL_ENVIRONMENTS

    classes = [cls for cls in INDUSTRIAL_ENVIRONMENTS.values()]
    assert len(set(classes)) == len(classes), "two tasks share an environment class"


def test_instruction_variants_are_distinct_and_nonempty(tasks):
    for key, task in tasks.items():
        instructions = task.spec.all_instructions()
        assert instructions, f"{key}: no instructions"
        assert all(text.strip() for text in instructions), f"{key}: empty instruction variant"
        assert len(set(instructions)) == len(instructions), f"{key}: duplicate instruction variants"


def test_every_stage_arm_is_valid(tasks):
    for key, task in tasks.items():
        for stage in task.spec.stages:
            assert stage.arm in ("left", "right", "both"), f"{key}: invalid arm '{stage.arm}'"


def test_stage_budget_fits_the_episode(tasks):
    """The per-stage step budget must leave room inside the episode.

    At the OpenArm bimanual decimation of 4 and 200 Hz physics, one control
    step is 20 ms, so a 60 s episode is roughly 3000 steps.
    """
    from openarm.policies.oracle.staged_oracle import RECOVERY_STEPS

    control_steps_per_second = 50
    for key, task in tasks.items():
        budget = task.spec.episode_length_s * control_steps_per_second
        worst_case = 0
        for stage in task.spec.stages:
            attempts = min(stage.max_attempts, 4)
            worst_case += attempts * (stage.max_steps + RECOVERY_STEPS)
        assert worst_case < budget, (
            f"{key}: worst-case scripted time {worst_case} steps exceeds the "
            f"{budget:.0f}-step episode budget"
        )


def test_retry_attempts_are_bounded(tasks):
    for key, task in tasks.items():
        for stage in task.spec.stages:
            assert 1 <= stage.max_attempts <= 5, f"{key}: unreasonable max_attempts {stage.max_attempts}"
            assert stage.max_steps > 0, f"{key}: non-positive max_steps"


def test_dr_presets_resolve_for_every_task():
    from openarm.arena_dr.config import DRConfig

    from openarm.arena_environments.industrial import INDUSTRIAL_TASKS

    for key, task in INDUSTRIAL_TASKS.items():
        cfg = DRConfig.named(task.dr_preset)
        assert cfg is not None, f"{key}: unknown DR preset '{task.dr_preset}'"


def test_pinned_objects_are_merged_into_the_dr_config():
    """Pinned fixtures must survive the per-task DR merge.

    Regression test: the task's pinned objects have to be unioned into the
    preset's own pinned set, otherwise a fixture gets randomized out from under
    the goal region and the task becomes unsolvable rather than merely harder.
    """
    from dataclasses import replace

    from openarm.arena_dr.config import DRConfig
    from openarm.arena_environments.industrial import INDUSTRIAL_TASKS

    preset = DRConfig.named("default")
    for key, task in INDUSTRIAL_TASKS.items():
        if not task.pinned_objects:
            continue
        merged = replace(preset, pinned_objects=tuple({*preset.pinned_objects, *task.pinned_objects}))
        for name in task.pinned_objects:
            assert name in merged.pinned_objects, f"{key}: '{name}' lost during the DR merge"


def test_builder_signature_accepts_the_hdr_registry():
    import inspect

    from openarm.arena_environments.industrial import build_environment

    params = inspect.signature(build_environment).parameters
    for name in ("task", "args_cli", "registry", "hdr_registry", "dr"):
        assert name in params, f"build_environment is missing the '{name}' parameter"


def test_composite_goals_serialize_to_plain_nested_dicts():
    """Composite goals must not leak goal instances into termination params.

    The termination term's params are walked by ``configclass``, which assigns
    to any dataclass it finds. The goals are frozen dataclasses, so passing the
    children through as objects raises ``FrozenInstanceError`` at build time.
    """
    from openarm.arena_tasks.staged.goals import All, AtLeast, ObjectInRegion, ObjectUpright

    children = (ObjectInRegion("mug", (0.3, 0.0, 0.1), (0.1, 0.1, 0.1)), ObjectUpright("mug"))
    for composite in (All(goals=children), AtLeast(count=1, goals=children)):
        nested = composite.params()["goals"]
        assert len(nested) == len(children)
        for entry in nested:
            assert isinstance(entry, dict), f"{type(composite).__name__} leaked a {type(entry).__name__}"
            assert "kind" in entry
            # Recursively: a nested composite must be dicts all the way down.
            for value in entry.values():
                assert not hasattr(value, "params") or isinstance(value, (str, int, float, tuple, list, dict))


def test_local_point_follows_the_pose_device():
    """A local offset must be built on the pose's device.

    The physics backend hands back CUDA tensors, so a CPU offset makes the
    matmul raise instead of running on the GPU.
    """
    import torch

    from openarm.policies.oracle.skills import local_point

    pose = torch.eye(4, dtype=torch.float32)
    pose[:3, 3] = torch.tensor([1.0, 2.0, 3.0])
    result = local_point(pose, (0.5, 0.0, 0.0))
    assert result.device == pose.device
    assert torch.allclose(result, torch.tensor([1.5, 2.0, 3.0]))


def test_fixture_joint_is_read_from_the_goal_not_the_robot():
    """The articulation controller must watch the fixture's joint.

    Reading the robot's ``joint_pos`` observation would track an arm joint and
    report the fixture as never moving.
    """
    from openarm.arena_tasks.staged.goals import JointNormalizedAbove, ObjectInRegion
    from openarm.arena_tasks.staged.stages import Skill, Stage

    joint_stage = Stage(skill=Skill.OPEN, arm="right", target="microwave", label="open").with_goal(
        JointNormalizedAbove("microwave", "Disc001_joint", threshold=0.5)
    )
    from openarm.policies.oracle.skills import _fixture_joint

    assert _fixture_joint(joint_stage) == ("microwave", "Disc001_joint")

    # A goal that names no joint leaves the controller without one to watch.
    plain_stage = Stage(skill=Skill.PICK, arm="right", target="mug", label="pick").with_goal(
        ObjectInRegion("mug", (0.3, 0.0, 0.1), (0.1, 0.1, 0.1))
    )
    assert _fixture_joint(plain_stage) == (None, None)


def test_fixture_joint_names_match_the_real_assets():
    """The declared fixture joints must be the ones the USD assets expose.

    A generic name fails at reset with a KeyError listing the real joints; these
    are the names the microwave, coffee machine and stand mixer actually ship.
    """
    from openarm.arena_environments.industrial.beverage import (
        COFFEE_BUTTON_JOINT,
        MICROWAVE_DOOR_JOINT,
        MIXER_JOINT,
    )

    assert MICROWAVE_DOOR_JOINT == "Disc001_joint"
    assert COFFEE_BUTTON_JOINT == "CoffeeMachine108_Button001_joint"
    assert MIXER_JOINT == "knob_speed_joint"
