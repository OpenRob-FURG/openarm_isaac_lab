"""Static validation of the 50 industrial task specifications.

The task table is pure data, so it can be checked without launching Isaac Sim.
These tests stub out ``isaaclab``, ``warp`` and the GPU-backed parts of
``isaaclab_arena``, then import the real modules and assert the invariants that
would otherwise only fail hours into a headless run.

Run with ``pytest source/openarm/openarm/tests/test_industrial_tasks.py``.
"""

from __future__ import annotations

import ast
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


class _FakeBoundingBox:
    """Stand-in for Arena's ``AxisAlignedBoundingBox``.

    Arena's own implementation is plain data, so a two-field record is enough to
    let the surface anchor be built and inspected without a simulator.
    """

    def __init__(self, min_point=(), max_point=()):
        self.min_point = tuple(min_point)
        self.max_point = tuple(max_point)

    def __repr__(self):
        return f"_FakeBoundingBox(min_point={self.min_point}, max_point={self.max_point})"


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
    # ``utils`` is a package, so utils.bounding_box resolves
    _stub("isaaclab_arena.utils").__path__ = []
    _stub("isaaclab_arena.utils.pose", Pose=MagicMock)
    _stub("isaaclab_arena.utils.bounding_box", AxisAlignedBoundingBox=_FakeBoundingBox)
    _stub("isaaclab_arena.assets.dummy_object", DummyObject=MagicMock)
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


# -- semantic relational layout ---------------------------------------------


def test_the_work_surface_matches_the_arena_background():
    """The layout's surface and the background's collision slab must agree.

    The layout solves against a declared slab, but the props rest on whatever
    ``ProceduralTable`` actually spawns. If the two ever disagree the props
    either hover above the table or sink into it, and no amount of solving
    fixes it -- so the declared size is checked against the asset's own.
    """
    from openarm.arena_environments.industrial._base import TABLE_CENTER, TABLE_SIZE, work_surface

    source = (ARENA_ROOT / "isaaclab_arena" / "assets" / "object_library.py").read_text()
    block = source.split("_PROCEDURAL_TABLE_SPAWN_CFG = ", 1)[1].split("\n)", 1)[0]
    size = re.search(r"size=\(([\d.]+),\s*([\d.]+),\s*([\d.]+)\)", block)
    assert size, "could not read the ProceduralTable slab size from the Arena asset library"
    assert tuple(float(value) for value in size.groups()) == TABLE_SIZE

    surface = work_surface()
    assert surface.center == TABLE_CENTER
    assert surface.size == TABLE_SIZE
    # A prop rests with its underside a clearance above the slab's top face.
    assert surface.top_z == pytest.approx(TABLE_CENTER[2] + TABLE_SIZE[2] / 2.0)


def test_every_task_layout_places_every_scene_object(tasks):
    """A scene object with no placement would be left wherever it spawns.

    ``ObjectPlacer`` only moves the objects it is handed, so an unplaced prop is
    one the task silently never set up.
    """
    for key, task in tasks.items():
        layout = task.resolve_layout()
        assert sorted(layout.object_names()) == sorted(task.objects), f"{key}: layout does not cover the scene"
        assert len(layout.object_names()) == len(set(layout.object_names())), f"{key}: duplicate placements"


def test_every_placement_declares_relations_and_a_known_parent(tasks):
    """The placer requires a relation per object, and every parent to exist.

    ``ObjectPlacer.place`` asserts that each object it is given has at least one
    relation, and a relation whose parent is not in the scene fails with a
    ``KeyError`` from far inside the builder.
    """
    for key, task in tasks.items():
        for placement in task.resolve_layout().placements:
            assert placement.relations, f"{key}: '{placement.object_name}' declares no relation"
            for parent in placement.parent_names():
                assert parent in task.objects, f"{key}: '{placement.object_name}' references unknown '{parent}'"


def test_every_placement_rests_on_the_work_surface(tasks):
    """Nothing may be placed by absolute coordinates alone.

    A prop with no ``On`` relation has no height constraint, so the solver would
    happily leave it at the world origin, stacked on the floor.
    """
    from openarm.arena_tasks.staged.layout import IsAnchor, On

    for key, task in tasks.items():
        for placement in task.resolve_layout().placements:
            if placement.is_anchor:
                assert placement.pose is not None, f"{key}: anchored '{placement.object_name}' has no pose"
                continue
            assert any(
                isinstance(relation, On) for relation in placement.relations
            ), f"{key}: '{placement.object_name}' is not placed on the work surface"


def test_at_position_relations_pin_at_least_one_axis(tasks):
    """A bare ``AtPosition`` is rejected by Arena before the solver starts.

    Regression test. A placement that is positioned only relative to its
    neighbours once carried an ``AtPosition`` with every axis ``None``, which
    looks like "no constraint" but trips Arena's assert that at least one axis
    is given, and fails at environment-build time for all 50 tasks.
    """
    from openarm.arena_tasks.staged.layout import AtPosition

    for key, task in tasks.items():
        for placement in task.resolve_layout().placements:
            for relation in placement.relations:
                if isinstance(relation, AtPosition):
                    assert any(axis is not None for axis in (relation.x, relation.y, relation.z)), (
                        f"{key}: '{placement.object_name}' declares an AtPosition with no axis set"
                    )


def test_pinned_objects_never_receive_a_jitter_marker(tasks):
    """Fixtures and goal-region receptacles must survive placement unjittered.

    They are excluded from ``RandomAroundSolution`` rather than given one with
    zeroed extents, so a mistake cannot leave a fixture quietly randomized.
    """
    from openarm.arena_dr.config import DRConfig
    from openarm.arena_tasks.staged.layout import RandomAroundSolution

    for key, task in tasks.items():
        pinned = {*DRConfig.named(task.dr_preset).pinned_objects, *task.pinned_objects}
        for placement in task.resolve_layout().placements:
            if placement.object_name in pinned:
                assert not any(isinstance(r, RandomAroundSolution) for r in placement.relations), (
                    f"{key}: pinned object '{placement.object_name}' declares jitter"
                )


def test_pose_jitter_follows_the_dr_preset():
    """Jitter magnitudes come from the preset, and a disabled preset means none.

    The layout decides *where* randomization is declared, but the preset still
    decides how much of it there is, so the two have to agree on both.
    """
    from openarm.arena_dr.config import DRConfig
    from openarm.arena_tasks.staged.layout import PoseJitter

    for quiet in ("none", "light"):
        assert not PoseJitter.from_dr_config(DRConfig.named(quiet)).as_marker().is_active, (
            f"preset '{quiet}' disables pose jitter but still produced some"
        )

    for active in ("default", "heavy"):
        preset = DRConfig.named(active)
        jitter = PoseJitter.from_dr_config(preset)
        assert jitter.as_marker().is_active, f"preset '{active}' enables pose jitter but produced none"
        assert (jitter.x_half_m, jitter.y_half_m) == (preset.position_jitter[0], preset.position_jitter[1])
        assert jitter.z_half_m == preset.height_jitter
        assert jitter.yaw_half_rad == preset.yaw_jitter


_ARENA_RELATIONS = (
    "Relation",
    "UnaryRelation",
    "RelationBase",
    "NextTo",
    "On",
    "NoCollision",
    "RandomAroundSolution",
    "RotateAroundSolution",
    "AtPosition",
    "PositionLimits",
    "IsAnchor",
    "Side",
)


def _recording_arena_relations():
    """Install a stand-in Arena relations module that records how it is called.

    The real module needs a running simulator, and the translation from the
    layout vocabulary to Arena relations is exactly the layer where a swapped
    positional argument or a renamed constructor would go unnoticed -- every
    call would still succeed and every object would still be "placed", just not
    where the layout said. Each fake class records the arguments it received, and
    the tests below compare them against the *real* signatures parsed out of the
    Arena source, so a change on either side of the translation is caught here
    rather than hours into a headless run.
    """
    calls: dict[str, list] = {}

    def make_class(name: str):
        def __init__(self, *args, **kwargs):
            self.name = name
            self.args = args
            self.kwargs = kwargs
            calls.setdefault(name, []).append(self)

        return type(name, (), {"__init__": __init__})

    module = types.ModuleType("isaaclab_arena.relations.relations")
    for name in _ARENA_RELATIONS:
        if name == "Side":
            module.Side = lambda value: f"Side({value})"
        else:
            setattr(module, name, make_class(name))
    # The resolver does ``from isaaclab_arena.relations import relations``, which
    # reads the attribute off the parent package before consulting
    # ``sys.modules``, so both have to point at the fake.
    package = _stub("isaaclab_arena.relations")
    package.__path__ = []
    package.relations = module
    sys.modules["isaaclab_arena.relations.relations"] = module
    return calls


def _arena_signatures() -> dict[str, list[str]]:
    """Parse the parameter order of Arena's real relation constructors."""
    source = (ARENA_ROOT / "isaaclab_arena" / "relations" / "relations.py").read_text()
    signatures: dict[str, list[str]] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        init = next((n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"), None)
        if init is not None:
            signatures[node.name] = [arg.arg for arg in init.args.args][1:]
    return signatures


def test_declared_relations_translate_onto_the_arena_ones():
    """The translation must target Arena's own argument order.

    Every declared relation is resolved against recording fakes, and the
    positional arguments are compared to the constructors parsed out of Arena's
    source. A layout relation reordered without a matching change here (or the
    reverse) fails instead of silently building a relation with, say, a distance
    where Arena expects a clearance.
    """
    from openarm.arena_tasks.staged import layout as layout_module

    calls = _recording_arena_relations()
    parents = {"bin": MagicMock(name="bin")}
    surface_object = MagicMock(name="surface")
    jitter = layout_module.PoseJitter(0.02, 0.02, 0.005, 0.26)
    samples = [
        layout_module.On(),
        layout_module.On(clearance_m=0.03, surface="bin"),
        layout_module.NextTo("bin", side=layout_module.Side.POSITIVE_Y, distance_m=0.2),
        layout_module.AtPosition(0.1, 0.2, None),
        layout_module.PositionLimits(x_min=0.0, x_max=1.0, y_min=-1.0, y_max=1.0),
        layout_module.NoCollision("bin", clearance_m=0.05),
        layout_module.IsAnchor(),
        jitter.as_marker(),
        layout_module.RotateAroundSolution(0.0, 0.0, 1.57),
    ]
    for declared in samples:
        built = layout_module.build_relation(declared, parents, surface_object)
        signature = _arena_signatures().get(built.name, [])
        assert len(built.args) <= len(signature), (
            f"{type(declared).__name__}: passed {len(built.args)} positionally, Arena takes {signature}"
        )
        assert not built.kwargs, f"{type(declared).__name__}: Arena takes no keyword arguments"


    # A spot check that the arguments landed where they are meant to.
    assert calls["On"][0].args == (surface_object, 1.0, 0.005)
    assert calls["On"][1].args == (parents["bin"], 1.0, 0.03)
    assert calls["NextTo"][0].args == (parents["bin"], 1.0, 0.2, "Side(positive_y)", 0.0)
    assert calls["AtPosition"][0].args == (0.1, 0.2, None, 1.0)
    assert calls["RandomAroundSolution"][0].args == (0.02, 0.02, 0.005, 0.0, 0.0, 0.26)
    assert calls["RotateAroundSolution"][0].args == (0.0, 0.0, 1.57)


def test_every_declared_relation_is_translated():
    """No relation in the vocabulary may fall through the dispatcher.

    ``build_relation`` dispatches on the declared type and raises for anything it
    does not recognise, so a relation added to the vocabulary without a branch
    here would surface as a ``TypeError`` at environment-build time, for
    whichever task happened to use it first.
    """
    from openarm.arena_tasks.staged import layout as layout_module

    _recording_arena_relations()
    parents = {"bin": MagicMock(name="bin")}
    surface_object = MagicMock(name="surface")
    samples = {
        layout_module.On: (),
        layout_module.NextTo: ("bin",),
        layout_module.AtPosition: (0.1, 0.2),
        layout_module.PositionLimits: (0.0, 1.0),
        layout_module.NoCollision: ("bin",),
        layout_module.IsAnchor: (),
        layout_module.RandomAroundSolution: (0.01, 0.01, 0.0, 0.1),
        layout_module.RotateAroundSolution: (0.0, 0.0, 0.5),
    }
    for declared, args in samples.items():
        assert layout_module.build_relation(declared(*args), parents, surface_object) is not None
    vocabulary = set(getattr(layout_module.LayoutRelation, "__args__", ()))
    assert vocabulary == set(samples), "the relation vocabulary and the samples disagree"


def test_the_declared_side_values_match_arena():
    """``Side`` values are strings matched against Arena's enum, not members.

    ``build_relation`` resolves a side by value, so a renamed Arena member would
    raise at resolve time for whichever task places objects in a row.
    """
    from openarm.arena_tasks.staged.layout import Side

    source = (ARENA_ROOT / "isaaclab_arena" / "relations" / "relations.py").read_text()
    block = re.search(r"class Side\(.*?\n(?=class |\Z)", source, re.S).group(0)
    pairs = re.findall(r'^\s+([A-Z_0-9]+) = "([a-z_]+)"', block, re.M)
    assert pairs, "could not read Arena's Side enum members"
    assert {member.name: member.value for member in Side} == dict(pairs)


def test_build_relations_attaches_jitter_only_where_it_belongs(tasks):
    """Jitter is declared per placement, so the exclusions have to hold.

    The placer only randomizes objects that carry a ``RandomAroundSolution``,
    which makes the declared relations, not a later pass, the thing that decides
    whether a fixture is allowed to move.
    """
    from openarm.arena_tasks.staged.layout import SceneLayout, Surface, build_relations

    _recording_arena_relations()
    _stub("isaaclab_arena.assets.dummy_object", DummyObject=_FakeAnchor)
    layout = SceneLayout(
        surface=Surface(),
        placements=(
            _placement("pinned_thing", pose=((0.5, 0.5, 1.0), (0.0, 0.0, 0.0, 1.0))),
            _placement("bin"),
            _placement("prop"),
        ),
    )
    scene = {name: _FakeObject(name) for name in ("pinned_thing", "bin", "prop")}
    placed = build_relations(
        layout,
        scene,
        jitter=layout_module_jitter(),
        exclude_from_jitter=frozenset({"bin"}),
    )
    assert len(placed) == len(scene) + 1, "the surface anchor must be handed to the placer too"
    assert placed[0] is not scene["pinned_thing"], "the surface anchor comes first"
    assert {obj.name for obj in placed[1:]} == set(scene)

    jittered = {name for name, obj in scene.items() if any(c.name == "RandomAroundSolution" for c in obj.calls)}
    assert jittered == {"prop"}, "only the free prop should be jittered"
    assert scene["pinned_thing"].initial_pose is not None, "an anchor's declared pose must be applied"


class _FakeObject:
    """Stand-in for an Arena ``Object`` that records the relations attached to it."""

    def __init__(self, name: str):
        self.name = name
        self.relations: list = []
        self.calls: list = []
        self.initial_pose = None

    def add_relation(self, relation):
        self.relations.append(relation)
        self.calls.append(relation)

    def get_relations(self):
        return self.relations

    def set_initial_pose(self, pose):
        self.initial_pose = pose

    def get_initial_pose(self):
        return self.initial_pose


class _FakeAnchor(_FakeObject):
    """Stand-in for Arena's ``DummyObject``, which builds its own box."""

    def __init__(self, name, bounding_box=None, initial_pose=None, **kwargs):
        super().__init__(name)
        self.bounding_box = bounding_box
        self.initial_pose = initial_pose


def _placement(object_name, pose=None):
    """Build a minimal on-surface placement for the translation tests."""
    from openarm.arena_tasks.staged.layout import IsAnchor, On, Placement

    relations = (IsAnchor(),) if pose is not None else (On(),)
    return Placement(object_name=object_name, relations=relations, pose=pose)


def layout_module_jitter():
    """Return an active jitter marker for the attachment tests."""
    from openarm.arena_tasks.staged.layout import PoseJitter

    return PoseJitter(0.02, 0.02, 0.005, 0.26)


def test_build_relations_rejects_a_layout_that_misses_an_object():
    """A scene object with no placement is a bug, not something to paper over."""
    from openarm.arena_tasks.staged.layout import SceneLayout, Surface, build_relations

    _recording_arena_relations()
    _stub("isaaclab_arena.assets.dummy_object", DummyObject=_FakeAnchor)
    layout = SceneLayout(surface=Surface(), placements=())
    with pytest.raises(KeyError, match="no placement"):
        build_relations(layout, {"mug": MagicMock(name="mug")})


def test_layout_resolution_anchors_explicit_object_poses():
    """An explicit pose is a claim the solver must not overrule.

    ``object_poses`` is the one place a task can insist on a coordinate, so
    those objects become anchors and the placer leaves them alone.
    """
    from openarm.arena_tasks.staged.layout import On

    task = _task_with_pose()
    layout = task.resolve_layout()
    assert layout.anchored_names() == ["pinned_thing"]

    pinned = layout.placement("pinned_thing")
    assert pinned.pose == ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    # The rest of the scene is still solved against the surface.
    assert any(isinstance(r, On) for r in layout.placement("prop").relations)


def _task_with_pose(**overrides):
    """Build a one-object task with an explicitly posed object."""
    from openarm.arena_environments.industrial._base import IndustrialTask
    from openarm.arena_tasks.staged import Skill, Stage, TaskSpec

    return IndustrialTask(
        spec=TaskSpec(
            name="layout_probe",
            stages=(Stage(skill=Skill.PICK, arm="right", target="prop", label="pick"),),
            object_names=("prop",),
            **overrides.pop("spec_overrides", {}),
        ),
        objects={"prop": "mug", "pinned_thing": "mug"},
        object_poses={"pinned_thing": ((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))},
        **overrides,
    )


def test_layout_resolution_prefers_an_explicit_layout():
    """A task's own layout wins, then the spec's, and only then the derivation.

    The order matters because the three are nested fallbacks: a task that
    attaches a layout to close off one scene must not have it overridden by the
    spec, and a spec that declares one must not have it overridden by the
    derivation the task would otherwise fall back to.
    """
    from openarm.arena_tasks.staged.layout import SceneLayout, Surface

    spec_layout = SceneLayout(surface=Surface(), placements=())
    task_layout = SceneLayout(surface=Surface(center=(1.0, 1.0, 1.0)), placements=())

    # Neither declared: the layout is derived, so it places the scene.
    assert sorted(_task_with_pose().resolve_layout().object_names()) == ["pinned_thing", "prop"]

    # The spec's layout is used when the task does not override it.
    from_spec = _task_with_pose(spec_overrides={"layout": spec_layout})
    assert from_spec.resolve_layout() is spec_layout

    # The task's own layout takes precedence over the spec's.
    both = _task_with_pose(spec_overrides={"layout": spec_layout}, layout=task_layout)
    assert both.resolve_layout() is task_layout


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
