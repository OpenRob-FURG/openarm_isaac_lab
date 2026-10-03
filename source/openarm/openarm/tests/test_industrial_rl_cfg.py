"""The industrial runner config is authored in RSL-RL's deprecated shape on purpose.

``IndustrialStagedRslRlRunnerCfg`` sets ``policy`` and leaves ``actor``/``critic``
unset, which is the only form that works across the rsl-rl versions this repo
supports (3.x understands ``policy``; 4.x and 5.x derive the models from it).
The translation to the installed version happens once, in
``handle_deprecated_rsl_rl_cfg``.

That translation has a consequence for the env config. Arena embeds the runner
config on ``IsaacLabArenaEnvironment``, the manager cfg embeds that, and
``gym.make`` validates every nested configclass -- including the runner config,
whose model fields are still ``MISSING`` until the translation runs. This file
pins both halves: the config stays in the deprecated form, and translating it
yields the model shape the installed rsl-rl reads.

The probe runs in a subprocess. ``test_industrial_tasks.py`` deliberately
replaces ``isaaclab`` with stubs process-wide when it is collected, which would
otherwise break the genuine ``isaaclab_rl`` import this file needs, independent
of test ordering.

Run with ``pytest source/openarm/openarm/tests/test_industrial_rl_cfg.py``.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
CFG_PATH = REPO_ROOT / "source" / "openarm" / "openarm" / "arena_environments" / "industrial" / "rsl_rl_cfg.py"

#: Observation groups the actor and critic are each meant to read. This split is
#: the point of the config: the actor sees only proprioception, the critic also
#: sees the privileged task state.
EXPECTED_OBS_GROUPS = {"actor": ["policy"], "critic": ["policy", "critic"]}


def _stub_omni_ext() -> None:
    """Let pytest import the ``openarm`` package without Isaac Sim.

    pytest treats ``openarm/tests`` as part of the installed ``openarm`` package,
    so collecting this file also imports ``openarm/__init__``, which imports
    ``omni.ext``. Register only that one stub in the collecting process; the
    probe subprocess imports the real ``isaaclab_rl`` on its own.
    """
    if "omni.ext" in sys.modules:
        return

    omni = sys.modules.setdefault("omni", types.ModuleType("omni"))
    ext = types.ModuleType("omni.ext")

    class IExt:  # minimal stand-in for omni.ext.IExt, only used as a base class
        pass

    ext.IExt = IExt
    ext.acquire_extension = lambda *args, **kwargs: None
    sys.modules["omni.ext"] = ext
    omni.ext = ext


_stub_omni_ext()


_CHILD_SCRIPT = """
import importlib.metadata
import importlib.util
import json
import sys
from dataclasses import MISSING

cfg_path, out_path = sys.argv[1], sys.argv[2]

spec = importlib.util.spec_from_file_location("industrial_rsl_rl_cfg", cfg_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)

from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

version = importlib.metadata.version("rsl-rl-lib")
cfg = module.IndustrialStagedRslRlRunnerCfg(run_name="thread_nut_on_bolt")

missing = sorted(key for key, value in vars(cfg).items() if type(value) is type(MISSING))
policy_present = type(cfg.policy) is not type(MISSING)
actor_hidden = [int(dim) for dim in cfg.policy.actor_hidden_dims]
critic_hidden = [int(dim) for dim in cfg.policy.critic_hidden_dims]
obs_groups_cfg = dict(cfg.obs_groups)

handle_deprecated_rsl_rl_cfg(cfg, version)
translated = cfg.to_dict()

actor = translated["actor"]
distribution = actor.get("distribution_cfg")
if not isinstance(distribution, dict):
    distribution = {}

result = {
    "version": version,
    "missing": missing,
    "policy_present": policy_present,
    "actor_hidden": actor_hidden,
    "critic_hidden": critic_hidden,
    "obs_groups_module": dict(module.OBS_GROUPS),
    "obs_groups_cfg": obs_groups_cfg,
    "translated_keys": sorted(translated.keys()),
    "translated_obs_groups": dict(translated["obs_groups"]),
    "translated_actor_hidden": [int(dim) for dim in actor["hidden_dims"]],
    "translated_critic_hidden": [int(dim) for dim in translated["critic"]["hidden_dims"]],
    "distribution_class": distribution.get("class_name"),
    "distribution_init_std": distribution.get("init_std"),
    "init_noise_std": actor.get("init_noise_std"),
}

with open(out_path, "w") as handle:
    json.dump(result, handle)
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory) -> dict:
    """Run the config probe in a clean interpreter and return its findings."""
    if importlib.util.find_spec("isaaclab_rl") is None:
        pytest.skip("isaaclab_rl is not installed")

    out_path = tmp_path_factory.mktemp("rl_cfg_probe") / "probe.json"
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD_SCRIPT, str(CFG_PATH), str(out_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if completed.returncode != 0:
        raise AssertionError(f"config probe failed:\n{completed.stdout}\n{completed.stderr}")
    return json.loads(out_path.read_text())


def test_config_uses_the_version_portable_deprecated_form(probe):
    """``policy`` set, ``actor``/``critic`` deliberately left for the translation."""
    assert probe["policy_present"] is True
    assert probe["actor_hidden"] == [512, 256, 128]
    assert probe["critic_hidden"] == [512, 256, 128]
    assert set(probe["missing"]) == {"actor", "critic", "empirical_normalization"}


def test_actor_and_critic_get_the_asymmetric_split(probe):
    assert probe["obs_groups_module"] == EXPECTED_OBS_GROUPS
    assert probe["obs_groups_cfg"] == EXPECTED_OBS_GROUPS


def test_deprecation_translation_produces_the_model_shape_rsl_rl_reads(probe):
    """The failure mode this guards: a runner handed a config whose model fields
    are MISSING, because nothing translated ``policy`` into ``actor``/``critic``."""
    for key in ("actor", "critic", "obs_groups", "algorithm", "num_steps_per_env"):
        assert key in probe["translated_keys"], f"translated config is missing '{key}'"

    assert probe["translated_obs_groups"] == EXPECTED_OBS_GROUPS

    # rsl-rl >= 4 reads ``actor``/``critic`` and resolves the class_name from
    # them; the networks must carry their hidden dims through the translation.
    assert probe["translated_actor_hidden"] == [512, 256, 128]
    assert probe["translated_critic_hidden"] == [512, 256, 128]

    if int(probe["version"].split(".")[0]) >= 5:
        # 5.x selects the action distribution from ``distribution_cfg``.
        assert probe["distribution_class"] == "GaussianDistribution"
        assert probe["distribution_init_std"] == pytest.approx(0.8)
    else:
        # 4.x still reads the legacy stochastic parameters.
        assert probe["init_noise_std"] == pytest.approx(0.8)
