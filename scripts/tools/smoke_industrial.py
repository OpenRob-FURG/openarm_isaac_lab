"""Runtime smoke test for the industrial task suite under real Isaac Sim.

Static validation proves the specs are self-consistent. This proves they can
actually be built: assets resolve, the scene instantiates, and the environment
steps.

Uses the same Arena build path as ``scripts/record_demos.py`` so nothing here
bypasses the real registration flow. ``--arena`` swaps in Arena's own
environment registry, which is useful for telling a regression in this repo
apart from a pre-existing problem in the container.

Run inside the container (subparser flags go *after* the environment name):

    /isaac-sim/python.sh scripts/tools/smoke_industrial.py --list
    /isaac-sim/python.sh scripts/tools/smoke_industrial.py --steps 3 openarm_bimanual_industrial_thread_nut_on_bolt
    /isaac-sim/python.sh scripts/tools/smoke_industrial.py --steps 3 openarm_bimanual_industrial_count_and_stack --no-cameras
"""

"""Launch Isaac Sim first."""

import sys

from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

# NOTE: openarm.arena_environments.cli is deliberately NOT imported here. It
# pulls in the embodiment registry, which imports omniverse extension modules
# that are not loadable until after AppLauncher below has started the app.

parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--steps", type=int, default=5, help="Env steps to run per task.")
parser.add_argument("--envs", type=int, default=1, help="Sub-environments per task.")
parser.add_argument("--list", action="store_true", help="List task names and exit.")
parser.add_argument(
    "--all",
    action="store_true",
    help="Probe every industrial task in one Isaac Sim session instead of just the named one.",
)
parser.add_argument(
    "--arena",
    action="store_true",
    help="Resolve the environment name from Arena's own registry instead of openarm's. "
    "Used to get a baseline that is free of openarm-specific code.",
)

# The environment subparsers have to exist before parsing, so decide the
# listing mode first and give the required positional a placeholder when we
# only want to print names.
args_arena = "--arena" in sys.argv
if "--list" in sys.argv:
    from openarm.arena_environments.industrial import INDUSTRIAL_TASKS

    # The positional is required, so feed it a real registered name. The CLI
    # choices are prefixed while INDUSTRIAL_TASKS is keyed by bare name.
    sys.argv.append(f"openarm_bimanual_industrial_{next(iter(INDUSTRIAL_TASKS))}")
if args_arena:
    from isaaclab_arena_environments.cli import add_example_environments_cli_args as _add
else:
    from openarm.arena_environments.cli import add_example_environments_cli_args as _add
_add(parser)
args_cli = parser.parse_args()
listing = args_cli.list

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Everything below runs with Isaac Sim initialized."""

import traceback

import gymnasium as gym
import torch

from openarm.arena_environments.industrial import INDUSTRIAL_TASKS


def run_listing() -> None:
    """Print every industrial task and whether the CLI actually registered it."""
    if args_arena:
        from isaaclab_arena_environments.cli import ExampleEnvironments
    else:
        from openarm.arena_environments.cli import ExampleEnvironments

    # ExampleEnvironments is keyed by the prefixed CLI name, while
    # INDUSTRIAL_TASKS is keyed by the bare task name.
    def cli_name(key: str) -> str:
        return f"openarm_bimanual_industrial_{key}"

    print(f"\n{len(INDUSTRIAL_TASKS)} tasks registered:\n", flush=True)
    for key in INDUSTRIAL_TASKS:
        mark = "" if cli_name(key) in ExampleEnvironments else "   <-- NOT REGISTERED"
        print(f"  {key}{mark}", flush=True)
    missing = [k for k in INDUSTRIAL_TASKS if cli_name(k) not in ExampleEnvironments]
    print(f"\nregistered: {len(INDUSTRIAL_TASKS) - len(missing)}/{len(INDUSTRIAL_TASKS)}", flush=True)
    raise SystemExit(1 if missing else 0)


def _builder_factory():
    """Return a zero-arg callable that builds the CLI-selected environment.

    The builder resolves ``args_cli.example_environment`` when it is called
    rather than when it is created, which is what lets --all retarget it
    between tasks.
    """

    def build():
        if args_arena:
            from isaaclab_arena_environments.cli import get_arena_builder_from_cli
        else:
            from openarm.arena_environments.cli import get_arena_builder_from_cli

        return get_arena_builder_from_cli(args_cli).build_registered()

    return build


def run_task(gym_id: str, env_cfg, label: str) -> bool:
    """Build one environment and run the full build/scene/reset/step/oracle probe.

    Each stage is reported separately so a failure says which part of the
    pipeline broke rather than just that the task did not work.

    Args:
        gym_id: The registered Gym id to build.
        env_cfg: The compiled environment config.
        label: Human-readable task label for the report.

    Returns:
        True when every probe stage passed.
    """
    print(f"\n--- {label} ---", flush=True)
    env = None
    try:
        env = gym.make(gym_id, cfg=env_cfg)
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  build: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        return False

    ok = True
    try:
        print(f"  PASS  build: {gym_id} -> {type(env).__name__}", flush=True)

        # gym.make wraps the env in OrderEnforcing; everything below needs the
        # manager-based env itself.
        base = env.unwrapped
        scene = base.scene
        names = sorted(scene.keys()) if hasattr(scene, "keys") else []
        if not names:
            raise AssertionError("scene has no assets")
        print(f"  PASS  scene: {len(names)} assets -> {', '.join(names)}", flush=True)

        obs, _ = env.reset()
        if not isinstance(obs, dict):
            raise AssertionError(f"expected dict obs, got {type(obs).__name__}")
        print(f"  PASS  reset: obs keys {sorted(obs)}", flush=True)

        # A zero action keeps the probe deterministic: it proves the whole
        # action/observation/termination pipeline executes without asserting
        # anything about task success, which the oracle is for.
        act_dim = int(base.action_manager.total_action_dim)
        zero_action = torch.zeros(base.num_envs, act_dim, device=base.device)
        shapes = None
        for _ in range(args_cli.steps):
            obs, rew, terminated, truncated, info = env.step(zero_action)
            shapes = {k: tuple(v.shape) for k, v in obs.items() if hasattr(v, "shape")}
        print(f"  PASS  step: {args_cli.steps} steps, action_dim={act_dim}", flush=True)
        print(f"        obs shapes {shapes}", flush=True)

        # Drive one step from the scripted oracle to prove the policy
        # integration is wired, not just importable.
        # The builder attaches the oracle to the Arena environment object, which
        # the RL env only keeps a reference to via its cfg, so probe both.
        cfg = getattr(base, "cfg", None)
        arena_env = getattr(cfg, "isaaclab_arena_env", None) if cfg is not None else None
        oracle = None
        for holder in (base, cfg, arena_env):
            oracle = getattr(holder, "oracle_policy", None)
            if oracle is not None:
                break

        if oracle is not None and act_dim:
            o = base.observation_manager.compute()
            # Arena's PolicyBase contract hands the policy the Gym env; the
            # oracle resolves the embodiment and scene itself via skills.
            oracle_action = oracle.get_action(base, o)
            if isinstance(oracle_action, dict):
                oracle_action = torch.cat(
                    [
                        torch.as_tensor(v, dtype=torch.float32).reshape(base.num_envs, -1)
                        for v in oracle_action.values()
                    ],
                    dim=-1,
                )
            oracle_action = torch.as_tensor(oracle_action, dtype=torch.float32, device=base.device)
            if oracle_action.dim() == 1:
                oracle_action = oracle_action.unsqueeze(0)
            assert oracle_action.shape == (base.num_envs, act_dim), (
                f"oracle action shape {tuple(oracle_action.shape)} != {(base.num_envs, act_dim)}"
            )
            assert torch.isfinite(oracle_action).all(), "oracle action has non-finite values"
            env.step(oracle_action)
            print(f"  PASS  oracle: action {tuple(oracle_action.shape)} accepted by the env", flush=True)
        else:
            print("  SKIP  oracle: no oracle_policy on the environment", flush=True)
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAIL  probe: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    finally:
        if env is not None:
            env.close()
    return ok


def main():
    """Probe the single environment named on the command line."""
    get_builder = _builder_factory()
    requested = args_cli.example_environment

    # Industrial CLI names are prefixed; INDUSTRIAL_TASKS is keyed by bare name.
    # Non-industrial envs (the pre-existing openarm ones) are allowed through so
    # this script can be used to tell a regression apart from a pre-existing
    # problem with the shared asset library.
    task_key = next(
        (k for k in INDUSTRIAL_TASKS if f"openarm_bimanual_industrial_{k}" == requested), requested
    )
    task = INDUSTRIAL_TASKS.get(task_key)
    label = f"{task_key} (background={task.background}, dr={task.dr_preset})" if task else task_key

    try:
        gym_id, env_cfg = get_builder()
    except Exception as exc:  # noqa: BLE001
        print(f"\n--- {label} ---", flush=True)
        print(f"  FAIL  build: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise SystemExit(1) from exc

    raise SystemExit(0 if run_task(gym_id, env_cfg, label) else 1)


def main_all():
    """Probe every industrial task in one Isaac Sim session.

    Restarting the app per task costs minutes each, and a single process also
    proves the tasks compose in one interpreter rather than only in isolation.
    """
    get_builder = _builder_factory()
    total = len(INDUSTRIAL_TASKS)
    passed, failures = 0, []
    for i, (task_key, task) in enumerate(INDUSTRIAL_TASKS.items(), start=1):
        gym_name = f"openarm_bimanual_industrial_{task_key}"
        label = f"[{i}/{total}] {task_key} (background={task.background}, dr={task.dr_preset})"
        # The builder reads this attribute when it is called, so pointing it at
        # the next task is all that is needed to build the next environment.
        args_cli.example_environment = gym_name
        try:
            gym_id, env_cfg = get_builder()
        except Exception as exc:  # noqa: BLE001
            print(f"\n--- {label} ---", flush=True)
            print(f"  FAIL  build: {type(exc).__name__}: {exc}", flush=True)
            failures.append((task_key, f"build: {exc}"))
            continue
        if run_task(gym_id, env_cfg, label):
            passed += 1
        else:
            failures.append((task_key, "probe failed"))

    print(f"\n===== industrial smoke summary: {passed}/{total} passed =====", flush=True)
    for task_key, why in failures:
        print(f"  FAIL  {task_key}: {why}", flush=True)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    if listing:
        run_listing()
    elif args_cli.all:
        main_all()
    else:
        main()
    simulation_app.close()
