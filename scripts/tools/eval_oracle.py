"""Measure an industrial task's oracle success rate and explain each failure.

Where ``smoke_industrial.py`` proves the pipeline runs, this script actually
closes the loop: it drives the scripted oracle for whole episodes, checks the
task's own success predicate, and reports why an episode ended the way it did.
That is the signal the oracle gets iterated against.

The success check re-evaluates the task's declared goals rather than trusting the
environment's ``terminated`` flag, because ``terminated`` is the OR of every
termination term and cannot say which one fired or whether the goal was ever met.
"""

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser

# The Arena base parser carries --num_envs/--env_spacing that the env builder
# reads, and the per-environment flags then hang off its subparsers.
parser = get_isaaclab_arena_cli_parser()
parser.add_argument("--episodes", type=int, default=3, help="Episodes to run.")
parser.add_argument("--max-steps", type=int, default=0, help="Override the step cap (0 = episode length).")
parser.add_argument("--trace", action="store_true", help="Print a per-step timeline for the first episode.")
parser.add_argument("--trace-stride", type=int, default=10, help="Steps between trace rows.")
parser.add_argument("--task", type=str, default=None, help="Override the industrial task key to evaluate.")
parser.add_argument("--verbose-fail", action="store_true", help="Print a full trace for every failed episode.")
parser.add_argument("--capture-dir", type=str, default=None, help="Save RGB observation frames here when cameras are enabled.")
parser.add_argument("--no-cameras", "--no_cameras", dest="no_cameras", action="store_true",
                    help="Disable the task RGB camera to speed up headless rollouts.")

from isaaclab.app import AppLauncher  # noqa: E402

from openarm.arena_environments.cli import add_example_environments_cli_args  # noqa: E402

# The environment name is a subparser positional, so the per-environment flags
# have to be added to this same parser and only parse after it.
add_example_environments_cli_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=False, enable_cameras=args_cli.enable_cameras and not args_cli.no_cameras)
simulation_app = app_launcher.app


import torch  # noqa: E402
from pathlib import Path  # noqa: E402

from openarm.arena_environments.cli import get_arena_builder_from_cli  # noqa: E402
from openarm.arena_environments.industrial import INDUSTRIAL_TASKS  # noqa: E402
from openarm.arena_tasks.staged.terminations import evaluate_goal  # noqa: E402
from openarm.policies.oracle.skills import (  # noqa: E402
    _eef_names,
    _usd_antipodal_geometry,
    _usd_path_for_object,
    eef_pose,
    object_pose,
)
from openarm.policies.oracle.staged_oracle import StagedOraclePolicy, StagedOraclePolicyCfg  # noqa: E402


def goal_specs(spec):
    """Return (label, params) for every goal the task declares."""
    out = []
    for stage in spec.stages:
        if stage.goal is None:
            continue
        for goal in getattr(stage.goal, "goals", ()) or ():
            out.append((goal.describe(), goal.params()))
        out.append((stage.goal.describe(), stage.goal.params()))
    return out


def evaluate(env, spec, env_id=0):
    """Return [(label, held_bool)] for every goal in the spec."""
    results = []
    for label, params in goal_specs(spec):
        try:
            results.append((label, bool(evaluate_goal(env, params)[env_id].item())))
        except Exception as exc:  # noqa: BLE001
            results.append((label, f"error: {type(exc).__name__}: {exc}"))
    return results


def main():
    requested = args_cli.example_environment
    task_key = next((k for k in INDUSTRIAL_TASKS if f"openarm_bimanual_industrial_{k}" == requested), requested)
    if args_cli.task:
        task_key = args_cli.task
    if task_key not in INDUSTRIAL_TASKS:
        raise SystemExit(f"unknown industrial task '{task_key}'")
    task = INDUSTRIAL_TASKS[task_key]
    spec = task.spec

    builder = get_arena_builder_from_cli(args_cli)
    gym_id, env_cfg = builder.build_registered()

    import gymnasium as gym

    env = gym.make(gym_id, cfg=env_cfg)
    base = env.unwrapped
    oracle = StagedOraclePolicy(StagedOraclePolicyCfg(task=task_key))

    # The stage budget already bounds a single stage; the environment's own time
    # out is the real episode cap, so only override it when asked.
    if args_cli.max_steps:
        base.cfg.terminations.time_out = None
        cap = args_cli.max_steps
    else:
        decimation = base.cfg.decimation
        dt = base.cfg.sim.dt
        cap = int(spec.episode_length_s / (dt * decimation))

    results = []
    print(f"\n=== {task_key} | {spec.name} | {spec.instruction}")
    print(f"    stages={[s.skill.value for s in spec.stages]} cap={cap} steps episodes={args_cli.episodes}")

    for ep in range(args_cli.episodes):
        base.cfg.seed = (args_cli.seed or 0) + ep
        obs, _ = env.reset()
        oracle.reset()
        if args_cli.trace and ep == 0:
            arms = {arm: fmt(eef_pose(base, arm, 0)[:3, 3]) for arm in _eef_names(base)}
            print(f"    initial_end_effectors={arms}", flush=True)
            if spec.stages[0].target:
                name = spec.stages[0].target
                usd_path = _usd_path_for_object(base, name)
                geometry = _usd_antipodal_geometry(usd_path) if usd_path else None
                geometry_summary = None if geometry is None else (geometry[0], len(geometry[1]), geometry[2].shape)
                print(
                    f"    target_usd={usd_path} mesh_geometry={geometry_summary}",
                    flush=True,
                )
                if geometry is not None:
                    eef = eef_pose(base, spec.stages[0].arm, 0)
                    obj_pose = object_pose(base, name, 0)
                    jaw_axis = obj_pose[:3, :3].T @ eef[:3, 1]
                    import numpy as np

                    jaw_np = jaw_axis.detach().cpu().numpy()
                    alignment = geometry[3] @ jaw_np
                    width = float((geometry[2] @ jaw_np).max() - (geometry[2] @ jaw_np).min())
                    print(
                        f"    tcp_rotation={eef[:3, :3].detach().cpu().tolist()} "
                        f"jaw_axis_local={jaw_np.tolist()} width={width:.4f} "
                        f"opposing_normals={float(alignment.max()):.3f}/{float((-alignment).max()):.3f}",
                        flush=True,
                    )
        if args_cli.capture_dir:
            save_rgb_frame(obs, Path(args_cli.capture_dir) / f"episode_{ep:03d}_step_00000.png")

        trace = []
        steps = 0
        outcome = "timeout"
        # ManagerBasedRLEnv resets a terminated sub-environment inside step() and
        # returns the *reset* state, so the goal and the poses are sampled before
        # stepping: that reading still describes the live episode, and the
        # termination manager's own term is the authority on success.
        goal_results = evaluate(base, spec)
        final, goal_box, dist = final_geometry(base, spec)
        while steps < cap:
            action = oracle.get_action(base, obs)
            if args_cli.trace and ep == 0 and steps == 0:
                print(f"    first_action={action[0].detach().cpu().tolist()}", flush=True)
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            if args_cli.capture_dir and args_cli.trace_stride > 0 and steps % args_cli.trace_stride == 0:
                save_rgb_frame(obs, Path(args_cli.capture_dir) / f"episode_{ep:03d}_step_{steps:05d}.png")
            if bool(terminated[0]) or bool(truncated[0]):
                outcome = "terminated" if bool(terminated[0]) else "truncated"
                break
            if args_cli.trace and args_cli.trace_stride > 0 and steps % args_cli.trace_stride == 0:
                trace.append((steps, snapshot(base, spec, oracle)))
            goal_results = evaluate(base, spec)
            final, goal_box, dist = final_geometry(base, spec)
        if args_cli.trace and ep == 0:
            print_trace(trace, spec, task_key)

        success = bool(env.unwrapped.termination_manager.get_term("success")[0])
        state = oracle.states[0]
        results.append(success)
        status = "SUCCESS" if success else outcome.upper()
        print(f"  [{ep}] {status:<9} steps={steps:<4} stage={state.index}/{len(spec.stages)} "
              f"attempts={state.attempt} results={state.stage_results}")
        print(f"       object={fmt(final)} goal={fmt(goal_box)} dist={dist:.3f}")
        for label, value in goal_results:
            mark = "ok " if value is True else "NO "
            print(f"       [{mark}] {label}" + ("" if value is True else f"  -> {value}"))
        if args_cli.verbose_fail and not success and ep > 0:
            print_trace(trace, spec, task_key)

    env.close()
    rate = sum(results) / max(1, len(results))
    print(f"\n=== SUCCESS RATE {sum(results)}/{len(results)} = {rate:.0%}\n")
    raise SystemExit(0 if rate == 1.0 else 1)


def snapshot(base, spec, oracle):
    """Grab the numbers worth watching at one step."""
    stage = spec.stages[min(oracle.states[0].index, len(spec.stages) - 1)]
    from openarm.policies.oracle.skills import eef_pose

    eef = eef_pose(base, stage.arm, 0)[:3, 3]
    arm_poses = {arm: eef_pose(base, arm, 0)[:3, 3] for arm in _eef_names(base)}
    obj = object_pose(base, stage.target, 0)[:3, 3] if stage.target else torch.zeros(3)
    ctrl = oracle.controllers[0]
    target = getattr(ctrl, "pregrasp", None)
    if target is None:
        target = getattr(ctrl, "target", None)
    target_pos = target[:3, 3] if target is not None else None
    robot = base.scene["robot"]
    from isaaclab.utils.array import convert_to_torch

    finger_ids = robot.find_joints(
        [f"openarm_{stage.arm}_finger_joint1", f"openarm_{stage.arm}_finger_joint2"], preserve_order=True
    )[0]
    finger_body_ids = robot.find_bodies(
        [f"openarm_{stage.arm}_right_finger", f"openarm_{stage.arm}_left_finger"], preserve_order=True
    )[0]
    joint_pos = convert_to_torch(robot.data.joint_pos, dtype=torch.float32, device=base.device)
    body_pos = convert_to_torch(robot.data.body_pos_w, dtype=torch.float32, device=base.device)
    return {
        "phase": getattr(ctrl, "phase", "?"),
        "eef": eef,
        "arms": {arm: fmt(pos) for arm, pos in arm_poses.items()},
        "obj": obj,
        "gap": float(torch.norm(eef - obj)),
        "target": target_pos,
        "target_gap": float(torch.norm(eef - target_pos)) if target_pos is not None else float("nan"),
        "grip": float(ctrl.grasped) if hasattr(ctrl, "grasped") else float("nan"),
        "fingers": joint_pos[0, finger_ids].detach().cpu().tolist(),
        "finger_bodies": body_pos[0, finger_body_ids].detach().cpu().tolist(),
    }


def print_trace(trace, spec, task_key):
    """Print a per-step timeline."""
    print(f"\n  --- trace: {task_key} ---")
    for steps, s in trace:
        target = "" if s["target"] is None else f" target={fmt(s['target'])} err={s['target_gap']:.3f}"
        print(
            f"    t={steps:<4} phase={s['phase']:<10} eef={fmt(s['eef'])} obj={fmt(s['obj'])} "
            f"gap={s['gap']:.3f} arms={s['arms']} fingers={s['fingers']} "
            f"finger_bodies={s['finger_bodies']}{target}"
        )


def final_geometry(base, spec):
    """Return (object position, goal centre, distance) for the primary object."""
    stage = spec.stages[0]
    if not stage.target:
        return torch.zeros(3), torch.zeros(3), float("nan")
    pos = object_pose(base, stage.target, 0)[:3, 3]
    goal = stage.goal
    centre = getattr(goal, "center", None)
    if centre is None and getattr(goal, "goals", None):
        centre = getattr(goal.goals[0], "center", None)
    if centre is None:
        return pos, torch.zeros(3), float("nan")
    centre_t = torch.as_tensor(centre, dtype=torch.float32, device=pos.device)
    return pos, centre_t, float(torch.norm(pos - centre_t))


def fmt(v):
    """Format a 3-vector compactly."""
    return "[" + " ".join(f"{float(x):+.3f}" for x in v) + "]"


def save_rgb_frame(obs, path):
    """Save the first task RGB observation for visual rollout inspection."""
    from PIL import Image

    rgb = obs.get("task_obs", {}).get("rgb") if isinstance(obs, dict) else None
    if rgb is None:
        raise RuntimeError("--capture-dir requested, but the observation has no task_obs.rgb camera image")
    frame = rgb[0].detach().cpu()
    if frame.dtype.is_floating_point and (frame.numel() == 0 or float(frame.max()) <= 1.0):
        frame = frame.clamp(0, 1).mul(255)
    frame = frame.clamp(0, 255).byte().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(path)


if __name__ == "__main__":
    main()
    simulation_app.close()
