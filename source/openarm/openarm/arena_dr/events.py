"""Custom event terms used by the OpenArm domain randomization.

All event terms in this module are plain functions suitable for
:class:`isaaclab.managers.EventTermCfg`. They are written to be safe under
parallel simulation: every term is batched over environments and never indexes
a single environment implicitly.
"""

from __future__ import annotations

import torch
import warp as wp
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg

# Number of photometric parameters sampled per environment. Layout:
#   0: brightness gain
#   1: contrast gain
#   2: gamma exponent
#   3: colour cast (shared scalar across channels, fine-grained variation comes
#      from the per-channel gain drawn in :func:`sample_photometric_params`)
#   4: pixel noise standard deviation
PHOTOMETRIC_PARAM_COUNT = 5

# Key used to stash the sampled photometric parameters on the environment so
# that the observation term can read them back.
PHOTOMETRIC_KEY = "openarm_photometric_params"


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
    return env_ids


def sample_photometric_params(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    brightness_range: tuple[float, float],
    contrast_range: tuple[float, float],
    gamma_range: tuple[float, float],
    color_cast_range: tuple[float, float],
    pixel_noise_std: float,
) -> None:
    """Resample per-environment photometric augmentation parameters on reset.

    The parameters are stored on ``env.extras`` rather than baked into the
    image buffer. This is deliberate: mutating the camera data buffer in place
    would compound the augmentation across resets, because each reset would
    perturb an already-perturbed image. Storing the parameters and applying
    them as a pure function inside the observation term keeps every reset
    independent.

    Args:
        env: The environment.
        env_ids: Environments being reset. ``None`` resets all.
        brightness_range: Multiplicative brightness gain range.
        contrast_range: Multiplicative contrast gain range.
        gamma_range: Gamma exponent range.
        color_cast_range: Per-channel colour cast gain range.
        pixel_noise_std: Standard deviation of additive pixel noise, in
            normalized [0, 1] image units.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    num = env_ids.shape[0]
    device = env.unwrapped.device

    def _uniform(bounds: tuple[float, float], shape: tuple[int, ...]) -> torch.Tensor:
        return torch.empty(shape, device=device).uniform_(bounds[0], bounds[1])

    params = torch.zeros((num, PHOTOMETRIC_PARAM_COUNT), device=device)
    params[:, 0] = _uniform(brightness_range, (num,))
    params[:, 1] = _uniform(contrast_range, (num,))
    params[:, 2] = _uniform(gamma_range, (num,))
    # Per-channel colour cast: expand the shared scalar into three independent
    # channel gains so that white-balance error is modelled per channel.
    cast = _uniform(color_cast_range, (num, 3))
    params[:, 3] = cast.mean(dim=1)
    params[:, 4] = float(pixel_noise_std)

    extras = env.unwrapped.extras
    if PHOTOMETRIC_KEY not in extras or extras[PHOTOMETRIC_KEY].shape[0] != env.unwrapped.num_envs:
        extras[PHOTOMETRIC_KEY] = torch.zeros((env.unwrapped.num_envs, PHOTOMETRIC_PARAM_COUNT), device=device)
        extras[f"{PHOTOMETRIC_KEY}_cast"] = torch.ones((env.unwrapped.num_envs, 3), device=device)
    extras[PHOTOMETRIC_KEY][env_ids] = params
    extras[f"{PHOTOMETRIC_KEY}_cast"][env_ids] = cast


def get_photometric_params(env: ManagerBasedEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Read back the photometric parameters sampled by :func:`sample_photometric_params`.

    Args:
        env: The environment.

    Returns:
        Tuple of ``(params, color_cast)`` with shapes ``(N, 5)`` and ``(N, 3)``.
        Neutral values are returned if reset has not run yet, so the
        observation term is safe to evaluate on the very first step.
    """
    extras = env.unwrapped.extras
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    params = extras.get(PHOTOMETRIC_KEY)
    cast = extras.get(f"{PHOTOMETRIC_KEY}_cast")
    if params is None:
        params = torch.zeros((num_envs, PHOTOMETRIC_PARAM_COUNT), device=device)
        params[:, 0] = 1.0
        params[:, 1] = 1.0
        params[:, 2] = 1.0
    if cast is None:
        cast = torch.ones((num_envs, 3), device=device)
    return params, cast


def perturb_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    action_noise_std: float,
) -> None:
    """Add zero-mean Gaussian noise to the commanded action buffer.

    Applied every step (not just on reset) to model actuatorcommand noise. The
    action is perturbed in place after the action manager has written it, so
    the controller observes the noisy command on the same step it is issued.

    Args:
        env: The environment.
        env_ids: Unused; the action term is a single global buffer.
        action_noise_std: Standard deviation of the additive noise, in action
            units (m/rad for the delta end-effector command).
    """
    del env_ids
    if action_noise_std <= 0.0:
        return
    action_manager = env.unwrapped.action_manager
    action_term = action_manager.active_terms
    for term_name in action_term:
        term = action_manager._terms[term_name]
        action = term.action
        if action is None:
            continue
        noise = torch.randn_like(action) * float(action_noise_std)
        term.action = action + noise


def delay_action(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_delay_steps: int,
) -> None:
    """Delay the commanded action by a random number of control steps.

    Models the latency between intent and actuation on real hardware. Each
    action term keeps a short history of its own command buffer; the term then
    reads back the entry that is ``delay`` steps old, where ``delay`` is
    resampled per environment by :func:`resample_action_delay`.

    The history is stacked into a single ``(history_len, num_envs,
    num_actions)`` tensor so the read-back is a batched gather rather than a
    Python loop over environments.

    Args:
        env: The environment.
        env_ids: Unused; the action buffer is global.
        max_delay_steps: Maximum delay in control steps. Sampled uniformly from
            ``[0, max_delay_steps]`` per environment.
    """
    del env_ids
    if max_delay_steps <= 0:
        return
    action_manager = env.unwrapped.action_manager
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device
    history_len = int(max_delay_steps) + 1

    state = getattr(action_manager, "_openarm_delay_state", None)
    if state is None or state["history_len"] != history_len:
        state = {
            "history_len": history_len,
            "index": 0,
            "steps": torch.zeros(num_envs, device=device, dtype=torch.long),
            "history": {},
        }
        for term_name in action_manager.active_terms:
            term = action_manager._terms[term_name]
            if term.action is None:
                continue
            state["history"][term_name] = term.action.unsqueeze(0).repeat(history_len, 1, 1).clone()
        setattr(action_manager, "_openarm_delay_state", state)

    steps = state["steps"]
    if steps.shape[0] != num_envs:
        state["steps"] = steps = torch.zeros(num_envs, device=device, dtype=torch.long)

    index = state["index"]
    slot = (index - steps) % history_len
    env_arange = torch.arange(num_envs, device=device)

    for term_name, history in state["history"].items():
        term = action_manager._terms[term_name]
        if term.action is None:
            continue
        # Record the freshly commanded action before overwriting it.
        history[index] = term.action
        # Gather each environment's delayed entry in one batched indexing op.
        term.action = history[slot, env_arange].clone()

    state["index"] = (index + 1) % history_len


def resample_action_delay(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    max_delay_steps: int,
) -> None:
    """Resample the per-environment action delay on episode reset.

    Registered in ``reset`` mode so that the first action of an episode already
    uses the delay drawn for that episode.

    Args:
        env: The environment.
        env_ids: Environments being reset. ``None`` resets all.
        max_delay_steps: Maximum delay in control steps.
    """
    env_ids = _resolve_env_ids(env, env_ids)
    if max_delay_steps <= 0:
        return
    state = getattr(env.unwrapped.action_manager, "_openarm_delay_state", None)
    if state is None:
        return
    delays = torch.randint(0, int(max_delay_steps) + 1, (env_ids.shape[0],), device=env.unwrapped.device)
    state["steps"][env_ids] = delays
    state["index"] = 0
    # Prime the history so a nonzero delay on the first step replays a
    # well-defined action rather than stale buffer contents.
    for history in state["history"].values():
        history[:] = history[-1:]

def randomize_ee_observation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    obs_name: str,
    noise_std: float,
) -> None:
    """Add Gaussian noise to a named end-effector observation term.

    Args:
        env: The environment.
        env_ids: Unused; the observation buffer is global.
        obs_name: Name of the observation term to perturb. Accepts either a bare
            term name such as ``"left_eef_pos"`` or a fully qualified
            ``"<group>.<term>"`` name such as ``"policy.left_eef_pos"``. The
            group is checked against ``"policy"`` so a term from another group
            is never perturbed by mistake.
        noise_std: Standard deviation of the additive noise [m].

    Raises:
        ValueError: If the qualified name refers to a group other than
            ``"policy"``.
    """
    del env_ids
    if noise_std <= 0.0:
        return
    obs_manager = env.unwrapped.observation_manager
    group = obs_manager.active_terms.get("policy")
    if group is None:
        return
    # The manager's term table is keyed by bare term name, so strip any group
    # prefix before looking up. Passing "policy.left_eef_pos" straight through
    # silently matched nothing and the term was never perturbed.
    term_name = obs_name.split(".", 1)[-1] if "." in obs_name else obs_name
    if "." in obs_name and obs_name.split(".", 1)[0] != "policy":
        raise ValueError(
            f"randomize_ee_observation only perturbs the 'policy' group, but was given '{obs_name}'"
        )
    term = obs_manager._terms[group]._terms.get(term_name)
    if term is None:
        available = sorted(obs_manager._terms[group]._terms)
        raise ValueError(
            f"End-effector observation term '{term_name}' is not in the policy group. Available: {available}"
        )
    obs = term._data
    if obs is None:
        return
    obs += torch.randn_like(obs) * float(noise_std)


def randomize_light_appearance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    prim_path: str,
    intensity_range: tuple[float, float],
    color_temp_range: tuple[float, float],
) -> None:
    """Randomize the intensity and colour temperature of a dome light.

    The arena's dome light lives on a single global prim (see
    :class:`isaaclab_arena.assets.object_library.DomeLight`), so this samples
    once per call and applies to every environment. The term is registered in
    ``startup`` mode to make the sampling happen once per rollout.

    Args:
        env: The environment.
        env_ids: Unused; the light is global.
        prim_path: USD path of the dome light prim, e.g. ``/World/Light``.
        intensity_range: Intensity multiplier range applied to the authored
            intensity.
        color_temp_range: Blackbody colour temperature range [K].
    """
    del env_ids
    try:
        import omni.usd
        from pxr import UsdGeom
    except ImportError:  # pragma: no cover - only inside a full Isaac Sim runtime
        return

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    intensity = float(torch.empty(1).uniform_(intensity_range[0], intensity_range[1]).item())
    temperature = float(torch.empty(1).uniform_(color_temp_range[0], color_temp_range[1]).item())

    light = UsdGeom.DomeLight(prim)
    authored = light.GetIntensityAttr().Get()
    light.CreateIntensityAttr().Set(authored * intensity)

    # Approximate blackbody colour so warm and cool casts are both reachable
    # without pulling in a full spectral model.
    temperature = max(1000.0, temperature)
    red = min(1.0, max(0.0, temperature / 6500.0))
    green = min(1.0, max(0.0, 1.6 - abs(temperature - 6500.0) / 8000.0))
    blue = min(1.0, max(0.0, (12000.0 - temperature) / 8000.0))
    light.CreateColorAttr().Set((red, green, blue))


def get_object_position(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the world position of a scene entity, batched over environments.

    Args:
        env: The environment.
        asset_cfg: Scene entity selector.

    Returns:
        Tensor of shape ``(N, 3)`` with world-frame positions.
    """
    asset = env.unwrapped.scene[asset_cfg.name]
    return wp.to_torch(asset.data.root_pos_w)
