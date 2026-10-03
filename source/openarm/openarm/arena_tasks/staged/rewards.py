"""Dense shaped reward terms for staged OpenArm tasks.

The success predicates in :mod:`openarm.arena_tasks.staged.terminations` are
boolean, which is the right thing to end an episode on and the wrong thing to
learn from: a policy that has to cross a whole pick-and-place before it learns
anything is a policy that never learns anything. Everything here is a
*continuous* counterpart of that same vocabulary, so one declarative
:class:`~openarm.arena_tasks.staged.stages.TaskSpec` drives both what ends an
episode and what the policy is rewarded for on the way there.

Three conventions hold throughout:

* Every function returns a tensor of shape ``(num_envs,)`` and is vectorized
  across parallel environments. There is no Python loop over environments and
  no ``.item()`` anywhere in this module.
* Shaping terms return a **non-negative score**, normally in ``[0, 1]``, and
  penalties return a **non-negative cost**. The sign lives in the
  ``RewardTermCfg`` weight, not in the return value, which is what lets a
  single term be reused with either sign.
* Terms that mirror a boolean goal use :func:`tolerance_score`, which is
  **exactly 1.0 across the whole success set** and decays smoothly outside it.
  Terms with no boolean counterpart -- end-effector approach, alignment, lift
  height, and the rest -- use :func:`tanh_score`, which peaks at the target.
  The split matters: a peaking kernel applied to a toleranced goal pays only
  0.24 at the tolerance boundary, so the shaping surface is nearly flat right
  where the episode's reward actually changes and the policy gets no push at
  the moment it is about to succeed. Gating instead makes the dense score and
  the boolean predicate agree by construction, since both are read off the same
  tolerance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from openarm.arena_tasks.staged.scene_state import (
    ee_orientation,
    ee_position,
    gripper_opening,
    normalized_joint_position,
    object_angular_speed,
    object_position,
    object_speed,
    tilt_from_vertical,
    to_torch,
)
from openarm.arena_tasks.staged.terminations import evaluate_goal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

#: How the per-goal scores of a multi-goal task are combined. Chosen per term
#: because they answer different questions: ``mean`` asks "how far along the
#: task am I overall", ``min`` asks "what is the worst part still missing", and
#: ``product`` asks "how close am I to the conjunction succeeding at once".
AGGREGATIONS = ("mean", "min", "product")

#: World-frame direction the end-effector's own +Z should point for a top-down
#: grasp. The OpenArm finger frames point their +Z out of the gripper, so a
#: grasp from above aligns this axis with world -Z.
TOP_DOWN_DIRECTION = (0.0, 0.0, -1.0)

#: Shaping scale used when a goal declares no tolerance of its own [m]. Chosen
#: an order of magnitude above the tightest tolerance in the goal vocabulary
#: (0.02 m) and an order below the loosest plausible region, so a goal without a
#: tolerance still gets a kernel that varies over the range the task spans
#: rather than one that is saturated everywhere it matters.
DEFAULT_SCALE = 0.1

#: Goal tolerances that double as the default decay scale. A goal that declares
#: none leaves the scale to each kernel's own default.
TOLERANCE_KEYS = ("tolerance", "max_distance")

#: Width of the ramp either side of a normalized-joint goal's threshold, in
#: normalized joint units. The joint goals are the one case with no tolerance
#: parameter to borrow, because their threshold *is* the bound: the predicate
#: is ``value >= threshold``, so a ramp that reaches 1.0 at the threshold is
#: already the gated kernel in all but name.
JOINT_SCALE = 0.1


# ----------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------


def tanh_score(error: torch.Tensor, std: float) -> torch.Tensor:
    """Map a non-negative error to a score in ``(0, 1]`` that peaks at zero.

    ``1 - tanh(error / std)`` is one at zero error, about 0.24 at one ``std``,
    and decays to zero far away. It is smooth everywhere, so there is no kink
    for a policy to get stuck on, and saturating, so a hopeless configuration
    cannot accumulate arbitrarily negative reward and dominate the gradient.

    Use this for shaping that has no boolean counterpart. For a goal that *does*
    have a success tolerance, use :func:`tolerance_score` instead.

    Args:
        error: Non-negative error tensor of any shape.
        std: Characteristic error scale at which the score is roughly halved.

    Returns:
        Tensor of the same shape as ``error``.

    Raises:
        ValueError: If ``std`` is not positive, which would make the kernel
            constant or infinite and silently remove the gradient.
    """
    if float(std) <= 0.0:
        raise ValueError(f"tanh_score() needs a positive std, got {std}.")
    rew = 1.0 - torch.tanh(error / float(std))
    return rew


def tolerance_score(
    error: torch.Tensor,
    tolerance: float,
    scale: float | None = None,
) -> torch.Tensor:
    """Map an error to 1.0 inside ``tolerance`` and decay smoothly outside it.

    This is the kernel every goal-mirroring term below is built from, and the
    reason it exists is the invariant it guarantees:

        score == 1.0  if and only if  the boolean predicate holds.

    Both sides are read off the same ``tolerance``, so dense shaping and episode
    termination cannot drift apart as a task is retuned. Note what the kernel
    deliberately does *not* do: there is no gradient inside the tolerance. That
    is correct rather than lazy -- once a goal is satisfied there is nothing
    left to earn from it, so paying more for being at the exact centre of a
    region than for merely being inside it would invite the policy to sit in
    the middle of a goal rather than to finish the task and move on.

    Args:
        error: Non-negative error tensor of any shape, in the goal's own units.
        tolerance: The success threshold, in the same units as ``error``.
        scale: Error beyond the tolerance at which the score falls to
            ``exp(-1)``. Defaults to ``tolerance``, which puts the score at
            0.37 once the error is twice the tolerance.

    Returns:
        Tensor of the same shape as ``error``, in ``(0, 1]``.

    Raises:
        ValueError: If ``tolerance`` is negative or ``scale`` is not positive.
    """
    tolerance = float(tolerance)
    if tolerance < 0.0:
        raise ValueError(f"tolerance_score() needs a non-negative tolerance, got {tolerance}.")
    if scale is None:
        scale = tolerance
    if float(scale) <= 0.0:
        raise ValueError(f"tolerance_score() needs a positive scale, got {scale}.")
    excess = (error - tolerance).clamp_min(0.0)
    return torch.exp(-excess / float(scale))


def ramp_score(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Linearly map a value in ``[low, high]`` to a score in ``[0, 1]``.

    Unlike :func:`tanh_score` this is exact rather than asymptotic, which is
    what the normalized-joint goals want: the underlying predicate is a
    threshold on a coordinate that is already normalized to ``[0, 1]``, so a
    kernel that reaches 1.0 and 0.0 at the declared bounds says exactly what
    the goal says, only continuously.

    The two bounds are pinned rather than left to the division, because
    ``(high - low) / (high - low)`` is not reliably 1.0 in float32. A score that
    reaches 0.9999999 on a threshold the task declares met is a small but real
    disagreement between a shaping term and the predicate it is meant to mirror.

    Args:
        value: Tensor of any shape.
        low: Value mapping to 0.
        high: Value mapping to 1.

    Returns:
        Tensor of the same shape as ``value``.

    Raises:
        ValueError: If ``high <= low``, which would make the ramp ill-defined.
    """
    if float(high) <= float(low):
        raise ValueError(f"ramp_score() needs low < high, got low={low}, high={high}.")
    ramped = ((value - float(low)) / (float(high) - float(low))).clamp(0.0, 1.0)
    ones = torch.ones_like(ramped)
    zeros = torch.zeros_like(ramped)
    return torch.where(value >= float(high), ones, torch.where(value <= float(low), zeros, ramped))


def _as_vector(values: tuple[float, ...] | list[float], reference: torch.Tensor) -> torch.Tensor:
    """Broadcast a Python 3-vector to a ``(1, 3)`` tensor on ``reference``'s device.

    Args:
        values: The three components.
        reference: Tensor supplying device and dtype.

    Returns:
        Tensor of shape ``(1, 3)``.
    """
    return torch.tensor(values, device=reference.device, dtype=reference.dtype).view(1, 3)


# ----------------------------------------------------------------------
# Per-goal dense scores
# ----------------------------------------------------------------------


def band_distance(value: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """Distance from a scalar to the band ``[low, high]``, zero inside it.

    Args:
        value: Tensor of any shape.
        low: Lower bound.
        high: Upper bound.

    Returns:
        Tensor of the same shape as ``value``.
    """
    below = (float(low) - value).clamp_min(0.0)
    above = (value - float(high)).clamp_min(0.0)
    return below + above


def in_region_score(
    env: "ManagerBasedRLEnv",
    object_cfg: str,
    center: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    scale: float | None = None,
) -> torch.Tensor:
    """Dense counterpart of the ``in_region`` goal.

    The tolerance is the region's own half-extent, applied per axis, because
    the boolean predicate tests each axis independently. The axes are combined
    with a product rather than a minimum so that a region the object is
    straddling one edge of scores near zero on that edge and can be seen to be
    improving along it.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        center: Region centre in the environment frame [m].
        half_extents: Region half-size along each axis [m].
        scale: Override for the decay scale, which defaults to each axis's own
            half-extent.

    Returns:
        Tensor of shape ``(N,)``: 1.0 inside the region, decaying outside it.

    Raises:
        ValueError: If any half-extent is not positive.
    """
    if any(float(half) <= 0.0 for half in half_extents):
        raise ValueError(f"in_region_score() needs positive half_extents, got {half_extents}.")
    position = object_position(env, object_cfg)
    offset = (position - _as_vector(center, position)).abs()
    half = _as_vector(half_extents, position)
    axes = [tolerance_score(offset[:, axis], float(half_extents[axis]), scale) for axis in range(3)]
    return axes[0] * axes[1] * axes[2]


def relative_to_score(
    env: "ManagerBasedRLEnv",
    object_cfg: str,
    reference_cfg: str,
    offset: tuple[float, float, float],
    tolerance: float,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense counterpart of the ``relative_to`` goal.

    Args:
        env: The environment.
        object_cfg: Scene name of the moving object.
        reference_cfg: Scene name of the reference object.
        offset: Required position offset from the reference [m].
        tolerance: Allowed Euclidean deviation [m]. This is the goal's own
            threshold, so the score is 1.0 exactly where the goal holds.
        scale: Override for the decay scale beyond ``tolerance``.

    Returns:
        Tensor of shape ``(N,)``: 1.0 within tolerance, decaying outside it.
    """
    position = object_position(env, object_cfg)
    reference = object_position(env, reference_cfg)
    offset_t = _as_vector(offset, position)
    return tolerance_score(torch.norm((position - reference) - offset_t, dim=1), tolerance, scale)


def upright_score(
    env: "ManagerBasedRLEnv",
    object_cfg: str,
    tolerance: float,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense counterpart of the ``upright`` goal.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        tolerance: Maximum tilt the goal allows [rad]. The score is 1.0 up to
            this tilt and decays past it.
        scale: Override for the decay scale beyond the tolerance.

    Returns:
        Tensor of shape ``(N,)``: 1.0 within tolerance, decaying outside it.
    """
    return tolerance_score(tilt_from_vertical(env, object_cfg), tolerance, scale)


def within_distance_score(
    env: "ManagerBasedRLEnv",
    object_cfg: str,
    other_cfg: str,
    max_distance: float,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense counterpart of the ``within_distance`` goal.

    Args:
        env: The environment.
        object_cfg: Scene name of the first object.
        other_cfg: Scene name of the second object.
        max_distance: Maximum centre-to-centre distance [m], the goal's own
            threshold.
        scale: Override for the decay scale beyond ``max_distance``.

    Returns:
        Tensor of shape ``(N,)``: 1.0 within range, decaying outside it.
    """
    position = object_position(env, object_cfg)
    other = object_position(env, other_cfg)
    return tolerance_score(torch.norm(position - other, dim=1), max_distance, scale)


def joint_above_score(
    env: "ManagerBasedRLEnv",
    asset_cfg: str,
    joint_name: str,
    threshold: float,
    std: float = JOINT_SCALE,
) -> torch.Tensor:
    """Dense counterpart of the ``joint_above`` goal.

    Shapes the approach to the threshold from below and saturates above it, so
    continuing to open a door further than required earns nothing extra.

    Args:
        env: The environment.
        asset_cfg: Scene name of the articulation.
        joint_name: Name of the joint.
        threshold: Normalized threshold the goal requires.
        std: Normalized band below the threshold over which the score ramps up.

    Returns:
        Tensor of shape ``(N,)`` in ``[0, 1]``.
    """
    value = normalized_joint_position(env, asset_cfg, joint_name)
    return ramp_score(value, float(threshold) - float(std), float(threshold))


def joint_below_score(
    env: "ManagerBasedRLEnv",
    asset_cfg: str,
    joint_name: str,
    threshold: float,
    std: float = JOINT_SCALE,
) -> torch.Tensor:
    """Dense counterpart of the ``joint_below`` goal.

    The normalized coordinate *is* the error here, so this is
    :func:`tolerance_score` with the goal's own threshold: 1.0 at or below the
    threshold, decaying as the joint travels away from it in the wrong
    direction.

    This deliberately does not use :func:`ramp_score` the way
    :func:`joint_above_score` does. A ramp from ``threshold`` to
    ``threshold + std`` would saturate at 1.0 for a joint that is far *above*
    the threshold, which is the opposite of what a "close it" goal wants, and
    would pay a policy in full for the single state that is furthest from the
    goal.

    Args:
        env: The environment.
        asset_cfg: Scene name of the articulation.
        joint_name: Name of the joint.
        threshold: Normalized threshold the goal requires.
        std: Normalized band above the threshold over which the score decays.

    Returns:
        Tensor of shape ``(N,)``: 1.0 at or below the threshold, decaying above.
    """
    value = normalized_joint_position(env, asset_cfg, joint_name)
    return tolerance_score(value, float(threshold), std)


def at_rest_score(
    env: "ManagerBasedRLEnv",
    object_cfg: str,
    max_speed: float,
    max_angular_speed: float,
    scale: float | None = None,
) -> torch.Tensor:
    """Dense counterpart of the ``at_rest`` goal.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        max_speed: Linear speed the goal allows [m/s].
        max_angular_speed: Angular speed the goal allows [rad/s].
        scale: Override for the decay scale beyond each threshold.

    Returns:
        Tensor of shape ``(N,)``. A product rather than a minimum, because an
            object drifting slowly while spinning is not settled and should not
            score as though it were.
    """
    linear = tolerance_score(object_speed(env, object_cfg), max_speed, scale)
    angular = tolerance_score(object_angular_speed(env, object_cfg), max_angular_speed, scale)
    return linear * angular


def above_plane_score(
    env: "ManagerBasedRLEnv",
    object_cfg: str,
    min_height: float,
    max_height: float,
    std: float = DEFAULT_SCALE,
) -> torch.Tensor:
    """Dense counterpart of the ``above_plane`` goal.

    The goal is a band, so its tolerance is zero and only the decay scale
    matters: the score is flat at 1.0 anywhere inside the band and falls off
    with height once outside it.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        min_height: Lower bound on world-frame height [m].
        max_height: Upper bound on world-frame height [m].
        std: Height outside the band at which the score is roughly halved [m].

    Returns:
        Tensor of shape ``(N,)``: 1.0 inside the band, decaying outside it.
    """
    height = object_position(env, object_cfg)[:, 2]
    return tolerance_score(band_distance(height, min_height, max_height), 0.0, std)


#: Dense handler per goal ``kind``, matching the boolean dispatch in
#: :mod:`openarm.arena_tasks.staged.terminations` one for one. Keyed on the
#: explicit ``kind`` the goals emit, so adding a goal kind cannot silently
#: change how an existing one is routed.
#:
#: Each handler receives the goal's *own* success threshold alongside the
#: optional decay ``scale``, which is what lets :func:`tolerance_score` put the
#: shaping surface and the success threshold on the same number. ``in_region``
#: and ``above_plane`` read their thresholds out of the goal's own geometry, so
#: they need no extra argument.
_DENSE_DISPATCH: dict[str, Any] = {
    "in_region": lambda env, p, scale: in_region_score(env, p["object_cfg"], p["center"], p["half_extents"], scale),
    "relative_to": lambda env, p, scale: relative_to_score(
        env, p["object_cfg"], p["reference_cfg"], p["offset"], p["tolerance"], scale
    ),
    "upright": lambda env, p, scale: upright_score(env, p["object_cfg"], p["tolerance"], scale),
    "within_distance": lambda env, p, scale: within_distance_score(
        env, p["object_cfg"], p["other_cfg"], p["max_distance"], scale
    ),
    "joint_above": lambda env, p, scale: joint_above_score(
        env, p["asset_cfg"], p["joint_name"], p["threshold"], JOINT_SCALE if scale is None else scale
    ),
    "joint_below": lambda env, p, scale: joint_below_score(
        env, p["asset_cfg"], p["joint_name"], p["threshold"], JOINT_SCALE if scale is None else scale
    ),
    "at_rest": lambda env, p, scale: at_rest_score(
        env, p["object_cfg"], p["max_speed"], p["max_angular_speed"], scale
    ),
    "above_plane": lambda env, p, scale: above_plane_score(
        env, p["object_cfg"], p["min_height"], p["max_height"], scale if scale is not None else DEFAULT_SCALE
    ),
    "all": lambda env, p, scale: aggregate_goal_scores(env, p["goals"], "mean", scale),
    "at_least": lambda env, p, scale: at_least_score(env, p["goals"], p["count"], scale),
}


def default_scale(params: dict[str, Any]) -> float | None:
    """Return the decay scale a goal implies, or ``None`` to use each kernel's own.

    The tolerant kernels already default their scale to the goal's tolerance,
    which is the only sensible default: a decay length unrelated to the
    threshold would be one more number per task to keep in step with the rest of
    the declaration. Only a caller that has a reason to pick a different decay
    should pass an explicit one.

    Args:
        params: Parameters produced by
            :meth:`openarm.arena_tasks.staged.goals.Goal.params`.

    Returns:
        The goal's own tolerance where it has one, else ``None``.
    """
    for key in TOLERANCE_KEYS:
        if key in params:
            return float(params[key])
    return None


def evaluate_goal_score(
    env: "ManagerBasedRLEnv",
    params: dict[str, Any],
    std: float | None = None,
) -> torch.Tensor:
    """Score one goal continuously, in ``[0, 1]``.

    Args:
        env: The environment.
        params: Parameters produced by
            :meth:`openarm.arena_tasks.staged.goals.Goal.params`.
        std: Decay scale beyond the goal's tolerance. ``None`` keeps each
            kernel's default, which for a toleranced goal is the tolerance
            itself.

    Returns:
        Tensor of shape ``(N,)``.

    Raises:
        KeyError: If the goal ``kind`` is missing or unknown.
    """
    kind = params.get("kind")
    if kind is None:
        raise KeyError(f"Goal parameters are missing the 'kind' key: {sorted(params)}")
    handler = _DENSE_DISPATCH.get(kind)
    if handler is None:
        raise KeyError(f"Unknown goal kind '{kind}'. Known kinds: {sorted(_DENSE_DISPATCH)}")
    return handler(env, params, default_scale(params) if std is None else float(std))


# ----------------------------------------------------------------------
# Goal aggregation
# ----------------------------------------------------------------------


def goal_scores(env: "ManagerBasedRLEnv", goals: list[Any], std: float | None = None) -> torch.Tensor:
    """Score every goal and stack the results.

    Args:
        env: The environment.
        goals: Parameter dictionaries produced by
            :meth:`openarm.arena_tasks.staged.goals.Goal.params`.
        std: Decay scale applied to every goal, overriding their tolerances.

    Returns:
        Tensor of shape ``(N, len(goals))``.

    Raises:
        ValueError: If there are no goals, which would make any aggregation
            below vacuous.
    """
    if not goals:
        raise ValueError("goal_scores() requires at least one goal.")
    return torch.stack([evaluate_goal_score(env, params, std) for params in goals], dim=1)


def aggregate_goal_scores(
    env: "ManagerBasedRLEnv",
    goals: list[Any],
    aggregation: str = "mean",
    std: float | None = None,
) -> torch.Tensor:
    """Combine several goals into one score.

    Args:
        env: The environment.
        goals: Parameter dictionaries produced by
            :meth:`openarm.arena_tasks.staged.goals.Goal.params`.
        aggregation: One of :data:`AGGREGATIONS`.
        std: Decay scale applied to every goal, overriding their tolerances.

    Returns:
        Tensor of shape ``(N,)`` in ``[0, 1]``.

    Raises:
        ValueError: If the aggregation is unknown, so a typo cannot silently
            degrade to a plain sum.
    """
    scores = goal_scores(env, goals, std)
    if aggregation == "mean":
        return scores.mean(dim=1)
    if aggregation == "min":
        return scores.amin(dim=1)
    if aggregation == "product":
        return scores.prod(dim=1)
    raise ValueError(f"Unknown aggregation '{aggregation}'. Known: {list(AGGREGATIONS)}")


def at_least_score(
    env: "ManagerBasedRLEnv",
    goals: list[Any],
    count: int,
    std: float | None = None,
) -> torch.Tensor:
    """Dense counterpart of the ``at_least`` goal: the mean of the best ``count``.

    "At least three of six cans placed" is a k-of-n condition, and the honest
    continuous version of it is the mean of the ``count`` best-scoring
    candidates. That is one when the best ``count`` are all satisfied, zero
    when none is, and -- the point -- a gradient that says which extra can is
    worth chasing, which a boolean count does not.

    Args:
        env: The environment.
        goals: Parameter dictionaries produced by
            :meth:`openarm.arena_tasks.staged.goals.Goal.params`.
        count: Number of candidates that must hold.
        std: Decay scale applied to every goal, overriding their tolerances.

    Returns:
        Tensor of shape ``(N,)`` in ``[0, 1]``.

    Raises:
        ValueError: If there are no goals or ``count`` exceeds their number.
    """
    if not goals:
        raise ValueError("at_least_score() requires at least one goal.")
    if count > len(goals):
        raise ValueError(f"at_least_score() count={count} exceeds the number of goals ({len(goals)}).")
    scores = goal_scores(env, goals, std)
    best = torch.topk(scores, int(count), dim=1).values
    return best.mean(dim=1)


def goal_progress(
    env: "ManagerBasedRLEnv",
    goals: list[Any],
    aggregation: str = "mean",
    std: float | None = None,
) -> torch.Tensor:
    """The main shaping term: continuous progress towards a task's full success.

    Args:
        env: The environment.
        goals: Every goal parameter dictionary of the task.
        aggregation: One of :data:`AGGREGATIONS`. ``mean`` for overall progress,
            ``min`` to put the weight on the single worst-unsatisfied part,
            ``product`` to approximate "all at once".
        std: Decay scale applied to every goal, overriding their tolerances.

    Returns:
        Tensor of shape ``(N,)`` in ``[0, 1]``.
    """
    return aggregate_goal_scores(env, goals, aggregation, std)


def goal_fraction(env: "ManagerBasedRLEnv", goals: list[Any]) -> torch.Tensor:
    """Fraction of a task's goals that currently hold, from the boolean predicates.

    Complements :func:`goal_progress`. Shaping alone is unboundedly gameable by
    creeping toward every target at once; this is the discrete quantity the
    episode is actually scored on, so a policy that can be trained to increase
    it is learning the real objective.

    Args:
        env: The environment.
        goals: Every goal parameter dictionary of the task.

    Returns:
        Tensor of shape ``(N,)`` in ``[0, 1]``.
    """
    if not goals:
        raise ValueError("goal_fraction() requires at least one goal.")
    satisfied = torch.stack([evaluate_goal(env, params) for params in goals], dim=0)
    return satisfied.to(torch.float32).mean(dim=0)


def success_bonus(env: "ManagerBasedRLEnv", goals: list[Any]) -> torch.Tensor:
    """Sparse bonus for satisfying every goal at once.

    Args:
        env: The environment.
        goals: Every goal parameter dictionary of the task.

    Returns:
        Tensor of shape ``(N,)``, one where all goals hold and zero elsewhere.
    """
    return goal_fraction(env, goals).round()


# ----------------------------------------------------------------------
# Manipulation shaping
# ----------------------------------------------------------------------


def ee_object_distance(
    env: "ManagerBasedRLEnv",
    arm: str,
    object_cfg: str,
    std: float = 0.08,
    approach_axis: int | None = None,
    approach_distance: float = 0.0,
) -> torch.Tensor:
    """Reward an end effector for closing in on an object.

    The single most useful shaping term for a pick stage, because it supplies a
    gradient over the whole approach instead of only at contact.

    Args:
        env: The environment.
        arm: Either ``"left"`` or ``"right"``.
        object_cfg: Scene name of the object.
        std: Distance at which the score is roughly halved [m].
        approach_axis: Optional world axis to offset the target along, to aim
            at a pre-grasp standoff rather than the object itself. ``0``, ``1``
            and ``2`` select x, y and z.
        approach_distance: How far along that axis to offset [m]. A negative
            value aims below the object, which is what a top-down grasp wants.

    Returns:
        Tensor of shape ``(N,)`` in ``(0, 1]``.
    """
    if arm == "closest":
        left_distance_reward = ee_object_distance(env=env, arm="left", object_cfg=object_cfg, std=std, approach_axis=approach_axis, approach_distance=approach_distance)
        right_distance_reward = ee_object_distance(env=env, arm="right", object_cfg=object_cfg, std=std, approach_axis=approach_axis, approach_distance=approach_distance)
        return torch.where(left_distance_reward > right_distance_reward, left_distance_reward, right_distance_reward)
    position = object_position(env, object_cfg)
    if approach_axis is not None:
        if approach_axis not in (0, 1, 2):
            raise ValueError(f"approach_axis must be 0, 1 or 2, got {approach_axis}.")
        position = position.clone()
        position[:, int(approach_axis)] += float(approach_distance)
    rew = tanh_score(torch.norm(ee_position(env, arm) - position, dim=1), std)
    #rew = -torch.norm(ee_position(env, arm) - position, dim=1)
    #print(rew)
    return rew


def ee_approach_alignment(
    env: "ManagerBasedRLEnv",
    arm: str,
    direction: tuple[float, float, float] = TOP_DOWN_DIRECTION,
    std: float = 0.6,
) -> torch.Tensor:
    """Reward an end effector for pointing its approach axis along a direction.

    Position alone will not teach a usable grasp: a gripper that arrives at the
    right spot from the side closes on nothing. This is the term that shapes
    the wrist.

    Args:
        env: The environment.
        arm: Either ``"left"`` or ``"right"``.
        direction: World-frame unit vector the end effector's +Z should align
            with. Defaults to straight down, for a top-down grasp.
        std: Angular error at which the score is roughly halved [rad].

    Returns:
        Tensor of shape ``(N,)`` in ``(0, 1]``.
    """
    quat = ee_orientation(env, arm)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    body_z = torch.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], dim=1)
    target = _as_vector(direction, body_z)
    target = target / target.norm(dim=1, keepdim=True).clamp_min(1e-6)
    alignment = (body_z * target).sum(dim=1).clamp(-1.0, 1.0)
    return tanh_score(torch.acos(alignment), std)


def grasp_attempt(
    env: "ManagerBasedRLEnv",
    arm: str,
    object_cfg: str,
    distance_std: float = 0.05,
    closing_std: float = 0.02,
) -> torch.Tensor:
    """Reward closing the gripper on an object that is actually within reach.

    The product is the point: closing on empty air is worth nothing, and being
    in position with the gripper still wide open is worth nothing either, so
    the term only pays when both halves of a grasp are true.

    Args:
        env: The environment.
        arm: Either ``"left"`` or ``"right"``.
        object_cfg: Scene name of the object.
        distance_std: Distance at which the proximity half is roughly halved [m].
        closing_std: Finger joint position at which the closing half is roughly
            halved [m]. The OpenArm finger is fully open at about 0.043 m.

    Returns:
        Tensor of shape ``(N,)`` in ``[0, 1]``.
    """
    if arm == "closest":
        left_distance_reward = ee_object_distance(env=env, arm="left", object_cfg=object_cfg)
        right_distance_reward = ee_object_distance(env=env, arm="right", object_cfg=object_cfg)
        left_grasp_reward = grasp_attempt(env=env, arm="left", object_cfg=object_cfg, distance_std=distance_std, closing_std=closing_std)
        right_grasp_reward = grasp_attempt(env=env, arm="right", object_cfg=object_cfg, distance_std=distance_std, closing_std=closing_std)
        return torch.where(left_distance_reward > right_distance_reward, left_grasp_reward, right_grasp_reward)
    separation = torch.norm(ee_position(env, arm) - object_position(env, object_cfg), dim=1)
    proximity = tanh_score(separation, distance_std)
    closing = tanh_score(gripper_opening(env, arm), closing_std)
    return proximity * closing


def object_lifted(
    env: "ManagerBasedRLEnv",
    object_cfg: str,
    base_height: float,
    std: float = 0.1,
) -> torch.Tensor:
    """Reward an object for being lifted clear of the height it started at.

    A ramp rather than a ``tanh`` score, because unlike a distance-to-target
    there is nothing to be gained past a comfortable lift, and a saturating
    kernel would keep paying for a wrist lift that costs the policy stability.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        base_height: Height the object rests at, in the environment frame [m].
        std: Lift over which the score reaches one [m].

    Returns:
        Tensor of shape ``(N,)`` in ``[0, 1]``.
    """
    height = object_position(env, object_cfg)[:, 2]
    return ramp_score(height, base_height, base_height + float(std))


def object_to_target(
    env: "ManagerBasedRLEnv",
    object_cfg: str,
    target: tuple[float, float, float],
    std: float = 0.1,
) -> torch.Tensor:
    """Reward an object for being carried toward a world-frame point.

    Args:
        env: The environment.
        object_cfg: Scene name of the object.
        target: Target point in the environment frame [m].
        std: Distance at which the score is roughly halved [m].

    Returns:
        Tensor of shape ``(N,)`` in ``(0, 1]``.
    """
    position = object_position(env, object_cfg)
    return tanh_score(torch.norm(position - _as_vector(target, position), dim=1), std)


def ee_to_anchor(
    env: "ManagerBasedRLEnv",
    arm: str,
    anchor: tuple[float, float, float],
    std: float = 0.08,
) -> torch.Tensor:
    """Reward an end effector for reaching a named world-frame anchor.

    The shaping term for the ``MOVE`` and ``DOCK`` stages, which declare an
    anchor and no goal of their own.

    Args:
        env: The environment.
        arm: Either ``"left"`` or ``"right"``.
        anchor: Target point in the environment frame [m].
        std: Distance at which the score is roughly halved [m].

    Returns:
        Tensor of shape ``(N,)`` in ``(0, 1]``.
    """
    position = ee_position(env, arm)
    return tanh_score(torch.norm(position - _as_vector(anchor, position), dim=1), std)


# ----------------------------------------------------------------------
# Penalties
# ----------------------------------------------------------------------


def object_motion_l2(
    env: "ManagerBasedRLEnv",
    object_names: list[str],
    lin_scale: float = 1.0,
    ang_scale: float = 0.5,
) -> torch.Tensor:
    """Squared-motion cost for a set of objects, summing over the set.

    Point the term at every scene object the task does *not* manipulate and it
    becomes the "leave the distractors alone" penalty, which matters for
    industrial tasks whose scenes are mostly props. Because it is a function of
    the current velocities only, it needs no memory of the reset state and so
    cannot desynchronize from an episode boundary the way a displacement term
    would.

    Args:
        env: The environment.
        object_names: Scene names of the objects to police.
        lin_scale: Linear speed that costs one unit [m/s].
        ang_scale: Angular speed that costs one unit [rad/s].

    Returns:
        Tensor of shape ``(N,)``, non-negative.

    Raises:
        ValueError: If ``lin_scale`` or ``ang_scale`` is not positive.
    """
    if float(lin_scale) <= 0.0 or float(ang_scale) <= 0.0:
        raise ValueError("object_motion_l2() needs positive lin_scale and ang_scale.")
    total = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.float32)
    for name in object_names:
        total = total + (object_speed(env, name) / float(lin_scale)) ** 2
        total = total + (object_angular_speed(env, name) / float(ang_scale)) ** 2
    return total


def ee_speed_l2(env: "ManagerBasedRLEnv", arm: str, scale: float = 1.0) -> torch.Tensor:
    """Squared end-effector speed, as a smoothness cost.

    Read from the robot's body-link velocities rather than differencing the
    frame transformer's positions: the transformer publishes pose only, so a
    differencing term would have to cache the previous step and would be the
    one term in this module able to desynchronize from an episode boundary.

    Args:
        env: The environment.
        arm: Either ``"left"`` or ``"right"``.
        scale: Speed that costs one unit [m/s].

    Returns:
        Tensor of shape ``(N,)``, non-negative.

    Raises:
        KeyError: If the robot has no body named after the arm's end effector.
    """
    if float(scale) <= 0.0:
        raise ValueError("ee_speed_l2() needs a positive scale.")
    robot = env.unwrapped.scene["robot"]
    body_name = f"openarm_{arm}_ee_tcp"
    if body_name not in list(robot.data.body_names):
        raise KeyError(f"Body '{body_name}' not found on 'robot'. Available: {list(robot.data.body_names)}")
    index = list(robot.data.body_names).index(body_name)
    velocity = to_torch(robot.data.body_link_lin_vel_w)[:, index, :]
    return (torch.norm(velocity, dim=1) / float(scale)) ** 2


def object_below_height_l2(
    env: "ManagerBasedRLEnv",
    object_names: list[str],
    minimum_height: float,
) -> torch.Tensor:
    """Squared depth below a height, as a smooth drop penalty.

    The boolean ``dropped`` termination ends the episode, which the policy
    learns about only through the value of that terminal. This gives it a
    gradient on the way down instead, and charges nothing until an object is
    actually below the line.

    Args:
        env: The environment.
        object_names: Scene names of the tracked objects.
        minimum_height: Height the objects must stay above [m].

    Returns:
        Tensor of shape ``(N,)``, non-negative, summed over the objects.
    """
    depth = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.float32)
    for name in object_names:
        depth = depth + (float(minimum_height) - object_position(env, name)[:, 2]).clamp_min(0.0) ** 2
    return depth
