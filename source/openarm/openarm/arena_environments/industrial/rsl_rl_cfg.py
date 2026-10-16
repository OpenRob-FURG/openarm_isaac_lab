"""RSL-RL runner configuration for the 50 staged industrial tasks.

The one thing this file decides is the shape of the actor-critic split. Every
other value follows from it, so the reasoning is worth stating.

A staged industrial task is scored on where its objects are. The OpenArm
bimanual embodiment's ``policy`` observation group is proprioception only --
joint states, end effectors, grippers, the robot's own links -- so an actor on
that group never sees whether the can is already in the bin. A symmetric
critic would be blind in exactly the same way, and its value estimates would
then be conditioned on a state that does not determine the reward, which shows
up as a value function that is confidently wrong about how close the task is
to done.

So the critic is given the privileged group that
:class:`~openarm.arena_tasks.staged.task.StagedTask` builds: full object poses,
velocities and heights, world-frame end effectors, applied torques, and each
goal object's displacement from its own target. The actor keeps the
proprioception it could actually obtain on hardware.

Two consequences follow from ``task_obs`` being left out of both groups. The
``task_obs`` group carries the 640x480 camera, and a stacked CNN over tens of
thousands of environments is not what this task is bottlenecked on; dropping it
keeps a training run inside a single GPU's memory. And it is the arrangement
that lets the same environment be trained from state and then evaluated through
the scripted oracle against pixels, since the observation groups are additive
rather than exclusive.
"""

from __future__ import annotations

from dataclasses import field, dataclass

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

#: Observation groups the actor and critic each read. ``policy`` is the
#: embodiment's proprioception; ``critic`` is the task's privileged state.
#: ``task_obs`` is deliberately absent from both.
#:
#: The split has to be spelled out rather than inherited, because rsl-rl does not
#: derive it: an ``obs_groups`` entry listing ``critic`` on the actor would hand
#: the policy privileged state it could not observe on hardware, and the run
#: would not transfer.
OBS_GROUPS: dict[str, list[str]] = {
    "actor": ["policy"],
    "critic": ["policy", "critic"],
}



@configclass
class IndustrialStagedRslRlRunnerCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL PPO runner for a staged industrial task, trained on privileged state.

    Attributes:
        num_steps_per_env: Rollout length per environment. 32 rather than
            Arena's 24, because these episodes are long: at the default 0.05 s
            control step a 60 s task is 1200 steps, and a 24-step rollout
            covers only 2% of it.
        max_iterations: Optimization steps. Industrial tasks with several
            sequential stages need a long horizon, and the discount of 0.98
            makes early progress on a long task worth very little, so this is
            set well above the reach-task default.
        experiment_name: Log directory root, shared by all 50 tasks so that a
            sweep over tasks stays in one place.
        obs_groups: The actor-critic split described in this module's
            docstring.
        policy: Actor and critic networks, in rsl-rl's deprecated shape. Both
            normalize their observations, which matters more than usual here: the
            privileged group stacks positions in metres, torques in
            newton-metres and normalized joint coordinates in [0, 1] into one
            vector, and the critic would otherwise see a few channels dominate
            the rest by orders of magnitude.

            ``actor``/``critic`` are deliberately not spelled out. rsl-rl 3.x
            reads ``policy`` and ignores them, rsl-rl 4.x and later read them and
            ignore ``policy``, and
            ``handle_deprecated_rsl_rl_cfg`` derives one from the other. Setting
            both would let the installed version win over the one field that is
            portable, and the explicit actor would have to repeat the
            ``stochastic=True`` that PPO needs for exploration.
        algorithm: PPO settings. The entropy coefficient is low because the
            shaped reward is informative from the first step and there is no
            exploration problem worth solving; the learning rate is on the
            adaptive schedule so it decays on its own as the KL budget is hit.
    """

    num_steps_per_env: int = 32
    max_iterations: int = 8000
    save_interval: int = 200
    experiment_name: str = "openarm_bimanual_industrial"
    run_name: str = ""
    resume: bool = False
    obs_groups: dict[str, list[str]] = field(default_factory=lambda: dict(OBS_GROUPS))
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=0.8,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
