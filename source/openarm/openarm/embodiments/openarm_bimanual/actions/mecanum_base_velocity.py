import numpy as np
import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.assets import Articulation
from isaacsim.robot.wheeled_robots.controllers.holonomic_controller import HolonomicController
from isaaclab.utils import configclass


class MecanumBaseVelocityAction(ActionTerm):
    """Action term that sets base velocity and synchronously solves wheel velocities."""

    cfg: "MecanumBaseVelocityActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "MecanumBaseVelocityActionCfg", env):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]

        # Cache wheel joint indices
        self._wheel_joint_ids, _ = self._asset.find_joints(cfg.wheel_joint_names)
        self._num_wheels = len(self._wheel_joint_ids)

        # Build Isaac Sim's built-in holonomic controller (precomputes kinematic matrix)
        self._holonomic = HolonomicController(
            "mecanum_base_controller",
            wheel_radius=np.array([cfg.wheel_radius] * self._num_wheels),
            wheel_positions=np.array(cfg.wheel_positions),          # shape (4, 3)
            wheel_orientations=np.array(cfg.wheel_orientations),    # shape (4, 3)
            mecanum_angles=np.array(cfg.mecanum_angles),            # e.g. [pi/4, -pi/4, pi/4, -pi/4]
            wheel_axis=np.array(cfg.wheel_axis),
            up_axis=np.array(cfg.up_axis),
        )

        # Raw actions: [vx, vy, wz]
        self._raw_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Store raw actions and apply scale."""
        self._raw_actions[:] = actions
        self._processed_actions[:] = actions * self.cfg.scale

    def apply_actions(self):
        """Write base velocity and wheel velocities to sim."""
        # Step 1: write root velocity
        root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        root_vel[:, 0] = self._processed_actions[:, 0]  # vx
        root_vel[:, 1] = self._processed_actions[:, 1]  # vy
        root_vel[:, 5] = self._processed_actions[:, 2]  # wz
        self._asset.write_root_velocity_to_sim(root_vel)

        # Step 2: solve wheel angular velocities via holonomic controller
        wheel_vel = torch.zeros(
            (self.num_envs, self._num_wheels), device=self.device
        )
        for i in range(self.num_envs):
            cmd = self._processed_actions[i].cpu().numpy()
            joint_vel = self._holonomic.forward(cmd).joint_velocities  # np.ndarray (4,)
            print(joint_vel)
            wheel_vel[i] = torch.tensor(
                joint_vel, device=self.device, dtype=torch.float32
            )

        # Step 3: write wheel velocities
        self._asset.write_joint_velocity_to_sim(
            wheel_vel, joint_ids=self._wheel_joint_ids
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


@configclass
class MecanumBaseVelocityActionCfg(ActionTermCfg):
    """Configuration for the mecanum base velocity action term."""

    class_type: type = MecanumBaseVelocityAction

    asset_name: str = "robot"
    wheel_joint_names: list[str] = None
    wheel_radius: float = 0.05
    wheel_positions: list[list[float]] = None   # 4x3, relative to base
    wheel_orientations: list[list[float]] = None
    mecanum_angles: list[float] = None          # roller angles, e.g. [pi/4, -pi/4, ...]
    wheel_axis: list[float] = [0.0, 1.0, 0.0]
    up_axis: list[float] = [0.0, 0.0, 1.0]
    scale: float = 1.0