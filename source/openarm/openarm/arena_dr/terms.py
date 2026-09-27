"""Observation terms for OpenArm bimanual tasks.

The OpenArm bimanual embodiment ships a top-down RGB camera but does not
include it in the policy observation group, so these terms are what put pixels
into the observation. They also carry the per-environment photometric
randomization used for sim-to-real, implemented as a pure function of the
image and the parameters sampled on reset (see
:func:`openarm.arena_dr.events.sample_photometric_params`).
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.array import convert_to_torch

from openarm.arena_dr.events import get_photometric_params


def rgb_image(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, data_type: str = "rgb") -> torch.Tensor:
    """Read a camera image batch as float RGB in [0, 1].

    Isaac Lab camera data is returned as ``(N, H, W, C)`` in [0, 1] for
    ``rgb``/``depth`` outputs, with an alpha channel for ``rgb``. This term
    drops the alpha channel so downstream consumers always see three channels.

    Args:
        env: The environment.
        sensor_cfg: Scene entity selector naming the camera.
        data_type: Camera output buffer to read.

    Returns:
        Tensor of shape ``(N, H, W, 3)`` with values in [0, 1].

    Raises:
        KeyError: If the camera does not expose the requested data type.
    """
    sensor = env.unwrapped.scene[sensor_cfg.name]
    if data_type not in sensor.data.output:
        available = sorted(sensor.data.output.keys())
        raise KeyError(f"Camera '{sensor_cfg.name}' has no '{data_type}' output. Available: {available}")
    image = sensor.data.output[data_type]
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def augmented_rgb(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, data_type: str = "rgb") -> torch.Tensor:
    """Camera RGB with per-environment photometric randomization applied.

    The augmentation models the appearance gap between simulation and a real
    onboard camera: exposure and white-balance error (brightness, colour cast),
    lens and sensor response variation (contrast, gamma), and sensor noise.

    Because the dome light in this arena setup is a single global prim, the
    only way to obtain *per-environment* visual diversity is to augment the
    image in observation space. Doing it here, as a pure function of the current
    image and the parameters for this episode, also avoids the compounding
    artefact that in-place buffer mutation would introduce across resets.

    Args:
        env: The environment.
        sensor_cfg: Scene entity selector naming the camera.
        data_type: Camera output buffer to read.

    Returns:
        Tensor of shape ``(N, H, W, 3)`` with values clamped to [0, 1].
    """
    image = rgb_image(env, sensor_cfg, data_type)
    params, color_cast = get_photometric_params(env)
    if params is None:
        return image

    brightness = params[:, 0].view(-1, 1, 1, 1)
    contrast = params[:, 1].view(-1, 1, 1, 1)
    gamma = params[:, 2].view(-1, 1, 1, 1)
    noise_std = params[:, 4].view(-1, 1, 1, 1)
    cast = color_cast.view(-1, 1, 1, 3)

    out = image * brightness * cast
    # Contrast around mid-grey, which keeps the mean roughly fixed when the
    # gain is near one.
    out = (out - 0.5) * contrast + 0.5
    out = out.clamp_min(1e-4).pow(gamma)
    if float(noise_std.max()) > 0.0:
        out = out + torch.randn_like(out) * noise_std
    return out.clamp(0.0, 1.0)


def object_pose(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Object root pose in the environment frame, as ``(x, y, z, qw, qx, qy, qz)``.

    Args:
        env: The environment.
        asset_cfg: Scene entity selector naming the object.

    Returns:
        Tensor of shape ``(N, 7)``.
    """
    asset = env.unwrapped.scene[asset_cfg.name]
    if hasattr(asset.data, "root_pose_w"):
        raw = asset.data.root_pose_w
    else:
        raw = torch.cat(asset.get_world_poses(), dim=-1)
    # The physics backend does not guarantee torch: isaaclab_physx hands back
    # warp arrays and the legacy backend hands back torch. Normalise here so the
    # term is agnostic to which one is active.
    pose = convert_to_torch(raw, dtype=torch.float32, device=env.unwrapped.device).clone()
    pose[:, :3] -= convert_to_torch(
        env.unwrapped.scene.env_origins, dtype=torch.float32, device=env.unwrapped.device
    )
    return pose


def ee_pose(env: ManagerBasedEnv, arm: str) -> torch.Tensor:
    """End-effector pose in the robot base frame, as ``(x, y, z, qw, qx, qy, qz)``.

    Args:
        env: The environment.
        arm: Either ``"left"`` or ``"right"``.

    Returns:
        Tensor of shape ``(N, 7)``.
    """
    pose = env.get_robot_eef_pose(eef_name=arm)
    quat = _matrix_to_quat_wxyz(pose[..., :3, :3])
    pos = pose[..., :3, 3]
    return torch.cat([pos, quat], dim=-1)


def _matrix_to_quat_wxyz(rotation: torch.Tensor) -> torch.Tensor:
    """Convert a batch of rotation matrices to ``(w, x, y, z)`` quaternions.

    Uses the branch-free formulation from :func:`torchvision`-style matrix
    conversion but written out directly to avoid an extra dependency.

    Args:
        rotation: Tensor of shape ``(..., 3, 3)``.

    Returns:
        Tensor of shape ``(..., 4)`` in ``(w, x, y, z)`` order.
    """
    m = rotation
    m00, m01, m02 = m[..., 0, 0], m[..., 0, 1], m[..., 0, 2]
    m10, m11, m12 = m[..., 1, 0], m[..., 1, 1], m[..., 1, 2]
    m20, m21, m22 = m[..., 2, 0], m[..., 2, 1], m[..., 2, 2]

    trace = m00 + m11 + m22
    quat = torch.zeros(m.shape[:-2] + (4,), device=m.device, dtype=m.dtype)

    # Pick the branch with the largest denominator for numerical stability.
    positive = trace > 0.0
    if positive.any():
        s = torch.sqrt((trace[positive] + 1.0).clamp_min(1e-12)) * 2.0
        quat[positive, 0] = 0.25 * s
        quat[positive, 1] = (m21[positive] - m12[positive]) / s
        quat[positive, 2] = (m02[positive] - m20[positive]) / s
        quat[positive, 3] = (m10[positive] - m01[positive]) / s

    neg_x = (~positive) & (m00 > m11) & (m00 > m22)
    if neg_x.any():
        s = torch.sqrt((1.0 + m00[neg_x] - m11[neg_x] - m22[neg_x]).clamp_min(1e-12)) * 2.0
        quat[neg_x, 0] = (m21[neg_x] - m12[neg_x]) / s
        quat[neg_x, 1] = 0.25 * s
        quat[neg_x, 2] = (m01[neg_x] + m10[neg_x]) / s
        quat[neg_x, 3] = (m02[neg_x] + m20[neg_x]) / s

    neg_y = (~positive) & (~neg_x) & (m11 > m22)
    if neg_y.any():
        s = torch.sqrt((1.0 + m11[neg_y] - m00[neg_y] - m22[neg_y]).clamp_min(1e-12)) * 2.0
        quat[neg_y, 0] = (m02[neg_y] - m20[neg_y]) / s
        quat[neg_y, 1] = (m01[neg_y] + m10[neg_y]) / s
        quat[neg_y, 2] = 0.25 * s
        quat[neg_y, 3] = (m12[neg_y] + m21[neg_y]) / s

    neg_z = (~positive) & (~neg_x) & (~neg_y)
    if neg_z.any():
        s = torch.sqrt((1.0 + m22[neg_z] - m00[neg_z] - m11[neg_z]).clamp_min(1e-12)) * 2.0
        quat[neg_z, 0] = (m10[neg_z] - m01[neg_z]) / s
        quat[neg_z, 1] = (m02[neg_z] + m20[neg_z]) / s
        quat[neg_z, 2] = (m12[neg_z] + m21[neg_z]) / s
        quat[neg_z, 3] = 0.25 * s

    return quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
