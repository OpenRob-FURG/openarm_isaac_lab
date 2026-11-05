# open_rob_interface.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

import isaaclab.utils.math as math_utils
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

import heapq
import numpy as np
from scipy.ndimage import binary_dilation

# Add to imports section
from typing import Tuple, List, Optional

# Add to imports
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from lerobot.datasets import LeRobotDataset
    _LEROBOT_AVAILABLE = True
except ImportError:
    _LEROBOT_AVAILABLE = False
    print("[OpenRobInterface] lerobot not installed; dataset recording disabled")


# ---------- constants for the action layout ----------
LEFT_POS = slice(0, 3)
LEFT_QUAT = slice(3, 7)
LEFT_GRIP = 7
RIGHT_POS = slice(8, 11)
RIGHT_QUAT = slice(11, 15)
RIGHT_GRIP = 15
BASE_CMD = slice(16, 19)
ACTION_DIM = 19


@dataclass
class ArmTarget:
    """Absolute Cartesian target for one arm."""
    pos: torch.Tensor   # (3,) xyz in world frame
    quat: torch.Tensor  # (4,) wxyz in world frame
    grip: float = 0.0   # 0.0 = open, 1.0 = close


class OpenRobInterface:
    """
    High-level skill interface for a bimanual mobile manipulator
    (mecanum base + two 7-DoF arms with parallel grippers) in IsaacLab Arena.

    Action layout (19-dim):
        [0:3]   left  arm xyz        (absolute, world frame)
        [3:7]   left  arm quat wxyz  (absolute, world frame)
        [7]     left  gripper        (0=open, 1=close)
        [8:11]  right arm xyz
        [11:15] right arm quat wxyz
        [15]    right gripper
        [16:19] base (vx, vy, wz)
    """

    # arm joint names assumed; adjust to your URDF/USD
    LEFT_ARM_JOINTS = [
        "openarm_left_joint1", "openarm_left_joint2", "openarm_left_joint3",
        "openarm_left_joint4", "openarm_left_joint5", "openarm_left_joint6", "openarm_left_joint7"
    ]
    RIGHT_ARM_JOINTS = [
        "openarm_right_joint1", "openarm_right_joint2", "openarm_right_joint3",
        "openarm_right_joint4", "openarm_right_joint5", "openarm_right_joint6", "openarm_right_joint7"
    ]
    LEFT_EE_BODY = "openarm_left_ee_tcp"
    RIGHT_EE_BODY = "openarm_right_ee_tcp"

    def __init__(self, env, device: str = "cuda:0"):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs
        self.robot = env.scene["robot"]
        self.dt = env.physics_dt * env.cfg.decimation

        # Base
        self.base_body = "base_link"
        self.max_base_v = 0.4      # m/s
        self.max_base_w = 1.0      # rad/s

        # Arms
        self.left_joint_ids, _ = self.robot.find_joints(self.LEFT_ARM_JOINTS)
        self.right_joint_ids, _ = self.robot.find_joints(self.RIGHT_ARM_JOINTS)
        self.left_ee_id = self.robot.find_bodies(self.LEFT_EE_BODY)[0][0]
        self.right_ee_id = self.robot.find_bodies(self.RIGHT_EE_BODY)[0][0]

        # Gripper joint names (two fingers each)
        self.left_grip_ids, _ = self.robot.find_joints(
            ["openarm_left_finger_joint.*"])
        self.right_grip_ids, _ = self.robot.find_joints(
            ["openarm_right_finger_joint.*"])
        # open / close positions – customize to your gripper travel
        self._grip_open = 0.04
        self._grip_close = 0.0

        # Differential IK for both arms
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
        )
        self.left_ik = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=device)
        self.right_ik = DifferentialIKController(
            ik_cfg, num_envs=self.num_envs, device=device)
        self.left_ik.set_command(torch.zeros((self.num_envs, 7), device=device))
        self.right_ik.set_command(torch.zeros((self.num_envs, 7), device=device))

        # default neutral targets (filled after first step)
        self._left_target = ArmTarget(
            pos=torch.tensor([0.35, 0.25, 0.9], device=device),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        )
        self._right_target = ArmTarget(
            pos=torch.tensor([0.35, -0.25, 0.9], device=device),
            quat=torch.tensor([1.0, 0.0, 0.0, 0.0], device=device),
        )

        # ---- cameras ----
        self.camera_names = ["top_camera", "left_wrist_camera", "right_wrist_camera"]
        self._lerobot_image_keys = {
            "top_camera":        "observation.images.top",
            "left_wrist_camera": "observation.images.left_wrist",
            "right_wrist_camera": "observation.images.right_wrist",
        }
        # Populated in start_recording() from the live sensor configs.
        self._camera_hw: dict[str, tuple[int, int]] = {}

        # ---- episode buffer for LeRobot recording ----
        self._episode_buffer: List[Dict[str, Any]] = []
        self._recording = False
        self._lerobot_dataset = None
        self._lerobot_fps = int(round(1.0 / self.dt))
        self._episode_idx = 0
        self._last_task = "unknown"

    def _policy_observation_terms(self) -> Dict[str, np.ndarray]:
        """
        Evaluate the same terms the policy's ObservationGroup sees, env 0 only.
        Returns a dict keyed by the term name in PolicyCfg.
        """
        from .mdp import rel_ee_pos, rel_ee_quat, gripper_opening
        # NOTE: adjust this import if OpenRobInterface lives in a different
        # package than the mdp module.  The terms only need `env` and `arm`.

        def _np(t: torch.Tensor) -> np.ndarray:
            return t[0].detach().cpu().numpy().astype(np.float32)

        return {
            "left_eef_pos":     _np(rel_ee_pos(self.env, arm="left")),
            "left_eef_quat":    _np(rel_ee_quat(self.env, arm="left")),
            "left_gripper_pos": _np(gripper_opening(self.env, arm="left")),
            "right_eef_pos":    _np(rel_ee_pos(self.env, arm="right")),
            "right_eef_quat":   _np(rel_ee_quat(self.env, arm="right")),
            "right_gripper_pos":_np(gripper_opening(self.env, arm="right")),
        }
    
    def _read_cameras(self) -> Dict[str, np.ndarray]:
        """Return RGB uint8 HWC frames for each configured camera, env 0 only."""
        frames: Dict[str, np.ndarray] = {}
        for name in self.camera_names:
            sensor = self.env.scene.sensors[name]
            # Shape for TiledCameraCfg: (num_envs, H, W, 4) RGBA uint8
            # Shape for CameraCfg:      (num_envs, H, W, 3) RGB uint8
            rgb = sensor.data.output["rgb"][0].detach().cpu().numpy()
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]          # drop alpha
            frames[self._lerobot_image_keys[name]] = rgb
        return frames

    def _lerobot_features(self) -> dict:
        """Feature schema for LeRobotDataset. Shapes must match exactly."""
        feats: dict = {
            # ---- observation.state: exactly the PolicyCfg terms ----
            "observation.state": {
                "dtype": "float32",
                "shape": (16,),
                "names": [
                    "left_eef_x", "left_eef_y", "left_eef_z",
                    "left_eef_qw", "left_eef_qx", "left_eef_qy", "left_eef_qz",
                    "left_gripper",
                    "right_eef_x", "right_eef_y", "right_eef_z",
                    "right_eef_qw", "right_eef_qx", "right_eef_qy", "right_eef_qz",
                    "right_gripper",
                ],
            },
            # ---- action: raw 19-dim from the environment ----
            "action": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": [
                    "l_pos_x", "l_pos_y", "l_pos_z",
                    "l_quat_w", "l_quat_x", "l_quat_y", "l_quat_z", "l_grip",
                    "r_pos_x", "r_pos_y", "r_pos_z",
                    "r_quat_w", "r_quat_x", "r_quat_y", "r_quat_z", "r_grip",
                    "base_vx", "base_vy", "base_wz",
                ],
            },
        }
        # ---- observation.images.* ----
        for name in self.camera_names:
            h, w = self._camera_hw[name]
            feats[self._lerobot_image_keys[name]] = {
                "dtype": "video" if self._lerobot_use_videos else "image",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            }
        return feats
    
    def start_recording(
        self,
        repo_id: str,
        root: str = "datasets/arena_recordings",
        fps: Optional[int] = None,
        use_videos: bool = True,
        robot_type: str = "openrob",
    ):
        """
        Initialize a writable LeRobotDataset and enable transition recording.

        Camera resolution is read from the live sensor configs so the schema
        shapes match the arrays produced by `_read_cameras()` exactly.
        """
        if not _LEROBOT_AVAILABLE:
            raise RuntimeError(
                "lerobot is not installed. "
                "Install with: pip install 'lerobot[dataset]>=0.6.1'")

        # Read camera resolutions from the scene configs
        self._camera_hw.clear()
        cam_cfg = self.env.cfg.scene  # InteractiveSceneCfg – but camera_cfg is on the task
        # Fall back to the live sensor's output tensor shape if config is not
        # reachable through cfg.  This is safer across Arena wrapper versions.
        for name in self.camera_names:
            sensor = self.env.scene.sensors[name]
            # A single env's sensor tensor is already built at this point.
            h, w = sensor.image_shape[:2] if hasattr(sensor, "image_shape") \
                else sensor.data.output["rgb"].shape[1:3]
            self._camera_hw[name] = (int(h), int(w))

        self._lerobot_use_videos = use_videos
        self._lerobot_fps = fps or int(round(1.0 / self.dt))

        self._lerobot_dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            fps=self._lerobot_fps,
            features=self._lerobot_features(),
            robot_type=robot_type,
            use_videos=use_videos,
        )
        self._episode_buffer.clear()
        self._episode_idx = 0
        self._recording = True
        print(
            f"[OpenRobInterface] Recording to {root}/{repo_id} "
            f"@ {self._lerobot_fps} fps, "
            f"cameras={self.camera_names}, "
            f"use_videos={use_videos}")

    def store_transition(self, action: torch.Tensor, task: str = "unknown"):
        """
        Snapshot the current MDP transition into the episode buffer.

        Call this *after* env.step(action) so observations reflect the
        post-step state, and pass the same `action` tensor that was sent.
        """
        if not self._recording:
            return

        self._last_task = self._current_task = task
        act_np = action.detach().cpu().numpy().flatten().astype(np.float32)
        if act_np.shape[0] != ACTION_DIM:
            raise ValueError(
                f"Action must be {ACTION_DIM}-dim, got {act_np.shape[0]}")

        # ---- observation.state: matches PolicyCfg exactly ----
        terms = self._policy_observation_terms()
        state = np.concatenate([
            terms["left_eef_pos"],
            terms["left_eef_quat"],
            terms["left_gripper_pos"],
            terms["right_eef_pos"],
            terms["right_eef_quat"],
            terms["right_gripper_pos"],
        ], axis=0).astype(np.float32)
        assert state.shape == (16,), state.shape

        frame: Dict[str, Any] = {
            "observation.state": state,
            "action": act_np,
            "task": task,
        }
        # ---- cameras ----
        frame.update(self._read_cameras())

        self._lerobot_dataset.add_frame(frame)


    def clear_episode_buffer(self):
        """Discard the current episode buffer and LeRobot's pending frames."""
        self._episode_buffer.clear()
        if self._lerobot_dataset is not None:
            self._lerobot_dataset.clear_episode_buffer(delete_images=True)


    def save_episode(self, task: Optional[str] = None):
        """
        Commit the current episode to the LeRobot dataset on disk.

        LeRobot flushes parquet + video, computes stats, updates metadata,
        and resets its internal buffer. Safe to call once per episode.
        """
        if not self._recording or self._lerobot_dataset is None:
            return False

        task_str = task or self._last_task
        self._lerobot_dataset.save_episode(
            episode_index=self._episode_idx,
            task=task_str,
        )
        self._episode_idx += 1
        self._episode_buffer.clear()
        print(f"[OpenRobInterface] Saved episode {self._episode_idx - 1} "
            f"(task='{task_str}')")
        return True


    def stop_recording(self):
        """Finalize the dataset. Call once at the very end of data collection."""
        if self._lerobot_dataset is not None:
            self._lerobot_dataset.finalize()
            self._lerobot_dataset = None
        self._recording = False
        self._episode_buffer.clear()
        print(f"[OpenRobInterface] Dataset finalized with "
            f"{self._episode_idx} episode(s)")

    # ------------------------------------------------------------------
    # Low-level: build and send a 19-dim action
    # ------------------------------------------------------------------
    def _compose_action(
        self,
        left: ArmTarget,
        right: ArmTarget,
        base_cmd: torch.Tensor,           # (3,) vx, vy, wz
    ) -> torch.Tensor:
        act = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        act[:, LEFT_POS] = left.pos
        act[:, LEFT_QUAT] = left.quat
        act[:, LEFT_GRIP] = left.grip
        act[:, RIGHT_POS] = right.pos
        act[:, RIGHT_QUAT] = right.quat
        act[:, RIGHT_GRIP] = right.grip
        act[:, BASE_CMD] = base_cmd
        return act

    def _step(self, action: torch.Tensor):
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action.to(self.device).clamp_(
            min=torch.tensor([-1.0] * ACTION_DIM, device=self.device),
            max=torch.tensor([1.0] * ACTION_DIM, device=self.device),
        )
        print('ACTION:', action)
        self._last_action = action[0].clone()
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self._recording:
            self.store_transition(action[0], task=self._current_task)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # IK helpers
    # ------------------------------------------------------------------
    def _solve_arm(
        self,
        ik: DifferentialIKController,
        joint_ids: Sequence[int],
        ee_id: int,
        target: ArmTarget,
    ) -> torch.Tensor:
        """Solve one arm's joint targets via differential IK."""
        ee_pos_b = torch.as_tensor(self.robot.data.body_pos_w)[:, ee_id].clone()
        ee_quat_b = torch.as_tensor(self.robot.data.body_quat_w)[:, ee_id].clone()

        cmd = torch.cat([target.pos, target.quat], dim=-1).unsqueeze(0)
        cmd = cmd.expand(self.num_envs, -1)
        ik.set_command(command=cmd)

        jac = self.robot.root_physx_view.get_jacobians()[
            :, ee_id, :, list(joint_ids)]
        joint_pos = torch.as_tensor(self.robot.data.joint_pos)[:, list(joint_ids)]
        joint_pos_des = ik.compute(
            ee_pos=ee_pos_b, ee_quat=ee_quat_b,
            jacobian=jac, joint_pos=joint_pos)
        return joint_pos_des

    def _apply_joint_targets(self, left_q, right_q):
        # Write joint targets directly so the action can be tiny / zero.
        jp = torch.as_tensor(self.robot.data.joint_pos).clone()
        jp[:, list(self.left_joint_ids)] = left_q
        jp[:, list(self.right_joint_ids)] = right_q
        self.robot.set_joint_position_target(jp)

    def _apply_grippers(self, left_close: float, right_close: float):
        jp = torch.as_tensor(self.robot.data.joint_pos).clone()
        lv = self._grip_open + left_close * (self._grip_close - self._grip_open)
        rv = self._grip_open + right_close * (self._grip_close - self._grip_open)
        jp[:, self.left_grip_ids] = lv
        jp[:, self.right_grip_ids] = rv
        self.robot.set_joint_position_target(jp)

    # ------------------------------------------------------------------
    # Base helper
    # ------------------------------------------------------------------
    def _base_cmd(self, vx: float, vy: float, wz: float) -> torch.Tensor:
        return torch.tensor(
            [max(-1.0, min(1.0, vx / self.max_base_v)),
             max(-1.0, min(1.0, vy / self.max_base_v)),
             max(-1.0, min(1.0, wz / self.max_base_w))],
            device=self.device,
        )

    def _apply_base_cmd(self, vx: float, vy: float, wz: float):
        # Base velocity command is expected to be one of the action terms.
        # If your env exposes `robot.write_base_velocity` (or a wheel
        # controller term), call that here.  Otherwise the value in the
        # 19-dim action is consumed by the env directly.
        pass

    def _build_occupancy_grid(
        self,
        resolution: float = 0.05,
        z_min: float = 0.05,
        z_max: float = 2.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build a 2D occupancy grid from the Isaac Lab scene.

        Tries the PhysX-based omap generator first.  Falls back to a
        USD bounding-box projection if the extension is unavailable.

        Returns
        -------
        grid : np.ndarray of shape (H, W)  — 1 = obstacle, 0 = free
        origin : np.ndarray of shape (2,)  — world (x, y) of grid[0, 0]
        """
        try:
            return self._build_grid_omap(resolution, z_min, z_max, bounds)
        except ImportError as e:
            raise e
            print("[OpenRobInterface] omap unavailable, using USD projection fallback")
            return self._build_grid_usd(resolution, z_min, z_max, bounds)

    def _build_grid_omap(
        self,
        resolution: float,
        z_min: float,
        z_max: float,
        bounds: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """PhysX-collider-based occupancy grid via isaacsim.asset.gen.omap."""
        from isaacsim.asset.gen.omap.bindings import _omap
        import omni.physx
        import omni.usd

        physx = omni.physx.get_physx_interface()
        stage_id = omni.usd.get_context().get_stage_id()

        generator = _omap.Generator(physx, stage_id)
        generator.update_settings(
            resolution,
            z_min,
            z_max if z_max > 0 else z_min,
            0.5,
        )

        if bounds is None:
            # default 20 m × 20 m around origin
            lo = (-10.0, -10.0, 0.0)
            hi = (10.0, 10.0, 0.0)
        else:
            x0, y0, x1, y1 = bounds
            lo = (x0, y0, 0.0)
            hi = (x1, y1, 0.0)

        generator.set_transform((0, 0, 0), lo, hi)
        generator.generate2d()
        raw = generator.get_buffer()

        # raw is a flat buffer; reshape according to generator metadata
        #info = generator.get_output_info()
        dimensions = generator.get_dimensions() # info.height, info.width
        #print(dimensions)
        H, W, _ = dimensions
        print(raw)
        grid = np.asarray(raw).reshape(H, W).copy()
        # Invert: omap gives 255 = free, 0 = occupied
        grid = (grid < z_max).astype(np.uint8)
        origin = np.array([lo[0], lo[1]], dtype=np.float32)
        return grid, origin

    def _build_grid_usd(
        self,
        resolution: float,
        z_min: float,
        z_max: float,
        bounds: Optional[Tuple[float, float, float, float]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: project USD collision prims onto a 2D grid by AABB."""
        from pxr import Usd, UsdGeom, UsdPhysics
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        # Collect obstacle AABBs
        obstacles = []  # list of (xmin, ymin, xmax, ymax)
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) and not prim.IsA(UsdGeom.Cube):
                continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue

            bbox = bbox_cache.ComputeWorldBound(prim)
            rng = bbox.ComputeAlignedRange()
            if rng.IsEmpty():
                continue
            min_pt = rng.GetMin()
            max_pt = rng.GetMax()

            # height filter — skip floor-level markings
            if max_pt[2] < z_min or min_pt[2] > z_max:
                continue
            obstacles.append((
                min_pt[0], min_pt[1], max_pt[0], max_pt[1]))

        if bounds is None:
            if obstacles:
                xs = [o[0] for o in obstacles] + [o[2] for o in obstacles]
                ys = [o[1] for o in obstacles] + [o[3] for o in obstacles]
                bounds = (min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2)
            else:
                bounds = (-10, -10, 10, 10)

        x0, y0, x1, y1 = bounds
        W = int(np.ceil((x1 - x0) / resolution))
        H = int(np.ceil((y1 - y0) / resolution))
        grid = np.zeros((H, W), dtype=np.uint8)

        for (ox0, oy0, ox1, oy1) in obstacles:
            ix0 = max(0, int((ox0 - x0) / resolution))
            iy0 = max(0, int((oy0 - y0) / resolution))
            ix1 = min(W, int(np.ceil((ox1 - x0) / resolution)))
            iy1 = min(H, int(np.ceil((oy1 - y0) / resolution)))
            grid[iy0:iy1, ix0:ix1] = 1

        origin = np.array([x0, y0], dtype=np.float32)
        return grid, origin

    @staticmethod
    def _inflate_grid(
        grid: np.ndarray,
        robot_radius: float,
        resolution: float,
    ) -> np.ndarray:
        """Dilate obstacles by the robot radius so A* never hugs walls."""
        radius_cells = int(np.ceil(robot_radius / resolution))
        if radius_cells < 1:
            return grid.copy()

        # Circular structuring element
        y, x = np.ogrid[-radius_cells:radius_cells + 1,
                        -radius_cells:radius_cells + 1]
        kernel = (x * x + y * y <= radius_cells * radius_cells).astype(bool)
        return binary_dilation(grid, structure=kernel).astype(np.uint8)

    @staticmethod
    def _astar_grid(
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        allow_diagonal: bool = True,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* on a 2D occupancy grid.

        Parameters
        ----------
        grid : (H, W) with 1 = blocked, 0 = free
        start, goal : (row, col) integer cell coordinates
        allow_diagonal : 8-connected if True, else 4-connected

        Returns
        -------
        List of (row, col) from start to goal, or None if unreachable.
        """
        H, W = grid.shape

        def in_bounds(r, c):
            return 0 <= r < H and 0 <= c < W

        if not in_bounds(*start) or not in_bounds(*goal):
            return None
        if grid[start] == 1 or grid[goal] == 1:
            return None

        # 8-connected with corner-safety: diagonal step only if
        # both orthogonal neighbours are free (prevents clipping corners)
        if allow_diagonal:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
            move_costs = [1.0, 1.0, 1.0, 1.0,
                          1.4142, 1.4142, 1.4142, 1.4142]
        else:
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            move_costs = [1.0, 1.0, 1.0, 1.0]

        def heuristic(a, b):
            dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
            if allow_diagonal:
                # octile distance
                return max(dr, dc) + (1.4142 - 1.0) * min(dr, dc)
            return dr + dc

        start_t = (0.0, heuristic(start, goal), start)
        open_heap = [start_t]
        g_score = {start: 0.0}
        came_from: dict = {}
        closed = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                # reconstruct
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed.add(current)
            r, c = current

            for (dr, dc), cost in zip(moves, move_costs):
                nr, nc = r + dr, c + dc
                if not in_bounds(nr, nc) or grid[nr, nc] == 1:
                    continue

                # corner-cutting guard for diagonal moves
                if dr != 0 and dc != 0:
                    if grid[r, nc] == 1 or grid[nr, c] == 1:
                        continue

                neighbor = (nr, nc)
                if neighbor in closed:
                    continue

                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, heuristic(neighbor, goal), neighbor))

        return None  # unreachable

    @staticmethod
    def _world_to_grid(
        xy: Tuple[float, float],
        origin: np.ndarray,
        resolution: float,
    ) -> Tuple[int, int]:
        """World (x, y) -> grid (row, col)."""
        col = int(round((xy[0] - origin[0]) / resolution))
        row = int(round((xy[1] - origin[1]) / resolution))
        return row, col

    @staticmethod
    def _grid_to_world(
        rc: Tuple[int, int],
        origin: np.ndarray,
        resolution: float,
    ) -> Tuple[float, float]:
        """Grid (row, col) -> world (x, y)."""
        row, col = rc
        x = origin[0] + col * resolution
        y = origin[1] + row * resolution
        return x, y

    @staticmethod
    def _smooth_path(
        path_xy: np.ndarray,
        grid: np.ndarray,
        origin: np.ndarray,
        resolution: float,
        iterations: int = 3,
    ) -> np.ndarray:
        """
        Simple shortcut smoothing: try to connect non-adjacent waypoints
        if the straight segment is collision-free on the grid.
        """
        if len(path_xy) <= 2:
            return path_xy

        def line_is_free(a, b):
            """Bresenham-ish check of grid cells along segment a→b."""
            ar, ac = OpenRobInterface._world_to_grid(a, origin, resolution)
            br, bc = OpenRobInterface._world_to_grid(b, origin, resolution)
            steps = max(abs(br - ar), abs(bc - ac), 1)
            for i in range(steps + 1):
                t = i / steps
                r = int(round(ar + t * (br - ar)))
                c = int(round(ac + t * (bc - ac)))
                if grid[r, c] == 1:
                    return False
            return True

        for _ in range(iterations):
            new_path = [path_xy[0]]
            i = 0
            while i < len(path_xy) - 1:
                # look ahead as far as possible
                j = len(path_xy) - 1
                while j > i + 1 and not line_is_free(path_xy[i], path_xy[j]):
                    j -= 1
                new_path.append(path_xy[j])
                i = j
            path_xy = np.array(new_path, dtype=np.float32)
        return path_xy

    # ------------------------------------------------------------------
    # Scene query helpers
    # ------------------------------------------------------------------
    def _get_obj_pose(self, name: str):
        obj = self.env.scene[name]
        return torch.as_tensor(obj.data.root_pos_w)[0].clone(), torch.as_tensor(obj.data.root_quat_w)[0].clone()

    def _ee_pose(self, arm: str):
        ee_id = self.left_ee_id if arm == "left" else self.right_ee_id
        return torch.as_tensor(self.robot.data.body_pos_w)[:, ee_id], torch.as_tensor(self.robot.data.body_quat_w)[:, ee_id]

    # ------------------------------------------------------------------
    # Public: generic step
    # ------------------------------------------------------------------
    def step_raw(self, action: torch.Tensor):
        """Send a raw 19-dim action straight to the environment."""
        return self._step(action)

    def hold(self, steps: int = 5, base: Iterable[float] = (0.0, 0.0, 0.0)):
        """Keep the current arm targets and gripper state for `steps` ticks."""
        base_t = self._base_cmd(*base)
        l = ArmTarget(self._left_target.pos, self._left_target.quat, self._left_target.grip)
        r = ArmTarget(self._right_target.pos, self._right_target.quat, self._right_target.grip)
        out = None
        for _ in range(steps):
            out = self._step(self._compose_action(l, r, base_t))
        return out

    # ------------------------------------------------------------------
    # 1) Navigation
    # ------------------------------------------------------------------
    def navigate(
        self,
        goal_xy: Sequence[float],
        goal_yaw: float = 0.0,
        resolution: float = 0.05,
        robot_radius: float = 0.35,
        z_min: float = 0.05,
        z_max: float = 2.0,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        allow_diagonal: bool = True,
        pos_tol: float = 0.05,
        yaw_tol: float = 0.05,
        v_max: float = 0.3,
        k_yaw: float = 2.0,
        lookahead_dist: float = 0.25,
        max_iters: int = 500,
    ):
        """
        Plan a collision-free path to (goal_x, goal_y) with A* and follow it
        using pure-pursuit on the mecanum base.

        1. Build an inflated occupancy grid from the scene.
        2. Run A* from the current base pose to the goal cell.
        3. Convert the grid path to world waypoints.
        4. Smooth the path by shortcutting.
        5. Follow waypoints with a lookahead pure-pursuit controller,
        writing (vx_body, vy_body, wz) into the 19-dim action.
        """
        # ---- 1. grid ----
        raw_grid, origin = self._build_occupancy_grid(
            resolution, z_min, z_max, bounds)
        grid = self._inflate_grid(raw_grid, robot_radius, resolution)

        # ---- 2. A* ----
        base_xy = torch.as_tensor(self.robot.data.root_pos_w)[0, :2].cpu().numpy()
        start_rc = self._world_to_grid(tuple(base_xy), origin, resolution)
        goal_rc = self._world_to_grid(tuple(goal_xy), origin, resolution)

        # clamp to grid bounds
        H, W = grid.shape
        start_rc = (max(0, min(H - 1, start_rc[0])),
                    max(0, min(W - 1, start_rc[1])))
        goal_rc = (max(0, min(H - 1, goal_rc[0])),
                max(0, min(W - 1, goal_rc[1])))

        grid_path = self._astar_grid(grid, start_rc, goal_rc, allow_diagonal)
        if grid_path is None:
            print(f"[navigate] A* found no path to {goal_xy}")
            return False

        # ---- 3. world waypoints ----
        path_xy = np.array(
            [self._grid_to_world(rc, origin, resolution) for rc in grid_path],
            dtype=np.float32,
        )
        # ensure final waypoint is exactly the requested goal
        path_xy[-1] = goal_xy

        # ---- 4. smooth ----
        path_xy = self._smooth_path(path_xy, grid, origin, resolution, iterations=4)

        # ---- 5. follow ----
        wp_idx = 0
        n_wp = len(path_xy)
        iters = 0

        while wp_idx < n_wp and iters < max_iters:
            iters += 1
            base_xy = torch.as_tensor(self.robot.data.root_pos_w)[0, :2].cpu().numpy()
            base_quat = torch.as_tensor(self.robot.data.root_quat_w)[0]
            base_yaw = math_utils.euler_xyz_from_quat(
                base_quat.unsqueeze(0))[2][0].item()

            # advance waypoints we've already reached
            while wp_idx < n_wp:
                d = np.linalg.norm(path_xy[wp_idx] - base_xy)
                if d < pos_tol:
                    wp_idx += 1
                else:
                    break
            if wp_idx >= n_wp:
                break

            # pure-pursuit lookahead
            look_idx = wp_idx
            for i in range(wp_idx, n_wp):
                d = np.linalg.norm(path_xy[i] - base_xy)
                if d >= lookahead_dist:
                    look_idx = i
                    break
                look_idx = i

            target = path_xy[look_idx]
            d_vec = target - base_xy
            d_norm = np.linalg.norm(d_vec)
            if d_norm < 1e-6:
                continue
            d_hat = d_vec / d_norm

            # world-frame velocity
            speed = min(v_max, k_yaw * d_norm)
            v_world = d_hat * speed

            # rotate into body frame for mecanum base
            c, s = np.cos(base_yaw), np.sin(base_yaw)
            vx_body = float(c * v_world[0] + s * v_world[1])
            vy_body = float(-s * v_world[0] + c * v_world[1])

            # yaw control toward final goal orientation
            dyaw = math.atan2(
                math.sin(goal_yaw - base_yaw),
                math.cos(goal_yaw - base_yaw))
            wz = float(np.clip(k_yaw * dyaw,
                            -self.max_base_w, self.max_base_w))

            self._step(self._compose_action(
                ArmTarget(self._left_target.pos, self._left_target.quat,
                        self._left_target.grip),
                ArmTarget(self._right_target.pos, self._right_target.quat,
                        self._right_target.grip),
                self._base_cmd(vx_body, vy_body, wz),
            ))

        # final yaw alignment
        for _ in range(60):
            base_quat = torch.as_tensor(self.robot.data.root_quat_w)[0]
            base_yaw = math_utils.euler_xyz_from_quat(
                base_quat.unsqueeze(0))[2][0].item()
            dyaw = math.atan2(
                math.sin(goal_yaw - base_yaw),
                math.cos(goal_yaw - base_yaw))
            if abs(dyaw) < yaw_tol:
                break
            wz = float(np.clip(k_yaw * dyaw,
                            -self.max_base_w, self.max_base_w))
            self._step(self._compose_action(
                ArmTarget(self._left_target.pos, self._left_target.quat,
                        self._left_target.grip),
                ArmTarget(self._right_target.pos, self._right_target.quat,
                        self._right_target.grip),
                self._base_cmd(0.0, 0.0, wz),
            ))

        return True

    # Add to OpenRobInterface

    def navigate_path(
        self,
        path_xy: torch.Tensor,       # (N, 2) waypoints in world frame
        goal_yaw: float = 0.0,
        pos_tol: float = 0.05,
        yaw_tol: float = 0.05,
        lookahead: float = 0.3,
        v_max: float = 0.3,
        k_yaw: float = 2.0,
        max_iters_per_wp: int = 100,
    ):
        """
        Follow a precomputed path of (x, y) waypoints using pure pursuit.
        The planner (A*, Nav2, or your own global planner) supplies path_xy.
        Mecanum base consumes body-frame (vx, vy, wz) as before.
        """
        path = path_xy.to(self.device)
        wp_idx = 0
        total_wp = path.shape[0]

        while wp_idx < total_wp:
            base_pos = torch.as_tensor(self.robot.data.root_pos_w)[0, :2]
            target = path[wp_idx]
            d = target - base_pos
            dist = d.norm()

            # advance waypoint if close enough
            if dist < pos_tol:
                wp_idx += 1
                continue

            # pure-pursuit lookahead
            look_idx = min(wp_idx + 1, total_wp - 1)
            look_target = path[look_idx]
            d_look = look_target - base_pos
            d_look_norm = d_look / (d_look.norm() + 1e-6)

            base_quat = torch.as_tensor(self.robot.data.root_quat_w)[0]
            base_yaw = math_utils.euler_xyz_from_quat(
                base_quat.unsqueeze(0))[2][0]

            # desired velocity toward lookahead point
            v_world = d_look_norm * min(v_max, d_look.norm() * 1.5)

            # rotate world -> body for mecanum
            c, s = torch.cos(base_yaw), torch.sin(base_yaw)
            vx_body = c * v_world[0] + s * v_world[1]
            vy_body = -s * v_world[0] + c * v_world[1]

            # yaw control toward final goal
            dyaw = math.atan2(math.sin(goal_yaw - base_yaw),
                            math.cos(goal_yaw - base_yaw))
            w = max(-self.max_base_w, min(self.max_base_w, k_yaw * dyaw))

            self._step(self._compose_action(
                ArmTarget(self._left_target.pos, self._left_target.quat,
                        self._left_target.grip),
                ArmTarget(self._right_target.pos, self._right_target.quat,
                        self._right_target.grip),
                self._base_cmd(float(vx_body), float(vy_body), float(w)),
            ))

        # final yaw alignment
        for _ in range(50):
            base_quat = torch.as_tensor(self.robot.data.root_quat_w)[0]
            base_yaw = math_utils.euler_xyz_from_quat(
                base_quat.unsqueeze(0))[2][0]
            dyaw = math.atan2(math.sin(goal_yaw - base_yaw),
                            math.cos(goal_yaw - base_yaw))
            if abs(dyaw) < yaw_tol:
                break
            w = max(-self.max_base_w, min(self.max_base_w, k_yaw * dyaw))
            self._step(self._compose_action(
                ArmTarget(self._left_target.pos, self._left_target.quat,
                        self._left_target.grip),
                ArmTarget(self._right_target.pos, self._right_target.quat,
                        self._right_target.grip),
                self._base_cmd(0.0, 0.0, float(w)),
            ))
    
    # ------------------------------------------------------------------
    # 2) Pick (single arm)
    # ------------------------------------------------------------------
    def pick(
        self,
        obj_name: str,
        arm: str = "right",
        approach_height: float = 0.15,
        pre_grasp_steps: int = 20,
        descend_steps: int = 25,
        lift_height: float = 0.10,
        settle_steps: int = 10,
        gripper_quat: Optional[torch.Tensor] = None,
    ):
        """
        Grasp an object by its scene name.  Assumes a top-down grasp is
        acceptable; pass `gripper_quat` to override orientation.
        """
        obj_pos, obj_quat = self._get_obj_pose(obj_name)
        if gripper_quat is None:
            gripper_quat = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], device=self.device)  # identity

        # 1) pre-grasp
        pre = obj_pos + torch.tensor([0.0, 0.0, approach_height],
                                     device=self.device)
        self._move_arm_to(arm, pre, gripper_quat, grip=0.0,
                          steps=pre_grasp_steps)
        # 2) descend
        down = obj_pos.clone()
        self._move_arm_to(arm, down, gripper_quat, grip=0.0,
                          steps=descend_steps)
        # 3) close gripper
        self._move_arm_to(arm, down, gripper_quat, grip=1.0,
                          steps=settle_steps)
        # 4) lift
        lift = down + torch.tensor([0.0, 0.0, lift_height],
                                   device=self.device)
        self._move_arm_to(arm, lift, gripper_quat, grip=1.0,
                          steps=pre_grasp_steps)
        return True

    # ------------------------------------------------------------------
    # 3) Place
    # ------------------------------------------------------------------
    def place(
        self,
        target_xyz: Sequence[float],
        arm: str = "right",
        target_quat: Optional[torch.Tensor] = None,
        approach_height: float = 0.10,
        steps: int = 25,
    ):
        pos = torch.tensor(target_xyz, device=self.device, dtype=torch.float32)
        if target_quat is None:
            target_quat = torch.tensor([1.0, 0.0, 0.0, 0.0],
                                       device=self.device)
        # move above
        self._move_arm_to(arm, pos + torch.tensor(
            [0, 0, approach_height], device=self.device),
            target_quat, grip=1.0, steps=steps)
        # descend
        self._move_arm_to(arm, pos, target_quat, grip=1.0, steps=steps)
        # open gripper
        self._move_arm_to(arm, pos, target_quat, grip=0.0, steps=steps // 2)
        # retreat
        self._move_arm_to(arm, pos + torch.tensor(
            [0, 0, approach_height], device=self.device),
            target_quat, grip=0.0, steps=steps)

    # ------------------------------------------------------------------
    # 4) Open / Close (door, drawer, ...)
    # ------------------------------------------------------------------
    def open(
        self,
        obj_name: str,
        arm: str = "right",
        pull_dir: Sequence[float] = (0.0, -1.0, 0.0),
        distance: float = 0.25,
        approach_height: float = 0.10,
        steps: int = 30,
    ):
        """Pull a door / drawer open along `pull_dir` (unit vector)."""
        handle_pos, handle_quat = self._get_obj_pose(obj_name)
        pre = handle_pos + torch.tensor(
            [0, 0, approach_height], device=self.device)
        self._move_arm_to(arm, pre, handle_quat, grip=0.0, steps=steps)
        self._move_arm_to(arm, handle_pos, handle_quat, grip=0.0, steps=steps)
        self._move_arm_to(arm, handle_pos, handle_quat, grip=1.0,
                          steps=steps // 2)
        pull = torch.tensor(pull_dir, device=self.device, dtype=torch.float32)
        pull = pull / (pull.norm() + 1e-9)
        goal = handle_pos + pull * distance
        self._move_arm_to(arm, goal, handle_quat, grip=1.0, steps=steps * 2)
        self._move_arm_to(arm, goal, handle_quat, grip=0.0, steps=steps // 2)

    def close(
        self,
        obj_name: str,
        arm: str = "right",
        push_dir: Sequence[float] = (0.0, 1.0, 0.0),
        distance: float = 0.25,
        approach_height: float = 0.10,
        steps: int = 30,
    ):
        """Push a door / drawer closed along `push_dir`."""
        handle_pos, handle_quat = self._get_obj_pose(obj_name)
        pre = handle_pos + torch.tensor(
            [0, 0, approach_height], device=self.device)
        self._move_arm_to(arm, pre, handle_quat, grip=0.0, steps=steps)
        self._move_arm_to(arm, handle_pos, handle_quat, grip=1.0, steps=steps)
        push = torch.tensor(push_dir, device=self.device, dtype=torch.float32)
        push = push / (push.norm() + 1e-9)
        goal = handle_pos + push * distance
        self._move_arm_to(arm, goal, handle_quat, grip=1.0, steps=steps * 2)
        self._move_arm_to(arm, goal, handle_quat, grip=0.0, steps=steps // 2)

    # ------------------------------------------------------------------
    # 5) Press button
    # ------------------------------------------------------------------
    def press_button(
        self,
        button_name: str,
        arm: str = "right",
        press_depth: float = 0.03,
        approach_height: float = 0.06,
        steps: int = 20,
    ):
        btn_pos, btn_quat = self._get_obj_pose(button_name)
        # point the gripper along +z (top-down); override via quat if needed
        pre = btn_pos + torch.tensor(
            [0, 0, approach_height], device=self.device)
        self._move_arm_to(arm, pre, btn_quat, grip=1.0, steps=steps)
        self._move_arm_to(arm, btn_pos, btn_quat, grip=1.0, steps=steps)
        self._move_arm_to(arm, btn_pos - torch.tensor(
            [0, 0, press_depth], device=self.device), btn_quat, grip=1.0,
            steps=steps)
        self._move_arm_to(arm, pre, btn_quat, grip=1.0, steps=steps)

    # ------------------------------------------------------------------
    # 6) Bimanual lift
    # ------------------------------------------------------------------
    def bimanual_lift(
        self,
        obj_name: str,
        lift_height: float = 0.30,
        approach_height: float = 0.15,
        lateral_offset: float = 0.15,
        steps: int = 30,
    ):
        """Grasp a large object with both arms and lift it."""
        obj_pos, obj_quat = self._get_obj_pose(obj_name)
        # left / right grasp offsets (local +/-y)
        off_l = torch.tensor([0.0, +lateral_offset, 0.0], device=self.device)
        off_r = torch.tensor([0.0, -lateral_offset, 0.0], device=self.device)
        grip_q = (self._left_target.quat + self._right_target.quat)/2.0

        # pre-grasp
        self._move_arms_to(
            obj_pos + off_l + torch.tensor([0, 0, approach_height],
                                           device=self.device),
            obj_pos + off_r + torch.tensor([0, 0, approach_height],
                                           device=self.device),
            grip_q, left_grip=0.0, right_grip=0.0, steps=steps)
        # descend
        self._move_arms_to(
            obj_pos + off_l, obj_pos + off_r,
            grip_q, left_grip=0.0, right_grip=0.0, steps=steps)
        # close
        self._move_arms_to(
            obj_pos + off_l, obj_pos + off_r,
            grip_q, left_grip=1.0, right_grip=1.0, steps=steps // 2)
        # lift
        up = torch.tensor([0, 0, lift_height], device=self.device)
        self._move_arms_to(
            obj_pos + off_l + up, obj_pos + off_r + up,
            grip_q, left_grip=1.0, right_grip=1.0, steps=steps)

    def bimanual_drop(
        self,
        release_height: float = 0.05,
        release_steps: int = 15,
        retreat_height: float = 0.15,
        retreat_steps: int = 20,
    ):
        """
        Release the object currently held by both arms (call after bimanual_lift).

        Phase 1: lower both arms slightly to `release_height`.
        Phase 2: open both grippers and hold for `release_steps`.
        Phase 3: retreat upward by `retreat_height` with grippers open.

        The object falls under gravity during phase 2-3, so the retreat should
        start *after* the grippers have opened enough to clear the object.
        """
        l_ee, _ = self._ee_pose("left")
        r_ee, _ = self._ee_pose("right")
        l_now = l_ee[0].clone()
        r_now = r_ee[0].clone()

        l_release = l_now + torch.tensor(
            [0.0, 0.0, -release_height], device=self.device)
        r_release = r_now + torch.tensor(
            [0.0, 0.0, -release_height], device=self.device)

        # Phase 1: lower with grippers still closed
        self._move_arms_to(
            l_release, r_release, self._left_target.quat,
            left_grip=1.0, right_grip=1.0, steps=release_steps)

        # Phase 2: open grippers — object starts falling
        self._move_arms_to(
            l_release, r_release, self._left_target.quat,
            left_grip=0.0, right_grip=0.0, steps=release_steps)

        # Phase 3: retreat upward with grippers open
        l_retreat = l_release + torch.tensor(
            [0.0, 0.0, retreat_height], device=self.device)
        r_retreat = r_release + torch.tensor(
            [0.0, 0.0, retreat_height], device=self.device)
        self._move_arms_to(
            l_retreat, r_retreat, self._left_target.quat,
            left_grip=0.0, right_grip=0.0, steps=retreat_steps)

        # Update stored targets so subsequent calls hold the new pose
        self._left_target = ArmTarget(l_retreat, self._left_target.quat, 0.0)
        self._right_target = ArmTarget(r_retreat, self._right_target.quat, 0.0)
    
    # ------------------------------------------------------------------
    # internal single/multi-arm motion helpers
    # ------------------------------------------------------------------
    def _move_arm_to(
        self, arm: str, pos: torch.Tensor, quat: torch.Tensor,
        grip: float, steps: int,
    ):
        tgt = ArmTarget(pos, quat, grip)
        if arm == "left":
            self._left_target = tgt
        else:
            self._right_target = tgt
        self._drive_to_targets(steps)

    def _move_arms_to(
        self, left_pos, right_pos, quat,
        left_grip: float, right_grip: float, steps: int,
    ):
        self._left_target = ArmTarget(left_pos, quat, left_grip)
        self._right_target = ArmTarget(right_pos, quat, right_grip)
        self._drive_to_targets(steps)

    def _drive_to_targets(self, steps: int):
        """Iterate: solve IK, apply joint targets, step."""
        for _ in range(steps):
            lq = self._solve_arm(
                self.left_ik, self.left_joint_ids, self.left_ee_id,
                self._left_target)
            rq = self._solve_arm(
                self.right_ik, self.right_joint_ids, self.right_ee_id,
                self._right_target)
            self._apply_joint_targets(lq, rq)
            self._apply_grippers(self._left_target.grip, self._right_target.grip)

            # Encoder both arms + grip into the 19-dim action as a fallback /
            # for envs whose action term uses them directly.
            action = self._compose_action(
                self._left_target, self._right_target,
                self._base_cmd(0.0, 0.0, 0.0))
            self._step(action)

    # ------------------------------------------------------------------
    # reset / state
    # ------------------------------------------------------------------
    def reset(self):
        return self.env.reset()

    def reset_arm_targets_to_current(self):
        lp, lq = self._ee_pose("left")
        rp, rq = self._ee_pose("right")
        self._left_target = ArmTarget(lp[0].clone(), lq[0].clone(), 0.0)
        self._right_target = ArmTarget(rp[0].clone(), rq[0].clone(), 0.0)

    def close_connection(self):
        self.env.close()