from isaaclab_arena.policy.policy_base import PolicyBase
from openai import OpenAI
import base64
import io
import json
import re
import numpy as np
import torch
from PIL import Image
import argparse
from collections import deque


class LLMPolicy(PolicyBase):
    """
    LLM-based policy: instead of a learned VLA (SmolVLA / Groot / XVLA) we query a
    vision-language model served by Ollama (OpenAI-compatible endpoint) and parse
    an action chunk out of the text response.
    """

    def __init__(
        self,
        task: str,
        policy_type: str = "llm",
        model: str = "qwen3.5",                     # Ollama model tag
        base_url: str = "http://localhost:11434/v1", # Ollama OpenAI-compat endpoint
        api_key: str = "ollama",
        action_dim: int = 14,                         # e.g. 6-DoF + gripper
        action_horizon: int = 1,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        image_size: tuple = (640, 480),
    ) -> None:
        super().__init__(config=None)
        self.task = task
        self.policy_type = policy_type
        self.model = model
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.image_size = image_size
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.buffer = deque([], maxlen=3)
        self.action_buffer = deque([], maxlen=3)

        # Ollama exposes an OpenAI-compatible REST API at /v1
        self.client = OpenAI(base_url=base_url, api_key=api_key)

        self.system_prompt = (
            "You are a robot policy controlling two 6-DoF arms with a parallel-jaw gripper.\n"
            "You receive one or more camera images, a natural-language task, and the robot's "
            "current proprioceptive state.\n\n"

            "Axis convention:\n"
            "  X: points forward\n"
            "  Y: points left\n"
            "  Z: points upward\n\n"

            "Proprioceptive state is provided as a concatenated list of:\n"
            "  left_ee_position_xyz_m        : [x, y, z] in meters (world frame)\n"
            "  left_ee_orientation_quat_wxyz : [x, y, z, w] unit quaternion (world frame)\n"
            "  left_gripper                  : prismatic joint position in meters\n"
            "  right_ee_position_xyz_m        : [x, y, z] in meters (world frame)\n"
            "  right_ee_orientation_quat_wxyz : [x, y, z, w] unit quaternion (world frame)\n"
            "  right_gripper                  : prismatic joint position in meters\n\n"

            "You must output ONLY a JSON object of the form:\n"
            '  {"actions": [[left_dx, left_dy, left_dz, left_droll, left_dpitch, left_dyaw, left_gripper, right_dx, right_dy, right_dz, right_droll, right_dpitch, right_dyaw, right_gripper], ...]}\n'
            f"with exactly {self.action_horizon} action(s), each having {self.action_dim} floats.\n\n"

            "Action semantics (per row):\n"
            "  .*_dx, .*_dy, .*_dz       : desired DELTA end-effector position of left and right arm in meters, "
            "world frame, clipped to [-0.05, 0.05] per step.\n"
            "  .*_droll, .*_dpitch, .*_dyaw : desired DELTA orientation of left and right arm end-effector in RADIANS, applied "
            "as intrinsic X-Y-Z Euler angles relative to the current orientation, "
            "each clipped to [-0.1, 0.1] per step.\n"
            "  .*_gripper          : target left and right gripper command in [-1, 1], -1=closed, 1=open.\n\n"

            "Rules:\n"
            "- Position is a DELTA from the current ee_position_xyz_m, NOT absolute.\n"
            "- Orientation is a DELTA RPY from the current quaternion, NOT absolute, "
            "and NOT a quaternion.\n"
            "- Output must be valid JSON. No prose, no markdown, no explanations."
        )

    # ------------------------------------------------------------------ helpers
    def _to_b64(self, img) -> str:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 3 and img.shape[0] in (1, 3):   # CHW -> HWC
                img = img.permute(1, 2, 0)
            img = img.numpy()
        if isinstance(img, np.ndarray):
            img = np.clip(img, 0, 255).astype(np.uint8) if img.max() > 1.0 \
                  else (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
        if not isinstance(img, Image.Image):
            raise TypeError(f"Unsupported image type: {type(img)}")
        img = img.convert("RGB").resize(self.image_size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _parse_actions(self, text: str) -> torch.Tensor:
        # Strip code fences if the model added them
        print(text)
        text = re.sub(r"```(?:json)?", "", text).strip("` \n")
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            match = re.search(r"\[.*\]", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in LLM output: {text!r}")
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            actions = np.asarray(data["actions"], dtype=np.float32)
        else:
            actions = np.asarray(data, dtype=np.float32)

        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action_dim={self.action_dim}, got {actions.shape[-1]}"
            )
        # Pad / truncate horizon
        if actions.shape[0] < self.action_horizon:
            pad = np.repeat(actions[-1:], self.action_horizon - actions.shape[0], axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        actions = actions[: self.action_horizon]

        return torch.from_numpy(actions)

    # ------------------------------------------------------------------ API
    @torch.no_grad()
    def get_action(self, env, observation: dict) -> torch.Tensor:
        """
        observation: dict with keys like
            - "image"      : HxWx3 uint8 / float, or tensor CHW
            - "image_*"    : additional camera views (optional)
            - "state"      : proprioceptive vector (optional)
            - "task"       : override task string (optional)
        Returns: tensor of shape (action_horizon, action_dim)
        """

        if len(self.action_buffer) != 0:
            return torch.as_tensor(self.action_buffer.pop(), device=env.device).reshape((1,self.action_dim))

        task = observation.get("task", self.task)

        content = [{"type": "text", "text": f"Task: {task}"}]

        # Collect all image-like entries
        for key, val in observation.items():
            if key == "camera_obs":
                for img_key in val.keys():
                    b64 = self._to_b64(val[img_key][0])
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })

        state = torch.concatenate([observation["policy"]["left_eef_pos"],
                            observation["policy"]["left_eef_quat"],
                            observation["policy"]["left_gripper_pos"],
                            observation["policy"]["right_eef_pos"],
                            observation["policy"]["right_eef_quat"],
                            observation["policy"]["right_gripper_pos"],], dim=-1).detach().cpu().numpy().ravel().tolist()
        print(state)
        self.buffer.append({"type": "text", "text": f"Robot state: {state}"})

        content.extend(self.buffer)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            reasoning_effort="none",
        )
        #self.buffer.clear()
        text = resp.choices[0].message.content
        try:
            actions = self._parse_actions(text)
            self.buffer.append({"type": "text", "text": f"Robot action: {actions.detach().cpu().numpy().ravel().tolist()}"})
            self.action_buffer.extend(actions.tolist())
            return torch.as_tensor(self.action_buffer.pop(), device=env.device).reshape((1,self.action_dim))
        except (ValueError, KeyError) as e:
            print(e)
            return self.get_action(env=env, observation=observation)
        
    def reset(self, env_ids = None):
        self.buffer.clear()
        self.action_buffer.clear()
        return super().reset(env_ids)
    
    @staticmethod
    def add_args_to_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    @staticmethod
    def from_args(args: argparse.Namespace) -> "LLMPolicy":
        raise NotImplementedError(
            "LLMPolicy is constructed directly from a task. "
            "Use policy_runner.py which builds the policy from CLI arguments."
        )