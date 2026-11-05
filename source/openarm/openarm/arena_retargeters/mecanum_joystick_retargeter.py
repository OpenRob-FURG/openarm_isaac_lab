import numpy as np
from dataclasses import dataclass

#from isaacteleop.retargeters.base_retargeter import RetargeterBase
from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource


@dataclass
class MecanumJoystickRetargeterConfig:
    """Config for mapping Quest 3 thumbsticks to a planar base command."""
    left_stick_translation: bool = True   # left stick -> (vx, vy)
    right_stick_yaw: bool = True          # right stick x -> wz
    deadzone: float = 0.15
    max_linear_speed: float = 0.5         # m/s at full deflection
    max_angular_speed: float = 1.0        # rad/s at full deflection
    invert_vy: bool = False
    invert_wz: bool = False


class MecanumJoystickRetargeter:
    """Convert Meta Quest 3 controller thumbsticks into a mecanum base command.

    Output: 3D array [vx, vy, wz] in the robot base frame.
    """

    def __init__(self, cfg: MecanumJoystickRetargeterConfig, name: str = "mecanum_base"):
        #super().__init__(name=name)
        self.name = name
        self.cfg = cfg
        self._outputs = {"base_command": np.zeros(3, dtype=np.float32)}

    def connect(self, inputs: dict):
        """Wire controller outputs into this retargeter.

        `inputs` is expected to contain the controller nodes for LEFT and RIGHT
        (the transformed controller outputs from `controllers.transformed(...)`).
        """
        self._left = inputs[ControllersSource.LEFT]
        self._right = inputs[ControllersSource.RIGHT]
        return self

    # ---- stick accessors ---------------------------------------------------
    # OpenXR exposes thumbstick values via the controller node's thumbstick
    # output. The exact attribute name depends on the IsaacTeleop version;
    # these helpers tolerate the common variants and fall back to zeros.
    @staticmethod
    def _read_stick(selector):
        """Resolve a controller output selector to a 2D thumbstick vector."""
        # Output selectors are callable; calling yields the current tensor/array.
        val = selector() if callable(selector) else selector
        if val is None:
            return np.zeros(2, dtype=np.float32)
        # If the selector exposes named outputs (dict-like), prefer "thumbstick".
        if isinstance(val, dict):
            for key in ("thumbstick", "joystick", "stick"):
                if key in val:
                    return np.asarray(val[key], dtype=np.float32).reshape(-1)[:2]
        return np.asarray(val, dtype=np.float32).reshape(-1)[:2]

    def _apply_deadzone(self, v: np.ndarray) -> np.ndarray:
        mag = np.linalg.norm(v)
        if mag < self.cfg.deadzone:
            return np.zeros_like(v)
        # Rescale so the response is continuous past the deadzone.
        return v * (mag - self.cfg.deadzone) / (mag * (1.0 - self.cfg.deadzone))

    def compute(self) -> dict:
        left = self._apply_deadzone(self._read_stick(self._left))
        right = self._apply_deadzone(self._read_stick(self._right))

        vx = vy = wz = 0.0

        if self.cfg.left_stick_translation:
            # OpenXR convention: +y is forward, +x is right.
            vx = left[1] * self.cfg.max_linear_speed
            vy = left[0] * self.cfg.max_linear_speed
            if self.cfg.invert_vy:
                vy = -vy

        if self.cfg.right_stick_yaw:
            wz = right[0] * self.cfg.max_angular_speed
            if self.cfg.invert_wz:
                wz = -wz

        self._outputs["base_command"] = np.array([vx, vy, wz], dtype=np.float32)
        return self._outputs

    def output(self, name: str):
        return self._outputs[name]