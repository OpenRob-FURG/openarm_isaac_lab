# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Keyboard device for teleoperating two arms with 6-DOF delta pose and binary gripper."""

import numpy as np
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Dict, Optional

import carb
import carb.input
import omni.graph.core as og
from omni.kit.widget.settings import SettingType
import omni.ui as ui
import omni.kit.commands
from omni.physx.scripts import utils

from isaaclab.devices import DeviceBase
from isaaclab.utils.math import convert_quat

from isaaclab_arena.teleop_devices import register_device, TeleopDeviceBase
from isaaclab.devices.device_base import DevicesCfg

import torch



class DualArmKeyboard(DeviceBase):
    """A keyboard device for teleoperating two arms with 6-DOF delta pose and binary gripper.

    This device maps keyboard keys to delta pose commands for two robot arms and allows switching
    between active arms. It maintains the same keymap as the default keyboard device for the active arm,
    with an additional key for switching arms.

    Keymap:
        - W/S: Move forward/backward along x-axis (for active arm)
        - A/D: Move left/right along y-axis (for active arm)
        - Q/E: Move up/down along z-axis (for active arm)
        - I/K: Rotate around x-axis (roll)
        - J/L: Rotate around y-axis (pitch)
        - U/O: Rotate around z-axis (yaw)
        - Space/Left Ctrl: Open/Close gripper (for active arm)
        - Tab: Switch active arm between arm0 and arm1
        - Numpad +/-: Increase/decrease movement sensitivity
        - Numpad *: Reset commands to zero
    """

    def __init__(self, cfg = None):
        """Initialize the keyboard layer.

        Args:
            cfg: The configuration for the device. If None, default configuration is used.
        """
        # Store configuration
        self.cfg = cfg or DualArmTeleopCfg()
        
        # Store arm indices
        if self.cfg.arm_indices is None:
            self.cfg.arm_indices = [0, 1]
        
        # Initialize commands for both arms: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
        self._delta_pose = {
            self.cfg.arm_indices[0]: np.zeros(6),
            self.cfg.arm_indices[1]: np.zeros(6)
        }
        self._gripper_command = {
            self.cfg.arm_indices[0]: 0.0,  # 0: open, 1: close
            self.cfg.arm_indices[1]: 0.0
        }
        
        # Active arm
        self._active_arm = self.cfg.active_arm
        
        # Sensitivity settings
        self.pos_sensitivity = self.cfg.pos_sensitivity
        self.rot_sensitivity = self.cfg.rot_sensitivity
        
        # Key states
        self._pressed_keys = set()
        
        # Bindings for key actions
        self._key_action_map = {
            # Forward/backward along x-axis
            'W': 'pos_x_positive',
            'S': 'pos_x_negative',
            # Left/right along y-axis
            'A': 'pos_y_positive',
            'D': 'pos_y_negative',
            # Up/down along z-axis
            'Q': 'pos_z_positive',
            'E': 'pos_z_negative',
            # Rotations
            'I': 'rot_x_positive',
            'K': 'rot_x_negative',
            'J': 'rot_y_negative',
            'L': 'rot_y_positive',
            'U': 'rot_z_positive',
            'O': 'rot_z_negative',
            # Gripper control
            'C': 'gripper_close',  # Space
            'G': 'gripper_open',
            # Switching arms
            'TAB': 'switch_arm',  # Tab
            # Sensitivity adjustment
            'NUMPAD_ADD': 'increase_sensitivity',
            'NUMPAD_SUBTRACT': 'decrease_sensitivity',
            # Reset
            'NUMPAD_MULTIPLY': 'reset',
        }
        
        # Additional mapping for number pad keys
        self._special_key_map = {
            carb.input.KeyboardInput.NUMPAD_ADD: 'increase_sensitivity',
            carb.input.KeyboardInput.NUMPAD_SUBTRACT: 'decrease_sensitivity',
            carb.input.KeyboardInput.NUMPAD_MULTIPLY: 'reset',
        }
        
        # Subscribe to keyboard events
        self._input_interface = carb.input.acquire_input_interface()
        self._app_window = omni.appwindow.get_default_app_window()
        self._keyboard_sub = self._input_interface.subscribe_to_keyboard_events(
            self._app_window.get_keyboard(), self._keyboard_event_handler
        )
        
        # Print info
        print("=========================================")
        print("Dual-Arm Keyboard Teleop Device")
        print("=========================================")
        print(f"Active arm: {self._active_arm} (use Tab to switch)")
        print(f"Arm indices: {self.cfg.arm_indices[0]} and {self.cfg.arm_indices[1]}")
        print("-----------------------------------------")
        print("Movement (active arm):")
        print("  - W/S: forward/backward (x-axis)")
        print("  - A/D: left/right (y-axis)")
        print("  - Q/E: up/down (z-axis)")
        print("Rotations (active arm):")
        print("  - I/K: roll (x-axis)")
        print("  - J/L: pitch (y-axis)")
        print("  - U/O: yaw (z-axis)")
        print("Gripper:")
        print("  - Space: Close")
        print("  - Left Ctrl: Open")
        print("-----------------------------------------")
        print("Switching:")
        print("  - Tab: Switch between arms")
        print("Sensitivity:")
        print("  - Numpad +: Increase")
        print("  - Numpad -: Decrease")
        print("  - Numpad *: Reset commands")
        print("=========================================")

    def __del__(self):
        """Unsubscribe from keyboard events when deleted."""
        self._input_interface.unsubscribe_to_keyboard_events(
            self._app_window.get_keyboard(), self._keyboard_sub
        )

    def _keyboard_event_handler(self, event, *args, **kwargs):
        """Handle keyboard events and update command state."""
        # Apply action on key press or hold
        if event.type == carb.input.KeyboardEventType.KEY_PRESS or \
           event.type == carb.input.KeyboardEventType.KEY_REPEAT:
            
            # Check if key is in special key map
            #print(event.input.name)
            action = self._special_key_map.get(event.input, None)
            if action is None:
                # Check normal key map
                action = self._key_action_map.get(event.input.name, None)
            
            if action is not None:
                self._apply_action(action)
                self._pressed_keys.add(event.input.name)
            #print(action)
        # Remove action on key release
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            action = self._key_action_map.get(event.input.name, None)
            if action is not None:
                self._remove_action(action)
                if event.input.name in self._pressed_keys:
                    self._pressed_keys.remove(event.input.name)

        return True

    def _apply_action(self, action: str):
        """Apply action to update command state."""
        active_idx = self.cfg.arm_indices[self._active_arm]
        
        # Position deltas
        if action == 'pos_x_positive':
            self._delta_pose[active_idx][0] = self.pos_sensitivity
        elif action == 'pos_x_negative':
            self._delta_pose[active_idx][0] = -self.pos_sensitivity
        elif action == 'pos_y_positive':
            self._delta_pose[active_idx][1] = self.pos_sensitivity
        elif action == 'pos_y_negative':
            self._delta_pose[active_idx][1] = -self.pos_sensitivity
        elif action == 'pos_z_positive':
            self._delta_pose[active_idx][2] = self.pos_sensitivity
        elif action == 'pos_z_negative':
            self._delta_pose[active_idx][2] = -self.pos_sensitivity
        
        # Rotation deltas
        elif action == 'rot_x_positive':
            self._delta_pose[active_idx][3] = self.rot_sensitivity
        elif action == 'rot_x_negative':
            self._delta_pose[active_idx][3] = -self.rot_sensitivity
        elif action == 'rot_y_positive':
            self._delta_pose[active_idx][4] = self.rot_sensitivity
        elif action == 'rot_y_negative':
            self._delta_pose[active_idx][4] = -self.rot_sensitivity
        elif action == 'rot_z_positive':
            self._delta_pose[active_idx][5] = self.rot_sensitivity
        elif action == 'rot_z_negative':
            self._delta_pose[active_idx][5] = -self.rot_sensitivity
        
        # Gripper control
        elif action == 'gripper_close':
            self._gripper_command[active_idx] = -1.0
        elif action == 'gripper_open':
            self._gripper_command[active_idx] = 1.0
        
        # Switch active arm
        elif action == 'switch_arm':
            self._active_arm = (self._active_arm + 1) % 2
            print(f"Switched to arm {self.cfg.arm_indices[self._active_arm]}")
        
        # Sensitivity adjustments
        elif action == 'increase_sensitivity':
            self.pos_sensitivity += 0.001
            self.rot_sensitivity += 0.002
            print(f"Sensitivity increased: pos={self.pos_sensitivity:.3f}, rot={self.rot_sensitivity:.3f}")
        elif action == 'decrease_sensitivity':
            self.pos_sensitivity = max(0.001, self.pos_sensitivity - 0.001)
            self.rot_sensitivity = max(0.001, self.rot_sensitivity - 0.002)
            print(f"Sensitivity decreased: pos={self.pos_sensitivity:.3f}, rot={self.rot_sensitivity:.3f}")
        
        # Reset commands
        elif action == 'reset':
            for arm_idx in self.cfg.arm_indices:
                self._delta_pose[arm_idx] = np.zeros(6)
                self._gripper_command[arm_idx] = 1.0
            print("Reset all commands to zero")

    def _remove_action(self, action: str):
        """Remove action from command state."""
        active_idx = self.cfg.arm_indices[self._active_arm]
        
        # Reset deltas to zero when key released
        if 'pos' in action or 'rot' in action:
            if 'pos_x' in action:
                self._delta_pose[active_idx][0] = 0.0
            elif 'pos_y' in action:
                self._delta_pose[active_idx][1] = 0.0
            elif 'pos_z' in action:
                self._delta_pose[active_idx][2] = 0.0
            elif 'rot_x' in action:
                self._delta_pose[active_idx][3] = 0.0
            elif 'rot_y' in action:
                self._delta_pose[active_idx][4] = 0.0
            elif 'rot_z' in action:
                self._delta_pose[active_idx][5] = 0.0
        
        # Gripper
        #elif 'gripper' in action:
        #    self._gripper_command[active_idx] = 0.0

    def reset(self):
        """Reset the device to initial state."""
        # Reset commands for both arms
        for arm_idx in self.cfg.arm_indices:
            self._delta_pose[arm_idx] = np.zeros(6)
            self._gripper_command[arm_idx] = 0.0
        
        # Reset active arm to default
        self._active_arm = self.cfg.active_arm
        
        # Clear pressed keys
        self._pressed_keys.clear()
        
        print("Device reset to initial state")

    def get_delta_pose(self, arm_index: Optional[int] = None) -> np.ndarray:
        """Get the current delta pose command for specified arm.
        
        Args:
            arm_index: The index of the arm. If None, returns delta pose for active arm.
            
        Returns:
            A 6-DOF delta pose command [dx, dy, dz, droll, dpitch, dyaw].
        """
        if arm_index is None:
            arm_index = self.cfg.arm_indices[self._active_arm]
        return self._delta_pose[arm_index].copy()

    def get_gripper_command(self, arm_index: Optional[int] = None) -> float:
        """Get the current gripper command for specified arm.
        
        Args:
            arm_index: The index of the arm. If None, returns command for active arm.
            
        Returns:
            Gripper command: 0.0 for open, 1.0 for close.
        """
        if arm_index is None:
            arm_index = self.cfg.arm_indices[self._active_arm]
        return self._gripper_command[arm_index]

    def get_active_arm(self) -> int:
        """Get the currently active arm index."""
        return self.cfg.arm_indices[self._active_arm]

    def get_all_commands(self) -> Dict[int, tuple[np.ndarray, float]]:
        """Get commands for all arms.
        
        Returns:
            Dictionary mapping arm indices to (delta_pose, gripper_command) tuples.
        """
        commands = {}
        for arm_idx in self.cfg.arm_indices:
            commands[arm_idx] = (self._delta_pose[arm_idx].copy(), self._gripper_command[arm_idx])
        return commands

    def add_callback(self, key, func):
        pass

    def advance(self):
        all_commands_merged = []
        all_commands = self.get_all_commands()
        for arm_idx in all_commands.keys():
            all_commands_merged.append(all_commands[arm_idx][0])
            all_commands_merged.append(np.asarray([all_commands[arm_idx][1],]))
        return torch.as_tensor(
            np.concatenate(all_commands_merged),
            device=torch.device('cuda')
        )

    @property
    def active_arm(self) -> int:
        """Get the currently active arm index."""
        return self.cfg.arm_indices[self._active_arm]
    

@dataclass
class DualArmTeleopCfg:
    """Configuration for the dual-arm keyboard teleop device."""
    
    arm_indices: list[int] = None
    """Indices of the two arms to control. Default: [0, 1]."""
    
    pos_sensitivity: float = 0.01
    """Sensitivity of moving the arm position. Default: 0.01."""
    
    rot_sensitivity: float = 0.02
    """Sensitivity of rotating the arm. Default: 0.02."""
    
    active_arm: int = 0
    """Initially active arm index. Default: 0."""

    class_type = DualArmKeyboard


@register_device
class KeyboardBimanualTeleopDevice(TeleopDeviceBase):
    name = "keyboard_bimanual"

    def get_teleop_device_cfg(self, embodiment = None):
        return DevicesCfg(
                devices=dict(
                    keyboard_bimanual=DualArmTeleopCfg(
                        arm_indices=[0,1]
                    ),
                )
        )