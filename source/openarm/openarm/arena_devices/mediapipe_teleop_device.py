# mediapipe_teleop_device.py
import cv2
import threading
import queue
import time
import numpy as np
import torch
from typing import Dict, Any, Optional

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from isaaclab.utils import configclass
from isaaclab.devices import DeviceBase
from isaaclab.devices.device_base import DeviceCfg
from isaaclab.devices.retargeter_base import RetargeterBase, RetargeterCfg

from isaaclab_arena.teleop_devices import register_device, TeleopDeviceBase

from dataclasses import dataclass, MISSING

from isaaclab.devices.device_base import DevicesCfg

# MediaPipe initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class MediaPipeHolisticProcessor:
    """Real-time MediaPipe Holistic processing on a separate thread"""
    
    def __init__(self, camera_id=0, image_width=640, image_height=480):
        self.camera_id = camera_id
        self.image_width = image_width
        self.image_height = image_height
        
        # Thread-safe data sharing
        self.latest_data = {
            'left_hand': None,
            'right_hand': None,
            'pose': None,
            'frame': None,
            'timestamp': 0
        }
        #self.data_lock = threading.Lock()
        self.running = False
        self.processing_thread = None
        
    def start(self):
        """Start the camera and processing thread"""
        if self.running:
            return
            
        self.running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
        
    def stop(self):
        """Stop the processing thread"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
            
    def get_latest_data(self) -> Dict[str, Any]:
        """Thread-safe access to latest processed data"""
        #with self.data_lock:
        #    return self.latest_data.copy()
        return self.latest_data.copy()
            
    def _processing_loop(self):
        """Main processing loop running on separate thread"""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        
        # Configure MediaPipe Holistic
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            enable_segmentation=False
        ) as holistic:
            
            while self.running and cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue
                    
                # Process the frame
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                
                # Extract and normalize landmarks
                hand_data = {}
                
                # Left hand landmarks (21 points in 3D)
                if results.left_hand_landmarks:
                    hand_data['left_hand'] = np.array([
                        [landmark.x, landmark.y, landmark.z] 
                        for landmark in results.left_hand_landmarks.landmark
                    ])
                    
                # Right hand landmarks
                if results.right_hand_landmarks:
                    hand_data['right_hand'] = np.array([
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in results.right_hand_landmarks.landmark
                    ])
                    
                # Pose landmarks for reference (shoulders, hips)
                if results.pose_landmarks:
                    hand_data['pose'] = np.array([
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in results.pose_landmarks.landmark
                    ])
                    
                # Draw landmarks on frame for visualization
                annotated_image = image.copy()
                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.pose_landmarks,
                        mp_holistic.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.
                        get_default_pose_landmarks_style()
                    )
                if results.left_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.left_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                if results.right_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        results.right_hand_landmarks,
                        mp_holistic.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Thread-safe update
                #with self.data_lock:
                #    self.latest_data = {
                #        **hand_data,
                #        'frame': annotated_image,
                #        'timestamp': time.time()
                #    }
                self.latest_data = {
                    **hand_data,
                    'frame': annotated_image,
                    'timestamp': time.time()
                }
                    
                # Display preview window
                cv2.imshow('MediaPipe Holistic - Hand Tracking', annotated_image)
                if cv2.waitKey(5) & 0xFF == 27:  # ESC to exit
                    self.running = False
                    break
                    
        cap.release()
        cv2.destroyAllWindows()

# hand_pose_retargeter.py
class HandPoseRetargeter(RetargeterBase):
    """Converts MediaPipe hand landmarks to OpenArm bimanual control commands"""
    
    def __init__(self, cfg: RetargeterCfg):
        super().__init__(cfg)
        
        # Arm configuration
        self.left_arm_id = cfg.left_arm_id
        self.right_arm_id = cfg.right_arm_id
        
        # Control parameters
        self.workspace_scale = torch.tensor(cfg.workspace_scale)
        self.base_position = torch.tensor(cfg.base_position)
        self.max_velocity = cfg.max_velocity
        
        # Calibration state
        self.calibrated = False
        self.left_hand_offset = torch.zeros(3)
        self.right_hand_offset = torch.zeros(3)
        self.shoulder_distance = 0.4
        
        # Filter for smooth control
        self.filter_alpha = cfg.filter_alpha
        self.prev_left_pos = torch.zeros(3)
        self.prev_right_pos = torch.zeros(3)
        self.prev_left_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        self.prev_right_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        
    def calibrate(self, left_hand_pos, right_hand_pos, shoulder_positions):
        """Calibrate hand positions to robot workspace"""
        if left_hand_pos is not None:
            self.left_hand_offset = torch.tensor(left_hand_pos[0])  # Wrist position
        if right_hand_pos is not None:
            self.right_hand_offset = torch.tensor(right_hand_pos[0])
            
        # Calculate shoulder distance for scaling
        if shoulder_positions is not None and len(shoulder_positions) >= 12:
            left_shoulder = shoulder_positions[11]  # MediaPipe left shoulder index
            right_shoulder = shoulder_positions[12]  # MediaPipe right shoulder index
            self.shoulder_distance = np.linalg.norm(
                np.array(right_shoulder) - np.array(left_shoulder)
            )
            
        self.calibrated = True
        print("Calibration complete. Shoulder distance:", self.shoulder_distance)
        
    def retarget(self, hand_data: Dict[str, Any]) -> torch.Tensor:
        """
        Convert hand landmarks to OpenArm control commands
        
        Returns:
            torch.Tensor: [num_envs, 16] control commands
            [pos_L(3), quat_L(4), gripper_L(1), pos_R(3), quat_R(4), gripper_R(1)]
        """
        num_envs = 1  # Single teleoperation environment
        commands = torch.zeros((num_envs, 16))
        
        # Process left hand (controls left robot arm)
        if hand_data.get('left_hand') is not None:
            left_pos, left_quat, left_grip = self._process_hand(
                hand_data['left_hand'], is_left=True
            )
            
            # Apply low-pass filter for smooth movement
            left_pos = self.filter_alpha * left_pos + (1 - self.filter_alpha) * self.prev_left_pos
            left_quat = self.filter_alpha * left_quat + (1 - self.filter_alpha) * self.prev_left_quat
            
            commands[0, 0:3] = left_pos  # Left arm position
            commands[0, 3:7] = left_quat  # Left arm orientation
            commands[0, 7] = left_grip  # Left gripper
            
            self.prev_left_pos = left_pos
            self.prev_left_quat = left_quat
            
        # Process right hand (controls right robot arm)
        if hand_data.get('right_hand') is not None:
            right_pos, right_quat, right_grip = self._process_hand(
                hand_data['right_hand'], is_left=False
            )
            
            right_pos = self.filter_alpha * right_pos + (1 - self.filter_alpha) * self.prev_right_pos
            right_quat = self.filter_alpha * right_quat + (1 - self.filter_alpha) * self.prev_right_quat
            
            commands[0, 8:11] = right_pos  # Right arm position
            commands[0, 11:15] = right_quat  # Right arm orientation
            commands[0, 15] = right_grip  # Right gripper
            
            self.prev_right_pos = right_pos
            self.prev_right_quat = right_quat
            
        return commands
    
    def _process_hand(self, hand_landmarks: np.ndarray, is_left: bool):
        """Process individual hand landmarks"""
        if len(hand_landmarks) < 21:
            # Return default position if hand not fully detected
            default_pos = torch.tensor([0.3, 0.3 if is_left else -0.3, 0.2])
            default_quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
            return default_pos, default_quat, torch.tensor([0.0])
        
        # Wrist position (landmark 0)
        wrist_pos = torch.tensor(hand_landmarks[0])
        
        # Convert to robot workspace coordinates
        if self.calibrated:
            offset = self.left_hand_offset if is_left else self.right_hand_offset
            scaled_pos = (wrist_pos - torch.tensor(offset)) * self.workspace_scale
        else:
            scaled_pos = wrist_pos * self.workspace_scale
            
        # Position in robot base frame
        robot_pos = self.base_position + scaled_pos * torch.tensor([1.0, -1.0, 1.0])
        
        # Estimate hand orientation from palm landmarks
        quaternion = self._estimate_hand_orientation(hand_landmarks, is_left)
        
        # Detect pinch for gripper control
        gripper_value = self._detect_pinch(hand_landmarks)
        
        return robot_pos, quaternion, torch.tensor([gripper_value])
    
    def _estimate_hand_orientation(self, landmarks: np.ndarray, is_left: bool) -> torch.Tensor:
        """Estimate hand orientation from palm landmarks"""
        # Use palm center and finger bases to estimate orientation
        palm_center = torch.tensor(landmarks[9])  # Middle finger MCP
        wrist = torch.tensor(landmarks[0])
        
        # Vector from wrist to palm center (forward direction)
        forward = palm_center - wrist
        
        # Use thumb and pinky MCPs for side vector
        thumb_mcp = torch.tensor(landmarks[1])
        pinky_mcp = torch.tensor(landmarks[17])
        side = pinky_mcp - thumb_mcp if is_left else thumb_mcp - pinky_mcp
        
        # Normalize vectors
        forward_norm = forward / torch.norm(forward)
        side_norm = side / torch.norm(side)
        
        # Calculate up vector (cross product)
        up = torch.cross(side_norm, forward_norm)
        
        # Create rotation matrix and convert to quaternion
        rot_matrix = torch.stack([side_norm, up, forward_norm], dim=1)
        return self._matrix_to_quaternion(rot_matrix)
    
    def _detect_pinch(self, landmarks: np.ndarray) -> float:
        """Detect pinch gesture for gripper control (0=open, 1=closed)"""
        if len(landmarks) < 21:
            return 0.0
            
        # Thumb tip (4) and index tip (8) distance
        thumb_tip = torch.tensor(landmarks[4])
        index_tip = torch.tensor(landmarks[8])
        
        distance = torch.norm(thumb_tip - index_tip)
        
        # Normalize to 0-1 range (pinch detected when distance < 0.1)
        gripper_value = float(max(0.0, min(1.0, 1.0 - (distance / 0.1))))
        
        # Add hysteresis for stable grip
        if gripper_value > 0.7:
            return 1.0
        elif gripper_value < 0.3:
            return 0.0
        else:
            return gripper_value
    
    def _matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to quaternion"""
        # Simplified conversion - for production use torchquaternion or scipy
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        
        if trace > 0:
            S = torch.sqrt(trace + 1.0) * 2
            w = 0.25 * S
            x = (R[2, 1] - R[1, 2]) / S
            y = (R[0, 2] - R[2, 0]) / S
            z = (R[1, 0] - R[0, 1]) / S
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            S = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / S
            x = 0.25 * S
            y = (R[0, 1] + R[1, 0]) / S
            z = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / S
            x = (R[0, 1] + R[1, 0]) / S
            y = 0.25 * S
            z = (R[1, 2] + R[2, 1]) / S
        else:
            S = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / S
            x = (R[0, 2] + R[2, 0]) / S
            y = (R[1, 2] + R[2, 1]) / S
            z = 0.25 * S
            
        return torch.tensor([w, x, y, z])

# mediapipe_teleop_device.py (continued)
class MediaPipeTeleopDevice(DeviceBase):
    """
    Monolithic IsaacLab device for MediaPipe-based teleoperation.
    Combines camera capture, hand tracking, and robot control in one process.
    """
    
    def __init__(self, cfg: Dict[str, Any]):
        # Initialize retargeter
        retargeter_cfg = MediaPipeRetargeterCfg(cfg.retargeter)
        self.retargeter = HandPoseRetargeter(retargeter_cfg)
        super().__init__(retargeters=[self.retargeter])
        
        # MediaPipe processor
        self.processor = MediaPipeHolisticProcessor(
            camera_id=cfg.camera_id,
            image_width=cfg.image_width,
            image_height=cfg.image_height
        )
        
        # Control state
        self.enabled = False
        self.control_mode = "position"  # "position" or "velocity"
        self.safety_limits = {
            "max_position": torch.tensor([0.5, 0.5, 0.5]),
            "min_position": torch.tensor([-0.5, -0.5, 0.1]),
            "max_velocity": 1.0
        }
        
        # Gesture detection
        self.gesture_callbacks = {}
        self._setup_gesture_callbacks()
        
        # Calibration flag
        self.needs_calibration = True
        
    def initialize(self):
        """Initialize the device and start camera processing"""
        print("Initializing MediaPipe teleoperation device...")
        self.processor.start()
        print("Camera and MediaPipe processing started")
        
        # Wait for camera to initialize
        time.sleep(2.0)

    def add_callback(self, key, func):
        pass
        
    def advance(self, dt: float = 0.01) -> torch.Tensor:
        """
        Main update function called by IsaacLab simulation loop
        
        Args:
            dt: Time step in seconds
            
        Returns:
            torch.Tensor: Robot control commands
        """
        if not self.enabled:
            return torch.zeros((1, 16))
        
        # Get latest hand tracking data
        hand_data = self.processor.get_latest_data()
        
        # Auto-calibration on first valid data
        if self.needs_calibration:
            if hand_data.get('left_hand') is not None or hand_data.get('right_hand') is not None:
                self._perform_auto_calibration(hand_data)
                self.needs_calibration = False
        
        # Detect special gestures
        self._check_gestures(hand_data)
        
        # Get control commands from retargeter
        commands = self.retargeter.retarget(hand_data)
        
        # Apply safety limits
        commands = self._apply_safety_limits(commands)
        
        return commands
    
    def _perform_auto_calibration(self, hand_data: Dict[str, Any]):
        """Perform automatic calibration using T-pose detection"""
        print("Performing auto-calibration...")
        print("Assume T-pose with arms extended sideways")
        
        # Wait for stable detection
        time.sleep(3.0)
        
        # Get calibration data
        calib_data = self.processor.get_latest_data()
        
        # Calibrate retargeter
        self.retargeter.calibrate(
            left_hand_pos=calib_data.get('left_hand'),
            right_hand_pos=calib_data.get('right_hand'),
            shoulder_positions=calib_data.get('pose')
        )
        
        print("Auto-calibration complete!")
        self.enabled = True
        
    def _setup_gesture_callbacks(self):
        """Setup gesture-based callbacks"""
        self.gesture_callbacks = {
            'fist': self._emergency_stop,
            'open_palm': self._enable_control,
            'thumbs_up': self._toggle_control_mode,
            'peace': self._reset_arms,
        }
        
    def _check_gestures(self, hand_data: Dict[str, Any]):
        """Check for special control gestures"""
        left_hand = hand_data.get('left_hand')
        right_hand = hand_data.get('right_hand')
        
        # Check for fist gesture (emergency stop)
        if left_hand is not None and self._is_fist_gesture(left_hand):
            self.gesture_callbacks['fist']()
            
        # Check for open palm (enable control)
        if right_hand is not None and self._is_open_palm_gesture(right_hand):
            self.gesture_callbacks['open_palm']()
    
    def _is_fist_gesture(self, landmarks: np.ndarray) -> bool:
        """Detect fist gesture (all fingers curled)"""
        if len(landmarks) < 21:
            return False
            
        # Check distance from finger tips to palm center
        palm_center = landmarks[9]
        tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        
        distances = [np.linalg.norm(tip - palm_center) for tip in tips]
        avg_distance = np.mean(distances)
        
        return avg_distance < 0.1  # All tips close to palm
    
    def _is_open_palm_gesture(self, landmarks: np.ndarray) -> bool:
        """Detect open palm gesture (all fingers extended)"""
        if len(landmarks) < 21:
            return False
            
        # Check if fingers are extended
        tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        mcp_joints = [landmarks[1], landmarks[5], landmarks[9], landmarks[13], landmarks[17]]
        
        # Calculate extension angles (simplified)
        extended = 0
        for tip, mcp in zip(tips, mcp_joints):
            if tip[1] < mcp[1]:  # Tip is above MCP (in image coordinates)
                extended += 1
                
        return extended >= 4  # At least 4 fingers extended
    
    def _emergency_stop(self):
        """Emergency stop callback"""
        print("EMERGENCY STOP ACTIVATED!")
        self.enabled = False
        
    def _enable_control(self):
        """Enable teleoperation control"""
        self.enabled = True
        print("Teleoperation enabled")
        
    def _toggle_control_mode(self):
        """Toggle between position and velocity control"""
        self.control_mode = "velocity" if self.control_mode == "position" else "position"
        print(f"Control mode switched to: {self.control_mode}")
        
    def _reset_arms(self):
        """Reset arms to neutral position"""
        print("Resetting arms to neutral position")
        # Implementation depends on your robot controller
        
    def _apply_safety_limits(self, commands: torch.Tensor) -> torch.Tensor:
        """Apply safety limits to control commands"""
        # Limit position commands
        commands[0, 0:3] = torch.clamp(
            commands[0, 0:3],
            self.safety_limits["min_position"],
            self.safety_limits["max_position"]
        )
        commands[0, 8:11] = torch.clamp(
            commands[0, 8:11],
            self.safety_limits["min_position"],
            self.safety_limits["max_position"]
        )
        
        return commands
        
    def reset(self):
        """Reset device state"""
        self.enabled = False
        #self.retargeter.reset() #TODO: Check if this reset is really required.
        
    def shutdown(self):
        """Clean shutdown of the device"""
        print("Shutting down MediaPipe teleoperation device...")
        self.processor.stop()
        cv2.destroyAllWindows()

@dataclass
class MediaPipeRetargeterCfg(RetargeterCfg):
    workspace_scale: tuple[float] = (0.4, 0.4, 0.3)
    base_position: tuple[float] = (0.0, 0.0, 1.2)
    max_velocity: float = 0.6
    filter_alpha: float = 0.4
    left_arm_id: str = "left_arm"
    right_arm_id: str = "right_arm"
    retargeter_type: type[RetargeterBase] = HandPoseRetargeter

@dataclass
class MediaPipeTeleopDeviceConfig(DeviceCfg):
    camera_id: int = 0
    image_width: int = 640
    image_height: int = 480
    retargeter: MediaPipeRetargeterCfg | None = None
    class_type: type[DeviceBase] = MediaPipeTeleopDevice


@register_device
class MediaPipeArenaTeleopDevice(TeleopDeviceBase):
    name = "mediapipe"

    def get_teleop_device_cfg(self, embodiment = None):
        return DevicesCfg(
                devices=dict(
                    mediapipe=MediaPipeTeleopDeviceConfig(
                        sim_device=self.sim_device,
                        camera_id=0,
                        image_height=480,
                        image_width=640,
                        retargeter=dict(
                            retargeter_type=HandPoseRetargeter,
                            workspace_scale=(0.4, 0.4, 0.3),
                            base_position=(0.0, 0.0, 1.2),
                            max_velocity=0.6,
                            filter_alpha=0.4,
                            left_arm_id="left_arm",
                            right_arm_id="right_arm"
                        )
                    )
                )
        )
        '''return dict(
                class_type=MediaPipeTeleopDevice,
                camera_id=0,
                image_height=480,
                image_width=640,
                retargeter=dict(
                    retargeter_type=HandPoseRetargeter,
                    workspace_scale=(0.4, 0.4, 0.3),
                    base_position=(0.0, 0.0, 1.2),
                    max_velocity=0.6,
                    filter_alpha=0.4,
                    left_arm_id="left_arm",
                    right_arm_id="right_arm"
                )
            )'''