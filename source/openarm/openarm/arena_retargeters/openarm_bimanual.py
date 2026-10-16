from collections.abc import Callable

from isaaclab_arena.assets.register import register_retargeter
from isaaclab_arena.assets.retargeter_library import RetargetterBase


def _build_openarm_pipeline():
    """Build an IsaacTeleop retargeting pipeline for GR1T2 pick-place teleoperation.

    Creates two Se3AbsRetargeters for left and right wrist pose tracking and
    two DexHandRetargeters for left and right dexterous hand finger control
    from hand tracking data. All outputs are flattened into a single action
    tensor via TensorReorderer.
    """
    #from isaacteleop.retargeters import (
    #    DexHandRetargeter,
    #    DexHandRetargeterConfig,
    #    Se3AbsRetargeter,
    #    Se3RetargeterConfig,
    #    TensorReorderer,
    #)
    from isaacteleop.retargeters.gripper_retargeter import GripperRetargeter, GripperRetargeterConfig
    from isaacteleop.retargeters.se3_retargeter import Se3AbsRetargeter, Se3RetargeterConfig
    from isaacteleop.retargeters.tensor_reorderer import TensorReorderer
    from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource, HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    # Create input sources (trackers are auto-discovered from pipeline)
    controllers = ControllersSource(name="controllers")

    # External input: world-to-anchor 4x4 transform matrix provided by IsaacTeleopDevice
    transform_input = ValueInput("world_T_anchor", TransformMatrix())

    # Apply the coordinate-frame transform to controller poses so that
    # downstream retargeters receive data in the simulation world frame.
    transformed_controllers = controllers.transformed(transform_input.output(ValueInput.VALUE))
    #transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # -------------------------------------------------------------------------
    # SE3 Absolute Pose Retargeters (left and right wrists)
    # -------------------------------------------------------------------------
    # Left wrist: identity rotation offset (passes through as-is in original retargeter)
    left_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=0.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT),
        }
    )

    # Right wrist: 180-degree Z rotation offset
    # From GR1T2Retargeter._retarget_abs: the USD control frame is 180 degrees
    # rotated around the Z axis w.r.t. the OpenXR frame.
    right_se3_cfg = Se3RetargeterConfig(
        input_device=ControllersSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=False,
        use_wrist_position=False,
        target_offset_roll=0.0,
        target_offset_pitch=0.0,
        target_offset_yaw=180.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT),
        }
    )

    left_gripper_cfg = GripperRetargeterConfig(
        hand_side="left"
    )
    left_gripper = GripperRetargeter(left_gripper_cfg, name="left_gripper")
    connected_left_gripper = left_gripper.connect(
        {
            ControllersSource.LEFT: transformed_controllers.output(ControllersSource.LEFT)
        }
    )

    right_gripper_cfg = GripperRetargeterConfig(
        hand_side="right"
    )
    right_gripper = GripperRetargeter(right_gripper_cfg, name="right_gripper")
    connected_right_gripper = right_gripper.connect(
        {
            ControllersSource.RIGHT: transformed_controllers.output(ControllersSource.RIGHT)
        }
    )

    # -------------------------------------------------------------------------
    # TensorReorderer: flatten into a 16D action tensor
    # -------------------------------------------------------------------------
    # Se3AbsRetargeter outputs 7D arrays: [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
    left_ee_elements = ["l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w"]
    right_ee_elements = ["r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w"]

    left_grip_elements = ['l_grip']
    right_grip_elements = ['r_grip']
    # Output order must match the PinkInverseKinematicsActionCfg expected tensor layout:
    #   [left_wrist(7), right_wrist(7), hand_joints(22)]
    # Hand joints follow hand_joint_names order from ActionsCfg.upper_body_ik.
    output_order = (
        left_ee_elements
        + left_grip_elements
        + right_ee_elements
        + right_grip_elements
    )

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": left_ee_elements,
            "right_ee_pose": right_ee_elements,
            "left_gripper": left_grip_elements,
            "right_gripper": right_grip_elements,
        },
        output_order=output_order,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_gripper": "scalar",
            "right_gripper": "scalar",
        },
    )
    connected_reorderer = reorderer.connect(
        {
            "left_ee_pose": connected_left_se3.output("ee_pose"),
            "right_ee_pose": connected_right_se3.output("ee_pose"),
            "left_gripper": connected_left_gripper.output("gripper_command"),
            "right_gripper": connected_right_gripper.output("gripper_command"),
        }
    )

    pipeline = OutputCombiner({"action": connected_reorderer.output("output")})
    return pipeline #, [left_gripper, right_gripper]

@register_retargeter
class BimanualOpenArmKeyboardRertargetter(RetargetterBase):
    device = "keyboard"
    embodiment = "openarm_bimanual"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class BimanualOpenArmKeyboardBimanualRetargetter(RetargetterBase):
    device = "keyboard_bimanual"
    embodiment = "openarm_bimanual"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class BimanualOpenArmKeyboardOpenArmRetargetter(RetargetterBase):
    device = "keyboard__openarm_bimanual"
    embodiment = "openarm_bimanual"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None


@register_retargeter
class BimanualOpenArmMediaPipeRetargetter(RetargetterBase):
    device = "mediapipe"
    embodiment = "openarm_bimanual"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return None
    

@register_retargeter
class BimanualOpenArmOpenXRRetargetter(RetargetterBase):
    device = "openxr"
    embodiment = "openarm_bimanual"

    def get_pipeline_builder(self, embodiment: object) -> Callable | None:
        return _build_openarm_pipeline

