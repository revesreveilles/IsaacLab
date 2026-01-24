# constants.py
"""Constants for mobile manipulator tasks."""
from __future__ import annotations
import math
from isaaclab.managers import SceneEntityCfg

# UR3 Arm joint names
UR_ARM_JOINT_NAMES = (
    "ur_shoulder_pan_joint",
    "ur_shoulder_lift_joint",
    "ur_elbow_joint",
    "ur_wrist_1_joint",
    "ur_wrist_2_joint",
    "ur_wrist_3_joint",
)

# SceneEntityCfg helper
UR_ARM_JOINT_CFG = SceneEntityCfg("robot", joint_names=list(UR_ARM_JOINT_NAMES))

# Wheel joint names
WHEEL_PIVOT_JOINTS = (
    "left_forward_pivot_joint",
    "right_forward_pivot_joint",
    "left_back_pivot_joint",
    "right_back_pivot_joint",
)

WHEEL_DRIVE_JOINTS = (
    "left_forward_drive_joint",
    "right_forward_drive_joint",
    "left_back_drive_joint",
    "right_back_drive_joint",
)

# Physics
PI = math.pi
