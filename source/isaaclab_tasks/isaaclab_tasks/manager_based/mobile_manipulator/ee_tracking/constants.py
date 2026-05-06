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


# ==============================================================================
# Dynamic Obstacle Registry
# ==============================================================================
# Number per category
NUM_DYN_TOTAL = 60
NUM_DYN_CUBOIDS = int(NUM_DYN_TOTAL * 0.6)
NUM_DYN_CYLINDERS = NUM_DYN_TOTAL - NUM_DYN_CUBOIDS

# Width bins: [0.25, 0.50, 0.75, 1.0] m
_N_W = 4

# Auto-generated names
DYN_CUBOID_NAMES: tuple[str, ...] = tuple(
    f"dyn_cuboid_{i}" for i in range(NUM_DYN_CUBOIDS)
)
DYN_CYLINDER_NAMES: tuple[str, ...] = tuple(
    f"dyn_cylinder_{i}" for i in range(NUM_DYN_CYLINDERS)
)
DYN_ALL_NAMES: tuple[str, ...] = DYN_CUBOID_NAMES + DYN_CYLINDER_NAMES

# Size per obstacle: (width, width, height)
# Cuboids: 3D floating, height=0.3m, width from bins
CUBOID_SIZES: tuple[tuple[float, float, float], ...] = tuple(
    (0.25 * (i % _N_W + 1), 0.25 * (i % _N_W + 1), 0.3)
    for i in range(NUM_DYN_CUBOIDS)
)
# Cylinders: 2D ground pillars, height=1.8m, width from bins
CYLINDER_SIZES: tuple[tuple[float, float, float], ...] = tuple(
    (0.25 * (i % _N_W + 1), 0.25 * (i % _N_W + 1), 1.8)
    for i in range(NUM_DYN_CYLINDERS)
)
DYN_ALL_SIZES: tuple[tuple[float, float, float], ...] = CUBOID_SIZES + CYLINDER_SIZES

# Half-extents for reward computation (cuboid)
CUBOID_HALF_EXTENTS: tuple[tuple[float, float, float], ...] = tuple(
    (s[0] / 2, s[1] / 2, s[2] / 2) for s in CUBOID_SIZES
)
# (radius, half_height) for reward computation (cylinder)
CYLINDER_PARAMS: tuple[tuple[float, float], ...] = tuple(
    (s[0] / 2, s[2] / 2) for s in CYLINDER_SIZES
)

# Color palette for visual variety
_CUBOID_COLORS = [
    (1.0, 0.3, 0.3), (1.0, 0.5, 0.2), (1.0, 0.7, 0.1), (0.9, 0.9, 0.0),
    (1.0, 0.4, 0.4), (1.0, 0.6, 0.3), (1.0, 0.8, 0.2), (0.9, 0.8, 0.1),
]
_CYLINDER_COLORS = [
    (0.2, 0.3, 1.0), (0.1, 0.1, 0.8), (0.3, 0.5, 1.0), (0.2, 0.2, 0.9),
    (0.25, 0.35, 0.95), (0.15, 0.15, 0.85), (0.35, 0.55, 0.95), (0.25, 0.25, 0.85),
]
