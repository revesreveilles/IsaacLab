"""Configuration classes for mobile manipulator actions."""

from dataclasses import MISSING
from typing import Literal

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from . import four_wheel_steer_actions


@configclass
class FourWheelFourSteerActionCfg(ActionTermCfg):
    """Configuration for a 4WD4WS base action term (GPU-optimized).

    Uses vectorized PyTorch controller for 10-15x performance improvement
    over the original numpy-based implementation.

    See :class:`FourWheelFourSteerAction` for details.
    """

    class_type: type[ActionTerm] = four_wheel_steer_actions.FourWheelFourSteerAction

    # ========== Required Parameters ==========
    # The target robot asset name in the scene
    asset_name: str = MISSING

    # Joint names (ordered [FL, FR, BL, BR])
    pivot_joint_names: tuple[str, str, str, str] = MISSING
    drive_joint_names: tuple[str, str, str, str] = MISSING

    # ========== Action Transformation ==========
    # Scaling fraction applied to max velocities [vx, vy, wz].
    # Effective scale = scale * [max_linear_velocity, max_linear_velocity, max_angular_velocity]
    # With default scale=(1,1,1): action=1 -> max_vel, action=-1 -> -max_vel
    # Raw actions from policy are always clamped to [-1, 1] before scaling.
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # ========== Frame Convention ==========
    # Whether the provided [vx, vy] are in world-frame (True) or body-frame (False)
    use_world_frame: bool = True

    # ========== Geometry Parameters ==========
    wheelbase: float = 0.6  # Distance between front and rear axles (m)
    track_width: float = 0.6  # Distance between left and right wheels (m)
    wheel_radii: tuple[float, float, float, float] | None = None  # [FL, FR, BL, BR] in meters
    wheel_positions: list[list[float]] | None = None  # [[x,y,z]*4] in body frame [FL, FR, BL, BR]

    # ========== Velocity Limits ==========
    # These define the actual velocity range the robot can reach.
    # action=1 -> max_linear_velocity m/s, action=-1 -> -max_linear_velocity m/s
    max_linear_velocity: float = 2.0  # m/s
    max_angular_velocity: float = 4.0  # rad/s
    max_steering_angle: float = 1.57  # rad (~90 degrees)
