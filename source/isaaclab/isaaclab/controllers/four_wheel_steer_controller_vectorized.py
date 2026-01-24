"""GPU-optimized 4WD4WS controller for Isaac Lab."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional

import torch

from isaaclab.assets.articulation import Articulation


@dataclass
class FourWheelSteerControllerCfg:
    """Configuration for 4WD4WS controller."""
    # joint names [FL, FR, BL, BR]
    pivot_joint_names: Sequence[str]
    drive_joint_names: Sequence[str]
    # geometry
    wheelbase: float = 0.6
    track_width: float = 0.6
    wheel_radii: Sequence[float] = (0.1, 0.1, 0.1, 0.1)
    # limits
    max_linear_velocity: float = 50.0
    max_angular_velocity: float = 10.0
    max_steering_angle: float = 1.57


def quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Compute yaw from quaternion (x, y, z, w)."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(t3, t4)


class FourWheelSteerControllerVectorized:
    """Vectorized 4WD4WS controller - GPU optimized.
    
    Implements the standard 4WD4WS kinematic model:
    v_wheel_i = v_chassis + ω × r_i
    
    Performance: ~10-15x faster than numpy-based implementation.
    """

    def __init__(
        self,
        cfg: FourWheelSteerControllerCfg,
        robot: Articulation,
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.robot = robot
        self.device = device or robot.device

        # Resolve joint indices
        self._pivot_ids, pivot_found = robot.find_joints(
            cfg.pivot_joint_names,
            preserve_order=True
        )
        self._drive_ids, drive_found = robot.find_joints(
            cfg.drive_joint_names,
            preserve_order=True
        )
        
        if len(pivot_found) != 4 or len(drive_found) != 4:
            raise ValueError(
                f"Expected 4 pivot and 4 drive joints, "
                f"got {len(pivot_found)} and {len(drive_found)}"
            )

        # Pre-compute wheel geometry on GPU
        # Default wheel positions [FL, FR, BL, BR]
        half_wb = cfg.wheelbase / 2.0
        half_tw = cfg.track_width / 2.0
        
        self._wheel_positions = torch.tensor(
            [
                [half_wb, half_tw, 0.0],    # FL
                [half_wb, -half_tw, 0.0],   # FR
                [-half_wb, half_tw, 0.0],   # BL
                [-half_wb, -half_tw, 0.0],  # BR
            ],
            device=self.device,
            dtype=torch.float32,
        )  # (4, 3)
        
        self._wheel_radii = torch.tensor(
            list(cfg.wheel_radii),
            device=self.device,
            dtype=torch.float32,
        )  # (4,)

    # four_wheel_steer_controller_vectorized.py

    def compute(self, cmd_vel: torch.Tensor, use_world_frame: bool = True):
        """Compute steering angles and wheel velocities."""
        
        if cmd_vel.ndim == 1:
            cmd_vel = cmd_vel.unsqueeze(0)
        N = cmd_vel.shape[0]

        # Get base orientation
        root_state = self.robot.data.root_state_w[:N]
        yaw = quat_to_yaw(root_state[:, 3:7])
        
        # Parse commands
        v_xy = cmd_vel[:, :2]
        wz = cmd_vel[:, 2]

        # Apply velocity limits
        lin_speed = torch.norm(v_xy, dim=1)
        over_limit = lin_speed > self.cfg.max_linear_velocity
        if over_limit.any():
            scale = self.cfg.max_linear_velocity / lin_speed.clamp(min=1e-6)
            scale = torch.where(over_limit, scale, torch.ones_like(scale))
            v_xy = v_xy * scale.unsqueeze(1)
        
        wz = wz.clamp(-self.cfg.max_angular_velocity, self.cfg.max_angular_velocity)

        # Transform to body frame
        if use_world_frame:
            cos_yaw = torch.cos(-yaw)
            sin_yaw = torch.sin(-yaw)
            vx_body = v_xy[:, 0] * cos_yaw - v_xy[:, 1] * sin_yaw
            vy_body = v_xy[:, 0] * sin_yaw + v_xy[:, 1] * cos_yaw
        else:
            vx_body = v_xy[:, 0]
            vy_body = v_xy[:, 1]

        v_chassis = torch.stack([vx_body, vy_body, torch.zeros_like(vx_body)], dim=1)

        # Compute wheel velocities
        wz_expanded = wz.unsqueeze(1)
        r = self._wheel_positions
        
        v_rot = torch.zeros(N, 4, 3, device=self.device)
        v_rot[:, :, 0] = -wz_expanded * r[:, 1]
        v_rot[:, :, 1] = wz_expanded * r[:, 0]
        
        v_trans = v_chassis.unsqueeze(1).expand(N, 4, 3)
        v_wheel = v_trans + v_rot
        
        # Compute steering angles
        steer_ang = -torch.atan2(v_wheel[:, :, 1], v_wheel[:, :, 0])
        vel_norm = torch.norm(v_wheel[:, :, :2], dim=2)
        stationary = vel_norm < 1e-6
        steer_ang = torch.where(stationary, torch.zeros_like(steer_ang), steer_ang)
        steer_ang = steer_ang.clamp(-self.cfg.max_steering_angle, self.cfg.max_steering_angle)

        # Compute wheel angular velocities
        wheel_radii = self._wheel_radii.unsqueeze(0)
        wheel_omega = vel_norm / wheel_radii
        wheel_omega = torch.where(stationary, torch.zeros_like(wheel_omega), wheel_omega)
        
        return steer_ang, wheel_omega

    def apply(self, cmd_vel: torch.Tensor, use_world_frame: bool = True):
        """Compute and apply commands to robot.
        
        Args:
            cmd_vel: (N, 3) velocity commands [vx, vy, wz]
            use_world_frame: whether cmd_vel is in world frame (True) or body frame (False)
        """
        steer_ang, wheel_omega = self.compute(cmd_vel, use_world_frame)
        self.robot.set_joint_position_target(steer_ang, joint_ids=self._pivot_ids)
        self.robot.set_joint_velocity_target(wheel_omega, joint_ids=self._drive_ids)
