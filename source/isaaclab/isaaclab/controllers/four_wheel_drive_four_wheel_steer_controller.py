"""
Lightweight 4WD4WS controller without Isaac Sim dependencies.

This module provides a minimal FourWheelDriveFourWheelSteerController that matches
the expected interface used by the adapter in isaaclab (forward(command) -> object
with joint_positions and joint_velocities). Implementation follows the semantics
of the Isaac Sim 5.0 4WD4WS controller:

- Inputs to forward: [chassis_linear_vel(vector3 or float vx),
                      chassis_angular_vel(vector3 or float wz),
                      yaw(float), dt(float)]
- Outputs: object with attributes:
        - joint_velocities: (FL, FR, BL, BR) wheel angular speeds in rad/s
        - joint_positions:  (FL, FR, BL, BR) steering angles in rad

Notes:
- This controller directly computes wheel angular velocities (rad/s) from
  linear velocities (m/s) using: ω_wheel = v_linear / wheel_radius,
  matching Isaac Sim 5.0 behavior.

Conventions:
- Wheel order: [front_left, front_right, rear_left, rear_right] (FL, FR, BL, BR)
- Coordinate frame: x-forward, y-left, z-up.
- For world-frame linear velocity inputs, the adapter passes yaw and we rotate
  v by R(-yaw) to body frame before computing wheel states.

This implementation is derived from NVIDIA Isaac Sim 5.0 examples and updated
to match the correct wheel velocity calculation method documented in the
Isaac Sim 5.0 controller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class _FWDSAction:
    joint_velocities: Tuple[float, float, float, float]
    joint_positions: Tuple[float, float, float, float]


class FourWheelDriveFourWheelSteerController:
    def __init__(
        self,
        name: str,
        wheel_base: float,
        track_width: float,
        wheel_radii: Sequence[float],
        wheel_positions: Optional[Sequence[Sequence[float]]] = None,
        max_wheel_velocity: float = 0.0,
        max_steering_angle: float = np.pi,
        max_acceleration: float = 0.0,  # not used in lightweight impl
        max_steering_velocity: float = 0.0,
    ) -> None:
        self._name = name
        self._wheel_base = float(wheel_base)
        self._track_width = float(track_width)
        self._half_wheel_base = self._wheel_base / 2.0
        self._half_track_width = self._track_width / 2.0

        # wheel radii
        wr = np.asarray(list(wheel_radii), dtype=float).reshape(4)
        self._wheel_radii = wr
        self._wheel_radius = float(np.mean(wr))

        # wheel positions (FL, FR, BL, BR)
        if wheel_positions is not None:
            self._wheel_positions = np.asarray(wheel_positions, dtype=float).reshape(4, 3)
        else:
            self._wheel_positions = np.asarray(
                [
                    [self._half_wheel_base, self._half_track_width, 0.0],   # FL
                    [self._half_wheel_base, -self._half_track_width, 0.0],  # FR
                    [-self._half_wheel_base, self._half_track_width, 0.0],  # BL
                    [-self._half_wheel_base, -self._half_track_width, 0.0],  # BR
                ],
                dtype=float,
            )

        # limits (0.0 interpreted as infinity where applicable)
        self._max_wheel_velocity = float(max_wheel_velocity)
        self._max_steering_angle = float(max_steering_angle)
        self._max_acceleration = float(max_acceleration)
        self._max_steering_velocity = float(max_steering_velocity)

    def forward(self, command) -> _FWDSAction:
        """Compute wheel velocities and steering angles for 4WD4WS given body/world twist.

        Args:
            command: [chassis_linear_vel(vector3 or float vx),
                      chassis_angular_vel(vector3 or float wz),
                      yaw(float), dt(float)]
        Returns:
            _FWDSAction with joint_velocities (FL, FR, BL, BR) and joint_positions (FL, FR, BL, BR).
        """
        # parse command
        if command is None or len(command) < 4:
            raise ValueError(
                "4WD4WS Controller expects [chassis_linear_vel, chassis_angular_vel, yaw, dt]"
            )

        lin = command[0]
        ang = command[1]
        yaw = float(command[2])
        # dt parsed but not used
        _ = float(command[3])

        # convert inputs to arrays
        if isinstance(lin, (list, tuple, np.ndarray)):
            v = np.asarray(lin, dtype=float).reshape(3)
        else:
            v = np.asarray([float(lin), 0.0, 0.0], dtype=float)

        if isinstance(ang, (list, tuple, np.ndarray)):
            w = np.asarray(ang, dtype=float).reshape(3)
        else:
            w = np.asarray([0.0, 0.0, float(ang)], dtype=float)

        # interpret zeros as infinity for limits to match common Isaac conventions
        max_wheel_vel = np.inf if self._max_wheel_velocity == 0.0 else self._max_wheel_velocity
        max_steer_ang = np.inf if self._max_steering_angle == 0.0 else self._max_steering_angle
        max_steer_vel = np.inf if self._max_steering_velocity == 0.0 else self._max_steering_velocity

        # clamp angular z to steering velocity limit
        w[2] = float(np.clip(w[2], -max_steer_vel, max_steer_vel))

        # compute max linear speed from mean wheel radius
        max_linear_speed = float(abs(max_wheel_vel * self._wheel_radius))
        lin_speed = float(np.linalg.norm(v[:2]))
        if lin_speed > max_linear_speed and max_linear_speed > 0.0 and not np.isinf(max_linear_speed):
            scale = max_linear_speed / lin_speed
            v[:2] *= scale

        # rotate world-frame velocity to body frame using R(-yaw)
        cy = np.cos(-yaw)
        sy = np.sin(-yaw)
        R = np.array([[cy, -sy], [sy, cy]], dtype=float)
        v_xy_body = R @ v[:2]
        v_body = np.array([v_xy_body[0], v_xy_body[1], 0.0], dtype=float)

        # compute per-wheel velocities and steering angles
        wheel_vel = np.zeros(4, dtype=float)
        steer_ang = np.zeros(4, dtype=float)
        w_vec = np.array([0.0, 0.0, w[2]], dtype=float)

        for i in range(4):
            r = self._wheel_positions[i]
            w_cross_r = np.cross(w_vec, r)
            vel = v_body + w_cross_r
            vel_norm = float(np.linalg.norm(vel))
            if vel_norm < 1e-6:
                wheel_vel[i] = 0.0
                steer_ang[i] = 0.0
            else:
                # Convert linear velocity (m/s) to angular velocity (rad/s)
                # wheel_angular_velocity = linear_velocity / wheel_radius
                wheel_vel[i] = vel_norm / self._wheel_radii[i]
                # negative sign matches adapter's convention (Isaac Sim)
                steer_ang[i] = -float(np.arctan2(vel[1], vel[0]))

        # apply limits
        if not np.isinf(max_wheel_vel):
            wheel_vel = np.clip(wheel_vel, -max_wheel_vel, max_wheel_vel)
        if not np.isinf(max_steer_ang):
            steer_ang = np.clip(steer_ang, -max_steer_ang, max_steer_ang)

        return _FWDSAction(
            joint_velocities=(wheel_vel[0], wheel_vel[1], wheel_vel[2], wheel_vel[3]),
            joint_positions=(steer_ang[0], steer_ang[1], steer_ang[2], steer_ang[3]),
        )
