"""
Adapter to use a 4WD4WS controller inside Isaac Lab.

This wraps a FourWheelDriveFourWheelSteerController and applies its outputs
to an Articulation via position targets for pivot joints and velocity targets
for drive joints.

Contract:
- Inputs per step: cmd body twist [vx, vy, wz] in world or body frame,
  dt, and current base yaw.
- Outputs applied: 4 pivot joint positions (FL, FR, BL, BR),
  4 drive joint velocities (FL, FR, BL, BR) in rad/s.

Notes:
- The adapter assumes pivot_names and drive_names are ordered
  [FL, FR, BL, BR].
- Yaw is computed from the articulation root state.
- Controller outputs wheel angular velocities (rad/s) directly,
  matching Isaac Sim 5.0 behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, cast

import torch
import numpy as np

from isaaclab.assets.articulation import Articulation


@dataclass
class FourWSAdapterCfg:
    # joint names (must be ordered [FL, FR, BL, BR])
    pivot_joint_names: Sequence[str]
    drive_joint_names: Sequence[str]
    # geometry (optional; if None, use controller's internal defaults)
    wheelbase: Optional[float] = 0.6
    track_width: Optional[float] = 0.6
    wheel_radii: Optional[Sequence[float]] = None  # [FL, FR, BL, BR]
    # optional explicit wheel positions in body frame (x-forward, y-left, z-up),
    # ordered as [FL, FR, BL, BR]; shape (4,3)
    wheel_positions: Optional[Sequence[Sequence[float]]] = None
    # limits (optional)
    max_wheel_velocity: float = 0.0
    max_steering_angle: float = 0.0
    max_steering_velocity: float = 0.0


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Compute yaw (Z) from quaternion [w, x, y, z] or [x, y, z, w].

    Accepts (..., 4) tensor. Returns (...,) yaw in radians.
    """
    # Try to detect layout: if last dim ~1 at index 0 often w-first; but be explicit:
    # Isaac Lab stores root_state as [..., 0:3]=pos, 3:7=orn (x,y,z,w).
    # We will expect (x, y, z, w).
    x = q[..., 0]
    y = q[..., 1]
    z = q[..., 2]
    w = q[..., 3]
    # yaw from quaternion (x, y, z, w)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(t3, t4)


class FourWheelSteerAdapter:
    """Thin wrapper to call user's 4WD4WS controller and apply to an Articulation.

    Usage:
        adapter = FourWheelSteerAdapter(robot, cfg)
        adapter.apply(cmd_vxvywz)  # (N,3) tensor in m/s, rad/s
    """

    def __init__(self, robot: Articulation, cfg: FourWSAdapterCfg, device: Optional[str] = None):
        self.robot = robot
        self.cfg = cfg
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        # Resolve joint ids in the provided FL/FR/BL/BR order
        self._pivot_ids, pivot_found = self.robot.find_joints(cfg.pivot_joint_names)
        self._drive_ids, drive_found = self.robot.find_joints(cfg.drive_joint_names)
        if len(pivot_found) != 4 or len(drive_found) != 4:
            raise RuntimeError(
                f"Expected 4 pivot and 4 drive joints, got {len(pivot_found)} and {len(drive_found)}.\n"
                f"Pivot found: {pivot_found}\nDrive found: {drive_found}"
            )

        # Import the lightweight controller (no Isaac Sim dependencies)
        from isaaclab.controllers.four_wheel_drive_four_wheel_steer_controller import (
            FourWheelDriveFourWheelSteerController,
        )

        # Provide fallback geometry and limits if not specified
        wb = float(cfg.wheelbase) if cfg.wheelbase is not None else 0.6
        tw = float(cfg.track_width) if cfg.track_width is not None else 0.6
        radii_seq = cfg.wheel_radii if cfg.wheel_radii is not None else (0.1, 0.1, 0.1, 0.1)
        wheel_radii_np = np.asarray(list(radii_seq), dtype=float)

        # Resolve wheel positions: explicit or computed from geometry
        if cfg.wheel_positions is not None:
            wheel_positions_np = np.asarray(cfg.wheel_positions, dtype=float)
        else:
            # Default positions: FL, FR, BL, BR
            wheel_positions_np = np.asarray(
                [
                    [wb / 2.0, tw / 2.0, 0.0],    # front left
                    [wb / 2.0, -tw / 2.0, 0.0],   # front right
                    [-wb / 2.0, tw / 2.0, 0.0],   # back left
                    [-wb / 2.0, -tw / 2.0, 0.0],  # back right
                ],
                dtype=float,
            )

        # Limits fallbacks (generous defaults)
        max_wheel_velocity = cfg.max_wheel_velocity if cfg.max_wheel_velocity > 0.0 else 50.0
        max_steering_angle = cfg.max_steering_angle if cfg.max_steering_angle > 0.0 else 0.8
        max_steering_velocity = cfg.max_steering_velocity if cfg.max_steering_velocity > 0.0 else 3.0

        self.ctrl = FourWheelDriveFourWheelSteerController(
            name="4wd4ws",
            wheel_base=wb,
            track_width=tw,
            wheel_radii=wheel_radii_np.tolist(),
            wheel_positions=wheel_positions_np.tolist(),
            max_wheel_velocity=max_wheel_velocity,
            max_steering_angle=max_steering_angle,
            max_acceleration=0.0,
            max_steering_velocity=max_steering_velocity,
        )

    def apply(
        self, cmd_vxvywz: torch.Tensor, dt: float, use_world_cmd: bool = True
    ):
        """Apply base command(s) to robot.

        Args:
            cmd_vxvywz: (N,3) tensor [vx, vy, wz] in world frame if
                use_world_cmd=True, else body frame.
            dt: step time in seconds.
            use_world_cmd: when True, [vx, vy] are in world frame;
                adapter rotates to body via -yaw.
        """
        if cmd_vxvywz.ndim == 1:
            cmd_vxvywz = cmd_vxvywz.unsqueeze(0)
        N = cmd_vxvywz.shape[0]

        # Get yaw from root state (x,y,z, qw?) -> In Isaac Lab: root_state_w[..., 3:7] = (x,y,z,w)
        root_state = self.robot.data.root_state_w  # (num_envs, 13)
        q_xyzw = root_state[:, 3:7]
        yaw_all = _quat_to_yaw(q_xyzw)

        # Broadcast or slice first N envs
        yaw = yaw_all[:N]

        # Prepare outputs (FL, FR, BL, BR) on same device as robot
        out_device = root_state.device
        steer_targets = torch.zeros((N, 4), device=out_device)
        omega_targets = torch.zeros((N, 4), device=out_device)

        # Loop per env (controller uses numpy per-call)
        for i in range(N):
            vx, vy, wz = cmd_vxvywz[i].tolist()
            # Controller rotates by -yaw internally if world-frame
            lin = [vx, vy, 0.0]
            ang = [0.0, 0.0, wz]
            yaw_i = float(yaw[i].item()) if use_world_cmd else 0.0
            act = self.ctrl.forward([lin, ang, yaw_i, dt])
            # Robustly extract fields from controller output
            js = getattr(act, "joint_positions", None)
            jv = getattr(act, "joint_velocities", None)
            if js is None or jv is None:
                raise RuntimeError(
                    "4WD4WS controller missing joint_positions/"
                    f"joint_velocities; got: pos={type(js)}, "
                    f"vel={type(jv)}"
                )
            js = cast(Sequence[float], js)
            jv = cast(Sequence[float], jv)

            # The controller uses FL, FR, BL, BR ordering
            steer_targets[i, 0] = js[0]
            steer_targets[i, 1] = js[1]
            steer_targets[i, 2] = js[2]
            steer_targets[i, 3] = js[3]

            # Controller outputs wheel angular velocities in rad/s
            # (Isaac Sim 5.0 aligned) - no conversion needed
            omega_targets[i, 0] = jv[0]
            omega_targets[i, 1] = jv[1]
            omega_targets[i, 2] = jv[2]
            omega_targets[i, 3] = jv[3]

        # Apply to articulation
        self.robot.set_joint_position_target(
            steer_targets, joint_ids=self._pivot_ids
        )
        self.robot.set_joint_velocity_target(
            omega_targets, joint_ids=self._drive_ids
        )
