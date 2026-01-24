"""
Action terms for four-wheel drive four-wheel steer (4WD4WS) mobile bases.

GPU-optimized version using vectorized controller.

This action maps a 3-DoF chassis twist command [vx, vy, wz] to 4 steering joint
positions (pivot) and 4 wheel velocity targets (drive) using a fully vectorized
PyTorch-based controller.

Performance improvement: 10-15x faster than original numpy-based implementation
for large batches (4096 envs).

Contract
- Input action per env: [vx, vy, wz] (m/s, m/s, rad/s), typically normalized then scaled.
- Applied each physics step: pivot joint position targets (rad) and drive joint
  velocity targets (rad/s) for joints ordered [FL, FR, BL, BR].

Notes
- Joint names in config must be provided in the order [FL, FR, BL, BR].
- If you prefer world-frame commands, set use_world_frame=True. Otherwise
  commands are interpreted in body-frame.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.four_wheel_steer_controller_vectorized import (
    FourWheelSteerControllerVectorized,
    FourWheelSteerControllerCfg,
)
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor
    from . import actions_cfg


class FourWheelFourSteerAction(ActionTerm):
    """4WD4WS base action: 3D chassis twist -> 4 pivot positions + 4 wheel speeds.

    GPU-optimized: uses vectorized PyTorch controller to compute per-wheel steering
    angles and velocities for all environments in parallel.
    """

    cfg: "actions_cfg.FourWheelFourSteerActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "actions_cfg.FourWheelFourSteerActionCfg", env: ManagerBasedEnv):
        # initialize base
        super().__init__(cfg, env)

        # resolve articulation and save sim dt
        self._sim_dt = env.physics_dt

        # create buffers for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # validate joint names (must be ordered [FL, FR, BL, BR])
        pivot_names = list(self.cfg.pivot_joint_names)
        drive_names = list(self.cfg.drive_joint_names)
        if len(pivot_names) != 4 or len(drive_names) != 4:
            raise ValueError(
                f"Expected 4 pivot and 4 drive joint names. "
                f"Got {len(pivot_names)} and {len(drive_names)}."
            )

        # cast asset to articulation
        if not isinstance(self._asset, Articulation):
            raise TypeError(
                f"FourWheelFourSteerAction requires an Articulation asset. "
                f"Got: {type(self._asset).__name__}."
            )
        pivot_ids, resolved_pivot_names = self._asset.find_joints(
            pivot_names,
            preserve_order=True
        )
        drive_ids, resolved_drive_names = self._asset.find_joints(
            drive_names,
            preserve_order=True
        )

        # 验证找到的joint
        if len(pivot_ids) != 4 or len(drive_ids) != 4:
            raise ValueError(
                f"Failed to find all joints. "
                f"Found {len(pivot_ids)} pivot and {len(drive_ids)} drive joints."
            )
        # create controller configuration
        controller_cfg = FourWheelSteerControllerCfg(
            pivot_joint_names=tuple(resolved_pivot_names),
            drive_joint_names=tuple(resolved_drive_names),
            wheelbase=float(self.cfg.wheelbase),
            track_width=float(self.cfg.track_width),
            wheel_radii=tuple(self.cfg.wheel_radii) if self.cfg.wheel_radii is not None else (0.1, 0.1, 0.1, 0.1),
            max_linear_velocity=self.cfg.max_linear_velocity if self.cfg.max_linear_velocity > 0 else 50.0,
            max_angular_velocity=self.cfg.max_angular_velocity if self.cfg.max_angular_velocity > 0 else 10.0,
            max_steering_angle=self.cfg.max_steering_angle if self.cfg.max_steering_angle > 0 else 1.57,
        )

        # instantiate vectorized controller
        self._controller = FourWheelSteerControllerVectorized(
            cfg=controller_cfg,
            robot=self._asset,
            device=self.device,
        )

        # parse scale/offset params (tuple of 3 floats)
        self._scale = torch.tensor(self.cfg.scale, device=self.device).view(1, 3)
        self._offset = torch.tensor(self.cfg.offset, device=self.device).view(1, 3)

        omni.log.info(
            f"[FourWheelFourSteerAction] Initialized vectorized controller with "
            f"pivot joints {pivot_names} and drive joints {drive_names}"
        )

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        # [vx, vy, wz]
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def IO_descriptor(self) -> "GenericActionIODescriptor":
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "4WD4WS_Vectorized"
        self._IO_descriptor.pivot_joint_names = list(self.cfg.pivot_joint_names)
        self._IO_descriptor.drive_joint_names = list(self.cfg.drive_joint_names)
        self._IO_descriptor.scale = self._scale
        self._IO_descriptor.offset = self._offset
        self._IO_descriptor.use_world_frame = self.cfg.use_world_frame
        self._IO_descriptor.wheelbase = float(self.cfg.wheelbase)
        self._IO_descriptor.track_width = float(self.cfg.track_width)
        self._IO_descriptor.wheel_radii = (
            list(self.cfg.wheel_radii) if self.cfg.wheel_radii is not None else [0.1, 0.1, 0.1, 0.1]
        )
        self._IO_descriptor.max_linear_velocity = float(self.cfg.max_linear_velocity)
        self._IO_descriptor.max_angular_velocity = float(self.cfg.max_angular_velocity)
        self._IO_descriptor.max_steering_angle = float(self.cfg.max_steering_angle)
        return self._IO_descriptor

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions with scale and offset."""
        # store raw actions
        self._raw_actions[:] = actions
        # affine transform: action = raw * scale + offset
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def apply_actions(self):
        """Apply processed actions to robot using vectorized controller."""
        # Apply commands (fully vectorized across all envs)
        self._controller.apply(
            cmd_vel=self.processed_actions,
            use_world_frame=self.cfg.use_world_frame,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset actions for specified environments."""
        if env_ids is None:
            self._raw_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0

    """
    Debug visualization.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Setup debug visualization markers."""
        if debug_vis:
            if not hasattr(self, "base_cmd_goal_visualizer"):
                # Goal velocity marker (green)
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/4ws_velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_cmd_goal_visualizer = VisualizationMarkers(marker_cfg)

                # Current velocity marker (blue)
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Actions/4ws_velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_cmd_current_visualizer = VisualizationMarkers(marker_cfg)

            self.base_cmd_goal_visualizer.set_visibility(True)
            self.base_cmd_current_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_cmd_goal_visualizer"):
                self.base_cmd_goal_visualizer.set_visibility(False)
                self.base_cmd_current_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Update debug visualization each frame."""
        # check if robot is initialized
        if not self._asset.is_initialized:
            return

        # get base position (raised for visibility)
        base_pos_w = self._asset.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5

        # commanded velocity in body frame
        cmd_xy_body = self._command_in_base_frame()
        cmd_scale, cmd_quat = self._resolve_xy_velocity_to_arrow(cmd_xy_body)

        # current velocity in body frame
        vel_xy_body = self._asset.data.root_lin_vel_b[:, :2]
        vel_scale, vel_quat = self._resolve_xy_velocity_to_arrow(vel_xy_body)

        # visualize
        self.base_cmd_goal_visualizer.visualize(base_pos_w, cmd_quat, cmd_scale)
        self.base_cmd_current_visualizer.visualize(base_pos_w, vel_quat, vel_scale)

    """
    Internal helpers.
    """

    def _command_in_base_frame(self) -> torch.Tensor:
        """Transform command to base frame if needed."""
        cmd_xy = self.processed_actions[:, :2]
        if self.cfg.use_world_frame:
            # Transform world-frame command to body frame
            cmd_vec = torch.zeros(self.num_envs, 3, device=self.device)
            cmd_vec[:, :2] = cmd_xy
            cmd_body = math_utils.quat_apply_inverse(
                self._asset.data.root_quat_w, cmd_vec
            )
            return cmd_body[:, :2]
        return cmd_xy

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert XY velocity to arrow visualization (scale and orientation)."""
        # obtain default scale of the marker
        default_scale = torch.tensor(
            self.base_cmd_goal_visualizer.cfg.markers["arrow"].scale,
            device=self.device,
        )
        
        # compute arrow scale based on velocity magnitude
        arrow_scale = default_scale.repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        
        # compute arrow orientation
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        # transform to world frame
        base_quat_w = self._asset.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)
        
        return arrow_scale, arrow_quat
