"""Fixed via-point route guidance command for EE tracking.

This command replaces the old projection-driven moving preview with a fixed
episode-level route made of SE(3) via-points / gates:

  via_0 = current EE pose at reset
  via_1 ... via_M = sampled once at episode start

The active target is always a fixed gate. It only advances when the current
gate pass condition is satisfied.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    compute_pose_error,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class EEViaPointRouteCommandCfg(CommandTermCfg):
    """Configuration for fixed via-point / gate route guidance."""

    class_type: type = None

    asset_name: str = MISSING
    body_name: str = MISSING

    # Prevent auto-resample inside an episode.
    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    # Nominal EE offset from the mobile base root.
    arm_base_offset: tuple[float, float, float] = (0.3, 0.0, 0.52)
    collision_box_lower: tuple[float, float, float] = (-0.5, -0.4, 0.0)
    collision_box_upper: tuple[float, float, float] = (0.5, 0.4, 0.5)

    # Fixed route length: via_0 is the current EE pose, via_1..via_M are sampled.
    num_via_points: int = 10

    # Local delta sampling for via-point recursion.
    d_forward_min: float = 0.35
    d_forward_max: float = 0.70
    d_lat_range: tuple[float, float] = (-0.25, 0.25)
    d_z_range: tuple[float, float] = (-0.15, 0.15)

    # Relative to env origin z.
    z_range: tuple[float, float] = (0.50, 1.00)

    # Via-point safety checks.
    max_resample_attempts: int = 64
    check_obstacle_collision: bool = True
    obstacle_collision_margin: float = 0.10
    # Inflate the via-point collision check by a small radius so the sampled
    # via-point does not lie exactly on obstacle boundaries.  This margin only
    # applies to the via_pos_w point itself (no base or whole-body check).
    via_point_safe_margin: float = 0.05

    # Residual orientation noise added on top of the segment-direction-aligned
    # quaternion.  These are small Euler perturbations (rad).
    residual_roll_range: tuple[float, float] = (-0.05, 0.05)
    residual_pitch_range: tuple[float, float] = (-0.10, 0.10)
    residual_yaw_range: tuple[float, float] = (-0.10, 0.10)

    # Polyline cache used only for visualization / logging metrics.
    num_path_cache_samples: int = 64

    # Prioritized base-arm weighting around the active gate.
    arm_workspace_radius: float = 0.55
    base_switch_k: float = 12.0
    ori_weight_switch_distance: float = 0.20
    ori_weight_k: float = 12.0

    # Gate pass condition.
    gate_plane_hys: float = 0.05
    gate_pass_radius: float = 0.20
    gate_capture_radius: float = 0.12

    # Pose tolerances layered on top of the geometric gate pass condition.
    gate_position_tolerance: float = 0.20
    gate_orientation_tolerance: float = 0.50

    # Visualization.
    debug_vis: bool = False
    vis_update_interval: int = 10
    goal_pose_visualizer_cfg: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/goal_pose",
            markers={
                "frame": FRAME_MARKER_CFG.markers["frame"].replace(
                    scale=(0.15, 0.15, 0.15)
                )
            },
        )
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/current_pose",
            markers={
                "frame": FRAME_MARKER_CFG.markers["frame"].replace(
                    scale=(0.12, 0.12, 0.12)
                )
            },
        )
    )
    start_pose_visualizer_cfg: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/start_pose",
            markers={
                "frame": FRAME_MARKER_CFG.markers["frame"].replace(
                    scale=(0.12, 0.12, 0.12)
                )
            },
        )
    )
    trajectory_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/Command/trajectory",
            markers={
                "path_sphere": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.8, 1.0),
                        opacity=0.9,
                    ),
                ),
            },
        )
    )
    num_trajectory_samples: int = 32

    def __post_init__(self):
        super().__post_init__()
        if self.class_type is None:
            self.class_type = EEViaPointRouteCommand


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batched quaternion slerp."""
    if t.dim() == 2:
        t = t.squeeze(-1)
    dot = (q0 * q1).sum(dim=-1)
    q1 = torch.where(dot.unsqueeze(-1) < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0 - 1e-6)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp(min=1e-8)
    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta

    small = theta.abs() < 1e-6
    w0 = torch.where(small, 1.0 - t, w0)
    w1 = torch.where(small, t, w1)

    quat = w0.unsqueeze(-1) * q0 + w1.unsqueeze(-1) * q1
    return quat / quat.norm(dim=-1, keepdim=True).clamp(min=1e-8)


class EEViaPointRouteCommand(CommandTerm):
    """Fixed via-point / gate route guidance for the EE."""

    cfg: EEViaPointRouteCommandCfg

    def __init__(self, cfg: EEViaPointRouteCommandCfg, env: ManagerBasedRLEnv):
        requested_debug_vis = cfg.debug_vis
        cfg.debug_vis = False
        super().__init__(cfg, env)
        cfg.debug_vis = requested_debug_vis

        self._env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]
        self._arm_base_offset = torch.tensor(
            cfg.arm_base_offset, device=self.device, dtype=torch.float32
        )
        self._collision_box_lower_t = torch.tensor(
            cfg.collision_box_lower, device=self.device, dtype=torch.float32
        )
        self._collision_box_upper_t = torch.tensor(
            cfg.collision_box_upper, device=self.device, dtype=torch.float32
        )
        self._grid_neighbor_offset_cache: dict[int, torch.Tensor] = {}
        self._attempt_chunk_size = min(16, max(1, cfg.max_resample_attempts))
        self._attempt_scales = torch.full(
            (cfg.max_resample_attempts,), 0.25, device=self.device, dtype=torch.float32
        )
        half_attempts = cfg.max_resample_attempts // 2
        three_quarter_attempts = (3 * cfg.max_resample_attempts) // 4
        self._attempt_scales[:half_attempts] = 1.0
        self._attempt_scales[half_attempts:three_quarter_attempts] = 0.5

        self._num_route_points = cfg.num_via_points + 1
        self._num_route_segments = cfg.num_via_points
        self._final_gate_idx = self._num_route_points - 1

        n = self.num_envs
        m = cfg.num_path_cache_samples
        self.lie_command_b = torch.zeros(n, 6, device=self.device)
        self.pose_command_b = torch.zeros(n, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros(n, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0

        self._via_pos_w = torch.zeros(n, self._num_route_points, 3, device=self.device)
        self._via_quat_w = torch.zeros(n, self._num_route_points, 4, device=self.device)
        self._via_quat_w[:, :, 0] = 1.0

        self._active_gate_idx = torch.ones(n, dtype=torch.long, device=self.device)
        self._num_passed_gates = torch.zeros(n, dtype=torch.long, device=self.device)
        self._gate_passed_this_step = torch.zeros(n, dtype=torch.bool, device=self.device)
        self._final_gate_passed_this_step = torch.zeros(
            n, dtype=torch.bool, device=self.device
        )
        self._route_completed = torch.zeros(n, dtype=torch.bool, device=self.device)

        self._segment_lengths = torch.zeros(
            n, self._num_route_segments, device=self.device
        )
        self._cum_lengths = torch.zeros(
            n, self._num_route_points, device=self.device
        )
        self._path_total_length = torch.ones(n, device=self.device)

        self._path_s_grid = torch.linspace(0.0, 1.0, m, device=self.device)
        self._path_pos_w = torch.zeros(n, m, 3, device=self.device)
        self._path_quat_w = torch.zeros(n, m, 4, device=self.device)
        self._path_quat_w[:, :, 0] = 1.0
        self._path_tangent_w = torch.zeros(n, m, 3, device=self.device)

        self._s_proj = torch.zeros(n, device=self.device)
        self._robot_progress_delta = torch.zeros(n, device=self.device)
        self._segment_progress = torch.zeros(n, device=self.device)
        self._segment_progress_delta = torch.zeros(n, device=self.device)
        self._segment_length_current = torch.zeros(n, device=self.device)
        self._remaining_length = torch.zeros(n, device=self.device)
        self._segment_tangent_w = torch.zeros(n, 3, device=self.device)

        self._contouring_error = torch.zeros(n, device=self.device)
        self._path_orientation_error = torch.zeros(n, device=self.device)
        self._lag_error = torch.zeros(n, device=self.device)

        self._position_error = torch.zeros(n, device=self.device)
        self._orientation_error = torch.zeros(n, device=self.device)
        self._final_position_error = torch.full((n,), 1e6, device=self.device)
        self._final_orientation_error = torch.full((n,), 1e6, device=self.device)

        self._prev_position_error = torch.full((n,), float("nan"), device=self.device)
        self._prev_orientation_error = torch.full(
            (n,), float("nan"), device=self.device
        )
        self._position_progress_delta = torch.zeros(n, device=self.device)
        self._orientation_progress_delta = torch.zeros(n, device=self.device)

        self._prev_final_position_error = torch.full(
            (n,), float("nan"), device=self.device
        )
        self._final_goal_progress_delta = torch.zeros(n, device=self.device)

        self._rho_base = torch.zeros(n, device=self.device)
        self._prev_rho_base = torch.full((n,), float("nan"), device=self.device)
        self._rho_base_delta = torch.zeros(n, device=self.device)
        self._base_task_weight = torch.zeros(n, device=self.device)

        self._min_clearance = torch.full((n,), 1e6, device=self.device)

        self._ep_sum_contouring = torch.zeros(n, device=self.device)
        self._ep_sum_proj_ori_error = torch.zeros(n, device=self.device)
        self._ep_step_count = torch.zeros(n, device=self.device)

        self.metrics["position_error"] = torch.zeros(n, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(n, device=self.device)
        self.metrics["final_position_error"] = torch.full(
            (n,), 1e6, device=self.device
        )
        self.metrics["final_orientation_error"] = torch.full(
            (n,), 1e6, device=self.device
        )
        self.metrics["contouring_error"] = torch.zeros(n, device=self.device)
        self.metrics["path_orientation_error"] = torch.zeros(n, device=self.device)
        self.metrics["lag_error_signed"] = torch.zeros(n, device=self.device)
        self.metrics["lag_error_abs"] = torch.zeros(n, device=self.device)
        self.metrics["s_proj"] = torch.zeros(n, device=self.device)
        self.metrics["robot_progress_delta"] = torch.zeros(n, device=self.device)
        self.metrics["segment_progress"] = torch.zeros(n, device=self.device)
        self.metrics["segment_progress_delta"] = torch.zeros(n, device=self.device)
        self.metrics["position_progress_delta"] = torch.zeros(n, device=self.device)
        self.metrics["orientation_progress_delta"] = torch.zeros(n, device=self.device)
        self.metrics["final_goal_progress_delta"] = torch.zeros(
            n, device=self.device
        )
        self.metrics["active_gate_idx"] = torch.ones(n, device=self.device)
        self.metrics["num_passed_gates"] = torch.zeros(n, device=self.device)
        self.metrics["gate_passed_this_step"] = torch.zeros(n, device=self.device)
        self.metrics["segment_length"] = torch.zeros(n, device=self.device)
        self.metrics["remaining_length"] = torch.zeros(n, device=self.device)
        self.metrics["path_total_length"] = torch.ones(n, device=self.device)
        self.metrics["base_reachability_distance"] = torch.zeros(
            n, device=self.device
        )
        self.metrics["base_task_weight"] = torch.zeros(n, device=self.device)
        self.metrics["rho_base_delta"] = torch.zeros(n, device=self.device)
        self.metrics["min_clearance"] = torch.full((n,), 1e6, device=self.device)
        self.metrics["ep_mean_contouring"] = torch.zeros(n, device=self.device)
        self.metrics["ep_mean_ori_error"] = torch.zeros(n, device=self.device)

        self._vis_step_counter = 0

        self._has_height_grid = False
        if cfg.check_obstacle_collision:
            self._build_obstacle_height_grid()

        if requested_debug_vis:
            self.set_debug_vis(True)

    def __str__(self) -> str:
        return (
            "EEViaPointRouteCommand (Fixed Gate Sequence):\n"
            f"  Command dimension: {tuple(self.command.shape[1:])}\n"
            f"  Route gates: {self.cfg.num_via_points}\n"
            f"  Delta forward range: [{self.cfg.d_forward_min}, {self.cfg.d_forward_max}] m\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    @staticmethod
    def _yaw_only_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w = quat_wxyz[:, 0]
        x = quat_wxyz[:, 1]
        y = quat_wxyz[:, 2]
        z = quat_wxyz[:, 3]
        yaw = torch.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        zeros = torch.zeros_like(yaw)
        return quat_from_euler_xyz(zeros, zeros, yaw)

    def _build_obstacle_height_grid(self):
        try:
            from pxr import UsdGeom
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            mesh_prim_path = "/World/ground/terrain/mesh"
            mesh_prim = stage.GetPrimAtPath(mesh_prim_path)

            if not mesh_prim.IsValid():
                print(
                    f"[WARN] EEViaPointRouteCommand: terrain mesh not found at {mesh_prim_path}"
                )
                return

            mesh = UsdGeom.Mesh(mesh_prim)
            points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)
            if len(points) == 0:
                print("[WARN] EEViaPointRouteCommand: terrain mesh has 0 vertices")
                return

            h_scale = 0.1
            x_min = float(points[:, 0].min())
            y_min = float(points[:, 1].min())

            ix = np.round((points[:, 0] - x_min) / h_scale).astype(int)
            iy = np.round((points[:, 1] - y_min) / h_scale).astype(int)

            grid_w = int(ix.max()) + 1
            grid_h = int(iy.max()) + 1
            height_grid = np.zeros((grid_w, grid_h), dtype=np.float32)
            np.maximum.at(height_grid, (ix, iy), points[:, 2])

            self._height_grid = torch.tensor(height_grid, device=self.device)
            self._grid_origin = torch.tensor([x_min, y_min], device=self.device)
            self._grid_h_scale = h_scale
            self._height_obstacle_mask = self._height_grid > 0.05
            self._has_height_grid = True

            n_obs = int((height_grid > 0.05).sum())
            print(
                f"[INFO] EEViaPointRouteCommand: built obstacle height grid {grid_w}x{grid_h}, {n_obs} obstacle cells"
            )
        except Exception as exc:
            print(
                f"[WARN] EEViaPointRouteCommand: failed to build height grid: {exc}"
            )

    def _get_grid_neighbor_offsets(self, radius: float) -> torch.Tensor:
        if radius <= 0.0:
            return torch.zeros((1, 2), device=self.device, dtype=torch.long)

        r_cells = max(1, int(radius / self._grid_h_scale + 0.5))
        offsets = self._grid_neighbor_offset_cache.get(r_cells)
        if offsets is None:
            delta = torch.arange(
                -r_cells, r_cells + 1, device=self.device, dtype=torch.long
            )
            dx, dy = torch.meshgrid(delta, delta, indexing="ij")
            offsets = torch.stack([dx.reshape(-1), dy.reshape(-1)], dim=-1)
            self._grid_neighbor_offset_cache[r_cells] = offsets
        return offsets

    def _gather_height_grid_window(
        self,
        pos_w: torch.Tensor,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        offsets = self._get_grid_neighbor_offsets(radius)
        ix = torch.round(
            (pos_w[:, 0] - self._grid_origin[0]) / self._grid_h_scale
        ).long()
        iy = torch.round(
            (pos_w[:, 1] - self._grid_origin[1]) / self._grid_h_scale
        ).long()

        ci = torch.clamp(
            ix.unsqueeze(1) + offsets[:, 0].unsqueeze(0),
            0,
            self._height_grid.shape[0] - 1,
        )
        cj = torch.clamp(
            iy.unsqueeze(1) + offsets[:, 1].unsqueeze(0),
            0,
            self._height_grid.shape[1] - 1,
        )
        heights = self._height_grid[ci, cj]
        return ci, cj, heights

    def _check_obstacle_collision(self, pos_w: torch.Tensor) -> torch.Tensor:
        if not self._has_height_grid:
            return torch.zeros(pos_w.shape[0], dtype=torch.bool, device=pos_w.device)

        margin = self.cfg.obstacle_collision_margin
        _, _, local_heights = self._gather_height_grid_window(pos_w, margin)
        max_height = local_heights.max(dim=1).values
        return (max_height > 0.05) & (pos_w[:, 2] < max_height + margin)

    def _has_static_clearance(
        self,
        pos_w: torch.Tensor,
        clearance_margin: float,
    ) -> torch.Tensor:
        if (not self._has_height_grid) or clearance_margin <= 0.0:
            return torch.ones(pos_w.shape[0], dtype=torch.bool, device=pos_w.device)

        ci, cj, cell_h = self._gather_height_grid_window(pos_w, clearance_margin)
        cell_x = self._grid_origin[0] + ci.float() * self._grid_h_scale
        cell_y = self._grid_origin[1] + cj.float() * self._grid_h_scale

        dx = pos_w[:, 0:1] - cell_x
        dy = pos_w[:, 1:2] - cell_y
        d_xy_sq = dx.square() + dy.square()
        near_obstacle = (
            (cell_h > 0.05)
            & (d_xy_sq <= clearance_margin * clearance_margin)
            & (pos_w[:, 2:3] < cell_h + clearance_margin)
        )
        return ~near_obstacle.any(dim=1)

    def _validate_via_point(
        self,
        env_ids: torch.Tensor,
        via_pos_w: torch.Tensor,
    ) -> torch.Tensor:
        """Position-only validity check (z range + obstacle collision + inflation)."""
        env_origin_z = self._env.scene.env_origins[env_ids, 2]
        rel_z = via_pos_w[:, 2] - env_origin_z
        valid = (rel_z >= self.cfg.z_range[0]) & (rel_z <= self.cfg.z_range[1])

        if self.cfg.check_obstacle_collision:
            valid &= ~self._check_obstacle_collision(via_pos_w)
            if self.cfg.via_point_safe_margin > 0.0:
                valid &= self._has_static_clearance(
                    via_pos_w,
                    self.cfg.via_point_safe_margin,
                )

        return valid

    def _sample_position_delta(
        self,
        batch_shape: int | tuple[int, ...],
        scale: float | torch.Tensor = 1.0,
    ) -> torch.Tensor:
        """Sample a position-only local delta (forward, lateral, vertical)."""
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)

        if torch.is_tensor(scale):
            scale_t = scale.to(device=self.device, dtype=torch.float32)
            scale_t = torch.broadcast_to(scale_t, batch_shape)
        else:
            scale_t = torch.full(
                batch_shape, float(scale), device=self.device, dtype=torch.float32
            )

        d_fwd = torch.empty(*batch_shape, device=self.device).uniform_(
            self.cfg.d_forward_min, self.cfg.d_forward_max
        )

        def _sample_scaled_range(range_cfg: tuple[float, float]) -> torch.Tensor:
            low = scale_t * range_cfg[0]
            high = scale_t * range_cfg[1]
            return low + torch.rand(*batch_shape, device=self.device) * (high - low)

        d_lat = _sample_scaled_range(self.cfg.d_lat_range)
        d_z = _sample_scaled_range(self.cfg.d_z_range)

        return torch.stack([d_fwd, d_lat, d_z], dim=-1)

    def _assign_orientation(
        self,
        prev_pos_w: torch.Tensor,
        next_pos_w: torch.Tensor,
        prev_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        """Derive the next via-point orientation from the previous one.

        Instead of constructing a quaternion from scratch (which ignores the
        EE's intrinsic roll/pitch), we:
          1. Compute the yaw & pitch *change* implied by the segment direction
             relative to the previous orientation's local x-axis.
          2. Apply that delta rotation to the previous quaternion.
          3. Optionally add a small residual perturbation.

        This keeps the orientation chain smooth and continuous with the actual
        EE pose at via_0.
        """
        seg = next_pos_w - prev_pos_w
        seg_len = seg.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        seg_dir = seg / seg_len  # (N, 3) world-frame unit direction

        # Previous via-point's local x-axis in world frame.
        local_x = torch.zeros_like(seg_dir)
        local_x[:, 0] = 1.0
        prev_x_w = quat_apply(prev_quat_w, local_x)  # (N, 3)

        # Project both into the XY plane to get the yaw delta.
        prev_yaw = torch.atan2(prev_x_w[:, 1], prev_x_w[:, 0])
        seg_yaw = torch.atan2(seg_dir[:, 1], seg_dir[:, 0])
        delta_yaw = seg_yaw - prev_yaw
        # Wrap to [-pi, pi].
        delta_yaw = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw))

        # Pitch delta: elevation angle of seg_dir vs prev_x_w.
        prev_xy_len = prev_x_w[:, :2].norm(dim=-1).clamp(min=1e-8)
        prev_pitch = torch.atan2(prev_x_w[:, 2], prev_xy_len)
        seg_xy_len = seg_dir[:, :2].norm(dim=-1).clamp(min=1e-8)
        seg_pitch = torch.atan2(seg_dir[:, 2], seg_xy_len)
        delta_pitch = seg_pitch - prev_pitch

        zeros = torch.zeros_like(delta_yaw)
        heading_delta_quat = quat_from_euler_xyz(zeros, delta_pitch, delta_yaw)
        base_quat = quat_unique(quat_mul(prev_quat_w, heading_delta_quat))

        # Small residual perturbation.
        n = prev_pos_w.shape[0]
        r_roll = torch.empty(n, device=self.device).uniform_(
            *self.cfg.residual_roll_range
        )
        r_pitch = torch.empty(n, device=self.device).uniform_(
            *self.cfg.residual_pitch_range
        )
        r_yaw = torch.empty(n, device=self.device).uniform_(
            *self.cfg.residual_yaw_range
        )
        residual_quat = quat_from_euler_xyz(r_roll, r_pitch, r_yaw)
        return quat_unique(quat_mul(base_quat, residual_quat))

    def _sample_next_via_points(
        self,
        prev_via: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Sample the next via-point positions, then assign orientations.

        1. Position sampling with collision rejection (chunked attempts).
        2. Last-resort: deterministic forward step (no zero-length fallback).
        3. Orientation is derived from segment direction + residual noise.
        """
        n = prev_via.shape[0]
        prev_pos = prev_via[:, :3]
        prev_quat = prev_via[:, 3:7]
        sampled_pos = prev_pos.clone()
        accepted = torch.zeros(n, dtype=torch.bool, device=self.device)

        # ── Phase 1: chunked random position sampling with rejection ──
        for attempt_start in range(
            0, self.cfg.max_resample_attempts, self._attempt_chunk_size
        ):
            pending = torch.where(~accepted)[0]
            if pending.numel() == 0:
                break

            attempt_end = min(
                attempt_start + self._attempt_chunk_size,
                self.cfg.max_resample_attempts,
            )
            attempt_scales = self._attempt_scales[attempt_start:attempt_end].view(-1, 1)
            num_attempts = attempt_scales.shape[0]
            num_pending = pending.numel()

            delta_pos_local = self._sample_position_delta(
                (num_attempts, num_pending), scale=attempt_scales
            )
            # Apply the local delta in the previous via-point's frame.
            prev_quat_expanded = (
                prev_quat[pending].unsqueeze(0).expand(num_attempts, -1, -1)
            )
            candidate_pos = (
                prev_pos[pending].unsqueeze(0).expand(num_attempts, -1, -1)
                + quat_apply(
                    prev_quat_expanded.reshape(-1, 4),
                    delta_pos_local.reshape(-1, 3),
                ).reshape(num_attempts, num_pending, 3)
            )

            env_ids_expanded = env_ids[pending].unsqueeze(0).expand(num_attempts, -1)
            valid = self._validate_via_point(
                env_ids_expanded.reshape(-1),
                candidate_pos.reshape(-1, 3),
            ).reshape(num_attempts, num_pending)

            has_valid = valid.any(dim=0)
            if has_valid.any():
                first_valid_idx = valid.float().argmax(dim=0)
                chosen_pos = candidate_pos[
                    first_valid_idx,
                    torch.arange(num_pending, device=self.device),
                ]
                ok = pending[has_valid]
                sampled_pos[ok] = chosen_pos[has_valid]
                accepted[ok] = True

        # ── Phase 2: deterministic forward step for remaining envs ──
        remaining = torch.where(~accepted)[0]
        if remaining.numel() > 0:
            delta_fwd = torch.zeros(remaining.numel(), 3, device=self.device)
            delta_fwd[:, 0] = self.cfg.d_forward_min
            candidate_pos = prev_pos[remaining] + quat_apply(
                prev_quat[remaining], delta_fwd
            )
            valid = self._validate_via_point(
                env_ids[remaining], candidate_pos
            )
            # Only accept valid deterministic candidates; invalid ones get
            # the deterministic step anyway (never produce zero-length segments).
            sampled_pos[remaining] = candidate_pos

        # ── Phase 3: assign orientation from segment direction ──
        sampled_quat = self._assign_orientation(prev_pos, sampled_pos, prev_quat)

        return torch.cat([sampled_pos, sampled_quat], dim=-1)

    @staticmethod
    def _interp_1d_batch(
        x_known: torch.Tensor,
        y_known: torch.Tensor,
        x_query: torch.Tensor,
    ) -> torch.Tensor:
        """Batched linear interpolation with shared query coordinates."""
        n, _, c = y_known.shape
        m = x_query.shape[0]
        x_q = x_query.unsqueeze(0).expand(n, m).contiguous()
        idx_right = torch.searchsorted(x_known.contiguous(), x_q).clamp(
            min=1, max=x_known.shape[1] - 1
        )
        idx_left = idx_right - 1

        x_left = x_known.gather(1, idx_left)
        x_right = x_known.gather(1, idx_right)
        t = ((x_q - x_left) / (x_right - x_left).clamp(min=1e-8)).clamp(0.0, 1.0)

        il = idx_left.unsqueeze(-1).expand(n, m, c)
        ir = idx_right.unsqueeze(-1).expand(n, m, c)
        y_left = y_known.gather(1, il)
        y_right = y_known.gather(1, ir)
        return y_left + t.unsqueeze(-1) * (y_right - y_left)

    @staticmethod
    def _interp_quat_batch(
        s_known: torch.Tensor,
        quat_known: torch.Tensor,
        s_query: torch.Tensor,
    ) -> torch.Tensor:
        n, _, _ = quat_known.shape
        m = s_query.shape[0]
        s_q = s_query.unsqueeze(0).expand(n, m).contiguous()
        idx_right = torch.searchsorted(s_known.contiguous(), s_q).clamp(
            min=1, max=s_known.shape[1] - 1
        )
        idx_left = idx_right - 1

        s_left = s_known.gather(1, idx_left)
        s_right = s_known.gather(1, idx_right)
        t = ((s_q - s_left) / (s_right - s_left).clamp(min=1e-8)).clamp(0.0, 1.0)

        il = idx_left.unsqueeze(-1).expand(n, m, 4)
        ir = idx_right.unsqueeze(-1).expand(n, m, 4)
        q_left = quat_known.gather(1, il).reshape(-1, 4)
        q_right = quat_known.gather(1, ir).reshape(-1, 4)
        quat = _quat_slerp(q_left, q_right, t.reshape(-1))
        return quat.reshape(n, m, 4)

    def _build_path_cache(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        via_pos = self._via_pos_w[env_ids]
        via_quat = self._via_quat_w[env_ids]

        seg_vec = via_pos[:, 1:] - via_pos[:, :-1]
        seg_len = seg_vec.norm(dim=-1).clamp(min=1e-6)
        cum_len = torch.zeros(
            len(env_ids), self._num_route_points, device=self.device
        )
        cum_len[:, 1:] = torch.cumsum(seg_len, dim=-1)
        total_len = cum_len[:, -1].clamp(min=1e-6)

        self._segment_lengths[env_ids] = seg_len
        self._cum_lengths[env_ids] = cum_len
        self._path_total_length[env_ids] = total_len

        s_known = cum_len / total_len.unsqueeze(-1)
        path_pos = self._interp_1d_batch(s_known, via_pos, self._path_s_grid)
        path_quat = self._interp_quat_batch(s_known, via_quat, self._path_s_grid)

        tangent = torch.zeros_like(path_pos)
        tangent[:, :-1] = path_pos[:, 1:] - path_pos[:, :-1]
        tangent[:, -1] = tangent[:, -2]
        tangent = tangent / tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)

        self._path_pos_w[env_ids] = path_pos
        self._path_quat_w[env_ids] = quat_unique(path_quat)
        self._path_tangent_w[env_ids] = tangent

    def _compute_min_clearance(self):
        try:
            from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.mdp.observations import (
                compute_full_min_clearance,
            )

            clearance = compute_full_min_clearance(self._env)
            self._min_clearance = clearance["min_clearance"]
        except Exception:
            pass

    def _current_segment_indices(self) -> tuple[torch.Tensor, torch.Tensor]:
        end_idx = self._active_gate_idx.clamp(1, self._final_gate_idx)
        start_idx = (end_idx - 1).clamp(0, self._final_gate_idx - 1)
        return start_idx, end_idx

    def _compute_segment_state(
        self,
        ee_pos_w: torch.Tensor,
        start_idx: torch.Tensor,
        end_idx: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        env_idx = torch.arange(self.num_envs, device=self.device)
        seg_start = self._via_pos_w[env_idx, start_idx]
        seg_end = self._via_pos_w[env_idx, end_idx]
        seg_vec = seg_end - seg_start
        seg_len = seg_vec.norm(dim=-1).clamp(min=1e-6)
        seg_dir = seg_vec / seg_len.unsqueeze(-1)
        lambda_raw = ((ee_pos_w - seg_start) * seg_vec).sum(dim=-1) / (seg_len**2)
        lambda_clamped = lambda_raw.clamp(0.0, 1.0)
        closest = seg_start + lambda_clamped.unsqueeze(-1) * seg_vec
        return seg_start, seg_end, seg_dir, seg_len, lambda_clamped, closest

    def _compute_gate_pass_mask(
        self,
        ee_pos_w: torch.Tensor,
        ee_quat_w: torch.Tensor,
        active_idx: torch.Tensor,
    ) -> torch.Tensor:
        env_idx = torch.arange(self.num_envs, device=self.device)
        gate_pos = self._via_pos_w[env_idx, active_idx]
        gate_quat = self._via_quat_w[env_idx, active_idx]

        rel = quat_apply_inverse(gate_quat, ee_pos_w - gate_pos)
        rel_yz = rel[:, 1:].norm(dim=-1)
        pos_err_vec, rot_err_vec = compute_pose_error(
            gate_pos, gate_quat, ee_pos_w, ee_quat_w
        )
        dist = pos_err_vec.norm(dim=-1)
        ori_err = rot_err_vec.norm(dim=-1)
        geometry_ok = (
            (rel[:, 0] >= -self.cfg.gate_plane_hys)
            & (rel_yz <= self.cfg.gate_pass_radius)
        ) | (dist <= self.cfg.gate_capture_radius)
        pose_ok = (
            (dist <= self.cfg.gate_position_tolerance)
            & (ori_err <= self.cfg.gate_orientation_tolerance)
        )
        return geometry_ok & pose_ok

    def _update_command(self):
        if not self.robot.is_initialized:
            return

        self._compute_min_clearance()

        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = quat_unique(self.robot.data.body_quat_w[:, self.body_idx])
        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w

        prev_s_proj = self._s_proj.clone()
        prev_segment_progress = self._segment_progress.clone()
        prev_active_gate_idx = self._active_gate_idx.clone()

        gate_pass = self._compute_gate_pass_mask(
            ee_pos_w, ee_quat_w, prev_active_gate_idx
        )
        gate_pass &= ~self._route_completed
        final_pass = gate_pass & (prev_active_gate_idx == self._final_gate_idx)
        intermediate_pass = gate_pass & ~final_pass

        self._gate_passed_this_step = gate_pass
        self._final_gate_passed_this_step = final_pass
        self._num_passed_gates = torch.clamp(
            self._num_passed_gates + gate_pass.long(),
            max=self._final_gate_idx,
        )
        self._active_gate_idx = torch.where(
            intermediate_pass,
            torch.clamp(self._active_gate_idx + 1, max=self._final_gate_idx),
            self._active_gate_idx,
        )
        self._route_completed |= final_pass

        start_idx, end_idx = self._current_segment_indices()
        env_idx = torch.arange(self.num_envs, device=self.device)
        seg_start, _, seg_dir, seg_len, seg_progress, closest = self._compute_segment_state(
            ee_pos_w, start_idx, end_idx
        )
        seg_progress = torch.where(
            self._route_completed,
            torch.ones_like(seg_progress),
            seg_progress,
        )

        start_len = self._cum_lengths[env_idx, start_idx]
        s_proj = (
            start_len + seg_progress * seg_len
        ) / self._path_total_length.clamp(min=1e-6)
        s_proj = torch.where(self._route_completed, torch.ones_like(s_proj), s_proj)

        segment_changed = gate_pass | torch.isnan(prev_segment_progress)
        self._segment_progress = seg_progress
        self._segment_progress_delta = torch.where(
            segment_changed,
            torch.zeros_like(seg_progress),
            (seg_progress - prev_segment_progress).clamp(min=0.0, max=1.0),
        )
        self._s_proj = s_proj.clamp(0.0, 1.0)
        self._robot_progress_delta = (
            self._s_proj - prev_s_proj
        ).clamp(min=0.0, max=0.2)
        self._segment_length_current = seg_len
        self._remaining_length = (
            self._path_total_length - (start_len + seg_progress * seg_len)
        ).clamp(min=0.0)
        self._segment_tangent_w = seg_dir
        self._contouring_error = torch.where(
            self._route_completed,
            torch.zeros_like(seg_len),
            (ee_pos_w - closest).norm(dim=-1),
        )

        seg_start_quat = self._via_quat_w[env_idx, start_idx]
        seg_end_quat = self._via_quat_w[env_idx, end_idx]
        proj_quat = _quat_slerp(seg_start_quat, seg_end_quat, seg_progress)
        _, proj_rot_err_vec = compute_pose_error(
            closest, proj_quat, ee_pos_w, ee_quat_w
        )
        self._path_orientation_error = torch.where(
            self._route_completed,
            torch.zeros_like(seg_len),
            proj_rot_err_vec.norm(dim=-1),
        )

        active_pos = self._via_pos_w[env_idx, end_idx]
        active_quat = self._via_quat_w[env_idx, end_idx]
        self._lag_error = ((active_pos - ee_pos_w) * seg_dir).sum(dim=-1)

        pos_err_vec, rot_err_vec = compute_pose_error(
            active_pos, active_quat, ee_pos_w, ee_quat_w
        )
        d_p = pos_err_vec.norm(dim=-1)
        d_o = rot_err_vec.norm(dim=-1)
        self._position_error = d_p
        self._orientation_error = d_o

        reset_gate_progress = torch.isnan(self._prev_position_error) | gate_pass
        self._position_progress_delta = torch.where(
            reset_gate_progress,
            torch.zeros_like(d_p),
            self._prev_position_error - d_p,
        )
        self._orientation_progress_delta = torch.where(
            reset_gate_progress,
            torch.zeros_like(d_o),
            self._prev_orientation_error - d_o,
        )
        self._prev_position_error = d_p.clone()
        self._prev_orientation_error = d_o.clone()

        base_pos_xy = root_pos_w[:, :2]
        rho_base = (active_pos[:, :2] - base_pos_xy).norm(dim=-1)
        reset_base_progress = torch.isnan(self._prev_rho_base) | gate_pass
        self._rho_base_delta = torch.where(
            reset_base_progress,
            torch.zeros_like(rho_base),
            self._prev_rho_base - rho_base,
        )
        self._prev_rho_base = rho_base.clone()
        self._rho_base = rho_base
        self._base_task_weight = torch.sigmoid(
            self.cfg.base_switch_k * (rho_base - self.cfg.arm_workspace_radius)
        )

        final_pos = self._via_pos_w[:, -1]
        final_quat = self._via_quat_w[:, -1]
        final_pos_err, final_rot_err = compute_pose_error(
            final_pos, final_quat, ee_pos_w, ee_quat_w
        )
        d_final = final_pos_err.norm(dim=-1)
        self._final_position_error = d_final
        self._final_orientation_error = final_rot_err.norm(dim=-1)
        first_final = torch.isnan(self._prev_final_position_error)
        self._final_goal_progress_delta = torch.where(
            first_final,
            torch.zeros_like(d_final),
            self._prev_final_position_error - d_final,
        )
        self._prev_final_position_error = d_final.clone()

        self.pose_command_w[:, :3] = active_pos
        self.pose_command_w[:, 3:7] = quat_unique(active_quat)
        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, active_pos, active_quat
        )
        self.pose_command_b[:, :3] = cmd_pos_b
        self.pose_command_b[:, 3:7] = cmd_quat_b

    def _resample_command(self, env_ids):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        if len(env_ids) == 0:
            return

        self._via_pos_w[env_ids] = 0.0
        self._via_quat_w[env_ids] = 0.0
        self._via_quat_w[env_ids, :, 0] = 1.0
        self._active_gate_idx[env_ids] = 1
        self._num_passed_gates[env_ids] = 0
        self._gate_passed_this_step[env_ids] = False
        self._final_gate_passed_this_step[env_ids] = False
        self._route_completed[env_ids] = False

        self._segment_lengths[env_ids] = 0.0
        self._cum_lengths[env_ids] = 0.0
        self._path_total_length[env_ids] = 1.0
        self._path_pos_w[env_ids] = 0.0
        self._path_quat_w[env_ids] = 0.0
        self._path_quat_w[env_ids, :, 0] = 1.0
        self._path_tangent_w[env_ids] = 0.0

        self._s_proj[env_ids] = 0.0
        self._robot_progress_delta[env_ids] = 0.0
        self._segment_progress[env_ids] = 0.0
        self._segment_progress_delta[env_ids] = 0.0
        self._segment_length_current[env_ids] = 0.0
        self._remaining_length[env_ids] = 0.0
        self._segment_tangent_w[env_ids] = 0.0

        self._contouring_error[env_ids] = 0.0
        self._path_orientation_error[env_ids] = 0.0
        self._lag_error[env_ids] = 0.0

        self._position_error[env_ids] = 0.0
        self._orientation_error[env_ids] = 0.0
        self._final_position_error[env_ids] = 1e6
        self._final_orientation_error[env_ids] = 1e6
        self._prev_position_error[env_ids] = float("nan")
        self._prev_orientation_error[env_ids] = float("nan")
        self._position_progress_delta[env_ids] = 0.0
        self._orientation_progress_delta[env_ids] = 0.0
        self._prev_final_position_error[env_ids] = float("nan")
        self._final_goal_progress_delta[env_ids] = 0.0

        self._rho_base[env_ids] = 0.0
        self._prev_rho_base[env_ids] = float("nan")
        self._rho_base_delta[env_ids] = 0.0
        self._base_task_weight[env_ids] = 0.0
        self._min_clearance[env_ids] = 1e6

        self._ep_sum_contouring[env_ids] = 0.0
        self._ep_sum_proj_ori_error[env_ids] = 0.0
        self._ep_step_count[env_ids] = 0.0

        self.pose_command_b[env_ids] = 0.0
        self.pose_command_b[env_ids, 3] = 1.0
        self.pose_command_w[env_ids] = 0.0
        self.pose_command_w[env_ids, 3] = 1.0

        if not self.robot.is_initialized:
            self._build_path_cache(env_ids)
            return

        ee_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]
        ee_quat_w = quat_unique(self.robot.data.body_quat_w[env_ids, self.body_idx])
        self._via_pos_w[env_ids, 0] = ee_pos_w
        self._via_quat_w[env_ids, 0] = ee_quat_w

        prev_via = torch.cat([ee_pos_w, ee_quat_w], dim=-1)
        for via_idx in range(1, self._num_route_points):
            next_via = self._sample_next_via_points(prev_via, env_ids)
            self._via_pos_w[env_ids, via_idx] = next_via[:, :3]
            self._via_quat_w[env_ids, via_idx] = next_via[:, 3:7]
            prev_via = next_via

        self._build_path_cache(env_ids)

        active_pos = self._via_pos_w[env_ids, 1]
        active_quat = self._via_quat_w[env_ids, 1]
        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]
        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            root_pos, root_quat, active_pos, active_quat
        )
        self.pose_command_w[env_ids, :3] = active_pos
        self.pose_command_w[env_ids, 3:7] = active_quat
        self.pose_command_b[env_ids, :3] = cmd_pos_b
        self.pose_command_b[env_ids, 3:7] = cmd_quat_b

    def _update_metrics(self):
        if not self.robot.is_initialized:
            return

        self._ep_sum_contouring += self._contouring_error
        self._ep_sum_proj_ori_error += self._path_orientation_error
        self._ep_step_count += 1.0
        count = self._ep_step_count.clamp(min=1.0)

        self.metrics["position_error"] = self._position_error
        self.metrics["orientation_error"] = self._orientation_error
        self.metrics["final_position_error"] = self._final_position_error
        self.metrics["final_orientation_error"] = self._final_orientation_error
        self.metrics["contouring_error"] = self._contouring_error
        self.metrics["path_orientation_error"] = self._path_orientation_error
        self.metrics["lag_error_signed"] = self._lag_error
        self.metrics["lag_error_abs"] = self._lag_error.abs()
        self.metrics["s_proj"] = self._s_proj
        self.metrics["robot_progress_delta"] = self._robot_progress_delta
        self.metrics["segment_progress"] = self._segment_progress
        self.metrics["segment_progress_delta"] = self._segment_progress_delta
        self.metrics["position_progress_delta"] = self._position_progress_delta
        self.metrics["orientation_progress_delta"] = self._orientation_progress_delta
        self.metrics["final_goal_progress_delta"] = self._final_goal_progress_delta
        self.metrics["active_gate_idx"] = self._active_gate_idx.float()
        self.metrics["num_passed_gates"] = self._num_passed_gates.float()
        self.metrics["gate_passed_this_step"] = self._gate_passed_this_step.float()
        self.metrics["segment_length"] = self._segment_length_current
        self.metrics["remaining_length"] = self._remaining_length
        self.metrics["path_total_length"] = self._path_total_length
        self.metrics["base_reachability_distance"] = self._rho_base
        self.metrics["base_task_weight"] = self._base_task_weight
        self.metrics["rho_base_delta"] = self._rho_base_delta
        self.metrics["min_clearance"] = self._min_clearance
        self.metrics["ep_mean_contouring"] = self._ep_sum_contouring / count
        self.metrics["ep_mean_ori_error"] = self._ep_sum_proj_ori_error / count

    def get_path_positions_at_s(self, s_query: torch.Tensor) -> torch.Tensor:
        """Get cached polyline positions at arbitrary s."""
        if s_query.dim() == 1:
            s_query = s_query.unsqueeze(-1)
            squeeze = True
        else:
            squeeze = False

        n, k = s_query.shape
        m = self.cfg.num_path_cache_samples
        flat = s_query.reshape(-1).clamp(0.0, 1.0)
        idx_f = flat * (m - 1)
        idx_left = idx_f.long().clamp(0, m - 2)
        idx_right = idx_left + 1
        t = idx_f - idx_left.float()
        env_idx = torch.arange(n, device=self.device).repeat_interleave(k)
        p_left = self._path_pos_w[env_idx, idx_left]
        p_right = self._path_pos_w[env_idx, idx_right]
        pos = p_left + t.unsqueeze(-1) * (p_right - p_left)
        pos = pos.reshape(n, k, 3)
        return pos[:, 0] if squeeze else pos

    def get_final_goal_pose_w(self) -> torch.Tensor:
        return torch.cat([self._via_pos_w[:, -1], self._via_quat_w[:, -1]], dim=-1)

    def get_active_gate_pose_w(self) -> torch.Tensor:
        env_idx = torch.arange(self.num_envs, device=self.device)
        idx = self._active_gate_idx.clamp(1, self._final_gate_idx)
        return torch.cat(
            [self._via_pos_w[env_idx, idx], self._via_quat_w[env_idx, idx]], dim=-1
        )

    def update_min_clearance(self, min_clearance: torch.Tensor):
        self._min_clearance = min_clearance

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(
                    self.cfg.goal_pose_visualizer_cfg
                )
            if not hasattr(self, "start_pose_visualizer"):
                self.start_pose_visualizer = VisualizationMarkers(
                    self.cfg.start_pose_visualizer_cfg
                )
            if not hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer = VisualizationMarkers(
                    self.cfg.current_pose_visualizer_cfg
                )
            if not hasattr(self, "trajectory_visualizer"):
                self.trajectory_visualizer = VisualizationMarkers(
                    self.cfg.trajectory_visualizer_cfg
                )

            self.goal_pose_visualizer.set_visibility(True)
            self.start_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.trajectory_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "start_pose_visualizer"):
                self.start_pose_visualizer.set_visibility(False)
            if hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer.set_visibility(False)
            if hasattr(self, "trajectory_visualizer"):
                self.trajectory_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self._vis_step_counter += 1
        if self._vis_step_counter < self.cfg.vis_update_interval:
            return
        self._vis_step_counter = 0

        if hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.visualize(
                self._via_pos_w[:, -1], self._via_quat_w[:, -1]
            )

        if hasattr(self, "start_pose_visualizer"):
            self.start_pose_visualizer.visualize(
                self._via_pos_w[:, 0], self._via_quat_w[:, 0]
            )

        if hasattr(self, "current_pose_visualizer"):
            self.current_pose_visualizer.visualize(
                self.pose_command_w[:, :3], self.pose_command_w[:, 3:7]
            )

        if hasattr(self, "trajectory_visualizer"):
            vis_n = min(self.cfg.num_trajectory_samples, self.cfg.num_path_cache_samples)
            step = max(1, self.cfg.num_path_cache_samples // vis_n)
            flat_pos = self._path_pos_w[:, ::step, :].reshape(-1, 3)
            self.trajectory_visualizer.visualize(translations=flat_pos)
