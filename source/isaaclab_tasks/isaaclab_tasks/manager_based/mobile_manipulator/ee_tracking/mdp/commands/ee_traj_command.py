"""SE(3) EE Path Command Generator – Bezier Path + Monotonic Virtual Progress.

Generates a coarse global centerline (quadratic/cubic Bezier) with arc-length
normalization.  Position and orientation are decoupled: position follows the
Bezier p(s), orientation follows quaternion slerp R(s), sharing the same
normalized progress parameter s ∈ [0, 1].

Progress uses a two-layer design:
  1. s_proj: geometric projection onto the cached path (can retreat).
     Used for global progress reward (Δs_proj).
  2. s_hat:  monotonic virtual progress, driven by MPCC-style lag/contouring
             trackability gates computed at the s_hat reference point itself.
             s_hat never decreases → eliminates reference point retreat.

The s_hat update decomposes the error e = P(s_hat) - ee into:
  - lag error (tangential): how far EE is behind s_hat along the path
  - contouring error (normal): how far EE deviates laterally from the path
Both errors pass through deadband + Gaussian gates to modulate the nominal
virtual speed.  A separate catch-up term accelerates s_hat when s_proj > s_hat.

Preview points are sampled from s_hat:
  s_hat → uniform preview sampling → q1=s_ref → local tracking.
No obstacle-aware filtering is applied to preview points; the RL policy
is fully responsible for obstacle avoidance.
Path-ahead clearance modulates an adaptive contouring tube deadband.

All delta rewards are signed (positive for improvement, negative for
regression) to prevent oscillation exploitation.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    compute_pose_error,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    quat_unique,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Configuration                                                       ║
# ╚═══════════════════════════════════════════════════════════════════════╝


@configclass
class EETrajectoryCommandCfg(CommandTermCfg):
    """Configuration for SE(3) Bezier path + projection-driven progress command."""

    class_type: type = None

    # Asset configuration
    asset_name: str = MISSING
    body_name: str = MISSING

    # Timing – very large to prevent auto-resample
    resampling_time_range: tuple[float, float] = (1e6, 1e6)

    # Orientation sampling
    sample_uniform_orientation: bool = False

    arm_base_offset: tuple[float, float, float] = (0.3, 0.0, 0.52)
    ranges_pos: object = {
        "rho_xy": (0.8, 2.0),
        "yaw": (-3 * torch.pi / 5, 3 * torch.pi / 5),
    }
    delta_euler_ranges: object = {
        "roll":  (-0.1, 0.1),
        "pitch": (-2 * torch.pi / 5, 2 * torch.pi / 5),
        "yaw":   (-3 * torch.pi / 5, 3 * torch.pi / 5),
    }
    z_range: tuple[float, float] = (0.2, 1.0)

    collision_box_lower: tuple[float, float, float] = (-0.5, -0.4, 0.0)
    collision_box_upper: tuple[float, float, float] = (0.5, 0.4, 0.5)

    max_resample_attempts: int = 10

    # Static obstacle collision checking
    check_obstacle_collision: bool = True
    obstacle_collision_margin: float = 0.1

    # --- Bezier path parameters ---
    num_control_points: int = 1
    """Number of intermediate control points (0 = line, 1 = quadratic, 2 = cubic)."""
    control_point_lateral_range: tuple[float, float] = (-0.3, 0.3)
    """Lateral offset range (m) for intermediate control points."""
    control_point_vertical_range: tuple[float, float] = (-0.15, 0.15)
    """Vertical offset range (m) for intermediate control points."""

    # --- Path discretisation ---
    num_path_cache_samples: int = 64
    """Number of uniformly-spaced s-grid points for the cached path."""
    num_bezier_dense_samples: int = 256
    """Number of dense samples on the raw Bezier for arc-length computation."""
    projection_window_backward_segments: int = 8
    """Number of cached segments searched behind the previous projected segment."""
    projection_window_forward_segments: int = 16
    """Number of cached segments searched ahead of the previous projected segment."""

    # --- Local preview stack ---
    num_preview_points: int = 4
    """Total number of preview points (N). q1=s_ref, q2..qN=future refs."""
    preview_spacing_s_base: float = 0.06
    """Base spacing in s-units between consecutive preview points."""
    preview_spacing_s_min: float = 0.03
    """Minimum adaptive spacing."""
    preview_spacing_s_max: float = 0.10
    """Maximum adaptive spacing."""
    preview_spacing_error_gain: float = 2.0
    """Gain for shrinking spacing with contouring error."""

    # --- Path-ahead clearance (static only) ---
    path_clearance_num_samples: int = 6
    """Number of sample points along nominal path ahead for clearance."""
    path_clearance_spacing_s: float = 0.04
    """Spacing in s-units between clearance sample points."""
    path_clearance_probe_radius: float = 0.05
    """Probe sphere radius subtracted from point cloud distance."""
    path_clearance_safe: float = 0.20
    """Safe clearance value — below this the tube deadband narrows."""

    # --- Adaptive tube deadband ---
    tube_deadband_min: float = 0.05
    """Minimum contouring tube deadband (m)."""
    tube_deadband_max: float = 0.25
    """Maximum contouring tube deadband (m)."""
    tube_deadband_gain: float = 1.0
    """Gain for expanding deadband when clearance is low."""
    min_clearance_update_interval: int = 4
    """Update interval in simulation steps for logging-only min_clearance."""

    # --- Virtual progress (s_hat) ---
    s_hat_initial_offset_s: float = 0.03
    """Initial s_hat value after episode reset (slightly ahead of s=0)."""
    s_hat_nominal_speed_mps: float = 0.35
    """Nominal virtual advance speed (m/s along arc length)."""
    s_hat_speed_max_mps: float = 3.0
    """Safety cap on virtual advance speed (m/s). Rarely active with gap catch-up."""
    s_hat_filter_tau: float = 0.15
    """First-order filter time constant (s) for s_hat_dot smoothing."""
    s_hat_catchup_tau: float = 0.15
    """Time constant (s) for s_proj→s_hat catch-up: v_catchup = gap / tau."""
    s_hat_lag_deadband_m: float = 0.05
    """Deadband (m) for lag error before the lag gate starts decaying."""
    s_hat_contouring_deadband_m: float = 0.03
    """Deadband (m) for contouring error before the contouring gate starts decaying."""
    s_hat_lag_sigma_m: float = 0.15
    """Gaussian sigma (m) for lag gate decay beyond the deadband."""
    s_hat_contouring_sigma_m: float = 0.12
    """Gaussian sigma (m) for contouring gate decay beyond the deadband."""
    s_hat_contouring_gate_power: float = 0.3
    """Power exponent for contouring gate (0–1). Lower = weaker contouring effect.
    v_track = v_nom * g_lag * g_contour^power.  At 0.3, g_contour=0.1 still
    gives 0.50 pace, so s_hat keeps moving during lateral obstacle detours."""
    s_hat_pos_sigma: float = 0.12
    """Deprecated: kept for config compat."""

    # --- Tail release (末段虚拟进度释放) ---
    s_hat_tail_release_start_s: float = 0.85
    """s_hat threshold where brake gates start releasing."""
    s_hat_tail_release_start_dist_m: float = 0.60
    """EE-to-goal distance (m) below which brake release also activates."""
    s_hat_tail_nominal_floor_ratio: float = 0.70
    """Minimum v_track as fraction of v_nom during tail release."""

    # --- Reward reference (s_ref) ---
    ori_weight_switch_distance: float = 0.10
    """Position error threshold (m) for orientation weight sigmoid gating."""
    ori_weight_k: float = 20.0
    """Sigmoid steepness for orientation weight gating."""

    # --- Base reachability (prioritized task reward) ---
    arm_workspace_radius: float = 0.55
    """Effective planar workspace radius (m) of the arm.
    Serves as the sigmoid switching centre for base-vs-arm task weight."""
    base_switch_k: float = 12.0
    """Sigmoid steepness for base/arm weight transition."""

    # Visualization
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
                "frame": FRAME_MARKER_CFG.markers[
                    "frame"
                ].replace(scale=(0.12, 0.12, 0.12))
            },
        )
    )
    trajectory_visualizer_cfg: VisualizationMarkersCfg = (
        VisualizationMarkersCfg(
            prim_path="/Visuals/Command/trajectory",
            markers={
                "path_sphere": sim_utils.SphereCfg(
                    radius=0.015,
                    visual_material=(
                        sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 0.8, 1.0),
                            opacity=0.9,
                        )
                    ),
                ),
            },
        )
    )
    num_trajectory_samples: int = 32

    def __post_init__(self):
        super().__post_init__()
        if self.class_type is None:
            self.class_type = EETrajectoryCommand


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Helper: Quaternion Slerp (batched, pure-torch)                      ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def _quat_slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Batched quaternion slerp.

    Args:
        q0: [N, 4] start quaternion (w, x, y, z).
        q1: [N, 4] end quaternion (w, x, y, z).
        t:  [N] or [N, 1] interpolation parameter in [0, 1].

    Returns:
        [N, 4] interpolated quaternion.
    """
    if t.dim() == 2:
        t = t.squeeze(-1)
    # Ensure positive dot (shortest arc)
    dot = (q0 * q1).sum(dim=-1)  # [N]
    q1 = torch.where(dot.unsqueeze(-1) < 0, -q1, q1)
    dot = dot.abs().clamp(max=1.0 - 1e-6)

    theta = torch.acos(dot)  # [N]
    sin_theta = torch.sin(theta).clamp(min=1e-8)

    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta

    # Fallback to lerp for near-identical quaternions
    small = theta.abs() < 1e-6
    w0 = torch.where(small, 1.0 - t, w0)
    w1 = torch.where(small, t, w1)

    q = w0.unsqueeze(-1) * q0 + w1.unsqueeze(-1) * q1
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Helper: Bezier evaluation (pure torch, GPU-friendly)                ║
# ╚═══════════════════════════════════════════════════════════════════════╝


def _eval_bezier(ctrl_pts: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Evaluate Bezier curve via De Casteljau (any degree, batched).

    Args:
        ctrl_pts: [N, K+1, 3] control points (K = degree).
        u: [L] parameter values in [0, 1].

    Returns:
        [N, L, 3] evaluated positions.
    """
    degree = ctrl_pts.shape[1] - 1
    u = u.view(1, -1, 1)
    one_minus_u = 1.0 - u

    if degree == 1:
        p0 = ctrl_pts[:, 0].unsqueeze(1)
        p1 = ctrl_pts[:, 1].unsqueeze(1)
        return one_minus_u * p0 + u * p1
    if degree == 2:
        p0 = ctrl_pts[:, 0].unsqueeze(1)
        p1 = ctrl_pts[:, 1].unsqueeze(1)
        p2 = ctrl_pts[:, 2].unsqueeze(1)
        return (
            one_minus_u.square() * p0
            + 2.0 * one_minus_u * u * p1
            + u.square() * p2
        )
    if degree == 3:
        p0 = ctrl_pts[:, 0].unsqueeze(1)
        p1 = ctrl_pts[:, 1].unsqueeze(1)
        p2 = ctrl_pts[:, 2].unsqueeze(1)
        p3 = ctrl_pts[:, 3].unsqueeze(1)
        return (
            one_minus_u.pow(3) * p0
            + 3.0 * one_minus_u.square() * u * p1
            + 3.0 * one_minus_u * u.square() * p2
            + u.pow(3) * p3
        )

    pts = ctrl_pts.unsqueeze(2).expand(-1, -1, u.shape[1], -1).clone()
    u_exp = u.unsqueeze(1)
    for k in range(ctrl_pts.shape[1] - 1, 0, -1):
        pts[:, :k] = (1.0 - u_exp) * pts[:, :k] + u_exp * pts[:, 1 : k + 1]
    return pts[:, 0]


# ╔═══════════════════════════════════════════════════════════════════════╗
# ║  Command Generator                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════╝


class EETrajectoryCommand(CommandTerm):
    """SE(3) Bezier path + monotonic virtual progress command generator.

    Single path / single goal / episodic task.
    Position follows arc-length-normalized Bezier p(s);
    Orientation follows quaternion slerp R(s);
    Progress uses two layers:
      s_proj (geometric, can retreat) → s_hat (monotonic, filtered)
      → s_ref = s_hat → preview stack → q1 = s_ref for local tracking.
    """

    cfg: EETrajectoryCommandCfg

    def __init__(self, cfg: EETrajectoryCommandCfg, env: ManagerBasedRLEnv):
        requested_debug_vis = cfg.debug_vis
        cfg.debug_vis = False
        super().__init__(cfg, env)
        cfg.debug_vis = requested_debug_vis

        self._env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # Sampling ranges (cylindrical coordinates)
        self.rho_xy_range = cfg.ranges_pos["rho_xy"]
        self.yaw_range = cfg.ranges_pos["yaw"]
        self.z_range = cfg.z_range

        # Debug counters for sampling fallback ratio
        self._last_sampling_valid_ratio = 1.0
        self._last_sampling_fallback_ratio = 0.0

        N = self.num_envs
        M = cfg.num_path_cache_samples
        self._env_indices = torch.arange(N, device=self.device)
        self._env_repeat_index_cache: dict[int, torch.Tensor] = {}
        self._path_s_grid_batch_cache: dict[int, torch.Tensor] = {}
        self._path_s_grid_flat_cache: dict[int, torch.Tensor] = {}
        self._control_point_frac_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        self._sphere_lidar_cache_ref = None
        self._compute_sphere_lidar_fn = None
        self._projection_window_backward = max(
            int(cfg.projection_window_backward_segments), 0
        )
        self._projection_window_forward = max(
            int(cfg.projection_window_forward_segments), 0
        )
        self._projection_local_segment_offsets = torch.arange(
            -self._projection_window_backward,
            self._projection_window_forward + 1,
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0)
        self._arm_base_offset = torch.tensor(cfg.arm_base_offset, device=self.device)
        self._collision_box_lower = torch.tensor(cfg.collision_box_lower, device=self.device)
        self._collision_box_upper = torch.tensor(cfg.collision_box_upper, device=self.device)
        self._world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._fallback_lateral = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        self._quat_identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._bezier_dense_u = torch.linspace(
            0.0, 1.0, cfg.num_bezier_dense_samples, device=self.device
        )

        # =====================================================================
        # Command buffers
        # =====================================================================
        self.lie_command_b = torch.zeros(N, 6, device=self.device)

        self.pose_command_b = torch.zeros(N, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0

        self.pose_command_w = torch.zeros(N, 7, device=self.device)
        self.pose_command_w[:, 3] = 1.0

        # =====================================================================
        # Path endpoints in {World}
        # =====================================================================
        self._T_init_w = torch.zeros(N, 7, device=self.device)
        self._T_init_w[:, 3] = 1.0

        self._T_end_w = torch.zeros(N, 7, device=self.device)
        self._T_end_w[:, 3] = 1.0

        # =====================================================================
        # Path cache  [N, M, ...]
        # =====================================================================
        self._path_s_grid = torch.linspace(0.0, 1.0, M, device=self.device)  # [M]
        self._path_pos_w = torch.zeros(N, M, 3, device=self.device)
        self._path_quat_w = torch.zeros(N, M, 4, device=self.device)
        self._path_quat_w[:, :, 0] = 1.0
        self._path_tangent_w = torch.zeros(N, M, 3, device=self.device)
        self._projection_window_indices = torch.zeros(
            N,
            self._projection_local_segment_offsets.shape[1],
            device=self.device,
            dtype=torch.long,
        )

        # Total arc length per env
        self._path_total_length = torch.ones(N, device=self.device)

        # =====================================================================
        # Progress state
        # =====================================================================
        self._s_proj = torch.zeros(N, device=self.device)
        self._prev_s_proj = torch.zeros(N, device=self.device)
        self._robot_progress_delta = torch.zeros(N, device=self.device)
        self._contouring_error = torch.zeros(N, device=self.device)           # at s_proj
        self._path_orientation_error = torch.zeros(N, device=self.device)      # at s_proj
        self._lag_error = torch.zeros(N, device=self.device)                   # at s_ref (signed)
        self._geometric_contouring_error = torch.zeros(N, device=self.device)  # at s_ref

        # ── Virtual progress (s_hat) ──
        self._s_hat = torch.full((N,), cfg.s_hat_initial_offset_s, device=self.device)
        self._prev_s_hat = torch.full((N,), cfg.s_hat_initial_offset_s, device=self.device)
        self._s_hat_dot = torch.zeros(N, device=self.device)

        # =====================================================================
        # Preview stack  [N, num_preview_points]
        # =====================================================================
        P = cfg.num_preview_points
        self._preview_s = torch.zeros(N, P, device=self.device)
        self._preview_pos_w = torch.zeros(N, P, 3, device=self.device)
        self._preview_quat_w = torch.zeros(N, P, 4, device=self.device)
        self._preview_quat_w[:, :, 0] = 1.0
        self._preview_step_offsets = torch.arange(P, device=self.device, dtype=torch.float32).unsqueeze(0)
        clearance_steps = torch.arange(
            1, cfg.path_clearance_num_samples + 1, device=self.device, dtype=torch.float32
        )
        self._path_clearance_step_offsets = (
            clearance_steps * cfg.path_clearance_spacing_s
        ).unsqueeze(0)

        # Path-ahead clearance / adaptive tube
        self._path_ahead_clearance = torch.full((N,), 1.0, device=self.device)
        self._tube_deadband_dynamic = torch.full((N,), cfg.tube_deadband_min, device=self.device)
        self._path_clearance_query_s = torch.zeros(
            N, cfg.path_clearance_num_samples, device=self.device
        )
        self._path_clearance_values = torch.full(
            (N, cfg.path_clearance_num_samples), 1e6, device=self.device
        )

        # Reward reference (s_ref based)
        self._s_ref = torch.zeros(N, device=self.device)
        self._prev_position_error = torch.zeros(N, device=self.device)
        self._prev_orientation_error = torch.zeros(N, device=self.device)
        self._has_prev_ref_error = torch.zeros(
            N, dtype=torch.bool, device=self.device
        )
        self._position_progress_delta = torch.zeros(N, device=self.device)
        self._orientation_progress_delta = torch.zeros(N, device=self.device)
        # Final-goal approach (delta reward)
        self._prev_final_position_error = torch.zeros(N, device=self.device)
        self._has_prev_final_position_error = torch.zeros(
            N, dtype=torch.bool, device=self.device
        )
        self._final_goal_progress_delta = torch.zeros(N, device=self.device)

        # Base reachability (prioritized task reward)
        self._rho_base = torch.zeros(N, device=self.device)
        self._prev_rho_base = torch.zeros(N, device=self.device)
        self._has_prev_rho_base = torch.zeros(
            N, dtype=torch.bool, device=self.device
        )
        self._rho_base_delta = torch.zeros(N, device=self.device)
        self._base_task_weight = torch.zeros(N, device=self.device)

        # Min clearance for logging
        self._min_clearance = torch.full((N,), 1e6, device=self.device)
        self._min_clearance_update_interval = max(int(cfg.min_clearance_update_interval), 1)
        self._last_min_clearance_step = -self._min_clearance_update_interval

        # Tail release state
        self._s_hat_tail_alpha = torch.zeros(N, device=self.device)
        self._s_hat_tail_alpha_s = torch.zeros(N, device=self.device)
        self._s_hat_tail_alpha_d = torch.zeros(N, device=self.device)

        # Episode-level mean errors (for quality assessment)
        self._ep_sum_contouring = torch.zeros(N, device=self.device)
        self._ep_sum_proj_ori_error = torch.zeros(N, device=self.device)
        self._ep_step_count = torch.zeros(N, device=self.device)

        # =====================================================================
        # Metrics
        # =====================================================================
        self.metrics["position_error"] = torch.zeros(N, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(N, device=self.device)
        self.metrics["final_position_error"] = torch.full((N,), 1e6, device=self.device)
        self.metrics["final_orientation_error"] = torch.full((N,), 1e6, device=self.device)
        self.metrics["contouring_error"] = torch.zeros(N, device=self.device)
        self.metrics["path_orientation_error"] = torch.zeros(N, device=self.device)
        self.metrics["lag_error_signed"] = torch.zeros(N, device=self.device)
        self.metrics["lag_error_abs"] = torch.zeros(N, device=self.device)
        self.metrics["s_proj"] = torch.zeros(N, device=self.device)
        self.metrics["s_hat"] = torch.full((N,), cfg.s_hat_initial_offset_s, device=self.device)
        self.metrics["s_hat_dot"] = torch.zeros(N, device=self.device)
        self.metrics["robot_progress_delta"] = torch.zeros(N, device=self.device)
        self.metrics["min_clearance"] = torch.full((N,), 1e6, device=self.device)
        # Reward reference metrics (s_ref based)
        self.metrics["s_ref"] = torch.zeros(N, device=self.device)
        self.metrics["position_progress_delta"] = torch.zeros(N, device=self.device)
        self.metrics["orientation_progress_delta"] = torch.zeros(N, device=self.device)
        self.metrics["final_goal_progress_delta"] = torch.zeros(N, device=self.device)
        # Path-ahead clearance metrics
        self.metrics["path_ahead_clearance"] = torch.ones(N, device=self.device)
        self.metrics["tube_deadband_dynamic"] = torch.full((N,), self.cfg.tube_deadband_min, device=self.device)
        # Episode-level mean errors (used by curriculum quality check)
        # ep_mean_contouring: based on contouring_error (s_proj)
        # ep_mean_ori_error: based on path_orientation_error (s_proj)
        self.metrics["ep_mean_contouring"] = torch.zeros(N, device=self.device)
        self.metrics["ep_mean_ori_error"] = torch.zeros(N, device=self.device)
        # Base reachability metrics
        self.metrics["base_reachability_distance"] = torch.zeros(N, device=self.device)
        self.metrics["base_task_weight"] = torch.zeros(N, device=self.device)

        # =====================================================================
        # Visualization
        # =====================================================================
        self._vis_step_counter = 0

        # =====================================================================
        # Static obstacle height grid
        # =====================================================================
        self._has_height_grid = False
        if cfg.check_obstacle_collision:
            self._build_obstacle_height_grid()

        if requested_debug_vis:
            self.set_debug_vis(True)

    def __str__(self) -> str:
        cfg = self.cfg
        return (
            f"EETrajectoryCommand (Bezier + Projection-Driven Progress):\n"
            f"  Command dimension: {tuple(self.command.shape[1:])}\n"
            f"  Path cache samples: {cfg.num_path_cache_samples}\n"
            f"  Control points: {cfg.num_control_points}\n"
            f"  Preview: N={cfg.num_preview_points}, spacing={cfg.preview_spacing_s_base}\n"
            f"  Sampling (Cylindrical): rho_xy={self.rho_xy_range} m, "
            f"yaw={self.yaw_range} rad\n"
            f"  Height constraint: z={self.z_range} m\n"
        )

    @property
    def command(self) -> torch.Tensor:
        return self.pose_command_b

    # =========================================================================
    # Core Methods
    # =========================================================================

    def _get_projected_frame(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if env_ids is None:
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
        else:
            root_pos = self.robot.data.root_pos_w[env_ids]
            root_quat = self.robot.data.root_quat_w[env_ids]

        proj_pos = root_pos.clone()
        proj_pos[:, 2] = 0.0

        w = root_quat[..., 0]
        x = root_quat[..., 1]
        y = root_quat[..., 2]
        z = root_quat[..., 3]
        yaw = torch.atan2(
            2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
        )
        zeros = torch.zeros_like(yaw)
        proj_quat = quat_from_euler_xyz(zeros, zeros, yaw)
        return proj_pos, proj_quat

    # -----------------------------------------------------------------
    # Static obstacle collision helpers
    # -----------------------------------------------------------------

    def _build_obstacle_height_grid(self):
        try:
            from pxr import UsdGeom
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            mesh_prim_path = "/World/ground/terrain/mesh"
            mesh_prim = stage.GetPrimAtPath(mesh_prim_path)

            if not mesh_prim.IsValid():
                print(
                    f"[WARN] EETrajectoryCommand: terrain mesh not found "
                    f"at {mesh_prim_path}, disabling obstacle collision check"
                )
                return

            mesh = UsdGeom.Mesh(mesh_prim)
            points = np.array(mesh.GetPointsAttr().Get(), dtype=np.float32)

            if len(points) == 0:
                print("[WARN] EETrajectoryCommand: terrain mesh has 0 vertices")
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
            r_cells = max(1, int(self.cfg.obstacle_collision_margin / h_scale + 0.5))
            kernel_size = 2 * r_cells + 1
            self._height_grid_dilated = F.max_pool2d(
                self._height_grid.unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=r_cells,
            ).squeeze(0).squeeze(0)
            self._has_height_grid = True

            n_obs = int((height_grid > 0.05).sum())
            print(
                f"[INFO] EETrajectoryCommand: built obstacle height grid "
                f"{grid_w}x{grid_h}, {n_obs} obstacle cells"
            )
        except Exception as e:
            print(f"[WARN] EETrajectoryCommand: failed to build height grid: {e}")

    def _check_obstacle_collision(self, pos_w: torch.Tensor) -> torch.Tensor:
        ix = torch.round(
            (pos_w[:, 0] - self._grid_origin[0]) / self._grid_h_scale
        ).long()
        iy = torch.round(
            (pos_w[:, 1] - self._grid_origin[1]) / self._grid_h_scale
        ).long()

        x_max_idx = self._height_grid_dilated.shape[0] - 1
        y_max_idx = self._height_grid_dilated.shape[1] - 1
        ix = ix.clamp(0, x_max_idx)
        iy = iy.clamp(0, y_max_idx)
        max_height = self._height_grid_dilated[ix, iy]

        collides = (
            (max_height > 0.05)
            & (pos_w[:, 2] < max_height + self.cfg.obstacle_collision_margin)
        )
        return collides

    def _sample_target_pose(
        self,
        n: int,
        env_ids: torch.Tensor,
        proj_pos: torch.Tensor | None = None,
        proj_quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = self.device
        arm_offset = self._arm_base_offset
        collision_lower = self._collision_box_lower
        collision_upper = self._collision_box_upper
        max_attempts = self.cfg.max_resample_attempts

        pose = torch.zeros(n, 7, device=device)
        pose[:, 3] = 1.0

        rho_xy = torch.empty(n, max_attempts, device=device).uniform_(*self.rho_xy_range)
        yaw = torch.empty(n, max_attempts, device=device).uniform_(*self.yaw_range)
        z = torch.empty(n, max_attempts, device=device).uniform_(*self.z_range)

        x = rho_xy * torch.cos(yaw) + arm_offset[0]
        y = rho_xy * torch.sin(yaw) + arm_offset[1]

        valid = ~(
            (x >= collision_lower[0]) & (x <= collision_upper[0]) &
            (y >= collision_lower[1]) & (y <= collision_upper[1]) &
            (z >= collision_lower[2]) & (z <= collision_upper[2])
        )

        if self._has_height_grid and proj_pos is not None and proj_quat is not None:
            pos_proj = torch.stack([x, y, z], dim=-1)
            num_candidates = n * max_attempts
            pos_w, _ = combine_frame_transforms(
                proj_pos.unsqueeze(1).expand(-1, max_attempts, -1).reshape(num_candidates, 3),
                proj_quat.unsqueeze(1).expand(-1, max_attempts, -1).reshape(num_candidates, 4),
                pos_proj.reshape(num_candidates, 3),
                self._quat_identity.unsqueeze(0).expand(num_candidates, -1),
            )
            obstacle_hit = self._check_obstacle_collision(pos_w).reshape(n, max_attempts)
            valid = valid & (~obstacle_hit)

        has_valid = valid.any(dim=1)
        first_valid_idx = valid.to(torch.int64).argmax(dim=1)
        env_idx = torch.arange(n, device=device)

        if has_valid.any():
            chosen_env_idx = env_idx[has_valid]
            chosen_attempt_idx = first_valid_idx[has_valid]
            pose[chosen_env_idx, 0] = x[chosen_env_idx, chosen_attempt_idx]
            pose[chosen_env_idx, 1] = y[chosen_env_idx, chosen_attempt_idx]
            pose[chosen_env_idx, 2] = z[chosen_env_idx, chosen_attempt_idx]

        needs_sample = ~has_valid

        # Fallback: use midpoint of valid cylindrical workspace
        remaining_indices = torch.where(needs_sample)[0]
        num_fallback = remaining_indices.numel()
        if num_fallback > 0:
            rho_mid = (self.rho_xy_range[0] + self.rho_xy_range[1]) / 2.0
            z_mid = (self.z_range[0] + self.z_range[1]) / 2.0
            yaw = torch.empty(num_fallback, device=device).uniform_(*self.yaw_range)
            pose[remaining_indices, 0] = rho_mid * torch.cos(yaw) + arm_offset[0]
            pose[remaining_indices, 1] = rho_mid * torch.sin(yaw) + arm_offset[1]
            pose[remaining_indices, 2] = z_mid

        # Update sampling debug metrics
        self._last_sampling_fallback_ratio = num_fallback / max(n, 1)
        self._last_sampling_valid_ratio = 1.0 - self._last_sampling_fallback_ratio

        if self.cfg.sample_uniform_orientation:
            u1 = torch.rand(n, device=device)
            u2 = torch.rand(n, device=device) * 2 * torch.pi
            u3 = torch.rand(n, device=device) * 2 * torch.pi
            sqrt_1_u1 = torch.sqrt(1 - u1)
            sqrt_u1 = torch.sqrt(u1)
            quat = torch.stack([
                sqrt_1_u1 * torch.sin(u2),
                sqrt_1_u1 * torch.cos(u2),
                sqrt_u1 * torch.sin(u3),
                sqrt_u1 * torch.cos(u3),
            ], dim=-1)
            pose[:, 3:7] = quat[:, [3, 0, 1, 2]]
        else:
            ranges = self.cfg.delta_euler_ranges
            roll = torch.empty(n, device=device).uniform_(*ranges["roll"])
            pitch = torch.empty(n, device=device).uniform_(*ranges["pitch"])
            yaw = torch.empty(n, device=device).uniform_(*ranges["yaw"])
            quat_noise = quat_from_euler_xyz(roll, pitch, yaw)
            pose[:, 3:7] = quat_noise

        return pose

    def _get_batched_path_s_grid(self, batch_size: int) -> torch.Tensor:
        """Return cached [batch_size, M] copies of the path s-grid."""
        cached = self._path_s_grid_batch_cache.get(batch_size)
        if cached is None:
            cached = self._path_s_grid.unsqueeze(0).expand(batch_size, -1).contiguous()
            self._path_s_grid_batch_cache[batch_size] = cached
        return cached

    def _get_flat_path_s_grid(self, batch_size: int) -> torch.Tensor:
        """Return cached flattened path s-grid repeated across the batch."""
        cached = self._path_s_grid_flat_cache.get(batch_size)
        if cached is None:
            cached = self._get_batched_path_s_grid(batch_size).reshape(-1)
            self._path_s_grid_flat_cache[batch_size] = cached
        return cached

    def _get_control_point_fracs(self, num_cp: int, dtype: torch.dtype) -> torch.Tensor:
        """Return cached [1, num_cp, 1] control-point fractions."""
        cache_key = (num_cp, dtype)
        cached = self._control_point_frac_cache.get(cache_key)
        if cached is None:
            cached = (
                torch.arange(1, num_cp + 1, device=self.device, dtype=dtype)
                .view(1, num_cp, 1)
                / (num_cp + 1.0)
            )
            self._control_point_frac_cache[cache_key] = cached
        return cached

    def _get_sphere_lidar_helpers(self):
        """Lazily cache sphere-lidar cache and compute function references."""
        if (
            self._sphere_lidar_cache_ref is not None
            and self._compute_sphere_lidar_fn is not None
        ):
            return self._sphere_lidar_cache_ref, self._compute_sphere_lidar_fn
        try:
            from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.mdp.observations import (
                _compute_sphere_lidar,
                _sphere_lidar_cache,
            )
            self._compute_sphere_lidar_fn = _compute_sphere_lidar
            self._sphere_lidar_cache_ref = _sphere_lidar_cache
        except Exception:
            return None, None
        return self._sphere_lidar_cache_ref, self._compute_sphere_lidar_fn

    # =====================================================================
    # (A) Bezier path generation with arc-length normalisation
    # =====================================================================

    def _build_path_cache(self, env_ids: torch.Tensor):
        """Build Bezier path with arc-length normalisation for given envs.

        Steps:
          1. Construct control points (start + intermediates + end).
          2. Evaluate dense Bezier p(u) with u ∈ [0, 1].
          3. Compute cumulative arc length → L(u); normalise to s = L/L_total.
          4. Resample to uniform s-grid → _path_pos_w, _path_tangent_w.
          5. Slerp orientation independently → _path_quat_w.
        """
        n = len(env_ids)
        if n == 0:
            return
        device = self.device
        cfg = self.cfg
        M = cfg.num_path_cache_samples
        D = cfg.num_bezier_dense_samples

        pos_start = self._T_init_w[env_ids, :3]   # [n, 3]
        pos_end = self._T_end_w[env_ids, :3]       # [n, 3]
        quat_start = self._T_init_w[env_ids, 3:7]  # [n, 4]
        quat_end = self._T_end_w[env_ids, 3:7]     # [n, 4]
        path_delta = pos_end - pos_start

        num_cp = max(0, min(cfg.num_control_points, 2))
        s_grid = self._path_s_grid
        s_grid_batch = self._get_batched_path_s_grid(n)
        s_grid_flat = self._get_flat_path_s_grid(n)

        # ── Build control points [n, K+2, 3] ──
        if num_cp == 0:
            total_len = path_delta.norm(dim=-1).clamp(min=1e-6)
            self._path_total_length[env_ids] = total_len

            path_pos = pos_start.unsqueeze(1) + s_grid.view(1, M, 1) * path_delta.unsqueeze(1)
            self._path_pos_w[env_ids] = path_pos

            tangent = path_delta / total_len.unsqueeze(-1)
            self._path_tangent_w[env_ids] = tangent.unsqueeze(1).expand(-1, M, -1)
            quat_start_exp = quat_start.unsqueeze(1).expand(-1, M, -1).reshape(-1, 4)
            quat_end_exp = quat_end.unsqueeze(1).expand(-1, M, -1).reshape(-1, 4)
            self._path_quat_w[env_ids] = _quat_slerp(
                quat_start_exp, quat_end_exp, s_grid_flat
            ).reshape(n, M, 4)
            return
        else:
            # Direction helpers
            fwd = path_delta  # [n, 3]
            fwd_len = fwd.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            fwd_unit = fwd / fwd_len

            # Construct a lateral basis vector via cross with world-up
            up = self._world_up.expand(n, 3)
            lat = torch.cross(fwd_unit, up, dim=-1)
            lat_len = lat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            # Fallback when fwd is near-vertical
            fallback = self._fallback_lateral.expand(n, 3)
            is_degenerate = (lat_len.squeeze(-1) < 1e-4)
            lat = torch.where(is_degenerate.unsqueeze(-1), fallback, lat / lat_len)

            vert = self._world_up.expand(n, 3)

            frac = self._get_control_point_fracs(num_cp, pos_start.dtype)
            base_pts = pos_start.unsqueeze(1) + frac * fwd.unsqueeze(1)
            lat_off = torch.empty(n, num_cp, 1, device=device).uniform_(
                *cfg.control_point_lateral_range
            )
            vert_off = torch.empty(n, num_cp, 1, device=device).uniform_(
                *cfg.control_point_vertical_range
            )
            ctrl_mid = (
                base_pts
                + lat_off * lat.unsqueeze(1)
                + vert_off * vert.unsqueeze(1)
            )
            ctrl = torch.cat(
                [pos_start.unsqueeze(1), ctrl_mid, pos_end.unsqueeze(1)],
                dim=1,
            )

        # ── Dense Bezier evaluation ──
        pts_dense = _eval_bezier(ctrl, self._bezier_dense_u)  # [n, D, 3]

        # ── Cumulative arc length ──
        seg_len = (pts_dense[:, 1:] - pts_dense[:, :-1]).norm(dim=-1)  # [n, D-1]
        cum_len = torch.cat(
            [torch.zeros(n, 1, device=device), torch.cumsum(seg_len, dim=-1)],
            dim=1,
        )
        total_len = cum_len[:, -1].clamp(min=1e-6)  # [n]
        self._path_total_length[env_ids] = total_len

        # s values at dense samples: s = L(u) / L_total
        s_dense = cum_len / total_len.unsqueeze(-1)  # [n, D]

        # ── Resample to uniform s-grid via linear interpolation ──
        # For each env, interpolate pts_dense at uniform s_grid values
        # We use torch-based piecewise linear interpolation
        path_pos = self._interp_1d_batch(s_dense, pts_dense, s_grid_batch)  # [n, M, 3]
        self._path_pos_w[env_ids] = path_pos

        # ── Tangent vectors via finite differences ──
        tangent = torch.empty_like(path_pos)
        tangent[:, :-1] = path_pos[:, 1:] - path_pos[:, :-1]
        tangent[:, -1] = tangent[:, -2]
        tang_norm = tangent.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        self._path_tangent_w[env_ids] = tangent / tang_norm

        # ── Orientation: quaternion slerp over s-grid (decoupled from position) ──
        quat_start_exp = quat_start.unsqueeze(1).expand(-1, M, -1).reshape(-1, 4)
        quat_end_exp = quat_end.unsqueeze(1).expand(-1, M, -1).reshape(-1, 4)
        self._path_quat_w[env_ids] = _quat_slerp(
            quat_start_exp, quat_end_exp, s_grid_flat
        ).reshape(n, M, 4)

    @staticmethod
    def _interp_1d_batch(
        x_known: torch.Tensor, y_known: torch.Tensor, x_query: torch.Tensor
    ) -> torch.Tensor:
        """Batched 1D linear interpolation (pure torch).

        Args:
            x_known: [N, D] monotone increasing x coordinates.
            y_known: [N, D, C] corresponding y values.
            x_query: [N, M] query x values.

        Returns:
            [N, M, C] interpolated values.
        """
        N, D, C = y_known.shape
        M = x_query.shape[1]

        # searchsorted: find right boundary index for each query
        # x_known: [N, D], x_query: [M] → expand for batched search
        # Use searchsorted per-batch
        idx_right = torch.searchsorted(x_known.contiguous(), x_query)  # [N, M] in [0, D]
        idx_right = idx_right.clamp(min=1, max=D - 1)
        idx_left = idx_right - 1

        # Gather x and y at left/right
        x_left = x_known.gather(1, idx_left)   # [N, M]
        x_right = x_known.gather(1, idx_right)  # [N, M]

        dx = (x_right - x_left).clamp(min=1e-8)
        t = ((x_query - x_left) / dx).clamp(0.0, 1.0)  # [N, M]

        # Gather y values
        il = idx_left.unsqueeze(-1).expand(N, M, C)
        ir = idx_right.unsqueeze(-1).expand(N, M, C)
        y_left = y_known.gather(1, il)    # [N, M, C]
        y_right = y_known.gather(1, ir)   # [N, M, C]

        return y_left + t.unsqueeze(-1) * (y_right - y_left)

    # =====================================================================
    # (B) Continuous interpolation at arbitrary s
    # =====================================================================

    def _interpolate_path_position(self, s_query: torch.Tensor) -> torch.Tensor:
        """Interpolate position at arbitrary s values.

        Args:
            s_query: [N] progress values in [0, 1].

        Returns:
            [N, 3] interpolated world position.
        """
        M = self.cfg.num_path_cache_samples
        s_query = s_query.clamp(0.0, 1.0)
        # Convert to continuous index
        idx_f = s_query * (M - 1)
        idx_left = idx_f.long().clamp(0, M - 2)
        idx_right = idx_left + 1
        t = idx_f - idx_left.float()  # fractional part

        env_idx = self._env_indices
        p_left = self._path_pos_w[env_idx, idx_left]    # [N, 3]
        p_right = self._path_pos_w[env_idx, idx_right]  # [N, 3]
        return p_left + t.unsqueeze(-1) * (p_right - p_left)

    def _interpolate_path_orientation(self, s_query: torch.Tensor) -> torch.Tensor:
        """Interpolate orientation at arbitrary s values via slerp.

        Args:
            s_query: [N] progress values in [0, 1].

        Returns:
            [N, 4] interpolated quaternion (w, x, y, z).
        """
        M = self.cfg.num_path_cache_samples
        s_query = s_query.clamp(0.0, 1.0)
        idx_f = s_query * (M - 1)
        idx_left = idx_f.long().clamp(0, M - 2)
        idx_right = idx_left + 1
        t = idx_f - idx_left.float()

        env_idx = self._env_indices
        q_left = self._path_quat_w[env_idx, idx_left]
        q_right = self._path_quat_w[env_idx, idx_right]
        return _quat_slerp(q_left, q_right, t)

    def _interpolate_path_tangent(self, s_query: torch.Tensor) -> torch.Tensor:
        """Interpolate tangent at arbitrary s values.

        Args:
            s_query: [N] progress values in [0, 1].

        Returns:
            [N, 3] interpolated unit tangent.
        """
        M = self.cfg.num_path_cache_samples
        s_query = s_query.clamp(0.0, 1.0)
        idx_f = s_query * (M - 1)
        idx_left = idx_f.long().clamp(0, M - 2)
        idx_right = idx_left + 1
        t = idx_f - idx_left.float()

        env_idx = self._env_indices
        t_left = self._path_tangent_w[env_idx, idx_left]
        t_right = self._path_tangent_w[env_idx, idx_right]
        tang = t_left + t.unsqueeze(-1) * (t_right - t_left)
        return tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _get_reference_pose_at_s(self, s_query: torch.Tensor):
        """Get full reference pose (pos, quat) at arbitrary s.

        Returns:
            (pos_w [N, 3], quat_w [N, 4])
        """
        return (
            self._interpolate_path_position(s_query),
            self._interpolate_path_orientation(s_query),
        )

    def _project_onto_segment_candidates(
        self,
        ee_pos_w: torch.Tensor,
        env_ids: torch.Tensor,
        segment_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project onto a subset of cached path segments for the given envs."""
        if env_ids is self._env_indices:
            path_pos = self._path_pos_w
        elif env_ids.numel() == self.num_envs and torch.equal(
            env_ids, self._env_indices
        ):
            path_pos = self._path_pos_w
        else:
            path_pos = self._path_pos_w[env_ids]
        if segment_indices is None:
            seg_start = path_pos[:, :-1]
            seg_end = path_pos[:, 1:]
        else:
            gather_idx = segment_indices.unsqueeze(-1).expand(-1, -1, 3)
            seg_start = path_pos.gather(1, gather_idx)
            seg_end = path_pos.gather(
                1, (segment_indices + 1).unsqueeze(-1).expand(-1, -1, 3)
            )

        seg_dir = seg_end - seg_start
        seg_len_sq = (seg_dir * seg_dir).sum(dim=-1).clamp(min=1e-12)
        to_ee = ee_pos_w.unsqueeze(1) - seg_start
        tau = ((to_ee * seg_dir).sum(dim=-1) / seg_len_sq).clamp(0.0, 1.0)

        closest = seg_start + tau.unsqueeze(-1) * seg_dir
        dist = (ee_pos_w.unsqueeze(1) - closest).norm(dim=-1)

        min_dist, min_local_idx = dist.min(dim=-1)
        best_tau = tau.gather(1, min_local_idx.unsqueeze(-1)).squeeze(-1)
        if segment_indices is None:
            best_seg_idx = min_local_idx
        else:
            best_seg_idx = segment_indices.gather(
                1, min_local_idx.unsqueeze(-1)
            ).squeeze(-1)

        denom = max(self.cfg.num_path_cache_samples - 1, 1)
        s_proj = (best_seg_idx.to(dtype=ee_pos_w.dtype) + best_tau) / denom
        return s_proj, min_dist, best_seg_idx

    # =====================================================================
    # (C) Continuous segment projection
    # =====================================================================

    def _project_onto_path_continuous(self, ee_pos_w: torch.Tensor):
        """Project EE positions onto path via local-window segment projection."""
        seg_count = self.cfg.num_path_cache_samples - 1
        if seg_count <= 0:
            self._s_proj.zero_()
            self._contouring_error.zero_()
            return self._s_proj, self._contouring_error

        if self._projection_window_indices.shape[1] >= seg_count:
            s_proj, min_dist, _ = self._project_onto_segment_candidates(
                ee_pos_w, self._env_indices
            )
            self._s_proj.copy_(s_proj)
            self._contouring_error.copy_(min_dist)
            return self._s_proj, self._contouring_error

        center_seg_idx = (self._prev_s_proj * seg_count).long().clamp(
            0, seg_count - 1
        )
        self._projection_window_indices.copy_(
            self._projection_local_segment_offsets.expand(self.num_envs, -1)
        )
        self._projection_window_indices.add_(center_seg_idx.unsqueeze(-1))
        self._projection_window_indices.clamp_(0, seg_count - 1)

        s_proj, min_dist, best_seg_idx = self._project_onto_segment_candidates(
            ee_pos_w, self._env_indices, self._projection_window_indices
        )

        left_bound = (center_seg_idx - self._projection_window_backward).clamp(
            min=0
        )
        right_bound = (center_seg_idx + self._projection_window_forward).clamp(
            max=seg_count - 1
        )
        needs_global_fallback = (
            ((best_seg_idx == left_bound) & (left_bound > 0))
            | ((best_seg_idx == right_bound) & (right_bound < seg_count - 1))
        )
        if torch.any(needs_global_fallback):
            fallback_env_ids = self._env_indices[needs_global_fallback]
            fallback_s_proj, fallback_dist, _ = self._project_onto_segment_candidates(
                ee_pos_w[needs_global_fallback], fallback_env_ids
            )
            s_proj[needs_global_fallback] = fallback_s_proj
            min_dist[needs_global_fallback] = fallback_dist

        self._s_proj.copy_(s_proj)
        self._contouring_error.copy_(min_dist)
        return self._s_proj, self._contouring_error


    # =====================================================================
    # (D) Geometric errors at reference s_ref
    # =====================================================================

    def _compute_geometric_path_errors(self, ee_pos_w: torch.Tensor):
        """Compute geometric lag and contouring errors at s_ref.

        Error vector convention: e = P(s_ref) - ee_pos_w.

        Decomposition at reference point P(s_ref) with unit tangent t_ref:
          lag_error_signed   = dot(e, t_ref)
          geometric_contouring_vec     = e - lag * t_ref
          contouring_error   = norm(geometric_contouring_vec)

        Sign of lag:
          positive → EE is BEHIND (needs to catch up)
          negative → EE is AHEAD
        """
        ref_pos = self._interpolate_path_position(self._s_ref)   # [N, 3]
        tangent = self._interpolate_path_tangent(self._s_ref)     # [N, 3]

        error_vec = ref_pos - ee_pos_w  # [N, 3]

        lag = (error_vec * tangent).sum(dim=-1)  # [N]
        perp = error_vec - lag.unsqueeze(-1) * tangent  # [N, 3]
        contouring = perp.norm(dim=-1)  # [N]

        self._lag_error.copy_(lag)
        self._geometric_contouring_error.copy_(contouring)

    # =====================================================================
    # (D-2) Virtual progress s_hat (monotonic, MPCC-style lag/contouring)
    # =====================================================================

    def _update_virtual_progress(self, ee_pos_w: torch.Tensor):
        """Update monotonic virtual progress s_hat.

        MPCC-style trackability-driven update at the s_hat reference point:

        1. Geometric decomposition at P(s_hat):
           error_vec   = P(s_hat) - ee_pos_w
           lag_signed  = dot(error_vec, tangent(s_hat))
                         >0 ⇒ EE behind s_hat,  <0 ⇒ EE ahead
           contour_mag = ||error_vec - lag_signed * tangent||

        2. Deadband + Gaussian gates (continuous, differentiable):
           lag_pos        = relu(lag_signed)          # only penalise "behind"
           contour_excess = relu(contour_mag - deadband_c)
           lag_excess     = relu(lag_pos     - deadband_l)
           g_contour = exp(-0.5 * (contour_excess / σ_c)²)
           g_lag     = exp(-0.5 * (lag_excess     / σ_l)²)

        3. Speed composition (lag-dominant):
           v_track   = v_nom/L * g_contour^w * g_lag   (w < 1 → contouring mild)
           v_catchup = relu(s_proj - s_hat) / τ_catch  (unconditional catch-up)
           v_raw     = clamp(v_track + v_catchup, max=v_max/L)

        4. First-order filter + monotonic integration:
           α         = dt / τ_filter
           s_hat_dot ← (1-α)*prev + α*v_raw
           s_hat     ← clamp(s_hat + s_hat_dot*dt, prev_s_hat, 1)
        """
        cfg = self.cfg
        dt = self._env.step_dt

        # --- 1. Geometric decomposition at P(s_hat) ---
        ref_pos_hat = self._interpolate_path_position(self._s_hat)   # [N, 3]
        ref_tang_hat = self._interpolate_path_tangent(self._s_hat)   # [N, 3]
        error_vec = ref_pos_hat - ee_pos_w                           # [N, 3]

        # Tangential (lag) and normal (contouring) decomposition
        lag_signed = (error_vec * ref_tang_hat).sum(dim=-1)          # [N]
        contour_vec = error_vec - lag_signed.unsqueeze(-1) * ref_tang_hat
        contour_mag = contour_vec.norm(dim=-1)                       # [N]

        # --- 2. Deadband + Gaussian gates ---
        lag_pos = torch.relu(lag_signed)  # only care about EE behind s_hat
        lag_excess = torch.relu(lag_pos - cfg.s_hat_lag_deadband_m)
        contour_excess = torch.relu(contour_mag - cfg.s_hat_contouring_deadband_m)

        sigma_l_sq = max(cfg.s_hat_lag_sigma_m ** 2, 1e-8)
        sigma_c_sq = max(cfg.s_hat_contouring_sigma_m ** 2, 1e-8)
        g_lag = torch.exp(-0.5 * lag_excess ** 2 / sigma_l_sq)
        g_contour = torch.exp(-0.5 * contour_excess ** 2 / sigma_c_sq)

        # --- 3. Speed composition ---
        L = self._path_total_length.clamp(min=1e-6)
        v_nom_s = cfg.s_hat_nominal_speed_mps / L
        v_max_s = cfg.s_hat_speed_max_mps / L

        # Trackability-gated nominal advance
        g_contour_eff = torch.pow(
            g_contour, cfg.s_hat_contouring_gate_power
        )

        # --- Tail release: fade out brake gates near path end ---
        goal_pos_w = self._T_end_w[:, :3]
        d_final = (ee_pos_w - goal_pos_w).norm(dim=-1)

        s_tail_range = max(
            1.0 - cfg.s_hat_tail_release_start_s, 1e-6
        )
        alpha_tail_s = (
            (self._s_hat - cfg.s_hat_tail_release_start_s)
            / s_tail_range
        ).clamp(0.0, 1.0)
        d_start = max(cfg.s_hat_tail_release_start_dist_m, 1e-6)
        alpha_tail_d = (
            (d_start - d_final) / d_start
        ).clamp(0.0, 1.0)
        alpha_tail = torch.maximum(alpha_tail_s, alpha_tail_d)

        # Store for metrics
        self._s_hat_tail_alpha.copy_(alpha_tail)
        self._s_hat_tail_alpha_s.copy_(alpha_tail_s)
        self._s_hat_tail_alpha_d.copy_(alpha_tail_d)

        # Release brake gates toward 1.0 in tail region
        g_lag_eff = (1.0 - alpha_tail) * g_lag + alpha_tail
        g_cont_eff_tail = (
            (1.0 - alpha_tail) * g_contour_eff + alpha_tail
        )

        v_track_raw = v_nom_s * g_lag_eff * g_cont_eff_tail

        # Tail nominal floor
        v_floor = (
            v_nom_s * cfg.s_hat_tail_nominal_floor_ratio
            * alpha_tail
        )
        v_track = torch.maximum(v_track_raw, v_floor)

        # Catch-up when s_proj has overtaken s_hat (robot genuinely ahead)
        gap_behind = (self._s_proj - self._s_hat).clamp(min=0.0)
        v_catchup = gap_behind / max(cfg.s_hat_catchup_tau, 1e-6)

        v_raw = (v_track + v_catchup).clamp(min=torch.zeros_like(v_max_s), max=v_max_s)

        # --- 4. First-order filter + monotonic integration ---
        alpha = min(dt / max(cfg.s_hat_filter_tau, 1e-6), 1.0)
        self._s_hat_dot.mul_(1.0 - alpha).add_(v_raw, alpha=alpha)

        self._prev_s_hat.copy_(self._s_hat)
        self._s_hat.add_(self._s_hat_dot, alpha=dt)
        torch.maximum(self._s_hat, self._prev_s_hat, out=self._s_hat)
        self._s_hat.clamp_(max=1.0)

    # =====================================================================
    # (E) Projection-driven progress + obstacle-aware preview stack
    # =====================================================================

    def _update_command(self):
        """Update command via projection-driven progress + monotonic s_hat.

        Data flow:
          s_proj (geometric projection, can retreat)
            → _update_virtual_progress → s_hat (monotonic, filtered)
            → s_ref = s_hat
            → preview stack sampled from s_hat with adaptive spacing
            → q1 = preview[0] for local tracking
        """
        if not self.robot.is_initialized:
            return

        cfg = self.cfg

        # 0. Compute min_clearance for logging only
        self._maybe_update_min_clearance()

        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]

        # 1. Continuous segment projection -> s_proj, contouring_error
        self._prev_s_proj.copy_(self._s_proj)
        s_proj, _ = self._project_onto_path_continuous(ee_pos_w)
        # Signed delta (can be negative when EE retreats)
        self._robot_progress_delta.copy_(s_proj)
        self._robot_progress_delta.sub_(self._prev_s_proj)
        self._robot_progress_delta.clamp_(-0.1, 0.1)

        # 2. Projected orientation error at s_proj (for curriculum quality)
        proj_quat = self._interpolate_path_orientation(self._s_proj)
        _, proj_rot_err_vec = compute_pose_error(
            self._interpolate_path_position(self._s_proj), proj_quat,
            ee_pos_w, ee_quat_w,
        )
        self._path_orientation_error.copy_(proj_rot_err_vec.norm(dim=-1))

        # 3. Update virtual progress s_hat (monotonic)
        self._update_virtual_progress(ee_pos_w)

        # 4. Contouring-error driven adaptive preview spacing
        spacing_s = (
            cfg.preview_spacing_s_base
            / (1.0 + cfg.preview_spacing_error_gain * self._contouring_error)
        ).clamp(cfg.preview_spacing_s_min, cfg.preview_spacing_s_max)

        # 5. Generate preview points from s_hat (not s_proj)
        self._preview_s.copy_(self._preview_step_offsets)
        self._preview_s.mul_(spacing_s.unsqueeze(1))
        self._preview_s.add_(self._s_hat.unsqueeze(1))
        self._preview_s.clamp_(0.0, 1.0)

        # 6. Compute preview positions and orientations (path-based)
        self._preview_pos_w.copy_(self.get_path_positions_at_s(self._preview_s))
        self._preview_quat_w.copy_(self.get_path_orientations_at_s(self._preview_s))

        # 7. s_ref = s_hat (first preview point = s_hat)
        self._s_ref.copy_(self._s_hat)

        # 8. Geometric errors at s_ref
        self._compute_geometric_path_errors(ee_pos_w)

        # 9. Path-ahead clearance
        self._compute_path_ahead_clearance()

        # 10. Adaptive tube deadband
        clearance = self._path_ahead_clearance
        self._tube_deadband_dynamic.copy_(clearance)
        self._tube_deadband_dynamic.mul_(-1.0).add_(cfg.path_clearance_safe)
        self._tube_deadband_dynamic.relu_()
        self._tube_deadband_dynamic.mul_(cfg.tube_deadband_gain)
        self._tube_deadband_dynamic.add_(cfg.tube_deadband_min)
        self._tube_deadband_dynamic.clamp_(
            cfg.tube_deadband_min, cfg.tube_deadband_max
        )

        # 11. Reference pose at q1 for observations + reward tracking
        pos_w = self._preview_pos_w[:, 0]
        quat_w = self._preview_quat_w[:, 0]

        pos_err_vec, rot_err_vec_ref = compute_pose_error(
            pos_w, quat_w, ee_pos_w, ee_quat_w,
        )
        d_p = torch.norm(pos_err_vec, dim=-1)
        d_o = torch.norm(rot_err_vec_ref, dim=-1)

        # Delta progress (signed), zeroed on the first valid step after reset
        self._position_progress_delta.copy_(self._prev_position_error)
        self._position_progress_delta.sub_(d_p)
        self._position_progress_delta.masked_fill_(~self._has_prev_ref_error, 0.0)
        self._orientation_progress_delta.copy_(self._prev_orientation_error)
        self._orientation_progress_delta.sub_(d_o)
        self._orientation_progress_delta.masked_fill_(
            ~self._has_prev_ref_error, 0.0
        )
        self._prev_position_error.copy_(d_p)
        self._prev_orientation_error.copy_(d_o)
        self._has_prev_ref_error.fill_(True)

        # 12. Base reachability: rho_base
        base_pos_xy = self.robot.data.root_pos_w[:, :2]
        ref_pos_xy = pos_w[:, :2]
        rho_base = (ref_pos_xy - base_pos_xy).norm(dim=-1)

        self._rho_base_delta.copy_(self._prev_rho_base)
        self._rho_base_delta.sub_(rho_base)
        self._rho_base_delta.masked_fill_(~self._has_prev_rho_base, 0.0)
        self._prev_rho_base.copy_(rho_base)
        self._rho_base.copy_(rho_base)
        self._has_prev_rho_base.fill_(True)

        self._base_task_weight.copy_(rho_base)
        self._base_task_weight.sub_(cfg.arm_workspace_radius)
        self._base_task_weight.mul_(cfg.base_switch_k)
        self._base_task_weight.sigmoid_()

        # 13. Store reference pose from q1
        self.pose_command_w[:, :3] = pos_w
        self.pose_command_w[:, 3:7] = quat_unique(quat_w)

        root_pos_w = self.robot.data.root_pos_w
        root_quat_w = self.robot.data.root_quat_w
        cmd_pos_b, cmd_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w, pos_w, quat_w
        )
        self.pose_command_b[:, :3] = cmd_pos_b
        self.pose_command_b[:, 3:7] = cmd_quat_b

    # -----------------------------------------------------------------
    # Path-ahead clearance (static only)
    # -----------------------------------------------------------------

    def _compute_path_ahead_clearance(self):
        """Compute minimum static clearance along nominal path ahead of s_hat.

        Uses the valid_hits cache from sphere_lidar if available, plus the
        static obstacle height grid for collision-inside checks.
        """
        cfg = self.cfg
        K = cfg.path_clearance_num_samples

        if K <= 0:
            self._path_ahead_clearance.fill_(10.0)
            return

        # Sample K points along nominal path ahead
        self._path_clearance_query_s.copy_(self._path_clearance_step_offsets)
        self._path_clearance_query_s.add_(self._s_hat.unsqueeze(1))
        self._path_clearance_query_s.clamp_(0.0, 1.0)

        query_pos = self.get_path_positions_at_s(self._path_clearance_query_s)  # [N, K, 3]

        # Try to get valid_hits from sphere_lidar cache.
        clearances = self._path_clearance_values
        clearances.fill_(1e6)
        sphere_lidar_cache, compute_sphere_lidar = self._get_sphere_lidar_helpers()
        if sphere_lidar_cache is not None and compute_sphere_lidar is not None:
            cache_key = id(self._env)
            cached = sphere_lidar_cache.get(cache_key)
            if cached is None or cached.get("_step") != self._env.common_step_counter:
                cached = compute_sphere_lidar(self._env)
            if cached is not None and "valid_hits" in cached:
                valid_hits = cached["valid_hits"]  # [E, R', 3]
                if valid_hits.shape[1] > 0:
                    dists = torch.cdist(query_pos, valid_hits)  # [E, K, R']
                    min_dists = dists.min(dim=-1).values  # [E, K]
                    clearances.copy_(min_dists)
                    clearances.sub_(cfg.path_clearance_probe_radius)

        # Override with 0 where height grid says inside obstacle
        if self._has_height_grid:
            collides = self._check_obstacle_collision(
                query_pos.reshape(-1, 3)
            ).reshape(self.num_envs, K)
            clearances.masked_fill_(collides, 0.0)

        self._path_ahead_clearance.copy_(clearances.min(dim=-1).values)
        self._path_ahead_clearance.clamp_(0.0, 10.0)

    def _compute_min_clearance(self):
        """Compute min clearance from static + dynamic obstacles.

        Used for logging / metric only, NOT for progress control.
        """
        from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.mdp.observations import (
            compute_full_min_clearance,
        )

        clearance = compute_full_min_clearance(self._env)
        self._min_clearance.copy_(clearance["min_clearance"])

    def _maybe_update_min_clearance(self):
        """Update logging-only min_clearance at a lower fixed frequency."""
        step = int(self._env.common_step_counter)
        if step - self._last_min_clearance_step < self._min_clearance_update_interval:
            return
        self._last_min_clearance_step = step
        self._compute_min_clearance()

    def _resample_command(self, env_ids):
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        n = len(env_ids)
        if n == 0:
            return

        # Reset progress state
        self._s_proj[env_ids] = 0.0
        self._prev_s_proj[env_ids] = 0.0
        self._robot_progress_delta[env_ids] = 0.0
        self._contouring_error[env_ids] = 0.0
        self._path_orientation_error[env_ids] = 0.0
        self._lag_error[env_ids] = 0.0
        self._geometric_contouring_error[env_ids] = 0.0
        self._min_clearance[env_ids] = 1e6
        # Reset virtual progress (s_hat)
        self._s_hat[env_ids] = self.cfg.s_hat_initial_offset_s
        self._prev_s_hat[env_ids] = self.cfg.s_hat_initial_offset_s
        self._s_hat_dot[env_ids] = 0.0
        self._rho_base[env_ids] = 0.0
        self._prev_rho_base[env_ids] = 0.0
        self._has_prev_rho_base[env_ids] = False
        self._rho_base_delta[env_ids] = 0.0
        self._base_task_weight[env_ids] = 0.0
        self._ep_sum_contouring[env_ids] = 0.0
        self._ep_sum_proj_ori_error[env_ids] = 0.0
        self._ep_step_count[env_ids] = 0.0
        # Reset preview state
        self._preview_s[env_ids] = 0.0
        self._preview_pos_w[env_ids] = 0.0
        self._preview_quat_w[env_ids] = 0.0
        self._preview_quat_w[env_ids, :, 0] = 1.0
        self._path_ahead_clearance[env_ids] = 1.0
        self._tube_deadband_dynamic[env_ids] = self.cfg.tube_deadband_min
        # Reset tail release state
        self._s_hat_tail_alpha[env_ids] = 0.0
        self._s_hat_tail_alpha_s[env_ids] = 0.0
        self._s_hat_tail_alpha_d[env_ids] = 0.0
        # Reset reward reference state
        self._s_ref[env_ids] = 0.0
        self._prev_position_error[env_ids] = 0.0
        self._prev_orientation_error[env_ids] = 0.0
        self._has_prev_ref_error[env_ids] = False
        self._position_progress_delta[env_ids] = 0.0
        self._orientation_progress_delta[env_ids] = 0.0
        self._prev_final_position_error[env_ids] = 0.0
        self._has_prev_final_position_error[env_ids] = False
        self._final_goal_progress_delta[env_ids] = 0.0

        if not self.robot.is_initialized:
            self._T_init_w[env_ids, :3] = 0.0
            self._T_init_w[env_ids, 3] = 1.0
            self._T_init_w[env_ids, 4:] = 0.0
            self._T_end_w[env_ids] = self._T_init_w[env_ids]
            self._build_path_cache(env_ids)
            self.metrics["final_position_error"][env_ids] = 1e6
            self.metrics["final_orientation_error"][env_ids] = 1e6
            return

        # 1. Initial pose = current EE pose in world frame
        ee_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[env_ids, self.body_idx]
        self._T_init_w[env_ids, :3] = ee_pos_w
        self._T_init_w[env_ids, 3:7] = quat_unique(ee_quat_w)

        # 2. Sample target in {Proj} frame
        proj_pos, proj_quat = self._get_projected_frame(env_ids)
        T_end_proj = self._sample_target_pose(
            n, env_ids, proj_pos, proj_quat
        )

        # 2b. Fix orientation
        _, ee_quat_proj = subtract_frame_transforms(
            proj_pos, proj_quat, ee_pos_w, ee_quat_w
        )
        T_end_proj[:, 3:7] = quat_mul(
            ee_quat_proj, T_end_proj[:, 3:7]
        )

        # 3. Convert to world frame
        end_pos_w, end_quat_w = combine_frame_transforms(
            proj_pos, proj_quat,
            T_end_proj[:, :3], T_end_proj[:, 3:7]
        )
        self._T_end_w[env_ids, :3] = end_pos_w
        self._T_end_w[env_ids, 3:7] = quat_unique(end_quat_w)

        # 4. Build Bezier path cache (arc-length normalised, decoupled ori)
        self._build_path_cache(env_ids)

        # 5. Compute initial metrics
        final_pos_error, final_rot_error = compute_pose_error(
            self._T_end_w[env_ids, :3],
            self._T_end_w[env_ids, 3:7],
            ee_pos_w,
            ee_quat_w,
        )
        self.metrics["final_position_error"][env_ids] = torch.norm(final_pos_error, dim=-1)
        self.metrics["final_orientation_error"][env_ids] = torch.norm(final_rot_error, dim=-1)

    # =====================================================================
    # Metrics
    # =====================================================================

    def _update_metrics(self):
        if not self.robot.is_initialized:
            return

        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]

        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:7],
            ee_pos_w,
            ee_quat_w,
        )

        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

        # Final goal error: EE vs world-fixed path endpoint (_T_end_w)
        final_pos_error, final_rot_error = compute_pose_error(
            self._T_end_w[:, :3],
            self._T_end_w[:, 3:7],
            ee_pos_w,
            ee_quat_w,
        )
        d_final = torch.norm(final_pos_error, dim=-1)
        self.metrics["final_position_error"] = d_final
        self.metrics["final_orientation_error"] = torch.norm(final_rot_error, dim=-1)

        # Final-goal approach delta (positive = getting closer)
        self._final_goal_progress_delta.copy_(self._prev_final_position_error)
        self._final_goal_progress_delta.sub_(d_final)
        self._final_goal_progress_delta.masked_fill_(
            ~self._has_prev_final_position_error, 0.0
        )
        self._prev_final_position_error.copy_(d_final)
        self._has_prev_final_position_error.fill_(True)
        self.metrics["final_goal_progress_delta"] = self._final_goal_progress_delta

        self.metrics["contouring_error"] = self._contouring_error
        self.metrics["path_orientation_error"] = self._path_orientation_error
        self.metrics["lag_error_signed"] = self._lag_error
        self.metrics["lag_error_abs"] = self._lag_error.abs()
        self.metrics["s_proj"] = self._s_proj
        self.metrics["s_hat"] = self._s_hat
        self.metrics["s_hat_dot"] = self._s_hat_dot
        self.metrics["robot_progress_delta"] = self._robot_progress_delta
        self.metrics["min_clearance"] = self._min_clearance
        # Reward reference metrics (s_ref based)
        self.metrics["s_ref"] = self._s_ref
        self.metrics["position_progress_delta"] = self._position_progress_delta
        self.metrics["orientation_progress_delta"] = self._orientation_progress_delta

        # Episode-level mean errors (running sum / step count)
        # ep_mean_contouring: uses contouring_error (s_proj, path-centerline quality)
        # ep_mean_ori_error: uses path_orientation_error (at s_proj)
        self._ep_sum_contouring += self._contouring_error
        self._ep_sum_proj_ori_error += self._path_orientation_error
        self._ep_step_count += 1.0
        cnt = self._ep_step_count
        self.metrics["ep_mean_contouring"] = self._ep_sum_contouring / cnt
        self.metrics["ep_mean_ori_error"] = self._ep_sum_proj_ori_error / cnt

        # Base reachability metrics
        self.metrics["base_reachability_distance"] = self._rho_base
        self.metrics["base_task_weight"] = self._base_task_weight
        self.metrics["rho_base_delta"] = self._rho_base_delta

        # Tail release metrics
        self.metrics["s_hat_tail_alpha"] = self._s_hat_tail_alpha
        self.metrics["s_hat_tail_alpha_s"] = self._s_hat_tail_alpha_s
        self.metrics["s_hat_tail_alpha_d"] = self._s_hat_tail_alpha_d

        # Projection-driven progress metrics
        self.metrics["path_ahead_clearance"] = self._path_ahead_clearance
        self.metrics["tube_deadband_dynamic"] = self._tube_deadband_dynamic

    # =========================================================================
    # Public helpers (for observation functions)
    # =========================================================================

    def _get_repeated_env_indices(self, repeats: int, batch_size: int | None = None) -> torch.Tensor:
        """Return cached env indices repeated along the query dimension."""
        if batch_size is not None and batch_size != self.num_envs:
            return torch.arange(batch_size, device=self.device).repeat_interleave(repeats)

        cached = self._env_repeat_index_cache.get(repeats)
        if cached is None:
            cached = self._env_indices.repeat_interleave(repeats)
            self._env_repeat_index_cache[repeats] = cached
        return cached

    def get_path_positions_at_s(self, s_query: torch.Tensor) -> torch.Tensor:
        """Get path positions at arbitrary s values. [N, 3] or [N, K, 3]."""
        if s_query.dim() == 1:
            return self._interpolate_path_position(s_query)
        # Batched: s_query [N, K]
        N, K = s_query.shape
        flat = s_query.reshape(-1)
        # Expand path cache to handle flat queries
        M = self.cfg.num_path_cache_samples
        idx_f = flat.clamp(0.0, 1.0) * (M - 1)
        idx_left = idx_f.long().clamp(0, M - 2)
        idx_right = idx_left + 1
        t = idx_f - idx_left.float()
        env_idx = self._get_repeated_env_indices(K, N)
        p_left = self._path_pos_w[env_idx, idx_left]
        p_right = self._path_pos_w[env_idx, idx_right]
        result = p_left + t.unsqueeze(-1) * (p_right - p_left)
        return result.reshape(N, K, 3)

    def get_path_orientations_at_s(self, s_query: torch.Tensor) -> torch.Tensor:
        """Get path orientations at arbitrary s values. [N, 4] or [N, K, 4]."""
        if s_query.dim() == 1:
            return self._interpolate_path_orientation(s_query)

        N, K = s_query.shape
        flat = s_query.reshape(-1)
        M = self.cfg.num_path_cache_samples
        idx_f = flat.clamp(0.0, 1.0) * (M - 1)
        idx_left = idx_f.long().clamp(0, M - 2)
        idx_right = idx_left + 1
        t = idx_f - idx_left.float()
        env_idx = self._get_repeated_env_indices(K, N)
        q_left = self._path_quat_w[env_idx, idx_left]
        q_right = self._path_quat_w[env_idx, idx_right]
        result = _quat_slerp(q_left, q_right, t)
        return result.reshape(N, K, 4)

    def get_final_goal_pose_w(self) -> torch.Tensor:
        """Get the world-fixed final goal pose [N, 7] (pos + quat)."""
        return self._T_end_w.clone()

    def update_min_clearance(self, min_clearance: torch.Tensor):
        """Override min_clearance externally (backward-compat, not required)."""
        self._min_clearance.copy_(min_clearance)

    # =========================================================================
    # Visualization
    # =========================================================================

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                cfg = self.cfg.goal_pose_visualizer_cfg
                self.goal_pose_visualizer = (
                    VisualizationMarkers(cfg)
                )
            if not hasattr(self, "start_pose_visualizer"):
                cfg = self.cfg.start_pose_visualizer_cfg
                self.start_pose_visualizer = (
                    VisualizationMarkers(cfg)
                )
            if not hasattr(self, "current_pose_visualizer"):
                cfg = self.cfg.current_pose_visualizer_cfg
                self.current_pose_visualizer = (
                    VisualizationMarkers(cfg)
                )
            if not hasattr(self, "trajectory_visualizer"):
                cfg = self.cfg.trajectory_visualizer_cfg
                self.trajectory_visualizer = (
                    VisualizationMarkers(cfg)
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

        # Goal pose (end of path)
        if hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.visualize(
                self._T_end_w[:, :3],
                self._T_end_w[:, 3:7],
            )

        # Start pose (beginning of path)
        if hasattr(self, "start_pose_visualizer"):
            self.start_pose_visualizer.visualize(
                self._T_init_w[:, :3],
                self._T_init_w[:, 3:7],
            )

        # Current reference pose
        if hasattr(self, "current_pose_visualizer"):
            self.current_pose_visualizer.visualize(
                self.pose_command_w[:, :3],
                self.pose_command_w[:, 3:7],
            )

        # Path trajectory
        if hasattr(self, "trajectory_visualizer"):
            M = self.cfg.num_path_cache_samples
            vis_n = min(self.cfg.num_trajectory_samples, M)
            step = max(1, M // vis_n)
            vis_pos = self._path_pos_w[:, ::step, :]
            flat_pos = vis_pos.reshape(-1, 3)
            self.trajectory_visualizer.visualize(translations=flat_pos)
