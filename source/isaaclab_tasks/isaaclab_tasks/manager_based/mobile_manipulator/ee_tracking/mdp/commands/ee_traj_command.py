"""SE(3) EE Trajectory Command Generator - Lie Group Interpolation with Workspace Constraints."""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    combine_frame_transforms,
    compute_pose_error,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_unique,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Configuration
# =============================================================================


@configclass
class EETrajectoryCommandCfg(CommandTermCfg):
    """Configuration for SE(3) trajectory with Lie group interpolation."""

    class_type: type = None

    # Asset configuration
    asset_name: str = MISSING
    body_name: str = MISSING

    # Timing
    resampling_time_range: tuple[float, float] = (1000.0, 1000.0)
    trajectory_time_range: tuple[float, float] = (3.0, 6.0)

    # Orientation sampling (uniform on SO(3))
    sample_uniform_orientation: bool = True

    # Workspace constraints (define sampling region relative to arm base)
    arm_length_min: float = 0.2      # Minimum arm length (avoid singularity)
    arm_length_max: float = 0.5      # Maximum arm length (reachable range)

    # Arm base offset relative to robot base (in base frame)
    # Format: (x_forward, y_lateral, z_vertical)
    arm_base_offset: tuple[float, float, float] = (0.3, 0.0, 0.45)

    # Collision constraints (relative to robot base center)
    chassis_radius: float = 0.35     # Chassis radius (avoid self-collision)
    min_height: float = 0.2          # Minimum height in {Proj} frame
    max_height: float = 1.2          # Maximum height in {Proj} frame

    max_resample_attempts: int = 10  # Max resampling attempts

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
    trajectory_visualizer_cfg: VisualizationMarkersCfg = (
        FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/trajectory",
            markers={
                "frame": FRAME_MARKER_CFG.markers["frame"].replace(
                    scale=(0.06, 0.06, 0.06)
                )
            },
        )
    )
    num_trajectory_samples: int = 10

    def __post_init__(self):
        super().__post_init__()
        if self.class_type is None:
            self.class_type = EETrajectoryCommand


# =============================================================================
# Lie Group Helper Functions
# =============================================================================


def se3_log(pos_delta: torch.Tensor, quat_rel: torch.Tensor) -> torch.Tensor:
    """Compute SE(3) logarithm map: T -> se(3) ∈ ℝ^6.

    Args:
        pos_delta: Relative position (N, 3)
        quat_rel: Relative quaternion (N, 4) [w, x, y, z]

    Returns:
        Lie algebra element (N, 6) [translation, rotation]
    """
    axis_angle = axis_angle_from_quat(quat_unique(quat_rel))
    return torch.cat([pos_delta, axis_angle], dim=-1)


def se3_exp(
    lie_algebra: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SE(3) exponential map: se(3) -> SE(3).

    Args:
        lie_algebra: Lie algebra element (N, 6) [translation, rotation]

    Returns:
        tuple of (position, quaternion)
    """
    pos = lie_algebra[..., :3]
    axis_angle = lie_algebra[..., 3:]

    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = torch.zeros_like(axis_angle)

    valid = angle.squeeze(-1) > 1e-8
    axis[valid] = axis_angle[valid] / angle[valid]
    axis[~valid, 0] = 1.0

    quat = quat_from_angle_axis(angle.squeeze(-1), axis)

    return pos, quat


def se3_minus(
    pos1: torch.Tensor,
    quat1: torch.Tensor,
    pos2: torch.Tensor,
    quat2: torch.Tensor,
) -> torch.Tensor:
    """Compute SE(3) 'minus': T2 ⊖ T1 = log(T1^{-1} T2).

    Args:
        pos1, quat1: First pose
        pos2, quat2: Second pose

    Returns:
        Lie algebra difference (N, 6)
    """
    pos_rel, quat_rel = subtract_frame_transforms(
        pos1, quat1, pos2, quat2
    )
    return se3_log(pos_rel, quat_rel)


def se3_plus(
    pos_base: torch.Tensor,
    quat_base: torch.Tensor,
    lie_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SE(3) 'plus': T ⊕ δ = T · exp(δ).

    Args:
        pos_base, quat_base: Base pose
        lie_delta: Lie algebra increment (N, 6)

    Returns:
        tuple of (new_position, new_quaternion)
    """
    pos_delta, quat_delta = se3_exp(lie_delta)
    return combine_frame_transforms(
        pos_base, quat_base, pos_delta, quat_delta
    )


# =============================================================================
# Command Generator
# =============================================================================


class EETrajectoryCommand(CommandTerm):
    """SE(3) trajectory command generator using Lie group interpolation.

    Implementation:
    1. Projects robot base onto ground to form {Proj} frame
    2. Samples poses in spherical shell [arm_length_min, arm_length_max] from arm base
    3. Applies workspace constraints (height, chassis collision)
    4. Computes ΔT = T_end ⊖ T_init in se(3)
    5. Uses cubic s(t) to interpolate: T_ref(t) = T_init ⊕ exp(s(t)·ΔT)
    6. Transforms result to body frame for observation

    Key feature: Accounts for arm base offset from robot base center.
    """

    cfg: EETrajectoryCommandCfg

    def __init__(self, cfg: EETrajectoryCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._env = env
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # Workspace constraints
        self.arm_length_min = cfg.arm_length_min
        self.arm_length_max = cfg.arm_length_max
        self.chassis_radius = cfg.chassis_radius
        self.min_height = cfg.min_height
        self.max_height = cfg.max_height

        # Arm base offset (convert to tensor)
        self.arm_base_offset = torch.tensor(
            cfg.arm_base_offset, device=self.device, dtype=torch.float32
        )

        # =====================================================================
        # Command buffers
        # =====================================================================
        self.lie_command_b = torch.zeros(
            self.num_envs, 6, device=self.device
        )

        self.pose_command_b = torch.zeros(
            self.num_envs, 7, device=self.device
        )
        self.pose_command_b[:, 3] = 1.0

        self.pose_command_w = torch.zeros(
            self.num_envs, 7, device=self.device
        )
        self.pose_command_w[:, 3] = 1.0

        # =====================================================================
        # Trajectory state in {Proj} frame
        # =====================================================================
        self._T_init_proj = torch.zeros(self.num_envs, 7, device=self.device)
        self._T_init_proj[:, 3] = 1.0

        self._T_end_proj = torch.zeros(self.num_envs, 7, device=self.device)
        self._T_end_proj[:, 3] = 1.0

        self._delta_lie = torch.zeros(self.num_envs, 6, device=self.device)

        self._traj_time = torch.zeros(self.num_envs, device=self.device)
        self._traj_duration = torch.zeros(self.num_envs, device=self.device)

        # 用于存储轨迹开始时的 {Proj} 坐标系
        self._proj_pos_at_start = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._proj_quat_at_start = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self._proj_quat_at_start[:, 0] = 1.0
        # =====================================================================
        # Metrics
        # =====================================================================
        self.metrics["position_error"] = torch.zeros(
            self.num_envs, device=self.device
        )
        self.metrics["orientation_error"] = torch.zeros(
            self.num_envs, device=self.device
        )

        # =====================================================================
        # Visualization
        # =====================================================================
        if cfg.debug_vis:
            self._vis_step_counter = 0
            self._vis_alphas = torch.linspace(
                0.0, 1.0, cfg.num_trajectory_samples, device=self.device
            )
            self._vis_traj_pos = torch.zeros(
                cfg.num_trajectory_samples, 3, device=self.device
            )
            self._vis_traj_quat = torch.zeros(
                cfg.num_trajectory_samples, 4, device=self.device
            )
            self._vis_traj_quat[:, 0] = 1.0
            self._vis_needs_update = True

    def __str__(self) -> str:
        return (
            f"EETrajectoryCommand (Lie Group Interpolation with Arm Base Offset):\n"
            f"  Command dimension: {tuple(self.command.shape[1:])}\n"
            f"  Trajectory duration: {self.cfg.trajectory_time_range}\n"
            f"  Arm length range: [{self.arm_length_min:.2f}, {self.arm_length_max:.2f}]m\n"
            f"  Arm base offset: {self.cfg.arm_base_offset}\n"
            f"  Height range: [{self.min_height:.2f}, {self.max_height:.2f}]m\n"
            f"  Chassis radius: {self.chassis_radius:.2f}m\n"
        )

    @property
    def command(self) -> torch.Tensor:
        """Return command in Lie algebra se(3) (6D)."""
        return self.lie_command_b

    # =========================================================================
    # Core Methods
    # =========================================================================

    def _get_projected_frame(
        self, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct projected frame {Proj} on the ground.

        Projects robot base onto ground plane:
        - Origin at base projection on ground (z=0)
        - Yaw aligned with base heading
        - Roll and pitch are zero (horizontal)

        Args:
            env_ids: Environment indices (None = all)

        Returns:
            (proj_pos, proj_quat): Position and orientation of {Proj}
        """
        if env_ids is None:
            root_pos = self.robot.data.root_pos_w
            root_quat = self.robot.data.root_quat_w
        else:
            root_pos = self.robot.data.root_pos_w[env_ids]
            root_quat = self.robot.data.root_quat_w[env_ids]

        proj_pos = root_pos.clone()
        proj_pos[:, 2] = 0.0

        # yaw = arctan2(2(wz + xy), 1 - 2(y^2 + z^2))
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

    def _get_arm_base_position_in_proj(
        self, proj_quat: torch.Tensor
    ) -> torch.Tensor:
        """Compute arm base position in {Proj} frame.

        Args:
            proj_quat: Projected frame orientation (n, 4)

        Returns:
            arm_base_pos_proj: Arm base position in {Proj} frame (n, 3)
        """
        n = proj_quat.shape[0]

        # Extract yaw angle from proj_quat
        w = proj_quat[:, 0]
        z = proj_quat[:, 3]
        yaw = torch.atan2(2.0 * w * z, w * w - z * z)

        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)

        # Rotate arm_base_offset by yaw
        # arm_base_offset is in base frame, rotate to {Proj} frame
        arm_base_pos_proj = torch.zeros(n, 3, device=self.device)
        arm_base_pos_proj[:, 0] = (
            self.arm_base_offset[0] * cos_yaw -
            self.arm_base_offset[1] * sin_yaw
        )
        arm_base_pos_proj[:, 1] = (
            self.arm_base_offset[0] * sin_yaw +
            self.arm_base_offset[1] * cos_yaw
        )
        arm_base_pos_proj[:, 2] = self.arm_base_offset[2]

        return arm_base_pos_proj

    def _check_workspace_constraints(
        self,
        pos_proj: torch.Tensor,
        arm_base_pos_proj: torch.Tensor
    ) -> torch.Tensor:
        """Check workspace constraints for sampled positions in {Proj} frame.

        Args:
            pos_proj: Target positions in {Proj} frame, shape (n, 3)
            arm_base_pos_proj: Arm base positions in {Proj} frame, shape (n, 3)

        Returns:
            valid_mask: Boolean tensor, True if position is valid
        """
        # 1. Height constraint (absolute height in {Proj})
        valid_height = (pos_proj[:, 2] >= self.min_height) & \
                       (pos_proj[:, 2] <= self.max_height)

        # 2. Chassis collision (xy distance from robot base center)
        dist_xy_from_base = torch.norm(pos_proj[:, :2], dim=1)
        valid_chassis = dist_xy_from_base > self.chassis_radius

        # 3. Arm reach constraint (distance from arm base)
        # Add tolerance for numerical errors
        dist_from_arm = torch.norm(pos_proj - arm_base_pos_proj, dim=1)
        valid_reach = (dist_from_arm >= self.arm_length_min * 0.95) & \
            (dist_from_arm <= self.arm_length_max * 1.05)

        return valid_height & valid_chassis & valid_reach

    def _sample_target_pose_in_shell(self, n: int, env_ids: torch.Tensor) -> torch.Tensor:
        """Sample target pose uniformly in spherical shell centered at arm base.

        Samples in shell [arm_length_min, arm_length_max] from arm base,
        accounting for arm_base_offset from robot base.

        Args:
            n: Number of samples
            env_ids: Environment indices

        Returns:
            Pose (n, 7) [pos, quat] in {Proj} frame, guaranteed to be valid
        """
        device = self.device

        pose = torch.zeros(n, 7, device=device)
        pose[:, 3] = 1.0

        # Get projected frame for these environments
        proj_pos, proj_quat = self._get_projected_frame(env_ids)

        # Compute arm base position in {Proj} frame
        arm_base_pos_proj = self._get_arm_base_position_in_proj(proj_quat)

        remaining_mask = torch.ones(n, dtype=torch.bool, device=device)
        attempts = 0
        max_attempts = self.cfg.max_resample_attempts

        while remaining_mask.any() and attempts < max_attempts:
            num_remaining = remaining_mask.sum().item()

            # Sample radius uniformly in spherical shell [r_min, r_max]
            u = torch.rand(num_remaining, device=device)
            r_min_cubed = self.arm_length_min ** 3
            r_max_cubed = self.arm_length_max ** 3
            r = torch.pow(r_min_cubed + u * (r_max_cubed - r_min_cubed), 1.0/3.0)

            # Sample direction uniformly on unit sphere
            direction = torch.randn(num_remaining, 3, device=device)
            direction = direction / (
                torch.norm(direction, dim=-1, keepdim=True) + 1e-8
            )

            # Position relative to arm base
            pos_from_arm_base = direction * r.unsqueeze(-1)

            # Transform to {Proj} frame (add arm base offset)
            remaining_indices = torch.where(remaining_mask)[0]
            candidate_pos = pos_from_arm_base + arm_base_pos_proj[remaining_indices]

            # Check constraints
            valid = self._check_workspace_constraints(
                candidate_pos, 
                arm_base_pos_proj[remaining_indices]
            )

            # Fill in valid samples
            valid_indices = remaining_indices[valid]
            pose[valid_indices, :3] = candidate_pos[valid]
            remaining_mask[valid_indices] = False
            attempts += 1

        # Fallback for remaining invalid samples
        if remaining_mask.any():
            remaining_indices = torch.where(remaining_mask)[0]
            fallback_r = (self.arm_length_min + self.arm_length_max) / 2
            fallback_height = (self.min_height + self.max_height) / 2

            # Place in front of arm at safe distance and height
            fallback_pos_from_arm = torch.tensor(
                [fallback_r * 0.7, 0.0, fallback_height - self.arm_base_offset[2]], 
                device=device
            ).unsqueeze(0).expand(remaining_mask.sum(), -1)

            pose[remaining_mask, :3] = (
                fallback_pos_from_arm + arm_base_pos_proj[remaining_indices]
            )

        # Sample orientation uniformly on SO(3)
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
            pose[:, 3] = 1.0
            pose[:, 4:] = 0.0

        return pose

    def _compute_lie_algebra_delta(self, env_ids: torch.Tensor):
        """Compute Lie algebra difference: ΔT = log(T_init^{-1} T_end).

        Args:
            env_ids: Environment indices
        """
        pos_init = self._T_init_proj[env_ids, :3]
        quat_init = self._T_init_proj[env_ids, 3:7]
        pos_end = self._T_end_proj[env_ids, :3]
        quat_end = self._T_end_proj[env_ids, 3:7]

        self._delta_lie[env_ids] = se3_minus(
            pos_init, quat_init, pos_end, quat_end
        )

    def _resample_command(self, env_ids):
        """Resample trajectory for specified environments."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(
                env_ids, device=self.device, dtype=torch.long
            )

        n = len(env_ids)
        if n == 0:
            return

        proj_pos, proj_quat = self._get_projected_frame(env_ids)
        self._proj_pos_at_start[env_ids] = proj_pos
        self._proj_quat_at_start[env_ids] = proj_quat

        t_min, t_max = self.cfg.trajectory_time_range
        self._traj_duration[env_ids] = torch.empty(
            n, device=self.device
        ).uniform_(t_min, t_max)
        self._traj_time[env_ids] = 0.0

        if not self.robot.is_initialized:
            self._T_init_proj[env_ids, :3] = 0.0
            self._T_init_proj[env_ids, 3] = 1.0
            self._T_init_proj[env_ids, 4:] = 0.0
            self._T_end_proj[env_ids] = self._sample_target_pose_in_shell(n, env_ids)
            self._compute_lie_algebra_delta(env_ids)
            return

        # get current EE pose in {Proj} frame
        ee_pos_w = self.robot.data.body_pos_w[env_ids, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[env_ids, self.body_idx]

        # Transform EE pose to {Proj} frame
        ee_pos_proj, ee_quat_proj = subtract_frame_transforms(
            self._proj_pos_at_start[env_ids],  # 使用保存的
            self._proj_quat_at_start[env_ids],
            ee_pos_w, ee_quat_w
        )

        # Set as initial pose
        self._T_init_proj[env_ids, :3] = ee_pos_proj
        self._T_init_proj[env_ids, 3:7] = quat_unique(ee_quat_proj)

        # Sample End pose (with workspace constraints)
        self._T_end_proj[env_ids] = self._sample_target_pose_in_shell(n, env_ids)

        self._compute_lie_algebra_delta(env_ids)

        self._vis_needs_update = True

    def _update_command(self):
        """Update command using cubic polynomial interpolation on SE(3)."""
        if not self.robot.is_initialized:
            return

        dt = self._env.step_dt
        self._traj_time += dt

        finished_mask = self._traj_time >= self._traj_duration
        if finished_mask.any():
            finished_ids = torch.where(finished_mask)[0]
            self._resample_command(finished_ids)

        # alpha = t / Traj_duration
        alpha = torch.clamp(self._traj_time / self._traj_duration, 0.0, 1.0)

        # Cubic polynomial interpolation s(α) = 3α^2 - 2α^3
        s = 3.0 * alpha ** 2 - 2.0 * alpha ** 3
        scaled_delta = s.unsqueeze(-1) * self._delta_lie

        pos_init = self._T_init_proj[:, :3]
        quat_init = self._T_init_proj[:, 3:7]

        # T_ref(t) = T_init ⊕ exp(s(t)·ΔT)
        pos_ref_proj, quat_ref_proj = se3_plus(
            pos_init, quat_init, scaled_delta
        )

        # Fix {Proj} frame to {World}
        proj_pos = self._proj_pos_at_start
        proj_quat = self._proj_quat_at_start

        # 转到世界系
        pos_w, quat_w = combine_frame_transforms(
            proj_pos, proj_quat,  # Fixed {Proj} frame
            pos_ref_proj, quat_ref_proj
        )

        # Store SE(3) command in {World} frame
        self.pose_command_w[:, :3] = pos_w
        self.pose_command_w[:, 3:7] = quat_unique(quat_w)

        # Store SE(3) command in {Proj} frame
        self.pose_command_b[:, :3] = pos_ref_proj
        self.pose_command_b[:, 3:7] = quat_unique(quat_ref_proj)

        # Compute Lie algebra command: current EE pose in {Proj} frame
        curr_proj_pos, curr_proj_quat = self._get_projected_frame()
        ee_pos_w = self.robot.data.body_pos_w[:, self.body_idx]
        ee_quat_w = self.robot.data.body_quat_w[:, self.body_idx]

        # Transform current EE pose to {Proj} frame
        ee_pos_curr_proj, ee_quat_curr_proj = subtract_frame_transforms(
            curr_proj_pos, curr_proj_quat, ee_pos_w, ee_quat_w
        )

        # 目标在当前 {Proj} 中的表示
        pos_ref_curr_proj, quat_ref_curr_proj = subtract_frame_transforms(
            curr_proj_pos, curr_proj_quat, pos_w, quat_w
        )
        # Compute se(3) difference: T_ref ⊖ T_ee
        self.lie_command_b[:] = se3_minus(
            ee_pos_curr_proj, ee_quat_curr_proj,
            pos_ref_curr_proj, quat_ref_curr_proj
        )

    def _update_metrics(self):
        """Update tracking error metrics."""
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

    # =========================================================================
    # Visualization
    # =========================================================================

    def _compute_trajectory_visualization(self):
        """Compute trajectory waypoints for visualization using SE(3)."""
        if not self.robot.is_initialized:
            return

        n_samples = self.cfg.num_trajectory_samples

        proj_pos = self._proj_pos_at_start[0:1]
        proj_quat = self._proj_quat_at_start[0:1]

        pos_init = self._T_init_proj[0, :3]
        quat_init = self._T_init_proj[0, 3:7]
        delta_lie = self._delta_lie[0]

        # Vectorized cubic interpolation for all samples
        # s(α) = 3α² - 2α³
        alphas = self._vis_alphas
        s_vals = 3.0 * alphas ** 2 - 2.0 * alphas ** 3

        scaled_deltas = s_vals.unsqueeze(-1) * delta_lie.unsqueeze(0)

        pos_deltas = scaled_deltas[:, :3]
        axis_angles = scaled_deltas[:, 3:]

        angles = torch.norm(axis_angles, dim=-1, keepdim=True)
        axes = torch.zeros_like(axis_angles)
        valid = (angles.squeeze(-1) > 1e-8)
        axes[valid] = axis_angles[valid] / angles[valid]
        axes[~valid, 0] = 1.0

        quats_delta = quat_from_angle_axis(angles.squeeze(-1), axes)

        pos_init_expanded = pos_init.unsqueeze(0).expand(n_samples, -1)
        quat_init_expanded = quat_init.unsqueeze(0).expand(n_samples, -1)

        pos_ref_proj, quat_ref_proj = combine_frame_transforms(
            pos_init_expanded, quat_init_expanded,
            pos_deltas, quats_delta
        )

        proj_pos_expanded = proj_pos.expand(n_samples, -1)
        proj_quat_expanded = proj_quat.expand(n_samples, -1)

        pos_w_all, quat_w_all = combine_frame_transforms(
            proj_pos_expanded, proj_quat_expanded,
            pos_ref_proj, quat_ref_proj
        )

        self._vis_traj_pos[:] = pos_w_all
        self._vis_traj_quat[:] = quat_w_all

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Enable/disable visualization."""
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                cfg = self.cfg.goal_pose_visualizer_cfg
                self.goal_pose_visualizer = VisualizationMarkers(cfg)
            if not hasattr(self, "current_pose_visualizer"):
                cfg = self.cfg.current_pose_visualizer_cfg
                self.current_pose_visualizer = VisualizationMarkers(cfg)
            if not hasattr(self, "trajectory_visualizer"):
                cfg = self.cfg.trajectory_visualizer_cfg
                self.trajectory_visualizer = VisualizationMarkers(cfg)

            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.trajectory_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
            if hasattr(self, "current_pose_visualizer"):
                self.current_pose_visualizer.set_visibility(False)
            if hasattr(self, "trajectory_visualizer"):
                self.trajectory_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        """Visualization callback with throttling."""
        if not self.robot.is_initialized:
            return

        self._vis_step_counter += 1
        if self._vis_step_counter < self.cfg.vis_update_interval:
            return
        self._vis_step_counter = 0

        # Goal pose(Red)
        if hasattr(self, "goal_pose_visualizer"):
            self.goal_pose_visualizer.visualize(
                self.pose_command_w[:, :3], self.pose_command_w[:, 3:7]
            )

        # Current pose (Blue)
        if hasattr(self, "current_pose_visualizer"):
            ee_pos = self.robot.data.body_pos_w[:, self.body_idx]
            ee_quat = self.robot.data.body_quat_w[:, self.body_idx]
            self.current_pose_visualizer.visualize(ee_pos, ee_quat)

        # Trajectory (Green)
        if hasattr(self, "trajectory_visualizer"):
            if self._vis_needs_update:
                self._compute_trajectory_visualization()
                self._vis_needs_update = False

            self.trajectory_visualizer.visualize(
                self._vis_traj_pos, self._vis_traj_quat
            )
