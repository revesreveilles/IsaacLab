"""End-effector pose command generator - GPU-optimized."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_unique,
    quat_mul,
    quat_conjugate,
    combine_frame_transforms,
    compute_pose_error,
)
from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.config.agents.rsl_rl_ppo_cfg import MobileManipulatorPPORunnerCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EEPoseCommand(UniformPoseCommand):
    """End-effector pose command generator with GPU-accelerated sampling."""

    cfg: UniformPoseCommandCfg

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator."""
        super().__init__(cfg, env)

        # Extract robot and body index
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # Create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w_z = torch.zeros(self.num_envs, 1, device=self.device)
        self.pose_command_w = torch.zeros_like(self.pose_command_b)

        # Metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # Environment reference
        self.env = env

        # Get num_steps_per_env from PPO runner config
        cfg_runner = MobileManipulatorPPORunnerCfg()
        self.num_env_step = cfg_runner.num_steps_per_env

        # Collision detection parameters
        self.chassis_radius = cfg.chassis_sphere_radius
        self.arm_length_min = 0.3
        self.arm_length_max = 0.5

        print("=" * 80)
        print(" EEPoseCommand initialized (GPU-optimized):")
        print(f"   - Robot: {cfg.asset_name}, Body: {cfg.body_name} (idx={self.body_idx})")
        print(f"   - Chassis sphere radius: {self.chassis_radius}m")
        print(f"   - Arm length range: [{self.arm_length_min}, {self.arm_length_max}]m")
        print(f"   - Curriculum coefficient: {cfg.curriculum_coeff} iterations")
        print(f"   - Debug visualization: {cfg.debug_vis}")
        print("=" * 80)

    def __str__(self) -> str:
        msg = "EEPoseCommand (GPU-optimized):\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tCurriculum coefficient: {self.cfg.curriculum_coeff} iterations\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command in base frame. Shape is (num_envs, 7)."""
        return self.pose_command_b

    def _check_constraints(self, pos_b: torch.Tensor) -> torch.Tensor:
        """Check workspace constraints for sampled positions (GPU-accelerated).

        Args:
            pos_b: Target positions in base frame, shape (n, 3)

        Constraints:
        1. Arm reach: EE must be within reachable range [arm_length_min, arm_length_max]
        2. Chassis clearance: EE xy distance must be outside chassis sphere

        Note: Height constraint is already handled by pos_z sampling range in config.
        """
        # 1. Arm reach constraint (3D distance from base)
        length_3d = torch.norm(pos_b, dim=1)
        valid_reach = (length_3d >= self.arm_length_min) & (length_3d <= self.arm_length_max)

        # 2. Chassis collision: EE must be outside chassis sphere in xy plane
        dist_xy = torch.norm(pos_b[:, :2], dim=1)
        valid_chassis = dist_xy > self.chassis_radius

        return valid_reach & valid_chassis

    def _sample_with_curriculum(
        self, n: int, range_init: tuple, range_final: tuple, progress: float
    ) -> torch.Tensor:
        """Sample values with curriculum interpolation.

        Args:
            n: Number of samples
            range_init: Initial (min, max) range
            range_final: Final (min, max) range  
            progress: Curriculum progress [0, 1]

        Returns:
            Sampled values interpolated between init and final ranges
        """
        val_init = torch.empty(n, device=self.device).uniform_(*range_init)
        val_final = torch.empty(n, device=self.device).uniform_(*range_final)
        return val_init * (1 - progress) + val_final * progress

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample pose command with GPU-accelerated constraint checking."""
        if not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        n = len(env_ids)

        # Compute curriculum progress [0, 1]
        # iteration = common_step_counter / num_env_step
        count = torch.clamp(
            torch.tensor(self.env.common_step_counter / self.num_env_step / self.cfg.curriculum_coeff),
            0.0, 1.0
        )

        # Get robot state
        chassis_quat_w = self.robot.data.root_quat_w[env_ids]
        root_z = self.robot.data.root_pos_w[env_ids, 2]

        # ========== Sample positions with curriculum ==========
        pos_x = self._sample_with_curriculum(n, self.cfg.ranges_init.pos_x, self.cfg.ranges_final.pos_x, count)
        pos_y = self._sample_with_curriculum(n, self.cfg.ranges_init.pos_y, self.cfg.ranges_final.pos_y, count)
        pos_z_world = self._sample_with_curriculum(n, self.cfg.ranges_init.pos_z, self.cfg.ranges_final.pos_z, count)
        pos_z = pos_z_world - root_z

        # Stack positions and check constraints
        pos_b = torch.stack([pos_x, pos_y, pos_z], dim=1)
        valid = self._check_constraints(pos_b)

        # ========== Iterative resampling for invalid samples ==========
        for _ in range(20):  # max iterations
            invalid_mask = ~valid
            n_invalid = invalid_mask.sum().item()
            if n_invalid == 0:
                break

            # Resample invalid positions
            pos_x[invalid_mask] = self._sample_with_curriculum(
                n_invalid, self.cfg.ranges_init.pos_x, self.cfg.ranges_final.pos_x, count)
            pos_y[invalid_mask] = self._sample_with_curriculum(
                n_invalid, self.cfg.ranges_init.pos_y, self.cfg.ranges_final.pos_y, count)
            pos_z_world[invalid_mask] = self._sample_with_curriculum(
                n_invalid, self.cfg.ranges_init.pos_z, self.cfg.ranges_final.pos_z, count)
            pos_z[invalid_mask] = pos_z_world[invalid_mask] - root_z[invalid_mask]
    
            # Update and re-check
            pos_b[invalid_mask] = torch.stack([pos_x[invalid_mask], pos_y[invalid_mask], pos_z[invalid_mask]], dim=1)
            valid[invalid_mask] = self._check_constraints(pos_b[invalid_mask])

        # Store sampled positions
        self.pose_command_b[env_ids, 0] = pos_x
        self.pose_command_b[env_ids, 1] = pos_y
        self.pose_command_b[env_ids, 2] = pos_z
        self.pose_command_w_z[env_ids, 0] = pos_z_world

        # ========== Sample orientation RELATIVE TO CURRENT EE FRAME ==========
        # Get current EE orientation in base frame: ee_quat_b = chassis_quat^{-1} * ee_quat_w
        ee_quat_w = self.robot.data.body_state_w[env_ids, self.body_idx, 3:7]
        ee_quat_b = quat_mul(quat_conjugate(chassis_quat_w), ee_quat_w)

        # Sample relative rotation (perturbation in EE local frame)
        roll_rel = self._sample_with_curriculum(n, self.cfg.ranges_init.roll, self.cfg.ranges_final.roll, count)
        pitch_rel = self._sample_with_curriculum(n, self.cfg.ranges_init.pitch, self.cfg.ranges_final.pitch, count)
        yaw_rel = self._sample_with_curriculum(n, self.cfg.ranges_init.yaw, self.cfg.ranges_final.yaw, count)
        quat_rel = quat_from_euler_xyz(roll_rel, pitch_rel, yaw_rel)

        # Target orientation: q_target_b = q_ee_b * q_rel
        quat_target_b = quat_mul(ee_quat_b, quat_rel)
        self.pose_command_b[env_ids, 3:] = quat_unique(quat_target_b) if self.cfg.make_quat_unique else quat_target_b

    def _update_command(self):
        pass

    def _update_metrics(self):
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        self.pose_command_w[:, 2] = self.pose_command_w_z[:, 0]

        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
        )

        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.current_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        body_pose_w = self.robot.data.body_state_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_pose_w[:, :3], body_pose_w[:, 3:7])


@configclass
class EEPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for end-effector pose command generator."""

    class_type: type = EEPoseCommand

    chassis_sphere_radius: float = None
    """Chassis collision sphere radius (meters)."""

    curriculum_coeff: float = None
    """Number of iterations to reach final ranges."""

    ranges_init: UniformPoseCommandCfg.Ranges = None
    """Initial sampling ranges."""

    ranges_final: UniformPoseCommandCfg.Ranges = None
    """Final sampling ranges."""
