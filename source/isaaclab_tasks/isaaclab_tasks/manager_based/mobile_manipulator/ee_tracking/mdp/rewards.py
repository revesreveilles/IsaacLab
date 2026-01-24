"""Reward functions for mobile manipulator tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ====================
# Core EE Tracking Rewards
# ====================


def ee_position_command_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 1.0,
) -> torch.Tensor:
    """Exponential kernel reward for end-effector position tracking.

    Higher reward when closer to target, smoothly decreasing with distance.

    Args:
        env: Environment instance.
        command_name: Name of the command term.
        asset_cfg: Asset configuration to specify which body to track.
        std: Standard deviation for exponential kernel.
    """
    # Get current EE position in world frame
    asset: Articulation = env.scene[asset_cfg.name]
    curr_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]

    # Get desired position in world frame (NOT base frame!)
    command_term = env.command_manager.get_term(command_name)
    des_pos_w = command_term.pose_command_w[:, :3]

    # Calculate distance and reward
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return torch.exp(-distance / std)


def orientation_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize orientation error between EE and target orientation.

    Returns axis-angle magnitude of orientation error (radians).
    """
    command_term = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    des_quat_w = command_term.pose_command_w[:, 3:7]

    curr_quat_inv = quat_conjugate(curr_quat_w)
    error_quat = quat_mul(des_quat_w, curr_quat_inv)

    axis_angle = axis_angle_from_quat(error_quat)
    return torch.norm(axis_angle, dim=1)


def orientation_command_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float = 0.5,
) -> torch.Tensor:
    """Exponential kernel reward for end-effector orientation tracking.

    Higher reward when orientation is closer to target.

    Args:
        env: Environment instance.
        command_name: Name of the command term.
        asset_cfg: Asset configuration to specify which body to track.
        std: Standard deviation for exponential kernel (radians).
             Smaller std = sharper reward curve.
    """
    command_term = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    des_quat_w = command_term.pose_command_w[:, 3:7]

    # Compute quaternion error: q_error = q_des * q_curr^{-1}
    curr_quat_inv = quat_conjugate(curr_quat_w)
    error_quat = quat_mul(des_quat_w, curr_quat_inv)

    # Convert to axis-angle and get rotation magnitude
    axis_angle = axis_angle_from_quat(error_quat)
    angle_error = torch.norm(axis_angle, dim=1)  # radians

    return torch.exp(-angle_error / std)


# ====================
# Action Regularization
# ====================

def arm_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of arm actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, 3:9] - env.action_manager.prev_action[:, 3:9]), dim=1)


def base_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of base actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, :3] - env.action_manager.prev_action[:, :3]), dim=1)


# ====================
# State Regularization (Safety)
# ====================

def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    When the robot is flat, projected_gravity_b ≈ [0, 0, -1], so xy components are ~0.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


# ====================
# Additional Useful Rewards
# ====================

def ee_pose_command_error_obs(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Return position error as observation (for critic).

    Returns:
        Scalar distance to target [num_envs, 1]
    """
    command_term = env.command_manager.get_term(command_name)
    return command_term.metrics["position_error"].unsqueeze(-1)


def ee_traj_pose_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std_pos: float = 0.2,
    std_rot: float = 0.5,
) -> torch.Tensor:
    """Exponential reward for SE(3) tracking with squared error."""
    command_term = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]

    pos_error, rot_error = compute_pose_error(
        command_term.pose_command_w[:, :3],
        command_term.pose_command_w[:, 3:7],
        asset.data.body_state_w[:, asset_cfg.body_ids[0], :3],
        asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7],
    )

    # 归一化后的 SE(3) "距离"
    normalized_pos = pos_error / std_pos
    normalized_rot = rot_error / std_rot

    # 统一6维向量范数
    se3_error = torch.cat([normalized_pos, normalized_rot], dim=1)
    error_norm = torch.norm(se3_error, dim=1)

    return torch.exp(-(error_norm ** 2))
