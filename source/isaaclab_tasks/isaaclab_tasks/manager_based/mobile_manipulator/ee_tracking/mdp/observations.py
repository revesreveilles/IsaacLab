"""Observation functions for mobile manipulator EE tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


# ====================
# Base State Observations
# ====================

def base_lin_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root linear velocity in base frame. Shape: (num_envs, 3)"""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root angular velocity in base frame. Shape: (num_envs, 3)"""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gravity vector projected into the root frame. Shape: (num_envs, 3)

    Note: Only needed for uneven terrain or slope navigation.
    For flat ground tasks, this can be removed.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


# ====================
# Arm Joint Observations
# ====================

def joint_pos_rel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Arm joint positions relative to default. Shape: (num_envs, num_joints)

    Uses joint_ids from asset_cfg for efficiency.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )


def joint_vel_rel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Arm joint velocities relative to default. Shape: (num_envs, num_joints)

    Uses joint_ids from asset_cfg for efficiency.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        asset.data.joint_vel[:, asset_cfg.joint_ids]
        - asset.data.default_joint_vel[:, asset_cfg.joint_ids]
    )


# ============================================================================
# End-Effector Observations
# ============================================================================

def body_pos_w(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Body position in world frame.

    Args:
        asset_cfg: Must specify body_names or body_ids

    Returns:
        Shape: (num_envs, 3)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    return asset.data.body_state_w[:, body_idx, :3]


def body_quat_w(
    env: ManagerBasedEnv, 
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Body orientation (quaternion) in world frame.

    Args:
        asset_cfg: Must specify body_names or body_ids

    Returns:
        Shape: (num_envs, 4) - [w, x, y, z]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_idx = asset_cfg.body_ids[0]
    return asset.data.body_state_w[:, body_idx, 3:7]


# ====================
# Command-Related Observations
# ====================

def ee_pose_command_error_obs(
    env: ManagerBasedEnv, command_name: str
) -> torch.Tensor:
    """End-effector pose error (position + orientation) as privileged obs.
    Shape: (num_envs, 2) - [position_error_norm, orientation_error_norm]
    """
    cmd_term = env.command_manager.get_term(command_name)
    pos_error = cmd_term.metrics["position_error"]
    ori_error = cmd_term.metrics["orientation_error"]
    return torch.stack([pos_error, ori_error], dim=-1)


def ee_command_error_obs(
    env: ManagerBasedEnv,
    command_name: str,
    std_pos: float,
    std_rot: float,
) -> torch.Tensor:
    """Extended tracking error as privileged observation.

    Returns:
        Shape: (num_envs, 4)
        - [0]: position error norm (meters)
        - [1]: orientation error norm (radians)
        - [2]: normalized position term: (pos_error/std_pos)²
        - [3]: normalized rotation term: (rot_error/std_rot)²
    """
    cmd_term = env.command_manager.get_term(command_name)

    pos_error = cmd_term.metrics["position_error"]  # (num_envs,)
    ori_error = cmd_term.metrics["orientation_error"]  # (num_envs,)

    # 归一化并平方（与 reward 函数一致）
    pos_term = (pos_error / std_pos) ** 2
    rot_term = (ori_error / std_rot) ** 2

    return torch.stack([pos_error, ori_error, pos_term, rot_term], dim=-1)
