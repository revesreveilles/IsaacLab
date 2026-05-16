"""Reward functions for mobile manipulator tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_mul, quat_conjugate, axis_angle_from_quat, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.constants import (
    DYN_ALL_NAMES,
    DYN_CUBOID_NAMES,
    DYN_CYLINDER_NAMES,
    CUBOID_HALF_EXTENTS,
    CYLINDER_PARAMS,
)


def _delta_reward_rate(
    env: ManagerBasedRLEnv,
    delta: torch.Tensor,
    min_delta: float,
    max_delta: float,
) -> torch.Tensor:
    """Return delta as a rate so RewardManager dt integration recovers delta."""
    step_dt = max(float(getattr(env, "step_dt", 1.0)), 1e-8)
    return delta.clamp(min_delta, max_delta) / step_dt


def _event_reward_rate(env: ManagerBasedRLEnv, event: torch.Tensor) -> torch.Tensor:
    """Return event indicator as a rate so one-shot weights are preserved."""
    step_dt = max(float(getattr(env, "step_dt", 1.0)), 1e-8)
    return event.float() / step_dt


# ====================
# Action Regularization
# ====================

def arm_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of arm actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, 3:9] - env.action_manager.prev_action[:, 3:9]), dim=1)


def base_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of base actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action[:, :3] - env.action_manager.prev_action[:, :3]), dim=1)


def arm_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the absolute magnitude of arm actions to prevent action drift."""
    return torch.sum(torch.square(env.action_manager.action[:, 3:9]), dim=1)


def base_action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the absolute magnitude of base actions to prevent action drift."""
    return torch.sum(torch.square(env.action_manager.action[:, :3]), dim=1)


# ====================
# State Regularization (Safety)
# ====================

def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize aggressive base motion that tends to cause tipping."""
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        0.8 * torch.square(asset.data.root_lin_vel_b[:, 2]) +
        0.2 * torch.sum(torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1)
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm(asset.data.projected_gravity_b[:, :2], dim=1)


# ====================
# Additional Useful Rewards
# ====================

def ee_traj_pose_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std_pos: float,
    std_rot: float,
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
    safe_std_pos = max(std_pos, 1e-6)
    safe_std_rot = max(std_rot, 1e-6)
    # 归一化后的 SE(3) "距离"
    normalized_pos = pos_error / safe_std_pos
    normalized_rot = rot_error / safe_std_rot

    # 统一6维向量范数
    se3_error = torch.cat([normalized_pos, normalized_rot], dim=1)
    error_norm = (torch.square(se3_error).sum(dim=1) + 1e-8).sqrt()

    return torch.exp(-(error_norm))


def ee_position_tracking_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
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
    ee_pos_error = torch.sum(torch.abs(curr_pos_w - des_pos_w), dim=1)
    return torch.exp(-ee_pos_error / std)


def ee_orientation_tracking_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Exponential kernel reward for end-effector orientation tracking.

    Higher reward when orientation is closer to target, smoothly decreasing with error.

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
    curr_quat_w = curr_quat_w / curr_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    curr_quat_inv = quat_conjugate(curr_quat_w)
    error_quat = quat_mul(des_quat_w, curr_quat_inv)
    axis_angle = axis_angle_from_quat(error_quat)
    angle_error = torch.sum(torch.abs(axis_angle), dim=1)
    return torch.exp(-angle_error / std)

# ================================================================
# Reference-based Main Tracking Rewards (at s_ref)
# ================================================================
#
# r_main = α_p * Δd_p + β_p * φ_p(d_p)
#        + w_o(d_p) * (α_o * Δd_o + β_o * φ_o(d_o))
#
# where:
#   d_p = position_error at s_ref
#   d_o = orientation_error at s_ref
#   Δd_p = d_p_prev − d_p_now   (>0 = approaching)
#   Δd_o = d_o_prev − d_o_now   (>0 = improving)
#   w_o  = sigmoid(k * (d_switch − d_p))  (position-first gating)
# ================================================================


def reference_position_approach_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_delta: float = 0.05,
) -> torch.Tensor:
    """Reward for EE approaching the s_ref reference position.

    Returns clamp(Δd_p, −max_delta, max_delta).  Signed: positive when
    the position error is decreasing, negative when increasing.
    This prevents reward exploitation via oscillation.

    Math:  α_p * clamp(d_p_prev − d_p_now, −max_delta, max_delta)
    """
    command_term = env.command_manager.get_term(command_name)
    delta = command_term.metrics["position_progress_delta"]
    w_arm = 1.0 - command_term._base_task_weight
    return w_arm * _delta_reward_rate(env, delta, -max_delta, max_delta)


def _local_reward_progress_gate(
    command_term,
    delta_s_norm: float = 0.01,
    delta_g_norm: float = 0.01,
) -> torch.Tensor:
    """Stall-aware gate for local tracking rewards.

    Returns a scalar gate in [0, 1] per environment.  The gate is high
    when the robot is making real progress (either along the path via
    Δs_proj, or toward the final goal via Δd_final), and low when the
    robot is stalled.

    This prevents "stand-still-and-collect" exploits where the policy
    parks near a fixed s_ref and harvests unconditional tracking reward.

    Args:
        command_term: The EETrajectoryCommand instance.
        delta_s_norm: Normalisation constant for Δs_proj gate.
        delta_g_norm: Normalisation constant for Δd_final gate.

    Returns:
        [N] tensor in [0, 1].
    """
    progress_gate = (
        command_term._robot_progress_delta / max(delta_s_norm, 1e-8)
    ).clamp(0.0, 1.0)

    goal_gate = (
        command_term._final_goal_progress_delta / max(delta_g_norm, 1e-8)
    ).clamp(0.0, 1.0)

    return torch.max(progress_gate, goal_gate)


def reference_position_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.10,
    delta_s_norm: float = 0.01,
    delta_g_norm: float = 0.01,
) -> torch.Tensor:
    """Progress-conditioned position tracking quality at s_ref.

    The tracking bonus exp(−d_p / std) is multiplied by a progress gate
    so it is only paid when the robot is actually advancing along the path
    or closing the final goal.  When stalled (Δs_proj ≈ 0 and Δd_final ≈ 0),
    the reward → 0, preventing the "park and collect" exploit.

    Math:  w_arm * active_gate * exp(−d_p / std_pos)
    """
    command_term = env.command_manager.get_term(command_name)
    d_p = command_term.metrics["position_error"]
    w_arm = 1.0 - command_term._base_task_weight
    gate = _local_reward_progress_gate(command_term, delta_s_norm, delta_g_norm)
    return w_arm * gate * torch.exp(-d_p / max(std, 1e-6))


def reference_orientation_approach_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_delta: float = 0.10,
) -> torch.Tensor:
    """Reward for improving orientation alignment toward s_ref reference.

    Gated by w_o(d_p): orientation matters only when position is close.
    Returns w_o * clamp(Δd_o, −max_delta, max_delta).

    Math:  α_o * w_o(d_p) * clamp(d_o_prev − d_o_now, −max_delta, max_delta)
    """
    command_term = env.command_manager.get_term(command_name)
    delta = command_term.metrics["orientation_progress_delta"]
    d_p = command_term.metrics["position_error"]
    cfg = command_term.cfg
    w_o = torch.sigmoid(cfg.ori_weight_k * (cfg.ori_weight_switch_distance - d_p))
    w_arm = 1.0 - command_term._base_task_weight
    return w_arm * w_o * _delta_reward_rate(env, delta, -max_delta, max_delta)


def reference_orientation_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float = 0.30,
    delta_s_norm: float = 0.01,
    delta_g_norm: float = 0.01,
) -> torch.Tensor:
    """Progress-conditioned orientation tracking quality at s_ref.

    Gated by w_o(d_p) (position-first) AND by the progress gate so
    orientation tracking reward is only paid during real task advance.

    Math:  w_arm * active_gate * w_o(d_p) * exp(−d_o / std_ori)
    """
    command_term = env.command_manager.get_term(command_name)
    d_p = command_term.metrics["position_error"]
    d_o = command_term.metrics["orientation_error"]
    cfg = command_term.cfg
    w_o = torch.sigmoid(cfg.ori_weight_k * (cfg.ori_weight_switch_distance - d_p))
    w_arm = 1.0 - command_term._base_task_weight
    gate = _local_reward_progress_gate(command_term, delta_s_norm, delta_g_norm)
    return w_arm * gate * w_o * torch.exp(-d_o / max(std, 1e-6))


# ================================================================
# Reachability-Gated Base Participation Rewards
# ================================================================
#
# rho_base = ||p_ref_xy(s_ref) - p_arm_mount_xy||
#
# r_base = alpha_b * base_reachability_approach_reward
#        + beta_b  * base_reachability_tracking_reward
#
# excess = relu(rho_base - arm_workspace_radius)
# w_base = 1 - exp(-(excess / base_switch_sigma_m)^2)
#
# Therefore:
# - rho_base <= arm_workspace_radius: w_base = 0, arm rewards dominate.
# - rho_base > arm_workspace_radius: w_base rises smoothly with excess reach.
#
# The approach term is multiplied by w_base. The tracking term is a positive
# workspace-membership bonus: it is 1 inside the comfort radius and decays
# outside it. Multiplying that bonus by w_base creates an unintended reward
# shell just outside the radius.
# ================================================================


def base_reachability_approach_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_delta: float = 0.05,
) -> torch.Tensor:
    """Delta reward encouraging the base to reduce rho_base.

    Returns clamp(rho_base_prev - rho_base_now, −max_delta, max_delta).
    Signed: positive when getting closer, negative when moving away.

    Math:  alpha_b * clamp(delta_rho, −max_delta, max_delta)
    """
    command_term = env.command_manager.get_term(command_name)
    delta = command_term._rho_base_delta
    w_base = command_term._base_task_weight
    return w_base * _delta_reward_rate(env, delta, -max_delta, max_delta)


def base_reachability_tracking_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    arm_workspace_radius: float = 0.55,
    std_base: float = 0.10,
) -> torch.Tensor:
    """Positive reachability membership bonus for the base-arm handoff.

    Inside the arm comfort radius this returns 1.0, giving no gradient to move
    the base closer than necessary. Outside the radius it smoothly decays with
    excess reachability distance. This keeps the reward positive while avoiding
    the old shell-shaped optimum from w_base * exp(-excess / std_base).

    Returns exp(-relu(rho_base - arm_workspace_radius) / std_base).
    """
    command_term = env.command_manager.get_term(command_name)
    rho = command_term._rho_base
    excess = torch.relu(rho - arm_workspace_radius)
    return torch.exp(-excess / max(std_base, 1e-6))


# ================================================================
# Global Path Progress Reward
# ================================================================


def path_progress_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_delta_s: float = 0.05,
) -> torch.Tensor:
    """Global path progress reward based on Δs_proj (robot_progress_delta).

    Returns clamp(robot_progress_delta, −max_delta_s, max_delta_s).
    Signed: positive when the robot's geometric projection advances,
    negative when retreating.  Prevents oscillation exploitation.
    """
    command_term = env.command_manager.get_term(command_name)
    delta = command_term._robot_progress_delta
    return _delta_reward_rate(env, delta, -max_delta_s, max_delta_s)


# ================================================================
# Tube Penalties (at s_proj — constrain deviation from path tube)
# ================================================================
#
# Tube penalties allow small deviations (e.g. for obstacle avoidance)
# but penalise once the error exceeds a deadband.
#
#   tube_penalty = −ReLU(error − deadband)²
# ================================================================
def adaptive_contouring_tube_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
) -> torch.Tensor:
    """Adaptive contouring tube penalty with path-ahead clearance modulated deadband.

    Based on contouring_error at s_proj; the deadband is the dynamic value
    computed by the command term from path_ahead_clearance.

    Returns −ReLU(contouring_error − deadband_dynamic)².
    """
    command_term = env.command_manager.get_term(command_name)
    e = command_term.metrics["contouring_error"]
    deadband = command_term._tube_deadband_dynamic
    return -(torch.relu(e - deadband) ** 2)


def contouring_tube_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 0.02,
) -> torch.Tensor:
    """Tube penalty for contouring error (perpendicular distance to path).

    Based on contouring_error at s_proj.
    Returns −ReLU(contouring_error − deadband)².
    """
    command_term = env.command_manager.get_term(command_name)
    e = command_term.metrics["contouring_error"]
    return -(torch.relu(e - deadband) ** 2)


def time_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant per-step time penalty. Use with negative weight in cfg.

    Returns 1.0 for all envs. Multiply by a small negative weight
    (e.g. -0.02) to discourage standing still and encourage fast completion.
    """
    return torch.ones(env.scene.num_envs, device=env.device)

# ================================================================

def orientation_tube_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    deadband: float = 0.10,
) -> torch.Tensor:
    """Tube penalty for orientation error at the path projection s_proj.

    Based on path_orientation_error at s_proj.
    Returns −ReLU(path_orientation_error − deadband)².
    """
    command_term = env.command_manager.get_term(command_name)
    e = command_term.metrics["path_orientation_error"]
    return -(torch.relu(e - deadband) ** 2)


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_tolerance: float = 0.1,
    ori_tolerance: float | None = None,
) -> torch.Tensor:
    """One-time bonus when goal is reached.

    Returns 1.0 when goal reached, 0.0 otherwise.
    """
    command_term = env.command_manager.get_term(command_name)
    pos_error = command_term.metrics["final_position_error"]

    reached = pos_error < pos_tolerance

    if ori_tolerance is not None:
        ori_error = command_term.metrics["final_orientation_error"]
        reached = reached & (ori_error < ori_tolerance)

    return _event_reward_rate(env, reached)


def terminated_event_reward(
    env: ManagerBasedRLEnv,
    term_keys: list[str],
) -> torch.Tensor:
    """One-shot termination event reward for non-timeout termination terms."""
    event = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    for key in term_keys:
        event = event | env.termination_manager.get_term(key)

    time_outs = getattr(env.termination_manager, "time_outs", None)
    if time_outs is not None:
        event = event & (~time_outs)

    return _event_reward_rate(env, event)


# ====================
# Collision Avoidance Reward
# ====================

def collision_avoidance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    d_safe: float,
    d_repulsion: float,
    repulsion_weight: float,
    hard_penalty: float,
    ground_clearance: float,
) -> torch.Tensor:
    """Dual-threshold exponential collision avoidance reward (collision spheres + lidar).

    For each collision sphere, computes the minimum surface distance to all
    lidar points, then applies:

      d_surface > d_repulsion        → 0  (safe, no penalty)
      d_safe < d_surface <= d_repulsion → exp(-w * penetration_depth) - 1
      d_surface <= d_safe             → hard_penalty

    where penetration_depth = (d_repulsion - d_surface).clamp(min=0).

    Args:
        env: Environment instance.
        sensor_cfg: LiDAR RayCaster sensor config.
        asset_cfg: Robot articulation config.
        d_safe: Hard-penalty threshold (m). Below this → hard_penalty.
        d_repulsion: Outer boundary of repulsion zone (m).
        repulsion_weight: Exponential decay rate in repulsion zone.
        hard_penalty: Constant penalty when d_surface <= d_safe.
        ground_clearance: Fixed absolute height threshold (m). Hits below this z are ground.

    Returns:
        torch.Tensor: shape [num_envs], sum of per-sphere rewards.
    """
    from .observations import _compute_sphere_lidar

    cache = _compute_sphere_lidar(
        env,
        sensor_cfg.name,
        asset_cfg.name,
        ground_clearance,
    )
    d_surface = cache["d_surface"]  # [E, S], signed surface clearance

    # 2. 计算侵入排斥区的深度
    penetration_depth = (d_repulsion - d_surface).clamp(min=0)

    # 3. 双阈值指数奖励
    #   d_surface > d_safe: exp(-w * penetration_depth) - 1  (safe 区 → 0, 排斥区 → 平滑惩罚)
    #   d_surface <= d_safe: hard_penalty
    hard_value = d_surface.new_full((), float(hard_penalty))
    reward_per_sphere = torch.where(
        d_surface > d_safe,
        torch.exp(-repulsion_weight * penetration_depth) - 1.0,
        hard_value,
    )

    # sum over all spheres → [E]
    return reward_per_sphere.sum(dim=-1)


def _relaxed_log_barrier(
    h: torch.Tensor,
    delta: float,
    mu: float,
) -> torch.Tensor:
    """Relaxed logarithmic barrier.

    For ``h >= delta``:

        B(h) = -mu * log(h)

    For ``h < delta``, use a quadratic extension that matches value and
    first derivative at ``h = delta``:

        B(h) = mu * (0.5 * ((h - 2*delta) / delta)^2 - log(delta) - 0.5)

    This keeps the barrier finite for ``h <= 0``, which is important for PPO
    value stability, while preserving increasing cost near the safety
    boundary.
    """
    delta_safe = max(float(delta), 1e-6)
    mu_safe = max(float(mu), 0.0)

    h_clamped = h.clamp_min(1e-12)
    log_branch = -mu_safe * torch.log(h_clamped)

    delta_tensor = torch.tensor(delta_safe, device=h.device, dtype=h.dtype)
    quad = 0.5 * ((h - 2.0 * delta_safe) / delta_safe).square()
    quad_branch = mu_safe * (quad - torch.log(delta_tensor) - 0.5)

    return torch.where(h >= delta_safe, log_branch, quad_branch)


def dynamic_obstacle_cbf_reward(
    env: ManagerBasedRLEnv,
    obstacle_asset_names: tuple[str, ...] = DYN_ALL_NAMES,
    cuboid_names: tuple[str, ...] = DYN_CUBOID_NAMES,
    cylinder_names: tuple[str, ...] = DYN_CYLINDER_NAMES,
    cuboid_half_extents: tuple[
        tuple[float, float, float], ...
    ] = CUBOID_HALF_EXTENTS,
    cylinder_params: tuple[
        tuple[float, float], ...
    ] = CYLINDER_PARAMS,
    d_safe: float = 0.12,
    d_trigger: float = 1.00,
    t_react: float = 0.80,
    barrier_delta: float = 0.08,
    barrier_mu: float = 0.05,
    max_penalty: float = 2.0,
) -> torch.Tensor:
    """Dynamic-obstacle relaxed barrier value reward.

    Uses the closest collision sphere for each dynamic obstacle.

    Define:

        d = signed surface clearance
        d_dot = distance rate along sphere -> obstacle center direction
        closing = relu(-d_dot)
        h_dyn = d - d_safe - t_react * closing

    ``h_dyn >= 0`` means the current clearance is large enough to preserve
    ``d_safe`` after ``t_react`` seconds at the current closing speed.  The
    reward penalizes the relaxed log-barrier value of ``h_dyn``, normalized so
    obstacles at ``d_trigger`` with zero closing speed have zero penalty.

    Parameter meanings:
        d_safe: Minimum desired signed surface clearance.
        d_trigger: Reward activation range; farther obstacles have no penalty.
        t_react: Reaction-time margin. Larger values are more conservative.
        barrier_delta: Relaxed barrier switch point.
        barrier_mu: Barrier scale.
        max_penalty: Per-step max obstacle penalty clamp.

    This is a barrier-inspired soft constraint for RL, not a formal CBF-QP
    safety filter.

    Returns:
        torch.Tensor: [num_envs], negative max-obstacle penalty.
    """
    from .curriculum_events import _get_dynamic_obstacle_active_limit
    from .observations import _compute_dynamic_obstacle_distances, _get_collision_bubbles

    num_envs = env.scene.num_envs
    active_limit = _get_dynamic_obstacle_active_limit(env, len(obstacle_asset_names))
    if active_limit <= 0:
        return torch.zeros(num_envs, device=env.device)

    dyn_cache = _compute_dynamic_obstacle_distances(
        env,
        obstacle_asset_names,
        cuboid_names,
        cylinder_names,
        cuboid_half_extents,
        cylinder_params,
    )
    if int(dyn_cache["n_total"]) == 0:
        return torch.zeros(num_envs, device=env.device)

    d_surface = dyn_cache["per_obstacle_min_dist"]  # [E, O]
    closest_sphere = dyn_cache["closest_sphere"]    # [E, O]
    sphere_pos_all = dyn_cache["centers_w"]         # [E, S, 3]
    obs_pos = dyn_cache["obs_pos"]                  # [E, O, 3]
    obs_vel = dyn_cache["obs_vel"]                  # [E, O, 3]
    active_mask = dyn_cache["active_mask"]          # [E, O]

    bubbles = _get_collision_bubbles(env)
    sphere_vel_all = bubbles.get_world_sphere_velocities()

    idx3 = closest_sphere.unsqueeze(-1).expand(-1, -1, 3)
    closest_sphere_pos = sphere_pos_all.gather(1, idx3)
    closest_sphere_vel = sphere_vel_all.gather(1, idx3)

    rel_pos = obs_pos - closest_sphere_pos
    n = rel_pos / rel_pos.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    v_rel = obs_vel - closest_sphere_vel
    d_dot = (n * v_rel).sum(dim=-1)

    closing = torch.relu(-d_dot)
    h_dyn = d_surface - d_safe - t_react * closing
    barrier = _relaxed_log_barrier(
        h_dyn,
        delta=barrier_delta,
        mu=barrier_mu,
    )

    h_free = torch.tensor(
        float(d_trigger - d_safe),
        device=env.device,
        dtype=d_surface.dtype,
    )
    barrier_free = _relaxed_log_barrier(
        h_free,
        delta=barrier_delta,
        mu=barrier_mu,
    )

    penalty = torch.relu(barrier - barrier_free)

    valid = active_mask & (d_surface < d_trigger) & (d_surface < 1e5)
    penalty = penalty.masked_fill(~valid, 0.0)
    max_penalty_value = max(float(max_penalty), 0.0)
    penalty = penalty.clamp(max=max_penalty_value)
    penalty = torch.nan_to_num(
        penalty,
        nan=0.0,
        posinf=max_penalty_value,
        neginf=0.0,
    )

    reward = -penalty.max(dim=-1).values
    return torch.nan_to_num(
        reward,
        nan=0.0,
        posinf=0.0,
        neginf=-max_penalty_value,
    )


def static_full_body_log_distance_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    d_trigger: float = 0.60,
    eps: float = 1e-3,
    max_penalty: float = 2.0,
    ground_clearance: float = 0.1,
    aggregation: str = "max",
) -> torch.Tensor:
    """Full-body static-obstacle log-distance reward baseline.

    Uses collision-sphere to LiDAR point-cloud clearance from
    _compute_sphere_lidar. This is a nominal distance-only baseline:
    no CBF, no barrier value, no velocity, no d_safe.

    penalty(d) = log((d_trigger + eps) / (clip(d, 0, d_trigger) + eps))

    reward = -aggregate_spheres(penalty)
    """
    from .observations import _compute_sphere_lidar

    cache = _compute_sphere_lidar(
        env,
        sensor_cfg.name,
        asset_cfg.name,
        ground_clearance,
    )

    d_surface = cache["d_surface"]  # [E, S]

    trigger = max(float(d_trigger), 1e-6)
    eps_val = max(float(eps), 1e-9)
    max_penalty_value = max(float(max_penalty), 0.0)

    valid = d_surface < trigger
    d_clip = d_surface.clamp(min=0.0, max=trigger)

    penalty = torch.log(
        torch.tensor(trigger + eps_val, device=env.device, dtype=d_surface.dtype)
        / (d_clip + eps_val)
    )

    penalty = penalty.masked_fill(~valid, 0.0)
    penalty = penalty.clamp(max=max_penalty_value)
    penalty = torch.nan_to_num(
        penalty,
        nan=0.0,
        posinf=max_penalty_value,
        neginf=0.0,
    )

    if aggregation == "mean":
        reward = -penalty.mean(dim=-1)
    else:
        reward = -penalty.max(dim=-1).values

    return torch.nan_to_num(
        reward,
        nan=0.0,
        posinf=0.0,
        neginf=-max_penalty_value,
    )


def dynamic_full_body_log_distance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_asset_names: tuple[str, ...] = DYN_ALL_NAMES,
    cuboid_names: tuple[str, ...] = DYN_CUBOID_NAMES,
    cylinder_names: tuple[str, ...] = DYN_CYLINDER_NAMES,
    cuboid_half_extents: tuple[
        tuple[float, float, float], ...
    ] = CUBOID_HALF_EXTENTS,
    cylinder_params: tuple[
        tuple[float, float], ...
    ] = CYLINDER_PARAMS,
    d_trigger: float = 1.00,
    eps: float = 1e-3,
    max_penalty: float = 2.0,
    aggregation: str = "max",
) -> torch.Tensor:
    """Full-body dynamic-obstacle log-distance reward baseline.

    Uses per-obstacle full-body closest signed surface clearance from
    _compute_dynamic_obstacle_distances.

    This is a nominal distance-only baseline:
    no velocity, no closing speed, no TTC, no CBF violation, no barrier value,
    no d_safe.

    penalty(d) = log((d_trigger + eps) / (clip(d, 0, d_trigger) + eps))

    reward = -aggregate_obstacles(penalty)
    """
    from .curriculum_events import _get_dynamic_obstacle_active_limit
    from .observations import _compute_dynamic_obstacle_distances

    _ = asset_cfg
    num_envs = env.scene.num_envs
    active_limit = _get_dynamic_obstacle_active_limit(
        env,
        len(obstacle_asset_names),
    )
    if active_limit <= 0:
        return torch.zeros(num_envs, device=env.device)

    dyn_cache = _compute_dynamic_obstacle_distances(
        env,
        obstacle_asset_names,
        cuboid_names,
        cylinder_names,
        cuboid_half_extents,
        cylinder_params,
    )
    if int(dyn_cache["n_total"]) == 0:
        return torch.zeros(num_envs, device=env.device)

    d_surface = dyn_cache["per_obstacle_min_dist"]  # [E, O]
    active_mask = dyn_cache["active_mask"]          # [E, O]

    trigger = max(float(d_trigger), 1e-6)
    eps_val = max(float(eps), 1e-9)
    max_penalty_value = max(float(max_penalty), 0.0)

    valid = active_mask & (d_surface < trigger) & (d_surface < 1e5)
    d_clip = d_surface.clamp(min=0.0, max=trigger)

    penalty = torch.log(
        torch.tensor(trigger + eps_val, device=env.device, dtype=d_surface.dtype)
        / (d_clip + eps_val)
    )

    penalty = penalty.masked_fill(~valid, 0.0)
    penalty = penalty.clamp(max=max_penalty_value)
    penalty = torch.nan_to_num(
        penalty,
        nan=0.0,
        posinf=max_penalty_value,
        neginf=0.0,
    )

    if aggregation == "mean":
        reward = -penalty.mean(dim=-1)
    else:
        reward = -penalty.max(dim=-1).values

    return torch.nan_to_num(
        reward,
        nan=0.0,
        posinf=0.0,
        neginf=-max_penalty_value,
    )


# ====================
# Time Penalty
# ====================

def timeout_terminal_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-shot penalty triggered only when the episode ends by timeout.

    Returns 1.0 for envs where the 'time_out' termination term fired
    on this step, 0.0 otherwise.
    """
    return _event_reward_rate(env, env.termination_manager.get_term("time_out"))


# ================================================================
# Final-Goal Approach Dense Reward
# ================================================================


def final_goal_approach_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    max_delta: float = 0.05,
) -> torch.Tensor:
    """Dense reward for reducing the distance between EE and the final goal.

    Returns clamp(final_pos_error_prev - final_pos_error_now, −max_delta, max_delta).
    Signed: positive when getting closer, negative when moving away.
    Zero on the first step of each episode (no previous error).

    This is a global completion signal that is always active, not gated
    by base_task_weight.
    """
    command_term = env.command_manager.get_term(command_name)
    delta = command_term._final_goal_progress_delta
    return _delta_reward_rate(env, delta, -max_delta, max_delta)
