from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.constants import (
    DYN_ALL_NAMES,
    DYN_CUBOID_NAMES,
    DYN_CYLINDER_NAMES,
    CUBOID_HALF_EXTENTS,
    CYLINDER_PARAMS,
)


_STALL_CACHE: dict[tuple[int, str, int], dict[str, torch.Tensor | int]] = {}


def ee_pose_goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    pos_tolerance: float = 0.1,
    ori_tolerance: float | None = None,
) -> torch.Tensor:
    """Terminate when the final goal pose is within tolerance.

    Uses only ``metrics[...]`` so this works with any command type that
    exposes ``final_position_error`` and ``final_orientation_error``.
    """
    command_term = env.command_manager.get_term(command_name)
    metrics = command_term.metrics
    pos_ok = metrics["final_position_error"] <= pos_tolerance
    if ori_tolerance is None:
        return pos_ok
    ori_ok = metrics["final_orientation_error"] <= ori_tolerance
    return pos_ok & ori_ok


def out_of_bounds(
    env: ManagerBasedRLEnv,
    max_dist: float = 8.0,
) -> torch.Tensor:
    """Terminate when robot moves too far from env origin.

    Checks 2D (x, y) distance from the environment origin.
    This prevents robots from walking off the sub-terrain.

    Args:
        env: The environment.
        max_dist: Maximum allowed 2D distance from
            env_origin (m). Default 8.0.

    Returns:
        Boolean tensor [num_envs].
    """
    robot = env.scene["robot"]
    pos_w = robot.data.root_pos_w[:, :2]
    origin = env.scene.env_origins[:, :2]
    dist = torch.norm(pos_w - origin, dim=-1)
    return dist > max_dist


def static_obstacle_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    static_collision_margin: float = 0.05,
    ground_clearance: float = 0.1,
) -> torch.Tensor:
    """Return True for environments where the robot collides with static obstacles.

    Static collision is evaluated using collision-sphere-based clearance
    against the static lidar point cloud. Points close to the ground below
    ``ground_clearance`` are ignored consistently with the existing static
    collision / reward implementation.

    Args:
        env: The environment.
        sensor_cfg: LiDAR sensor SceneEntityCfg.
        asset_cfg: Robot SceneEntityCfg.
        static_collision_margin: Safety margin for static obstacles (m).
            d_surface < margin -> collision. Default 0.05.
        ground_clearance: Absolute z threshold for ground filtering (m).

    Returns:
        Boolean tensor [num_envs] indicating static collision.
    """
    from .observations import _compute_sphere_lidar

    cache = _compute_sphere_lidar(
        env,
        sensor_cfg.name,
        asset_cfg.name,
        ground_clearance,
    )
    static_d_surface = cache["d_surface"]  # [E, S]
    return (static_d_surface < static_collision_margin).any(dim=-1)


def dynamic_obstacle_collision(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dynamic_collision_margin: float = 0.05,
    cuboid_names: tuple[str, ...] = DYN_CUBOID_NAMES,
    cylinder_names: tuple[str, ...] = DYN_CYLINDER_NAMES,
    cuboid_half_extents: tuple[
        tuple[float, float, float], ...
    ] = CUBOID_HALF_EXTENTS,
    cylinder_params: tuple[
        tuple[float, float], ...
    ] = CYLINDER_PARAMS,
) -> torch.Tensor:
    """Terminate on analytical dynamic obstacle collision.

    Dynamic collision is evaluated using the same collision-sphere and
    analytical cuboid/cylinder signed-distance geometry used by the dynamic
    obstacle rewards.

    Args:
        env: The environment.
        asset_cfg: Robot SceneEntityCfg. Kept for DoneTerm interface symmetry.
        dynamic_collision_margin: Safety margin for dynamic obstacles (m).
            d_surface < margin -> collision. Default 0.05.
        cuboid_names: Names of dynamic cuboid assets.
        cylinder_names: Names of dynamic cylinder assets.
        cuboid_half_extents: Half-extents per cuboid.
        cylinder_params: (radius, half_height) per cylinder.

    Returns:
        Boolean tensor [num_envs] indicating dynamic collision.
    """
    from .observations import _compute_dynamic_obstacle_distances

    del asset_cfg

    device = env.device
    num_envs = env.scene.num_envs

    dyn_cache = _compute_dynamic_obstacle_distances(
        env,
        DYN_ALL_NAMES,
        cuboid_names,
        cylinder_names,
        cuboid_half_extents,
        cylinder_params,
    )
    if int(dyn_cache["n_total"]) == 0:
        return torch.zeros(num_envs, device=device, dtype=torch.bool)

    return (dyn_cache["per_sphere_min_dist"] < dynamic_collision_margin).any(dim=-1)


def obstacle_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    static_collision_margin: float = 0.05,
    dynamic_collision_margin: float = 0.05,
    ground_clearance: float = 0.1,
    cuboid_names: tuple[str, ...] = DYN_CUBOID_NAMES,
    cylinder_names: tuple[str, ...] = DYN_CYLINDER_NAMES,
    cuboid_half_extents: tuple[
        tuple[float, float, float], ...
    ] = CUBOID_HALF_EXTENTS,
    cylinder_params: tuple[
        tuple[float, float], ...
    ] = CYLINDER_PARAMS,
) -> torch.Tensor:
    """Terminate on any static or dynamic obstacle collision.

    This remains the training/curriculum collision key. Static and dynamic
    collision terms are exposed separately only for logging/diagnostics.

    Args:
        env: The environment.
        sensor_cfg: LiDAR sensor SceneEntityCfg.
        asset_cfg: Robot SceneEntityCfg.
        static_collision_margin: Safety margin for static obstacles (m).
            d_surface < margin → collision. Default 0.05.
        dynamic_collision_margin: Safety margin for dynamic obstacles (m).
            d_surface < margin → collision. Default 0.05.
        ground_clearance: Absolute z threshold for ground filtering (m).
        cuboid_names: Names of dynamic cuboid assets.
        cylinder_names: Names of dynamic cylinder assets.
        cuboid_half_extents: Half-extents per cuboid.
        cylinder_params: (radius, half_height) per cylinder.

    Returns:
        Boolean tensor [num_envs] indicating collision.
    """
    static_hit = static_obstacle_collision(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        static_collision_margin=static_collision_margin,
        ground_clearance=ground_clearance,
    )
    dynamic_hit = dynamic_obstacle_collision(
        env,
        asset_cfg=asset_cfg,
        dynamic_collision_margin=dynamic_collision_margin,
        cuboid_names=cuboid_names,
        cylinder_names=cylinder_names,
        cuboid_half_extents=cuboid_half_extents,
        cylinder_params=cylinder_params,
    )
    return static_hit | dynamic_hit


def proportional_time_out(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_traj",
    nominal_speed: float = 1.0,
    safety_factor: float = 2.0,
    min_timeout_s: float = 20.0,
) -> torch.Tensor:
    """Terminate each environment with a path-length-proportional time budget.

    The timeout budget is computed as

        timeout_s = max(min_timeout_s, safety_factor * path_length / nominal_speed)

    where path_length is the total geometric length of the sampled
    end-effector path. The resulting timeout is converted into simulation
    steps using env.step_dt.
    """
    command_term = env.command_manager.get_term(command_name)

    path_length = command_term.metrics.get(
        "path_total_length",
        command_term._path_total_length,
    )  # [num_envs], meters

    timeout_s = torch.clamp(
        safety_factor * path_length / nominal_speed,
        min=min_timeout_s,
    )

    max_steps = torch.ceil(timeout_s / env.step_dt).long()
    return env.episode_length_buf >= max_steps
