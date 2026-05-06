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
    """Terminate on collision with static or dynamic obstacles.

    Collision check:
      1. **Static**: Use collision spheres + LiDAR point cloud.
         Collision when ``d_surface < static_collision_margin``.
      2. **Dynamic**: Use collision spheres + analytical signed distance
         to cuboids (point-to-AABB) and cylinders.
         Collision when ``d_surface < dynamic_collision_margin``.

    Either collision triggers termination for that environment.

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
    from .observations import _compute_sphere_lidar

    device = env.device
    num_envs = env.scene.num_envs

    # ==================================================================
    # 1. Static collision: collision spheres vs LiDAR point cloud
    # ==================================================================
    cache = _compute_sphere_lidar(
        env, sensor_cfg.name, asset_cfg.name, ground_clearance,
    )
    static_d_surface = cache["d_surface"]  # [E, S]
    # Any sphere closer than margin → collision
    static_collision = (static_d_surface < static_collision_margin).any(dim=-1)  # [E]

    from .observations import _compute_dynamic_obstacle_distances

    dyn_cache = _compute_dynamic_obstacle_distances(
        env,
        DYN_ALL_NAMES,
        cuboid_names,
        cylinder_names,
        cuboid_half_extents,
        cylinder_params,
    )
    if int(dyn_cache["n_total"]) == 0:
        dynamic_collision = torch.zeros(num_envs, device=device, dtype=torch.bool)
    else:
        dynamic_collision = (dyn_cache["per_sphere_min_dist"] < dynamic_collision_margin).any(dim=-1)
    return static_collision | dynamic_collision


def proportional_time_out(
    env: ManagerBasedRLEnv,
    command_name: str = "ee_traj",
    max_speed: float = 1.0,
    safety_factor: float = 4.0,
    min_timeout_s: float = 20.0,
) -> torch.Tensor:
    """Per-env timeout proportional to the episode's route length.

    timeout_s = max(min_timeout_s, safety_factor * path_length / max_speed)
    max_steps = timeout_s / step_dt

    Returns a boolean tensor [num_envs] — True when the env has exceeded
    its individual step budget.
    """
    command_term = env.command_manager.get_term(command_name)
    path_length = command_term.metrics.get(
        "path_total_length", command_term._path_total_length
    )  # [N]
    timeout_s = torch.clamp(safety_factor * path_length / max_speed, min=min_timeout_s)
    max_steps = (timeout_s / env.step_dt).long()
    return env.episode_length_buf >= max_steps
