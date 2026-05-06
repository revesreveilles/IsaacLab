"""Observation functions for mobile manipulator EE tracking task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    subtract_frame_transforms,
    quat_apply_inverse,
    axis_angle_from_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)
from .curriculum_events import (
    _get_dynamic_obstacle_active_limit,
)
from .dynamic_obstacle_buffers import get_dyn_obs_pos_vel, get_dyn_source_indices
from .model import RobotCollisionBubbles

from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.constants import (
    DYN_ALL_NAMES,
    DYN_ALL_SIZES,
    DYN_CUBOID_NAMES,
    DYN_CYLINDER_NAMES,
    CUBOID_HALF_EXTENTS,
    CYLINDER_PARAMS,
)

# ====================
# Base State Observations
# ====================


@generic_io_descriptor(
    units="m/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


@generic_io_descriptor(
    units="rad/s", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


@generic_io_descriptor(
    units="m/s^2", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


# ====================
# Arm Joint Observations
# ====================

@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s"
)
def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_vel_offsets],
    units="rad/s",
)
def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


# ============================================================================
# End-Effector Observations
# ============================================================================

def ee_pose_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """End-effector pose in robot's base frame (7 dim: pos + quat)."""
    asset: Articulation = env.scene[asset_cfg.name]

    # 获取 EE 和 Base 的世界坐标
    ee_pos_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], :3]
    ee_quat_w = asset.data.body_state_w[:, asset_cfg.body_ids[0], 3:7]

    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    ee_quat_w = ee_quat_w / ee_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    root_quat_w = root_quat_w / root_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, ee_pos_w, ee_quat_w
    )

    return torch.cat([ee_pos_b, ee_quat_b], dim=-1)


# ====================
# Command-Related Observations
# ====================


# ====================
# Collision Sphere to Point Cloud Distance Observation
# ====================

# 模块级缓存: 避免每步都重新创建 RobotCollisionBubbles
_collision_bubbles_cache: dict = {}
# 每步共享缓存: obs 计算后, reward 直接复用 d_surface
_sphere_lidar_cache: dict = {}


def _compute_sphere_lidar(
    env,
    sensor_cfg_name: str = "lidar",
    asset_cfg_name: str = "robot",
    ground_clearance: float = 0.1,
    ray_subsample: int = 1,
):
    """球-点云距离的共享计算 (obs 和 reward 复用)。

    Step-based 缓存: 同一步内 Policy → Critic → Reward 共享结果,
    避免重复 cdist [E, 24, 144] 计算。

    Args:
        env: 环境实例。
        sensor_cfg_name: LiDAR sensor 名称。
        asset_cfg_name: 机器人 asset 名称。
        ground_clearance: 固定绝对高度阈值 (m)，低于此高度的点云视为地面并过滤。
        ray_subsample: 射线降采样步长。

    Returns:
        dict with d_surface, nearest, sphere_centers_w, etc.
    """
    # ── Step-based cache: Policy → Critic → Reward 共享 ──
    cache_key = id(env)
    step = env.common_step_counter
    cached = _sphere_lidar_cache.get(cache_key)
    if cached is not None and cached.get("_step") == step:
        return cached

    device = env.device
    num_envs = env.scene.num_envs

    if cache_key not in _collision_bubbles_cache:
        _collision_bubbles_cache[cache_key] = (
            RobotCollisionBubbles(env)
        )
    bubbles = _collision_bubbles_cache[cache_key]

    # 获取 LiDAR 数据
    lidar = env.scene[sensor_cfg_name]
    hits_w = lidar.data.ray_hits_w  # [E, R, 3]

    # 先降采样 (view, 无拷贝), 再 clone (更小的张量)
    hits_sub = hits_w[:, ::ray_subsample, :]  # [E, R', 3]
    valid_hits = hits_sub.clone()

    # 替换 inf (未击中射线)
    invalid = torch.isinf(valid_hits).any(dim=-1)
    valid_hits[invalid] = 1e6

    # Ground filtering — 使用固定绝对高度阈值
    # 地形生成在 z≈0, 低于 ground_clearance 的点都是地面
    is_ground = valid_hits[:, :, 2] < ground_clearance
    valid_hits[is_ground] = 1e6

    # 碰撞球
    sphere_centers_w, sphere_radii = (
        bubbles.get_world_spheres()
    )
    num_spheres = bubbles.num_bubbles

    # 单次 batched cdist [E, S, R']
    all_dist = torch.cdist(sphere_centers_w, valid_hits)
    min_center_dist, min_idx = all_dist.min(dim=-1)
    d_surface = (
        min_center_dist - sphere_radii.unsqueeze(0)
    )

    # 最近点 [E, S, 3]
    env_idx = torch.arange(
        num_envs, device=device
    ).unsqueeze(1)
    nearest = valid_hits[env_idx, min_idx]

    result = {
        "d_surface": d_surface,
        "all_dist": all_dist,
        "nearest": nearest,
        "sphere_centers_w": sphere_centers_w,
        "sphere_radii": sphere_radii,
        "num_spheres": num_spheres,
        "valid_hits": valid_hits,
        "_step": step,
    }
    _sphere_lidar_cache[cache_key] = result
    return result


def sphere_pointcloud_distance(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_range: float | None = None,
    ground_clearance: float = 0.1,
) -> torch.Tensor:
    """碰撞球到点云最短距离观测 (全向量化)。

    每个碰撞球 4 维: [distance_norm, dir_x_b, dir_y_b, dir_z_b]。
    共 N_spheres × 4 维。

    使用 _compute_sphere_lidar 共享计算,
    reward 函数复用相同结果。

    Args:
        env: 环境实例。
        sensor_cfg: LiDAR sensor SceneEntityCfg。
        asset_cfg: 机器人 SceneEntityCfg。
        max_range: 最大有效距离 (m)。若为 None，则读取传感器
            ``sensor.cfg.max_distance``。
        ground_clearance: 固定绝对高度阈值 (m)，低于此高度的点云视为地面。

    Returns:
        torch.Tensor: [num_envs, num_spheres * 4]
    """
    num_envs = env.scene.num_envs
    device = env.device

    # ── 共享计算 (obs 先于 reward 运行, 结果缓存) ──
    cache = _compute_sphere_lidar(
        env, sensor_cfg.name, asset_cfg.name,
        ground_clearance,
    )

    if max_range is None:
        lidar_sensor = env.scene[sensor_cfg.name]
        sensor_cfg_obj = getattr(lidar_sensor, "cfg", None)
        max_range = float(getattr(sensor_cfg_obj, "max_distance", 5.0))

    d_surface = cache["d_surface"]
    nearest = cache["nearest"]
    sphere_centers_w = cache["sphere_centers_w"]
    num_spheres = cache["num_spheres"]

    # 机器人基座姿态
    robot: Articulation = env.scene[asset_cfg.name]
    root_quat_w = robot.data.root_quat_w  # [E, 4]

    # 方向向量 (世界系)
    direction_w = nearest - sphere_centers_w
    dir_norm = direction_w.norm(
        dim=-1, keepdim=True
    ).clamp(min=1e-6)
    direction_w_unit = direction_w / dir_norm

    # Batched quat_apply_inverse: [E*S, 3] 单次调用
    quat_exp = root_quat_w.unsqueeze(1).expand(
        -1, num_spheres, -1
    ).reshape(-1, 4)
    direction_b = quat_apply_inverse(
        quat_exp, direction_w_unit.reshape(-1, 3)
    ).reshape(num_envs, num_spheres, 3)

    # 无有效 hit 时方向置零 (nearest 为 1e6 假点, 方向无语义)
    no_valid_hit = (d_surface > max_range).unsqueeze(-1)  # [E, S, 1]
    direction_b = direction_b.masked_fill(no_valid_hit, 0.0)

    # 距离归一化
    dist_normalized = (
        d_surface.clamp(min=0.0) / max_range
    ).clamp(max=1.0)

    # 组装输出 [E, S*4]
    output = torch.stack([
        dist_normalized,
        direction_b[:, :, 0],
        direction_b[:, :, 1],
        direction_b[:, :, 2],
    ], dim=-1).reshape(num_envs, num_spheres * 4)

    return output


# ===========================================================
# Raw LiDAR Range Scan Observation
# ===========================================================

_lidar_scan_cache: dict = {}


def lidar_range_scan_flat(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_range: float | None = None,
    ground_clearance: float = 0.1,
    ray_subsample: int = 1,
    normalize: bool = True,
) -> torch.Tensor:
    """Raw lidar range scan as a flat 2-D observation.

    Computes per-ray range as the Euclidean distance from the **sensor
    origin** (``pos_w``) to each ray hit point.  Rays with no hit (inf),
    NaN values, and ground reflections are replaced with ``max_range``
    (or normalised 1.0).

    This observation is **additive** — it does NOT replace
    ``sphere_pointcloud_distance``.  Both observations use the same raw
    lidar data but serve different purposes:

    * ``sphere_pointcloud_distance`` → per-sphere closest-point features
    * ``lidar_range_scan_flat``      → dense per-ray range for the policy

    Ground filtering is applied independently in each function because
    they operate on different derived quantities.

    Args:
        env: Environment instance.
        sensor_cfg: LiDAR sensor SceneEntityCfg.
        max_range: Maximum valid range (m). If None, uses the sensor's
            ``max_distance`` setting.
        ground_clearance: Absolute height threshold (m).  Hit points with
            z < ``ground_clearance`` are treated as ground and masked out.
        ray_subsample: Ray stride for optional down-sampling (1 = keep all).
        normalize: If True, output is in [0, 1]; otherwise in metres.

    Returns:
        torch.Tensor: **[num_envs, num_rays']** where
        ``num_rays' = ceil(num_rays / ray_subsample)``.
    """
    # ── Step-based cache: Policy → Critic 共享 ──
    cache_key = (
        id(env),
        sensor_cfg_name,
        asset_cfg_name,
        float(ground_clearance),
        float(max_clearance),
    )
    step = env.common_step_counter
    cached = _lidar_scan_cache.get(cache_key)
    if cached is not None and cached.get("_step") == step:
        return cached["result"]

    lidar = env.scene[sensor_cfg.name]
    if max_range is None:
        sensor_cfg_obj = getattr(lidar, "cfg", None)
        max_range = float(getattr(sensor_cfg_obj, "max_distance", 5.0))

    hits_w = lidar.data.ray_hits_w   # [E, R, 3]
    origin_w = lidar.data.pos_w      # [E, 3]

    # Optional ray sub-sampling (view — zero-copy)
    if ray_subsample > 1:
        hits_sub = hits_w[:, ::ray_subsample, :]
    else:
        hits_sub = hits_w              # [E, R', 3]

    # Range = Euclidean distance from sensor origin to hit point
    displacement = hits_sub - origin_w.unsqueeze(1)   # [E, R', 3]
    ranges = displacement.norm(dim=-1)                 # [E, R']

    # Mask: inf (no hit) / nan
    invalid = torch.isinf(ranges) | torch.isnan(ranges)

    # Ground filtering: hit-points below ground_clearance are ground
    # reflections.  inf z-values will NOT trigger this check.
    is_ground = hits_sub[..., 2] < ground_clearance

    # Apply mask → set to max_range
    ranges = ranges.masked_fill(invalid | is_ground, max_range)

    # Clamp to [0, max_range]
    ranges = ranges.clamp(min=0.0, max=max_range)

    # Normalise to [0, 1] if requested
    if normalize:
        ranges = ranges / max_range

    # Final nan safety net
    fill_val = 1.0 if normalize else max_range
    ranges = torch.nan_to_num(
        ranges, nan=fill_val, posinf=fill_val, neginf=0.0,
    )

    _lidar_scan_cache[cache_key] = {"_step": step, "result": ranges}
    return ranges


# ===========================================================
# Dynamic Obstacle Observation
# ===========================================================

# 模块级缓存: 存储每种动态障碍物的尺寸信息
_dyn_obs_size_cache: dict = {}
_dyn_geom_table_cache: dict = {}
# Step-based cache: shared dynamic-obstacle signed-distance core results.
_dyn_distance_step_cache: dict = {}
# Step-based 缓存: 同一步内 Policy → Critic 共享 dynamic_obstacles 结果
_dyn_obs_step_cache: dict = {}


def _get_collision_bubbles(env) -> RobotCollisionBubbles:
    """Fetch or lazily create the shared robot collision bubbles helper."""
    cache_key = id(env)
    bubbles = _collision_bubbles_cache.get(cache_key)
    if bubbles is None:
        bubbles = RobotCollisionBubbles(env)
        _collision_bubbles_cache[cache_key] = bubbles
    return bubbles


def _freeze_cache_value(value):
    """Recursively convert nested config containers into hashable tuples."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(v) for v in value)
    return value


def _get_cached_static_table(
    cache_name: str,
    values: tuple,
    device: torch.device,
    width: int,
) -> torch.Tensor:
    """Cache small immutable geometry tables on the target device."""
    cache_key = (cache_name, _freeze_cache_value(values), str(device))
    table = _dyn_geom_table_cache.get(cache_key)
    if table is None:
        if len(values) == 0:
            table = torch.zeros(0, width, device=device, dtype=torch.float32)
        else:
            table = torch.tensor(values, device=device, dtype=torch.float32)
        _dyn_geom_table_cache[cache_key] = table
    return table


def _compute_dynamic_obstacle_distances(
    env,
    obstacle_asset_names: tuple[str, ...] = DYN_ALL_NAMES,
    cuboid_names: tuple[str, ...] = DYN_CUBOID_NAMES,
    cylinder_names: tuple[str, ...] = DYN_CYLINDER_NAMES,
    cuboid_half_extents: tuple[tuple[float, float, float], ...] = CUBOID_HALF_EXTENTS,
    cylinder_params: tuple[tuple[float, float], ...] = CYLINDER_PARAMS,
) -> dict[str, torch.Tensor | int]:
    """Shared step-level dynamic obstacle signed-distance cache."""
    original_obstacle_asset_names = tuple(obstacle_asset_names)
    active_limit = _get_dynamic_obstacle_active_limit(
        env,
        len(original_obstacle_asset_names),
    )
    cache_key = (
        id(env),
        int(active_limit),
        _freeze_cache_value(original_obstacle_asset_names),
        _freeze_cache_value(cuboid_names),
        _freeze_cache_value(cylinder_names),
        _freeze_cache_value(cuboid_half_extents),
        _freeze_cache_value(cylinder_params),
    )
    step = env.common_step_counter
    cached = _dyn_distance_step_cache.get(cache_key)
    if cached is not None and cached.get("_step") == step:
        return cached

    device = env.device
    num_envs = env.scene.num_envs

    if active_limit <= 0:
        result = {
            "centers_w": torch.zeros(num_envs, 0, 3, device=device),
            "radii": torch.zeros(0, device=device),
            "obs_pos": torch.zeros(num_envs, 0, 3, device=device),
            "obs_vel": torch.zeros(num_envs, 0, 3, device=device),
            "active_mask": torch.zeros(num_envs, 0, dtype=torch.bool, device=device),
            "is_cuboid": torch.zeros(0, dtype=torch.bool, device=device),
            "source_indices": torch.zeros(0, dtype=torch.long, device=device),
            "per_obstacle_min_dist": torch.full((num_envs, 0), 1e6, device=device),
            "closest_sphere": torch.zeros(num_envs, 0, dtype=torch.long, device=device),
            "per_sphere_min_dist": torch.full((num_envs, 0), 1e6, device=device),
            "n_total": 0,
            "_step": step,
        }
        _dyn_distance_step_cache[cache_key] = result
        return result

    source_indices = get_dyn_source_indices(env, active_limit, original_obstacle_asset_names)
    obs_pos, obs_vel, active_mask, source_idx_t = get_dyn_obs_pos_vel(
        env,
        source_indices=source_indices,
    )
    n_total = obs_pos.shape[1]

    if n_total == 0:
        result = {
            "centers_w": torch.zeros(num_envs, 0, 3, device=device),
            "radii": torch.zeros(0, device=device),
            "obs_pos": obs_pos,
            "obs_vel": obs_vel,
            "active_mask": active_mask,
            "is_cuboid": torch.zeros(0, dtype=torch.bool, device=device),
            "source_indices": source_idx_t,
            "per_obstacle_min_dist": torch.full((num_envs, 0), 1e6, device=device),
            "closest_sphere": torch.zeros(num_envs, 0, dtype=torch.long, device=device),
            "per_sphere_min_dist": torch.full((num_envs, 0), 1e6, device=device),
            "n_total": 0,
            "_step": step,
        }
        _dyn_distance_step_cache[cache_key] = result
        return result

    bubbles = _get_collision_bubbles(env)
    centers_w, radii = bubbles.get_world_spheres()
    n_spheres = centers_w.shape[1]

    is_cuboid = source_idx_t < len(DYN_CUBOID_NAMES)
    geom_idx = torch.where(
        is_cuboid,
        source_idx_t,
        source_idx_t - len(DYN_CUBOID_NAMES),
    )

    per_obstacle_min_dist = torch.full((num_envs, n_total), 1e6, device=device)
    closest_sphere = torch.zeros(num_envs, n_total, dtype=torch.long, device=device)
    per_sphere_min_dist = torch.full((num_envs, n_spheres), 1e6, device=device)

    cuboid_indices = torch.nonzero(is_cuboid, as_tuple=False).squeeze(-1)
    if cuboid_indices.numel() > 0:
        cub_pos = obs_pos.index_select(1, cuboid_indices)
        cub_he_table = _get_cached_static_table(
            "cuboid_half_extents", cuboid_half_extents, device, 3
        )
        cub_geom_idx = geom_idx.index_select(0, cuboid_indices)
        cub_he = cub_he_table.index_select(0, cub_geom_idx)

        delta = centers_w.unsqueeze(2) - cub_pos.unsqueeze(1)
        d_xyz = delta.abs() - cub_he[None, None, :, :]
        outside = d_xyz.clamp(min=0).norm(dim=-1)
        inside = d_xyz.max(dim=-1).values
        signed = torch.where(inside < 0, inside, outside) - radii[None, :, None]
        cub_active = active_mask.index_select(1, cuboid_indices)
        signed[~cub_active[:, None, :].expand_as(signed)] = 1e6

        cub_min_dist, cub_closest_sphere = signed.min(dim=1)
        cub_per_sphere_min = signed.min(dim=-1).values
        per_obstacle_min_dist[:, cuboid_indices] = cub_min_dist
        closest_sphere[:, cuboid_indices] = cub_closest_sphere
        per_sphere_min_dist = torch.minimum(per_sphere_min_dist, cub_per_sphere_min)

    cylinder_indices = torch.nonzero(~is_cuboid, as_tuple=False).squeeze(-1)
    if cylinder_indices.numel() > 0:
        cyl_pos = obs_pos.index_select(1, cylinder_indices)
        cyl_param_table = _get_cached_static_table(
            "cylinder_params", cylinder_params, device, 2
        )
        cyl_geom_idx = geom_idx.index_select(0, cylinder_indices)
        cyl_p = cyl_param_table.index_select(0, cyl_geom_idx)

        delta = centers_w.unsqueeze(2) - cyl_pos.unsqueeze(1)
        d_radial = delta[:, :, :, :2].norm(dim=-1) - cyl_p[None, None, :, 0]
        d_axial = delta[:, :, :, 2].abs() - cyl_p[None, None, :, 1]
        outside = (d_radial.clamp(min=0).square() + d_axial.clamp(min=0).square()).sqrt()
        inside = torch.stack([d_radial, d_axial], dim=-1).max(dim=-1).values
        signed = torch.where(inside < 0, inside, outside) - radii[None, :, None]
        cyl_active = active_mask.index_select(1, cylinder_indices)
        signed[~cyl_active[:, None, :].expand_as(signed)] = 1e6

        cyl_min_dist, cyl_closest_sphere = signed.min(dim=1)
        cyl_per_sphere_min = signed.min(dim=-1).values
        per_obstacle_min_dist[:, cylinder_indices] = cyl_min_dist
        closest_sphere[:, cylinder_indices] = cyl_closest_sphere
        per_sphere_min_dist = torch.minimum(per_sphere_min_dist, cyl_per_sphere_min)

    per_obstacle_min_dist = per_obstacle_min_dist.masked_fill(~active_mask, 1e6)

    result = {
        "centers_w": centers_w,
        "radii": radii,
        "obs_pos": obs_pos,
        "obs_vel": obs_vel,
        "active_mask": active_mask,
        "is_cuboid": is_cuboid,
        "source_indices": source_idx_t,
        "per_obstacle_min_dist": per_obstacle_min_dist,
        "closest_sphere": closest_sphere,
        "per_sphere_min_dist": per_sphere_min_dist,
        "n_total": n_total,
        "_step": step,
    }
    _dyn_distance_step_cache[cache_key] = result
    return result


def _get_dynamic_obstacle_sizes(
    source_indices: torch.Tensor,
    obstacle_sizes: tuple[tuple[float, float, float], ...],
    device: torch.device,
) -> torch.Tensor:
    """Map present obstacle source indices to their static size vectors."""
    if source_indices.numel() == 0:
        return torch.zeros(0, 3, device=device)

    cache_key = (_freeze_cache_value(obstacle_sizes), str(device))
    size_table = _dyn_obs_size_cache.get(cache_key)
    if size_table is None:
        size_table = torch.tensor(obstacle_sizes, device=device, dtype=torch.float32)
        _dyn_obs_size_cache[cache_key] = size_table

    return size_table.index_select(0, source_indices)


def _get_dynamic_obstacle_size_scale(
    obstacle_sizes: tuple[tuple[float, float, float], ...],
    device: torch.device,
) -> torch.Tensor:
    """Return per-axis size normalization scale.

    Scale is computed from constants.py obstacle_sizes:
        scale = max(abs(size), dim=0)

    For current dynamic obstacle constants, this should be roughly:
        [1.0, 1.0, 1.8]
    so size becomes normalized to [0, 1].
    """
    cache_key = ("size_scale", _freeze_cache_value(obstacle_sizes), str(device))
    scale = _dyn_obs_size_cache.get(cache_key)
    if scale is None:
        if len(obstacle_sizes) == 0:
            scale = torch.ones(3, device=device, dtype=torch.float32)
        else:
            size_table = torch.tensor(obstacle_sizes, device=device, dtype=torch.float32)
            scale = size_table.abs().max(dim=0).values.clamp_min(1e-6)
        _dyn_obs_size_cache[cache_key] = scale
    return scale


def _make_dynamic_obstacles_padding(
    num_envs: int,
    top_k: int,
    feat_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """All-padding output for dynamic_obstacles."""
    out = torch.zeros(num_envs, top_k * feat_dim, device=device)
    if top_k > 0:
        out[:, 3::feat_dim] = 1.0
    return out


def _dynamic_obstacles_impl(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    obstacle_asset_names: tuple[str, ...],
    obstacle_sizes: tuple[tuple[float, float, float], ...],
    cuboid_names: tuple[str, ...],
    cylinder_names: tuple[str, ...],
    cuboid_half_extents: tuple[tuple[float, float, float], ...],
    cylinder_params: tuple[tuple[float, float], ...],
    top_k: int,
    max_range: float,
    max_vel: float,
) -> torch.Tensor:
    feat_dim = 10
    active_limit = _get_dynamic_obstacle_active_limit(env, len(obstacle_asset_names))
    cache_key = (
        "dynamic_obstacles_10d",
        id(env),
        asset_cfg.name,
        int(active_limit),
        _freeze_cache_value(obstacle_asset_names),
        _freeze_cache_value(obstacle_sizes),
        _freeze_cache_value(cuboid_names),
        _freeze_cache_value(cylinder_names),
        _freeze_cache_value(cuboid_half_extents),
        _freeze_cache_value(cylinder_params),
        int(top_k),
        float(max_range),
        float(max_vel),
    )
    step = env.common_step_counter
    cached = _dyn_obs_step_cache.get(cache_key)
    if cached is not None and cached["_step"] == step:
        return cached["result"]

    device = env.device
    num_envs = env.scene.num_envs

    if top_k <= 0:
        result = torch.zeros(num_envs, 0, device=device)
        _dyn_obs_step_cache[cache_key] = {"_step": step, "result": result}
        return result

    if active_limit <= 0:
        result = _make_dynamic_obstacles_padding(num_envs, top_k, feat_dim, device)
        _dyn_obs_step_cache[cache_key] = {"_step": step, "result": result}
        return result

    robot: Articulation = env.scene[asset_cfg.name]
    robot_quat_w = robot.data.root_quat_w
    robot_quat_w = robot_quat_w / robot_quat_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    dyn_cache = _compute_dynamic_obstacle_distances(
        env,
        obstacle_asset_names,
        cuboid_names,
        cylinder_names,
        cuboid_half_extents,
        cylinder_params,
    )

    n_total = int(dyn_cache["n_total"])
    if n_total == 0:
        result = _make_dynamic_obstacles_padding(num_envs, top_k, feat_dim, device)
        _dyn_obs_step_cache[cache_key] = {"_step": step, "result": result}
        return result

    range_scale = max(float(max_range), 1e-6)
    vel_scale = max(float(max_vel), 1e-6)
    centers_w = dyn_cache["centers_w"]
    obs_pos = dyn_cache["obs_pos"]
    obs_vel = dyn_cache["obs_vel"]
    active_mask = dyn_cache["active_mask"]
    min_dist = dyn_cache["per_obstacle_min_dist"]
    closest_sphere = dyn_cache["closest_sphere"]
    obs_size = _get_dynamic_obstacle_sizes(dyn_cache["source_indices"], obstacle_sizes, device)
    size_scale = _get_dynamic_obstacle_size_scale(obstacle_sizes, device)

    actual_k = min(top_k, n_total)

    # Only expose dynamic obstacles within observation range.
    # min_dist is signed surface clearance.  Range-out obstacles become
    # standard padding after selection.
    valid_for_obs = active_mask & (min_dist < range_scale)
    rank_dist = min_dist.masked_fill(~valid_for_obs, 1e6)
    topk_dist, topk_idx = rank_dist.topk(actual_k, dim=1, largest=False)

    idx3 = topk_idx.unsqueeze(-1).expand(-1, -1, 3)
    sel_pos_w = obs_pos.gather(1, idx3)
    sel_vel_w = obs_vel.gather(1, idx3)
    sel_active = valid_for_obs.gather(1, topk_idx)
    sel_size = obs_size[topk_idx].clone()
    sel_size = (sel_size / size_scale).clamp(0.0, 1.0)

    sel_sphere_idx = closest_sphere.gather(1, topk_idx)
    sphere_idx3 = sel_sphere_idx.unsqueeze(-1).expand(-1, -1, 3)
    sel_sphere_pos = centers_w.gather(1, sphere_idx3)

    rel_pos_w = sel_pos_w - sel_sphere_pos
    dir_norm_val = rel_pos_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    dir_w = rel_pos_w / dir_norm_val

    quat_exp = robot_quat_w.unsqueeze(1).expand(-1, actual_k, -1).reshape(-1, 4)
    dir_b = quat_apply_inverse(quat_exp, dir_w.reshape(-1, 3)).reshape(num_envs, actual_k, 3)
    vel_b = quat_apply_inverse(quat_exp, sel_vel_w.reshape(-1, 3)).reshape(num_envs, actual_k, 3)

    dist_norm = (topk_dist / range_scale).clamp(-1.0, 1.0)
    vel_b = vel_b.clamp(-vel_scale, vel_scale) / vel_scale

    inactive = ~sel_active
    dir_b[inactive] = 0.0
    vel_b[inactive] = 0.0
    dist_norm[inactive] = 1.0
    sel_size[inactive] = 0.0

    obs_block = torch.cat(
        [
            dir_b,
            dist_norm.unsqueeze(-1),
            vel_b,
            sel_size,
        ],
        dim=-1,
    )

    out = _make_dynamic_obstacles_padding(num_envs, top_k, feat_dim, device)
    out[:, : actual_k * feat_dim] = obs_block.reshape(num_envs, actual_k * feat_dim)

    _dyn_obs_step_cache[cache_key] = {"_step": step, "result": out}
    return out


def _compute_dynamic_min_clearance_impl(
    env,
    obstacle_asset_names: tuple[str, ...],
    cuboid_names: tuple[str, ...],
    cylinder_names: tuple[str, ...],
    cuboid_half_extents: tuple[tuple[float, float, float], ...],
    cylinder_params: tuple[tuple[float, float], ...],
) -> torch.Tensor:
    """Compute minimum dynamic clearance from the shared step cache."""
    active_limit = _get_dynamic_obstacle_active_limit(env, len(obstacle_asset_names))
    if active_limit <= 0:
        return torch.full((env.scene.num_envs,), 1e6, device=env.device)

    dyn_cache = _compute_dynamic_obstacle_distances(
        env,
        obstacle_asset_names,
        cuboid_names,
        cylinder_names,
        cuboid_half_extents,
        cylinder_params,
    )
    if int(dyn_cache["n_total"]) == 0:
        return torch.full((env.scene.num_envs,), 1e6, device=env.device)
    return dyn_cache["per_sphere_min_dist"].min(dim=-1).values


def dynamic_obstacles(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    obstacle_asset_names: tuple[str, ...] = DYN_ALL_NAMES,
    obstacle_sizes: tuple[tuple[float, float, float], ...] = DYN_ALL_SIZES,
    cuboid_names: tuple[str, ...] = DYN_CUBOID_NAMES,
    cylinder_names: tuple[str, ...] = DYN_CYLINDER_NAMES,
    cuboid_half_extents: tuple[
        tuple[float, float, float], ...
    ] = CUBOID_HALF_EXTENTS,
    cylinder_params: tuple[
        tuple[float, float], ...
    ] = CYLINDER_PARAMS,
    top_k: int = 5,
    max_range: float = 5.0,
    max_vel: float = 5.0,
) -> torch.Tensor:
    """K nearest dynamic obstacles — 10D feature vector (base frame).

    Uses collision-sphere-based signed distance (same analytical geometry
    as reward functions) to rank obstacles by minimum signed surface
    clearance over all robot collision spheres.

    Per-obstacle feature vector (**10 dims**):

    ===  ============  =====================================================
    Dim  Name          Description
    ===  ============  =====================================================
     3   direction_b   Unit direction from closest collision sphere to
                       obstacle centre, in base frame.
     1   dist_norm     Signed surface distance normalised by ``max_range``,
                       clipped to [-1, 1].
     3   vel_b         Obstacle velocity in base frame, clipped to
                       ``[-max_vel, max_vel]`` and divided by ``max_vel``.
     3   size          Normalized continuous obstacle size [sx, sy, sz].
                       Cuboid → full extents; Cylinder → [diam, diam, h],
                       divided by the per-axis max dynamic obstacle size.
    ===  ============  =====================================================

    Obstacles are sorted by signed distance (nearest first).

    Active obstacles with signed clearance >= ``max_range`` are treated
    as unobserved.  Padding slots (including range-out obstacles) use:
    direction=0, dist=1, vel=0, size=0, so padding is not read as a
    dangerous obstacle.

    Args:
        env: Environment instance.
        asset_cfg: Robot SceneEntityCfg.
        obstacle_asset_names: Scene names for all dynamic obstacles.
        obstacle_sizes: (sx, sy, sz) per obstacle type.
        cuboid_names: Names of cuboid obstacles.
        cylinder_names: Names of cylinder obstacles.
        cuboid_half_extents: (hx, hy, hz) per cuboid type.
        cylinder_params: (radius, half_height) per cylinder type.
        top_k: Number of nearest obstacles to report.
        max_range: Normalisation range (m) for distance.
        max_vel: Clip and normalisation bound for obstacle velocity (m/s).

    Returns:
        torch.Tensor: [num_envs, top_k * 10]
    """
    return _dynamic_obstacles_impl(
        env,
        asset_cfg,
        obstacle_asset_names,
        obstacle_sizes,
        cuboid_names,
        cylinder_names,
        cuboid_half_extents,
        cylinder_params,
        top_k,
        max_range,
        max_vel,
    )


# ===========================================================
# Clearance Helpers (shared by command / reward / termination)
# ===========================================================

# Step-based cache for min clearance computation
_clearance_cache: dict = {}


def compute_static_min_clearance(
    env,
    sensor_cfg_name: str = "lidar",
    asset_cfg_name: str = "robot",
    ground_clearance: float = 0.1,
) -> torch.Tensor:
    """Compute minimum surface distance from collision spheres to static obstacles.

    Returns:
        [num_envs] tensor of minimum static clearance.
    """
    cache = _compute_sphere_lidar(
        env, sensor_cfg_name, asset_cfg_name, ground_clearance,
    )
    d_surface = cache["d_surface"]  # [E, S]
    return d_surface.min(dim=-1).values  # [E]


def compute_dynamic_min_clearance(
    env,
    asset_cfg_name: str = "robot",
    obstacle_asset_names: tuple[str, ...] = DYN_ALL_NAMES,
    cuboid_names: tuple[str, ...] = DYN_CUBOID_NAMES,
    cylinder_names: tuple[str, ...] = DYN_CYLINDER_NAMES,
    cuboid_half_extents: tuple[tuple[float, float, float], ...] = CUBOID_HALF_EXTENTS,
    cylinder_params: tuple[tuple[float, float], ...] = CYLINDER_PARAMS,
) -> torch.Tensor:
    """Compute minimum surface distance from collision spheres to dynamic obstacles.

    Returns:
        [num_envs] tensor of minimum dynamic clearance.
    """
    return _compute_dynamic_min_clearance_impl(
        env,
        obstacle_asset_names,
        cuboid_names,
        cylinder_names,
        cuboid_half_extents,
        cylinder_params,
    )


def compute_full_min_clearance(
    env,
    sensor_cfg_name: str = "lidar",
    asset_cfg_name: str = "robot",
    ground_clearance: float = 0.1,
    cuboid_names: tuple[str, ...] = DYN_CUBOID_NAMES,
    cylinder_names: tuple[str, ...] = DYN_CYLINDER_NAMES,
    cuboid_half_extents: tuple[tuple[float, float, float], ...] = CUBOID_HALF_EXTENTS,
    cylinder_params: tuple[tuple[float, float], ...] = CYLINDER_PARAMS,
    max_clearance: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Compute full min clearance (static + dynamic), with step-based cache.

    All outputs are clamped to [0, max_clearance] so that values remain
    physically meaningful even when no obstacles are nearby.

    Returns:
        dict with keys: static_min_clearance, dynamic_min_clearance, min_clearance
    """
    cache_key = (
        id(env),
        sensor_cfg_name,
        asset_cfg_name,
        float(ground_clearance),
        float(max_clearance),
    )
    step = env.common_step_counter
    cached = _clearance_cache.get(cache_key)
    if cached is not None and cached.get("_step") == step:
        return cached

    static_mc = compute_static_min_clearance(
        env,
        sensor_cfg_name=sensor_cfg_name,
        asset_cfg_name=asset_cfg_name,
        ground_clearance=ground_clearance,
    )
    dynamic_mc = compute_dynamic_min_clearance(
        env,
        asset_cfg_name=asset_cfg_name,
        obstacle_asset_names=DYN_ALL_NAMES,
        cuboid_names=cuboid_names,
        cylinder_names=cylinder_names,
        cuboid_half_extents=cuboid_half_extents,
        cylinder_params=cylinder_params,
    )

    # Clamp all clearance values to [0, max_clearance]
    static_mc = static_mc.clamp(min=0.0, max=max_clearance)
    dynamic_mc = dynamic_mc.clamp(min=0.0, max=max_clearance)
    min_clearance = torch.min(static_mc, dynamic_mc)

    result = {
        "static_min_clearance": static_mc,
        "dynamic_min_clearance": dynamic_mc,
        "min_clearance": min_clearance,
        "_step": step,
    }
    _clearance_cache[cache_key] = result
    return result


def ee_traj_progress_features(
    env: ManagerBasedEnv,
    command_name: str = "ee_traj",
) -> torch.Tensor:
    """Path progress features observation.

    Output: [s_hat, progress_speed, 1 - s_hat]
    Shape: [num_envs, 3]

    Where:
      s_hat           = monotonic virtual progress (filtered, never retreats)
      progress_speed  = s_hat_dot (virtual progress rate in s-units/s)
      1 - s_hat       = remaining progress
    """
    command_term = env.command_manager.get_term(command_name)

    s_hat = command_term._s_hat
    progress_speed = command_term._s_hat_dot
    remaining = 1.0 - s_hat

    return torch.stack([s_hat, progress_speed, remaining], dim=-1)


def ee_traj_final_goal_pose_b(
    env: ManagerBasedEnv,
    command_name: str = "ee_traj",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Final goal pose in robot base frame [pos(3) + quat(4)] = 7 dim.

    Uses the world-fixed path endpoint (sampled at episode start).
    """
    command_term = env.command_manager.get_term(command_name)
    goal_w = command_term.get_final_goal_pose_w()  # [N, 7]

    asset = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    root_quat_w = root_quat_w / root_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    goal_pos_b, goal_quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, goal_w[:, :3], goal_w[:, 3:7]
    )
    return torch.cat([goal_pos_b, goal_quat_b], dim=-1)


def ee_traj_preview_points_b(
    env: ManagerBasedEnv,
    command_name: str = "ee_traj",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_points: int = 3,
) -> torch.Tensor:
    """Future preview relative pose chain (num_points * 6 dim).

    Returns [num_envs, num_points * 6] flattened.
    Each link i encodes the relative pose from preview[i] to
    preview[i+1]:
      - delta_pos (3): position displacement in prev frame
      - delta_axis_angle (3): orientation delta as axis-angle

    The chain is: rel(q2,q1), rel(q3,q2), rel(q4,q3), ...
    q1 is separately provided via ee_traj_command.
    ``num_points`` should be N-1 where N = total preview points.
    """
    command_term = env.command_manager.get_term(command_name)
    num_envs = env.scene.num_envs
    device = env.device

    if num_points <= 0:
        return torch.zeros(num_envs, 0, device=device)

    p = command_term._preview_pos_w.shape[1]
    actual_pts = min(num_points, max(p - 1, 0))

    if actual_pts <= 0:
        return torch.zeros(
            num_envs, num_points * 6, device=device
        )

    out = torch.zeros(
        num_envs, num_points, 6, device=device
    )
    prev_pos = command_term._preview_pos_w[:, :actual_pts].reshape(-1, 3)
    prev_quat = command_term._preview_quat_w[:, :actual_pts].reshape(-1, 4)
    next_pos = command_term._preview_pos_w[:, 1 : actual_pts + 1].reshape(-1, 3)
    next_quat = command_term._preview_quat_w[:, 1 : actual_pts + 1].reshape(-1, 4)

    rel_pos, rel_quat = subtract_frame_transforms(
        prev_pos, prev_quat, next_pos, next_quat
    )
    rel_aa = axis_angle_from_quat(rel_quat).reshape(num_envs, actual_pts, 3)
    out[:, :actual_pts, :3] = rel_pos.reshape(num_envs, actual_pts, 3)
    out[:, :actual_pts, 3:] = rel_aa

    return out.reshape(num_envs, num_points * 6)


def ee_traj_path_pose_at_s_proj_b(
    env: ManagerBasedEnv,
    command_name: str = "ee_traj",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Path pose at geometric projection s_proj in base frame.

    Returns [num_envs, 7]: pos_b(3) + quat_b(4).
    Privileged observation for critic — provides the ground-truth
    geometric projection point on the path.
    """
    command_term = env.command_manager.get_term(command_name)
    s_proj = command_term._s_proj  # [N]

    pos_w = command_term._interpolate_path_position(s_proj)
    quat_w = command_term._interpolate_path_orientation(s_proj)

    asset = env.scene[asset_cfg.name]
    root_pos_w = asset.data.root_pos_w
    root_quat_w = asset.data.root_quat_w
    root_quat_w = root_quat_w / (
        root_quat_w.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    )

    pos_b, quat_b = subtract_frame_transforms(
        root_pos_w, root_quat_w, pos_w, quat_w,
    )
    return torch.cat([pos_b, quat_b], dim=-1)
