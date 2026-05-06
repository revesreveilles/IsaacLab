"""Batched state buffers for kinematic dynamic obstacles."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.constants import (
    NUM_DYN_TOTAL,
    NUM_DYN_CUBOIDS,
    NUM_DYN_CYLINDERS,
    DYN_ALL_NAMES,
    DYN_CUBOID_NAMES,
    DYN_CYLINDER_NAMES,
)


__all__ = [
    "assert_dynamic_obstacle_collection_order",
    "compare_dyn_buffer_to_sim",
    "get_dyn_name_to_index",
    "get_dyn_obs_pos_vel",
    "get_dyn_source_index_tuple",
    "get_dyn_source_indices",
    "get_dyn_write_split_indices",
    "get_dyn_state",
    "sync_dyn_state_to_sim_for_visualization",
    "write_dyn_state_to_sim",
]


_DYN_STATE_CACHE: dict[int, dict[str, torch.Tensor | int]] = {}
_DYN_NAME_TO_INDEX: dict[str, int] | None = None
_DYN_INTERLEAVED_NAMES: tuple[str, ...] | None = None
_DYN_INTERLEAVED_INDEX_CACHE: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
_DYN_COLLECTION_INDEX_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
_DYN_COLLECTION_ORDER_CHECKED: set[int] = set()
_DYN_WRITE_SPLIT_CACHE: dict[
    tuple[str, tuple[int, ...]],
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
] = {}


def _get_interleaved_dynamic_obstacle_names() -> tuple[str, ...]:
    """Return the reset-time dynamic obstacle activation order."""
    global _DYN_INTERLEAVED_NAMES
    if _DYN_INTERLEAVED_NAMES is not None:
        return _DYN_INTERLEAVED_NAMES

    names: list[str] = []
    ic = iy = 0
    nc, ny = len(DYN_CUBOID_NAMES), len(DYN_CYLINDER_NAMES)
    while ic < nc or iy < ny:
        if ic < nc and (iy >= ny or ic * ny <= iy * nc):
            names.append(DYN_CUBOID_NAMES[ic])
            ic += 1
        else:
            names.append(DYN_CYLINDER_NAMES[iy])
            iy += 1
    _DYN_INTERLEAVED_NAMES = tuple(names)
    return _DYN_INTERLEAVED_NAMES


def get_dyn_state(env) -> dict[str, torch.Tensor | int]:
    """Return the batched dynamic obstacle state buffer for this env."""
    key = id(env)
    state = _DYN_STATE_CACHE.get(key)
    if state is not None:
        return state

    assert_dynamic_obstacle_collection_order(env)

    num_envs = env.scene.num_envs
    device = env.device

    state = {
        "pos_w": torch.zeros(num_envs, NUM_DYN_TOTAL, 3, device=device),
        "quat_w": torch.zeros(num_envs, NUM_DYN_TOTAL, 4, device=device),
        "vel_w": torch.zeros(num_envs, NUM_DYN_TOTAL, 6, device=device),
        "goal_w": torch.zeros(num_envs, NUM_DYN_TOTAL, 3, device=device),
        "goal_age": torch.zeros(num_envs, NUM_DYN_TOTAL, device=device),
        "origin_w": torch.zeros(num_envs, NUM_DYN_TOTAL, 3, device=device),
        "speed_mag": torch.zeros(num_envs, NUM_DYN_TOTAL, 1, device=device),
        "active_mask": torch.zeros(num_envs, NUM_DYN_TOTAL, dtype=torch.bool, device=device),
        "_last_step": -1,
    }
    state["pos_w"][..., 2] = -100.0
    state["quat_w"][..., 0] = 1.0
    state["goal_w"][..., 2] = -100.0
    state["origin_w"][..., 2] = -100.0
    _DYN_STATE_CACHE[key] = state
    return state


def get_dyn_name_to_index() -> dict[str, int]:
    """Return mapping from dynamic obstacle scene name to buffer index."""
    global _DYN_NAME_TO_INDEX
    if _DYN_NAME_TO_INDEX is None:
        _DYN_NAME_TO_INDEX = {name: i for i, name in enumerate(DYN_ALL_NAMES)}
    return _DYN_NAME_TO_INDEX


def assert_dynamic_obstacle_collection_order(env) -> None:
    """Check collection object names once against the dynamic obstacle constants."""
    cache_key = id(env)
    if cache_key in _DYN_COLLECTION_ORDER_CHECKED:
        return

    cuboids = env.scene["dynamic_cuboids"]
    cylinders = env.scene["dynamic_cylinders"]
    cub_names = tuple(getattr(cuboids, "object_names", ()))
    cyl_names = tuple(getattr(cylinders, "object_names", ()))

    if len(cub_names) > 0 and cub_names != tuple(DYN_CUBOID_NAMES):
        raise AssertionError(
            f"dynamic_cuboids object order mismatch: {cub_names} != {DYN_CUBOID_NAMES}"
        )
    if len(cyl_names) > 0 and cyl_names != tuple(DYN_CYLINDER_NAMES):
        raise AssertionError(
            f"dynamic_cylinders object order mismatch: {cyl_names} != {DYN_CYLINDER_NAMES}"
        )

    _DYN_COLLECTION_ORDER_CHECKED.add(cache_key)


def get_dyn_collection_object_ids(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Return collection object ids ordered exactly like the dynamic constants."""
    cache_key = str(env.device)
    cached = _DYN_COLLECTION_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    cached = (
        torch.arange(NUM_DYN_CUBOIDS, device=env.device, dtype=torch.long),
        torch.arange(NUM_DYN_CYLINDERS, device=env.device, dtype=torch.long),
    )
    _DYN_COLLECTION_INDEX_CACHE[cache_key] = cached
    return cached


def get_dyn_source_index_tuple(
    active_limit: int,
    obstacle_names: Sequence[str] | None = None,
) -> tuple[int, ...]:
    """Return global source indices as a Python tuple.

    This is based only on constants and active_limit, so it does not touch
    CUDA.  Activation order follows the interleaved curriculum order.
    """
    active_limit = max(0, min(int(active_limit), NUM_DYN_TOTAL))
    if active_limit <= 0:
        return ()

    names = _get_interleaved_dynamic_obstacle_names()[:active_limit]
    if obstacle_names is not None:
        allowed = set(obstacle_names)
        names = tuple(name for name in names if name in allowed)

    name_to_index = get_dyn_name_to_index()
    return tuple(name_to_index[name] for name in names if name in name_to_index)


def get_dyn_source_indices(
    env,
    active_limit: int,
    obstacle_names: Sequence[str] | None = None,
) -> torch.Tensor:
    """Return global source indices for currently possible active obstacles."""
    idx_tuple = get_dyn_source_index_tuple(active_limit, obstacle_names)
    if len(idx_tuple) == 0:
        return torch.zeros(0, dtype=torch.long, device=env.device)

    cache_key = (str(env.device), idx_tuple)
    cached = _DYN_INTERLEAVED_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tensor = torch.tensor(idx_tuple, device=env.device, dtype=torch.long)
    _DYN_INTERLEAVED_INDEX_CACHE[cache_key] = tensor
    return tensor


def get_dyn_write_split_indices(
    env,
    source_indices: torch.Tensor | None = None,
    active_limit: int | None = None,
    obstacle_names: Sequence[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return split tensors for collection subset write.

    Returns:
        cub_global_ids: global buffer indices for cuboids.
        cub_object_ids: local object ids in ``dynamic_cuboids``.
        cyl_global_ids: global buffer indices for cylinders.
        cyl_object_ids: local object ids in ``dynamic_cylinders``.
    """
    if source_indices is None:
        if active_limit is None:
            active_limit = NUM_DYN_TOTAL
        idx_tuple = get_dyn_source_index_tuple(active_limit, obstacle_names)

        if len(idx_tuple) == 0:
            empty = torch.zeros(0, dtype=torch.long, device=env.device)
            return empty, empty, empty, empty

        cache_key = (str(env.device), idx_tuple)
        cached = _DYN_WRITE_SPLIT_CACHE.get(cache_key)
        if cached is not None:
            return cached

        cub_global = [idx for idx in idx_tuple if idx < NUM_DYN_CUBOIDS]
        cyl_global = [idx for idx in idx_tuple if idx >= NUM_DYN_CUBOIDS]
        cyl_local = [idx - NUM_DYN_CUBOIDS for idx in cyl_global]

        cub_global_ids = torch.tensor(cub_global, dtype=torch.long, device=env.device)
        cyl_global_ids = torch.tensor(cyl_global, dtype=torch.long, device=env.device)
        cyl_local_ids = torch.tensor(cyl_local, dtype=torch.long, device=env.device)

        cuboid_ids, cylinder_ids = get_dyn_collection_object_ids(env)

        if cub_global_ids.numel() > 0:
            cub_object_ids = cuboid_ids.index_select(0, cub_global_ids)
        else:
            cub_object_ids = torch.zeros(0, dtype=torch.long, device=env.device)

        if cyl_local_ids.numel() > 0:
            cyl_object_ids = cylinder_ids.index_select(0, cyl_local_ids)
        else:
            cyl_object_ids = torch.zeros(0, dtype=torch.long, device=env.device)

        cached = (cub_global_ids, cub_object_ids, cyl_global_ids, cyl_object_ids)
        _DYN_WRITE_SPLIT_CACHE[cache_key] = cached
        return cached

    source_indices = source_indices.to(device=env.device, dtype=torch.long)
    if source_indices.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=env.device)
        return empty, empty, empty, empty

    cuboid_ids, cylinder_ids = get_dyn_collection_object_ids(env)
    cub_mask = source_indices < NUM_DYN_CUBOIDS
    cub_global_ids = source_indices[cub_mask]
    cyl_global_ids = source_indices[~cub_mask]

    if cub_global_ids.numel() > 0:
        cub_object_ids = cuboid_ids.index_select(0, cub_global_ids)
    else:
        cub_object_ids = torch.zeros(0, dtype=torch.long, device=env.device)

    if cyl_global_ids.numel() > 0:
        cyl_local_ids = cyl_global_ids - NUM_DYN_CUBOIDS
        cyl_object_ids = cylinder_ids.index_select(0, cyl_local_ids)
    else:
        cyl_object_ids = torch.zeros(0, dtype=torch.long, device=env.device)

    return cub_global_ids, cub_object_ids, cyl_global_ids, cyl_object_ids


def get_dyn_obs_pos_vel(
    env,
    active_limit: int | None = None,
    obstacle_names: Sequence[str] | None = None,
    source_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return batched obstacle position, velocity, active mask and source indices."""
    state = get_dyn_state(env)
    if source_indices is None:
        if active_limit is None:
            active_limit = NUM_DYN_TOTAL
        source_indices = get_dyn_source_indices(env, active_limit, obstacle_names)
    else:
        source_indices = source_indices.to(device=env.device, dtype=torch.long)
    if source_indices.numel() == 0:
        return (
            state["pos_w"][:, :0],
            state["vel_w"][:, :0, :3],
            state["active_mask"][:, :0],
            source_indices,
        )
    return (
        state["pos_w"].index_select(1, source_indices),
        state["vel_w"].index_select(1, source_indices)[..., :3],
        state["active_mask"].index_select(1, source_indices),
        source_indices,
    )


def compare_dyn_buffer_to_sim(env, active_limit: int | None = None, max_objects: int | None = None) -> torch.Tensor:
    """Debug helper: max position error between buffer and collection state."""
    state = get_dyn_state(env)
    if active_limit is None:
        active_limit = NUM_DYN_TOTAL
    source_indices = get_dyn_source_indices(env, active_limit, None)
    if max_objects is not None:
        source_indices = source_indices[: max(0, int(max_objects))]
    if source_indices.numel() == 0:
        return torch.zeros((), device=env.device)

    cuboid_ids, cylinder_ids = get_dyn_collection_object_ids(env)
    sim_pos = torch.cat(
        [
            env.scene["dynamic_cuboids"].data.object_pos_w.index_select(1, cuboid_ids),
            env.scene["dynamic_cylinders"].data.object_pos_w.index_select(1, cylinder_ids),
        ],
        dim=1,
    )
    return (state["pos_w"].index_select(1, source_indices) - sim_pos.index_select(1, source_indices)).abs().max()


def write_dyn_state_to_sim(
    env,
    env_ids: torch.Tensor | None = None,
    active_limit: int | None = None,
    source_indices: torch.Tensor | None = None,
) -> bool:
    """Batched write selected dynamic obstacle state to RigidObjectCollections.

    Strict mode: ``object_ids`` subset write must work.
    """
    state = get_dyn_state(env)
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    if source_indices is None:
        if active_limit is None:
            active_limit = NUM_DYN_TOTAL
        source_indices = get_dyn_source_indices(env, active_limit)
    source_indices = source_indices.to(device=env.device, dtype=torch.long)
    if source_indices.numel() == 0:
        return True

    cuboids = env.scene["dynamic_cuboids"]
    cylinders = env.scene["dynamic_cylinders"]

    # Full-active fast path for the case where all objects are written.
    if source_indices.numel() == NUM_DYN_TOTAL:
        pose = torch.cat([state["pos_w"], state["quat_w"]], dim=-1)
        vel = state["vel_w"]

        pose_env = pose.index_select(0, env_ids)
        vel_env = vel.index_select(0, env_ids)
        cuboids.write_object_pose_to_sim(
            pose_env[:, :NUM_DYN_CUBOIDS],
            env_ids=env_ids,
        )
        cuboids.write_object_velocity_to_sim(
            vel_env[:, :NUM_DYN_CUBOIDS],
            env_ids=env_ids,
        )
        cylinders.write_object_pose_to_sim(
            pose_env[:, NUM_DYN_CUBOIDS:],
            env_ids=env_ids,
        )
        cylinders.write_object_velocity_to_sim(
            vel_env[:, NUM_DYN_CUBOIDS:],
            env_ids=env_ids,
        )
        return True

    if active_limit is not None:
        cub_global_ids, cub_object_ids, cyl_global_ids, cyl_object_ids = get_dyn_write_split_indices(
            env,
            active_limit=active_limit,
        )
    else:
        cub_global_ids, cub_object_ids, cyl_global_ids, cyl_object_ids = get_dyn_write_split_indices(
            env,
            source_indices=source_indices,
        )

    pos_env = state["pos_w"].index_select(0, env_ids)
    quat_env = state["quat_w"].index_select(0, env_ids)
    vel_env = state["vel_w"].index_select(0, env_ids)

    if cub_global_ids.numel() > 0:
        cub_pos = pos_env.index_select(1, cub_global_ids)
        cub_quat = quat_env.index_select(1, cub_global_ids)
        cub_pose = torch.cat([cub_pos, cub_quat], dim=-1)
        cub_vel = vel_env.index_select(1, cub_global_ids)
        cuboids.write_object_pose_to_sim(
            cub_pose,
            env_ids=env_ids,
            object_ids=cub_object_ids,
        )
        cuboids.write_object_velocity_to_sim(
            cub_vel,
            env_ids=env_ids,
            object_ids=cub_object_ids,
        )

    if cyl_global_ids.numel() > 0:
        cyl_pos = pos_env.index_select(1, cyl_global_ids)
        cyl_quat = quat_env.index_select(1, cyl_global_ids)
        cyl_pose = torch.cat([cyl_pos, cyl_quat], dim=-1)
        cyl_vel = vel_env.index_select(1, cyl_global_ids)
        cylinders.write_object_pose_to_sim(
            cyl_pose,
            env_ids=env_ids,
            object_ids=cyl_object_ids,
        )
        cylinders.write_object_velocity_to_sim(
            cyl_vel,
            env_ids=env_ids,
            object_ids=cyl_object_ids,
        )

    return True


def sync_dyn_state_to_sim_for_visualization(
    env,
    env_ids: torch.Tensor | None = None,
    active_limit: int | None = None,
) -> bool:
    """Manually sync dynamic obstacle buffer to sim for visualization/debug."""
    return write_dyn_state_to_sim(
        env,
        env_ids=env_ids,
        active_limit=active_limit,
    )
