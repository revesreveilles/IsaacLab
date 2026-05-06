"""Curriculum and event functions for mobile manipulator training.

All curricula use unified goal_reach/quality semantics:

- **goal_reach**: ``goal_reached`` — the end-effector position (and optionally
  orientation) is within tolerance of the **global sampled goal / path
  endpoint** for the current episode.  This is a world-fixed target sampled
  at episode start; it is **not** a local body-frame reference, **not** a
  moving path waypoint, and **not** survival-until-timeout.  The same check
  is used by ``terminations.ee_pose_goal_reached``, the
  ``rewards.goal_reached_bonus``, and every curriculum term.

- **quality**: contouring + orientation threshold check, evaluated only
  for successful episodes (quality = 0 when goal_reach = 0).

- **dynamic_obstacle_curriculum**: driven by success/collision/tip rates.
  Quality-rate is NOT used for obstacle difficulty progression.
  The task objective is "reach goal + avoid collision + avoid tipping".

Progression for command-range and reward-std curricula is driven by
goal_reach_rate + quality_rate via ring buffers.  Dynamic obstacle
curriculum is driven by success_rate / collision_rate / tip_rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_tasks.manager_based.mobile_manipulator.ee_tracking.constants import (
    NUM_DYN_TOTAL,
    NUM_DYN_CUBOIDS,
    DYN_CUBOID_NAMES,
    DYN_CYLINDER_NAMES,
    DYN_ALL_NAMES,
    DYN_ALL_SIZES,
    CUBOID_SIZES,
    CYLINDER_SIZES,
)

from .dynamic_obstacle_buffers import (
    get_dyn_source_index_tuple,
    get_dyn_source_indices,
    get_dyn_state,
    write_dyn_state_to_sim,
)


__all__ = [
    "ee_traj_command_ranges_curriculum",
    "ee_tracking_reward_std_curriculum",
    "dynamic_obstacle_curriculum",
    "get_active_dynamic_obstacle_names",
    "reset_dynamic_obstacles_navrl_style",
    "step_kinematic_obstacles",
]


# per-env step counter for velocity magnitude resampling
_kinematic_step_counter: dict[int, int] = {}
_dyn_all_size_table_cache: dict[str, torch.Tensor] = {}

# ==============================================================================
# Per-env obstacle difficulty buffer (module-level, shared)
# ==============================================================================
# Maps env_id → [num_envs] integer difficulty (0..max_active).
# Written by dynamic_obstacle_curriculum,
# read by reset_dynamic_obstacles_navrl_style.
_obstacle_difficulty: dict[int, torch.Tensor] = {}
# Python-side max active count.  Keeps the per-step obstacle motion and
# distance code from synchronizing on ``diff_buf.max().item()``.
_obstacle_active_count: dict[int, int] = {}
# Previous max active count seen by reset.  Used to hide obstacles that were
# active before a curriculum downgrade without looping over all obstacles.
_obstacle_prev_active_count: dict[int, int] = {}


def _interleaved_dynamic_obstacle_names() -> tuple[str, ...]:
    """Return the reset-time dynamic obstacle activation order."""
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
    return tuple(names)


_DYN_INTERLEAVED_NAMES = _interleaved_dynamic_obstacle_names()


def _get_dynamic_obstacle_active_limit(env, n_total: int) -> int:
    """Return the current max active dynamic obstacle count across envs."""
    cached = _obstacle_active_count.get(id(env))
    if cached is not None:
        return max(0, min(int(cached), n_total))

    return _refresh_dynamic_obstacle_active_limit(env, n_total)


def _refresh_dynamic_obstacle_active_limit(env, n_total: int) -> int:
    """Refresh the Python-side max active obstacle count from the tensor buffer."""
    diff_buf = _obstacle_difficulty.get(id(env))
    if diff_buf is None:
        active_limit = n_total
    elif diff_buf.numel() == 0 or n_total <= 0:
        active_limit = 0
    else:
        active_limit = int(diff_buf.max().clamp(0, n_total).item())
    _obstacle_active_count[id(env)] = active_limit
    return active_limit


def get_active_dynamic_obstacle_names(
    active_limit: int,
    obstacle_names: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Return active obstacle names according to interleaved curriculum order.

    If ``obstacle_names`` is provided, keep only names that are both in
    ``obstacle_names`` and in the first ``active_limit`` entries of the
    interleaved curriculum order.
    """
    if active_limit <= 0:
        return ()
    if obstacle_names is None:
        return _DYN_INTERLEAVED_NAMES[:active_limit]
    names = tuple(obstacle_names)
    if active_limit >= len(DYN_ALL_NAMES):
        return names
    active_set = set(_DYN_INTERLEAVED_NAMES[:active_limit])
    return tuple(name for name in names if name in active_set)


def _dynamic_obstacle_size_from_source_idx(source_idx: int) -> tuple[bool, tuple[float, float, float]]:
    """Return (is_cuboid, size) for a global dynamic obstacle index."""
    if source_idx < NUM_DYN_CUBOIDS:
        return True, CUBOID_SIZES[source_idx]
    return False, CYLINDER_SIZES[source_idx - NUM_DYN_CUBOIDS]


def _get_dyn_all_size_table(device: torch.device) -> torch.Tensor:
    """Return cached dynamic obstacle sizes on the requested device."""
    cache_key = str(device)
    size_table = _dyn_all_size_table_cache.get(cache_key)
    if size_table is None:
        size_table = torch.tensor(DYN_ALL_SIZES, device=device, dtype=torch.float32)
        _dyn_all_size_table_cache[cache_key] = size_table
    return size_table


def _hide_dynamic_obstacle_state(env, env_ids, source_idx: int) -> None:
    """Hide one dynamic obstacle in the batched state buffer."""
    state = get_dyn_state(env)
    state["pos_w"][env_ids, source_idx] = 0.0
    state["pos_w"][env_ids, source_idx, 2] = -100.0
    state["quat_w"][env_ids, source_idx] = 0.0
    state["quat_w"][env_ids, source_idx, 0] = 1.0
    state["vel_w"][env_ids, source_idx] = 0.0
    state["goal_w"][env_ids, source_idx] = 0.0
    state["goal_w"][env_ids, source_idx, 2] = -100.0
    state["goal_age"][env_ids, source_idx] = 0.0
    state["origin_w"][env_ids, source_idx] = 0.0
    state["origin_w"][env_ids, source_idx, 2] = -100.0
    state["speed_mag"][env_ids, source_idx] = 0.0
    state["active_mask"][env_ids, source_idx] = False


def step_kinematic_obstacles(
    env,
    env_ids,
    local_range: tuple[float, float, float] = (3.0, 3.0, 1.0),
    goal_reach_threshold: float = 0.5,
    vel_resample_interval: float = 2.0,
    vel_range: tuple[float, float] = (0.5, 2.0),
    cuboid_hover_range: tuple[float, float] = (0.5, 1.0),
    goal_check_interval_s: float = 0.10,
    goal_timeout_s: float = 4.0,
    write_to_sim: bool = False,
):
    """Interval event: vectorized buffer-based random walk for active dynamic obstacles."""

    active_limit = _get_dynamic_obstacle_active_limit(env, NUM_DYN_TOTAL)
    if active_limit <= 0:
        return

    source_indices = get_dyn_source_indices(env, active_limit)
    num_active_sources = source_indices.numel()
    if num_active_sources == 0:
        return

    dt = env.step_dt
    env_key = id(env)

    # ── 全局步数计数 ──
    if env_key not in _kinematic_step_counter:
        _kinematic_step_counter[env_key] = 0
    _kinematic_step_counter[env_key] += 1
    step_count = _kinematic_step_counter[env_key]

    resample_steps = max(1, int(vel_resample_interval / dt))
    do_vel_resample = (step_count % resample_steps == 0)
    goal_check_steps = max(1, int(goal_check_interval_s / dt))
    do_goal_check = (step_count % goal_check_steps) == 0

    # ── Terrain info ──
    terrain = getattr(env.scene, 'terrain', None)
    if terrain is None:
        return
    terrain_origins = terrain.env_origins
    t_cfg = terrain.cfg.terrain_generator
    mr_x = t_cfg.size[0] / 2.0
    mr_y = t_cfg.size[1] / 2.0

    num_envs = env.scene.num_envs
    lr_x, lr_y, lr_z = local_range
    state = get_dyn_state(env)

    pos = state["pos_w"].index_select(1, source_indices).clone()
    goal = state["goal_w"].index_select(1, source_indices).clone()
    origin = state["origin_w"].index_select(1, source_indices).clone()
    vel = state["vel_w"].index_select(1, source_indices).clone()
    mag = state["speed_mag"].index_select(1, source_indices).clone()
    goal_age = state["goal_age"].index_select(1, source_indices).clone()
    active = state["active_mask"].index_select(1, source_indices)
    goal_age = torch.where(active, goal_age + dt, torch.zeros_like(goal_age))

    sizes = _get_dyn_all_size_table(env.device).index_select(0, source_indices)
    is_cuboid = source_indices < NUM_DYN_CUBOIDS
    half_h = sizes[:, 2] * 0.5
    hover_lo, hover_hi = cuboid_hover_range
    z_min = torch.where(
        is_cuboid,
        torch.full_like(half_h, hover_lo) + half_h,
        half_h,
    )
    z_max = torch.where(
        is_cuboid,
        torch.full_like(half_h, hover_hi) + half_h,
        half_h,
    )

    if do_goal_check:
        goal_dist = (pos - goal).norm(dim=-1)
        need_new_goal = active & (
            (goal_dist < goal_reach_threshold)
            | (goal_age > goal_timeout_s)
        )

        rand = torch.rand(num_envs, num_active_sources, 3, device=env.device)
        offset = torch.empty(num_envs, num_active_sources, 3, device=env.device)
        offset[..., 0] = (rand[..., 0] * 2.0 - 1.0) * lr_x
        offset[..., 1] = (rand[..., 1] * 2.0 - 1.0) * lr_y
        offset[..., 2] = (rand[..., 2] * 2.0 - 1.0) * lr_z

        new_goal = origin + offset
        new_goal[..., 0] = new_goal[..., 0].clamp(
            min=terrain_origins[:, None, 0] - mr_x,
            max=terrain_origins[:, None, 0] + mr_x,
        )
        new_goal[..., 1] = new_goal[..., 1].clamp(
            min=terrain_origins[:, None, 1] - mr_y,
            max=terrain_origins[:, None, 1] + mr_y,
        )
        new_goal_z_min = terrain_origins[:, None, 2] + z_min.view(1, num_active_sources)
        new_goal_z_max = terrain_origins[:, None, 2] + z_max.view(1, num_active_sources)
        new_goal[..., 2] = new_goal[..., 2].clamp(
            min=new_goal_z_min,
            max=new_goal_z_max,
        )
        goal = torch.where(need_new_goal.unsqueeze(-1), new_goal, goal)
        goal_age = torch.where(need_new_goal, torch.zeros_like(goal_age), goal_age)

    if do_vel_resample:
        new_mag = vel_range[0] + (vel_range[1] - vel_range[0]) * torch.rand(
            num_envs,
            num_active_sources,
            1,
            device=env.device,
        )
        mag = torch.where(active.unsqueeze(-1), new_mag, mag)

    direction = goal - pos
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp(min=1e-4)

    vel3 = direction * mag
    vel3[..., 2] = torch.where(
        is_cuboid.view(1, num_active_sources),
        vel3[..., 2],
        torch.zeros_like(vel3[..., 2]),
    )
    vel[..., :3] = torch.where(active.unsqueeze(-1), vel3, torch.zeros_like(vel[..., :3]))

    new_pos = pos + vel[..., :3] * dt
    new_pos[..., 0] = new_pos[..., 0].clamp(
        min=terrain_origins[:, None, 0] - mr_x,
        max=terrain_origins[:, None, 0] + mr_x,
    )
    new_pos[..., 1] = new_pos[..., 1].clamp(
        min=terrain_origins[:, None, 1] - mr_y,
        max=terrain_origins[:, None, 1] + mr_y,
    )
    pos_z_min = terrain_origins[:, None, 2] + z_min.view(1, num_active_sources)
    pos_z_max = terrain_origins[:, None, 2] + z_max.view(1, num_active_sources)
    new_pos[..., 2] = new_pos[..., 2].clamp(min=pos_z_min, max=pos_z_max)
    pos = torch.where(active.unsqueeze(-1), new_pos, pos)

    state["pos_w"].index_copy_(1, source_indices, pos)
    state["goal_w"].index_copy_(1, source_indices, goal)
    state["vel_w"].index_copy_(1, source_indices, vel)
    state["speed_mag"].index_copy_(1, source_indices, mag)
    state["goal_age"].index_copy_(1, source_indices, goal_age)

    state["_last_step"] = env.common_step_counter
    if write_to_sim:
        write_dyn_state_to_sim(env, source_indices=source_indices)


# ==============================================================================
# Curriculum Helper Functions
# ==============================================================================

# Default warmup thresholds (can be overridden per-curriculum via params):
# - warmup_min_samples: minimum samples before computing meaningful rates
# - warmup_min_adjust_samples: minimum samples before allowing level/difficulty adjustment
_DEFAULT_WARMUP_MIN_SAMPLES = 100
_DEFAULT_WARMUP_MIN_ADJUST_SAMPLES = 200


def _linear_interpolation(
    initial_value: float,
    final_value: float,
    progress: float,
) -> float:
    """Linear interpolation between initial and final values."""
    return initial_value + (final_value - initial_value) * progress


def _interpolate_range(
    initial_range: tuple[float, float],
    final_range: tuple[float, float],
    progress: float,
) -> tuple[float, float]:
    """Interpolate a range (min, max) tuple."""
    return (
        _linear_interpolation(initial_range[0], final_range[0], progress),
        _linear_interpolation(initial_range[1], final_range[1], progress),
    )


def _interpolate_dict_ranges(
    initial_dict: dict,
    final_dict: dict,
    progress: float,
) -> dict:
    """Interpolate all ranges in a dictionary."""
    result = {}
    for key in initial_dict:
        result[key] = _interpolate_range(initial_dict[key], final_dict[key], progress)
    return result


def _goal_reached_check(
    command_term,
    env_ids: torch.Tensor,
    goal_pos_tolerance: float = 0.1,
    goal_ori_tolerance: float | None = None,
) -> torch.Tensor:
    """Check whether the final goal pose is reached within tolerances.

    Uses only ``metrics[...]`` so this works with any command type that
    exposes ``final_position_error`` and ``final_orientation_error``.
    """
    metrics = command_term.metrics
    pos_ok = metrics["final_position_error"][env_ids] <= goal_pos_tolerance
    if goal_ori_tolerance is None:
        return pos_ok
    ori_ok = (
        metrics["final_orientation_error"][env_ids] <= goal_ori_tolerance
    )
    return pos_ok & ori_ok


def _compute_quality(
    command_term,
    env_ids: torch.Tensor,
    contouring_threshold: float = 0.08,
    orientation_threshold: float = 0.5,
) -> torch.Tensor:
    """Compute path tracking quality from episode-mean errors.

    Uses the *average* contouring and orientation errors accumulated
    over the entire episode, reflecting overall tracking fidelity
    rather than a single worst-case or snapshot.

    Binary threshold: quality = 1 if ALL episode-mean errors are within
    their respective thresholds, 0 otherwise.

    All curricula use this function to ensure consistent quality semantics.
    The returned quality should be multiplied by goal_reach to get final quality.

    Args:
        command_term: The command term with metrics.
        env_ids: Environment indices to check.
        contouring_threshold: Max mean contouring error (m).
        orientation_threshold: Max mean orientation error (rad).

    Returns:
        Float tensor [len(env_ids)] in {0, 1}.
    """
    contouring = command_term.metrics["ep_mean_contouring"][env_ids]
    ori_err = command_term.metrics["ep_mean_ori_error"][env_ids]
    return (
        (contouring < contouring_threshold)
        & (ori_err < orientation_threshold)
    ).float()


def _write_ring_buffer(
    goal_reach_buf: torch.Tensor,
    quality_buf: torch.Tensor,
    goal_reach: torch.Tensor,
    quality: torch.Tensor,
    ptr: int,
    filled: bool,
    window_size: int,
) -> tuple[int, bool]:
    """Write goal_reach/quality samples to ring buffer.

    Unified helper used by all three curricula.

    Args:
        goal_reach_buf: Ring buffer for goal_reach values [window_size].
        quality_buf: Ring buffer for quality values [window_size].
        goal_reach: New goal_reach values to write [n].
        quality: New quality values to write [n].
        ptr: Current write pointer.
        filled: Whether buffer has been filled at least once.
        window_size: Size of the ring buffer.

    Returns:
        (new_ptr, new_filled) tuple.
    """
    n = goal_reach.shape[0]
    ws = window_size

    if n >= ws:
        goal_reach_buf[:] = goal_reach[-ws:]
        quality_buf[:] = quality[-ws:]
        return 0, True

    end = ptr + n
    if end <= ws:
        goal_reach_buf[ptr:end] = goal_reach
        quality_buf[ptr:end] = quality
    else:
        overflow = end - ws
        remain = n - overflow
        goal_reach_buf[ptr:] = goal_reach[:remain]
        goal_reach_buf[:overflow] = goal_reach[remain:]
        quality_buf[ptr:] = quality[:remain]
        quality_buf[:overflow] = quality[remain:]

    new_ptr = (ptr + n) % ws
    new_filled = filled or (end >= ws)
    return new_ptr, new_filled


def _get_buffer_rates(
    goal_reach_buf: torch.Tensor,
    quality_buf: torch.Tensor,
    ptr: int,
    filled: bool,
    warmup_min_samples: int = _DEFAULT_WARMUP_MIN_SAMPLES,
) -> tuple[float, float]:
    """Get goal_reach_rate and quality_rate from ring buffer.

    Unified helper used by all three curricula.
    Returns (0.0, 0.0) during warmup (ptr <= warmup_min_samples).

    Args:
        goal_reach_buf: Ring buffer for goal_reach values.
        quality_buf: Ring buffer for quality values.
        ptr: Current write pointer.
        filled: Whether buffer has been filled at least once.
        warmup_min_samples: Minimum samples before computing meaningful rates.

    Returns:
        (goal_reach_rate, quality_rate) tuple.
    """
    if filled:
        return float(goal_reach_buf.mean()), float(quality_buf.mean())
    elif ptr > warmup_min_samples:
        return (
            float(goal_reach_buf[:ptr].mean()),
            float(quality_buf[:ptr].mean()),
        )
    return 0.0, 0.0


# ==============================================================================
# Curriculum Terms
# ==============================================================================


class ee_traj_command_ranges_curriculum(ManagerTermBase):
    """Curriculum for gradually expanding EE path command sampling ranges.

    Uses unified goal_reach + quality semantics:
    - **goal_reach**: ``goal_reached`` — EE reached the **global sampled goal
      / path endpoint** (world-fixed target sampled at episode start).
      Not a local target, not timeout, not survival.
    - **quality**: contouring + orientation via ``_compute_quality`` (same as all curricula).

    Execution order per call:
    1. Compute goal_reach and quality for current batch
    2. Write batch to ring buffer
    3. Read rates from ring buffer (post-write)
    4. Adjust discrete level based on rates
    5. Interpolate sampling ranges from level → progress

    Warmup: rates return 0.0 until ``warmup_min_samples``; level adjustment
    deferred until ``warmup_min_adjust_samples`` accumulated.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._command_term = env.command_manager.get_term(cfg.params["command_name"])

        self._initial_range_pos = cfg.params["initial_range_pos"]
        self._final_range_pos = cfg.params["final_range_pos"]
        self._initial_delta_euler = cfg.params.get("initial_delta_euler_ranges", None)
        self._final_delta_euler = cfg.params.get("final_delta_euler_ranges", None)
        self._initial_z_range = cfg.params.get("initial_z_range", None)
        self._final_z_range = cfg.params.get("final_z_range", None)

        # Ring buffer
        ws = cfg.params.get("window_size", 2000)
        self._window_size = ws
        self._goal_reach_buf = torch.zeros(ws, device=env.device)
        self._quality_buf = torch.zeros(ws, device=env.device)
        self._ptr = 0
        self._filled = False

        # Discrete difficulty level (0..max_level)
        self._max_level = cfg.params.get("max_level", 10)
        self._level = 0

        # Warmup thresholds (configurable via params)
        self._warmup_min_samples = cfg.params.get(
            "warmup_min_samples", _DEFAULT_WARMUP_MIN_SAMPLES
        )
        self._warmup_min_adjust_samples = cfg.params.get(
            "warmup_min_adjust_samples", _DEFAULT_WARMUP_MIN_ADJUST_SAMPLES
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str,
        initial_range_pos: dict,
        final_range_pos: dict,
        goal_pos_tolerance: float = 0.1,
        goal_ori_tolerance: float | None = None,
        num_steps: int | None = None,
        move_up_rate: float = 0.7,
        move_down_rate: float = 0.3,
        quality_up_rate: float = 0.6,
        contouring_threshold: float = 0.08,
        orientation_threshold: float = 0.5,
        window_size: int = 2000,
        max_level: int = 10,
        initial_delta_euler_ranges: dict | None = None,
        final_delta_euler_ranges: dict | None = None,
        initial_z_range: tuple[float, float] | None = None,
        final_z_range: tuple[float, float] | None = None,
        warmup_min_samples: int = _DEFAULT_WARMUP_MIN_SAMPLES,
        warmup_min_adjust_samples: int = _DEFAULT_WARMUP_MIN_ADJUST_SAMPLES,
    ) -> dict[str, float]:
        """Update command sampling ranges based on goal_reach + quality."""
        n_envs = len(env_ids) if not isinstance(env_ids, torch.Tensor) else env_ids.shape[0]
        sr, qr = 0.0, 0.0

        if num_steps is not None:
            # Simple linear schedule.
            progress = min(1.0, env.common_step_counter / num_steps)
        else:
            # Step 1-2: compute goal_reach/quality and write to ring buffer
            if n_envs > 0:
                env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
                goal_reach = _goal_reached_check(
                    self._command_term, env_ids_t,
                    goal_pos_tolerance, goal_ori_tolerance,
                ).float()
                quality = goal_reach * _compute_quality(
                    self._command_term, env_ids_t,
                    contouring_threshold,
                    orientation_threshold,
                )
                self._ptr, self._filled = _write_ring_buffer(
                    self._goal_reach_buf, self._quality_buf,
                    goal_reach, quality,
                    self._ptr, self._filled, self._window_size,
                )

            # Step 3: read rates from buffer (post-write)
            sr, qr = _get_buffer_rates(
                self._goal_reach_buf, self._quality_buf,
                self._ptr, self._filled,
                self._warmup_min_samples,
            )

            # Step 4: adjust level (deferred until warmup_min_adjust_samples)
            if self._filled or self._ptr > self._warmup_min_adjust_samples:
                if sr > move_up_rate and qr > quality_up_rate and self._level < self._max_level:
                    self._level += 1
                elif sr < move_down_rate and self._level > 0:
                    self._level -= 1

            progress = self._level / max(self._max_level, 1)

        # Step 5: interpolate sampling ranges
        new_range_pos = _interpolate_dict_ranges(
            initial_range_pos, final_range_pos, progress
        )
        self._command_term.r_range = new_range_pos["r"]
        self._command_term.p_range = new_range_pos["p"]
        self._command_term.y_range = new_range_pos["y"]
        self._command_term.cfg.ranges_pos = new_range_pos

        new_z_range = None
        if initial_z_range is not None and final_z_range is not None:
            new_z_range = _interpolate_range(
                initial_z_range, final_z_range, progress
            )
            self._command_term.z_range = new_z_range
            self._command_term.cfg.z_range = new_z_range

        if initial_delta_euler_ranges is not None and final_delta_euler_ranges is not None:
            new_delta_euler = _interpolate_dict_ranges(
                initial_delta_euler_ranges, final_delta_euler_ranges, progress
            )
            self._command_term.cfg.delta_euler_ranges = new_delta_euler

        result = {
            "progress": progress,
            "level": float(self._level),
            "goal_reach_rate": sr,
            "quality_rate": qr,
            "r_min": new_range_pos["r"][0],
            "r_max": new_range_pos["r"][1],
            "p_min": new_range_pos["p"][0],
            "p_max": new_range_pos["p"][1],
            "y_min": new_range_pos["y"][0],
            "y_max": new_range_pos["y"][1],
        }

        if new_z_range is not None:
            result["z_min"] = new_z_range[0]
            result["z_max"] = new_z_range[1]

        return result


class ee_tracking_reward_std_curriculum(ManagerTermBase):
    """Curriculum for gradually tightening reward std (tolerance).

    Uses unified goal_reach + quality semantics:
    - **goal_reach**: ``goal_reached`` — EE reached the **global sampled goal
      / path endpoint** (world-fixed target sampled at episode start).
      Not a local target, not timeout, not survival.
    - **quality**: contouring/lag via ``_compute_quality`` (same as all curricula).

    Execution order per call (same as all curricula):
    1. Compute goal_reach and quality for current batch
    2. Write batch to ring buffer
    3. Read rates from ring buffer (post-write)
    4. Adjust discrete level based on rates
    5. Interpolate std values from level -> progress

    Warmup: rates return 0.0 until ``warmup_min_samples``; level adjustment
    deferred until ``warmup_min_adjust_samples`` accumulated.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(self._term_name)

        self._std_configs = cfg.params.get("std_configs", None)
        if self._std_configs is None:
            self._std_configs = [{
                "param_name": cfg.params["param_name"],
                "initial_std": cfg.params["initial_std"],
                "final_std": cfg.params["final_std"],
            }]

        self._command_term = env.command_manager.get_term(cfg.params["command_name"])

        # Ring buffer
        ws = cfg.params.get("window_size", 2000)
        self._window_size = ws
        self._goal_reach_buf = torch.zeros(ws, device=env.device)
        self._quality_buf = torch.zeros(ws, device=env.device)
        self._ptr = 0
        self._filled = False

        self._max_level = cfg.params.get("max_level", 10)
        self._level = 0

        # Warmup thresholds (configurable via params)
        self._warmup_min_samples = cfg.params.get(
            "warmup_min_samples", _DEFAULT_WARMUP_MIN_SAMPLES
        )
        self._warmup_min_adjust_samples = cfg.params.get(
            "warmup_min_adjust_samples", _DEFAULT_WARMUP_MIN_ADJUST_SAMPLES
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        command_name: str,
        goal_pos_tolerance: float = 0.1,
        goal_ori_tolerance: float | None = None,
        num_steps: int | None = None,
        move_up_rate: float = 0.7,
        move_down_rate: float = 0.3,
        quality_up_rate: float = 0.6,
        contouring_threshold: float = 0.08,
        orientation_threshold: float = 0.5,
        window_size: int = 2000,
        max_level: int = 10,
        std_configs: list[dict] | None = None,
        warmup_min_samples: int = _DEFAULT_WARMUP_MIN_SAMPLES,
        warmup_min_adjust_samples: int = _DEFAULT_WARMUP_MIN_ADJUST_SAMPLES,
        # Legacy params (ignored, kept for backwards compat)
        param_name: str | None = None,
        initial_std: float | None = None,
        final_std: float | None = None,
    ) -> dict[str, float]:
        """Update reward std based on goal_reach + quality."""
        n_envs = len(env_ids) if not isinstance(env_ids, torch.Tensor) else env_ids.shape[0]
        sr, qr = 0.0, 0.0

        if num_steps is not None:
            progress = min(1.0, env.common_step_counter / num_steps)
        else:
            # Step 1-2: compute goal_reach/quality and write to ring buffer
            if n_envs > 0:
                env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
                goal_reach = _goal_reached_check(
                    self._command_term, env_ids_t,
                    goal_pos_tolerance, goal_ori_tolerance,
                ).float()
                quality = goal_reach * _compute_quality(
                    self._command_term, env_ids_t,
                    contouring_threshold,
                    orientation_threshold,
                )
                self._ptr, self._filled = _write_ring_buffer(
                    self._goal_reach_buf, self._quality_buf,
                    goal_reach, quality,
                    self._ptr, self._filled, self._window_size,
                )

            # Step 3: read rates from buffer (post-write)
            sr, qr = _get_buffer_rates(
                self._goal_reach_buf, self._quality_buf,
                self._ptr, self._filled,
                self._warmup_min_samples,
            )

            # Step 4: adjust level (deferred until warmup_min_adjust_samples)
            if self._filled or self._ptr > self._warmup_min_adjust_samples:
                if sr > move_up_rate and qr > quality_up_rate and self._level < self._max_level:
                    self._level += 1
                elif sr < move_down_rate and self._level > 0:
                    self._level -= 1

            progress = self._level / max(self._max_level, 1)

        result = {
            "progress": progress,
            "level": float(self._level),
            "goal_reach_rate": sr,
            "quality_rate": qr,
        }

        # Step 5: update all std params
        for std_cfg in self._std_configs:
            p_name = std_cfg["param_name"]
            init_std = std_cfg["initial_std"]
            fin_std = std_cfg["final_std"]

            new_std = _linear_interpolation(init_std, fin_std, progress)
            self._term_cfg.params[p_name] = new_std
            result[p_name] = new_std

        env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return result


class dynamic_obstacle_curriculum(ManagerTermBase):
    """Quantized-step dynamic obstacle curriculum driven by success/collision/tip.

    Progression is driven by three episode-outcome rates:
    - **success_rate**: fraction of episodes where ``goal_reached`` fired
    - **collision_rate**: fraction of episodes terminated by obstacle collision
    - **tip_rate**: fraction of episodes terminated by tipping over

    Quality-rate (contouring/orientation) is NOT used for upgrade decisions.
    The task objective is "reach the goal without collision or tipping";
    the path serves only as coarse guidance.

    ``global_difficulty`` directly represents the number of active
    dynamic obstacles per environment (0 .. ``max_active``).
    There is **no fixed stage list** — the upper bound always comes
    from ``max_active`` (defaults to ``NUM_DYN_TOTAL``).

    Each adjustment changes ``global_difficulty`` by
    ``±difficulty_step`` (default 5) and clamps to
    ``[0, max_active]``.

    Adjustment happens at most once per ``adjust_every_samples``
    new samples, and only when the streak condition is met:
    - upgrade: ``_up_streak >= up_hold_windows`` consecutive windows
      with success_rate >= threshold AND collision_rate <= threshold
      AND tip_rate <= threshold.
    - downgrade: ``_down_streak >= down_hold_windows`` consecutive
      windows with success_rate <= threshold OR collision_rate >= threshold
      OR tip_rate >= threshold.

    Execution order per call:
    1. Classify episode outcomes for current batch (success/collision/tip)
    2. Write outcomes to ring buffers
    3. Accumulate sample counter; if < adjust_every_samples skip 4
    4. Read rates → update streaks → adjust difficulty (±step)
    5. Push difficulty to resetting envs

    Warmup: rates return 0.0 until ``warmup_min_samples``;
    difficulty adjustment deferred until
    ``warmup_min_adjust_samples`` accumulated.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._command_name: str = cfg.params.get(
            "command_name", "ee_traj",
        )

        # Upper bound & quantization step
        self._max_active: int = cfg.params.get(
            "max_active", NUM_DYN_TOTAL,
        )
        self._difficulty_step: int = cfg.params.get(
            "difficulty_step", 5,
        )

        # Active obstacle count (0 .. max_active)
        self.global_difficulty: int = 0

        # Ring buffers for three outcome channels
        ws = cfg.params.get("window_size", 8000)
        self._window_size = ws
        self._success_buf = torch.zeros(
            ws, device=env.device, dtype=torch.float,
        )
        self._collision_buf = torch.zeros(
            ws, device=env.device, dtype=torch.float,
        )
        self._tip_buf = torch.zeros(
            ws, device=env.device, dtype=torch.float,
        )
        self._ptr = 0
        self._filled = False

        # Per-env difficulty buffer
        _obstacle_difficulty[id(env)] = torch.zeros(
            env.scene.num_envs,
            dtype=torch.long,
            device=env.device,
        )
        _obstacle_active_count[id(env)] = self.global_difficulty
        _obstacle_prev_active_count[id(env)] = self.global_difficulty

        # Sample-based adjustment interval
        self._adjust_every: int = cfg.params.get(
            "adjust_every_samples", 8000,
        )
        self._samples_since_adjust: int = 0

        # Hysteresis streaks
        self._up_streak: int = 0
        self._down_streak: int = 0
        self._up_hold: int = cfg.params.get(
            "up_hold_windows", 3,
        )
        self._down_hold: int = cfg.params.get(
            "down_hold_windows", 3,
        )

        # Warmup thresholds
        self._warmup_min_samples = cfg.params.get(
            "warmup_min_samples",
            _DEFAULT_WARMUP_MIN_SAMPLES,
        )
        self._warmup_min_adjust_samples = cfg.params.get(
            "warmup_min_adjust_samples",
            _DEFAULT_WARMUP_MIN_ADJUST_SAMPLES,
        )

        # Total accumulated sample count (for warmup gate)
        self._total_samples: int = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        max_active: int = NUM_DYN_TOTAL,
        difficulty_step: int = 5,
        command_name: str = "ee_traj",
        goal_pos_tolerance: float = 0.15,
        goal_ori_tolerance: float | None = 0.30,
        # Upgrade thresholds
        move_up_success_rate: float = 0.82,
        move_up_collision_rate: float = 0.18,
        move_up_tip_rate: float = 0.01,
        # Downgrade thresholds
        move_down_success_rate: float = 0.45,
        move_down_collision_rate: float = 0.30,
        move_down_tip_rate: float = 0.03,
        # Ring buffer & timing
        window_size: int = 8000,
        warmup_min_samples: int = _DEFAULT_WARMUP_MIN_SAMPLES,
        warmup_min_adjust_samples: int = _DEFAULT_WARMUP_MIN_ADJUST_SAMPLES,
        adjust_every_samples: int = 8000,
        up_hold_windows: int = 3,
        down_hold_windows: int = 3,
    ) -> dict[str, float]:
        """Record episode outcomes and adjust obstacle count."""
        sr, cr, tr = 0.0, 0.0, 0.0

        n_envs = len(env_ids)
        if n_envs == 0:
            sr, cr, tr = self._get_rates()
            return self._build_log(sr, cr, tr)

        # Step 1: classify episode outcomes for current batch
        command_term = env.command_manager.get_term(command_name)
        env_ids_t = torch.as_tensor(
            env_ids, device=env.device, dtype=torch.long,
        )

        # Success: goal reached
        success = _goal_reached_check(
            command_term, env_ids_t,
            goal_pos_tolerance, goal_ori_tolerance,
        ).float()

        # Collision: obstacle_collision termination fired
        collision = env.termination_manager.get_term("obstacle_collision")[env_ids_t].float()

        # Tip: tipped_over termination fired
        tip = env.termination_manager.get_term("tipped_over")[env_ids_t].float()

        # Step 2: write to ring buffers
        self._write_buffers(success, collision, tip)

        # Track sample counts
        self._samples_since_adjust += n_envs
        self._total_samples += n_envs

        # Step 3: read rates (always, for logging)
        sr, cr, tr = self._get_rates()

        # Step 4: evaluate difficulty adjustment
        warmup_ok = (
            self._filled
            or self._total_samples > self._warmup_min_adjust_samples
        )
        do_adjust = (
            warmup_ok
            and self._samples_since_adjust >= self._adjust_every
        )

        if do_adjust:
            # Upgrade condition: high success, low collision, low tip
            up_condition = (
                sr >= move_up_success_rate
                and cr <= move_up_collision_rate
                and tr <= move_up_tip_rate
            )
            # Downgrade condition: low success OR high collision OR high tip
            down_condition = (
                sr <= move_down_success_rate
                or cr >= move_down_collision_rate
                or tr >= move_down_tip_rate
            )

            # Update streaks
            if up_condition:
                self._up_streak += 1
            else:
                self._up_streak = 0

            if down_condition:
                self._down_streak += 1
            else:
                self._down_streak = 0

            # Quantized difficulty adjustment (±step)
            step = self._difficulty_step
            if (
                self._up_streak >= self._up_hold
                and self.global_difficulty < self._max_active
            ):
                self.global_difficulty = min(
                    self._max_active,
                    self.global_difficulty + step,
                )
                self._up_streak = 0
                self._down_streak = 0
            elif (
                self._down_streak >= self._down_hold
                and self.global_difficulty > 0
            ):
                self.global_difficulty = max(
                    0,
                    self.global_difficulty - step,
                )
                self._up_streak = 0
                self._down_streak = 0

            # Reset sample counter after evaluation
            self._samples_since_adjust = 0

        # Step 5: push difficulty to resetting envs
        diff_buf = _obstacle_difficulty[id(env)]
        diff_buf[env_ids_t] = self.global_difficulty
        _refresh_dynamic_obstacle_active_limit(env, self._max_active)

        return self._build_log(sr, cr, tr)

    def _write_buffers(
        self,
        success: torch.Tensor,
        collision: torch.Tensor,
        tip: torch.Tensor,
    ) -> None:
        """Write outcome samples to the three ring buffers."""
        n = success.shape[0]
        ws = self._window_size

        if n >= ws:
            self._success_buf[:] = success[-ws:]
            self._collision_buf[:] = collision[-ws:]
            self._tip_buf[:] = tip[-ws:]
            self._ptr = 0
            self._filled = True
            return

        end = self._ptr + n
        if end <= ws:
            self._success_buf[self._ptr:end] = success
            self._collision_buf[self._ptr:end] = collision
            self._tip_buf[self._ptr:end] = tip
        else:
            overflow = end - ws
            remain = n - overflow
            self._success_buf[self._ptr:] = success[:remain]
            self._success_buf[:overflow] = success[remain:]
            self._collision_buf[self._ptr:] = collision[:remain]
            self._collision_buf[:overflow] = collision[remain:]
            self._tip_buf[self._ptr:] = tip[:remain]
            self._tip_buf[:overflow] = tip[remain:]

        self._ptr = (self._ptr + n) % ws
        self._filled = self._filled or (end >= ws)

    def _get_rates(self) -> tuple[float, float, float]:
        """Get success/collision/tip rates from ring buffers."""
        if self._filled:
            return (
                float(self._success_buf.mean()),
                float(self._collision_buf.mean()),
                float(self._tip_buf.mean()),
            )
        elif self._ptr > self._warmup_min_samples:
            return (
                float(self._success_buf[:self._ptr].mean()),
                float(self._collision_buf[:self._ptr].mean()),
                float(self._tip_buf[:self._ptr].mean()),
            )
        return 0.0, 0.0, 0.0

    def _build_log(
        self, sr: float, cr: float, tr: float,
    ) -> dict[str, float]:
        """Build logging dict."""
        ma = max(self._max_active, 1)
        return {
            "progress": self.global_difficulty / ma,
            "level": float(self.global_difficulty),
            "global_difficulty": float(self.global_difficulty),
            "max_possible_difficulty": float(self._max_active),
            "goal_reach_rate": sr,
            "obstacle_collision_rate": cr,
            "tipped_over_rate": tr,
        }


# ==============================================================================
# Event Functions
# ==============================================================================


def reset_dynamic_obstacles_navrl_style(
    env,
    env_ids,
    safe_radius: float = 1.5,
    cuboid_hover_range: tuple[float, float] = (0.5, 1.0),
    vel_range: tuple[float, float] = (0.5, 2.0),
    local_range: tuple[float, float, float] = (3.0, 3.0, 1.0),
    write_to_sim: bool = False,
):
    """渐进式动态障碍物重置 + 目标点随机游走初始化。

    map_half_size 从 terrain.cfg.terrain_generator.size 读取,
    dynamic obstacle size 从 constants.py 读取。

    障碍物分两大类:
      1. Cuboid (悬浮桥型): gravity off, 随机悬浮高度
      2. Cylinder (地面柱体): 站立在地面

    激活策略 (quantized difficulty mask):
      per-env difficulty d = 当前激活障碍物数量,
      由 dynamic_obstacle_curriculum 维护 (按 difficulty_step
      量化增减, 上限 max_active / NUM_DYN_TOTAL)。
      第 i 个 obstacle 在 difficulty > i 时激活,
      否则隐藏到 z=-100。

    运动策略 (参考 env.py move_dynamic_obstacle):
      每个障碍物维护 origin / goal / vel_mag。
      step_kinematic_obstacles 中:
        - 方向每步实时重算 (pos → goal)
        - 到达目标后重采样目标
        - 速度大小周期性重采样

    Args:
        env: 环境实例。
        env_ids: 需要重置的环境索引。
        safe_radius: 安全半径 (m)。
        cuboid_hover_range: (min_bottom, max_bottom)
            方块底面的悬浮高度范围 (m)。
            扰动角度范围 (rad)。
        vel_range: (min, max) 速度大小采样范围.
        local_range: (x, y, z) 目标点采样半范围.
        write_to_sim: 是否把 buffer 状态同步到 RigidObjectCollection，
            仅用于可视化/debug；训练逻辑只读 buffer。
    """
    env_key = id(env)
    if not isinstance(env_ids, torch.Tensor):
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    active_limit = _refresh_dynamic_obstacle_active_limit(env, NUM_DYN_TOTAL)
    prev_limit = _obstacle_prev_active_count.get(env_key, 0)
    write_limit = min(max(active_limit, prev_limit), NUM_DYN_TOTAL)

    if write_limit <= 0:
        _obstacle_prev_active_count[env_key] = active_limit
        return

    # ── Per-env difficulty from curriculum ──
    diff_buf = _obstacle_difficulty.get(env_key)
    if diff_buf is not None:
        difficulty_int = diff_buf[env_ids]
    else:
        difficulty_int = torch.full(
            (len(env_ids),), NUM_DYN_TOTAL,
            device=env.device, dtype=torch.long,
        )

    # ── Terrain ──
    terrain = getattr(env.scene, 'terrain', None)
    if terrain is None:
        return
    t_cfg = terrain.cfg.terrain_generator
    mr_x = t_cfg.size[0] / 2.0
    mr_y = t_cfg.size[1] / 2.0

    lr_x, lr_y, lr_z = local_range
    state = get_dyn_state(env)
    write_source_idx_tuple = get_dyn_source_index_tuple(write_limit)

    for rank, source_idx in enumerate(write_source_idx_tuple):
        if rank >= active_limit:
            _hide_dynamic_obstacle_state(env, env_ids, source_idx)
            continue

        is_cuboid, size = _dynamic_obstacle_size_from_source_idx(source_idx)
        if is_cuboid:
            half_h = size[2] / 2.0
            lo, hi = cuboid_hover_range
            z_min_off = lo + half_h
            z_max_off = hi + half_h
        else:
            fixed_z = size[2] / 2.0

        active_mask = difficulty_int > rank
        if (~active_mask).any():
            _hide_dynamic_obstacle_state(env, env_ids[~active_mask], source_idx)

        if not active_mask.any():
            continue

        ids_on = env_ids[active_mask]
        n_active = len(ids_on)
        origins = terrain.env_origins[ids_on]
        offset = torch.zeros((n_active, 3), device=env.device)

        offset[:, 0] = (torch.rand(n_active, device=env.device) * 2 - 1.0) * (mr_x - 0.5)
        offset[:, 1] = (torch.rand(n_active, device=env.device) * 2 - 1.0) * (mr_y - 0.5)
        pos_w_xy = origins[:, :2] + torch.stack((offset[:, 0], offset[:, 1]), dim=-1)
        robot = env.scene["robot"]
        robot_pos_xy = robot.data.root_pos_w[ids_on, :2]
        to_obstacle = pos_w_xy - robot_pos_xy
        dist_to_robot = to_obstacle.norm(dim=-1, keepdim=True)
        too_close = (dist_to_robot < safe_radius).squeeze(-1)
        if too_close.any():
            push_dir = to_obstacle[too_close] / dist_to_robot[too_close].clamp(min=1e-4)
            pos_w_xy[too_close] = robot_pos_xy[too_close] + push_dir * safe_radius
            pos_w_xy[:, 0] = pos_w_xy[:, 0].clamp(origins[:, 0] - mr_x + 0.5, origins[:, 0] + mr_x - 0.5)
            pos_w_xy[:, 1] = pos_w_xy[:, 1].clamp(origins[:, 1] - mr_y + 0.5, origins[:, 1] + mr_y - 0.5)

        offset[:, 0] = pos_w_xy[:, 0] - origins[:, 0]
        offset[:, 1] = pos_w_xy[:, 1] - origins[:, 1]
        if is_cuboid:
            bottom = lo + (hi - lo) * torch.rand(n_active, device=env.device)
            offset[:, 2] = bottom + half_h
        else:
            offset[:, 2] = fixed_z

        pos_w = origins + offset
        dx = 2.0 * lr_x * torch.rand(n_active, device=env.device) - lr_x
        dy = 2.0 * lr_y * torch.rand(n_active, device=env.device) - lr_y
        dz = 2.0 * lr_z * torch.rand(n_active, device=env.device) - lr_z
        init_goal = pos_w.clone()
        init_goal[:, 0] += dx
        init_goal[:, 1] += dy
        init_goal[:, 2] += dz

        t_orig = terrain.env_origins[ids_on]
        init_goal[:, 0] = init_goal[:, 0].clamp(
            min=t_orig[:, 0] - mr_x,
            max=t_orig[:, 0] + mr_x,
        )
        init_goal[:, 1] = init_goal[:, 1].clamp(
            min=t_orig[:, 1] - mr_y,
            max=t_orig[:, 1] + mr_y,
        )
        if is_cuboid:
            init_goal[:, 2] = init_goal[:, 2].clamp(
                min=t_orig[:, 2] + z_min_off,
                max=t_orig[:, 2] + z_max_off,
            )
        else:
            init_goal[:, 2] = fixed_z

        direction = init_goal - pos_w
        dist = direction.norm(dim=-1, keepdim=True).clamp(min=1e-4)
        direction = direction / dist
        init_mag = vel_range[0] + (
            vel_range[1] - vel_range[0]
        ) * torch.rand(n_active, 1, device=env.device)

        vel = torch.zeros((n_active, 6), device=env.device)
        vel[:, :3] = direction * init_mag
        if not is_cuboid:
            vel[:, 2] = 0.0

        state["pos_w"][ids_on, source_idx] = pos_w
        state["quat_w"][ids_on, source_idx] = 0.0
        state["quat_w"][ids_on, source_idx, 0] = 1.0
        state["vel_w"][ids_on, source_idx] = vel
        state["goal_w"][ids_on, source_idx] = init_goal
        state["goal_age"][ids_on, source_idx] = 0.0
        state["origin_w"][ids_on, source_idx] = pos_w
        state["speed_mag"][ids_on, source_idx] = init_mag
        state["active_mask"][ids_on, source_idx] = True

    state["_last_step"] = env.common_step_counter
    if write_to_sim:
        write_source_indices = get_dyn_source_indices(env, write_limit)
        write_dyn_state_to_sim(env, env_ids=env_ids, source_indices=write_source_indices)
    _obstacle_prev_active_count[env_key] = active_limit
    return
