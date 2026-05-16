import numpy as np

from isaaclab.utils import configclass
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import (
    HfDiscreteObstaclesTerrainCfg,
)
from isaaclab.terrains.height_field.utils import (
    height_field_to_mesh,
)


# ─────────────────────────────────────────────────────────
#  Static obstacles terrain
# ─────────────────────────────────────────────────────────

@height_field_to_mesh
def static_obstacles_terrain(
    difficulty: float,
    cfg: "StaticObstaclesCfg",
) -> np.ndarray:
    """生成固定数量的随机静态障碍物地形。

    difficulty 参数仅用于满足接口要求，实际障碍物数量固定。
    中心区域 (platform_width) 保持平坦，作为机器人安全生成区。

    Args:
        difficulty: [0, 1] 之间的难度参数 (未使用)
        cfg: 地形配置

    Returns:
        np.ndarray: 高度场 (int16)
    """
    w = int(cfg.size[0] / cfg.horizontal_scale)
    h = int(cfg.size[1] / cfg.horizontal_scale)

    hf = np.zeros((w, h))

    n_obs = cfg.num_obstacles

    # 障碍物高度 (pixels)
    obs_h = int(
        cfg.obstacle_height / cfg.vertical_scale
    )

    # 障碍物宽度范围 (pixels)
    w_min = int(
        cfg.obstacle_width_range[0] / cfg.horizontal_scale
    )
    w_max = int(
        cfg.obstacle_width_range[1] / cfg.horizontal_scale
    )
    obs_w_range = np.arange(
        w_min, max(w_max, w_min + 1), 4
    )
    if len(obs_w_range) == 0:
        obs_w_range = np.array([w_min])
    obs_l_range = obs_w_range

    # 中心安全区尺寸 (pixels)
    plat_px = int(
        cfg.platform_width / cfg.horizontal_scale
    )
    x1 = (w - plat_px) // 2
    x2 = (w + plat_px) // 2
    y1 = (h - plat_px) // 2
    y2 = (h + plat_px) // 2

    # ── 随机最小间距范围 ──
    # spacing_range 表示每个障碍物独立采样的最小间距 (m)，
    # 比固定 1.6m 更容易在高密度场景中放满，同时保留随机性。
    spacing_low, spacing_high = cfg.obstacle_spacing_range
    assert spacing_low > 0.0 and spacing_high >= spacing_low

    # ── core + buffer 双 mask 放置 ──
    # core_mask 只记录真实障碍物核心区域和中心安全出生区核心区域；
    # buffer_mask 记录每个障碍物核心区按自身 spacing_m 膨胀后的禁入区。
    #
    # 接受条件：
    #   1. 候选核心不能进入已有 buffer；
    #   2. 候选 buffer 不能覆盖已有 core。
    # 这样同时避免「候选障碍物进入已有障碍物 buffer」和
    #「已有障碍物进入候选障碍物 buffer」。
    core_mask = np.zeros((w, h), dtype=bool)
    buffer_mask = np.zeros((w, h), dtype=bool)
    core_mask[x1:x2, y1:y2] = True

    # 高密度静态障碍物会频繁采样到冲突位置，需要比 n_obs * 10
    # 更高的尝试上限，避免过早停止。
    placed_count = 0
    attempts_used = 0
    max_attempts = n_obs * cfg.max_place_attempts_per_obstacle
    for _ in range(max_attempts):
        if placed_count >= n_obs:
            break
        attempts_used += 1

        bw = int(np.random.choice(obs_w_range))
        bl = int(np.random.choice(obs_l_range))
        bx = np.random.randint(0, w - bw + 1)
        by = np.random.randint(0, h - bl + 1)
        spacing_m = np.random.uniform(spacing_low, spacing_high)
        spacing_px = max(1, int(np.ceil(spacing_m / cfg.horizontal_scale)))

        buf_x1 = max(0, bx - spacing_px)
        buf_x2 = min(w, bx + bw + spacing_px)
        buf_y1 = max(0, by - spacing_px)
        buf_y2 = min(h, by + bl + spacing_px)

        core_overlaps_buffer = np.any(
            buffer_mask[bx:bx + bw, by:by + bl]
        )
        buffer_overlaps_core = np.any(
            core_mask[buf_x1:buf_x2, buf_y1:buf_y2]
        )
        if core_overlaps_buffer or buffer_overlaps_core:
            continue

        hf[bx:bx + bw, by:by + bl] = obs_h
        core_mask[bx:bx + bw, by:by + bl] = True
        buffer_mask[buf_x1:buf_x2, buf_y1:buf_y2] = True
        placed_count += 1

    if placed_count < n_obs:
        print(
            "Warning: static obstacle placement only placed "
            f"{placed_count}/{n_obs} obstacles after "
            f"{attempts_used}/{max_attempts} attempts "
            f"(spacing_range={cfg.obstacle_spacing_range})."
        )
    elif cfg.debug_obstacle_placement:
        print(
            "Static obstacle placement: placed "
            f"{placed_count}/{n_obs} obstacles after "
            f"{attempts_used}/{max_attempts} attempts "
            f"(spacing_range={cfg.obstacle_spacing_range})."
        )

    # 确保中心安全区干净
    hf[x1:x2, y1:y2] = 0

    return np.rint(hf).astype(np.int16)


# ─────────────────────────────────────────────────────────
#  Sub-terrain config
# ─────────────────────────────────────────────────────────

@configclass
class StaticObstaclesCfg(HfDiscreteObstaclesTerrainCfg):
    """固定障碍物地形配置。

    障碍物数量固定，中心有安全出生区。
    """
    function = static_obstacles_terrain

    obstacle_height: float = 1.2
    """障碍物固定高度 (m)。"""

    num_obstacles: int = 400
    """障碍物数量 (固定)。"""

    obstacle_spacing_range: tuple[float, float] = (1.2, 1.6)
    """随机采样的障碍物最小间距范围，单位 m。"""

    max_place_attempts_per_obstacle: int = 80
    """每个障碍物最大尝试次数，用于提升高密度场景下的成功放置率。"""

    debug_obstacle_placement: bool = False
    """是否打印静态障碍物放置统计信息。"""


# ─────────────────────────────────────────────────────────
#  Terrain generator config
# ─────────────────────────────────────────────────────────

@configclass
class MobileManipulatorTerrainCfg(TerrainGeneratorCfg):
    """单块地形配置（无课程学习）。

    单一 50×50m 地形块，固定 300 个静态障碍物。
    所有机器人在同一块地形上训练。

    布局：
      - 1 row × 1 col = 单块 50×50m 地形
      - 中心 1×1m 安全出生区（无障碍物）
      - border_width=20m 外围平坦区域
      - border_height=0 与地面齐平
      - curriculum=False 无难度递进

    动态障碍物通过 RigidObject 单独管理。
    """
    num_rows: int = 1
    num_cols: int = 1
    size: tuple[float, float] = (50.0, 50.0)
    border_width: float = 5.0
    border_height: float = 0.0
    curriculum: bool = False

    def __post_init__(self):
        self.sub_terrains = {
            "static_obstacles": StaticObstaclesCfg(
                proportion=1.0,
                obstacle_height=1.2,
                obstacle_width_range=(0.4, 1.0),
                obstacle_height_range=(1.2, 1.6),
                obstacle_height_mode="fixed",
                num_obstacles=300,
                platform_width=0.5,
                obstacle_spacing_range=(1.3, 1.6),
                max_place_attempts_per_obstacle=80,
                debug_obstacle_placement=True,
            ),
        }
