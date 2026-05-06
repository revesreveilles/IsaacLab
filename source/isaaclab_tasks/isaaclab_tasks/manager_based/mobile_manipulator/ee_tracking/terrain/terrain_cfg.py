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

    # ── 带最小间距约束的放置 ──
    min_spacing_m = 1.6
    spacing_px = int(min_spacing_m / cfg.horizontal_scale)

    occupied_mask = np.zeros((w, h), dtype=bool)
    # 标记安全区 + 间距缓冲为已占用
    safe_x1 = max(0, x1 - spacing_px)
    safe_x2 = min(w, x2 + spacing_px)
    safe_y1 = max(0, y1 - spacing_px)
    safe_y2 = min(h, y2 + spacing_px)
    occupied_mask[safe_x1:safe_x2, safe_y1:safe_y2] = True

    placed_count = 0
    max_attempts = n_obs * 10
    for _ in range(max_attempts):
        if placed_count >= n_obs:
            break
        bw = int(np.random.choice(obs_w_range))
        bl = int(np.random.choice(obs_l_range))
        bx = np.random.randint(0, w - bw + 1)
        by = np.random.randint(0, h - bl + 1)
        if not np.any(occupied_mask[bx:bx + bw, by:by + bl]):
            hf[bx:bx + bw, by:by + bl] = obs_h
            occ_x1 = max(0, bx - spacing_px)
            occ_x2 = min(w, bx + bw + spacing_px)
            occ_y1 = max(0, by - spacing_px)
            occ_y2 = min(h, by + bl + spacing_px)
            occupied_mask[occ_x1:occ_x2, occ_y1:occ_y2] = True
            placed_count += 1
    if placed_count < n_obs:
        print(
            f"Warning: Only placed {placed_count}/{n_obs}"
            f" obstacles after {max_attempts} attempts."
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


# ─────────────────────────────────────────────────────────
#  Terrain generator config
# ─────────────────────────────────────────────────────────

@configclass
class MobileManipulatorTerrainCfg(TerrainGeneratorCfg):
    """单块地形配置（无课程学习）。

    单一 45×45m 地形块，固定 350 个静态障碍物。
    所有机器人在同一块地形上训练。

    布局：
      - 1 row × 1 col = 单块 45×45m 地形
      - 中心 1×1m 安全出生区（无障碍物）
      - border_width=20m 外围平坦区域
      - border_height=0 与地面齐平
      - curriculum=False 无难度递进

    动态障碍物通过 RigidObject 单独管理。
    """
    num_rows: int = 1
    num_cols: int = 1
    size: tuple[float, float] = (45.0, 45.0)
    border_width: float = 5.0
    border_height: float = 0.0
    curriculum: bool = False

    def __post_init__(self):
        self.sub_terrains = {
            "static_obstacles": StaticObstaclesCfg(
                proportion=1.0,
                obstacle_height=1.2,
                obstacle_width_range=(0.4, 1.0),
                obstacle_height_range=(1.2, 1.2),
                obstacle_height_mode="fixed",
                num_obstacles=350,
                platform_width=0.5,
            ),
        }
