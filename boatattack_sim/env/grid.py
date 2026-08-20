"""
boatattack_sim/env/grid.py — 고해상 격자 (충돌·그물 painting·포획 판정 전용)

순수 numpy. 액션·관측과 무관하며, world(m) 좌표를 cell로 rasterize 해
그물 띠 painting / 포획 / 충돌만 계산한다. **해상도를 키워도 액션 차원 불변.**

채널:
  painted   [G,G] bool   그물이 칠한 cell (포획 판정용)
  obstacle  [G,G] bool   장애물 cell (1차엔 비활성/전부 False)
"""
import numpy as np

from .config import SimConfig, DEFAULT_CONFIG


class Grid:
    """모선 방어 영역의 고해상 격자."""

    def __init__(self, cfg: SimConfig = DEFAULT_CONFIG):
        self.cfg = cfg
        self.G = cfg.grid_size
        self.cell = cfg.cell_size
        self.painted = np.zeros((self.G, self.G), dtype=bool)
        self.obstacle = np.zeros((self.G, self.G), dtype=bool)

    # ── 좌표 변환 (rasterize) ─────────────────────────────────────────

    def world_to_cell(self, x: float, y: float):
        """world(m) → (i, j) cell 인덱스. 맵 밖은 가장자리로 clip."""
        i = int(np.clip(x // self.cell, 0, self.G - 1))
        j = int(np.clip(y // self.cell, 0, self.G - 1))
        return i, j

    def world_to_cell_vec(self, xy: np.ndarray):
        """[K,2] world → ([K] i, [K] j) 벡터화."""
        ij = np.clip((np.asarray(xy) // self.cell).astype(np.int64),
                     0, self.G - 1)
        return ij[:, 0], ij[:, 1]

    # ── 그물 cell-painting (띠) ───────────────────────────────────────

    def paint_at(self, x: float, y: float, width: int = None):
        """world 위치 (x,y) 주변 width×width cell 블록을 painted 로 마킹.
        선박이 그물 전개 중 매 스텝 호출하면 진행 방향으로 폭 width 띠가 누적된다."""
        w = (self.cfg.net_width if width is None else width)
        h = max(0, (w - 1) // 2)
        i, j = self.world_to_cell(x, y)
        i0, i1 = max(0, i - h), min(self.G, i + h + 1)
        j0, j1 = max(0, j - h), min(self.G, j + h + 1)
        self.painted[i0:i1, j0:j1] = True

    # ── 포획 판정 ─────────────────────────────────────────────────────

    def captured_mask(self, enemy_xy: np.ndarray, alive: np.ndarray) -> np.ndarray:
        """painted cell 에 진입한 살아있는 적 → True. 반환 [M] bool."""
        if len(enemy_xy) == 0:
            return np.zeros(0, dtype=bool)
        i, j = self.world_to_cell_vec(enemy_xy)
        on_paint = self.painted[i, j]
        return on_paint & alive

    # ── 충돌 (장애물) ─────────────────────────────────────────────────

    def is_obstacle(self, x: float, y: float) -> bool:
        i, j = self.world_to_cell(x, y)
        return bool(self.obstacle[i, j])

    # ── 리셋 ──────────────────────────────────────────────────────────

    def reset(self):
        self.painted.fill(False)
        self.obstacle.fill(False)

    @property
    def painted_ratio(self) -> float:
        """칠해진 cell 비율 (효율/디버그 지표)."""
        return float(self.painted.mean())
