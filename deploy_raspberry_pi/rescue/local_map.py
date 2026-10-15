"""
地面端局部地图更新：用小车激光雷达修正航拍栅格地图

    命中点  -> 标记为障碍（补上空中不可见、漏检的障碍）
    射线经过的近处格子 -> 清除障碍（修正航拍检测框偏大、误检造成的“假障碍”）

car_node.py 与 sim.py 共用本模块，保证仿真与真机的地图更新逻辑一致。
"""

from __future__ import annotations

import math

import numpy as np

from .geo_mapping import GridMap, inflate_manhattan


class LidarMapper:
    def __init__(self, gm: GridMap, inflate_m: float, carve_range: float = 4.0):
        self.gm = gm
        self.raw = (gm.raw if gm.raw is not None else gm.grid).copy()
        self.r = int(math.ceil(inflate_m / gm.resolution))
        self.carve_range = carve_range

    def update(self, pose, rel_deg, ranges):
        """
        Args:
            pose: 小车位姿 (x, y, theta)，地面坐标系
            rel_deg: 扫描点角度（车体系，逆时针为正，度）
            ranges: 距离（米）；inf = 无回波；0 / NaN = 无效测量（丢弃）
        """
        x, y, th = pose
        rel = np.asarray(rel_deg, float)
        rng = np.asarray(ranges, float)
        ok = ~np.isnan(rng) & (rng > 0.02)
        rel, rng = rel[ok], rng[ok]
        a = th + np.radians(rel)
        gm = self.gm
        # 清除：射线上距离 < min(命中-0.3, carve_range) 的点
        lim = np.minimum(np.where(np.isfinite(rng), rng - 0.3, np.inf), self.carve_range)
        r = np.arange(0.2, self.carve_range, gm.resolution / 2)
        mask = r[None, :] < lim[:, None]
        xs = (x + np.cos(a)[:, None] * r)[mask]
        ys = (y + np.sin(a)[:, None] * r)[mask]
        self._set(xs, ys, 0)
        hit = np.isfinite(rng)
        self._set(x + np.cos(a[hit]) * rng[hit], y + np.sin(a[hit]) * rng[hit], 1)
        gm.raw = self.raw
        gm.grid = inflate_manhattan(self.raw, self.r)

    def _set(self, xs, ys, val):
        gm = self.gm
        rows = np.floor((gm.y_max - ys) / gm.resolution).astype(int)
        cols = np.floor((xs - gm.x_min) / gm.resolution).astype(int)
        ok = (rows >= 0) & (rows < gm.grid.shape[0]) & (cols >= 0) & (cols < gm.grid.shape[1])
        self.raw[rows[ok], cols[ok]] = val

    def path_blocked(self, path_xy, skip: int = 2) -> bool:
        """当前路径（地面坐标点列）是否被更新后的地图阻断；skip 跳过小车脚下的前几个点"""
        gm = self.gm
        for x, y in list(path_xy)[skip:]:
            r, c = gm.world_to_cell(x, y)
            if gm.in_bounds(r, c) and gm.grid[r, c]:
                return True
        return False
