#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
栅格地图构建模块（研究内容③：检测框 -> 栅格地图）

实现申报书中描述的完整流程：
1. 通过无人机高度 + 相机 FOV + 像素数计算 GSD（每像素代表的地面尺寸，米/像素）；
2. 以图像为基础建系，把检测框四角投影到世界坐标（米），再映射到二维栅格；
3. 对障碍栅格做膨胀（按曼哈顿距离扩张，给小车留安全冗余）；
4. 在前方区域自动扫描，选取"障碍最少、空白最多"的可通行块作为终点。

该模块只依赖 numpy，可脱离摄像头/模型单独测试。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

# 障碍框输入格式：(x1, y1, x2, y2) 像素坐标（左上、右下）
Box = Tuple[float, float, float, float]


@dataclass
class CameraModel:
    """下视（nadir）相机 + 无人机位姿的最小几何模型。

    Attributes:
        height_m:  相机（无人机）离地高度，单位米。
        hfov_deg:  水平视场角，单位度。
        img_w:     图像宽度，单位像素。
        img_h:     图像高度，单位像素。
        vfov_deg:  垂直视场角，单位度；为 None 时假设方形像素，
                   由水平 GSD 推出垂直方向足迹。
    """

    height_m: float
    hfov_deg: float
    img_w: int
    img_h: int
    vfov_deg: Optional[float] = None

    def gsd(self) -> Tuple[float, float]:
        """返回 (gsd_x, gsd_y)，单位 米/像素。

        下视相机地面足迹宽度 = 2 * H * tan(FOV/2)，
        GSD = 足迹 / 像素数。
        """
        if self.height_m <= 0:
            raise ValueError("height_m 必须为正")
        if not (0 < self.hfov_deg < 180):
            raise ValueError("hfov_deg 必须在 (0, 180) 之间")

        ground_w = 2.0 * self.height_m * math.tan(math.radians(self.hfov_deg) / 2.0)
        gsd_x = ground_w / self.img_w

        if self.vfov_deg is not None:
            ground_h = 2.0 * self.height_m * math.tan(math.radians(self.vfov_deg) / 2.0)
            gsd_y = ground_h / self.img_h
        else:
            # 方形像素：垂直方向与水平方向每像素地面尺寸一致
            gsd_y = gsd_x
        return gsd_x, gsd_y

    def footprint_m(self) -> Tuple[float, float]:
        """返回地面足迹 (width_m, height_m)。"""
        gx, gy = self.gsd()
        return gx * self.img_w, gy * self.img_h

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        """像素坐标 -> 世界坐标（米）。

        世界系：原点在图像中心（相机正下方地面点），
        x 轴向右为正，y 轴"向前"（图像向上）为正。
        """
        gx, gy = self.gsd()
        wx = (px - self.img_w / 2.0) * gx
        wy = (self.img_h / 2.0 - py) * gy  # 图像 y 向下，翻转成"向前为正"
        return wx, wy


@dataclass
class GridMap:
    """世界坐标下的二维占据栅格。

    grid[row, col]: 0=可通行, 1=障碍。
    row 0 位于地图最"前方"（世界 y 最大），col 0 位于最左（世界 x 最小）。
    """

    grid: np.ndarray
    cell_size_m: float
    origin_col_m: float  # 世界 x 最小值（对应 col 0 左边界）
    origin_row_m: float  # 世界 y 最大值（对应 row 0 上边界）
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    def world_to_cell(self, wx: float, wy: float) -> Tuple[int, int]:
        col = int((wx - self.origin_col_m) / self.cell_size_m)
        row = int((self.origin_row_m - wy) / self.cell_size_m)
        return row, col


def build_grid(
    boxes: Iterable[Box],
    cam: CameraModel,
    cell_size_m: float = 0.20,
    inflate_m: float = 0.30,
) -> GridMap:
    """把障碍检测框投影为世界坐标栅格地图（含膨胀）。

    Args:
        boxes:        障碍框列表，(x1,y1,x2,y2) 像素坐标。
        cam:          相机/无人机几何模型。
        cell_size_m:  栅格分辨率，单位 米/格（默认 0.20m）。
        inflate_m:    障碍膨胀半径，单位米（默认 0.30m，约等于半个车宽的安全冗余）。

    Returns:
        GridMap，其中 grid 为 uint8 (0 空闲 / 1 障碍)。
    """
    if cell_size_m <= 0:
        raise ValueError("cell_size_m 必须为正")

    foot_w, foot_h = cam.footprint_m()
    n_cols = max(1, math.ceil(foot_w / cell_size_m))
    n_rows = max(1, math.ceil(foot_h / cell_size_m))

    grid = np.zeros((n_rows, n_cols), dtype=np.uint8)
    gmap = GridMap(
        grid=grid,
        cell_size_m=cell_size_m,
        origin_col_m=-foot_w / 2.0,
        origin_row_m=foot_h / 2.0,
        meta={
            "gsd_m_per_px": cam.gsd(),
            "footprint_m": (foot_w, foot_h),
            "height_m": cam.height_m,
            "hfov_deg": cam.hfov_deg,
        },
    )

    for x1, y1, x2, y2 in boxes:
        # 障碍框四角 -> 世界坐标 -> 栅格，取覆盖的行列范围置为 1
        corners_px = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        rows, cols = [], []
        for px, py in corners_px:
            wx, wy = cam.pixel_to_world(px, py)
            r, c = gmap.world_to_cell(wx, wy)
            rows.append(r)
            cols.append(c)
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        # 裁剪到地图范围
        r0 = max(0, r0)
        c0 = max(0, c0)
        r1 = min(n_rows - 1, r1)
        c1 = min(n_cols - 1, c1)
        if r0 <= r1 and c0 <= c1:
            grid[r0 : r1 + 1, c0 : c1 + 1] = 1

    if inflate_m > 0:
        inflate_cells = int(math.ceil(inflate_m / cell_size_m))
        gmap.grid = inflate_obstacles(grid, inflate_cells)

    return gmap


def inflate_obstacles(grid: np.ndarray, radius_cells: int) -> np.ndarray:
    """按曼哈顿距离膨胀障碍。

    把与任一障碍格曼哈顿距离 <= radius_cells 的空闲格也标记为障碍，
    等价于对 4-邻域二值膨胀迭代 radius_cells 次，给小车留安全冗余。
    """
    if radius_cells <= 0:
        return grid.copy()

    out = (grid != 0).astype(np.uint8)
    for _ in range(radius_cells):
        shifted = out.copy()
        shifted[1:, :] |= out[:-1, :]   # 下
        shifted[:-1, :] |= out[1:, :]   # 上
        shifted[:, 1:] |= out[:, :-1]   # 右
        shifted[:, :-1] |= out[:, 1:]   # 左
        out = shifted
    return out


def select_goal(
    grid: np.ndarray,
    window: int = 3,
    front_fraction: float = 0.5,
) -> Optional[Tuple[int, int]]:
    """在前方区域自动选取"障碍最少、空白最多"的块作为终点。

    在栅格前方（row 较小 = 世界 y 较大 = 更远/更前方）区域滑动一个
    window×window 窗口，选取空闲格最多的窗口中心；空闲数相同时优先
    更靠前（row 更小）、更居中的窗口。

    Args:
        grid:            占据栅格（0 空闲 / 1 障碍）。
        window:          扫描窗口边长（格）。
        front_fraction:  参与扫描的前方行占比（0~1），默认前半张图。

    Returns:
        (row, col) 终点栅格坐标；若无空闲块则返回 None。
    """
    H, W = grid.shape
    free = (grid == 0).astype(np.int32)
    half = window // 2
    front_rows = max(1, int(H * front_fraction))
    center_col = (W - 1) / 2.0

    best = None  # (free_count, -row, -|col-center|, (r,c))
    r_hi = min(front_rows, H - half)
    for r in range(half, max(half + 1, r_hi)):
        for c in range(half, W - half):
            block = free[r - half : r + half + 1, c - half : c + half + 1]
            if grid[r, c] != 0:
                continue  # 中心必须可站立
            score = int(block.sum())
            key = (score, -r, -abs(c - center_col))
            if best is None or key > best[0]:
                best = (key, (r, c))
    return best[1] if best else None


def to_ascii(
    grid: np.ndarray,
    path: Optional[Sequence[Tuple[int, int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    max_rows: int = 40,
    max_cols: int = 80,
) -> str:
    """把栅格（可选叠加路径/起终点）渲染成 ASCII 字符串，便于终端可视化。"""
    H, W = grid.shape
    vis = np.array(grid, dtype=np.int32)
    if path:
        for (x, y) in path:
            if 0 <= x < H and 0 <= y < W:
                vis[x, y] = 2
    if start is not None:
        vis[start[0], start[1]] = 3
    if goal is not None:
        vis[goal[0], goal[1]] = 4

    sym = {0: ".", 1: "#", 2: "o", 3: "S", 4: "G"}
    rows = min(H, max_rows)
    cols = min(W, max_cols)
    lines = ["Legend: .=free  #=obstacle  o=path  S=start  G=goal"]
    for i in range(rows):
        lines.append("".join(sym.get(int(vis[i, j]), "?") for j in range(cols)))
    return "\n".join(lines)
