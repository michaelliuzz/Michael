#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A* 路径规划模块（研究内容③下游：在栅格地图上寻路）

从根目录的 `astar` 脚本重构而来，改为可导入、可测试的函数式接口，
供 detect_to_path.py 端到端流水线调用。
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

import numpy as np

Cell = Tuple[int, int]

# 4 邻域 / 8 邻域移动
MOVES_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MOVES_8 = MOVES_4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]


def nearest_free(grid: np.ndarray, p: Cell, radius: int = 10) -> Cell:
    """若 p 落在障碍上，则在半径内螺旋搜索最近的可通行格。"""
    x0, y0 = p
    H, W = grid.shape
    if 0 <= x0 < H and 0 <= y0 < W and grid[x0, y0] == 0:
        return p
    for r in range(1, radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                x, y = x0 + dx, y0 + dy
                if 0 <= x < H and 0 <= y < W and grid[x, y] == 0:
                    return (x, y)
    return p


def heuristic(a: Cell, b: Cell, allow_diag: bool) -> float:
    """4 邻域用曼哈顿距离；8 邻域用对角(八边形)距离，保证可采纳性。"""
    dx, dy = abs(a[0] - b[0]), abs(a[1] - b[1])
    if allow_diag:
        # octile 距离：diag * √2 + 剩余直行
        return (dx + dy) + (1.4142 - 2.0) * min(dx, dy)
    return dx + dy


def astar(
    grid: np.ndarray,
    start: Cell,
    goal: Cell,
    allow_diag: bool = False,
) -> Optional[List[Cell]]:
    """在占据栅格 (0 空闲 / 非0 障碍) 上运行 A*。

    Returns:
        从 start 到 goal 的格坐标列表；不可达时返回 None。
    """
    H, W = grid.shape
    moves = MOVES_8 if allow_diag else MOVES_4

    def step_cost(dx: int, dy: int) -> float:
        return 1.4142 if (dx != 0 and dy != 0) else 1.0

    open_heap: List[Tuple[float, Cell]] = [(0.0, start)]
    came_from: dict[Cell, Cell] = {}
    g_score: dict[Cell, float] = {start: 0.0}
    closed: set[Cell] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < H and 0 <= ny < W):
                continue
            if grid[nx, ny] != 0:
                continue
            # 8 邻域禁止穿越障碍夹角（防止"擦角"穿墙）
            if allow_diag and dx != 0 and dy != 0:
                if grid[cx + dx, cy] != 0 and grid[cx, cy + dy] != 0:
                    continue
            neighbor = (nx, ny)
            tentative = g_score[current] + step_cost(dx, dy)
            if neighbor not in g_score or tentative < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f = tentative + heuristic(neighbor, goal, allow_diag)
                heapq.heappush(open_heap, (f, neighbor))

    return None


def plan(
    grid: np.ndarray,
    start: Cell,
    goal: Cell,
    allow_diag: bool = False,
    snap_radius: int = 10,
) -> Tuple[Optional[List[Cell]], Cell, Cell]:
    """便捷入口：自动把起终点吸附到最近可通行格后再规划。

    Returns:
        (path, snapped_start, snapped_goal)
    """
    s = nearest_free(grid, start, snap_radius)
    g = nearest_free(grid, goal, snap_radius)
    return astar(grid, s, g, allow_diag=allow_diag), s, g
