"""
A* 全局路径规划（在 geo_mapping 生成的栅格地图上运行）

相比最初的 astar 脚本的改进:
    - 8 邻域使用八方向距离（octile）启发式，保证启发式可采纳、结果最优
      （原脚本 8 邻域仍用曼哈顿距离，会高估代价，路径不一定最短）
    - 8 邻域禁止“切角”穿过两个障碍格之间的缝
    - 输出路径的真实长度（米）与精简后的航点，供小车跟踪
"""

from __future__ import annotations

import heapq
import math

import numpy as np

SQRT2 = math.sqrt(2.0)


def heuristic(a, b, allow_diag: bool) -> float:
    dr, dc = abs(a[0] - b[0]), abs(a[1] - b[1])
    if allow_diag:
        return (dr + dc) + (SQRT2 - 2) * min(dr, dc)
    return dr + dc


def astar(grid: np.ndarray, start, goal, allow_diag: bool = True):
    """
    Args:
        grid: 0=可通行 1=障碍
        start, goal: (row, col)
    Returns:
        路径格子列表 [(row, col), ...]（含起终点），无路径返回 None
    """
    H, W = grid.shape
    start, goal = tuple(start), tuple(goal)
    if grid[start] != 0 or grid[goal] != 0:
        return None
    if allow_diag:
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]
    else:
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]

    counter = 0  # 打破 f 值并列，保证堆中元素可比较
    open_heap = [(heuristic(start, goal, allow_diag), 0.0, counter, start)]
    came_from = {}
    g_score = {start: 0.0}
    closed = set()

    while open_heap:
        _, g, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            return path[::-1]
        closed.add(cur)
        r, c = cur
        for dr, dc, cost in moves:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or grid[nr, nc] != 0:
                continue
            if dr and dc and (grid[r + dr, c] != 0 or grid[r, c + dc] != 0):
                continue  # 禁止切角
            ng = g + cost
            nb = (nr, nc)
            if ng < g_score.get(nb, math.inf):
                g_score[nb] = ng
                came_from[nb] = cur
                counter += 1
                heapq.heappush(open_heap, (ng + heuristic(nb, goal, allow_diag), ng, counter, nb))
    return None


def path_length_cells(path) -> float:
    if not path or len(path) < 2:
        return 0.0
    p = np.asarray(path, float)
    return float(np.hypot(*np.diff(p, axis=0).T).sum())


def _line_free(grid, a, b) -> bool:
    """栅格直线（Bresenham 风格采样）是否全部可通行"""
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    rs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    cs = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(np.all(grid[rs, cs] == 0))


def simplify_path(grid: np.ndarray, path):
    """视线法精简路径：只保留必须的拐点，作为小车跟踪的航点"""
    if not path or len(path) <= 2:
        return list(path or [])
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_free(grid, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


def path_to_world(gridmap, path):
    """栅格路径 -> 地面坐标（米）"""
    return [gridmap.cell_to_world(r, c) for r, c in path]


def render_ascii(grid: np.ndarray, path=None, start=None, goal=None, margin: int = 15) -> str:
    """字符可视化（保留原 astar 脚本的输出风格）: . 空闲  # 障碍  o 路径  S 起点  G 终点"""
    vis = np.where(grid != 0, "#", ".").astype("<U1")
    for r, c in path or []:
        vis[r, c] = "o"
    if start is not None:
        vis[start] = "S"
    if goal is not None:
        vis[goal] = "G"
    pts = [p for p in (start, goal) if p is not None]
    if pts:
        r0 = max(0, min(p[0] for p in pts) - margin)
        r1 = min(grid.shape[0], max(p[0] for p in pts) + margin + 1)
        c0 = max(0, min(p[1] for p in pts) - margin)
        c1 = min(grid.shape[1], max(p[1] for p in pts) + margin + 1)
        vis = vis[r0:r1, c0:c1]
    return "\n".join("".join(row) for row in vis)
