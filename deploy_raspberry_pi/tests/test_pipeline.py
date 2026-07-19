#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""流水线核心逻辑的可运行测试（只需 numpy，无需模型/相机）。

运行：python3 deploy_raspberry_pi/tests/test_pipeline.py
"""

import math
import sys
from pathlib import Path

import numpy as np

# 允许直接运行：把 src 加入路径
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from grid_map import (  # noqa: E402
    CameraModel,
    build_grid,
    inflate_obstacles,
    select_goal,
)
from astar_planner import astar, nearest_free, plan  # noqa: E402


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_gsd_math():
    # H=10m, HFOV=90° -> 足迹宽 = 2*10*tan(45°) = 20m；640px -> 0.03125 m/px
    cam = CameraModel(height_m=10.0, hfov_deg=90.0, img_w=640, img_h=480)
    gx, gy = cam.gsd()
    assert approx(gx, 20.0 / 640), gx
    assert approx(gy, gx), (gx, gy)  # 方形像素
    fw, fh = cam.footprint_m()
    assert approx(fw, 20.0), fw
    assert approx(fh, 20.0 / 640 * 480), fh
    print("✓ test_gsd_math")


def test_pixel_to_world_center():
    cam = CameraModel(height_m=10.0, hfov_deg=90.0, img_w=640, img_h=480)
    # 图像中心 -> 世界原点
    wx, wy = cam.pixel_to_world(320, 240)
    assert approx(wx, 0.0) and approx(wy, 0.0), (wx, wy)
    # 右上角像素 -> 世界 x>0, y>0
    wx, wy = cam.pixel_to_world(640, 0)
    assert wx > 0 and wy > 0, (wx, wy)
    print("✓ test_pixel_to_world_center")


def test_build_grid_marks_obstacle():
    cam = CameraModel(height_m=10.0, hfov_deg=90.0, img_w=640, img_h=480)
    # 图像正中的一个框，应在栅格中部落下障碍
    boxes = [(300, 220, 340, 260)]
    gmap = build_grid(boxes, cam, cell_size_m=0.20, inflate_m=0.0)
    assert gmap.grid.sum() > 0, "障碍应被标记"
    # 中心行列附近应为障碍
    H, W = gmap.grid.shape
    assert gmap.grid[H // 2, W // 2] == 1 or gmap.grid[H // 2 - 1, W // 2] == 1
    print("✓ test_build_grid_marks_obstacle")


def test_inflate():
    g = np.zeros((7, 7), dtype=np.uint8)
    g[3, 3] = 1
    out = inflate_obstacles(g, 1)
    # 曼哈顿距离<=1 的 4 个邻居被填充，总计 5 格
    assert out.sum() == 5, out.sum()
    assert out[2, 3] == 1 and out[4, 3] == 1 and out[3, 2] == 1 and out[3, 4] == 1
    # 半径 2
    out2 = inflate_obstacles(g, 2)
    assert out2.sum() == 13, out2.sum()  # 曼哈顿半径2的菱形 = 13 格
    print("✓ test_inflate")


def test_select_goal_prefers_free_front():
    grid = np.zeros((20, 20), dtype=np.uint8)
    # 在前方左侧堆障碍，前方右侧留空 -> 终点应偏向空的一侧
    grid[0:6, 0:8] = 1
    goal = select_goal(grid, window=3, front_fraction=0.5)
    assert goal is not None
    r, c = goal
    assert r < 10, f"终点应在前方: {goal}"
    assert c >= 8, f"终点应在空闲侧: {goal}"
    print("✓ test_select_goal_prefers_free_front")


def test_select_goal_none_when_full():
    grid = np.ones((10, 10), dtype=np.uint8)
    assert select_goal(grid) is None
    print("✓ test_select_goal_none_when_full")


def test_astar_basic_and_wall():
    grid = np.zeros((10, 10), dtype=np.uint8)
    path = astar(grid, (9, 5), (0, 5))
    assert path and path[0] == (9, 5) and path[-1] == (0, 5)
    # 一堵横墙留一个缺口
    grid[5, :] = 1
    grid[5, 2] = 0
    path2 = astar(grid, (9, 5), (0, 5))
    assert path2 is not None, "有缺口应能绕行"
    assert (5, 2) in path2, "必须穿过缺口"
    # 封死缺口 -> 不可达
    grid[5, 2] = 1
    assert astar(grid, (9, 5), (0, 5)) is None
    print("✓ test_astar_basic_and_wall")


def test_astar_diag_no_corner_cut():
    # 两个障碍构成对角夹角，8邻域不应擦角穿过
    grid = np.zeros((3, 3), dtype=np.uint8)
    grid[0, 1] = 1
    grid[1, 0] = 1
    path = astar(grid, (0, 0), (1, 1), allow_diag=True)
    # (0,0)->(1,1) 直接对角被夹角封锁；此处唯一对角被禁 => 需绕行或不可达
    if path is not None:
        assert (0, 0) in path and (1, 1) in path
        # 不能是长度2的直接对角
        assert len(path) > 2, path
    print("✓ test_astar_diag_no_corner_cut")


def test_nearest_free_snaps():
    grid = np.zeros((10, 10), dtype=np.uint8)
    grid[5, 5] = 1
    snapped = nearest_free(grid, (5, 5))
    assert snapped != (5, 5) and grid[snapped] == 0
    print("✓ test_nearest_free_snaps")


def test_end_to_end_demo():
    """模拟 detect_to_path 的核心链路：框->栅格->选终点->A*。"""
    cam = CameraModel(height_m=10.0, hfov_deg=66.0, img_w=640, img_h=480)
    boxes = [
        (192, 144, 352, 264),
        (64, 288, 224, 408),
        (448, 72, 576, 216),
    ]
    gmap = build_grid(boxes, cam, cell_size_m=0.20, inflate_m=0.30)
    grid = gmap.grid
    H, W = grid.shape
    start = (H - 2, W // 2)
    goal = select_goal(grid, window=3, front_fraction=0.5)
    assert goal is not None
    path, s, g = plan(grid, start, goal, allow_diag=True)
    assert path is not None, "demo 场景应存在可行路径"
    assert path[0] == s and path[-1] == g
    print(f"✓ test_end_to_end_demo (grid={grid.shape}, path_len={len(path)})")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n全部 {len(tests)} 项测试通过 ✅")


if __name__ == "__main__":
    main()
