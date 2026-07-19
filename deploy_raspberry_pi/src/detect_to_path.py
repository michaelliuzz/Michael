#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
端到端流水线：捕获/图像 -> YOLO 检测 -> GSD 栅格地图 -> A* 路径规划

把原本三个孤立脚本（修改后的yolo / astar / detect_camera_cli）串成
一个闭环，实现申报书研究内容③的完整链路：
  检测框 --GSD投影--> 栅格 --膨胀--> 自动选终点 --A*--> 可通行路径

支持三种输入来源，便于在无摄像头/无模型的开发机上验证几何与规划逻辑：
  --picamera / (默认USB)  : 真机摄像头
  --image path.jpg        : 离线单帧图片
  --demo                  : 合成障碍框，纯逻辑自检（无需模型/相机）

产物：grid.npy（占据栅格）、path.npy（路径 Nx2）、终端 ASCII 可视化。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# 兼容"脚本直接运行"与"作为包导入"
try:
    from .grid_map import Box, CameraModel, build_grid, select_goal, to_ascii
    from .astar_planner import plan
except ImportError:  # 直接 python detect_to_path.py 运行时
    from grid_map import Box, CameraModel, build_grid, select_goal, to_ascii
    from astar_planner import plan


# ---------------------------------------------------------------------------
# 输入来源
# ---------------------------------------------------------------------------
def grab_frame(use_picamera: bool, image_path: Optional[str]) -> "np.ndarray":
    """获取一帧 BGR 图像（来自图片文件或摄像头）。"""
    import cv2

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            raise SystemExit(f"[✗] 无法读取图片: {image_path}")
        return frame

    if use_picamera:
        from picamera2 import Picamera2  # type: ignore
        import time

        picam2 = Picamera2()
        cfg = picam2.create_preview_configuration(
            main={"format": "RGB888", "size": (640, 480)}
        )
        picam2.configure(cfg)
        picam2.start()
        time.sleep(1)
        rgb = picam2.capture_array()
        picam2.stop()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("[✗] 无法打开摄像头")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("[✗] 摄像头读取失败")
    return frame


def detect_boxes(
    frame: "np.ndarray",
    model_path: str,
    obstacle_classes: Optional[List[str]],
    conf: float,
) -> List[Box]:
    """运行 YOLO，返回属于障碍类别的检测框像素坐标列表。

    model_path 可为 .pt / .onnx / ncnn 目录（Ultralytics 原生支持）。
    obstacle_classes 为 None 时把所有检测目标都视为障碍。
    """
    from ultralytics import YOLO

    model = YOLO(model_path)
    result = model(frame, conf=conf, verbose=False)[0]
    boxes: List[Box] = []
    for b in result.boxes:
        cls_name = model.names[int(b.cls[0])]
        if obstacle_classes and cls_name not in obstacle_classes:
            continue
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
        boxes.append((x1, y1, x2, y2))
    return boxes


def synth_boxes(img_w: int, img_h: int) -> List[Box]:
    """合成一组障碍框用于 --demo 自检（无需模型/相机）。"""
    return [
        (0.30 * img_w, 0.30 * img_h, 0.55 * img_w, 0.55 * img_h),   # 中部障碍
        (0.10 * img_w, 0.60 * img_h, 0.25 * img_h + 0.10 * img_w, 0.85 * img_h),
        (0.70 * img_w, 0.15 * img_h, 0.90 * img_w, 0.45 * img_h),
    ]


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> int:
    img_w, img_h = args.img_w, args.img_h

    # 1) 取框
    if args.demo:
        boxes = synth_boxes(img_w, img_h)
        print(f"[demo] 合成障碍框 {len(boxes)} 个（无需相机/模型）")
    else:
        frame = grab_frame(args.picamera, args.image)
        img_h, img_w = frame.shape[:2]
        classes = None if args.all_classes else args.classes
        boxes = detect_boxes(frame, args.model, classes, args.conf)
        print(f"[检测] 图像 {img_w}x{img_h}，障碍框 {len(boxes)} 个")

    # 2) GSD 投影 -> 栅格 -> 膨胀
    cam = CameraModel(
        height_m=args.height,
        hfov_deg=args.hfov,
        img_w=img_w,
        img_h=img_h,
        vfov_deg=args.vfov,
    )
    gsd_x, gsd_y = cam.gsd()
    foot_w, foot_h = cam.footprint_m()
    gmap = build_grid(boxes, cam, cell_size_m=args.cell, inflate_m=args.inflate)
    grid = gmap.grid
    print(
        f"[栅格] {grid.shape[0]}x{grid.shape[1]} 格 | GSD≈{gsd_x:.3f}m/px | "
        f"地面足迹≈{foot_w:.1f}x{foot_h:.1f}m | 分辨率 {args.cell}m/格"
    )

    # 3) 起点：小车在地图底部中央（世界系正下方近处）；终点：自动选取
    H, W = grid.shape
    start = (H - 2, W // 2)
    goal = select_goal(grid, window=args.goal_window, front_fraction=args.front)
    if goal is None:
        print("[✗] 前方无可通行块，无法选取终点")
        np.save(args.grid_out, grid)
        return 2
    print(f"[目标] 自动终点 = {goal}")

    # 4) A* 规划
    path, s, g = plan(grid, start, goal, allow_diag=args.diag)

    np.save(args.grid_out, grid)
    print(f"[保存] 栅格 -> {args.grid_out}")

    if path is None:
        print("[✗] 无可行路径（可能被障碍完全阻断）")
        print(to_ascii(grid, start=s, goal=g))
        return 3

    arr = np.array(path, dtype=np.int32)
    np.save(args.path_out, arr)
    print(f"[✓] 找到路径，长度 {len(path)} -> {args.path_out}")
    print(to_ascii(grid, path=path, start=s, goal=g))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="检测->栅格->A* 端到端流水线",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # 输入来源
    p.add_argument("--model", default="best_person.pt", help="YOLO 权重 (.pt/.onnx/ncnn)")
    p.add_argument("--image", default=None, help="离线单帧图片路径")
    p.add_argument("--picamera", action="store_true", help="使用树莓派 CSI 摄像头")
    p.add_argument("--demo", action="store_true", help="合成障碍框自检(无需相机/模型)")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--classes", nargs="*", default=["person", "rock", "tree"],
                   help="视为障碍的类别名")
    p.add_argument("--all-classes", action="store_true", help="所有检测目标都视为障碍")
    # 相机/无人机几何（GSD）
    p.add_argument("--height", type=float, default=10.0, help="无人机离地高度(米)")
    p.add_argument("--hfov", type=float, default=66.0, help="相机水平视场角(度)")
    p.add_argument("--vfov", type=float, default=None, help="相机垂直视场角(度,可选)")
    p.add_argument("--img-w", type=int, default=640, help="demo 模式图像宽(像素)")
    p.add_argument("--img-h", type=int, default=480, help="demo 模式图像高(像素)")
    # 栅格/规划
    p.add_argument("--cell", type=float, default=0.20, help="栅格分辨率(米/格)")
    p.add_argument("--inflate", type=float, default=0.30, help="障碍膨胀半径(米)")
    p.add_argument("--goal-window", type=int, default=3, help="自动选终点扫描窗口(格)")
    p.add_argument("--front", type=float, default=0.5, help="参与选终点的前方行占比")
    p.add_argument("--diag", action="store_true", help="A* 允许 8 邻域对角移动")
    # 产物
    p.add_argument("--grid-out", default="grid.npy")
    p.add_argument("--path-out", default="path.npy")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
