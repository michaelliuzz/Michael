"""
各功能模块的可视化与仿真动画（需要 matplotlib；导出 mp4 需要 ffmpeg，导出 gif 只需 pillow）

用法（在 deploy_raspberry_pi 目录下）:
    python3 -m rescue.viz all --out viz                 # 生成下列全部图片与动画
    python3 -m rescue.viz mapping --out viz             # 航拍检测框 -> 地面投影 -> 栅格地图
    python3 -m rescue.viz astar --out viz               # A* 路径：8 邻域/4 邻域、航点精简
    python3 -m rescue.viz gap --out viz                 # 激光雷达缝隙检测：俯视图 + 距离曲线 d(β)
    python3 -m rescue.viz person --out viz              # 车载相机人体测距测向与多帧确认
    python3 -m rescue.viz control --out viz             # 路径跟踪与速度/角速度/左右履带指令
    python3 -m rescue.viz comm --out viz                # 栅格地图压缩编码前后对比
    python3 -m rescue.viz baro --out viz                # 气压测高仿真（按 BMP388 手册精度加噪声）
    python3 -m rescue.viz animate --policy coop --out viz/sim_coop.mp4   # 搜救过程动画
    python3 -m rescue.viz animate --policy ground_search --out viz/sim_search.gif

所有图均来自仿真，不是实地测试数据。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .comm import decode_map, encode_map
from .control import DriveConfig, HeadingController
from .geo_mapping import (DronePose, GridMap, altitude_from_pressure, fill_polygon, ground_to_pixels,
                          pixels_to_ground, select_goal)
from .lidar_gap import GapConfig, plan_local
from .person_locate import CarCamera, PersonTracker, car_to_world, locate_person
from .planner import astar, path_length_cells, path_to_world, simplify_path
from .sim import lidar_scan, prepare_trial, project_person_to_car_camera, run_mission

COLORS = {"debris": "#8c6d46", "stone": "#7f7f7f", "wood": "#4f7f3a", "person": "#d62728"}
NAVY, BLUE, ORANGE, GREEN, RED = "#1B3A6B", "#2F6DB5", "#E07B24", "#2E8B57", "#C0392B"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
                                       "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _extent(gm):
    return [gm.x_min, gm.x_min + gm.shape[1] * gm.resolution,
            gm.y_max - gm.shape[0] * gm.resolution, gm.y_max]


def _save(fig, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, dpi=150, bbox_inches="tight")
    print(f"[✓] {out / name}")


def _record(scene, aerial, policy, seed):
    frames = []
    res = run_mission(scene, policy, np.random.default_rng(seed + 1000), aerial,
                      on_step=lambda st: frames.append(st | {"map_grid": st["map"].grid.copy()}))
    return res, frames


# ---------------------------------------------------------------------------
# 研究内容(3)：检测框 -> 地面投影 -> 栅格地图
# ---------------------------------------------------------------------------

def fig_mapping(seed, out):
    plt = _plt()
    from matplotlib.patches import Polygon, Rectangle
    scene, dets, aerial, map_ms = prepare_trial(seed)
    gm = aerial.gridmap
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))

    ax = axes[0, 0]
    for o in scene.obstacles:
        uv = ground_to_pixels(o.poly, scene.cam, scene.true_pose)
        ax.add_patch(Polygon(uv, fc=COLORS[o.cls_name], alpha=1.0 if o.visible_from_air else 0.25))
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=COLORS.get(d.cls_name, BLUE), lw=1.2))
    ax.set_xlim(0, scene.cam.width)
    ax.set_ylim(scene.cam.height, 0)
    ax.set_aspect("equal")
    ax.set_title(f"(a) 无人机图像与 YOLO 检测框（{len(dets)} 个，半透明为空中不可见）")

    ax = axes[0, 1]
    for o in scene.obstacles:
        ax.add_patch(Polygon(o.poly, fc="#d9d9d9", ec="#999999", lw=0.6))
    level = DronePose(altitude_m=scene.measured_pose.altitude_m)
    for k, d in enumerate(dets):
        x1, y1, x2, y2 = d.xyxy
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        for pose, c, ls, lab in ((scene.measured_pose, BLUE, "-", "有姿态补偿"), (level, RED, "--", "无姿态补偿")):
            g = pixels_to_ground(corners, scene.cam, pose)
            ax.add_patch(Polygon(g, fill=False, ec=c, ls=ls, lw=1.0, label=lab if k == 0 else None))
    ax.plot(*scene.person_xy, "*", c=COLORS["person"], ms=14, label="受困人员（真实）")
    ax.set_xlim(*_extent(gm)[:2])
    ax.set_ylim(*_extent(gm)[2:])
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"(b) 检测框地面投影（灰色为真实障碍；俯仰 {scene.true_pose.pitch_deg:.1f}°，"
                 f"横滚 {scene.true_pose.roll_deg:.1f}°）")

    ax = axes[1, 0]
    ax.imshow(gm.raw, cmap="Greys", extent=_extent(gm), vmin=0, vmax=1)
    ax.set_title(f"(c) 栅格化结果（{gm.resolution} m/格，{gm.shape[0]}×{gm.shape[1]}）")

    ax = axes[1, 1]
    ax.imshow(gm.grid.astype(float) * 0.45 + gm.raw.astype(float) * 0.55, cmap="Greys", extent=_extent(gm), vmin=0, vmax=1)
    goal_cell, _ = select_goal(gm, aerial.persons_xy)
    ax.plot(*gm.cell_to_world(*goal_cell), "*", c=COLORS["person"], ms=15, label="终点（航拍人员坐标）")
    ax.plot(*scene.start_pose[:2], "s", c=GREEN, ms=9, label="小车起点")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(f"(d) 曼哈顿膨胀 0.3 m（浅灰）与终点选取，建图耗时 {map_ms:.1f} ms")
    for a in axes[0, 1], axes[1, 0], axes[1, 1]:
        a.set_xlabel("x / m")
        a.set_ylabel("y / m")
    fig.suptitle("识别结果栅格化：像素检测框 → 地面坐标 → 栅格地图", fontsize=14)
    fig.tight_layout()
    _save(fig, out, "viz_mapping.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 研究内容(3)：A* 全局路径
# ---------------------------------------------------------------------------

def fig_astar(seed, out):
    plt = _plt()
    scene, _, aerial, _ = prepare_trial(seed)
    gm = aerial.gridmap
    s = gm.world_to_cell(*scene.start_pose[:2])
    g, _ = select_goal(gm, aerial.persons_xy)
    p8 = astar(gm.grid, s, g, allow_diag=True)
    p4 = astar(gm.grid, s, g, allow_diag=False)
    wps = path_to_world(gm, simplify_path(gm.grid, p8))
    show = gm.grid.astype(float) * 0.45 + gm.raw.astype(float) * 0.55
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax in axes:
        ax.imshow(show, cmap="Greys", extent=_extent(gm), vmin=0, vmax=1)
        ax.plot(*gm.cell_to_world(*s), "s", c=GREEN, ms=10, label="起点")
        ax.plot(*gm.cell_to_world(*g), "*", c=COLORS["person"], ms=16, label="终点")
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
    w8 = np.array(path_to_world(gm, p8))
    ax = axes[0]
    ax.plot(w8[:, 0], w8[:, 1], c=ORANGE, lw=1.5, label=f"A* 原始路径（{len(p8)} 格）")
    w = np.array(wps)
    ax.plot(w[:, 0], w[:, 1], "-o", c=BLUE, lw=2.2, ms=6, label=f"视线法精简后航点（{len(wps)} 个）")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("(a) 8 邻域 A* 路径与航点精简")
    ax = axes[1]
    w4 = np.array(path_to_world(gm, p4))
    ax.plot(w4[:, 0], w4[:, 1], c=RED, lw=1.6, label=f"4 邻域：{path_length_cells(p4) * gm.resolution:.1f} m")
    ax.plot(w8[:, 0], w8[:, 1], c=BLUE, lw=1.6, label=f"8 邻域：{path_length_cells(p8) * gm.resolution:.1f} m")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("(b) 4 邻域与 8 邻域路径长度对比")
    fig.suptitle("全局路径规划（深灰为障碍，浅灰为膨胀区域）", fontsize=14)
    fig.tight_layout()
    _save(fig, out, "viz_astar.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 研究内容(1)：激光雷达缝隙检测
# ---------------------------------------------------------------------------

def gap_demo_scan(seed=1):
    """构造典型局部场景：车头正前方 0.7 m 有障碍，左前方缝隙过窄，右前方缝隙可通行"""
    gm = GridMap(np.zeros((175, 200), np.uint8), 0.02, x_min=-1.0, y_max=1.75)
    boxes = [(0.70, 1.00, -0.05, 0.19),   # 正前方
             (0.68, 0.95, 0.40, 0.62),    # 左前方（与正前方障碍之间缝隙过窄）
             (0.55, 0.85, -1.05, -0.72),  # 右前方
             (2.20, 2.40, -1.30, 1.40),   # 远处墙体
             (-0.60, -0.40, 0.55, 1.20)]  # 左后方
    for x0, x1, y0, y1 in boxes:
        fill_polygon(gm, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
    rel, ranges = lidar_scan(gm, (0.0, 0.0, 0.0), np.random.default_rng(seed), step=0.01)
    return rel, ranges


def fig_gap(seed, out):
    plt = _plt()
    from matplotlib.patches import Circle, Polygon, Wedge
    cfg = GapConfig(car_width=0.25, margin=0.08)
    rel_deg, rng_ = gap_demo_scan(seed)
    lp = plan_local(rel_deg, rng_, 0.0, cfg)
    st = {"t": 0.0, "target": 0.0}
    rel = np.radians(rel_deg)
    fin = np.isfinite(rng_)
    to_plot = lambda b_deg, r: (-r * np.sin(np.radians(b_deg)), r * np.cos(np.radians(b_deg)))  # 车头朝上

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for gp in lp.gaps:
        ax.add_patch(Wedge((0, 0), min(gp.mean_dist, 3.2), 90 + gp.start_deg, 90 + gp.end_deg,
                           fc=GREEN if gp.passable else RED, alpha=0.18, ec="none"))
    ax.add_patch(Circle((0, 0), cfg.safe_dist, fill=False, ls="--", ec="#666666"))
    for b in (-cfg.fov_deg / 2, cfg.fov_deg / 2):
        ax.plot(*zip((0, 0), to_plot(b, 3.2)), c="#999999", lw=0.8, ls=":")
    xs, ys = -rng_[fin] * np.sin(rel[fin]), rng_[fin] * np.cos(rel[fin])
    near = rng_[fin] < cfg.safe_dist
    ax.scatter(xs[~near], ys[~near], s=6, c="#555555", label="激光雷达扫描点")
    ax.scatter(xs[near], ys[near], s=10, c=RED, label="1 m 内障碍边界点")
    tgt = math.degrees(st["target"])
    ax.annotate("", to_plot(tgt, 2.6), (0, 0), arrowprops=dict(arrowstyle="->", color="#888888", lw=1.6, ls="--"))
    ax.annotate("", to_plot(lp.heading_deg, 2.6), (0, 0), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.4))
    hw, L, h = cfg.required_width / 2, 1.0, math.radians(lp.heading_deg)
    corr = np.array([[0, -hw], [L, -hw], [L, hw], [0, hw]])
    c, s = math.cos(h), math.sin(h)
    xy = corr @ np.array([[c, s], [-s, c]])
    ax.add_patch(Polygon(np.column_stack([-xy[:, 1], xy[:, 0]]), fc=BLUE, alpha=0.15, ec=BLUE, lw=1, label="车体走廊"))
    ax.add_patch(Polygon([[-0.125, -0.17], [0.125, -0.17], [0.125, 0.17], [-0.125, 0.17]], fc=NAVY))
    ax.plot([], [], c="#888888", ls="--", label=f"全局目标方位 {tgt:+.0f}°")
    ax.plot([], [], c=BLUE, lw=2.4, label=f"期望航向 {lp.heading_deg:+.0f}°")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-1.0, 3.5)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_xlabel("左右 / m")
    ax.set_ylabel("前方 / m")
    ax.set_title("(a) 小车视角俯视图（绿色为可通行缝隙，红色为过窄缝隙）")

    ax = axes[1]
    centers, prof = lp.profile
    ax.plot(centers, np.minimum(prof, 4.0), c=NAVY, lw=1.6, label="距离曲线 d(β)")
    ax.axhline(cfg.safe_dist, c=RED, ls="--", lw=1.2, label="安全距离 1 m")
    for gp in lp.gaps:
        ax.axvspan(gp.start_deg, gp.end_deg, color=GREEN if gp.passable else RED, alpha=0.15)
        ax.text(gp.center_deg, 3.75, f"{gp.width_m:.2f} m\n{'可通行' if gp.passable else '过窄'}",
                ha="center", va="top", fontsize=8.5, color=GREEN if gp.passable else RED)
    ax.axvline(tgt, c="#888888", ls="--", lw=1.2, label="全局目标方位")
    ax.axvline(lp.heading_deg, c=BLUE, lw=2, label="期望航向")
    ax.set_xlim(cfg.fov_deg / 2, -cfg.fov_deg / 2)
    ax.set_ylim(0, 4.0)
    ax.set_xlabel("扫描角 β / °（左为正）")
    ax.set_ylabel("最近距离 / m（大于 4 m 截断显示）")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title(f"(b) 前方 120° 距离曲线与缝隙（所需宽度 {cfg.required_width:.2f} m）")
    fig.suptitle("基于二维激光雷达缝隙检测的局部避障（典型场景：全局目标方向正前方 0.7 m 处被障碍阻挡）", fontsize=13)
    fig.tight_layout()
    _save(fig, out, "viz_gap.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 研究内容(4)：车载相机人体定位
# ---------------------------------------------------------------------------

def fig_person(seed, out):
    plt = _plt()
    from matplotlib.patches import Rectangle, Wedge
    cam = CarCamera()
    rng = np.random.default_rng(seed)
    cases = [("站立", SimpleNamespace(person_xy=(5.0, 0.9), person_lying=False), NAVY),
             ("倒地", SimpleNamespace(person_xy=(3.6, -1.1), person_lying=True), ORANGE)]
    pose = (0.0, 0.0, 0.0)
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05])
    horizon = cam.height / 2 - cam.f * math.tan(math.radians(cam.pitch_down_deg))
    top = fig.add_subplot(gs[0, 2])
    half = cam.hfov_deg / 2
    top.add_patch(Wedge((0, 0), 7.5, 90 - half, 90 + half, fc="#EAF1FA", ec="#9DB9D9"))
    top.add_patch(Rectangle((-0.125, -0.35), 0.25, 0.35, fc=NAVY))
    for k, (name, sc, col) in enumerate(cases):
        bb = project_person_to_car_camera(sc, pose, cam)
        ax = fig.add_subplot(gs[0, k])
        ax.add_patch(Rectangle((0, 0), cam.width, horizon, fc="#E8EEF5"))
        ax.add_patch(Rectangle((0, horizon), cam.width, cam.height - horizon, fc="#D8CDBE"))
        x1, y1, x2, y2 = bb
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=RED, lw=2))
        ax.plot((x1 + x2) / 2, y2, "o", c=RED, ms=7)
        ax.annotate("接地点", ((x1 + x2) / 2, y2), (12, 14), textcoords="offset points", color=RED, fontsize=10)
        fix = locate_person(bb, cam)
        ax.set_xlim(0, cam.width)
        ax.set_ylim(cam.height, 0)
        ax.set_aspect("equal")
        note = "（倒地时接地点为身体近端）" if sc.person_lying else ""
        ax.set_title(f"({'ab'[k]}) 车载相机画面：{name}人员{note}\n测得前方 {fix.x:.2f} m、左侧 {fix.y:+.2f} m"
                     f"（人体中心 {sc.person_xy[0]:.2f} m、{sc.person_xy[1]:+.2f} m）", fontsize=10.5)
        tracker, pts = PersonTracker(), []
        for i in range(8):
            noisy = np.asarray(bb) + rng.normal(0, 2.0, 4)
            f = locate_person(noisy, cam)
            xy = car_to_world(f, pose)
            pts.append(xy)
            tracker.add(xy, i * 0.3)
        conf = tracker.confirmed(8 * 0.3)
        P = np.array(pts)
        top.scatter(-P[:, 1], P[:, 0], s=14, c=col, alpha=0.6, label=f"{name}：单帧定位（8 帧）")
        top.plot(-sc.person_xy[1], sc.person_xy[0], "*", c=col, ms=16, mec="k", label=f"{name}：人体中心")
        if conf:
            err = math.hypot(conf[0] - sc.person_xy[0], conf[1] - sc.person_xy[1])
            top.plot(-conf[1], conf[0], "x", c=RED, ms=11, mew=2.5)
            top.annotate(f"多帧确认\n误差 {err:.2f} m", (-conf[1], conf[0]), (10, -28), textcoords="offset points", fontsize=9)
    top.set_xlim(-4, 4)
    top.set_ylim(-0.6, 7.5)
    top.set_aspect("equal")
    top.set_xlabel("左右 / m（左为负）")
    top.set_ylabel("前方 / m")
    top.legend(loc="upper left", fontsize=8)
    top.set_title("(c) 小车坐标系下的定位结果（浅蓝为相机视场）", fontsize=10.5)
    fig.suptitle(f"人体识别与定位：检测框底边接地点 + 相机安装高度 {cam.mount_height_m} m、俯角 {cam.pitch_down_deg}°", fontsize=14)
    fig.tight_layout()
    _save(fig, out, "viz_person.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 运动控制
# ---------------------------------------------------------------------------

def fig_control(seed, out):
    plt = _plt()
    scene, _, aerial, _ = prepare_trial(seed)
    res, frames = _record(scene, aerial, "coop", seed)
    ctrl = HeadingController(DriveConfig())
    t = np.array([f["t"] for f in frames])
    v = np.array([f["v"] for f in frames])
    w = np.array([f["w"] for f in frames])
    err = np.array([math.degrees(f["target"]) for f in frames])
    wheels = np.array([ctrl.to_wheels(a, b) for a, b in zip(v, w)])
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3])
    ax = fig.add_subplot(gs[:, 0])
    tr = scene.truth
    ax.imshow(tr.grid, cmap="Greys", extent=_extent(tr), vmin=0, vmax=1.6)
    p = np.array(res.initial_path)
    ax.plot(p[:, 0], p[:, 1], "--o", c=ORANGE, ms=4, lw=1.5, label="A* 初始航点")
    traj = np.array(res.trajectory)
    ax.plot(traj[:, 0], traj[:, 1], c=BLUE, lw=2, label=f"实际轨迹（{res.distance_m:.1f} m，{res.time_s:.0f} s）")
    ax.plot(*scene.person_xy, "*", c=COLORS["person"], ms=15, label="受困人员")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("x / m")
    ax.set_ylabel("y / m")
    ax.set_title("(a) 纯追踪 + 缝隙避障的行驶轨迹")
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, err, c="#888888", lw=1, label="前视点方位（航向误差）/ °")
    ax.set_ylabel("航向误差 / °")
    ax2 = ax.twinx()
    ax2.plot(t, v, c=BLUE, lw=1.6, label="线速度 v / (m/s)")
    ax2.plot(t, w, c=ORANGE, lw=1.2, label="角速度 ω / (rad/s)")
    ax2.set_ylabel("v / ω")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8.5)
    ax.set_title("(b) PID 航向控制输出")
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, wheels[:, 0], c=NAVY, lw=1.4, label="左履带")
    ax.plot(t, wheels[:, 1], c=GREEN, lw=1.4, label="右履带")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("PWM 占空比（归一化）")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.set_title("(c) 差速分配后的左右履带指令")
    fig.suptitle("运动控制：纯追踪 → PID 航向控制 → 差速分配", fontsize=14)
    fig.tight_layout()
    _save(fig, out, "viz_control.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 空地通信
# ---------------------------------------------------------------------------

def fig_comm(seed, out):
    plt = _plt()
    _, _, aerial, _ = prepare_trial(seed)
    gm = aerial.gridmap
    payload = encode_map(gm, aerial.persons_xy)
    gm2, persons, _ = decode_map(payload)
    same = bool(np.array_equal(gm.grid, gm2.grid))
    raw = gm.grid.size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, g, title in ((axes[0], gm.grid, "(a) 空中端发送的栅格地图"), (axes[1], gm2.grid, "(b) 地面端解码得到的栅格地图")):
        ax.imshow(g, cmap="Greys", extent=_extent(gm), vmin=0, vmax=1)
        for px, py, *_ in (aerial.persons_xy if g is gm.grid else persons):
            ax.plot(px, py, "*", c=COLORS["person"], ms=13)
        ax.set_title(title)
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
    fig.suptitle(f"MQTT 地图消息：{gm.shape[0]}×{gm.shape[1]} 栅格，逐格存储 {raw / 1024:.1f} KB → "
                 f"按位压缩 + zlib + Base64 后 {len(payload) / 1024:.2f} KB；解码结果{'完全一致' if same else '不一致'}",
                 fontsize=12.5)
    fig.tight_layout()
    _save(fig, out, "viz_comm.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 气压测高
# ---------------------------------------------------------------------------

def fig_baro(seed, out, noise_pa=3.6):
    plt = _plt()
    rng = np.random.default_rng(seed)
    t = np.arange(0, 120, 0.1)
    h = np.interp(t, [0, 10, 35, 95, 115, 120], [0, 0, 25, 25, 0, 0]) + np.where((t > 35) & (t < 95), 0.4 * np.sin(t / 3), 0)
    p0, temp = 1013.25, 20.0
    p = p0 * np.exp(-h / (29.271 * (temp + 273.15)))
    meas = p + rng.normal(0, noise_pa / 100, len(t))
    ground = meas[:50].mean()
    alt = np.array([altitude_from_pressure(x, ground, temp) for x in meas])
    k = 10
    smooth = np.convolve(alt, np.ones(k) / k, mode="same")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(t, alt, c="#9DB9D9", lw=0.8, label="单次读数换算高度")
    ax.plot(t, smooth, c=BLUE, lw=1.8, label="滑动平均（1 s）")
    ax.plot(t, h, c="k", ls="--", lw=1.2, label="真实飞行高度")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("相对高度 / m")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("(a) 起飞 → 25 m 悬停 → 降落")
    ax = axes[1]
    e1, e2 = alt - h, smooth - h
    ax.hist(e1, bins=40, color="#9DB9D9", alpha=0.9, label=f"单次读数 RMS {np.sqrt(np.mean(e1 ** 2)):.2f} m")
    ax.hist(e2[k:-k], bins=40, color=BLUE, alpha=0.7, label=f"滑动平均 RMS {np.sqrt(np.mean(e2[k:-k] ** 2)):.2f} m")
    ax.set_xlabel("高度误差 / m")
    ax.legend(fontsize=9)
    ax.set_title("(b) 测高误差分布")
    fig.suptitle(f"气压测高仿真：气压噪声标准差 {noise_pa} Pa（按 BMP388 相对精度 ±8 Pa 取 2 倍标准差），起飞前 5 s 采样为地面参考", fontsize=12.5)
    fig.tight_layout()
    _save(fig, out, "viz_baro.png")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 搜救过程动画
# ---------------------------------------------------------------------------

POLICY_NAME = {"coop": "空地协同", "ground_known": "仅地面（已知坐标）", "ground_search": "仅地面（往返搜索）"}


def animate(seed, policy, out_file, every=None, fps=10):
    plt = _plt()
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle, Polygon
    scene, _, aerial, _ = prepare_trial(seed)
    res, frames = _record(scene, aerial, policy, seed)
    every = every or max(1, len(frames) // 150)
    frames = frames[::every] + [frames[-1]] * fps * 2
    speed = every * 0.1 * fps

    tr = scene.truth
    gm0 = frames[0]["map"]
    init = aerial.gridmap.grid if policy == "coop" else np.zeros_like(aerial.gridmap.grid)

    def map_rgb(cur):
        rgb = np.ones(cur.shape + (3,))
        rgb[(cur == 1) & (init == 1)] = (0.40, 0.40, 0.40)   # 航拍障碍
        rgb[(cur == 1) & (init == 0)] = (0.88, 0.48, 0.14)   # 雷达新增
        rgb[(cur == 0) & (init == 1)] = (0.74, 0.85, 0.96)   # 雷达清除的误检
        return rgb

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))
    ax1.imshow(tr.grid, cmap="Greys", extent=_extent(tr), vmin=0, vmax=1.6)
    for o in scene.obstacles:
        if not o.visible_from_air:
            ax1.add_patch(Polygon(o.poly, fill=False, ec=ORANGE, lw=1.4))
    ax1.plot(*scene.person_xy, "*", c=COLORS["person"], ms=15)
    img = ax2.imshow(map_rgb(frames[0]["map_grid"]), extent=_extent(gm0), interpolation="nearest")
    ax2.plot(*scene.person_xy, "*", c=COLORS["person"], ms=13, alpha=0.35)
    rays = LineCollection([], colors="#E07B24", linewidths=0.5, alpha=0.5)
    ax1.add_collection(rays)
    art = {}
    for k, ax in (("1", ax1), ("2", ax2)):
        art["traj" + k], = ax.plot([], [], c=BLUE, lw=2)
        art["path" + k], = ax.plot([], [], c=GREEN, lw=1.4, ls="--")
        art["car" + k] = ax.add_patch(Circle((0, 0), 0.55, fc=NAVY, ec="w", lw=1.5, zorder=5))
        art["head" + k], = ax.plot([], [], c=NAVY, lw=2.5, zorder=6)
        ax.set_xlim(*_extent(tr)[:2])
        ax.set_ylim(*_extent(tr)[2:])
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
    ax1.set_title("真实场景（橙框为空中不可见障碍，橙线为激光雷达）", fontsize=10.5)
    ax2.set_title("小车地图（灰：航拍障碍　橙：雷达新增　浅蓝：雷达清除；绿线为当前路径）", fontsize=10.5)
    title = fig.suptitle("", fontsize=13)
    xs, ys = [], []

    def status(st, last):
        lp = st["local"]
        if last:
            return "已确认受困人员，搜救完成" if res.success else "超时未完成"
        if st["fixes"] >= 1:
            return f"车载相机已发现人员（确认 {min(st['fixes'], 3)}/3）"
        if lp.heading_deg is None:
            return "前方无可通行缝隙，执行脱困"
        if abs(lp.heading_deg - math.degrees(st["target"])) > 5:
            return "缝隙检测修正航向，绕行障碍"
        return "沿 A* 路径行驶" if st["homing"] else "往返搜索中"

    def update(i):
        st = frames[i]
        last = i >= len(frames) - fps * 2
        x, y, th = st["pose"]
        if not last:
            xs.append(x)
            ys.append(y)
        a = th + np.radians(st["rel"][::4])
        r = np.minimum(st["ranges"][::4], 8.0)
        rays.set_segments([[(x, y), (x + rr * math.cos(aa), y + rr * math.sin(aa))] for aa, rr in zip(a, r)])
        img.set_data(map_rgb(st["map_grid"]))
        path = st["path"]
        for k in "12":
            art["traj" + k].set_data(xs, ys)
            art["path" + k].set_data(path[:, 0], path[:, 1])
            art["car" + k].center = (x, y)
            art["head" + k].set_data([x, x + 1.2 * math.cos(th)], [y, y + 1.2 * math.sin(th)])
        title.set_text(f"{POLICY_NAME[policy]}　t = {st['t']:5.1f} s　{status(st, last)}　（{speed:.0f} 倍速）")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=fps) if out_file.suffix == ".gif" else FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(out_file, writer=writer, dpi=90 if out_file.suffix == ".gif" else 110)
    plt.close(fig)
    print(f"[✓] {out_file}（{POLICY_NAME[policy]}：{'成功' if res.success else '失败'}，"
          f"{res.time_s:.1f} s，{res.distance_m:.1f} m）")


FIGS = {"mapping": fig_mapping, "astar": fig_astar, "gap": fig_gap, "person": fig_person,
        "control": fig_control, "comm": fig_comm, "baro": fig_baro}


def main():
    ap = argparse.ArgumentParser(description="各功能模块可视化与仿真动画")
    ap.add_argument("what", choices=[*FIGS, "animate", "all"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="viz", help="输出目录（animate 时为输出文件 .mp4/.gif）")
    ap.add_argument("--policy", choices=list(POLICY_NAME), default="coop")
    ap.add_argument("--every", type=int, default=None, help="动画每隔几个仿真步取一帧（默认自动）")
    ap.add_argument("--gif", action="store_true", help="all 模式下动画导出为 gif（默认 mp4，需要 ffmpeg）")
    args = ap.parse_args()
    if args.what in FIGS:
        FIGS[args.what](args.seed, args.out)
    elif args.what == "animate":
        out = args.out if Path(args.out).suffix in (".mp4", ".gif") else str(Path(args.out) / f"sim_{args.policy}.mp4")
        animate(args.seed, args.policy, out, args.every)
    else:
        for fn in FIGS.values():
            fn(args.seed, args.out)
        ext = ".gif" if args.gif else ".mp4"
        for pol in ("coop", "ground_search"):
            animate(args.seed, pol, str(Path(args.out) / f"sim_{pol}{ext}"), args.every)


if __name__ == "__main__":
    main()
