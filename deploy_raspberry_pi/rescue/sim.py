"""
空地协同全流程仿真（无需无人机、小车实物）

仿真内容:
    1. 随机生成灾区场景：碎石/废墟块/木材/倒塌墙体（含 U 型死路），以及 1 名受困人员
       其中一部分障碍物从空中不可见（被遮挡或漏检），只能由地面激光雷达发现
    2. 无人机：按真实相机模型 + 带噪声的 IMU/气压计位姿，把障碍物投影成像素检测框，
       再调用 geo_mapping.build_grid_map —— 与真机代码完全相同的建图路径
    3. 小车：A* 全局规划 + 纯追踪 + 激光雷达缝隙检测避障 + PID 航向控制，
       边走边用雷达更新地图并重规划；车载相机发现人员后用 person_locate 定位
    4. 对比实验：
         空地协同        航拍地图 + 航拍人员坐标
         仅地面(已知坐标) 无航拍地图，雷达逐步建图，假设已知人员坐标（理想化对照）
         仅地面(搜索)     无航拍地图且不知道人员位置，割草机式往返搜索（实际情况）
       以及 IMU 姿态补偿对建图精度的影响

用法:
    python -m rescue.sim --seed 1 --figure docs/figures/sim_demo.png
    python -m rescue.sim --trials 50 --seed 0 --json results.json

注意：仿真结果仅用于验证算法逻辑与相对性能，不能等同于实地测试数据。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .control import DriveConfig, HeadingController, PurePursuit, recovery, wrap_angle
from .geo_mapping import (CameraModel, Detection, DronePose, GridMap, build_grid_map,
                          empty_map_for_footprint, fill_polygon, ground_to_pixels,
                          inflate_manhattan, nearest_free, select_goal)
from .lidar_gap import GapConfig, plan_local
from .local_map import LidarMapper
from .person_locate import CarCamera, car_to_world, locate_person
from .planner import astar, path_to_world, simplify_path

CAR_RADIUS = 0.17


# ---------------------------------------------------------------------------
# 场景
# ---------------------------------------------------------------------------

@dataclass
class Obstacle:
    cls_name: str
    poly: np.ndarray
    visible_from_air: bool = True


@dataclass
class Scene:
    obstacles: list
    person_xy: tuple
    person_lying: bool
    start_pose: tuple
    truth: GridMap
    cam: CameraModel
    true_pose: DronePose
    measured_pose: DronePose


def _rect(cx, cy, w, h, ang):
    c, s = math.cos(ang), math.sin(ang)
    pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return pts @ np.array([[c, s], [-s, c]]) + [cx, cy]


def _octagon(cx, cy, r):
    a = np.arange(8) * math.pi / 4 + math.pi / 8
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def random_scene(rng: np.random.Generator, trap: bool = True, hidden_ratio: float = 0.15,
                 n_debris: int = 14, n_stones: int = 22, n_wood: int = 6) -> Scene:
    cam = CameraModel()
    true_pose = DronePose(altitude_m=25.0, pitch_deg=float(rng.uniform(-3, 3)),
                          roll_deg=float(rng.uniform(-3, 3)))
    # IMU 姿态误差 ±0.5°，气压计高度误差 ±0.3 m
    measured = DronePose(altitude_m=true_pose.altitude_m + float(rng.normal(0, 0.3)),
                         pitch_deg=true_pose.pitch_deg + float(rng.normal(0, 0.5)),
                         roll_deg=true_pose.roll_deg + float(rng.normal(0, 0.5)))
    truth = empty_map_for_footprint(cam, true_pose, 0.05)
    H, W = truth.grid.shape
    x0, x1 = truth.x_min + 2.0, truth.x_min + W * 0.05 - 2.0
    y0, y1 = truth.y_max - H * 0.05 + 2.0, truth.y_max - 2.0

    start = (float(rng.uniform(-2, 2)), y0 + 0.5, math.pi / 2)
    person = (float(rng.uniform(x0 + 2, x1 - 2)), float(rng.uniform(y1 - 5, y1 - 1)))
    keep_clear = [(start[:2], 2.0), (person, 1.5)]

    def clear(cx, cy, r):
        return all(math.hypot(cx - px, cy - py) > r + rr for (px, py), rr in keep_clear)

    obs = []
    if trap:
        # U 型死路：开口朝向小车，位于起点与人员之间。从地面看是一条“通向目标的通道”
        mx = (start[0] + person[0]) / 2 + float(rng.uniform(-1, 1))
        my = (start[1] + person[1]) / 2 + float(rng.uniform(-1, 1))
        w, d = float(rng.uniform(4.5, 6.5)), float(rng.uniform(3.0, 4.0))
        obs.append(Obstacle("debris", _rect(mx, my + d / 2, w, 0.4, 0)))
        obs.append(Obstacle("debris", _rect(mx - w / 2, my, 0.4, d, 0)))
        obs.append(Obstacle("debris", _rect(mx + w / 2, my, 0.4, d, 0)))
        keep_clear.append(((mx, my), max(w, d) / 2 + 0.3))
    kinds = ["debris"] * n_debris + ["stone"] * n_stones + ["wood"] * n_wood
    for k in kinds:
        for _ in range(50):
            cx, cy = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if k == "debris":
                w, h = rng.uniform(0.6, 2.5), rng.uniform(0.4, 1.2)
                poly, r = _rect(cx, cy, w, h, rng.uniform(0, math.pi)), math.hypot(w, h) / 2
            elif k == "stone":
                r = rng.uniform(0.15, 0.5)
                poly = _octagon(cx, cy, r)
            else:
                w = rng.uniform(1.5, 3.5)
                poly, r = _rect(cx, cy, w, rng.uniform(0.15, 0.3), rng.uniform(0, math.pi)), w / 2
            if clear(cx, cy, r):
                obs.append(Obstacle(k, poly, bool(rng.random() > hidden_ratio)))
                break
    for o in obs:
        fill_polygon(truth, o.poly)
    lying = bool(rng.random() < 0.5)
    return Scene(obs, person, lying, start, truth, cam, true_pose, measured)


def scene_solvable(scene: Scene) -> bool:
    """在真实地图上（按车体半径膨胀）检查起点到人员附近是否存在通路"""
    gm = scene.truth
    H, W = gm.grid.shape
    g = gm.grid[:H - H % 4, :W - W % 4]
    coarse = g.reshape(g.shape[0] // 4, 4, -1, 4).max(axis=(1, 3))
    cgm = GridMap(inflate_manhattan(coarse, 1), 0.2, gm.x_min, gm.y_max)
    s = nearest_free(cgm.grid, cgm.world_to_cell(*scene.start_pose[:2]), 5)
    g = nearest_free(cgm.grid, cgm.world_to_cell(*scene.person_xy), 8)
    return s is not None and g is not None and astar(cgm.grid, s, g) is not None


# ---------------------------------------------------------------------------
# 传感器仿真
# ---------------------------------------------------------------------------

def simulate_drone_detections(scene: Scene, rng, recall: float = 0.92, px_noise: float = 2.0):
    """把场景投影到无人机图像中，生成像素检测框（模拟 YOLO 输出）"""
    cam, pose = scene.cam, scene.true_pose
    dets = []

    def box(poly, cls, conf):
        uv = ground_to_pixels(poly, cam, pose)
        x1, y1 = uv.min(0) + rng.normal(0, px_noise, 2)
        x2, y2 = uv.max(0) + rng.normal(0, px_noise, 2)
        x1, x2 = np.clip([x1, x2], 0, cam.width - 1)
        y1, y2 = np.clip([y1, y2], 0, cam.height - 1)
        if x2 > x1 and y2 > y1:
            dets.append(Detection(cls, conf, (float(x1), float(y1), float(x2), float(y2))))

    for o in scene.obstacles:
        if o.visible_from_air and rng.random() < recall:
            box(o.poly, o.cls_name, float(rng.uniform(0.4, 0.95)))
    px, py = scene.person_xy
    half = (0.85, 0.3) if scene.person_lying else (0.3, 0.3)
    box(_rect(px, py, half[0] * 2, half[1] * 2, 0), "person", 0.8)
    return dets


def lidar_scan(truth: GridMap, pose, rng, n_beams=360, max_range=8.0, step=0.04, noise=0.01):
    x, y, th = pose
    rel = np.linspace(-180, 180, n_beams, endpoint=False)
    a = th + np.radians(rel)
    r = np.arange(step, max_range, step)
    xs = x + np.cos(a)[:, None] * r
    ys = y + np.sin(a)[:, None] * r
    rows = np.floor((truth.y_max - ys) / truth.resolution).astype(int)
    cols = np.floor((xs - truth.x_min) / truth.resolution).astype(int)
    H, W = truth.grid.shape
    inb = (rows >= 0) & (rows < H) & (cols >= 0) & (cols < W)
    occ = np.zeros_like(inb)
    occ[inb] = truth.grid[rows[inb], cols[inb]] != 0
    hit = occ.any(axis=1)
    ranges = np.full(n_beams, np.inf)
    ranges[hit] = r[occ[hit].argmax(axis=1)] + rng.normal(0, noise, hit.sum())
    return rel, ranges


def collides(truth: GridMap, x, y, radius=CAR_RADIUS, heading=None) -> bool:
    """车体（圆盘）是否与障碍重叠；给定运动方向时只检查前半圆（向后/侧面贴靠不算碰撞）"""
    if heading is None:
        a = np.linspace(0, 2 * math.pi, 12, endpoint=False)
    else:
        a = heading + np.linspace(-math.pi / 2, math.pi / 2, 9)
    pts = [(x, y)] + [(x + radius * math.cos(t), y + radius * math.sin(t)) for t in a]
    for px, py in pts:
        r, c = truth.world_to_cell(px, py)
        if truth.in_bounds(r, c) and truth.grid[r, c]:
            return True
    return False


def project_person_to_car_camera(scene: Scene, pose, cam: CarCamera):
    """把人员（站立圆柱 / 倒地长方体）投影到车载相机图像，返回像素框或 None"""
    x, y, th = pose
    px, py = scene.person_xy
    c, s = math.cos(th), math.sin(th)
    dx, dy = px - x, py - y
    X, Y = c * dx + s * dy, -s * dx + c * dy   # 车体系
    hgt, hw, hl = (0.4, 0.3, 0.85) if scene.person_lying else (1.7, 0.25, 0.25)
    pts = np.array([[X + sx * hl, Y + sy * hw, z] for sx in (-1, 1) for sy in (-1, 1)
                    for z in (0.0, hgt)])
    tpd = math.radians(cam.pitch_down_deg)
    P = pts - [0, 0, cam.mount_height_m]
    xc = -P[:, 1]
    yc = -math.sin(tpd) * P[:, 0] - math.cos(tpd) * P[:, 2]
    zc = math.cos(tpd) * P[:, 0] - math.sin(tpd) * P[:, 2]
    if np.any(zc < 0.1):
        return None
    u = cam.f * xc / zc + cam.width / 2
    v = cam.f * yc / zc + cam.height / 2
    x1, x2 = np.clip([u.min(), u.max()], 0, cam.width - 1)
    y1, y2 = np.clip([v.min(), v.max()], 0, cam.height - 1)
    if x2 - x1 < 4 or y2 - y1 < 10:
        return None
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# 任务仿真
# ---------------------------------------------------------------------------

@dataclass
class MissionResult:
    policy: str
    success: bool
    time_s: float
    distance_m: float
    replans: int
    collisions: int
    person_err_m: float | None
    trajectory: list = field(default_factory=list, repr=False)
    initial_path: list = field(default_factory=list, repr=False)
    final_map: GridMap | None = field(default=None, repr=False)


def _plan(gm: GridMap, pose, goal_cell):
    s = nearest_free(gm.grid, gm.world_to_cell(pose[0], pose[1]), 6)
    if s is None or goal_cell is None:
        return None
    g = nearest_free(gm.grid, goal_cell, 8)
    if g is None:
        return None
    path = astar(gm.grid, s, g, allow_diag=True)
    if path is None:
        return None
    wps = path_to_world(gm, simplify_path(gm.grid, path))
    wps[0] = (pose[0], pose[1])
    return wps


def lawnmower_goals(gm: GridMap, start_xy, spacing=5.0, margin=1.5):
    """仅地面搜索时的“割草机”式往返扫描航点（从起点一侧开始，逐条向前推进）"""
    W = gm.shape[1] * gm.resolution
    x0, x1 = gm.x_min + margin, gm.x_min + W - margin
    ys = np.arange(start_xy[1] + spacing / 2, gm.y_max - margin, spacing)
    goals = []
    for k, y in enumerate(ys):
        xs = (x0, x1) if k % 2 == 0 else (x1, x0)
        goals += [(xs[0], float(y)), (xs[1], float(y))]
    return goals


def run_mission(scene: Scene, policy: str, rng, aerial=None, max_time=None, dt=0.1,
                inflate_m=0.3) -> MissionResult:
    """
    policy:
        "coop"         空地协同：航拍栅格地图 + 航拍人员坐标作为终点
        "ground_known" 仅地面（理想化对照）：空白地图，只靠雷达逐步建图，但假设已知人员真实坐标
        "ground_search"仅地面（实际情况）：空白地图且不知道人员位置，按割草机路线搜索，
                       车载相机发现人员后再前往
    """
    base = aerial.gridmap
    if policy == "coop":
        gm = copy.deepcopy(base)
        goal_cell, _ = select_goal(gm, aerial.persons_xy)
        goals = [gm.cell_to_world(*goal_cell)]
    else:
        gm = GridMap(np.zeros_like(base.grid), base.resolution, base.x_min, base.y_max,
                     raw=np.zeros_like(base.grid))
        if policy == "ground_known":
            goals = [scene.person_xy]
        else:
            goals = lawnmower_goals(gm, scene.start_pose[:2])
    if max_time is None:
        max_time = 900.0 if policy == "ground_search" else 300.0

    mapper = LidarMapper(gm, inflate_m)
    gap_cfg = GapConfig(car_width=0.25, margin=0.08)
    ctrl = HeadingController(DriveConfig())
    car_cam = CarCamera()
    pose = list(scene.start_pose)

    rel, ranges = lidar_scan(scene.truth, pose, rng)
    mapper.update(pose, rel, ranges)

    def new_follower(goal_xy):
        wps = _plan(gm, pose, gm.world_to_cell(*goal_xy))
        return PurePursuit(wps or [(pose[0], pose[1]), goal_xy], lookahead=0.8), wps

    follower, wps = new_follower(goals[0])
    initial = list(wps) if wps else []
    goal_idx = 0
    homing = policy != "ground_search"   # 是否已在前往人员
    replans, collisions, dist = 0, 0, 0.0
    traj = [tuple(pose)]
    fixes = []
    t = 0.0
    next_replan = 1.0
    last_progress = (t, pose[0], pose[1])
    success = False

    while t < max_time:
        rel, ranges = lidar_scan(scene.truth, pose, rng)
        if t >= next_replan:
            mapper.update(pose, rel, ranges)
            if mapper.path_blocked(follower.path[follower.idx::5]):
                follower, wps = new_follower(goals[goal_idx])
                replans += 1
            next_replan = t + 0.5

        target = follower.target_bearing(pose)
        local = plan_local(rel, ranges, math.degrees(target), gap_cfg)
        if local.heading_deg is None:
            v, w = recovery(rel, ranges, target, ctrl, dt, gap_cfg)
        else:
            v, w = ctrl.compute(math.radians(local.heading_deg), dt, local.min_front_dist)
            if local.emergency_stop:
                v = 0.0

        nx = pose[0] + v * math.cos(pose[2]) * dt
        ny = pose[1] + v * math.sin(pose[2]) * dt
        nth = wrap_angle(pose[2] + w * dt)
        if v != 0 and collides(scene.truth, nx, ny,
                               heading=pose[2] if v > 0 else wrap_angle(pose[2] + math.pi)):
            collisions += 1
            nx, ny = pose[0], pose[1]
        dist += math.hypot(nx - pose[0], ny - pose[1])
        pose = [nx, ny, nth]
        traj.append(tuple(pose))
        t += dt

        # 车载相机人员检测与定位（模型召回率按 0.9 计，需视线无遮挡）
        bb = project_person_to_car_camera(scene, pose, car_cam)
        if bb is not None and rng.random() < 0.9:
            dperson = math.hypot(scene.person_xy[0] - pose[0], scene.person_xy[1] - pose[1])
            bearing = math.atan2(scene.person_xy[1] - pose[1], scene.person_xy[0] - pose[0])
            _, rr = lidar_scan(scene.truth, (pose[0], pose[1], bearing), rng, n_beams=1,
                               max_range=max(0.1, dperson - 0.5))
            if not np.isfinite(rr[0]) and dperson < 8.0:
                fix = locate_person(bb, car_cam)
                if fix is not None:
                    fixes.append(car_to_world(fix, pose))

        near = math.hypot(scene.person_xy[0] - pose[0], scene.person_xy[1] - pose[1]) < 2.0
        if len(fixes) >= 3 and near:
            success = True
            break

        # 发现人员（或到达航拍终点但人员不在附近）-> 以车载相机定位结果为新终点
        reached = follower.reached(pose)
        if fixes and (not homing or (reached and not near)):
            homing = True
            goals = [tuple(np.mean(fixes[-5:], axis=0))]
            goal_idx = 0
            follower, _ = new_follower(goals[0])
            replans += 1
            continue
        # 搜索模式：到达航点或长时间无进展（航点不可达）则前往下一个航点
        stuck = t - last_progress[0] > 15.0
        if not homing and (reached or stuck):
            goal_idx += 1
            if goal_idx >= len(goals):
                break
            follower, _ = new_follower(goals[goal_idx])
            last_progress = (t, pose[0], pose[1])
        if math.hypot(pose[0] - last_progress[1], pose[1] - last_progress[2]) > 1.0:
            last_progress = (t, pose[0], pose[1])

    err = None
    if fixes:
        ex, ey = np.mean(fixes, axis=0)
        err = float(math.hypot(ex - scene.person_xy[0], ey - scene.person_xy[1]))
    return MissionResult(policy, success, round(t, 2), round(dist, 2), replans, collisions, err,
                         traj, initial, gm)


def map_quality(scene: Scene, aerial) -> dict:
    """航拍地图（膨胀前）与真实障碍物的比较：召回率 / 精确率 / 人员定位误差"""
    gm = aerial.gridmap
    raw = gm.raw
    truth_c = np.zeros_like(raw)
    H, W = raw.shape
    rr, cc = np.mgrid[0:H, 0:W]
    xs = gm.x_min + (cc + 0.5) * gm.resolution
    ys = gm.y_max - (rr + 0.5) * gm.resolution
    tr = np.floor((scene.truth.y_max - ys) / scene.truth.resolution).astype(int)
    tc = np.floor((xs - scene.truth.x_min) / scene.truth.resolution).astype(int)
    ok = (tr >= 0) & (tr < scene.truth.grid.shape[0]) & (tc >= 0) & (tc < scene.truth.grid.shape[1])
    truth_c[ok] = scene.truth.grid[tr[ok], tc[ok]]
    tp = int(((raw == 1) & (truth_c == 1)).sum())
    recall = tp / max(1, int(truth_c.sum()))
    precision = tp / max(1, int(raw.sum()))
    covered = int(((gm.grid == 1) & (truth_c == 1)).sum()) / max(1, int(truth_c.sum()))
    perr = None
    if aerial.persons_xy:
        px, py, _ = aerial.persons_xy[0]
        perr = math.hypot(px - scene.person_xy[0], py - scene.person_xy[1])
    return {"obstacle_recall": recall, "obstacle_precision": precision,
            "obstacle_coverage_inflated": covered, "aerial_person_err_m": perr}


# ---------------------------------------------------------------------------
# 可视化
# ---------------------------------------------------------------------------

def plot_mission(scene, dets, aerial, results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei",
                                       "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"debris": "#8c6d46", "stone": "#7f7f7f", "wood": "#4f7f3a", "person": "#d62728"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    # (a) 无人机视角 + 检测框
    ax = axes[0]
    for o in scene.obstacles:
        uv = ground_to_pixels(o.poly, scene.cam, scene.true_pose)
        ax.add_patch(Polygon(uv, closed=True, fc=colors[o.cls_name],
                             alpha=1.0 if o.visible_from_air else 0.25))
    for d in dets:
        x1, y1, x2, y2 = d.xyxy
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               ec=colors.get(d.cls_name, "b"), lw=1.2))
    ax.set_xlim(0, scene.cam.width)
    ax.set_ylim(scene.cam.height, 0)
    ax.set_aspect("equal")
    ax.set_title("(a) 无人机视角与检测框（半透明为空中不可见障碍）", fontsize=10.5)

    # (b) 航拍栅格地图 + 初始 A* 路径
    gm = aerial.gridmap
    ext = [gm.x_min, gm.x_min + gm.shape[1] * gm.resolution,
           gm.y_max - gm.shape[0] * gm.resolution, gm.y_max]
    ax = axes[1]
    show = gm.grid.astype(float) * 0.45 + gm.raw.astype(float) * 0.55
    ax.imshow(show, cmap="Greys", extent=ext, vmin=0, vmax=1)
    coop = next(r for r in results if r.policy == "coop")
    if coop.initial_path:
        p = np.array(coop.initial_path)
        ax.plot(p[:, 0], p[:, 1], "-o", c="#1f77b4", ms=3, lw=2, label="A* 初始路径")
    for px, py, _ in aerial.persons_xy:
        ax.plot(px, py, "r*", ms=14, label="航拍人员坐标")
    ax.plot(*scene.start_pose[:2], "gs", ms=9, label="小车起点")
    ax.set_title(f"(b) 栅格地图与 A* 初始路径（{gm.resolution} m/格）", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8)

    # (c) 真实环境中的实际轨迹
    ax = axes[2]
    t = scene.truth
    text = [t.x_min, t.x_min + t.grid.shape[1] * t.resolution,
            t.y_max - t.grid.shape[0] * t.resolution, t.y_max]
    ax.imshow(t.grid, cmap="Greys", extent=text, vmin=0, vmax=1.6)
    for o in scene.obstacles:
        if not o.visible_from_air:
            ax.add_patch(Polygon(o.poly, closed=True, fc="none", ec="orange", lw=1.5))
    style = {"coop": ("#1f77b4", "空地协同"), "ground_known": ("#ff7f0e", "仅地面(已知坐标)"),
             "ground_search": ("#9467bd", "仅地面(搜索)")}
    for r in results:
        tr = np.array(r.trajectory)
        c, name = style[r.policy]
        ax.plot(tr[:, 0], tr[:, 1], c=c, lw=2,
                label=f"{name}: {'成功' if r.success else '失败'} {r.time_s:.0f}s {r.distance_m:.1f}m")
    ax.plot(*scene.person_xy, "r*", ms=14, label="受困人员（真实）")
    ax.plot(*scene.start_pose[:2], "gs", ms=9)
    ax.set_title("(c) 三种策略的行驶轨迹（橙框为仅雷达可见障碍）", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8)
    for ax in axes[1:]:
        ax.set_xlabel("x / m")
        ax.set_ylabel("y / m")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------

POLICIES = ("coop", "ground_known", "ground_search")


def run_trial(seed: int, trap=True, figure=None):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        scene = random_scene(rng, trap=trap)
        if scene_solvable(scene):
            break
    dets = simulate_drone_detections(scene, rng)
    t0 = time.perf_counter()
    aerial = build_grid_map(dets, scene.cam, scene.measured_pose, resolution=0.25,
                            inflate_radius_m=0.3)
    map_ms = (time.perf_counter() - t0) * 1000
    results = [run_mission(scene, p, np.random.default_rng(seed + 1000), aerial)
               for p in POLICIES]
    if figure:
        plot_mission(scene, dets, aerial, results, figure)
    q = map_quality(scene, aerial)
    q["map_build_ms"] = map_ms
    # 对照：不做 IMU 姿态补偿（假设相机始终竖直向下）时的建图质量
    level = DronePose(altitude_m=scene.measured_pose.altitude_m)
    q_level = map_quality(scene, build_grid_map(dets, scene.cam, level, 0.25, 0.3))
    q["no_imu_obstacle_recall"] = q_level["obstacle_recall"]
    q["no_imu_person_err_m"] = q_level["aerial_person_err_m"]
    return scene, results, q


def summarize(rows):
    out = {}
    for pol in POLICIES:
        rs = [r[pol] for r in rows]
        ok = [r for r in rs if r["success"]]
        out[pol] = {
            "trials": len(rs),
            "success_rate": len(ok) / max(1, len(rs)),
            "mean_time_s_success": float(np.mean([r["time_s"] for r in ok])) if ok else None,
            "median_time_s_success": float(np.median([r["time_s"] for r in ok])) if ok else None,
            "mean_distance_m_success": float(np.mean([r["distance_m"] for r in ok])) if ok else None,
            "collisions_total": int(sum(r["collisions"] for r in rs)),
            "mean_person_err_m": float(np.mean([r["person_err_m"] for r in rs
                                                if r["person_err_m"] is not None] or [np.nan])),
        }
    for other in POLICIES[1:]:
        both = [r for r in rows if r["coop"]["success"] and r[other]["success"]]
        if both:
            out[f"paired_time_ratio_coop_over_{other}"] = float(np.median(
                [r["coop"]["time_s"] / r[other]["time_s"] for r in both]))
            out[f"paired_trials_{other}"] = len(both)
    for k in ("obstacle_recall", "obstacle_precision", "obstacle_coverage_inflated",
              "aerial_person_err_m", "no_imu_obstacle_recall", "no_imu_person_err_m",
              "map_build_ms"):
        vals = [r["map"][k] for r in rows if r["map"][k] is not None]
        out[k] = float(np.mean(vals)) if vals else None
    return out


def main():
    ap = argparse.ArgumentParser(description="空地协同搜救全流程仿真")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--no-trap", action="store_true", help="不生成 U 型死路")
    ap.add_argument("--figure", type=str, default=None, help="保存第一个试验的可视化图")
    ap.add_argument("--json", type=str, default=None, help="保存统计结果")
    args = ap.parse_args()

    rows = []
    for i in range(args.trials):
        seed = args.seed + i
        _, results, q = run_trial(seed, trap=not args.no_trap,
                                  figure=args.figure if i == 0 else None)
        row = {"seed": seed, "map": q}
        for r in results:
            row[r.policy] = {k: getattr(r, k) for k in
                             ("success", "time_s", "distance_m", "replans", "collisions", "person_err_m")}
        rows.append(row)
        cols = "  ".join(f"{p}: {'OK ' if row[p]['success'] else 'FAIL'} {row[p]['time_s']:6.1f}s"
                         for p in POLICIES)
        print(f"[{i + 1}/{args.trials}] seed={seed}  {cols}  map recall={q['obstacle_recall']:.2f}",
              flush=True)
    summary = summarize(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "trials": rows}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
