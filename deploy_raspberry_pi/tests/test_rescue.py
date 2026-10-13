"""
算法模块单元测试：python -m pytest deploy_raspberry_pi/tests -q
只依赖 numpy，无需摄像头、雷达、模型文件
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rescue.comm import LoopbackLink, TOPIC_MAP, decode_map, encode_map  # noqa: E402
from rescue.control import DriveConfig, HeadingController, PurePursuit  # noqa: E402
from rescue.geo_mapping import (CameraModel, Detection, DronePose, altitude_from_pressure,  # noqa: E402
                                build_grid_map, ground_sample_distance, ground_to_pixels,
                                inflate_manhattan, pixels_to_ground, select_goal,
                                select_goal_free_block)
from rescue.lidar_gap import GapConfig, plan_local, rplidar_to_standard  # noqa: E402
from rescue.person_locate import CarCamera, locate_person  # noqa: E402
from rescue.planner import astar, path_length_cells, simplify_path  # noqa: E402


# ---------------- geo_mapping ----------------

def test_gsd_nadir():
    cam = CameraModel(width=640, hfov_deg=62.2)
    gsd = ground_sample_distance(cam, 25.0)
    # 25 m 高度覆盖宽度 = 2*25*tan(31.1°) ≈ 30.17 m
    assert gsd * 640 == pytest.approx(2 * 25 * math.tan(math.radians(31.1)), rel=1e-6)


def test_pixel_ground_roundtrip_with_attitude():
    cam = CameraModel()
    pose = DronePose(altitude_m=20, pitch_deg=8, roll_deg=-5, yaw_deg=3)
    xy = np.array([[0, 0], [3.5, -2.0], [-6, 7]])
    uv = ground_to_pixels(xy, cam, pose)
    back = pixels_to_ground(uv, cam, pose)
    assert np.allclose(back, xy, atol=1e-6)


def test_image_center_projects_below_drone_when_level():
    cam = CameraModel()
    xy = pixels_to_ground(np.array([[cam.cx, cam.cy]]), cam, DronePose(altitude_m=15))
    assert np.allclose(xy, 0, atol=1e-9)


def test_pitch_moves_center_forward():
    cam = CameraModel()
    xy = pixels_to_ground(np.array([[cam.cx, cam.cy]]), cam, DronePose(altitude_m=10, pitch_deg=10))
    assert xy[0, 1] == pytest.approx(10 * math.tan(math.radians(10)))


def test_barometric_altitude():
    assert altitude_from_pressure(1013.25, 1013.25) == pytest.approx(0)
    # 约 12 hPa ≈ 100 m
    assert altitude_from_pressure(1001.3, 1013.25) == pytest.approx(100, abs=3)


def test_inflate_manhattan_diamond():
    g = np.zeros((7, 7), np.uint8)
    g[3, 3] = 1
    out = inflate_manhattan(g, 2)
    rr, cc = np.nonzero(out)
    assert out.sum() == 13  # 曼哈顿半径 2 的菱形
    assert np.all(np.abs(rr - 3) + np.abs(cc - 3) <= 2)


def test_build_grid_map_places_obstacle_at_true_location():
    cam = CameraModel()
    pose = DronePose(altitude_m=25, pitch_deg=2, roll_deg=-1)
    # 地面上 (4, 5) 处 2m x 1m 的障碍
    poly = np.array([[3, 4.5], [5, 4.5], [5, 5.5], [3, 5.5]])
    uv = ground_to_pixels(poly, cam, pose)
    det = Detection("debris", 0.9, (*uv.min(0), *uv.max(0)))
    person_uv = ground_to_pixels(np.array([[-3.0, -2.0]]), cam, pose)[0]
    pdet = Detection("person", 0.8, (person_uv[0] - 2, person_uv[1] - 2, person_uv[0] + 2, person_uv[1] + 2))
    res = build_grid_map([det, pdet], cam, pose, resolution=0.25, inflate_radius_m=0.0)
    gm = res.gridmap
    assert gm.raw[gm.world_to_cell(4, 5)] == 1
    assert gm.raw[gm.world_to_cell(0, 0)] == 0
    assert gm.raw.sum() * 0.25 ** 2 == pytest.approx(2.0, rel=0.35)
    # 人员不是障碍，而是目标
    assert len(res.persons_xy) == 1
    assert res.persons_xy[0][:2] == pytest.approx((-3, -2), abs=0.05)
    goal, kind = select_goal(gm, res.persons_xy)
    assert kind == "person"
    assert gm.cell_to_world(*goal) == pytest.approx((-3, -2), abs=0.25)


def test_grid_update_rate_over_5hz():
    """申报书指标：栅格地图更新频率 > 5 Hz（这里测 50 个检测框的建图耗时）"""
    rng = np.random.default_rng(0)
    cam = CameraModel()
    dets = []
    for _ in range(50):
        x, y = rng.uniform(0, 600), rng.uniform(0, 440)
        dets.append(Detection("stone", 0.5, (x, y, x + rng.uniform(5, 40), y + rng.uniform(5, 40))))
    t0 = time.perf_counter()
    for _ in range(10):
        build_grid_map(dets, cam, DronePose(altitude_m=25), resolution=0.25)
    per_frame = (time.perf_counter() - t0) / 10
    assert per_frame < 0.2


def test_free_block_goal_selection():
    g = np.ones((40, 40), np.uint8)
    g[2:10, 25:35] = 0   # 前方唯一的空旷块
    g[30:, :] = 0
    r, c = select_goal_free_block(g, block=6, front_ratio=0.35)
    assert 2 <= r < 10 and 25 <= c < 35


# ---------------- planner ----------------

def test_astar_optimal_8_connected():
    g = np.zeros((20, 20), np.uint8)
    path = astar(g, (19, 0), (0, 19), allow_diag=True)
    assert path_length_cells(path) == pytest.approx(19 * math.sqrt(2))


def test_astar_detours_wall_and_no_corner_cutting():
    g = np.zeros((10, 10), np.uint8)
    g[5, :9] = 1
    path = astar(g, (9, 0), (0, 0), allow_diag=True)
    assert path is not None
    assert all(g[p] == 0 for p in path)
    assert (5, 9) in path
    # 两个对角相邻障碍之间不能穿过
    g2 = np.zeros((3, 3), np.uint8)
    g2[0, 1] = g2[1, 0] = 1
    assert astar(g2, (0, 0), (1, 1), allow_diag=True) is None


def test_astar_no_path():
    g = np.zeros((10, 10), np.uint8)
    g[5, :] = 1
    assert astar(g, (9, 0), (0, 0)) is None


def test_simplify_path_keeps_free_lines():
    g = np.zeros((10, 10), np.uint8)
    g[5, :9] = 1
    path = astar(g, (9, 0), (0, 0))
    simp = simplify_path(g, path)
    assert simp[0] == path[0] and simp[-1] == path[-1]
    assert len(simp) < len(path)


# ---------------- lidar_gap ----------------

def _scan_with_wall(gap_center_deg, gap_half_deg, wall_dist=0.8):
    ang = np.arange(-180, 180, 0.5)
    rng = np.full_like(ang, np.inf)
    front = np.abs(ang) <= 60
    rng[front] = wall_dist
    opening = np.abs(ang - gap_center_deg) <= gap_half_deg
    rng[opening] = 5.0
    return ang, rng


def test_gap_found_and_heading_inside():
    ang, rng = _scan_with_wall(25, 20)
    plan = plan_local(ang, rng, target_deg=0.0)
    passable = [g for g in plan.gaps if g.passable]
    assert len(passable) == 1
    assert 5 <= passable[0].start_deg and passable[0].end_deg <= 45.5
    assert 5 < plan.heading_deg < 45


def test_narrow_gap_rejected():
    cfg = GapConfig(car_width=0.4, margin=0.1)
    ang, rng = _scan_with_wall(0, 4, wall_dist=0.8)  # 宽度约 0.8*2*sin(4.5°) ≈ 0.13 m
    plan = plan_local(ang, rng, target_deg=0.0, cfg=cfg)
    assert plan.heading_deg is None


def test_open_space_follows_target():
    ang = np.arange(-180, 180, 1.0)
    plan = plan_local(ang, np.full_like(ang, np.inf), target_deg=17.0)
    assert plan.heading_deg == pytest.approx(17.0)
    assert not plan.emergency_stop


def test_corridor_rejects_heading_that_scrapes_side_obstacle():
    """回归测试：右前方 0.15 m 处的障碍不在缝隙边界上，但位于车体扫过的走廊内"""
    ang = np.arange(-180, 180, 1.0)
    rng = np.full_like(ang, np.inf)
    rng[(ang >= -62) & (ang <= -52)] = 0.15
    rng[(ang >= -51) & (ang <= -25)] = 0.47
    cfg = GapConfig(car_width=0.25, margin=0.08)
    plan = plan_local(ang, rng, target_deg=0.0, cfg=cfg)
    if plan.heading_deg is not None:
        a = np.radians(ang - plan.heading_deg)
        fx, fy = rng * np.cos(a), rng * np.sin(a)
        assert not np.any((fx > 0) & (fx < 1.0) & (np.abs(fy) < 0.205))


def test_emergency_stop():
    ang = np.arange(-180, 180, 1.0)
    rng = np.full_like(ang, np.inf)
    rng[np.abs(ang) < 5] = 0.15
    assert plan_local(ang, rng, 0.0).emergency_stop


def test_rplidar_angle_conversion():
    assert np.allclose(rplidar_to_standard(np.array([0, 90, 270, 180])), [0, -90, 90, -180])


# ---------------- control ----------------

def test_heading_controller_turns_toward_error():
    ctrl = HeadingController(DriveConfig())
    v, w = ctrl.compute(math.radians(20), 0.1)
    assert v > 0 and w > 0
    left, right = ctrl.to_wheels(v, w)
    assert right > left
    assert max(abs(left), abs(right)) <= 1.0
    v, _ = ctrl.compute(math.radians(120), 0.1)
    assert v == 0.0  # 大误差原地转向


def test_pure_pursuit_bearing():
    pp = PurePursuit([(0, 0), (0, 5)], lookahead=1.0)
    assert pp.target_bearing((0, 0, math.pi / 2)) == pytest.approx(0, abs=1e-6)
    assert pp.target_bearing((0, 0, 0)) == pytest.approx(math.pi / 2, abs=1e-6)
    assert pp.reached((0, 4.9, 0))


# ---------------- person_locate ----------------

def _project(cam, X, Y, Z):
    th = math.radians(cam.pitch_down_deg)
    P = np.array([X, Y, Z - cam.mount_height_m])
    xc = -P[1]
    yc = -math.sin(th) * P[0] - math.cos(th) * P[2]
    zc = math.cos(th) * P[0] - math.sin(th) * P[2]
    return cam.f * xc / zc + cam.width / 2, cam.f * yc / zc + cam.height / 2


@pytest.mark.parametrize("X,Y", [(3.0, 0.0), (4.0, 1.0), (2.5, -0.8)])
def test_person_ground_contact(X, Y):
    cam = CarCamera()
    u_foot, v_foot = _project(cam, X, Y, 0.0)
    _, v_head = _project(cam, X, Y, 1.7)
    fix = locate_person((u_foot - 10, v_head, u_foot + 10, v_foot), cam)
    assert fix.method == "ground_contact"
    assert fix.x == pytest.approx(X, abs=1e-6)
    assert fix.y == pytest.approx(Y, abs=1e-6)


def test_person_height_prior_when_truncated():
    cam = CarCamera()
    u, _ = _project(cam, 2.0, 0.0, 0.0)
    _, v_head = _project(cam, 2.0, 0.0, 1.7)
    _, v_foot = _project(cam, 2.0, 0.0, 0.0)
    fix = locate_person((u - 10, v_head, u + 10, cam.height), cam)
    assert fix.method == "height_prior"
    assert 1.0 < fix.distance < 6.0


# ---------------- comm ----------------

def test_map_message_roundtrip_and_size():
    rng = np.random.default_rng(1)
    from rescue.geo_mapping import GridMap
    g = (rng.random((96, 128)) < 0.1).astype(np.uint8)
    gm = GridMap(g, 0.25, -15.0, 11.3)
    payload = encode_map(gm, [(1.0, 2.0, 0.9)])
    link = LoopbackLink()
    got = {}
    link.subscribe(TOPIC_MAP, lambda t, p: got.setdefault("m", decode_map(p)))
    link.publish(TOPIC_MAP, payload)
    gm2, persons, _ = got["m"]
    assert np.array_equal(gm2.grid, g)
    assert gm2.resolution == 0.25 and gm2.x_min == -15.0
    assert persons[0] == (1.0, 2.0, 0.9)
    assert len(payload) < 4000


# ---------------- 全流程仿真（冒烟测试）----------------

def test_simulation_smoke():
    from rescue.sim import run_trial
    _, results, q = run_trial(seed=3)
    coop = next(r for r in results if r.policy == "coop")
    assert coop.collisions == 0
    assert q["obstacle_recall"] > 0.5


# ---------------- local_map（雷达修正地图）----------------

def test_lidar_mapper_marks_hits_carves_free_and_ignores_invalid():
    from rescue.geo_mapping import GridMap
    from rescue.local_map import LidarMapper
    gm = GridMap(np.zeros((40, 40), np.uint8), 0.25, -5.0, 5.0, raw=np.zeros((40, 40), np.uint8))
    gm.raw[20, 30] = 1                           # 航拍误检：小车正前方 2.5 m 处的“假障碍”
    mapper = LidarMapper(gm, inflate_m=0.0)
    # 小车在原点朝 +x；正前方 4 m 有真实障碍；另有一条 0 读数（无效）与一条 inf（开阔）
    mapper.update((0.0, 0.0, 0.0), [0.0, 90.0, 180.0], [4.0, 0.0, float("inf")])
    assert gm.raw[gm.world_to_cell(4.0, 0.0)] == 1      # 命中点标记为障碍
    assert gm.raw[gm.world_to_cell(2.6, 0.0)] == 0      # 射线经过处的假障碍被清除
    assert gm.raw[gm.world_to_cell(0.0, 0.1)] == 0      # 0 读数没有在车身处标障碍
    assert mapper.path_blocked([(0, 0), (0.5, 0), (1, 0), (4.0, 0.0)])
    assert not mapper.path_blocked([(0, 0), (0, -0.5), (0, -1), (0, -2)])


def test_recovery_ignores_invalid_zero_readings():
    from rescue.control import recovery
    ctrl = HeadingController(DriveConfig())
    cfg = GapConfig()
    ang = np.arange(-180, 180, 1.0)
    rng = np.full_like(ang, 3.0)
    rng[np.abs(ang) < 10] = 0.0                 # 前方全是无效读数
    v, _ = recovery(ang, rng, 0.5, ctrl, 0.1, cfg)
    assert v == 0.0                             # 不应误判为“贴近障碍”而后退
    rng[np.abs(ang) < 10] = 0.15                # 真的贴近障碍
    v, _ = recovery(ang, rng, 0.5, ctrl, 0.1, cfg)
    assert v < 0


# ---------------- PersonTracker（车载多帧确认）----------------

def test_person_tracker_confirms_cluster_and_rejects_outlier():
    from rescue.person_locate import PersonTracker
    tr = PersonTracker(min_hits=3, window_s=3.0, gate_m=1.0)
    tr.add((5.0, 5.0), 0.0)
    tr.add((20.0, -3.0), 0.1)                   # 单帧误检
    assert tr.confirmed(0.2) is None
    tr.add((5.2, 4.9), 0.5)
    tr.add((4.9, 5.1), 1.0)
    xy = tr.confirmed(1.1)
    assert xy == pytest.approx((5.03, 5.0), abs=0.05)
    assert tr.confirmed(10.0) is None           # 超出时间窗口后失效


# ---------------- car_node 集成（无硬件 dry-run）----------------

def test_car_node_dry_run_plans_and_reaches_goal(monkeypatch, capsys):
    import car_node
    from rescue.comm import LoopbackLink, encode_map
    from rescue.geo_mapping import GridMap
    g = np.zeros((60, 60), np.uint8)
    g[30, 5:55] = 1                             # 一堵墙，必须绕行
    gm = GridMap(g, 0.25, -7.5, 7.5)
    payload = encode_map(gm, [(0.0, 5.0, 0.9)])

    class FakeLink(LoopbackLink):
        def __init__(self, *a, **k):
            super().__init__()

        def subscribe(self, topic, cb):
            super().subscribe(topic, cb)
            cb(topic, payload)

    def fake_print(*a, **k):
        print_orig(*a, **k)
        if a and "到达终点" in str(a[0]):
            raise KeyboardInterrupt

    print_orig = print
    monkeypatch.setattr(car_node, "MqttLink", FakeLink)
    monkeypatch.setattr(car_node.time, "sleep", lambda s: None)
    monkeypatch.setattr(car_node, "print", fake_print, raising=False)
    monkeypatch.setattr(sys, "argv", ["car_node.py", "--mqtt", "x", "--dry-run"])
    car_node.main()
    out = capsys.readouterr().out
    assert "[规划·新地图]" in out and "到达终点" in out
