#!/usr/bin/env python3
"""
地面端节点：订阅航拍栅格地图 -> A* 全局规划 -> 激光雷达缝隙检测避障 -> 航向控制 -> 电机
         行驶中用雷达修正地图并在路径受阻时重规划；车载相机确认人员后以其位置为新终点

示例:
    # 只验证“接收地图 + 规划路径”，不驱动电机、不接雷达（目前无小车实物时使用）
    python3 car_node.py --mqtt localhost --dry-run

    # 真车：RPLIDAR 接 /dev/ttyUSB0，电机驱动板接 GPIO（引脚按实际接线修改），车载相机找人
    python3 car_node.py --mqtt <无人机IP> --lidar /dev/ttyUSB0 --left-pins 17 27 --right-pins 22 23 \
        --person-model best_person.pt --camera picamera

    # 只测试雷达缝隙检测（打印每圈扫描的缝隙与期望航向）
    python3 car_node.py --lidar /dev/ttyUSB0 --lidar-only

说明:
    小车位姿目前按航迹推算（指令速度积分）估计，起点为航拍地图中的 --start-x/--start-y。
    航迹推算会累积误差，真车部署时建议接入轮式编码器或 IMU 航向角进行修正。
"""

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rescue.comm import TOPIC_MAP, TOPIC_STATUS, MqttLink, decode_map  # noqa: E402
from rescue.control import (DriveConfig, DummyMotorDriver, GpioZeroMotorDriver,  # noqa: E402
                            HeadingController, PurePursuit, recovery, wrap_angle)
from rescue.geo_mapping import DEFAULT_TARGET_CLASSES, nearest_free, select_goal  # noqa: E402
from rescue.lidar_gap import GapConfig, plan_local, rplidar_to_standard  # noqa: E402
from rescue.local_map import LidarMapper  # noqa: E402
from rescue.person_locate import CarCamera, PersonTracker, car_to_world, locate_person  # noqa: E402
from rescue.planner import astar, path_to_world, render_ascii, simplify_path  # noqa: E402


class LidarReader(threading.Thread):
    """后台线程持续读取 RPLIDAR，保存最新一圈扫描（需 pip install rplidar-roboticia）"""

    def __init__(self, port):
        super().__init__(daemon=True)
        from rplidar import RPLidar
        self.lidar = RPLidar(port)
        self.latest = None
        self.running = True

    def run(self):
        for scan in self.lidar.iter_scans(max_buf_meas=3000):
            if not self.running:
                break
            # scan: [(quality, angle_deg, distance_mm), ...]
            ang = rplidar_to_standard([s[1] for s in scan])
            rng = [s[2] / 1000.0 for s in scan]
            self.latest = (ang, rng, time.time())

    def stop(self):
        self.running = False
        self.lidar.stop()
        self.lidar.stop_motor()
        self.lidar.disconnect()


class PersonWatcher(threading.Thread):
    """后台线程：车载相机 + YOLO 人员模型，检测到人员后用 locate_person 估计车体系位置"""

    def __init__(self, model_path, source, cam: CarCamera, conf=0.4, imgsz=320):
        super().__init__(daemon=True)
        from ultralytics import YOLO
        from drone_node import FrameSource
        self.model = YOLO(model_path)
        self.src = FrameSource(source, cam.width, cam.height)
        self.cam, self.conf, self.imgsz = cam, conf, imgsz
        self.targets = {c.lower() for c in DEFAULT_TARGET_CLASSES}
        self.fixes = []          # [(t, PersonFix)]，由主循环取走
        self.lock = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            frame = self.src.read()
            if frame is None:
                time.sleep(0.05)
                continue
            t = time.time()
            res = self.model(frame, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
            for box in res.boxes:
                if res.names[int(box.cls[0])].lower() not in self.targets:
                    continue
                fix = locate_person(tuple(float(v) for v in box.xyxy[0]), self.cam)
                if fix is not None:
                    with self.lock:
                        self.fixes.append((t, fix))

    def take(self):
        with self.lock:
            out, self.fixes = self.fixes, []
        return out

    def stop(self):
        self.running = False
        self.src.close()


def plan_path(gm, start_xy, goal_cell):
    """A* 规划，返回 (航点列表, 起点格, 终点格)；无路径时航点为 None"""
    s = nearest_free(gm.grid, gm.world_to_cell(*start_xy), 10)
    g = nearest_free(gm.grid, tuple(goal_cell), 10) if goal_cell is not None else None
    if s is None or g is None:
        return None, s, g, None
    path = astar(gm.grid, s, g, allow_diag=True)
    if path is None:
        return None, s, g, None
    wps = path_to_world(gm, simplify_path(gm.grid, path))
    wps[0] = tuple(start_xy)
    return wps, s, g, path


def main():
    ap = argparse.ArgumentParser(description="地面端：地图接收 + 路径规划 + 雷达避障 + 运动控制")
    ap.add_argument("--mqtt", default=None)
    ap.add_argument("--lidar", default=None, help="RPLIDAR 串口，如 /dev/ttyUSB0")
    ap.add_argument("--lidar-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="不驱动电机")
    ap.add_argument("--start-x", type=float, default=0.0)
    ap.add_argument("--start-y", type=float, default=None, help="默认取地图下边缘上方 1 m")
    ap.add_argument("--left-pins", type=int, nargs=2, default=(17, 27))
    ap.add_argument("--right-pins", type=int, nargs=2, default=(22, 23))
    ap.add_argument("--max-speed", type=float, default=0.3)
    ap.add_argument("--safe-dist", type=float, default=1.0)
    ap.add_argument("--car-width", type=float, default=0.25)
    ap.add_argument("--inflate", type=float, default=0.3, help="雷达修正地图时的膨胀半径（米）")
    ap.add_argument("--person-model", default=None, help="车载人员识别模型，如 best_person.pt")
    ap.add_argument("--camera", default="picamera", help="车载相机：picamera / 摄像头编号 / 视频")
    ap.add_argument("--cam-height", type=float, default=0.15, help="车载相机离地高度（米）")
    ap.add_argument("--cam-pitch", type=float, default=10.0, help="车载相机俯角（度）")
    ap.add_argument("--cam-hfov", type=float, default=62.2, help="车载相机水平视场角（度）")
    args = ap.parse_args()

    if args.lidar_only and not args.lidar:
        ap.error("--lidar-only 需要同时指定 --lidar")
    if not args.lidar_only and not args.mqtt:
        ap.error("需要 --mqtt 指定 broker 以接收地图")

    gap_cfg = GapConfig(safe_dist=args.safe_dist, car_width=args.car_width)
    lidar = LidarReader(args.lidar) if args.lidar else None
    if lidar:
        lidar.start()

    if args.lidar_only:
        try:
            while True:
                time.sleep(0.2)
                if lidar.latest is None:
                    continue
                ang, rng, _ = lidar.latest
                lp = plan_local(ang, rng, 0.0, gap_cfg)
                gaps = ", ".join(f"[{g.start_deg:+.0f}°,{g.end_deg:+.0f}° {g.width_m:.2f}m"
                                 f"{'✓' if g.passable else '✗'}]" for g in lp.gaps)
                hd = "无" if lp.heading_deg is None else f"{lp.heading_deg:+.1f}°"
                print(f"前方最近 {lp.min_front_dist:.2f}m | 期望航向 {hd} | 缝隙 {gaps}")
        except KeyboardInterrupt:
            pass
        finally:
            lidar.stop()
        return

    state = {"gm": None, "persons": [], "goal": None, "new": False}
    lock = threading.Lock()

    def on_map(_, payload):
        gm, persons, msg = decode_map(payload)
        with lock:
            state.update(gm=gm, persons=persons, goal=msg.get("goal"), new=True)

    link = MqttLink(args.mqtt, client_id="car")
    link.subscribe(TOPIC_MAP, on_map)

    car_cam = CarCamera(hfov_deg=args.cam_hfov, mount_height_m=args.cam_height,
                        pitch_down_deg=args.cam_pitch)
    watcher = PersonWatcher(args.person_model, args.camera, car_cam) if args.person_model else None
    if watcher:
        watcher.start()
    tracker = PersonTracker()

    motors = DummyMotorDriver() if args.dry_run else GpioZeroMotorDriver(args.left_pins, args.right_pins)
    ctrl = HeadingController(DriveConfig(max_speed=args.max_speed))
    pose, follower, gm, mapper, goal_cell = None, None, None, None, None
    person_xy = None            # 车载相机确认的人员地面坐标
    dt, next_map_update, arrived = 0.1, 0.0, False

    def replan(reason, quiet=False):
        nonlocal follower, arrived
        arrived = False
        wps, s, g, path = plan_path(gm, pose[:2], goal_cell)
        if wps is None:
            if not quiet:
                print(f"[⚠] 无可行路径（{reason}），等待雷达修正地图后重试。起点{s} 终点{g}")
            follower = None
            return
        follower = PurePursuit(wps)
        length = sum(math.dist(a, b) for a, b in zip(wps[:-1], wps[1:]))
        print(f"[规划·{reason}] 地图 {gm.shape} 起点 {s} -> 终点 {g}，航点 {len(wps)} 个，路径长 {length:.1f} m")
        if args.dry_run and reason == "新地图":
            print(render_ascii(gm.grid, path, s, g, margin=5))

    print("[等待] 等待无人机发布地图 ...")
    try:
        while True:
            t0 = time.time()
            with lock:
                fresh, new_gm, persons, goal_hint = state["new"], state["gm"], state["persons"], state["goal"]
                state["new"] = False
            if fresh:
                gm = new_gm
                mapper = LidarMapper(gm, args.inflate)
                if pose is None:
                    sy = args.start_y if args.start_y is not None else \
                        gm.y_max - gm.shape[0] * gm.resolution + 1.0
                    pose = [args.start_x, sy, math.pi / 2]
                if person_xy is not None:
                    goal_cell = gm.world_to_cell(*person_xy)
                else:
                    goal_cell = tuple(goal_hint) if goal_hint else select_goal(gm, persons)[0]
                replan("新地图")
            if pose is None:
                time.sleep(dt)
                continue

            # 车载相机：多帧确认人员后，以其位置为新终点
            if watcher:
                for t, fix in watcher.take():
                    tracker.add(car_to_world(fix, pose), t)
                found = tracker.confirmed(time.time())
                if found and (person_xy is None or math.dist(found, person_xy) > 1.0):
                    person_xy = found
                    goal_cell = gm.world_to_cell(*person_xy)
                    print(f"[发现] 车载相机确认人员，位置 ({person_xy[0]:.1f}, {person_xy[1]:.1f}) m")
                    replan("人员确认")

            scan = lidar.latest if lidar is not None else None
            # 雷达修正地图，路径被新发现的障碍阻断时重规划
            if scan is not None and time.time() >= next_map_update:
                mapper.update(pose, scan[0], scan[1])
                if follower is not None and mapper.path_blocked(follower.path[follower.idx::5]):
                    replan("路径受阻")
                elif follower is None and not arrived:
                    replan("重试", quiet=True)
                next_map_update = time.time() + 0.5

            if follower is None:
                motors.stop()
                time.sleep(dt)
                continue
            if follower.reached(pose):
                motors.stop()
                print("[✓] 到达终点")
                follower, arrived = None, True
                continue

            target = follower.target_bearing(pose)
            if scan is not None:
                ang, rng, _ = scan
                lp = plan_local(ang, rng, math.degrees(target), gap_cfg)
                if lp.heading_deg is None:
                    v, w = recovery(ang, rng, target, ctrl, dt, gap_cfg)
                else:
                    v, w = ctrl.compute(math.radians(lp.heading_deg), dt, lp.min_front_dist)
                    if lp.emergency_stop:
                        v = 0.0
            else:
                v, w = ctrl.compute(target, dt)
            motors.set(*ctrl.to_wheels(v, w))
            # 航迹推算
            pose[0] += v * math.cos(pose[2]) * dt
            pose[1] += v * math.sin(pose[2]) * dt
            pose[2] = wrap_angle(pose[2] + w * dt)
            link.publish(TOPIC_STATUS, json.dumps({"t": time.time(), "pose": pose,
                                                   "person": person_xy}).encode())
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        motors.close()
        if lidar:
            lidar.stop()
        if watcher:
            watcher.stop()
        link.close()


if __name__ == "__main__":
    main()
