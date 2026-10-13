#!/usr/bin/env python3
"""
地面端节点：订阅航拍栅格地图 -> A* 全局规划 -> 激光雷达缝隙检测避障 -> 航向控制 -> 电机

示例:
    # 只验证“接收地图 + 规划路径”，不驱动电机、不接雷达（目前无小车实物时使用）
    python3 car_node.py --mqtt localhost --dry-run

    # 真车：RPLIDAR 接 /dev/ttyUSB0，电机驱动板接 GPIO（引脚按实际接线修改）
    python3 car_node.py --mqtt <无人机IP> --lidar /dev/ttyUSB0 --left-pins 17 27 --right-pins 22 23

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
from rescue.geo_mapping import nearest_free, select_goal  # noqa: E402
from rescue.lidar_gap import GapConfig, plan_local, rplidar_to_standard  # noqa: E402
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


def plan_on_map(gm, start_xy, persons, goal_hint=None):
    s = nearest_free(gm.grid, gm.world_to_cell(*start_xy), 10)
    goal = tuple(goal_hint) if goal_hint else select_goal(gm, persons)[0]
    goal = nearest_free(gm.grid, goal, 10) if goal else None
    if s is None or goal is None:
        return None, s, goal
    return astar(gm.grid, s, goal, allow_diag=True), s, goal


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
    args = ap.parse_args()

    gap_cfg = GapConfig(safe_dist=args.safe_dist, car_width=args.car_width)
    lidar = LidarReader(args.lidar) if args.lidar else None
    if lidar:
        lidar.start()

    if args.lidar_only:
        try:
            while True:
                time.sleep(0.2)
                if lidar is None or lidar.latest is None:
                    continue
                ang, rng, _ = lidar.latest
                lp = plan_local(ang, rng, 0.0, gap_cfg)
                gaps = ", ".join(f"[{g.start_deg:+.0f}°,{g.end_deg:+.0f}° {g.width_m:.2f}m"
                                 f"{'✓' if g.passable else '✗'}]" for g in lp.gaps)
                hd = "无" if lp.heading_deg is None else f"{lp.heading_deg:+.1f}°"
                print(f"前方最近 {lp.min_front_dist:.2f}m | 期望航向 {hd} | 缝隙 {gaps}")
        except KeyboardInterrupt:
            lidar.stop()
        return

    state = {"gm": None, "persons": [], "goal": None, "new": False}
    lock = threading.Lock()

    def on_map(_, payload):
        gm, persons, msg = decode_map(payload)
        with lock:
            state.update(gm=gm, persons=persons, goal=msg.get("goal"), new=True)

    if not args.mqtt:
        print("[✗] 需要 --mqtt 指定 broker 以接收地图")
        return
    link = MqttLink(args.mqtt, client_id="car")
    link.subscribe(TOPIC_MAP, on_map)

    motors = DummyMotorDriver() if args.dry_run else GpioZeroMotorDriver(args.left_pins, args.right_pins)
    ctrl = HeadingController(DriveConfig(max_speed=args.max_speed))
    pose = None
    follower = None
    dt = 0.1
    print("[等待] 等待无人机发布地图 ...")
    try:
        while True:
            t0 = time.time()
            with lock:
                fresh, gm, persons, goal_hint = state["new"], state["gm"], state["persons"], state["goal"]
                state["new"] = False
            if fresh:
                if pose is None:
                    sy = args.start_y if args.start_y is not None else \
                        gm.y_max - gm.shape[0] * gm.resolution + 1.0
                    pose = [args.start_x, sy, math.pi / 2]
                path, s, g = plan_on_map(gm, pose[:2], persons, goal_hint)
                if path is None:
                    print(f"[⚠] 无可行路径 起点{s} 终点{g}")
                else:
                    wps = path_to_world(gm, simplify_path(gm.grid, path))
                    wps[0] = tuple(pose[:2])
                    follower = PurePursuit(wps)
                    length = sum(math.dist(a, b) for a, b in zip(wps[:-1], wps[1:]))
                    print(f"[规划] 地图 {gm.shape} 起点 {s} -> 终点 {g}，航点 {len(wps)} 个，"
                          f"路径长 {length:.1f} m")
                    if args.dry_run:
                        print(render_ascii(gm.grid, path, s, g, margin=5))
            if follower is None or pose is None:
                time.sleep(dt)
                continue
            if follower.reached(pose):
                motors.stop()
                print("[✓] 到达终点")
                follower = None
                continue

            target = follower.target_bearing(pose)
            if lidar is not None and lidar.latest is not None:
                ang, rng, _ = lidar.latest
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
            link.publish(TOPIC_STATUS, json.dumps({"t": time.time(), "pose": pose}).encode())
            time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        motors.close()
        if lidar:
            lidar.stop()
        link.close()


if __name__ == "__main__":
    main()
