#!/usr/bin/env python3
"""
空中端位姿节点：BMP388 气压计 -> 相对高度，九轴 IMU（ROS 2 话题）-> 俯仰/横滚，
以 JSON {altitude, pitch, roll, yaw} 发布到 MQTT 话题 rescue/pose，供 drone_node.py --pose-topic 使用。

示例:
    # 只读气压计，姿态取 0（先验证气压测高），同时记录到 CSV
    python3 pose_publisher.py --bmp388 --print --csv baro_test.csv

    # 气压计 + IMU（IMU 厂家 ROS 2 驱动已发布 sensor_msgs/Imu 到 /imu/data）
    python3 pose_publisher.py --bmp388 --ros-imu /imu/data --mqtt localhost

    # 空中端主程序订阅该话题
    python3 drone_node.py --model best_collapsed.pt best_person.pt --mqtt localhost --pose-topic rescue/pose

姿态符号约定与 rescue.geo_mapping.DronePose 一致：pitch>0 表示相机朝机头方向倾斜，roll>0 表示朝右侧倾斜。
IMU 按 ROS 机体坐标系（x 前、y 左、z 上）安装且相机图像上方朝向机头时无需调整；
安装方向不同时用 --pitch-sign / --roll-sign 修正，首次联调请手动倾斜机体核对符号。
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def quaternion_to_rpy_deg(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """四元数 -> (横滚, 俯仰, 航向)，单位度，ZYX 欧拉角"""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


class RosImu:
    """订阅 ROS 2 sensor_msgs/Imu，保存最近一次姿态"""

    def __init__(self, topic: str):
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Imu
        rclpy.init()
        self._rclpy = rclpy
        self.node = Node("rescue_pose_publisher")
        self.rpy = None
        self.node.create_subscription(Imu, topic, self._on_imu, 10)

    def _on_imu(self, msg):
        q = msg.orientation
        self.rpy = quaternion_to_rpy_deg(q.x, q.y, q.z, q.w)

    def poll(self):
        self._rclpy.spin_once(self.node, timeout_sec=0.0)
        return self.rpy


def main():
    ap = argparse.ArgumentParser(description="空中端位姿节点：气压计高度 + IMU 姿态 -> MQTT")
    ap.add_argument("--mqtt", default=None, help="MQTT broker 地址，不填则只打印")
    ap.add_argument("--topic", default="rescue/pose")
    ap.add_argument("--rate", type=float, default=10.0, help="发布频率（Hz）")
    ap.add_argument("--bmp388", action="store_true", help="从 I2C 读取 BMP388 气压计")
    ap.add_argument("--i2c-bus", type=int, default=1)
    ap.add_argument("--i2c-addr", type=lambda s: int(s, 0), default=0x77)
    ap.add_argument("--alt-offset", type=float, default=0.0, help="起飞时相机离地高度（米），加到相对高度上")
    ap.add_argument("--altitude", type=float, default=10.0, help="不接气压计时使用的固定高度（米）")
    ap.add_argument("--ros-imu", default=None, help="IMU 的 ROS 2 话题（sensor_msgs/Imu），如 /imu/data")
    ap.add_argument("--pitch-sign", type=float, default=1.0, choices=[1.0, -1.0])
    ap.add_argument("--roll-sign", type=float, default=1.0, choices=[1.0, -1.0])
    ap.add_argument("--use-yaw", action="store_true", help="同时发布航向角（默认 0，使地图与图像方向一致）")
    ap.add_argument("--print", action="store_true", help="在终端打印每次发布的数据")
    ap.add_argument("--csv", default=None, help="把每次读数追加记录到 CSV 文件（实测数据整理用）")
    args = ap.parse_args()

    altimeter = None
    if args.bmp388:
        from rescue.baro import BMP388, RelativeAltimeter
        print("[*] 正在采集地面参考气压，请保持静止 ...")
        altimeter = RelativeAltimeter(BMP388(args.i2c_bus, args.i2c_addr))
        print(f"[✓] 地面参考气压 {altimeter.ground_hpa:.2f} hPa")
    imu = RosImu(args.ros_imu) if args.ros_imu else None
    link = None
    if args.mqtt:
        from rescue.comm import MqttLink
        link = MqttLink(args.mqtt, client_id="pose_publisher")

    log = None
    if args.csv:
        import csv
        new = not Path(args.csv).exists()
        log_file = open(args.csv, "a", newline="", encoding="utf-8")
        log = csv.writer(log_file)
        if new:
            log.writerow(["t", "altitude_m", "pressure_hpa", "temp_c", "pitch_deg", "roll_deg", "yaw_deg"])

    period = 1.0 / args.rate
    try:
        while True:
            t0 = time.time()
            msg = {"t": t0, "altitude": args.altitude, "pitch": 0.0, "roll": 0.0, "yaw": 0.0}
            if altimeter:
                alt, p, temp = altimeter.altitude()
                msg.update(altitude=alt + args.alt_offset, pressure_hpa=round(p, 3), temp_c=round(temp, 2))
            rpy = imu.poll() if imu else None
            if rpy:
                msg["roll"] = args.roll_sign * rpy[0]
                msg["pitch"] = args.pitch_sign * rpy[1]
                if args.use_yaw:
                    msg["yaw"] = rpy[2]
            if link:
                link.publish(args.topic, json.dumps(msg).encode())
            if log:
                log.writerow([f"{t0:.3f}", f"{msg['altitude']:.3f}", msg.get("pressure_hpa", ""),
                              msg.get("temp_c", ""), f"{msg['pitch']:.2f}", f"{msg['roll']:.2f}", f"{msg['yaw']:.2f}"])
                log_file.flush()
            if args.print or not link:
                print(f"高度 {msg['altitude']:7.2f} m  俯仰 {msg['pitch']:6.2f}°  横滚 {msg['roll']:6.2f}°"
                      + (f"  气压 {msg['pressure_hpa']:.2f} hPa  温度 {msg['temp_c']:.1f} °C" if altimeter else ""))
            time.sleep(max(0.0, period - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if log:
            log_file.close()


if __name__ == "__main__":
    main()
