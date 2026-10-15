#!/usr/bin/env python3
"""
空中端节点：摄像头 -> YOLO11 -> GSD 投影栅格地图 -> MQTT 发布给地面小车

示例:
    # 树莓派 CSI 摄像头，同时运行废墟模型与人员模型，持续建图并发布（高度/姿态先用固定值）
    python3 drone_node.py --model best_collapsed.pt best_person.pt --source picamera --altitude 10 --mqtt localhost

    # 用一张航拍图片离线测试（不需要摄像头 / MQTT）
    python3 drone_node.py --model best_collapsed.pt --source test.jpg --altitude 25 --once

    # 姿态与高度由 IMU/气压计节点通过 MQTT 发布到 rescue/pose（JSON: altitude, pitch, roll, yaw）
    python3 drone_node.py --model best_collapsed.pt --source picamera --mqtt localhost --pose-topic rescue/pose

输出: grid.npy（与原 astar 脚本兼容）、grid_map.png（可视化）、grid_meta.json（地图参考信息）
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rescue.comm import TOPIC_MAP, MqttLink, encode_map  # noqa: E402
from rescue.geo_mapping import (CameraModel, DronePose, build_grid_map,  # noqa: E402
                                detections_from_ultralytics, select_goal)


class FrameSource:
    """统一的帧来源：picamera2 / USB 摄像头 / 图片 / 视频，输出 BGR"""

    def __init__(self, source: str, width=640, height=480):
        import cv2
        self.cv2 = cv2
        self.kind = None
        self.image = None
        if source == "picamera":
            from picamera2 import Picamera2
            self.cam = Picamera2()
            self.cam.configure(self.cam.create_preview_configuration(
                main={"format": "RGB888", "size": (width, height)}))
            self.cam.start()
            time.sleep(1)
            self.kind = "picamera"
        elif source.isdigit():
            self.cam = cv2.VideoCapture(int(source))
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.kind = "cv"
        elif Path(source).suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            self.image = cv2.imread(source)
            if self.image is None:
                raise FileNotFoundError(source)
            self.kind = "image"
        else:
            self.cam = cv2.VideoCapture(source)
            self.kind = "cv"

    def read(self):
        if self.kind == "image":
            return self.image.copy()
        if self.kind == "picamera":
            # picamera2 的 "RGB888" 实际按 BGR 字节序输出，可直接给 OpenCV / YOLO 使用
            return self.cam.capture_array()
        ok, frame = self.cam.read()
        return frame if ok else None

    def close(self):
        if self.kind == "picamera":
            self.cam.stop()
        elif self.kind == "cv":
            self.cam.release()


def save_visualization(path, gm, persons, goal, annotated=None):
    import cv2
    vis = np.full((*gm.grid.shape, 3), 255, np.uint8)
    vis[gm.grid == 1] = (170, 170, 170)
    vis[gm.raw == 1] = (40, 40, 40)
    for x, y, _ in persons:
        r, c = gm.world_to_cell(x, y)
        if gm.in_bounds(r, c):
            vis[r, c] = (0, 0, 255)
    if goal is not None:
        vis[goal] = (0, 200, 0)
    scale = max(1, 480 // max(gm.grid.shape))
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    if annotated is not None:
        h = vis.shape[0]
        ann = cv2.resize(annotated, (int(annotated.shape[1] * h / annotated.shape[0]), h))
        vis = np.hstack([ann, vis])
    cv2.imwrite(str(path), vis)


def main():
    ap = argparse.ArgumentParser(description="空中端：YOLO11 检测 -> 栅格地图 -> MQTT")
    ap.add_argument("--model", nargs="+", required=True,
                    help="一个或多个模型，如 best_collapsed.pt（障碍）best_person.pt（人员）")
    ap.add_argument("--source", default="picamera", help="picamera / 摄像头编号 / 图片 / 视频路径")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--hfov", type=float, default=62.2, help="相机水平视场角（度），按实际镜头标定")
    ap.add_argument("--altitude", type=float, default=10.0, help="飞行高度（米）")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--res", type=float, default=0.25, help="栅格分辨率（米/格）")
    ap.add_argument("--inflate", type=float, default=0.3, help="膨胀半径（米）= 小车半宽 + 冗余")
    ap.add_argument("--mqtt", default=None, help="MQTT broker 地址，不填则不发布")
    ap.add_argument("--pose-topic", default=None, help="从该 MQTT 话题读取实时高度/姿态")
    ap.add_argument("--rate", type=float, default=5.0, help="最大发布频率（Hz）")
    ap.add_argument("--once", action="store_true", help="只处理一帧（兼容原“修改后的yolo”用法）")
    ap.add_argument("--out", default=".", help="grid.npy / grid_map.png 输出目录")
    args = ap.parse_args()

    from ultralytics import YOLO
    models = [YOLO(m) for m in args.model]
    for m, path in zip(models, args.model):
        print(f"[✓] {path} 类别: {m.names}")

    pose = DronePose(args.altitude, args.pitch, args.roll)
    link = MqttLink(args.mqtt, client_id="drone") if args.mqtt else None
    if link and args.pose_topic:
        def on_pose(_, payload):
            m = json.loads(payload)
            pose.altitude_m = float(m.get("altitude", pose.altitude_m))
            pose.pitch_deg = float(m.get("pitch", pose.pitch_deg))
            pose.roll_deg = float(m.get("roll", pose.roll_deg))
            pose.yaw_deg = float(m.get("yaw", pose.yaw_deg))
        link.subscribe(args.pose_topic, on_pose)

    src = FrameSource(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n, t_start = 0, time.time()
    try:
        while True:
            t0 = time.time()
            frame = src.read()
            if frame is None:
                break
            h, w = frame.shape[:2]
            cam = CameraModel(w, h, args.hfov)
            results = [m(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0] for m in models]
            t_inf = time.time()
            dets = [d for r in results for d in detections_from_ultralytics(r)]
            res = build_grid_map(dets, cam, pose, args.res, args.inflate)
            gm = res.gridmap
            goal, goal_kind = select_goal(gm, res.persons_xy)
            t_map = time.time()
            np.save(out / "grid.npy", gm.grid)
            (out / "grid_meta.json").write_text(json.dumps(
                {**gm.to_dict(), "gsd_m_per_px": res.gsd, "persons": res.persons_xy,
                 "goal": goal, "goal_kind": goal_kind}, ensure_ascii=False, indent=1))
            if link:
                link.publish(TOPIC_MAP, encode_map(gm, res.persons_xy,
                                                   {"goal": goal, "goal_kind": goal_kind}))
            n += 1
            print(f"[帧 {n}] 检测 {len(dets)} 个，人员 {len(res.persons_xy)} 名 | 地图 {gm.shape} "
                  f"GSD={res.gsd * 100:.1f}cm/px | 推理 {1000 * (t_inf - t0):.0f}ms "
                  f"建图 {1000 * (t_map - t_inf):.1f}ms | 终点 {goal} ({goal_kind})")
            if args.once:
                annotated = frame
                for r in results:
                    annotated = r.plot(img=annotated)
                save_visualization(out / "grid_map.png", gm, res.persons_xy, goal, annotated)
                print(f"[✓] 已保存 {out / 'grid.npy'}、{out / 'grid_map.png'}")
                break
            time.sleep(max(0.0, 1.0 / args.rate - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        if link:
            link.close()
        if n:
            print(f"[统计] {n} 帧，平均建图+发布频率 {n / (time.time() - t_start):.2f} Hz")


if __name__ == "__main__":
    main()
