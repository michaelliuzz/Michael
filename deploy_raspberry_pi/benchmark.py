#!/usr/bin/env python3
"""
难点(1)：YOLO11 在树莓派上的部署优化与稳定性测试

功能:
    export   把 .pt 模型导出为 ONNX / NCNN（可选 FP16、ONNX 动态 INT8 量化）
    speed    对比不同格式、不同输入尺寸的推理速度（FPS、P50/P95 延迟）
    soak     长时间连续运行测试（如 3 小时），每 10 秒记录 FPS、CPU 温度、降频状态、内存

示例（在树莓派上运行，结果才可作为结项数据）:
    python3 benchmark.py export --model best_person.pt --formats onnx ncnn --imgsz 320
    python3 benchmark.py speed --models best_person.pt best_person.onnx best_person_ncnn_model \
        --imgsz 640 320 --frames 200 --source test.jpg --csv speed.csv
    python3 benchmark.py soak --model best_person_ncnn_model --imgsz 320 --hours 3 \
        --source picamera --csv soak.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cpu_temp_c():
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except (OSError, ValueError):
        return None


def throttled():
    """树莓派 vcgencmd get_throttled，非 0 表示发生过欠压或降频"""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2)
        return out.stdout.strip().split("=")[-1]
    except (OSError, subprocess.SubprocessError):
        return None


def rss_mb():
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def frame_source(source, imgsz):
    if source == "random":
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (480, 640, 3), np.uint8)
        return lambda: img
    from drone_node import FrameSource
    src = FrameSource(source)
    return src.read


def cmd_export(args):
    from ultralytics import YOLO
    for fmt in args.formats:
        model = YOLO(args.model)
        kw = {"format": fmt, "imgsz": args.imgsz}
        if args.half and fmt in ("ncnn", "onnx"):
            kw["half"] = True
        path = model.export(**kw)
        print(f"[✓] {fmt}: {path}")
        if fmt == "onnx" and args.int8:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            q = str(Path(path).with_name(Path(path).stem + "_int8.onnx"))
            quantize_dynamic(path, q, weight_type=QuantType.QUInt8)
            print(f"[✓] onnx int8 (动态量化): {q}")


def measure(model, read, imgsz, frames, warmup=10):
    for _ in range(warmup):
        model(read(), imgsz=imgsz, verbose=False)
    lat = []
    t0 = time.perf_counter()
    for _ in range(frames):
        t = time.perf_counter()
        model(read(), imgsz=imgsz, verbose=False)
        lat.append(time.perf_counter() - t)
    total = time.perf_counter() - t0
    lat = np.array(lat) * 1000
    return frames / total, float(np.percentile(lat, 50)), float(np.percentile(lat, 95))


def cmd_speed(args):
    from ultralytics import YOLO
    read = frame_source(args.source, None)
    rows = []
    for m in args.models:
        for sz in args.imgsz:
            model = YOLO(m, task="detect")
            fps, p50, p95 = measure(model, read, sz, args.frames)
            row = {"model": m, "imgsz": sz, "fps": round(fps, 2), "p50_ms": round(p50, 1),
                   "p95_ms": round(p95, 1), "cpu_temp_c": cpu_temp_c()}
            rows.append(row)
            print(row, flush=True)
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def cmd_soak(args):
    from ultralytics import YOLO
    model = YOLO(args.model, task="detect")
    read = frame_source(args.source, args.imgsz)
    end = time.time() + args.hours * 3600
    f = open(args.csv, "w", newline="") if args.csv else None
    w = None
    n_total, t_start = 0, time.time()
    errors = 0
    try:
        while time.time() < end:
            n, t0 = 0, time.time()
            while time.time() - t0 < args.interval:
                try:
                    frame = read()
                    if frame is None:
                        errors += 1
                        continue
                    model(frame, imgsz=args.imgsz, verbose=False)
                    n += 1
                except Exception as e:  # 记录异常但不中断测试
                    errors += 1
                    print(f"[⚠] {e}")
            n_total += n
            row = {"elapsed_s": round(time.time() - t_start), "fps": round(n / (time.time() - t0), 2),
                   "cpu_temp_c": cpu_temp_c(), "throttled": throttled(), "rss_mb": rss_mb(),
                   "errors": errors}
            print(row, flush=True)
            if f:
                if w is None:
                    w = csv.DictWriter(f, fieldnames=list(row))
                    w.writeheader()
                w.writerow(row)
                f.flush()
    except KeyboardInterrupt:
        pass
    finally:
        if f:
            f.close()
        dur = time.time() - t_start
        print(f"[统计] 运行 {dur / 3600:.2f} h，共 {n_total} 帧，平均 {n_total / max(dur, 1e-9):.2f} FPS，"
              f"异常 {errors} 次")


def main():
    ap = argparse.ArgumentParser(description="YOLO11 部署优化与稳定性测试")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", required=True)
    e.add_argument("--formats", nargs="+", default=["onnx", "ncnn"])
    e.add_argument("--imgsz", type=int, default=320)
    e.add_argument("--half", action="store_true")
    e.add_argument("--int8", action="store_true", help="对 ONNX 额外做动态 INT8 量化")
    s = sub.add_parser("speed")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--imgsz", type=int, nargs="+", default=[640, 320])
    s.add_argument("--frames", type=int, default=100)
    s.add_argument("--source", default="random")
    s.add_argument("--csv", default=None)
    k = sub.add_parser("soak")
    k.add_argument("--model", required=True)
    k.add_argument("--imgsz", type=int, default=320)
    k.add_argument("--hours", type=float, default=3.0)
    k.add_argument("--interval", type=float, default=10.0)
    k.add_argument("--source", default="random")
    k.add_argument("--csv", default="soak.csv")
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    {"export": cmd_export, "speed": cmd_speed, "soak": cmd_soak}[args.cmd](args)


if __name__ == "__main__":
    main()
