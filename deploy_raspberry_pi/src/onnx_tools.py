#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
边缘推理加速工具（难点①：低算力平台部署，目标 >20 FPS）

申报书的加速路线是：PyTorch(.pt) -> 中间格式(ONNX) -> 算子优化/量化
(FP16 / INT8) -> 边缘推理框架。本模块提供一键导出与量化，导出的
模型可直接被 detect_camera_cli.py / detect_to_path.py 的 --model 加载
（Ultralytics 的 YOLO() 原生支持 .onnx 与 ncnn 权重目录，底层走
 onnxruntime / ncnn，无需改动推理代码）。

树莓派 5 为 ARM CPU，实测通常 ncnn > onnx(int8) > onnx(fp32) > pytorch。
因此提供两条导出路径：
  * onnx  —— 通用，支持 FP16 / INT8 动态量化；
  * ncnn  —— 树莓派 CPU 上一般最快，推荐首选。

用法示例：
  # 导出 ncnn（Pi 上最快，推荐）
  python onnx_tools.py export best_person.pt --format ncnn --imgsz 640

  # 导出 ONNX 并做 INT8 动态量化（无需标定集）
  python onnx_tools.py export best_person.pt --format onnx --int8 --imgsz 640

  # 对已有 onnx 单独做动态量化
  python onnx_tools.py quantize best_person.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def export_model(
    pt_path: str,
    fmt: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    int8: bool = False,
    simplify: bool = True,
) -> str:
    """用 Ultralytics 导出 .pt 到 onnx / ncnn。

    Args:
        pt_path:  源 .pt 权重路径。
        fmt:      "onnx" 或 "ncnn"。
        imgsz:    推理输入边长（越小越快，精度略降；Pi 上可试 480/512）。
        half:     FP16（onnx 有效；CPU 上未必更快，GPU/部分加速器有效）。
        int8:     INT8 量化（onnx：Ultralytics 需标定，见 --int8 说明）。
        simplify: 是否用 onnx-simplifier 精简计算图。

    Returns:
        导出产物路径（.onnx 文件或 *_ncnn_model 目录）。
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover - 依赖在 Pi 上安装
        raise SystemExit(
            "需要 ultralytics 才能导出模型：pip install ultralytics"
        ) from e

    src = Path(pt_path)
    if not src.exists():
        raise SystemExit(f"[✗] 权重不存在: {pt_path}")

    model = YOLO(str(src))
    kwargs = dict(format=fmt, imgsz=imgsz, half=half)
    if fmt == "onnx":
        kwargs.update(simplify=simplify, int8=int8, dynamic=False)
    print(f"[导出] {src.name} -> {fmt} (imgsz={imgsz}, half={half}, int8={int8})")
    out = model.export(**kwargs)
    print(f"[✓] 导出完成: {out}")
    return str(out)


def quantize_dynamic_onnx(onnx_path: str, out_path: str | None = None) -> str:
    """对 ONNX 模型做 INT8 动态量化（无需标定数据集）。

    动态量化把权重转为 INT8、激活在运行时量化，是在 ARM CPU 上
    降低内存与延迟最省事的方案；相比需要标定集的静态量化精度略低，
    但足以在树莓派上把帧率推向 >20 FPS 目标。
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:  # pragma: no cover
        raise SystemExit(
            "需要 onnxruntime：pip install onnxruntime onnx"
        ) from e

    src = Path(onnx_path)
    if not src.exists():
        raise SystemExit(f"[✗] onnx 不存在: {onnx_path}")
    dst = Path(out_path) if out_path else src.with_name(src.stem + "_int8.onnx")

    print(f"[量化] {src.name} -> {dst.name} (INT8 动态量化)")
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    print(f"[✓] 量化完成: {dst}")
    return str(dst)


def benchmark(model_path: str, imgsz: int = 640, runs: int = 50) -> float:
    """粗测模型平均推理帧率（在当前机器上），用于验证是否达到 >20 FPS。"""
    try:
        import time

        import numpy as np
        from ultralytics import YOLO
    except ImportError as e:  # pragma: no cover
        raise SystemExit("需要 ultralytics + numpy 才能基准测试") from e

    model = YOLO(model_path)
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    model(dummy, verbose=False)  # 预热
    t0 = time.time()
    for _ in range(runs):
        model(dummy, verbose=False)
    dt = time.time() - t0
    fps = runs / dt if dt > 0 else 0.0
    print(f"[基准] {model_path}: {fps:.1f} FPS (imgsz={imgsz}, {runs} 次)")
    return fps


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO 边缘推理加速：导出/量化/基准")
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("export", help="导出 .pt 到 onnx/ncnn")
    pe.add_argument("pt", help="源 .pt 权重")
    pe.add_argument("--format", default="onnx", choices=["onnx", "ncnn"])
    pe.add_argument("--imgsz", type=int, default=640)
    pe.add_argument("--half", action="store_true", help="FP16 导出")
    pe.add_argument("--int8", action="store_true", help="INT8 导出(onnx)")
    pe.add_argument("--no-simplify", action="store_true")

    pq = sub.add_parser("quantize", help="对已有 onnx 做 INT8 动态量化")
    pq.add_argument("onnx", help="源 .onnx")
    pq.add_argument("-o", "--out", default=None)

    pb = sub.add_parser("benchmark", help="测量模型帧率")
    pb.add_argument("model", help=".pt / .onnx / ncnn 目录")
    pb.add_argument("--imgsz", type=int, default=640)
    pb.add_argument("--runs", type=int, default=50)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "export":
        export_model(
            args.pt,
            fmt=args.format,
            imgsz=args.imgsz,
            half=args.half,
            int8=args.int8,
            simplify=not args.no_simplify,
        )
    elif args.cmd == "quantize":
        quantize_dynamic_onnx(args.onnx, args.out)
    elif args.cmd == "benchmark":
        benchmark(args.model, imgsz=args.imgsz, runs=args.runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
