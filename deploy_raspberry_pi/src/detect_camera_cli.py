#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
树莓派专用版本 - YOLO11 摄像头实时检测（无 GUI，性能优化）
轻量化设计，专为树莓派等资源受限的设备优化
"""

import cv2
from ultralytics import YOLO
from pathlib import Path
import argparse
import time
import sys

# 尝试导入 libcamera（新树莓派系统）
try:
    from picamera2 import Picamera2
    from picamera2.encoders import JpegEncoder
    from picamera2.outputs import FileOutput
    LIBCAMERA_AVAILABLE = True
except ImportError:
    LIBCAMERA_AVAILABLE = False

class CameraDetector:
    """树莓派摄像头检测器"""
    
    def __init__(self, model_path, conf=0.25, use_picamera=False):
        """
        初始化检测器
        
        Args:
            model_path: YOLO 模型路径
            conf: 置信度阈值 (0-1)
            use_picamera: 是否使用树莓派专用摄像头接口
        """
        self.model_path = model_path
        self.conf = conf
        self.use_picamera = use_picamera
        
        # 加载模型
        print("[加载中] 正在加载 YOLO 模型...")
        try:
            self.model = YOLO(model_path)
            print("[✓] 模型加载成功")
        except Exception as e:
            print(f"[✗] 模型加载失败: {e}")
            sys.exit(1)
    
    def open_camera(self):
        """打开摄像头"""
        print("[加载中] 正在打开摄像头...")
        
        # 优先尝试 libcamera（新树莓派系统）
        if LIBCAMERA_AVAILABLE:
            try:
                from picamera2 import Picamera2
                
                picam2 = Picamera2()
                config = picam2.create_preview_configuration(
                    main={"format": 'XRGB8888', "size": (640, 480)}
                )
                picam2.configure(config)
                picam2.start()
                print("[✓] 树莓派 libcamera 已打开 (640x480)")
                return picam2, True
            except Exception as e:
                print(f"[⚠] libcamera 打开失败: {e}")
                print("[⚠] 尝试使用 OpenCV...")
        
        # 备选：标准 OpenCV 方式（USB 摄像头或旧系统）
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("[✗] 无法打开摄像头")
            sys.exit(1)
            sys.exit(1)
        
        # 设置分辨率和帧率
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"[✓] 摄像头已打开 ({width}x{height}, {fps} FPS)")
        return cap, False
    
    def run(self, save_video=False, duration=0, show_fps=True, flip_h=False, flip_v=False):
        """
        运行实时检测
        
        Args:
            save_video: 是否保存检测结果视频
            duration: 运行时长（秒），0 表示持续运行
            show_fps: 是否显示 FPS
            flip_h: 水平翻转
            flip_v: 垂直翻转
        """
        camera, is_picamera = self.open_camera()
        
        # 准备输出视频
        out = None
        output_path = None
        if save_video:
            output_dir = Path("runs/detect_camera")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"camera_detected_{timestamp}.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 30, (640, 480))
            print(f"[✓] 输出视频: {output_path}")
        
        # 主循环
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        loop_start = time.time()
        
        print("\n[运行中] 开始实时检测...")
        print("按 'q' 停止，按 's' 保存截图\n")
        
        try:
            while True:
                # 读取帧
                if is_picamera:
                    array = camera.capture_array()
                    frame = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame = camera.read()
                    if not ret:
                        break
                
                frame_count += 1
                
                # 翻转（如需要）
                if flip_h:
                    frame = cv2.flip(frame, 1)
                if flip_v:
                    frame = cv2.flip(frame, 0)
                
                # 运行检测
                results = self.model(frame, conf=self.conf, verbose=False)
                detections = results[0]
                num_detections = len(detections.boxes)
                total_detections += num_detections
                
                # 绘制检测结果
                annotated = detections.plot()
                
                # 计算 FPS
                elapsed_total = time.time() - start_time
                fps = frame_count / elapsed_total if elapsed_total > 0 else 0
                
                # 添加信息文本
                if show_fps:
                    info_text = f"Frame: {frame_count} | Detections: {num_detections} | FPS: {fps:.1f}"
                    cv2.putText(annotated, info_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示帧
                cv2.imshow("YOLO11 Person Detection", annotated)
                
                # 保存帧
                if out:
                    out.write(annotated)
                
                # 检查互动和退出条件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[停止] 用户停止")
                    break
                elif key == ord('s'):
                    # 保存截图
                    screenshot_dir = Path("runs/screenshots")
                    screenshot_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    screenshot_path = screenshot_dir / f"screenshot_{timestamp}.jpg"
                    cv2.imwrite(str(screenshot_path), annotated)
                    print(f"[✓] 截图已保存: {screenshot_path}")
                
                # 检查时长限制
                if duration > 0 and (time.time() - loop_start) > duration:
                    print(f"\n[完成] 达到设定时长 ({duration}s)")
                    break
        
        except KeyboardInterrupt:
            print("\n[中断] Ctrl+C")
        
        finally:
            # 清理资源
            if is_picamera:
                camera.stop()
            else:
                camera.release()
            
            if out:
                out.release()
            
            cv2.destroyAllWindows()
            
            # 统计信息
            total_time = time.time() - start_time
            print("\n" + "="*50)
            print("[统计]")
            print(f"  总帧数: {frame_count}")
            print(f"  总检测人数: {total_detections}")
            print(f"  平均人数/帧: {total_detections/frame_count if frame_count > 0 else 0:.2f}")
            print(f"  平均 FPS: {frame_count/total_time if total_time > 0 else 0:.1f}")
            print(f"  总耗时: {total_time:.1f}s")
            if output_path:
                print(f"  结果视频: {output_path}")
            print("="*50)


def main():
    parser = argparse.ArgumentParser(
        description="树莓派专用 YOLO11 摄像头实时检测 (无 GUI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础使用 (USB 摄像头)
  python detect_camera_cli.py --model best.pt
  
  # 树莓派 CSI 摄像头
  python detect_camera_cli.py --model best.pt --picamera
  
  # 保存结果，运行 30 秒
  python detect_camera_cli.py --model best.pt --save --duration 30
  
  # 调整置信度，翻转视频
  python detect_camera_cli.py --model best.pt --conf 0.35 --flip-h --flip-v
        """
    )
    
    parser.add_argument('--model', type=str, default='best.pt',
                       help='YOLO 模型路径 (默认: best.pt)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='置信度阈值 0-1 (默认: 0.25)')
    parser.add_argument('--save', action='store_true',
                       help='保存检测结果视频')
    parser.add_argument('--duration', type=int, default=0,
                       help='运行时长（秒），0 为持续运行 (默认: 0)')
    parser.add_argument('--picamera', action='store_true',
                       help='使用树莓派 CSI 摄像头接口 (picamera2)')
    parser.add_argument('--flip-h', action='store_true',
                       help='水平翻转视频')
    parser.add_argument('--flip-v', action='store_true',
                       help='垂直翻转视频')
    parser.add_argument('--no-fps', action='store_true',
                       help='不显示 FPS')
    
    args = parser.parse_args()
    
    # 验证模型文件
    if not Path(args.model).exists():
        print(f"[✗] 模型文件不存在: {args.model}")
        sys.exit(1)
    
    # 创建检测器并运行
    detector = CameraDetector(
        model_path=args.model,
        conf=args.conf,
        use_picamera=args.picamera
    )
    
    detector.run(
        save_video=args.save,
        duration=args.duration,
        show_fps=not args.no_fps,
        flip_h=args.flip_h,
        flip_v=args.flip_v
    )


if __name__ == "__main__":
    main()
