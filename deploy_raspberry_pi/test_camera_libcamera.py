#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速摄像头测试脚本（支持 libcamera 和 OpenCV）
用于验证摄像头是否能正常访问
"""

import sys
import time

def test_libcamera():
    """使用 libcamera 测试"""
    print("\n方式 1: 测试 libcamera...")
    try:
        from picamera2 import Picamera2
        
        print("  初始化 libcamera...")
        picam2 = Picamera2()
        
        # 获取可用的配置
        config = picam2.create_preview_configuration(
            main={"format": 'XRGB8888', "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        
        # 读取几帧
        print("  读取摄像头数据...")
        for i in range(5):
            array = picam2.capture_array()
            print(f"    ✓ 帧 {i+1}: 成功 ({array.shape})")
        
        picam2.stop()
        print("libcamera 测试成功！\n")
        return True
        
    except Exception as e:
        print(f"libcamera 失败: {e}\n")
        return False

def test_opencv():
    """使用 OpenCV 测试"""
    print("方式 2: 测试 OpenCV (V4L2)...")
    try:
        import cv2
        
        print("  打开摄像头...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("OpenCV 无法打开摄像头\n")
            return False
        
        # 获取属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        print(f"  分辨率: {width}x{height}")
        print(f"  帧率: {fps} FPS")
        
        # 读取几帧
        print("  读取摄像头数据...")
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"    ✓ 帧 {i+1}: 成功 ({frame.shape})")
            else:
                print(f"    帧 {i+1}: 读取失败")
                cap.release()
                return False
        
        cap.release()
        print("OpenCV 测试成功！\n")
        return True
        
    except Exception as e:
        print(f"OpenCV 失败: {e}\n")
        return False

def main():
    print("╔════════════════════════════════════════════════════╗")
    print("║  树莓派摄像头全面测试                              ║")
    print("╚════════════════════════════════════════════════════╝\n")
    
    libcamera_ok = test_libcamera()
    opencv_ok = test_opencv()
    
    print("─" * 50)
    print("\n 测试结果：\n")
    print(f"  libcamera: {'正常' if libcamera_ok else '失败'}")
    print(f"  OpenCV:    {'正常' if opencv_ok else '失败'}\n")
    
    if libcamera_ok:
        print("推荐使用 libcamera（新树莓派系统标准）")
        print("   脚本会自动优先使用 libcamera\n")
    elif opencv_ok:
        print("可以使用 OpenCV (V4L2)")
        print("   但新系统推荐升级到 libcamera\n")
    else:
        print("   摄像头无法访问！")
        print("   建议检查：")
        print("   1. CSI 排线是否插紧")
        print("   2. raspi-config 是否启用摄像头")
        print("   3. 运行: vcgencmd get_camera")
        sys.exit(1)

if __name__ == "__main__":
    main()
