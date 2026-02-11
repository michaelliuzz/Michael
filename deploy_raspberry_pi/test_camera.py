#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速摄像头测试脚本
用于验证摄像头是否能正常访问
"""

import cv2
import sys

def test_camera():
    """测试摄像头"""
    print("正在测试摄像头...")
    
    # 尝试打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误：无法打开摄像头！")
        print("请检查：")
        print("  1. 摄像头是否已连接")
        print("  2. 其他程序是否占用了摄像头")
        print("  3. 摄像头权限是否开启")
        return False
    
    # 获取摄像头属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    
    print(f" 摄像头访问成功！")
    print(f"   分辨率: {width}x{height}")
    print(f"   帧率: {fps} FPS")
    
    # 读取几帧来进一步验证
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"   ✓ 帧 {i+1}: 成功读取 ({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"   帧 {i+1}: 读取失败")
            cap.release()
            return False
    
    cap.release()
    print("\n摄像头测试完成！可以开始使用。")
    return True

if __name__ == "__main__":
    success = test_camera()
    sys.exit(0 if success else 1)
