# 树莓派部署说明

部署时间: 2026-02-08 18:54:02

## 快速部署 (3 步)

### 步骤 1: 上传文件到树莓派

```bash
# 从 Windows 上传（推荐）
scp -r deploy_raspberry_pi/* pi@your-raspberry-ip:~/yolo-detection/

# 或使用 WinSCP/FileZilla
```

### 步骤 2: 连接树莓派并安装依赖

```bash
ssh pi@your-raspberry-ip

cd ~/yolo-detection

# 自动安装所有依赖
bash setup.sh
```

### 步骤 3: 运行检测

```bash
# 基础运行
bash start.sh

# 或使用完整命令
python3 src/detect_camera_cli.py --model best.pt

# 保存检测结果
python3 src/detect_camera_cli.py --model best.pt --save

# 运行 30 秒
python3 src/detect_camera_cli.py --model best.pt --duration 30
```

## 文件清单

```
deploy_raspberry_pi/
├── src/
│   └── detect_camera_cli.py      # 检测脚本
├── requirements_raspberry_pi.txt  # Python 依赖
├── setup.sh                       # 自动部署脚本
├── start.sh                       # 快速启动脚本
├── best.pt                        # YOLO 模型 [已包含]
├── test_camera.py                 # 摄像头测试
├── autostart_detection.sh         # 自启动脚本（可选）
└── README_RASPBERRY_PI.md         # 详细指南
```

## 常见问题

### Q: 模型文件太大？
A: 只需上传 best.pt，其他文件不是必需的

### Q: 自动安装失败？
A: 参考 README_RASPBERRY_PI.md 的"手动部署"部分

### Q: 摄像头无法识别？
A:
```bash
# 启用摄像头
sudo raspi-config
# 选择 Interface 点击 Camera 点击 Enable

# 测试
python3 test_camera.py
```

## 性能提示

树莓派 3B+:
```bash
python3 src/detect_camera_cli.py --model best.pt --no-fps
```

树莓派 4:
```bash
python3 src/detect_camera_cli.py --model best.pt --save
```

## 需要帮助？

- 查看 README_RASPBERRY_PI.md 获得详细说明
- 检查日志: tail -f detection.log
- 查看系统资源: top 或 free -h

祝部署顺利！
