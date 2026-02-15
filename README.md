# 🍓 树莓派 YOLO11 人员检测部署包

快速部署 YOLO11 人员检测到树莓派 5。

---

## ⚡ 3 步快速部署

### 1️⃣ 自动安装依赖
```bash
bash raspberry_pi_setup_venv.sh
```
自动安装所有依赖并创建虚拟环境。

### 2️⃣ 启动检测
```bash
bash start_venv.sh
```

### 3️⃣ 查看结果
检测结果保存到 `detection_results/` 目录。

---

## 🎯 基础命令

### 实时摄像头检测
```bash
python3 src/detect_camera_cli.py --model best.pt
```

### 检测并保存
```bash
python3 src/detect_camera_cli.py --model best.pt --save
```

### 性能优化模式（无 FPS 显示）
```bash
python3 src/detect_camera_cli.py --model best.pt --no-fps
```

### 运行指定时间（秒）
```bash
python3 src/detect_camera_cli.py --model best.pt --duration 60
```

### 跳帧处理（加快处理）
```bash
python3 src/detect_camera_cli.py --model best.pt --skip-frames 2
```

### 调整置信度
```bash
python3 src/detect_camera_cli.py --model best.pt --conf 0.3
```

---

## 🔧 故障排查

### 摄像头无法识别
```bash
# 检查摄像头连接
ls -la /dev/video*

# 测试摄像头
python3 test_camera.py
```

### 启用摄像头（如需要）
```bash
sudo raspi-config
# 选择: Interface Options → Camera → Enable
# 重启后生效
```

### 内存不足
使用性能优化选项：
```bash
python3 src/detect_camera_cli.py --model best.pt --skip-frames 3 --no-fps
```

### Python 模块缺失
重新安装依赖：
```bash
source venv/bin/activate
pip3 install -r requirements_raspberry_pi.txt
```

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `src/detect_camera_cli.py` | 主检测脚本 |
| `best.pt` | YOLO11 模型 |
| `requirements_raspberry_pi.txt` | 依赖列表 |
| `raspberry_pi_setup_venv.sh` | 自动部署脚本 ⭐ |
| `start_venv.sh` | 启动脚本 |
| `test_camera.py` | 摄像头测试工具 |
| `autostart_detection.sh` | 自启动脚本（可选） |

---

## 🚀 高级功能

### 自启动检测
设置在树莓派启动时自动运行：
```bash
# 编辑 crontab
crontab -e

# 添加以下行（启动后延迟 30 秒）
@reboot sleep 30 && /home/pi/yolo-detection/autostart_detection.sh
```

### 查看日志
```bash
tail -f detection.log
```

### 检查系统资源
```bash
# CPU 和内存使用率
top

# 内存信息
free -h

# 磁盘使用
df -h
```

---

## 📊 性能参考

**树莓派 4B（4GB RAM）**:
- FPS: 2-3（完整模式）
- 内存: 60-85%
- CPU: 70-95%

**优化建议**:
```bash
python3 src/detect_camera_cli.py --model best.pt --skip-frames 2 --no-fps
# 预期: FPS 8-12, 内存 40-60%, CPU 50-70%
```

---

## 🔌 硬件要求

- **树莓派**: 5 或更高版本
- **摄像头**: Raspberry Pi Camera Module 2 或 USB 摄像头
- **电源**: 5V/3A+
- **存储**: 32GB+ MicroSD 卡

### 摄像头连接
1. 打开树莓派上的带状电缆接口
2. 插入相机带状电缆（蓝色端朝向以太网口）
3. 轻轻按压直到固定

---

## 📖 更多信息

**完整部署指南**: 
查看主目录的 [RASPBERRY_PI_DEPLOYMENT.md](../RASPBERRY_PI_DEPLOYMENT.md)

**主项目说明**:
查看主目录的 [README.md](../README.md)

**命令速查**:
查看主目录的 [CHEATSHEET.md](../CHEATSHEET.md)

---

## 💡 常见问题

**Q: 部署脚本失败怎么办？**
A: 尝试手动安装：
```bash
sudo apt-get update
sudo apt-get install -y python3-pip
pip3 install ultralytics opencv-python
```

**Q: 如何更新模型？**
A: 替换 `best.pt` 文件，重启程序即可。

**Q: 能否提高检测准确度？**
A: 降低置信度阈值：`--conf 0.2`（更敏感但可能误检）

**Q: 摄像头很卡怎么办？**
A: 使用跳帧处理：`--skip-frames 3`

**Q: 如何远程访问检测结果？**
A: 使用 SSH 或 SCP 复制文件：
```bash
scp -r pi@your-pi-ip:~/yolo-detection/detection_results ./
```

---

## 🆘 需要帮助？

1. 检查 `test_camera.py` 是否能拍照
2. 查看 `detection.log` 中的错误信息
3. 确保 Python 3.11+ 已安装
4. 检查依赖: `pip3 list | grep torch`

---

**最后更新**: 2026年2月15日  
**部署状态**: ✅ 适配树莓派 5
