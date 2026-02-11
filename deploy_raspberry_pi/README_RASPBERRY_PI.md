# 树莓派部署指南

## 📋 目录

1. [系统要求](#系统要求)
2. [快速部署](#快速部署)
3. [手动部署](#手动部署)
4. [使用方法](#使用方法)
5. [树莓派摄像头](#树莓派摄像头)
6. [性能优化](#性能优化)
7. [常见问题](#常见问题)

---

## 系统要求

### 硬件
- **树莓派 3B+** 或更新版本（推荐树莓派 4/5）
- **内存**：至少 2GB（推荐 4GB+）
- **存储**：至少 16GB MicroSD 卡（推荐 32GB+）
- **摄像头**：
  - 树莓派 CSI 摄像头

### 软件
- **操作系统**：Raspberry Pi OS
- **Python**：3.13.5
- **依赖**： `requirements_raspberry_pi.txt`

### 网络（初始化时需要）
- 互联网连接（下载依赖和模型）

---

## 快速部署

### 1️⃣ 脚本化一键部署

```bash
# 克隆或复制项目到树莓派
cd ~
# 假设项目在 /path/to/project

# 运行部署脚本
bash raspberry_pi_setup.sh
```

脚本将自动处理：
- ✅ 系统包更新
- ✅ 系统依赖安装
- ✅ Python 依赖安装
- ✅ 模型检查

### ⏱️ 预计时间

| 步骤 | 时间 |
|------|------|
| 系统更新 | 5-10 分钟 |
| 依赖安装 | 10-20 分钟 |
| PyTorch 编译 | 5-10 分钟 |
| **总计** | **20-40 分钟** |

---

## 手动部署

如果脚本部署有问题，可以手动操作：

### 1. 更新系统

```bash
sudo apt update
sudo apt upgrade -y
```

### 2. 安装系统依赖

```bash
sudo apt install -y \
    python3-dev \
    python3-pip \
    python3-numpy \
    libjasper-dev \
    libtiff5 \
    libjasper1 \
    libharfbuzz0b \
    libwebp6 \
    libtk8.6 \
    libatlas-base-dev \
    build-essential
```

### 3. 升级 pip

```bash
python3 -m pip install --upgrade pip
```

### 4. 安装 Python 依赖

```bash
# 方式 1：使用树莓派优化版（推荐）
pip install -r requirements_raspberry_pi.txt

# 方式 2：手动安装（如果 txt 文件不可用）
pip install torch==2.0.1 torchvision==0.15.2 ultralytics opencv-python numpy pyyaml
```

### 5. 验证安装

```bash
python3 -c "from ultralytics import YOLO; print('YOLO 安装成功')"
python3 -c "import cv2; print('OpenCV 安装成功')"
```

---

## 使用方法

所有命令都在项目目录下运行：

```bash
cd ~/yolo11-detection  # 或您的项目目录
```

### 基础用法

#### 模式 1: USB 摄像头（最简单）

```bash
python3 src/detect_camera_cli.py --model best.pt
```

**效果**：
- 打开实时摄像头画面
- 显示检测框和 FPS
- 按 `q` 停止，按 `s` 保存截图

#### 模式 2: 树莓派 CSI 摄像头

```bash
# 需要先安装 picamera2
sudo apt install -y python3-picamera2

# 然后运行
python3 src/detect_camera_cli.py --model best.pt --picamera
```

#### 模式 3: 保存视频

```bash
python3 src/detect_camera_cli.py --model best.pt --save
```

**结果**：自动保存到 `runs/detect_camera/camera_detected_YYYYMMDD_HHMMSS.mp4`

#### 模式 4: 指定运行时长

```bash
# 运行 30 秒，然后自动停止
python3 src/detect_camera_cli.py --model best.pt --duration 30 --save
```

#### 模式 5: 调整参数

```bash
# 置信度 0.35，翻转画面，显示 FPS
python3 src/detect_camera_cli.py \
    --model best.pt \
    --conf 0.35 \
    --flip-h \
    --flip-v
```

### 完整参数列表

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--model` | YOLO 模型路径 | `best.pt` | `best.pt` |
| `--conf` | 置信度 (0-1) | `0.25` | `0.35` |
| `--save` | 保存检测视频 | 不保存 | `--save` |
| `--duration` | 运行时长(秒) | `0`(无限) | `--duration 60` |
| `--picamera` | 使用树莓派摄像头 | 不使用 | `--picamera` |
| `--flip-h` | 水平翻转 | 不翻转 | `--flip-h` |
| `--flip-v` | 垂直翻转 | 不翻转 | `--flip-v` |
| `--no-fps` | 不显示 FPS | 显示 | `--no-fps` |

### 查看帮助

```bash
python3 src/detect_camera_cli.py --help
```

---

## 树莓派摄像头

### CSI 摄像头支持

#### 安装 picamera2

```bash
sudo apt install -y python3-picamera2
pip install picamera2
```

#### 检查摄像头

```bash
libcamera-hello --list-cameras
```

#### 运行检测

```bash
python3 src/detect_camera_cli.py --model best.pt --picamera
```

---

## 性能优化

### 1. 降低分辨率

```bash
# 修改 detect_camera_cli.py 中的这两行：
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # 从 640 改为 320
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # 从 480 改为 240
```

### 2. 使用更小的模型

```bash
# 使用 nano 版本（比 best.pt 快得多）
python3 src/detect_camera_cli.py --model yolov11n.pt
```

### 3. 降低帧率处理

修改 `detect_camera_cli.py` 的这行：

```python
# 每隔 3 帧处理一次
if frame_count % 3 != 0:
    continue
```

### 4. 设置最大内存使用

```bash
# 限制内存使用（例如 512MB）
python3 -X dev src/detect_camera_cli.py --model yolov11n.pt
```

### 性能对比

| 配置 | FPS | 内存 | 推荐场景 |
|------|-----|------|---------|
| yolov11n + 320x240 | 15-25 | ~300MB | 树莓派 3B+ |
| yolov11n + 640x480 | 8-12 | ~400MB | 树莓派 4 |
| best.pt + 640x480 | 3-5 | ~600MB | 树莓派 5 (仅参考) |

---

## 自启动配置

### 使用 systemd（推荐）

#### 1. 创建服务文件

```bash
sudo nano /etc/systemd/system/yolo-detection.service
```

#### 2. 添加以下内容

```ini
[Unit]
Description=YOLO11 Person Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/yolo11-detection
ExecStart=/usr/bin/python3 /home/pi/yolo11-detection/src/detect_camera_cli.py --model best.pt --save
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 3. 启用服务

```bash
sudo systemctl enable yolo-detection.service
sudo systemctl start yolo-detection.service
```

#### 4. 查看状态

```bash
sudo systemctl status yolo-detection.service
sudo journalctl -u yolo-detection.service -f  # 实时日志
```

### 使用 crontab

#### 1. 编辑 crontab

```bash
crontab -e
```

#### 2. 添加自启动命令

```bash
@reboot sleep 30 && cd /home/pi/yolo11-detection && python3 src/detect_camera_cli.py --model best.pt --save >> detection.log 2>&1
```

---

## 常见问题

### Q1: ImportError: No module named 'ultralytics'

**A:** 依赖未安装，运行：
```bash
pip install -r requirements_raspberry_pi.txt
```

### Q2: 摄像头无法打开

**A:** 检查以下几点：
1. 摄像头是否连接
2. 树莓派配置中是否启用了摄像头

```bash
# 启用摄像头接口
sudo raspi-config
# 选择 Interface → Camera → Enable
```

### Q3: 内存不足 (Out of Memory)

**A:** 
- 使用更小的模型：`yolov11n.pt`
- 降低分辨率
- 关闭其他应用

```bash
# 检查内存使用
free -h
```

### Q4: 检测速度太慢 (FPS 很低)

**A:**
- 使用 nano 模型
- 降低置信度（会更快但检测可能不准）
- 考虑使用树莓派 4/5（树莓派 3B+ 性能有限）

### Q5: 如何停止自启动检测？

**A:** 如果使用 systemd 服务：
```bash
sudo systemctl stop yolo-detection.service
sudo systemctl disable yolo-detection.service
```

### Q6: 检测结果视频无法播放

**A:** 某些播放器可能不支持 mp4v 编码，尝试转换格式：
```bash
ffmpeg -i input.mp4 -c:v libx264 output.mp4
```

### Q7: 树莓派摄像头无法识别

**A:** 检查 picamera2 是否安装：
```bash
python3 -c "import picamera2; print('OK')"

# 如果失败，安装：
sudo apt install -y python3-picamera2
```

### Q8: GPIO/硬件加速支持

**A:** 目前脚本不支持树莓派的硬件加速（H.264 编码等），但可以扩展。联系开发者了解更多。

---

## 性能调整建议

### 树莓派 3B+

```bash
# 使用最小化配置
python3 src/detect_camera_cli.py \
    --model yolov11n.pt \
    --no-fps
```

### 树莓派 4

```bash
# 平衡配置
python3 src/detect_camera_cli.py \
    --model yolov11n.pt \
    --conf 0.3
```

### 树莓派 5

```bash
# 完整配置
python3 src/detect_camera_cli.py \
    --model best.pt \
    --save \
    --picamera
```

---

## 故障排除步骤

### 当遇到问题时：

1. **查看日志**
   ```bash
   python3 src/detect_camera_cli.py --model best.pt 2>&1 | tee debug.log
   cat debug.log  # 检查错误
   ```

2. **测试摄像头**
   ```bash
   python3 test_camera.py
   ```

3. **检查系统资源**
   ```bash
   top  # CPU 使用情况
   free -h  # 内存
   df -h  # 磁盘空间
   ```

4. **重新安装依赖**
   ```bash
   pip install --upgrade -r requirements_raspberry_pi.txt
   ```

---


