# 🤖 基于 YOLO11 + A* 的人体识别与路径规划系统

本项目将 **YOLO11 目标检测**与 **A\* 路径规划算法**结合，实现在树莓派上实时识别人体及障碍物，并自动规划可行路径。

---

## 📐 系统架构总览

```
摄像头采集画面
      │
      ▼
 YOLO11 推理
 (人体 / 障碍物检测)
      │
      ▼
 生成栅格地图 grid.npy
 (障碍物 → 格子标记为 1，空闲 → 0)
      │
      ▼
 A* 路径规划
 (在栅格地图上搜索最短可行路径)
      │
      ▼
 输出路径坐标 / 可视化
```

---

## 🧠 一、人体识别过程

### 1. 模型与检测目标

使用 **YOLO11**（`best.pt` / `best_person.pt`）对摄像头帧进行推理，检测目标包括：
- `person`（人体）
- `rock`（岩石）
- `tree`（树木）
- 以及其他自定义障碍类别

### 2. 检测核心流程（`修改后的yolo`）

```python
# 1. 加载 YOLO11 模型
model = YOLO("best.pt")

# 2. 通过树莓派摄像头采集一帧画面
picam2 = Picamera2()
frame = picam2.capture_array()   # 输出 RGB888 格式，尺寸 640×480

# 3. 执行 YOLO 推理，获取所有检测框
results = model(frame, verbose=False)[0]

# 4. 遍历检测结果，将目标区域映射到栅格坐标
GRID_SIZE = 20  # 每个格子对应 20×20 像素
for box in results.boxes:
    cls_name = model.names[int(box.cls[0])]
    if cls_name not in OBSTACLE_CLASSES:
        continue
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    # 将像素坐标转换为栅格坐标并标记为障碍
    grid[y1//GRID_SIZE : y2//GRID_SIZE+1,
         x1//GRID_SIZE : x2//GRID_SIZE+1] = 1

# 5. 保存栅格地图供路径规划使用
np.save("grid.npy", grid)
```

**关键映射关系**：
- 图像分辨率 640×480，栅格大小 20×20 → 栅格地图为 **24×32**
- 检测框像素坐标 `(x1,y1,x2,y2)` 整除 `GRID_SIZE` 得到栅格索引
- 被检测目标覆盖的格子标记为 `1`（障碍），其余为 `0`（可通行）

### 3. 实时摄像头检测（`deploy_raspberry_pi/src/detect_camera_cli.py`）

对于持续视频流场景，`CameraDetector` 类封装了完整的检测循环：

```
打开摄像头（优先 libcamera，备选 OpenCV）
      │
      ▼
  循环读帧
      │
  YOLO 推理 → 绘制检测框 + 置信度
      │
  显示 FPS、检测数量等统计信息
      │
  按 'q' 退出 / 按 's' 截图保存
```

---

## 🗺️ 二、路径规划过程

### 1. 加载栅格地图

```python
grid = np.load("grid.npy")   # 由 YOLO 检测步骤生成
H, W = grid.shape
# 0 = 可通行，1 = 障碍物（人体 / 障碍物区域）
```

### 2. 确定起点与终点

```python
start = (H - 2, W // 2)   # 默认：底部中间（机器人当前位置）
goal  = (1,     W // 2)   # 默认：顶部中间（目标位置）

# 若起点/终点落在障碍物上，自动寻找最近可通行格子
start = nearest_free(grid, start)
goal  = nearest_free(grid, goal)
```

### 3. A* 算法核心（`astar`）

A\* 综合利用**已走代价 g(n)** 和**启发式估计 h(n)**，优先扩展最有希望的节点：

```
f(n) = g(n) + h(n)
```

| 符号 | 含义 |
|------|------|
| `g(n)` | 从起点到节点 n 的实际代价 |
| `h(n)` | 节点 n 到终点的曼哈顿距离（启发函数） |
| `f(n)` | 综合评分，越小越优先扩展 |

**搜索过程**：

```
初始化：将起点加入 open 堆（优先队列）
      │
      ▼
弹出 f 值最小的节点 current
      │
      ├─ 若 current == goal → 回溯 came_from 得到完整路径，结束
      │
      └─ 否则 → 扩展四邻域（或八邻域）
            对每个未访问的可通行邻居 neighbor：
              tentative_g = g(current) + 移动代价(1.0 或 1.414)
              若 tentative_g < g(neighbor)：
                更新 came_from[neighbor] = current
                更新 g(neighbor) = tentative_g
                将 (f, neighbor) 压入 open 堆
```

**启发函数**（4 邻域使用曼哈顿距离）：
```python
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])
```

**支持两种移动模式**：
- **4 邻域**（上下左右，代价均为 1.0）
- **8 邻域**（含对角，对角代价为 √2 ≈ 1.414）

### 4. 路径可视化

找到路径后，将路径标记到可视化矩阵并打印字符地图：

```
图例：
  '.'  →  0  可通行空格
  '#'  →  1  障碍物（YOLO 检测到的人体/障碍）
  'o'  →  2  路径节点
  'S'  →  3  起点
  'G'  →  4  终点
```

示例输出：
```
......S......
......o......
......o......
....##o......
....##o......
......o......
......G......
```

---

## 🔗 完整运行流程

```bash
# 步骤 1：在树莓派上采集一帧并生成栅格地图
python3 修改后的yolo

# 步骤 2：对栅格地图运行 A* 路径规划
python3 astar
```

或使用实时摄像头持续检测：
```bash
python3 deploy_raspberry_pi/src/detect_camera_cli.py --model deploy_raspberry_pi/best_person.pt
```

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
