# 树莓派虚拟环境部署

## 问题：系统包不可用

如果你遇到：
- `Unable to locate package libjasper-dev`
- `Package 'libatlas-base-dev' has no installation candidate`
- Debian trixie/testing 包名已改变

## 解决方案：使用 Python 虚拟环境

虚拟环境的优势：
- ✅ 不依赖系统包，不会破坏系统 Python
- ✅ 独立环境，干净隔离
- ✅ 可以随时删除重建
- ✅ 适用于任何 Linux 发行版

---

## 部署步骤（3 步）

### 第 1 步：上传文件到树莓派

### 第 2 步：运行虚拟环境部署脚本

```bash
cd ~/yolo-detection

# 使用虚拟环境版本
bash raspberry_pi_setup_venv.sh
```

这将：
- 安装最小系统依赖（python3-pip, python3-venv, python3-dev）
- 创建 Python 虚拟环境 `venv/`
- 在虚拟环境中安装所有 Python 包
- 不破坏系统 Python 环境

**耗时**：10-20 分钟

### 第 3 步：运行检测

```bash
# 使用快速启动脚本（自动激活虚拟环境）
bash start_venv.sh
```

或手动运行：

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行检测
python3 src/detect_camera_cli.py --model best.pt --save

# 退出虚拟环境
deactivate
```

---

## 文件清单

新增的虚拟环境相关文件：

```
deploy_raspberry_pi/
├── raspberry_pi_setup_venv.sh   ⭐ 虚拟环境部署脚本
├── start_venv.sh                ⭐ 虚拟环境快速启动
├── test_camera_venv.sh          ⭐ 虚拟环境摄像头测试
├── raspberry_pi_setup.sh        (旧版，可能在 trixie 上失败)
├── start.sh                     (旧版)
└── ... (其他文件)
```

---

## 常用命令

### 激活虚拟环境
```bash
source venv/bin/activate
```

### 运行检测
```bash
# 基础运行
python3 src/detect_camera_cli.py --model best.pt

# 保存结果
python3 src/detect_camera_cli.py --model best.pt --save

# 运行 30 秒
python3 src/detect_camera_cli.py --model best.pt --duration 30 --save

# 树莓派 CSI 摄像头
python3 src/detect_camera_cli.py --model best.pt --picamera
```

### 退出虚拟环境
```bash
deactivate
```

### 测试摄像头
```bash
bash test_camera_venv.sh
```

---

## 虚拟环境管理

### 查看虚拟环境安装的包
```bash
source venv/bin/activate
pip list
deactivate
```

### 重建虚拟环境
```bash
# 删除旧环境
rm -rf venv/

# 重新部署
bash raspberry_pi_setup_venv.sh
```

### 虚拟环境大小
大约 500-800 MB（包含 PyTorch）

---

## 对比：旧版 vs 虚拟环境版

| 方面 | raspberry_pi_setup.sh | raspberry_pi_setup_venv.sh |
|------|---------------------|--------------------------|
| **系统包依赖** | 需要安装很多系统包 | 只需要最小依赖 |
| **兼容性** | Debian bookworm | 任何 Linux 发行版 ✓ |
| **系统污染** | 可能破坏系统 Python | 完全隔离 ✓ |
| **卸载** | 难以清理 | `rm -rf venv/` 即可 ✓ |
| **适用场景** | 稳定版系统 | Testing/不稳定系统 ✓ |

---

## 故障排除

### Q: 虚拟环境激活失败

**A:** 检查是否已创建：
```bash
ls -l venv/bin/activate
bash raspberry_pi_setup_venv.sh  # 重新创建
```

### Q: PyTorch 安装太慢

**A:** 树莓派编译需要时间，耐心等待 10-20 分钟

### Q: 内存不足

**A:** 树莓派 3B+ 编译 PyTorch 可能需要增加 swap：
```bash
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # 修改 CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

### Q: 想删除虚拟环境

**A:** 简单删除文件夹即可：
```bash
rm -rf venv/
```

---

## 在树莓派上的完整流程

```bash
# 1. 上传文件（在 Windows 上）
scp -r deploy_raspberry_pi/* pi@树莓派IP:~/yolo-detection/

# 2. 连接到树莓派
ssh pi@树莓派IP

# 3. 进入目录
cd ~/yolo-detection

# 4. 运行虚拟环境部署
bash raspberry_pi_setup_venv.sh

# 5. 测试摄像头（可选）
bash test_camera_venv.sh

# 6. 运行检测
bash start_venv.sh
```

---

## 推荐使用场景

### 使用虚拟环境版本（raspberry_pi_setup_venv.sh）当：
- ✅ Debian trixie/testing/sid
- ✅ 系统包不可用或名称改变
- ✅ 不想破坏系统 Python
- ✅ 需要独立环境
- ✅ 准备实验和测试

### 使用传统版本（raspberry_pi_setup.sh）当：
- Debian bookworm (stable)
- 所有系统包都可用
- 生产环境

---

**现在你可以：**

1. 将更新后的 `deploy_raspberry_pi/` 文件夹上传到树莓派
2. 运行 `bash raspberry_pi_setup_venv.sh`
3. 等待 10-20 分钟安装完成
4. 运行 `bash start_venv.sh` 开始检测

祝部署顺利！🚀
