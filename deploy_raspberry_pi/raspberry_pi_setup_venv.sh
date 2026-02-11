#!/bin/bash
# 树莓派部署脚本 - 虚拟环境版本
# 使用 Python 虚拟环境，不破坏系统环境
# 适用于 Debian trixie/bookworm

set -e  # 任何命令失败时停止

echo "╔════════════════════════════════════════════════════╗"
echo "║  树莓派 YOLO11 - 虚拟环境部署                       ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# 检查 Python 版本
echo "[1/5] 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  发现 Python $python_version"

if ! command -v python3 &> /dev/null; then
    echo "[X] Python3 未安装，请先安装: sudo apt install python3"
    exit 1
fi

# 安装必要的系统包（最小化）
echo "[2/5] 安装最小系统依赖..."
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    libopencv-dev \
    libcap-dev \
    libffi-dev \
    pkg-config \
    python3-picamera2 \
    || echo "  [警告] 部分包安装失败，但可以继续"

echo "  ✓ 核心依赖已安装（包括 picamera2）"

# 创建虚拟环境
echo "[3/5] 创建 Python 虚拟环境..."
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "  虚拟环境已存在，跳过创建"
else
    python3 -m venv $VENV_DIR --system-site-packages
    echo "  ✓ 虚拟环境创建成功: $VENV_DIR/ (允许访问系统包)"
fi

# 激活虚拟环境
echo "[4/5] 激活虚拟环境并安装 Python 依赖..."
source $VENV_DIR/bin/activate

# 升级 pip
echo "  升级 pip..."
pip install --upgrade pip

# 配置国内镜像源（网络波动时更稳定）
PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_FALLBACK="https://mirrors.aliyun.com/pypi/simple"
PIP_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
PIP_FALLBACK_HOST="mirrors.aliyun.com"

# 安装依赖（使用镜像源，失败后自动切换）
echo "  安装 PyTorch 和相关包（可能需要 10-20 分钟）..."
pip install \
    -i $PIP_INDEX --trusted-host $PIP_TRUSTED_HOST \
    torch \
    torchvision \
    ultralytics \
    opencv-python \
    numpy \
    pyyaml \
    pillow \
    || pip install \
        -i $PIP_FALLBACK --trusted-host $PIP_FALLBACK_HOST \
        torch \
        torchvision \
        ultralytics \
        opencv-python \
        numpy \
        pyyaml \
        pillow

echo "  ✓ Python 依赖安装完成"

# 检查安装
echo "[5/5] 验证安装..."
python3 -c "import cv2; print(f'  ✓ OpenCV {cv2.__version__}')" || echo "  [X] OpenCV 导入失败"
python3 -c "import torch; print(f'  ✓ PyTorch {torch.__version__}')" || echo "  [X] PyTorch 导入失败"
python3 -c "from ultralytics import YOLO; print('  ✓ Ultralytics YOLO')" || echo "  [X] YOLO 导入失败"

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  安装完成！                                         ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""
echo "使用方法："
echo ""
echo "1. 激活虚拟环境（每次使用前）："
echo "   source venv/bin/activate"
echo ""
echo "2. 运行检测："
echo "   python3 src/detect_camera_cli.py --model best.pt"
echo ""
echo "3. 退出虚拟环境："
echo "   deactivate"
echo ""
echo "快速启动脚本已更新，可以直接运行："
echo "   bash start_venv.sh"
echo ""
