#!/bin/bash
# 树莓派快速启动脚本 - 虚拟环境版本
# 自动激活虚拟环境并运行检测

echo "YOLO11 人员检测 - 树莓派版本 (虚拟环境)"
echo "=========================================="
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "[错误] 虚拟环境不存在"
    echo "请先运行: bash raspberry_pi_setup_venv.sh"
    exit 1
fi

# 检查模型文件
if [ ! -f "best.pt" ]; then
    echo "[警告] 未找到模型文件 best.pt"
    echo "将使用默认模型（首次运行会自动下载）"
    MODEL_ARG=""
else
    MODEL_ARG="--model best.pt"
    echo "[✓] 找到模型: best.pt"
fi

# 激活虚拟环境
echo "[启动] 激活虚拟环境..."
source venv/bin/activate

# 运行检测
echo "[运行] 开始人员检测..."
echo "按 'q' 停止，按 's' 保存截图"
echo ""

python3 src/detect_camera_cli.py $MODEL_ARG --save

# 退出虚拟环境
deactivate

echo ""
echo "[完成] 检测已停止"
