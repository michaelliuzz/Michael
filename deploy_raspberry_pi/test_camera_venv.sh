#!/bin/bash
# 测试摄像头脚本 - 虚拟环境版本

if [ ! -d "venv" ]; then
    echo "[错误] 虚拟环境不存在"
    echo "请先运行: bash raspberry_pi_setup_venv.sh"
    exit 1
fi

echo "激活虚拟环境..."
source venv/bin/activate

echo "测试摄像头..."
python3 test_camera.py

deactivate
