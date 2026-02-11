#!/bin/bash
# 树莓派自启动脚本
# 安装方式：
#   1. chmod +x autostart_detection.sh
#   2. crontab -e
#   3. 添加：@reboot /path/to/autostart_detection.sh

# 配置参数
PROJECT_DIR="/home/pi/yolo11-detection"  # 修改为您的项目目录
MODEL_PATH="best.pt"
DURATION=0  # 0=持续运行，或指定秒数
LOG_FILE="$PROJECT_DIR/detection.log"

# 等待系统启动完成
sleep 30

# 启动检测
cd $PROJECT_DIR

echo "启动检测系统..." >> $LOG_FILE
echo "时间: $(date)" >> $LOG_FILE

python3 src/detect_camera_cli.py \
    --model $MODEL_PATH \
    --duration $DURATION \
    --save \
    >> $LOG_FILE 2>&1

echo "检测系统已停止" >> $LOG_FILE
echo "时间: $(date)" >> $LOG_FILE
