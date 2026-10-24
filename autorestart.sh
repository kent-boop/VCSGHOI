#!/bin/bash

# ================= 配置区域 =================
# 你要检测的主程序关键词 (能唯一识别出你的进程即可)
TARGET_PROCESS="STIP_main.py"

# 你的启动脚本路径 (建议用绝对路径，或者确保在同级目录)
START_SCRIPT="./run.sh"

# 检测间隔 (秒)
CHECK_INTERVAL=30

# 日志文件 (记录重启时间)
LOG_FILE="autorestart.log"
# ===========================================

echo "🐶 看门狗已启动！正在监控: $TARGET_PROCESS"
echo "日志将写入: $LOG_FILE"

while true; do
    # 使用 pgrep -f 查找包含关键词的进程
    # > /dev/null 表示不输出查到的 PID，只通过返回值判断是否存在
    if pgrep -f "$TARGET_PROCESS" > /dev/null; then
        # 1. 如果进程存在，什么都不做，打印个时间戳证明活着
        current_time=$(date "+%Y-%m-%d %H:%M:%S")
        echo "[$current_time] ✅ $TARGET_PROCESS 正在运行中..."
    else
        # 2. 如果进程不存在，记录日志并重启
        current_time=$(date "+%Y-%m-%d %H:%M:%S")
        echo "----------------------------------------" | tee -a "$LOG_FILE"
        echo "[$current_time] ⚠️  检测到 $TARGET_PROCESS 已停止！" | tee -a "$LOG_FILE"
        echo "[$current_time] 🚀 正在尝试重启..." | tee -a "$LOG_FILE"
        
        # 运行启动脚本
        # nohup ... & 确保启动的新进程在后台运行，不会卡住监控脚本
        nohup bash "$START_SCRIPT" >> "$LOG_FILE" 2>&1 &
        
        # 等待一段时间让程序完成初始化，避免刚启动还没出现进程就又重启
        echo "等待 60 秒让新程序初始化..."
        sleep 60
    fi

    # 休息一会儿再进行下一次检测
    sleep "$CHECK_INTERVAL"
done