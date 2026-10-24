#!/bin/bash
#bash start_guard.sh
# ================= 配置区 =================
# 在这里写你想占哪些卡，多少G
# 格式：python manual_guard.py --gpu 卡号 --mem 大小 &

echo "🛡️  正在批量启动显存防御..."


python manual_guard.py --gpu 7 --mem 1.0 &
PID_LIST+="$! "  # 记录 PID 方便后面杀

python manual_guard.py --gpu 6 --mem 1.0 &
PID_LIST+="$! "

python manual_guard.py --gpu 5 --mem 1.0 &
PID_LIST+="$! "

python manual_guard.py --gpu 4 --mem 1.0 &
PID_LIST+="$! "
python manual_guard.py --gpu 3 --mem 1.0 &
PID_LIST+="$! "
python manual_guard.py --gpu 2 --mem 1.0 &
PID_LIST+="$! "
python manual_guard.py --gpu 1 --mem 1.0 &
PID_LIST+="$! "
python manual_guard.py --gpu 0 --mem 1.0 &
PID_LIST+="$! "
# =======================
=================

echo "✅ 所有防御进程已在后台启动 (PIDs: $PID_LIST)"
echo "🔴 按任意键(或 Ctrl+C) 停止所有防御并释放显存..."

# 定义清理函数：当脚本退出时，杀掉所有后台进程
cleanup() {
    echo ""
    echo "👋 正在清理所有占位进程..."
    kill $PID_LIST 2>/dev/null
    echo "✅ 清理完毕。"
    exit
}

# 捕获退出信号
trap cleanup SIGINT SIGTERM EXIT

# 等待用户输入，保持脚本运行
read -p "Press any key to stop..."