import torch
import time
import argparse
import sys
#python multi_guard.py --gpus 0,2,3 --mem 1.0
def multi_occupy(gpu_ids, mem_gb):
    """
    同时在多张卡上占用显存
    """
    holders = [] # 必须持有引用，防止被释放
    
    print(f"🛡️  准备在以下显卡启动防御: {gpu_ids}")
    
    # 1. 循环申请
    for gpu_id in gpu_ids:
        # 检查ID合法性
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            print(f"❌ 跳过 GPU {gpu_id}: 设备不存在或不可见。")
            continue

        device = torch.device(f"cuda:{gpu_id}")
        props = torch.cuda.get_device_properties(device)
        
        # 计算大小
        num_elements = int(mem_gb * (1024**3) / 4)
        
        print(f"   [GPU {gpu_id} | {props.name}] 正在申请 {mem_gb} GB...")
        
        try:
            # 申请并存入列表
            dummy = torch.empty(num_elements, dtype=torch.float32, device=device)
            holders.append(dummy)
            print(f"   ✅ GPU {gpu_id} 占位成功！")
        except RuntimeError as e:
            print(f"   ❌ GPU {gpu_id} 失败: {e}")

    if not holders:
        print("⚠️  没有成功占用任何显存，程序退出。")
        return

    print(f"\n🔒 已成功锁定 {len(holders)} 张显卡，每张占用 {mem_gb} GB。")
    print("⚡ 按 Ctrl+C 可一次性释放所有显存并退出。")

    # 2. 挂起
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n👋 收到停止指令，正在释放所有显存...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多卡显存批量占位工具")
    # 接收逗号分隔的字符串，如 "0,1,2"
    parser.add_argument('--gpus', type=str, required=True, help="卡号列表，用逗号分隔 (例如: 0,2,3)")
    parser.add_argument('--mem', type=float, required=True, help="每张卡占用的大小(GB) (例如: 2.0)")
    
    args = parser.parse_args()
    
    # 处理输入的字符串 "0,1,2" -> [0, 1, 2]
    try:
        gpu_list = [int(x) for x in args.gpus.split(',')]
    except ValueError:
        print("❌ 错误: 卡号格式不对，请使用逗号分隔的数字，例如 0,1,2")
        sys.exit(1)
        
    multi_occupy(gpu_list, args.mem)