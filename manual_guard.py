import torch
import time
import argparse
import sys

def manual_occupy(gpu_id, mem_gb):
    if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
        print(f"❌ 错误: 找不到 GPU {gpu_id}。")
        return

    device = torch.device(f"cuda:{gpu_id}")
    num_elements = int(mem_gb * (1024**3) / 4)
    
    print(f"⏳ [GPU {gpu_id}] 正在尝试占用 {mem_gb} GB 显存...")
    
    try:
        dummy = torch.empty(num_elements, dtype=torch.float32, device=device)
        print(f"✅ [GPU {gpu_id}] 占用成功！PID: {sys.argv[0]}")
        while True:
            time.sleep(10)
    except RuntimeError as e:
        print(f"❌ [GPU {gpu_id}] 占用失败: {e}")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--mem', type=float, required=True)
    args = parser.parse_args()
    manual_occupy(args.gpu, args.mem)