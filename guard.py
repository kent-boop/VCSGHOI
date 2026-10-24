import torch
import time
import sys

def hold_memory(gib=1.0):
    try:
        # 1. 检测当前可见的 GPU 数量 (由 bash 的 CUDA_VISIBLE_DEVICES 决定)
        gpu_count = torch.cuda.device_count()
        print(f"【Guard】检测到 {gpu_count} 张显卡可用。")

        holders = [] # 必须把tensor存起来，防止被垃圾回收

        # 2. 循环对每一张卡进行占位
        for i in range(gpu_count):
            device = torch.device(f'cuda:{i}')
            
            # 计算元素数量
            num_elements = int(gib * 1024 * 1024 * 1024 / 4)
            
            print(f"【Guard】正在显卡 cuda:{i} 上申请 {gib} GB 显存...")
            
            # 申请并存入列表
            dummy = torch.empty(num_elements, dtype=torch.float32, device=device)
            holders.append(dummy)
            
            print(f"【Guard】显卡 cuda:{i} 占位成功！")

        print(f"【Guard】所有显卡占位完毕 (PID: {sys.argv[0]})。进入守护模式...")
        
        # 3. 死循环挂起
        while True:
            time.sleep(10)
            
    except Exception as e:
        print(f"【Guard】占位发生错误: {e}")

if __name__ == '__main__':
    # 这里设置每张卡占用的显存大小
    # 如果你想总共占位防御，建议稍微小一点，比如 0.8 或者 1.0
    hold_memory(gib=0.8)