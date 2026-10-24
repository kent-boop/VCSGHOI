import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
from pathlib import Path
from PIL import Image

# --- 1. 导入项目依赖 ---
# 确保可以导入当前目录下的模块
sys.path.append(os.getcwd())

try:
    from src.models.stip import STIP  # <--- 直接导入你的类
    from src.data.datasets.hico import make_hico_transforms, HICODetection
    from src.util import box_ops
    # 导入 SAM (根据你的环境调整)
    from src.lm.SAM.build_sam import sam_model_registry
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保你在项目根目录下运行，且 segment_anything 已安装")
    sys.exit(1)

# --- 2. 配置参数 ---
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='/data1/datasets/hico_20160224_det/hico_20160224_det/', type=str)
parser.add_argument('--sam_checkpoint', default='/home/zijian/SAM-BLIP-HOI3/pth/sam/sam_vit_b_01ec64.pth', type=str)
parser.add_argument('--sam_type', default='vit_b', type=str)
args = parser.parse_args()

# --- 3. 可视化辅助函数 ---
def denormalize(tensor):
    # STIP 中使用的 mean/std
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862854, 0.26130258, 0.27577711]).view(3, 1, 1)
    return tensor * std + mean

def main():
    # A. 准备 SAM 模型
    print(f"正在加载 SAM ({args.sam_type})...")
    sam_model = sam_model_registry[args.sam_type](checkpoint=args.sam_checkpoint)
    sam_model.to(device='cuda')

    # B. 准备数据
    root = Path(args.data_path)
    img_folder = root / 'images' / 'train2015'
    anno_file = root / 'annotations' / 'trainval_hico.json'
    action_list = root / 'list_action.txt'
    
    dataset = HICODetection(
        img_set='train', 
        img_folder=img_folder, 
        anno_file=anno_file, 
        action_list_file=action_list,
        transforms=make_hico_transforms('val'),
        num_queries=100
    )

    # C. 测试循环
    # 选取几个样本进行测试
    # test_indices = [0, 15, 30, 45] 
    test_indices = [i for i in range(15)]
    print("开始调用 stip.py 中的 get_mask_from_sam...")

    for i in test_indices:
        img_tensor, target, raw_img = dataset[i]
        
        # 准备 get_mask_from_sam 需要的参数
        # 1. boxes: 需要是归一化的 cxcywh 格式 (HICO dataset加载出来已经是这个格式)
        boxes = target['boxes'].to('cuda')
        
        # 2. size: 原始图片尺寸 (h, w)
        # 注意：dataset 返回的 target['orig_size'] 通常是 tensor，STIP 中可能需要它作为 Tensor 或 numpy
        orig_size = torch.tensor([raw_img.height, raw_img.width]).to('cuda') # H, W
        
        # --- 核心调用 ---
        # 语法：类名.方法名(self占位符, 参数...)
        # 我们传入 None 作为 self，因为该函数内部没有使用 self.属性
        try:
            # 1. 这里需要接收 3 个返回值！
            outputs = STIP.get_mask_from_sam(
                None,           
                sam_model,      
                raw_img,        
                orig_size,      
                boxes           
            )
            
            # 2. 手动解包 (对应你 return 的顺序: masked_instances, boxes_xy, scaled_img_tensor)
            if isinstance(outputs, tuple):
                masked_instances, sam_boxes, scaled_img_tensor = outputs
            else:
                masked_instances = outputs #以此兼容旧版本或单返回值情况
                sam_boxes = None
                
        except AttributeError as e:
            print(f"调用失败: {e}")
            break

        if masked_instances is None:
            print(f"Sample {i}: No output")
            continue
            
        # --- 数据转回 CPU 方便绘图 ---
        masked_instances = masked_instances.cpu() # 现在 masked_instances 已经是 Tensor 了，可以调 .cpu()
        if sam_boxes is not None:
            # 注意：boxes_xy 已经是 numpy 数组了(你在函数里转了)，不需要 .cpu()，但可能需要检查维度
            pass 
        
        # --- 可视化 (修改以适应新的返回值) ---
        num_boxes = min(3, len(masked_instances))
        fig, axs = plt.subplots(num_boxes, 2, figsize=(10, 4 * num_boxes))
        if num_boxes == 1: axs = axs[None, :]

        for k in range(num_boxes):
            # [左图] 显示原始图片 + SAM 使用的 Box
            # 由于你返回了 SAM 实际使用的 boxes_xy，我们可以直接画这个，验证坐标转换是否正确！
            
            # 为了显示方便，我们使用你返回的 scaled_img_tensor (归一化过的) 还原回图片
            # 这样能看到 SAM 真正看到的 384x384 图像是什么样
            debug_img = denormalize(scaled_img_tensor.cpu()).permute(1, 2, 0).numpy()
            debug_img = np.clip(debug_img, 0, 1)
            
            axs[k, 0].imshow(debug_img)
            
            # 画框: 使用你函数返回的 boxes_xy (已经是 384 尺度的绝对坐标了)
            # boxes_xy shape 通常是 [1, N, 4] 或 [N, 4]，根据你 numpy 处理后的结果
            if sam_boxes is not None:
                # 你的代码里：boxes_xy = boxes_xy.cpu().numpy() * scale_fct[None, None, :] 
                # 这会让 shape 变成 (1, N, 4)，所以取 [0, k]
                current_box = sam_boxes[0][k] 
                x0, y0, x1, y1 = current_box
                w_box, h_box = x1 - x0, y1 - y0
                
                rect = plt.Rectangle((x0, y0), w_box, h_box, fill=False, edgecolor='green', linewidth=2, label='SAM Box')
                axs[k, 0].add_patch(rect)
                
            axs[k, 0].set_title(f"SAM Input (384x384) + Box")
            axs[k, 0].axis('off')

            # [右图] 显示 Masked Instance
            out_tensor = masked_instances[k] 
            out_img = denormalize(out_tensor).permute(1, 2, 0).numpy()
            out_img = np.clip(out_img, 0, 1)
            
            axs[k, 1].imshow(out_img)
            axs[k, 1].set_title(f"Masked Output")
            axs[k, 1].axis('off')

        save_path = f"test_sample_{i}_fixed.png"
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Sample {i} 结果已保存: {save_path}")
        plt.close()

if __name__ == "__main__":
    main()