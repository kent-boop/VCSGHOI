import os
# =========================================================
#  1. 显卡设置 (必须在 import torch 前设置)
# =========================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 指定使用 1 号卡

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import argparse

# 导入项目模块
# 确保 src 在 python path 中
from src.models import build_model
from src.engine.arg_parser import get_args_parser
from src.util.misc import nested_tensor_from_tensor_list

# ================= 配置区域 =================
# 权重路径 (对应 run.sh 中的 output_dir 下的 checkpoint)
CHECKPOINT_PATH = '/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/checkpoints/hico-det4/STIP_opensource/hico_run1/best.pth' 
# 测试图片
IMAGE_PATH = '/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/visual2/HICO_train2015_00000127.jpg' 
# 输出目录
SAVE_DIR = 'vis_outputs'
os.makedirs(SAVE_DIR, exist_ok=True)
# ===========================================

def get_args():
    parser = argparse.ArgumentParser('Visualization Script', parents=[get_args_parser()])
    
    # ------------------ 参数补全区 ------------------
    parser.add_argument('--detr_weights', default=None, type=str)
    parser.add_argument('--train_detr', action='store_true', default=False)
    parser.add_argument('--finetune_detr_weight', default=0.1, type=float)
    parser.add_argument('--lr_detr', default=1e-5, type=float)
    parser.add_argument('--reduce_lr_on_plateau_patience', default=2, type=int)
    parser.add_argument('--reduce_lr_on_plateau_factor', default=0.1, type=float)
    parser.add_argument('--proposal_focal_loss_alpha', default=0.75, type=float)
    parser.add_argument('--action_focal_loss_alpha', default=0.5, type=float)
    parser.add_argument('--proposal_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--action_focal_loss_gamma', default=2, type=float)
    parser.add_argument('--proposal_loss_coef', default=1, type=float)
    parser.add_argument('--action_loss_coef', default=1, type=float)
    parser.add_argument('--action_sampled_loss_coef', default=1.0, type=float)
    parser.add_argument('--action_fore_loss_coef', default=1.0, type=float)
    parser.add_argument('--global_action_fore_loss_coef', default=1.0, type=float)
    parser.add_argument('--action_fea_loss_coef', default=1.0, type=float)
    parser.add_argument('--no_hard_mining_for_relation_discovery', dest='use_hard_mining_for_relation_discovery', action='store_false', default=True)
    parser.add_argument('--no_relation_dependency_encoding', dest='use_relation_dependency_encoding', action='store_false', default=True)
    parser.add_argument('--no_memory_layout_encoding', dest='use_memory_layout_encoding', action='store_false', default=True)
    parser.add_argument('--no_nms_on_detr', dest='apply_nms_on_detr', action='store_false', default=True)
    parser.add_argument('--no_tail_semantic_feature', dest='use_tail_semantic_feature', action='store_false', default=True)
    parser.add_argument('--no_spatial_feature', dest='use_spatial_feature', action='store_false', default=True)
    parser.add_argument('--no_interaction_decoder', action='store_true', default=False)
    parser.add_argument('--use_memory_union_mask', action='store_true', default=False)
    parser.add_argument('--use_union_feature', action='store_true', default=False)
    parser.add_argument('--adaptive_relation_query_num', action='store_true', default=False)
    parser.add_argument('--use_relation_tgt_mask', action='store_true', default=False)
    parser.add_argument('--use_relation_tgt_mask_attend_topk', default=10, type=int)
    parser.add_argument('--use_prior_verb_label_mask', action='store_true', default=False)
    parser.add_argument('--relation_feature_map_from', default='backbone')
    parser.add_argument('--use_query_fourier_encoding', action='store_true', default=False)
    parser.add_argument('--proj_blip', action='store_true', default=True)
    parser.add_argument('--merge_blip', action='store_true', default=True)
    parser.add_argument('--only_blip', action='store_true', default=False)
    parser.add_argument('--use_moe_visual_router', action='store_true', default=True)
    parser.add_argument('--use_moe_aux_loss', action='store_true', default=True)
    parser.add_argument('--use_moe_noisy_gating', action='store_true', default=True)
    # --------------------------------------------------

    args = parser.parse_args([])
    args.dataset_file = 'hico-det'
    args.data_path = '/data1/datasets/hico_20160224_det/hico_20160224_det/'
    args.backbone = 'resnet50'
    args.device = 'cuda'
    args.batch_size = 1
    args.num_hoi_queries = 32
    args.HOIDet = True
    args.aux_loss = False
    args.hoi_aux_loss = True
    
    # Dataset constants
    args.num_classes = 91 
    args.num_actions = 117
    args.valid_obj_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
        22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
        43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
        62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 
        85, 86, 87, 88, 89, 90
    ]
    args.STIP_relation_head = True
    args.share_enc = False
    args.pretrained_dec = False
    
    return args

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")
    raw_img = Image.open(image_path).convert('RGB')
    img_tensor = transform(raw_img).unsqueeze(0).to('cuda')
    return raw_img, img_tensor

def save_tensor_images(tensors, prefix="crop", mean=None, std=None):
    """
    将 Tensor 批次保存为图片
    Args:
        tensors: (B, C, H, W) 形状的 Tensor
        prefix: 保存文件的前缀名
        mean, std: 用于反归一化的均值和方差 (CLIP 和 ImageNet 不同)
    """
    if tensors is None:
        return

    # 默认使用 CLIP 的归一化参数 (stip.py 中 blip_transform 使用的参数)
    if mean is None:
        mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
        std = torch.tensor((0.26862854, 0.26130258, 0.27577711)).view(3, 1, 1)
    
    # 反归一化: pixel = (tensor * std) + mean
    # 这里的 tensors 已经在 cpu 上了
    tensors = tensors * std + mean
    tensors = torch.clamp(tensors, 0, 1) # 限制在 0-1 之间
    
    # 转换为 Numpy 并保存
    # tensors shape: [N, 3, H, W]
    for i in range(min(tensors.shape[0], 20)): # 最多保存前20张，防止文件太多
        img_tensor = tensors[i]
        img_np = img_tensor.permute(1, 2, 0).numpy() # [H, W, 3]
        img_np = (img_np * 255).astype(np.uint8)
        
        img_pil = Image.fromarray(img_np)
        save_path = os.path.join(SAVE_DIR, f"{prefix}_{i}.jpg")
        img_pil.save(save_path)
    
    print(f"Saved {min(tensors.shape[0], 20)} images with prefix '{prefix}' to {SAVE_DIR}")

# def draw_heatmap(raw_image, attn_map, save_name="heatmap.jpg"):
#     img_np = np.array(raw_image)
#     w, h = raw_image.size
    
#     if isinstance(attn_map, torch.Tensor):
#         attn_map = attn_map.detach().cpu().squeeze().numpy()
        
#     n_tokens = attn_map.shape[0]
#     grid_size = int(np.sqrt(n_tokens))
#     if grid_size * grid_size != n_tokens:
#         if int(np.sqrt(n_tokens - 1)) ** 2 == n_tokens - 1:
#             attn_map = attn_map[1:]
#             grid_size = int(np.sqrt(n_tokens - 1))
#         else:
#             print(f"Warning: Token count {n_tokens} not square, skipping.")
#             return

#     attn_grid = attn_map.reshape(grid_size, grid_size)
#     attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
#     attn_grid = np.uint8(255 * attn_grid)
#     heatmap = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_CUBIC)
#     heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#     heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
#     alpha = 0.5
#     overlay = (1 - alpha) * img_np + alpha * heatmap_color
#     overlay = np.uint8(overlay)
#     save_path = os.path.join(SAVE_DIR, save_name)
#     Image.fromarray(overlay).save(save_path)
#     print(f"Saved: {save_path}")

def draw_heatmap(raw_image, attn_map, save_name="heatmap.jpg", keep_ratio=0.5, sharpen=2.0):
    """
    自适应热力图绘制 (修复版)
    Args:
        keep_ratio (0.0 ~ 1.0): 相对阈值。例如 0.5 表示只保留最大值 50% 以上的区域。
        sharpen (float): 锐化系数。>1.0 会让高亮更集中。
    """
    img_np = np.array(raw_image)
    w, h = raw_image.size
    
    if isinstance(attn_map, torch.Tensor):
        attn_map = attn_map.detach().cpu().squeeze().numpy()
        
    n_tokens = attn_map.shape[0]
    grid_size = int(np.sqrt(n_tokens))
    if grid_size * grid_size != n_tokens:
        if int(np.sqrt(n_tokens - 1)) ** 2 == n_tokens - 1:
            attn_map = attn_map[1:]
            grid_size = int(np.sqrt(n_tokens - 1))
        else:
            return

    attn_grid = attn_map.reshape(grid_size, grid_size)
    
    # 1. 归一化到 0~1
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
    
    # 2. 锐化
    if sharpen > 1.0:
        attn_grid = attn_grid ** sharpen

    # 3. 相对阈值过滤
    threshold_val = attn_grid.max() * keep_ratio
    indices = attn_grid < threshold_val
    attn_grid[indices] = 0 
    
    # 4. 再次归一化 (为了显示)
    if attn_grid.max() > 0:
        attn_grid = attn_grid / attn_grid.max()
        
    # 5. 上采样
    heatmap = cv2.resize(attn_grid, (w, h), interpolation=cv2.INTER_CUBIC)
    
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 叠加
    mask = heatmap[..., np.newaxis] 
    alpha_map = 0.6 * mask 
    
    overlay = img_np * (1 - alpha_map) + heatmap_color * alpha_map
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    save_path = os.path.join(SAVE_DIR, save_name)
    Image.fromarray(overlay).save(save_path)
    print(f"Saved: {save_path} (Ratio: {keep_ratio}, Sharpen: {sharpen})")

def main():
    args = get_args()
    print(f"Building model with num_classes={args.num_classes}, num_actions={args.num_actions}...")
    
    # 1. Build Model
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        
        # 1. 去除 module. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        # 2. 【修改这里】过滤形状不匹配 OR 指定名称的权重
        model_dict = model.state_dict()
        filtered_dict = {}
        for k, v in new_state_dict.items():
            # ===> [新增] 显式跳过 instance_adapter <===
            if "instance_adapter" in k:
                # print(f"Skipping ablation layer: {k}") # 可选：打印跳过的层
                continue
            
            # 形状检查 (保留原有的防崩溃逻辑)
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    print(f"Skipping mismatch layer: {k} (ckpt: {v.shape}, model: {model_dict[k].shape})")
        
        # 3. 加载权重
        model.load_state_dict(filtered_dict, strict=False)
        print("Weights loaded (Instance Adapter skipped).")
    else:
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # 3. Preprocess & Inference
    raw_img, img_tensor = preprocess_image(IMAGE_PATH)
    samples = nested_tensor_from_tensor_list(img_tensor)
    
    # 构造 dummy targets，避免 forward 内部报错 (orig_size)
    dummy_targets = [{
        "orig_size": torch.as_tensor([raw_img.height, raw_img.width]).to('cuda'),
        "image_id": 0
    }]

    print("Running inference...")
    try:
        with torch.no_grad():
            # 修复：传入 images 列表和 targets
            _ = model(samples, [raw_img], targets=dummy_targets)
    except Exception as e:
        print(f"Inference interrupted by error: {e}")
        print("Attempting to visualize attention maps anyway...")

    # 4. Extract Attention
    print("Extracting attention maps...")
    found = False
    if hasattr(model, 'scene_optimizer') and hasattr(model.scene_optimizer, 'last_attn_map'):
        print(f"Found Scene Attention: {model.scene_optimizer.last_attn_map.shape}")
        # 场景图：Max=0.21 (信心足)，所以参数要温柔一点，保留更多背景
        # keep_ratio=0.1: 保留前10%的强度，避免场景断裂
        # sharpen=1.5: 轻微锐化
        draw_heatmap(raw_img, model.scene_optimizer.last_attn_map, "vis_scene.jpg", 
                     keep_ratio=0.5, sharpen=1.5)
        # draw_heatmap(raw_img,model.scene_optimizer.last_attn_map, "vis_scene.jpg")

    if hasattr(model, 'instance_optimizer') and hasattr(model.instance_optimizer, 'last_attn_map'):
        print(f"Found Instance Attention: {model.instance_optimizer.last_attn_map.shape}")
        # 实例图：Max=0.005 (信心弱)，需要强力清洗背景
        # keep_ratio=0.5: 只保留最大值一半以上的区域 (杀掉杂乱背景)
        # sharpen=4.0: 强力锐化，把中心点“提”出来
        draw_heatmap(raw_img, model.instance_optimizer.last_attn_map, "vis_instance.jpg", 
                     keep_ratio=0.1, sharpen=1.5)
        # draw_heatmap(raw_img, model.instance_optimizer.last_attn_map, "vis_instance.jpg")
            
    if not found:
        print("No attention maps found. Please ensure 'fusion_module.py' is modified to save 'self.last_attn_map'.")

if __name__ == "__main__":
    main()