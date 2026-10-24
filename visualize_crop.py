import os
# =========================================================
#  1. 显卡设置 (必须在 import torch 前设置)
# =========================================================
os.environ['CUDA_VISIBLE_DEVICES'] = '3' # 指定使用 1 号卡

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
IMAGE_PATH = '/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/visual2/HICO_train2015_00000115.jpg' 
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
    # 这里的 transforms 仅用于模型输入，不影响我们手动抠图
    transform = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到图片: {image_path}")
    
    # 1. 读取原始图片 (保持原始尺寸)
    raw_img_pil = Image.open(image_path).convert('RGB')
    
    # 2. 准备送入模型的 Tensor (Resize 到 800x800)
    # 这一步是为了让模型正常运行，生成 outputs_coord
    raw_img_resized = raw_img_pil.resize((800, 800))
    img_tensor = transform(raw_img_pil).unsqueeze(0).to('cuda')
    
    return raw_img_pil, raw_img_resized, img_tensor

def crop_union_from_data(raw_img_pil, pred_boxes, rel_pairs):
    """
    完全在外部进行坐标换算和抠图
    Args:
        raw_img_pil: 原始 PIL 图片 (或者 Resize 过的 PIL，取决于你想在哪张图上扣)
        pred_boxes: [N, 4] 归一化坐标 (cx, cy, w, h)
        rel_pairs: [K, 2] 关系对索引
    """
    W, H = raw_img_pil.size
    print(f"Cropping union regions from image size: {W}x{H}")

    # 只保存前 10 个 Pair，避免太多
    num_to_save = min(rel_pairs.shape[0], 10)
    
    for i in range(num_to_save):
        # 1. 获取索引
        sub_idx = rel_pairs[i, 0]
        obj_idx = rel_pairs[i, 1]
        
        # 2. 获取对应的归一化框 (cx, cy, w, h)
        sub_box_norm = pred_boxes[sub_idx]
        obj_box_norm = pred_boxes[obj_idx]
        
        # 3. 定义转换函数 (Norm -> Pixel xyxy)
        def to_pixel_xyxy(box):
            cx, cy, w, h = box
            x1 = (cx - 0.5 * w) * W
            y1 = (cy - 0.5 * h) * H
            x2 = (cx + 0.5 * w) * W
            y2 = (cy + 0.5 * h) * H
            return float(x1), float(y1), float(x2), float(y2)
        
        s_rect = to_pixel_xyxy(sub_box_norm)
        o_rect = to_pixel_xyxy(obj_box_norm)
        
        # 4. 计算 Union Box
        u_x1 = min(s_rect[0], o_rect[0])
        u_y1 = min(s_rect[1], o_rect[1])
        u_x2 = max(s_rect[2], o_rect[2])
        u_y2 = max(s_rect[3], o_rect[3])
        
        # 5. 边界处理 (防止切出图片外)
        u_x1 = max(0, int(u_x1))
        u_y1 = max(0, int(u_y1))
        u_x2 = min(W, int(u_x2))
        u_y2 = min(H, int(u_y2))
        
        # 检查有效性
        if u_x2 <= u_x1 or u_y2 <= u_y1:
            print(f"Skipping invalid crop {i}: {u_x1},{u_y1},{u_x2},{u_y2}")
            continue
            
        # 6. 裁剪并保存
        union_crop = raw_img_pil.crop((u_x1, u_y1, u_x2, u_y2))
        union_crop.save(os.path.join(SAVE_DIR, f"union_crop_ext_{i}.jpg"))
        print(f"Saved union_crop_ext_{i}.jpg")

def main():
    args = get_args()
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading weights...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cuda')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(filtered_dict, strict=False)
    else:
        return

    # 1. 预处理
    # raw_img_pil: 原始尺寸 (例如 640x480)
    # raw_img_resized: 800x800 (用于给模型“看”的坐标参考，如果你想切出来的图不失真，可以用这个)
    raw_img_pil, raw_img_resized, img_tensor = preprocess_image(IMAGE_PATH)
    
    samples = nested_tensor_from_tensor_list(img_tensor)
    
    # 构造 target 里的 orig_size，告诉模型这是 800x800 的图
    # 这样 outputs_coord 出来的就是相对于 800x800 的归一化坐标
    dummy_targets = [{"orig_size": torch.as_tensor([800, 800]).to('cuda'), "image_id": 0}]

    print("Running inference...")
    with torch.no_grad():
        _ = model(samples, [raw_img_resized], targets=dummy_targets)

    # 2. 从模型中取出数据
    if hasattr(model, 'last_pred_boxes') and hasattr(model, 'last_rel_pairs'):
        print("Found internal prediction data!")
        
        # 选项 A: 在 800x800 的图上切 (和模型输入一致，坐标最准)
        crop_union_from_data(raw_img_resized, model.last_pred_boxes, model.last_rel_pairs)
        
        # 选项 B: 如果你想在原始大图上切 (比如原图是 4K 的)，把 raw_img_resized 换成 raw_img_pil 即可
        # crop_union_from_data(raw_img_pil, model.last_pred_boxes, model.last_rel_pairs)
        
    else:
        print("Error: 'last_pred_boxes' not found. Please modify stip.py to save outputs_coord[-1].")
    
    # 3. 顺便保存一下 SAM Mask (如果之前改过的话)
    if hasattr(model, 'last_sam_masked_imgs'):
         # 简单的保存逻辑...
         pass

if __name__ == "__main__":
    main()