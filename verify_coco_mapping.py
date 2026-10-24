import torch
import numpy as np

# 1. 导入你的 HICO 标签定义
try:
    from hico_text_label import hico_obj_text_label
except ImportError:
    print("错误: 找不到 hico_text_label.py，请确保它在当前目录")
    exit()

# 2. 定义标准的 COCO 91 类名称列表 (包含 'N/A' 和空缺位置)
# 这是 DETR 预训练模型通用的输出定义 (index 0-91)
# 索引 0 通常保留给背景或 N/A，物体 ID 从 1 开始
COCO_CLASSES_FULL = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', # ID 12 is empty
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', # ID 26 is empty
    'backpack', 'umbrella', 'N/A', 'N/A', # ID 29, 30 empty
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'N/A', # ID 45 empty
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
    'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', # ID 66 empty
    'dining table', 'N/A', 'N/A', # ID 68, 69 empty
    'toilet', 'N/A', # ID 71 empty
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'N/A', # ID 83 empty
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# 3. 复制我们刚才定义的映射索引 (需要验证的对象)
coco_indices = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 
    85, 86, 87, 88, 89, 90
]

def verify():
    print(f"正在验证映射表... (共 {len(coco_indices)} 个索引)")
    
    # 构建 HICO 字典: ID -> Name (清洗过的)
    hico_dict = {}
    for oid, text in hico_obj_text_label:
        # 去掉 "a photo of a/an " 并转小写
        clean_name = text.replace('a photo of a ', '').replace('a photo of an ', '').strip().lower()
        hico_dict[oid] = clean_name

    match_count = 0
    mismatch_count = 0

    # 4. 逐一比对
    for hico_id, coco_idx in enumerate(coco_indices):
        if hico_id not in hico_dict:
            print(f"Error: HICO ID {hico_id} 不在 hico_text_label 中!")
            continue

        hico_name = hico_dict[hico_id]
        coco_name = COCO_CLASSES_FULL[coco_idx]

        # 简单的名称标准化匹配
        # 注意: 有些名称可能略有不同 (如 'monitor' vs 'tv')，需要人工检查
        is_match = (hico_name == coco_name) or \
                   (hico_name == 'airplane' and coco_name == 'airplane') or \
                   (hico_name == 'dining table' and coco_name == 'dining table') or \
                   (hico_name == 'sports ball' and coco_name == 'sports ball') or \
                   (hico_name == 'potted plant' and coco_name == 'potted plant') 

        # 特殊别名处理 (如果名称拼写不一样但指代相同)
        # HICO: 'couch' <-> COCO: 'couch' (Standard COCO often uses couch or sofa)
        # HICO: 'tv' <-> COCO: 'tv' 
        
        status = "✅ 匹配" if is_match else "❌ 不匹配"
        if not is_match:
            # 允许一些常见的同义词差异
            if hico_name == 'tv' and 'tv' in coco_name: status = "⚠️ 警告 (需人工确认)"
            else: mismatch_count += 1
        else:
            match_count += 1

        print(f"HICO ID {hico_id:<2} ({hico_name:<15}) <==> COCO Index {coco_idx:<2} ({coco_name:<15}) | {status}")

    print("="*60)
    print(f"验证完成: 成功 {match_count} 个, 失败 {mismatch_count} 个")
    if mismatch_count == 0:
        print("🎉 恭喜！映射表与标准 COCO 定义完全一致。")
    else:
        print("🛑 请检查上述不匹配的项！")

if __name__ == "__main__":
    verify()