import torch
import argparse

def extract_detr_weights(source_path, output_path):
    print(f"Loading source model from {source_path}...")
    checkpoint = torch.load(source_path, map_location='cpu')
    
    # 获取原始的状态字典
    if 'model' in checkpoint:
        source_state_dict = checkpoint['model']
    else:
        source_state_dict = checkpoint
    
    new_state_dict = {}
    
    # 我们需要提取的前缀列表
    # input_proj 就是你提到的“通过卷积改变通道数的适配层”
    prefixes_to_keep = [
        "detr.backbone", 
        "detr.transformer", 
        "detr.input_proj", 
        "detr.query_embed"  # 强烈建议保留，这是 Object Queries，Transformer Decoder 必须用
    ]
    
    print("Extracting keys...")
    converted_count = 0
    for key, value in source_state_dict.items():
        # 处理可能存在的 module. 前缀 (如果是 DDP 保存的)
        if key.startswith('module.'):
            key = key[7:]
            
        # 检查是否是我们需要的层
        for prefix in prefixes_to_keep:
            if key.startswith(prefix):
                # 去掉 'detr.' 前缀，使其适配 model.detr.load_state_dict()
                # 例如: detr.backbone.0.body.layer1... -> backbone.0.body.layer1...
                new_key = key.replace("detr.", "", 1)
                new_state_dict[new_key] = value
                converted_count += 1
                break
    
    print(f"Extracted {converted_count} keys.")
    
    # 简单的验证
    has_backbone = any('backbone.0.body.layer4' in k for k in new_state_dict)
    has_transformer = any('transformer.encoder.layers' in k for k in new_state_dict)
    has_proj = any('input_proj.weight' in k for k in new_state_dict)
    
    print(f"Check extraction:")
    print(f"  - ResNet Backbone found: {has_backbone}")
    print(f"  - Transformer found: {has_transformer}")
    print(f"  - Input Projection (Adaptation Layer) found: {has_proj}")
    
    if converted_count == 0:
        print("[WARNING] No keys were extracted! Please check if the source model has 'detr.' prefix.")
        # 备选方案：如果源模型没有 'detr.' 前缀 (比如它本身就是个纯 DETR 权重)
        # 你可以尝试修改脚本直接复制
    else:
        # 保存为新格式，包裹在 'model' key 中以符合 STIP_main.py 的读取习惯
        torch.save({'model': new_state_dict}, output_path)
        print(f"Saved extracted weights to {output_path}")

if __name__ == "__main__":
    # 修改这里的路径
    SOURCE_MODEL_PATH = "/share/share/ljg/SICHOI/checkpoints/vcoco/KakaoBrain/run_000001/best_large.pth" # 你的 sichoi 模型路径
    OUTPUT_MODEL_PATH = "vcoco_resnet101.pth"     # 输出路径
    
    extract_detr_weights(SOURCE_MODEL_PATH, OUTPUT_MODEL_PATH)