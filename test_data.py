import torch
import torch.nn as nn
from src.lm.BLIP.build_blip import blip_itm
import numpy as np

# ================= 配置区域 =================
# 1. 设置你的 BLIP-Large 权重文件路径 (Coco Retrieval 版本)
pth_path = "/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_base_retrieval_coco.pth" 

# 2. 你的 BERT config 路径 (根据你的 build_blip.py，通常是这个)
bert_config_path = './src/lm/BLIP/bert-base-uncased/config.json'
# ===========================================

def check_layer_status(model_tensor, checkpoint_tensor, layer_name):
    """
    对比模型中的 Tensor 和 Checkpoint 中的 Tensor 是否一致
    """
    # 1. 检查形状是否一致
    if model_tensor.shape != checkpoint_tensor.shape:
        print(f"[❌ 形状不匹配] {layer_name}")
        print(f"    - 模型定义形状: {model_tensor.shape}")
        print(f"    - 权重文件形状: {checkpoint_tensor.shape}")
        return False
    
    # 2. 检查数值是否一致 (允许极小的浮点误差)
    if torch.allclose(model_tensor, checkpoint_tensor, atol=1e-6):
        print(f"[✅ 加载成功] {layer_name} (Shape: {model_tensor.shape})")
        return True
    else:
        print(f"[⚠️ 数值不一致] {layer_name}")
        print(f"    - 警告: 该层形状匹配，但数值不匹配！可能是被随机初始化覆盖了。")
        print(f"    - Model Mean: {model_tensor.mean().item():.6f}, Std: {model_tensor.std().item():.6f}")
        print(f"    - Ckpt  Mean: {checkpoint_tensor.mean().item():.6f}, Std: {checkpoint_tensor.std().item():.6f}")
        return False

def main():
    print(f"正在加载 Checkpoint 文件: {pth_path} ...")
    # 加载原始权重字典
    checkpoint = torch.load(pth_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    print("正在初始化 BLIP-Large 模型 (代码定义)...")
    # 初始化模型结构 (不加载权重，只看结构)
    model = blip_itm(pretrained='', med_config=bert_config_path, vit='base', image_size=384)
    
    # ============== 请插入下面这 3 行代码 ==============
    print("正在模拟加载权重...")
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"加载反馈: {msg}")  # 看看 missing_keys 是否为空，或者只有无关紧要的层
    # =================================================

    print("\n" + "="*20 + " 开始核心层检查 " + "="*20)

    # -----------------------------------------------------------
    # 检查点 1: Vision Projection (关键瓶颈层)
    # 作用: 将 ViT-Large 的 1024 维特征 压缩到 256 维
    # -----------------------------------------------------------
    layer_name = "vision_proj.weight"
    if layer_name in state_dict:
        # 注意: build_blip.py 中定义为 self.vision_proj
        check_layer_status(model.vision_proj.weight, state_dict[layer_name], layer_name)
    else:
        print(f"[❌ 缺失] Checkpoint 中找不到 {layer_name}！这解释了为什么效果差。")

    # -----------------------------------------------------------
    # 检查点 2: Text Projection
    # 作用: 将 Text Encoder 的 768 维特征 压缩到 256 维
    # -----------------------------------------------------------
    layer_name = "text_proj.weight"
    if layer_name in state_dict:
        check_layer_status(model.text_proj.weight, state_dict[layer_name], layer_name)
    else:
        print(f"[❌ 缺失] Checkpoint 中找不到 {layer_name}！")

    # -----------------------------------------------------------
    # 检查点 3: Text Embeddings (词表陷阱)
    # 作用: 检查 vocab_size 是否匹配 (30522 vs 30524)
    # -----------------------------------------------------------
    layer_name = "text_encoder.embeddings.word_embeddings.weight"
    if layer_name in state_dict:
        status = check_layer_status(model.text_encoder.embeddings.word_embeddings.weight, state_dict[layer_name], layer_name)
        if not status:
            print("    >>> [诊断建议] 请检查 config.json 中的 vocab_size 是否比权重文件少 2 个 (特殊Token)。")
    else:
        # 有时候 key 名字里没有 text_encoder 前缀，尝试 bert 前缀
        alt_name = "bert.embeddings.word_embeddings.weight"
        if alt_name in state_dict:
            check_layer_status(model.text_encoder.embeddings.word_embeddings.weight, state_dict[alt_name], alt_name)
        else:
            print(f"[❌ 缺失] Checkpoint 中找不到 Embedding 层！")

    # -----------------------------------------------------------
    # 检查点 4: Cross Attention Key (维度适配陷阱)
    # 作用: 检查 Text Encoder 是否正确适配了 1024 维的视觉特征
    # -----------------------------------------------------------
    # 这里的路径比较深，对应 BERT 的第0层 CrossAttention
    layer_name = "text_encoder.encoder.layer.0.crossattention.self.key.weight"
    # 预期形状: [768, 1024] (因为输入是 encoder_width=1024)
    if layer_name in state_dict:
        check_layer_status(model.text_encoder.encoder.layer[0].crossattention.self.key.weight, state_dict[layer_name], layer_name)
    else:
        # 有可能是 bert 前缀
        alt_name = "bert.encoder.layer.0.crossattention.self.key.weight"
        if alt_name in state_dict:
             check_layer_status(model.text_encoder.encoder.layer[0].crossattention.self.key.weight, state_dict[alt_name], alt_name)
        else:
            print(f"[❓ 未找到/单模态] Checkpoint 中找不到 CrossAttention Key。如果这是单模态权重，这是正常的；如果是多模态，则有问题。")

    print("\n" + "="*20 + " 检查结束 " + "="*20)

if __name__ == "__main__":
    main()