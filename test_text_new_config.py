import torch
import torch.nn.functional as F
from src.lm.BLIP.build_blip import blip_itm

# ================= 配置 =================
# 1. 权重路径 (Large)
pth_path = "/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_large_retrieval_coco.pth"
# 2. 配置文件路径 (已修复的)
config_path = './src/lm/BLIP/bert-base-uncased/config.json'
# =======================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在加载模型... (Device: {device})")
    
    # 初始化并加载模型
    model = blip_itm(pretrained=pth_path, med_config=config_path, vit='large', image_size=384)
    model.to(device)
    model.eval()
    
    # 准备测试文本
    # 我们设计两组：一组是语义非常相近的，一组是完全无关的
    texts = [
        "a person riding a bicycle",   # A
        "someone holds a bike",        # B (和 A 语义相关)
        "a cat eating a fish",         # C (和 A 完全无关)
        "a delicious pizza"            # D (和 A 完全无关)
    ]
    
    print("\n正在计算文本特征...")
    embeddings = []
    with torch.no_grad():
        for t in texts:
            # 模拟你生成 npy 时的调用方式 (match_head='text_encoding')
            # 注意：text_encoding 模式下，blip_itm 会返回归一化后的 text_feat
            emb = model(None, t, match_head='text_encoding')
            embeddings.append(emb)
    
    # 堆叠起来: [4, 256]
    embeddings = torch.cat(embeddings, dim=0)
    
    # 计算两两之间的余弦相似度 (Cosine Similarity)
    # 因为 embeddings 已经是归一化的 (F.normalize)，直接矩阵相乘即可
    sim_matrix = embeddings @ embeddings.t()
    
    print("\n" + "="*20 + " 验证结果 " + "="*20)
    
    # 1. 验证相关性 (A 和 B 应该相似)
    score_ab = sim_matrix[0, 1].item()
    print(f"1. [相似句测试] '骑自行车' vs '扶着自行车':")
    print(f"   相似度: {score_ab:.4f}")
    
    if score_ab > 0.5:
        print("   ✅ 结果正常：相似度较高，模型理解语义。")
    else:
        print("   ❌ 结果异常：相似度过低，模型可能仍在输出随机噪声。")
        
    print("-" * 30)

    # 2. 验证无关性 (A 和 C 应该不相似)
    score_ac = sim_matrix[0, 2].item()
    print(f"2. [无关句测试] '骑自行车' vs '猫吃鱼':")
    print(f"   相似度: {score_ac:.4f}")
    
    if score_ac < 0.4 and score_ac < score_ab:
        print("   ✅ 结果正常：相似度较低，且明显低于相似句。")
    else:
        print("   ❌ 结果异常：相似度异常高，或者模型无法区分语义。")

    print("-" * 30)
    
    # 3. 检查特征分布 (防止全是0或者NaN)
    mean_val = embeddings.mean().item()
    std_val = embeddings.std().item()
    print(f"3. [数值统计] Mean: {mean_val:.6f}, Std: {std_val:.6f}")
    if abs(mean_val) < 1e-5 and std_val < 1e-5:
         print("   ❌ 结果异常：特征向量似乎全是 0 或失效。")
    else:
         print("   ✅ 数值分布看起来是正常的非零向量。")

    print("="*50)

if __name__ == "__main__":
    main()