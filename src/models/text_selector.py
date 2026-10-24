'''
select text_feature from img_blip
top_k to select
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HierarchicalSimilarityCalculator(nn.Module):
    def __init__(self, text_lib_path,img_dim=256,text_dim=256, top_k_dict={'scene': 3, 'instance': 10, 'action': 20}):
        super().__init__()
        self.top_k_dict = top_k_dict
        self.text_dim = text_dim

        #1.load text features(520,3,256)
        print("loading text features from {}".format(text_lib_path))
        try:
            text_lib_np = np.load(text_lib_path)
        except:
            raise ValueError("Failed to load text features from {}".format(text_lib_path))
        
        text_tensor = torch.tensor(text_lib_np,dtype=torch.float32)  #(520,3,256)
        self.text_lib = F.normalize(text_tensor, dim=-1)  #normalize
        #buffer text features:no grad
        self.register_buffer("text_library", self.text_lib) 
        #logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.fusion_proj = nn.Sequential(
            nn.Linear(3*text_dim, text_dim*2),
            nn.ReLU(),
            nn.Linear(text_dim*2, text_dim)
        )

    def forward(self, image_features):
        """
        image_features: (B,num_patches,img_dim) select cls token features
        return:
        scene, object, action similarities: (B,520)
        indices_dict: 3 keys: 'scene', 'object', 'action', each value is (B,top_k)
        sim_dict: 
        seleted_text_features: (B,3*text_dim) fused text features
        """

        if image_features.dim() == 3:
            image_features = image_features.squeeze(1)  #(B,256)

        #logit scale
        logit_scale = self.logit_scale.exp()
        text_features_dict = {}
        indices_dict = {}
        sim_dict = {}

        bs = image_features.shape[0]
        num_hois = self.text_library.shape[0]

        total_sim_logits = 0
        layer_names = ['scene', 'instance', 'action']

        for i,name in enumerate(layer_names):
            k = self.top_k_dict[name]
            #text features for this layer 
            level_text_bank = self.text_library[:,i,:]
            level_text_bank = level_text_bank.squeeze(1) if level_text_bank.dim() ==3 else level_text_bank  #(520,256)  

            #similarity (B,256)*(520,256).T = (B,520)
            sim = torch.matmul(image_features, level_text_bank.t()) * self.logit_scale.exp()
            probs = torch.softmax(sim, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k, dim=-1) # (B, K)

            #select topk similarities 
            indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, level_text_bank.shape[-1])#(B,K,256)
            level_text_batch = level_text_bank.unsqueeze(0).expand(bs, -1, -1)  #(B,520,256)
            selected_text = torch.gather(level_text_batch, 1, indices_expanded)  #(B,K,256)
            selected_text = selected_text * topk_probs.unsqueeze(-1)  #(B,K,256) weighted by probs
            text_features_dict[name] = selected_text
            indices_dict[name] = topk_indices

        return text_features_dict, indices_dict
    

def test_selector_with_mock_data():
    import json
    import torch
    import numpy as np
    import torch.nn.functional as F
    import ast

    print("="*50)
    print("开始测试 Text Selector 模块 (诊断模式)...")
    print("="*50)

    embed_path = "/home/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/hico_embeds_256.npy"
    json_path = "/home/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/hico_text_descriptions.json"
    
    # --- 1. 强力诊断加载逻辑 ---
    idx2desc = {}
    
    try:
        with open(json_path, 'r') as f:
            raw_descs = json.load(f)
            
        print(f"原始 JSON 包含 {len(raw_descs)} 个键值对。")
        
        # 检查键的范围
        raw_keys = [int(k) for k in raw_descs.keys() if k.isdigit()]
        if raw_keys:
            print(f"JSON 键范围: Min={min(raw_keys)}, Max={max(raw_keys)}")
            if min(raw_keys) == 1:
                print("【注意】检测到 JSON 是从 ID 1 开始的，将自动执行 -1 修正！")
        
        # 遍历加载
        parse_fail_count = 0
        loaded_count = 0
        
        for k, v in raw_descs.items():
            try:
                # 尝试转换 ID
                raw_idx = int(k)
                
                # 【修正逻辑】：如果是 1-based (1~520)，需要转为 0-based (0~519)
                # 这里的判断依据是：如果 raw_keys 最小是 1 且最大是 520
                if min(raw_keys) == 1 and max(raw_keys) == 520:
                    idx = raw_idx - 1
                else:
                    idx = raw_idx

                # 提取文本
                desc_str = ""
                if isinstance(v, str):
                    if v.strip().startswith("{") and "verb_phrase" in v:
                        try:
                            data = ast.literal_eval(v)
                            desc_str = f"Human {data.get('verb_phrase', '')} {data.get('obj_phrase', '')}"
                        except:
                            print(f"  [解析警告] ID {k} 的 Value 格式错误，保留原字符串。")
                            desc_str = v
                    else:
                        desc_str = v
                elif isinstance(v, dict):
                    desc_str = f"Human {v.get('verb_phrase', '')} {v.get('obj_phrase', '')}"
                else:
                    desc_str = str(v)
                
                idx2desc[idx] = desc_str
                loaded_count += 1
                
            except ValueError:
                print(f"  [跳过] 键 '{k}' 无法转换为整数。")
                parse_fail_count += 1

        print(f"最终加载有效描述: {len(idx2desc)} 条。")
        
        # 检查缺失的 ID
        expected_ids = set(range(520))
        loaded_ids = set(idx2desc.keys())
        missing_ids = expected_ids - loaded_ids
        if missing_ids:
            print(f"【严重警告】以下 {len(missing_ids)} 个 ID (0-519范围) 缺失:")
            print(f"  {sorted(list(missing_ids))}")
        else:
            print("所有 0-519 的 ID 均已就位！")

    except Exception as e:
        print(f"警告: 描述文件加载失败 ({e})")
        return

    # --- 2. 初始化模型 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # top_k 稍微改大一点以便观察
    top_k_dict = {'scene': 3, 'instance': 10, 'action': 10} 
    selector = HierarchicalSimilarityCalculator(text_lib_path=embed_path, top_k_dict=top_k_dict).to(device)
    selector.eval()

    # --- 3. 构造测试 ---
    full_lib = torch.tensor(np.load(embed_path)).to(device)
    
    # 测试那些之前显示无描述的 ID
    test_indices = [0, 50, 100] 
    
    for test_idx in test_indices:
        gt_desc = idx2desc.get(test_idx, f"ID_{test_idx} (缺失)")
        print(f"\n" + "-"*60)
        print(f"【测试输入 HOI ID: {test_idx}】")
        print(f"  -> Ground Truth: \"{gt_desc}\"")
        print("-" * 60)
        
        # 模拟特征
        target_feat = full_lib[test_idx].mean(dim=0).unsqueeze(0)
        noise = torch.randn_like(target_feat) * 0.02 
        mock_img_feat = F.normalize(target_feat + noise, dim=-1)

        with torch.no_grad():
            outputs = selector(mock_img_feat)
            if isinstance(outputs, tuple):
                feats_dict, indices_dict = outputs[0], outputs[1]
            else:
                feats_dict, indices_dict = outputs, getattr(selector, 'last_indices', {})

            print("选取的文本特征形状:")
            print(feats_dict['scene'][0].shape)
        # 验证 Instance 和 Action
        for branch in ['instance', 'action']:
            if branch in indices_dict:
                top_indices = indices_dict[branch][0].cpu().numpy()
                print(f"  [{branch.upper()} Branch] Top-5:")
                for rank, idx in enumerate(top_indices[:5]):
                    res_text = idx2desc.get(idx, f"ID_{idx}")
                    
                    # 标记
                    marker = ""
                    if idx == test_idx: marker = " <--- (原ID)"
                    
                    print(f"    {rank+1}. ID {idx:<3}: {res_text}{marker}")

if __name__ == "__main__":
    # 确保依赖库已安装
    try:
        import torch
        import numpy as np
        import torch.nn.functional as F
        test_selector_with_mock_data()
    except ImportError as e:
        print(f"缺少必要的库: {e}")
    except Exception as e:
        print(f"测试运行出错: {e}")
        import traceback
        traceback.print_exc()