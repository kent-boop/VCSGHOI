import torch
import torch.nn as nn

class StaticTextAligner(nn.Module):
    """
    用于消融实验：直接加载全量静态文本库，跳过 Top-K 筛选。
    输入: (Num_Classes, 3, Dim)
    输出: Dict of (1, Num_Classes, Dim) - Batch维设为1以便广播
    """
    def __init__(self):
        super().__init__()

    def forward(self, text_library):
        # text_library shape: [520, 3, 256]
        # 目标: 将其拆解并增加维度，使其能与 Visual Features (B, N, D) 进行计算
        
        # 1. 拆分三个分支 (Scene, Instance, Action)
        # 形状变化: [520, 3, 256] -> [520, 256]
        scene_feat = text_library[:, 0, :]
        instance_feat = text_library[:, 1, :]
        action_feat = text_library[:, 2, :]

        # 2. 增加 Batch 维度 (设为1)
        # 形状变化: [520, 256] -> [1, 520, 256]
        # 这样当它和图像特征 [B, N, 256] 结合时，PyTorch会自动广播 B 维度
        res_dict = {
            'scene': scene_feat.unsqueeze(0),
            'instance': instance_feat.unsqueeze(0),
            'action': action_feat.unsqueeze(0)
        }
        
        # 3. 返回字典，indices 为 None (因为没有筛选)
        return res_dict, None