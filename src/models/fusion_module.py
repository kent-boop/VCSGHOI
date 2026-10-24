'''
models.blipdetr_fusion
fusion blip with detr for HOI detection
corss-attention
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
class TextGuidedBlipOptimizer(nn.Module):
    def __init__(self, hidden_dim=256, use_gate=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_gate = use_gate
        if self.use_gate:
            self.gate_alpha = nn.Parameter(torch.zeros(1))

    def forward(self, blip_visual_feat, text_feat):

        B,N,C = blip_visual_feat.shape  #(B,num_patches,dim)
        #normalize
        v_norm = F.normalize(blip_visual_feat, dim=-1)  
        t_norm = F.normalize(text_feat, dim=-1)  #(B,K,dim)
        if t_norm.dim() == 2:
            t_norm = t_norm.unsqueeze(0) # [B, C] -> [B, 1, C]
        #similarity
        similarity = torch.matmul(v_norm, t_norm.permute(0,2,1))  #(B, N, K)
        spatial_weight = similarity.max(dim=-1, keepdim=True)[0] #(B,N,1)
        spatial_attn = F.softmax(spatial_weight * 10.0, dim=1)
        # print(f"DEBUG [{self.__class__.__name__}] Max: {spatial_attn.max().item():.6f}, Min: {spatial_attn.min().item():.6f}")
        self.last_attn_map = spatial_attn.detach().cpu()#visualation for visualization

        if self.use_gate:
            alpha = torch.tanh(self.gate_alpha)
            weighted_feat = blip_visual_feat * spatial_attn
            optimized_feat = blip_visual_feat + alpha * weighted_feat
        else:
            optimized_feat = blip_visual_feat + (blip_visual_feat * spatial_attn)

        return optimized_feat

class VisualAdapter(nn.Module):#scene and instance
    def __init__(self, visual_dim=256, text_dim=256, bottleneck=64):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Linear(visual_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, text_dim)
        )
        nn.init.constant_(self.meta_net[-1].weight, 0)
        nn.init.constant_(self.meta_net[-1].bias, 0)

    def forward(self, text_bank, visual_features):
        #compute bias
        bias = self.meta_net(visual_features) 
        bias = bias.unsqueeze(1) 
        text_dynamic = text_bank + bias
        return text_dynamic

class VisualTextAdapter(nn.Module):#action
    def __init__(self, visual_dim=1024, text_dim=256, bottleneck=64):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Linear(visual_dim+text_dim, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, text_dim)
        )
        nn.init.constant_(self.meta_net[-1].weight, 0)
        nn.init.constant_(self.meta_net[-1].bias, 0)

    def forward(self, text_bank, visual_features):
        K = text_bank.shape[1]
        visual_expand = visual_features.unsqueeze(1).expand(-1,K,-1)
        combined = torch.cat([text_bank, visual_expand], dim=-1)
        #compute bias
        bias = self.meta_net(combined) 
        text_dynamic = text_bank + bias
        return text_dynamic
