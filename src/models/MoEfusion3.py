import torch
import torch.nn.functional as F
from torch import nn

class MoEFusion(nn.Module):
    """ Mixture of Experts Fusion Module """

    def __init__(self, visual_dim_dict, text_dim, out_dim, hidden_dim=1024,
                 use_visual_router=True, use_aux_loss=True, noisy_gating=True):
        super().__init__()
        self.use_visual_router = use_visual_router
        self.use_aux_loss = use_aux_loss
        self.noisy_gating = noisy_gating
        self.num_experts = 3
        self.semantic_out_dim = 512
        #norm
        self.text_norm = nn.LayerNorm(text_dim)
        self.visual_ctx_norm = nn.LayerNorm(hidden_dim) if use_visual_router else nn.Identity()
        self.visual_norm = nn.LayerNorm(128)
        self.rel_norm = nn.LayerNorm(visual_dim_dict['rel'])
        self.moe_out_norm = nn.LayerNorm(self.semantic_out_dim)
        #specialized experts
        #1.expert for SAM
        self.expert_sam = nn.Sequential(
            nn.Linear(visual_dim_dict['sam'], hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.semantic_out_dim)
        )
        #2.expert for Union
        self.expert_union = nn.Sequential(
            nn.Linear(visual_dim_dict['union'], hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.semantic_out_dim)
        )
        #3.expert for Global
        self.expert_global = nn.Sequential(
            nn.Linear(visual_dim_dict['global'], hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.semantic_out_dim)
        )

        #gating network
        if self.use_visual_router:
            self.router_visual_compression = 128 
            self.router_visual_encoder = nn.Sequential(
                nn.Linear(visual_dim_dict['global'], 512),
                nn.ReLU(),
                nn.Linear(512, self.router_visual_compression) 
            )
            self.router_input_dim = text_dim + self.router_visual_compression
        else:
            self.router_input_dim = text_dim

        self.router_input_norm = nn.LayerNorm(self.router_input_dim)

        self.router = nn.Sequential(
            nn.Linear(self.router_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.num_experts)
        )
        #noisy_gating
        if self.noisy_gating:
            self.w_noise = nn.Linear(self.router_input_dim,self.num_experts)
        
        #geometric fusion
        final_in_dim = visual_dim_dict['rel'] + self.semantic_out_dim
        self.final_proj = nn.Linear(final_in_dim, out_dim)

    def forward(self, rel_reps, blip_sam, blip_union, visual_global, text_features):
        # [DIAGNOSIS START]
        # if self.training and torch.rand(1).item() < 0.1:
        #     print(f"[DEBUG] Visual Global Shape: {visual_global.shape}")
        #     print(f"[DEBUG] Visual Global Norm : {visual_global.norm(dim=-1).mean().item()}")
        # if self.training and torch.rand(1).item() < 0.1:
        #     print("\n[BLIP-L Alignment Check]")
            
        #     # 1. 拿到视觉特征 (Global)
        #     # visual_global 来自 BLIP-L Image Encoder (256d)
        #     v_feat = F.normalize(visual_global, dim=-1)
            
        #     # 2. 拿到文本特征
        #     # text_features 来自 BLIP-L Text Encoder (256d)
        #     t_feat = F.normalize(text_features, dim=-1)
            
        #     # 3. 计算点积 (Cosine Similarity)
        #     # 理论上，匹配的图文对应该 > 0.3
        #     sim = (v_feat * t_feat).sum(dim=-1)
            
        #     print(f"Similarity Mean: {sim.mean().item():.4f}")
        #     print(f"Similarity Max : {sim.max().item():.4f}")
        #     print(f"Similarity Min : {sim.min().item():.4f}")
            
        #     if sim.mean().item() < 0.1:
        #         print(">>> 🚨 警告：特征空间未对齐！模型在瞎猜！")
        #     else:
        #         print(">>> ✅ 特征空间正常。")
        #     print("-" * 30)
        # [DIAGNOSIS END]
        # visual_feat: [Batch, N, Visual_Dim] 
        # text_feat:   [Batch, N, Text_Dim]   
        text_features = self.text_norm(text_features)

        #align visual features to text space
        #1.experts processing
        feat_sam = self.expert_sam(blip_sam)
        feat_union = self.expert_union(blip_union)
        feat_global = self.expert_global(visual_global)

        #stack experts
        expert_outputs = torch.stack([feat_sam, feat_union, feat_global], dim=2)  # [B, N, num_experts, D]

        #dynamic routing
        if self.use_visual_router:
            visual_ctx = self.router_visual_encoder(visual_global)
            visual_ctx_norm = self.visual_norm(visual_ctx)
            visual_ctx_scaled = visual_ctx_norm * 0.6
            router_input = torch.cat([text_features, visual_ctx_scaled], dim=-1)
        else:
            router_input = text_features
        
        if hasattr(self, 'router_input_norm'):
            router_input = self.router_input_norm(router_input)
        
        #router weights
        router_logits = self.router(router_input)  # [B, N, num_experts]
        if self.noisy_gating and self.training:
            noise_std = F.softplus(self.w_noise(router_input))
            noise = torch.randn_like(router_logits) * noise_std
            router_logits = router_logits + noise

        #softmax weights
        router_weights = F.softmax(router_logits, dim=-1)  # [B, N, num_experts]
        self.last_weights = router_weights.detach()
        # if not self.training:
        #     # 获取当前 batch 第一张图、第一个 query 的权重
        #     # router_weights shape: [Batch_Size, Num_Queries, 3]
        #     w = router_weights[0, 0].detach().cpu().numpy()
            
        #     # 设置一个概率打印，或者打印特定格式，方便你肉眼观察
        #     # 打印格式：[SAM权重, Union权重, Global权重]
        #     print(f"\n[DEBUG R50-CHECK] Route: SAM={w[0]:.2f} | Union={w[1]:.2f} | Global={w[2]:.2f}")
            
        #     # 如果你想看是不是“坍塌”了，可以加个判断
        #     if w.max() > 0.95:
        #         print("   >>> ⚠️ 单专家主导 (Dominant)")
        #     elif abs(w[0]-0.33) < 0.05 and abs(w[1]-0.33) < 0.05:
        #         print("   >>> ⚠️ 平均分配 (Average)")
        #semantic fusion
        semantic_feat = torch.sum(router_weights.unsqueeze(-1) * expert_outputs, dim=2)

        #aux loss
        if self.use_aux_loss and self.training:
            # expert_usage = router_weights.mean(dim=[0, 1])
            # std = expert_usage.std()
            # mean = expert_usage.mean() + 1e-6
            # aux_loss = (std / mean) * 0.1
            aux_loss = torch.tensor(0.0, device=rel_reps.device)
        else:
            aux_loss = torch.tensor(0.0, device=rel_reps.device)
        
        #norm
        rel_reps_norm = self.rel_norm(rel_reps)
        semantic_feat_norm = self.moe_out_norm(semantic_feat)
        
        # # ---------------------------------------------------------
        # # [DEBUG PRINT] - 修正版：打印真实使用的变量
        # # ---------------------------------------------------------
        # if self.training and torch.rand(1).item() < 0.05:
        #     print("\n" + "="*40)
        #     print("[MoE DEBUG] Corrected Statistics (R101 Fixed)")
            
        #     def print_stat(name, tensor):
        #         print(f"{name:<20} | Mean: {tensor.mean().item():.4f} | "
        #               f"Std: {tensor.std().item():.4f} | "
        #               f"Norm: {tensor.norm(p=2, dim=-1).mean().item():.4f}")

        #     # 打印归一化后的特征，确认比例是否接近 1.4 : 1
        #     print_stat("rel_reps (Normed)", rel_reps_norm)
        #     print_stat("MoE_Out (Normed)", semantic_feat_norm)
            
        #     # 打印 Router 输入，确认 Norm 正常
        #     print_stat("Router Input", router_input)
            
        #     # 打印 Logits，确认有区分度
        #     print(f"Router Logits (Raw): {router_logits[0, 0, :].detach().cpu().numpy()}")
        #     print("="*40 + "\n")

        #geometric fusion
        combined_feat = torch.cat([rel_reps, semantic_feat], dim=-1)
        fused_output = self.final_proj(combined_feat)  # [B, N, out_dim]
        
        return fused_output,aux_loss