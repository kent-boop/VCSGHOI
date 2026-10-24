import torch
import torch.nn.functional as F
from torch import nn

class MoEFusion(nn.Module):
    """ Mixture of Experts Fusion Module """

    def __init__(self, visual_dim, text_dim, out_dim, num_experts=4, hidden_dim=1024,
                use_visual_router=False,use_aux_loss=False,noisy_gating=False):
        super().__init__()
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.num_experts = num_experts
        self.use_visual_router = use_visual_router
        self.use_aux_loss = use_aux_loss
        self.noisy_gating = noisy_gating
        
        #Text Aligner
        self.aligner_proj = nn.Linear(visual_dim, text_dim)
        self.temperature = text_dim ** -0.5
        #experts bank
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(visual_dim, hidden_dim), 
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            ) for _ in range(num_experts)
        ])
        #gating network
        if self.use_visual_router:
            self.visual_compression_dim = 256
            self.router_visual_proj = nn.Sequential(
                nn.Linear(visual_dim,512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512,self.visual_compression_dim)
            )
            self.router_input_dim = self.visual_compression_dim + self.text_dim
        else:
            self.router_input_dim = text_dim

        self.router = nn.Sequential(
            nn.Linear(self.router_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_experts)
        )
        #noisy_gating
        if self.noisy_gating:
            self.w_noise = nn.Linear(self.router_input_dim,self.num_experts)

        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, visual_features, text_features):
        # visual_feat: [Batch, N, Visual_Dim] 
        # text_feat:   [Batch, N, Text_Dim]   

        #align visual features to text space
        visual_query_mapped = self.aligner_proj(visual_features) 
        attn_logits = torch.matmul(visual_query_mapped, text_features.transpose(0, 1)) * self.temperature 
        attn_weights = F.softmax(attn_logits, dim=-1)
        aligned_text = torch.matmul(attn_weights, text_features)#aligned_fused_text

        if self.use_visual_router:
            visual_summary = self.router_visual_proj(visual_features)
            router_input = torch.cat([aligned_text,visual_summary],dim=-1)
        else:
            router_input = aligned_text

        #router weights
        router_logits = self.router(router_input)  # [B, N, num_experts]
        #add noisy
        if self.noisy_gating and self.training:
            noise_std = F.softplus(self.w_noise(router_input))
            noise = torch.randn_like(router_logits)*noise_std
            router_logits = router_logits + noise
        
        router_weights = F.softmax(router_logits, dim=-1)  # [B, N, num_experts]

        #aux loss
        if self.use_aux_loss and self.training:
            expert_usage = router_weights.mean(dim=0)
            aux_loss = torch.sum(expert_usage ** 2) * self.num_experts
            aux_loss = aux_loss - 1.0
        else:
            aux_loss = torch.tensor(0.0, device=visual_features.device)
        
        #expert outputs
        expert_outputs = []
        for expert in self.experts:
            out = expert(visual_features)
            expert_outputs.append(out)  # list of [B, N, out_dim]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        #fuse expert outputs
        router_weights = router_weights.unsqueeze(-1)  # [B, N, num_experts, 1]
        final_output = torch.sum(router_weights * expert_outputs, dim=1)  # [B, N, out_dim]
        #final normalization
        fused_output = self.output_norm(final_output) 
        return fused_output,aux_loss