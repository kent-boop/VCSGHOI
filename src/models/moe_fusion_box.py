"""Three-expert MoE used by the SAM-free OSSM ROI variant."""

import torch
import torch.nn.functional as F
from torch import nn


class OSSMBoxMoEFusion(nn.Module):
    """Fuse the global, instance and action visual experts.

    The router and output projection intentionally follow the old OSSM MoE.
    Only the expert names and their inputs are made explicit for scratch
    training of the ROI-based model.
    """

    EXPERT_ORDER = ("global", "instance", "action")

    def __init__(
        self,
        visual_dim_dict,
        text_dim,
        out_dim,
        hidden_dim=1024,
        use_visual_router=True,
        use_aux_loss=True,
        noisy_gating=True,
    ):
        super().__init__()
        self.use_visual_router = use_visual_router
        self.use_aux_loss = use_aux_loss
        self.noisy_gating = noisy_gating
        self.num_experts = len(self.EXPERT_ORDER)
        self.semantic_out_dim = 512

        self.text_norm = nn.LayerNorm(text_dim)
        self.visual_norm = nn.LayerNorm(128)
        self.rel_norm = nn.LayerNorm(visual_dim_dict["rel"])
        self.moe_out_norm = nn.LayerNorm(self.semantic_out_dim)

        self.expert_global = self._make_expert(
            visual_dim_dict["global"], hidden_dim
        )
        self.expert_instance = self._make_expert(
            visual_dim_dict["instance"], hidden_dim
        )
        self.expert_action = self._make_expert(
            visual_dim_dict["action"], hidden_dim
        )

        if self.use_visual_router:
            self.router_visual_encoder = nn.Sequential(
                nn.Linear(visual_dim_dict["global"], 512),
                nn.ReLU(),
                nn.Linear(512, 128),
            )
            router_input_dim = text_dim + 128
        else:
            router_input_dim = text_dim
        self.router_input_norm = nn.LayerNorm(router_input_dim)
        self.router = nn.Sequential(
            nn.Linear(router_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, self.num_experts),
        )
        if self.noisy_gating:
            self.w_noise = nn.Linear(router_input_dim, self.num_experts)

        self.final_proj = nn.Linear(
            visual_dim_dict["rel"] + self.semantic_out_dim, out_dim
        )
        self.last_weights = None

    def _make_expert(self, input_dim, hidden_dim):
        return nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.semantic_out_dim),
        )

    def forward(
        self,
        rel_reps,
        visual_global,
        visual_instance,
        visual_action,
        text_features,
    ):
        text_features = self.text_norm(text_features)

        expert_outputs = torch.stack(
            [
                self.expert_global(visual_global),
                self.expert_instance(visual_instance),
                self.expert_action(visual_action),
            ],
            dim=2,
        )

        if self.use_visual_router:
            visual_context = self.visual_norm(
                self.router_visual_encoder(visual_global)
            )
            router_input = torch.cat(
                [text_features, visual_context * 0.6], dim=-1
            )
        else:
            router_input = text_features
        router_input = self.router_input_norm(router_input)

        router_logits = self.router(router_input)
        if self.noisy_gating and self.training:
            noise_std = F.softplus(self.w_noise(router_input))
            router_logits = router_logits + torch.randn_like(router_logits) * noise_std
        router_weights = F.softmax(router_logits, dim=-1)
        self.last_weights = router_weights.detach()

        semantic_feature = torch.sum(
            router_weights.unsqueeze(-1) * expert_outputs, dim=2
        )
        # The selected old MoE implementation returns a zero auxiliary loss.
        aux_loss = rel_reps.new_zeros(())

        # Keep the selected old MoE's effective path: its norm tensors were
        # computed only for diagnostics, while final_proj consumed raw values.
        combined = torch.cat([rel_reps, semantic_feature], dim=-1)
        return self.final_proj(combined), aux_loss


__all__ = ["OSSMBoxMoEFusion"]
