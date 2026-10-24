"""Language Prototype Topology for the B4-VET-LPT ablation.

This module performs object-conditioned action-object prototype retrieval,
structured prototype slot extraction, and masked prototype slot aggregation into
a compact language context. It intentionally excludes SETM/STEM, explicit
evidence-prototype matching, visual-text token updates, and per-layer
synchronization with visual evidence.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.text_scene_graph_oov3 import (
    TextGraphFeatureBank,
    TextHOIRetriever,
    TextSceneGraphReasoner,
)


@dataclass
class LPTOutput:
    prototype_slots: Tensor              # [B, K, R, S, D]
    prototype_slot_mask: Tensor          # [B, K, R, S]
    selected_hoi_indices: Tensor         # [B, K, R]
    selected_hoi_mask: Tensor            # [B, K, R]
    selected_scores: Tensor              # [B, K, R]
    text_tokens: Tensor                  # [B, K, T, D]
    text_token_mask: Tensor              # [B, K, T]
    pooled_human_node: Tensor            # [B, K, D]
    pooled_object_node: Tensor           # [B, K, D]
    pooled_ho_edge: Tensor               # [B, K, D]
    pooled_discriminative: Tensor        # [B, K, D]
    pooled_scene_context: Tensor         # [B, K, D]
    pooled_oo_context: Tensor            # [B, K, D]
    pooled_all: Tensor                   # [B, K, D]
    slot_summary: Tensor                 # [B, K, D]
    prototype_context: Tensor            # [B, K, D]
    pair_valid_mask: Tensor              # [B, K]
    debug_info: Dict[str, Any]


class LanguagePrototypeTopology(nn.Module):
    VALID_CONTEXT_SOURCES = {"slots", "pooled_all", "concat_slots_pooled"}

    def __init__(
        self,
        text_features_path: str,
        text_mask_path: str,
        hoi_ids_path: str,
        object_to_hoi_indices_path: str,
        field_order_path: str,
        d_model: int = 256,
        strict_shape: bool = True,
        topk: int = 5,
        pre_topk: int = 20,
        temperature: float = 0.07,
        rerank_weight: float = 0.5,
        use_discriminative_rerank: bool = True,
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_scene_token: bool = True,
        use_oo_context_token: bool = True,
        prototype_slot_fallback_to_raw: bool = True,
        prototype_slot_debug: bool = False,
        context_source: str = "slots",
        use_selected_score_weight: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if context_source not in self.VALID_CONTEXT_SOURCES:
            raise ValueError(
                f"context_source must be one of {sorted(self.VALID_CONTEXT_SOURCES)}, "
                f"got {context_source!r}"
            )

        self.d_model = int(d_model)
        self.context_source = context_source
        self.use_selected_score_weight = bool(use_selected_score_weight)
        self.debug = bool(debug)

        self.bank = TextGraphFeatureBank(
            text_features_path=text_features_path,
            text_mask_path=text_mask_path,
            hoi_ids_path=hoi_ids_path,
            object_to_hoi_indices_path=object_to_hoi_indices_path,
            field_order_path=field_order_path,
            d_model=self.d_model,
            strict_shape=strict_shape,
            debug=debug,
        )
        self.retriever = TextHOIRetriever(
            bank=self.bank,
            d_model=self.d_model,
            topk=topk,
            pre_topk=pre_topk,
            temperature=temperature,
            rerank_weight=rerank_weight,
            use_discriminative_rerank=use_discriminative_rerank,
            debug=debug,
        )
        self.reasoner = TextSceneGraphReasoner(
            retriever=self.retriever,
            d_model=self.d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_scene_token=use_scene_token,
            use_oo_context_token=use_oo_context_token,
            enable_prototype_slots=True,
            prototype_slot_fallback_to_raw=prototype_slot_fallback_to_raw,
            prototype_slot_debug=prototype_slot_debug,
            debug=debug,
        )

        context_in_dim = self.d_model * 2 if context_source == "concat_slots_pooled" else self.d_model
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_in_dim),
            nn.Linear(context_in_dim, self.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.last_debug: Dict[str, Any] = {}

    def forward(
        self,
        visual_pair_feats: Tensor,
        pair_object_labels: Tensor,
        pair_valid_mask: Optional[Tensor] = None,
    ) -> LPTOutput:
        self._check_inputs(visual_pair_feats, pair_object_labels, pair_valid_mask)
        if pair_object_labels.device != visual_pair_feats.device:
            pair_object_labels = pair_object_labels.to(device=visual_pair_feats.device)
        if pair_valid_mask is None:
            pair_valid_mask = torch.ones(
                visual_pair_feats.shape[:2],
                dtype=torch.bool,
                device=visual_pair_feats.device,
            )
        else:
            pair_valid_mask = pair_valid_mask.to(device=visual_pair_feats.device, dtype=torch.bool)

        try:
            text_out = self.reasoner(
                visual_pair_feats=visual_pair_feats,
                pair_object_labels=pair_object_labels,
                pair_valid_mask=pair_valid_mask,
            )
        except TypeError:
            text_out = self.reasoner(visual_pair_feats, pair_object_labels, pair_valid_mask)

        self._require_text_output_fields(text_out)

        prototype_slots = text_out.prototype_slots
        prototype_slot_mask = text_out.prototype_slot_mask.to(dtype=torch.bool)
        selected_hoi_mask = text_out.selected_hoi_mask.to(dtype=torch.bool)
        selected_scores = text_out.selected_scores.to(device=visual_pair_feats.device, dtype=visual_pair_feats.dtype)

        slot_summary = self._aggregate_slots(
            prototype_slots=prototype_slots,
            prototype_slot_mask=prototype_slot_mask,
            selected_scores=selected_scores,
            selected_hoi_mask=selected_hoi_mask,
        )

        if self.context_source == "slots":
            prototype_context = self.context_proj(slot_summary)
        elif self.context_source == "pooled_all":
            prototype_context = self.context_proj(text_out.pooled_all)
        else:
            prototype_context = self.context_proj(torch.cat([slot_summary, text_out.pooled_all], dim=-1))

        pair_valid_float = pair_valid_mask.unsqueeze(-1).to(dtype=prototype_context.dtype)
        prototype_context = torch.nan_to_num(prototype_context, nan=0.0, posinf=0.0, neginf=0.0)
        prototype_context = prototype_context * pair_valid_float
        slot_summary = torch.nan_to_num(slot_summary, nan=0.0, posinf=0.0, neginf=0.0) * pair_valid_float

        debug_info = dict(getattr(text_out, "debug_info", {}))
        debug_info.update(self._build_debug_info(
            slot_summary=slot_summary,
            prototype_context=prototype_context,
            prototype_slot_mask=prototype_slot_mask,
            selected_hoi_mask=selected_hoi_mask,
            pair_valid_mask=pair_valid_mask,
        ))
        self.last_debug = debug_info

        return LPTOutput(
            prototype_slots=prototype_slots,
            prototype_slot_mask=prototype_slot_mask,
            selected_hoi_indices=text_out.selected_hoi_indices,
            selected_hoi_mask=selected_hoi_mask,
            selected_scores=selected_scores,
            text_tokens=text_out.text_tokens,
            text_token_mask=text_out.text_token_mask,
            pooled_human_node=text_out.pooled_human_node,
            pooled_object_node=text_out.pooled_object_node,
            pooled_ho_edge=text_out.pooled_ho_edge,
            pooled_discriminative=text_out.pooled_discriminative,
            pooled_scene_context=text_out.pooled_scene_context,
            pooled_oo_context=text_out.pooled_oo_context,
            pooled_all=text_out.pooled_all,
            slot_summary=slot_summary,
            prototype_context=prototype_context,
            pair_valid_mask=pair_valid_mask,
            debug_info=debug_info,
        )

    def _aggregate_slots(
        self,
        prototype_slots: Tensor,
        prototype_slot_mask: Tensor,
        selected_scores: Tensor,
        selected_hoi_mask: Tensor,
    ) -> Tensor:
        slot_valid = prototype_slot_mask.to(dtype=torch.bool)
        per_proto_slot_sum = (prototype_slots * slot_valid.unsqueeze(-1).to(dtype=prototype_slots.dtype)).sum(dim=-2)
        per_proto_slot_count = slot_valid.sum(dim=-1).clamp(min=1)
        per_proto_summary = per_proto_slot_sum / per_proto_slot_count.unsqueeze(-1).to(dtype=prototype_slots.dtype)

        if self.use_selected_score_weight:
            score_logits = selected_scores.masked_fill(~selected_hoi_mask, -1e4)
            proto_weights = torch.softmax(score_logits, dim=-1)
            proto_weights = proto_weights * selected_hoi_mask.to(dtype=proto_weights.dtype)
            proto_weights = proto_weights / proto_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        else:
            proto_weights = selected_hoi_mask.to(dtype=prototype_slots.dtype)
            proto_weights = proto_weights / proto_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        slot_summary = (proto_weights.unsqueeze(-1).to(dtype=per_proto_summary.dtype) * per_proto_summary).sum(dim=-2)
        return torch.nan_to_num(slot_summary, nan=0.0, posinf=0.0, neginf=0.0)

    def _require_text_output_fields(self, text_out: Any) -> None:
        required_fields = (
            "prototype_slots",
            "prototype_slot_mask",
            "selected_hoi_indices",
            "selected_hoi_mask",
            "selected_scores",
            "text_tokens",
            "text_token_mask",
            "pooled_human_node",
            "pooled_object_node",
            "pooled_ho_edge",
            "pooled_discriminative",
            "pooled_scene_context",
            "pooled_oo_context",
            "pooled_all",
        )
        missing = [field_name for field_name in required_fields if not hasattr(text_out, field_name)]
        if missing:
            raise RuntimeError(f"TextSceneGraphReasoner output is missing required LPT fields: {missing}")

    def _check_inputs(
        self,
        visual_pair_feats: Tensor,
        pair_object_labels: Tensor,
        pair_valid_mask: Optional[Tensor],
    ) -> None:
        if visual_pair_feats.dim() != 3:
            raise ValueError(f"visual_pair_feats must be [B, K, D], got {tuple(visual_pair_feats.shape)}")
        if visual_pair_feats.shape[-1] != self.d_model:
            raise ValueError(f"visual_pair_feats last dim must be {self.d_model}, got {visual_pair_feats.shape[-1]}")
        if pair_object_labels.shape != visual_pair_feats.shape[:2]:
            raise ValueError(f"pair_object_labels must be [B, K], got {tuple(pair_object_labels.shape)}")
        if pair_object_labels.dtype != torch.long:
            raise ValueError(f"pair_object_labels must be torch.long, got {pair_object_labels.dtype}")
        if pair_valid_mask is not None and pair_valid_mask.shape != visual_pair_feats.shape[:2]:
            raise ValueError(f"pair_valid_mask must be [B, K], got {tuple(pair_valid_mask.shape)}")

    def _build_debug_info(
        self,
        slot_summary: Tensor,
        prototype_context: Tensor,
        prototype_slot_mask: Tensor,
        selected_hoi_mask: Tensor,
        pair_valid_mask: Tensor,
    ) -> Dict[str, Any]:
        valid_pairs = int(pair_valid_mask.sum().detach().cpu().item()) if pair_valid_mask.numel() > 0 else 0
        slot_valid_count = int(prototype_slot_mask.sum().detach().cpu().item()) if prototype_slot_mask.numel() > 0 else 0
        selected_count = int(selected_hoi_mask.sum().detach().cpu().item()) if selected_hoi_mask.numel() > 0 else 0
        return {
            "lpt_enabled": True,
            "lpt_context_source": self.context_source,
            "lpt_use_selected_score_weight": self.use_selected_score_weight,
            "lpt_pair_valid_count": valid_pairs,
            "lpt_selected_hoi_count": selected_count,
            "lpt_prototype_slot_valid_count": slot_valid_count,
            "lpt_slot_summary_shape": tuple(slot_summary.shape),
            "lpt_prototype_context_shape": tuple(prototype_context.shape),
            "lpt_slot_summary_norm": float(slot_summary.detach().float().norm(dim=-1).mean().cpu().item()) if slot_summary.numel() > 0 else 0.0,
            "lpt_prototype_context_norm": float(prototype_context.detach().float().norm(dim=-1).mean().cpu().item()) if prototype_context.numel() > 0 else 0.0,
            "lpt_has_nan": bool(
                torch.isnan(slot_summary).any().detach().cpu().item()
                or torch.isnan(prototype_context).any().detach().cpu().item()
            ),
            "no_setm": True,
            "no_evidence_prototype_matching": True,
            "no_visual_text_token_updates": True,
            "no_" + "layer" + "wise_synchronization": True,
        }
