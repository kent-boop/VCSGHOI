from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class SynchronizedEvidenceOutput:
    match_logits: Tensor
    visual_to_text_attn: Tensor
    text_to_visual_attn: Tensor
    updated_visual_tokens: Tensor
    updated_prototype_slots: Tensor
    evidence_weights: Tensor
    prototype_scores: Tensor
    prototype_weights: Tensor
    prototype_coverage: Tensor
    matched_context: Tensor
    anchor_delta: Tensor
    debug_info: Dict[str, Any]


class SynchronizedEvidenceTopologyMatcher(nn.Module):
    """
    Match candidate visual evidence tokens with language-derived prototype slots.

    Inputs:
        visual_tokens: [B, K, Mv, D]
        visual_mask: [B, K, Mv]
        prototype_slots: [B, K, R, S, D]
        prototype_slot_mask: [B, K, R, S]

    Core:
        match_logits: [B, K, R, Mv, S]
        visual_to_text_attn: visual evidence -> text slots
        text_to_visual_attn: text slots -> visual evidence

    Outputs:
        bidirectionally updated visual tokens and prototype slots,
        topology/evidence weights, matched_context, and anchor_delta.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_slots: int = 6,
        dropout: float = 0.1,
        visual_gate_init: float = 0.0,
        text_gate_init: float = 0.0,
        anchor_gate_init: float = 0.0,
        normalize_inputs: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        self.d_model = int(d_model)
        self.num_slots = int(num_slots)
        self.normalize_inputs = bool(normalize_inputs)
        self.debug = bool(debug)

        self.visual_proj = nn.Linear(d_model, d_model)
        self.text_proj = nn.Linear(d_model, d_model)

        self.visual_update_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.text_update_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.evidence_score_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.prototype_score_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 2 + num_slots),
            nn.Linear(d_model * 2 + num_slots, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        self.anchor_delta_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        self.visual_gate = nn.Parameter(torch.tensor(float(visual_gate_init)))
        self.text_gate = nn.Parameter(torch.tensor(float(text_gate_init)))
        self.anchor_gate = nn.Parameter(torch.tensor(float(anchor_gate_init)))

    def forward(
        self,
        visual_tokens: Tensor,
        visual_mask: Tensor,
        prototype_slots: Tensor,
        prototype_slot_mask: Tensor,
        pair_valid_mask: Optional[Tensor] = None,
    ) -> SynchronizedEvidenceOutput:
        self._check_inputs(
            visual_tokens=visual_tokens,
            visual_mask=visual_mask,
            prototype_slots=prototype_slots,
            prototype_slot_mask=prototype_slot_mask,
            pair_valid_mask=pair_valid_mask,
        )
        batch_size, num_pairs, num_visual_tokens, dim = visual_tokens.shape
        _, _, num_prototypes, num_slots, _ = prototype_slots.shape

        visual_tokens = torch.nan_to_num(visual_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        prototype_slots = torch.nan_to_num(prototype_slots, nan=0.0, posinf=0.0, neginf=0.0)
        visual_mask = visual_mask.to(dtype=torch.bool)
        prototype_slot_mask = prototype_slot_mask.to(dtype=torch.bool)
        if pair_valid_mask is None:
            pair_valid_mask = torch.ones((batch_size, num_pairs), dtype=torch.bool, device=visual_tokens.device)
        else:
            pair_valid_mask = pair_valid_mask.to(dtype=torch.bool, device=visual_tokens.device)

        visual_tokens = visual_tokens * visual_mask.unsqueeze(-1).to(dtype=visual_tokens.dtype)
        prototype_slots = prototype_slots * prototype_slot_mask.unsqueeze(-1).to(dtype=prototype_slots.dtype)

        visual_q = self.visual_proj(visual_tokens)
        text_k = self.text_proj(prototype_slots)
        if self.normalize_inputs:
            visual_q = F.normalize(visual_q, dim=-1, eps=1e-6)
            text_k = F.normalize(text_k, dim=-1, eps=1e-6)

        raw_match_logits = torch.einsum("bkmd,bkrsd->bkrms", visual_q, text_k) / math.sqrt(self.d_model)
        raw_match_logits = torch.nan_to_num(raw_match_logits, nan=0.0, posinf=0.0, neginf=0.0)

        valid_match_mask = (
            pair_valid_mask[:, :, None, None, None]
            & visual_mask[:, :, None, :, None]
            & prototype_slot_mask[:, :, :, None, :]
        )
        valid_proto = prototype_slot_mask.any(dim=-1) & pair_valid_mask[:, :, None]
        valid_visual_for_proto = visual_mask[:, :, None, :] & valid_proto[:, :, :, None]
        valid_slot_for_proto = prototype_slot_mask & pair_valid_mask[:, :, None, None]
        match_logits = raw_match_logits.masked_fill(~valid_match_mask, 0.0)
        visual_to_text_attn = self._masked_softmax(match_logits, valid_match_mask, dim=-1)
        text_to_visual_attn = self._masked_softmax(
            match_logits.transpose(-1, -2),
            valid_match_mask.transpose(-1, -2),
            dim=-1,
        )

        visual_slot_context = torch.einsum(
            "bkrms,bkrsd->bkrmd",
            visual_to_text_attn,
            prototype_slots,
        )
        visual_broadcast = visual_tokens.unsqueeze(2).expand(-1, -1, num_prototypes, -1, -1)

        visual_update_input = torch.cat([visual_broadcast, visual_slot_context], dim=-1)
        visual_delta = self.visual_update_proj(visual_update_input)
        updated_visual_tokens = visual_broadcast + torch.tanh(self.visual_gate).to(dtype=visual_delta.dtype) * visual_delta
        updated_visual_tokens = updated_visual_tokens * valid_visual_for_proto.unsqueeze(-1).to(dtype=updated_visual_tokens.dtype)
        updated_visual_tokens = torch.nan_to_num(updated_visual_tokens, nan=0.0, posinf=0.0, neginf=0.0)

        text_visual_context = torch.einsum(
            "bkrsm,bkrmd->bkrsd",
            text_to_visual_attn,
            visual_broadcast,
        )
        text_update_input = torch.cat([prototype_slots, text_visual_context], dim=-1)
        text_delta = self.text_update_proj(text_update_input)
        updated_prototype_slots = prototype_slots + torch.tanh(self.text_gate).to(dtype=text_delta.dtype) * text_delta
        updated_prototype_slots = updated_prototype_slots * valid_slot_for_proto.unsqueeze(-1).to(dtype=updated_prototype_slots.dtype)
        updated_prototype_slots = torch.nan_to_num(updated_prototype_slots, nan=0.0, posinf=0.0, neginf=0.0)

        evidence_score_input = torch.cat([
            updated_visual_tokens,
            visual_slot_context,
            updated_visual_tokens * visual_slot_context,
        ], dim=-1)
        evidence_logits = self.evidence_score_mlp(evidence_score_input).squeeze(-1)
        evidence_logits = torch.nan_to_num(evidence_logits, nan=0.0, posinf=0.0, neginf=0.0)
        evidence_mask = visual_mask[:, :, None, :] & valid_proto[:, :, :, None]
        evidence_weights = self._masked_softmax(evidence_logits, evidence_mask, dim=-1)

        matched_context_per_proto = torch.einsum(
            "bkrm,bkrmd->bkrd",
            evidence_weights,
            updated_visual_tokens,
        )
        matched_context_per_proto = torch.nan_to_num(matched_context_per_proto, nan=0.0, posinf=0.0, neginf=0.0)
        matched_context_per_proto = matched_context_per_proto * valid_proto.unsqueeze(-1).to(dtype=matched_context_per_proto.dtype)

        prototype_coverage = text_to_visual_attn.max(dim=-1).values
        prototype_coverage = prototype_coverage * prototype_slot_mask.to(dtype=prototype_coverage.dtype)
        prototype_coverage = torch.nan_to_num(prototype_coverage, nan=0.0, posinf=0.0, neginf=0.0)

        slot_weights = prototype_slot_mask.to(dtype=updated_prototype_slots.dtype)
        prototype_text_context = (
            updated_prototype_slots * slot_weights.unsqueeze(-1)
        ).sum(dim=-2) / slot_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
        prototype_text_context = torch.nan_to_num(prototype_text_context, nan=0.0, posinf=0.0, neginf=0.0)
        prototype_text_context = prototype_text_context * valid_proto.unsqueeze(-1).to(dtype=prototype_text_context.dtype)

        coverage_summary = prototype_coverage
        prototype_score_input = torch.cat([
            matched_context_per_proto,
            prototype_text_context,
            coverage_summary,
        ], dim=-1)
        prototype_scores = self.prototype_score_mlp(prototype_score_input).squeeze(-1)
        prototype_scores = torch.nan_to_num(prototype_scores, nan=0.0, posinf=0.0, neginf=0.0)
        prototype_scores = prototype_scores.masked_fill(~valid_proto, -1e4)
        prototype_weights = self._masked_softmax(prototype_scores, valid_proto, dim=-1)

        matched_context = torch.einsum(
            "bkr,bkrd->bkd",
            prototype_weights,
            matched_context_per_proto,
        )
        matched_context = matched_context * pair_valid_mask.unsqueeze(-1).to(dtype=matched_context.dtype)
        matched_context = torch.nan_to_num(matched_context, nan=0.0, posinf=0.0, neginf=0.0)

        anchor_delta_raw = self.anchor_delta_proj(matched_context)
        anchor_delta = torch.tanh(self.anchor_gate).to(dtype=anchor_delta_raw.dtype) * anchor_delta_raw
        anchor_delta = anchor_delta * pair_valid_mask.unsqueeze(-1).to(dtype=anchor_delta.dtype)
        anchor_delta = torch.nan_to_num(anchor_delta, nan=0.0, posinf=0.0, neginf=0.0)

        debug_info = self._build_debug_info(
            match_logits=match_logits,
            updated_visual_tokens=updated_visual_tokens,
            updated_prototype_slots=updated_prototype_slots,
            evidence_weights=evidence_weights,
            prototype_scores=prototype_scores,
            visual_mask=visual_mask,
            prototype_slot_mask=prototype_slot_mask,
            pair_valid_mask=pair_valid_mask,
            valid_proto=valid_proto,
            anchor_delta=anchor_delta,
        )

        return SynchronizedEvidenceOutput(
            match_logits=match_logits,
            visual_to_text_attn=visual_to_text_attn,
            text_to_visual_attn=text_to_visual_attn,
            updated_visual_tokens=updated_visual_tokens,
            updated_prototype_slots=updated_prototype_slots,
            evidence_weights=evidence_weights,
            prototype_scores=prototype_scores,
            prototype_weights=prototype_weights,
            prototype_coverage=prototype_coverage,
            matched_context=matched_context,
            anchor_delta=anchor_delta,
            debug_info=debug_info,
        )

    def _masked_softmax(self, logits: Tensor, mask: Tensor, dim: int) -> Tensor:
        mask = mask.to(dtype=torch.bool)
        safe_logits = logits.masked_fill(~mask, -1e4)
        attn = torch.softmax(safe_logits, dim=dim)
        attn = attn * mask.to(dtype=attn.dtype)
        denom = attn.sum(dim=dim, keepdim=True)
        attn = torch.where(denom > 0, attn / denom.clamp(min=1e-6), torch.zeros_like(attn))
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        return attn

    def _build_debug_info(
        self,
        match_logits: Tensor,
        updated_visual_tokens: Tensor,
        updated_prototype_slots: Tensor,
        evidence_weights: Tensor,
        prototype_scores: Tensor,
        visual_mask: Tensor,
        prototype_slot_mask: Tensor,
        pair_valid_mask: Tensor,
        valid_proto: Tensor,
        anchor_delta: Tensor,
    ) -> Dict[str, Any]:
        return {
            "sync_evidence_enabled": True,
            "sync_match_logits_shape": tuple(match_logits.shape),
            "sync_updated_visual_tokens_shape": tuple(updated_visual_tokens.shape),
            "sync_updated_prototype_slots_shape": tuple(updated_prototype_slots.shape),
            "sync_evidence_weights_shape": tuple(evidence_weights.shape),
            "sync_prototype_scores_shape": tuple(prototype_scores.shape),
            "sync_visual_token_valid_ratio": float(visual_mask.float().mean().detach().cpu().item()) if visual_mask.numel() > 0 else 0.0,
            "sync_prototype_slot_valid_ratio": float(prototype_slot_mask.float().mean().detach().cpu().item()) if prototype_slot_mask.numel() > 0 else 0.0,
            "sync_valid_pair_count": int(pair_valid_mask.sum().detach().cpu().item()) if pair_valid_mask.numel() > 0 else 0,
            "sync_valid_prototype_count": int(valid_proto.sum().detach().cpu().item()) if valid_proto.numel() > 0 else 0,
            "sync_anchor_delta_norm_mean": float(anchor_delta.detach().float().norm(dim=-1).mean().cpu().item()) if anchor_delta.numel() > 0 else 0.0,
            "sync_visual_gate": float(torch.tanh(self.visual_gate).detach().cpu().item()),
            "sync_text_gate": float(torch.tanh(self.text_gate).detach().cpu().item()),
            "sync_anchor_gate": float(torch.tanh(self.anchor_gate).detach().cpu().item()),
            "sync_has_nan": bool(
                torch.isnan(match_logits).any().detach().cpu().item()
                or torch.isnan(updated_visual_tokens).any().detach().cpu().item()
                or torch.isnan(updated_prototype_slots).any().detach().cpu().item()
                or torch.isnan(anchor_delta).any().detach().cpu().item()
            ),
        }

    def _check_inputs(
        self,
        visual_tokens: Tensor,
        visual_mask: Tensor,
        prototype_slots: Tensor,
        prototype_slot_mask: Tensor,
        pair_valid_mask: Optional[Tensor],
    ) -> None:
        if visual_tokens.dim() != 4:
            raise ValueError(f"visual_tokens must be [B,K,Mv,D], got {tuple(visual_tokens.shape)}")
        if visual_mask.shape != visual_tokens.shape[:3]:
            raise ValueError(
                f"visual_mask must match visual_tokens shape[:3]={tuple(visual_tokens.shape[:3])}, "
                f"got {tuple(visual_mask.shape)}"
            )
        if prototype_slots.dim() != 5:
            raise ValueError(f"prototype_slots must be [B,K,R,S,D], got {tuple(prototype_slots.shape)}")
        if prototype_slot_mask.shape != prototype_slots.shape[:4]:
            raise ValueError(
                f"prototype_slot_mask must match prototype_slots shape[:4]={tuple(prototype_slots.shape[:4])}, "
                f"got {tuple(prototype_slot_mask.shape)}"
            )
        if visual_tokens.shape[-1] != self.d_model:
            raise ValueError(f"visual_tokens D must equal d_model={self.d_model}, got {visual_tokens.shape[-1]}")
        if prototype_slots.shape[-1] != self.d_model:
            raise ValueError(f"prototype_slots D must equal d_model={self.d_model}, got {prototype_slots.shape[-1]}")
        if prototype_slots.shape[0:2] != visual_tokens.shape[0:2]:
            raise ValueError(
                f"prototype_slots [B,K] must match visual_tokens [B,K]={tuple(visual_tokens.shape[0:2])}, "
                f"got {tuple(prototype_slots.shape[0:2])}"
            )
        if prototype_slots.shape[3] != self.num_slots:
            raise ValueError(f"prototype_slots S must equal num_slots={self.num_slots}, got {prototype_slots.shape[3]}")
        if pair_valid_mask is not None and tuple(pair_valid_mask.shape) != tuple(visual_tokens.shape[:2]):
            raise ValueError(
                f"pair_valid_mask must be [B,K]={tuple(visual_tokens.shape[:2])}, got {tuple(pair_valid_mask.shape)}"
            )
