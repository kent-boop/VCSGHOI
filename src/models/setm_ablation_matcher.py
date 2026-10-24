"""Configurable Synchronized Evidence-Topology Matching ablations."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class SETMAblationOutput:
    match_logits: Tensor
    visual_to_text_attn: Tensor
    text_to_visual_attn: Tensor
    updated_visual_tokens: Tensor
    updated_prototype_slots: Tensor
    evidence_weights: Tensor
    prototype_scores: Tensor
    prototype_weights: Tensor
    prototype_coverage: Tensor
    matched_context_per_proto: Tensor
    matched_context: Tensor
    anchor_delta: Tensor
    pair_valid_mask: Tensor
    valid_proto_mask: Tensor
    stage: int
    stage_name: str
    debug_info: Dict[str, Any]


class SETMAblationMatcher(nn.Module):
    """Match visual evidence and prototype slots with stage-controlled updates."""

    STAGE_NAMES = {
        1: "one_way_match",
        2: "bidirectional_update",
        3: "evidence_reweighting",
        4: "prototype_reweighting",
        5: "layerwise_synchronization",
    }

    def __init__(
        self,
        d_model: int = 256,
        num_slots: int = 6,
        dropout: float = 0.1,
        visual_gate_init: float = 0.0,
        text_gate_init: float = 0.0,
        anchor_gate_init: float = 0.0,
        normalize_inputs: bool = True,
        stage: int = 5,
        debug: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_slots <= 0:
            raise ValueError(f"num_slots must be positive, got {num_slots}")
        self._validate_stage(stage)

        self.d_model = int(d_model)
        self.num_slots = int(num_slots)
        self.stage = int(stage)
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

    def get_stage_flags(self) -> Dict[str, bool]:
        return self._stage_flags(self.stage)

    def _stage_flags(self, stage: int) -> Dict[str, bool]:
        self._validate_stage(stage)
        return {
            "enable_visual_update": stage >= 1,
            "enable_text_update": stage >= 2,
            "enable_evidence_reweight": stage >= 3,
            "enable_prototype_reweight": stage >= 4,
            "enable_prototype_coverage": stage >= 4,
        }

    def forward(
        self,
        visual_tokens: Tensor,
        visual_mask: Tensor,
        prototype_slots: Tensor,
        prototype_slot_mask: Tensor,
        pair_valid_mask: Optional[Tensor] = None,
        stage: Optional[int] = None,
    ) -> SETMAblationOutput:
        active_stage = self.stage if stage is None else int(stage)
        self._validate_stage(active_stage)
        self._check_inputs(
            visual_tokens,
            visual_mask,
            prototype_slots,
            prototype_slot_mask,
            pair_valid_mask,
        )
        flags = self._stage_flags(active_stage)

        batch_size, num_pairs, num_visual_tokens, _ = visual_tokens.shape
        num_prototypes = prototype_slots.shape[2]
        visual_tokens = self._sanitize(visual_tokens)
        prototype_slots = self._sanitize(prototype_slots)
        visual_mask = visual_mask.to(device=visual_tokens.device, dtype=torch.bool)
        prototype_slot_mask = prototype_slot_mask.to(device=visual_tokens.device, dtype=torch.bool)

        inferred_pair_mask = visual_mask.any(dim=-1) & prototype_slot_mask.flatten(2).any(dim=-1)
        if pair_valid_mask is None:
            pair_valid_mask = inferred_pair_mask
        else:
            pair_valid_mask = pair_valid_mask.to(device=visual_tokens.device, dtype=torch.bool)

        valid_proto_mask = prototype_slot_mask.any(dim=-1) & pair_valid_mask[:, :, None]
        valid_visual_for_proto = visual_mask[:, :, None, :] & valid_proto_mask[:, :, :, None]
        valid_slot_for_proto = prototype_slot_mask & valid_proto_mask[:, :, :, None]
        valid_match_mask = (
            valid_visual_for_proto[:, :, :, :, None]
            & valid_slot_for_proto[:, :, :, None, :]
        )

        visual_tokens = visual_tokens * visual_mask.unsqueeze(-1).to(dtype=visual_tokens.dtype)
        prototype_slots = prototype_slots * prototype_slot_mask.unsqueeze(-1).to(dtype=prototype_slots.dtype)
        visual_proj = self.visual_proj(visual_tokens)
        text_proj = self.text_proj(prototype_slots)
        if self.normalize_inputs:
            visual_proj = F.normalize(visual_proj, dim=-1, eps=1e-6)
            text_proj = F.normalize(text_proj, dim=-1, eps=1e-6)

        match_logits = torch.einsum("bkmd,bkrsd->bkrms", visual_proj, text_proj)
        match_logits = self._sanitize(match_logits / math.sqrt(self.d_model))
        match_logits = match_logits.masked_fill(~valid_match_mask, 0.0)
        visual_to_text_attn = self._masked_softmax(match_logits, valid_match_mask, dim=-1)
        if flags["enable_text_update"]:
            text_to_visual_attn = self._masked_softmax(
                match_logits.transpose(-1, -2),
                valid_match_mask.transpose(-1, -2),
                dim=-1,
            )
        else:
            text_to_visual_attn = match_logits.new_zeros(
                batch_size,
                num_pairs,
                num_prototypes,
                self.num_slots,
                num_visual_tokens,
            )

        visual_broadcast = visual_tokens.unsqueeze(2).expand(-1, -1, num_prototypes, -1, -1)
        visual_slot_context = torch.einsum(
            "bkrms,bkrsd->bkrmd",
            visual_to_text_attn,
            prototype_slots,
        )
        visual_delta = self.visual_update_proj(
            torch.cat([visual_broadcast, visual_slot_context], dim=-1)
        )
        visual_gate = torch.tanh(self.visual_gate).to(dtype=visual_delta.dtype)
        updated_visual_tokens = visual_broadcast + visual_gate * visual_delta
        updated_visual_tokens = self._mask_and_sanitize(
            updated_visual_tokens,
            valid_visual_for_proto,
        )

        updated_prototype_slots = prototype_slots
        visual_slot_context_for_scoring = visual_slot_context
        if flags["enable_text_update"]:
            text_visual_context = torch.einsum(
                "bkrsm,bkrmd->bkrsd",
                text_to_visual_attn,
                visual_broadcast,
            )
            text_delta = self.text_update_proj(
                torch.cat([prototype_slots, text_visual_context], dim=-1)
            )
            text_gate = torch.tanh(self.text_gate).to(dtype=text_delta.dtype)
            updated_prototype_slots = prototype_slots + text_gate * text_delta
            updated_prototype_slots = self._mask_and_sanitize(
                updated_prototype_slots,
                valid_slot_for_proto,
            )

            visual_slot_context_for_scoring = torch.einsum(
                "bkrms,bkrsd->bkrmd",
                visual_to_text_attn,
                updated_prototype_slots,
            )
            visual_delta_refined = self.visual_update_proj(
                torch.cat(
                    [updated_visual_tokens, visual_slot_context_for_scoring],
                    dim=-1,
                )
            )
            updated_visual_tokens = updated_visual_tokens + visual_gate * visual_delta_refined
            updated_visual_tokens = self._mask_and_sanitize(
                updated_visual_tokens,
                valid_visual_for_proto,
            )

        if flags["enable_evidence_reweight"]:
            evidence_score_input = torch.cat(
                [
                    updated_visual_tokens,
                    visual_slot_context_for_scoring,
                    updated_visual_tokens * visual_slot_context_for_scoring,
                ],
                dim=-1,
            )
            evidence_logits = self._sanitize(
                self.evidence_score_mlp(evidence_score_input).squeeze(-1)
            )
            evidence_weights = self._masked_softmax(
                evidence_logits,
                valid_visual_for_proto,
                dim=-1,
            )
        else:
            evidence_weights = self._uniform_weights(
                valid_visual_for_proto,
                dim=-1,
                dtype=updated_visual_tokens.dtype,
            )

        matched_context_per_proto = torch.einsum(
            "bkrm,bkrmd->bkrd",
            evidence_weights,
            updated_visual_tokens,
        )
        matched_context_per_proto = self._mask_and_sanitize(
            matched_context_per_proto,
            valid_proto_mask,
        )

        if flags["enable_prototype_coverage"]:
            prototype_coverage = text_to_visual_attn.max(dim=-1).values
            prototype_coverage = prototype_coverage * prototype_slot_mask.to(
                dtype=prototype_coverage.dtype
            )
            prototype_coverage = self._sanitize(prototype_coverage)
        else:
            prototype_coverage = prototype_slots.new_zeros(prototype_slot_mask.shape)

        if flags["enable_prototype_reweight"]:
            slot_weights = prototype_slot_mask.to(dtype=updated_prototype_slots.dtype)
            prototype_text_context = (
                updated_prototype_slots * slot_weights.unsqueeze(-1)
            ).sum(dim=-2) / slot_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
            prototype_text_context = self._mask_and_sanitize(
                prototype_text_context,
                valid_proto_mask,
            )
            prototype_score_input = torch.cat(
                [
                    matched_context_per_proto,
                    prototype_text_context,
                    prototype_coverage,
                ],
                dim=-1,
            )
            prototype_scores = self._sanitize(
                self.prototype_score_mlp(prototype_score_input).squeeze(-1)
            )
            prototype_scores = prototype_scores.masked_fill(~valid_proto_mask, -1e4)
            prototype_weights = self._masked_softmax(
                prototype_scores,
                valid_proto_mask,
                dim=-1,
            )
        else:
            prototype_scores = prototype_slots.new_zeros(
                batch_size,
                num_pairs,
                num_prototypes,
            )
            prototype_weights = self._uniform_weights(
                valid_proto_mask,
                dim=-1,
                dtype=matched_context_per_proto.dtype,
            )

        matched_context = torch.einsum(
            "bkr,bkrd->bkd",
            prototype_weights,
            matched_context_per_proto,
        )
        matched_context = self._mask_and_sanitize(matched_context, pair_valid_mask)
        anchor_delta_raw = self.anchor_delta_proj(matched_context)
        anchor_delta = torch.tanh(self.anchor_gate).to(
            dtype=anchor_delta_raw.dtype
        ) * anchor_delta_raw
        anchor_delta = self._mask_and_sanitize(anchor_delta, pair_valid_mask)

        outputs = {
            "match_logits": self._sanitize(match_logits),
            "visual_to_text_attn": self._sanitize(visual_to_text_attn),
            "text_to_visual_attn": self._sanitize(text_to_visual_attn),
            "updated_visual_tokens": self._sanitize(updated_visual_tokens),
            "updated_prototype_slots": self._sanitize(updated_prototype_slots),
            "evidence_weights": self._sanitize(evidence_weights),
            "prototype_scores": self._sanitize(prototype_scores),
            "prototype_weights": self._sanitize(prototype_weights),
            "prototype_coverage": self._sanitize(prototype_coverage),
            "matched_context_per_proto": self._sanitize(matched_context_per_proto),
            "matched_context": self._sanitize(matched_context),
            "anchor_delta": self._sanitize(anchor_delta),
        }
        stage_name = self.STAGE_NAMES[active_stage]
        debug_info = self._build_debug_info(
            active_stage,
            stage_name,
            flags,
            pair_valid_mask,
            valid_proto_mask,
            outputs,
        )

        return SETMAblationOutput(
            **outputs,
            pair_valid_mask=pair_valid_mask,
            valid_proto_mask=valid_proto_mask,
            stage=active_stage,
            stage_name=stage_name,
            debug_info=debug_info,
        )

    @staticmethod
    def _validate_stage(stage: int) -> None:
        if isinstance(stage, bool) or stage not in {1, 2, 3, 4, 5}:
            raise ValueError(f"stage must be one of {{1,2,3,4,5}}, got {stage}")

    @staticmethod
    def _sanitize(tensor: Tensor) -> Tensor:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _mask_and_sanitize(self, tensor: Tensor, mask: Tensor) -> Tensor:
        tensor = tensor * mask.unsqueeze(-1).to(dtype=tensor.dtype)
        return self._sanitize(tensor)

    def _masked_softmax(self, logits: Tensor, mask: Tensor, dim: int) -> Tensor:
        mask = mask.to(dtype=torch.bool)
        safe_logits = logits.masked_fill(~mask, -1e4)
        weights = torch.softmax(safe_logits, dim=dim)
        weights = weights * mask.to(dtype=weights.dtype)
        denominator = weights.sum(dim=dim, keepdim=True)
        weights = torch.where(
            denominator > 0,
            weights / denominator.clamp(min=1e-6),
            torch.zeros_like(weights),
        )
        return self._sanitize(weights)

    def _uniform_weights(self, mask: Tensor, dim: int, dtype: torch.dtype) -> Tensor:
        weights = mask.to(dtype=dtype)
        denominator = weights.sum(dim=dim, keepdim=True)
        weights = torch.where(
            denominator > 0,
            weights / denominator.clamp(min=1.0),
            torch.zeros_like(weights),
        )
        return self._sanitize(weights)

    def _build_debug_info(
        self,
        stage: int,
        stage_name: str,
        flags: Dict[str, bool],
        pair_valid_mask: Tensor,
        valid_proto_mask: Tensor,
        outputs: Dict[str, Tensor],
    ) -> Dict[str, Any]:
        has_nan = any(
            bool(torch.isnan(value).any().detach().cpu().item())
            for value in outputs.values()
        )
        return {
            "setm_ablation_stage": stage,
            "setm_stage_name": stage_name,
            **flags,
            "match_logits_shape": tuple(outputs["match_logits"].shape),
            "visual_to_text_attn_shape": tuple(outputs["visual_to_text_attn"].shape),
            "text_to_visual_attn_shape": tuple(outputs["text_to_visual_attn"].shape),
            "updated_visual_tokens_shape": tuple(outputs["updated_visual_tokens"].shape),
            "updated_prototype_slots_shape": tuple(outputs["updated_prototype_slots"].shape),
            "evidence_weights_shape": tuple(outputs["evidence_weights"].shape),
            "prototype_weights_shape": tuple(outputs["prototype_weights"].shape),
            "prototype_coverage_shape": tuple(outputs["prototype_coverage"].shape),
            "matched_context_shape": tuple(outputs["matched_context"].shape),
            "anchor_delta_shape": tuple(outputs["anchor_delta"].shape),
            "visual_gate": float(torch.tanh(self.visual_gate).detach().cpu().item()),
            "text_gate": float(torch.tanh(self.text_gate).detach().cpu().item()),
            "anchor_gate": float(torch.tanh(self.anchor_gate).detach().cpu().item()),
            "valid_pair_count": int(pair_valid_mask.sum().detach().cpu().item()),
            "valid_proto_count": int(valid_proto_mask.sum().detach().cpu().item()),
            "has_nan": has_nan,
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
            raise ValueError(
                f"visual_tokens must be [B,K,Mv,D], got {tuple(visual_tokens.shape)}"
            )
        if tuple(visual_mask.shape) != tuple(visual_tokens.shape[:3]):
            raise ValueError(
                "visual_mask must be [B,K,Mv] matching visual_tokens; "
                f"expected {tuple(visual_tokens.shape[:3])}, got {tuple(visual_mask.shape)}"
            )
        if prototype_slots.dim() != 5:
            raise ValueError(
                "prototype_slots must be [B,K,R,S,D], "
                f"got {tuple(prototype_slots.shape)}"
            )
        if tuple(prototype_slot_mask.shape) != tuple(prototype_slots.shape[:4]):
            raise ValueError(
                "prototype_slot_mask must be [B,K,R,S] matching prototype_slots; "
                f"expected {tuple(prototype_slots.shape[:4])}, "
                f"got {tuple(prototype_slot_mask.shape)}"
            )
        if tuple(prototype_slots.shape[:2]) != tuple(visual_tokens.shape[:2]):
            raise ValueError(
                "prototype_slots [B,K] must match visual_tokens [B,K]; "
                f"got {tuple(prototype_slots.shape[:2])} and {tuple(visual_tokens.shape[:2])}"
            )
        if visual_tokens.shape[-1] != self.d_model:
            raise ValueError(
                f"visual_tokens D must equal d_model={self.d_model}, "
                f"got {visual_tokens.shape[-1]}"
            )
        if prototype_slots.shape[-1] != self.d_model:
            raise ValueError(
                f"prototype_slots D must equal d_model={self.d_model}, "
                f"got {prototype_slots.shape[-1]}"
            )
        if prototype_slots.shape[3] != self.num_slots:
            raise ValueError(
                f"prototype_slots S must equal num_slots={self.num_slots}, "
                f"got {prototype_slots.shape[3]}"
            )
        if pair_valid_mask is not None and tuple(pair_valid_mask.shape) != tuple(
            visual_tokens.shape[:2]
        ):
            raise ValueError(
                f"pair_valid_mask must be [B,K]={tuple(visual_tokens.shape[:2])}, "
                f"got {tuple(pair_valid_mask.shape)}"
            )
