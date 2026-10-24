"""Candidate evidence construction for the isolated VET internal ablation.

This module deliberately lives beside (rather than replacing)
``candidate_evidence_topology.py``.  Every ablation keeps the same tensor
shape and the same parameter set; modes only change state propagation,
evidence masks, path selection, and VET-side path weighting.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class VETAblationSpec:
    label: str
    use_svrm: bool
    use_endpoints: bool
    context_selection: str
    context_weighting: str
    candidate_writeback: bool


VET_ABLATION_SPECS: Mapping[str, VETAblationSpec] = {
    "anchor_raw": VETAblationSpec("V0", False, False, "none", "none", False),
    "anchor_svrm": VETAblationSpec("V1", True, False, "none", "none", False),
    "endpoints": VETAblationSpec("V2", True, True, "none", "none", False),
    "context_uniform": VETAblationSpec("V3", True, True, "fixed", "uniform", True),
    "path_selection": VETAblationSpec("V4", True, True, "learned", "uniform", True),
    "path_reweight": VETAblationSpec("V5", True, True, "learned", "learned", True),
}


def get_vet_ablation_spec(mode: str) -> VETAblationSpec:
    try:
        return VET_ABLATION_SPECS[mode]
    except KeyError as exc:
        choices = ", ".join(VET_ABLATION_SPECS)
        raise ValueError(f"Unknown graph_vet_ablation_mode={mode!r}; expected one of: {choices}") from exc


@dataclass
class CandidateEvidenceAblationOutput:
    evidence_tokens: Tensor
    evidence_mask: Tensor
    evidence_weights: Tensor
    evidence_context: Tensor
    anchor_delta: Tensor
    selected_context_indices: Tensor
    selected_context_mask: Tensor
    context_path_weights: Tensor
    selected_context_scores: Tensor
    soft_selection_weights: Tensor
    effective_token_count: Tensor
    debug_info: Dict[str, Any]


class CandidateEvidenceTopologyVETAblation(nn.Module):
    """Build fixed-length evidence tokens for V0--V5.

    Learned selection uses a deterministic hard top-k in the forward pass.
    A straight-through gate has forward value one and a softmax surrogate in
    backward, allowing V4's selector to train without turning V4 into a path
    reweighting experiment.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_context_paths: int = 4,
        dropout: float = 0.1,
        gate_init: float = 0.0,
        allow_node_only_context: bool = False,
        ablation_mode: str = "path_reweight",
        selection_temperature: float = 1.0,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_context_paths < 0:
            raise ValueError(f"max_context_paths must be non-negative, got {max_context_paths}")
        if selection_temperature <= 0:
            raise ValueError("selection_temperature must be positive")
        get_vet_ablation_spec(ablation_mode)

        self.d_model = int(d_model)
        self.max_context_paths = int(max_context_paths)
        self.allow_node_only_context = bool(allow_node_only_context)
        self.ablation_mode = str(ablation_mode)
        self.selection_temperature = float(selection_temperature)
        self.debug = bool(debug)

        self.path_mlp = nn.Sequential(
            nn.Linear(self.d_model * 5, self.d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
        )
        self.path_score_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, 1),
        )
        self.anchor_delta_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.anchor_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.role_embedding = nn.Embedding(3 + self.max_context_paths, self.d_model)

    @property
    def spec(self) -> VETAblationSpec:
        return get_vet_ablation_spec(self.ablation_mode)

    def forward(
        self,
        node_feats: Tensor,
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        node_mask: Tensor,
        pair_indices: Tensor,
        pair_valid_mask: Tensor,
        edge_score_dense: Optional[Tensor] = None,
        anchor_feats: Optional[Tensor] = None,
        human_feats: Optional[Tensor] = None,
        object_feats: Optional[Tensor] = None,
    ) -> CandidateEvidenceAblationOutput:
        self._check_inputs(
            node_feats, edge_attr_dense, edge_mask_dense, node_mask,
            pair_indices, pair_valid_mask, edge_score_dense,
            anchor_feats, human_feats, object_feats,
        )
        spec = self.spec
        batch_size, num_nodes, dim = node_feats.shape
        num_pairs = pair_indices.shape[1]
        context_count = self.max_context_paths
        evidence_count = 3 + context_count
        device = node_feats.device

        node_mask = node_mask.to(dtype=torch.bool)
        edge_mask_dense = edge_mask_dense.to(dtype=torch.bool)
        pair_valid_mask = pair_valid_mask.to(dtype=torch.bool)
        raw_h_idx, raw_o_idx = pair_indices[..., 0], pair_indices[..., 1]
        in_bounds = (
            (raw_h_idx >= 0) & (raw_h_idx < num_nodes)
            & (raw_o_idx >= 0) & (raw_o_idx < num_nodes)
        )
        h_idx = raw_h_idx.clamp(0, max(num_nodes - 1, 0))
        o_idx = raw_o_idx.clamp(0, max(num_nodes - 1, 0))
        batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, num_pairs)
        pair_valid = pair_valid_mask & in_bounds
        pair_valid = pair_valid & node_mask[batch_idx, h_idx] & node_mask[batch_idx, o_idx]

        anchor = edge_attr_dense[batch_idx, h_idx, o_idx] if anchor_feats is None else anchor_feats
        human = node_feats[batch_idx, h_idx] if human_feats is None else human_feats
        obj = node_feats[batch_idx, o_idx] if object_feats is None else object_feats
        anchor = self._sanitize(anchor)
        human = self._sanitize(human)
        obj = self._sanitize(obj)

        if edge_score_dense is None:
            edge_score_dense = edge_attr_dense.new_zeros(edge_mask_dense.shape)
        (
            path_tokens_all,
            learned_scores_all,
            fixed_scores_all,
            path_candidate_mask,
            two_hop_mask,
        ) = self._build_path_candidates(
            node_feats, edge_attr_dense, edge_mask_dense, edge_score_dense,
            node_mask, h_idx, o_idx, anchor, pair_valid,
        )

        if spec.context_selection == "fixed":
            selection_scores = fixed_scores_all
        elif spec.context_selection == "learned":
            selection_scores = learned_scores_all
        else:
            selection_scores = learned_scores_all.new_full(learned_scores_all.shape, float("-inf"))
            path_candidate_mask = torch.zeros_like(path_candidate_mask)

        selected_indices, selected_mask, selected_scores = self._safe_topk_context(
            selection_scores, path_candidate_mask, context_count,
        )
        soft_selection_weights = self._masked_softmax(
            learned_scores_all / self.selection_temperature,
            path_candidate_mask,
            dim=-1,
        )

        if context_count:
            gather_index = selected_indices.unsqueeze(-1).expand(batch_size, num_pairs, context_count, dim)
            context_tokens = torch.gather(path_tokens_all, 2, gather_index)
            if spec.context_selection == "learned":
                selected_soft = torch.gather(soft_selection_weights, 2, selected_indices)
                selected_soft = selected_soft * selected_mask.to(dtype=selected_soft.dtype)
                # Exactly one in forward; differentiable through the soft selector in backward.
                st_gate = torch.ones_like(selected_soft) + selected_soft - selected_soft.detach()
                context_tokens = context_tokens * st_gate.unsqueeze(-1)
            context_tokens = context_tokens * selected_mask.unsqueeze(-1).to(dtype=context_tokens.dtype)
        else:
            context_tokens = node_feats.new_zeros((batch_size, num_pairs, 0, dim))

        base_tokens = torch.stack([anchor, human, obj], dim=2)
        evidence_tokens = torch.cat([base_tokens, context_tokens], dim=2)
        role_ids = torch.arange(evidence_count, device=device)
        evidence_tokens = evidence_tokens + self.role_embedding(role_ids).view(1, 1, evidence_count, dim)

        anchor_mask = pair_valid.unsqueeze(-1)
        endpoint_mask = pair_valid.unsqueeze(-1).expand(batch_size, num_pairs, 2)
        if spec.use_endpoints:
            base_mask = torch.cat([anchor_mask, endpoint_mask], dim=-1)
        else:
            base_mask = torch.cat([anchor_mask, torch.zeros_like(endpoint_mask)], dim=-1)
        evidence_mask = torch.cat([base_mask, selected_mask], dim=-1)
        evidence_tokens = evidence_tokens * evidence_mask.unsqueeze(-1).to(dtype=evidence_tokens.dtype)

        evidence_weights, context_path_weights = self._build_evidence_weights(
            evidence_mask, selected_mask, selected_scores, spec.context_weighting,
        )
        evidence_context = (evidence_tokens * evidence_weights.unsqueeze(-1)).sum(dim=2)
        evidence_context = evidence_context * pair_valid.unsqueeze(-1).to(dtype=evidence_context.dtype)

        anchor_delta_raw = self.anchor_delta_proj(torch.cat([anchor, evidence_context], dim=-1))
        anchor_delta = torch.tanh(self.anchor_gate).to(dtype=anchor_delta_raw.dtype) * anchor_delta_raw
        has_context = selected_mask.any(dim=-1) & pair_valid
        if not spec.candidate_writeback:
            has_context = torch.zeros_like(has_context)
        anchor_delta = anchor_delta * has_context.unsqueeze(-1).to(dtype=anchor_delta.dtype)

        evidence_tokens = self._sanitize(evidence_tokens)
        evidence_weights = self._sanitize(evidence_weights)
        context_path_weights = self._sanitize(context_path_weights)
        evidence_context = self._sanitize(evidence_context)
        anchor_delta = self._sanitize(anchor_delta)
        effective_token_count = evidence_mask.sum(dim=-1)

        invalid = ~pair_valid
        if invalid.any():
            evidence_tokens = evidence_tokens.masked_fill(invalid[..., None, None], 0.0)
            evidence_mask = evidence_mask & pair_valid.unsqueeze(-1)
            evidence_weights = evidence_weights.masked_fill(invalid.unsqueeze(-1), 0.0)
            context_path_weights = context_path_weights.masked_fill(invalid.unsqueeze(-1), 0.0)
            evidence_context = evidence_context.masked_fill(invalid.unsqueeze(-1), 0.0)
            anchor_delta = anchor_delta.masked_fill(invalid.unsqueeze(-1), 0.0)
            selected_indices = selected_indices.masked_fill(invalid.unsqueeze(-1), 0)
            selected_mask = selected_mask & pair_valid.unsqueeze(-1)

        debug_info = self._build_debug_info(
            spec, pair_valid, evidence_mask, evidence_weights, context_path_weights,
            selected_indices, selected_mask, path_candidate_mask, two_hop_mask,
            anchor_delta, evidence_tokens,
        )
        return CandidateEvidenceAblationOutput(
            evidence_tokens=evidence_tokens,
            evidence_mask=evidence_mask,
            evidence_weights=evidence_weights,
            evidence_context=evidence_context,
            anchor_delta=anchor_delta,
            selected_context_indices=selected_indices,
            selected_context_mask=selected_mask,
            context_path_weights=context_path_weights,
            selected_context_scores=selected_scores,
            soft_selection_weights=soft_selection_weights,
            effective_token_count=effective_token_count,
            debug_info=debug_info,
        )

    def _build_path_candidates(
        self,
        node_feats: Tensor,
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        edge_score_dense: Tensor,
        node_mask: Tensor,
        h_idx: Tensor,
        o_idx: Tensor,
        anchor: Tensor,
        pair_valid: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_nodes, dim = node_feats.shape
        num_pairs = h_idx.shape[1]
        batch_idx = torch.arange(batch_size, device=node_feats.device).view(batch_size, 1).expand(batch_size, num_pairs)
        node_ids = torch.arange(num_nodes, device=node_feats.device).view(1, 1, num_nodes)

        e_hk = edge_attr_dense[batch_idx, h_idx]
        e_ko = edge_attr_dense[batch_idx.unsqueeze(-1), node_ids, o_idx.unsqueeze(-1)]
        x_k = node_feats.unsqueeze(1).expand(batch_size, num_pairs, num_nodes, dim)
        anchor_expanded = anchor.unsqueeze(2).expand_as(x_k)
        path_input = torch.cat([anchor_expanded, e_hk, e_ko, x_k, x_k * anchor_expanded], dim=-1)
        path_tokens = self._sanitize(self.path_mlp(self._sanitize(path_input)))
        learned_scores = self._sanitize(
            self.path_score_mlp(torch.cat([path_tokens, anchor_expanded], dim=-1)).squeeze(-1)
        )

        edge_hk = edge_mask_dense[batch_idx, h_idx]
        edge_ko = edge_mask_dense[batch_idx.unsqueeze(-1), node_ids, o_idx.unsqueeze(-1)]
        two_hop = edge_hk & edge_ko
        one_sided = edge_hk | edge_ko
        candidate_mask = node_mask.unsqueeze(1).expand(batch_size, num_pairs, num_nodes)
        candidate_mask = candidate_mask & pair_valid.unsqueeze(-1)
        candidate_mask = candidate_mask & (node_ids != h_idx.unsqueeze(-1)) & (node_ids != o_idx.unsqueeze(-1))
        if not self.allow_node_only_context:
            candidate_mask = candidate_mask & one_sided

        score_hk = self._sanitize(edge_score_dense[batch_idx, h_idx])
        score_ko = self._sanitize(edge_score_dense[batch_idx.unsqueeze(-1), node_ids, o_idx.unsqueeze(-1)])
        connectivity_rank = two_hop.to(dtype=path_tokens.dtype) * 2.0 + one_sided.to(dtype=path_tokens.dtype)
        # Connectivity is lexicographically dominant; lower node id breaks exact ties.
        fixed_scores = connectivity_rank * 1_000_000.0
        fixed_scores = fixed_scores + (score_hk + score_ko).clamp(-10_000.0, 10_000.0) * 10.0
        fixed_scores = fixed_scores - node_ids.to(dtype=path_tokens.dtype) * 1e-4

        path_tokens = path_tokens * candidate_mask.unsqueeze(-1).to(dtype=path_tokens.dtype)
        learned_scores = learned_scores.masked_fill(~candidate_mask, float("-inf"))
        fixed_scores = fixed_scores.masked_fill(~candidate_mask, float("-inf"))
        return path_tokens, learned_scores, fixed_scores, candidate_mask, two_hop & candidate_mask

    def _build_evidence_weights(
        self,
        evidence_mask: Tensor,
        selected_mask: Tensor,
        selected_scores: Tensor,
        weighting: str,
    ) -> Tuple[Tensor, Tensor]:
        dtype = selected_scores.dtype
        active = evidence_mask.to(dtype=dtype)
        uniform = active / active.sum(dim=-1, keepdim=True).clamp(min=1.0)
        num_context = selected_mask.sum(dim=-1, keepdim=True).to(dtype=dtype)
        uniform_context = selected_mask.to(dtype=dtype) / num_context.clamp(min=1.0)
        if weighting != "learned":
            return uniform, uniform_context

        alpha = self._masked_softmax(
            selected_scores / self.selection_temperature,
            selected_mask,
            dim=-1,
        )
        denom = 3.0 + num_context
        base_weight = (1.0 / denom).expand(*evidence_mask.shape[:-1], 3)
        base_weight = base_weight * evidence_mask[..., :3].to(dtype=dtype)
        context_mass = num_context / denom
        context_weights = alpha * context_mass
        weights = torch.cat([base_weight, context_weights], dim=-1)
        weights = weights * evidence_mask.to(dtype=dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return weights, alpha

    @staticmethod
    def _safe_topk_context(scores: Tensor, mask: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
        shape = tuple(scores.shape[:-1]) + (k,)
        indices = torch.zeros(shape, dtype=torch.long, device=scores.device)
        selected_mask = torch.zeros(shape, dtype=torch.bool, device=scores.device)
        selected_scores = scores.new_zeros(shape)
        if k <= 0 or scores.shape[-1] == 0:
            return indices, selected_mask, selected_scores
        safe_scores = scores.masked_fill(~mask, float("-inf"))
        actual_k = min(k, scores.shape[-1])
        values, top_indices = safe_scores.topk(actual_k, dim=-1, largest=True, sorted=True)
        valid = torch.isfinite(values)
        indices[..., :actual_k] = torch.where(valid, top_indices, torch.zeros_like(top_indices))
        selected_mask[..., :actual_k] = valid
        selected_scores[..., :actual_k] = torch.where(valid, values, torch.zeros_like(values))
        return indices, selected_mask, selected_scores

    @staticmethod
    def _masked_softmax(logits: Tensor, mask: Tensor, dim: int) -> Tensor:
        mask = mask.to(dtype=torch.bool)
        safe_logits = torch.where(mask, logits, torch.zeros_like(logits))
        safe_logits = safe_logits - safe_logits.amax(dim=dim, keepdim=True)
        exp = torch.exp(safe_logits) * mask.to(dtype=logits.dtype)
        return exp / exp.sum(dim=dim, keepdim=True).clamp(min=1e-12)

    @staticmethod
    def _sanitize(tensor: Tensor) -> Tensor:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_debug_info(
        self,
        spec: VETAblationSpec,
        pair_valid: Tensor,
        evidence_mask: Tensor,
        evidence_weights: Tensor,
        context_path_weights: Tensor,
        selected_indices: Tensor,
        selected_mask: Tensor,
        candidate_mask: Tensor,
        two_hop_mask: Tensor,
        anchor_delta: Tensor,
        evidence_tokens: Tensor,
    ) -> Dict[str, Any]:
        valid_count = int(pair_valid.sum().detach().cpu().item())
        if valid_count:
            active_mean = float(evidence_mask.sum(-1)[pair_valid].float().mean().detach().cpu().item())
            selected_mean = float(selected_mask.sum(-1)[pair_valid].float().mean().detach().cpu().item())
            delta_norm = float(anchor_delta[pair_valid].float().norm(dim=-1).mean().detach().cpu().item())
        else:
            active_mean = selected_mean = delta_norm = 0.0
        info: Dict[str, Any] = {
            "vet_ablation_mode": self.ablation_mode,
            "vet_ablation_label": spec.label,
            "svrm_enabled": spec.use_svrm,
            "endpoint_tokens_enabled": spec.use_endpoints,
            "context_selection": spec.context_selection,
            "context_weighting": spec.context_weighting,
            "candidate_writeback": spec.candidate_writeback,
            "evidence_tokens_shape": tuple(evidence_tokens.shape),
            "evidence_mask_shape": tuple(evidence_mask.shape),
            "valid_pair_count": valid_count,
            "effective_token_count_mean": active_mean,
            "selected_context_count_mean": selected_mean,
            "path_candidate_count": int(candidate_mask.sum().detach().cpu().item()),
            "two_hop_candidate_count": int(two_hop_mask.sum().detach().cpu().item()),
            "candidate_anchor_delta_norm_mean": delta_norm,
            "anchor_gate": float(torch.tanh(self.anchor_gate).detach().cpu().item()),
            "selection_temperature": self.selection_temperature,
            "straight_through_selection": spec.context_selection == "learned",
            "has_nan_or_inf": not bool(torch.isfinite(evidence_tokens).all().detach().cpu().item()),
            "evidence_weight_sum_min": float(evidence_weights.sum(-1).min().detach().cpu().item()),
            "evidence_weight_sum_max": float(evidence_weights.sum(-1).max().detach().cpu().item()),
            "context_weight_sum_max": float(context_path_weights.sum(-1).max().detach().cpu().item()),
        }
        if self.debug:
            info.update({
                "effective_token_count": evidence_mask.sum(-1).detach(),
                "selected_context_indices": selected_indices.detach(),
                "selected_context_mask": selected_mask.detach(),
                "evidence_weights": evidence_weights.detach(),
                "context_path_weights": context_path_weights.detach(),
            })
        return info

    def _check_inputs(
        self,
        node_feats: Tensor,
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        node_mask: Tensor,
        pair_indices: Tensor,
        pair_valid_mask: Tensor,
        edge_score_dense: Optional[Tensor],
        anchor_feats: Optional[Tensor],
        human_feats: Optional[Tensor],
        object_feats: Optional[Tensor],
    ) -> None:
        if node_feats.dim() != 3:
            raise ValueError(f"node_feats must be [B,N,D], got {tuple(node_feats.shape)}")
        batch_size, num_nodes, dim = node_feats.shape
        num_pairs = pair_indices.shape[1] if pair_indices.dim() >= 2 else -1
        expected_edge = (batch_size, num_nodes, num_nodes, dim)
        if tuple(edge_attr_dense.shape) != expected_edge:
            raise ValueError(f"edge_attr_dense must be {expected_edge}, got {tuple(edge_attr_dense.shape)}")
        if tuple(edge_mask_dense.shape) != expected_edge[:3]:
            raise ValueError("edge_mask_dense shape does not match node_feats")
        if tuple(node_mask.shape) != (batch_size, num_nodes):
            raise ValueError("node_mask shape does not match node_feats")
        if pair_indices.dim() != 3 or pair_indices.shape[-1] != 2:
            raise ValueError("pair_indices must be [B,Q,2]")
        if tuple(pair_valid_mask.shape) != (batch_size, num_pairs):
            raise ValueError("pair_valid_mask must be [B,Q]")
        if dim != self.d_model:
            raise ValueError(f"feature dim {dim} does not match d_model={self.d_model}")
        if edge_score_dense is not None and tuple(edge_score_dense.shape) != expected_edge[:3]:
            raise ValueError("edge_score_dense must be [B,N,N]")
        expected_pair = (batch_size, num_pairs, dim)
        for name, value in {
            "anchor_feats": anchor_feats,
            "human_feats": human_feats,
            "object_feats": object_feats,
        }.items():
            if value is not None and tuple(value.shape) != expected_pair:
                raise ValueError(f"{name} must be {expected_pair}, got {tuple(value.shape)}")

