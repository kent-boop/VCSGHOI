from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class CandidateEvidenceOutput:
    evidence_tokens: Tensor
    evidence_mask: Tensor
    evidence_weights: Tensor
    evidence_context: Tensor
    anchor_delta: Tensor
    selected_context_indices: Tensor
    selected_context_mask: Tensor
    debug_info: Dict[str, Any]


class CandidateEvidenceTopologyInducer(nn.Module):
    """Build candidate-conditioned evidence tokens for HOI pair anchors.

    For each candidate pair q=(h,o), the module induces an evidence set
    [e_ho, x_h, x_o, p_q,k1, ...] without materializing B x K x N x N x D
    tensors. Context path candidates are only B x K x N x D before top-k.
    """

    def __init__(
        self,
        d_model: int = 256,
        max_context_paths: int = 4,
        dropout: float = 0.1,
        gate_init: float = 0.0,
        allow_node_only_context: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if max_context_paths < 0:
            raise ValueError(f"max_context_paths must be non-negative, got {max_context_paths}")

        self.d_model = int(d_model)
        self.max_context_paths = int(max_context_paths)
        self.allow_node_only_context = bool(allow_node_only_context)
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

    def forward(
        self,
        node_feats: Tensor,
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        node_mask: Tensor,
        pair_indices: Tensor,
        pair_valid_mask: Tensor,
        anchor_feats: Optional[Tensor] = None,
        human_feats: Optional[Tensor] = None,
        object_feats: Optional[Tensor] = None,
    ) -> CandidateEvidenceOutput:
        self._check_inputs(
            node_feats=node_feats,
            edge_attr_dense=edge_attr_dense,
            edge_mask_dense=edge_mask_dense,
            node_mask=node_mask,
            pair_indices=pair_indices,
            pair_valid_mask=pair_valid_mask,
            anchor_feats=anchor_feats,
            human_feats=human_feats,
            object_feats=object_feats,
        )

        batch_size, num_nodes, dim = node_feats.shape
        _, num_pairs, _ = pair_indices.shape
        context_count = self.max_context_paths
        evidence_count = 3 + context_count
        device = node_feats.device

        pair_valid_mask = pair_valid_mask.to(dtype=torch.bool)
        node_mask = node_mask.to(dtype=torch.bool)
        edge_mask_dense = edge_mask_dense.to(dtype=torch.bool)

        raw_h_idx = pair_indices[..., 0]
        raw_o_idx = pair_indices[..., 1]
        raw_pair_in_bounds = (
            (raw_h_idx >= 0)
            & (raw_h_idx < num_nodes)
            & (raw_o_idx >= 0)
            & (raw_o_idx < num_nodes)
        )
        h_idx = raw_h_idx.clamp(min=0, max=max(num_nodes - 1, 0))
        o_idx = raw_o_idx.clamp(min=0, max=max(num_nodes - 1, 0))
        batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1).expand(batch_size, num_pairs)

        endpoint_valid = node_mask[batch_idx, h_idx] & node_mask[batch_idx, o_idx]
        pair_valid = pair_valid_mask & raw_pair_in_bounds & endpoint_valid

        if anchor_feats is None:
            anchor = edge_attr_dense[batch_idx, h_idx, o_idx]
        else:
            anchor = anchor_feats
        if human_feats is None:
            human = node_feats[batch_idx, h_idx]
        else:
            human = human_feats
        if object_feats is None:
            obj = node_feats[batch_idx, o_idx]
        else:
            obj = object_feats

        anchor = torch.nan_to_num(anchor, nan=0.0, posinf=0.0, neginf=0.0)
        human = torch.nan_to_num(human, nan=0.0, posinf=0.0, neginf=0.0)
        obj = torch.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)

        path_tokens_all, path_scores_all, path_candidate_mask = self._build_path_candidates(
            node_feats=node_feats,
            edge_attr_dense=edge_attr_dense,
            edge_mask_dense=edge_mask_dense,
            node_mask=node_mask,
            h_idx=h_idx,
            o_idx=o_idx,
            anchor=anchor,
            pair_valid=pair_valid,
        )
        selected_context_indices, selected_context_mask, selected_context_scores = self._safe_topk_context(
            path_scores_all,
            path_candidate_mask,
            context_count,
        )

        if context_count > 0:
            gather_index = selected_context_indices.unsqueeze(-1).expand(batch_size, num_pairs, context_count, dim)
            context_tokens = torch.gather(path_tokens_all, dim=2, index=gather_index)
            context_tokens = context_tokens * selected_context_mask.unsqueeze(-1).to(dtype=context_tokens.dtype)
        else:
            context_tokens = node_feats.new_zeros((batch_size, num_pairs, 0, dim))

        base_tokens = torch.stack([anchor, human, obj], dim=2)
        evidence_tokens = torch.cat([base_tokens, context_tokens], dim=2)
        role_ids = torch.arange(evidence_count, device=device)
        evidence_tokens = evidence_tokens + self.role_embedding(role_ids).view(1, 1, evidence_count, dim)

        evidence_mask = torch.cat(
            [pair_valid.unsqueeze(-1).expand(batch_size, num_pairs, 3), selected_context_mask],
            dim=-1,
        )
        evidence_tokens = evidence_tokens * evidence_mask.unsqueeze(-1).to(dtype=evidence_tokens.dtype)

        evidence_weights = self._build_evidence_weights(
            evidence_mask=evidence_mask,
            selected_context_scores=selected_context_scores,
            pair_valid=pair_valid,
        )
        evidence_context = (evidence_tokens * evidence_weights.unsqueeze(-1)).sum(dim=2)
        evidence_context = evidence_context * pair_valid.unsqueeze(-1).to(dtype=evidence_context.dtype)

        anchor_delta_raw = self.anchor_delta_proj(torch.cat([anchor, evidence_context], dim=-1))
        anchor_delta = torch.tanh(self.anchor_gate).to(dtype=anchor_delta_raw.dtype) * anchor_delta_raw
        anchor_delta = anchor_delta * pair_valid.unsqueeze(-1).to(dtype=anchor_delta.dtype)
        anchor_delta = torch.nan_to_num(anchor_delta, nan=0.0, posinf=0.0, neginf=0.0)

        evidence_tokens = torch.nan_to_num(evidence_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        evidence_context = torch.nan_to_num(evidence_context, nan=0.0, posinf=0.0, neginf=0.0)
        evidence_weights = torch.nan_to_num(evidence_weights, nan=0.0, posinf=0.0, neginf=0.0)

        invalid_pair = ~pair_valid
        if invalid_pair.any():
            evidence_tokens = evidence_tokens.masked_fill(invalid_pair.unsqueeze(-1).unsqueeze(-1), 0.0)
            evidence_mask = evidence_mask & pair_valid.unsqueeze(-1)
            evidence_weights = evidence_weights.masked_fill(invalid_pair.unsqueeze(-1), 0.0)
            evidence_context = evidence_context.masked_fill(invalid_pair.unsqueeze(-1), 0.0)
            anchor_delta = anchor_delta.masked_fill(invalid_pair.unsqueeze(-1), 0.0)
            selected_context_indices = selected_context_indices.masked_fill(invalid_pair.unsqueeze(-1), 0)
            selected_context_mask = selected_context_mask & pair_valid.unsqueeze(-1)

        debug_info = self._build_debug_info(
            evidence_tokens=evidence_tokens,
            evidence_context=evidence_context,
            anchor_delta=anchor_delta,
            pair_valid=pair_valid,
            path_candidate_mask=path_candidate_mask,
        )

        return CandidateEvidenceOutput(
            evidence_tokens=evidence_tokens,
            evidence_mask=evidence_mask,
            evidence_weights=evidence_weights,
            evidence_context=evidence_context,
            anchor_delta=anchor_delta,
            selected_context_indices=selected_context_indices,
            selected_context_mask=selected_context_mask,
            debug_info=debug_info,
        )

    def _build_path_candidates(
        self,
        node_feats: Tensor,
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        node_mask: Tensor,
        h_idx: Tensor,
        o_idx: Tensor,
        anchor: Tensor,
        pair_valid: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_size, num_nodes, dim = node_feats.shape
        _, num_pairs = h_idx.shape
        batch_idx = torch.arange(batch_size, device=node_feats.device).view(batch_size, 1).expand(batch_size, num_pairs)

        e_hk = edge_attr_dense[batch_idx, h_idx]
        e_ko = edge_attr_dense[batch_idx.unsqueeze(-1), torch.arange(num_nodes, device=node_feats.device).view(1, 1, num_nodes), o_idx.unsqueeze(-1)]
        x_k = node_feats.unsqueeze(1).expand(batch_size, num_pairs, num_nodes, dim)
        anchor_expanded = anchor.unsqueeze(2).expand(batch_size, num_pairs, num_nodes, dim)

        path_input = torch.cat([anchor_expanded, e_hk, e_ko, x_k, x_k * anchor_expanded], dim=-1)
        path_input = torch.nan_to_num(path_input, nan=0.0, posinf=0.0, neginf=0.0)
        path_tokens = self.path_mlp(path_input)
        path_tokens = torch.nan_to_num(path_tokens, nan=0.0, posinf=0.0, neginf=0.0)

        score_input = torch.cat([path_tokens, anchor_expanded], dim=-1)
        path_scores = self.path_score_mlp(score_input).squeeze(-1)
        path_scores = torch.nan_to_num(path_scores, nan=0.0, posinf=0.0, neginf=0.0)

        edge_hk = edge_mask_dense[batch_idx, h_idx]
        edge_ko = edge_mask_dense[batch_idx.unsqueeze(-1), torch.arange(num_nodes, device=node_feats.device).view(1, 1, num_nodes), o_idx.unsqueeze(-1)]
        path_mask = node_mask.unsqueeze(1).expand(batch_size, num_pairs, num_nodes)
        path_mask = path_mask & pair_valid.unsqueeze(-1)
        path_mask = path_mask & (torch.arange(num_nodes, device=node_feats.device).view(1, 1, num_nodes) != h_idx.unsqueeze(-1))
        path_mask = path_mask & (torch.arange(num_nodes, device=node_feats.device).view(1, 1, num_nodes) != o_idx.unsqueeze(-1))
        if not self.allow_node_only_context:
            path_mask = path_mask & (edge_hk | edge_ko)

        path_tokens = path_tokens * path_mask.unsqueeze(-1).to(dtype=path_tokens.dtype)
        path_scores = path_scores.masked_fill(~path_mask, float("-inf"))
        return path_tokens, path_scores, path_mask

    def _safe_topk_context(self, scores: Tensor, mask: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
        out_shape = tuple(scores.shape[:-1]) + (k,)
        indices = torch.zeros(out_shape, dtype=torch.long, device=scores.device)
        selected_mask = torch.zeros(out_shape, dtype=torch.bool, device=scores.device)
        selected_scores = scores.new_zeros(out_shape)
        if k <= 0 or scores.shape[-1] == 0:
            return indices, selected_mask, selected_scores

        safe_scores = scores.masked_fill(~mask, float("-inf"))
        finite_mask = torch.isfinite(safe_scores)
        valid_count = finite_mask.sum(dim=-1)
        actual_k = min(k, scores.shape[-1])
        values, top_indices = safe_scores.topk(k=actual_k, dim=-1, largest=True)
        rank = torch.arange(actual_k, device=scores.device).view(*([1] * (scores.dim() - 1)), actual_k)
        top_valid = torch.isfinite(values) & (rank < valid_count.unsqueeze(-1))

        indices[..., :actual_k] = torch.where(top_valid, top_indices, torch.zeros_like(top_indices))
        selected_mask[..., :actual_k] = top_valid
        selected_scores[..., :actual_k] = torch.where(top_valid, values, torch.zeros_like(values))
        return indices, selected_mask, selected_scores

    def _build_evidence_weights(
        self,
        evidence_mask: Tensor,
        selected_context_scores: Tensor,
        pair_valid: Tensor,
    ) -> Tensor:
        batch_size, num_pairs, evidence_count = evidence_mask.shape
        base_scores = selected_context_scores.new_zeros((batch_size, num_pairs, 3))
        evidence_scores = torch.cat([base_scores, selected_context_scores], dim=-1)
        evidence_scores = evidence_scores.masked_fill(~evidence_mask, float("-inf"))
        weights = torch.softmax(evidence_scores, dim=-1)
        weights = torch.where(evidence_mask, weights, torch.zeros_like(weights))
        weights = weights * pair_valid.unsqueeze(-1).to(dtype=weights.dtype)

        denom = weights.sum(dim=-1, keepdim=True)
        uniform = evidence_mask.to(dtype=weights.dtype) / evidence_mask.sum(dim=-1, keepdim=True).clamp(min=1).to(dtype=weights.dtype)
        weights = torch.where(denom > 0, weights / denom.clamp(min=1e-6), uniform)
        weights = weights * pair_valid.unsqueeze(-1).to(dtype=weights.dtype)
        return weights.reshape(batch_size, num_pairs, evidence_count)

    def _build_debug_info(
        self,
        evidence_tokens: Tensor,
        evidence_context: Tensor,
        anchor_delta: Tensor,
        pair_valid: Tensor,
        path_candidate_mask: Tensor,
    ) -> Dict[str, Any]:
        valid_pair_count = int(pair_valid.sum().detach().cpu().item()) if pair_valid.numel() > 0 else 0
        if valid_pair_count > 0:
            path_count_mean = float(path_candidate_mask.sum(dim=-1)[pair_valid].float().mean().detach().cpu().item())
            anchor_delta_norm_mean = float(anchor_delta[pair_valid].detach().float().norm(dim=-1).mean().cpu().item())
        else:
            path_count_mean = 0.0
            anchor_delta_norm_mean = 0.0

        has_nan = bool(
            torch.isnan(evidence_tokens).any().detach().cpu().item()
            or torch.isnan(evidence_context).any().detach().cpu().item()
            or torch.isnan(anchor_delta).any().detach().cpu().item()
        )
        return {
            "candidate_evidence_enabled": True,
            "candidate_evidence_tokens_shape": tuple(evidence_tokens.shape),
            "candidate_evidence_valid_pair_count": valid_pair_count,
            "candidate_evidence_context_path_count_mean": path_count_mean,
            "candidate_evidence_anchor_delta_norm_mean": anchor_delta_norm_mean,
            "candidate_evidence_gate": float(torch.tanh(self.anchor_gate).detach().cpu().item()),
            "candidate_evidence_has_nan": has_nan,
        }

    def _check_inputs(
        self,
        node_feats: Tensor,
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        node_mask: Tensor,
        pair_indices: Tensor,
        pair_valid_mask: Tensor,
        anchor_feats: Optional[Tensor],
        human_feats: Optional[Tensor],
        object_feats: Optional[Tensor],
    ) -> None:
        if node_feats.dim() != 3:
            raise ValueError(f"node_feats must be [B,N,D], got {tuple(node_feats.shape)}")
        if edge_attr_dense.dim() != 4:
            raise ValueError(f"edge_attr_dense must be [B,N,N,D], got {tuple(edge_attr_dense.shape)}")
        if edge_mask_dense.dim() != 3:
            raise ValueError(f"edge_mask_dense must be [B,N,N], got {tuple(edge_mask_dense.shape)}")
        if node_mask.dim() != 2:
            raise ValueError(f"node_mask must be [B,N], got {tuple(node_mask.shape)}")
        if pair_indices.dim() != 3 or pair_indices.shape[-1] != 2:
            raise ValueError(f"pair_indices must be [B,K,2], got {tuple(pair_indices.shape)}")
        if pair_valid_mask.dim() != 2:
            raise ValueError(f"pair_valid_mask must be [B,K], got {tuple(pair_valid_mask.shape)}")

        batch_size, num_nodes, dim = node_feats.shape
        _, num_pairs, _ = pair_indices.shape
        if dim != self.d_model:
            raise ValueError(f"node_feats D must equal d_model={self.d_model}, got {dim}")
        expected_edge_attr = (batch_size, num_nodes, num_nodes, dim)
        if tuple(edge_attr_dense.shape) != expected_edge_attr:
            raise ValueError(f"edge_attr_dense must be {expected_edge_attr}, got {tuple(edge_attr_dense.shape)}")
        expected_edge_mask = (batch_size, num_nodes, num_nodes)
        if tuple(edge_mask_dense.shape) != expected_edge_mask:
            raise ValueError(f"edge_mask_dense must be {expected_edge_mask}, got {tuple(edge_mask_dense.shape)}")
        if tuple(node_mask.shape) != (batch_size, num_nodes):
            raise ValueError(f"node_mask must be {(batch_size, num_nodes)}, got {tuple(node_mask.shape)}")
        if tuple(pair_valid_mask.shape) != (batch_size, num_pairs):
            raise ValueError(f"pair_valid_mask must be {(batch_size, num_pairs)}, got {tuple(pair_valid_mask.shape)}")

        optional_shapes = {
            "anchor_feats": anchor_feats,
            "human_feats": human_feats,
            "object_feats": object_feats,
        }
        expected_optional = (batch_size, num_pairs, dim)
        for name, value in optional_shapes.items():
            if value is not None and tuple(value.shape) != expected_optional:
                raise ValueError(f"{name} must be {expected_optional}, got {tuple(value.shape)}")
