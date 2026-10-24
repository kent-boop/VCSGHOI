"""Layerwise VET/SETM scheduler for the isolated V0--V5 experiment."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from src.models.visual_evidence_topology_vet import VETOutput
from src.models.visual_evidence_topology_vet_ablation import (
    VisualEvidenceTopologyVETAblation,
    vet_ablation_config_from_args,
)


@dataclass
class VETAblationLayerwiseOutput(VETOutput):
    layerwise_setm_debug: Any
    final_prototype_slots: Tensor
    final_prototype_slot_mask: Tensor


class VisualEvidenceTopologyVETAblationLayerwiseSETM(VisualEvidenceTopologyVETAblation):
    """Run an identical SETM schedule for all modes and vary VET evidence only."""

    def forward(
        self,
        node_feats: Tensor,
        boxes: Tensor,
        labels: Tensor,
        node_types: Tensor,
        node_mask: Tensor,
        pair_indices: Tensor,
        pair_scores: Tensor,
        pair_feats: Optional[Tensor] = None,
        union_feats: Optional[Tensor] = None,
        global_feat: Optional[Tensor] = None,
        node_blip_feats: Optional[Tensor] = None,
        prototype_slots: Optional[Tensor] = None,
        prototype_slot_mask: Optional[Tensor] = None,
        setm_matcher: Optional[nn.Module] = None,
        setm_stage: int = 4,
        write_visual: bool = True,
        write_text: bool = True,
        write_reverse_edge: bool = False,
        detach_text_update: bool = False,
        debug: bool = False,
    ) -> VETAblationLayerwiseOutput:
        self._check_inputs(
            node_feats=node_feats,
            node_blip_feats=node_blip_feats,
            boxes=boxes,
            labels=labels,
            node_types=node_types,
            node_mask=node_mask,
            pair_indices=pair_indices,
            pair_scores=pair_scores,
            pair_feats=pair_feats,
            union_feats=union_feats,
            global_feat=global_feat,
        )
        self._check_setm_inputs(
            node_feats, pair_indices, prototype_slots, prototype_slot_mask,
            setm_matcher, setm_stage,
        )
        spec = self.ablation_spec
        fused_node_feats = self._fuse_node_inputs(
            node_feats, node_blip_feats, boxes, labels, node_types
        )
        fused_node_feats = fused_node_feats * node_mask.unsqueeze(-1).to(dtype=fused_node_feats.dtype)
        if pair_feats is None:
            base_pair_feats = self._build_pair_feats_from_endpoints(
                fused_node_feats, pair_indices, pair_scores
            )
            pair_source = "endpoint_node_feats"
        else:
            base_pair_feats = pair_feats
            pair_source = "provided_pair_feats"

        initial_edge_attr, edge_mask_dense, edge_score_dense, counts = self._build_dense_edges(
            node_feats=fused_node_feats,
            boxes=boxes,
            labels=labels,
            node_types=node_types,
            node_mask=node_mask,
            pair_indices=pair_indices,
            pair_scores=pair_scores,
            pair_feats=base_pair_feats,
            union_feats=union_feats,
        )
        current_prototype_slots = prototype_slots
        current_prototype_slot_mask = prototype_slot_mask
        updated_node_feats = fused_node_feats
        updated_edge_attr = initial_edge_attr
        layer_debug = []
        layerwise_setm_debug = []

        for layer_id, layer in enumerate(self.layers):
            if spec.use_svrm:
                updated_node_feats, updated_edge_attr, layer_info = layer(
                    updated_node_feats, updated_edge_attr, edge_mask_dense, node_mask
                )
                layer_info["svrm_applied"] = True
            else:
                # Identity scheduling preserves depth and SETM call count while
                # excluding shared visual relation message passing in V0.
                layer_info = {
                    "layer_index": layer_id,
                    "svrm_applied": False,
                    "identity_visual_update": True,
                    "node_in_shape": tuple(updated_node_feats.shape),
                    "node_out_shape": tuple(updated_node_feats.shape),
                    "edge_in_shape": tuple(updated_edge_attr.shape),
                    "edge_out_shape": tuple(updated_edge_attr.shape),
                    "has_nan": False,
                }
            layer_info["layer_index"] = layer_id
            layer_debug.append(layer_info)

            refined_edge_feats, valid_pair_mask = self._gather_pair_edge_feats(
                updated_edge_attr, edge_mask_dense, pair_indices, node_mask
            )
            refined_edge_feats = refined_edge_feats * valid_pair_mask.unsqueeze(-1).to(
                dtype=refined_edge_feats.dtype
            )
            node_h = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 0])
            node_o = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 1])
            candidate_out = self.candidate_evidence(
                node_feats=updated_node_feats,
                edge_attr_dense=updated_edge_attr,
                edge_mask_dense=edge_mask_dense,
                edge_score_dense=edge_score_dense,
                node_mask=node_mask,
                pair_indices=pair_indices,
                pair_valid_mask=valid_pair_mask,
                anchor_feats=refined_edge_feats,
                human_feats=node_h,
                object_feats=node_o,
            )
            setm_out = setm_matcher(
                visual_tokens=candidate_out.evidence_tokens,
                visual_mask=candidate_out.evidence_mask,
                prototype_slots=current_prototype_slots,
                prototype_slot_mask=current_prototype_slot_mask,
                pair_valid_mask=valid_pair_mask,
                stage=4,
            )
            candidate_norm = self._valid_norm(candidate_out.anchor_delta, valid_pair_mask)
            setm_norm = self._valid_norm(setm_out.anchor_delta, valid_pair_mask)
            total_delta = candidate_out.anchor_delta + setm_out.anchor_delta
            total_norm = self._valid_norm(total_delta, valid_pair_mask)
            if write_visual:
                updated_edge_attr = self._scatter_pair_delta_to_dense_edges(
                    updated_edge_attr, edge_mask_dense, pair_indices,
                    valid_pair_mask, total_delta, write_reverse_edge,
                )
            if write_text:
                current_prototype_slots = setm_out.updated_prototype_slots
                if detach_text_update:
                    current_prototype_slots = current_prototype_slots.detach()

            layerwise_setm_debug.append({
                "layer_id": layer_id,
                "svrm_applied": spec.use_svrm,
                "candidate_evidence_debug": candidate_out.debug_info,
                "setm_debug": setm_out.debug_info,
                "candidate_anchor_delta_norm_mean": candidate_norm,
                "setm_anchor_delta_norm_mean": setm_norm,
                "total_writeback_delta_norm_mean": total_norm,
                "write_visual": bool(write_visual),
                "write_text": bool(write_text),
                "write_reverse_edge": bool(write_reverse_edge),
                "detach_text_update": bool(detach_text_update),
            })

        refined_edge_feats, valid_pair_mask = self._gather_pair_edge_feats(
            updated_edge_attr, edge_mask_dense, pair_indices, node_mask
        )
        refined_edge_feats = refined_edge_feats * valid_pair_mask.unsqueeze(-1).to(
            dtype=refined_edge_feats.dtype
        )
        node_h = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 0])
        node_o = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 1])
        final_candidate_out = self.candidate_evidence(
            node_feats=updated_node_feats,
            edge_attr_dense=updated_edge_attr,
            edge_mask_dense=edge_mask_dense,
            edge_score_dense=edge_score_dense,
            node_mask=node_mask,
            pair_indices=pair_indices,
            pair_valid_mask=valid_pair_mask,
            anchor_feats=refined_edge_feats,
            human_feats=node_h,
            object_feats=node_o,
        )

        union_input = union_feats if union_feats is not None else torch.zeros_like(base_pair_feats)
        pair_out = self.pair_out_mlp(torch.cat(
            [refined_edge_feats, node_h, node_o, base_pair_feats, union_input], dim=-1
        ))
        fallback = base_pair_feats if pair_feats is not None else torch.zeros_like(base_pair_feats)
        pair_out = torch.where(valid_pair_mask.unsqueeze(-1), pair_out, fallback)

        debug_info = self._build_debug_info(
            pair_source=pair_source,
            input_node_feats=node_feats,
            node_blip_feats=node_blip_feats,
            fused_node_feats=fused_node_feats,
            updated_node_feats=updated_node_feats,
            pair_indices=pair_indices,
            pair_out=pair_out,
            refined_edge_feats=refined_edge_feats,
            initial_edge_attr=initial_edge_attr,
            edge_attr_dense=updated_edge_attr,
            edge_mask_dense=edge_mask_dense,
            edge_score_dense=edge_score_dense,
            valid_pair_mask=valid_pair_mask,
            counts=counts,
            union_feats=union_feats,
            global_feat=global_feat,
            layer_debug=layer_debug,
            candidate_debug_info=final_candidate_out.debug_info,
            evidence_tokens=final_candidate_out.evidence_tokens,
            evidence_context=final_candidate_out.evidence_context,
            anchor_delta=final_candidate_out.anchor_delta,
            selected_context_indices=final_candidate_out.selected_context_indices,
        )
        debug_info.update({
            "vet_ablation_mode": self.vet_ablation_mode,
            "vet_ablation_label": spec.label,
            "anchor_source": "pre_svrm_dense_ho_edge" if not spec.use_svrm else "svrm_refined_ho_edge",
            "svrm_enabled": spec.use_svrm,
            "endpoint_tokens_enabled": spec.use_endpoints,
            "context_selection": spec.context_selection,
            "context_weighting": spec.context_weighting,
            "evidence_token_count": 3 + self.candidate_evidence.max_context_paths,
            "effective_token_count_shape": tuple(final_candidate_out.effective_token_count.shape),
            "effective_token_count": final_candidate_out.effective_token_count.detach().cpu(),
            "selected_context_indices": final_candidate_out.selected_context_indices.detach().cpu(),
            "selected_context_mask": final_candidate_out.selected_context_mask.detach().cpu(),
            "evidence_weights": final_candidate_out.evidence_weights.detach().cpu(),
            "context_path_weights": final_candidate_out.context_path_weights.detach().cpu(),
            "layerwise_setm_enabled": True,
            "layerwise_setm_stage": int(setm_stage),
            "layerwise_setm_layer_count": len(layerwise_setm_debug),
            "expected_setm_calls": self.num_layers,
            "actual_setm_calls": len(layerwise_setm_debug),
            "layerwise_setm_write_visual": bool(write_visual),
            "layerwise_setm_write_text": bool(write_text),
            "layerwise_setm_write_reverse_edge": bool(write_reverse_edge),
            "layerwise_setm_detach_text_update": bool(detach_text_update),
            "layerwise_setm_final_matching": False,
            "layerwise_setm_debug": layerwise_setm_debug,
            "final_prototype_slots_shape": tuple(current_prototype_slots.shape),
            "final_prototype_slot_mask_shape": tuple(current_prototype_slot_mask.shape),
            "layerwise_setm_has_nan": self._has_nan(
                pair_out, updated_node_feats, updated_edge_attr, current_prototype_slots
            ),
        })
        if debug:
            debug_info["effective_token_count"] = final_candidate_out.effective_token_count.detach().cpu()
            debug_info["context_path_weights"] = final_candidate_out.context_path_weights.detach().cpu()

        return VETAblationLayerwiseOutput(
            pair_feats=pair_out,
            node_feats=updated_node_feats,
            edge_attr_dense=updated_edge_attr,
            edge_mask_dense=edge_mask_dense,
            edge_score_dense=edge_score_dense,
            refined_edge_feats=refined_edge_feats,
            valid_pair_mask=valid_pair_mask,
            evidence_tokens=final_candidate_out.evidence_tokens,
            evidence_mask=final_candidate_out.evidence_mask,
            evidence_weights=final_candidate_out.evidence_weights,
            evidence_context=final_candidate_out.evidence_context,
            anchor_delta=final_candidate_out.anchor_delta,
            selected_context_indices=final_candidate_out.selected_context_indices,
            selected_context_mask=final_candidate_out.selected_context_mask,
            debug_info=debug_info,
            layerwise_setm_debug=layerwise_setm_debug,
            final_prototype_slots=current_prototype_slots,
            final_prototype_slot_mask=current_prototype_slot_mask,
        )

    @staticmethod
    def _valid_norm(delta: Tensor, valid_mask: Tensor) -> float:
        if not bool(valid_mask.any().detach().cpu().item()):
            return 0.0
        return float(delta[valid_mask].detach().float().norm(dim=-1).mean().cpu().item())

    @staticmethod
    def _scatter_pair_delta_to_dense_edges(
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        pair_indices: Tensor,
        pair_valid_mask: Tensor,
        pair_delta: Tensor,
        write_reverse_edge: bool,
    ) -> Tensor:
        updated = edge_attr_dense.clone()
        batch_size, num_nodes = edge_mask_dense.shape[:2]
        for batch_id in range(batch_size):
            for pair_id in range(pair_indices.shape[1]):
                if not bool(pair_valid_mask[batch_id, pair_id].detach().cpu().item()):
                    continue
                h = int(pair_indices[batch_id, pair_id, 0].detach().cpu().item())
                o = int(pair_indices[batch_id, pair_id, 1].detach().cpu().item())
                if h < 0 or o < 0 or h >= num_nodes or o >= num_nodes:
                    continue
                if bool(edge_mask_dense[batch_id, h, o].detach().cpu().item()):
                    updated[batch_id, h, o] = updated[batch_id, h, o] + pair_delta[batch_id, pair_id]
                if write_reverse_edge and bool(edge_mask_dense[batch_id, o, h].detach().cpu().item()):
                    updated[batch_id, o, h] = updated[batch_id, o, h] + pair_delta[batch_id, pair_id]
        return torch.nan_to_num(updated, nan=0.0, posinf=0.0, neginf=0.0)

    def _check_setm_inputs(
        self,
        node_feats: Tensor,
        pair_indices: Tensor,
        prototype_slots: Optional[Tensor],
        prototype_slot_mask: Optional[Tensor],
        setm_matcher: Optional[nn.Module],
        setm_stage: int,
    ) -> None:
        if setm_matcher is None:
            raise ValueError("setm_matcher is required")
        if setm_stage != 4:
            raise ValueError("layerwise VET ablation fixes SETM matcher behavior at Stage 4")
        if prototype_slots is None or prototype_slot_mask is None:
            raise ValueError("prototype slots and mask are required")
        batch_size, num_pairs = pair_indices.shape[:2]
        if prototype_slots.dim() != 5 or tuple(prototype_slots.shape[:2]) != (batch_size, num_pairs):
            raise ValueError("prototype_slots must be [B,Q,R,S,D]")
        if prototype_slots.shape[-1] != self.d_model:
            raise ValueError("prototype slot feature dim must equal VET d_model")
        if tuple(prototype_slot_mask.shape) != tuple(prototype_slots.shape[:4]):
            raise ValueError("prototype_slot_mask must match prototype_slots[:4]")
        if prototype_slots.device != node_feats.device or prototype_slot_mask.device != node_feats.device:
            raise ValueError("prototype state must share the visual device")


def build_vet_internal_ablation_reasoner(args=None, **kwargs):
    if args is None:
        return VisualEvidenceTopologyVETAblationLayerwiseSETM(**kwargs)
    config: Dict[str, Any] = vet_ablation_config_from_args(args)
    config.update(kwargs)
    return VisualEvidenceTopologyVETAblationLayerwiseSETM(**config)


build_visual_evidence_topology_vet_ablation = build_vet_internal_ablation_reasoner

