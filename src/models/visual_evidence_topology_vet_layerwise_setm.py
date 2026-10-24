"""Visual Evidence Topology with layerwise SETM synchronization."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from src.models.visual_evidence_topology_vet import VETOutput, VisualEvidenceTopology


@dataclass
class LayerwiseSETMVETOutput(VETOutput):
    layerwise_setm_debug: Any
    final_prototype_slots: Tensor
    final_prototype_slot_mask: Tensor


class VisualEvidenceTopologyLayerwiseSETM(VisualEvidenceTopology):
    """Run Stage-4 SETM after every visual relation reasoning layer."""

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
    ) -> LayerwiseSETMVETOutput:
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
            node_feats=node_feats,
            pair_indices=pair_indices,
            prototype_slots=prototype_slots,
            prototype_slot_mask=prototype_slot_mask,
            setm_matcher=setm_matcher,
            setm_stage=setm_stage,
        )

        fused_node_feats = self._fuse_node_inputs(
            node_feats,
            node_blip_feats,
            boxes,
            labels,
            node_types,
        )
        fused_node_feats = fused_node_feats * node_mask.unsqueeze(-1).to(
            dtype=fused_node_feats.dtype
        )
        if pair_feats is None:
            base_pair_feats = self._build_pair_feats_from_endpoints(
                fused_node_feats,
                pair_indices,
                pair_scores,
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
            updated_node_feats, updated_edge_attr, layer_info = layer(
                updated_node_feats,
                updated_edge_attr,
                edge_mask_dense,
                node_mask,
            )
            layer_info["layer_index"] = layer_id
            layer_debug.append(layer_info)

            refined_edge_feats, valid_pair_mask = self._gather_pair_edge_feats(
                updated_edge_attr,
                edge_mask_dense,
                pair_indices,
                node_mask,
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
                node_mask=node_mask,
                pair_indices=pair_indices,
                pair_valid_mask=valid_pair_mask,
                anchor_feats=refined_edge_feats,
                human_feats=node_h,
                object_feats=node_o,
            )
            layer_setm_out = setm_matcher(
                visual_tokens=candidate_out.evidence_tokens,
                visual_mask=candidate_out.evidence_mask,
                prototype_slots=current_prototype_slots,
                prototype_slot_mask=current_prototype_slot_mask,
                pair_valid_mask=valid_pair_mask,
                stage=4,
            )

            if write_visual:
                refined_edge_delta = candidate_out.anchor_delta + layer_setm_out.anchor_delta
                updated_edge_attr = self._scatter_pair_delta_to_dense_edges(
                    edge_attr_dense=updated_edge_attr,
                    edge_mask_dense=edge_mask_dense,
                    pair_indices=pair_indices,
                    pair_valid_mask=valid_pair_mask,
                    pair_delta=refined_edge_delta,
                    write_reverse_edge=write_reverse_edge,
                )
            if write_text:
                current_prototype_slots = layer_setm_out.updated_prototype_slots
                if detach_text_update:
                    current_prototype_slots = current_prototype_slots.detach()

            layerwise_setm_debug.append({
                "layer_id": layer_id,
                "candidate_evidence_debug": candidate_out.debug_info,
                "setm_debug": layer_setm_out.debug_info,
                "write_visual": bool(write_visual),
                "write_text": bool(write_text),
                "write_reverse_edge": bool(write_reverse_edge),
                "detach_text_update": bool(detach_text_update),
            })

        refined_edge_feats, valid_pair_mask = self._gather_pair_edge_feats(
            updated_edge_attr,
            edge_mask_dense,
            pair_indices,
            node_mask,
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
            node_mask=node_mask,
            pair_indices=pair_indices,
            pair_valid_mask=valid_pair_mask,
            anchor_feats=refined_edge_feats,
            human_feats=node_h,
            object_feats=node_o,
        )
        # The post-loop CETI pass exports final evidence only. Its delta is not
        # injected again after the last layerwise CETI+SETM edge write-back.
        final_refined_edge_feats = refined_edge_feats

        union_input = union_feats if union_feats is not None else torch.zeros_like(base_pair_feats)
        pair_out_input = torch.cat(
            [
                final_refined_edge_feats,
                node_h,
                node_o,
                base_pair_feats,
                union_input,
            ],
            dim=-1,
        )
        pair_out = self.pair_out_mlp(pair_out_input)
        fallback_pair_feats = base_pair_feats if pair_feats is not None else torch.zeros_like(
            base_pair_feats
        )
        pair_out = torch.where(valid_pair_mask.unsqueeze(-1), pair_out, fallback_pair_feats)

        debug_info = self._build_debug_info(
            pair_source=pair_source,
            input_node_feats=node_feats,
            node_blip_feats=node_blip_feats,
            fused_node_feats=fused_node_feats,
            updated_node_feats=updated_node_feats,
            pair_indices=pair_indices,
            pair_out=pair_out,
            refined_edge_feats=final_refined_edge_feats,
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
            "layerwise_setm_enabled": True,
            "layerwise_setm_stage": int(setm_stage),
            "layerwise_setm_layer_count": len(layerwise_setm_debug),
            "layerwise_setm_write_visual": bool(write_visual),
            "layerwise_setm_write_text": bool(write_text),
            "layerwise_setm_write_reverse_edge": bool(write_reverse_edge),
            "layerwise_setm_detach_text_update": bool(detach_text_update),
            "layerwise_setm_debug_enabled": bool(debug),
            "layerwise_setm_final_matching": False,
            "layerwise_setm_debug": layerwise_setm_debug,
            "final_prototype_slots_shape": tuple(current_prototype_slots.shape),
            "final_prototype_slot_mask_shape": tuple(current_prototype_slot_mask.shape),
            "layerwise_setm_has_nan": self._has_nan(
                pair_out,
                updated_node_feats,
                updated_edge_attr,
                current_prototype_slots,
            ),
        })

        return LayerwiseSETMVETOutput(
            pair_feats=pair_out,
            node_feats=updated_node_feats,
            edge_attr_dense=updated_edge_attr,
            edge_mask_dense=edge_mask_dense,
            edge_score_dense=edge_score_dense,
            refined_edge_feats=final_refined_edge_feats,
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

    def _scatter_pair_delta_to_dense_edges(
        self,
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        pair_indices: Tensor,
        pair_valid_mask: Tensor,
        pair_delta: Tensor,
        write_reverse_edge: bool,
    ) -> Tensor:
        updated_dense = edge_attr_dense.clone()
        batch_size, num_nodes = edge_mask_dense.shape[:2]
        for batch_idx in range(batch_size):
            for pair_idx in range(pair_indices.shape[1]):
                if not bool(pair_valid_mask[batch_idx, pair_idx].detach().cpu().item()):
                    continue
                human_idx = int(pair_indices[batch_idx, pair_idx, 0].detach().cpu().item())
                object_idx = int(pair_indices[batch_idx, pair_idx, 1].detach().cpu().item())
                if human_idx < 0 or object_idx < 0 or human_idx >= num_nodes or object_idx >= num_nodes:
                    continue
                if bool(edge_mask_dense[batch_idx, human_idx, object_idx].detach().cpu().item()):
                    updated_dense[batch_idx, human_idx, object_idx] = (
                        updated_dense[batch_idx, human_idx, object_idx]
                        + pair_delta[batch_idx, pair_idx]
                    )
                if (
                    write_reverse_edge
                    and bool(edge_mask_dense[batch_idx, object_idx, human_idx].detach().cpu().item())
                ):
                    updated_dense[batch_idx, object_idx, human_idx] = (
                        updated_dense[batch_idx, object_idx, human_idx]
                        + pair_delta[batch_idx, pair_idx]
                    )
        return torch.nan_to_num(updated_dense, nan=0.0, posinf=0.0, neginf=0.0)

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
            raise ValueError("setm_matcher is required for layerwise SETM")
        if setm_stage != 4:
            raise ValueError(f"layerwise SETM uses Stage-4 matcher behavior, got stage={setm_stage}")
        if prototype_slots is None or prototype_slot_mask is None:
            raise ValueError("prototype_slots and prototype_slot_mask are required for layerwise SETM")
        batch_size, num_pairs = pair_indices.shape[:2]
        if prototype_slots.dim() != 5:
            raise ValueError(
                "prototype_slots must be [B,K,R,S,D], "
                f"got {tuple(prototype_slots.shape)}"
            )
        if tuple(prototype_slots.shape[:2]) != (batch_size, num_pairs):
            raise ValueError(
                "prototype_slots [B,K] must match pair_indices; "
                f"expected {(batch_size, num_pairs)}, got {tuple(prototype_slots.shape[:2])}"
            )
        if prototype_slots.shape[-1] != self.d_model:
            raise ValueError(
                f"prototype_slots D must equal d_model={self.d_model}, "
                f"got {prototype_slots.shape[-1]}"
            )
        if tuple(prototype_slot_mask.shape) != tuple(prototype_slots.shape[:4]):
            raise ValueError(
                "prototype_slot_mask must match prototype_slots shape[:4]; "
                f"expected {tuple(prototype_slots.shape[:4])}, "
                f"got {tuple(prototype_slot_mask.shape)}"
            )
        if prototype_slots.device != node_feats.device:
            raise ValueError("prototype_slots must be on the same device as node_feats")
        if prototype_slot_mask.device != node_feats.device:
            raise ValueError("prototype_slot_mask must be on the same device as node_feats")


def build_vet_layerwise_setm_reasoner(args=None, **kwargs) -> VisualEvidenceTopologyLayerwiseSETM:
    if args is None:
        return VisualEvidenceTopologyLayerwiseSETM(**kwargs)

    dropout = getattr(args, "graph_dropout", 0.1)
    graph_oo_mode = getattr(args, "graph_oo_mode", "hybrid")
    if graph_oo_mode == "text_hybrid":
        raise ValueError("B5 layerwise VET does not allow text_hybrid OO mode.")
    config = {
        "d_model": getattr(args, "hidden_dim", 256),
        "num_obj_classes": getattr(args, "num_obj_classes", 80),
        "num_edge_types": getattr(args, "graph_num_edge_types", 4),
        "num_node_types": getattr(args, "graph_num_node_types", 3),
        "num_layers": getattr(args, "graph_num_layers", 2),
        "num_heads": getattr(args, "graph_num_heads", 8),
        "k_oo": getattr(args, "graph_k_oo", 3),
        "enable_oo_edges": getattr(args, "graph_enable_oo_edges", True),
        "enable_hh_edges": getattr(args, "graph_enable_hh_edges", False),
        "dropout": dropout,
        "spatial_sigma": getattr(args, "graph_spatial_sigma", 0.5),
        "graph_oo_mode": graph_oo_mode,
        "graph_oo_lambda_geo": getattr(args, "graph_oo_lambda_geo", 1.0),
        "graph_oo_lambda_share": getattr(args, "graph_oo_lambda_share", 0.5),
        "graph_oo_min_score": getattr(args, "graph_oo_min_score", 0.0),
        "graph_oo_fallback_to_spatial": getattr(args, "graph_oo_fallback_to_spatial", True),
        "graph_oo_share_eps": getattr(args, "graph_oo_share_eps", 1e-6),
        "graph_evidence_topk": getattr(args, "graph_evidence_topk", 4),
        "graph_evidence_gate_init": getattr(args, "graph_evidence_gate_init", 0.0),
        "graph_evidence_dropout": getattr(args, "graph_evidence_dropout", dropout),
        "graph_evidence_allow_node_only_context": getattr(
            args,
            "graph_evidence_allow_node_only_context",
            False,
        ),
        "graph_evidence_debug": getattr(args, "graph_evidence_debug", False),
        "debug": getattr(args, "vet_debug", getattr(args, "graph_debug", False)),
        "store_dense_debug_tensors": getattr(args, "vet_store_dense_debug_tensors", False),
    }
    config.update(kwargs)
    return VisualEvidenceTopologyLayerwiseSETM(**config)


build_visual_evidence_topology_layerwise_setm = build_vet_layerwise_setm_reasoner
