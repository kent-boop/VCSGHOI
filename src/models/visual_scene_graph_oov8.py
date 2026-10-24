"""Visual scene graph oov8 with Stage 4-standard layerwise SETM flags."""

import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


def _load_oov6_classes():
    try:
        from .visual_scene_graph_oov6 import GraphReasonerOutput, VisualSceneGraphReasoner
    except ImportError:
        module_path = Path(__file__).resolve().with_name("visual_scene_graph_oov6.py")
        spec = importlib.util.spec_from_file_location("visual_scene_graph_oov6_direct", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        GraphReasonerOutput = module.GraphReasonerOutput
        VisualSceneGraphReasoner = module.VisualSceneGraphReasoner
    return GraphReasonerOutput, VisualSceneGraphReasoner


def _load_sync_matcher_classes():
    try:
        from .synchronized_evidence_topology_matcher import SynchronizedEvidenceTopologyMatcher
    except ImportError:
        module_path = Path(__file__).resolve().with_name("synchronized_evidence_topology_matcher.py")
        spec = importlib.util.spec_from_file_location("synchronized_evidence_topology_matcher_direct", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        SynchronizedEvidenceTopologyMatcher = module.SynchronizedEvidenceTopologyMatcher
    return SynchronizedEvidenceTopologyMatcher


def _load_text_scene_graph_oov3_reasoner():
    try:
        from .text_scene_graph_oov3 import TextSceneGraphReasoner
    except ImportError:
        module_path = Path(__file__).resolve().with_name("text_scene_graph_oov3.py")
        spec = importlib.util.spec_from_file_location("text_scene_graph_oov3_direct", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        TextSceneGraphReasoner = module.TextSceneGraphReasoner
    return TextSceneGraphReasoner


GraphReasonerOutput, _BaseVisualSceneGraphReasoner = _load_oov6_classes()


class VisualSceneGraphReasoner(_BaseVisualSceneGraphReasoner):
    def __init__(
        self,
        *args: Any,
        graph_enable_sync_evidence_matching: bool = False,
        graph_sync_visual_gate_init: float = 0.0,
        graph_sync_text_gate_init: float = 0.0,
        graph_sync_anchor_gate_init: float = 0.0,
        graph_sync_dropout: Optional[float] = None,
        graph_sync_debug: bool = False,
        graph_text_force_oov3: bool = True,
        graph_text_prototype_slot_fallback_to_raw: bool = True,
        graph_text_prototype_slot_debug: bool = False,
        graph_enable_layerwise_sync_evidence_matching: bool = False,
        graph_layerwise_sync_write_visual: bool = True,
        graph_layerwise_sync_write_text: bool = True,
        graph_layerwise_sync_final_matching: bool = False,
        graph_layerwise_sync_disable_old_dense_injection: bool = True,
        graph_layerwise_sync_write_reverse_edge: bool = False,
        graph_layerwise_sync_detach_text_update: bool = False,
        graph_layerwise_sync_debug: bool = False,
        graph_store_dense_debug_tensors: bool = False,
        graph_export_evidence_trace: bool = False,
        graph_export_trace_each_layer: bool = False,
        graph_export_trace_to_cpu: bool = True,
        graph_export_trace_float_dtype: str = "float16",
        graph_export_trace_include_match_logits: bool = False,
        graph_export_trace_include_features: bool = False,
        graph_export_mechanism_validation_trace: bool = False,
        graph_mechanism_validation_variant: str = "full",
        **kwargs: Any,
    ):
        if graph_enable_layerwise_sync_evidence_matching and bool(kwargs.get("graph_text_dynamic_retrieval", False)):
            raise ValueError(
                "Layerwise SETM requires graph_text_dynamic_retrieval=False because rebuilding text_state erases visual-to-text updates."
            )
        if graph_enable_layerwise_sync_evidence_matching and graph_layerwise_sync_final_matching:
            raise ValueError(
                "graph_layerwise_sync_final_matching=True is not implemented for Stage 4-standard. "
                "Use graph_layerwise_sync_final_matching=False to avoid duplicate final SETM injection."
            )
        super().__init__(*args, **kwargs)
        self.graph_enable_sync_evidence_matching = bool(graph_enable_sync_evidence_matching)
        self.graph_sync_debug = bool(graph_sync_debug)
        self.graph_text_force_oov3 = bool(graph_text_force_oov3)
        self.graph_text_prototype_slot_fallback_to_raw = bool(graph_text_prototype_slot_fallback_to_raw)
        self.graph_text_prototype_slot_debug = bool(graph_text_prototype_slot_debug)
        self.graph_enable_layerwise_sync_evidence_matching = bool(graph_enable_layerwise_sync_evidence_matching)
        self.graph_layerwise_sync_write_visual = bool(graph_layerwise_sync_write_visual)
        self.graph_layerwise_sync_write_text = bool(graph_layerwise_sync_write_text)
        self.graph_layerwise_sync_final_matching = bool(graph_layerwise_sync_final_matching)
        self.graph_layerwise_sync_disable_old_dense_injection = bool(graph_layerwise_sync_disable_old_dense_injection)
        self.graph_layerwise_sync_write_reverse_edge = bool(graph_layerwise_sync_write_reverse_edge)
        self.graph_layerwise_sync_detach_text_update = bool(graph_layerwise_sync_detach_text_update)
        self.graph_layerwise_sync_debug = bool(graph_layerwise_sync_debug)
        self.graph_store_dense_debug_tensors = bool(graph_store_dense_debug_tensors)
        self.graph_export_evidence_trace = bool(graph_export_evidence_trace)
        self.graph_export_trace_each_layer = bool(graph_export_trace_each_layer)
        self.graph_export_trace_to_cpu = bool(graph_export_trace_to_cpu)
        self.graph_export_trace_float_dtype = str(graph_export_trace_float_dtype).lower()
        self.graph_export_trace_include_match_logits = bool(graph_export_trace_include_match_logits)
        self.graph_export_trace_include_features = bool(graph_export_trace_include_features)
        self.graph_export_mechanism_validation_trace = bool(graph_export_mechanism_validation_trace)
        self.graph_mechanism_validation_variant = str(graph_mechanism_validation_variant)
        valid_mechanism_variants = {
            "full",
            "no_anchor_writeback",
            "no_prototype_state_writeback",
            "no_setm",
            "endpoint_only_setm",
        }
        if self.graph_mechanism_validation_variant not in valid_mechanism_variants:
            raise ValueError(f"Unsupported graph_mechanism_validation_variant={self.graph_mechanism_validation_variant!r}")
        valid_trace_dtypes = {"float16", "float32", "bfloat16"}
        if self.graph_export_trace_float_dtype not in valid_trace_dtypes:
            raise ValueError(
                "graph_export_trace_float_dtype must be one of "
                f"{sorted(valid_trace_dtypes)}, got {self.graph_export_trace_float_dtype}"
            )

        if self.graph_enable_layerwise_sync_evidence_matching and self.graph_layerwise_sync_final_matching:
            raise ValueError(
                "graph_layerwise_sync_final_matching=True is not implemented for Stage 4-standard. "
                "Use graph_layerwise_sync_final_matching=False to avoid duplicate final SETM injection."
            )

        if self.enable_text_graph and self.graph_text_force_oov3 and self.text_graph_reasoner is not None:
            TextSceneGraphReasonerV3 = _load_text_scene_graph_oov3_reasoner()
            old_reasoner = self.text_graph_reasoner
            self.text_graph_reasoner = TextSceneGraphReasonerV3(
                retriever=old_reasoner.retriever,
                d_model=self.d_model,
                num_layers=getattr(old_reasoner, "num_layers", getattr(self, "num_layers", 1)),
                num_heads=getattr(old_reasoner, "num_heads", getattr(self, "num_heads", 8)),
                dropout=kwargs.get("graph_text_dropout", kwargs.get("dropout", 0.1)),
                use_field_tokens=getattr(old_reasoner, "use_field_tokens", ("human_node", "object_node", "ho_edge", "discriminative", "scene_context", "oo_context")),
                use_scene_token=getattr(old_reasoner, "use_scene_token", True),
                use_oo_context_token=getattr(old_reasoner, "use_oo_context_token", True),
                use_score_embedding=getattr(old_reasoner, "use_score_embedding", True),
                use_rank_embedding=getattr(old_reasoner, "use_rank_embedding", True),
                max_topk=getattr(old_reasoner, "max_topk", max(int(kwargs.get("graph_text_topk", 5)), 1)),
                field_dropout=0.0,
                debug=getattr(old_reasoner, "debug", False),
                enable_prototype_slots=True,
                prototype_slot_fallback_to_raw=self.graph_text_prototype_slot_fallback_to_raw,
                prototype_slot_debug=self.graph_text_prototype_slot_debug or self.graph_sync_debug,
            )

        if self.graph_enable_sync_evidence_matching:
            if not self.enable_text_graph:
                raise ValueError("graph_enable_sync_evidence_matching=True requires enable_text_graph=True")
            if not self.graph_enable_candidate_evidence:
                raise ValueError("graph_enable_sync_evidence_matching=True requires graph_enable_candidate_evidence=True")

        if self.graph_enable_layerwise_sync_evidence_matching:
            if self.graph_text_dynamic_retrieval:
                raise ValueError(
                    "Layerwise SETM requires graph_text_dynamic_retrieval=False because rebuilding text_state erases visual-to-text updates."
                )
            if not self.graph_enable_sync_evidence_matching:
                raise ValueError(
                    "graph_enable_layerwise_sync_evidence_matching=True requires graph_enable_sync_evidence_matching=True"
                )
            if not self.enable_text_graph:
                raise ValueError("graph_enable_layerwise_sync_evidence_matching=True requires enable_text_graph=True")
            if not self.graph_enable_candidate_evidence:
                raise ValueError("graph_enable_layerwise_sync_evidence_matching=True requires graph_enable_candidate_evidence=True")
            if self.graph_text_sync_mode != "layerwise":
                raise ValueError('graph_enable_layerwise_sync_evidence_matching=True requires graph_text_sync_mode="layerwise"')

        self.sync_evidence_matcher = None
        if self.graph_enable_sync_evidence_matching or self.graph_enable_layerwise_sync_evidence_matching:
            SynchronizedEvidenceTopologyMatcher = _load_sync_matcher_classes()
            sync_dropout = kwargs.get("dropout", 0.1) if graph_sync_dropout is None else graph_sync_dropout
            self.sync_evidence_matcher = SynchronizedEvidenceTopologyMatcher(
                d_model=self.d_model,
                num_slots=6,
                dropout=float(sync_dropout),
                visual_gate_init=graph_sync_visual_gate_init,
                text_gate_init=graph_sync_text_gate_init,
                anchor_gate_init=graph_sync_anchor_gate_init,
                normalize_inputs=True,
                debug=graph_sync_debug,
            )
        if self.graph_enable_layerwise_sync_evidence_matching and self.candidate_evidence_inducer is None:
            raise ValueError("graph_enable_layerwise_sync_evidence_matching=True requires candidate_evidence_inducer to be initialized")
        if self.graph_enable_layerwise_sync_evidence_matching and self.sync_evidence_matcher is None:
            raise ValueError("graph_enable_layerwise_sync_evidence_matching=True requires sync_evidence_matcher to be initialized")

    def _scatter_pair_delta_to_dense_edges(
        self,
        edge_attr_dense: Tensor,
        pair_indices: Tensor,
        pair_delta: Tensor,
        valid_pair_mask: Tensor,
        edge_mask_dense: Tensor,
        write_reverse: bool = False,
    ) -> Tensor:
        new_edge_attr = edge_attr_dense.clone()
        if edge_attr_dense.numel() == 0 or pair_indices.numel() == 0:
            return torch.nan_to_num(new_edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

        batch_size, num_nodes, _, dim = edge_attr_dense.shape
        pair_delta = torch.nan_to_num(pair_delta, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=edge_attr_dense.dtype)
        h_idx = pair_indices[..., 0].long()
        o_idx = pair_indices[..., 1].long()
        valid_pair_mask = valid_pair_mask.to(dtype=torch.bool)
        in_bounds = valid_pair_mask & (h_idx >= 0) & (h_idx < num_nodes) & (o_idx >= 0) & (o_idx < num_nodes)

        safe_h = h_idx.clamp(min=0, max=max(num_nodes - 1, 0))
        safe_o = o_idx.clamp(min=0, max=max(num_nodes - 1, 0))
        batch_idx = torch.arange(batch_size, device=edge_attr_dense.device).unsqueeze(1).expand_as(safe_h)

        def scatter_delta(src_idx: Tensor, dst_idx: Tensor, write_mask: Tensor) -> None:
            existing_edge = edge_mask_dense[batch_idx, src_idx, dst_idx].to(dtype=torch.bool)
            final_mask = write_mask & existing_edge
            if not bool(final_mask.any().detach().cpu().item()):
                return
            flat_index = (
                batch_idx[final_mask] * num_nodes * num_nodes
                + src_idx[final_mask] * num_nodes
                + dst_idx[final_mask]
            )
            flat_edge_attr = new_edge_attr.reshape(batch_size * num_nodes * num_nodes, dim)
            flat_edge_attr.index_add_(0, flat_index, pair_delta[final_mask])

        scatter_delta(safe_h, safe_o, in_bounds)
        if write_reverse:
            scatter_delta(safe_o, safe_h, in_bounds)
        return torch.nan_to_num(new_edge_attr, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_prototype_slot_update_to_text_state(
        self,
        text_state,
        updated_prototype_slots: Tensor,
        prototype_slot_mask: Tensor,
        detach_update: bool = False,
    ):
        if detach_update:
            updated_prototype_slots = updated_prototype_slots.detach()
        updated_prototype_slots = torch.nan_to_num(updated_prototype_slots, nan=0.0, posinf=0.0, neginf=0.0)
        prototype_slot_mask = prototype_slot_mask.to(dtype=torch.bool)

        def masked_replace(old: Tensor, new: Tensor, mask: Tensor) -> Tensor:
            new = torch.nan_to_num(new.to(dtype=old.dtype), nan=0.0, posinf=0.0, neginf=0.0)
            return torch.where(mask.unsqueeze(-1), new, old)

        node_feats = text_state.node_feats.clone()
        edge_attr = text_state.edge_attr.clone()
        raw_field_feats = text_state.raw_field_feats.clone()

        node_feats[..., 0, :] = masked_replace(
            node_feats[..., 0, :],
            updated_prototype_slots[..., 0, :],
            prototype_slot_mask[..., 0],
        )
        node_feats[..., 1, :] = masked_replace(
            node_feats[..., 1, :],
            updated_prototype_slots[..., 1, :],
            prototype_slot_mask[..., 1],
        )
        edge_attr[..., 0, 1, :] = masked_replace(
            edge_attr[..., 0, 1, :],
            updated_prototype_slots[..., 2, :],
            prototype_slot_mask[..., 2],
        )
        if self.graph_layerwise_sync_write_reverse_edge:
            edge_attr[..., 1, 0, :] = masked_replace(
                edge_attr[..., 1, 0, :],
                updated_prototype_slots[..., 2, :],
                prototype_slot_mask[..., 2],
            )
        raw_field_feats[..., 3, :] = masked_replace(
            raw_field_feats[..., 3, :],
            updated_prototype_slots[..., 3, :],
            prototype_slot_mask[..., 3],
        )
        node_feats[..., 2, :] = masked_replace(
            node_feats[..., 2, :],
            updated_prototype_slots[..., 4, :],
            prototype_slot_mask[..., 4],
        )
        node_feats[..., 3, :] = masked_replace(
            node_feats[..., 3, :],
            updated_prototype_slots[..., 5, :],
            prototype_slot_mask[..., 5],
        )

        node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        raw_field_feats = torch.nan_to_num(raw_field_feats, nan=0.0, posinf=0.0, neginf=0.0)
        has_nan = bool(
            torch.isnan(node_feats).any().detach().cpu().item()
            or torch.isnan(edge_attr).any().detach().cpu().item()
            or torch.isnan(raw_field_feats).any().detach().cpu().item()
        )
        debug_info = dict(text_state.debug_info)
        debug_info.update({
            "layerwise_text_state_updated": True,
            "layerwise_text_state_detached": bool(detach_update),
            "layerwise_text_state_updated_slot_count": int(prototype_slot_mask.sum().detach().cpu().item()),
            "layerwise_text_state_has_nan": has_nan,
        })
        return replace(
            text_state,
            node_feats=node_feats,
            edge_attr=edge_attr,
            raw_field_feats=raw_field_feats,
            debug_info=debug_info,
        )

    def _trace_tensor(self, tensor: Optional[Tensor]) -> Optional[Tensor]:
        """Detach a tensor for evaluation-only evidence tracing.

        Tracing is strictly opt-in. Floating tensors are compressed to the
        configured dtype and can be moved to CPU before leaving the graph
        reasoner, preventing the caller from retaining the autograd graph.
        """
        if tensor is None:
            return None
        out = tensor.detach()
        if torch.is_floating_point(out):
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
            }
            out = out.to(dtype=dtype_map[self.graph_export_trace_float_dtype])
        if self.graph_export_trace_to_cpu:
            out = out.cpu()
        return out

    def _selected_path_support(
        self,
        pair_indices: Tensor,
        selected_context_indices: Tensor,
        selected_context_mask: Tensor,
        edge_mask_dense: Tensor,
        edge_score_dense: Tensor,
    ) -> Dict[str, Tensor]:
        """Gather whether every selected context forms h->k and k->o edges."""
        batch_size, num_pairs, context_count = selected_context_indices.shape
        num_nodes = edge_mask_dense.shape[1]
        device = pair_indices.device
        batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        batch_idx = batch_idx.expand(batch_size, num_pairs, context_count)

        h_idx = pair_indices[..., 0].unsqueeze(-1).expand_as(selected_context_indices)
        o_idx = pair_indices[..., 1].unsqueeze(-1).expand_as(selected_context_indices)
        k_idx = selected_context_indices.long()
        in_bounds = (
            (h_idx >= 0)
            & (h_idx < num_nodes)
            & (o_idx >= 0)
            & (o_idx < num_nodes)
            & (k_idx >= 0)
            & (k_idx < num_nodes)
            & selected_context_mask.to(dtype=torch.bool)
        )
        safe_h = h_idx.clamp(min=0, max=max(num_nodes - 1, 0))
        safe_o = o_idx.clamp(min=0, max=max(num_nodes - 1, 0))
        safe_k = k_idx.clamp(min=0, max=max(num_nodes - 1, 0))

        hk_mask = edge_mask_dense[batch_idx, safe_h, safe_k].to(dtype=torch.bool) & in_bounds
        ko_mask = edge_mask_dense[batch_idx, safe_k, safe_o].to(dtype=torch.bool) & in_bounds
        hk_score = edge_score_dense[batch_idx, safe_h, safe_k]
        ko_score = edge_score_dense[batch_idx, safe_k, safe_o]
        hk_score = torch.where(hk_mask, hk_score, torch.zeros_like(hk_score))
        ko_score = torch.where(ko_mask, ko_score, torch.zeros_like(ko_score))
        return {
            "selected_context_edge_hk_mask": hk_mask,
            "selected_context_edge_ko_mask": ko_mask,
            "selected_context_two_hop_mask": hk_mask & ko_mask,
            "selected_context_edge_hk_score": hk_score,
            "selected_context_edge_ko_score": ko_score,
        }

    def _selected_hoi_semantic_ids(self, text_out: Any) -> Dict[str, Tensor]:
        """Map retrieved text-bank rows to action/object ids when available."""
        if text_out is None or not hasattr(text_out, "selected_hoi_indices"):
            return {}
        selected = text_out.selected_hoi_indices.long()
        selected_mask = text_out.selected_hoi_mask.to(dtype=torch.bool)
        bank = getattr(getattr(self.text_graph_reasoner, "retriever", None), "bank", None)
        if bank is None:
            return {}

        out: Dict[str, Tensor] = {}
        for source_name, output_name in (
            ("hoi_action_ids", "selected_hoi_action_ids"),
            ("hoi_object_ids", "selected_hoi_object_ids"),
        ):
            source = getattr(bank, source_name, None)
            if source is None or source.numel() == 0:
                continue
            source = source.to(device=selected.device)
            safe = selected.clamp(min=0, max=max(source.shape[0] - 1, 0))
            mapped = source[safe]
            if source_name == "hoi_action_ids" and source.numel() > 0:
                raw_mapped = torch.where(selected_mask, mapped, torch.full_like(mapped, -1))
                out["selected_hoi_raw_action_ids"] = raw_mapped
                if int(source.min().detach().cpu().item()) >= 1:
                    mapped = mapped - 1
            mapped = torch.where(selected_mask, mapped, torch.full_like(mapped, -1))
            out[output_name] = mapped
        return out

    def _build_evidence_trace(
        self,
        *,
        stage: str,
        layer_idx: int,
        pair_indices: Tensor,
        pair_valid_mask: Tensor,
        candidate_out: Any,
        text_out: Any,
        sync_out: Any,
        edge_mask_dense: Tensor,
        edge_score_dense: Tensor,
        pair_anchor_pre_sync: Optional[Tensor] = None,
        pair_anchor_post_sync: Optional[Tensor] = None,
        pair_anchor_post_sync_actual: Optional[Tensor] = None,
        sync_anchor_delta_applied: Optional[Tensor] = None,
        candidate_anchor_delta_applied: Optional[Tensor] = None,
        prototype_slots_post_writeback: Optional[Tensor] = None,
        graph_pair_features_after_layer: Optional[Tensor] = None,
        current_human_node_state: Optional[Tensor] = None,
        current_object_node_state: Optional[Tensor] = None,
        sync_visual_mask: Optional[Tensor] = None,
        setm_called: bool = True,
        visual_writeback_applied: bool = False,
        text_writeback_applied: bool = False,
    ) -> Dict[str, Any]:
        """Create the compact tensor trace used by the example-mining pipeline."""
        selected_context_indices = candidate_out.selected_context_indices
        selected_context_mask = candidate_out.selected_context_mask.to(dtype=torch.bool)
        path_support = self._selected_path_support(
            pair_indices=pair_indices,
            selected_context_indices=selected_context_indices,
            selected_context_mask=selected_context_mask,
            edge_mask_dense=edge_mask_dense,
            edge_score_dense=edge_score_dense,
        )
        evidence_count = int(candidate_out.evidence_tokens.shape[-2])
        context_count = max(evidence_count - 3, 0)
        trace: Dict[str, Any] = {
            "schema_version": 2,
            "stage": str(stage),
            "layer_idx": int(layer_idx),
            "mechanism_validation_variant": self.graph_mechanism_validation_variant,
            "setm_called": bool(setm_called),
            "setm_correction_count": int(bool(setm_called)),
            "visual_writeback_applied": bool(visual_writeback_applied),
            "text_writeback_applied": bool(text_writeback_applied),
            "visual_token_roles": ["anchor", "human", "object"]
            + [f"context_{idx + 1}" for idx in range(context_count)],
            "prototype_slot_names": [
                "human_state",
                "object_state",
                "anchor_relation",
                "discriminative",
                "scene",
                "context_object",
            ],
            "pair_indices": self._trace_tensor(pair_indices),
            "pair_valid_mask": self._trace_tensor(pair_valid_mask.to(dtype=torch.bool)),
            "selected_context_indices": self._trace_tensor(selected_context_indices),
            "selected_context_mask": self._trace_tensor(selected_context_mask),
            "candidate_evidence_mask": self._trace_tensor(candidate_out.evidence_mask.to(dtype=torch.bool)),
            "candidate_evidence_weights": self._trace_tensor(candidate_out.evidence_weights),
        }
        original_visual_mask = candidate_out.evidence_mask.to(dtype=torch.bool)
        if sync_visual_mask is None:
            sync_visual_mask = original_visual_mask
        sync_visual_mask = sync_visual_mask.to(dtype=torch.bool)
        mask_difference = original_visual_mask & ~sync_visual_mask
        if mask_difference.dim() > 1:
            # torch<=1.13 only accepts a single integer reduction dimension.
            reduced_difference = mask_difference.reshape(
                -1, mask_difference.shape[-1]
            ).any(dim=0)
        else:
            reduced_difference = mask_difference
        trace["sync_visual_mask"] = self._trace_tensor(sync_visual_mask)
        trace["masked_visual_indices"] = reduced_difference.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        trace["requested_masked_visual_indices"] = (
            list(range(3, evidence_count))
            if self.graph_mechanism_validation_variant == "endpoint_only_setm" else []
        )
        trace.update({name: self._trace_tensor(value) for name, value in path_support.items()})

        if text_out is not None:
            trace.update({
                "selected_hoi_indices": self._trace_tensor(text_out.selected_hoi_indices),
                "selected_hoi_mask": self._trace_tensor(text_out.selected_hoi_mask.to(dtype=torch.bool)),
                "selected_hoi_scores": self._trace_tensor(text_out.selected_scores),
                "prototype_slot_mask": self._trace_tensor(text_out.prototype_slot_mask.to(dtype=torch.bool)),
            })
            trace.update({
                name: self._trace_tensor(value)
                for name, value in self._selected_hoi_semantic_ids(text_out).items()
            })

        if sync_out is not None:
            trace.update({
                "sync_visual_to_text_attn": self._trace_tensor(sync_out.visual_to_text_attn),
                "sync_text_to_visual_attn": self._trace_tensor(sync_out.text_to_visual_attn),
                "sync_evidence_weights": self._trace_tensor(sync_out.evidence_weights),
                "sync_prototype_scores": self._trace_tensor(sync_out.prototype_scores),
                "sync_prototype_weights": self._trace_tensor(sync_out.prototype_weights),
                "sync_prototype_coverage": self._trace_tensor(sync_out.prototype_coverage),
                "sync_anchor_delta": self._trace_tensor(sync_out.anchor_delta),
            })
            if self.graph_export_trace_include_match_logits:
                trace["sync_match_logits"] = self._trace_tensor(sync_out.match_logits)

        if self.graph_export_trace_include_features:
            trace["candidate_evidence_tokens"] = self._trace_tensor(candidate_out.evidence_tokens)
            trace["candidate_evidence_context"] = self._trace_tensor(candidate_out.evidence_context)
            trace["candidate_anchor_delta"] = self._trace_tensor(candidate_out.anchor_delta)
            trace["candidate_anchor_delta_proposed"] = self._trace_tensor(candidate_out.anchor_delta)
            if candidate_anchor_delta_applied is not None:
                trace["candidate_anchor_delta_applied"] = self._trace_tensor(candidate_anchor_delta_applied)
            if text_out is not None:
                trace["prototype_slots"] = self._trace_tensor(text_out.prototype_slots)
                trace["prototype_slots_pre_sync"] = self._trace_tensor(text_out.prototype_slots)
            if prototype_slots_post_writeback is not None:
                trace["prototype_slots_post_writeback_actual"] = self._trace_tensor(prototype_slots_post_writeback)
            if sync_out is not None:
                trace["updated_visual_tokens"] = self._trace_tensor(sync_out.updated_visual_tokens)
                trace["updated_visual_tokens_proposed"] = self._trace_tensor(sync_out.updated_visual_tokens)
                trace["updated_prototype_slots"] = self._trace_tensor(sync_out.updated_prototype_slots)
                trace["updated_prototype_slots_proposed"] = self._trace_tensor(sync_out.updated_prototype_slots)
                trace["sync_matched_context"] = self._trace_tensor(sync_out.matched_context)
                trace["sync_anchor_delta_proposed"] = self._trace_tensor(sync_out.anchor_delta)
            if pair_anchor_pre_sync is not None:
                trace["pair_anchor_pre_sync"] = self._trace_tensor(pair_anchor_pre_sync)
            if pair_anchor_post_sync is not None:
                trace["pair_anchor_post_sync"] = self._trace_tensor(pair_anchor_post_sync)
            if pair_anchor_post_sync_actual is not None:
                trace["pair_anchor_post_sync_actual"] = self._trace_tensor(pair_anchor_post_sync_actual)
                trace["current_refined_edge_after_writeback"] = self._trace_tensor(pair_anchor_post_sync_actual)
            if sync_anchor_delta_applied is not None:
                trace["sync_anchor_delta_applied"] = self._trace_tensor(sync_anchor_delta_applied)
            if graph_pair_features_after_layer is not None:
                trace["graph_pair_features_after_layer"] = self._trace_tensor(graph_pair_features_after_layer)
            if current_human_node_state is not None:
                trace["current_human_node_state"] = self._trace_tensor(current_human_node_state)
            if current_object_node_state is not None:
                trace["current_object_node_state"] = self._trace_tensor(current_object_node_state)
        return trace

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
    ) -> GraphReasonerOutput:
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

        mechanism_variant = self.graph_mechanism_validation_variant
        mechanism_trace_enabled = bool(self.graph_export_mechanism_validation_trace and not self.training)
        if self.training and (self.graph_export_mechanism_validation_trace or mechanism_variant != "full"):
            raise RuntimeError("Mechanism validation trace/variants are evaluation-only")
        if mechanism_variant != "full" and not self.graph_export_mechanism_validation_trace:
            raise RuntimeError("Non-full mechanism variants require graph_export_mechanism_validation_trace=True")
        if mechanism_trace_enabled and not self.graph_export_evidence_trace:
            raise RuntimeError("Mechanism validation trace requires graph_export_evidence_trace=True")

        effective_write_visual = bool(self.graph_layerwise_sync_write_visual)
        effective_write_text = bool(self.graph_layerwise_sync_write_text)
        setm_enabled_for_variant = mechanism_variant != "no_setm"
        if mechanism_variant == "no_anchor_writeback":
            effective_write_visual = False
        elif mechanism_variant == "no_prototype_state_writeback":
            effective_write_text = False
        elif mechanism_variant == "no_setm":
            effective_write_visual = False
            effective_write_text = False

        layerwise_sync_enabled = bool(self.graph_enable_layerwise_sync_evidence_matching)
        layerwise_sync_debug = []
        mechanism_layer_pair_features: List[Tensor] = []
        evidence_trace_layers: List[Dict[str, Any]] = []
        final_evidence_trace: Optional[Dict[str, Any]] = None
        layerwise_sync_total_anchor_delta_norm = 0.0
        layerwise_sync_has_nan = False
        layerwise_skip_final_matching = layerwise_sync_enabled
        text_out_for_matching = None
        sync_evidence_debug_info = {
            "sync_evidence_enabled": bool(self.graph_enable_sync_evidence_matching),
            "sync_evidence_applied": False,
            "sync_evidence_skip_reason": "disabled" if not self.graph_enable_sync_evidence_matching else "not_attempted",
        }
        if not setm_enabled_for_variant:
            sync_evidence_debug_info["sync_evidence_variant_enabled"] = False
            sync_evidence_debug_info["sync_evidence_skip_reason"] = "mechanism_variant_no_setm"
        else:
            sync_evidence_debug_info["sync_evidence_variant_enabled"] = True

        fused_node_feats = self._fuse_node_inputs(node_feats, node_blip_feats, boxes, labels, node_types)
        fused_node_feats = fused_node_feats * node_mask.unsqueeze(-1).to(dtype=fused_node_feats.dtype)

        if pair_feats is None:
            base_pair_feats = self._build_pair_feats_from_endpoints(fused_node_feats, pair_indices, pair_scores)
            pair_source = "endpoint_node_feats"
        else:
            base_pair_feats = pair_feats
            pair_source = "provided_pair_feats"

        edge_attr_dense, edge_mask_dense, edge_score_dense, counts = self._build_dense_edges(
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

        union_input = union_feats if union_feats is not None else torch.zeros_like(base_pair_feats)
        text_state = None
        text_pair_valid_mask = None
        pair_object_labels = None
        pair_label_valid_mask = None
        text_layer_debug = []
        text_dense_injection_debug_per_layer = []
        text_layerwise_enabled = self.enable_text_graph and self.graph_text_sync_mode == "layerwise"
        if text_layerwise_enabled:
            initial_refined_edge, initial_valid_pair_mask = self._gather_pair_edge_feats(edge_attr_dense, pair_indices, node_mask)
            pair_object_labels, pair_label_valid_mask = self._gather_pair_object_labels(
                labels=labels,
                pair_indices=pair_indices,
                node_mask=node_mask,
            )
            text_pair_valid_mask = initial_valid_pair_mask & pair_label_valid_mask
            text_query_feats = self._build_text_query_feats(
                source=self.graph_text_sync_query_source,
                refined_edge=initial_refined_edge,
                base_pair_feats=base_pair_feats,
                union_feats=union_input,
            )
            text_state = self.text_graph_reasoner.build_initial_state(
                visual_pair_feats=text_query_feats,
                pair_object_labels=pair_object_labels,
                pair_valid_mask=text_pair_valid_mask,
            )
        if layerwise_sync_enabled and text_state is None:
            raise ValueError("graph_enable_layerwise_sync_evidence_matching=True requires text_state to be initialized")

        layer_debug = []
        updated_node_feats = fused_node_feats
        updated_edge_attr = edge_attr_dense
        for layer_idx, layer in enumerate(self.layers):
            updated_node_feats, updated_edge_attr, layer_info = layer(
                updated_node_feats,
                updated_edge_attr,
                edge_mask_dense,
                node_mask,
            )
            layer_info["layer_index"] = layer_idx
            layer_debug.append(layer_info)
            if text_layerwise_enabled and text_state is not None:
                refined_edge_l, valid_pair_mask_l = self._gather_pair_edge_feats(updated_edge_attr, pair_indices, node_mask)
                layer_text_pair_valid_mask = valid_pair_mask_l if text_pair_valid_mask is None else (valid_pair_mask_l & text_pair_valid_mask)
                if layerwise_sync_enabled:
                    if self.candidate_evidence_inducer is None:
                        raise ValueError("graph_enable_layerwise_sync_evidence_matching=True requires candidate_evidence_inducer to be initialized")
                    if self.sync_evidence_matcher is None:
                        raise ValueError("graph_enable_layerwise_sync_evidence_matching=True requires sync_evidence_matcher to be initialized")
                    node_h_l = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 0])
                    node_o_l = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 1])
                    text_state = self.text_graph_reasoner.step(text_state, layer_idx)
                    text_out_l = self.text_graph_reasoner.export_layer_outputs(text_state)
                    if not hasattr(text_out_l, "prototype_slots") or not hasattr(text_out_l, "prototype_slot_mask"):
                        raise ValueError(
                            "graph_enable_layerwise_sync_evidence_matching=True requires text_graph_reasoner.export_layer_outputs(...) "
                            "to return prototype_slots and prototype_slot_mask"
                        )
                    candidate_out_l = self.candidate_evidence_inducer(
                        node_feats=updated_node_feats,
                        edge_attr_dense=updated_edge_attr,
                        edge_mask_dense=edge_mask_dense,
                        node_mask=node_mask,
                        pair_indices=pair_indices,
                        pair_valid_mask=layer_text_pair_valid_mask,
                        anchor_feats=refined_edge_l,
                        human_feats=node_h_l,
                        object_feats=node_o_l,
                    )
                    sync_visual_mask_l = candidate_out_l.evidence_mask.to(dtype=torch.bool)
                    if mechanism_variant == "endpoint_only_setm":
                        sync_visual_mask_l = sync_visual_mask_l.clone()
                        sync_visual_mask_l[..., 3:] = False
                    sync_out_l = None
                    if setm_enabled_for_variant:
                        sync_out_l = self.sync_evidence_matcher(
                            visual_tokens=candidate_out_l.evidence_tokens,
                            visual_mask=sync_visual_mask_l,
                            prototype_slots=text_out_l.prototype_slots,
                            prototype_slot_mask=text_out_l.prototype_slot_mask,
                            pair_valid_mask=layer_text_pair_valid_mask,
                        )
                    sync_anchor_delta_applied_l = torch.zeros_like(refined_edge_l)
                    candidate_anchor_delta_applied_l = torch.zeros_like(candidate_out_l.anchor_delta)
                    visual_writeback_applied_l = bool(effective_write_visual and sync_out_l is not None)
                    text_writeback_applied_l = bool(effective_write_text and sync_out_l is not None)
                    if visual_writeback_applied_l:
                        sync_anchor_delta_applied_l = torch.where(
                            layer_text_pair_valid_mask.unsqueeze(-1), sync_out_l.anchor_delta, sync_anchor_delta_applied_l)
                        updated_edge_attr = self._scatter_pair_delta_to_dense_edges(
                            edge_attr_dense=updated_edge_attr,
                            pair_indices=pair_indices,
                            pair_delta=sync_out_l.anchor_delta,
                            valid_pair_mask=layer_text_pair_valid_mask,
                            edge_mask_dense=edge_mask_dense,
                            write_reverse=self.graph_layerwise_sync_write_reverse_edge,
                        )
                    if text_writeback_applied_l:
                        write_text_mask = text_out_l.prototype_slot_mask & layer_text_pair_valid_mask[:, :, None, None]
                        text_state = self._apply_prototype_slot_update_to_text_state(
                            text_state=text_state,
                            updated_prototype_slots=sync_out_l.updated_prototype_slots,
                            prototype_slot_mask=write_text_mask,
                            detach_update=self.graph_layerwise_sync_detach_text_update,
                        )
                    if (
                        self.graph_text_export_each_layer
                        and self.graph_text_dense_injection
                        and not self.graph_layerwise_sync_disable_old_dense_injection
                    ):
                        text_context_l = self._cross_attend_text_tokens(
                            query=refined_edge_l,
                            text_tokens=text_out_l.text_tokens,
                            text_token_mask=text_out_l.text_token_mask,
                            valid_pair_mask=layer_text_pair_valid_mask,
                        )
                        text_layer_debug.append(self._build_text_layer_debug_info(
                            layer_idx=layer_idx,
                            text_out=text_out_l,
                            text_context=text_context_l,
                            refined_edge=refined_edge_l,
                        ))
                        updated_node_feats, updated_edge_attr, dense_debug_l = self._inject_text_to_visual_state(
                            updated_node_feats=updated_node_feats,
                            updated_edge_attr=updated_edge_attr,
                            edge_mask_dense=edge_mask_dense,
                            pair_indices=pair_indices,
                            text_out=text_out_l,
                            text_context=text_context_l,
                            valid_pair_mask=layer_text_pair_valid_mask,
                            layer_idx=layer_idx,
                        )
                        text_dense_injection_debug_per_layer.append(dense_debug_l)
                    if self.graph_export_evidence_trace and not self.training:
                        refined_edge_after_writeback_l, valid_pair_mask_after_writeback_l = self._gather_pair_edge_feats(
                            updated_edge_attr, pair_indices, node_mask,
                        )
                        node_h_after_writeback_l = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 0])
                        node_o_after_writeback_l = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 1])
                        graph_pair_features_after_layer_l = None
                        if mechanism_trace_enabled:
                            layer_pair_input_l = torch.cat([
                                refined_edge_after_writeback_l,
                                node_h_after_writeback_l,
                                node_o_after_writeback_l,
                                base_pair_feats,
                                union_input,
                            ], dim=-1)
                            graph_pair_features_after_layer_l = self.pair_out_mlp(layer_pair_input_l)
                            graph_pair_features_after_layer_l = torch.where(
                                valid_pair_mask_after_writeback_l.unsqueeze(-1),
                                graph_pair_features_after_layer_l,
                                base_pair_feats,
                            )
                            mechanism_layer_pair_features.append(graph_pair_features_after_layer_l)
    
                        text_out_post_writeback_l = self.text_graph_reasoner.export_layer_outputs(text_state)
                        pair_anchor_post_sync_proposed_l = (
                            refined_edge_l
                            if sync_out_l is None
                            else refined_edge_l + sync_out_l.anchor_delta
                        )
                        if self.graph_export_evidence_trace and not self.training:
                            trace_l = self._build_evidence_trace(
                                stage="layerwise",
                                layer_idx=layer_idx,
                                pair_indices=pair_indices,
                                pair_valid_mask=layer_text_pair_valid_mask,
                                candidate_out=candidate_out_l,
                                text_out=text_out_l,
                                sync_out=sync_out_l,
                                edge_mask_dense=edge_mask_dense,
                                edge_score_dense=edge_score_dense,
                                pair_anchor_pre_sync=refined_edge_l,
                                pair_anchor_post_sync=pair_anchor_post_sync_proposed_l,
                                pair_anchor_post_sync_actual=refined_edge_after_writeback_l,
                                sync_anchor_delta_applied=sync_anchor_delta_applied_l,
                                candidate_anchor_delta_applied=candidate_anchor_delta_applied_l,
                                prototype_slots_post_writeback=text_out_post_writeback_l.prototype_slots,
                                graph_pair_features_after_layer=graph_pair_features_after_layer_l,
                                current_human_node_state=node_h_after_writeback_l,
                                current_object_node_state=node_o_after_writeback_l,
                                sync_visual_mask=sync_visual_mask_l,
                                setm_called=sync_out_l is not None,
                                visual_writeback_applied=visual_writeback_applied_l,
                                text_writeback_applied=text_writeback_applied_l,
                            )
                            final_evidence_trace = trace_l
                            if self.graph_export_trace_each_layer:
                                evidence_trace_layers.append(trace_l)
                    valid_anchor_delta_source_l = (
                        sync_out_l.anchor_delta if sync_out_l is not None else torch.zeros_like(refined_edge_l)
                    )
                    valid_anchor_delta = valid_anchor_delta_source_l[layer_text_pair_valid_mask]
                    if valid_anchor_delta.numel() > 0:
                        anchor_delta_norm = valid_anchor_delta.norm(dim=-1)
                        anchor_delta_norm_mean = float(anchor_delta_norm.mean().detach().cpu().item())
                        layerwise_sync_total_anchor_delta_norm += float(anchor_delta_norm.sum().detach().cpu().item())
                    else:
                        anchor_delta_norm_mean = 0.0
                    valid_prototype_mask = text_out_l.prototype_slot_mask & layer_text_pair_valid_mask[:, :, None, None]
                    layer_sync_has_nan = False
                    if sync_out_l is not None:
                        layer_sync_has_nan = bool(
                            torch.isnan(sync_out_l.match_logits).any().detach().cpu().item()
                            or torch.isnan(sync_out_l.anchor_delta).any().detach().cpu().item()
                            or torch.isnan(sync_out_l.updated_prototype_slots).any().detach().cpu().item()
                        )
                    layerwise_sync_has_nan = layerwise_sync_has_nan or layer_sync_has_nan
                    layerwise_sync_debug.append({
                        "layer_idx": int(layer_idx),
                        "candidate_evidence_tokens_shape": tuple(candidate_out_l.evidence_tokens.shape),
                        "prototype_slots_shape": tuple(text_out_l.prototype_slots.shape),
                        "sync_match_logits_shape": None if sync_out_l is None else tuple(sync_out_l.match_logits.shape),
                        "mechanism_validation_variant": mechanism_variant,
                        "setm_called": bool(sync_out_l is not None),
                        "visual_writeback_applied": visual_writeback_applied_l,
                        "text_writeback_applied": text_writeback_applied_l,
                        "sync_visual_mask_valid_count": int(sync_visual_mask_l.sum().detach().cpu().item()),
                        "sync_valid_pair_count": int(layer_text_pair_valid_mask.sum().detach().cpu().item()),
                        "sync_valid_prototype_count": int(valid_prototype_mask.sum().detach().cpu().item()),
                        "sync_anchor_delta_norm_mean": anchor_delta_norm_mean,
                        "sync_visual_gate": float(torch.tanh(self.sync_evidence_matcher.visual_gate).detach().cpu().item()),
                        "sync_text_gate": float(torch.tanh(self.sync_evidence_matcher.text_gate).detach().cpu().item()),
                        "sync_anchor_gate": float(torch.tanh(self.sync_evidence_matcher.anchor_gate).detach().cpu().item()),
                        "sync_has_nan": layer_sync_has_nan,
                    })
                elif self.graph_text_dynamic_retrieval:
                    text_query_feats_l = self._build_text_query_feats(
                        source="refined_edge",
                        refined_edge=refined_edge_l,
                        base_pair_feats=base_pair_feats,
                        union_feats=union_input,
                    )
                    text_state = self.text_graph_reasoner.build_initial_state(
                        visual_pair_feats=text_query_feats_l,
                        pair_object_labels=pair_object_labels,
                        pair_valid_mask=layer_text_pair_valid_mask,
                    )
                    text_state = self.text_graph_reasoner.step(text_state, layer_idx)
                    if self.graph_text_export_each_layer:
                        text_out_l = self.text_graph_reasoner.export_layer_outputs(text_state)
                        text_context_l = self._cross_attend_text_tokens(
                            query=refined_edge_l,
                            text_tokens=text_out_l.text_tokens,
                            text_token_mask=text_out_l.text_token_mask,
                            valid_pair_mask=layer_text_pair_valid_mask,
                        )
                        text_layer_debug.append(self._build_text_layer_debug_info(
                            layer_idx=layer_idx,
                            text_out=text_out_l,
                            text_context=text_context_l,
                            refined_edge=refined_edge_l,
                        ))
                        if self.graph_text_dense_injection:
                            updated_node_feats, updated_edge_attr, dense_debug_l = self._inject_text_to_visual_state(
                                updated_node_feats=updated_node_feats,
                                updated_edge_attr=updated_edge_attr,
                                edge_mask_dense=edge_mask_dense,
                                pair_indices=pair_indices,
                                text_out=text_out_l,
                                text_context=text_context_l,
                                valid_pair_mask=layer_text_pair_valid_mask,
                                layer_idx=layer_idx,
                            )
                            text_dense_injection_debug_per_layer.append(dense_debug_l)
                else:
                    text_state = self.text_graph_reasoner.step(text_state, layer_idx)
                    if self.graph_text_export_each_layer:
                        text_out_l = self.text_graph_reasoner.export_layer_outputs(text_state)
                        text_context_l = self._cross_attend_text_tokens(
                            query=refined_edge_l,
                            text_tokens=text_out_l.text_tokens,
                            text_token_mask=text_out_l.text_token_mask,
                            valid_pair_mask=layer_text_pair_valid_mask,
                        )
                        text_layer_debug.append(self._build_text_layer_debug_info(
                            layer_idx=layer_idx,
                            text_out=text_out_l,
                            text_context=text_context_l,
                            refined_edge=refined_edge_l,
                        ))
                        if self.graph_text_dense_injection:
                            updated_node_feats, updated_edge_attr, dense_debug_l = self._inject_text_to_visual_state(
                                updated_node_feats=updated_node_feats,
                                updated_edge_attr=updated_edge_attr,
                                edge_mask_dense=edge_mask_dense,
                                pair_indices=pair_indices,
                                text_out=text_out_l,
                                text_context=text_context_l,
                                valid_pair_mask=layer_text_pair_valid_mask,
                                layer_idx=layer_idx,
                            )
                            text_dense_injection_debug_per_layer.append(dense_debug_l)

        refined_edge, valid_pair_mask = self._gather_pair_edge_feats(updated_edge_attr, pair_indices, node_mask)
        if text_layerwise_enabled and text_state is not None:
            text_out_for_matching = self.text_graph_reasoner.export_layer_outputs(text_state)
        node_h = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 0])
        node_o = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 1])
        text_debug_info = self._empty_text_debug_info(refined_edge)
        text_debug_info.update({
            "text_sync_mode": self.graph_text_sync_mode,
            "text_layerwise_enabled": text_layerwise_enabled,
            "text_layer_count": len(text_layer_debug),
            "text_layer_debug": text_layer_debug,
            "graph_text_dense_injection": self.graph_text_dense_injection,
            "graph_text_final_pair_injection": self.graph_text_final_pair_injection,
            "text_dynamic_retrieval": self.graph_text_dynamic_retrieval,
            "text_sync_layers_with_visual": self.graph_text_sync_layers_with_visual,
            "text_dense_injection_debug_per_layer": text_dense_injection_debug_per_layer,
            "text_dense_total_edge_update_count": sum(
                item.get("text_dense_edge_update_count", 0) + item.get("text_dense_reverse_edge_update_count", 0)
                for item in text_dense_injection_debug_per_layer
            ),
            "text_dense_total_node_update_count": sum(
                item.get("text_dense_human_node_update_count", 0) + item.get("text_dense_object_node_update_count", 0)
                for item in text_dense_injection_debug_per_layer
            ),
            "text_layerwise_final_injected": bool(text_layerwise_enabled and self.graph_text_dense_injection and len(text_dense_injection_debug_per_layer) > 0),
        })
        if self.enable_text_graph and self.graph_text_sync_mode == "posthoc":
            text_query_feats = self._build_text_query_feats(
                source=self.graph_text_query_source,
                refined_edge=refined_edge,
                base_pair_feats=base_pair_feats,
                union_feats=union_input,
            )
            pair_object_labels, pair_label_valid_mask = self._gather_pair_object_labels(
                labels=labels,
                pair_indices=pair_indices,
                node_mask=node_mask,
            )
            text_pair_valid_mask = valid_pair_mask & pair_label_valid_mask
            text_out = self.text_graph_reasoner(
                visual_pair_feats=text_query_feats,
                pair_object_labels=pair_object_labels,
                pair_valid_mask=text_pair_valid_mask,
            )
            text_out_for_matching = text_out
            text_context = self._cross_attend_text_tokens(
                query=refined_edge,
                text_tokens=text_out.text_tokens,
                text_token_mask=text_out.text_token_mask,
                valid_pair_mask=text_pair_valid_mask,
            )
            text_injection = torch.tanh(self.text_gate).to(dtype=text_context.dtype) * text_context
            text_injection = text_injection * valid_pair_mask.unsqueeze(-1).to(dtype=text_injection.dtype)
            refined_edge = refined_edge + text_injection
            node_h, node_o, refined_edge, text_prior_debug_info = self._inject_text_priors(
                text_out=text_out,
                node_h=node_h,
                node_o=node_o,
                refined_edge=refined_edge,
                valid_pair_mask=valid_pair_mask,
            )
            text_debug_info = self._build_text_debug_info(
                text_out=text_out,
                text_context=text_context,
                text_injection=text_injection,
                text_pair_valid_mask=text_pair_valid_mask,
                text_prior_debug_info=text_prior_debug_info,
            )
            text_debug_info.update({
                "text_sync_mode": self.graph_text_sync_mode,
                "text_layerwise_enabled": False,
                "text_layer_count": 0,
                "text_layer_debug": [],
                "graph_text_dense_injection": self.graph_text_dense_injection,
                "graph_text_final_pair_injection": self.graph_text_final_pair_injection,
                "text_dynamic_retrieval": False,
                "text_sync_layers_with_visual": self.graph_text_sync_layers_with_visual,
                "text_dense_injection_debug_per_layer": [],
                "text_dense_total_edge_update_count": 0,
                "text_dense_total_node_update_count": 0,
                "text_layerwise_final_injected": False,
            })

        candidate_out = None
        candidate_evidence_debug_info = {
            "candidate_evidence_enabled": bool(self.graph_enable_candidate_evidence),
            "candidate_evidence_applied": False,
        }
        if layerwise_skip_final_matching:
            candidate_evidence_debug_info["candidate_evidence_skip_reason"] = "layerwise_sync_final_matching_disabled"
            if setm_enabled_for_variant:
                sync_evidence_debug_info["sync_evidence_skip_reason"] = "layerwise_sync_final_matching_disabled"
        if self.graph_enable_candidate_evidence and self.candidate_evidence_inducer is not None and not layerwise_skip_final_matching:
            candidate_out = self.candidate_evidence_inducer(
                node_feats=updated_node_feats,
                edge_attr_dense=updated_edge_attr,
                edge_mask_dense=edge_mask_dense,
                node_mask=node_mask,
                pair_indices=pair_indices,
                pair_valid_mask=valid_pair_mask,
                anchor_feats=refined_edge,
                human_feats=node_h,
                object_feats=node_o,
            )
            refined_edge = refined_edge + candidate_out.anchor_delta
            candidate_evidence_debug_info.update(candidate_out.debug_info)
            candidate_evidence_debug_info["candidate_evidence_applied"] = True

        if text_out_for_matching is not None and hasattr(text_out_for_matching, "debug_info"):
            sync_evidence_debug_info["text_prototype_slots_shape"] = text_out_for_matching.debug_info.get("prototype_slots_shape", None)
            sync_evidence_debug_info["text_prototype_slot_valid_ratio"] = text_out_for_matching.debug_info.get("prototype_slot_valid_ratio", 0.0)

        sync_out = None
        if self.graph_enable_sync_evidence_matching and self.sync_evidence_matcher is not None and not layerwise_skip_final_matching:
            if candidate_out is None:
                sync_evidence_debug_info["sync_evidence_skip_reason"] = "candidate_evidence_output_missing"
            elif text_out_for_matching is None:
                sync_evidence_debug_info["sync_evidence_skip_reason"] = "text_output_missing"
            elif not hasattr(text_out_for_matching, "prototype_slots") or not hasattr(text_out_for_matching, "prototype_slot_mask"):
                sync_evidence_debug_info["sync_evidence_skip_reason"] = "prototype_slots_missing"
            else:
                sync_out = self.sync_evidence_matcher(
                    visual_tokens=candidate_out.evidence_tokens,
                    visual_mask=candidate_out.evidence_mask,
                    prototype_slots=text_out_for_matching.prototype_slots,
                    prototype_slot_mask=text_out_for_matching.prototype_slot_mask,
                    pair_valid_mask=valid_pair_mask,
                )
                refined_edge = refined_edge + sync_out.anchor_delta
                sync_evidence_debug_info.update(sync_out.debug_info)
                sync_evidence_debug_info["sync_evidence_applied"] = True
                sync_evidence_debug_info.pop("sync_evidence_skip_reason", None)

        if (
            self.graph_export_evidence_trace
            and not self.training
            and not layerwise_skip_final_matching
            and candidate_out is not None
        ):
            final_evidence_trace = self._build_evidence_trace(
                stage="final",
                layer_idx=max(len(self.layers) - 1, 0),
                pair_indices=pair_indices,
                pair_valid_mask=valid_pair_mask,
                candidate_out=candidate_out,
                text_out=text_out_for_matching,
                sync_out=sync_out,
                edge_mask_dense=edge_mask_dense,
                edge_score_dense=edge_score_dense,
            )
            if self.graph_export_trace_each_layer:
                evidence_trace_layers.append(final_evidence_trace)

        pair_out_input = torch.cat([refined_edge, node_h, node_o, base_pair_feats, union_input], dim=-1)
        pair_out = self.pair_out_mlp(pair_out_input)
        fallback_pair_feats = base_pair_feats if pair_feats is not None else torch.zeros_like(base_pair_feats)
        pair_out = torch.where(valid_pair_mask.unsqueeze(-1), pair_out, fallback_pair_feats)

        has_nan = bool(
            torch.isnan(pair_out).any().item()
            or torch.isnan(updated_node_feats).any().item()
            or torch.isnan(updated_edge_attr).any().item()
        )
        debug_info = {
            "pair_source": pair_source,
            "node_feats_shape": tuple(node_feats.shape),
            "node_blip_feats_shape": None if node_blip_feats is None else tuple(node_blip_feats.shape),
            "fused_node_feats_shape": tuple(fused_node_feats.shape),
            "updated_node_feats_shape": tuple(updated_node_feats.shape),
            "pair_indices_shape": tuple(pair_indices.shape),
            "pair_feats_shape": tuple(pair_out.shape),
            "pair_edge_feats_shape": tuple(refined_edge.shape),
            "edge_attr_dense_shape": tuple(edge_attr_dense.shape),
            "updated_edge_attr_dense_shape": tuple(updated_edge_attr.shape),
            "edge_mask_dense_shape": tuple(edge_mask_dense.shape),
            "edge_score_dense_shape": tuple(edge_score_dense.shape),
            "e_ho_edge_count": counts["e_ho_edge_count"],
            "e_oo_edge_count": counts["e_oo_edge_count"],
            "e_hh_edge_count": counts["e_hh_edge_count"],
            "edge_count_per_batch": counts["edge_count_per_batch"],
            "e_ho_edge_count_per_batch": counts["e_ho_edge_count_per_batch"],
            "e_oo_edge_count_per_batch": counts["e_oo_edge_count_per_batch"],
            "e_hh_edge_count_per_batch": counts["e_hh_edge_count_per_batch"],
            "oo_mode": counts["oo_mode"],
            "oo_fallback_per_batch": counts["oo_fallback_per_batch"],
            "oo_geo_mean_per_batch": counts["oo_geo_mean_per_batch"],
            "oo_geo_max_per_batch": counts["oo_geo_max_per_batch"],
            "oo_share_mean_per_batch": counts["oo_share_mean_per_batch"],
            "oo_share_max_per_batch": counts["oo_share_max_per_batch"],
            "oo_text_used_per_batch": counts["oo_text_used_per_batch"],
            "oo_text_inactive_per_batch": counts["oo_text_inactive_per_batch"],
            "oo_text_mean_per_batch": counts["oo_text_mean_per_batch"],
            "oo_text_max_per_batch": counts["oo_text_max_per_batch"],
            "oo_text_nonzero_ratio_per_batch": counts["oo_text_nonzero_ratio_per_batch"],
            "oo_text_invalid_label_count_per_batch": counts["oo_text_invalid_label_count_per_batch"],
            "oo_score_mean_per_batch": counts["oo_score_mean_per_batch"],
            "oo_score_max_per_batch": counts["oo_score_max_per_batch"],
            "oo_topk_count_per_batch": counts["oo_topk_count_per_batch"],
            "oo_geo_mean": counts["oo_geo_mean"],
            "oo_geo_max": counts["oo_geo_max"],
            "oo_share_mean": counts["oo_share_mean"],
            "oo_share_max": counts["oo_share_max"],
            "oo_text_mean": counts["oo_text_mean"],
            "oo_text_max": counts["oo_text_max"],
            "oo_score_mean": counts["oo_score_mean"],
            "oo_score_max": counts["oo_score_max"],
            "valid_pair_count_per_batch": valid_pair_mask.sum(dim=1).detach().cpu().tolist(),
            "has_union_feats": union_feats is not None,
            "has_global_feat": global_feat is not None,
            "enable_oo_edges": self.enable_oo_edges,
            "enable_hh_edges": self.enable_hh_edges,
            **text_debug_info,
            **candidate_evidence_debug_info,
            **sync_evidence_debug_info,
            "mechanism_validation_variant": mechanism_variant,
            "mechanism_trace_enabled": mechanism_trace_enabled,
            "effective_write_visual": effective_write_visual,
            "effective_write_text": effective_write_text,
            "setm_enabled_for_variant": setm_enabled_for_variant,
            "setm_call_count": sum(int(item.get("setm_called", False)) for item in layerwise_sync_debug),
            "layerwise_sync_enabled": layerwise_sync_enabled,
            "layerwise_sync_applied": bool(any(item.get("setm_called", False) for item in layerwise_sync_debug)),
            "layerwise_sync_layer_count": len(layerwise_sync_debug),
            "layerwise_sync_debug": layerwise_sync_debug,
            "layerwise_sync_total_anchor_delta_norm": layerwise_sync_total_anchor_delta_norm,
            "layerwise_sync_has_nan": layerwise_sync_has_nan,
            "message_passing_enabled": True,
            "layer_debug": layer_debug,
            "has_nan": has_nan,
        }
        if self.graph_export_evidence_trace and not self.training:
            debug_info["evidence_trace_enabled"] = True
            debug_info["evidence_trace"] = final_evidence_trace
            if self.graph_export_trace_each_layer:
                debug_info["evidence_trace_layers"] = evidence_trace_layers
        if mechanism_trace_enabled:
            debug_info["mechanism_layer_pair_features"] = mechanism_layer_pair_features
            if self.candidate_evidence_inducer is not None:
                candidate_gate_raw = self.candidate_evidence_inducer.anchor_gate
                debug_info["candidate_evidence_gate_raw"] = candidate_gate_raw.detach()
                debug_info["candidate_evidence_gate_tanh"] = torch.tanh(candidate_gate_raw).detach()
            if self.sync_evidence_matcher is not None:
                visual_gate_raw = self.sync_evidence_matcher.visual_gate
                text_gate_raw = self.sync_evidence_matcher.text_gate
                anchor_gate_raw = self.sync_evidence_matcher.anchor_gate
                debug_info["setm_visual_gate_raw"] = visual_gate_raw.detach()
                debug_info["setm_visual_gate_tanh"] = torch.tanh(visual_gate_raw).detach()
                debug_info["setm_text_gate_raw"] = text_gate_raw.detach()
                debug_info["setm_text_gate_tanh"] = torch.tanh(text_gate_raw).detach()
                debug_info["setm_anchor_gate_raw"] = anchor_gate_raw.detach()
                debug_info["setm_anchor_gate_tanh"] = torch.tanh(anchor_gate_raw).detach()
        if self.graph_store_dense_debug_tensors:
            debug_info.update({
                "fused_node_feats": fused_node_feats.detach(),
                "updated_node_feats": updated_node_feats.detach(),
                "edge_attr_dense": edge_attr_dense.detach(),
                "updated_edge_attr_dense": updated_edge_attr.detach(),
                "edge_mask_dense": edge_mask_dense.detach(),
                "edge_score_dense": edge_score_dense.detach(),
                "final_pair_feats": pair_out.detach(),
                "refined_pair_edge_feats": refined_edge.detach(),
            })
        return GraphReasonerOutput(
            pair_feats=pair_out,
            node_feats=updated_node_feats,
            debug_info=debug_info,
        )


def build_visual_scene_graph_reasoner(args=None, **kwargs) -> VisualSceneGraphReasoner:
    if args is None:
        return VisualSceneGraphReasoner(**kwargs)

    enable_text_graph = getattr(args, "enable_text_graph", getattr(args, "graph_enable_text_graph", False))
    required_text_paths = {
        "graph_text_features_path": getattr(args, "text_features_path", getattr(args, "graph_text_features_path", "")),
        "graph_text_mask_path": getattr(args, "text_mask_path", getattr(args, "graph_text_mask_path", "")),
        "graph_text_hoi_ids_path": getattr(args, "hoi_ids_path", getattr(args, "graph_text_hoi_ids_path", "")),
        "graph_text_object_to_hoi_indices_path": getattr(
            args,
            "object_to_hoi_indices_path",
            getattr(args, "graph_text_object_to_hoi_indices_path", ""),
        ),
        "graph_text_field_order_path": getattr(args, "field_order_path", getattr(args, "graph_text_field_order_path", "")),
    }
    if enable_text_graph:
        for name, path in required_text_paths.items():
            if not path:
                raise FileNotFoundError(f"{name} must be provided when graph_enable_text_graph=True")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name} file not found: {path}")

    graph_num_layers = getattr(args, "graph_num_layers", 2)
    graph_text_sync_mode = getattr(args, "graph_text_sync_mode", "layerwise")
    graph_text_sync_layers_with_visual = getattr(args, "graph_text_sync_layers_with_visual", True)
    graph_text_num_layers_arg = getattr(args, "graph_text_num_layers", None)
    explicit_text_num_layers = graph_text_num_layers_arg is not None
    if explicit_text_num_layers:
        graph_text_num_layers = graph_text_num_layers_arg
    elif graph_text_sync_layers_with_visual:
        graph_text_num_layers = graph_num_layers
    else:
        graph_text_num_layers = 0
    graph_text_query_source_default = "base_pair" if graph_text_sync_mode == "layerwise" else "refined_edge"
    graph_text_prior_default = "edge_only" if enable_text_graph and graph_text_sync_mode == "layerwise" else "none"
    graph_text_query_source = getattr(args, "graph_text_query_source", None)
    if graph_text_query_source is None:
        graph_text_query_source = graph_text_query_source_default
    graph_text_prior_injection = getattr(args, "graph_text_prior_injection", None)
    if graph_text_prior_injection is None:
        graph_text_prior_injection = graph_text_prior_default

    config = {
        "d_model": getattr(args, "hidden_dim", getattr(args, "d_model", 256)),
        "num_obj_classes": getattr(args, "num_classes", 80),
        "num_edge_types": getattr(args, "graph_num_edge_types", 4),
        "num_node_types": getattr(args, "graph_num_node_types", 3),
        "num_layers": graph_num_layers,
        "num_heads": getattr(args, "graph_num_heads", 8),
        "k_oo": getattr(args, "graph_k_oo", 3),
        "enable_oo_edges": getattr(args, "graph_enable_oo_edges", True),
        "enable_hh_edges": getattr(args, "graph_enable_hh_edges", False),
        "dropout": getattr(args, "graph_dropout", getattr(args, "dropout", 0.1)),
        "spatial_sigma": getattr(args, "graph_spatial_sigma", 0.5),
        "graph_oo_mode": getattr(args, "graph_oo_mode", "spatial"),
        "graph_oo_lambda_geo": getattr(args, "graph_oo_lambda_geo", 1.0),
        "graph_oo_lambda_share": getattr(args, "graph_oo_lambda_share", 0.5),
        "graph_oo_lambda_text": getattr(args, "graph_oo_lambda_text", 0.0),
        "graph_oo_min_score": getattr(args, "graph_oo_min_score", 0.0),
        "graph_oo_fallback_to_spatial": getattr(args, "graph_oo_fallback_to_spatial", True),
        "graph_oo_share_eps": getattr(args, "graph_oo_share_eps", 1e-6),
        "graph_oo_prior_path": getattr(args, "graph_oo_prior_path", ""),
        "graph_oo_prior_normalize": getattr(args, "graph_oo_prior_normalize", "none"),
        "graph_oo_prior_symmetric": getattr(args, "graph_oo_prior_symmetric", True),
        "graph_oo_prior_eps": getattr(args, "graph_oo_prior_eps", 1e-6),
        "graph_oo_prior_label_ids": getattr(args, "graph_oo_prior_label_ids", getattr(args, "valid_obj_ids", None)),
        "enable_text_graph": enable_text_graph,
        "text_features_path": getattr(args, "text_features_path", getattr(args, "graph_text_features_path", "")),
        "text_mask_path": getattr(args, "text_mask_path", getattr(args, "graph_text_mask_path", "")),
        "hoi_ids_path": getattr(args, "hoi_ids_path", getattr(args, "graph_text_hoi_ids_path", "")),
        "object_to_hoi_indices_path": getattr(
            args,
            "object_to_hoi_indices_path",
            getattr(args, "graph_text_object_to_hoi_indices_path", ""),
        ),
        "field_order_path": getattr(args, "field_order_path", getattr(args, "graph_text_field_order_path", "")),
        "graph_text_human_node_features_path": getattr(args, "graph_text_human_node_features_path", ""),
        "graph_text_object_node_features_path": getattr(args, "graph_text_object_node_features_path", ""),
        "graph_text_ho_edge_features_path": getattr(args, "graph_text_ho_edge_features_path", ""),
        "graph_text_discriminative_features_path": getattr(args, "graph_text_discriminative_features_path", ""),
        "graph_text_scene_context_features_path": getattr(args, "graph_text_scene_context_features_path", ""),
        "graph_text_oo_context_features_path": getattr(args, "graph_text_oo_context_features_path", ""),
        "oo_affordance_prior_path": getattr(args, "oo_affordance_prior_path", getattr(args, "graph_text_oo_affordance_prior_path", "")),
        "graph_text_topk": getattr(args, "graph_text_topk", 5),
        "graph_text_pre_topk": getattr(args, "graph_text_pre_topk", 20),
        "graph_text_temperature": getattr(args, "graph_text_temperature", 0.07),
        "graph_text_rerank_weight": getattr(args, "graph_text_rerank_weight", 0.5),
        "graph_text_use_discriminative_rerank": getattr(args, "graph_text_use_discriminative_rerank", True),
        "graph_text_label_ids": getattr(args, "graph_text_label_ids", getattr(args, "valid_obj_ids", None)),
        "graph_text_num_layers": graph_text_num_layers,
        "graph_text_num_heads": getattr(args, "graph_text_num_heads", 8),
        "graph_text_dropout": getattr(args, "graph_text_dropout", 0.1),
        "graph_text_use_scene_token": getattr(args, "graph_text_use_scene_token", True),
        "graph_text_use_oo_context_token": getattr(args, "graph_text_use_oo_context_token", True),
        "graph_text_cross_num_heads": getattr(args, "graph_text_cross_num_heads", 8),
        "graph_text_query_source": graph_text_query_source,
        "graph_text_sync_mode": graph_text_sync_mode,
        "graph_text_dense_injection": getattr(args, "graph_text_dense_injection", True),
        "graph_text_final_pair_injection": getattr(args, "graph_text_final_pair_injection", False),
        "graph_text_sync_query_source": getattr(args, "graph_text_sync_query_source", "base_pair"),
        "graph_text_export_each_layer": getattr(args, "graph_text_export_each_layer", True),
        "graph_text_dynamic_retrieval": getattr(args, "graph_text_dynamic_retrieval", False),
        "graph_text_sync_layers_with_visual": graph_text_sync_layers_with_visual,
        "graph_text_edge_inject_reverse": getattr(args, "graph_text_edge_inject_reverse", True),
        "graph_text_node_scatter_reduce": getattr(args, "graph_text_node_scatter_reduce", "mean"),
        "graph_text_gate_init": getattr(args, "graph_text_gate_init", 0.0),
        "graph_text_debug": getattr(args, "graph_text_debug", getattr(args, "graph_debug", False)),
        "graph_text_prior_injection": graph_text_prior_injection,
        "graph_text_use_human_node_prior": getattr(args, "graph_text_use_human_node_prior", False),
        "graph_text_use_object_node_prior": getattr(args, "graph_text_use_object_node_prior", False),
        "graph_text_use_ho_edge_prior": getattr(args, "graph_text_use_ho_edge_prior", True),
        "graph_text_use_discriminative_prior": getattr(args, "graph_text_use_discriminative_prior", False),
        "graph_text_use_scene_prior": getattr(args, "graph_text_use_scene_prior", False),
        "graph_text_use_oo_prior": getattr(args, "graph_text_use_oo_prior", False),
        "graph_text_node_prior_gate_init": getattr(args, "graph_text_node_prior_gate_init", 0.0),
        "graph_text_edge_prior_gate_init": getattr(args, "graph_text_edge_prior_gate_init", 0.0),
        "graph_text_scene_prior_gate_init": getattr(args, "graph_text_scene_prior_gate_init", 0.0),
        "graph_text_oo_prior_gate_init": getattr(args, "graph_text_oo_prior_gate_init", 0.0),
        "graph_enable_candidate_evidence": getattr(args, "graph_enable_candidate_evidence", False),
        "graph_evidence_topk": getattr(args, "graph_evidence_topk", 4),
        "graph_evidence_gate_init": getattr(args, "graph_evidence_gate_init", 0.0),
        "graph_evidence_dropout": getattr(args, "graph_evidence_dropout", None),
        "graph_evidence_allow_node_only_context": getattr(args, "graph_evidence_allow_node_only_context", False),
        "graph_evidence_debug": getattr(args, "graph_evidence_debug", False),
        "graph_enable_sync_evidence_matching": getattr(args, "graph_enable_sync_evidence_matching", False),
        "graph_sync_visual_gate_init": getattr(args, "graph_sync_visual_gate_init", 0.0),
        "graph_sync_text_gate_init": getattr(args, "graph_sync_text_gate_init", 0.0),
        "graph_sync_anchor_gate_init": getattr(args, "graph_sync_anchor_gate_init", 0.0),
        "graph_sync_dropout": getattr(args, "graph_sync_dropout", None),
        "graph_sync_debug": getattr(args, "graph_sync_debug", getattr(args, "graph_debug", False)),
        "graph_text_force_oov3": getattr(args, "graph_text_force_oov3", True),
        "graph_text_prototype_slot_fallback_to_raw": getattr(args, "graph_text_prototype_slot_fallback_to_raw", True),
        "graph_text_prototype_slot_debug": getattr(args, "graph_text_prototype_slot_debug", getattr(args, "graph_debug", False)),
        "graph_enable_layerwise_sync_evidence_matching": getattr(args, "graph_enable_layerwise_sync_evidence_matching", False),
        "graph_layerwise_sync_write_visual": getattr(args, "graph_layerwise_sync_write_visual", True),
        "graph_layerwise_sync_write_text": getattr(args, "graph_layerwise_sync_write_text", True),
        "graph_layerwise_sync_final_matching": getattr(args, "graph_layerwise_sync_final_matching", False),
        "graph_layerwise_sync_disable_old_dense_injection": getattr(args, "graph_layerwise_sync_disable_old_dense_injection", True),
        "graph_layerwise_sync_write_reverse_edge": getattr(args, "graph_layerwise_sync_write_reverse_edge", False),
        "graph_layerwise_sync_detach_text_update": getattr(args, "graph_layerwise_sync_detach_text_update", False),
        "graph_layerwise_sync_debug": getattr(args, "graph_layerwise_sync_debug", False),
        "graph_store_dense_debug_tensors": getattr(args, "graph_store_dense_debug_tensors", False),
        "graph_export_evidence_trace": getattr(args, "graph_export_evidence_trace", False),
        "graph_export_trace_each_layer": getattr(args, "graph_export_trace_each_layer", False),
        "graph_export_trace_to_cpu": getattr(args, "graph_export_trace_to_cpu", True),
        "graph_export_trace_float_dtype": getattr(args, "graph_export_trace_float_dtype", "float16"),
        "graph_export_trace_include_match_logits": getattr(args, "graph_export_trace_include_match_logits", False),
        "graph_export_trace_include_features": getattr(args, "graph_export_trace_include_features", False),
        "graph_export_mechanism_validation_trace": getattr(args, "graph_export_mechanism_validation_trace", False),
        "graph_mechanism_validation_variant": getattr(args, "graph_mechanism_validation_variant", "full"),
    }
    config.update(kwargs)
    return VisualSceneGraphReasoner(**config)
