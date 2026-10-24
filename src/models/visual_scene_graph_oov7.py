"""Visual scene graph oov7 with final-layer synchronized evidence matching."""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Optional

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
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.graph_enable_sync_evidence_matching = bool(graph_enable_sync_evidence_matching)
        self.graph_sync_debug = bool(graph_sync_debug)
        self.graph_text_force_oov3 = bool(graph_text_force_oov3)
        self.graph_text_prototype_slot_fallback_to_raw = bool(graph_text_prototype_slot_fallback_to_raw)
        self.graph_text_prototype_slot_debug = bool(graph_text_prototype_slot_debug)

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

        self.sync_evidence_matcher = None
        if self.graph_enable_sync_evidence_matching:
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

        text_out_for_matching = None
        sync_evidence_debug_info = {
            "sync_evidence_enabled": bool(self.graph_enable_sync_evidence_matching),
            "sync_evidence_applied": False,
            "sync_evidence_skip_reason": "disabled" if not self.graph_enable_sync_evidence_matching else "not_attempted",
        }

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
                if self.graph_text_dynamic_retrieval:
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
        if self.graph_enable_candidate_evidence and self.candidate_evidence_inducer is not None:
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

        if self.graph_enable_sync_evidence_matching and self.sync_evidence_matcher is not None:
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
            "message_passing_enabled": True,
            "layer_debug": layer_debug,
            "has_nan": has_nan,
            "edge_attr_dense": edge_attr_dense.detach(),
            "updated_edge_attr_dense": updated_edge_attr.detach(),
            "edge_mask_dense": edge_mask_dense.detach(),
            "edge_score_dense": edge_score_dense.detach(),
        }
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
    }
    config.update(kwargs)
    return VisualSceneGraphReasoner(**config)
