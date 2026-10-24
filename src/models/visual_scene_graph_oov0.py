import math
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn


def _build_text_graph_pair_prior(**kwargs):
    try:
        from .text_scene_graph import TextGraphPairPrior
    except ImportError:
        module_path = Path(__file__).resolve().with_name("text_scene_graph.py")
        spec = importlib.util.spec_from_file_location("text_scene_graph_direct", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        TextGraphPairPrior = module.TextGraphPairPrior
    return TextGraphPairPrior(**kwargs)


@dataclass
class GraphReasonerOutput:
    pair_feats: Tensor
    node_feats: Optional[Tensor]
    debug_info: Dict[str, Any]


class EdgeAwareGraphLayer(nn.Module):
    """Lightweight edge-aware graph transformer layer.

    Node update uses dense edge-masked multi-head attention with an additive edge
    bias. Edge update uses endpoint node features and only updates existing edges.
    """

    def __init__(self, d_model: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Linear(d_model, num_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.node_dropout = nn.Dropout(dropout)
        self.node_norm1 = nn.LayerNorm(d_model)
        self.node_norm2 = nn.LayerNorm(d_model)
        self.node_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

        self.edge_ffn = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.edge_res_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        node_feats: Tensor,
        edge_attr: Tensor,
        edge_mask: Tensor,
        node_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
        batch_size, num_nodes, dim = node_feats.shape
        valid_node_mask = node_mask.to(dtype=torch.bool)

        eye = torch.eye(num_nodes, dtype=torch.bool, device=node_feats.device).unsqueeze(0)
        self_loop_mask = eye & valid_node_mask.unsqueeze(1) & valid_node_mask.unsqueeze(2)
        attn_allowed = (edge_mask | self_loop_mask) & valid_node_mask.unsqueeze(1) & valid_node_mask.unsqueeze(2)

        q = self.q_proj(node_feats).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(node_feats).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(node_feats).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        edge_bias = self.edge_bias(edge_attr).permute(0, 3, 1, 2)
        logits = logits + edge_bias
        logits = logits.masked_fill(~attn_allowed.unsqueeze(1), -1e4)
        attn = torch.softmax(logits, dim=-1)
        attn = attn.masked_fill(~attn_allowed.unsqueeze(1), 0.0)
        attn = self.attn_dropout(attn)

        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(batch_size, num_nodes, dim)
        attn_out = self.out_proj(attn_out)
        node_new = self.node_norm1(node_feats + self.node_dropout(attn_out))
        node_new = self.node_norm2(node_new + self.node_dropout(self.node_ffn(node_new)))
        node_new = node_new * valid_node_mask.unsqueeze(-1).to(dtype=node_new.dtype)

        src_nodes = node_new.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, dim)
        dst_nodes = node_new.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, dim)
        edge_update_input = torch.cat([edge_attr, src_nodes, dst_nodes, src_nodes * dst_nodes], dim=-1)
        edge_delta = self.edge_ffn(edge_update_input)
        edge_new = edge_attr + self.edge_res_scale * edge_delta
        edge_new = edge_new * edge_mask.unsqueeze(-1).to(dtype=edge_new.dtype)

        debug = {
            "node_in_shape": tuple(node_feats.shape),
            "node_out_shape": tuple(node_new.shape),
            "edge_in_shape": tuple(edge_attr.shape),
            "edge_out_shape": tuple(edge_new.shape),
            "attention_mask_shape": tuple(attn_allowed.shape),
            "has_nan": bool(torch.isnan(node_new).any().item() or torch.isnan(edge_new).any().item()),
        }
        return node_new, edge_new, debug


class VisualSceneGraphReasoner(nn.Module):
    """Visual scene graph reasoner with E_ho/E_oo and edge-aware updates."""

    EDGE_HUMAN_TO_OBJECT = 0
    EDGE_OBJECT_TO_HUMAN = 1
    EDGE_OBJECT_OBJECT = 2
    EDGE_HUMAN_HUMAN = 3
    GEO_DIM = 9
    BOX_DIM = 4

    def __init__(
        self,
        d_model: int = 256,
        num_obj_classes: int = 80,
        num_edge_types: int = 4,
        num_node_types: int = 3,
        num_layers: int = 2,
        num_heads: int = 8,
        k_oo: int = 3,
        enable_oo_edges: bool = True,
        enable_hh_edges: bool = False,
        dropout: float = 0.1,
        spatial_sigma: float = 0.5,
        graph_enable_text_graph: bool = False,
        graph_text_features_path: str = "",
        graph_text_mask_path: str = "",
        graph_text_hoi_ids_path: str = "",
        graph_text_object_to_hoi_indices_path: str = "",
        graph_text_field_order_path: str = "",
        graph_text_field_names: Optional[List[str]] = None,
        graph_text_max_candidates: int = 0,
        graph_text_gate_init: float = 0.0,
        graph_text_normalize: bool = True,
        graph_text_strict_shape: bool = True,
        graph_text_strict_consistency: bool = False,
        graph_text_human_node_features_path: str = "",
        graph_text_object_node_features_path: str = "",
        graph_text_ho_edge_features_path: str = "",
        graph_text_discriminative_features_path: str = "",
        graph_text_scene_context_features_path: str = "",
        graph_text_oo_context_features_path: str = "",
        graph_text_oo_affordance_prior_path: str = "",
        graph_text_label_ids: Optional[List[int]] = None,
        graph_debug: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_obj_classes = num_obj_classes
        self.num_edge_types = num_edge_types
        self.num_node_types = num_node_types
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.k_oo = k_oo
        self.enable_oo_edges = enable_oo_edges
        self.enable_hh_edges = enable_hh_edges
        self.spatial_sigma = spatial_sigma
        self.graph_enable_text_graph = bool(graph_enable_text_graph)
        self.graph_debug = bool(graph_debug)

        self.class_embedding = nn.Embedding(num_obj_classes + 1, d_model)
        self.node_type_embedding = nn.Embedding(num_node_types, d_model)
        self.box_mlp = nn.Sequential(
            nn.Linear(self.BOX_DIM, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.node_fusion = nn.Sequential(
            nn.LayerNorm(d_model * 5),
            nn.Linear(d_model * 5, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.endpoint_pair_proj = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.geo_mlp = nn.Sequential(
            nn.Linear(self.GEO_DIM, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.edge_type_embedding = nn.Embedding(num_edge_types, d_model)
        self.edge_mlp = nn.Sequential(
            nn.Linear(d_model * 5 + 1, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList([
            EdgeAwareGraphLayer(d_model=d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.pair_out_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 5),
            nn.Linear(d_model * 5, d_model * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.text_graph_prior = None
        if self.graph_enable_text_graph:
            self.text_graph_prior = _build_text_graph_pair_prior(
                text_features_path=graph_text_features_path,
                text_mask_path=graph_text_mask_path,
                hoi_ids_path=graph_text_hoi_ids_path,
                object_to_hoi_indices_path=graph_text_object_to_hoi_indices_path,
                field_order_path=graph_text_field_order_path,
                d_model=d_model,
                field_names=graph_text_field_names,
                max_candidates=graph_text_max_candidates,
                gate_init=graph_text_gate_init,
                normalize=graph_text_normalize,
                strict_shape=graph_text_strict_shape,
                strict_consistency=graph_text_strict_consistency,
                human_node_features_path=graph_text_human_node_features_path,
                object_node_features_path=graph_text_object_node_features_path,
                ho_edge_features_path=graph_text_ho_edge_features_path,
                discriminative_features_path=graph_text_discriminative_features_path,
                scene_context_features_path=graph_text_scene_context_features_path,
                oo_context_features_path=graph_text_oo_context_features_path,
                oo_affordance_prior_path=graph_text_oo_affordance_prior_path,
                label_ids=graph_text_label_ids,
                debug=graph_debug,
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

        refined_edge, valid_pair_mask = self._gather_pair_edge_feats(updated_edge_attr, pair_indices, node_mask)
        node_h = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 0])
        node_o = self._gather_pair_nodes(updated_node_feats, pair_indices[..., 1])
        union_input = union_feats if union_feats is not None else torch.zeros_like(base_pair_feats)
        pair_out_input = torch.cat([refined_edge, node_h, node_o, base_pair_feats, union_input], dim=-1)
        pair_out = self.pair_out_mlp(pair_out_input)
        fallback_pair_feats = base_pair_feats if pair_feats is not None else torch.zeros_like(base_pair_feats)
        pair_out = torch.where(valid_pair_mask.unsqueeze(-1), pair_out, fallback_pair_feats)
        text_graph_debug = {
            "enabled": False,
            "applied": False,
        }
        if self.graph_enable_text_graph and self.text_graph_prior is not None:
            text_delta, text_graph_debug = self.text_graph_prior(
                pair_indices=pair_indices,
                labels=labels,
                pair_mask=valid_pair_mask,
            )
            text_delta = text_delta.to(device=pair_out.device, dtype=pair_out.dtype)
            pair_out = torch.where(valid_pair_mask.unsqueeze(-1), pair_out + text_delta, pair_out)
            text_graph_debug["applied"] = True

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
            "valid_pair_count_per_batch": valid_pair_mask.sum(dim=1).detach().cpu().tolist(),
            "has_union_feats": union_feats is not None,
            "has_global_feat": global_feat is not None,
            "enable_oo_edges": self.enable_oo_edges,
            "enable_hh_edges": self.enable_hh_edges,
            "graph_enable_text_graph": self.graph_enable_text_graph,
            "text_graph_debug": text_graph_debug,
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

    def _build_dense_edges(
        self,
        node_feats: Tensor,
        boxes: Tensor,
        labels: Tensor,
        node_types: Tensor,
        node_mask: Tensor,
        pair_indices: Tensor,
        pair_scores: Tensor,
        pair_feats: Tensor,
        union_feats: Optional[Tensor],
    ):
        batch_size, num_nodes, _ = node_feats.shape
        edge_attr_dense = node_feats.new_zeros((batch_size, num_nodes, num_nodes, self.d_model))
        edge_mask_dense = torch.zeros((batch_size, num_nodes, num_nodes), dtype=torch.bool, device=node_feats.device)
        edge_score_dense = node_feats.new_zeros((batch_size, num_nodes, num_nodes))

        e_ho_counts = []
        e_oo_counts = []
        e_hh_counts = []
        edge_counts = []

        for batch_idx in range(batch_size):
            e_ho = self._add_ho_edges(
                batch_idx=batch_idx,
                boxes=boxes,
                labels=labels,
                node_mask=node_mask,
                pair_indices=pair_indices,
                pair_scores=pair_scores,
                pair_feats=pair_feats,
                union_feats=union_feats,
                edge_attr_dense=edge_attr_dense,
                edge_mask_dense=edge_mask_dense,
                edge_score_dense=edge_score_dense,
            )
            e_oo = 0
            if self.enable_oo_edges:
                e_oo = self._add_spatial_edges(
                    batch_idx=batch_idx,
                    boxes=boxes,
                    labels=labels,
                    node_types=node_types,
                    node_mask=node_mask,
                    source_type=1,
                    edge_type=self.EDGE_OBJECT_OBJECT,
                    edge_attr_dense=edge_attr_dense,
                    edge_mask_dense=edge_mask_dense,
                    edge_score_dense=edge_score_dense,
                )
            e_hh = 0
            if self.enable_hh_edges:
                e_hh = self._add_spatial_edges(
                    batch_idx=batch_idx,
                    boxes=boxes,
                    labels=labels,
                    node_types=node_types,
                    node_mask=node_mask,
                    source_type=0,
                    edge_type=self.EDGE_HUMAN_HUMAN,
                    edge_attr_dense=edge_attr_dense,
                    edge_mask_dense=edge_mask_dense,
                    edge_score_dense=edge_score_dense,
                )
            e_ho_counts.append(e_ho)
            e_oo_counts.append(e_oo)
            e_hh_counts.append(e_hh)
            edge_counts.append(int(edge_mask_dense[batch_idx].sum().item()))

        counts = {
            "e_ho_edge_count": int(sum(e_ho_counts)),
            "e_oo_edge_count": int(sum(e_oo_counts)),
            "e_hh_edge_count": int(sum(e_hh_counts)),
            "edge_count_per_batch": edge_counts,
            "e_ho_edge_count_per_batch": e_ho_counts,
            "e_oo_edge_count_per_batch": e_oo_counts,
            "e_hh_edge_count_per_batch": e_hh_counts,
        }
        return edge_attr_dense, edge_mask_dense, edge_score_dense, counts

    def _add_ho_edges(
        self,
        batch_idx: int,
        boxes: Tensor,
        labels: Tensor,
        node_mask: Tensor,
        pair_indices: Tensor,
        pair_scores: Tensor,
        pair_feats: Tensor,
        union_feats: Optional[Tensor],
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        edge_score_dense: Tensor,
    ) -> int:
        count = 0
        num_nodes = boxes.shape[1]
        for pair_idx in range(pair_indices.shape[1]):
            human_idx = int(pair_indices[batch_idx, pair_idx, 0].item())
            object_idx = int(pair_indices[batch_idx, pair_idx, 1].item())
            if human_idx < 0 or object_idx < 0 or human_idx >= num_nodes or object_idx >= num_nodes:
                continue
            if not bool(node_mask[batch_idx, human_idx]) or not bool(node_mask[batch_idx, object_idx]):
                continue
            score = pair_scores[batch_idx, pair_idx]
            union_feat = None if union_feats is None else union_feats[batch_idx, pair_idx]
            self._write_edge(
                batch_idx=batch_idx,
                src_idx=human_idx,
                dst_idx=object_idx,
                score=score,
                edge_type=self.EDGE_HUMAN_TO_OBJECT,
                boxes=boxes,
                labels=labels,
                base_feat=pair_feats[batch_idx, pair_idx],
                union_feat=union_feat,
                edge_attr_dense=edge_attr_dense,
                edge_mask_dense=edge_mask_dense,
                edge_score_dense=edge_score_dense,
            )
            self._write_edge(
                batch_idx=batch_idx,
                src_idx=object_idx,
                dst_idx=human_idx,
                score=score,
                edge_type=self.EDGE_OBJECT_TO_HUMAN,
                boxes=boxes,
                labels=labels,
                base_feat=pair_feats[batch_idx, pair_idx],
                union_feat=union_feat,
                edge_attr_dense=edge_attr_dense,
                edge_mask_dense=edge_mask_dense,
                edge_score_dense=edge_score_dense,
            )
            count += 2
        return count

    def _add_spatial_edges(
        self,
        batch_idx: int,
        boxes: Tensor,
        labels: Tensor,
        node_types: Tensor,
        node_mask: Tensor,
        source_type: int,
        edge_type: int,
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        edge_score_dense: Tensor,
    ) -> int:
        node_ids = torch.logical_and(node_mask[batch_idx], node_types[batch_idx] == source_type).nonzero(as_tuple=False).squeeze(1)
        if node_ids.numel() < 2:
            return 0

        centers = self._box_centers(boxes[batch_idx, node_ids])
        distances = torch.cdist(centers, centers, p=2)
        self_mask = torch.eye(node_ids.numel(), dtype=torch.bool, device=boxes.device)
        distances = distances.masked_fill(self_mask, float("inf"))
        topk = min(self.k_oo, node_ids.numel() - 1)
        if topk <= 0:
            return 0

        count = 0
        _, nearest = distances.topk(k=topk, dim=1, largest=False)
        for local_src in range(node_ids.numel()):
            src_idx = int(node_ids[local_src].item())
            for local_dst in nearest[local_src]:
                dst_idx = int(node_ids[int(local_dst.item())].item())
                if src_idx == dst_idx:
                    continue
                distance = distances[local_src, int(local_dst.item())]
                score = torch.exp(-distance / self.spatial_sigma)
                for edge_src, edge_dst in ((src_idx, dst_idx), (dst_idx, src_idx)):
                    is_new_edge = not bool(edge_mask_dense[batch_idx, edge_src, edge_dst])
                    if not is_new_edge:
                        continue
                    self._write_edge(
                        batch_idx=batch_idx,
                        src_idx=edge_src,
                        dst_idx=edge_dst,
                        score=score,
                        edge_type=edge_type,
                        boxes=boxes,
                        labels=labels,
                        base_feat=None,
                        union_feat=None,
                        edge_attr_dense=edge_attr_dense,
                        edge_mask_dense=edge_mask_dense,
                        edge_score_dense=edge_score_dense,
                    )
                    count += 1
        return count

    def _write_edge(
        self,
        batch_idx: int,
        src_idx: int,
        dst_idx: int,
        score: Tensor,
        edge_type: int,
        boxes: Tensor,
        labels: Tensor,
        base_feat: Optional[Tensor],
        union_feat: Optional[Tensor],
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        edge_score_dense: Tensor,
    ) -> None:
        src_box = boxes[batch_idx, src_idx]
        dst_box = boxes[batch_idx, dst_idx]
        geo_feat = self.encode_geometry(src_box.unsqueeze(0), dst_box.unsqueeze(0)).squeeze(0)
        geo_emb = self.geo_mlp(geo_feat)
        type_ids = torch.tensor(edge_type, dtype=torch.long, device=boxes.device)
        type_emb = self.edge_type_embedding(type_ids)
        src_class_emb = self.class_embedding(labels[batch_idx, src_idx].clamp(min=0, max=self.num_obj_classes))
        dst_class_emb = self.class_embedding(labels[batch_idx, dst_idx].clamp(min=0, max=self.num_obj_classes))
        if base_feat is None:
            base_feat = geo_emb.new_zeros(self.d_model)
        if union_feat is None:
            union_feat = geo_emb.new_zeros(self.d_model)
        score_feat = score.reshape(1).to(dtype=geo_emb.dtype)
        edge_input = torch.cat([score_feat, geo_emb, type_emb, union_feat, base_feat, src_class_emb + dst_class_emb], dim=-1)
        edge_attr = self.edge_mlp(edge_input)
        edge_attr_dense[batch_idx, src_idx, dst_idx] = edge_attr
        edge_mask_dense[batch_idx, src_idx, dst_idx] = True
        edge_score_dense[batch_idx, src_idx, dst_idx] = score.to(dtype=edge_score_dense.dtype)

    def _gather_pair_edge_feats(self, edge_attr_dense: Tensor, pair_indices: Tensor, node_mask: Tensor) -> Tuple[Tensor, Tensor]:
        batch_feats = []
        valid_masks = []
        num_nodes = edge_attr_dense.shape[1]
        for batch_idx in range(pair_indices.shape[0]):
            raw_src = pair_indices[batch_idx, :, 0]
            raw_dst = pair_indices[batch_idx, :, 1]
            valid = (raw_src >= 0) & (raw_dst >= 0) & (raw_src < num_nodes) & (raw_dst < num_nodes)
            src = raw_src.clamp(min=0, max=num_nodes - 1)
            dst = raw_dst.clamp(min=0, max=num_nodes - 1)
            valid = valid & node_mask[batch_idx, src] & node_mask[batch_idx, dst]
            batch_feats.append(edge_attr_dense[batch_idx, src, dst])
            valid_masks.append(valid)
        return torch.stack(batch_feats, dim=0), torch.stack(valid_masks, dim=0)

    def _gather_pair_nodes(self, node_feats: Tensor, node_indices: Tensor) -> Tensor:
        batch_feats = []
        num_nodes = node_feats.shape[1]
        for batch_idx in range(node_indices.shape[0]):
            idx = node_indices[batch_idx].clamp(min=0, max=num_nodes - 1)
            batch_feats.append(node_feats[batch_idx, idx])
        return torch.stack(batch_feats, dim=0)

    def _build_pair_feats_from_endpoints(
        self,
        node_feats: Tensor,
        pair_indices: Tensor,
        pair_scores: Tensor,
    ) -> Tensor:
        batch_size, num_nodes, dim = node_feats.shape
        _, num_pairs, _ = pair_indices.shape

        safe_pair_indices = pair_indices.clamp(min=0, max=max(num_nodes - 1, 0))
        gather_index = safe_pair_indices.unsqueeze(-1).expand(batch_size, num_pairs, 2, dim)
        expanded_nodes = node_feats.unsqueeze(1).expand(batch_size, num_pairs, num_nodes, dim)
        endpoint_feats = torch.gather(expanded_nodes, dim=2, index=gather_index)
        human_feats = endpoint_feats[:, :, 0, :]
        object_feats = endpoint_feats[:, :, 1, :]
        score_feats = pair_scores.unsqueeze(-1).to(dtype=node_feats.dtype)

        return self.endpoint_pair_proj(torch.cat([human_feats, object_feats, score_feats], dim=-1))

    def _fuse_node_inputs(
        self,
        node_feats: Tensor,
        node_blip_feats: Optional[Tensor],
        boxes: Tensor,
        labels: Tensor,
        node_types: Tensor,
    ) -> Tensor:
        if node_blip_feats is None:
            node_blip_feats = torch.zeros_like(node_feats)
        label_ids = labels.clamp(min=0, max=self.num_obj_classes)
        type_ids = node_types.clamp(min=0, max=self.num_node_types - 1)
        class_feats = self.class_embedding(label_ids)
        type_feats = self.node_type_embedding(type_ids)
        box_feats = self.box_mlp(boxes.clamp(0, 1))
        return self.node_fusion(torch.cat([node_feats, node_blip_feats, class_feats, box_feats, type_feats], dim=-1))

    def encode_geometry(self, boxes_i: Tensor, boxes_j: Tensor, eps: float = 1e-6) -> Tensor:
        xi1, yi1, xi2, yi2 = boxes_i.unbind(dim=-1)
        xj1, yj1, xj2, yj2 = boxes_j.unbind(dim=-1)
        wi = (xi2 - xi1).clamp(min=eps)
        hi = (yi2 - yi1).clamp(min=eps)
        wj = (xj2 - xj1).clamp(min=eps)
        hj = (yj2 - yj1).clamp(min=eps)
        cxi = (xi1 + xi2) * 0.5
        cyi = (yi1 + yi2) * 0.5
        cxj = (xj1 + xj2) * 0.5
        cyj = (yj1 + yj2) * 0.5
        dx = cxj - cxi
        dy = cyj - cyi
        dw = torch.log((wj + eps) / (wi + eps))
        dh = torch.log((hj + eps) / (hi + eps))
        center_dist = torch.sqrt(dx.square() + dy.square() + eps)
        area_i = wi * hi
        area_j = wj * hj
        inter_x1 = torch.maximum(xi1, xj1)
        inter_y1 = torch.maximum(yi1, yj1)
        inter_x2 = torch.minimum(xi2, xj2)
        inter_y2 = torch.minimum(yi2, yj2)
        inter_area = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
        union_area = area_i + area_j - inter_area
        iou = inter_area / union_area.clamp(min=eps)
        return torch.stack([dx, dy, dw, dh, center_dist, iou, union_area, area_i, area_j], dim=-1)

    def _box_centers(self, boxes: Tensor) -> Tensor:
        return torch.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], dim=-1)

    def _check_inputs(
        self,
        node_feats: Tensor,
        node_blip_feats: Optional[Tensor],
        boxes: Tensor,
        labels: Tensor,
        node_types: Tensor,
        node_mask: Tensor,
        pair_indices: Tensor,
        pair_scores: Tensor,
        pair_feats: Optional[Tensor],
        union_feats: Optional[Tensor],
        global_feat: Optional[Tensor],
    ) -> None:
        if node_feats.dim() != 3:
            raise ValueError(f"node_feats must be [B, N, D], got {tuple(node_feats.shape)}")
        batch_size, num_nodes, dim = node_feats.shape
        if dim != self.d_model:
            raise ValueError(f"node_feats last dim must be {self.d_model}, got {dim}")
        if node_blip_feats is not None and node_blip_feats.shape != (batch_size, num_nodes, self.d_model):
            raise ValueError(f"node_blip_feats must be [B, N, D], got {tuple(node_blip_feats.shape)}")
        if boxes.shape != (batch_size, num_nodes, 4):
            raise ValueError(f"boxes must be [B, N, 4], got {tuple(boxes.shape)}")
        if labels.shape != (batch_size, num_nodes):
            raise ValueError(f"labels must be [B, N], got {tuple(labels.shape)}")
        if node_types.shape != (batch_size, num_nodes):
            raise ValueError(f"node_types must be [B, N], got {tuple(node_types.shape)}")
        if node_mask.shape != (batch_size, num_nodes):
            raise ValueError(f"node_mask must be [B, N], got {tuple(node_mask.shape)}")
        if pair_indices.dim() != 3 or pair_indices.shape[0] != batch_size or pair_indices.shape[-1] != 2:
            raise ValueError(f"pair_indices must be [B, K, 2], got {tuple(pair_indices.shape)}")
        num_pairs = pair_indices.shape[1]
        if pair_scores.shape != (batch_size, num_pairs):
            raise ValueError(f"pair_scores must be [B, K], got {tuple(pair_scores.shape)}")
        if pair_indices.dtype != torch.long:
            raise ValueError(f"pair_indices must be torch.long, got {pair_indices.dtype}")
        if labels.dtype != torch.long:
            raise ValueError(f"labels must be torch.long, got {labels.dtype}")
        if node_types.dtype != torch.long:
            raise ValueError(f"node_types must be torch.long, got {node_types.dtype}")
        if node_mask.dtype != torch.bool:
            raise ValueError(f"node_mask must be torch.bool, got {node_mask.dtype}")
        if pair_feats is not None and pair_feats.shape != (batch_size, num_pairs, self.d_model):
            raise ValueError(f"pair_feats must be [B, K, D], got {tuple(pair_feats.shape)}")
        if union_feats is not None and union_feats.shape != (batch_size, num_pairs, self.d_model):
            raise ValueError(f"union_feats must be [B, K, D], got {tuple(union_feats.shape)}")
        if global_feat is not None and global_feat.shape != (batch_size, self.d_model):
            raise ValueError(f"global_feat must be [B, D], got {tuple(global_feat.shape)}")


def build_visual_scene_graph_reasoner(args=None, **kwargs) -> VisualSceneGraphReasoner:
    if args is None:
        return VisualSceneGraphReasoner(**kwargs)

    config = {
        "d_model": getattr(args, "hidden_dim", getattr(args, "d_model", 256)),
        "num_obj_classes": getattr(args, "num_classes", 80),
        "num_edge_types": getattr(args, "graph_num_edge_types", 4),
        "num_node_types": getattr(args, "graph_num_node_types", 3),
        "num_layers": getattr(args, "graph_num_layers", 2),
        "num_heads": getattr(args, "graph_num_heads", 8),
        "k_oo": getattr(args, "graph_k_oo", 3),
        "enable_oo_edges": getattr(args, "graph_enable_oo_edges", True),
        "enable_hh_edges": getattr(args, "graph_enable_hh_edges", False),
        "dropout": getattr(args, "graph_dropout", getattr(args, "dropout", 0.1)),
        "spatial_sigma": getattr(args, "graph_spatial_sigma", 0.5),
        "graph_enable_text_graph": getattr(args, "graph_enable_text_graph", False),
        "graph_text_features_path": getattr(args, "graph_text_features_path", ""),
        "graph_text_mask_path": getattr(args, "graph_text_mask_path", ""),
        "graph_text_hoi_ids_path": getattr(args, "graph_text_hoi_ids_path", ""),
        "graph_text_object_to_hoi_indices_path": getattr(args, "graph_text_object_to_hoi_indices_path", ""),
        "graph_text_field_order_path": getattr(args, "graph_text_field_order_path", ""),
        "graph_text_field_names": getattr(args, "graph_text_field_names", None),
        "graph_text_max_candidates": getattr(args, "graph_text_max_candidates", 0),
        "graph_text_gate_init": getattr(args, "graph_text_gate_init", 0.0),
        "graph_text_normalize": getattr(args, "graph_text_normalize", True),
        "graph_text_strict_shape": getattr(args, "graph_text_strict_shape", True),
        "graph_text_strict_consistency": getattr(args, "graph_text_strict_consistency", False),
        "graph_text_human_node_features_path": getattr(args, "graph_text_human_node_features_path", ""),
        "graph_text_object_node_features_path": getattr(args, "graph_text_object_node_features_path", ""),
        "graph_text_ho_edge_features_path": getattr(args, "graph_text_ho_edge_features_path", ""),
        "graph_text_discriminative_features_path": getattr(args, "graph_text_discriminative_features_path", ""),
        "graph_text_scene_context_features_path": getattr(args, "graph_text_scene_context_features_path", ""),
        "graph_text_oo_context_features_path": getattr(args, "graph_text_oo_context_features_path", ""),
        "graph_text_oo_affordance_prior_path": getattr(args, "graph_text_oo_affordance_prior_path", ""),
        "graph_text_label_ids": getattr(args, "graph_text_label_ids", getattr(args, "valid_obj_ids", None)),
        "graph_debug": getattr(args, "graph_debug", False),
    }
    config.update(kwargs)
    return VisualSceneGraphReasoner(**config)
