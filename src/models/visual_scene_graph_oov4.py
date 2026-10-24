import math
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


def _load_text_scene_graph_classes():
    try:
        from .text_scene_graph import TextGraphFeatureBank, TextHOIRetriever, TextSceneGraphReasoner
    except ImportError:
        module_path = Path(__file__).resolve().with_name("text_scene_graph.py")
        spec = importlib.util.spec_from_file_location("text_scene_graph_direct", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        TextGraphFeatureBank = module.TextGraphFeatureBank
        TextHOIRetriever = module.TextHOIRetriever
        TextSceneGraphReasoner = module.TextSceneGraphReasoner
    return TextGraphFeatureBank, TextHOIRetriever, TextSceneGraphReasoner


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
        graph_oo_mode: str = "spatial",
        graph_oo_lambda_geo: float = 1.0,
        graph_oo_lambda_share: float = 0.5,
        graph_oo_lambda_text: float = 0.0,
        graph_oo_min_score: float = 0.0,
        graph_oo_fallback_to_spatial: bool = True,
        graph_oo_share_eps: float = 1e-6,
        graph_oo_prior_path: str = "",
        graph_oo_prior_normalize: str = "none",
        graph_oo_prior_symmetric: bool = True,
        graph_oo_prior_eps: float = 1e-6,
        graph_oo_prior_label_ids: Optional[List[int]] = None,
        enable_text_graph: bool = False,
        text_features_path: str = "",
        text_mask_path: str = "",
        hoi_ids_path: str = "",
        object_to_hoi_indices_path: str = "",
        field_order_path: str = "",
        graph_text_human_node_features_path: str = "",
        graph_text_object_node_features_path: str = "",
        graph_text_ho_edge_features_path: str = "",
        graph_text_discriminative_features_path: str = "",
        graph_text_scene_context_features_path: str = "",
        graph_text_oo_context_features_path: str = "",
        oo_affordance_prior_path: str = "",
        graph_text_topk: int = 5,
        graph_text_pre_topk: int = 20,
        graph_text_temperature: float = 0.07,
        graph_text_rerank_weight: float = 0.5,
        graph_text_use_discriminative_rerank: bool = True,
        graph_text_label_ids: Optional[List[int]] = None,
        graph_text_num_layers: int = 1,
        graph_text_num_heads: int = 8,
        graph_text_dropout: float = 0.1,
        graph_text_use_scene_token: bool = True,
        graph_text_use_oo_context_token: bool = True,
        graph_text_cross_num_heads: int = 8,
        graph_text_query_source: str = "refined_edge",
        graph_text_gate_init: float = 0.0,
        graph_text_debug: bool = False,
        graph_text_prior_injection: str = "none",
        graph_text_use_human_node_prior: bool = True,
        graph_text_use_object_node_prior: bool = True,
        graph_text_use_ho_edge_prior: bool = True,
        graph_text_use_discriminative_prior: bool = False,
        graph_text_use_scene_prior: bool = False,
        graph_text_use_oo_prior: bool = False,
        graph_text_node_prior_gate_init: float = 0.0,
        graph_text_edge_prior_gate_init: float = 0.0,
        graph_text_scene_prior_gate_init: float = 0.0,
        graph_text_oo_prior_gate_init: float = 0.0,
    ):
        super().__init__()
        valid_oo_modes = {"spatial", "share_human", "hybrid", "text_hybrid"}
        if graph_oo_mode not in valid_oo_modes:
            raise ValueError(f"graph_oo_mode must be one of {sorted(valid_oo_modes)}, got {graph_oo_mode}")
        valid_text_query_sources = {"base_pair", "refined_edge", "union", "refined_plus_union"}
        if graph_text_query_source not in valid_text_query_sources:
            raise ValueError(
                f"graph_text_query_source must be one of {sorted(valid_text_query_sources)}, got {graph_text_query_source}"
            )
        valid_text_prior_injection = {"none", "edge_only", "node_edge", "all"}
        if graph_text_prior_injection not in valid_text_prior_injection:
            raise ValueError(
                f"graph_text_prior_injection must be one of {sorted(valid_text_prior_injection)}, got {graph_text_prior_injection}"
            )
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
        self.graph_oo_mode = graph_oo_mode
        self.graph_oo_lambda_geo = graph_oo_lambda_geo
        self.graph_oo_lambda_share = graph_oo_lambda_share
        self.graph_oo_lambda_text = graph_oo_lambda_text
        self.graph_oo_min_score = graph_oo_min_score
        self.graph_oo_fallback_to_spatial = graph_oo_fallback_to_spatial
        self.graph_oo_share_eps = graph_oo_share_eps
        self.graph_oo_prior_normalize = graph_oo_prior_normalize
        self.graph_oo_prior_symmetric = graph_oo_prior_symmetric
        self.graph_oo_prior_eps = graph_oo_prior_eps
        self.enable_text_graph = bool(enable_text_graph)
        self.graph_text_query_source = graph_text_query_source
        self.graph_text_debug = bool(graph_text_debug)
        self.graph_text_prior_injection = graph_text_prior_injection
        self.graph_text_use_human_node_prior = bool(graph_text_use_human_node_prior)
        self.graph_text_use_object_node_prior = bool(graph_text_use_object_node_prior)
        self.graph_text_use_ho_edge_prior = bool(graph_text_use_ho_edge_prior)
        self.graph_text_use_discriminative_prior = bool(graph_text_use_discriminative_prior)
        self.graph_text_use_scene_prior = bool(graph_text_use_scene_prior)
        self.graph_text_use_oo_prior = bool(graph_text_use_oo_prior)
        self._load_oo_affordance_prior(
            prior_path=graph_oo_prior_path,
            normalize=graph_oo_prior_normalize,
            symmetric=graph_oo_prior_symmetric,
            eps=graph_oo_prior_eps,
        )
        self._build_oo_prior_label_map(graph_oo_prior_label_ids)

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
        self.text_graph_reasoner = None
        self.text_cross_attn = None
        self.text_cross_norm = None
        self.text_context_proj = None
        self.text_refined_union_query_proj = None
        self.text_gate = nn.Parameter(torch.tensor(float(graph_text_gate_init)))
        self.text_human_node_proj = self._make_text_prior_mlp(d_model)
        self.text_object_node_proj = self._make_text_prior_mlp(d_model)
        self.text_ho_edge_proj = self._make_text_prior_mlp(d_model)
        self.text_discriminative_proj = self._make_text_prior_mlp(d_model)
        self.text_scene_proj = self._make_text_prior_mlp(d_model)
        self.text_oo_proj = self._make_text_prior_mlp(d_model)
        self.text_human_node_gate = nn.Parameter(torch.tensor(float(graph_text_node_prior_gate_init)))
        self.text_object_node_gate = nn.Parameter(torch.tensor(float(graph_text_node_prior_gate_init)))
        self.text_ho_edge_gate = nn.Parameter(torch.tensor(float(graph_text_edge_prior_gate_init)))
        self.text_discriminative_gate = nn.Parameter(torch.tensor(float(graph_text_edge_prior_gate_init)))
        self.text_scene_gate = nn.Parameter(torch.tensor(float(graph_text_scene_prior_gate_init)))
        self.text_oo_gate = nn.Parameter(torch.tensor(float(graph_text_oo_prior_gate_init)))
        if self.enable_text_graph:
            TextGraphFeatureBank, TextHOIRetriever, TextSceneGraphReasoner = _load_text_scene_graph_classes()
            text_bank = TextGraphFeatureBank(
                text_features_path=text_features_path,
                text_mask_path=text_mask_path,
                hoi_ids_path=hoi_ids_path,
                object_to_hoi_indices_path=object_to_hoi_indices_path,
                field_order_path=field_order_path,
                d_model=d_model,
                normalize=True,
                strict_shape=False,
                human_node_features_path=graph_text_human_node_features_path,
                object_node_features_path=graph_text_object_node_features_path,
                ho_edge_features_path=graph_text_ho_edge_features_path,
                discriminative_features_path=graph_text_discriminative_features_path,
                scene_context_features_path=graph_text_scene_context_features_path,
                oo_context_features_path=graph_text_oo_context_features_path,
                oo_affordance_prior_path=oo_affordance_prior_path,
                label_ids=graph_text_label_ids,
            )
            text_retriever = TextHOIRetriever(
                bank=text_bank,
                d_model=d_model,
                topk=graph_text_topk,
                pre_topk=graph_text_pre_topk,
                temperature=graph_text_temperature,
                rerank_weight=graph_text_rerank_weight,
                use_discriminative_rerank=graph_text_use_discriminative_rerank,
                query_proj=True,
                debug=graph_text_debug,
            )
            self.text_graph_reasoner = TextSceneGraphReasoner(
                retriever=text_retriever,
                d_model=d_model,
                num_layers=graph_text_num_layers,
                num_heads=graph_text_num_heads,
                dropout=graph_text_dropout,
                use_scene_token=graph_text_use_scene_token,
                use_oo_context_token=graph_text_use_oo_context_token,
                max_topk=max(int(graph_text_topk), 1),
                debug=graph_text_debug,
            )
            self.text_cross_attn = nn.MultiheadAttention(
                d_model,
                graph_text_cross_num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.text_cross_norm = nn.LayerNorm(d_model)
            self.text_context_proj = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
            )
            self.text_refined_union_query_proj = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.ReLU(inplace=True),
                nn.Linear(d_model, d_model),
            )

    def _make_text_prior_mlp(self, d_model: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def _load_oo_affordance_prior(
        self,
        prior_path: str,
        normalize: str = "none",
        symmetric: bool = True,
        eps: float = 1e-6,
    ) -> Tensor:
        if not prior_path:
            A = torch.zeros((80, 80), dtype=torch.float32)
            self.register_buffer("oo_affordance_prior", A)
            self.oo_prior_loaded = False
            self.oo_prior_path = ""
            self.oo_prior_shape = tuple(A.shape)
            return A

        if not os.path.isfile(prior_path):
            raise FileNotFoundError(f"OO affordance prior file not found: {prior_path}")

        ext = os.path.splitext(prior_path)[1].lower()
        if ext == ".npy":
            A = torch.as_tensor(np.load(prior_path), dtype=torch.float32)
        elif ext in {".pt", ".pth"}:
            prior_data = torch.load(prior_path, map_location="cpu")
            if isinstance(prior_data, dict):
                for key in ("oo_affordance_prior", "prior", "A"):
                    if key in prior_data:
                        prior_data = prior_data[key]
                        break
            A = torch.as_tensor(prior_data, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported OO affordance prior extension: {ext}")

        if A.dim() != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"OO affordance prior must be a 2D square matrix, got {tuple(A.shape)}")
        A = torch.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)
        if normalize == "none":
            A = A.clamp(0.0, 1.0)
        elif normalize == "minmax":
            finite = torch.isfinite(A)
            if bool(finite.any().item()):
                finite_vals = A[finite]
                denom = (finite_vals.max() - finite_vals.min()).clamp(min=eps)
                A = (A - finite_vals.min()) / denom
            else:
                A = torch.zeros_like(A)
            A = A.clamp(0.0, 1.0)
        elif normalize == "sigmoid":
            A = torch.sigmoid(A).clamp(0.0, 1.0)
        else:
            raise ValueError("graph_oo_prior_normalize must be one of ['none', 'minmax', 'sigmoid']")
        if symmetric:
            A = (0.5 * (A + A.t())).clamp(0.0, 1.0)
        A.fill_diagonal_(0.0)
        self.register_buffer("oo_affordance_prior", A)
        self.oo_prior_loaded = bool(prior_path)
        self.oo_prior_path = prior_path
        self.oo_prior_shape = tuple(A.shape)
        return A

    def _build_oo_prior_label_map(self, graph_oo_prior_label_ids: Optional[List[int]]) -> None:
        if graph_oo_prior_label_ids is None or len(graph_oo_prior_label_ids) == 0:
            return
        label_ids = [int(label_id) for label_id in graph_oo_prior_label_ids if int(label_id) >= 0]
        if not label_ids:
            return
        label_to_prior_index = torch.full((max(label_ids) + 1,), -1, dtype=torch.long)
        for prior_idx, label_id in enumerate(graph_oo_prior_label_ids):
            label_id = int(label_id)
            if label_id < 0 or prior_idx >= self.oo_affordance_prior.shape[0]:
                continue
            label_to_prior_index[label_id] = prior_idx
        self.register_buffer("oo_prior_label_to_index", label_to_prior_index)

    def _map_labels_to_oo_prior_indices(self, labels_img: Tensor) -> Tuple[Tensor, Tensor]:
        labels_long = labels_img.to(dtype=torch.long)
        if hasattr(self, "oo_prior_label_to_index"):
            mapping = self.oo_prior_label_to_index.to(device=labels_img.device)
            valid_range = (labels_long >= 0) & (labels_long < mapping.shape[0])
            safe_labels = labels_long.clamp(min=0, max=max(mapping.shape[0] - 1, 0))
            prior_indices = mapping[safe_labels]
            valid_mask = valid_range & (prior_indices >= 0)
            prior_indices = prior_indices.clamp(min=0, max=max(self.oo_affordance_prior.shape[0] - 1, 0))
            return prior_indices, valid_mask
        prior_size = self.oo_affordance_prior.shape[0]
        valid_mask = (labels_long >= 0) & (labels_long < prior_size)
        prior_indices = labels_long.clamp(min=0, max=max(prior_size - 1, 0))
        return prior_indices, valid_mask

    def _build_oo_text_score_matrix(self, labels_img: Tensor, object_ids: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        num_objects = int(object_ids.numel())
        s_text = torch.zeros((num_objects, num_objects), dtype=torch.float32, device=labels_img.device)
        stats = self._empty_oo_text_stats()
        if self.graph_oo_mode != "text_hybrid" or self.graph_oo_lambda_text <= 0.0:
            return s_text, stats
        stats["oo_prior_loaded"] = self.oo_prior_loaded
        stats["oo_prior_shape"] = self.oo_prior_shape
        prior = self.oo_affordance_prior.to(device=labels_img.device)
        if prior.numel() == 0 or not bool((prior > 0).any().item()):
            return s_text, stats
        obj_labels = labels_img[object_ids]
        prior_idx, valid_obj_mask = self._map_labels_to_oo_prior_indices(obj_labels)
        valid_pair = valid_obj_mask[:, None] & valid_obj_mask[None, :]
        idx_i = prior_idx.clamp(0, prior.shape[0] - 1)
        idx_j = prior_idx.clamp(0, prior.shape[1] - 1)
        raw = prior[idx_i[:, None], idx_j[None, :]].to(dtype=s_text.dtype)
        s_text = torch.where(valid_pair, raw, torch.zeros_like(raw))
        if num_objects > 0:
            s_text.fill_diagonal_(0.0)
        stats.update(self._oo_text_matrix_stats(s_text, valid_pair))
        stats["oo_text_invalid_label_count"] = int((~valid_obj_mask).sum().detach().cpu().item())
        stats["oo_text_used"] = bool(stats["oo_text_max"] > 0.0)
        stats["oo_text_inactive"] = not stats["oo_text_used"]
        return s_text, stats

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
        text_debug_info = self._empty_text_debug_info(refined_edge)
        if self.enable_text_graph:
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
        oo_stats_per_batch = []

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
            oo_stats = self._empty_oo_stats(oo_mode=self.graph_oo_mode)
            if self.enable_oo_edges:
                if self.graph_oo_mode == "spatial":
                    oo_stats = self._spatial_oo_stats(
                        batch_idx=batch_idx,
                        boxes=boxes,
                        node_types=node_types,
                        node_mask=node_mask,
                    )
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
                    oo_stats["oo_topk_count"] = e_oo
                else:
                    e_oo, oo_stats = self._add_human_conditioned_oo_edges(
                        batch_idx=batch_idx,
                        boxes=boxes,
                        labels=labels,
                        node_types=node_types,
                        node_mask=node_mask,
                        pair_indices=pair_indices,
                        pair_scores=pair_scores,
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
            oo_stats_per_batch.append(oo_stats)

        def _mean(values):
            return float(sum(values) / len(values)) if values else 0.0

        def _max(values):
            return float(max(values)) if values else 0.0

        oo_geo_means = [stats["oo_geo_mean"] for stats in oo_stats_per_batch]
        oo_geo_maxes = [stats["oo_geo_max"] for stats in oo_stats_per_batch]
        oo_share_means = [stats["oo_share_mean"] for stats in oo_stats_per_batch]
        oo_share_maxes = [stats["oo_share_max"] for stats in oo_stats_per_batch]
        oo_text_means = [stats["oo_text_mean"] for stats in oo_stats_per_batch]
        oo_text_maxes = [stats["oo_text_max"] for stats in oo_stats_per_batch]
        oo_score_means = [stats["oo_score_mean"] for stats in oo_stats_per_batch]
        oo_score_maxes = [stats["oo_score_max"] for stats in oo_stats_per_batch]

        counts = {
            "e_ho_edge_count": int(sum(e_ho_counts)),
            "e_oo_edge_count": int(sum(e_oo_counts)),
            "e_hh_edge_count": int(sum(e_hh_counts)),
            "edge_count_per_batch": edge_counts,
            "e_ho_edge_count_per_batch": e_ho_counts,
            "e_oo_edge_count_per_batch": e_oo_counts,
            "e_hh_edge_count_per_batch": e_hh_counts,
            "oo_mode": self.graph_oo_mode,
            "oo_fallback_per_batch": [stats["oo_fallback"] for stats in oo_stats_per_batch],
            "oo_geo_mean_per_batch": oo_geo_means,
            "oo_geo_max_per_batch": oo_geo_maxes,
            "oo_share_mean_per_batch": oo_share_means,
            "oo_share_max_per_batch": oo_share_maxes,
            "oo_text_used_per_batch": [stats["oo_text_used"] for stats in oo_stats_per_batch],
            "oo_text_inactive_per_batch": [stats["oo_text_inactive"] for stats in oo_stats_per_batch],
            "oo_text_mean_per_batch": oo_text_means,
            "oo_text_max_per_batch": oo_text_maxes,
            "oo_text_nonzero_ratio_per_batch": [stats["oo_text_nonzero_ratio"] for stats in oo_stats_per_batch],
            "oo_text_invalid_label_count_per_batch": [stats["oo_text_invalid_label_count"] for stats in oo_stats_per_batch],
            "oo_score_mean_per_batch": oo_score_means,
            "oo_score_max_per_batch": oo_score_maxes,
            "oo_topk_count_per_batch": [stats["oo_topk_count"] for stats in oo_stats_per_batch],
            "oo_geo_mean": _mean(oo_geo_means),
            "oo_geo_max": _max(oo_geo_maxes),
            "oo_share_mean": _mean(oo_share_means),
            "oo_share_max": _max(oo_share_maxes),
            "oo_text_mean": _mean(oo_text_means),
            "oo_text_max": _max(oo_text_maxes),
            "oo_score_mean": _mean(oo_score_means),
            "oo_score_max": _max(oo_score_maxes),
        }
        return edge_attr_dense, edge_mask_dense, edge_score_dense, counts

    def _empty_oo_stats(self, oo_mode: str, fallback: bool = False) -> Dict[str, Any]:
        return {
            "oo_mode": oo_mode,
            "oo_fallback": fallback,
            "oo_prior_loaded": self.oo_prior_loaded,
            "oo_prior_shape": self.oo_prior_shape,
            "oo_geo_mean": 0.0,
            "oo_geo_max": 0.0,
            "oo_share_mean": 0.0,
            "oo_share_max": 0.0,
            "oo_text_used": False,
            "oo_text_inactive": True,
            "oo_text_mean": 0.0,
            "oo_text_max": 0.0,
            "oo_text_min_valid": 0.0,
            "oo_text_nonzero_ratio": 0.0,
            "oo_text_invalid_label_count": 0,
            "oo_score_mean": 0.0,
            "oo_score_max": 0.0,
            "oo_topk_count": 0,
        }

    def _empty_oo_text_stats(self) -> Dict[str, Any]:
        return {
            "oo_text_used": False,
            "oo_prior_loaded": self.oo_prior_loaded,
            "oo_prior_shape": self.oo_prior_shape,
            "oo_text_invalid_label_count": 0,
            "oo_text_nonzero_ratio": 0.0,
            "oo_text_mean": 0.0,
            "oo_text_max": 0.0,
            "oo_text_min_valid": 0.0,
            "oo_text_inactive": True,
        }

    def _oo_text_matrix_stats(self, scores: Tensor, valid_pair: Optional[Tensor] = None) -> Dict[str, Any]:
        stats = self._empty_oo_text_stats()
        if scores.numel() == 0:
            return stats
        valid = torch.isfinite(scores)
        if valid_pair is not None:
            valid = valid & valid_pair
        if scores.dim() == 2 and scores.shape[0] == scores.shape[1]:
            valid = valid & ~torch.eye(scores.shape[0], dtype=torch.bool, device=scores.device)
        if not bool(valid.any().item()):
            return stats
        valid_scores = scores[valid].detach().float()
        stats["oo_text_mean"] = float(valid_scores.mean().item())
        stats["oo_text_max"] = float(valid_scores.max().item())
        stats["oo_text_min_valid"] = float(valid_scores.min().item())
        stats["oo_text_nonzero_ratio"] = float((valid_scores > 0).float().mean().item())
        stats["oo_text_used"] = bool(stats["oo_text_max"] > 0.0)
        stats["oo_text_inactive"] = not stats["oo_text_used"]
        return stats

    def _valid_score_stats(self, scores: Tensor) -> Tuple[float, float]:
        valid = torch.isfinite(scores)
        if not bool(valid.any().item()):
            return 0.0, 0.0
        valid_scores = scores[valid].detach().float()
        return float(valid_scores.mean().item()), float(valid_scores.max().item())

    def _spatial_oo_stats(
        self,
        batch_idx: int,
        boxes: Tensor,
        node_types: Tensor,
        node_mask: Tensor,
    ) -> Dict[str, Any]:
        stats = self._empty_oo_stats(oo_mode="spatial")
        object_ids = torch.logical_and(node_mask[batch_idx], node_types[batch_idx] == 1).nonzero(as_tuple=False).squeeze(1)
        if object_ids.numel() < 2:
            return stats
        centers = self._box_centers(boxes[batch_idx, object_ids])
        distances = torch.cdist(centers, centers, p=2)
        s_geo = torch.exp(-distances / self.spatial_sigma)
        diag = torch.eye(object_ids.numel(), dtype=torch.bool, device=boxes.device)
        s_geo = s_geo.masked_fill(diag, float("-inf"))
        stats["oo_geo_mean"], stats["oo_geo_max"] = self._valid_score_stats(s_geo)
        stats["oo_score_mean"], stats["oo_score_max"] = stats["oo_geo_mean"], stats["oo_geo_max"]
        return stats

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

    def _build_ho_score_matrix(
        self,
        pair_indices_img: Tensor,
        pair_scores_img: Tensor,
        node_types_img: Tensor,
        node_mask_img: Tensor,
        num_nodes: int,
    ) -> Tensor:
        ho_score_flat = pair_scores_img.new_zeros(num_nodes * num_nodes)
        if pair_indices_img.numel() == 0:
            return ho_score_flat.view(num_nodes, num_nodes)

        human_idx = pair_indices_img[:, 0]
        object_idx = pair_indices_img[:, 1]
        valid = (human_idx >= 0) & (object_idx >= 0) & (human_idx < num_nodes) & (object_idx < num_nodes)
        if not bool(valid.any().item()):
            return ho_score_flat.view(num_nodes, num_nodes)

        safe_human_idx = human_idx.clamp(min=0, max=max(num_nodes - 1, 0))
        safe_object_idx = object_idx.clamp(min=0, max=max(num_nodes - 1, 0))
        valid = (
            valid
            & node_mask_img[safe_human_idx]
            & node_mask_img[safe_object_idx]
            & (node_types_img[safe_human_idx] == 0)
            & (node_types_img[safe_object_idx] == 1)
        )
        if not bool(valid.any().item()):
            return ho_score_flat.view(num_nodes, num_nodes)

        flat_idx = safe_human_idx[valid] * num_nodes + safe_object_idx[valid]
        scores = torch.sigmoid(pair_scores_img[valid])
        ho_score_flat = ho_score_flat.scatter_reduce(0, flat_idx, scores, reduce="amax", include_self=True)
        return ho_score_flat.view(num_nodes, num_nodes)

    def _add_human_conditioned_oo_edges(
        self,
        batch_idx: int,
        boxes: Tensor,
        labels: Tensor,
        node_types: Tensor,
        node_mask: Tensor,
        pair_indices: Tensor,
        pair_scores: Tensor,
        edge_attr_dense: Tensor,
        edge_mask_dense: Tensor,
        edge_score_dense: Tensor,
    ) -> Tuple[int, Dict[str, Any]]:
        valid_nodes = node_mask[batch_idx]
        human_ids = torch.logical_and(valid_nodes, node_types[batch_idx] == 0).nonzero(as_tuple=False).squeeze(1)
        object_ids = torch.logical_and(valid_nodes, node_types[batch_idx] == 1).nonzero(as_tuple=False).squeeze(1)
        stats = self._empty_oo_stats(oo_mode=self.graph_oo_mode)

        if object_ids.numel() < 2:
            return 0, stats

        num_nodes = boxes.shape[1]
        if human_ids.numel() > 0:
            ho_score = self._build_ho_score_matrix(
                pair_indices_img=pair_indices[batch_idx],
                pair_scores_img=pair_scores[batch_idx],
                node_types_img=node_types[batch_idx],
                node_mask_img=node_mask[batch_idx],
                num_nodes=num_nodes,
            )
            score_h_obj = ho_score[human_ids[:, None], object_ids[None, :]]
            share = score_h_obj[:, :, None] * score_h_obj[:, None, :]
            s_share = share.max(dim=0).values
        else:
            s_share = boxes.new_zeros((object_ids.numel(), object_ids.numel()))

        centers = self._box_centers(boxes[batch_idx, object_ids])
        distances = torch.cdist(centers, centers, p=2)
        s_geo = torch.exp(-distances / self.spatial_sigma)
        s_text, text_stats = self._build_oo_text_score_matrix(labels[batch_idx], object_ids)
        stats.update(text_stats)

        diag = torch.eye(object_ids.numel(), dtype=torch.bool, device=boxes.device)
        s_geo_stats = s_geo.masked_fill(diag, float("-inf"))
        s_share_stats = s_share.masked_fill(diag, float("-inf"))

        stats["oo_geo_mean"], stats["oo_geo_max"] = self._valid_score_stats(s_geo_stats)
        stats["oo_share_mean"], stats["oo_share_max"] = self._valid_score_stats(s_share_stats)

        if self.graph_oo_mode == "share_human" and stats["oo_share_max"] <= self.graph_oo_share_eps:
            if not self.graph_oo_fallback_to_spatial:
                return 0, stats
            count = self._add_spatial_edges(
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
            stats["oo_fallback"] = True
            stats["oo_topk_count"] = count
            return count, stats

        if self.graph_oo_mode == "share_human":
            s_oo = self.graph_oo_lambda_share * s_share
        elif self.graph_oo_mode == "hybrid":
            s_oo = (
                self.graph_oo_lambda_geo * s_geo
                + self.graph_oo_lambda_share * s_share
            )
        elif self.graph_oo_mode == "text_hybrid":
            s_oo = (
                self.graph_oo_lambda_geo * s_geo
                + self.graph_oo_lambda_share * s_share
                + self.graph_oo_lambda_text * s_text.to(dtype=s_geo.dtype)
            )
        else:
            raise ValueError(f"Unsupported human-conditioned OO mode: {self.graph_oo_mode}")
        s_oo = s_oo.masked_fill(diag, float("-inf"))

        stats["oo_score_mean"], stats["oo_score_max"] = self._valid_score_stats(s_oo)

        topk = min(self.k_oo, object_ids.numel() - 1)
        if topk <= 0:
            return 0, stats

        count = 0
        _, nearest = s_oo.topk(k=topk, dim=1, largest=True)
        for local_src in range(object_ids.numel()):
            src_idx = int(object_ids[local_src].item())
            for local_dst_tensor in nearest[local_src]:
                local_dst = int(local_dst_tensor.item())
                dst_idx = int(object_ids[local_dst].item())
                if src_idx == dst_idx:
                    continue
                score = s_oo[local_src, local_dst]
                if not bool(torch.isfinite(score).item()):
                    continue
                if self.graph_oo_min_score > 0.0 and float(score.detach().item()) < self.graph_oo_min_score:
                    continue
                for edge_src, edge_dst in ((src_idx, dst_idx), (dst_idx, src_idx)):
                    if bool(edge_mask_dense[batch_idx, edge_src, edge_dst]):
                        continue
                    self._write_edge(
                        batch_idx=batch_idx,
                        src_idx=edge_src,
                        dst_idx=edge_dst,
                        score=score,
                        edge_type=self.EDGE_OBJECT_OBJECT,
                        boxes=boxes,
                        labels=labels,
                        base_feat=None,
                        union_feat=None,
                        edge_attr_dense=edge_attr_dense,
                        edge_mask_dense=edge_mask_dense,
                        edge_score_dense=edge_score_dense,
                    )
                    count += 1

        if count == 0 and self.graph_oo_mode in {"hybrid", "text_hybrid"} and self.graph_oo_fallback_to_spatial:
            count = self._add_spatial_edges(
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
            stats["oo_fallback"] = True

        stats["oo_topk_count"] = count
        return count, stats

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

    def _build_text_query_feats(
        self,
        source: str,
        refined_edge: Tensor,
        base_pair_feats: Tensor,
        union_feats: Tensor,
    ) -> Tensor:
        if source == "base_pair":
            return base_pair_feats
        if source == "refined_edge":
            return refined_edge
        if source == "union":
            return union_feats if union_feats is not None else base_pair_feats
        if source == "refined_plus_union":
            if self.text_refined_union_query_proj is None:
                return refined_edge
            union_input = union_feats if union_feats is not None else torch.zeros_like(refined_edge)
            return self.text_refined_union_query_proj(torch.cat([refined_edge, union_input], dim=-1))
        raise ValueError(f"Unsupported graph_text_query_source: {source}")

    def _gather_pair_object_labels(
        self,
        labels: Tensor,
        pair_indices: Tensor,
        node_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        batch_size, num_nodes = labels.shape
        object_indices = pair_indices[..., 1]
        valid = (object_indices >= 0) & (object_indices < num_nodes)
        safe_indices = object_indices.clamp(min=0, max=max(num_nodes - 1, 0))
        gathered = torch.gather(labels, dim=1, index=safe_indices)
        valid = valid & torch.gather(node_mask, dim=1, index=safe_indices)
        gathered = torch.where(valid, gathered, torch.full_like(gathered, -1))
        return gathered, valid

    def _cross_attend_text_tokens(
        self,
        query: Tensor,
        text_tokens: Tensor,
        text_token_mask: Tensor,
        valid_pair_mask: Tensor,
    ) -> Tensor:
        context = query.new_zeros(query.shape)
        if self.text_cross_attn is None or text_tokens.numel() == 0:
            return context
        batch_size, num_pairs, seq_len, dim = text_tokens.shape
        if seq_len == 0:
            return context
        flat_query = query.reshape(batch_size * num_pairs, 1, dim)
        flat_tokens = text_tokens.reshape(batch_size * num_pairs, seq_len, dim)
        flat_mask = text_token_mask.reshape(batch_size * num_pairs, seq_len)
        flat_valid_pair = valid_pair_mask.reshape(batch_size * num_pairs)
        has_text = flat_valid_pair & flat_mask.any(dim=-1)
        if not bool(has_text.any().detach().cpu().item()):
            return context

        attn_out, _ = self.text_cross_attn(
            flat_query[has_text],
            flat_tokens[has_text],
            flat_tokens[has_text],
            key_padding_mask=~flat_mask[has_text],
            need_weights=False,
        )
        attn_out = attn_out.squeeze(1)
        if self.text_cross_norm is not None:
            attn_out = self.text_cross_norm(flat_query[has_text].squeeze(1) + attn_out) - flat_query[has_text].squeeze(1)
        if self.text_context_proj is not None:
            attn_out = self.text_context_proj(attn_out)
        attn_out = torch.nan_to_num(attn_out, nan=0.0, posinf=0.0, neginf=0.0)
        flat_context = context.reshape(batch_size * num_pairs, dim)
        flat_context[has_text] = attn_out
        flat_context = flat_context * flat_valid_pair.unsqueeze(-1).to(dtype=flat_context.dtype)
        return flat_context.reshape(batch_size, num_pairs, dim)

    def _empty_text_debug_info(self, like_tensor: Tensor) -> Dict[str, Any]:
        return {
            "text_graph_enabled": self.enable_text_graph,
            "text_gate": float(torch.tanh(self.text_gate).detach().cpu().item()),
            "text_query_source": self.graph_text_query_source,
            **self._empty_text_prior_debug_info(),
            "text_tokens_shape": None,
            "text_valid_token_count_mean": 0.0,
            "text_no_candidate_pair_count": 0,
            "text_valid_pair_count": 0,
            "text_selected_scores_mean": 0.0,
            "text_selected_scores_max": 0.0,
            "text_context_norm_mean": 0.0,
            "text_injection_norm": 0.0,
            "text_has_nan": bool(torch.isnan(like_tensor).any().detach().cpu().item()) if like_tensor.numel() > 0 else False,
        }

    def _empty_text_prior_debug_info(self) -> Dict[str, Any]:
        return {
            "text_prior_injection_mode": self.graph_text_prior_injection,
            "text_human_prior_norm": 0.0,
            "text_object_prior_norm": 0.0,
            "text_ho_edge_prior_norm": 0.0,
            "text_discriminative_prior_norm": 0.0,
            "text_scene_prior_norm": 0.0,
            "text_oo_prior_norm": 0.0,
            "text_node_prior_injection_norm": 0.0,
            "text_edge_prior_injection_norm": 0.0,
            "text_scene_prior_injection_norm": 0.0,
            "text_oo_prior_injection_norm": 0.0,
            "text_prior_has_nan": False,
        }

    def _project_text_prior(self, pooled: Tensor, proj: nn.Module, gate: Tensor, valid_pair_mask: Tensor) -> Tuple[Tensor, Tensor]:
        prior = proj(pooled)
        prior = torch.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
        injection = torch.tanh(gate).to(dtype=prior.dtype) * prior
        pooled_valid = pooled.detach().abs().sum(dim=-1) > 0
        prior_mask = (valid_pair_mask & pooled_valid).unsqueeze(-1).to(dtype=injection.dtype)
        prior = prior * prior_mask
        injection = injection * prior_mask
        return prior, injection

    def _inject_text_priors(
        self,
        text_out: Any,
        node_h: Tensor,
        node_o: Tensor,
        refined_edge: Tensor,
        valid_pair_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
        debug_info = self._empty_text_prior_debug_info()
        if self.graph_text_prior_injection == "none":
            return node_h, node_o, refined_edge, debug_info

        node_injections = []
        edge_injections = []
        scene_injections = []
        oo_injections = []
        prior_tensors = []
        use_nodes = self.graph_text_prior_injection in {"node_edge", "all"}
        use_edges = self.graph_text_prior_injection in {"edge_only", "node_edge", "all"}
        use_all = self.graph_text_prior_injection == "all"

        if use_nodes and self.graph_text_use_human_node_prior:
            prior, inj = self._project_text_prior(text_out.pooled_human_node, self.text_human_node_proj, self.text_human_node_gate, valid_pair_mask)
            node_h = node_h + inj
            node_injections.append(inj)
            prior_tensors.append(prior)
            debug_info["text_human_prior_norm"] = self._norm_mean(prior)
        if use_nodes and self.graph_text_use_object_node_prior:
            prior, inj = self._project_text_prior(text_out.pooled_object_node, self.text_object_node_proj, self.text_object_node_gate, valid_pair_mask)
            node_o = node_o + inj
            node_injections.append(inj)
            prior_tensors.append(prior)
            debug_info["text_object_prior_norm"] = self._norm_mean(prior)
        if use_edges and self.graph_text_use_ho_edge_prior:
            prior, inj = self._project_text_prior(text_out.pooled_ho_edge, self.text_ho_edge_proj, self.text_ho_edge_gate, valid_pair_mask)
            refined_edge = refined_edge + inj
            edge_injections.append(inj)
            prior_tensors.append(prior)
            debug_info["text_ho_edge_prior_norm"] = self._norm_mean(prior)
        if use_all and self.graph_text_use_discriminative_prior:
            prior, inj = self._project_text_prior(text_out.pooled_discriminative, self.text_discriminative_proj, self.text_discriminative_gate, valid_pair_mask)
            refined_edge = refined_edge + inj
            edge_injections.append(inj)
            prior_tensors.append(prior)
            debug_info["text_discriminative_prior_norm"] = self._norm_mean(prior)
        if use_all and self.graph_text_use_scene_prior:
            prior, inj = self._project_text_prior(text_out.pooled_scene_context, self.text_scene_proj, self.text_scene_gate, valid_pair_mask)
            refined_edge = refined_edge + inj
            scene_injections.append(inj)
            prior_tensors.append(prior)
            debug_info["text_scene_prior_norm"] = self._norm_mean(prior)
        if use_all and self.graph_text_use_oo_prior:
            prior, inj = self._project_text_prior(text_out.pooled_oo_context, self.text_oo_proj, self.text_oo_gate, valid_pair_mask)
            refined_edge = refined_edge + inj
            oo_injections.append(inj)
            prior_tensors.append(prior)
            debug_info["text_oo_prior_norm"] = self._norm_mean(prior)

        debug_info["text_node_prior_injection_norm"] = self._combined_norm_mean(node_injections)
        debug_info["text_edge_prior_injection_norm"] = self._combined_norm_mean(edge_injections)
        debug_info["text_scene_prior_injection_norm"] = self._combined_norm_mean(scene_injections)
        debug_info["text_oo_prior_injection_norm"] = self._combined_norm_mean(oo_injections)
        debug_info["text_prior_has_nan"] = any(
            bool(torch.isnan(tensor).any().detach().cpu().item()) for tensor in prior_tensors + node_injections + edge_injections + scene_injections + oo_injections
        )
        return node_h, node_o, refined_edge, debug_info

    def _norm_mean(self, tensor: Tensor) -> float:
        return float(tensor.detach().float().norm(dim=-1).mean().item()) if tensor.numel() > 0 else 0.0

    def _combined_norm_mean(self, tensors: List[Tensor]) -> float:
        if not tensors:
            return 0.0
        total = torch.stack([tensor.detach().float().norm(dim=-1) for tensor in tensors], dim=0).sum(dim=0)
        return float(total.mean().item()) if total.numel() > 0 else 0.0

    def _build_text_debug_info(
        self,
        text_out: Any,
        text_context: Tensor,
        text_injection: Tensor,
        text_pair_valid_mask: Tensor,
        text_prior_debug_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        selected_valid = text_out.selected_hoi_mask
        if bool(selected_valid.any().detach().cpu().item()):
            valid_scores = text_out.selected_scores[selected_valid].detach().float()
            score_mean = float(valid_scores.mean().item())
            score_max = float(valid_scores.max().item())
        else:
            score_mean = 0.0
            score_max = 0.0
        if text_context.numel() > 0:
            context_norm_mean = float(text_context.detach().float().norm(dim=-1).mean().item())
            injection_norm = float(text_injection.detach().float().norm(dim=-1).mean().item())
        else:
            context_norm_mean = 0.0
            injection_norm = 0.0
        no_candidate_pair_count = int(
            (text_pair_valid_mask & ~text_out.selected_hoi_mask.any(dim=-1)).sum().detach().cpu().item()
        ) if text_out.selected_hoi_mask.numel() > 0 else int(text_pair_valid_mask.sum().detach().cpu().item())
        debug_info = {
            "text_graph_enabled": True,
            "text_gate": float(torch.tanh(self.text_gate).detach().cpu().item()),
            "text_query_source": self.graph_text_query_source,
            **text_prior_debug_info,
            "text_tokens_shape": tuple(text_out.text_tokens.shape),
            "text_valid_token_count_mean": text_out.debug_info.get("text_valid_token_count_mean", 0.0),
            "text_no_candidate_pair_count": no_candidate_pair_count,
            "text_valid_pair_count": int(text_pair_valid_mask.sum().detach().cpu().item()),
            "text_selected_scores_mean": score_mean,
            "text_selected_scores_max": score_max,
            "text_context_norm_mean": context_norm_mean,
            "text_injection_norm": injection_norm,
            "text_has_nan": bool(
                text_out.debug_info.get("text_has_nan", False)
                or torch.isnan(text_context).any().detach().cpu().item()
                or torch.isnan(text_injection).any().detach().cpu().item()
                or text_prior_debug_info.get("text_prior_has_nan", False)
            ),
        }
        if self.graph_text_debug and "selected_hoi_ids_preview" in text_out.debug_info:
            debug_info["text_selected_hoi_ids_preview"] = text_out.debug_info["selected_hoi_ids_preview"]
        return debug_info

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
        "graph_text_num_layers": getattr(args, "graph_text_num_layers", 1),
        "graph_text_num_heads": getattr(args, "graph_text_num_heads", 8),
        "graph_text_dropout": getattr(args, "graph_text_dropout", 0.1),
        "graph_text_use_scene_token": getattr(args, "graph_text_use_scene_token", True),
        "graph_text_use_oo_context_token": getattr(args, "graph_text_use_oo_context_token", True),
        "graph_text_cross_num_heads": getattr(args, "graph_text_cross_num_heads", 8),
        "graph_text_query_source": getattr(args, "graph_text_query_source", "refined_edge"),
        "graph_text_gate_init": getattr(args, "graph_text_gate_init", 0.0),
        "graph_text_debug": getattr(args, "graph_text_debug", getattr(args, "graph_debug", False)),
        "graph_text_prior_injection": getattr(args, "graph_text_prior_injection", "none"),
        "graph_text_use_human_node_prior": getattr(args, "graph_text_use_human_node_prior", True),
        "graph_text_use_object_node_prior": getattr(args, "graph_text_use_object_node_prior", True),
        "graph_text_use_ho_edge_prior": getattr(args, "graph_text_use_ho_edge_prior", True),
        "graph_text_use_discriminative_prior": getattr(args, "graph_text_use_discriminative_prior", False),
        "graph_text_use_scene_prior": getattr(args, "graph_text_use_scene_prior", False),
        "graph_text_use_oo_prior": getattr(args, "graph_text_use_oo_prior", False),
        "graph_text_node_prior_gate_init": getattr(args, "graph_text_node_prior_gate_init", 0.0),
        "graph_text_edge_prior_gate_init": getattr(args, "graph_text_edge_prior_gate_init", 0.0),
        "graph_text_scene_prior_gate_init": getattr(args, "graph_text_scene_prior_gate_init", 0.0),
        "graph_text_oo_prior_gate_init": getattr(args, "graph_text_oo_prior_gate_init", 0.0),
    }
    config.update(kwargs)
    return VisualSceneGraphReasoner(**config)
