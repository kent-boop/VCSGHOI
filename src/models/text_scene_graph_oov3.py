"""Text scene graph oov3 with structured prototype slot outputs."""

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import Tensor


def _load_text_scene_graph_oov2_classes():
    try:
        from .text_scene_graph_oov2 import (
            ExplicitTextGraphState,
            TextCandidateOutput,
            TextGraphFeatureBank,
            TextGraphFeatureOutput,
            TextHOIRetriever,
            TextSceneGraphReasoner,
        )
    except ImportError:
        module_path = Path(__file__).resolve().with_name("text_scene_graph_oov2.py")
        spec = importlib.util.spec_from_file_location("text_scene_graph_oov2_direct", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        ExplicitTextGraphState = module.ExplicitTextGraphState
        TextCandidateOutput = module.TextCandidateOutput
        TextGraphFeatureBank = module.TextGraphFeatureBank
        TextGraphFeatureOutput = module.TextGraphFeatureOutput
        TextHOIRetriever = module.TextHOIRetriever
        TextSceneGraphReasoner = module.TextSceneGraphReasoner
    return (
        ExplicitTextGraphState,
        TextCandidateOutput,
        TextGraphFeatureBank,
        TextGraphFeatureOutput,
        TextHOIRetriever,
        TextSceneGraphReasoner,
    )


def _load_text_prototype_slot_builder():
    try:
        from .text_prototype_slots import TextPrototypeSlotBuilder
    except ImportError:
        module_path = Path(__file__).resolve().with_name("text_prototype_slots.py")
        spec = importlib.util.spec_from_file_location("text_prototype_slots_direct", module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        TextPrototypeSlotBuilder = module.TextPrototypeSlotBuilder
    return TextPrototypeSlotBuilder


(
    ExplicitTextGraphState,
    TextCandidateOutput,
    TextGraphFeatureBank,
    TextGraphFeatureOutput,
    TextHOIRetriever,
    _BaseTextSceneGraphReasoner,
) = _load_text_scene_graph_oov2_classes()


@dataclass
class TextGraphOutput:
    text_tokens: Tensor
    text_token_mask: Tensor
    selected_hoi_indices: Tensor
    selected_hoi_mask: Tensor
    selected_scores: Tensor
    pooled_human_node: Tensor
    pooled_object_node: Tensor
    pooled_ho_edge: Tensor
    pooled_discriminative: Tensor
    pooled_scene_context: Tensor
    pooled_oo_context: Tensor
    pooled_all: Tensor
    prototype_slots: Tensor
    prototype_slot_mask: Tensor
    debug_info: Dict[str, Any]


class TextSceneGraphReasoner(_BaseTextSceneGraphReasoner):
    def __init__(
        self,
        *args: Any,
        enable_prototype_slots: bool = True,
        prototype_slot_fallback_to_raw: bool = True,
        prototype_slot_debug: bool = False,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self.enable_prototype_slots = bool(enable_prototype_slots)
        self.prototype_slot_fallback_to_raw = bool(prototype_slot_fallback_to_raw)
        self.prototype_slot_debug = bool(prototype_slot_debug)
        self.prototype_slot_builder = None
        if self.enable_prototype_slots:
            TextPrototypeSlotBuilder = _load_text_prototype_slot_builder()
            self.prototype_slot_builder = TextPrototypeSlotBuilder(
                d_model=self.d_model,
                fallback_to_raw_fields=self.prototype_slot_fallback_to_raw,
                sanitize=True,
                debug=self.prototype_slot_debug or self.debug,
            )

    def export_layer_outputs(self, state: ExplicitTextGraphState) -> TextGraphOutput:
        batch_size, num_pairs, topk, num_nodes, dim = state.node_feats.shape
        node_tokens = state.node_feats.reshape(batch_size, num_pairs, topk * num_nodes, dim)
        node_token_mask = state.node_mask.reshape(batch_size, num_pairs, topk * num_nodes)
        edge_tokens = state.edge_attr.reshape(batch_size, num_pairs, topk * num_nodes * num_nodes, dim)
        edge_token_mask = state.edge_mask.reshape(batch_size, num_pairs, topk * num_nodes * num_nodes)
        text_tokens = torch.cat([node_tokens, edge_tokens], dim=-2)
        text_token_mask = torch.cat([node_token_mask, edge_token_mask], dim=-1)
        text_tokens = torch.nan_to_num(text_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        text_tokens = text_tokens * text_token_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)

        pooled_human_node = self._weighted_pool_candidates(state.node_feats[..., 0, :], state.node_mask[..., 0], state)
        pooled_object_node = self._weighted_pool_candidates(state.node_feats[..., 1, :], state.node_mask[..., 1], state)
        pooled_ho_edge = self._weighted_pool_candidates(state.edge_attr[..., 0, 1, :], state.edge_mask[..., 0, 1], state)
        pooled_discriminative = self._weighted_pool_field(state, "discriminative")

        pooled_scene_node = self._weighted_pool_candidates(state.node_feats[..., 2, :], state.node_mask[..., 2], state)
        pooled_scene_raw = self._weighted_pool_field(state, "scene_context")
        has_scene_node = state.node_mask[..., 2].any(dim=-1, keepdim=True)
        pooled_scene_context = torch.where(has_scene_node, pooled_scene_node, pooled_scene_raw)

        pooled_oo_node = self._weighted_pool_candidates(state.node_feats[..., 3, :], state.node_mask[..., 3], state)
        pooled_oo_raw = self._weighted_pool_field(state, "oo_context")
        has_oo_node = state.node_mask[..., 3].any(dim=-1, keepdim=True)
        pooled_oo_context = torch.where(has_oo_node, pooled_oo_node, pooled_oo_raw)
        pooled_all = self._masked_average(text_tokens, text_token_mask)

        if self.prototype_slot_builder is not None:
            prototype_slot_out = self.prototype_slot_builder(
                node_feats=state.node_feats,
                node_mask=state.node_mask,
                edge_attr=state.edge_attr,
                edge_mask=state.edge_mask,
                raw_field_feats=state.raw_field_feats,
                raw_field_mask=state.raw_field_mask,
                selected_hoi_mask=state.selected_hoi_mask,
                pair_valid_mask=state.pair_valid_mask,
                selected_hoi_indices=state.selected_hoi_indices,
            )
            prototype_slots = prototype_slot_out.prototype_slots
            prototype_slot_mask = prototype_slot_out.prototype_slot_mask
            prototype_slot_debug = prototype_slot_out.debug_info
            prototype_slot_debug["prototype_slot_builder_enabled"] = True
        else:
            prototype_slots = state.raw_field_feats
            prototype_slot_mask = (
                state.raw_field_mask
                & state.selected_hoi_mask.unsqueeze(-1)
                & state.pair_valid_mask.unsqueeze(-1).unsqueeze(-1)
            )
            prototype_slots = prototype_slots * prototype_slot_mask.unsqueeze(-1).to(dtype=prototype_slots.dtype)
            prototype_slots = torch.nan_to_num(prototype_slots, nan=0.0, posinf=0.0, neginf=0.0)
            prototype_slot_debug = {
                "prototype_slots_shape": tuple(prototype_slots.shape),
                "prototype_slot_mask_shape": tuple(prototype_slot_mask.shape),
                "prototype_slot_valid_ratio": (
                    float(prototype_slot_mask.float().mean().detach().cpu().item())
                    if prototype_slot_mask.numel() > 0
                    else 0.0
                ),
                "prototype_valid_count": int((state.selected_hoi_mask & state.pair_valid_mask.unsqueeze(-1)).sum().detach().cpu().item()),
                "prototype_slot_valid_count_per_slot": prototype_slot_mask.sum(dim=(0, 1, 2)).detach().cpu().tolist(),
                "prototype_slot_names": [
                    "human_state",
                    "object_state",
                    "anchor_relation",
                    "discriminative",
                    "scene",
                    "context_object",
                ],
                "prototype_slot_has_nan": bool(torch.isnan(prototype_slots).any().detach().cpu().item()) if prototype_slots.numel() > 0 else False,
                "fallback_to_raw_fields": False,
                "prototype_slot_builder_enabled": False,
            }

        clear_scene_slot = not self.use_scene_token
        clear_oo_context_slot = not self.use_oo_context_token
        if clear_scene_slot or clear_oo_context_slot:
            prototype_slot_mask = prototype_slot_mask.clone()
            prototype_slots = prototype_slots.clone()
            if clear_scene_slot:
                prototype_slot_mask[..., 4] = False
                prototype_slots[..., 4, :] = 0.0
            if clear_oo_context_slot:
                prototype_slot_mask[..., 5] = False
                prototype_slots[..., 5, :] = 0.0

        prototype_slot_debug = dict(prototype_slot_debug)
        prototype_slot_debug["prototype_slot_valid_ratio"] = (
            float(prototype_slot_mask.float().mean().detach().cpu().item())
            if prototype_slot_mask.numel() > 0
            else 0.0
        )
        prototype_slot_debug["prototype_slot_valid_count_per_slot"] = prototype_slot_mask.sum(dim=(0, 1, 2)).detach().cpu().tolist()
        prototype_slot_debug["prototype_slot_has_nan"] = (
            bool(torch.isnan(prototype_slots).any().detach().cpu().item()) if prototype_slots.numel() > 0 else False
        )

        debug_info = dict(state.debug_info)
        debug_info.update(self._build_debug_info(
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            selected_hoi_indices=state.selected_hoi_indices,
            selected_hoi_mask=state.selected_hoi_mask,
            selected_scores=state.selected_scores,
            pooled_by_field={
                "human_node": pooled_human_node,
                "object_node": pooled_object_node,
                "ho_edge": pooled_ho_edge,
                "discriminative": pooled_discriminative,
                "scene_context": pooled_scene_context,
                "oo_context": pooled_oo_context,
            },
            pooled_all=pooled_all,
            topk=topk,
            num_fields=num_nodes + num_nodes * num_nodes,
        ))
        debug_info.update(prototype_slot_debug)
        self.last_debug = debug_info
        return TextGraphOutput(
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            selected_hoi_indices=state.selected_hoi_indices,
            selected_hoi_mask=state.selected_hoi_mask,
            selected_scores=state.selected_scores,
            pooled_human_node=pooled_human_node,
            pooled_object_node=pooled_object_node,
            pooled_ho_edge=pooled_ho_edge,
            pooled_discriminative=pooled_discriminative,
            pooled_scene_context=pooled_scene_context,
            pooled_oo_context=pooled_oo_context,
            pooled_all=pooled_all,
            prototype_slots=prototype_slots,
            prototype_slot_mask=prototype_slot_mask,
            debug_info=debug_info,
        )
