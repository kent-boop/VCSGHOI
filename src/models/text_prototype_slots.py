from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class PrototypeSlotOutput:
    prototype_slots: Tensor
    prototype_slot_mask: Tensor
    debug_info: Dict[str, Any]


class TextPrototypeSlotBuilder(nn.Module):
    """
    Build structured language-derived interaction prototype slots.

    Slot order:
        0: human_state        from human_node
        1: object_state       from object_node
        2: anchor_relation    from ho_edge / HO edge
        3: discriminative     from discriminative field
        4: scene              from scene_context / scene node
        5: context_object     from oo_context / context node
    """

    SLOT_NAMES = (
        "human_state",
        "object_state",
        "anchor_relation",
        "discriminative",
        "scene",
        "context_object",
    )

    def __init__(
        self,
        d_model: int = 256,
        fallback_to_raw_fields: bool = True,
        sanitize: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.d_model = int(d_model)
        self.fallback_to_raw_fields = bool(fallback_to_raw_fields)
        self.sanitize = bool(sanitize)
        self.debug = bool(debug)

    def forward(
        self,
        node_feats: Tensor,
        node_mask: Tensor,
        edge_attr: Tensor,
        edge_mask: Tensor,
        raw_field_feats: Tensor,
        raw_field_mask: Tensor,
        selected_hoi_mask: Tensor,
        pair_valid_mask: Tensor,
        selected_hoi_indices: Optional[Tensor] = None,
    ) -> PrototypeSlotOutput:
        self._check_inputs(
            node_feats=node_feats,
            node_mask=node_mask,
            edge_attr=edge_attr,
            edge_mask=edge_mask,
            raw_field_feats=raw_field_feats,
            raw_field_mask=raw_field_mask,
            selected_hoi_mask=selected_hoi_mask,
            pair_valid_mask=pair_valid_mask,
            selected_hoi_indices=selected_hoi_indices,
        )

        if self.sanitize:
            node_feats = torch.nan_to_num(node_feats, nan=0.0, posinf=0.0, neginf=0.0)
            edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
            raw_field_feats = torch.nan_to_num(raw_field_feats, nan=0.0, posinf=0.0, neginf=0.0)

        node_mask = node_mask.to(dtype=torch.bool)
        edge_mask = edge_mask.to(dtype=torch.bool)
        raw_field_mask = raw_field_mask.to(dtype=torch.bool)
        selected_hoi_mask = selected_hoi_mask.to(dtype=torch.bool)
        pair_valid_mask = pair_valid_mask.to(dtype=torch.bool)

        human_slot, human_mask = self._choose(
            node_feats[..., 0, :],
            node_mask[..., 0],
            raw_field_feats[..., 0, :],
            raw_field_mask[..., 0],
        )
        object_slot, object_mask = self._choose(
            node_feats[..., 1, :],
            node_mask[..., 1],
            raw_field_feats[..., 1, :],
            raw_field_mask[..., 1],
        )
        anchor_slot, anchor_mask = self._choose(
            edge_attr[..., 0, 1, :],
            edge_mask[..., 0, 1],
            raw_field_feats[..., 2, :],
            raw_field_mask[..., 2],
        )
        disc_slot, disc_mask = raw_field_feats[..., 3, :], raw_field_mask[..., 3]
        scene_slot, scene_mask = self._choose(
            node_feats[..., 2, :],
            node_mask[..., 2],
            raw_field_feats[..., 4, :],
            raw_field_mask[..., 4],
        )
        ctx_slot, ctx_mask = self._choose(
            node_feats[..., 3, :],
            node_mask[..., 3],
            raw_field_feats[..., 5, :],
            raw_field_mask[..., 5],
        )

        prototype_slots = torch.stack([
            human_slot,
            object_slot,
            anchor_slot,
            disc_slot,
            scene_slot,
            ctx_slot,
        ], dim=-2)
        prototype_slot_mask = torch.stack([
            human_mask,
            object_mask,
            anchor_mask,
            disc_mask,
            scene_mask,
            ctx_mask,
        ], dim=-1)

        valid_proto = selected_hoi_mask & pair_valid_mask.unsqueeze(-1)
        prototype_slot_mask = prototype_slot_mask & valid_proto.unsqueeze(-1)
        prototype_slots = prototype_slots * prototype_slot_mask.unsqueeze(-1).to(dtype=prototype_slots.dtype)
        prototype_slots = torch.nan_to_num(prototype_slots, nan=0.0, posinf=0.0, neginf=0.0)

        debug_info = self._build_debug_info(
            prototype_slots=prototype_slots,
            prototype_slot_mask=prototype_slot_mask,
            valid_proto=valid_proto,
        )
        return PrototypeSlotOutput(
            prototype_slots=prototype_slots,
            prototype_slot_mask=prototype_slot_mask,
            debug_info=debug_info,
        )

    def get_slot_names(self) -> Tuple[str, ...]:
        return self.SLOT_NAMES

    def _choose(
        self,
        primary_value: Tensor,
        primary_mask: Tensor,
        fallback_value: Tensor,
        fallback_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        primary_mask = primary_mask.to(dtype=torch.bool)
        fallback_mask = fallback_mask.to(dtype=torch.bool)
        if self.fallback_to_raw_fields:
            use_primary = primary_mask.unsqueeze(-1)
            out_value = torch.where(use_primary, primary_value, fallback_value)
            out_mask = primary_mask | fallback_mask
        else:
            out_value = primary_value
            out_mask = primary_mask
        return out_value, out_mask

    def _build_debug_info(
        self,
        prototype_slots: Tensor,
        prototype_slot_mask: Tensor,
        valid_proto: Tensor,
    ) -> Dict[str, Any]:
        valid_ratio = (
            float(prototype_slot_mask.float().mean().detach().cpu().item())
            if prototype_slot_mask.numel() > 0
            else 0.0
        )
        valid_count = int(valid_proto.sum().detach().cpu().item()) if valid_proto.numel() > 0 else 0
        per_slot = prototype_slot_mask.sum(dim=(0, 1, 2)).detach().cpu().tolist()
        has_nan = bool(torch.isnan(prototype_slots).any().detach().cpu().item()) if prototype_slots.numel() > 0 else False
        return {
            "prototype_slots_shape": tuple(prototype_slots.shape),
            "prototype_slot_mask_shape": tuple(prototype_slot_mask.shape),
            "prototype_slot_valid_ratio": valid_ratio,
            "prototype_valid_count": valid_count,
            "prototype_slot_valid_count_per_slot": per_slot,
            "prototype_slot_names": list(self.SLOT_NAMES),
            "prototype_slot_has_nan": has_nan,
            "fallback_to_raw_fields": self.fallback_to_raw_fields,
        }

    def _check_inputs(
        self,
        node_feats: Tensor,
        node_mask: Tensor,
        edge_attr: Tensor,
        edge_mask: Tensor,
        raw_field_feats: Tensor,
        raw_field_mask: Tensor,
        selected_hoi_mask: Tensor,
        pair_valid_mask: Tensor,
        selected_hoi_indices: Optional[Tensor],
    ) -> None:
        if node_feats.dim() != 5:
            raise ValueError(f"node_feats must be [B,K,R,4,D] or [B,K,R,>=4,D], got {tuple(node_feats.shape)}")
        if node_mask.dim() != 4:
            raise ValueError(f"node_mask must be [B,K,R,4] or [B,K,R,>=4], got {tuple(node_mask.shape)}")
        if edge_attr.dim() != 6:
            raise ValueError(f"edge_attr must be [B,K,R,4,4,D] or [B,K,R,>=4,>=4,D], got {tuple(edge_attr.shape)}")
        if edge_mask.dim() != 5:
            raise ValueError(f"edge_mask must be [B,K,R,4,4] or [B,K,R,>=4,>=4], got {tuple(edge_mask.shape)}")
        if raw_field_feats.dim() != 5:
            raise ValueError(f"raw_field_feats must be [B,K,R,6,D], got {tuple(raw_field_feats.shape)}")
        if raw_field_mask.dim() != 4:
            raise ValueError(f"raw_field_mask must be [B,K,R,6], got {tuple(raw_field_mask.shape)}")
        if selected_hoi_mask.dim() != 3:
            raise ValueError(f"selected_hoi_mask must be [B,K,R], got {tuple(selected_hoi_mask.shape)}")
        if pair_valid_mask.dim() != 2:
            raise ValueError(f"pair_valid_mask must be [B,K], got {tuple(pair_valid_mask.shape)}")

        batch_size, num_pairs, num_proto, num_nodes, dim = node_feats.shape
        if dim != self.d_model:
            raise ValueError(f"node_feats D must equal d_model={self.d_model}, got {dim}")
        if num_nodes < 4:
            raise ValueError(f"node_feats fourth dimension must be at least 4, got {num_nodes}")
        if tuple(node_mask.shape[:3]) != (batch_size, num_pairs, num_proto) or node_mask.shape[3] < 4:
            raise ValueError(
                "node_mask must share [B,K,R] with node_feats and have fourth dimension >= 4, "
                f"got {tuple(node_mask.shape)}"
            )
        if tuple(edge_attr.shape[:3]) != (batch_size, num_pairs, num_proto):
            raise ValueError(f"edge_attr must share [B,K,R] with node_feats, got {tuple(edge_attr.shape)}")
        if edge_attr.shape[3] < 4 or edge_attr.shape[4] < 4:
            raise ValueError(f"edge_attr fourth and fifth dimensions must be >= 4, got {tuple(edge_attr.shape[3:5])}")
        if edge_attr.shape[-1] != self.d_model:
            raise ValueError(f"edge_attr D must equal d_model={self.d_model}, got {edge_attr.shape[-1]}")
        if tuple(edge_mask.shape[:3]) != (batch_size, num_pairs, num_proto):
            raise ValueError(f"edge_mask must share [B,K,R] with node_feats, got {tuple(edge_mask.shape)}")
        if edge_mask.shape[3] < 4 or edge_mask.shape[4] < 4:
            raise ValueError(f"edge_mask fourth and fifth dimensions must be >= 4, got {tuple(edge_mask.shape[3:5])}")
        if tuple(raw_field_feats.shape) != (batch_size, num_pairs, num_proto, 6, self.d_model):
            raise ValueError(
                f"raw_field_feats must be {(batch_size, num_pairs, num_proto, 6, self.d_model)}, "
                f"got {tuple(raw_field_feats.shape)}"
            )
        if tuple(raw_field_mask.shape) != (batch_size, num_pairs, num_proto, 6):
            raise ValueError(
                f"raw_field_mask must be {(batch_size, num_pairs, num_proto, 6)}, got {tuple(raw_field_mask.shape)}"
            )
        if tuple(selected_hoi_mask.shape) != (batch_size, num_pairs, num_proto):
            raise ValueError(
                f"selected_hoi_mask must be {(batch_size, num_pairs, num_proto)}, got {tuple(selected_hoi_mask.shape)}"
            )
        if tuple(pair_valid_mask.shape) != (batch_size, num_pairs):
            raise ValueError(f"pair_valid_mask must be {(batch_size, num_pairs)}, got {tuple(pair_valid_mask.shape)}")
        if selected_hoi_indices is not None and tuple(selected_hoi_indices.shape) != (batch_size, num_pairs, num_proto):
            raise ValueError(
                f"selected_hoi_indices must be {(batch_size, num_pairs, num_proto)}, "
                f"got {tuple(selected_hoi_indices.shape)}"
            )
