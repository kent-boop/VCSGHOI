import json
import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class TextGraphFeatureOutput:
    features: Tensor
    mask: Tensor


@dataclass
class TextCandidateOutput:
    selected_hoi_indices: Tensor
    selected_hoi_mask: Tensor
    selected_scores: Tensor
    coarse_scores: Optional[Tensor]
    rerank_scores: Optional[Tensor]
    pair_object_ids: Tensor
    pair_object_valid_mask: Tensor
    debug_info: Dict[str, Any]


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
    debug_info: Dict[str, Any]


class TextGraphFeatureBank(nn.Module):
    """Feature bank for offline HICO text graph embeddings.

    This module only loads precomputed JSON/NPY assets during initialization.
    Forward-time callers should use the query methods, which operate on buffers.
    """

    DEFAULT_FIELD_ORDER = [
        "human_node",
        "object_node",
        "ho_edge",
        "discriminative",
        "scene_context",
        "oo_context",
    ]

    OPTIONAL_FIELD_ATTRS = {
        "human_node": "human_node_features",
        "object_node": "object_node_features",
        "ho_edge": "ho_edge_features",
        "discriminative": "discriminative_features",
        "scene_context": "scene_context_features",
        "oo_context": "oo_context_features",
    }

    def __init__(
        self,
        text_features_path: str,
        text_mask_path: str,
        hoi_ids_path: str,
        object_to_hoi_indices_path: str,
        field_order_path: str,
        d_model: int = 256,
        normalize: bool = True,
        strict_shape: bool = True,
        strict_consistency: bool = False,
        human_node_features_path: str = "",
        object_node_features_path: str = "",
        ho_edge_features_path: str = "",
        discriminative_features_path: str = "",
        scene_context_features_path: str = "",
        oo_context_features_path: str = "",
        oo_affordance_prior_path: str = "",
        label_ids: Optional[List[int]] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.normalize = bool(normalize)
        self.strict_shape = bool(strict_shape)
        self.strict_consistency = bool(strict_consistency)
        self.debug = bool(debug)

        field_order = self._load_json(field_order_path, "field_order")
        self._check_field_order(field_order)
        self.index_to_field = {idx: name for idx, name in enumerate(self.DEFAULT_FIELD_ORDER)}
        self.field_to_index = {name: idx for idx, name in self.index_to_field.items()}

        self.hoi_ids = self._load_json(hoi_ids_path, "hoi_ids")
        if not isinstance(self.hoi_ids, list) or not all(isinstance(item, str) for item in self.hoi_ids):
            raise ValueError("hoi_ids must be a List[str]")
        self.hoi_to_index = {hoi_id: idx for idx, hoi_id in enumerate(self.hoi_ids)}
        if len(self.hoi_to_index) != len(self.hoi_ids):
            raise ValueError("hoi_ids contains duplicate entries")

        text_features = self._load_npy(text_features_path, "text_features").to(dtype=torch.float32)
        text_mask = self._load_npy(text_mask_path, "text_mask")
        self._check_main_shapes(text_features, text_mask)

        text_mask = text_mask.to(dtype=torch.bool)
        text_features = self._sanitize_tensor(text_features, "text_features")
        raw_text_features = text_features * text_mask.unsqueeze(-1).to(dtype=text_features.dtype)
        text_features = raw_text_features
        if self.normalize:
            text_features = F.normalize(text_features, dim=-1, eps=1e-6)
            text_features = text_features * text_mask.unsqueeze(-1).to(dtype=text_features.dtype)

        self.register_buffer("text_features", text_features)
        self.register_buffer("text_mask", text_mask)

        optional_paths = {
            "human_node": human_node_features_path,
            "object_node": object_node_features_path,
            "ho_edge": ho_edge_features_path,
            "discriminative": discriminative_features_path,
            "scene_context": scene_context_features_path,
            "oo_context": oo_context_features_path,
        }
        for field_name, attr_name in self.OPTIONAL_FIELD_ATTRS.items():
            field_idx = self.field_to_index[field_name]
            field_features = text_features[:, field_idx, :].clone()
            optional_path = optional_paths[field_name]
            if optional_path:
                optional_features = self._load_optional_field(optional_path, field_name, raw_text_features[:, field_idx, :])
                if optional_features is not None and self.debug:
                    warnings.warn(
                        f"Loaded optional {field_name} features for consistency check; main 6x256 features remain authoritative.",
                        RuntimeWarning,
                    )
            self.register_buffer(attr_name, field_features)

        action_ids, object_ids = self._parse_hoi_ids(self.hoi_ids)
        self.register_buffer("hoi_action_ids", torch.tensor(action_ids, dtype=torch.long))
        self.register_buffer("hoi_object_ids", torch.tensor(object_ids, dtype=torch.long))

        object_to_raw = self._load_json(object_to_hoi_indices_path, "object_to_hoi_indices")
        self.object_to_hoi_indices = self._parse_object_to_hoi_indices(object_to_raw, object_ids)

        label_to_hico_object_id = self._build_label_mapping(label_ids)
        self.register_buffer("label_to_hico_object_id", label_to_hico_object_id)

        oo_prior = self._load_oo_prior(oo_affordance_prior_path)
        self.register_buffer("oo_affordance_prior", oo_prior)
        self.oo_prior_loaded = bool(oo_affordance_prior_path)

    def map_labels_to_hico_object_ids(self, labels: Tensor) -> Tuple[Tensor, Tensor]:
        labels_long = labels.to(dtype=torch.long)
        if self.label_to_hico_object_id.numel() == 0:
            valid_mask = (labels_long >= 0) & (labels_long < 80)
            hico_object_ids = labels_long.clamp(min=0, max=79)
            hico_object_ids = torch.where(valid_mask, hico_object_ids, torch.full_like(hico_object_ids, -1))
            return hico_object_ids, valid_mask

        mapping = self.label_to_hico_object_id.to(device=labels.device)
        valid_range = (labels_long >= 0) & (labels_long < mapping.shape[0])
        safe_labels = labels_long.clamp(min=0, max=max(mapping.shape[0] - 1, 0))
        mapped = mapping[safe_labels]
        valid_mask = valid_range & (mapped >= 0)
        mapped = torch.where(valid_mask, mapped, torch.full_like(mapped, -1))
        return mapped, valid_mask

    def get_candidate_hoi_indices(self, object_ids: Tensor) -> Tuple[Tensor, Tensor]:
        object_ids_cpu = object_ids.detach().to(dtype=torch.long, device="cpu")
        flat_objects = object_ids_cpu.reshape(-1).tolist()
        candidate_lists = [self.object_to_hoi_indices.get(int(object_id), []) for object_id in flat_objects]
        max_candidates = max((len(items) for items in candidate_lists), default=0)

        out_shape = tuple(object_ids.shape) + (max_candidates,)
        candidates = torch.full(out_shape, -1, dtype=torch.long, device=object_ids.device)
        mask = torch.zeros(out_shape, dtype=torch.bool, device=object_ids.device)
        if max_candidates == 0:
            return candidates, mask

        flat_candidates = candidates.reshape(-1, max_candidates)
        flat_mask = mask.reshape(-1, max_candidates)
        for row, items in enumerate(candidate_lists):
            if not items:
                continue
            values = torch.tensor(items, dtype=torch.long, device=object_ids.device)
            flat_candidates[row, : values.numel()] = values
            flat_mask[row, : values.numel()] = True
        return candidates, mask

    def gather_fields(self, hoi_indices: Tensor, field_names: List[str]) -> Tuple[Tensor, Tensor]:
        if not field_names:
            feature_shape = tuple(hoi_indices.shape) + (0, self.d_model)
            mask_shape = tuple(hoi_indices.shape) + (0,)
            return (
                self.text_features.new_zeros(feature_shape),
                torch.zeros(mask_shape, dtype=torch.bool, device=hoi_indices.device),
            )

        field_indices = []
        for field_name in field_names:
            if field_name not in self.field_to_index:
                raise KeyError(f"Unknown text graph field: {field_name}")
            field_indices.append(self.field_to_index[field_name])

        hoi_long = hoi_indices.to(dtype=torch.long)
        num_hoi = self.text_features.shape[0]
        valid_hoi = (hoi_long >= 0) & (hoi_long < num_hoi)
        safe_hoi = hoi_long.clamp(min=0, max=max(num_hoi - 1, 0))

        text_features = self.text_features.to(device=hoi_indices.device)
        text_mask = self.text_mask.to(device=hoi_indices.device)
        field_index_tensor = torch.tensor(field_indices, dtype=torch.long, device=hoi_indices.device)

        gathered_all_fields = text_features[safe_hoi]
        gathered_all_masks = text_mask[safe_hoi]
        features = gathered_all_fields.index_select(dim=-2, index=field_index_tensor)
        masks = gathered_all_masks.index_select(dim=-1, index=field_index_tensor)
        masks = masks & valid_hoi.unsqueeze(-1)
        features = features * masks.unsqueeze(-1).to(dtype=features.dtype)
        return features, masks

    def get_debug_summary(self) -> Dict[str, Any]:
        counts = [len(indices) for indices in self.object_to_hoi_indices.values()]
        if counts:
            count_min = int(min(counts))
            count_max = int(max(counts))
            count_mean = float(sum(counts) / len(counts))
        else:
            count_min = 0
            count_max = 0
            count_mean = 0.0

        valid_ratio = self.text_mask.float().mean(dim=0).detach().cpu().tolist()
        has_nan = bool(
            torch.isnan(self.text_features).any().item()
            or torch.isnan(self.oo_affordance_prior).any().item()
        )
        return {
            "num_hoi": int(self.text_features.shape[0]),
            "d_model": self.d_model,
            "field_order": list(self.DEFAULT_FIELD_ORDER),
            "object_candidate_count_min": count_min,
            "object_candidate_count_max": count_max,
            "object_candidate_count_mean": count_mean,
            "text_mask_valid_ratio_per_field": valid_ratio,
            "oo_prior_loaded": self.oo_prior_loaded,
            "has_nan": has_nan,
        }

    def _load_json(self, path: str, name: str) -> Any:
        if not path:
            raise ValueError(f"{name} path must be provided")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _load_npy(self, path: str, name: str) -> Tensor:
        if not path:
            raise ValueError(f"{name} path must be provided")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} file not found: {path}")
        return torch.as_tensor(np.load(path))

    def _check_field_order(self, field_order: Any) -> None:
        if field_order != self.DEFAULT_FIELD_ORDER:
            raise ValueError(f"field_order must be {self.DEFAULT_FIELD_ORDER}, got {field_order}")

    def _check_main_shapes(self, text_features: Tensor, text_mask: Tensor) -> None:
        if text_features.dim() != 3:
            raise ValueError(f"text_features must be [C, 6, D], got {tuple(text_features.shape)}")
        if text_features.shape[1] != len(self.DEFAULT_FIELD_ORDER):
            raise ValueError(f"text_features second dim must be 6, got {text_features.shape[1]}")
        if text_features.shape[2] != self.d_model:
            raise ValueError(f"text_features last dim must be {self.d_model}, got {text_features.shape[2]}")
        if text_mask.shape != text_features.shape[:2]:
            raise ValueError(f"text_mask must be [C, 6], got {tuple(text_mask.shape)}")
        if len(self.hoi_ids) != text_features.shape[0]:
            raise ValueError(f"hoi_ids length must match C={text_features.shape[0]}, got {len(self.hoi_ids)}")
        if self.strict_shape and text_features.shape[0] != 520:
            raise ValueError(f"strict_shape expects 520 HOI classes, got {text_features.shape[0]}")

    def _sanitize_tensor(self, tensor: Tensor, name: str) -> Tensor:
        if not bool(torch.isfinite(tensor).all().item()):
            warnings.warn(f"{name} contains NaN/Inf; replacing with finite values", RuntimeWarning)
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        return tensor

    def _load_optional_field(self, path: str, field_name: str, main_field: Tensor) -> Optional[Tensor]:
        if not os.path.isfile(path):
            message = f"Optional {field_name} feature file not found: {path}"
            if self.strict_consistency:
                raise FileNotFoundError(message)
            warnings.warn(message, RuntimeWarning)
            return None
        optional = self._load_npy(path, f"{field_name}_features").to(dtype=torch.float32)
        expected_shape = (main_field.shape[0], self.d_model)
        if tuple(optional.shape) != expected_shape:
            message = f"Optional {field_name} features must be {expected_shape}, got {tuple(optional.shape)}"
            if self.strict_consistency:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)
            return None
        optional = self._sanitize_tensor(optional, f"{field_name}_features")
        diff = (optional - main_field.detach().cpu()).abs().max().item() if optional.numel() > 0 else 0.0
        if diff > 1e-3:
            message = f"Optional {field_name} features differ from text_features field by max diff {diff:.6g}"
            if self.strict_consistency:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)
        return optional

    def _parse_hoi_ids(self, hoi_ids: Sequence[str]) -> Tuple[List[int], List[int]]:
        action_ids = []
        object_ids = []
        for hoi_id in hoi_ids:
            parts = hoi_id.split("_")
            if len(parts) != 2:
                raise ValueError(f"Invalid hoi_id format: {hoi_id}")
            try:
                action_id = int(parts[0])
                object_id = int(parts[1])
            except ValueError as exc:
                raise ValueError(f"Invalid hoi_id format: {hoi_id}") from exc
            action_ids.append(action_id)
            object_ids.append(object_id)
        return action_ids, object_ids

    def _parse_object_to_hoi_indices(self, raw_mapping: Any, hoi_object_ids: Sequence[int]) -> Dict[int, List[int]]:
        if not isinstance(raw_mapping, dict):
            raise ValueError("object_to_hoi_indices must be a dict")
        num_hoi = len(self.hoi_ids)
        parsed: Dict[int, List[int]] = {}
        for raw_key, raw_values in raw_mapping.items():
            try:
                object_id = int(raw_key)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"object_to_hoi_indices key must be int-like, got {raw_key}") from exc
            if raw_values is None:
                parsed[object_id] = []
                continue
            if not isinstance(raw_values, list):
                raise ValueError(f"object_to_hoi_indices[{raw_key}] must be a list")

            indices = []
            for item in raw_values:
                if isinstance(item, str):
                    if item not in self.hoi_to_index:
                        raise ValueError(f"Unknown HOI id in object_to_hoi_indices[{raw_key}]: {item}")
                    index = self.hoi_to_index[item]
                else:
                    index = int(item)
                    if index < 0 or index >= num_hoi:
                        raise ValueError(f"HOI index out of range in object_to_hoi_indices[{raw_key}]: {index}")
                if int(hoi_object_ids[index]) != object_id:
                    warnings.warn(
                        f"object_to_hoi_indices[{object_id}] includes HOI {index} with object_id {hoi_object_ids[index]}",
                        RuntimeWarning,
                    )
                indices.append(index)
            parsed[object_id] = indices
        return parsed

    def _build_label_mapping(self, label_ids: Optional[List[int]]) -> Tensor:
        if label_ids is None or len(label_ids) == 0:
            return torch.empty((0,), dtype=torch.long)
        label_ids_int = [int(label_id) for label_id in label_ids]
        valid_labels = [label_id for label_id in label_ids_int if label_id >= 0]
        if not valid_labels:
            return torch.empty((0,), dtype=torch.long)
        mapping = torch.full((max(valid_labels) + 1,), -1, dtype=torch.long)
        for hico_object_id, detector_label_id in enumerate(label_ids_int):
            if detector_label_id < 0:
                continue
            mapping[detector_label_id] = hico_object_id
        return mapping

    def _load_oo_prior(self, path: str) -> Tensor:
        if not path:
            return torch.zeros((80, 80), dtype=torch.float32)
        prior = self._load_npy(path, "oo_affordance_prior").to(dtype=torch.float32)
        if tuple(prior.shape) != (80, 80):
            message = f"oo_affordance_prior must be [80, 80], got {tuple(prior.shape)}"
            if self.strict_shape:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning)
            return torch.zeros((80, 80), dtype=torch.float32)
        prior = self._sanitize_tensor(prior, "oo_affordance_prior")
        return prior.clamp(min=0.0)


class TextGraphPairPrior(nn.Module):
    """Gated residual pair prior backed by offline text graph features.

    All JSON/NPY assets are loaded by TextGraphFeatureBank during __init__.
    Forward-time logic only queries registered buffers and is safe to disable.
    """

    DEFAULT_FIELD_NAMES = ["object_node", "ho_edge", "discriminative"]

    def __init__(
        self,
        text_features_path: str,
        text_mask_path: str,
        hoi_ids_path: str,
        object_to_hoi_indices_path: str,
        field_order_path: str,
        d_model: int = 256,
        field_names: Optional[List[str]] = None,
        max_candidates: int = 0,
        gate_init: float = 0.0,
        normalize: bool = True,
        strict_shape: bool = True,
        strict_consistency: bool = False,
        human_node_features_path: str = "",
        object_node_features_path: str = "",
        ho_edge_features_path: str = "",
        discriminative_features_path: str = "",
        scene_context_features_path: str = "",
        oo_context_features_path: str = "",
        oo_affordance_prior_path: str = "",
        label_ids: Optional[List[int]] = None,
        debug: bool = False,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.field_names = list(field_names or self.DEFAULT_FIELD_NAMES)
        self.max_candidates = max(int(max_candidates), 0)
        self.debug = bool(debug)
        self.bank = TextGraphFeatureBank(
            text_features_path=text_features_path,
            text_mask_path=text_mask_path,
            hoi_ids_path=hoi_ids_path,
            object_to_hoi_indices_path=object_to_hoi_indices_path,
            field_order_path=field_order_path,
            d_model=d_model,
            normalize=normalize,
            strict_shape=strict_shape,
            strict_consistency=strict_consistency,
            human_node_features_path=human_node_features_path,
            object_node_features_path=object_node_features_path,
            ho_edge_features_path=ho_edge_features_path,
            discriminative_features_path=discriminative_features_path,
            scene_context_features_path=scene_context_features_path,
            oo_context_features_path=oo_context_features_path,
            oo_affordance_prior_path=oo_affordance_prior_path,
            label_ids=label_ids,
            debug=debug,
        )
        self.prior_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.residual_gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.last_debug: Dict[str, Any] = self._empty_debug(device=torch.device("cpu"))

    def forward(
        self,
        pair_indices: Tensor,
        labels: Tensor,
        pair_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        self._check_inputs(pair_indices, labels, pair_mask)
        if labels.device != pair_indices.device:
            labels = labels.to(device=pair_indices.device)
        if pair_mask is not None and pair_mask.device != pair_indices.device:
            pair_mask = pair_mask.to(device=pair_indices.device)
        batch_size, num_pairs, _ = pair_indices.shape
        out = self.bank.text_features.new_zeros((batch_size, num_pairs, self.d_model)).to(device=pair_indices.device)
        debug = self._empty_debug(device=pair_indices.device)
        debug["enabled"] = True
        debug["field_names"] = list(self.field_names)
        debug["residual_gate"] = float(torch.tanh(self.residual_gate).detach().cpu().item())

        if num_pairs == 0 or not self.field_names:
            self.last_debug = debug
            return out, debug

        num_nodes = labels.shape[1]
        raw_object_indices = pair_indices[..., 1]
        valid_object_index = (raw_object_indices >= 0) & (raw_object_indices < num_nodes)
        safe_object_indices = raw_object_indices.clamp(min=0, max=max(num_nodes - 1, 0))
        object_labels = torch.gather(labels, dim=1, index=safe_object_indices)
        hico_object_ids, valid_object_label = self.bank.map_labels_to_hico_object_ids(object_labels)
        valid_pair = valid_object_index & valid_object_label
        if pair_mask is not None:
            valid_pair = valid_pair & pair_mask.to(dtype=torch.bool)

        candidates, candidate_mask = self.bank.get_candidate_hoi_indices(hico_object_ids)
        if self.max_candidates > 0 and candidates.shape[-1] > self.max_candidates:
            candidates = candidates[..., : self.max_candidates]
            candidate_mask = candidate_mask[..., : self.max_candidates]

        field_features, field_masks = self.bank.gather_fields(candidates, self.field_names)
        candidate_field_mask = field_masks & candidate_mask.unsqueeze(-1) & valid_pair.unsqueeze(-1).unsqueeze(-1)
        weights = candidate_field_mask.unsqueeze(-1).to(dtype=field_features.dtype)
        denom = weights.sum(dim=(-3, -2)).clamp(min=1.0)
        text_context = (field_features * weights).sum(dim=(-3, -2)) / denom
        text_context = torch.nan_to_num(text_context, nan=0.0, posinf=0.0, neginf=0.0)

        has_text = candidate_field_mask.flatten(start_dim=2).any(dim=-1)
        residual = self.prior_proj(text_context)
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        residual = residual * has_text.unsqueeze(-1).to(dtype=residual.dtype)
        residual = residual * valid_pair.unsqueeze(-1).to(dtype=residual.dtype)
        out = torch.tanh(self.residual_gate).to(dtype=residual.dtype) * residual

        valid_text_counts = has_text.sum(dim=1).detach().cpu().tolist()
        candidate_counts = (candidate_mask & valid_pair.unsqueeze(-1)).sum(dim=-1)
        unknown_count = int((valid_object_index & ~valid_object_label).sum().detach().cpu().item())
        debug.update({
            "valid_pair_count_per_batch": valid_pair.sum(dim=1).detach().cpu().tolist(),
            "valid_text_pair_count_per_batch": valid_text_counts,
            "unknown_object_label_count": unknown_count,
            "candidate_count_max": int(candidate_counts.max().detach().cpu().item()) if candidate_counts.numel() > 0 else 0,
            "candidate_count_min": int(candidate_counts.min().detach().cpu().item()) if candidate_counts.numel() > 0 else 0,
            "max_candidates": self.max_candidates,
            "has_nan": bool(torch.isnan(out).any().detach().cpu().item()),
        })
        self.last_debug = debug
        return out, debug

    def _empty_debug(self, device: torch.device) -> Dict[str, Any]:
        return {
            "enabled": False,
            "field_names": list(getattr(self, "field_names", [])),
            "residual_gate": 0.0,
            "valid_pair_count_per_batch": [],
            "valid_text_pair_count_per_batch": [],
            "unknown_object_label_count": 0,
            "candidate_count_max": 0,
            "candidate_count_min": 0,
            "max_candidates": int(getattr(self, "max_candidates", 0)),
            "has_nan": False,
            "device": str(device),
        }

    def _check_inputs(self, pair_indices: Tensor, labels: Tensor, pair_mask: Optional[Tensor]) -> None:
        if pair_indices.dim() != 3 or pair_indices.shape[-1] != 2:
            raise ValueError(f"pair_indices must be [B, K, 2], got {tuple(pair_indices.shape)}")
        if labels.dim() != 2 or labels.shape[0] != pair_indices.shape[0]:
            raise ValueError(f"labels must be [B, N], got {tuple(labels.shape)}")
        if pair_indices.dtype != torch.long:
            raise ValueError(f"pair_indices must be torch.long, got {pair_indices.dtype}")
        if labels.dtype != torch.long:
            raise ValueError(f"labels must be torch.long, got {labels.dtype}")
        if pair_mask is not None and pair_mask.shape != pair_indices.shape[:2]:
            raise ValueError(f"pair_mask must be [B, K], got {tuple(pair_mask.shape)}")


class TextHOIRetriever(nn.Module):
    """Object-conditioned top-k HOI retrieval with optional discriminative reranking."""

    def __init__(
        self,
        bank: TextGraphFeatureBank,
        d_model: int = 256,
        topk: int = 5,
        pre_topk: int = 20,
        temperature: float = 0.07,
        retrieval_fields: Tuple[str, ...] = ("ho_edge",),
        retrieval_field_weights: Optional[Dict[str, float]] = None,
        rerank_fields: Tuple[str, ...] = ("discriminative",),
        rerank_field_weights: Optional[Dict[str, float]] = None,
        rerank_weight: float = 0.5,
        use_discriminative_rerank: bool = True,
        query_proj: bool = True,
        debug: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.bank = bank
        self.d_model = int(d_model)
        self.topk = max(int(topk), 0)
        self.pre_topk = max(int(pre_topk), 0)
        self.temperature = max(float(temperature), 1e-6)
        self.retrieval_fields = tuple(retrieval_fields)
        self.rerank_fields = tuple(rerank_fields)
        self.retrieval_field_weights = self._normalize_field_weights(self.retrieval_fields, retrieval_field_weights)
        self.rerank_field_weights = self._normalize_field_weights(self.rerank_fields, rerank_field_weights)
        self.rerank_weight = float(rerank_weight)
        self.use_discriminative_rerank = bool(use_discriminative_rerank)
        self.debug = bool(debug)
        self.query_proj = nn.Linear(self.d_model, self.d_model) if query_proj else nn.Identity()
        self.last_debug: Dict[str, Any] = self._empty_debug()

        for field_name in self.retrieval_fields + self.rerank_fields:
            if field_name not in self.bank.field_to_index:
                raise KeyError(f"Unknown text graph field: {field_name}")

    def forward(
        self,
        visual_pair_feats: Tensor,
        pair_object_labels: Tensor,
        pair_valid_mask: Optional[Tensor] = None,
    ) -> TextCandidateOutput:
        self._check_inputs(visual_pair_feats, pair_object_labels, pair_valid_mask)
        if pair_object_labels.device != visual_pair_feats.device:
            pair_object_labels = pair_object_labels.to(device=visual_pair_feats.device)
        if pair_valid_mask is not None and pair_valid_mask.device != visual_pair_feats.device:
            pair_valid_mask = pair_valid_mask.to(device=visual_pair_feats.device)

        batch_size, num_pairs, _ = visual_pair_feats.shape
        out_shape = (batch_size, num_pairs, self.topk)
        selected_hoi_indices = torch.zeros(out_shape, dtype=torch.long, device=visual_pair_feats.device)
        selected_hoi_mask = torch.zeros(out_shape, dtype=torch.bool, device=visual_pair_feats.device)
        selected_scores = visual_pair_feats.new_zeros(out_shape)
        coarse_scores_out = visual_pair_feats.new_zeros(out_shape)
        rerank_scores_out = visual_pair_feats.new_zeros(out_shape) if self.use_discriminative_rerank else None

        pair_object_ids, object_valid_mask = self.bank.map_labels_to_hico_object_ids(pair_object_labels)
        if pair_object_ids.device != visual_pair_feats.device:
            pair_object_ids = pair_object_ids.to(device=visual_pair_feats.device)
            object_valid_mask = object_valid_mask.to(device=visual_pair_feats.device)
        valid_pair_mask = object_valid_mask
        if pair_valid_mask is not None:
            valid_pair_mask = valid_pair_mask & pair_valid_mask.to(dtype=torch.bool)

        candidates, candidate_mask = self.bank.get_candidate_hoi_indices(pair_object_ids)
        candidate_mask = candidate_mask & valid_pair_mask.unsqueeze(-1)
        no_candidate_pair_count = int((valid_pair_mask & ~candidate_mask.any(dim=-1)).sum().detach().cpu().item()) if candidate_mask.numel() > 0 else int(valid_pair_mask.sum().detach().cpu().item())

        debug_info = self._empty_debug()
        debug_info.update({
            "object_valid_pair_count": int(valid_pair_mask.sum().detach().cpu().item()),
            "no_candidate_pair_count": no_candidate_pair_count,
        })

        if self.topk == 0 or num_pairs == 0 or candidates.shape[-1] == 0 or not bool(candidate_mask.any().detach().cpu().item()):
            debug_info.update(self._score_debug(selected_scores, selected_hoi_mask))
            self.last_debug = debug_info
            return TextCandidateOutput(
                selected_hoi_indices=selected_hoi_indices,
                selected_hoi_mask=selected_hoi_mask,
                selected_scores=selected_scores,
                coarse_scores=coarse_scores_out,
                rerank_scores=rerank_scores_out,
                pair_object_ids=pair_object_ids,
                pair_object_valid_mask=object_valid_mask,
                debug_info=debug_info,
            )

        query = self.query_proj(visual_pair_feats)
        query = torch.nan_to_num(query, nan=0.0, posinf=0.0, neginf=0.0)
        query = F.normalize(query, dim=-1, eps=1e-6)

        coarse_keys, coarse_key_mask = self._build_text_keys(candidates, self.retrieval_fields, self.retrieval_field_weights)
        coarse_scores = self._score_candidates(query, coarse_keys, coarse_key_mask & candidate_mask)
        pre_count = min(max(self.pre_topk, self.topk), candidates.shape[-1])
        pre_indices, pre_mask, pre_scores = self._safe_topk(coarse_scores, coarse_key_mask & candidate_mask, pre_count)
        pre_hoi_indices = torch.gather(candidates.clamp(min=0), dim=-1, index=pre_indices) if pre_count > 0 else candidates[..., :0]

        rerank_pre_scores = None
        final_pool_scores = pre_scores
        final_pool_mask = pre_mask
        if self.use_discriminative_rerank and pre_count > 0:
            rerank_keys, rerank_key_mask = self._build_text_keys(pre_hoi_indices, self.rerank_fields, self.rerank_field_weights)
            raw_rerank_scores = self._score_candidates(query, rerank_keys, rerank_key_mask & pre_mask)
            rerank_valid = rerank_key_mask & pre_mask & torch.isfinite(raw_rerank_scores)
            rerank_pre_scores = torch.where(rerank_valid, raw_rerank_scores, torch.zeros_like(raw_rerank_scores))
            final_pool_scores = pre_scores + self.rerank_weight * rerank_pre_scores
            final_pool_scores = final_pool_scores.masked_fill(~(pre_mask & torch.isfinite(pre_scores)), float("-inf"))

        final_indices, final_mask, final_scores = self._safe_topk(final_pool_scores, final_pool_mask, self.topk)
        if self.topk > 0 and pre_count > 0:
            selected_hoi_indices = torch.gather(pre_hoi_indices.clamp(min=0), dim=-1, index=final_indices)
            selected_hoi_indices = selected_hoi_indices * final_mask.to(dtype=selected_hoi_indices.dtype)
            selected_scores = torch.where(final_mask, final_scores, torch.zeros_like(final_scores))
            coarse_selected = torch.gather(pre_scores, dim=-1, index=final_indices)
            coarse_scores_out = torch.where(final_mask, coarse_selected, torch.zeros_like(coarse_selected))
            if rerank_scores_out is not None and rerank_pre_scores is not None:
                rerank_selected = torch.gather(rerank_pre_scores, dim=-1, index=final_indices)
                rerank_scores_out = torch.where(final_mask, rerank_selected, torch.zeros_like(rerank_selected))
            selected_hoi_mask = final_mask

        selected_scores = torch.nan_to_num(selected_scores, nan=0.0, posinf=0.0, neginf=0.0)
        coarse_scores_out = torch.nan_to_num(coarse_scores_out, nan=0.0, posinf=0.0, neginf=0.0)
        if rerank_scores_out is not None:
            rerank_scores_out = torch.nan_to_num(rerank_scores_out, nan=0.0, posinf=0.0, neginf=0.0)

        debug_info.update(self._score_debug(selected_scores, selected_hoi_mask))
        if self.debug:
            debug_info["selected_hoi_ids_preview"] = self._preview_hoi_ids(selected_hoi_indices, selected_hoi_mask)
        self.last_debug = debug_info
        return TextCandidateOutput(
            selected_hoi_indices=selected_hoi_indices,
            selected_hoi_mask=selected_hoi_mask,
            selected_scores=selected_scores,
            coarse_scores=coarse_scores_out,
            rerank_scores=rerank_scores_out,
            pair_object_ids=pair_object_ids,
            pair_object_valid_mask=object_valid_mask,
            debug_info=debug_info,
        )

    def _normalize_field_weights(
        self,
        fields: Tuple[str, ...],
        weights: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        if weights is None:
            return {field_name: 1.0 for field_name in fields}
        return {field_name: float(weights.get(field_name, 1.0)) for field_name in fields}

    def _build_text_keys(
        self,
        hoi_indices: Tensor,
        field_names: Tuple[str, ...],
        field_weights: Dict[str, float],
    ) -> Tuple[Tensor, Tensor]:
        if not field_names:
            key = self.bank.text_features.new_zeros(tuple(hoi_indices.shape) + (self.d_model,)).to(device=hoi_indices.device)
            return key, torch.zeros(tuple(hoi_indices.shape), dtype=torch.bool, device=hoi_indices.device)

        field_features, field_masks = self.bank.gather_fields(hoi_indices, list(field_names))
        weights = torch.tensor(
            [max(float(field_weights.get(field_name, 1.0)), 0.0) for field_name in field_names],
            dtype=field_features.dtype,
            device=field_features.device,
        )
        valid_weights = field_masks.to(dtype=field_features.dtype) * weights.view(*([1] * (field_masks.dim() - 1)), -1)
        denom = valid_weights.sum(dim=-1, keepdim=True)
        key = (field_features * valid_weights.unsqueeze(-1)).sum(dim=-2) / denom.clamp(min=1e-6)
        key_mask = denom.squeeze(-1) > 0
        key = torch.where(key_mask.unsqueeze(-1), key, torch.zeros_like(key))
        key = torch.nan_to_num(key, nan=0.0, posinf=0.0, neginf=0.0)
        key = F.normalize(key, dim=-1, eps=1e-6)
        key = key * key_mask.unsqueeze(-1).to(dtype=key.dtype)
        return key, key_mask

    def _score_candidates(self, query: Tensor, keys: Tensor, key_mask: Tensor) -> Tensor:
        scores = (query.unsqueeze(-2) * keys).sum(dim=-1) / self.temperature
        scores = torch.nan_to_num(scores, nan=float("-inf"), posinf=float("inf"), neginf=float("-inf"))
        return scores.masked_fill(~key_mask, float("-inf"))

    def _safe_topk(self, scores: Tensor, mask: Tensor, k: int) -> Tuple[Tensor, Tensor, Tensor]:
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
        top_valid = torch.isfinite(values) & (torch.arange(actual_k, device=scores.device).view(*([1] * (scores.dim() - 1)), actual_k) < valid_count.unsqueeze(-1))
        indices[..., :actual_k] = torch.where(top_valid, top_indices, torch.zeros_like(top_indices))
        selected_mask[..., :actual_k] = top_valid
        selected_scores[..., :actual_k] = torch.where(top_valid, values, torch.zeros_like(values))
        return indices, selected_mask, selected_scores

    def _empty_debug(self) -> Dict[str, Any]:
        return {
            "text_topk": self.topk,
            "text_pre_topk": self.pre_topk,
            "retrieval_fields": tuple(self.retrieval_fields),
            "rerank_fields": tuple(self.rerank_fields),
            "rerank_weight": self.rerank_weight,
            "object_valid_pair_count": 0,
            "no_candidate_pair_count": 0,
            "selected_score_mean": 0.0,
            "selected_score_max": 0.0,
            "selected_mask_ratio": 0.0,
            "has_nan": False,
        }

    def _score_debug(self, selected_scores: Tensor, selected_mask: Tensor) -> Dict[str, Any]:
        has_selected = bool(selected_mask.any().detach().cpu().item()) if selected_mask.numel() > 0 else False
        if has_selected:
            valid_scores = selected_scores[selected_mask].detach().float()
            selected_mean = float(valid_scores.mean().item())
            selected_max = float(valid_scores.max().item())
        else:
            selected_mean = 0.0
            selected_max = 0.0
        has_nan = bool(torch.isnan(selected_scores).any().detach().cpu().item()) if selected_scores.numel() > 0 else False
        mask_ratio = float(selected_mask.float().mean().detach().cpu().item()) if selected_mask.numel() > 0 else 0.0
        return {
            "selected_score_mean": selected_mean,
            "selected_score_max": selected_max,
            "selected_mask_ratio": mask_ratio,
            "has_nan": has_nan,
        }

    def _preview_hoi_ids(self, selected_hoi_indices: Tensor, selected_hoi_mask: Tensor) -> List[List[List[str]]]:
        previews: List[List[List[str]]] = []
        batch_limit = min(selected_hoi_indices.shape[0], 1)
        pair_limit = min(selected_hoi_indices.shape[1], 3) if selected_hoi_indices.dim() >= 2 else 0
        for batch_idx in range(batch_limit):
            pair_previews: List[List[str]] = []
            for pair_idx in range(pair_limit):
                hoi_ids = []
                for item_idx in range(selected_hoi_indices.shape[2]):
                    if not bool(selected_hoi_mask[batch_idx, pair_idx, item_idx].detach().cpu().item()):
                        continue
                    hoi_index = int(selected_hoi_indices[batch_idx, pair_idx, item_idx].detach().cpu().item())
                    if 0 <= hoi_index < len(self.bank.hoi_ids):
                        hoi_ids.append(self.bank.hoi_ids[hoi_index])
                pair_previews.append(hoi_ids)
            previews.append(pair_previews)
        return previews

    def _check_inputs(
        self,
        visual_pair_feats: Tensor,
        pair_object_labels: Tensor,
        pair_valid_mask: Optional[Tensor],
    ) -> None:
        if visual_pair_feats.dim() != 3:
            raise ValueError(f"visual_pair_feats must be [B, K, D], got {tuple(visual_pair_feats.shape)}")
        if visual_pair_feats.shape[-1] != self.d_model:
            raise ValueError(f"visual_pair_feats last dim must be {self.d_model}, got {visual_pair_feats.shape[-1]}")
        if pair_object_labels.shape != visual_pair_feats.shape[:2]:
            raise ValueError(f"pair_object_labels must be [B, K], got {tuple(pair_object_labels.shape)}")
        if pair_object_labels.dtype != torch.long:
            raise ValueError(f"pair_object_labels must be torch.long, got {pair_object_labels.dtype}")
        if pair_valid_mask is not None and pair_valid_mask.shape != visual_pair_feats.shape[:2]:
            raise ValueError(f"pair_valid_mask must be [B, K], got {tuple(pair_valid_mask.shape)}")


class TextSceneGraphReasoner(nn.Module):
    """Builds text graph tokens from retrieved HOIs and runs lightweight message passing."""

    FIELD_TYPE_IDS = {
        "human_node": 0,
        "object_node": 1,
        "ho_edge": 2,
        "discriminative": 3,
        "scene_context": 4,
        "oo_context": 5,
    }

    def __init__(
        self,
        retriever: TextHOIRetriever,
        d_model: int = 256,
        num_layers: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_field_tokens: Tuple[str, ...] = ("human_node", "object_node", "ho_edge", "discriminative", "scene_context", "oo_context"),
        use_scene_token: bool = True,
        use_oo_context_token: bool = True,
        use_score_embedding: bool = True,
        use_rank_embedding: bool = True,
        max_topk: int = 10,
        field_dropout: float = 0.0,
        debug: bool = False,
    ):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.retriever = retriever
        self.bank = retriever.bank
        self.d_model = int(d_model)
        self.num_layers = max(int(num_layers), 0)
        self.num_heads = int(num_heads)
        self.use_scene_token = bool(use_scene_token)
        self.use_oo_context_token = bool(use_oo_context_token)
        self.use_score_embedding = bool(use_score_embedding)
        self.use_rank_embedding = bool(use_rank_embedding)
        self.max_topk = max(int(max_topk), 1)
        self.field_dropout = nn.Dropout(float(field_dropout))
        self.debug = bool(debug)

        filtered_fields = []
        for field_name in use_field_tokens:
            if field_name == "scene_context" and not self.use_scene_token:
                continue
            if field_name == "oo_context" and not self.use_oo_context_token:
                continue
            if field_name not in self.FIELD_TYPE_IDS:
                raise KeyError(f"Unknown text graph field: {field_name}")
            if field_name not in self.bank.field_to_index:
                raise KeyError(f"Unknown text graph bank field: {field_name}")
            filtered_fields.append(field_name)
        self.use_field_tokens = tuple(filtered_fields)

        self.field_type_embedding = nn.Embedding(len(self.FIELD_TYPE_IDS), self.d_model)
        self.rank_embedding = nn.Embedding(self.max_topk, self.d_model) if self.use_rank_embedding else None
        self.score_embedding = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.d_model, self.d_model),
        ) if self.use_score_embedding else None
        if self.num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.num_heads,
                dim_feedforward=self.d_model * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        else:
            self.encoder = nn.Identity()
        self.last_debug: Dict[str, Any] = {}

    def forward(
        self,
        visual_pair_feats: Tensor,
        pair_object_labels: Tensor,
        pair_valid_mask: Optional[Tensor] = None,
    ) -> TextGraphOutput:
        self._check_inputs(visual_pair_feats, pair_object_labels, pair_valid_mask)
        candidate_output = self.retriever(
            visual_pair_feats=visual_pair_feats,
            pair_object_labels=pair_object_labels,
            pair_valid_mask=pair_valid_mask,
        )
        batch_size, num_pairs, topk = candidate_output.selected_hoi_indices.shape
        num_fields = len(self.use_field_tokens)
        seq_len = topk * num_fields
        text_tokens = visual_pair_feats.new_zeros((batch_size, num_pairs, seq_len, self.d_model))
        text_token_mask = torch.zeros((batch_size, num_pairs, seq_len), dtype=torch.bool, device=visual_pair_feats.device)

        if topk > 0 and num_fields > 0:
            field_features, field_masks = self.bank.gather_fields(
                candidate_output.selected_hoi_indices,
                list(self.use_field_tokens),
            )
            selected_mask = candidate_output.selected_hoi_mask.unsqueeze(-1)
            field_masks = field_masks & selected_mask
            field_features = field_features.to(device=visual_pair_feats.device, dtype=visual_pair_feats.dtype)
            field_masks = field_masks.to(device=visual_pair_feats.device)

            field_type_ids = torch.tensor(
                [self.FIELD_TYPE_IDS[field_name] for field_name in self.use_field_tokens],
                dtype=torch.long,
                device=visual_pair_feats.device,
            )
            field_emb = self.field_type_embedding(field_type_ids).view(1, 1, 1, num_fields, self.d_model)
            token_features = field_features + field_emb.to(dtype=field_features.dtype)

            if self.rank_embedding is not None:
                rank_ids = torch.arange(topk, dtype=torch.long, device=visual_pair_feats.device).clamp(max=self.max_topk - 1)
                rank_emb = self.rank_embedding(rank_ids).view(1, 1, topk, 1, self.d_model)
                token_features = token_features + rank_emb.to(dtype=token_features.dtype)

            if self.score_embedding is not None:
                safe_scores = torch.nan_to_num(candidate_output.selected_scores, nan=0.0, posinf=0.0, neginf=0.0)
                score_emb = self.score_embedding(safe_scores.unsqueeze(-1)).unsqueeze(-2)
                token_features = token_features + score_emb.to(dtype=token_features.dtype)

            token_features = self.field_dropout(token_features)
            token_features = token_features * field_masks.unsqueeze(-1).to(dtype=token_features.dtype)
            text_tokens = token_features.reshape(batch_size, num_pairs, seq_len, self.d_model)
            text_token_mask = field_masks.reshape(batch_size, num_pairs, seq_len)

        text_tokens = self._run_message_passing(text_tokens, text_token_mask)
        text_tokens = text_tokens * text_token_mask.unsqueeze(-1).to(dtype=text_tokens.dtype)

        pooled_by_field = self._pool_fields(
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            selected_scores=candidate_output.selected_scores,
            batch_size=batch_size,
            num_pairs=num_pairs,
            topk=topk,
            num_fields=num_fields,
        )
        pooled_all = self._masked_average(text_tokens, text_token_mask)

        debug_info = self._build_debug_info(
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            selected_hoi_indices=candidate_output.selected_hoi_indices,
            selected_hoi_mask=candidate_output.selected_hoi_mask,
            selected_scores=candidate_output.selected_scores,
            pooled_by_field=pooled_by_field,
            pooled_all=pooled_all,
            topk=topk,
            num_fields=num_fields,
        )
        debug_info["retriever_object_valid_pair_count"] = candidate_output.debug_info.get("object_valid_pair_count", 0)
        debug_info["retriever_no_candidate_pair_count"] = candidate_output.debug_info.get("no_candidate_pair_count", 0)
        self.last_debug = debug_info
        return TextGraphOutput(
            text_tokens=text_tokens,
            text_token_mask=text_token_mask,
            selected_hoi_indices=candidate_output.selected_hoi_indices,
            selected_hoi_mask=candidate_output.selected_hoi_mask,
            selected_scores=candidate_output.selected_scores,
            pooled_human_node=pooled_by_field["human_node"],
            pooled_object_node=pooled_by_field["object_node"],
            pooled_ho_edge=pooled_by_field["ho_edge"],
            pooled_discriminative=pooled_by_field["discriminative"],
            pooled_scene_context=pooled_by_field["scene_context"],
            pooled_oo_context=pooled_by_field["oo_context"],
            pooled_all=pooled_all,
            debug_info=debug_info,
        )

    def _run_message_passing(self, text_tokens: Tensor, text_token_mask: Tensor) -> Tensor:
        if self.num_layers <= 0 or text_tokens.numel() == 0:
            return text_tokens
        batch_size, num_pairs, seq_len, dim = text_tokens.shape
        flat_tokens = text_tokens.reshape(batch_size * num_pairs, seq_len, dim)
        flat_mask = text_token_mask.reshape(batch_size * num_pairs, seq_len)
        has_token = flat_mask.any(dim=-1)
        if not bool(has_token.any().detach().cpu().item()):
            return text_tokens

        updated = flat_tokens.clone()
        valid_tokens = flat_tokens[has_token]
        valid_padding_mask = ~flat_mask[has_token]
        encoded = self.encoder(valid_tokens, src_key_padding_mask=valid_padding_mask)
        encoded = torch.nan_to_num(encoded, nan=0.0, posinf=0.0, neginf=0.0)
        encoded = encoded * flat_mask[has_token].unsqueeze(-1).to(dtype=encoded.dtype)
        updated[has_token] = encoded
        return updated.reshape(batch_size, num_pairs, seq_len, dim)

    def _pool_fields(
        self,
        text_tokens: Tensor,
        text_token_mask: Tensor,
        selected_scores: Tensor,
        batch_size: int,
        num_pairs: int,
        topk: int,
        num_fields: int,
    ) -> Dict[str, Tensor]:
        pooled = {
            field_name: text_tokens.new_zeros((batch_size, num_pairs, self.d_model))
            for field_name in self.FIELD_TYPE_IDS
        }
        if topk == 0 or num_fields == 0:
            return pooled

        tokens_by_field = text_tokens.reshape(batch_size, num_pairs, topk, num_fields, self.d_model)
        masks_by_field = text_token_mask.reshape(batch_size, num_pairs, topk, num_fields)
        score_weights = torch.softmax(
            selected_scores.masked_fill(~masks_by_field.any(dim=-1), float("-inf")),
            dim=-1,
        )
        score_weights = torch.nan_to_num(score_weights, nan=0.0, posinf=0.0, neginf=0.0)
        for local_field_idx, field_name in enumerate(self.use_field_tokens):
            field_tokens = tokens_by_field[:, :, :, local_field_idx, :]
            field_mask = masks_by_field[:, :, :, local_field_idx]
            weights = score_weights * field_mask.to(dtype=score_weights.dtype)
            denom = weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            pooled[field_name] = (field_tokens * weights.unsqueeze(-1)).sum(dim=-2) / denom
            pooled[field_name] = torch.where(
                field_mask.any(dim=-1, keepdim=True),
                pooled[field_name],
                torch.zeros_like(pooled[field_name]),
            )
        return pooled

    def _masked_average(self, tokens: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(dtype=tokens.dtype)
        denom = weights.sum(dim=-2).clamp(min=1.0)
        pooled = (tokens * weights).sum(dim=-2) / denom
        return torch.where(mask.any(dim=-1, keepdim=True), pooled, torch.zeros_like(pooled))

    def _build_debug_info(
        self,
        text_tokens: Tensor,
        text_token_mask: Tensor,
        selected_hoi_indices: Tensor,
        selected_hoi_mask: Tensor,
        selected_scores: Tensor,
        pooled_by_field: Dict[str, Tensor],
        pooled_all: Tensor,
        topk: int,
        num_fields: int,
    ) -> Dict[str, Any]:
        valid_counts = text_token_mask.sum(dim=-1).detach().float()
        if valid_counts.numel() > 0:
            count_mean = float(valid_counts.mean().item())
            count_min = int(valid_counts.min().item())
            count_max = int(valid_counts.max().item())
        else:
            count_mean = 0.0
            count_min = 0
            count_max = 0

        field_valid_ratio: Dict[str, float] = {field_name: 0.0 for field_name in self.FIELD_TYPE_IDS}
        if topk > 0 and num_fields > 0 and text_token_mask.numel() > 0:
            masks_by_field = text_token_mask.reshape(text_token_mask.shape[0], text_token_mask.shape[1], topk, num_fields)
            for local_field_idx, field_name in enumerate(self.use_field_tokens):
                field_valid_ratio[field_name] = float(masks_by_field[:, :, :, local_field_idx].float().mean().detach().cpu().item())

        pooled_norm_per_field = {
            field_name: float(value.detach().float().norm(dim=-1).mean().item()) if value.numel() > 0 else 0.0
            for field_name, value in pooled_by_field.items()
        }
        pooled_norm_per_field["all"] = float(pooled_all.detach().float().norm(dim=-1).mean().item()) if pooled_all.numel() > 0 else 0.0
        debug_info = {
            "text_tokens_shape": tuple(text_tokens.shape),
            "text_valid_token_count_mean": count_mean,
            "text_valid_token_count_min": count_min,
            "text_valid_token_count_max": count_max,
            "selected_hoi_mask_ratio": float(selected_hoi_mask.float().mean().detach().cpu().item()) if selected_hoi_mask.numel() > 0 else 0.0,
            "field_valid_ratio": field_valid_ratio,
            "pooled_norm_per_field": pooled_norm_per_field,
            "no_text_pair_count": int((~text_token_mask.any(dim=-1)).sum().detach().cpu().item()) if text_token_mask.numel() > 0 else int(text_token_mask.shape[0] * text_token_mask.shape[1]),
            "text_has_nan": bool(
                torch.isnan(text_tokens).any().detach().cpu().item()
                or torch.isnan(selected_scores).any().detach().cpu().item()
                or any(torch.isnan(value).any().detach().cpu().item() for value in pooled_by_field.values())
                or torch.isnan(pooled_all).any().detach().cpu().item()
            ),
        }
        if self.debug:
            debug_info["selected_hoi_ids_preview"] = self.retriever._preview_hoi_ids(selected_hoi_indices, selected_hoi_mask)
        return debug_info

    def _check_inputs(
        self,
        visual_pair_feats: Tensor,
        pair_object_labels: Tensor,
        pair_valid_mask: Optional[Tensor],
    ) -> None:
        if visual_pair_feats.dim() != 3:
            raise ValueError(f"visual_pair_feats must be [B, K, D], got {tuple(visual_pair_feats.shape)}")
        if visual_pair_feats.shape[-1] != self.d_model:
            raise ValueError(f"visual_pair_feats last dim must be {self.d_model}, got {visual_pair_feats.shape[-1]}")
        if pair_object_labels.shape != visual_pair_feats.shape[:2]:
            raise ValueError(f"pair_object_labels must be [B, K], got {tuple(pair_object_labels.shape)}")
        if pair_object_labels.dtype != torch.long:
            raise ValueError(f"pair_object_labels must be torch.long, got {pair_object_labels.dtype}")
        if pair_valid_mask is not None and pair_valid_mask.shape != visual_pair_feats.shape[:2]:
            raise ValueError(f"pair_valid_mask must be [B, K], got {tuple(pair_valid_mask.shape)}")
