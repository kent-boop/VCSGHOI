#!/usr/bin/env python3
"""Shared utilities for reproducible EvidenceHOI efficiency benchmarks.

This module intentionally calls the repository's real dataset/model builders and
the official ``STIP.forward(samples, images, targets)`` path.  It never carries a
second implementation of the model.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class BenchmarkError(RuntimeError):
    """Base error for an invalid or incomplete benchmark run."""


class ManifestError(BenchmarkError):
    """Raised when the variant manifest is ambiguous or invalid."""


class CheckpointError(BenchmarkError):
    """Raised when a checkpoint cannot be loaded exactly."""

    def __init__(self, message: str, report: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.report = dict(report) if report is not None else None


ROOT_FIELDS = {"schema_version", "baseline_variant", "variants"}
VARIANT_FIELDS = {
    "display_name",
    "enabled",
    "paper_components",
    "backbone",
    "visual_encoder",
    "model_file",
    "config_args",
    "resources",
    "checkpoint",
    "expected",
    "paper_configuration_verified",
    "notes",
}
EXPECTED_FIELDS = {"d", "L", "k_oo", "M", "K", "coarse_pool"}
COMPONENTS = ("VET", "KPT", "SETM")


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ManifestError("PyYAML is required: install it with `pip install pyyaml`.") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ManifestError(f"Manifest root must be a mapping: {path}")
    return data


_ENV_PATTERN = re.compile(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))")


def expand_environment(value: Any) -> Any:
    """Expand environment placeholders recursively, preserving unresolved tokens."""
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            name = match.group(1) or match.group(2)
            return os.environ.get(name, match.group(0))
        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_environment(item) for key, item in value.items()}
    return value


def has_unresolved_environment(value: Any) -> bool:
    if isinstance(value, str):
        return _ENV_PATTERN.search(value) is not None
    if isinstance(value, list):
        return any(has_unresolved_environment(item) for item in value)
    if isinstance(value, dict):
        return any(has_unresolved_environment(item) for item in value.values())
    return False


def load_and_validate_manifest(path: os.PathLike) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise ManifestError(f"Variant manifest not found: {path}")
    manifest = _read_yaml(path)
    unknown_root = set(manifest) - ROOT_FIELDS
    missing_root = ROOT_FIELDS - set(manifest)
    if unknown_root or missing_root:
        raise ManifestError(
            f"Invalid manifest root fields; missing={sorted(missing_root)}, unknown={sorted(unknown_root)}"
        )
    if manifest["schema_version"] != 1:
        raise ManifestError(f"Unsupported schema_version={manifest['schema_version']!r}; expected 1")
    variants = manifest.get("variants")
    if not isinstance(variants, dict) or not variants:
        raise ManifestError("`variants` must be a non-empty mapping")
    if manifest["baseline_variant"] not in variants:
        raise ManifestError("baseline_variant must name an entry in `variants`")

    for name, raw in variants.items():
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9_]+", name):
            raise ManifestError(f"Invalid variant name: {name!r}")
        if not isinstance(raw, dict):
            raise ManifestError(f"Variant {name!r} must be a mapping")
        unknown = set(raw) - VARIANT_FIELDS
        missing = VARIANT_FIELDS - set(raw)
        if unknown or missing:
            raise ManifestError(
                f"Variant {name!r} fields invalid; missing={sorted(missing)}, unknown={sorted(unknown)}"
            )
        if not isinstance(raw["enabled"], bool) or not isinstance(raw["paper_configuration_verified"], bool):
            raise ManifestError(f"Variant {name!r}: enabled/verified must be booleans")
        for field in ("display_name", "backbone", "visual_encoder", "model_file", "checkpoint", "notes"):
            if not isinstance(raw[field], str):
                raise ManifestError(f"Variant {name!r}: {field} must be a string")
        if raw["checkpoint"] and Path(raw["checkpoint"]).is_absolute():
            raise ManifestError(f"Variant {name!r}: checkpoint must use an environment placeholder, not an absolute path")
        components = raw["paper_components"]
        if set(components) != set(COMPONENTS) or not all(isinstance(components[k], bool) for k in COMPONENTS):
            raise ManifestError(f"Variant {name!r}: paper_components must contain boolean VET/KPT/SETM")
        if raw["model_file"] != "src/models/stip.py":
            raise ManifestError(
                f"Variant {name!r}: model_file must be src/models/stip.py (user-confirmed final model)"
            )
        if not isinstance(raw["config_args"], dict) or not isinstance(raw["resources"], dict):
            raise ManifestError(f"Variant {name!r}: config_args/resources must be mappings")
        if not all(isinstance(key, str) for key in raw["config_args"]):
            raise ManifestError(f"Variant {name!r}: config_args keys must be strings")
        for key, value in raw["resources"].items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ManifestError(f"Variant {name!r}: resource keys/values must be strings")
            if value and Path(value).is_absolute():
                raise ManifestError(f"Variant {name!r}: resource {key} must not be an absolute personal path")
        expected = raw["expected"]
        if not isinstance(expected, dict) or set(expected) != EXPECTED_FIELDS:
            raise ManifestError(f"Variant {name!r}: expected fields must be {sorted(EXPECTED_FIELDS)}")
        for key in EXPECTED_FIELDS:
            if not isinstance(expected[key], int) or expected[key] < 0:
                raise ManifestError(f"Variant {name!r}: expected.{key} must be a non-negative integer")
        _validate_component_switches(name, raw)

    expanded = expand_environment(manifest)
    expanded["_manifest_path"] = str(path)
    expanded["_manifest_sha256"] = sha256_file(path)
    return expanded


def _validate_component_switches(name: str, variant: Mapping[str, Any]) -> None:
    flags = variant["config_args"]
    components = variant["paper_components"]
    use_graph = bool(flags.get("use_visual_scene_graph", False))
    candidate = bool(flags.get("graph_enable_candidate_evidence", False))
    text = bool(flags.get("graph_enable_text_graph", False))
    sync = bool(flags.get("graph_enable_sync_evidence_matching", False))
    layerwise = bool(flags.get("graph_enable_layerwise_sync_evidence_matching", False))

    if components["VET"] and (not use_graph or not candidate):
        raise ManifestError(f"Variant {name!r}: VET requires visual graph and candidate evidence")
    if components["KPT"] and (not use_graph or not text):
        raise ManifestError(f"Variant {name!r}: KPT requires the graph forward and text prototype graph")
    if components["SETM"] and not (components["VET"] and components["KPT"] and sync and layerwise):
        raise ManifestError(f"Variant {name!r}: SETM requires VET, KPT, sync, and layerwise sync")
    if not components["SETM"] and (sync or layerwise):
        raise ManifestError(f"Variant {name!r}: non-SETM variant enables a SETM switch")
    if layerwise and flags.get("graph_text_dynamic_retrieval", False):
        raise ManifestError(f"Variant {name!r}: layerwise SETM requires retrieval-once")
    if layerwise and flags.get("graph_layerwise_sync_final_matching", False):
        raise ManifestError(f"Variant {name!r}: duplicate final SETM matching is forbidden")


def resolve_variants(manifest: Mapping[str, Any], variant: Optional[str], all_enabled: bool) -> List[str]:
    if bool(variant) == bool(all_enabled):
        raise ManifestError("Select exactly one of --variant or --all-enabled")
    if variant:
        if variant not in manifest["variants"]:
            raise ManifestError(f"Unknown variant {variant!r}; choices={sorted(manifest['variants'])}")
        return [variant]
    names = [name for name, cfg in manifest["variants"].items() if cfg["enabled"]]
    if not names:
        raise ManifestError("No enabled variants in manifest")
    return names


def sha256_file(path: os.PathLike, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: os.PathLike, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def write_csv(path: os.PathLike, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(temporary), str(path))


def _official_runtime_defaults() -> Dict[str, Any]:
    """Defaults added by STIP_main.py after get_args_parser()."""
    return {
        "detr_weights": None,
        "train_detr": False,
        "finetune_detr_weight": 0.1,
        "lr_detr": 1e-5,
        "reduce_lr_on_plateau_patience": 2,
        "reduce_lr_on_plateau_factor": 0.1,
        "proposal_focal_loss_alpha": 0.75,
        "action_focal_loss_alpha": 0.5,
        "proposal_focal_loss_gamma": 2.0,
        "action_focal_loss_gamma": 2.0,
        "proposal_loss_coef": 1.0,
        "action_loss_coef": 1.0,
        "action_sampled_loss_coef": 1.0,
        "action_fore_loss_coef": 1.0,
        "global_action_fore_loss_coef": 1.0,
        "action_fea_loss_coef": 1.0,
        "use_hard_mining_for_relation_discovery": True,
        "use_relation_dependency_encoding": True,
        "use_memory_layout_encoding": True,
        "apply_nms_on_detr": True,
        "use_tail_semantic_feature": True,
        "use_spatial_feature": True,
        "no_interaction_decoder": False,
        "use_memory_union_mask": False,
        "use_union_feature": False,
        "adaptive_relation_query_num": False,
        "use_relation_tgt_mask": False,
        "use_relation_tgt_mask_attend_topk": 10,
        "use_prior_verb_label_mask": False,
        "relation_feature_map_from": "backbone",
        "use_query_fourier_encoding": False,
        "proj_blip": True,
        "merge_blip": True,
        "only_blip": False,
        "zero_shot_type": "default",
    }


def build_repository_args(
    variant: Mapping[str, Any],
    dataset_file: str,
    data_root: str,
    device: str,
    batch_size: int,
    num_workers: int,
) -> Any:
    # Loading ``src.engine.arg_parser`` normally executes ``src.engine.__init__``,
    # which eagerly imports optional V-COCO plotting dependencies (pyheatmap).
    # The benchmark only needs the parser, so load that source file directly.
    import importlib.util

    parser_path = REPO_ROOT / "src" / "engine" / "arg_parser.py"
    spec = importlib.util.spec_from_file_location("evidencehoi_benchmark_arg_parser", parser_path)
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"Cannot load official argument parser: {parser_path}")
    parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser_module)
    get_args_parser = parser_module.get_args_parser

    args = get_args_parser().parse_args([])
    for key, value in _official_runtime_defaults().items():
        setattr(args, key, value)
    base = {
        "HOIDet": True,
        "STIP_relation_head": True,
        "eval": True,
        "validate": False,
        "dataset_file": dataset_file,
        "data_path": data_root,
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "output_dir": "",
        "resume": "",
        "stip_model_variant": "default",
        "model_variant": None,
        "distributed": False,
        "world_size": 1,
        "rank": 0,
        "gpu": 0,
    }
    for key, value in base.items():
        setattr(args, key, value)

    known = set(vars(args))
    for source in (variant["config_args"], variant["resources"]):
        for key, value in source.items():
            if key not in known:
                raise ManifestError(f"Unknown repository argument in manifest: {key!r}")
            setattr(args, key, value)

    args.backbone = variant["backbone"]
    args.hidden_dim = variant["expected"]["d"]
    return args


def build_official_dataset(args: Any, split: str):
    from src.data.datasets import build_dataset

    dataset = build_dataset(image_set=split, args=args)
    args.num_classes = dataset.num_category()
    args.num_actions = dataset.num_action()
    args.action_names = dataset.get_actions()
    if args.share_enc:
        args.hoi_enc_layers = args.enc_layers
    if args.pretrained_dec:
        args.hoi_dec_layers = args.dec_layers
    if args.dataset_file == "hico-det":
        args.valid_obj_ids = dataset.get_valid_obj_ids()
        args.correct_mat = torch.tensor(dataset.correct_mat)
    return dataset


def build_model_from_variant(args: Any, expected_model_file: str = "src/models/stip.py"):
    from src.models import build_model

    model, criterion, postprocessors = build_model(args)
    module_file = Path(sys.modules[type(model).__module__].__file__).resolve()
    expected = (REPO_ROOT / expected_model_file).resolve()
    if module_file != expected:
        raise BenchmarkError(f"Wrong model implementation: built {module_file}, expected {expected}")
    return model, criterion, postprocessors


def load_checkpoint_strict(
    model: torch.nn.Module,
    checkpoint_path: os.PathLike,
    checkpoint_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise CheckpointError(f"Checkpoint not found: {path}")
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise CheckpointError("Checkpoint root must be a mapping")
    if "model" in checkpoint:
        state = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        raise CheckpointError("Checkpoint must contain `model` or `state_dict`")
    if not isinstance(state, Mapping):
        raise CheckpointError("Checkpoint state dict is not a mapping")
    try:
        incompatible = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise CheckpointError(f"Checkpoint tensor shape/type mismatch: {exc}") from exc
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    report = {
        "path_basename": path.name,
        "sha256": checkpoint_sha256 or sha256_file(path),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "epoch": checkpoint.get("epoch"),
        "strict": True,
    }
    if missing or unexpected:
        raise CheckpointError(
            "Strict checkpoint validation failed: "
            f"missing={missing[:20]}{'...' if len(missing) > 20 else ''}, "
            f"unexpected={unexpected[:20]}{'...' if len(unexpected) > 20 else ''}",
            report=report,
        )
    return report


def validate_model_mode(model: torch.nn.Module) -> None:
    training = [name or "<root>" for name, module in model.named_modules() if module.training]
    if training:
        raise BenchmarkError(f"Modules still in training mode: {training[:20]}")


def prepare_official_batch(batch: Sequence[Any], device: torch.device):
    if not isinstance(batch, (tuple, list)) or len(batch) != 3:
        raise BenchmarkError("Official batch must be (samples, targets, images)")
    samples, targets, images = batch
    samples = samples.to(device)
    moved_targets = []
    for target in targets:
        if "orig_size" not in target:
            raise BenchmarkError("Target is missing required field `orig_size`")
        moved_targets.append({key: (value.to(device) if hasattr(value, "to") else value) for key, value in target.items()})
    # Match STIP.forward(samples, images, targets), not the DataLoader tuple.
    return samples, images, moved_targets


def _iter_tensors(value: Any, prefix: str = "output") -> Iterator[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(value):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _iter_tensors(child, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _iter_tensors(child, f"{prefix}[{index}]")


def validate_outputs(outputs: Any) -> None:
    if not isinstance(outputs, Mapping):
        raise BenchmarkError("Model output must be a mapping")
    required = {"pred_logits", "pred_boxes", "pred_rel_pairs", "pred_actions", "pred_action_exists"}
    missing = required - set(outputs)
    if missing:
        raise BenchmarkError(f"Model output missing keys: {sorted(missing)}")
    tensor_count = 0
    for name, tensor in _iter_tensors(outputs):
        tensor_count += 1
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all().item()):
            raise BenchmarkError(f"Non-finite tensor in {name}")
    if tensor_count == 0:
        raise BenchmarkError("Model output contains no tensors")


def output_tensor_snapshot(outputs: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in _iter_tensors(outputs)
        if name.startswith("output.pred_")
    }


def compare_output_snapshots(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], atol=1e-5, rtol=1e-4) -> None:
    if set(left) != set(right):
        raise BenchmarkError("Profiled/unprofiled output structures differ")
    for name in left:
        if left[name].shape != right[name].shape or not torch.allclose(left[name], right[name], atol=atol, rtol=rtol):
            raise BenchmarkError(f"Profiled/unprofiled output mismatch at {name}")


def sample_identifier(target: Mapping[str, Any]) -> int:
    value = target.get("image_id", target.get("id"))
    if torch.is_tensor(value):
        value = value.reshape(-1)[0].item()
    return int(value)


def select_stratified_sample_ids(dataset: Any, num_images: int, seed: int) -> List[int]:
    if num_images <= 0:
        raise BenchmarkError("num_images must be positive")
    count = min(num_images, len(dataset))
    if count < num_images:
        raise BenchmarkError(f"Dataset contains {len(dataset)} images, fewer than requested {num_images}")
    annotations = getattr(dataset, "annotations", None)
    if not isinstance(annotations, list) or len(annotations) < len(dataset):
        # Deterministic fallback; never select using predictions/accuracy.
        import random
        rng = random.Random(seed)
        ids = list(range(len(dataset)))
        rng.shuffle(ids)
        return sorted(ids[:count])

    scored = []
    dataset_ids = getattr(dataset, "ids", list(range(len(dataset))))
    for dataset_index, annotation_index in enumerate(dataset_ids):
        record = annotations[annotation_index]
        complexity = len(record.get("annotations", [])) + len(record.get("hoi_annotation", []))
        scored.append((complexity, dataset_index))
    scored.sort()
    strata = [[], [], []]
    for rank, (_, index) in enumerate(scored):
        stratum = min(2, (3 * rank) // max(len(scored), 1))
        strata[stratum].append(index)

    import random
    rng = random.Random(seed)
    quotas = [count // 3] * 3
    for i in range(count % 3):
        quotas[i] += 1
    chosen: List[int] = []
    leftovers: List[int] = []
    for bucket, quota in zip(strata, quotas):
        rng.shuffle(bucket)
        chosen.extend(bucket[:quota])
        leftovers.extend(bucket[quota:])
    if len(chosen) < count:
        rng.shuffle(leftovers)
        chosen.extend(leftovers[: count - len(chosen)])
    return sorted(chosen)


def write_sample_ids(path: os.PathLike, sample_ids: Sequence[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(f"{int(item)}\n" for item in sample_ids)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(path))


def read_sample_ids(path: os.PathLike) -> List[int]:
    path = Path(path)
    if not path.is_file():
        raise BenchmarkError(f"sample_ids file not found: {path}")
    values = [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise BenchmarkError("sample_ids must be non-empty and unique")
    return values


def build_fixed_loader(dataset: Any, sample_ids: Sequence[int], batch_size: int, num_workers: int):
    from src.util.misc import collate_fn

    if batch_size <= 0:
        raise BenchmarkError("batch_size must be positive")
    invalid = [idx for idx in sample_ids if idx < 0 or idx >= len(dataset)]
    if invalid:
        raise BenchmarkError(f"sample_ids out of range: {invalid[:10]}")
    subset = Subset(dataset, list(sample_ids))
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )


def cache_preprocessed_batches(loader: Iterable[Any]) -> Tuple[List[Any], Dict[str, Any]]:
    cached: List[Any] = []
    resolutions: List[Tuple[int, int]] = []
    annotations: List[int] = []
    image_ids: List[int] = []
    for batch in loader:
        samples, targets, images = batch
        cached.append(batch)
        for target in targets:
            height, width = [int(x) for x in target["orig_size"].tolist()]
            resolutions.append((height, width))
            annotations.append(int(target.get("hois", torch.empty(0)).shape[0]))
            image_ids.append(sample_identifier(target))
    if not cached:
        raise BenchmarkError("Benchmark subset produced no batches")
    areas = [h * w for h, w in resolutions]
    stats = {
        "image_count": len(image_ids),
        "image_ids": image_ids,
        "resolution": {
            "height_min": min(h for h, _ in resolutions),
            "height_median": statistics.median(h for h, _ in resolutions),
            "height_max": max(h for h, _ in resolutions),
            "width_min": min(w for _, w in resolutions),
            "width_median": statistics.median(w for _, w in resolutions),
            "width_max": max(w for _, w in resolutions),
            "area_mean": (sum(areas) / len(areas)),
        },
        "annotation_count": summarize_numbers(annotations),
        "complexity_proxy": "ground-truth HOI annotation count; samples are not selected by accuracy",
    }
    return cached, stats


def unique_named_parameters(model: torch.nn.Module) -> Iterator[Tuple[str, torch.nn.Parameter]]:
    seen = set()
    for name, parameter in model.named_parameters():
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        yield name, parameter


def parameter_group(name: str) -> str:
    if name.startswith("detr."):
        return "DETR"
    if name.startswith("blip."):
        return "BLIP"
    if name.startswith("visual_scene_graph.sync_evidence_matcher."):
        return "SETM"
    if name.startswith("visual_scene_graph.text_"):
        return "KPT"
    if name.startswith("visual_scene_graph."):
        return "VET"
    if name.startswith("interaction_decoder."):
        return "HOI decoder"
    if name.startswith(("action_embed.", "sampled_actions.", "relation_proposal_mlp.", "rel_query_pre_proj.")):
        return "HOI heads"
    return "Other"


def count_parameters(model: torch.nn.Module) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows_by_group: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"parameters": 0, "trainable": 0, "frozen": 0})
    dtype_counts: Counter = Counter()
    total = trainable = 0
    for name, parameter in unique_named_parameters(model):
        count = parameter.numel()
        group = parameter_group(name)
        total += count
        trainable += count if parameter.requires_grad else 0
        rows_by_group[group]["parameters"] += count
        rows_by_group[group]["trainable"] += count if parameter.requires_grad else 0
        rows_by_group[group]["frozen"] += 0 if parameter.requires_grad else count
        dtype_counts[str(parameter.dtype)] += count
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    component_state = active_component_state(model) if hasattr(model, "visual_scene_graph") else {}
    active_groups = {
        "VET": bool(component_state.get("use_visual_scene_graph", False) and component_state.get("candidate_evidence_initialized", False)),
        "KPT": bool(component_state.get("use_visual_scene_graph", False) and component_state.get("kpt_initialized", False)),
        "SETM": bool(component_state.get("use_visual_scene_graph", False) and component_state.get("setm_initialized", False)),
    }
    rows = [
        {"module_group": group, "active_in_forward": active_groups.get(group, True), **values}
        for group, values in sorted(rows_by_group.items())
    ]
    if sum(row["parameters"] for row in rows) != total:
        raise BenchmarkError("Module parameter accounting does not reconcile with total")
    summary = {
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "buffers": buffer_count,
        "fp32_storage_bytes": total * 4,
        "fp16_storage_bytes": total * 2,
        "dtype_distribution": dict(dtype_counts),
        "deduplicated_by_object_id": True,
    }
    return summary, rows


def summarize_numbers(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None, "p90": None, "min": None, "max": None}
    ordered = sorted(float(value) for value in values)
    position = 0.9 * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    p90 = ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {
        "count": len(ordered),
        "mean": (sum(ordered) / len(ordered)),
        "std": statistics.pstdev(ordered),
        "median": statistics.median(ordered),
        "p90": p90,
        "min": ordered[0],
        "max": ordered[-1],
    }


def inference_context():
    """Use inference_mode when available, with a PyTorch 1.7 no_grad fallback."""
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()

def precision_context(precision: str, device: torch.device):
    if precision == "fp32":
        return contextlib.nullcontext()
    if precision != "amp":
        raise BenchmarkError(f"Unsupported precision: {precision}")
    if device.type != "cuda":
        raise BenchmarkError("AMP benchmark requires CUDA")
    return torch.cuda.amp.autocast(enabled=True)


def require_cuda(device: torch.device, metric: str) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise BenchmarkError(f"{metric} requires an available CUDA device")


def dynamic_workload_rows(outputs: Mapping[str, Any], variant: str, image_ids: Sequence[int]) -> List[Dict[str, Any]]:
    inputs = outputs.get("graph_input_debug", [])
    graph_debug = outputs.get("graph_reasoner_debug", [])
    rows: List[Dict[str, Any]] = []
    for index, image_id in enumerate(image_ids):
        input_info = inputs[index] if index < len(inputs) else {}
        graph_info = graph_debug[index] if index < len(graph_debug) else {}
        layer_debug = graph_info.get("layerwise_sync_debug", []) or []
        first_layer = next((item for item in layer_debug if isinstance(item, Mapping)), {})
        match_shapes = [
            item.get("sync_match_logits_shape")
            for item in layer_debug
            if isinstance(item, Mapping) and item.get("sync_match_logits_shape") is not None
        ]
        evidence_shape = graph_info.get("candidate_evidence_tokens_shape") or first_layer.get(
            "candidate_evidence_tokens_shape"
        )
        prototype_shape = (
            graph_info.get("prototype_slots_shape")
            or graph_info.get("text_prototype_slots_shape")
            or first_layer.get("prototype_slots_shape")
        )
        selected_m = None
        if isinstance(evidence_shape, (list, tuple)) and len(evidence_shape) >= 2:
            # Evidence tokens contain Human State, Object State, Action Cue, then M contextual paths.
            selected_m = max(0, int(evidence_shape[-2]) - 3)
        retrieved_k = None
        if isinstance(prototype_shape, (list, tuple)) and len(prototype_shape) >= 3:
            retrieved_k = int(prototype_shape[2])
        kpt_reasoning_layers = graph_info.get("text_layer_count")
        if not kpt_reasoning_layers and layer_debug:
            # In layerwise SETM, old dense text injection can be disabled while
            # KPT still advances once per visual layer.
            kpt_reasoning_layers = len(layer_debug)
        row = {
            "variant": variant,
            "image_id": image_id,
            "valid_nodes": input_info.get("detected_object_count"),
            "candidate_hoi_pairs": input_info.get("proposal_count"),
            "valid_hoi_pairs": _first_scalar(graph_info.get("valid_pair_count_per_batch")),
            "context_path_candidates_before_topM": None,
            "selected_context_paths_mean": graph_info.get("candidate_evidence_context_path_count_mean"),
            "selected_paths_M": selected_m,
            "retrieved_prototypes_K": retrieved_k,
            "prototype_valid_count": graph_info.get("prototype_valid_count"),
            "kpt_reasoning_layers": kpt_reasoning_layers,
            "setm_reasoning_layers": graph_info.get("layerwise_sync_layer_count"),
            "setm_match_shapes": json.dumps(match_shapes),
        }
        rows.append(row)
    return rows


def _first_scalar(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value


def _command_output(command: Sequence[str]) -> Optional[str]:
    try:
        return subprocess.check_output(command, stderr=subprocess.DEVNULL, text=True, timeout=10).strip()
    except Exception:
        return None


def _torchvision_version() -> Optional[str]:
    try:
        import torchvision
        return torchvision.__version__
    except Exception:
        return None


def collect_environment(device: str, precision: str, benchmark_script: os.PathLike) -> Dict[str, Any]:
    cuda_device = torch.device(device)
    gpu: Dict[str, Any] = {"available": torch.cuda.is_available(), "requested_device": device}
    if torch.cuda.is_available() and cuda_device.type == "cuda":
        index = cuda_device.index if cuda_device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu.update({"name": properties.name, "total_memory_bytes": properties.total_memory})
    driver = _command_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if driver:
        gpu["driver_version"] = driver.splitlines()[0]
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu_model = platform.processor() or None
    ram_bytes = None
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                ram_bytes = int(line.split()[1]) * 1024
                break
    except OSError:
        pass
    git_commit = _command_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    git_status = _command_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"])
    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    environment = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": _torchvision_version(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn,
        "gpu": gpu,
        "cpu": {"model": cpu_model, "logical_cores": os.cpu_count()},
        "ram_bytes": ram_bytes,
        "git": {"commit": git_commit, "dirty": bool(git_status)},
        "benchmark_script_sha256": sha256_file(benchmark_script),
        "precision": precision,
        "backend": {
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
            "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
        },
    }
    return environment


def sanitize_path(value: Optional[str], placeholder: str) -> Optional[str]:
    if not value:
        return value
    return f"<{placeholder}>/{Path(value).name}"


def environment_signature(environment: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "pytorch": environment.get("pytorch"),
        "torchvision": environment.get("torchvision"),
        "cuda_runtime": environment.get("cuda_runtime"),
        "cudnn": environment.get("cudnn"),
        "gpu_name": environment.get("gpu", {}).get("name"),
        "driver_version": environment.get("gpu", {}).get("driver_version"),
        "precision": environment.get("precision"),
        "backend": environment.get("backend"),
    }


def compare_environments(environments: Sequence[Mapping[str, Any]], allow_mixed: bool = False) -> None:
    if len(environments) < 2:
        return
    reference = environment_signature(environments[0])
    differences = []
    for index, environment in enumerate(environments[1:], 1):
        signature = environment_signature(environment)
        if signature != reference:
            differences.append({"index": index, "signature": signature})
    if differences and not allow_mixed:
        raise BenchmarkError(f"Critical benchmark environments differ: {differences}")


def active_component_state(model: torch.nn.Module) -> Dict[str, Any]:
    graph = getattr(model, "visual_scene_graph", None)
    return {
        "model_module": type(model).__module__,
        "use_visual_scene_graph": bool(getattr(model, "use_visual_scene_graph", False)),
        "candidate_evidence_initialized": getattr(graph, "candidate_evidence_inducer", None) is not None,
        "kpt_initialized": getattr(graph, "text_graph_reasoner", None) is not None,
        "setm_initialized": getattr(graph, "sync_evidence_matcher", None) is not None,
        "layerwise_setm": bool(getattr(graph, "graph_enable_layerwise_sync_evidence_matching", False)),
        "dynamic_retrieval": bool(getattr(graph, "graph_text_dynamic_retrieval", False)),
        "visual_layers": len(getattr(graph, "layers", [])) if graph is not None else 0,
    }


def validate_component_state(model: torch.nn.Module, variant: Mapping[str, Any]) -> Dict[str, Any]:
    state = active_component_state(model)
    expected = variant["paper_components"]
    checks = {
        "VET": state["use_visual_scene_graph"] and state["candidate_evidence_initialized"],
        "KPT": state["use_visual_scene_graph"] and state["kpt_initialized"],
        "SETM": state["use_visual_scene_graph"] and state["setm_initialized"] and state["layerwise_setm"],
    }
    mismatches = [name for name in COMPONENTS if checks[name] != expected[name]]
    if mismatches:
        raise BenchmarkError(f"Built component state differs from manifest for: {mismatches}; state={state}")
    if expected["SETM"] and state["dynamic_retrieval"]:
        raise BenchmarkError("Full SETM must retrieve KPT candidates once, not once per layer")
    return state


def profiler_modules(model: torch.nn.Module) -> Dict[str, List[torch.nn.Module]]:
    detr_modules = [
        module for module in (
            getattr(model.detr, "backbone", None),
            getattr(model.detr, "transformer", None),
            getattr(model.detr, "input_proj", None),
            getattr(model.detr, "class_embed", None),
            getattr(model.detr, "bbox_embed", None),
        ) if isinstance(module, torch.nn.Module)
    ]
    groups: Dict[str, List[torch.nn.Module]] = {
        "DETR": detr_modules,
        "BLIP": [model.blip],
        "HOI decoder": [model.interaction_decoder] if hasattr(model, "interaction_decoder") else [],
        "Graph total (inclusive)": [],
        "VET": [],
        "KPT retrieval": [],
        "KPT reasoning": [],
        "KPT fusion": [],
        "SETM": [],
    }
    graph = getattr(model, "visual_scene_graph", None)
    if graph is None:
        return groups
    if bool(getattr(model, "use_visual_scene_graph", False)):
        groups["Graph total (inclusive)"].append(graph)
    vet_names = (
        "class_embedding", "node_type_embedding", "box_mlp", "node_fusion",
        "endpoint_pair_proj", "geo_mlp", "edge_type_embedding", "edge_mlp",
        "pair_out_mlp",
    )
    groups["VET"].extend(
        module for module in (getattr(graph, name, None) for name in vet_names)
        if isinstance(module, torch.nn.Module)
    )
    groups["VET"].extend(list(getattr(graph, "layers", [])))
    candidate = getattr(graph, "candidate_evidence_inducer", None)
    if candidate is not None:
        groups["VET"].append(candidate)
    text = getattr(graph, "text_graph_reasoner", None)
    if text is not None:
        retriever = getattr(text, "retriever", None)
        if retriever is not None:
            groups["KPT retrieval"].append(retriever)
        text_layers = getattr(text, "graph_layers", getattr(text, "layers", []))
        groups["KPT reasoning"].extend(list(text_layers))
        fusion_names = (
            "text_cross_attn", "text_cross_norm", "text_context_proj",
            "text_refined_union_query_proj", "text_human_node_proj",
            "text_object_node_proj", "text_ho_edge_proj",
            "text_discriminative_proj", "text_scene_proj", "text_oo_proj",
        )
        groups["KPT fusion"].extend(
            module for module in (getattr(graph, name, None) for name in fusion_names)
            if isinstance(module, torch.nn.Module)
        )
    matcher = getattr(graph, "sync_evidence_matcher", None)
    if matcher is not None:
        groups["SETM"].append(matcher)
    return groups


class CudaModuleProfiler:
    """CUDA-event module timer; intervals may be inclusive/overlapping."""

    def __init__(self, groups: Mapping[str, Sequence[torch.nn.Module]]):
        self.groups = groups
        self.handles = []
        self.pending: Dict[int, List[torch.cuda.Event]] = defaultdict(list)
        self.events: List[Tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.call_counts: Counter = Counter()

    def __enter__(self):
        for group, modules in self.groups.items():
            for module in modules:
                module_id = id(module)

                def pre_hook(_module, _inputs, group=group, module_id=module_id):
                    start = torch.cuda.Event(enable_timing=True)
                    start.record()
                    self.pending[module_id].append(start)
                    self.call_counts[group] += 1

                def post_hook(_module, _inputs, _outputs, group=group, module_id=module_id):
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    if not self.pending[module_id]:
                        raise BenchmarkError(f"Unbalanced profiler hook for {group}")
                    start = self.pending[module_id].pop()
                    self.events.append((group, start, end))

                self.handles.append(module.register_forward_pre_hook(pre_hook))
                self.handles.append(module.register_forward_hook(post_hook))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def results(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        torch.cuda.synchronize()
        totals: Dict[str, float] = defaultdict(float)
        for group, start, end in self.events:
            totals[group] += float(start.elapsed_time(end))
        return dict(totals), dict(self.call_counts)


def verify_call_semantics(call_counts: Mapping[str, int], variant: Mapping[str, Any], image_count: int) -> List[str]:
    expected = variant["paper_components"]
    reasons: List[str] = []
    if expected["KPT"]:
        retrieval_calls = call_counts.get("KPT retrieval", 0)
        if retrieval_calls != image_count:
            reasons.append(f"KPT retrieval count {retrieval_calls} != image count {image_count}")
        reasoning_calls = call_counts.get("KPT reasoning", 0)
        expected_reasoning = image_count * int(variant["expected"]["L"])
        if reasoning_calls != expected_reasoning:
            reasons.append(f"KPT reasoning count {reasoning_calls} != image_count*L ({expected_reasoning})")
    if expected["SETM"]:
        expected_calls = image_count * int(variant["expected"]["L"])
        observed = call_counts.get("SETM", 0)
        if observed != expected_calls:
            reasons.append(f"SETM call count {observed} != image_count*L ({expected_calls})")
    return reasons
