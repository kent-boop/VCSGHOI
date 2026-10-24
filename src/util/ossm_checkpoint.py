"""Checkpoint initialization utilities for the SAM-free OSSM variant."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import torch


@dataclass(frozen=True)
class OSSMLoadReport:
    checkpoint: str
    loaded: List[str]
    filtered_sam: List[str]
    missing: List[str]
    unexpected: List[str]
    shape_mismatches: Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...]]]

    @property
    def loaded_count(self):
        return len(self.loaded)

    def summary(self):
        return (
            f"loaded={len(self.loaded)}, filtered_sam={len(self.filtered_sam)}, "
            f"missing={len(self.missing)}, unexpected={len(self.unexpected)}, "
            f"shape_mismatches={len(self.shape_mismatches)}"
        )


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping, got {type(checkpoint).__name__}")
    for key in ("model", "state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint has no model/state_dict/model_state_dict tensor mapping")


def _canonical_key(key):
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def load_ossm_initialization(model, checkpoint_path, map_location="cpu"):
    """Initialize ``model`` from an old OSSM checkpoint without resume state.

    Only the exact top-level ``sam.`` namespace is excluded.  Keys are loaded
    only when both the canonical name and tensor shape match the new model.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"OSSM initialization checkpoint not found: {path}")
    checkpoint = torch.load(str(path), map_location=map_location)
    source = _extract_state_dict(checkpoint)
    destination = model.state_dict()

    loadable = {}
    filtered_sam = []
    unexpected = []
    shape_mismatches = {}
    for raw_key, value in source.items():
        key = _canonical_key(raw_key)
        if key.startswith("sam."):
            filtered_sam.append(raw_key)
            continue
        if key not in destination:
            unexpected.append(raw_key)
            continue
        if tuple(value.shape) != tuple(destination[key].shape):
            shape_mismatches[key] = (tuple(value.shape), tuple(destination[key].shape))
            continue
        loadable[key] = value

    if not loadable:
        raise RuntimeError(
            f"No compatible parameters found in OSSM checkpoint {path}; "
            f"unexpected={len(unexpected)}, shape_mismatches={len(shape_mismatches)}"
        )
    incompatible = model.load_state_dict(loadable, strict=False)
    missing = sorted(set(incompatible.missing_keys))
    report = OSSMLoadReport(
        checkpoint=str(path),
        loaded=sorted(loadable),
        filtered_sam=sorted(filtered_sam),
        missing=missing,
        unexpected=sorted(unexpected),
        shape_mismatches=dict(sorted(shape_mismatches.items())),
    )
    return report


def format_ossm_load_report(report, max_examples=12):
    lines = [f"[OSSM init] {report.checkpoint}", f"[OSSM init] {report.summary()}"]
    groups = (
        ("filtered SAM", report.filtered_sam),
        ("missing", report.missing),
        ("unexpected", report.unexpected),
        ("shape mismatch", list(report.shape_mismatches)),
    )
    for label, values in groups:
        if values:
            preview = ", ".join(values[:max_examples])
            suffix = " ..." if len(values) > max_examples else ""
            lines.append(f"[OSSM init] {label}: {preview}{suffix}")
    return "\n".join(lines)


__all__ = [
    "OSSMLoadReport",
    "format_ossm_load_report",
    "load_ossm_initialization",
]
