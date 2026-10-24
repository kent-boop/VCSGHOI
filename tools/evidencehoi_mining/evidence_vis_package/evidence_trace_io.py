"""I/O helpers for EvidenceHOI qualitative-example mining.

The model stays side-effect free: it returns ``outputs['evidence_traces']``.
The evaluation loop owns this writer and decides where/when records are stored.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _slug(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "unknown"


class EvidenceTraceWriter:
    """Persist per-image trace dictionaries as atomic ``.pt`` records.

    Parameters
    ----------
    output_dir:
        Root directory. Records are stored in ``records/`` and a compact
        ``manifest_rank{rank}.jsonl`` is appended for quick inspection.
    rank:
        Distributed rank. Each rank writes an independent manifest and unique
        filenames; merge manifests after evaluation.
    overwrite:
        If ``False``, an existing record is kept and only its path is returned.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike[str],
        *,
        rank: int = 0,
        overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.records_dir = self.output_dir / "records"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.overwrite = bool(overwrite)
        self.manifest_path = self.output_dir / f"manifest_rank{self.rank}.jsonl"
        self._counter = 0

    def write_outputs(self, outputs: Mapping[str, Any]) -> List[Path]:
        traces = outputs.get("evidence_traces", [])
        if traces is None:
            return []
        return self.write_records(traces)

    def write_records(self, records: Iterable[Mapping[str, Any]]) -> List[Path]:
        paths: List[Path] = []
        for record in records:
            paths.append(self.write_record(dict(record)))
        return paths

    def write_record(self, record: Dict[str, Any]) -> Path:
        image = record.get("image", {})
        image_id = image.get("image_id", "unknown")
        stem = f"{_slug(image_id)}_r{self.rank}_{self._counter:07d}"
        self._counter += 1
        path = self.records_dir / f"{stem}.pt"

        if self.overwrite or not path.exists():
            temp_path = path.with_suffix(path.suffix + ".tmp")
            torch.save(record, temp_path)
            os.replace(temp_path, path)

        manifest_row = self._build_manifest_row(record, path)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
        return path

    def write_run_metadata(self, metadata: Mapping[str, Any]) -> Path:
        path = self.output_dir / f"run_metadata_rank{self.rank}.json"
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(
            json.dumps(_json_safe(metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temp_path, path)
        return path

    def _build_manifest_row(self, record: Mapping[str, Any], path: Path) -> Dict[str, Any]:
        image = record.get("image", {})
        pairs = record.get("pairs", {})
        graph = record.get("graph") or {}

        pair_indices = pairs.get("indices")
        pair_count = int(pair_indices.shape[0]) if isinstance(pair_indices, torch.Tensor) else 0

        gt_targets = pairs.get("gt_action_targets")
        positive_pair_count = 0
        if isinstance(gt_targets, torch.Tensor) and gt_targets.numel() > 0:
            positive_pair_count = int(gt_targets.to(dtype=torch.bool).any(dim=-1).sum().item())

        action_probs = pairs.get("action_probs")
        max_action_prob = 0.0
        min_top2_margin = None
        if isinstance(action_probs, torch.Tensor) and action_probs.numel() > 0:
            probs = action_probs.float()
            max_action_prob = float(probs.max().item())
            if probs.shape[-1] >= 2:
                top2 = probs.topk(k=2, dim=-1).values
                min_top2_margin = float((top2[:, 0] - top2[:, 1]).min().item())

        context_mask = graph.get("selected_context_mask")
        two_hop_mask = graph.get("selected_context_two_hop_mask")
        mean_context_count = 0.0
        mean_two_hop_count = 0.0
        if isinstance(context_mask, torch.Tensor) and context_mask.numel() > 0:
            mean_context_count = float(context_mask.to(dtype=torch.float32).sum(dim=-1).mean().item())
        if isinstance(two_hop_mask, torch.Tensor) and two_hop_mask.numel() > 0:
            mean_two_hop_count = float(two_hop_mask.to(dtype=torch.float32).sum(dim=-1).mean().item())

        return {
            "record_path": str(path.relative_to(self.output_dir)),
            "image_id": _json_safe(image.get("image_id")),
            "file_name": _json_safe(image.get("file_name")),
            "orig_size_hw": _json_safe(image.get("orig_size_hw")),
            "pair_count": pair_count,
            "positive_pair_count": positive_pair_count,
            "graph_trace_available": bool(record.get("metadata", {}).get("graph_trace_available", False)),
            "mean_context_count": mean_context_count,
            "mean_two_hop_context_count": mean_two_hop_count,
            "max_action_prob": max_action_prob,
            "min_top2_margin": min_top2_margin,
        }


def merge_manifests(trace_dir: str | os.PathLike[str], output_name: str = "manifest.jsonl") -> Path:
    """Merge all per-rank manifests deterministically."""
    root = Path(trace_dir)
    rows: List[str] = []
    for path in sorted(root.glob("manifest_rank*.jsonl")):
        rows.extend(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    output_path = root / output_name
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return output_path
