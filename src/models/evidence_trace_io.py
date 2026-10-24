"""I/O helpers for EvidenceHOI qualitative-example mining.

The model stays side-effect free: it returns ``outputs['evidence_traces']``.
The evaluation loop owns this writer and decides where/when records are stored.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
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
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _slug(value: Any) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or "unknown"


def _git_metadata(repository_root: Optional[str | os.PathLike[str]] = None) -> Dict[str, Any]:
    root = Path(repository_root) if repository_root is not None else Path(__file__).resolve().parents[2]

    def run_git(*args: str) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    status = run_git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def build_run_metadata(args: Any, *, dataset_split: str, rank: int, world_size: int) -> Dict[str, Any]:
    """Build reproducibility metadata written once per distributed rank."""
    config = dict(vars(args)) if hasattr(args, "__dict__") else {}
    variant = config.get("model_variant") or config.get("stip_model_variant", "default")
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rank": int(rank),
        "world_size": int(world_size),
        "checkpoint": config.get("resume"),
        "detr_weights": config.get("detr_weights"),
        "git": _git_metadata(),
        "dataset": {
            "name": config.get("dataset_file"),
            "split": str(dataset_split),
            "data_path": config.get("data_path"),
        },
        "model": {
            "variant": variant,
            "backbone": config.get("backbone"),
            "graph_evidence_topk": config.get("graph_evidence_topk"),
            "graph_text_topk": config.get("graph_text_topk"),
        },
        "seed": config.get("seed"),
        "label_sources": {
            "action_names": config.get("action_names"),
            "valid_action_ids": config.get("valid_ids"),
            "valid_object_ids": config.get("valid_obj_ids"),
            "text_hoi_ids_path": config.get("graph_text_hoi_ids_path"),
            "text_object_to_hoi_path": config.get("graph_text_object_to_hoi_indices_path"),
        },
        "config": config,
    }


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

    def write_outputs(
        self,
        outputs: Mapping[str, Any],
        *,
        expected_count: Optional[int] = None,
    ) -> List[Path]:
        if "evidence_traces" not in outputs:
            if expected_count is not None:
                raise RuntimeError(
                    "Trace export is enabled, but the model output has no "
                    "'evidence_traces'. Select the evidence_trace STIP variant."
                )
            return []
        traces = outputs.get("evidence_traces")
        if traces is None:
            traces = []
        records = list(traces)
        if expected_count is not None and len(records) != int(expected_count):
            raise RuntimeError(
                f"Expected {int(expected_count)} evidence traces for this batch, "
                f"but the model returned {len(records)}."
            )
        return self.write_records(records)

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
    """Merge rank manifests and archive distributed-sampler duplicates.

    ``DistributedSampler`` pads an evaluation split when its size is not
    divisible by the world size. Those padded samples are valid model outputs,
    but they must appear only once in the active Trace set used for ranking.
    Duplicate record files are retained under ``duplicate_records/``.
    """
    root = Path(trace_dir)
    parsed_rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("manifest_rank*.jsonl")):
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"Manifest row in {path}:{line_number} is not an object"
                )
            parsed_rows.append(dict(row))

    rows: List[Dict[str, Any]] = []
    duplicates: List[Dict[str, Any]] = []
    seen: Dict[Any, Dict[str, Any]] = {}
    for row in parsed_rows:
        image_id = row.get("image_id")
        if image_id is not None:
            identity = (
                "image_id",
                json.dumps(
                    _json_safe(image_id),
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        elif row.get("file_name"):
            identity = ("file_name", str(row["file_name"]))
        else:
            identity = ("record_path", str(row.get("record_path", "")))

        if identity in seen:
            duplicate = dict(row)
            duplicate["duplicate_of_record_path"] = seen[identity].get(
                "record_path"
            )
            duplicates.append(duplicate)
            continue
        seen[identity] = row
        rows.append(row)

    active_record_paths = {
        str(row["record_path"])
        for row in rows
        if row.get("record_path")
    }
    archive_dir = root / "duplicate_records"
    archived_count = 0
    for duplicate in duplicates:
        record_path = str(duplicate.get("record_path", ""))
        if not record_path or record_path in active_record_paths:
            continue
        relative_path = Path(record_path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe record path in manifest: {record_path}")

        source = root / relative_path
        destination = archive_dir / relative_path.name
        if source.is_file():
            archive_dir.mkdir(parents=True, exist_ok=True)
            suffix_index = 1
            while destination.exists():
                destination = archive_dir / (
                    f"{source.stem}_{suffix_index}{source.suffix}"
                )
                suffix_index += 1
            os.replace(source, destination)
            archived_count += 1
        if destination.is_file():
            duplicate["archived_record_path"] = str(
                destination.relative_to(root)
            )

    def write_jsonl(path: Path, values: List[Dict[str, Any]]) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        payload = "\n".join(
            json.dumps(value, ensure_ascii=False) for value in values
        )
        temp_path.write_text(
            payload + ("\n" if payload else ""),
            encoding="utf-8",
        )
        os.replace(temp_path, path)

    output_path = root / output_name
    write_jsonl(output_path, rows)
    write_jsonl(root / "manifest_duplicates.jsonl", duplicates)

    summary = {
        "raw_manifest_rows": len(parsed_rows),
        "unique_manifest_rows": len(rows),
        "duplicate_manifest_rows": len(duplicates),
        "archived_duplicate_records": archived_count,
    }
    summary_path = root / "manifest_merge_summary.json"
    temp_summary_path = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temp_summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_summary_path, summary_path)
    return output_path
