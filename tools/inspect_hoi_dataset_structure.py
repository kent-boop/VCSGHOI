#!/usr/bin/env python3
"""Inspect local HOI dataset layout and annotation schemas.

This script is intentionally read-only for dataset files. It recursively finds
common annotation/mapping formats and image folders, samples parseable files,
and writes a Markdown report for later Figure 1 candidate selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TEXT_EXTS = {".json", ".mat", ".pkl", ".p", ".txt", ".csv", ".npy"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
MAX_PARSE_BYTES = 256 * 1024 * 1024
MAX_TEXT_PREVIEW_LINES = 8
MAX_JSON_SAMPLE_ITEMS = 3
MAX_PICKLE_SAMPLE_ITEMS = 3


IMAGE_KEYS = {
    "image_id",
    "img_id",
    "id",
    "file_name",
    "filename",
    "image",
    "img",
    "global_id",
}
FILENAME_KEYS = {"file_name", "filename", "image", "img", "name"}
BBOX_KEYS = {"bbox", "boxes", "box", "bboxes"}
HUMAN_KEYS = {"subject", "sub", "human", "person", "agent"}
OBJECT_KEYS = {"object", "obj", "target", "role"}
ACTION_KEYS = {
    "action",
    "verb",
    "interaction",
    "hoi",
    "category_id",
    "hoi_category_id",
    "action_name",
    "label",
}
PAIR_KEYS = {
    "hoi_annotation",
    "hois",
    "pairs",
    "pair",
    "subject_id",
    "object_id",
    "subj_id",
    "obj_id",
    "role_object_id",
}


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def safe_stat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.stat()
    except OSError:
        return None


def iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d
            not in {
                ".git",
                "__pycache__",
                "node_modules",
                "checkpoints",
                ".kilo",
                ".vscode",
                ".run",
            }
        ]
        for name in filenames:
            yield Path(dirpath) / name


def find_files(root: Path) -> Tuple[List[Path], List[Path]]:
    ann_files: List[Path] = []
    image_files: List[Path] = []
    if not root.exists():
        return ann_files, image_files
    for path in iter_files(root):
        suffix = path.suffix
        if suffix in TEXT_EXTS:
            ann_files.append(path)
        elif suffix in IMAGE_EXTS:
            image_files.append(path)
    return sorted(ann_files), sorted(image_files)


def summarize_image_folders(image_files: Sequence[Path]) -> List[Dict[str, Any]]:
    counts: Counter[Path] = Counter(path.parent for path in image_files)
    out: List[Dict[str, Any]] = []
    for folder, count in counts.most_common():
        samples = [p.name for p in image_files if p.parent == folder][:5]
        out.append({"path": folder, "count": count, "samples": samples})
    return out


def is_mapping_like(path: Path) -> bool:
    name = path.name.lower()
    return any(
        token in name
        for token in [
            "list",
            "action",
            "verb",
            "object",
            "category",
            "mapping",
            "coco",
            "hico80",
            "corre",
            "correct",
            "lib",
            "label",
            "description",
        ]
    )


def split_hint(path: Path) -> str:
    text = str(path).lower()
    hints = []
    for token in ["trainval", "train", "val", "test", "2015", "2014", "50%", "25%", "15%", "5%"]:
        if token in text:
            hints.append(token)
    return ", ".join(hints) if hints else "not obvious"


def is_relevant_nonprimary_path(path: Path) -> bool:
    text = str(path).lower()
    name = path.name.lower()
    if "/checkpoint" in text or "train_log" in name or name == "args.txt":
        return False
    if any(part in text for part in ["/configs/", "/.git/", "/node_modules/", "/.kilo/"]):
        return False
    strong_names = [
        "instances_train2015.json",
        "instances_test2015.json",
        "instances_vcoco_train.json",
        "instances_vcoco_trainval.json",
        "instances_vcoco_val.json",
        "instances_vcoco_test.json",
        "coco80tohico80.json",
        "coco91tohico80.json",
        "hico_hoi_graph",
        "vcoco_hoi_graph",
        "hico_text_descriptions.json",
        "hico_lib.txt",
        "hico_list_vb.txt",
        "list_action.txt",
        "vcoco_lib.txt",
        "vcoco_verb_names.json",
        "hicodet_test_bbox",
        "hicodet_train_bbox",
        "vcoco_test_bbox",
        "vcoco_train_bbox",
    ]
    if any(token in name for token in strong_names):
        return True
    if "/hoi_txtdescrip/" in text and any(token in name for token in ["hico", "vcoco"]):
        return True
    if "/v-coco/" in text and any(token in name for token in ["vcoco", "instances"]):
        return True
    return False


def type_name(obj: Any) -> str:
    if obj is None:
        return "None"
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return f"{type(obj).__name__}(shape={tuple(obj.shape)}, dtype={obj.dtype})"
    if isinstance(obj, (list, tuple, set)):
        return f"{type(obj).__name__}(len={len(obj)})"
    if isinstance(obj, dict):
        return f"dict(len={len(obj)})"
    return type(obj).__name__


def truncate_repr(obj: Any, limit: int = 240) -> str:
    try:
        text = repr(obj)
    except Exception as exc:  # pragma: no cover - defensive
        text = f"<repr failed: {exc}>"
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def key_matches(key: Any, candidates: set[str]) -> bool:
    k = str(key).lower()
    return any(c in k for c in candidates)


def recursive_key_scan(obj: Any, max_depth: int = 4, max_items: int = 50) -> Counter[str]:
    found: Counter[str] = Counter()

    def visit(x: Any, depth: int) -> None:
        if depth > max_depth:
            return
        if isinstance(x, dict):
            for i, (k, v) in enumerate(x.items()):
                if i >= max_items:
                    break
                lk = str(k).lower()
                if key_matches(lk, IMAGE_KEYS):
                    found["image_id_or_filename"] += 1
                if key_matches(lk, FILENAME_KEYS):
                    found["filename"] += 1
                if key_matches(lk, BBOX_KEYS):
                    found["bbox"] += 1
                if key_matches(lk, HUMAN_KEYS):
                    found["human"] += 1
                if key_matches(lk, OBJECT_KEYS):
                    found["object"] += 1
                if key_matches(lk, ACTION_KEYS):
                    found["action_or_verb_or_hoi"] += 1
                if key_matches(lk, PAIR_KEYS):
                    found["hoi_pair"] += 1
                visit(v, depth + 1)
        elif isinstance(x, (list, tuple)):
            for item in list(x)[:max_items]:
                visit(item, depth + 1)

    visit(obj, 0)
    return found


def first_samples(obj: Any, n: int) -> List[Any]:
    if isinstance(obj, list):
        return obj[:n]
    if isinstance(obj, tuple):
        return list(obj[:n])
    if isinstance(obj, dict):
        samples = []
        for i, (k, v) in enumerate(obj.items()):
            if i >= n:
                break
            samples.append({k: v})
        return samples
    return [obj]


def summarize_sample(sample: Any) -> Dict[str, Any]:
    scan = recursive_key_scan(sample)
    summary: Dict[str, Any] = {
        "sample_type": type_name(sample),
        "has_image_id_or_filename": bool(scan["image_id_or_filename"]),
        "has_filename": bool(scan["filename"]),
        "has_bbox": bool(scan["bbox"]),
        "has_human_bbox_hint": bool(scan["bbox"] and scan["human"]),
        "has_object_bbox_hint": bool(scan["bbox"] and scan["object"]),
        "has_action_verb_interaction_label": bool(scan["action_or_verb_or_hoi"]),
        "has_hoi_pair_info": bool(scan["hoi_pair"]),
        "detected_signal_counts": dict(scan),
    }
    if isinstance(sample, dict):
        summary["keys"] = list(sample.keys())[:30]
        summary["field_types"] = {str(k): type_name(v) for k, v in list(sample.items())[:20]}
    return summary


def inspect_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    top: Dict[str, Any] = {
        "parse_status": "ok",
        "format": "json",
        "top_level_type": type_name(data),
    }
    if isinstance(data, dict):
        top["top_level_keys"] = list(data.keys())[:50]
        top["top_level_field_types"] = {str(k): type_name(v) for k, v in list(data.items())[:20]}
    elif isinstance(data, list):
        top["top_level_length"] = len(data)
    samples = first_samples(data, MAX_JSON_SAMPLE_ITEMS)
    top["sample_summaries"] = [summarize_sample(sample) for sample in samples]
    top["sample_previews"] = [truncate_repr(sample) for sample in samples]
    return top


def inspect_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        data = pickle.load(f)
    top: Dict[str, Any] = {
        "parse_status": "ok",
        "format": "pickle",
        "top_level_type": type_name(data),
    }
    if isinstance(data, dict):
        top["top_level_keys"] = list(data.keys())[:50]
        top["top_level_field_types"] = {str(k): type_name(v) for k, v in list(data.items())[:20]}
    elif isinstance(data, (list, tuple)):
        top["top_level_length"] = len(data)
    samples = first_samples(data, MAX_PICKLE_SAMPLE_ITEMS)
    top["sample_summaries"] = [summarize_sample(sample) for sample in samples]
    top["sample_previews"] = [truncate_repr(sample) for sample in samples]
    return top


def inspect_mat(path: Path) -> Dict[str, Any]:
    try:
        from scipy.io import loadmat
    except Exception as exc:
        return {"parse_status": "failed", "format": "mat", "reason": f"scipy unavailable: {exc}"}
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    user_keys = [k for k in data.keys() if not k.startswith("__")]
    info: Dict[str, Any] = {
        "parse_status": "ok",
        "format": "mat",
        "top_level_keys": user_keys,
        "top_level_field_types": {k: type_name(data[k]) for k in user_keys[:20]},
    }
    samples = [{k: data[k]} for k in user_keys[:MAX_JSON_SAMPLE_ITEMS]]
    info["sample_summaries"] = [summarize_mat_sample(s) for s in samples]
    info["sample_previews"] = [truncate_repr(s) for s in samples]
    return info


def summarize_mat_sample(sample: Any) -> Dict[str, Any]:
    summary = summarize_sample(sample)
    mat_fields: Dict[str, Any] = {}
    if isinstance(sample, dict):
        for k, v in sample.items():
            fields = getattr(v, "_fieldnames", None)
            if fields:
                mat_fields[str(k)] = list(fields)
            elif hasattr(v, "dtype") and getattr(v.dtype, "names", None):
                mat_fields[str(k)] = list(v.dtype.names)
    if mat_fields:
        summary["mat_struct_fields"] = mat_fields
    return summary


def inspect_text(path: Path) -> Dict[str, Any]:
    rows: List[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _, line in zip(range(MAX_TEXT_PREVIEW_LINES), f):
                rows.append(line.rstrip("\n"))
    except Exception as exc:
        return {"parse_status": "failed", "format": path.suffix.lstrip("."), "reason": str(exc)}
    info: Dict[str, Any] = {
        "parse_status": "ok",
        "format": path.suffix.lstrip("."),
        "preview_lines": rows,
    }
    if path.suffix.lower() == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                info["csv_preview_rows"] = [row for _, row in zip(range(5), reader)]
        except Exception as exc:
            info["csv_error"] = str(exc)
    return info


def inspect_npy(path: Path) -> Dict[str, Any]:
    try:
        import numpy as np
    except Exception as exc:
        return {"parse_status": "failed", "format": "npy", "reason": f"numpy unavailable: {exc}"}
    arr = np.load(path, allow_pickle=False)
    info: Dict[str, Any] = {
        "parse_status": "ok",
        "format": "npy",
        "top_level_type": type_name(arr),
        "shape": tuple(arr.shape),
        "dtype": str(arr.dtype),
    }
    if arr.size and arr.size <= 10000:
        info["sample_previews"] = [truncate_repr(arr.reshape(-1)[:20].tolist())]
    return info


def inspect_file(path: Path) -> Dict[str, Any]:
    stat = safe_stat(path)
    info: Dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size if stat else None,
        "split_hint": split_hint(path),
        "mapping_like": is_mapping_like(path),
    }
    if stat and stat.st_size > MAX_PARSE_BYTES and path.suffix.lower() not in {".txt", ".csv"}:
        info.update(
            {
                "parse_status": "skipped",
                "reason": f"larger than MAX_PARSE_BYTES={MAX_PARSE_BYTES}",
            }
        )
        return info
    try:
        suffix = path.suffix.lower()
        if suffix == ".json":
            info.update(inspect_json(path))
        elif suffix in {".pkl", ".p"}:
            info.update(inspect_pickle(path))
        elif suffix == ".mat":
            info.update(inspect_mat(path))
        elif suffix == ".npy":
            info.update(inspect_npy(path))
        elif suffix in {".txt", ".csv"}:
            info.update(inspect_text(path))
        else:
            info.update({"parse_status": "skipped", "reason": f"unsupported suffix {suffix}"})
    except Exception as exc:
        info.update({"parse_status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
    return info


def collect_root_summary(root: Path) -> Dict[str, Any]:
    ann_files, image_files = find_files(root)
    return {
        "root": root,
        "exists": root.exists(),
        "ann_files": ann_files,
        "image_files": image_files,
        "image_folders": summarize_image_folders(image_files),
    }


def score_annotation(info: Dict[str, Any]) -> int:
    score = 0
    text = str(info.get("path", "")).lower()
    if info.get("parse_status") == "ok":
        score += 2
    if "trainval_hico" in text or "test_hico" in text or "instances_train2015" in text or "instances_test2015" in text:
        score += 4
    if "annotations" in text or "instances" in text:
        score += 2
    sample_summaries = info.get("sample_summaries") or []
    for sample in sample_summaries:
        if sample.get("has_image_id_or_filename"):
            score += 1
        if sample.get("has_bbox"):
            score += 1
        if sample.get("has_human_bbox_hint"):
            score += 1
        if sample.get("has_object_bbox_hint"):
            score += 1
        if sample.get("has_action_verb_interaction_label"):
            score += 1
        if sample.get("has_hoi_pair_info"):
            score += 2
    return score


def md_bool(value: Any) -> str:
    return "yes" if value else "no"


def write_report(
    out_path: Path,
    primary_root: Path,
    root_summaries: Sequence[Dict[str, Any]],
    inspections: Sequence[Dict[str, Any]],
) -> None:
    ranked = sorted(inspections, key=score_annotation, reverse=True)
    best = [x for x in ranked if x.get("parse_status") == "ok" and not x.get("mapping_like")]
    mapping_files = [x for x in inspections if x.get("mapping_like")]

    lines: List[str] = []
    lines.append("# Dataset Inspection Report")
    lines.append("")
    lines.append("Generated by `tools/inspect_hoi_dataset_structure.py`.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Primary HICO-DET root: `{primary_root}`")
    for summary in root_summaries:
        lines.append(f"- Scanned root: `{summary['root']}` exists={summary['exists']}")
    lines.append("")

    lines.append("## Discovered Dataset Roots")
    lines.append("")
    for summary in root_summaries:
        root = summary["root"]
        ann_count = len(summary["ann_files"])
        img_count = len(summary["image_files"])
        lines.append(f"- `{root}`: {ann_count} annotation/mapping-like files, {img_count} image files")
    lines.append("")

    lines.append("## Image Folders")
    lines.append("")
    for summary in root_summaries:
        lines.append(f"### `{summary['root']}`")
        if not summary["image_folders"]:
            lines.append("")
            lines.append("No image folders containing `.jpg/.jpeg/.png` were found.")
            lines.append("")
            continue
        lines.append("")
        lines.append("| folder | image_count | sample_files |")
        lines.append("| --- | ---: | --- |")
        for folder_info in summary["image_folders"]:
            samples = ", ".join(f"`{x}`" for x in folder_info["samples"])
            lines.append(f"| `{folder_info['path']}` | {folder_info['count']} | {samples} |")
        lines.append("")

    lines.append("## Annotation And Mapping Files")
    lines.append("")
    lines.append("| path | size | split hint | mapping-like | parse status |")
    lines.append("| --- | ---: | --- | --- | --- |")
    for info in sorted(inspections, key=lambda x: str(x.get("path", ""))):
        lines.append(
            f"| `{info.get('path')}` | {info.get('size_bytes')} | {info.get('split_hint')} | "
            f"{md_bool(info.get('mapping_like'))} | {info.get('parse_status')} |"
        )
    lines.append("")

    lines.append("## Schema Summaries")
    lines.append("")
    for info in sorted(inspections, key=lambda x: str(x.get("path", ""))):
        lines.append(f"### `{info.get('path')}`")
        lines.append("")
        lines.append(f"- Format: `{info.get('format', info.get('path', 'unknown'))}`")
        lines.append(f"- Parse status: `{info.get('parse_status')}`")
        lines.append(f"- Split hint: `{info.get('split_hint')}`")
        lines.append(f"- Mapping-like: {md_bool(info.get('mapping_like'))}")
        if info.get("reason"):
            lines.append(f"- Parse note: {info.get('reason')}")
        if info.get("top_level_type"):
            lines.append(f"- Top-level type: `{info.get('top_level_type')}`")
        if info.get("top_level_length") is not None:
            lines.append(f"- Top-level length: `{info.get('top_level_length')}`")
        if info.get("top_level_keys"):
            keys = ", ".join(f"`{k}`" for k in info.get("top_level_keys", [])[:30])
            lines.append(f"- Top-level keys: {keys}")
        if info.get("top_level_field_types"):
            fields = ", ".join(
                f"`{k}`: `{v}`" for k, v in list(info["top_level_field_types"].items())[:12]
            )
            lines.append(f"- Top-level field types: {fields}")
        if info.get("preview_lines"):
            lines.append("- Text preview:")
            lines.append("")
            lines.append("```text")
            lines.extend(info["preview_lines"])
            lines.append("```")
        sample_summaries = info.get("sample_summaries") or []
        if sample_summaries:
            lines.append("- Sample schema signals:")
            lines.append("")
            lines.append(
                "| sample | type | image id/file | filename | bbox | human bbox hint | object bbox hint | action/verb/interaction | HOI pair | keys |"
            )
            lines.append("| ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for i, sample in enumerate(sample_summaries):
                keys = sample.get("keys") or []
                key_text = ", ".join(f"`{k}`" for k in keys[:12])
                lines.append(
                    f"| {i} | `{sample.get('sample_type')}` | "
                    f"{md_bool(sample.get('has_image_id_or_filename'))} | "
                    f"{md_bool(sample.get('has_filename'))} | "
                    f"{md_bool(sample.get('has_bbox'))} | "
                    f"{md_bool(sample.get('has_human_bbox_hint'))} | "
                    f"{md_bool(sample.get('has_object_bbox_hint'))} | "
                    f"{md_bool(sample.get('has_action_verb_interaction_label'))} | "
                    f"{md_bool(sample.get('has_hoi_pair_info'))} | {key_text} |"
                )
            lines.append("")
        if info.get("sample_previews"):
            lines.append("- Sample previews:")
            lines.append("")
            for preview in info["sample_previews"][:3]:
                lines.append(f"  - `{preview}`")
            lines.append("")

    lines.append("## Recommended Files For Next Candidate Screening")
    lines.append("")
    if best:
        for info in best[:5]:
            lines.append(f"- `{info.get('path')}` score={score_annotation(info)} split={info.get('split_hint')}")
    else:
        lines.append("- No fully parseable non-mapping annotation file was found.")
    lines.append("")

    lines.append("## Label Mapping Completeness")
    lines.append("")
    for info in sorted(mapping_files, key=lambda x: str(x.get("path", ""))):
        lines.append(f"- `{info.get('path')}` parse={info.get('parse_status')} split={info.get('split_hint')}")
    lines.append("")
    lines.append("Assessment:")
    list_action = primary_root / "list_action.txt"
    hico_list_vb = primary_root / "hico_list_vb.txt"
    corre = primary_root / "annotations" / "corre_hico.npy"
    lines.append(f"- Verb/action mapping available: {md_bool(list_action.exists())} at `{list_action}`")
    lines.append(f"- Alternate verb list available: {md_bool(hico_list_vb.exists())} at `{hico_list_vb}`")
    lines.append(f"- HOI/object compatibility matrix `corre_hico.npy` available: {md_bool(corre.exists())} at `{corre}`")
    lines.append("- COCO object-name mapping is not a standalone file in the primary dataset root; this project derives COCO class names from `src/data/datasets/builtin_meta.py`.")
    lines.append("- Explicit 600-class HOI triplet name mapping was not found as a standalone primary-root file; HOI labels can be inferred from annotation `hoi_category_id` plus action/object metadata, but a human-readable HOI triplet list may need project code or external HICO metadata.")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/share/share/Datasets/hico_20160224_det")
    parser.add_argument("--project-root", default="/wzj/zijian/OSSM-HOI-main")
    parser.add_argument("--extra-root", action="append", default=[])
    parser.add_argument(
        "--out",
        default="/wzj/zijian/OSSM-HOI-main/outputs/figure1_candidate_search/dataset_inspection.md",
    )
    args = parser.parse_args(argv)

    primary_root = Path(args.data_root).resolve()
    project_root = Path(args.project_root).resolve()
    roots = [primary_root, project_root]
    roots.extend(Path(x).resolve() for x in args.extra_root)

    seen = set()
    unique_roots: List[Path] = []
    for root in roots:
        if root not in seen:
            unique_roots.append(root)
            seen.add(root)

    root_summaries = [collect_root_summary(root) for root in unique_roots]

    inspection_paths: Dict[Path, None] = {}
    for summary in root_summaries:
        for path in summary["ann_files"]:
            path_text = str(path).lower()
            primary = Path(summary["root"]) == primary_root
            relevant = primary or is_relevant_nonprimary_path(path)
            if not relevant:
                continue
            if path.suffix.lower() in TEXT_EXTS:
                inspection_paths[path] = None
            elif any(token in path_text for token in ["hico", "vcoco", "v-coco", "coco"]):
                inspection_paths[path] = None

    inspections = [inspect_file(path) for path in sorted(inspection_paths.keys())]
    write_report(Path(args.out), primary_root, root_summaries, inspections)

    print(f"Wrote report: {args.out}")
    print(f"Inspected files: {len(inspections)}")
    for summary in root_summaries:
        print(
            f"Root {summary['root']}: {len(summary['ann_files'])} annotation/mapping files, "
            f"{len(summary['image_files'])} images, {len(summary['image_folders'])} image folders"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
