#!/usr/bin/env python3
"""Find HICO images with many GT humans and objects.

This script uses only ground-truth HICO annotations. It does not run any object
detector, and it computes IoU-based de-duplicated counts so repeated boxes do
not inflate the candidate list.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


PERSON_CATEGORY_ID = 1


CSV_COLUMNS = [
    "rank",
    "gt_many_score",
    "image_id",
    "filename",
    "image_path",
    "human_count_raw",
    "object_count_raw",
    "human_count_unique",
    "object_count_unique",
    "hoi_pair_count",
    "verb_count",
    "object_category_count",
    "distinct_humans_in_pairs",
    "distinct_objects_in_pairs",
    "duplicate_human_boxes_removed",
    "duplicate_object_boxes_removed",
    "object_categories",
    "verbs",
    "pair_summary",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find GT-only many-human many-object HICO candidates.")
    parser.add_argument("--dataset-root", default="/share/share/Datasets/hico_20160224_det")
    parser.add_argument("--annotation", default="annotations/trainval_hico.json")
    parser.add_argument("--image-root", default="images/train2015")
    parser.add_argument("--output-dir", default="outputs/figure1_candidate_search/gt_many")
    parser.add_argument("--min-humans", type=int, default=3)
    parser.add_argument("--min-objects", type=int, default=4)
    parser.add_argument("--min-pairs", type=int, default=3)
    parser.add_argument("--min-verbs", type=int, default=2)
    parser.add_argument("--dedup-iou", type=float, default=0.92)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


def resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base / path


def load_action_names(dataset_root: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for path in [dataset_root / "list_action.txt", dataset_root / "hico_list_vb.txt"]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    mapping[int(parts[0])] = parts[1]
        if mapping:
            break
    return mapping


def load_object_names() -> Dict[int, str]:
    repo_root = Path(__file__).resolve().parents[1]
    meta_path = repo_root / "src" / "data" / "datasets" / "builtin_meta.py"
    if not meta_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_gt_many_builtin_meta", meta_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    out: Dict[int, str] = {}
    for item in getattr(module, "COCO_CATEGORIES", []):
        if isinstance(item, dict) and isinstance(item.get("id"), int) and isinstance(item.get("name"), str):
            if item["name"] != "N/A":
                out[item["id"]] = item["name"]
    return out


def bbox(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [float(x) for x in raw]
    except Exception:
        return None


def area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def dedup_instances(instances: Sequence[Dict[str, Any]], threshold: float) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    """Deduplicate by category + high IoU, returning unique reps and original->rep index."""
    reps: List[Dict[str, Any]] = []
    original_to_rep: Dict[int, int] = {}
    for inst in instances:
        inst_box = inst.get("bbox")
        cat = inst.get("category_id")
        matched_rep = None
        for rep_i, rep in enumerate(reps):
            if rep.get("category_id") != cat:
                continue
            rep_box = rep.get("bbox")
            if inst_box and rep_box and iou(inst_box, rep_box) >= threshold:
                matched_rep = rep_i
                break
        if matched_rep is None:
            matched_rep = len(reps)
            reps.append(inst)
        original_to_rep[int(inst["index"])] = matched_rep
    return reps, original_to_rep


def image_id(sample: Dict[str, Any], fallback: int) -> Any:
    for key in ["image_id", "img_id", "id"]:
        if key in sample:
            return sample[key]
    name = sample.get("file_name")
    if isinstance(name, str):
        digits = "".join(ch for ch in Path(name).stem if ch.isdigit())
        if digits:
            return int(digits)
    return fallback


def summarize_pairs(pairs: Sequence[Dict[str, Any]], max_items: int = 10) -> str:
    parts = []
    for p in pairs[:max_items]:
        parts.append(
            f"h{p['human_index']}-o{p['object_index']}:{p['verb_name'] or p['verb_id']}/{p['object_name'] or p['object_category_id']}"
        )
    if len(pairs) > max_items:
        parts.append(f"+{len(pairs) - max_items} more")
    return "; ".join(parts)


def analyze_sample(
    sample: Dict[str, Any],
    idx: int,
    image_root: Path,
    object_names: Dict[int, str],
    action_names: Dict[int, str],
    dedup_iou: float,
) -> Dict[str, Any]:
    filename = sample.get("file_name") or sample.get("filename") or f"missing_{idx}.jpg"
    raw_annotations = sample.get("annotations") if isinstance(sample.get("annotations"), list) else []
    humans: List[Dict[str, Any]] = []
    objects: List[Dict[str, Any]] = []
    all_instances: Dict[int, Dict[str, Any]] = {}
    for inst_idx, raw in enumerate(raw_annotations):
        if not isinstance(raw, dict):
            continue
        cat = raw.get("category_id")
        inst = {
            "index": inst_idx,
            "bbox": bbox(raw.get("bbox")),
            "category_id": cat,
            "category_name": object_names.get(cat),
        }
        all_instances[inst_idx] = inst
        if cat == PERSON_CATEGORY_ID:
            humans.append(inst)
        else:
            objects.append(inst)

    unique_humans, human_original_to_rep = dedup_instances(humans, dedup_iou)
    unique_objects, object_original_to_rep = dedup_instances(objects, dedup_iou)

    pairs: List[Dict[str, Any]] = []
    for pair_idx, raw in enumerate(sample.get("hoi_annotation") if isinstance(sample.get("hoi_annotation"), list) else []):
        if not isinstance(raw, dict):
            continue
        h = raw.get("subject_id")
        o = raw.get("object_id")
        verb_id = raw.get("category_id")
        obj = all_instances.get(o) if isinstance(o, int) else None
        pairs.append(
            {
                "pair_index": pair_idx,
                "human_index": h,
                "object_index": o,
                "human_unique_index": human_original_to_rep.get(h) if isinstance(h, int) else None,
                "object_unique_index": object_original_to_rep.get(o) if isinstance(o, int) else None,
                "verb_id": verb_id,
                "verb_name": action_names.get(verb_id) if isinstance(verb_id, int) else None,
                "object_category_id": obj.get("category_id") if obj else None,
                "object_name": obj.get("category_name") if obj else None,
            }
        )

    verbs = {p["verb_id"] for p in pairs if p.get("verb_id") is not None}
    cats = {p["object_category_id"] for p in pairs if p.get("object_category_id") is not None}
    distinct_h = {p["human_unique_index"] for p in pairs if p.get("human_unique_index") is not None}
    distinct_o = {p["object_unique_index"] for p in pairs if p.get("object_unique_index") is not None}
    object_category_names = sorted({object_names.get(c, str(c)) for c in cats})
    verb_names = sorted({action_names.get(v, str(v)) for v in verbs})
    object_cat_hist = Counter(p["object_name"] or p["object_category_id"] for p in pairs)
    verb_hist = Counter(p["verb_name"] or p["verb_id"] for p in pairs)

    # GT-only many-entity score: favors true unique entities, moderate pair count, and interaction diversity.
    score = 0.0
    score += min(30.0, len(unique_humans) * 5.0)
    score += min(30.0, len(unique_objects) * 4.0)
    score += min(15.0, len(distinct_h) * 3.0)
    score += min(15.0, len(distinct_o) * 2.5)
    score += min(20.0, len(verbs) * 4.0)
    score += min(15.0, len(cats) * 5.0)
    score += 8.0 if 4 <= len(pairs) <= 12 else 0.0
    score -= max(0, len(pairs) - 18) * 0.8
    score -= max(0, len(humans) - len(unique_humans)) * 1.0
    score -= max(0, len(objects) - len(unique_objects)) * 1.0

    return {
        "gt_many_score": round(score, 4),
        "image_id": image_id(sample, idx),
        "filename": filename,
        "image_path": str(image_root / filename),
        "human_count_raw": len(humans),
        "object_count_raw": len(objects),
        "human_count_unique": len(unique_humans),
        "object_count_unique": len(unique_objects),
        "hoi_pair_count": len(pairs),
        "verb_count": len(verbs),
        "object_category_count": len(cats),
        "distinct_humans_in_pairs": len(distinct_h),
        "distinct_objects_in_pairs": len(distinct_o),
        "duplicate_human_boxes_removed": len(humans) - len(unique_humans),
        "duplicate_object_boxes_removed": len(objects) - len(unique_objects),
        "object_categories": object_category_names,
        "verbs": verb_names,
        "object_category_histogram": dict(object_cat_hist),
        "verb_histogram": dict(verb_hist),
        "pairs": pairs,
        "human_boxes_unique": unique_humans,
        "object_boxes_unique": unique_objects,
        "pair_summary": summarize_pairs(pairs),
    }


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else value


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    annotation = resolve(args.annotation, dataset_root).resolve()
    image_root = resolve(args.image_root, dataset_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(annotation.read_text(encoding="utf-8"))
    if args.max_images is not None:
        data = data[: max(0, args.max_images)]
    object_names = load_object_names()
    action_names = load_action_names(dataset_root)

    rows = [
        analyze_sample(sample, idx, image_root, object_names, action_names, args.dedup_iou)
        for idx, sample in enumerate(data)
        if isinstance(sample, dict)
    ]
    filtered = [
        r
        for r in rows
        if r["human_count_unique"] >= args.min_humans
        and r["object_count_unique"] >= args.min_objects
        and r["hoi_pair_count"] >= args.min_pairs
        and r["verb_count"] >= args.min_verbs
    ]
    filtered.sort(
        key=lambda r: (
            r["gt_many_score"],
            r["human_count_unique"],
            r["object_count_unique"],
            r["verb_count"],
            r["object_category_count"],
        ),
        reverse=True,
    )
    top = filtered[: args.top_k]

    csv_path = out_dir / "gt_many_humans_objects_candidates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for rank, row in enumerate(top, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow({col: csv_value(out.get(col)) for col in CSV_COLUMNS})

    json_path = out_dir / "gt_many_humans_objects_candidates.json"
    payload = {
        "metadata": {
            "dataset_root": str(dataset_root),
            "annotation": str(annotation),
            "image_root": str(image_root),
            "dedup_iou": args.dedup_iou,
            "min_humans": args.min_humans,
            "min_objects": args.min_objects,
            "min_pairs": args.min_pairs,
            "min_verbs": args.min_verbs,
            "matched_image_count": len(filtered),
            "top_k": args.top_k,
            "note": "Counts are GT-only; no detector outputs are used.",
        },
        "candidates": [dict(row, rank=i) for i, row in enumerate(top, start=1)],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Matched images: {len(filtered)}")
    print("Top 20 GT-only many-human/many-object candidates")
    for rank, row in enumerate(top[:20], start=1):
        print(
            f"{rank:02d}. {row['filename']} score={row['gt_many_score']:.1f} "
            f"Huniq={row['human_count_unique']} Ouniq={row['object_count_unique']} "
            f"pairs={row['hoi_pair_count']} verbs={row['verb_count']} cats={row['object_category_count']}"
        )
        print(f"    objects: {', '.join(row['object_categories'])}")
        print(f"    verbs: {', '.join(row['verbs'])}")
        print(f"    path: {row['image_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
