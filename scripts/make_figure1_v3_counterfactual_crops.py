#!/usr/bin/env python3
"""Generate counterfactual context crops for Figure 1 v3 Track C/D candidates.

The crops are diagnostic review material. They are not EvidenceHOI attention,
causal evidence, or model explanations. Optional CLIP/VLM margins are recorded
only as [AUTO] proxies when a local model is actually available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive/counterfactual_crops"
DEFAULT_TRACK_C = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive/tracks/track_C.csv"
DEFAULT_TRACK_D = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive/tracks/track_D.csv"
DEFAULT_TRACK_A = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive/tracks/track_A.csv"
DEFAULT_TRACK_B = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive/tracks/track_B.csv"
DEFAULT_INSTANCES = PROJECT_ROOT / "outputs/figure1_search_v2/index/instances.parquet"


CROP_COLUMNS = [
    "candidate_source_track",
    "source_rank",
    "candidate_id",
    "pair_id",
    "image_key",
    "file_name",
    "target_verbs",
    "target_object_category",
    "competing_verbs",
    "context_source",
    "context_instance_id",
    "context_category",
    "second_evidence_pair_id",
    "second_evidence_type",
    "pair_only_crop",
    "pair_plus_single_context_crop",
    "full_scene_crop",
    "context_masked_crop",
    "target_pair_plus_second_person_evidence_crop",
    "target_pair_plus_correct_evidence_crop",
    "m_pair",
    "m_context",
    "context_gain_auto_proxy",
    "clip_vlm_status",
    "manual_review_instruction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track-c", type=Path, default=DEFAULT_TRACK_C)
    parser.add_argument("--track-d", type=Path, default=DEFAULT_TRACK_D)
    parser.add_argument("--track-a", type=Path, default=DEFAULT_TRACK_A)
    parser.add_argument("--track-b", type=Path, default=DEFAULT_TRACK_B)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit-c", type=int, default=40)
    parser.add_argument("--limit-d", type=int, default=20)
    parser.add_argument("--limit-a", type=int, default=0)
    parser.add_argument("--limit-b", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    try:
        decoded = json.loads(str(value))
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def bbox_from_row(row: dict[str, Any], prefix: str) -> tuple[float, float, float, float]:
    return (
        float(row[f"{prefix}_bbox_x1"]),
        float(row[f"{prefix}_bbox_y1"]),
        float(row[f"{prefix}_bbox_x2"]),
        float(row[f"{prefix}_bbox_y2"]),
    )


def bbox_from_instance(inst: Any) -> tuple[float, float, float, float]:
    return (float(inst.bbox_x1), float(inst.bbox_y1), float(inst.bbox_x2), float(inst.bbox_y2))


def union_boxes(boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def expand_box(box: Sequence[float], width: int, height: int, pad_ratio: float = 0.18) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = max(bw, bh) * pad_ratio
    return (
        max(0, int(math.floor(x1 - pad))),
        max(0, int(math.floor(y1 - pad))),
        min(width, int(math.ceil(x2 + pad))),
        min(height, int(math.ceil(y2 + pad))),
    )


def crop_and_resize(img: Image.Image, box: Sequence[float], size: int) -> Image.Image:
    crop_box = expand_box(box, img.width, img.height)
    crop = img.crop(crop_box)
    crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(crop.convert("RGB"), ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas


def mask_context(img: Image.Image, context_box: Sequence[float]) -> Image.Image:
    masked = img.convert("RGB").copy()
    draw = ImageDraw.Draw(masked)
    box = tuple(int(round(v)) for v in context_box)
    draw.rectangle(box, fill=(128, 128, 128), outline=(255, 0, 0), width=3)
    return masked


def draw_boxes(img: Image.Image, boxes: list[tuple[Sequence[float], str, tuple[int, int, int]]]) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    for box, label, color in boxes:
        ibox = tuple(int(round(v)) for v in box)
        draw.rectangle(ibox, outline=color, width=4)
        draw.text((ibox[0] + 3, max(0, ibox[1] - 14)), label, fill=color)
    return out


def parse_track_c_context(row: dict[str, Any], instances_by_id: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[tuple[float, float, float, float]], str]:
    contexts = load_json_list(row.get("decisive_context_candidates"))
    for item in contexts:
        if not isinstance(item, dict) or item.get("type") != "track_c_hico_context_cue_candidates":
            continue
        candidates = item.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            inst_id = str(candidate.get("instance_id", ""))
            if not inst_id or inst_id == row.get("target_object_id") or inst_id == row.get("target_human_id"):
                continue
            inst = instances_by_id.get(inst_id)
            if inst is None:
                continue
            return inst_id, str(candidate.get("category", getattr(inst, "category_name", ""))), bbox_from_instance(inst), str(candidate.get("source", "[GT] HICO object/person instance"))
    return None, None, None, "[MISSING] no HICO context instance selected"


def parse_second_evidence(row: dict[str, Any], instances_by_id: dict[str, Any]) -> tuple[Optional[str], Optional[str], Optional[tuple[float, float, float, float]]]:
    neighbors = load_json_list(row.get("misleading_neighbor"))
    contrast_human = str(row.get("contrast_human_id", ""))
    chosen: Optional[dict[str, Any]] = None
    for item in neighbors:
        if isinstance(item, dict) and item.get("human_id") == contrast_human:
            chosen = item
            break
    if chosen is None:
        for item in neighbors:
            if isinstance(item, dict):
                chosen = item
                break
    if chosen:
        inst_id = str(chosen.get("object_id", ""))
        inst = instances_by_id.get(inst_id)
        if inst is not None:
            families = chosen.get("verb_families") if isinstance(chosen.get("verb_families"), list) else []
            second_type = f"{chosen.get('object_category','')}/{','.join(str(x) for x in families)}"
            return str(chosen.get("pair_id", "")), second_type, bbox_from_instance(inst)
    return None, None, None


def competing_verbs(row: dict[str, Any]) -> list[str]:
    target = {str(v) for v in load_json_list(row.get("target_verbs"))}
    confusion = [str(v) for v in load_json_list(row.get("confusion_verb_set"))]
    return [v for v in confusion if v not in target and v != "no_interaction"][:8]


def clip_vlm_margin_status() -> tuple[Optional[float], Optional[float], Optional[float], str]:
    # Deliberately do not download weights or call remote services. If a future
    # run wires a local CLIP/VLM model here, keep the output labelled [AUTO].
    return None, None, None, "not_computed_no_local_clip_vlm_loaded; [AUTO] proxy unavailable"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_track_rows(path: Path, limit: int, source_track: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["candidate_source_track"] = source_track
    return rows[:limit]


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    instances = pd.read_parquet(args.instances)
    instances_by_image = {
        str(image_key): {str(row.canonical_instance_id): row for row in df.itertuples(index=False)}
        for image_key, df in instances.groupby("image_key", sort=False)
    }

    rows = (
        load_track_rows(args.track_d, args.limit_d, "D")
        + load_track_rows(args.track_c, args.limit_c, "C")
        + load_track_rows(args.track_a, args.limit_a, "A")
        + load_track_rows(args.track_b, args.limit_b, "B")
    )
    seen: set[str] = set()
    unique_rows: list[dict[str, Any]] = []
    for row in rows:
        key = f"{row.get('candidate_source_track')}:{row.get('pair_id')}"
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)

    manifest_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(unique_rows, start=1):
        image_path = Path(str(row.get("image_path", "")))
        if not image_path.exists():
            continue
        image_key = str(row["image_key"])
        inst_map = instances_by_image.get(image_key, {})
        context_id, context_category, context_box, context_source = parse_track_c_context(row, inst_map)
        second_pair_id, second_type, second_box = parse_second_evidence(row, inst_map)
        if context_box is None:
            # Without a concrete HICO context box, the non-HICO cue still remains
            # in CSV for manual/VLM review, but no reliable masked/context crop can be drawn.
            continue

        with Image.open(image_path) as img_raw:
            img = img_raw.convert("RGB")
            human_box = (
                float(row["target_human_id"].split(":h")[-1]) if False else 0.0,
                0.0,
                0.0,
                0.0,
            )
            # Use target boxes from audited pair fields if present in CSV is not enough;
            # recover them from instances for exact v2 canonical boxes.
            target_human_inst = inst_map.get(str(row["target_human_id"]))
            target_object_inst = inst_map.get(str(row["target_object_id"]))
            if target_human_inst is None or target_object_inst is None:
                continue
            human_box = bbox_from_instance(target_human_inst)
            object_box = bbox_from_instance(target_object_inst)
            pair_box = union_boxes([human_box, object_box])
            pair_context_box = union_boxes([human_box, object_box, context_box])
            full_scene_box = (0.0, 0.0, float(img.width), float(img.height))
            masked_img = mask_context(img, context_box)
            second_crop_box = union_boxes([human_box, object_box, second_box]) if second_box is not None else pair_box

            candidate_dir = args.output_dir / f"{idx:03d}_{row['candidate_source_track']}_{row['split']}_{row['pair_id'].replace(':','_')}"
            candidate_dir.mkdir(parents=True, exist_ok=True)

            pair_only = crop_and_resize(draw_boxes(img, [(human_box, "target human", (0, 255, 0)), (object_box, "target object", (255, 0, 0))]), pair_box, args.image_size)
            pair_context = crop_and_resize(draw_boxes(img, [(human_box, "target human", (0, 255, 0)), (object_box, "target object", (255, 0, 0)), (context_box, "context", (0, 0, 255))]), pair_context_box, args.image_size)
            full_scene = crop_and_resize(draw_boxes(img, [(human_box, "target human", (0, 255, 0)), (object_box, "target object", (255, 0, 0)), (context_box, "context", (0, 0, 255))]), full_scene_box, args.image_size)
            context_masked = crop_and_resize(draw_boxes(masked_img, [(human_box, "target human", (0, 255, 0)), (object_box, "target object", (255, 0, 0))]), pair_context_box, args.image_size)
            second_evidence = crop_and_resize(draw_boxes(img, [(human_box, "target human", (0, 255, 0)), (object_box, "target object", (255, 0, 0))] + ([(second_box, "second evidence", (255, 165, 0))] if second_box is not None else [])), second_crop_box, args.image_size)
            correct_evidence = crop_and_resize(draw_boxes(img, [(human_box, "target human", (0, 255, 0)), (object_box, "target object", (255, 0, 0)), (context_box, "correct evidence", (0, 0, 255))]), pair_context_box, args.image_size)

            paths = {
                "pair_only_crop": candidate_dir / "pair_only_crop.jpg",
                "pair_plus_single_context_crop": candidate_dir / "pair_plus_single_context_crop.jpg",
                "full_scene_crop": candidate_dir / "full_scene_crop.jpg",
                "context_masked_crop": candidate_dir / "context_masked_crop.jpg",
                "target_pair_plus_second_person_evidence_crop": candidate_dir / "target_pair_plus_second_person_evidence_crop.jpg",
                "target_pair_plus_correct_evidence_crop": candidate_dir / "target_pair_plus_correct_evidence_crop.jpg",
            }
            pair_only.save(paths["pair_only_crop"], quality=92)
            pair_context.save(paths["pair_plus_single_context_crop"], quality=92)
            full_scene.save(paths["full_scene_crop"], quality=92)
            context_masked.save(paths["context_masked_crop"], quality=92)
            second_evidence.save(paths["target_pair_plus_second_person_evidence_crop"], quality=92)
            correct_evidence.save(paths["target_pair_plus_correct_evidence_crop"], quality=92)

        m_pair, m_context, context_gain, clip_status = clip_vlm_margin_status()
        manifest_rows.append({
            "candidate_source_track": row.get("candidate_source_track", ""),
            "source_rank": row.get("rank", ""),
            "candidate_id": row.get("candidate_id", ""),
            "pair_id": row.get("pair_id", ""),
            "image_key": image_key,
            "file_name": row.get("file_name", ""),
            "target_verbs": row.get("target_verbs", "[]"),
            "target_object_category": row.get("target_object_category", ""),
            "competing_verbs": json.dumps(competing_verbs(row), ensure_ascii=False),
            "context_source": context_source,
            "context_instance_id": context_id or "",
            "context_category": context_category or "",
            "second_evidence_pair_id": second_pair_id or "",
            "second_evidence_type": second_type or "",
            **{key: rel(value) for key, value in paths.items()},
            "m_pair": "" if m_pair is None else f"{m_pair:.6f}",
            "m_context": "" if m_context is None else f"{m_context:.6f}",
            "context_gain_auto_proxy": "" if context_gain is None else f"{context_gain:.6f}",
            "clip_vlm_status": clip_status,
            "manual_review_instruction": "[MANUAL_REQUIRED] Compare pair-only vs context/masked crops: masking the context should make GT vs competing verb harder; adding correct context should support GT more than borrowed second-person evidence.",
        })

    manifest_path = args.output_dir / "counterfactual_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CROP_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary = {
        "schema_version": "figure1_search_v3.counterfactual_crops",
        "input_track_c": rel(args.track_c),
        "input_track_d": rel(args.track_d),
        "input_track_a": rel(args.track_a),
        "input_track_b": rel(args.track_b),
        "output_dir": rel(args.output_dir),
        "rows_requested": len(unique_rows),
        "rows_with_crops": len(manifest_rows),
        "crop_types": [
            "pair-only crop",
            "pair + single context crop",
            "full-scene crop",
            "context-masked crop",
            "target pair + second-person evidence crop",
            "target pair + correct evidence crop",
        ],
        "clip_vlm_margin_status": "not_computed_no_local_clip_vlm_loaded; margins would be [AUTO] proxies only",
        "manual_review_required": True,
        "manual_review_note": "Final judgment must inspect whether masking/adding context changes action interpretability; these crops are not EvidenceHOI attention or causal evidence.",
        "manifest": rel(manifest_path),
    }
    (args.output_dir / "counterfactual_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
