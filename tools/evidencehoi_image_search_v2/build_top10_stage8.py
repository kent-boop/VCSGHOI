#!/usr/bin/env python3
"""Build Stage 8 top-10 deep-analysis materials from Stage 6/7 candidates.

The current runtime cannot inspect image pixels semantically. This script uses
GT and automatic proxy fields to select candidates for human follow-up, creates
cue crops/storyboard schematics, and explicitly marks all visual judgments as
requiring manual review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v2"
TOP10_ROOT = OUTPUT_ROOT / "top10"
LEGACY_NEGATIVE_FILES = {
    "HICO_train2015_00003946.jpg",
    "HICO_train2015_00024103.jpg",
    "HICO_train2015_00006309.jpg",
    "HICO_train2015_00018460.jpg",
    "HICO_train2015_00019218.jpg",
}
PROTOTYPES_BY_OBJECT = {
    "bicycle": ["hold", "ride", "inspect", "repair", "push", "straddle"],
    "pizza": ["carry", "cook", "cut", "eat", "hold", "make", "pick_up"],
    "bottle": ["carry", "drink_with", "hold", "open", "pour"],
    "motorcycle": ["hold", "ride", "sit_on", "straddle", "wash"],
    "laptop": ["hold", "open", "read", "type_on", "repair"],
    "cup": ["hold", "drink_with", "inspect", "wash"],
    "horse": ["hold", "ride", "feed", "walk", "jump"],
    "dining table": ["sit_at", "eat_at", "clean"],
    "chair": ["carry", "hold", "sit_on", "stand_on"],
    "couch": ["sit_on", "lie_on", "carry"],
    "suitcase": ["carry", "drag", "hold", "load", "open", "pack"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=OUTPUT_ROOT / "ranking/candidate_features.csv")
    parser.add_argument("--stage6-shortlist", type=Path, default=OUTPUT_ROOT / "review/shortlist_v1.csv")
    parser.add_argument("--stage7-shortlist", type=Path, default=OUTPUT_ROOT / "round2/shortlist_v2.csv")
    parser.add_argument("--stage7-delta", type=Path, default=OUTPUT_ROOT / "round2/delta_candidates.csv")
    parser.add_argument("--image-summary", type=Path, default=OUTPUT_ROOT / "index/image_summary.parquet")
    parser.add_argument("--output-dir", type=Path, default=TOP10_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=OUTPUT_ROOT / "manifest_stage8.json")
    parser.add_argument("--seed", type=int, default=20260710)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project_relative(path: Path) -> str:
    path = path if path.is_absolute() else PROJECT_ROOT / path
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def json_loads(value: Any) -> Any:
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, (list, dict)):
        return value
    return []


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT = font(14)
FONT_BOLD = font(16, True)
FONT_BIG = font(22, True)


def box_from(row: dict[str, Any], prefix: str) -> tuple[float, float, float, float]:
    return (safe_float(row[f"{prefix}_bbox_x1"]), safe_float(row[f"{prefix}_bbox_y1"]), safe_float(row[f"{prefix}_bbox_x2"]), safe_float(row[f"{prefix}_bbox_y2"]))


def union_box(*boxes: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes))


def expand_box(box: tuple[float, float, float, float], width: int, height: int, ratio: float = 0.45, min_pad: float = 28.0) -> tuple[int, int, int, int]:
    pad_x = max((box[2] - box[0]) * ratio, min_pad)
    pad_y = max((box[3] - box[1]) * ratio, min_pad)
    x1 = max(0, min(int(round(box[0] - pad_x)), width - 1))
    y1 = max(0, min(int(round(box[1] - pad_y)), height - 1))
    x2 = max(x1 + 1, min(int(round(box[2] + pad_x)), width))
    y2 = max(y1 + 1, min(int(round(box[3] + pad_y)), height))
    return x1, y1, x2, y2


def resolve_image_path(row: dict[str, Any], image_by_key: dict[str, dict[str, Any]]) -> Path:
    image_row = image_by_key.get(row["image_key"], {})
    path = Path(str(image_row.get("image_path", "")))
    if path.exists():
        return path
    folder = "train2015" if row.get("split") == "trainval" else "test2015"
    return Path("/share/share/Datasets/hico_20160224_det/images") / folder / str(row.get("file_name"))


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def select_top10(stage7: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if stage7.empty:
        return pd.DataFrame()
    ids = stage7["candidate_id"].astype(str).tolist()
    merged = features[features["candidate_id"].astype(str).isin(ids)].copy()
    metadata = stage7[["candidate_id", "round2_id", "experiment_id", "gap_addressed", "manual_status"]].drop_duplicates("candidate_id")
    merged = merged.merge(metadata, on="candidate_id", how="left")
    merged = merged[
        (~merged["file_name"].isin(LEGACY_NEGATIVE_FILES))
        & (merged["target_object_scale_risk_auto"] == "good_ge50px")
        & (merged["combined_distinct_verbset_count_auto"] >= 2)
        & ((merged["target_human_other_gt_hoi_count"] >= 1) | (merged["recognizable_context_count_auto"] >= 2) | merged["second_pair_recognizable_auto"].map(safe_bool))
    ].copy()
    if merged.empty:
        return merged
    merged["narrative_diversity_score"] = (
        merged["target_human_other_gt_hoi_count"].clip(upper=4)
        + merged["recognizable_context_count_auto"].clip(upper=4)
        + merged["combined_distinct_verbset_count_auto"].clip(upper=4)
        + merged["second_pair_recognizable_auto"].map(lambda value: 2 if safe_bool(value) else 0)
        + merged["layout_empty_grid_fraction_auto"] * 3
        - merged["bbox_overlap_mean_iou_auto"].clip(upper=1) * 2
    )
    merged = merged.sort_values(["narrative_diversity_score", "target_object_scaled_short_edge_px", "auto_pre_score"], ascending=[False, False, False])
    selected = []
    used_objects: Counter[str] = Counter()
    used_experiments: Counter[str] = Counter()
    used_images: set[str] = set()
    for row in merged.to_dict("records"):
        if len(selected) >= 10:
            break
        object_name = str(row["object_category_name"])
        experiment_id = str(row.get("experiment_id", "unknown"))
        if used_objects[object_name] >= 2:
            continue
        if used_experiments[experiment_id] >= 3:
            continue
        if row["image_key"] in used_images and len(selected) < 7:
            continue
        used_objects[object_name] += 1
        used_experiments[experiment_id] += 1
        used_images.add(str(row["image_key"]))
        selected.append(row)
    if len(selected) < min(10, len(merged)):
        used_ids = {row["candidate_id"] for row in selected}
        for row in merged.to_dict("records"):
            if len(selected) >= 10:
                break
            if row["candidate_id"] not in used_ids:
                selected.append(row)
                used_ids.add(row["candidate_id"])
    top = pd.DataFrame(selected)
    top["top10_rank"] = range(1, len(top) + 1)
    return top


def crop_tile(image: Image.Image, box: tuple[float, float, float, float], title: str, color: tuple[int, int, int]) -> Image.Image:
    crop = image.crop(expand_box(box, image.width, image.height))
    crop.thumbnail((240, 190), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (250, 230), (248, 250, 252))
    tile.paste(crop, ((250 - crop.width) // 2, 0))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, 249, 191), outline=color, width=3)
    draw.text((6, 200), title[:30], fill=color, font=FONT)
    return tile


def make_cue_crops(row: dict[str, Any], image_by_key: dict[str, dict[str, Any]], out_path: Path) -> None:
    image = load_image(resolve_image_path(row, image_by_key))
    human = box_from(row, "human")
    obj = box_from(row, "object")
    union = union_box(human, obj)
    crop_box = (safe_float(row["suggested_crop_x1"]), safe_float(row["suggested_crop_y1"]), safe_float(row["suggested_crop_x2"]), safe_float(row["suggested_crop_y2"]))
    tiles = [
        crop_tile(image, human, "[GT] target human", (220, 38, 38)),
        crop_tile(image, obj, "[GT] target object", (37, 99, 235)),
        crop_tile(image, union, "[AUTO] union", (245, 158, 11)),
        crop_tile(image, crop_box, "[AUTO] suggested crop", (71, 85, 105)),
    ]
    canvas = Image.new("RGB", (500, 500), (226, 232, 240))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"{row['top10_id']} cue crops", fill=(15, 23, 42), font=FONT_BIG)
    for idx, tile in enumerate(tiles):
        x = (idx % 2) * 250
        y = 50 + (idx // 2) * 230
        canvas.paste(tile, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def make_storyboard(row: dict[str, Any], out_path: Path) -> None:
    width, height = 1600, 900
    canvas = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    panels = [
        ("Pair-centric", "Only target human + object.\nAmbiguity remains among prototypes."),
        ("Scene-level", "Global entities may help or mislead.\nNo model attention/probability shown."),
        ("Neighborhood", "Nearby/same-class nodes retained.\nProximity is not evidence."),
        ("EvidenceHOI", "Keep pair-specific local + context cues.\nDo not copy evidence from second person."),
    ]
    panel_w, panel_h = 760, 350
    positions = [(30, 90), (810, 90), (30, 500), (810, 500)]
    draw.text((30, 25), f"{row['top10_id']} conceptual storyboard / schematic (not model output)", fill=(15, 23, 42), font=FONT_BIG)
    for (title, body), (x, y) in zip(panels, positions):
        draw.rectangle((x, y, x + panel_w, y + panel_h), outline=(15, 23, 42), width=3)
        draw.text((x + 15, y + 12), title, fill=(15, 23, 42), font=FONT_BIG)
        # Schematic nodes; not copied from model internals.
        hx, hy = x + 150, y + 175
        ox, oy = x + 320, y + 175
        draw.ellipse((hx - 34, hy - 34, hx + 34, hy + 34), outline=(220, 38, 38), width=5)
        draw.ellipse((ox - 34, oy - 34, ox + 34, oy + 34), outline=(37, 99, 235), width=5)
        draw.line((hx + 34, hy, ox - 34, oy), fill=(245, 158, 11), width=4)
        draw.text((hx - 22, hy - 8), "H", fill=(220, 38, 38), font=FONT_BOLD)
        draw.text((ox - 22, oy - 8), "O", fill=(37, 99, 235), font=FONT_BOLD)
        if title in {"Scene-level", "Neighborhood", "EvidenceHOI"}:
            cx, cy = x + 520, y + 130
            nx, ny = x + 560, y + 250
            draw.ellipse((cx - 28, cy - 28, cx + 28, cy + 28), outline=(22, 163, 74), width=4)
            draw.ellipse((nx - 28, ny - 28, nx + 28, ny + 28), outline=(107, 114, 128), width=4)
            draw.text((cx - 14, cy - 8), "C", fill=(22, 163, 74), font=FONT_BOLD)
            draw.text((nx - 14, ny - 8), "N", fill=(107, 114, 128), font=FONT_BOLD)
            if title == "EvidenceHOI":
                draw.line((hx + 25, hy - 20, cx - 28, cy), fill=(22, 163, 74), width=3)
            else:
                draw.line((ox + 20, oy + 20, nx - 28, ny), fill=(107, 114, 128), width=2)
        draw.text((x + 15, y + 245), body, fill=(51, 65, 85), font=FONT)
        draw.text((x + 15, y + 305), f"GT verb: {row['gt_verb_names_json']} | object: {row['object_category_name']}", fill=(15, 23, 42), font=FONT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)


def prototypes_for(row: dict[str, Any]) -> list[str]:
    object_name = str(row["object_category_name"])
    gt = set(json_loads(row.get("gt_verb_names_json")))
    proto = PROTOTYPES_BY_OBJECT.get(object_name, [])
    merged = []
    for value in list(gt) + proto:
        if value not in merged:
            merged.append(value)
    return merged[:8]


def build_rows(top: pd.DataFrame, image_by_key: dict[str, dict[str, Any]], cue_dir: Path, storyboard_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix_rows = []
    risk_rows = []
    for row in top.to_dict("records"):
        top_id = f"T10_{safe_int(row['top10_rank']):02d}"
        row["top10_id"] = top_id
        cue_path = cue_dir / f"{top_id}_cue_crops.jpg"
        storyboard_path = storyboard_dir / f"{top_id}_storyboard_schematic.jpg"
        make_cue_crops(row, image_by_key, cue_path)
        make_storyboard(row, storyboard_path)
        human_bbox = [safe_float(row["human_bbox_x1"]), safe_float(row["human_bbox_y1"]), safe_float(row["human_bbox_x2"]), safe_float(row["human_bbox_y2"])]
        object_bbox = [safe_float(row["object_bbox_x1"]), safe_float(row["object_bbox_y1"]), safe_float(row["object_bbox_x2"]), safe_float(row["object_bbox_y2"])]
        local_risk = "requires_manual_visual_review; do not infer hand/contact/state from bbox proxies"
        max_risk = "manual visual evidence unavailable; top10 is a proxy shortlist, not final candidate"
        if safe_float(row["bbox_overlap_mean_iou_auto"]) > 0.3:
            max_risk += "; crowded/overlap proxy high"
        if safe_float(row["target_object_scaled_short_edge_px"]) < 60:
            max_risk += "; near paper-scale readability threshold"
        matrix_rows.append({
            "top10_id": top_id,
            "candidate_id": row["candidate_id"],
            "pair_id": row["pair_id"],
            "image_id": row["image_id"],
            "split": row["split"],
            "dataset_source": "HICO-DET",
            "target_human_id": row["canonical_human_id"],
            "target_human_bbox": json_dumps(human_bbox),
            "target_object_id": row["canonical_object_id"],
            "target_object_bbox": json_dumps(object_bbox),
            "target_object_category": row["object_category_name"],
            "target_gt_verb": row["gt_verb_names_json"],
            "confusing_verb_prototypes": json_dumps(prototypes_for(row)),
            "tier": row["target_tier"],
            "second_human_or_second_pair": "[AUTO] second_pair_recognizable_auto=" + str(row["second_pair_recognizable_auto"]),
            "target_human_state": "requires_manual_visual_review",
            "target_object_state": "requires_manual_visual_review",
            "ho_pair_union_relation": f"[AUTO] union_norm_area={safe_float(row['target_union_bbox_norm_area']):.4f}; human_object_iou={safe_float(row['human_object_bbox_iou']):.4f}",
            "discriminative_local_cue": "requires_manual_visual_review; cue crop generated",
            "scene_cue": f"[AUTO] recognizable_context_count={safe_int(row['recognizable_context_count_auto'])}; scene cue requires manual validation",
            "context_object_cue": f"[GT/AUTO] target_human_other_gt_hoi_count={safe_int(row['target_human_other_gt_hoi_count'])}; possible_context_count={safe_int(row['possible_context_count'])}",
            "context_basis": "GT if auxiliary HOI exists; otherwise auto candidate requiring manual review",
            "scene_level_noise": "requires_manual_visual_review; unannotated objects are not confirmed noise",
            "proximity_noise": "requires_manual_visual_review; proximity alone is not evidence",
            "max_narrative_risk": max_risk,
            "pair_centric_answer": "Only human/object boxes and object category are available here; prototypes such as " + ", ".join(prototypes_for(row)[:4]) + " may remain difficult without local state/contact cues.",
            "scene_level_answer": "Global entities may provide scene priors, but unrelated entities can propagate errors; no attention/probability is claimed.",
            "neighborhood_answer": "Nearby/same-class entities may be retained by spatial/category heuristics, but closeness/similarity is not pair-specific evidence.",
            "evidencehoi_answer": "Should retain current-pair local cue, target-human auxiliary context, and third-party object cues only when visually confirmed for this pair; evidence must not be copied from second person.",
            "one_sentence_answer": "For this target pair, GT/AUTO proxies suggest candidate local/context cues, but manual review must decide which cues change action judgment and which nearby/scene entities cause false propagation.",
            "cue_crops_path": project_relative(cue_path),
            "storyboard_mockup_path": project_relative(storyboard_path),
            "storyboard_label": "conceptual storyboard / schematic; not model output; no attention/probability shown",
        })
        risk_rows.append({
            "top10_id": top_id,
            "candidate_id": row["candidate_id"],
            "risk_type": "manual_visual_unavailable",
            "risk_level": "high",
            "risk_description": max_risk,
            "mitigation": "Requires image-capable manual review before final-three selection.",
        })
        risk_rows.append({
            "top10_id": top_id,
            "candidate_id": row["candidate_id"],
            "risk_type": "gt_vs_manual_boundary",
            "risk_level": "medium",
            "risk_description": "Context and noise labels are proxies unless backed by auxiliary GT and visual confirmation.",
            "mitigation": "Keep [GT], [AUTO], and [MANUAL] labels separate in final figure design.",
        })
    return matrix_rows, risk_rows


def write_analysis(path: Path, matrix_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Stage 8 Top-10 Deep Analysis",
        "",
        "Status: proxy top-10 selected for image-capable manual analysis.",
        "",
        "Stage 6 manual visual review was blocked by the runtime's lack of image input. Therefore this report uses [GT] fields and [AUTO] proxy fields only. It does not claim target human state, target object state, hand contact, scene cue, noise, or model behavior as confirmed.",
        "",
        "Storyboard images are labeled `conceptual storyboard / schematic`; they are not model outputs and contain no attention weights, probabilities, or inferred model selections.",
        "",
    ]
    for row in matrix_rows:
        lines.extend([
            f"## {row['top10_id']} | {row['dataset_source']} | {row['split']} | image {row['image_id']} | {row['tier']} {row['target_object_category']}",
            "",
            f"- [GT] Target pair: human `{row['target_human_id']}` to object `{row['target_object_id']}`; GT verb {row['target_gt_verb']}.",
            f"- [AUTO] Candidate confusing prototypes: `{row['confusing_verb_prototypes']}`.",
            f"- [AUTO] Union relation: {row['ho_pair_union_relation']}.",
            f"- [MANUAL] Target human state: {row['target_human_state']}.",
            f"- [MANUAL] Target object state: {row['target_object_state']}.",
            f"- [MANUAL] Local cue: {row['discriminative_local_cue']}.",
            f"- [AUTO/MANUAL] Scene cue: {row['scene_cue']}.",
            f"- [GT/AUTO/MANUAL] Context cue: {row['context_object_cue']} | basis: {row['context_basis']}.",
            f"- [MANUAL] Scene-level noise: {row['scene_level_noise']}.",
            f"- [MANUAL] Proximity noise: {row['proximity_noise']}.",
            f"- Risk: {row['max_narrative_risk']}.",
            "",
            "Four-method narrative:",
            f"- Pair-centric: {row['pair_centric_answer']}",
            f"- Scene-level: {row['scene_level_answer']}",
            f"- Neighborhood-based: {row['neighborhood_answer']}",
            f"- EvidenceHOI: {row['evidencehoi_answer']}",
            "",
            f"One-sentence check: {row['one_sentence_answer']}",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception:
        return None
    return result.stdout.strip()


def build_stage8(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cue_dir = args.output_dir / "cue_crops"
    storyboard_dir = args.output_dir / "storyboard_mockups"
    cue_dir.mkdir(parents=True, exist_ok=True)
    storyboard_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features)
    stage7 = pd.read_csv(args.stage7_shortlist) if args.stage7_shortlist.exists() else pd.DataFrame()
    stage6 = pd.read_csv(args.stage6_shortlist) if args.stage6_shortlist.exists() else pd.DataFrame()
    images = pd.read_parquet(args.image_summary)
    image_by_key = {row["image_key"]: row for row in images.to_dict("records")}

    top = select_top10(stage7, features)
    matrix_rows, risk_rows = build_rows(top, image_by_key, cue_dir, storyboard_dir)
    matrix_path = args.output_dir / "top10_matrix.csv"
    risk_path = args.output_dir / "risk_register.csv"
    analysis_path = args.output_dir / "top10_analysis.md"
    final_three_path = args.output_dir / "final_three.csv"
    pd.DataFrame(matrix_rows).to_csv(matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)
    pd.DataFrame(risk_rows).to_csv(risk_path, index=False, quoting=csv.QUOTE_MINIMAL)
    write_analysis(analysis_path, matrix_rows)
    final_three_columns = [
        "top10_id", "candidate_id", "pair_id", "image_id", "split", "dataset_source", "target_object_category",
        "target_gt_verb", "final_three_status", "reason",
    ]
    pd.DataFrame(columns=final_three_columns).to_csv(final_three_path, index=False, quoting=csv.QUOTE_MINIMAL)

    output_paths = [matrix_path, risk_path, analysis_path, final_three_path, *sorted(cue_dir.glob("*.jpg")), *sorted(storyboard_dir.glob("*.jpg")), Path(__file__)]
    acceptance = {
        "top10_candidates_have_specific_target_pair": bool(matrix_rows) and all(row.get("pair_id") for row in matrix_rows),
        "six_visual_evidence_classes_not_claimed_without_manual_review": True,
        "four_method_differences_are_conceptual_not_color_only": True,
        "context_noise_pair_specific_explanations_present_as_require_manual_review": bool(matrix_rows) and all(row.get("context_basis") and row.get("proximity_noise") for row in matrix_rows),
        "no_manual_inference_claimed_as_gt_or_model_output": True,
        "final_three_deferred_due_to_missing_manual_visual_review": True,
        "passed": False,
    }
    manifest = {
        "schema_version": "figure1_search_v2.stage8",
        "stage": 8,
        "stage_name": "top10_proxy_deep_analysis_and_storyboard_schematics",
        "status": "completed_with_final_three_blocked",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "random_seed": args.seed,
        "successful_command": " ".join(shlex.quote(part) for part in [sys.executable, "-m", "tools.evidencehoi_image_search_v2.build_top10_stage8", "--features", project_relative(args.features), "--stage6-shortlist", project_relative(args.stage6_shortlist), "--stage7-shortlist", project_relative(args.stage7_shortlist), "--stage7-delta", project_relative(args.stage7_delta), "--image-summary", project_relative(args.image_summary), "--output-dir", project_relative(args.output_dir), "--manifest-path", project_relative(args.manifest_path), "--seed", str(args.seed)]),
        "inputs": {
            "stage6_shortlist_rows": int(len(stage6)),
            "stage7_shortlist_rows": int(len(stage7)),
        },
        "outputs": {
            "top10_analysis": project_relative(analysis_path),
            "top10_matrix": project_relative(matrix_path),
            "cue_crops": project_relative(cue_dir),
            "storyboard_mockups": project_relative(storyboard_dir),
            "risk_register": project_relative(risk_path),
            "final_three": project_relative(final_three_path),
        },
        "statistics": {
            "top10_rows": len(matrix_rows),
            "risk_rows": len(risk_rows),
            "cue_crop_sheets": len(list(cue_dir.glob("*.jpg"))),
            "storyboard_mockups": len(list(storyboard_dir.glob("*.jpg"))),
            "final_three_rows": 0,
        },
        "artifact_hashes": {project_relative(path): sha256(path) for path in output_paths if path.exists()},
        "acceptance": acceptance,
        "blocker": "Current runtime cannot inspect image content; final_three requires manual visual confirmation of six evidence classes.",
        "next_stage_ready": True,
        "next_stage_recommendation": "Use an image-capable reviewer to validate top10_matrix rows, fill true MANUAL fields, and then choose final_three.",
        "next_stage_started": False,
        "git_commit": git_commit(),
    }
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = build_stage8(args)
    print(json.dumps({"status": manifest["status"], "statistics": manifest["statistics"], "acceptance": manifest["acceptance"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
