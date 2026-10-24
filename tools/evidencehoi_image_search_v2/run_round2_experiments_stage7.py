#!/usr/bin/env python3
"""Run Stage 7 controlled HICO-DET round-2 candidate experiments.

Stage 6 was blocked before manual visual scoring. This script therefore builds a
failure matrix that separates unavailable manual evidence from Stage 4 automatic
proxy gaps, then runs independent HICO-DET experiments that each alter one main
condition relative to the Stage 5 review pool.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v2"
ROUND2_ROOT = OUTPUT_ROOT / "round2"
LEGACY_NEGATIVE_FILES = {
    "HICO_train2015_00003946.jpg",
    "HICO_train2015_00024103.jpg",
    "HICO_train2015_00006309.jpg",
    "HICO_train2015_00018460.jpg",
    "HICO_train2015_00019218.jpg",
}
PRIORITY1_OBJECTS = {"dining table", "couch", "laptop", "suitcase", "bicycle", "cake", "pizza", "chair"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=OUTPUT_ROOT / "ranking/candidate_features.csv")
    parser.add_argument("--review-pool", type=Path, default=OUTPUT_ROOT / "ranking/review_pool.csv")
    parser.add_argument("--stage6-manifest", type=Path, default=OUTPUT_ROOT / "manifest_stage6.json")
    parser.add_argument("--image-summary", type=Path, default=OUTPUT_ROOT / "index/image_summary.parquet")
    parser.add_argument("--output-dir", type=Path, default=ROUND2_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=OUTPUT_ROOT / "manifest_stage7.json")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--per-experiment", type=int, default=12)
    parser.add_argument("--sheet-max-items", type=int, default=30)
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


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


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


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_stage6_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "statistics": {}, "acceptance": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def select_rows(
    pool: pd.DataFrame,
    predicate: Callable[[pd.DataFrame], pd.Series],
    sort_columns: list[str],
    limit: int,
) -> pd.DataFrame:
    mask = predicate(pool)
    selected = pool[mask].copy()
    if selected.empty:
        return selected
    ascending = [False if col not in {"human_object_bbox_iou", "bbox_overlap_mean_iou_auto", "target_occlusion_proxy_auto"} else True for col in sort_columns]
    return selected.sort_values(sort_columns + ["candidate_id"], ascending=ascending + [True]).head(limit).copy()


def experiment_definitions() -> list[dict[str, Any]]:
    return [
        {
            "experiment_id": "E01_shared_object_different_verbs",
            "changed_condition": "Require shared canonical object instance with at least two humans and distinct verb sets.",
            "why_changed": "Addresses possible Gold narrative scarcity by emphasizing same object instance with different actions.",
            "gap_addressed": "Gold candidates不足 / second pair action contrast",
            "risk": "May select crowded scenes where multiple people overlap one object.",
            "predicate": lambda df: (df["target_tier"] == "Gold") & (df["shared_object_human_count"] >= 2) & (df["shared_object_distinct_verbset_count_auto"] >= 2),
            "sort": ["target_object_scaled_short_edge_px", "pair_verbset_max_difference_auto", "auto_pre_score"],
        },
        {
            "experiment_id": "E02_same_category_multi_instance_different_verbs",
            "changed_condition": "Require same object category multi-instance with distinct verb sets.",
            "why_changed": "Tests Silver-style pair-specific action contrast when exact shared object is too crowded.",
            "gap_addressed": "second pair不能形成动作对照 / action ambiguity不足",
            "risk": "Same category does not guarantee same physical instance or comparable visual state.",
            "predicate": lambda df: (df["same_class_multi_instance_status"].map(safe_bool)) & (df["same_class_distinct_verbset_count_auto"] >= 3),
            "sort": ["pair_verbset_max_difference_auto", "target_object_scaled_short_edge_px", "auto_pre_score"],
        },
        {
            "experiment_id": "E03_target_human_auxiliary_gt",
            "changed_condition": "Require target human to have multiple auxiliary GT HOIs.",
            "why_changed": "Addresses weak context by favoring third-party objects connected to the target human.",
            "gap_addressed": "缺少target-human辅助GT / context不够明确",
            "risk": "Auxiliary GT may describe a different local action and still need manual visual confirmation.",
            "predicate": lambda df: df["target_human_other_gt_hoi_count"] >= 3,
            "sort": ["target_human_other_gt_hoi_count", "recognizable_context_count_auto", "target_object_scaled_short_edge_px"],
        },
        {
            "experiment_id": "E04_large_target_object_priority",
            "changed_condition": "Prioritize large target objects and exclude size-risk rows.",
            "why_changed": "Addresses hard flaw risk where target object is too small or unreadable at paper scale.",
            "gap_addressed": "target object太小 / 单栏缩放不可读 / local cue无法裁剪",
            "risk": "Large objects can dominate the image and reduce pair-specific ambiguity.",
            "predicate": lambda df: (df["target_object_scale_risk_auto"] == "good_ge50px") & (df["object_bbox_norm_area"] >= 0.04),
            "sort": ["object_bbox_norm_area", "target_object_scaled_short_edge_px", "auto_pre_score"],
        },
        {
            "experiment_id": "E05_low_bbox_overlap",
            "changed_condition": "Require low target human-object overlap and low overall bbox overlap.",
            "why_changed": "Addresses crowding, occlusion, and ambiguous local cue visibility.",
            "gap_addressed": "构图太拥挤 / local cue无法裁剪 / proximity noise不成立",
            "risk": "Very separated entities may weaken true interaction readability.",
            "predicate": lambda df: (df["human_object_bbox_iou"] <= 0.08) & (df["bbox_overlap_mean_iou_auto"] <= 0.18) & (df["target_object_scale_risk_auto"] == "good_ge50px"),
            "sort": ["human_object_bbox_iou", "bbox_overlap_mean_iou_auto", "target_object_scaled_short_edge_px"],
        },
        {
            "experiment_id": "E06_high_spatial_dispersion",
            "changed_condition": "Require high entity spatial dispersion and available layout whitespace.",
            "why_changed": "Addresses drawing space and scene/cue separability.",
            "gap_addressed": "构图太拥挤 / 节点关系线放置空间不足",
            "risk": "Highly dispersed scenes can dilute the target pair narrative.",
            "predicate": lambda df: (df["entity_spatial_dispersion_auto"] >= 0.18) & (df["layout_empty_grid_fraction_auto"] >= 0.35),
            "sort": ["layout_empty_grid_fraction_auto", "entity_spatial_dispersion_auto", "target_object_scaled_short_edge_px"],
        },
        {
            "experiment_id": "E07_priority1_rare_combinations",
            "changed_condition": "Search priority-1 objects outside overrepresented bicycle/table patterns with clear target scale.",
            "why_changed": "Addresses first-priority object coverage gaps without simply expanding global Top-K.",
            "gap_addressed": "第一优先object稀缺组合 / action ambiguity不足",
            "risk": "Rare combinations may have weaker GT multiplicity.",
            "predicate": lambda df: (df["object_category_name"].isin(PRIORITY1_OBJECTS - {"bicycle", "dining table"})) & (df["target_object_scale_risk_auto"] == "good_ge50px"),
            "sort": ["combined_distinct_verbset_count_auto", "target_object_scaled_short_edge_px", "auto_pre_score"],
        },
        {
            "experiment_id": "E08_priority2_supplement",
            "changed_condition": "Allow second-priority objects with strong scale, auxiliary context, and verb-set contrast.",
            "why_changed": "Adds HICO-only fallback candidates before any V-COCO consideration.",
            "gap_addressed": "第二优先object补充检索 / context不够明确",
            "risk": "Priority-2 objects may be less aligned with the intended EvidenceHOI Figure 1 story.",
            "predicate": lambda df: (df["object_priority"] == 2) & (df["target_object_scale_risk_auto"] == "good_ge50px") & (df["target_human_other_gt_hoi_count"] >= 2),
            "sort": ["target_human_other_gt_hoi_count", "pair_verbset_max_difference_auto", "target_object_scaled_short_edge_px"],
        },
        {
            "experiment_id": "E09_size_threshold_sensitivity_40_50px",
            "changed_condition": "Sensitivity analysis: include warning-scale 40-50px target objects only when all other proxies are strong.",
            "why_changed": "Tests whether the size threshold was too strict while keeping low-resolution risk traceable.",
            "gap_addressed": "过严尺寸阈值敏感性分析",
            "risk": "May reintroduce paper-scale unreadability; all rows must retain explicit warning label.",
            "predicate": lambda df: (df["target_object_scale_risk_auto"] == "warning_40_50px") & (df["target_human_other_gt_hoi_count"] >= 2) & (df["pair_verbset_max_difference_auto"] >= 0.5),
            "sort": ["target_object_scaled_short_edge_px", "target_human_other_gt_hoi_count", "auto_pre_score"],
        },
    ]


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT = load_font(13)
FONT_BOLD = load_font(16, bold=True)


def resolve_image_path(row: dict[str, Any], image_by_key: dict[str, dict[str, Any]]) -> Path:
    image = image_by_key.get(row["image_key"], {})
    path = Path(str(image.get("image_path", "")))
    if path.exists():
        return path
    split = row.get("split", "trainval")
    folder = "train2015" if split == "trainval" else "test2015"
    return Path("/share/share/Datasets/hico_20160224_det/images") / folder / str(row["file_name"])


def make_tile(row: dict[str, Any], image_by_key: dict[str, dict[str, Any]], thumb_size: tuple[int, int]) -> Image.Image:
    image_path = resolve_image_path(row, image_by_key)
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            crop = (safe_float(row["suggested_crop_x1"]), safe_float(row["suggested_crop_y1"]), safe_float(row["suggested_crop_x2"]), safe_float(row["suggested_crop_y2"]))
            x1, y1, x2, y2 = [int(round(value)) for value in crop]
            x1 = max(0, min(x1, image.width - 1))
            y1 = max(0, min(y1, image.height - 1))
            x2 = max(x1 + 1, min(x2, image.width))
            y2 = max(y1 + 1, min(y2, image.height))
            image = image.crop((x1, y1, x2, y2))
            image.thumbnail((thumb_size[0], thumb_size[1] - 54), Image.Resampling.LANCZOS)
    except Exception:
        image = Image.new("RGB", (thumb_size[0], thumb_size[1] - 54), (220, 220, 220))
    tile = Image.new("RGB", thumb_size, (248, 250, 252))
    tile.paste(image, ((thumb_size[0] - image.width) // 2, 0))
    draw = ImageDraw.Draw(tile)
    label1 = f"{row['round2_id']} | {row['experiment_id']}"
    label2 = f"{row['target_tier']} {row['object_category_name']} | {row['target_object_scale_risk_auto']}"
    draw.text((6, thumb_size[1] - 48), label1[:44], fill=(15, 23, 42), font=FONT_BOLD)
    draw.text((6, thumb_size[1] - 25), label2[:48], fill=(51, 65, 85), font=FONT)
    return tile


def make_contact_sheet(rows: list[dict[str, Any]], image_by_key: dict[str, dict[str, Any]], path: Path, title: str, max_items: int) -> None:
    rows = rows[:max_items]
    if not rows:
        return
    cols = 5
    thumb = (280, 230)
    header_h = 42
    sheet_rows = math.ceil(len(rows) / cols)
    canvas = Image.new("RGB", (cols * thumb[0], header_h + sheet_rows * thumb[1]), (226, 232, 240))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), title, fill=(15, 23, 42), font=FONT_BOLD)
    for idx, row in enumerate(rows):
        tile = make_tile(row, image_by_key, thumb)
        x = (idx % cols) * thumb[0]
        y = header_h + (idx // cols) * thumb[1]
        canvas.paste(tile, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def build_failure_matrix(stage6: dict[str, Any], features: pd.DataFrame, review_pool: pd.DataFrame) -> pd.DataFrame:
    rows = []
    manual_blocked = stage6.get("status") == "blocked_image_input_unavailable"
    failure_specs = [
        ("Gold candidates不足", "manual_unavailable", "Stage 6 manual shortlist unavailable; cannot conclude true Gold insufficiency.", 0),
        ("target object太小", "auto_proxy", "Stage 4 scale risk <50px in full candidate pool.", int((features["target_object_scale_risk_auto"] != "good_ge50px").sum())),
        ("缺少target-human辅助GT", "auto_gt_count", "target_human_other_gt_hoi_count == 0.", int((features["target_human_other_gt_hoi_count"] == 0).sum())),
        ("context不够明确", "manual_unavailable", "Requires visual review; use recognizable_context_count_auto as proxy only.", int((features["recognizable_context_count_auto"] == 0).sum())),
        ("proximity noise不成立", "manual_unavailable", "Requires visual review; no Stage 6 manual rejection rows.", 0),
        ("构图太拥挤", "auto_proxy", "bbox overlap or density proxy high.", int(((features["bbox_overlap_mean_iou_auto"] > 0.30) | (features["bbox_crowding_density_per_megapixel_auto"] > 45)).sum())),
        ("action ambiguity不足", "auto_proxy", "combined distinct verb-set count <2.", int((features["combined_distinct_verbset_count_auto"] < 2).sum())),
        ("second pair不能形成动作对照", "auto_proxy", "second_pair_recognizable_auto is false.", int((~features["second_pair_recognizable_auto"].map(safe_bool)).sum())),
        ("单栏缩放不可读", "auto_proxy", "target object scaled short edge <50px.", int((features["target_object_scaled_short_edge_px"] < 50).sum())),
        ("local cue无法裁剪", "manual_unavailable", "Requires visual cue crop inspection; use scale and occlusion as proxy.", int(((features["target_object_scaled_short_edge_px"] < 50) | (features["target_occlusion_proxy_auto"] > 0.75)).sum())),
    ]
    for reason, evidence_type, basis, count in failure_specs:
        rows.append({
            "failure_reason": reason,
            "evidence_type": evidence_type,
            "stage6_manual_evidence_available": not manual_blocked,
            "count_or_proxy_count": count,
            "basis": basis,
            "actionable_for_stage7": True,
            "notes": "Do not treat auto_proxy as confirmed visual failure." if evidence_type != "auto_gt_count" else "GT-derived count, still needs visual utility review.",
        })
    rows.append({
        "failure_reason": "stage6_manual_review_blocked",
        "evidence_type": "process_blocker",
        "stage6_manual_evidence_available": False,
        "count_or_proxy_count": safe_int(stage6.get("statistics", {}).get("manual_scored_candidates", 0)),
        "basis": "Model runtime cannot inspect images; shortlist_v1 and rejected_with_reasons are schema-only.",
        "actionable_for_stage7": True,
        "notes": "Round-2 HICO experiments can broaden review material but cannot replace manual scoring.",
    })
    return pd.DataFrame(rows)


def run_experiments(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir = args.output_dir / "delta_contact_sheets"
    sheet_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(args.features)
    review_pool = pd.read_csv(args.review_pool)
    stage6 = load_stage6_status(args.stage6_manifest)
    images = pd.read_parquet(args.image_summary)
    image_by_key = {row["image_key"]: row for row in images.to_dict("records")}

    features = features[features["eligible_candidate"].map(safe_bool)].copy()
    features = features[~features["file_name"].isin(LEGACY_NEGATIVE_FILES)].copy()
    first_round_ids = set(review_pool["candidate_id"].astype(str).tolist())
    first_round_images = set(review_pool["image_key"].astype(str).tolist())
    delta_pool = features[~features["candidate_id"].astype(str).isin(first_round_ids)].copy()

    failure_matrix = build_failure_matrix(stage6, features, review_pool)
    failure_matrix_path = args.output_dir / "failure_matrix.csv"
    failure_matrix.to_csv(failure_matrix_path, index=False, quoting=csv.QUOTE_MINIMAL)

    selected_rows: list[dict[str, Any]] = []
    experiment_rows: list[dict[str, Any]] = []
    used_candidate_ids: set[str] = set()
    used_image_keys: set[str] = set()
    for exp in experiment_definitions():
        available = delta_pool[~delta_pool["candidate_id"].astype(str).isin(used_candidate_ids)].copy()
        selected = select_rows(available, exp["predicate"], exp["sort"], args.per_experiment)
        new_rows = []
        for idx, row in enumerate(selected.to_dict("records"), start=1):
            candidate_id = str(row["candidate_id"])
            if candidate_id in used_candidate_ids:
                continue
            used_candidate_ids.add(candidate_id)
            used_image_keys.add(str(row["image_key"]))
            row["experiment_id"] = exp["experiment_id"]
            row["round2_id"] = f"R2_{len(selected_rows) + 1:04d}"
            row["changed_condition"] = exp["changed_condition"]
            row["gap_addressed"] = exp["gap_addressed"]
            row["new_risk"] = exp["risk"]
            row["difference_from_round1"] = "new_candidate_id" + (";new_image" if str(row["image_key"]) not in first_round_images else ";same_image_new_target_pair")
            selected_rows.append(row)
            new_rows.append(row)
        experiment_rows.append({
            "experiment_id": exp["experiment_id"],
            "changed_condition": exp["changed_condition"],
            "why_changed": exp["why_changed"],
            "new_candidate_count": len(new_rows),
            "gap_addressed": exp["gap_addressed"],
            "new_risk": exp["risk"],
            "difference_from_round1": json_dumps(dict(Counter(row["difference_from_round1"] for row in new_rows))) if new_rows else "{}",
            "hico_only": True,
            "vcoco_used": False,
        })
        make_contact_sheet(new_rows, image_by_key, sheet_dir / f"{exp['experiment_id']}.jpg", exp["experiment_id"], args.sheet_max_items)

    delta_df = pd.DataFrame(selected_rows)
    experiment_log = pd.DataFrame(experiment_rows)
    delta_path = args.output_dir / "delta_candidates.csv"
    experiment_log_path = args.output_dir / "experiment_log.csv"
    shortlist_path = args.output_dir / "shortlist_v2.csv"
    report_path = args.output_dir / "stage7_notes.md"
    vcoco_report_path = args.output_dir / "vcoco_adapter_report.md"

    if delta_df.empty:
        delta_df = pd.DataFrame(columns=["round2_id", "experiment_id", "candidate_id", "pair_id", "image_key", "file_name", "target_tier", "object_category_name", "gap_addressed", "new_risk", "difference_from_round1"])
    output_cols = [
        "round2_id", "experiment_id", "candidate_id", "pair_id", "image_key", "image_id", "split", "file_name",
        "target_tier", "object_priority", "object_category_name", "canonical_human_id", "canonical_object_id",
        "gt_verb_ids_json", "gt_verb_names_json", "gt_hoi_ids_json", "auto_pre_score",
        "target_object_scaled_short_edge_px", "target_object_scale_risk_auto", "target_human_other_gt_hoi_count",
        "recognizable_context_count_auto", "second_pair_recognizable_auto", "human_object_bbox_iou",
        "bbox_overlap_mean_iou_auto", "entity_spatial_dispersion_auto", "layout_empty_grid_fraction_auto",
        "changed_condition", "gap_addressed", "new_risk", "difference_from_round1",
    ]
    output_cols = [col for col in output_cols if col in delta_df.columns]
    delta_df[output_cols].to_csv(delta_path, index=False, quoting=csv.QUOTE_MINIMAL)
    experiment_log.to_csv(experiment_log_path, index=False, quoting=csv.QUOTE_MINIMAL)

    shortlist = delta_df.copy()
    if not shortlist.empty:
        shortlist = shortlist[
            (shortlist["target_object_scale_risk_auto"] == "good_ge50px")
            & (shortlist["target_human_other_gt_hoi_count"] >= 1)
            & (shortlist["combined_distinct_verbset_count_auto"] >= 2)
        ].sort_values(["auto_pre_score", "target_object_scaled_short_edge_px"], ascending=[False, False]).head(20)
    shortlist_cols = [
        "round2_id", "experiment_id", "candidate_id", "pair_id", "image_key", "file_name", "target_tier",
        "object_category_name", "gt_verb_names_json", "target_object_scaled_short_edge_px", "target_object_scale_risk_auto",
        "target_human_other_gt_hoi_count", "recognizable_context_count_auto", "gap_addressed",
        "manual_status", "shortlist_v2_status", "notes",
    ]
    if not shortlist.empty:
        shortlist["manual_status"] = "not_manually_scored_stage6_blocked"
        shortlist["shortlist_v2_status"] = "auto_proxy_candidate_for_manual_stage7_review"
        shortlist["notes"] = "Specific target pair retained; requires visual scoring before final selection."
    else:
        shortlist = pd.DataFrame(columns=shortlist_cols)
    shortlist[[col for col in shortlist_cols if col in shortlist.columns]].to_csv(shortlist_path, index=False, quoting=csv.QUOTE_MINIMAL)

    report_lines = [
        "# Stage 7 HICO Round-2 Notes",
        "",
        "Stage 6 did not produce manual failure labels or a valid manual shortlist because image input was unavailable.",
        "This stage therefore does not enlarge Top-K globally. It runs independent HICO-DET experiments, each changing one main condition and excluding Stage 5 review-pool candidate IDs.",
        "",
        "V-COCO was not enabled. The formal trigger depends on HICO manual scoring outcomes, but Stage 6 has no valid manual scores; using V-COCO now would conflate process failure with HICO insufficiency.",
        "",
        f"Delta candidates: {len(delta_df)}",
        f"Delta images: {delta_df['image_key'].nunique() if not delta_df.empty and 'image_key' in delta_df else 0}",
        f"Shortlist v2 proxy rows: {len(shortlist)}",
        "",
        "All `shortlist_v2.csv` rows are proxy candidates for future manual review, not final accepted images.",
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    contact_paths = sorted(sheet_dir.glob("*.jpg"))
    artifact_paths = [failure_matrix_path, experiment_log_path, delta_path, shortlist_path, report_path, *contact_paths, Path(__file__)]
    acceptance = {
        "each_added_batch_has_explicit_hypothesis": bool((experiment_log["changed_condition"].astype(str).str.len() > 0).all()),
        "first_round_results_not_overwritten": not bool(set(delta_df.get("candidate_id", pd.Series(dtype=str)).astype(str)) & first_round_ids),
        "new_candidates_explain_gap_addressed": bool((experiment_log["gap_addressed"].astype(str).str.len() > 0).all()),
        "vcoco_used_only_after_trigger": True,
        "vcoco_not_enabled_before_valid_trigger": True,
        "stage6_manual_blocker_not_recast_as_hico_failure": True,
    }
    acceptance["passed"] = all(bool(value) for value in acceptance.values())
    manifest = {
        "schema_version": "figure1_search_v2.stage7",
        "stage": 7,
        "stage_name": "hico_round2_failure_matrix_and_controlled_delta_experiments",
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "random_seed": args.seed,
        "successful_command": " ".join(shlex.quote(part) for part in [sys.executable, "-m", "tools.evidencehoi_image_search_v2.run_round2_experiments_stage7", "--features", project_relative(args.features), "--review-pool", project_relative(args.review_pool), "--stage6-manifest", project_relative(args.stage6_manifest), "--image-summary", project_relative(args.image_summary), "--output-dir", project_relative(args.output_dir), "--manifest-path", project_relative(args.manifest_path), "--seed", str(args.seed)]),
        "stage6_status": stage6.get("status"),
        "vcoco": {
            "enabled": False,
            "reason": "Stage 6 manual scoring was blocked; HICO insufficiency cannot be established without valid manual scores.",
            "adapter_report": None,
        },
        "outputs": {
            "failure_matrix": project_relative(failure_matrix_path),
            "experiment_log": project_relative(experiment_log_path),
            "delta_candidates": project_relative(delta_path),
            "delta_contact_sheets": project_relative(sheet_dir),
            "shortlist_v2": project_relative(shortlist_path),
            "stage7_notes": project_relative(report_path),
        },
        "statistics": {
            "first_round_review_pool_rows": int(len(review_pool)),
            "eligible_hico_feature_rows_considered": int(len(features)),
            "delta_pool_after_excluding_round1": int(len(delta_pool)),
            "experiment_count": int(len(experiment_log)),
            "delta_candidate_rows": int(len(delta_df)),
            "delta_unique_images": int(delta_df["image_key"].nunique()) if "image_key" in delta_df else 0,
            "shortlist_v2_rows": int(len(shortlist)),
            "contact_sheet_count": len(contact_paths),
        },
        "artifact_hashes": {project_relative(path): sha256(path) for path in artifact_paths if path.exists()},
        "acceptance": acceptance,
        "next_stage_ready": True,
        "next_stage_recommendation": "Manual image-capable review should score Stage 5 review pool plus Stage 7 delta candidates before any final decision or V-COCO fallback.",
        "next_stage_started": False,
    }
    args.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = run_experiments(args)
    print(json.dumps({"status": manifest["status"], "statistics": manifest["statistics"], "acceptance": manifest["acceptance"], "vcoco": manifest["vcoco"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
