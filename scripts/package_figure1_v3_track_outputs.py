#!/usr/bin/env python3
"""Package v3 contrastive search outputs into per-track review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
BASE = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive"
DEFAULT_OUTPUT_DIR = BASE / "final_review_package"
TRACKS_DIR = BASE / "tracks"
COUNTERFACTUAL = BASE / "counterfactual_crops/counterfactual_manifest.csv"
MANUAL_SHORTLIST = BASE / "manual_scoring/figure1_shortlist_manual_scoring.csv"
MANUAL_EXCLUDED = BASE / "manual_scoring/manual_scoring_excluded.csv"
PRIORITY = BASE / "priority_candidates.csv"
HARD_AUDIT = BASE / "hard_negative_audit.csv"
ALL_SCORED = BASE / "candidates_all_scored.csv"


CONTRASTIVE_COLUMNS = [
    "image_id",
    "candidate_id",
    "pair_id",
    "image_key",
    "file_name",
    "target_human_id",
    "target_object_id",
    "target_object_class",
    "target_gt_verbs",
    "confusion_verbs",
    "contrast_human_id",
    "contrast_object_id",
    "contrast_gt_verbs",
    "target_pair_type",
    "contrast_pair_type",
    "human_signature_distance",
    "dominant_pair_type_ratio",
    "duplicate_instance_ratio",
    "target_is_activity_outlier",
    "decisive_context",
    "context_source",
    "context_gain",
    "misleading_neighbor",
    "scene_mismatch_description",
    "candidate_swap_risk",
    "search_track",
    "heterogeneity_score",
    "ambiguity_score",
    "context_score",
    "interference_score",
    "readability_score",
    "final_score",
    "hard_negative_reason",
    "eligible_for_final",
    "one_sentence_rationale",
    "counterfactual_pair_only_crop",
    "counterfactual_pair_plus_context_crop",
    "counterfactual_context_masked_crop",
    "top10_rank",
    "top10_bucket",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=360)
    parser.add_argument("--thumb-height", type=int, default=300)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def clean_generated_jpgs(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.glob("*.jpg"):
        item.unlink()


def load_json_list(value: Any) -> list[Any]:
    try:
        decoded = json.loads(str(value))
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def row_float(row: dict[str, Any], field: str) -> float:
    try:
        return float(row.get(field, 0) or 0)
    except Exception:
        return 0.0


def text_join(items: list[Any], limit: int = 4) -> str:
    values = [str(item) for item in items if str(item)]
    if len(values) > limit:
        return "; ".join(values[:limit]) + f"; +{len(values) - limit} more"
    return "; ".join(values)


def first_neighbor(row: dict[str, str]) -> tuple[str, str, str, str]:
    neighbors = load_json_list(row.get("misleading_neighbor", "[]"))
    for item in neighbors:
        if isinstance(item, dict):
            families = item.get("verb_families") if isinstance(item.get("verb_families"), list) else []
            return str(item.get("human_id", "")), str(item.get("object_id", "")), str(item.get("pair_id", "")), text_join(families, 3)
    return "", "", "", ""


def context_from_manual(row: dict[str, str]) -> str:
    q3 = row.get("q3_decisive_context_object_part_relation", "")
    if q3:
        return q3.split(";", 1)[0]
    return ""


def clean_manual_text(value: str) -> str:
    value = str(value or "").strip()
    for prefix in [
        "Pair-only may be confused with:",
        "Second pair/activity:",
        "Scene-level may mix in dominant/contrast activity evidence:",
        "Neighborhood-based method may retain nearby conflicting evidence:",
    ]:
        value = value.replace(prefix, "").strip()
    return value.strip().strip(".").strip()


def parse_pair_only_confusions(row: dict[str, str]) -> list[str]:
    raw = clean_manual_text(row.get("q2_pair_only_confusing_action", ""))
    if not raw:
        return []
    out: list[str] = []
    for item in raw.replace(",", ";").split(";"):
        item = item.strip()
        if not item or item.startswith("+") or item == "no_interaction":
            continue
        out.append(item)
    return out


def second_activity_from_manual(row: dict[str, str]) -> str:
    q5 = clean_manual_text(row.get("q5_second_person_different_activity", ""))
    if "/" in q5:
        return q5.rsplit("/", 1)[-1].strip() if q5.count("/") == 1 else q5.split(" / ")[-1].strip()
    q6 = row.get("q6_why_second_pair_evidence_may_mislead_target", "")
    if "(" in q6 and ")" in q6:
        return q6.split("(", 1)[1].split(")", 1)[0].strip()
    q7 = row.get("q7_scene_level_wrongly_mixes", "")
    if "[" in q7 and "]" in q7:
        return q7.split("[", 1)[1].split("]", 1)[0].replace('"', "").strip()
    return ""


def sentence_gate_reason(row: dict[str, str]) -> str:
    if not row:
        return "missing_manual_shortlist_row"
    if len(parse_pair_only_confusions(row)) < 2:
        return "unfillable_required_sentence_less_than_two_pair_only_confusions"
    if not context_from_manual(row):
        return "unfillable_required_sentence_missing_decisive_context"
    if not second_activity_from_manual(row):
        return "unfillable_required_sentence_missing_interference_source"
    return ""


def sentence_from_manual(row: dict[str, str]) -> str:
    context = context_from_manual(row)
    confusions = parse_pair_only_confusions(row)
    interference = second_activity_from_manual(row)
    if len(confusions) < 2 or not interference or not context:
        return ""
    return f"对于 target pair，单看人和目标物容易混淆为 {confusions[0]} / {confusions[1]}；场景或另一人物的 {interference} 活动证据会产生干扰；真正决定性的证据是 {context}。"


def has_triptych(cf: dict[str, str]) -> bool:
    return all(cf.get(field, "") for field in ["pair_only_crop", "pair_plus_single_context_crop", "context_masked_crop"])


def image_key_for(priority: dict[str, str], manual: dict[str, str]) -> str:
    return priority.get("image_key", "") or manual.get("image_key", "") or priority.get("file_name", "")


def cf_by_pair() -> dict[str, dict[str, str]]:
    return {row.get("pair_id", ""): row for row in read_csv(COUNTERFACTUAL)}


def priority_by_pair() -> dict[str, dict[str, str]]:
    out = {row.get("pair_id", ""): row for row in read_csv(PRIORITY)}
    for tid in ["D", "A", "B", "C"]:
        for row in track_rows(tid):
            pid = row.get("pair_id", "")
            if pid and pid not in out:
                row = dict(row)
                row.setdefault("priority_group", f"track_{tid}_directed_pool")
                out[pid] = row
    return out


def track_rows(track_id: str) -> list[dict[str, str]]:
    return read_csv(TRACKS_DIR / f"track_{track_id}.csv")


def to_contrastive_row(
    row: dict[str, str],
    manual: Optional[dict[str, str]],
    cf: Optional[dict[str, str]],
    all_pairs_by_id: Optional[dict[str, dict[str, str]]] = None,
    top_rank: str = "",
    bucket: str = "",
) -> dict[str, Any]:
    target_verbs = load_json_list(row.get("target_verbs", "[]"))
    confusion = load_json_list(row.get("confusion_verb_set", "[]"))
    contrast_human, contrast_object, contrast_pair_id, contrast_families = first_neighbor(row)
    contrast_verbs = contrast_families
    if all_pairs_by_id and contrast_pair_id in all_pairs_by_id:
        contrast_verbs = text_join(load_json_list(all_pairs_by_id[contrast_pair_id].get("target_verbs", "[]")), 6)
    manual = manual or {}
    cf = cf or {}
    heterogeneity = min(5.0, 5.0 * row_float(row, "max_human_signature_distance"))
    ambiguity = min(4.0, 4.0 * row_float(row, "pair_crop_ambiguity"))
    context_score = min(5.0, 5.0 * row_float(row, "context_gain"))
    interference = min(3.0, 3.0 * max(row_float(row, "scene_mismatch_proxy"), row_float(row, "candidate_swap_risk")))
    readability = min(3.0, 1.5 * min(1.0, row_float(row, "object_short_edge_px") / 80.0) + 1.5 * min(1.0, row_float(row, "target_human_short_edge_px") / 100.0))
    final_score = heterogeneity + ambiguity + context_score + interference + readability
    sentence = sentence_from_manual(manual)
    return {
        "image_id": row.get("image_id", ""),
        "candidate_id": row.get("candidate_id", ""),
        "pair_id": row.get("pair_id", ""),
        "image_key": row.get("image_key", ""),
        "file_name": row.get("file_name", ""),
        "target_human_id": row.get("target_human_id", ""),
        "target_object_id": row.get("target_object_id", ""),
        "target_object_class": row.get("target_object_category", ""),
        "target_gt_verbs": json.dumps(target_verbs, ensure_ascii=False),
        "confusion_verbs": json.dumps([v for v in confusion if v != "no_interaction"][:8], ensure_ascii=False),
        "contrast_human_id": row.get("contrast_human_id", "") or contrast_human,
        "contrast_object_id": contrast_object,
        "contrast_gt_verbs": contrast_verbs,
        "target_pair_type": row.get("target_pair_type", ""),
        "contrast_pair_type": row.get("contrast_pair_type", ""),
        "human_signature_distance": row.get("max_human_signature_distance", ""),
        "dominant_pair_type_ratio": row.get("dominant_pair_type_ratio", ""),
        "duplicate_instance_ratio": row.get("duplicate_instance_ratio", ""),
        "target_is_activity_outlier": row.get("target_is_activity_outlier", ""),
        "decisive_context": context_from_manual(manual),
        "context_source": row.get("context_source", ""),
        "context_gain": row.get("context_gain", ""),
        "misleading_neighbor": row.get("misleading_neighbor", ""),
        "scene_mismatch_description": manual.get("q7_scene_level_wrongly_mixes", ""),
        "candidate_swap_risk": row.get("candidate_swap_risk", ""),
        "search_track": row.get("track_id", "") or row.get("search_track", ""),
        "heterogeneity_score": f"{heterogeneity:.3f}",
        "ambiguity_score": f"{ambiguity:.3f}",
        "context_score": f"{context_score:.3f}",
        "interference_score": f"{interference:.3f}",
        "readability_score": f"{readability:.3f}",
        "final_score": f"{final_score:.3f}",
        "hard_negative_reason": row.get("hard_negative_reason", ""),
        "eligible_for_final": row.get("eligible_for_final", ""),
        "one_sentence_rationale": sentence,
        "counterfactual_pair_only_crop": cf.get("pair_only_crop", ""),
        "counterfactual_pair_plus_context_crop": cf.get("pair_plus_single_context_crop", ""),
        "counterfactual_context_masked_crop": cf.get("context_masked_crop", ""),
        "top10_rank": top_rank,
        "top10_bucket": bucket,
    }


def draw_contact_sheet(rows: list[dict[str, Any]], path: Path, cols: int, thumb_w: int, thumb_h: int, title: str) -> None:
    if not rows:
        img = Image.new("RGB", (thumb_w * cols, 160), "white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 20), f"{title}: no candidates", fill=(0, 0, 0))
        img.save(path, quality=92)
        return
    rows_count = math.ceil(len(rows) / cols)
    header_h = 42
    img = Image.new("RGB", (thumb_w * cols, rows_count * thumb_h + header_h), "white")
    draw = ImageDraw.Draw(img)
    draw.text((12, 12), title, fill=(0, 0, 0))
    for idx, row in enumerate(rows):
        crop_path = row.get("counterfactual_pair_plus_context_crop") or row.get("counterfactual_pair_only_crop")
        src = PROJECT_ROOT / crop_path if crop_path else None
        col = idx % cols
        r = idx // cols
        x = col * thumb_w
        y = header_h + r * thumb_h
        tile = Image.new("RGB", (thumb_w, thumb_h), "white")
        if src and src.exists():
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb_w, thumb_h - 70), Image.Resampling.LANCZOS)
                tile.paste(im, ((thumb_w - im.width) // 2, 0))
        tdraw = ImageDraw.Draw(tile)
        label = f"#{idx+1} {row.get('pair_id','')}"
        tdraw.text((6, thumb_h - 66), label[:58], fill=(0, 0, 0))
        tdraw.text((6, thumb_h - 48), f"GT {row.get('target_gt_verbs','')} conf {row.get('confusion_verbs','')}"[:64], fill=(0, 0, 0))
        tdraw.text((6, thumb_h - 30), f"ctx {row.get('decisive_context','')}"[:64], fill=(0, 0, 0))
        tdraw.text((6, thumb_h - 14), f"score {row.get('final_score','')} track {row.get('search_track','')}"[:64], fill=(0, 0, 0))
        img.paste(tile, (x, y))
    img.save(path, quality=92)


def draw_triptych(row: dict[str, Any], path: Path) -> bool:
    paths = [row.get("counterfactual_pair_only_crop"), row.get("counterfactual_pair_plus_context_crop"), row.get("counterfactual_context_masked_crop")]
    if not all(paths):
        return False
    ims = []
    for p in paths:
        src = PROJECT_ROOT / str(p)
        if not src.exists():
            return False
        with Image.open(src) as im:
            tile = im.convert("RGB")
            tile.thumbnail((360, 360), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (360, 400), "white")
            canvas.paste(tile, ((360 - tile.width) // 2, 0))
            ims.append(canvas)
    labels = ["pair-only", "context-added", "context-masked"]
    out = Image.new("RGB", (1080, 430), "white")
    draw = ImageDraw.Draw(out)
    draw.text((10, 8), row.get("pair_id", ""), fill=(0, 0, 0))
    for i, im in enumerate(ims):
        x = i * 360
        out.paste(im, (x, 30))
        draw.text((x + 8, 405), labels[i], fill=(0, 0, 0))
    out.save(path, quality=92)
    return True


def select_top10(
    shortlist_rows: list[dict[str, str]],
    priority_rows: dict[str, dict[str, str]],
    cf_rows: dict[str, dict[str, str]],
    all_pairs_by_id: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    candidate_rows = []
    selection_audit: list[dict[str, str]] = []
    for manual in shortlist_rows:
        pair_id = manual.get("pair_id", "")
        priority = priority_rows.get(pair_id)
        if not priority:
            selection_audit.append({"pair_id": pair_id, "image_key": manual.get("image_key", ""), "reason": "missing_priority_row"})
            continue
        cf = cf_rows.get(pair_id, {})
        gate_reason = sentence_gate_reason(manual)
        if gate_reason:
            selection_audit.append({"pair_id": pair_id, "image_key": image_key_for(priority, manual), "reason": gate_reason})
            continue
        if not has_triptych(cf):
            selection_audit.append({"pair_id": pair_id, "image_key": image_key_for(priority, manual), "reason": "missing_pair_only_context_added_or_context_masked_crop"})
            continue
        candidate_rows.append((manual, priority, cf))

    selected: list[tuple[str, dict[str, str], dict[str, str], dict[str, str]]] = []
    used: set[str] = set()
    used_images: set[str] = set()

    def add_bucket(bucket: str, pred, quota: int) -> int:
        count = 0
        for manual, priority, cf in candidate_rows:
            pid = manual.get("pair_id", "")
            if pid in used or not pred(manual, priority):
                continue
            img_key = image_key_for(priority, manual)
            if img_key in used_images:
                selection_audit.append({"pair_id": pid, "image_key": img_key, "reason": f"not_selected_for_top10_duplicate_image_key_in_strict_diversity_gate:{bucket}"})
                continue
            selected.append((bucket, manual, priority, cf))
            used.add(pid)
            if img_key:
                used_images.add(img_key)
            count += 1
            if count >= quota:
                break
        return count

    got_d_ac = add_bucket("Track D or A+C", lambda m, p: "D" in p.get("track_id", "") or ("A" in p.get("track_id", "") and "C" in p.get("track_id", "")), 3)
    got_a = add_bucket("strong Track A", lambda m, p: "A" in p.get("track_id", "") and "C" not in p.get("track_id", "") and "D" not in p.get("track_id", ""), 2)
    got_b = add_bucket("strong Track B", lambda m, p: "B" in p.get("track_id", "") and "C" not in p.get("track_id", "") and "D" not in p.get("track_id", ""), 2)
    got_c = add_bucket("strong Track C", lambda m, p: "C" in p.get("track_id", "") and "A" not in p.get("track_id", "") and "B" not in p.get("track_id", "") and "D" not in p.get("track_id", ""), 3)

    # Do not backfill with homogeneous alternatives; only add remaining unselected
    # if they still belong to an underfilled requested bucket.
    rows: list[dict[str, Any]] = []
    for rank, (bucket, manual, priority, cf) in enumerate(selected, start=1):
        rows.append(to_contrastive_row(priority, manual, cf, all_pairs_by_id, str(rank), bucket))
    gaps = {
        "Track D or A+C": {"target": 3, "selected": got_d_ac, "gap": max(0, 3 - got_d_ac)},
        "strong Track A": {"target": 2, "selected": got_a, "gap": max(0, 2 - got_a)},
        "strong Track B": {"target": 2, "selected": got_b, "gap": max(0, 2 - got_b)},
        "strong Track C": {"target": 3, "selected": got_c, "gap": max(0, 3 - got_c)},
    }
    selected_pids = {row.get("pair_id", "") for row in rows}
    for manual, priority, _cf in candidate_rows:
        pid = manual.get("pair_id", "")
        if pid not in selected_pids and pid not in used:
            selection_audit.append({"pair_id": pid, "image_key": image_key_for(priority, manual), "reason": "eligible_but_not_selected_due_to_track_quota_or_lower_priority"})
    return rows, gaps, selection_audit


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    clean_generated_jpgs(args.output_dir / "contact_sheets")
    clean_generated_jpgs(args.output_dir / "counterfactual_triptychs")

    cf_rows = cf_by_pair()
    priority_rows = priority_by_pair()
    all_pairs_by_id = {row.get("pair_id", ""): row for row in read_csv(ALL_SCORED)}
    all_pairs_by_id.update(priority_rows)
    manual_shortlist = read_csv(MANUAL_SHORTLIST)
    manual_by_pair = {row.get("pair_id", ""): row for row in manual_shortlist}

    packaged: list[dict[str, Any]] = []
    for row in read_csv(PRIORITY):
        pid = row.get("pair_id", "")
        manual = manual_by_pair.get(pid)
        cf = cf_rows.get(pid)
        packaged.append(to_contrastive_row(row, manual, cf, all_pairs_by_id))
    write_csv(args.output_dir / "contrastive_candidates.csv", packaged, CONTRASTIVE_COLUMNS)

    sheet_sources: dict[str, list[dict[str, Any]]] = {}
    for tid in ["A", "B", "C", "D"]:
        rows = []
        for row in track_rows(tid)[:40]:
            pid = row.get("pair_id", "")
            rows.append(to_contrastive_row(row, manual_by_pair.get(pid), cf_rows.get(pid), all_pairs_by_id))
        sheet_sources[f"Track {tid}"] = rows

    ac_rows = [row for row in packaged if "A" in row.get("search_track", "") and "C" in row.get("search_track", "")][:40]
    bc_rows = [row for row in packaged if "B" in row.get("search_track", "") and "C" in row.get("search_track", "")][:40]
    duplicate_rows = []
    for row in read_csv(HARD_AUDIT):
        if "dominant_pair_type_ratio" in row and ("dominant_pair_type_ratio" in row.get("hard_negative_rule_hits", "") or "same_person_or_image_has_over_3_same_class" in row.get("hard_negative_rule_hits", "")):
            duplicate_rows.append(to_contrastive_row(row, None, cf_rows.get(row.get("pair_id", "")), all_pairs_by_id))
        if len(duplicate_rows) >= 40:
            break
    sheet_sources["A+C mixed"] = ac_rows
    sheet_sources["B+C mixed"] = bc_rows
    sheet_sources["duplicate-dominance negatives"] = duplicate_rows

    for name, rows in sheet_sources.items():
        safe = name.lower().replace("+", "plus").replace(" ", "_").replace("-", "_")
        draw_contact_sheet(rows[:24], args.output_dir / "contact_sheets" / f"{safe}_contact_sheet.jpg", args.sheet_cols, args.thumb_width, args.thumb_height, name)

    top10, gaps, selection_audit = select_top10(manual_shortlist, priority_rows, cf_rows, all_pairs_by_id)
    write_csv(args.output_dir / "top10_candidates.csv", top10, CONTRASTIVE_COLUMNS)
    for row in top10:
        draw_triptych(row, args.output_dir / "counterfactual_triptychs" / f"top{int(row['top10_rank']):02d}_{row['pair_id'].replace(':','_')}.jpg")

    primary = top10[0] if top10 else None
    backups = top10[1:3]
    with (args.output_dir / "top10_explanations.md").open("w", encoding="utf-8") as handle:
        handle.write("# Figure 1 v3 Top-10 Review Notes\n\n")
        handle.write("No final Figure 1 is produced in this step; these are bottom-image suggestions pending human confirmation.\n\n")
        handle.write("Top-10 candidates are gated by complete pair-only/context-added/context-masked crops, the required natural sentence, and strict image-level diversity. Homogeneous same-image candidates are reported as exclusions instead of being used to fill quotas.\n\n")
        handle.write("## Top-10\n\n")
        for row in top10:
            handle.write(f"## Rank {row['top10_rank']} | {row['pair_id']} | {row['top10_bucket']}\n\n")
            handle.write(f"- Target GT: `{row['target_gt_verbs']}`; target object: `{row['target_object_class']}`.\n")
            handle.write(f"- Required sentence: {row['one_sentence_rationale']}\n")
            handle.write(f"- Decisive context: {row['decisive_context']} ({row['context_source']}).\n")
            handle.write(f"- Scene/neighborhood interference: {row['scene_mismatch_description']} / {row['misleading_neighbor'][:400]}.\n")
            handle.write(f"- Triptych: `counterfactual_triptychs/top{int(row['top10_rank']):02d}_{row['pair_id'].replace(':','_')}.jpg`.\n\n")
        handle.write("## Primary And Backups\n\n")
        if primary:
            handle.write(f"- Primary: `{primary['pair_id']}`.\n")
        for idx, row in enumerate(backups, start=1):
            handle.write(f"- Backup {idx}: `{row['pair_id']}`.\n")
        handle.write("\n## Quota Gaps And Directed Re-search\n\n")
        for bucket, info in gaps.items():
            handle.write(f"- {bucket}: selected {info['selected']} / target {info['target']}; gap {info['gap']}.\n")
            if info["gap"]:
                handle.write(f"- Directed re-search for {bucket}: raise recall with concrete context evidence, complete triptychs, and a naturally fillable required sentence; do not fill with same-image or same-interaction homogeneous candidates.\n")

    excluded_rows = read_csv(MANUAL_EXCLUDED)
    with (args.output_dir / "excluded_candidates_report.md").open("w", encoding="utf-8") as handle:
        handle.write("# Excluded Candidates\n\n")
        handle.write("## Manual Scoring Exclusions\n\n")
        for row in excluded_rows[:120]:
            handle.write(f"- `{row.get('pair_id','')}`: {row.get('shortlist_gate_reason','')}\n")
        handle.write("\n## Top-10 Strict Gate Exclusions\n\n")
        reason_counts = Counter(item["reason"] for item in selection_audit)
        for reason, count in reason_counts.most_common():
            handle.write(f"- {reason}: {count}\n")
        handle.write("\n## Top-10 Strict Gate Examples\n\n")
        for item in selection_audit[:160]:
            handle.write(f"- `{item.get('pair_id','')}` ({item.get('image_key','')}): {item.get('reason','')}\n")

    with (args.output_dir / "new_vs_old_scoring_diff.md").open("w", encoding="utf-8") as handle:
        handle.write("# New vs Old Scoring Difference\n\n")
        handle.write("- Old v3 ranking used automatic contrastive proxies and track scores.\n")
        handle.write("- The manual-scoring sheet leaves the five 20-point rubric fields blank for human review; this final package keeps automatic proxy sub-scores only as sortable audit metadata in `contrastive_candidates.csv` and `top10_candidates.csv`.\n")
        handle.write("- Top-10 selection is now gated by answerability of the required confusion/context/interference sentence, complete pair-only/context-added/context-masked crops, and image-level diversity.\n")
        handle.write("- Duplicate instances, human count, object count, and raw HOI count are not direct bonuses.\n")
        handle.write(f"- Manual scoring shortlist contains {len(manual_shortlist)} candidates; Top-10 selection contains {len(top10)} candidates because quotas are not backfilled with homogeneous candidates.\n")
        handle.write("- Candidate-level old proxy scores remain in source CSVs for audit only, not as final manual scores.\n")

    summary = {
        "schema_version": "figure1_search_v3.final_review_package",
        "output_dir": str(args.output_dir.relative_to(PROJECT_ROOT)),
        "contrastive_candidates": str((args.output_dir / "contrastive_candidates.csv").relative_to(PROJECT_ROOT)),
        "top10_candidates": str((args.output_dir / "top10_candidates.csv").relative_to(PROJECT_ROOT)),
        "contact_sheets": sorted(str(p.relative_to(PROJECT_ROOT)) for p in (args.output_dir / "contact_sheets").glob("*.jpg")),
        "triptychs": sorted(str(p.relative_to(PROJECT_ROOT)) for p in (args.output_dir / "counterfactual_triptychs").glob("*.jpg")),
        "quota_gaps": gaps,
        "top10_strict_gate_exclusion_counts": dict(Counter(item["reason"] for item in selection_audit)),
        "primary": primary["pair_id"] if primary else "",
        "backups": [row["pair_id"] for row in backups],
        "final_figure1_created": False,
    }
    (args.output_dir / "package_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
