#!/usr/bin/env python3
"""Assign scored EvidenceHOI candidates to overlapping A/B/C review buckets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any


INDOOR_OBJECTS = {
    "chair",
    "couch",
    "bed",
    "dining table",
    "laptop",
    "keyboard",
    "mouse",
    "book",
    "cup",
    "bottle",
    "cell phone",
    "tv",
    "remote",
    "sink",
    "refrigerator",
    "oven",
    "knife",
    "fork",
    "spoon",
    "bowl",
}

STREET_TRAVEL_OBJECTS = {
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "train",
    "truck",
    "boat",
    "backpack",
    "suitcase",
    "umbrella",
    "traffic light",
    "bench",
    "skateboard",
}

AMBIGUOUS_ACTION_PATTERNS = [
    ("bicycle", {"ride", "push", "hold", "walk"}, "ride/push/hold bicycle"),
    ("cup", {"hold", "drink", "drink_with", "sip"}, "hold/drink cup"),
    ("bottle", {"hold", "drink", "drink_with", "sip", "pour"}, "hold/drink bottle"),
    ("knife", {"hold", "cut", "cut_with", "wield"}, "hold/cut knife"),
    ("chair", {"sit_on", "sit", "stand", "stand_on", "no_interaction"}, "sit on/stand near chair"),
    ("bench", {"sit_on", "sit", "stand", "no_interaction"}, "sit on/stand near bench"),
    ("backpack", {"hold", "carry", "wear"}, "hold/carry/wear bag"),
    ("handbag", {"hold", "carry"}, "hold/carry bag"),
    ("suitcase", {"hold", "carry", "pull"}, "hold/carry/pull bag"),
]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def triangular(value: float, low: float, high: float, ideal_low: float | None = None, ideal_high: float | None = None) -> float:
    if ideal_low is None:
        ideal_low = low
    if ideal_high is None:
        ideal_high = high
    if ideal_low <= value <= ideal_high:
        return 1.0
    if value < low or value > high:
        return 0.0
    if value < ideal_low:
        return (value - low) / max(ideal_low - low, 1e-9)
    return (high - value) / max(high - ideal_high, 1e-9)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_candidates(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL line {line_no} in {path}: {exc}") from exc
    return rows


def context_names(pair: dict[str, Any]) -> list[str]:
    names = pair.get("context_object_names")
    if isinstance(names, list):
        return [str(x) for x in names]
    return [str(item.get("category_name")) for item in pair.get("context_set", []) if item.get("category_name")]


def all_pair_object_names(row: dict[str, Any]) -> list[str]:
    return [str(pair.get("object_category_name")) for pair in row.get("top_pairs", []) if pair.get("object_category_name")]


def scene_preference(row: dict[str, Any]) -> tuple[float, str]:
    names = set(all_pair_object_names(row))
    for pair in row.get("top_3_pairs", []):
        names.update(context_names(pair))
    indoor = len(names & INDOOR_OBJECTS)
    street = len(names & STREET_TRAVEL_OBJECTS)
    if indoor >= 2:
        return 1.0, "indoor daily scene"
    if street >= 2:
        return 1.0, "street/travel scene"
    if indoor or street:
        return 0.6, "partial indoor/street cues"
    return 0.25, "scene type unclear"


def pair_spatial_separation(row: dict[str, Any]) -> float:
    pairs = row.get("top_3_pairs", [])[:3]
    if len(pairs) < 2:
        return 0.0
    distances: list[float] = []
    for i, a in enumerate(pairs):
        for b in pairs[i + 1 :]:
            ax, ay = float(a.get("center_x", 0.5)), float(a.get("center_y", 0.5))
            bx, by = float(b.get("center_x", 0.5)), float(b.get("center_y", 0.5))
            distances.append(math.dist((ax, ay), (bx, by)))
    avg = mean(distances)
    # Prefer separated but still same scene: roughly 0.18 to 0.55 normalized distance.
    return triangular(avg, 0.05, 0.8, 0.18, 0.55)


def circular_node_score(row: dict[str, Any]) -> float:
    pairs = row.get("top_3_pairs", [])[:3]
    primary = row.get("selected_primary_pair") or (pairs[0] if pairs else {})
    secondary_count = min(len(row.get("selected_secondary_pairs", [])), 2)
    node_count = 0
    if primary:
        node_count += 2
        node_count += min(int(primary.get("context_count", 0)), 3)
    node_count += 2 * secondary_count
    return triangular(node_count, 5, 11, 7, 9)


def ambiguous_action_tags(row: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    for pair in row.get("top_pairs", [])[:8]:
        obj = str(pair.get("object_category_name", ""))
        actions = {str(a) for a in pair.get("actions", [])}
        for expected_obj, action_set, tag in AMBIGUOUS_ACTION_PATTERNS:
            if obj == expected_obj and actions & action_set:
                tags.add(tag)
    # Cross-pair ambiguity: same object type, multiple action groups/verbs.
    per_obj: dict[str, set[str]] = {}
    for pair in row.get("top_pairs", [])[:8]:
        obj = str(pair.get("object_category_name", ""))
        per_obj.setdefault(obj, set()).update(str(a) for a in pair.get("actions", []))
    for obj, actions in per_obj.items():
        if len(actions) >= 2:
            tags.add(f"multi-action {obj}")
    return sorted(tags)


def possible_shuffle_pairs(row: dict[str, Any]) -> list[str]:
    pairs = row.get("top_pairs", [])[:8]
    results: list[str] = []
    for i, a in enumerate(pairs):
        for b in pairs[i + 1 :]:
            if a.get("object_category_name") == b.get("object_category_name") and set(a.get("actions", [])) != set(b.get("actions", [])):
                results.append(
                    f"h{a.get('subject_id')}-o{a.get('object_id')} <-> h{b.get('subject_id')}-o{b.get('object_id')} ({a.get('object_category_name')})"
                )
            elif set(a.get("action_groups", [])) & set(b.get("action_groups", [])) and a.get("object_category_name") != b.get("object_category_name"):
                results.append(
                    f"h{a.get('subject_id')}-o{a.get('object_id')} ({a.get('object_category_name')}) <-> h{b.get('subject_id')}-o{b.get('object_id')} ({b.get('object_category_name')})"
                )
            if len(results) >= 5:
                return results
    return results


def evidence_deletion_paths(row: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for pair in row.get("top_3_pairs", [])[:3]:
        for ctx in pair.get("context_set", [])[:3]:
            paths.append(f"P{pair.get('subject_id')} -> C{ctx.get('instance_id')}({ctx.get('category_name')}) -> O{pair.get('object_id')}")
            if len(paths) >= 8:
                return paths
    return paths


def score_a(row: dict[str, Any]) -> tuple[float, list[str]]:
    counts = row.get("counts", {})
    comps = row.get("component_scores", {})
    scene_score, scene_reason = scene_preference(row)
    pair_count = int(counts.get("num_valid_hoi_pairs", 0))
    context_j = float(comps.get("context_jaccard_mean", 0.0))
    primary_context = int(counts.get("primary_pair_context_count", 0))
    sports_penalty = 1.0 if row.get("reject_flags", {}).get("reject_severe_sports") else 0.0
    score = (
        1.6 * float(comps.get("PairDiversity", 0.0))
        + 1.4 * float(comps.get("ActionDiversity", 0.0))
        + 1.4 * float(comps.get("ObjectDiversity", 0.0))
        + 1.2 * float(comps.get("EvidenceContextScore", 0.0))
        + 0.9 * triangular(pair_count, 2, 9, 3, 7)
        + 0.8 * (1.0 if int(counts.get("num_unique_action_groups", 0)) >= 3 else 0.0)
        + 0.8 * min(int(counts.get("num_unique_object_categories", 0)) / 4.0, 1.0)
        + 0.7 * (1.0 if 0.2 <= context_j <= 0.7 else 0.4)
        + 0.7 * (1.0 if primary_context >= 2 else 0.0)
        + 0.7 * pair_spatial_separation(row)
        + 0.5 * circular_node_score(row)
        + 0.5 * scene_score
        - 1.3 * sports_penalty
        - 0.6 * float(comps.get("HomogeneityPenalty", 0.0))
    )
    reasons = [
        f"pair_count={pair_count}",
        f"action_groups={counts.get('num_unique_action_groups')}",
        f"object_categories={counts.get('num_unique_object_categories')}",
        f"context_jaccard={context_j:.3f}",
        scene_reason,
    ]
    return round(score, 6), reasons


def score_b(row: dict[str, Any]) -> tuple[float, list[str]]:
    counts = row.get("counts", {})
    comps = row.get("component_scores", {})
    pair_count = int(counts.get("num_valid_hoi_pairs", 0))
    primary_context = int(counts.get("primary_pair_context_count", 0))
    score = (
        1.8 * float(comps.get("VisualClarity", 0.0))
        + 1.5 * float(comps.get("LayoutScore", 0.0))
        + 1.4 * float(comps.get("EvidenceContextScore", 0.0))
        + 0.9 * triangular(pair_count, 1, 8, 3, 5)
        + 0.8 * (1.0 if primary_context >= 2 else 0.0)
        + 0.7 * min(int(counts.get("visible_pair_candidates", 0)) / 3.0, 1.0)
        + 0.5 * min(int(counts.get("num_gt_boxes", 0)) / 8.0, 1.0)
        - 0.8 * float(comps.get("CrowdingPenalty", 0.0))
        - 0.5 * float(comps.get("HomogeneityPenalty", 0.0))
    )
    reasons = [
        f"visual_clarity={comps.get('VisualClarity')}",
        f"layout={comps.get('LayoutScore')}",
        f"primary_context={primary_context}",
        f"pair_count={pair_count}",
    ]
    return round(score, 6), reasons


def score_c(row: dict[str, Any], tags: list[str], shuffle_pairs: list[str], deletion_paths: list[str]) -> tuple[float, list[str]]:
    counts = row.get("counts", {})
    comps = row.get("component_scores", {})
    whether_test = str(row.get("split", "")).lower().startswith("test")
    warning_bonus = min(len(row.get("warnings", [])) / 3.0, 1.0)
    score = (
        1.4 * min(len(tags) / 3.0, 1.0)
        + 1.0 * min(len(shuffle_pairs) / 3.0, 1.0)
        + 1.0 * min(len(deletion_paths) / 4.0, 1.0)
        + 0.8 * float(comps.get("EvidenceContextScore", 0.0))
        + 0.7 * min(int(counts.get("num_valid_hoi_pairs", 0)) / 6.0, 1.0)
        + 0.5 * warning_bonus
        + (0.4 if whether_test else 0.0)
        + 0.4 * float(comps.get("ActionDiversity", 0.0))
    )
    reasons = [
        f"ambiguous_tags={tags[:3]}",
        f"shuffle_pairs={len(shuffle_pairs)}",
        f"deletion_paths={len(deletion_paths)}",
        f"test_split={whether_test}",
    ]
    return round(score, 6), reasons


def enrich_candidate(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    tags = ambiguous_action_tags(row)
    shuffle = possible_shuffle_pairs(row)
    deletion = evidence_deletion_paths(row)
    a_score, a_reasons = score_a(row)
    b_score, b_reasons = score_b(row)
    c_score, c_reasons = score_c(row, tags, shuffle, deletion)
    row["bucket_scores"] = {"A_score": a_score, "B_score": b_score, "C_score": c_score}
    row["bucket_reasons"] = {"A": a_reasons, "B": b_reasons, "C": c_reasons}
    row["ambiguous_action_tags"] = tags
    row["possible_prototype_shuffle_pairs"] = shuffle
    row["evidence_deletion_candidate_paths"] = deletion
    row["whether_test_split"] = str(row.get("split", "")).lower().startswith("test")
    return row


def assign_buckets(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    buckets = {"A": [], "B": [], "C": []}
    for row in rows:
        enriched = enrich_candidate(row)
        scores = enriched["bucket_scores"]
        if scores["A_score"] >= thresholds["A"]:
            buckets["A"].append(enriched)
        if scores["B_score"] >= thresholds["B"]:
            buckets["B"].append(enriched)
        if scores["C_score"] >= thresholds["C"]:
            buckets["C"].append(enriched)
    buckets["A"].sort(key=lambda item: item["bucket_scores"]["A_score"], reverse=True)
    buckets["B"].sort(key=lambda item: item["bucket_scores"]["B_score"], reverse=True)
    buckets["C"].sort(key=lambda item: item["bucket_scores"]["C_score"], reverse=True)
    return buckets


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def pair_summary(pair: dict[str, Any] | None) -> str:
    if not pair:
        return ""
    return f"h{pair.get('subject_id')}-o{pair.get('object_id')} {pair.get('object_category_name')} {'/'.join(pair.get('actions', []))} ctx={pair.get('context_count')}"


def write_csv(path: Path, rows: list[dict[str, Any]], bucket: str) -> None:
    fields = [
        "image_id",
        "file_name",
        "image_path",
        "total_score",
        "A_score",
        "B_score",
        "C_score",
        "bucket_suggestion",
        "num_persons",
        "num_valid_pairs",
        "num_unique_actions",
        "num_unique_action_groups",
        "num_unique_objects",
        "primary_pair",
        "secondary_pairs",
        "top_evidence_paths",
        "ambiguous_action_tags",
        "possible_prototype_shuffle_pairs",
        "whether_test_split",
        "why_selected",
        "possible_figure_use",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            counts = row.get("counts", {})
            scores = row.get("bucket_scores", {})
            secondary = " | ".join(pair_summary(pair) for pair in row.get("selected_secondary_pairs", []))
            writer.writerow(
                {
                    "image_id": row.get("image_id"),
                    "file_name": row.get("file_name"),
                    "image_path": row.get("image_path"),
                    "total_score": row.get("total_score", row.get("score")),
                    "A_score": scores.get("A_score"),
                    "B_score": scores.get("B_score"),
                    "C_score": scores.get("C_score"),
                    "bucket_suggestion": row.get("bucket_suggestion"),
                    "num_persons": counts.get("num_persons"),
                    "num_valid_pairs": counts.get("num_valid_hoi_pairs"),
                    "num_unique_actions": counts.get("num_unique_action_categories"),
                    "num_unique_action_groups": counts.get("num_unique_action_groups"),
                    "num_unique_objects": counts.get("num_unique_object_categories"),
                    "primary_pair": pair_summary(row.get("selected_primary_pair")),
                    "secondary_pairs": secondary,
                    "top_evidence_paths": " | ".join(row.get("evidence_deletion_candidate_paths", [])[:5]),
                    "ambiguous_action_tags": ";".join(row.get("ambiguous_action_tags", [])),
                    "possible_prototype_shuffle_pairs": " | ".join(row.get("possible_prototype_shuffle_pairs", [])[:3]),
                    "whether_test_split": row.get("whether_test_split"),
                    "why_selected": "; ".join(row.get("bucket_reasons", {}).get(bucket, [])),
                    "possible_figure_use": {"A": "Figure 1 motivation / conceptual comparison", "B": "Figure 2 model pipeline / framework", "C": "Qualitative / diagnostic visualization"}[bucket],
                }
            )


def write_shortlist(path: Path, buckets: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# EvidenceHOI Final Shortlist", ""]
    uses = {"A": "Figure 1 motivation / conceptual comparison", "B": "Figure 2 model pipeline / framework", "C": "Qualitative / diagnostic visualization"}
    for bucket, title in [("A", "A: Figure 1 Motivation"), ("B", "B: Figure 2 Pipeline"), ("C", "C: Diagnostic")]:
        lines.append(f"## {title}")
        lines.append("")
        rows = buckets[bucket][:10]
        if not rows:
            lines.append("No candidates assigned to this bucket.")
            lines.append("")
            continue
        for idx, row in enumerate(rows, start=1):
            scores = row.get("bucket_scores", {})
            primary = row.get("selected_primary_pair")
            secondary = [pair_summary(pair) for pair in row.get("selected_secondary_pairs", [])[:2]]
            paths = row.get("evidence_deletion_candidate_paths", [])[:3]
            why = row.get("bucket_reasons", {}).get(bucket, [])
            lines.append(f"### {idx}. image_id={row.get('image_id')} file={row.get('file_name')}")
            lines.append(f"- bucket_score: `{scores.get(bucket + '_score')}` total_score: `{row.get('total_score', row.get('score'))}`")
            lines.append(f"- why selected: {'; '.join(why)}")
            lines.append(f"- primary pair: {pair_summary(primary)}")
            lines.append(f"- secondary pairs: {secondary if secondary else 'none'}")
            lines.append(f"- top evidence paths: {paths if paths else 'none'}")
            lines.append(f"- possible figure use: {uses[bucket]}")
            if bucket == "C":
                lines.append(f"- ambiguous tags: {row.get('ambiguous_action_tags', [])}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_contact_sheet(script_path: Path, bucket_jsonl: Path, image_root: Path | None, output_dir: Path, top_k: int) -> None:
    command = [
        sys.executable,
        str(script_path),
        "--candidates",
        str(bucket_jsonl),
        "--top_k",
        str(top_k),
        "--bucket",
        "all",
        "--output_dir",
        str(output_dir),
    ]
    if image_root is not None:
        command.extend(["--image_root", str(image_root)])
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket scored EvidenceHOI candidates into A/B/C sets.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--contact_sheet_top_k", type=int, default=50)
    parser.add_argument("--a_threshold", type=float, default=4.8)
    parser.add_argument("--b_threshold", type=float, default=4.0)
    parser.add_argument("--c_threshold", type=float, default=2.4)
    parser.add_argument("--skip_contact_sheets", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_candidates(args.candidates)
    thresholds = {"A": args.a_threshold, "B": args.b_threshold, "C": args.c_threshold}
    buckets = assign_buckets(rows, thresholds)
    specs = {
        "A": ("bucket_A_figure1", "A"),
        "B": ("bucket_B_figure2", "B"),
        "C": ("bucket_C_diagnostic", "C"),
    }
    for bucket, (stem, bucket_name) in specs.items():
        jsonl_path = args.output_dir / f"{stem}.jsonl"
        csv_path = args.output_dir / f"{stem}.csv"
        write_jsonl(jsonl_path, buckets[bucket])
        write_csv(csv_path, buckets[bucket], bucket)
    shortlist_path = args.output_dir / "final_shortlist.md"
    write_shortlist(shortlist_path, buckets)

    if not args.skip_contact_sheets:
        sheet_script = Path(__file__).resolve().parent / "make_contact_sheet.py"
        for bucket, (stem, _) in specs.items():
            run_contact_sheet(
                sheet_script,
                args.output_dir / f"{stem}.jsonl",
                args.image_root,
                args.output_dir / f"{stem}_contact_sheet_top{args.contact_sheet_top_k}",
                args.contact_sheet_top_k,
            )

    summary = {
        "input_candidates": len(rows),
        "bucket_counts": {bucket: len(items) for bucket, items in buckets.items()},
        "thresholds": thresholds,
    }
    (args.output_dir / "bucket_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"final_shortlist: {shortlist_path}")
    for bucket, (stem, _) in specs.items():
        print(f"bucket_{bucket}: {args.output_dir / (stem + '.jsonl')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
