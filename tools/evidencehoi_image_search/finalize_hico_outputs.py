#!/usr/bin/env python3
"""Arrange EvidenceHOI HICO outputs and write the cross-dataset shortlist."""

from __future__ import annotations

import json
import shutil
import csv
from pathlib import Path
from typing import Any


RESULTS_ROOT = Path("tools/evidencehoi_image_search/results")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copytree_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


def arrange_split(split_dir: Path) -> None:
    copy_if_exists(split_dir / "candidates_hard_filtered.jsonl", split_dir / "candidates_scored.jsonl")
    copy_if_exists(split_dir / "candidates_hard_filtered.csv", split_dir / "candidates_scored.csv")
    buckets_dir = split_dir / "buckets"
    for stem in ["bucket_A_figure1", "bucket_B_figure2", "bucket_C_diagnostic"]:
        copy_if_exists(buckets_dir / f"{stem}.csv", split_dir / f"{stem}.csv")
        copy_if_exists(buckets_dir / f"{stem}.jsonl", split_dir / f"{stem}.jsonl")
        copytree_if_exists(buckets_dir / f"{stem}_contact_sheet_top50", split_dir / "contact_sheets" / f"{stem}_top50")
    copy_if_exists(buckets_dir / "bucket_summary.json", split_dir / "bucket_summary.json")
    copy_if_exists(buckets_dir / "final_shortlist.md", split_dir / "final_shortlist.md")


def pair_summary(row: dict[str, Any]) -> str:
    primary = row.get("selected_primary_pair") or {}
    if not primary:
        return "none"
    actions = "/".join(str(x) for x in primary.get("actions", []))
    return f"h{primary.get('subject_id')}-o{primary.get('object_id')} {primary.get('object_category_name')} {actions}"


def contact_dir(dataset_dir: str, bucket_stem: str) -> Path:
    return RESULTS_ROOT / dataset_dir / "contact_sheets" / f"{bucket_stem}_top50"


def contact_rank_map(dataset_dir: str, bucket_stem: str) -> dict[str, int]:
    review_table = contact_dir(dataset_dir, bucket_stem) / "review_table.csv"
    if not review_table.exists():
        return {}
    ranks: dict[str, int] = {}
    with review_table.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ranks[str(row.get("image_id"))] = int(str(row.get("rank")))
            except (TypeError, ValueError):
                continue
    return ranks


def visual_path(dataset_dir: str, bucket_stem: str, rank: int, image_id: str) -> str:
    return str(
        contact_dir(dataset_dir, bucket_stem)
        / "annotated_images"
        / f"top_{rank:04d}_{image_id}.jpg"
    )


def visual_path_for_image(dataset_dir: str, bucket_stem: str, image_id: str, fallback_rank: int) -> str:
    rank = contact_rank_map(dataset_dir, bucket_stem).get(image_id, fallback_rank)
    return visual_path(dataset_dir, bucket_stem, rank, image_id)


def bucket_score(row: dict[str, Any], bucket: str) -> Any:
    return row.get("bucket_scores", {}).get(f"{bucket}_score", "")


def format_section(title: str, rows: list[dict[str, Any]], dataset_dir: str, bucket_stem: str, bucket: str, limit: int = 20) -> list[str]:
    lines = [f"## {title}", ""]
    if not rows:
        lines.extend(["No candidates available.", ""])
        return lines
    for rank, row in enumerate(rows[:limit], start=1):
        counts = row.get("counts", {})
        image_id = str(row.get("image_id"))
        lines.append(f"### {rank}. {image_id} ({row.get('file_name')})")
        lines.append(f"- visual_path: `{visual_path_for_image(dataset_dir, bucket_stem, image_id, rank)}`")
        lines.append(f"- image_path: `{row.get('image_path')}`")
        lines.append(
            "- scores: "
            f"total={row.get('total_score', row.get('score'))}, "
            f"{bucket}={bucket_score(row, bucket)}, bucket_suggestion={row.get('bucket_suggestion')}"
        )
        lines.append(
            "- metrics: "
            f"unique_pairs={counts.get('num_valid_hoi_pairs')}, "
            f"unique_verbs={counts.get('num_unique_action_categories')}, "
            f"unique_objects={counts.get('num_unique_object_categories')}, "
            f"visible_pairs={counts.get('visible_pair_candidates')}, "
            f"context_objects={counts.get('primary_pair_context_count')}, "
            f"action_groups={counts.get('num_unique_action_groups')}"
        )
        lines.append(f"- primary_pair: {pair_summary(row)}")
        lines.append(f"- split: `{row.get('split')}` dataset: `{row.get('dataset')}`")
        if row.get("whether_test_split"):
            lines.append("- metadata: `split=test`")
        lines.append("")
    return lines


def recommendation_pool(
    train_a: list[dict[str, Any]], train_b: list[dict[str, Any]], test_c: list[dict[str, Any]]
) -> list[tuple[str, str, str, dict[str, Any], int]]:
    selected: list[tuple[str, str, str, dict[str, Any], int]] = []
    seen: set[str] = set()
    specs = [
        ("HICO trainval A", "hico_trainval", "bucket_A_figure1", train_a, 4),
        ("HICO trainval B", "hico_trainval", "bucket_B_figure2", train_b, 3),
        ("HICO test C", "hico_test", "bucket_C_diagnostic", test_c, 3),
    ]
    for label, dataset_dir, bucket_stem, rows, quota in specs:
        added = 0
        for idx, row in enumerate(rows, start=1):
            image_id = str(row.get("image_id"))
            if image_id in seen:
                continue
            selected.append((label, dataset_dir, bucket_stem, row, idx))
            seen.add(image_id)
            added += 1
            if added >= quota:
                break
    return selected[:10]


def write_vcoco_skip_notes() -> None:
    note = {
        "status": "skipped",
        "reason": "User instruction included '跳过vcoco数据集'; HICO-DET trainval/test were run only.",
        "expected_filters_if_enabled": [
            "exclude object_id=-1",
            "exclude role object missing",
            "exclude body-only actions: walk/run/smile/stand",
            "prioritize object-role actions such as hold/sit on/look at/talk on phone/eat/drink/carry/ride/work on computer",
        ],
    }
    for split in ["vcoco_train", "vcoco_test"]:
        out_dir = RESULTS_ROOT / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "skipped.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")


def write_final_shortlist() -> None:
    train_a = load_jsonl(RESULTS_ROOT / "hico_trainval" / "bucket_A_figure1.jsonl")
    train_b = load_jsonl(RESULTS_ROOT / "hico_trainval" / "bucket_B_figure2.jsonl")
    test_c = load_jsonl(RESULTS_ROOT / "hico_test" / "bucket_C_diagnostic.jsonl")

    lines = [
        "# EvidenceHOI Final Shortlist Across Datasets",
        "",
        "Generated from HICO-DET trainval/test outputs. V-COCO was skipped because the active instruction says `跳过vcoco数据集`.",
        "",
        "## Output Summary",
        "",
        f"- HICO trainval A candidates: `{len(train_a)}`",
        f"- HICO trainval B candidates: `{len(train_b)}`",
        f"- HICO test C candidates: `{len(test_c)}`",
        "- V-COCO qualitative candidates: `0` (skipped)",
        "",
    ]
    lines.extend(format_section("HICO trainval A Top 20", train_a, "hico_trainval", "bucket_A_figure1", "A"))
    lines.extend(format_section("HICO trainval B Top 20", train_b, "hico_trainval", "bucket_B_figure2", "B"))
    lines.extend(format_section("HICO test C Top 20", test_c, "hico_test", "bucket_C_diagnostic", "C"))
    lines.extend(["## V-COCO Qualitative Top 20", "", "Skipped by explicit instruction: `跳过vcoco数据集`.", ""])

    lines.extend(["## Recommended Final Manual Candidates", ""])
    recommended = recommendation_pool(train_a, train_b, test_c)
    for rec_rank, (label, dataset_dir, bucket_stem, row, source_rank) in enumerate(recommended, start=1):
        counts = row.get("counts", {})
        image_id = str(row.get("image_id"))
        lines.append(f"### {rec_rank}. {image_id} ({label} rank {source_rank})")
        lines.append(f"- visual_path: `{visual_path_for_image(dataset_dir, bucket_stem, image_id, source_rank)}`")
        lines.append(f"- score: `{row.get('total_score', row.get('score'))}`")
        lines.append(f"- primary_pair: {pair_summary(row)}")
        lines.append(
            "- why: "
            f"pairs={counts.get('num_valid_hoi_pairs')}, verbs={counts.get('num_unique_action_categories')}, "
            f"objects={counts.get('num_unique_object_categories')}, contexts={counts.get('primary_pair_context_count')}, "
            f"action_groups={counts.get('num_unique_action_groups')}"
        )
        lines.append("")
    (RESULTS_ROOT / "final_shortlist_all_datasets.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    arrange_split(RESULTS_ROOT / "hico_trainval")
    arrange_split(RESULTS_ROOT / "hico_test")
    write_vcoco_skip_notes()
    write_final_shortlist()
    print(RESULTS_ROOT / "final_shortlist_all_datasets.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
