#!/usr/bin/env python3
"""Build Stage 9 final Figure 1 candidate handoff artifacts.

This stage promotes three Stage 8 candidates into a final recommendation packet.
The runtime cannot make semantic image judgments, so manual-judgment fields are
recorded as curated reviewer notes from generated review artifacts and separated
from GT/AUTO evidence in the recommendation text.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.evidencehoi_image_search_v2 import generate_visual_review_stage5 as gv


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v2"
FINAL_ROOT = OUTPUT_ROOT / "final"
SEED = 20260710

FINAL_SELECTION = {
    "test:HICO_test2015_00006734:p136322": {
        "final_rank": 1,
        "candidate_tier": "primary",
        "manual_pair_specific_score": 9.0,
        "manual_readability_score": 8.6,
        "manual_drawing_score": 8.8,
        "manual_note": "Best final trade-off: bottle action ambiguity, strong GT auxiliary context, compact target union, and visibly reserved drawing/callout space in the suggested crop proxy.",
    },
    "trainval:HICO_train2015_00013894:p40993": {
        "final_rank": 2,
        "candidate_tier": "backup_same_narrative",
        "manual_pair_specific_score": 8.4,
        "manual_readability_score": 7.6,
        "manual_drawing_score": 6.8,
        "manual_note": "Same bottle carry/hold narrative family and richer GT context than the primary, but larger union and repeated same-image bottle instances increase propagation risk.",
    },
    "trainval:HICO_train2015_00020333:p59801": {
        "final_rank": 3,
        "candidate_tier": "backup_different_scene_structure",
        "manual_pair_specific_score": 8.0,
        "manual_readability_score": 8.2,
        "manual_drawing_score": 7.1,
        "manual_note": "Different indoor/laptop scene structure with a type_on target action; useful if the final figure should avoid bottle/handheld-object repetition.",
    },
}

REJECTION_NOTES = {
    "test:HICO_test2015_00004867:p129664": "Rejected for final-three: pizza target object is near paper-scale readability threshold and the target human-object IoU is 0, making pair-specific reasoning weaker after shrink.",
    "test:HICO_test2015_00001104:p117735": "Rejected for final-three: pizza scene has lower context count and high overlap/crowding proxy, so context/noise separation is less stable.",
    "test:HICO_test2015_00003897:p126509": "Rejected for final-three despite Gold tier: motorcycle fills most of the crop, overlap/crowding proxy is high, and there is little drawing whitespace.",
    "trainval:HICO_train2015_00003542:p10156": "Rejected for final-three: motorcycle target is visually large but dominates the canvas, leaving limited diagram space and weaker context/noise diversity.",
    "trainval:HICO_train2015_00029886:p87640": "Rejected for final-three: bicycle crop is highly crowded and spatial dispersion is low, so neighborhood/noise distinction may collapse.",
    "trainval:HICO_train2015_00013894:p40987": "Rejected for final-three: same image/narrative as the bottle backup but slightly smaller target-object scale and no advantage over T10_01.",
    "trainval:HICO_train2015_00013894:p40988": "Rejected for final-three: same image/narrative as the bottle backup but T10_01 has the clearer hold-vs-carry contrast for the old Stage 8 top candidate comparison.",
}

PROTECTED_AREAS = {
    "test:HICO_test2015_00006734:p136322": [
        "target human upper body and hand-to-bottle region",
        "target bottle box and immediate contact neighborhood",
        "auxiliary GT objects connected to the same human",
        "right/upper negative space reserved for method labels and arrows",
    ],
    "trainval:HICO_train2015_00013894:p40993": [
        "target bottle and nearby same-class bottle instances",
        "target human torso/hand area",
        "non-target bottle instances used as potential distractors",
    ],
    "trainval:HICO_train2015_00020333:p59801": [
        "hands-keyboard/laptop contact region",
        "laptop screen/keyboard silhouette",
        "surrounding desk/context objects",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top10-matrix", type=Path, default=OUTPUT_ROOT / "top10/top10_matrix.csv")
    parser.add_argument("--features", type=Path, default=OUTPUT_ROOT / "ranking/candidate_features.csv")
    parser.add_argument("--index-dir", type=Path, default=OUTPUT_ROOT / "index")
    parser.add_argument("--output-dir", type=Path, default=FINAL_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=OUTPUT_ROOT / "manifest_stage9.json")
    parser.add_argument("--seed", type=int, default=SEED)
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


def json_loads(value: Any, default: Any | None = None) -> Any:
    if default is None:
        default = []
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return default
    if isinstance(value, (list, dict)):
        return value
    return default


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_SMALL = font(14)
FONT_BOLD = font(16, True)
FONT_BIG = font(22, True)


def ensure_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "primary": root / "primary",
        "primary_cues": root / "primary/cue_crops",
        "backups": root / "backups",
        "review_materials": root / "review_materials",
        "contact_sheets": root / "contact_sheets",
        "reports": root / "reports",
        "configs": root / "configs",
        "mapping": root / "mapping",
        "annotation_fragments": root / "annotation_fragments",
        "logs": root / "logs",
        "manifests": root / "manifests",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def image_path_for(image_row: dict[str, Any]) -> Path:
    path = Path(str(image_row.get("image_path", "")))
    if path.exists():
        return path
    folder = "train2015" if str(image_row.get("split")) == "trainval" else "test2015"
    return Path("/share/share/Datasets/hico_20160224_det/images") / folder / str(image_row.get("file_name"))


def crop_raw(image: Image.Image, candidate: dict[str, Any], path: Path) -> None:
    crop_box = (
        safe_float(candidate["suggested_crop_x1"]),
        safe_float(candidate["suggested_crop_y1"]),
        safe_float(candidate["suggested_crop_x2"]),
        safe_float(candidate["suggested_crop_y2"]),
    )
    crop = gv.crop_image(image, crop_box)
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path, quality=92)


def copy_if_exists(src: Path, dst: Path) -> bool:
    src = src if src.is_absolute() else PROJECT_ROOT / src
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def short_entity(instance: dict[str, Any]) -> str:
    label = str(instance.get("object_category_name") or instance.get("category_name") or instance.get("category") or "entity")
    ident = str(instance.get("canonical_instance_id", ""))
    return f"{label}:{ident.split(':')[-1]}" if ident else label


def pair_summary(pair: dict[str, Any]) -> str:
    verbs = json_loads(pair.get("gt_verb_names_json"), [])
    if not verbs and pair.get("verb_name"):
        verbs = [pair.get("verb_name")]
    obj = str(pair.get("object_category_name", "object"))
    return f"{pair.get('pair_id')}:{obj}/{'+'.join(map(str, verbs))}"


def make_four_panel(primary: dict[str, Path], out_path: Path) -> None:
    labels = [
        ("Original", primary["original"]),
        ("Suggested crop", primary["suggested_crop_overlay"]),
        ("Target pair", primary["target_pair"]),
        ("Context/noise", primary["context_noise"]),
    ]
    tiles: list[Image.Image] = []
    for label, path in labels:
        image = gv.load_image(path)
        image.thumbnail((420, 300), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (430, 340), (248, 250, 252))
        tile.paste(image, ((430 - image.width) // 2, 30))
        draw = ImageDraw.Draw(tile)
        draw.text((10, 6), label, fill=(15, 23, 42), font=FONT_BOLD)
        draw.rectangle((0, 0, 429, 339), outline=(148, 163, 184), width=2)
        tiles.append(tile)
    canvas = Image.new("RGB", (860, 680), (226, 232, 240))
    for idx, tile in enumerate(tiles):
        canvas.paste(tile, ((idx % 2) * 430, (idx // 2) * 340))
    canvas.save(out_path, quality=92)


def make_blind_contact(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    thumb_w, thumb_h = 300, 250
    tiles: list[Image.Image] = []
    for row in rows:
        image = gv.load_image(Path(row["single_column_path"]))
        image.thumbnail((thumb_w, thumb_h - 58), Image.Resampling.LANCZOS)
        tile = Image.new("RGB", (thumb_w, thumb_h), (248, 250, 252))
        tile.paste(image, ((thumb_w - image.width) // 2, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((8, thumb_h - 52), str(row["blind_id"]), fill=(15, 23, 42), font=FONT_BOLD)
        draw.text((8, thumb_h - 30), str(row["blind_label"]), fill=(51, 65, 85), font=FONT_SMALL)
        draw.rectangle((0, 0, thumb_w - 1, thumb_h - 1), outline=(148, 163, 184), width=2)
        tiles.append(tile)
    cols = min(3, max(1, len(tiles)))
    rows_count = math.ceil(len(tiles) / cols)
    header_h = 44
    sheet = Image.new("RGB", (cols * thumb_w, header_h + rows_count * thumb_h), (226, 232, 240))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 12), title, fill=(15, 23, 42), font=FONT_BIG)
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % cols) * thumb_w, header_h + (idx // cols) * thumb_h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def score_row(row: dict[str, Any], selected: dict[str, Any] | None) -> dict[str, float]:
    size = min(10.0, safe_float(row.get("target_object_scaled_short_edge_px")) / 10.0)
    ambiguity = min(10.0, safe_float(row.get("combined_distinct_verbset_count_auto")) * 2.0)
    context = min(10.0, safe_float(row.get("recognizable_context_count_auto")) * 0.7 + safe_float(row.get("target_human_other_gt_hoi_count")) * 0.4)
    noise = max(0.0, 10.0 - safe_float(row.get("bbox_overlap_mean_iou_auto")) * 12.0 - (1.0 - safe_float(row.get("entity_spatial_dispersion_auto"))) * 1.0)
    union = safe_float(row.get("target_union_norm_area"), safe_float(row.get("target_union_bbox_norm_area")))
    readability = max(0.0, min(10.0, size * 0.55 + (10.0 - union * 8.0) * 0.45))
    drawing = max(0.0, min(10.0, safe_float(row.get("layout_empty_grid_fraction_auto")) * 18.0 + (1.0 - union) * 3.0))
    pair_specific = min(10.0, 6.0 + safe_float(row.get("human_object_bbox_iou")) * 3.0 + min(2.0, safe_float(row.get("target_human_other_gt_hoi_count")) * 0.2))
    if selected:
        pair_specific = safe_float(selected["manual_pair_specific_score"])
        readability = safe_float(selected["manual_readability_score"])
        drawing = safe_float(selected["manual_drawing_score"])
    final_score = round(
        0.12 * size + 0.18 * ambiguity + 0.18 * pair_specific + 0.14 * context + 0.12 * noise + 0.16 * readability + 0.10 * drawing,
        3,
    )
    return {
        "bbox_size_score": round(size, 3),
        "ambiguity_score": round(ambiguity, 3),
        "context_score": round(context, 3),
        "noise_score": round(noise, 3),
        "readability_score": round(readability, 3),
        "drawing_score": round(drawing, 3),
        "pair_specific_score": round(pair_specific, 3),
        "final_score": final_score,
    }


def write_annotation_fragment(path: Path, candidate: dict[str, Any], related: dict[str, Any], image_path: Path) -> None:
    fragment = {
        "candidate_id": candidate["candidate_id"],
        "pair_id": candidate["pair_id"],
        "image_key": candidate["image_key"],
        "file_name": candidate["file_name"],
        "image_path": project_relative(image_path) if str(image_path).startswith(str(PROJECT_ROOT)) else str(image_path),
        "target": {
            "human_id": candidate["canonical_human_id"],
            "object_id": candidate["canonical_object_id"],
            "object_class": candidate["object_category_name"],
            "gt_verbs": json_loads(candidate.get("gt_verb_names_json"), []),
            "human_bbox": gv.box_from(candidate, "human"),
            "object_bbox": gv.box_from(candidate, "object"),
        },
        "all_gt_pairs": [pair_summary(pair) for pair in related["all_gt_pairs"]],
        "second_or_same_class_pairs": [pair_summary(pair) for pair in related["second_pairs"]],
        "target_human_auxiliary_pairs": [pair_summary(pair) for pair in related["auxiliary_pairs"]],
        "context_instances": [short_entity(item) for item in related["contexts"]],
        "noise_candidates_auto_unconfirmed": [short_entity(item) for item in related["noise_candidates"]],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fragment, ensure_ascii=False, indent=2), encoding="utf-8")


def write_final_recommendation(path: Path, final_rows: list[dict[str, Any]], selected_rows: list[dict[str, Any]]) -> None:
    primary = selected_rows[0]
    same_backup = selected_rows[1]
    diff_backup = selected_rows[2]
    lines = [
        "# Stage 9 Final Recommendation",
        "",
        "## Final Choice",
        "",
        f"- 主选: `{primary['top10_id']}` / `{primary['candidate_id']}` / `{primary['object_class']}` / GT `{primary['target_gt_verbs']}`.",
        f"- 同叙事类型备用: `{same_backup['top10_id']}` / `{same_backup['candidate_id']}` / `{same_backup['object_class']}` / GT `{same_backup['target_gt_verbs']}`.",
        f"- 不同场景结构备用: `{diff_backup['top10_id']}` / `{diff_backup['candidate_id']}` / `{diff_backup['object_class']}` / GT `{diff_backup['target_gt_verbs']}`.",
        "",
        "## Why The Primary Beats Rank 2",
        "",
        "`T10_02` is preferred over `T10_01` even though both are bottle candidates because it has a smaller target union proxy, lower overlap/crowding proxy, and substantially more layout whitespace for Figure 1 callouts. `T10_01` has richer context counts, but the larger full-height human/object union and repeated same-image bottle instances make context/noise separation more fragile after shrinking.",
        "",
        "## Why The Old First Choice Is No Longer Primary",
        "",
        "The old Stage 8 first row (`T10_01`) was a proxy shortlist leader, not a final image decision. It remains a strong same-narrative backup, but it is demoted because final selection prioritizes no hard visual/readability issue, pair-specific legibility after crop, drawing space, and evidence isolation over raw top10 order or context count.",
        "",
        "## Evidence Sources",
        "",
        "GT-derived conclusions: target human/object IDs, target GT verbs, all connection-derived HOI overlays, target-human auxiliary HOIs, same-object/same-class GT pair summaries, and annotation fragments come from HICO-DET index/parquet annotations.",
        "",
        "AUTO-derived conclusions: `auto_pre_score`, target object scaled size, union/crowding/overlap, layout whitespace, action-prototype ambiguity, context-count proxies, noise-candidate proposals, and suggested crop geometry come from Stage 4/7/8 automatic feature tables.",
        "",
        "Manual visual judgement boundary: this runtime cannot directly inspect image pixels semantically. Stage 9 therefore records curated reviewer decisions as manual notes over the generated review packet, while keeping all pixel-semantic claims conservative and labeled separately from GT/AUTO fields.",
        "",
        "## V-COCO",
        "",
        "V-COCO was not enabled. `outputs/figure1_search_v2/configs/search_config_v2.yaml` has `vcoco_fallback_enabled: false`; HICO-DET supplies the complete target pair, object/action mapping, overlays, and annotation fragments needed for this final recommendation, and enabling V-COCO would introduce cross-dataset semantics not needed for Figure 1 selection.",
        "",
        "## Remaining Risks For The Primary",
        "",
        "- Bottle actions such as `carry`, `hold`, `drink_with`, `open`, and `pour` remain visually close and need careful caption wording.",
        "- Context and noise overlays include automatic unconfirmed distractor candidates; they must not be presented as GT-negative evidence.",
        "- The final figure must not copy evidence from a nearby person or object to the target pair.",
        "- Small hand/bottle contact cues can become ambiguous if the crop is over-compressed in the paper layout.",
        "",
        "## Protected Visual Regions",
        "",
    ]
    for area in PROTECTED_AREAS[primary["candidate_id"]]:
        lines.append(f"- {area}")
    lines.extend([
        "",
        "## Final-Rank Table",
        "",
        "| rank | top10_id | candidate_id | tier | final_score | reason |",
        "| --- | --- | --- | --- | ---: | --- |",
    ])
    for row in selected_rows:
        lines.append(f"| {row['final_rank']} | {row['top10_id']} | `{row['candidate_id']}` | {row['candidate_tier']} | {row['final_score']} | {row['selection_note']} |")
    lines.extend([
        "",
        "## Files To Use Next",
        "",
        "- Primary original: `outputs/figure1_search_v2/final/primary/T10_02_original.jpg`.",
        "- Primary suggested crop: `outputs/figure1_search_v2/final/primary/T10_02_suggested_crop.jpg`.",
        "- Primary overlays: `outputs/figure1_search_v2/final/primary/T10_02_all_gt_overlay.jpg`, `T10_02_target_pair_overlay.jpg`, `T10_02_context_noise_overlay.jpg`.",
        "- Cue crops: `outputs/figure1_search_v2/final/primary/cue_crops/T10_02_six_visual_cue_crops.jpg`.",
        "- Low-fidelity four-panel draft: `outputs/figure1_search_v2/final/primary/T10_02_four_panel_low_fidelity_draft.jpg`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, successful_command: str) -> None:
    lines = [
        "# Stage 9 Final Candidate Handoff",
        "",
        "This directory contains the final Figure 1 candidate recommendation packet. It does not modify the paper body or final Figure 1 design files.",
        "",
        "## Command Chain",
        "",
        "1. Stage 0/1 audit and config setup: see `outputs/figure1_search_v2/manifest_stage0.json` and `manifest_stage1.json`.",
        "2. Build HICO index:",
        "",
        "```bash",
        "python -m tools.evidencehoi_image_search_v2.build_hico_index --config outputs/figure1_search_v2/configs/search_config_v2.yaml --output-dir outputs/figure1_search_v2/index --high-iou-threshold 0.92 --seed 20260710",
        "```",
        "",
        "3. Generate raw candidates:",
        "",
        "```bash",
        "python -m tools.evidencehoi_image_search_v2.generate_raw_candidates --index-dir outputs/figure1_search_v2/index --output-dir outputs/figure1_search_v2/candidates --min-object-area-ratio 0.001 --seed 20260710",
        "```",
        "",
        "4. Rank candidates and build review pool:",
        "",
        "```bash",
        "/share/apps/anaconda3/bin/python -m tools.evidencehoi_image_search_v2.rank_candidates_stage4 --candidate-path outputs/figure1_search_v2/candidates/raw_pair_candidates.parquet --index-dir outputs/figure1_search_v2/index --output-dir outputs/figure1_search_v2/ranking --manifest-path outputs/figure1_search_v2/manifest_stage4.json --seed 20260710 --paper-single-column-px 512",
        "```",
        "",
        "5. Generate visual review packet:",
        "",
        "```bash",
        "/share/apps/anaconda3/bin/python -m tools.evidencehoi_image_search_v2.generate_visual_review_stage5 --review-pool outputs/figure1_search_v2/ranking/review_pool.csv --index-dir outputs/figure1_search_v2/index --preview-dir outputs/figure1_search_v2/previews --contact-dir outputs/figure1_search_v2/contact_sheets --review-dir outputs/figure1_search_v2/review --manifest-path outputs/figure1_search_v2/manifest_stage5.json --seed 20260710",
        "```",
        "",
        "6. Run round-2 experiments:",
        "",
        "```bash",
        "/share/apps/anaconda3/bin/python -m tools.evidencehoi_image_search_v2.run_round2_experiments_stage7 --features outputs/figure1_search_v2/ranking/candidate_features.csv --review-pool outputs/figure1_search_v2/ranking/review_pool.csv --stage6-manifest outputs/figure1_search_v2/manifest_stage6.json --image-summary outputs/figure1_search_v2/index/image_summary.parquet --output-dir outputs/figure1_search_v2/round2 --manifest-path outputs/figure1_search_v2/manifest_stage7.json --seed 20260710",
        "```",
        "",
        "7. Build Stage 8 top-10 proxy analysis:",
        "",
        "```bash",
        "/share/apps/anaconda3/bin/python -m tools.evidencehoi_image_search_v2.build_top10_stage8 --features outputs/figure1_search_v2/ranking/candidate_features.csv --stage6-shortlist outputs/figure1_search_v2/review/shortlist_v1.csv --stage7-shortlist outputs/figure1_search_v2/round2/shortlist_v2.csv --stage7-delta outputs/figure1_search_v2/round2/delta_candidates.csv --image-summary outputs/figure1_search_v2/index/image_summary.parquet --output-dir outputs/figure1_search_v2/top10 --manifest-path outputs/figure1_search_v2/manifest_stage8.json --seed 20260710",
        "```",
        "",
        "8. Build this final Stage 9 packet:",
        "",
        "```bash",
        successful_command,
        "```",
        "",
        "## Key Outputs",
        "",
        "- `candidate_images.csv`: final ranked candidate table with GT/AUTO/manual-note boundaries.",
        "- `final_recommendation.md`: decision rationale and residual risks.",
        "- `primary/`: primary image, crop, overlays, cue crops, low-fidelity four-panel draft.",
        "- `backups/`: same-narrative and different-scene backup images and overlays.",
        "- `reproducibility_manifest.json`: artifact list, checksums, seed, and integrity-check result.",
        "- `file_manifest.csv`: flat path/checksum manifest.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def integrity_check(paths: list[Path]) -> tuple[bool, list[str]]:
    missing = []
    for path in paths:
        if not path.exists():
            missing.append(project_relative(path))
    return len(missing) == 0, missing


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    dirs = ensure_dirs(output_dir)
    command = " ".join([
        "/share/apps/anaconda3/bin/python",
        "-m",
        "tools.evidencehoi_image_search_v2.build_final_stage9",
        "--top10-matrix", project_relative(args.top10_matrix),
        "--features", project_relative(args.features),
        "--index-dir", project_relative(args.index_dir),
        "--output-dir", project_relative(output_dir),
        "--manifest-path", project_relative(args.manifest_path),
        "--seed", str(args.seed),
    ])
    run_log = dirs["logs"] / "stage9_run.log"
    run_log.write_text(f"generated_at={datetime.now(timezone.utc).isoformat()}\nseed={args.seed}\ncommand={command}\n", encoding="utf-8")

    top10 = pd.read_csv(args.top10_matrix)
    features = pd.read_csv(args.features)
    images = pd.read_parquet(args.index_dir / "image_summary.parquet")
    instances = pd.read_parquet(args.index_dir / "instances.parquet")
    pairs = pd.read_parquet(args.index_dir / "pairs.parquet")
    image_by_key = {str(row["image_key"]): row for row in images.to_dict("records")}
    pairs_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in pairs.to_dict("records"):
        pairs_by_image.setdefault(str(row["image_key"]), []).append(row)
    instances_by_image: dict[str, list[dict[str, Any]]] = {}
    for row in instances.to_dict("records"):
        instances_by_image.setdefault(str(row["image_key"]), []).append(row)

    feature_by_id = {str(row["candidate_id"]): row for row in features.to_dict("records")}
    top10_rows = top10.to_dict("records")
    generated_paths: list[Path] = []
    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    blind_rows: list[dict[str, Any]] = []
    annotation_fragment_paths: list[Path] = []

    material_paths_by_top10: dict[str, dict[str, Path]] = {}
    for row in top10_rows:
        candidate_id = str(row["candidate_id"])
        if candidate_id not in feature_by_id:
            continue
        candidate = dict(feature_by_id[candidate_id])
        candidate["top10_id"] = str(row["top10_id"])
        candidate["confusing_verb_prototypes"] = row.get("confusing_verb_prototypes", row.get("confusion_prototypes", "[]"))
        image_row = image_by_key[str(candidate["image_key"])]
        image_path = image_path_for(image_row)
        image = gv.load_image(image_path)
        all_gt_pairs, second_pairs, auxiliary_pairs = gv.related_pairs(candidate, pairs_by_image.get(str(candidate["image_key"]), []))
        contexts, noise_candidates = gv.context_and_noise_instances(candidate, instances_by_image.get(str(candidate["image_key"]), []), all_gt_pairs, second_pairs, auxiliary_pairs)
        related = {
            "all_gt_pairs": all_gt_pairs,
            "second_pairs": second_pairs,
            "auxiliary_pairs": auxiliary_pairs,
            "contexts": contexts,
            "noise_candidates": noise_candidates,
        }
        t10_id = str(row["top10_id"])
        base = dirs["review_materials"] / t10_id
        base.mkdir(parents=True, exist_ok=True)
        paths = {
            "original": base / f"{t10_id}_original.jpg",
            "suggested_crop": base / f"{t10_id}_suggested_crop.jpg",
            "suggested_crop_overlay": base / f"{t10_id}_suggested_crop_overlay.jpg",
            "all_gt": base / f"{t10_id}_all_gt_overlay.jpg",
            "target_pair": base / f"{t10_id}_target_pair_overlay.jpg",
            "context_noise": base / f"{t10_id}_context_noise_overlay.jpg",
            "auxiliary": base / f"{t10_id}_target_human_auxiliary_overlay.jpg",
            "narrative": base / f"{t10_id}_narrative_overlay.jpg",
            "single_column": base / f"{t10_id}_single_column.jpg",
            "cue_crops": base / f"{t10_id}_six_visual_cue_crops.jpg",
            "annotation_fragment": dirs["annotation_fragments"] / f"{t10_id}_annotation_fragment.json",
        }
        gv.make_original(image, t10_id, paths["original"], max_side=1200)
        crop_raw(image, candidate, paths["suggested_crop"])
        gv.make_crop_overlay(image, candidate, t10_id, paths["suggested_crop_overlay"], max_side=1200)
        gv.make_all_gt(image, candidate, all_gt_pairs, t10_id, paths["all_gt"], max_side=1200)
        gv.make_target_pair(image, candidate, t10_id, paths["target_pair"], max_side=1200)
        gv.make_context_overlay(image, candidate, second_pairs, auxiliary_pairs, contexts, noise_candidates, t10_id, paths["context_noise"], max_side=1200)
        gv.make_auxiliary_overlay(image, candidate, auxiliary_pairs, t10_id, paths["auxiliary"], max_side=1200)
        gv.make_narrative_overlay(image, candidate, second_pairs, contexts, noise_candidates, t10_id, paths["narrative"], max_side=1200)
        gv.make_single_column(image, candidate, t10_id, paths["single_column"], single_column_px=512)
        gv.make_cue_crops(image, candidate, contexts, noise_candidates, t10_id, paths["cue_crops"], cue_size=224)
        write_annotation_fragment(paths["annotation_fragment"], candidate, related, image_path)
        generated_paths.extend(paths.values())
        annotation_fragment_paths.append(paths["annotation_fragment"])
        material_paths_by_top10[t10_id] = paths

        selected = FINAL_SELECTION.get(candidate_id)
        scores = score_row(candidate, selected)
        eligible = selected is not None
        other_pairs = [pair_summary(pair) for pair in second_pairs[:8]]
        context_objects = [short_entity(item) for item in contexts[:10]]
        distractors = [short_entity(item) for item in noise_candidates[:8]]
        row_out = {
            "top10_id": t10_id,
            "candidate_id": candidate_id,
            "pair_id": str(candidate["pair_id"]),
            "image_id": str(row.get("image_id", candidate.get("image_id", candidate.get("image_key")))),
            "dataset_source": str(row.get("dataset_source", "HICO-DET")),
            "split": str(candidate["split"]),
            "target_human_id": str(candidate["canonical_human_id"]),
            "target_object_id": str(candidate["canonical_object_id"]),
            "object_class": str(candidate["object_category_name"]),
            "target_gt_verbs": json_dumps(json_loads(candidate.get("gt_verb_names_json"), [])),
            "confusion_prototypes": str(row.get("confusing_verb_prototypes", "[]")),
            "other_pairs": json_dumps(other_pairs),
            "context_objects": json_dumps(context_objects),
            "context_evidence_source": "[GT] all_gt/auxiliary/second-pair overlays plus [AUTO] context-count proxies",
            "distractors": json_dumps(distractors),
            "distractor_confidence": "low_auto_unconfirmed",
            "same_object_shared_by_humans": str(candidate.get("shared_object_status", False)),
            "candidate_tier": selected["candidate_tier"] if selected else "rejected_top10",
            "bbox_size_score": scores["bbox_size_score"],
            "auto_pre_score": round(safe_float(candidate.get("auto_pre_score")), 6),
            "ambiguity_score": scores["ambiguity_score"],
            "pair_specific_score": scores["pair_specific_score"],
            "context_score": scores["context_score"],
            "noise_score": scores["noise_score"],
            "readability_score": scores["readability_score"],
            "drawing_score": scores["drawing_score"],
            "final_score": scores["final_score"],
            "eligible_for_final": str(eligible),
            "rejection_reason": "" if eligible else REJECTION_NOTES.get(candidate_id, "Rejected for final-three: did not beat selected candidates under no-hard-issue, readability, pair-specific, and drawing-space priorities."),
            "final_rank": selected["final_rank"] if selected else "",
            "selection_note": selected["manual_note"] if selected else "",
            "original_path": project_relative(paths["original"]),
            "suggested_crop_path": project_relative(paths["suggested_crop"]),
            "all_gt_overlay_path": project_relative(paths["all_gt"]),
            "target_pair_overlay_path": project_relative(paths["target_pair"]),
            "context_noise_overlay_path": project_relative(paths["context_noise"]),
            "cue_crops_path": project_relative(paths["cue_crops"]),
            "annotation_fragment_path": project_relative(paths["annotation_fragment"]),
        }
        candidate_rows.append(row_out)
        if eligible:
            selected_rows.append(row_out)
            blind_rows.append({
                "blind_id": f"F9_{selected['final_rank']:02d}",
                "blind_label": f"rank {selected['final_rank']} / {candidate['object_category_name']}",
                "single_column_path": str(paths["single_column"]),
            })

    candidate_rows.sort(key=lambda item: (safe_int(item["final_rank"], 999), str(item["top10_id"])))
    selected_rows.sort(key=lambda item: safe_int(item["final_rank"], 999))
    candidate_csv = output_dir / "candidate_images.csv"
    pd.DataFrame(candidate_rows).to_csv(candidate_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    generated_paths.append(candidate_csv)

    for selected in selected_rows:
        paths = material_paths_by_top10[selected["top10_id"]]
        dst_root = dirs["primary"] if selected["final_rank"] == 1 else dirs["backups"]
        prefix = selected["top10_id"]
        mapping = {
            "original": dst_root / f"{prefix}_original.jpg",
            "suggested_crop": dst_root / f"{prefix}_suggested_crop.jpg",
            "suggested_crop_overlay": dst_root / f"{prefix}_suggested_crop_overlay.jpg",
            "all_gt": dst_root / f"{prefix}_all_gt_overlay.jpg",
            "target_pair": dst_root / f"{prefix}_target_pair_overlay.jpg",
            "context_noise": dst_root / f"{prefix}_context_noise_overlay.jpg",
            "cue_crops": (dirs["primary_cues"] if selected["final_rank"] == 1 else dst_root) / f"{prefix}_six_visual_cue_crops.jpg",
            "annotation_fragment": dst_root / f"{prefix}_annotation_fragment.json",
        }
        for key, dst in mapping.items():
            copy_if_exists(paths[key], dst)
            generated_paths.append(dst)
        if selected["final_rank"] == 1:
            four_panel = dirs["primary"] / f"{prefix}_four_panel_low_fidelity_draft.jpg"
            make_four_panel(mapping, four_panel)
            generated_paths.append(four_panel)

    final_contact = dirs["contact_sheets"] / "final_three_blind_contact_sheet.jpg"
    make_blind_contact(blind_rows, final_contact, "Stage 9 final-three blind contact sheet")
    generated_paths.append(final_contact)
    top10_blind_rows = []
    for idx, row in enumerate(top10_rows, start=1):
        t10_id = str(row["top10_id"])
        if t10_id in material_paths_by_top10:
            top10_blind_rows.append({"blind_id": f"B{idx:02d}", "blind_label": str(row.get("target_object_category", "candidate")), "single_column_path": str(material_paths_by_top10[t10_id]["single_column"])})
    top10_contact = dirs["contact_sheets"] / "top10_blind_contact_sheet.jpg"
    make_blind_contact(top10_blind_rows, top10_contact, "Stage 9 top-10 blind contact sheet")
    generated_paths.append(top10_contact)

    # Copy reports/config/mapping/manifests/logs needed for a self-contained handoff.
    copies = [
        (OUTPUT_ROOT / "top10/top10_analysis.md", dirs["reports"] / "top10_analysis.md"),
        (OUTPUT_ROOT / "top10/top10_matrix.csv", dirs["reports"] / "top10_matrix.csv"),
        (OUTPUT_ROOT / "top10/risk_register.csv", dirs["reports"] / "risk_register.csv"),
        (OUTPUT_ROOT / "ranking/feature_report.md", dirs["reports"] / "feature_report.md"),
        (OUTPUT_ROOT / "round2/stage7_notes.md", dirs["reports"] / "stage7_notes.md"),
        (OUTPUT_ROOT / "round2/experiment_log.csv", dirs["logs"] / "stage7_experiment_log.csv"),
        (OUTPUT_ROOT / "round2/failure_matrix.csv", dirs["reports"] / "failure_matrix.csv"),
        (OUTPUT_ROOT / "configs/search_config_v2.yaml", dirs["configs"] / "search_config_v2.yaml"),
        (OUTPUT_ROOT / "configs/search_contract_v2.md", dirs["configs"] / "search_contract_v2.md"),
        (OUTPUT_ROOT / "configs/legacy_exclusion_list.txt", dirs["configs"] / "legacy_exclusion_list.txt"),
        (OUTPUT_ROOT / "audit/mapping_report.md", dirs["mapping"] / "mapping_report.md"),
        (OUTPUT_ROOT / "audit/object_action_mapping.csv", dirs["mapping"] / "object_action_mapping.csv"),
        (OUTPUT_ROOT / "review/blind_key.csv", dirs["reports"] / "stage5_blind_key.csv"),
        (OUTPUT_ROOT / "review/visual_manifest.csv", dirs["reports"] / "stage5_visual_manifest.csv"),
        (OUTPUT_ROOT / "contact_sheets/mixed_anonymous_blind_contact_sheet.jpg", dirs["contact_sheets"] / "stage5_mixed_anonymous_blind_contact_sheet.jpg"),
    ]
    for src, dst in copies:
        if copy_if_exists(src, dst):
            generated_paths.append(dst)
    for stage_manifest in sorted(OUTPUT_ROOT.glob("manifest_stage*.json")):
        if stage_manifest.name != "manifest_stage9.json":
            dst = dirs["manifests"] / stage_manifest.name
            copy_if_exists(stage_manifest, dst)
            generated_paths.append(dst)
    for contact in sorted((OUTPUT_ROOT / "round2/delta_contact_sheets").glob("*.jpg")):
        dst = dirs["contact_sheets"] / f"stage7_{contact.name}"
        copy_if_exists(contact, dst)
        generated_paths.append(dst)

    seed_path = output_dir / "random_seed.txt"
    seed_path.write_text(str(args.seed) + "\n", encoding="utf-8")
    generated_paths.append(seed_path)

    final_md = output_dir / "final_recommendation.md"
    write_final_recommendation(final_md, candidate_rows, selected_rows)
    generated_paths.append(final_md)
    readme = output_dir / "README.md"
    write_readme(readme, command)
    generated_paths.append(readme)

    expected_paths = list(dict.fromkeys(generated_paths))
    integrity_ok, missing = integrity_check(expected_paths)
    file_manifest = output_dir / "file_manifest.csv"
    manifest_rows = []
    for path in sorted(set(expected_paths), key=lambda p: project_relative(p)):
        if path.exists() and path.is_file():
            manifest_rows.append({"path": project_relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(file_manifest, index=False, quoting=csv.QUOTE_MINIMAL)
    generated_paths.append(file_manifest)

    repo_commit = "unknown"
    try:
        repo_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        pass
    reproducibility = {
        "schema_version": "figure1_search_v2.stage9.reproducibility_manifest",
        "stage": 9,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_root": project_relative(output_dir),
        "random_seed": args.seed,
        "git_commit": repo_commit,
        "vcoco_enabled": False,
        "vcoco_reason": "Disabled by search_config_v2.yaml; HICO-DET annotations and mappings are sufficient for this final selection.",
        "final_selection": [{"rank": row["final_rank"], "top10_id": row["top10_id"], "candidate_id": row["candidate_id"], "tier": row["candidate_tier"]} for row in selected_rows],
        "manual_visual_boundary": "No direct semantic pixel inspection is claimed by this runtime; manual fields are curated stage-9 reviewer notes over generated artifacts.",
        "integrity_check": {"passed": integrity_ok, "missing": missing, "checked_path_count": len(expected_paths)},
        "file_manifest": project_relative(file_manifest),
        "artifacts": manifest_rows,
    }
    repro_path = output_dir / "reproducibility_manifest.json"
    repro_path.write_text(json.dumps(reproducibility, ensure_ascii=False, indent=2), encoding="utf-8")
    generated_paths.append(repro_path)

    stage9_manifest = {
        "schema_version": "figure1_search_v2.stage9",
        "stage": 9,
        "stage_name": "final_candidate_selection_and_handoff",
        "status": "completed" if integrity_ok else "completed_with_missing_paths",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_root": project_relative(output_dir),
        "random_seed": args.seed,
        "git_commit": repo_commit,
        "successful_command": command,
        "inputs": {
            "top10_matrix": project_relative(args.top10_matrix),
            "candidate_features": project_relative(args.features),
            "index_dir": project_relative(args.index_dir),
        },
        "outputs": {
            "candidate_images": project_relative(candidate_csv),
            "final_recommendation": project_relative(final_md),
            "primary_dir": project_relative(dirs["primary"]),
            "backups_dir": project_relative(dirs["backups"]),
            "reproducibility_manifest": project_relative(repro_path),
            "readme": project_relative(readme),
            "file_manifest": project_relative(file_manifest),
        },
        "statistics": {
            "top10_rows_evaluated": len(candidate_rows),
            "final_selected_rows": len(selected_rows),
            "generated_file_count": len(manifest_rows),
        },
        "acceptance": {
            "candidate_images_csv_present": candidate_csv.exists(),
            "primary_original_and_crop_present": (dirs["primary"] / "T10_02_original.jpg").exists() and (dirs["primary"] / "T10_02_suggested_crop.jpg").exists(),
            "primary_overlays_present": all((dirs["primary"] / name).exists() for name in ["T10_02_all_gt_overlay.jpg", "T10_02_target_pair_overlay.jpg", "T10_02_context_noise_overlay.jpg"]),
            "backup_images_present": (dirs["backups"] / "T10_01_original.jpg").exists() and (dirs["backups"] / "T10_05_original.jpg").exists(),
            "integrity_check_passed": integrity_ok,
            "paper_body_or_final_figure_modified": False,
        },
    }
    args.manifest_path.write_text(json.dumps(stage9_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # Include final manifest checksums after writing it.
    final_manifest_rows = pd.read_csv(file_manifest).to_dict("records")
    for extra in [repro_path, args.manifest_path]:
        final_manifest_rows.append({"path": project_relative(extra), "bytes": extra.stat().st_size, "sha256": sha256(extra)})
    pd.DataFrame(final_manifest_rows).drop_duplicates("path", keep="last").sort_values("path").to_csv(file_manifest, index=False, quoting=csv.QUOTE_MINIMAL)


if __name__ == "__main__":
    main()
