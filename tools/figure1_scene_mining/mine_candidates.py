#!/usr/bin/env python3
"""Mine pair-centric EvidenceHOI Figure 1 candidates from an audited HICO index.

The script deliberately produces an automatic priority score, not a final
semantic score. Pair ambiguity, decisive evidence and narrative suitability
must be confirmed from the rendered contact sheets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("scenario_config.yaml")
DEFAULT_OUTPUT = Path(__file__).with_name("outputs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-per-scenario", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def loads(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            result = json.loads(value)
            return result if isinstance(result, list) else []
        except json.JSONDecodeError:
            return []
    return []


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def bbox(row: pd.Series, prefix: str) -> list[float]:
    return [float(row[f"{prefix}_bbox_x1"]), float(row[f"{prefix}_bbox_y1"]),
            float(row[f"{prefix}_bbox_x2"]), float(row[f"{prefix}_bbox_y2"])]


def box_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def union_box(a: list[float], b: list[float]) -> list[float]:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def overlap_ratio(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / max(1.0, min(box_area(a), box_area(b)))


def instance_payload(row: pd.Series) -> dict[str, Any]:
    return {
        "instance_id": str(row["canonical_instance_id"]),
        "role": str(row["role"]),
        "category": str(row["category_name"]),
        "bbox": [float(row["bbox_x1"]), float(row["bbox_y1"]), float(row["bbox_x2"]), float(row["bbox_y2"])],
        "bbox_area": float(row["bbox_area"]),
    }


def clipped(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def score_candidate(
    scenario: str,
    spec: dict[str, Any],
    pair: pd.Series,
    image: pd.Series,
    contexts: list[dict[str, Any]],
    decisive: list[dict[str, Any]],
    distractors: list[dict[str, Any]],
) -> dict[str, float]:
    width, height = float(image["width"]), float(image["height"])
    image_area = max(1.0, width * height)
    hbox, obox = bbox(pair, "human"), bbox(pair, "object")
    ubox = union_box(hbox, obox)
    h_ratio = box_area(hbox) / image_area
    o_ratio = box_area(obox) / image_area
    union_ratio = box_area(ubox) / image_area
    hc, oc = box_center(hbox), box_center(obox)
    center_distance = math.dist(hc, oc) / max(1.0, math.hypot(width, height))
    free_space = max(ubox[0], width - ubox[2], ubox[1], height - ubox[3]) / max(width, height)
    people = int(image["canonical_human_count"])
    entity_count = int(image["canonical_instance_count"])
    context_count = len(contexts)
    verbs = loads(pair["verb_names_json"])

    pair_legibility = 0.45 * clipped(h_ratio / 0.12, 0, 1) + 0.55 * clipped(o_ratio / 0.02, 0, 1)
    composition = clipped(free_space / 0.30, 0, 1)
    entity_fit = 1.0 if 4 <= entity_count <= 10 else clipped(1.0 - abs(entity_count - 7) / 12.0, 0, 1)
    one_person = 1.0 if people == 1 else (0.55 if people == 2 else 0.20)
    context_fit = clipped(context_count / 4.0, 0, 1)
    distractor_fit = clipped(len(distractors) / 2.0, 0, 1)

    if decisive:
        decisive_fit = 1.0
    elif scenario == "bicycle":
        decisive_fit = 0.55 + 0.25 * clipped(overlap_ratio(hbox, obox) / 0.25, 0, 1)
    elif scenario == "kite":
        decisive_fit = 0.45 + 0.35 * clipped(center_distance / 0.35, 0, 1)
    elif scenario in {"tv", "laptop"}:
        decisive_fit = 0.35 + 0.25 * clipped(context_count / 3.0, 0, 1)
    else:
        decisive_fit = 0.15

    ambiguity_proxy = clipped(0.45 + 0.20 * (len(verbs) >= 2) + 0.20 * (str(pair["pair_structural_tier"]) in {"Gold", "Silver"}) + 0.15 * (context_count >= 2), 0, 1)
    long_range = clipped(center_distance / 0.45, 0, 1) if spec.get("long_range") else clipped(center_distance / 0.65, 0, 0.65)
    topology = clipped((decisive_fit + distractor_fit + context_fit + pair_legibility) / 4.0, 0, 1)

    auto_score = (
        14 * ambiguity_proxy + 14 * decisive_fit + 9 * distractor_fit + 9 * topology
        + 8 * pair_legibility + 5 * composition + 4 * long_range + 4 * one_person
        + 3 * entity_fit + 2 * context_fit
    )
    return {
        "auto_priority_score": round(auto_score, 4),
        "auto_ambiguity_proxy": round(ambiguity_proxy, 4),
        "auto_decisive_proxy": round(decisive_fit, 4),
        "auto_distractor_proxy": round(distractor_fit, 4),
        "auto_topology_proxy": round(topology, 4),
        "auto_cropability_proxy": round(pair_legibility, 4),
        "auto_composition_proxy": round(composition, 4),
        "auto_long_range_proxy": round(long_range, 4),
        "human_area_ratio": round(h_ratio, 7),
        "object_area_ratio": round(o_ratio, 7),
        "target_union_area_ratio": round(union_ratio, 7),
        "target_pair_center_distance": round(center_distance, 7),
        "target_pair_overlap_ratio": round(overlap_ratio(hbox, obox), 7),
        "layout_whitespace_proxy": round(free_space, 7),
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(args.seed if args.seed is not None else config.get("random_seed", 20260714))
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_parquet(resolve(config["index"]["pairs"]))
    instances = pd.read_parquet(resolve(config["index"]["instances"]))
    images = pd.read_parquet(resolve(config["index"]["images"]))
    image_lookup = images.set_index("image_key", drop=False)
    instances_by_image = {key: frame for key, frame in instances.groupby("image_key", sort=False)}

    specs = config["scenarios"]
    object_to_scenario = {spec["target_object"]: name for name, spec in specs.items()}
    target_objects = set(object_to_scenario)
    target_pairs = pairs[pairs["object_category_name"].isin(target_objects)].copy()
    target_pairs = target_pairs[(target_pairs["valid"] == True) & (target_pairs["invisible"] == False) & (target_pairs["eligible_for_final"] == True)]

    mining = config.get("mining", {})
    min_h = float(mining.get("min_human_area_ratio", 0.035))
    min_o = float(mining.get("min_object_area_ratio", 0.0008))
    max_per = int(args.max_per_scenario or mining.get("max_raw_per_scenario", 250))

    rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    scanned_by_scenario: dict[str, int] = {name: 0 for name in specs}
    for _, pair in target_pairs.iterrows():
        scenario = object_to_scenario[str(pair["object_category_name"])]
        scanned_by_scenario[scenario] += 1
        image = image_lookup.loc[pair["image_key"]]
        if isinstance(image, pd.DataFrame):
            image = image.iloc[0]
        if not Path(str(image["image_path"])).exists():
            rejected["image_missing"] = rejected.get("image_missing", 0) + 1
            continue
        verbs = loads(pair["verb_names_json"])
        if "no_interaction" in verbs:
            rejected["no_interaction"] = rejected.get("no_interaction", 0) + 1
            continue

        width, height = float(image["width"]), float(image["height"])
        image_area = max(1.0, width * height)
        hbox, obox = bbox(pair, "human"), bbox(pair, "object")
        if box_area(hbox) / image_area < min_h:
            rejected["human_too_small"] = rejected.get("human_too_small", 0) + 1
            continue
        scenario_min_o = min_o * (0.35 if scenario in {"kite", "tv"} else 1.0)
        if box_area(obox) / image_area < scenario_min_o:
            rejected["object_too_small"] = rejected.get("object_too_small", 0) + 1
            continue

        frame = instances_by_image.get(pair["image_key"])
        if frame is None:
            continue
        excluded = {str(pair["canonical_human_id"]), str(pair["canonical_object_id"])}
        contexts = [instance_payload(item) for _, item in frame.iterrows() if str(item["canonical_instance_id"]) not in excluded]
        spec = specs[scenario]
        decisive_names = set(spec.get("decisive_cues", []))
        distractor_names = set(spec.get("distractor_cues", []))
        decisive = [item for item in contexts if item["category"] in decisive_names]
        distractors = [item for item in contexts if item["category"] in distractor_names and item["category"] not in decisive_names]

        scores = score_candidate(scenario, spec, pair, image, contexts, decisive, distractors)
        row = {
            "candidate_id": str(pair["pair_id"]),
            "scenario": scenario,
            "dataset": "HICO-DET",
            "split": str(pair["split"]),
            "image_id": str(pair["image_id"]),
            "file_name": str(pair["file_name"]),
            "image_path": str(image["image_path"]),
            "image_width": int(image["width"]),
            "image_height": int(image["height"]),
            "human_bbox": dumps(hbox),
            "object_bbox": dumps(obox),
            "canonical_human_id": str(pair["canonical_human_id"]),
            "canonical_object_id": str(pair["canonical_object_id"]),
            "target_object": str(pair["object_category_name"]),
            "annotated_actions": dumps(verbs),
            "verb_count": int(pair["verb_count"]),
            "multi_action_pair": bool(int(pair["verb_count"]) >= 2),
            "alternative_plausible_actions_auto": dumps(spec.get("alternative_plausible_actions", [])),
            "number_of_people": int(image["canonical_human_count"]),
            "number_of_entities": int(image["canonical_instance_count"]),
            "number_of_context_objects": len(contexts),
            "context_categories": dumps(sorted({item["category"] for item in contexts})),
            "context_instances": dumps(contexts),
            "decisive_cue_candidates": dumps(decisive),
            "distractor_candidates": dumps(distractors),
            "scene_evidence_auto": str(spec.get("scene_evidence", "")),
            "pair_structural_tier": str(pair["pair_structural_tier"]),
            "raw_annotation_provenance": str(pair["raw_annotation_provenance_json"]),
            "license_or_source_note": "HICO-DET dataset image; retain dataset split, image ID, and annotation provenance.",
            "random_seed": seed,
            **scores,
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("No candidates survived the structural filters")
    frame = frame.sort_values(["scenario", "auto_priority_score", "candidate_id"], ascending=[True, False, True], kind="mergesort")
    frame["scenario_rank"] = frame.groupby("scenario").cumcount() + 1
    frame = frame[frame["scenario_rank"] <= max_per].copy()
    frame = frame.sort_values(["auto_priority_score", "scenario", "candidate_id"], ascending=[False, True, True], kind="mergesort")
    frame["overall_auto_rank"] = range(1, len(frame) + 1)
    frame.to_csv(output_dir / "raw_candidates.csv", index=False)

    scenario_counts = {name: int((frame["scenario"] == name).sum()) for name in specs}
    report = {
        "schema_version": "evidencehoi.figure1_scene_mining.raw.v1",
        "random_seed": seed,
        "dataset": "HICO-DET",
        "index_paths": {key: str(resolve(value)) for key, value in config["index"].items()},
        "scanned_target_pairs_by_scenario": scanned_by_scenario,
        "retained_raw_candidates_by_scenario": scenario_counts,
        "retained_raw_total": int(len(frame)),
        "rejected": rejected,
        "manual_review_required": True,
        "warning": "Automatic scores are structural proxies and must not be reported as final semantic evidence scores.",
    }
    (output_dir / "mining_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
