#!/usr/bin/env python3
"""Contrastive Figure 1 image search over the audited HICO-DET GT index.

This v3 search deliberately treats a canonical human-object pair as the
retrieval unit. It avoids ranking primarily by raw numbers of humans, objects,
HOIs, or duplicate same-class instances. Instead it scores whether a target pair
has a semantically different second pair/activity, decisive action-context cues,
and plausible scene/neighborhood distractors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs/figure1_search_v2/index"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/figure1_search_v3_contrastive"
DEFAULT_BLIND_KEY = PROJECT_ROOT / "outputs/figure1_search_v2/review/blind_key.csv"
DEFAULT_V2_FINAL = PROJECT_ROOT / "outputs/figure1_search_v2/final/candidate_images.csv"
DEFAULT_ACTIONS = PROJECT_ROOT / "hico_20160224_det/hico_list_vb.txt"

INITIAL_SIGNATURE_THRESHOLD = 0.65


CSV_COLUMNS = [
    "rank",
    "contrastive_score",
    "candidate_id",
    "pair_id",
    "image_key",
    "image_id",
    "split",
    "file_name",
    "image_path",
    "eligible_for_final",
    "target_human_id",
    "target_object_id",
    "target_object_category",
    "target_verbs",
    "target_verb_families",
    "target_human_short_edge_px",
    "target_human_area_ratio",
    "effective_human_count",
    "distinct_cross_human_pair_types",
    "dominant_pair_type_ratio",
    "duplicate_instance_ratio",
    "same_object_category_same_family_peer_count",
    "max_human_signature_distance",
    "target_is_activity_outlier",
    "target_pair_type",
    "contrast_human_id",
    "contrast_human_short_edge_px",
    "contrast_pair_type",
    "other_human_activity_summary",
    "confusion_verb_set",
    "decisive_context_candidates",
    "context_source",
    "misleading_neighbor",
    "neighbor_conflict_count",
    "scene_mismatch_proxy",
    "pair_crop_ambiguity",
    "context_gain",
    "candidate_swap_risk",
    "search_track",
    "hard_negative_reason",
    "hard_negative_rule_hits",
    "diagnostic_rule_hits",
    "eligible_after_hard_negative_rules",
    "diagnostic_only",
    "threshold_used",
    "threshold_pass",
    "target_union_norm_area",
    "object_short_edge_px",
]

TRACK_COLUMNS = [
    *CSV_COLUMNS,
    "track_id",
    "track_score",
    "track_reason",
    "manual_review_question",
    "second_pair_id",
    "second_pair_type",
    "borrowed_evidence_failure_mode",
]

PRIORITY_COLUMNS = [
    "priority_rank",
    "priority_group",
    *TRACK_COLUMNS,
]


TRACK_C_OBJECT_SPECS = {
    "cake": {
        "verbs": {"cut", "blow", "eat", "hold", "make"},
        "hico_context": {"knife", "fork", "dining table", "cup", "bottle", "person", "chair"},
        "manual_context": {"candles", "plate", "face/mouth pose", "birthday crowd"},
    },
    "pizza": {
        "verbs": {"cut", "eat", "make", "hold"},
        "hico_context": {"knife", "fork", "dining table", "chair", "oven", "person"},
        "manual_context": {"plate"},
    },
    "tv": {
        "verbs": {"watch", "repair", "carry"},
        "hico_context": {"remote", "couch", "chair", "person", "laptop", "cell phone"},
        "manual_context": {"tools", "indoor/outdoor layout"},
    },
    "laptop": {
        "verbs": {"type_on", "read", "open", "hold", "repair"},
        "hico_context": {"mouse", "keyboard", "book", "cell phone", "chair", "dining table", "backpack", "suitcase"},
        "manual_context": {"tools", "bag contents"},
    },
    "suitcase": {
        "verbs": {"drag", "carry", "open", "pack", "load"},
        "hico_context": {"bed", "car", "backpack", "handbag", "person"},
        "manual_context": {"wheels", "extended handle", "clothes"},
    },
    "bicycle": {
        "verbs": {"ride", "push", "repair", "wash", "hold"},
        "hico_context": {"person", "backpack", "helmet", "car", "truck", "motorcycle"},
        "manual_context": {"road", "tools", "handlebar", "pedal", "standing pose"},
    },
    "motorcycle": {
        "verbs": {"ride", "push", "repair", "wash", "hold"},
        "hico_context": {"person", "bicycle", "car", "truck", "backpack"},
        "manual_context": {"road", "tools", "helmet", "handlebar", "standing pose"},
    },
    "dining table": {
        "verbs": {"eat_at", "sit_at", "clean"},
        "hico_context": {"chair", "cup", "bottle", "wine glass", "fork", "knife", "spoon", "bowl", "pizza", "cake", "sandwich"},
        "manual_context": {"cloth", "food layout"},
    },
    "couch": {
        "verbs": {"sit_on", "lie_on", "stand_on", "carry"},
        "hico_context": {"tv", "book", "chair", "remote", "person", "laptop"},
        "manual_context": {"body pose", "other furniture"},
    },
    "chair": {
        "verbs": {"sit_on", "lie_on", "stand_on", "carry"},
        "hico_context": {"tv", "book", "couch", "dining table", "person", "laptop"},
        "manual_context": {"body pose", "other furniture"},
    },
    "bottle": {
        "verbs": {"drink_with", "pour", "hold", "carry"},
        "hico_context": {"cup", "wine glass", "dining table", "bowl", "person", "pizza", "cake", "sandwich"},
        "manual_context": {"another container", "food", "other diners"},
    },
    "cup": {
        "verbs": {"drink_with", "pour", "hold", "carry"},
        "hico_context": {"bottle", "wine glass", "dining table", "bowl", "person", "pizza", "cake", "sandwich"},
        "manual_context": {"another container", "food", "other diners"},
    },
    "wine glass": {
        "verbs": {"drink_with", "pour", "hold", "carry"},
        "hico_context": {"bottle", "cup", "dining table", "bowl", "person", "pizza", "cake", "sandwich"},
        "manual_context": {"another container", "food", "other diners"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--blind-key", type=Path, default=DEFAULT_BLIND_KEY)
    parser.add_argument("--v2-final-candidates", type=Path, default=DEFAULT_V2_FINAL)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--initial-threshold", type=float, default=INITIAL_SIGNATURE_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--copy-top-images", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def jdump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def read_action_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            try:
                names[int(fields[0])] = fields[1]
            except ValueError:
                continue
    if set(names) != set(range(1, 118)):
        raise RuntimeError(f"Action mapping is incomplete: {path}")
    return names


def verb_family(verb: str) -> str:
    """Map local HICO verbs to broader visual/action families.

    The mapping is intentionally conservative: visually similar possession and
    transport verbs such as hold/carry/pick_up/lift are collapsed, so they cannot
    create artificial strong heterogeneity by name alone.
    """

    v = verb.strip().lower()
    groups = {
        "possession_transport": {
            "hold", "carry", "pick", "pick_up", "lift", "load", "pack", "release", "throw",
            "catch", "move", "drag", "pull", "push", "fill", "pour", "open", "zip", "tie",
        },
        "consume_food_drink": {"eat", "eat_at", "drink_with", "sip", "smell", "cut", "cut_with", "peel", "cook", "stir", "serve", "toast"},
        "ride_mount_vehicle": {"ride", "straddle", "sit_on", "sit_at", "hop_on", "board", "drive", "row", "sail", "fly", "race", "park", "exit"},
        "sports_play": {"kick", "hit", "catch", "throw", "dribble", "swing", "spin", "flip", "jump", "slide", "launch", "block", "serve"},
        "tool_device_operate": {"type_on", "talk_on", "text_on", "operate", "control", "watch", "read", "check", "inspect", "sign"},
        "repair_make_modify": {"assemble", "break", "repair", "install", "make", "paint", "adjust", "clean", "wash", "dry", "light", "grind"},
        "person_animal_social": {"feed", "pet", "groom", "hug", "kiss", "greet", "teach", "chase", "herd", "hunt", "lasso", "milk", "shear"},
        "body_posture_contact": {"lie_on", "stand_on", "stand_under", "walk", "run", "wear", "wield", "point", "wave", "scratch", "lick", "stick", "stop_at"},
        "hygiene_surface": {"brush_with", "hose", "squeeze", "stab", "blow", "flush", "lose", "direct", "pay"},
        "none": {"no_interaction"},
    }
    for family, verbs in groups.items():
        if v in verbs:
            return family
    return f"other:{v}"


def build_family_mapping(action_names: dict[int, str]) -> list[dict[str, Any]]:
    return [
        {"verb_id": verb_id, "verb_name": action_names[verb_id], "verb_family": verb_family(action_names[verb_id])}
        for verb_id in sorted(action_names)
    ]


def bbox(row: pd.Series, prefix: str) -> tuple[float, float, float, float]:
    return (
        float(row[f"{prefix}_bbox_x1"]),
        float(row[f"{prefix}_bbox_y1"]),
        float(row[f"{prefix}_bbox_x2"]),
        float(row[f"{prefix}_bbox_y2"]),
    )


def inst_bbox(row: pd.Series) -> tuple[float, float, float, float]:
    return (float(row.bbox_x1), float(row.bbox_y1), float(row.bbox_x2), float(row.bbox_y2))


def area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def union_box(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float, float]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def diag(box: Sequence[float]) -> float:
    return math.hypot(max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1]))


def center_distance(a: Sequence[float], b: Sequence[float]) -> float:
    ac = center(a)
    bc = center(b)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])


def is_near(a: Sequence[float], b: Sequence[float], factor: float = 1.5) -> bool:
    return center_distance(a, b) <= factor * max(diag(a), diag(b), 1.0)


def jaccard_distance(left: set[Any], right: set[Any]) -> float:
    if not left and not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return 1.0 - (len(left & right) / len(union))


def signature_distance(a: dict[str, set[Any]], b: dict[str, set[Any]]) -> dict[str, float]:
    object_distance = jaccard_distance(a["objects"], b["objects"])
    verb_distance = jaccard_distance(a["verb_families"], b["verb_families"])
    pair_type_distance = jaccard_distance(a["pair_types"], b["pair_types"])
    total = 0.3 * object_distance + 0.3 * verb_distance + 0.4 * pair_type_distance
    return {
        "object_distance": object_distance,
        "verb_distance": verb_distance,
        "pair_type_distance": pair_type_distance,
        "human_signature_distance": total,
    }


def quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {key: 0.0 for key in ["min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"]}
    s = pd.Series(values, dtype="float64")
    return {
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "mean": float(s.mean()),
    }


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def summarize_pair_types(pair_types: Iterable[tuple[str, str]], limit: int = 4) -> str:
    items = sorted({f"{obj}/{family}" for obj, family in pair_types})
    if len(items) > limit:
        return "; ".join(items[:limit]) + f"; +{len(items) - limit} more"
    return "; ".join(items)


def parse_previous_negative_images(
    blind_key_path: Path,
    final_candidates_path: Path,
    output_dir: Path,
) -> tuple[set[str], list[dict[str, Any]], dict[str, Any]]:
    """Read the previous blind key and resolve T10_01/T10_02/T10_05 image IDs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    blind_rows: list[dict[str, str]] = []
    if blind_key_path.exists():
        with blind_key_path.open("r", encoding="utf-8", newline="") as handle:
            blind_rows = list(csv.DictReader(handle))

    target_top10 = {"T10_01", "T10_02", "T10_05"}
    resolved: list[dict[str, Any]] = []
    negative_image_keys: set[str] = set()

    def file_name_from_image_key(image_key: str) -> str:
        stem = image_key.split(":", 1)[1] if ":" in image_key else image_key
        return f"{stem}.jpg" if stem else ""

    # The blind key is intentionally anonymized and does not expose T10 IDs in
    # this run; keep it as the required audit source, then resolve the final T10
    # IDs through the previous final manifest.
    if final_candidates_path.exists():
        with final_candidates_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("top10_id") not in target_top10:
                    continue
                candidate_id = row.get("candidate_id", "")
                image_key = candidate_id.rsplit(":p", 1)[0] if ":p" in candidate_id else ""
                if image_key:
                    negative_image_keys.add(image_key)
                resolved.append({
                    "top10_id": row.get("top10_id"),
                    "candidate_id": candidate_id,
                    "pair_id": row.get("pair_id"),
                    "image_id": row.get("image_id"),
                    "image_key": image_key,
                    "file_name": file_name_from_image_key(image_key),
                    "eligible_for_final": False,
                    "negative_reason": "previous_round_T10_01_T10_02_T10_05_true_image_id",
                    "blind_key_rows_with_same_image_key": sum(1 for br in blind_rows if br.get("image_key") == image_key),
                    "resolution_source": str(final_candidates_path.relative_to(PROJECT_ROOT)),
                })

    summary = {
        "blind_key_path": str(blind_key_path.relative_to(PROJECT_ROOT)) if blind_key_path.exists() else str(blind_key_path),
        "blind_key_rows_read": len(blind_rows),
        "blind_key_contains_top10_ids": any(any(t in jdump(row) for t in target_top10) for row in blind_rows),
        "resolution_note": "blind_key.csv was read for audit; it is anonymized and has no T10 IDs, so true image IDs were resolved from the previous final candidate manifest.",
        "negative_image_keys": sorted(negative_image_keys),
    }

    with (output_dir / "negative_image_ids_from_previous_round.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["top10_id", "candidate_id", "pair_id", "image_id", "image_key", "file_name", "eligible_for_final", "negative_reason", "blind_key_rows_with_same_image_key", "resolution_source"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(resolved)

    return negative_image_keys, resolved, summary


def add_pair_families(pairs: pd.DataFrame, action_names: dict[int, str]) -> pd.DataFrame:
    rows = []
    id_to_family = {verb_id: verb_family(name) for verb_id, name in action_names.items()}
    for row in pairs.itertuples(index=False):
        verb_ids = [int(v) for v in load_json_list(getattr(row, "verb_ids_json"))]
        verb_names = [str(v) for v in load_json_list(getattr(row, "verb_names_json"))]
        families = sorted({id_to_family.get(v, verb_family(name)) for v, name in zip(verb_ids, verb_names)} or {verb_family(v) for v in verb_names})
        rows.append(jdump(families))
    pairs = pairs.copy()
    pairs["verb_families_json"] = rows
    return pairs


def build_image_context(
    image_key: str,
    image_pairs: pd.DataFrame,
    image_instances: pd.DataFrame,
    image_summary: pd.Series,
) -> dict[str, Any]:
    human_signatures: dict[str, dict[str, set[Any]]] = defaultdict(lambda: {"objects": set(), "verb_families": set(), "pair_types": set()})
    human_pair_rows: dict[str, list[pd.Series]] = defaultdict(list)
    pair_type_counter: Counter[tuple[str, str]] = Counter()
    raw_activity_edges = 0
    unique_activity_edges: set[tuple[str, str, str]] = set()

    for _, row in image_pairs.iterrows():
        human_id = str(row.canonical_human_id)
        obj = str(row.object_category_name)
        families = load_json_list(row.verb_families_json)
        if not families:
            families = ["unknown"]
        human_pair_rows[human_id].append(row)
        for family in families:
            pair_type = (obj, str(family))
            human_signatures[human_id]["objects"].add(obj)
            human_signatures[human_id]["verb_families"].add(str(family))
            human_signatures[human_id]["pair_types"].add(pair_type)
            pair_type_counter[pair_type] += 1
            raw_activity_edges += 1
            unique_activity_edges.add((human_id, obj, str(family)))

    effective_humans = sorted(h for h, sig in human_signatures.items() if sig["pair_types"])
    distances_by_human: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_distances: list[float] = []
    for i, left in enumerate(effective_humans):
        for right in effective_humans[i + 1:]:
            dist = signature_distance(human_signatures[left], human_signatures[right])
            payload = {"other_human_id": right, **dist}
            distances_by_human[left].append(payload)
            payload_rev = {"other_human_id": left, **dist}
            distances_by_human[right].append(payload_rev)
            all_distances.append(dist["human_signature_distance"])

    human_avg = {
        human_id: (sum(item["human_signature_distance"] for item in values) / len(values) if values else 0.0)
        for human_id, values in distances_by_human.items()
    }
    human_avg_values = sorted(human_avg.values())
    outlier_cutoff = human_avg_values[int(0.75 * (len(human_avg_values) - 1))] if human_avg_values else 0.0
    total_pair_type_assignments = sum(pair_type_counter.values())
    dominant_pair_type_ratio = (max(pair_type_counter.values()) / total_pair_type_assignments) if total_pair_type_assignments else 0.0
    duplicate_instance_ratio = 1.0 - (len(unique_activity_edges) / raw_activity_edges) if raw_activity_edges else 0.0
    instances_by_id = {str(row.canonical_instance_id): row for row in image_instances.itertuples(index=False)}

    return {
        "image_key": image_key,
        "summary": image_summary,
        "human_signatures": human_signatures,
        "human_pair_rows": human_pair_rows,
        "distances_by_human": distances_by_human,
        "human_avg_distance": human_avg,
        "outlier_cutoff": outlier_cutoff,
        "effective_humans": effective_humans,
        "all_distances": all_distances,
        "distinct_cross_human_pair_types": len(pair_type_counter),
        "dominant_pair_type_ratio": dominant_pair_type_ratio,
        "duplicate_instance_ratio": duplicate_instance_ratio,
        "pair_type_counter": pair_type_counter,
        "instances_by_id": instances_by_id,
    }


def contrast_for_target(context: dict[str, Any], target_human_id: str) -> tuple[float, Optional[str], list[dict[str, Any]]]:
    distances = context["distances_by_human"].get(target_human_id, [])
    if not distances:
        return 0.0, None, []
    distances = sorted(distances, key=lambda item: item["human_signature_distance"], reverse=True)
    return float(distances[0]["human_signature_distance"]), str(distances[0]["other_human_id"]), distances


def target_pair_types(row: pd.Series) -> list[tuple[str, str]]:
    obj = str(row.object_category_name)
    return [(obj, str(family)) for family in load_json_list(row.verb_families_json)]


def build_candidate_row(
    pair_row: pd.Series,
    context: dict[str, Any],
    global_confusion_verbs: dict[str, list[str]],
    threshold: float,
    negative_image_keys: set[str],
) -> dict[str, Any]:
    target_human_id = str(pair_row.canonical_human_id)
    target_object_id = str(pair_row.canonical_object_id)
    target_types = target_pair_types(pair_row)
    target_type_set = set(target_types)
    max_distance, contrast_human, distances = contrast_for_target(context, target_human_id)
    contrast_types: set[tuple[str, str]] = set()
    if contrast_human:
        contrast_types = set(context["human_signatures"][contrast_human]["pair_types"]) - target_type_set
        if not contrast_types:
            contrast_types = set(context["human_signatures"][contrast_human]["pair_types"])

    target_family_set = {family for _, family in target_types}
    target_obj = str(pair_row.object_category_name)
    target_verbs = [str(v) for v in load_json_list(pair_row.verb_names_json)]
    target_families = [str(v) for v in load_json_list(pair_row.verb_families_json)]
    confusion_verbs = [v for v in global_confusion_verbs.get(target_obj, []) if verb_family(v) not in target_family_set]
    confusion_verbs = confusion_verbs[:12]

    human_box = bbox(pair_row, "human")
    object_box = bbox(pair_row, "object")
    pair_union = union_box(human_box, object_box)
    image_area = float(context["summary"].width or 0) * float(context["summary"].height or 0)
    target_union_norm_area = area(pair_union) / image_area if image_area > 0 else 0.0
    object_short_edge = min(max(0.0, object_box[2] - object_box[0]), max(0.0, object_box[3] - object_box[1]))
    target_human_short_edge = min(max(0.0, human_box[2] - human_box[0]), max(0.0, human_box[3] - human_box[1]))
    target_human_area_ratio = area(human_box) / image_area if image_area > 0 else 0.0
    contrast_human_short_edge = 0.0
    if contrast_human and contrast_human in context["instances_by_id"]:
        contrast_human_short_edge = min(
            max(0.0, context["instances_by_id"][contrast_human].bbox_x2 - context["instances_by_id"][contrast_human].bbox_x1),
            max(0.0, context["instances_by_id"][contrast_human].bbox_y2 - context["instances_by_id"][contrast_human].bbox_y1),
        )

    decisive_context: list[dict[str, Any]] = []
    if contrast_human and contrast_types:
        decisive_context.append({
            "type": "semantically_different_second_pair_or_activity",
            "source": "GT_human_activity_signature",
            "human_id": contrast_human,
            "pair_types": [f"{obj}/{family}" for obj, family in sorted(contrast_types)[:6]],
            "distance": round(max_distance, 4),
        })
    for other in context["human_pair_rows"].get(target_human_id, []):
        if str(other.pair_id) == str(pair_row.pair_id):
            continue
        other_types = target_pair_types(other)
        if set(other_types) - target_type_set:
            decisive_context.append({
                "type": "target_human_auxiliary_pair",
                "source": "GT_same_human_other_pair",
                "pair_id": str(other.pair_id),
                "pair_types": [f"{obj}/{family}" for obj, family in other_types],
            })
            if len(decisive_context) >= 5:
                break

    context_source_parts = {"[GT] HICO canonical pairs/signatures"}
    spec = TRACK_C_OBJECT_SPECS.get(target_obj)
    if spec:
        hico_context = []
        for inst_id, inst in context["instances_by_id"].items():
            category = str(getattr(inst, "category_name", ""))
            if inst_id == target_object_id or category not in spec["hico_context"]:
                continue
            inst_box = inst_bbox(inst)
            near_pair = is_near(pair_union, inst_box, factor=1.8)
            hico_context.append({
                "instance_id": inst_id,
                "category": category,
                "source": "[GT] HICO object/person instance",
                "near_target_pair_proxy": near_pair,
            })
        hico_context.sort(key=lambda item: (not item["near_target_pair_proxy"], item["category"], item["instance_id"]))
        if hico_context:
            decisive_context.append({
                "type": "track_c_hico_context_cue_candidates",
                "source": "[GT] HICO object/person instance; [AUTO] near-target geometry proxy",
                "candidates": hico_context[:8],
            })
            context_source_parts.add("[AUTO] geometry-only context proximity proxy")
        manual_context = sorted(spec["manual_context"])
        if manual_context:
            decisive_context.append({
                "type": "track_c_non_hico_context_cue_candidates",
                "source": "[MANUAL_REQUIRED] or [VLM/detected] non-HICO cue; not used as GT",
                "candidates": manual_context,
            })
            context_source_parts.add("[MANUAL_REQUIRED]/[VLM_OR_DETECTOR_REQUIRED] non-HICO context cue names")

    misleading_neighbors: list[dict[str, Any]] = []
    neighbor_conflict_count = 0
    same_object_family_peers = 0
    same_object_category_peers = 0
    same_object_category_same_family_peers = 0
    for other in context["human_pair_rows"].get(target_human_id, []):
        if str(other.pair_id) == str(pair_row.pair_id):
            continue
        if str(other.object_category_name) == target_obj:
            same_object_category_peers += 1
            other_families = {family for _, family in target_pair_types(other)}
            if other_families & target_family_set:
                same_object_family_peers += 1
                same_object_category_same_family_peers += 1
    all_rows = []
    for rows in context["human_pair_rows"].values():
        all_rows.extend(rows)
    for other in all_rows:
        if str(other.pair_id) == str(pair_row.pair_id):
            continue
        other_human = str(other.canonical_human_id)
        other_obj = str(other.object_category_name)
        other_families = {family for _, family in target_pair_types(other)}
        other_object_box = bbox(other, "object")
        near_target = is_near(pair_union, other_object_box, factor=1.35)
        family_conflict = bool(other_families - target_family_set)
        category_overlap = other_obj == target_obj
        if near_target and (family_conflict or category_overlap or other_human != target_human_id):
            neighbor_conflict_count += 1 if family_conflict or other_human != target_human_id else 0
            if len(misleading_neighbors) < 8:
                misleading_neighbors.append({
                    "pair_id": str(other.pair_id),
                    "human_id": other_human,
                    "object_id": str(other.canonical_object_id),
                    "object_category": other_obj,
                    "verb_families": sorted(other_families),
                    "reason": "near_target_pair_with_conflicting_family_or_same_category",
                })

    dominant_types = [ptype for ptype, _ in context["pair_type_counter"].most_common(3)]
    target_is_dominant = any(ptype in target_type_set for ptype in dominant_types)
    scene_mismatch_proxy = clamp((0.6 if not target_is_dominant else 0.0) + 0.4 * max_distance)
    pair_crop_ambiguity = clamp(0.35 * (len(confusion_verbs) / 8.0) + 0.35 * context["duplicate_instance_ratio"] + 0.3 * min(1.0, same_object_category_peers / 4.0))
    context_gain = clamp(0.50 * max_distance + 0.30 * min(1.0, len(decisive_context) / 3.0) + 0.20 * min(1.0, neighbor_conflict_count / 4.0))
    candidate_swap_risk = clamp(0.40 * min(1.0, same_object_family_peers / 3.0) + 0.35 * context["duplicate_instance_ratio"] + 0.25 * min(1.0, same_object_category_peers / 5.0))

    threshold_pass = max_distance >= threshold
    target_avg = context["human_avg_distance"].get(target_human_id, 0.0)
    target_is_outlier = bool(threshold_pass and target_avg >= context["outlier_cutoff"])
    base_eligible = bool(pair_row.eligible_for_final)
    negative_round = str(pair_row.image_key) in negative_image_keys
    eligible = bool(base_eligible and not negative_round and threshold_pass)

    clarity = clamp(0.5 * min(1.0, object_short_edge / 70.0) + 0.5 * (1.0 - min(1.0, target_union_norm_area / 0.85)))
    contrastive_score = (
        35.0 * max_distance
        + 25.0 * context_gain
        + 15.0 * scene_mismatch_proxy
        + 15.0 * pair_crop_ambiguity
        + 10.0 * clarity
        - 12.0 * candidate_swap_risk
    )
    if not eligible:
        contrastive_score -= 20.0

    if threshold_pass and max_distance >= 0.80:
        search_track = "contrastive_cross_human"
    elif threshold_pass and decisive_context:
        search_track = "decisive_context_cue"
    elif threshold_pass and misleading_neighbors:
        search_track = "neighborhood_distractor"
    else:
        search_track = "below_contrast_threshold"

    hard_negative_reason = ""
    if negative_round:
        hard_negative_reason = "previous_round_T10_01_T10_02_T10_05_image_excluded"
    elif not base_eligible:
        hard_negative_reason = "legacy_negative_control_eligible_false"
    elif candidate_swap_risk >= 0.65:
        hard_negative_reason = "high_swap_risk_same_category_or_same_family_neighbors"
    elif misleading_neighbors:
        hard_negative_reason = "scene_or_neighborhood_method_may_import_distractor"

    summary_items = []
    for dist in distances[:4]:
        other_id = str(dist["other_human_id"])
        summary_items.append(f"{other_id}: {summarize_pair_types(context['human_signatures'][other_id]['pair_types'], limit=3)} (d={dist['human_signature_distance']:.2f})")

    return {
        "rank": 0,
        "contrastive_score": round(float(contrastive_score), 6),
        "candidate_id": str(pair_row.pair_id),
        "pair_id": str(pair_row.pair_id),
        "image_key": str(pair_row.image_key),
        "image_id": str(pair_row.image_id),
        "split": str(pair_row.split),
        "file_name": str(pair_row.file_name),
        "image_path": str(context["summary"].image_path),
        "eligible_for_final": eligible,
        "target_human_id": target_human_id,
        "target_object_id": target_object_id,
        "target_object_category": target_obj,
        "target_verbs": jdump(target_verbs),
        "target_verb_families": jdump(target_families),
        "target_human_short_edge_px": round(float(target_human_short_edge), 3),
        "target_human_area_ratio": round(float(target_human_area_ratio), 6),
        "effective_human_count": len(context["effective_humans"]),
        "distinct_cross_human_pair_types": int(context["distinct_cross_human_pair_types"]),
        "dominant_pair_type_ratio": round(float(context["dominant_pair_type_ratio"]), 6),
        "duplicate_instance_ratio": round(float(context["duplicate_instance_ratio"]), 6),
        "max_human_signature_distance": round(float(max_distance), 6),
        "target_is_activity_outlier": target_is_outlier,
        "target_pair_type": jdump([f"{obj}/{family}" for obj, family in target_types]),
        "contrast_human_id": contrast_human or "",
        "contrast_human_short_edge_px": round(float(contrast_human_short_edge), 3),
        "contrast_pair_type": jdump([f"{obj}/{family}" for obj, family in sorted(contrast_types)[:8]]),
        "other_human_activity_summary": " | ".join(summary_items),
        "confusion_verb_set": jdump(confusion_verbs),
        "decisive_context_candidates": jdump(decisive_context),
        "context_source": "; ".join(sorted(context_source_parts | {"[AUTO] geometry-only neighbor and scene-mismatch proxies"})),
        "misleading_neighbor": jdump(misleading_neighbors),
        "neighbor_conflict_count": int(neighbor_conflict_count),
        "scene_mismatch_proxy": round(float(scene_mismatch_proxy), 6),
        "pair_crop_ambiguity": round(float(pair_crop_ambiguity), 6),
        "context_gain": round(float(context_gain), 6),
        "candidate_swap_risk": round(float(candidate_swap_risk), 6),
        "search_track": search_track,
        "hard_negative_reason": hard_negative_reason,
        "hard_negative_rule_hits": "[]",
        "diagnostic_rule_hits": "[]",
        "eligible_after_hard_negative_rules": eligible,
        "diagnostic_only": False,
        "same_object_category_same_family_peer_count": int(same_object_category_same_family_peers),
        "threshold_used": round(float(threshold), 6),
        "threshold_pass": threshold_pass,
        "target_union_norm_area": round(float(target_union_norm_area), 6),
        "object_short_edge_px": round(float(object_short_edge), 3),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def copy_top_images(rows: list[dict[str, Any]], output_dir: Path, limit: int) -> None:
    target_dir = output_dir / "top_candidate_images"
    target_dir.mkdir(parents=True, exist_ok=True)
    for row in rows[:limit]:
        src = Path(row["image_path"])
        if not src.exists():
            continue
        safe_rank = int(row["rank"])
        dst = target_dir / f"rank_{safe_rank:03d}_{row['split']}_{Path(row['file_name']).name}"
        if not dst.exists():
            shutil.copy2(src, dst)


def parse_json_field(row: dict[str, Any], field: str) -> list[Any]:
    try:
        value = row.get(field, "[]")
        decoded = json.loads(value) if isinstance(value, str) else value
    except Exception:
        return []
    return decoded if isinstance(decoded, list) else []


def row_float(row: dict[str, Any], field: str) -> float:
    try:
        return float(row.get(field, 0) or 0)
    except Exception:
        return 0.0


def row_int(row: dict[str, Any], field: str) -> int:
    try:
        return int(float(row.get(field, 0) or 0))
    except Exception:
        return 0


def has_track_c_hico_context(row: dict[str, Any]) -> bool:
    for item in parse_json_field(row, "decisive_context_candidates"):
        if isinstance(item, dict) and item.get("type") == "track_c_hico_context_cue_candidates":
            candidates = item.get("candidates")
            if isinstance(candidates, list) and candidates:
                return True
    return False


def apply_hard_negative_rules(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply explicit v3 hard-negative/diagnostic rules.

    Rules that can be judged from GT/proxy fields remove final eligibility.
    Rules that require visual confirmation are downgraded to diagnostic_only.
    """

    clean: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for source in rows:
        row = dict(source)
        hard_hits: list[str] = []
        diag_hits: list[str] = []
        target_verbs = {str(item) for item in parse_json_field(row, "target_verbs")}
        target_families = {str(item) for item in parse_json_field(row, "target_verb_families")}
        target_types = [str(item) for item in parse_json_field(row, "target_pair_type")]
        contrast_types = [str(item) for item in parse_json_field(row, "contrast_pair_type")]
        target_type_set = set(target_types)
        contrast_type_set = set(contrast_types)
        target_type_objects = {item.split("/", 1)[0] for item in target_types if "/" in item}
        contrast_type_objects = {item.split("/", 1)[0] for item in contrast_types if "/" in item}
        target_type_families = {item.split("/", 1)[1] for item in target_types if "/" in item}
        contrast_type_families = {item.split("/", 1)[1] for item in contrast_types if "/" in item}
        dominant = row_float(row, "dominant_pair_type_ratio")
        max_dist = row_float(row, "max_human_signature_distance")
        context_gain = row_float(row, "context_gain")
        pair_ambiguity = row_float(row, "pair_crop_ambiguity")
        swap_risk = row_float(row, "candidate_swap_risk")
        target_human_short = row_float(row, "target_human_short_edge_px")
        contrast_human_short = row_float(row, "contrast_human_short_edge_px")
        object_short = row_float(row, "object_short_edge_px")
        duplicate_ratio = row_float(row, "duplicate_instance_ratio")
        neighbor_conflicts = row_int(row, "neighbor_conflict_count")
        same_cat_same_family = row_int(row, "same_object_category_same_family_peer_count")
        effective_humans = row_int(row, "effective_human_count")

        if "no_interaction" in target_verbs or "none" in target_families:
            hard_hits.append("target_is_no_interaction")

        if same_cat_same_family >= 3 and duplicate_ratio >= 0.45:
            hard_hits.append("same_person_or_image_has_over_3_same_class_same_family_instances")

        if target_type_set and contrast_type_set and target_type_set == contrast_type_set and dominant >= 0.60:
            hard_hits.append("multiple_people_same_object_category_and_verb_family")

        if dominant >= 0.80 and not bool(row.get("target_is_activity_outlier")):
            hard_hits.append("dominant_pair_type_ratio_ge_0_80_without_clear_minor_activity")

        if dominant > 0.70 and max_dist < 0.80:
            hard_hits.append("dominant_pair_type_ratio_gt_0_70_target_not_obvious_outlier")

        if target_type_objects == contrast_type_objects and target_type_families == {"possession_transport"} and contrast_type_families == {"possession_transport"}:
            hard_hits.append("difference_only_hold_carry_or_possession_synonym")

        if contrast_human_short < 35 or not contrast_types:
            hard_hits.append("second_person_too_small_or_no_confirmed_positive_hoi")

        if context_gain < 0.25 and not has_track_c_hico_context(row):
            hard_hits.append("context_repeated_target_or_common_cooccurrence_no_action_gain")

        if neighbor_conflicts == 0 and swap_risk < 0.20:
            diag_hits.append("noise_lacks_reasonable_baseline_misselection_risk")

        if row_float(row, "scene_mismatch_proxy") >= 0.55 and context_gain < 0.35:
            diag_hits.append("anomalous_scene_only_low_frequency_or_auto_proxy")

        if object_short < 25:
            hard_hits.append("decisive_context_or_target_object_disappears_at_paper_scale")

        if pair_ambiguity < 0.20 and context_gain < 0.35:
            diag_hits.append("pair_internal_pose_already_determines_action_no_extra_context_gain")

        if effective_humans >= 8 and neighbor_conflicts >= 8 and dominant >= 0.60:
            hard_hits.append("overlay_relation_lines_too_dense_to_separate_target_context_contrast")

        if max_dist < 0.65 and context_gain < 0.35:
            hard_hits.append("ranking_supported_only_by_counts_not_contrast_or_context")

        if target_human_short < 35:
            hard_hits.append("target_human_too_small_for_readable_pair")

        if hard_hits:
            row["eligible_after_hard_negative_rules"] = False
        else:
            row["eligible_after_hard_negative_rules"] = bool(row.get("eligible_for_final"))
        row["diagnostic_only"] = bool(diag_hits) or bool(hard_hits)
        row["hard_negative_rule_hits"] = jdump(sorted(set(hard_hits)))
        row["diagnostic_rule_hits"] = jdump(sorted(set(diag_hits)))
        if hard_hits:
            existing = str(row.get("hard_negative_reason", ""))
            joined = ";".join(sorted(set(hard_hits)))
            row["hard_negative_reason"] = f"{existing};{joined}" if existing else joined

        if row["eligible_after_hard_negative_rules"] and not row["diagnostic_only"]:
            clean.append(row)
        elif row["diagnostic_only"]:
            diagnostics.append(row)
        if hard_hits or diag_hits:
            audit.append(row)

    return clean, diagnostics, audit


def track_c_context_strength(row: dict[str, Any]) -> tuple[float, str, str]:
    contexts = parse_json_field(row, "decisive_context_candidates")
    hico_count = 0
    manual_count = 0
    sources: set[str] = set()
    for item in contexts:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source", ""))
        typ = str(item.get("type", ""))
        if "track_c_hico_context" in typ:
            candidates = item.get("candidates")
            if isinstance(candidates, list):
                hico_count += len(candidates)
            sources.add(source)
        if "track_c_non_hico_context" in typ:
            candidates = item.get("candidates")
            if isinstance(candidates, list):
                manual_count += len(candidates)
            sources.add(source)
    strength = clamp(0.65 * min(1.0, hico_count / 3.0) + 0.35 * min(1.0, manual_count / 3.0))
    return strength, "; ".join(sorted(sources)), f"hico_context={hico_count}; manual_or_vlm_required_context={manual_count}"


def contrast_object_family(row: dict[str, Any]) -> tuple[set[str], set[str]]:
    target_types = [str(item) for item in parse_json_field(row, "target_pair_type")]
    contrast_types = [str(item) for item in parse_json_field(row, "contrast_pair_type")]
    target_objects = {item.split("/", 1)[0] for item in target_types if "/" in item}
    target_families = {item.split("/", 1)[1] for item in target_types if "/" in item}
    contrast_objects = {item.split("/", 1)[0] for item in contrast_types if "/" in item}
    contrast_families = {item.split("/", 1)[1] for item in contrast_types if "/" in item}
    return target_objects | contrast_objects, target_families | contrast_families


def first_second_pair(row: dict[str, Any]) -> tuple[str, str]:
    neighbors = parse_json_field(row, "misleading_neighbor")
    contrast_human = str(row.get("contrast_human_id", ""))
    selected: Optional[dict[str, Any]] = None
    for item in neighbors:
        if isinstance(item, dict) and item.get("human_id") == contrast_human:
            selected = item
            break
    if selected is None:
        for item in neighbors:
            if isinstance(item, dict):
                selected = item
                break
    if selected:
        families = selected.get("verb_families") if isinstance(selected.get("verb_families"), list) else []
        pair_type = f"{selected.get('object_category','')}/{','.join(str(x) for x in families)}"
        return str(selected.get("pair_id", "")), pair_type
    contrast_types = parse_json_field(row, "contrast_pair_type")
    return "", ";".join(str(x) for x in contrast_types[:3])


def make_track_row(
    row: dict[str, Any],
    track_id: str,
    track_score: float,
    track_reason: str,
    manual_review_question: str,
    borrowed_evidence_failure_mode: str = "",
) -> dict[str, Any]:
    second_pair_id, second_pair_type = first_second_pair(row)
    out = dict(row)
    out.update({
        "track_id": track_id,
        "track_score": round(float(track_score), 6),
        "track_reason": track_reason,
        "manual_review_question": manual_review_question,
        "second_pair_id": second_pair_id,
        "second_pair_type": second_pair_type,
        "borrowed_evidence_failure_mode": borrowed_evidence_failure_mode,
    })
    return out


def assign_tracks(eligible_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    tracks: dict[str, list[dict[str, Any]]] = {"A": [], "B": [], "C": [], "D": []}
    homogeneous_objects = {"laptop", "bicycle", "motorcycle", "bottle", "cup", "wine glass"}

    for row in eligible_rows:
        target_obj = str(row.get("target_object_category", ""))
        max_dist = row_float(row, "max_human_signature_distance")
        context_gain = row_float(row, "context_gain")
        pair_ambiguity = row_float(row, "pair_crop_ambiguity")
        swap_risk = row_float(row, "candidate_swap_risk")
        scene_mismatch = row_float(row, "scene_mismatch_proxy")
        neighbor_conflicts = row_int(row, "neighbor_conflict_count")
        object_short_edge = row_float(row, "object_short_edge_px")
        human_short_edge = row_float(row, "target_human_short_edge_px")
        contrast_human_short_edge = row_float(row, "contrast_human_short_edge_px")
        target_types = [str(item) for item in parse_json_field(row, "target_pair_type")]
        contrast_types = [str(item) for item in parse_json_field(row, "contrast_pair_type")]
        confusion_verbs = [str(item) for item in parse_json_field(row, "confusion_verb_set")]
        target_verbs = {str(item) for item in parse_json_field(row, "target_verbs")}
        target_families = {str(item) for item in parse_json_field(row, "target_verb_families")}
        contrast_families = {item.split("/", 1)[1] for item in contrast_types if "/" in item}
        positive_target_gt = "no_interaction" not in target_verbs and "none" not in target_families
        positive_contrast_gt = any(family != "none" for family in contrast_families)
        objects, families = contrast_object_family(row)
        same_object_only = len(objects) == 1
        family_diff = len(families) >= 2
        object_diff = len(objects) >= 2
        clear_two_humans = human_short_edge >= 45 and contrast_human_short_edge >= 45
        contrast_pair_present = bool(contrast_types)
        not_background = clear_two_humans and max_dist >= 0.65

        # Track A: cross-human heterogeneous pair.
        track_a_pass = (
            positive_target_gt
            and positive_contrast_gt
            and
            clear_two_humans
            and contrast_pair_present
            and max_dist >= 0.65
            and (object_diff and family_diff or (not object_diff and family_diff and context_gain >= 0.55) or row.get("target_is_activity_outlier") is True)
            and not (same_object_only and target_obj in homogeneous_objects and max_dist < 0.80)
        )
        if track_a_pass:
            score = 100 * (0.45 * max_dist + 0.25 * context_gain + 0.20 * min(1.0, min(human_short_edge, contrast_human_short_edge) / 90.0) + 0.10 * min(1.0, len(objects) / 2.0))
            tracks["A"].append(make_track_row(
                row,
                "A_cross_human_heterogeneous_pair",
                score,
                "Two readable positive-GT human instances have materially different pair/activity signatures; second person is not only a background passerby by size/distance proxy.",
                "Do the two clear people really perform different target-pair activities, rather than the same laptop/bicycle/bottle activity repeated?",
            ))

        # Track B: scene/group activity outlier.
        track_b_pass = (
            positive_target_gt
            and bool(row.get("target_is_activity_outlier"))
            and max_dist >= 0.65
            and scene_mismatch >= 0.55
            and context_gain >= 0.40
            and object_short_edge >= 30
            and (contrast_pair_present or neighbor_conflicts >= 1)
        )
        if track_b_pass:
            score = 100 * (0.40 * scene_mismatch + 0.35 * max_dist + 0.15 * context_gain + 0.10 * min(1.0, object_short_edge / 80.0))
            tracks["B"].append(make_track_row(
                row,
                "B_scene_or_group_activity_outlier",
                score,
                "Target activity differs from dominant other-human activity signatures or from scene/neighborhood co-activity proxies; scene labels are AUTO only.",
                "场景或其他人大多在做什么，目标人物为什么明显不同？目标动作是否仍有清晰局部或第三方证据？",
            ))

        # Track C: action disambiguation via extra context.
        spec = TRACK_C_OBJECT_SPECS.get(target_obj)
        c_context_strength, _context_sources, context_counts = track_c_context_strength(row)
        spec_verbs = spec["verbs"] if spec else set()
        plausible_competing = sorted((set(confusion_verbs) | spec_verbs) - target_verbs)
        track_c_pass = (
            positive_target_gt
            and spec is not None
            and len(spec_verbs) >= 2
            and len(plausible_competing) >= 1
            and pair_ambiguity >= 0.25
            and c_context_strength > 0
            and object_short_edge >= 25
            and human_short_edge >= 35
        )
        if track_c_pass:
            score = 100 * (0.35 * pair_ambiguity + 0.35 * c_context_strength + 0.20 * context_gain + 0.10 * min(1.0, len(plausible_competing) / 5.0))
            reason = (
                f"Target object supports competing prototypes {plausible_competing[:6]}; {context_counts}. "
                "Non-HICO cues are explicitly marked manual/VLM-required, not GT."
            )
            tracks["C"].append(make_track_row(
                row,
                "C_context_dependent_action_disambiguation",
                score,
                reason,
                "遮掉 pair 外 context 后动作 A/B 是否难区分？加入该 cue 后是否更支持 GT 动作？",
            ))

        # Track D: two pairs, two evidence paths, possible evidence borrowing failure.
        track_d_pass = (
            track_a_pass
            and track_c_pass
            and neighbor_conflicts >= 1
            and swap_risk >= 0.25
            and context_gain >= 0.50
            and object_diff
            and family_diff
        )
        if track_d_pass:
            score = 100 * (0.30 * max_dist + 0.25 * c_context_strength + 0.20 * context_gain + 0.15 * min(1.0, neighbor_conflicts / 4.0) + 0.10 * swap_risk)
            tracks["D"].append(make_track_row(
                row,
                "D_two_pairs_two_evidence_paths",
                score,
                "Same image has a clear target pair and a semantically different second pair; target action needs its own context while the second pair supplies a plausible misleading evidence source.",
                "两个 pair 是否各自依赖不同证据？把第二人物证据复制给 target pair 是否会推动错误动作解释？EvidenceHOI 是否能保留 target 自己的证据？",
                "Global/neighborhood method may borrow the second human's object/activity evidence and attach it to the target pair, changing the action interpretation.",
            ))

    for rows in tracks.values():
        rows.sort(key=lambda item: (-row_float(item, "track_score"), item["image_key"], item["pair_id"]))
        for rank, item in enumerate(rows, start=1):
            item["rank"] = rank
    return tracks


def write_track_outputs(tracks: dict[str, list[dict[str, Any]]], output_dir: Path, limit: int = 300) -> None:
    tracks_dir = output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    for track_id, rows in tracks.items():
        limited = rows[:limit]
        write_csv(tracks_dir / f"track_{track_id}.csv", limited, TRACK_COLUMNS)
        with (tracks_dir / f"track_{track_id}.jsonl").open("w", encoding="utf-8") as handle:
            for row in limited:
                handle.write(jdump(row) + "\n")
        copy_track_images(limited, tracks_dir / f"track_{track_id}_images", limit=40)


def copy_track_images(rows: list[dict[str, Any]], output_dir: Path, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    copied = 0
    for row in rows:
        if copied >= limit:
            break
        src = Path(str(row.get("image_path", "")))
        if not src.exists() or str(src) in seen:
            continue
        seen.add(str(src))
        dst = output_dir / f"rank_{int(row['rank']):03d}_{row['track_id']}_{row['split']}_{Path(row['file_name']).name}"
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1


def build_priority_rows(tracks: dict[str, list[dict[str, Any]]], limit: int = 300) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for track_id, rows in tracks.items():
        for row in rows:
            by_pair[str(row["pair_id"])][track_id] = row

    selected: list[tuple[str, dict[str, Any]]] = []
    used: set[str] = set()

    def add_group(group: str, pair_ids: list[str]) -> None:
        for pair_id in pair_ids:
            if pair_id in used or len(selected) >= limit:
                continue
            track_map = by_pair[pair_id]
            preferred = "D" if "D" in track_map else "C" if "C" in track_map else "A" if "A" in track_map else "B"
            row = dict(track_map[preferred])
            row["priority_group"] = group
            row["track_id"] = "+".join(sorted(track_map))
            selected.append((pair_id, row))
            used.add(pair_id)

    add_group("1_track_D", [row["pair_id"] for row in tracks["D"]])

    ac = [pid for pid, tmap in by_pair.items() if "A" in tmap and "C" in tmap and "D" not in tmap]
    ac.sort(key=lambda pid: -(row_float(by_pair[pid]["A"], "track_score") + row_float(by_pair[pid]["C"], "track_score")))
    add_group("2_track_A_plus_C", ac)

    bc = [pid for pid, tmap in by_pair.items() if "B" in tmap and "C" in tmap and "D" not in tmap and pid not in used]
    bc.sort(key=lambda pid: -(row_float(by_pair[pid]["B"], "track_score") + row_float(by_pair[pid]["C"], "track_score")))
    add_group("3_track_B_plus_C", bc)

    c_only = [row["pair_id"] for row in tracks["C"]]
    add_group("4_strong_track_C", c_only)
    a_only = [row["pair_id"] for row in tracks["A"]]
    add_group("5_strong_track_A", a_only)
    b_only = [row["pair_id"] for row in tracks["B"]]
    add_group("6_single_track_B", b_only)

    out: list[dict[str, Any]] = []
    for rank, (_, row) in enumerate(selected, start=1):
        payload = dict(row)
        payload["priority_rank"] = rank
        out.append(payload)
    return out


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    negative_image_keys, negative_rows, negative_summary = parse_previous_negative_images(
        args.blind_key, args.v2_final_candidates, args.output_dir
    )

    action_names = read_action_names(args.actions)
    action_family_rows = build_family_mapping(action_names)
    write_csv(args.output_dir / "action_verb_family_mapping.csv", action_family_rows, ["verb_id", "verb_name", "verb_family"])

    pairs = pd.read_parquet(args.input_dir / "pairs.parquet")
    instances = pd.read_parquet(args.input_dir / "instances.parquet")
    summaries = pd.read_parquet(args.input_dir / "image_summary.parquet")
    pairs = add_pair_families(pairs, action_names)

    global_confusion_verbs: dict[str, list[str]] = defaultdict(list)
    global_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pairs.itertuples(index=False):
        obj = str(row.object_category_name)
        for verb in load_json_list(row.verb_names_json):
            global_counts[obj][str(verb)] += 1
    for obj, counts in global_counts.items():
        global_confusion_verbs[obj] = [verb for verb, _ in counts.most_common()]

    summaries_by_key = {str(row.image_key): row for row in summaries.itertuples(index=False)}
    instances_by_image = {key: df for key, df in instances.groupby("image_key", sort=False)}

    distance_distribution_values: list[float] = []
    image_contexts: dict[str, dict[str, Any]] = {}
    for image_key, image_pairs in pairs.groupby("image_key", sort=False):
        key = str(image_key)
        summary = summaries_by_key.get(key)
        if summary is None:
            continue
        context = build_image_context(key, image_pairs, instances_by_image.get(key, instances.iloc[0:0]), summary)
        image_contexts[key] = context
        distance_distribution_values.extend(context["all_distances"])

    distribution = quantiles(distance_distribution_values)
    threshold = float(args.initial_threshold)
    threshold_reason = "initial_threshold_retained"
    pass_count = sum(1 for value in distance_distribution_values if value >= threshold)
    if pass_count == 0 and distance_distribution_values:
        threshold = float(pd.Series(distance_distribution_values).quantile(0.90))
        threshold_reason = "adjusted_to_p90_because_initial_threshold_recalled_zero_human_pairs"
    elif pass_count < 100 and distance_distribution_values:
        threshold = float(min(args.initial_threshold, pd.Series(distance_distribution_values).quantile(0.85)))
        threshold_reason = "adjusted_to_p85_because_initial_threshold_recalled_too_few_human_pairs"

    candidate_rows: list[dict[str, Any]] = []
    for _, row in pairs.iterrows():
        context = image_contexts.get(str(row.image_key))
        if context is None or len(context["effective_humans"]) < 2:
            continue
        candidate_rows.append(build_candidate_row(row, context, global_confusion_verbs, threshold, negative_image_keys))

    candidate_rows.sort(key=lambda item: (not item["eligible_for_final"], -float(item["contrastive_score"]), item["image_key"], item["pair_id"]))
    for rank, row in enumerate(candidate_rows, start=1):
        row["rank"] = rank

    eligible_rows_pre_rules = [row for row in candidate_rows if row["eligible_for_final"]]
    clean_eligible_rows, diagnostic_rows, hard_negative_audit_rows = apply_hard_negative_rules(eligible_rows_pre_rules)
    clean_eligible_rows.sort(key=lambda item: (-float(item["contrastive_score"]), item["image_key"], item["pair_id"]))
    for rank, row in enumerate(clean_eligible_rows, start=1):
        row["rank"] = rank

    ineligible_previous = [row for row in candidate_rows if row["image_key"] in negative_image_keys]

    write_csv(args.output_dir / "candidates_all_scored.csv", candidate_rows, CSV_COLUMNS)
    write_csv(args.output_dir / "top_candidates.csv", clean_eligible_rows[: args.top_k], CSV_COLUMNS)
    write_csv(args.output_dir / "diagnostic_candidates.csv", diagnostic_rows, CSV_COLUMNS)
    write_csv(args.output_dir / "hard_negative_audit.csv", hard_negative_audit_rows, CSV_COLUMNS)
    write_csv(args.output_dir / "previous_round_negative_rows_scored.csv", ineligible_previous, CSV_COLUMNS)
    with (args.output_dir / "top_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in clean_eligible_rows[: args.top_k]:
            handle.write(jdump(row) + "\n")

    copy_top_images(clean_eligible_rows, args.output_dir, args.copy_top_images)
    tracks = assign_tracks(clean_eligible_rows)
    write_track_outputs(tracks, args.output_dir, limit=max(args.top_k, 300))
    priority_rows = build_priority_rows(tracks, limit=max(args.top_k, 300))
    write_csv(args.output_dir / "priority_candidates.csv", priority_rows, PRIORITY_COLUMNS)
    with (args.output_dir / "priority_candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in priority_rows:
            handle.write(jdump(row) + "\n")
    copy_track_images(priority_rows, args.output_dir / "priority_candidate_images", limit=args.copy_top_images)

    summary = {
        "schema_version": "figure1_search_v3_contrastive.summary",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "input_dir": str(args.input_dir.relative_to(PROJECT_ROOT)),
        "output_dir": str(args.output_dir.relative_to(PROJECT_ROOT)),
        "candidate_unit": "target canonical human-object pair",
        "primary_retrieval_goal": "cross-human activity-signature contrast plus decisive context/distractor cues; not raw human/object/HOI counts",
        "action_mapping": str(args.actions.relative_to(PROJECT_ROOT) if args.actions.is_relative_to(PROJECT_ROOT) else args.actions),
        "verb_family_note": "hold/carry/pick_up/lift and other visually close possession/transport verbs are collapsed before signature distance is computed.",
        "initial_human_signature_threshold": args.initial_threshold,
        "threshold_used": threshold,
        "threshold_adjustment_reason": threshold_reason,
        "human_signature_distance_distribution": distribution,
        "human_pair_distance_count": len(distance_distribution_values),
        "human_pair_distance_pass_count_at_initial_threshold": pass_count,
        "candidate_rows_scored": len(candidate_rows),
        "eligible_rows_after_negative_and_threshold_filters": len(eligible_rows_pre_rules),
        "clean_eligible_rows_after_hard_negative_rules": len(clean_eligible_rows),
        "diagnostic_or_hard_negative_rows": len(diagnostic_rows),
        "hard_negative_audit_rows": len(hard_negative_audit_rows),
        "previous_round_negative_rows_scored": len(ineligible_previous),
        "track_counts": {track_id: len(rows) for track_id, rows in tracks.items()},
        "priority_candidate_count": len(priority_rows),
        "priority_order": ["Track D", "A+C hybrid", "B+C hybrid", "strong Track C", "strong Track A", "single Track B"],
        "negative_resolution": negative_summary,
        "negative_rows": negative_rows,
        "outputs": {
            "top_candidates_csv": str((args.output_dir / "top_candidates.csv").relative_to(PROJECT_ROOT)),
            "top_candidates_jsonl": str((args.output_dir / "top_candidates.jsonl").relative_to(PROJECT_ROOT)),
            "all_scored_csv": str((args.output_dir / "candidates_all_scored.csv").relative_to(PROJECT_ROOT)),
            "diagnostic_candidates_csv": str((args.output_dir / "diagnostic_candidates.csv").relative_to(PROJECT_ROOT)),
            "hard_negative_audit_csv": str((args.output_dir / "hard_negative_audit.csv").relative_to(PROJECT_ROOT)),
            "previous_round_negative_rows_scored": str((args.output_dir / "previous_round_negative_rows_scored.csv").relative_to(PROJECT_ROOT)),
            "negative_image_ids": str((args.output_dir / "negative_image_ids_from_previous_round.csv").relative_to(PROJECT_ROOT)),
            "verb_family_mapping": str((args.output_dir / "action_verb_family_mapping.csv").relative_to(PROJECT_ROOT)),
            "top_candidate_images": str((args.output_dir / "top_candidate_images").relative_to(PROJECT_ROOT)),
            "track_outputs": str((args.output_dir / "tracks").relative_to(PROJECT_ROOT)),
            "priority_candidates_csv": str((args.output_dir / "priority_candidates.csv").relative_to(PROJECT_ROOT)),
        },
    }
    (args.output_dir / "search_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Figure 1 Search v3 Contrastive",
        "",
        f"Generated at: {summary['generated_at']}",
        "",
        "## Candidate Definition",
        "",
        "The retrieval unit is a target canonical human-object pair. Each human receives an activity signature S_h = {(object_category, verb_family)}. Duplicate same-class instances for the same person and synonym-level GT verbs on the same physical pair do not add activity types.",
        "",
        "## Threshold",
        "",
        f"Initial human_signature_distance threshold: {args.initial_threshold:.3f}.",
        f"Threshold used: {threshold:.3f} ({threshold_reason}).",
        f"Distribution: {json.dumps(distribution, ensure_ascii=False)}.",
        "",
        "## Previous-Round Negatives",
        "",
        "T10_01, T10_02, and T10_05 true image IDs were marked eligible_for_final=false in v3 outputs.",
        "",
    ]
    for row in negative_rows:
        report_lines.append(f"- {row['top10_id']}: `{row['image_key']}` / `{row['candidate_id']}`")
    report_lines.extend(["", "## Top Eligible Candidates", ""])
    for row in clean_eligible_rows[:20]:
        report_lines.append(
            f"- rank {row['rank']:03d}: `{row['candidate_id']}` score={row['contrastive_score']:.3f} "
            f"target={row['target_pair_type']} contrast={row['contrast_pair_type']} "
            f"max_d={row['max_human_signature_distance']:.3f} track={row['search_track']}"
        )
    report_lines.extend(["", "## Parallel Track Counts", ""])
    for track_id in ["D", "A", "C", "B"]:
        report_lines.append(f"- Track {track_id}: {len(tracks[track_id])} candidates")
    report_lines.extend(["", "## Priority Candidates", ""])
    for row in priority_rows[:20]:
        report_lines.append(
            f"- priority {row['priority_rank']:03d}: group={row['priority_group']} tracks={row['track_id']} "
            f"`{row['candidate_id']}` track_score={row['track_score']:.3f} reason={row['track_reason']}"
        )
    (args.output_dir / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "threshold_used": threshold,
        "threshold_reason": threshold_reason,
        "eligible_top_candidates_before_hard_negative_rules": len(eligible_rows_pre_rules),
        "clean_eligible_top_candidates": len(clean_eligible_rows),
        "diagnostic_or_hard_negative_rows": len(diagnostic_rows),
        "track_counts": {track_id: len(rows) for track_id, rows in tracks.items()},
        "priority_candidate_count": len(priority_rows),
        "negative_image_keys": sorted(negative_image_keys),
        "top_candidates_csv": str(args.output_dir / "top_candidates.csv"),
        "priority_candidates_csv": str(args.output_dir / "priority_candidates.csv"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
