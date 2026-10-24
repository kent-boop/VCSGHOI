#!/usr/bin/env python3
"""Figure 1 v4 structure-first pair gate search.

This run is intentionally a post-index filtering pass. It reads the already
audited v2 canonical HICO-DET index and writes a new output directory only.
It does not modify the GT parser, canonicalization, or overlay modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd
from PIL import Image, ImageDraw


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from search_figure1_v3_contrastive import (  # noqa: E402
    TRACK_C_OBJECT_SPECS,
    add_pair_families,
    read_action_names,
    verb_family,
)


DEFAULT_INPUT_DIR = PROJECT_ROOT / "outputs/figure1_search_v2/index"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/figure1_search_v4_pair_gate"
DEFAULT_ACTIONS = PROJECT_ROOT / "hico_20160224_det/hico_list_vb.txt"
DEFAULT_V2_TOP10 = PROJECT_ROOT / "outputs/figure1_search_v2/final/candidate_images.csv"

PANEL_HUMAN_SHORT_EDGE = 35.0
CONTACT_SHEET_LIMIT = 48


@dataclass(frozen=True)
class GateConfig:
    name: str
    signature_distance: float
    dominant_activity_ratio_max: float
    target_object_short_edge: float
    contrast_object_short_edge: float


CONFIGS = [
    GateConfig("recall", 0.50, 0.75, 32.0, 24.0),
    GateConfig("balanced", 0.60, 0.67, 40.0, 32.0),
    GateConfig("strict", 0.70, 0.60, 48.0, 40.0),
]


GATE_ORDER = [
    "legacy_or_invalid_pair",
    "target_positive_gt_missing",
    "target_human_not_readable",
    "target_object_not_readable",
    "effective_human_count_lt_2",
    "cross_human_effective_pair_count_lt_2",
    "distinct_activity_signature_count_lt_2",
    "dominant_activity_ratio_too_high",
    "multi_person_same_activity_family",
    "insufficient_readable_contrast_pair",
    "nontrivial_contrast_axis_count_lt_2",
    "activity_signature_distance_below_threshold",
    "hold_carry_only_contrast",
    "bicycle_motorcycle_same_ride_activity",
    "laptop_same_use_activity",
    "duplicate_same_physical_or_repeated_instance",
    "pair_only_crop_already_unambiguous_proxy",
    "pair_external_evidence_not_confirmed",
    "no_scene_or_neighborhood_failure_mode",
    "answerability_gate_failed",
]

MAIN_COLUMNS = [
    "config",
    "rank",
    "pool",
    "score",
    "first_failed_gate",
    "candidate_id",
    "pair_id",
    "image_key",
    "image_id",
    "split",
    "file_name",
    "image_path",
    "target_human_id",
    "target_object_id",
    "target_object_category",
    "target_verbs",
    "target_verb_families",
    "target_activity_signatures",
    "target_human_short_edge_px",
    "target_object_short_edge_px",
    "contrast_pair_id",
    "contrast_human_id",
    "contrast_object_id",
    "contrast_object_category",
    "contrast_verbs",
    "contrast_verb_families",
    "contrast_activity_signatures",
    "contrast_human_short_edge_px",
    "contrast_object_short_edge_px",
    "effective_human_count",
    "cross_human_effective_pair_count",
    "distinct_activity_signature_count",
    "dominant_activity_ratio",
    "nontrivial_contrast_axis_count",
    "activity_signature_distance",
    "pair_external_evidence_confirmed",
    "both_pairs_readable_in_full_scene",
    "second_pair_role",
    "candidate_swap_risk_proxy",
    "scene_level_error_proxy",
    "neighborhood_error_proxy",
    "target_evidence_needed",
    "borrowed_evidence_error_answer",
    "main_required_answer",
    "single_pair_exception_answer",
    "concrete_context_cue",
    "object_function_group",
    "verb_family_primary",
    "scene_family",
    "is_single_person_single_object_reject",
    "is_false_multi_background_reject",
    "is_homogeneous_multi_reject",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--v2-top10", type=Path, default=DEFAULT_V2_TOP10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=390)
    parser.add_argument("--thumb-height", type=int, default=310)
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


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def bbox_from_row(row: Any, prefix: str) -> tuple[float, float, float, float]:
    return (
        float(getattr(row, f"{prefix}_bbox_x1")),
        float(getattr(row, f"{prefix}_bbox_y1")),
        float(getattr(row, f"{prefix}_bbox_x2")),
        float(getattr(row, f"{prefix}_bbox_y2")),
    )


def bbox_from_instance(inst: Any) -> tuple[float, float, float, float]:
    return (float(inst.bbox_x1), float(inst.bbox_y1), float(inst.bbox_x2), float(inst.bbox_y2))


def short_edge(box: Sequence[float]) -> float:
    return min(max(0.0, float(box[2]) - float(box[0])), max(0.0, float(box[3]) - float(box[1])))


def area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def union_box(boxes: Iterable[Sequence[float]]) -> tuple[float, float, float, float]:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def center(box: Sequence[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def diag(box: Sequence[float]) -> float:
    return math.hypot(max(0.0, float(box[2]) - float(box[0])), max(0.0, float(box[3]) - float(box[1])))


def near(a: Sequence[float], b: Sequence[float], factor: float = 1.45) -> bool:
    ac = center(a)
    bc = center(b)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1]) <= factor * max(diag(a), diag(b), 1.0)


def object_function_group(obj: str) -> str:
    o = obj.strip().lower()
    groups = {
        "two_wheel_vehicle": {"bicycle", "motorcycle"},
        "road_vehicle": {"car", "bus", "truck", "train"},
        "ride_animal": {"horse", "elephant", "zebra", "giraffe", "cow"},
        "food": {"pizza", "cake", "sandwich", "hot dog", "apple", "banana", "orange", "broccoli", "carrot", "donut"},
        "drink_container": {"bottle", "cup", "wine glass"},
        "tableware_tool": {"fork", "knife", "spoon", "bowl"},
        "device_screen": {"laptop", "tv", "cell phone", "remote", "keyboard", "mouse"},
        "seat_furniture": {"chair", "bench", "couch", "bed", "dining table"},
        "bag_container": {"backpack", "handbag", "suitcase"},
        "sports_equipment": {"sports ball", "baseball bat", "baseball glove", "skateboard", "skis", "snowboard", "surfboard", "tennis racket", "kite"},
        "wearable": {"tie", "umbrella"},
        "animal_pet": {"dog", "cat", "bird", "sheep", "cow", "horse", "elephant", "bear", "zebra", "giraffe"},
        "tool_surface": {"sink", "toilet", "toothbrush", "hair drier"},
    }
    for group, values in groups.items():
        if o in values:
            return group
    return f"object:{o}"


def evidence_template(family: str, obj_group: str) -> str:
    if family == "ride_mount_vehicle" and obj_group in {"two_wheel_vehicle", "road_vehicle"}:
        return "mounted_body_pose_plus_vehicle_state_and_road"
    if family == "ride_mount_vehicle" and obj_group == "seat_furniture":
        return "seated_body_pose_plus_support_surface"
    if family == "tool_device_operate" and obj_group == "device_screen":
        return "gaze_or_hand_contact_plus_active_device_surface"
    if family == "consume_food_drink" and obj_group in {"food", "drink_container", "tableware_tool"}:
        return "mouth_hand_utensil_food_layout"
    if family == "possession_transport":
        return "hand_contact_or_load_carry_pose"
    if family == "repair_make_modify":
        return "tool_or_work_pose_plus_modified_object_state"
    if family == "person_animal_social":
        return "body_orientation_contact_with_living_target"
    if family == "sports_play":
        return "sport_pose_equipment_motion_context"
    if family == "body_posture_contact":
        return "body_pose_spatial_contact_relation"
    return f"{family}_evidence"


def scene_family_from_objects(objects: Iterable[str]) -> str:
    groups = {object_function_group(obj) for obj in objects}
    if groups & {"two_wheel_vehicle", "road_vehicle"}:
        return "street_vehicle"
    if groups & {"food", "drink_container", "tableware_tool"}:
        return "food_dining"
    if groups & {"device_screen"}:
        return "indoor_device_media"
    if groups & {"seat_furniture"}:
        return "indoor_furniture"
    if groups & {"sports_equipment"}:
        return "sports_outdoor"
    if groups & {"animal_pet", "ride_animal"}:
        return "animal_interaction"
    if groups & {"bag_container", "wearable"}:
        return "personal_item_travel"
    return "mixed_other"


def positive_verbs(row: Any) -> list[str]:
    return [str(v) for v in load_json_list(getattr(row, "verb_names_json", "[]")) if str(v) != "no_interaction"]


def positive_families(row: Any) -> list[str]:
    names = positive_verbs(row)
    families = [str(f) for f in load_json_list(getattr(row, "verb_families_json", "[]")) if str(f) != "none"]
    if not families:
        families = [verb_family(v) for v in names]
    return sorted({f for f in families if f != "none"})


def activity_signatures_for(row: Any) -> list[tuple[str, str, str]]:
    obj_group = object_function_group(str(getattr(row, "object_category_name")))
    out = []
    for family in positive_families(row):
        out.append((family, obj_group, evidence_template(family, obj_group)))
    return sorted(set(out))


def signature_label(sig: tuple[str, str, str]) -> str:
    return "/".join(sig)


def signature_distance(a: tuple[str, str, str], b: tuple[str, str, str]) -> tuple[float, int]:
    axes = int(a[0] != b[0]) + int(a[1] != b[1]) + int(a[2] != b[2])
    distance = 0.40 * int(a[0] != b[0]) + 0.25 * int(a[1] != b[1]) + 0.35 * int(a[2] != b[2])
    return float(distance), axes


def best_signature_distance(left: list[tuple[str, str, str]], right: list[tuple[str, str, str]]) -> tuple[float, int, Optional[tuple[str, str, str]], Optional[tuple[str, str, str]]]:
    best = (0.0, 0, None, None)
    for a in left:
        for b in right:
            dist, axes = signature_distance(a, b)
            if (dist, axes) > (best[0], best[1]):
                best = (dist, axes, a, b)
    return best


def first_failed(hits: dict[str, bool]) -> str:
    for gate in GATE_ORDER:
        if not hits.get(gate, False):
            return gate
    return "pass"


def first_context_cue(row: Any, inst_map: dict[str, Any], pair_union: Sequence[float]) -> tuple[str, str, Optional[tuple[float, float, float, float]]]:
    spec = TRACK_C_OBJECT_SPECS.get(str(getattr(row, "object_category_name")))
    if not spec:
        return "", "", None
    candidates = []
    for inst_id, inst in inst_map.items():
        category = str(getattr(inst, "category_name", ""))
        if inst_id in {str(getattr(row, "canonical_human_id")), str(getattr(row, "canonical_object_id"))}:
            continue
        if category not in spec["hico_context"]:
            continue
        box = bbox_from_instance(inst)
        candidates.append((not near(pair_union, box, factor=1.8), category, inst_id, box))
    candidates.sort()
    if candidates:
        _, category, inst_id, box = candidates[0]
        return category, inst_id, box
    manual = sorted(spec.get("manual_context", []))
    if manual:
        return f"manual_or_vlm:{manual[0]}", "", None
    return "", "", None


def competing_actions(row: Any, global_confusion: dict[str, list[str]]) -> list[str]:
    target = set(positive_verbs(row))
    target_families = set(positive_families(row))
    obj = str(getattr(row, "object_category_name"))
    out = []
    for verb in global_confusion.get(obj, []):
        if verb in target or verb == "no_interaction":
            continue
        if verb_family(verb) in target_families:
            continue
        out.append(verb)
    spec = TRACK_C_OBJECT_SPECS.get(obj)
    if spec:
        for verb in sorted(spec["verbs"]):
            if verb not in target and verb_family(verb) not in target_families and verb not in out:
                out.append(verb)
    return out[:10]


def pair_ambiguity_proxy(row: Any, global_confusion: dict[str, list[str]], duplicate_ratio: float) -> float:
    conf = competing_actions(row, global_confusion)
    spec_bonus = 0.25 if str(getattr(row, "object_category_name")) in TRACK_C_OBJECT_SPECS else 0.0
    return min(1.0, 0.18 * len(conf) + spec_bonus + 0.25 * duplicate_ratio)


def build_image_context(
    image_key: str,
    image_pairs: pd.DataFrame,
    image_instances: pd.DataFrame,
    summary: Any,
    cfg: GateConfig,
) -> dict[str, Any]:
    instances_by_id = {str(row.canonical_instance_id): row for row in image_instances.itertuples(index=False)}
    human_rows = {iid: inst for iid, inst in instances_by_id.items() if str(getattr(inst, "role", "")) == "human"}
    object_rows = {iid: inst for iid, inst in instances_by_id.items() if str(getattr(inst, "role", "")) == "object"}
    effective_pairs = []
    effective_by_human: dict[str, list[dict[str, Any]]] = defaultdict(list)
    activity_counter: Counter[tuple[str, str, str]] = Counter()
    family_counter: Counter[str] = Counter()
    physical_seen: set[tuple[str, str, tuple[str, str, str]]] = set()
    duplicate_records = 0

    for row in image_pairs.itertuples(index=False):
        if not bool(getattr(row, "eligible_for_final", False)) or not bool(getattr(row, "valid", True)) or bool(getattr(row, "invisible", False)):
            continue
        verbs = positive_verbs(row)
        sigs = activity_signatures_for(row)
        if not verbs or not sigs:
            continue
        human_box = bbox_from_row(row, "human")
        object_box = bbox_from_row(row, "object")
        h_short = short_edge(human_box)
        o_short = short_edge(object_box)
        readable_effective = h_short >= PANEL_HUMAN_SHORT_EDGE and o_short >= cfg.contrast_object_short_edge
        if not readable_effective:
            continue
        unique_sigs = []
        for sig in sigs:
            physical_key = (str(row.canonical_human_id), str(row.canonical_object_id), sig)
            if physical_key in physical_seen:
                duplicate_records += 1
                continue
            physical_seen.add(physical_key)
            unique_sigs.append(sig)
            activity_counter[sig] += 1
            family_counter[sig[0]] += 1
        if not unique_sigs:
            continue
        record = {
            "row": row,
            "human_id": str(row.canonical_human_id),
            "object_id": str(row.canonical_object_id),
            "object_category": str(row.object_category_name),
            "verbs": verbs,
            "families": positive_families(row),
            "signatures": unique_sigs,
            "human_box": human_box,
            "object_box": object_box,
            "pair_union": union_box([human_box, object_box]),
            "human_short": h_short,
            "object_short": o_short,
        }
        effective_pairs.append(record)
        effective_by_human[record["human_id"]].append(record)

    total_activity = sum(activity_counter.values())
    dominant_ratio = max(activity_counter.values()) / total_activity if total_activity else 0.0
    duplicate_ratio = duplicate_records / max(1, duplicate_records + len(physical_seen))
    objects = [str(row.object_category_name) for row in image_pairs.itertuples(index=False)]
    return {
        "image_key": image_key,
        "summary": summary,
        "instances_by_id": instances_by_id,
        "human_rows": human_rows,
        "object_rows": object_rows,
        "effective_pairs": effective_pairs,
        "effective_by_human": effective_by_human,
        "effective_humans": sorted(effective_by_human),
        "activity_counter": activity_counter,
        "family_counter": family_counter,
        "dominant_activity_ratio": dominant_ratio,
        "duplicate_ratio": duplicate_ratio,
        "scene_family": scene_family_from_objects(objects),
        "canonical_human_count": int(getattr(summary, "canonical_human_count", len(human_rows))),
        "canonical_object_count": int(getattr(summary, "canonical_object_count", len(object_rows))),
    }


def choose_contrast(target: Any, target_sigs: list[tuple[str, str, str]], context: dict[str, Any]) -> tuple[Optional[dict[str, Any]], float, int, Optional[tuple[str, str, str]], Optional[tuple[str, str, str]]]:
    target_human = str(getattr(target, "canonical_human_id"))
    best_record: Optional[dict[str, Any]] = None
    best_dist = 0.0
    best_axes = 0
    best_target_sig = None
    best_contrast_sig = None
    for rec in context["effective_pairs"]:
        if rec["human_id"] == target_human:
            continue
        dist, axes, left_sig, right_sig = best_signature_distance(target_sigs, rec["signatures"])
        if (dist, axes, rec["object_short"], rec["human_short"]) > (best_dist, best_axes, 0 if best_record is None else best_record["object_short"], 0 if best_record is None else best_record["human_short"]):
            best_record = rec
            best_dist = dist
            best_axes = axes
            best_target_sig = left_sig
            best_contrast_sig = right_sig
    return best_record, best_dist, best_axes, best_target_sig, best_contrast_sig


def second_pair_roles(target: Any, target_union: Sequence[float], contrast: Optional[dict[str, Any]], target_sig: Optional[tuple[str, str, str]], contrast_sig: Optional[tuple[str, str, str]]) -> tuple[list[str], float, float, float]:
    if contrast is None or target_sig is None or contrast_sig is None:
        return [], 0.0, 0.0, 0.0
    roles = []
    different_path = target_sig[2] != contrast_sig[2] or target_sig[0] != contrast_sig[0]
    close = near(target_union, contrast["pair_union"], factor=1.55) or near(target_union, contrast["object_box"], factor=1.35)
    same_group = target_sig[1] == contrast_sig[1]
    same_family = target_sig[0] == contrast_sig[0]
    swap_risk = min(1.0, 0.45 * same_group + 0.35 * close + 0.20 * same_family)
    scene_error = min(1.0, 0.55 * int(different_path) + 0.25 * same_group + 0.20 * int(not close))
    neighbor_error = min(1.0, 0.70 * close + 0.30 * same_group)
    if different_path:
        roles.append("different_evidence_path")
    if swap_risk >= 0.35:
        roles.append("candidate_swap_risk")
    if scene_error >= 0.45:
        roles.append("scene_level_global_propagation_error")
    if neighbor_error >= 0.40:
        roles.append("neighborhood_wrong_retention")
    return roles, swap_risk, scene_error, neighbor_error


def is_bicycle_motorcycle_same_ride(target_sig: Optional[tuple[str, str, str]], contrast_sig: Optional[tuple[str, str, str]]) -> bool:
    return bool(target_sig and contrast_sig and target_sig[0] == "ride_mount_vehicle" and contrast_sig[0] == "ride_mount_vehicle" and target_sig[1] == "two_wheel_vehicle" and contrast_sig[1] == "two_wheel_vehicle" and target_sig[2] == contrast_sig[2])


def is_laptop_same_use(target_obj: str, contrast_obj: str, target_sig: Optional[tuple[str, str, str]], contrast_sig: Optional[tuple[str, str, str]]) -> bool:
    return target_obj == "laptop" and contrast_obj == "laptop" and bool(target_sig and contrast_sig and target_sig[0] == contrast_sig[0])


def answer_main_question(target: Any, contrast: dict[str, Any], roles: list[str], target_sig: tuple[str, str, str], contrast_sig: tuple[str, str, str]) -> tuple[str, str, str]:
    target_verbs = ",".join(positive_verbs(target))
    contrast_verbs = ",".join(contrast["verbs"])
    wrong_action = contrast_verbs or contrast_sig[0]
    target_need = target_sig[2]
    role_text = ",".join(roles)
    borrowed = (
        f"If the contrast person's {contrast['object_category']} evidence path ({contrast_sig[2]}) is propagated to the target, "
        f"it can incorrectly support {wrong_action} for the target pair."
    )
    answer = (
        f"Target: {target.canonical_human_id} interacts with {target.object_category_name} via {target_verbs}. "
        f"Other: {contrast['human_id']} interacts with {contrast['object_category']} via {contrast_verbs}. "
        f"Wrong propagation: {borrowed} "
        f"Target needs its own evidence template: {target_need}. Roles: {role_text}."
    )
    return answer, borrowed, target_need


def answer_single_exception(target: Any, cue: str, confusions: list[str]) -> str:
    target_verbs = ",".join(positive_verbs(target))
    confused = ",".join(confusions[:3])
    return (
        f"Pair-only target {target.object_category_name} can be confused with {confused}. "
        f"Masking cue {cue} removes the deciding evidence, making the action less separable. "
        f"Adding {cue} better supports the GT action {target_verbs}."
    )


def classify_pair(
    row: Any,
    context: dict[str, Any],
    cfg: GateConfig,
    global_confusion: dict[str, list[str]],
) -> dict[str, Any]:
    target_box = bbox_from_row(row, "object")
    human_box = bbox_from_row(row, "human")
    pair_union = union_box([human_box, target_box])
    target_sigs = activity_signatures_for(row)
    target_verbs = positive_verbs(row)
    target_fams = positive_families(row)
    confusions = competing_actions(row, global_confusion)
    ambiguity = pair_ambiguity_proxy(row, global_confusion, context["duplicate_ratio"])
    contrast, dist, axes, target_sig, contrast_sig = choose_contrast(row, target_sigs, context)
    cue_category, cue_id, cue_box = first_context_cue(row, context["instances_by_id"], pair_union)
    cue_label = f"{cue_category}:{cue_id}" if cue_id else cue_category
    roles, swap_risk, scene_error, neighbor_error = second_pair_roles(row, pair_union, contrast, target_sig, contrast_sig)
    effective_human_count = len(context["effective_humans"])
    cross_human_effective_pair_count = len(context["effective_pairs"]) if effective_human_count >= 2 else 0
    distinct_activity_signature_count = len(context["activity_counter"])
    target_h_short = short_edge(human_box)
    target_o_short = short_edge(target_box)
    contrast_h_short = contrast["human_short"] if contrast else 0.0
    contrast_o_short = contrast["object_short"] if contrast else 0.0
    target_obj = str(row.object_category_name)
    contrast_obj = contrast["object_category"] if contrast else ""
    dominant = float(context["dominant_activity_ratio"])
    all_families = set(context["family_counter"])
    hold_carry_only = bool(target_sig and contrast_sig and target_sig[0] == contrast_sig[0] == "possession_transport")
    bimoto_same_ride = is_bicycle_motorcycle_same_ride(target_sig, contrast_sig)
    laptop_same_use = is_laptop_same_use(target_obj, contrast_obj, target_sig, contrast_sig)
    repeated_same_physical = context["duplicate_ratio"] > 0.0 and str(getattr(row, "shared_object_status", "False")) == "True" and str(getattr(row, "same_class_multi_instance_status", "False")) == "True"
    pair_external_evidence_confirmed = bool(contrast and roles and target_sig and contrast_sig)
    both_pairs_readable = target_h_short >= PANEL_HUMAN_SHORT_EDGE and target_o_short >= cfg.target_object_short_edge and contrast_h_short >= PANEL_HUMAN_SHORT_EDGE and contrast_o_short >= cfg.contrast_object_short_edge
    answerable = bool(contrast and target_sig and contrast_sig and roles)
    not_pair_only_obvious = ambiguity >= 0.25 or len(confusions) >= 2 or bool(cue_label)
    first_answer = ""
    borrowed = ""
    target_need = ""
    if answerable and contrast is not None and target_sig is not None and contrast_sig is not None:
        first_answer, borrowed, target_need = answer_main_question(row, contrast, roles, target_sig, contrast_sig)

    gates = {
        "legacy_or_invalid_pair": bool(getattr(row, "eligible_for_final", False)) and bool(getattr(row, "valid", True)) and not bool(getattr(row, "invisible", False)),
        "target_positive_gt_missing": bool(target_verbs and target_sigs),
        "target_human_not_readable": target_h_short >= PANEL_HUMAN_SHORT_EDGE,
        "target_object_not_readable": target_o_short >= cfg.target_object_short_edge,
        "effective_human_count_lt_2": effective_human_count >= 2,
        "cross_human_effective_pair_count_lt_2": cross_human_effective_pair_count >= 2,
        "distinct_activity_signature_count_lt_2": distinct_activity_signature_count >= 2,
        "dominant_activity_ratio_too_high": dominant <= cfg.dominant_activity_ratio_max,
        "multi_person_same_activity_family": len(all_families) >= 2,
        "insufficient_readable_contrast_pair": bool(contrast and both_pairs_readable),
        "nontrivial_contrast_axis_count_lt_2": axes >= 2,
        "activity_signature_distance_below_threshold": dist >= cfg.signature_distance,
        "hold_carry_only_contrast": not hold_carry_only,
        "bicycle_motorcycle_same_ride_activity": not bimoto_same_ride,
        "laptop_same_use_activity": not laptop_same_use,
        "duplicate_same_physical_or_repeated_instance": not repeated_same_physical,
        "pair_only_crop_already_unambiguous_proxy": not_pair_only_obvious,
        "pair_external_evidence_not_confirmed": pair_external_evidence_confirmed,
        "no_scene_or_neighborhood_failure_mode": bool(roles),
        "answerability_gate_failed": bool(first_answer),
    }
    fail = first_failed(gates)

    single_primary_pair = effective_human_count <= 1 and cross_human_effective_pair_count < 2
    exception_gates = {
        "legacy_or_invalid_pair": gates["legacy_or_invalid_pair"],
        "target_positive_gt_missing": gates["target_positive_gt_missing"],
        "target_human_not_readable": gates["target_human_not_readable"],
        "target_object_not_readable": gates["target_object_not_readable"],
        "single_primary_pair_structure": single_primary_pair,
        "target_object_two_competing_actions": len(confusions) >= 2,
        "pair_only_crop_true_ambiguity_proxy": ambiguity >= 0.35,
        "concrete_pair_external_cue": bool(cue_label),
        "masked_cue_more_ambiguous_proxy": bool(cue_label) and ambiguity >= 0.35,
        "cue_supports_gt_proxy": bool(cue_label) and target_obj in TRACK_C_OBJECT_SPECS,
        "cue_visible_in_final_panel_proxy": cue_box is None or short_edge(cue_box) >= cfg.contrast_object_short_edge,
        "not_multi_person_main_candidate": fail != "pass",
    }
    exception_fail = next((name for name, ok in exception_gates.items() if not ok), "pass")
    exception_answer = answer_single_exception(row, cue_label, confusions) if exception_fail == "pass" else ""
    pool = "main" if fail == "pass" else "single_pair_context_exception" if exception_fail == "pass" else "reject"
    readability = min(1.0, min(target_h_short, target_o_short, max(contrast_h_short, PANEL_HUMAN_SHORT_EDGE), max(contrast_o_short, cfg.contrast_object_short_edge)) / 90.0)
    main_score = 100.0 * (
        0.34 * dist
        + 0.18 * min(1.0, axes / 3.0)
        + 0.16 * (1.0 - min(1.0, dominant))
        + 0.12 * min(1.0, len(roles) / 4.0)
        + 0.10 * min(1.0, ambiguity)
        + 0.10 * readability
    )
    exception_score = 100.0 * (0.36 * min(1.0, ambiguity) + 0.30 * int(bool(cue_label)) + 0.20 * min(1.0, len(confusions) / 4.0) + 0.14 * min(1.0, target_o_short / 80.0))
    score = main_score if pool == "main" else exception_score if pool == "single_pair_context_exception" else main_score
    if pool == "reject":
        score -= 40.0

    single_person_single_object = effective_human_count < 2 and context["canonical_human_count"] <= 1 and context["canonical_object_count"] <= 1
    false_multi = effective_human_count < 2 and context["canonical_human_count"] >= 2
    homogeneous_multi = effective_human_count >= 2 and (distinct_activity_signature_count < 2 or len(all_families) < 2 or dominant > cfg.dominant_activity_ratio_max)
    primary_sig = target_sig or (target_sigs[0] if target_sigs else ("", "", ""))

    return {
        "config": cfg.name,
        "rank": 0,
        "pool": pool,
        "score": round(score, 6),
        "first_failed_gate": fail if pool != "single_pair_context_exception" else f"single_pair_exception:{exception_fail}",
        "candidate_id": str(row.pair_id),
        "pair_id": str(row.pair_id),
        "image_key": str(row.image_key),
        "image_id": str(row.image_id),
        "split": str(row.split),
        "file_name": str(row.file_name),
        "image_path": str(context["summary"].image_path),
        "target_human_id": str(row.canonical_human_id),
        "target_object_id": str(row.canonical_object_id),
        "target_object_category": target_obj,
        "target_verbs": jdump(target_verbs),
        "target_verb_families": jdump(target_fams),
        "target_activity_signatures": jdump([signature_label(sig) for sig in target_sigs]),
        "target_human_short_edge_px": round(target_h_short, 3),
        "target_object_short_edge_px": round(target_o_short, 3),
        "contrast_pair_id": str(contrast["row"].pair_id) if contrast else "",
        "contrast_human_id": contrast["human_id"] if contrast else "",
        "contrast_object_id": contrast["object_id"] if contrast else "",
        "contrast_object_category": contrast_obj,
        "contrast_verbs": jdump(contrast["verbs"] if contrast else []),
        "contrast_verb_families": jdump(contrast["families"] if contrast else []),
        "contrast_activity_signatures": jdump([signature_label(sig) for sig in contrast["signatures"]] if contrast else []),
        "contrast_human_short_edge_px": round(contrast_h_short, 3),
        "contrast_object_short_edge_px": round(contrast_o_short, 3),
        "effective_human_count": effective_human_count,
        "cross_human_effective_pair_count": cross_human_effective_pair_count,
        "distinct_activity_signature_count": distinct_activity_signature_count,
        "dominant_activity_ratio": round(dominant, 6),
        "nontrivial_contrast_axis_count": axes,
        "activity_signature_distance": round(dist, 6),
        "pair_external_evidence_confirmed": bool(pair_external_evidence_confirmed),
        "both_pairs_readable_in_full_scene": bool(both_pairs_readable),
        "second_pair_role": jdump(roles),
        "candidate_swap_risk_proxy": round(swap_risk, 6),
        "scene_level_error_proxy": round(scene_error, 6),
        "neighborhood_error_proxy": round(neighbor_error, 6),
        "target_evidence_needed": target_need,
        "borrowed_evidence_error_answer": borrowed,
        "main_required_answer": first_answer,
        "single_pair_exception_answer": exception_answer,
        "concrete_context_cue": cue_label,
        "object_function_group": primary_sig[1],
        "verb_family_primary": primary_sig[0],
        "scene_family": context["scene_family"],
        "is_single_person_single_object_reject": single_person_single_object,
        "is_false_multi_background_reject": false_multi,
        "is_homogeneous_multi_reject": homogeneous_multi,
    }


def coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        "object": dict(Counter(str(r["target_object_category"]) for r in rows)),
        "verb_family": dict(Counter(str(r["verb_family_primary"]) for r in rows)),
        "scene_family": dict(Counter(str(r["scene_family"]) for r in rows)),
    }


def draw_boxes_on_image(image: Image.Image, row: dict[str, Any], pair_lookup: dict[str, Any]) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    target_pair = pair_lookup.get(row.get("pair_id", ""))
    contrast_pair = pair_lookup.get(row.get("contrast_pair_id", ""))
    if target_pair is not None:
        for box, color, label in [
            (bbox_from_row(target_pair, "human"), (0, 160, 0), "target H"),
            (bbox_from_row(target_pair, "object"), (255, 80, 0), "target O"),
        ]:
            ib = tuple(int(round(x)) for x in box)
            draw.rectangle(ib, outline=color, width=4)
            draw.text((ib[0] + 3, max(0, ib[1] - 15)), label, fill=color)
    if contrast_pair is not None:
        for box, color, label in [
            (bbox_from_row(contrast_pair, "human"), (0, 80, 255), "contrast H"),
            (bbox_from_row(contrast_pair, "object"), (170, 0, 255), "contrast O"),
        ]:
            ib = tuple(int(round(x)) for x in box)
            draw.rectangle(ib, outline=color, width=3)
            draw.text((ib[0] + 3, max(0, ib[1] - 15)), label, fill=color)
    return out


def draw_contact_sheet(
    rows: list[dict[str, Any]],
    path: Path,
    title: str,
    pair_lookup: dict[str, Any],
    cols: int,
    thumb_w: int,
    thumb_h: int,
    anonymize: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = rows[:CONTACT_SHEET_LIMIT]
    if not rows:
        img = Image.new("RGB", (thumb_w * cols, 160), "white")
        draw = ImageDraw.Draw(img)
        draw.text((18, 20), f"{title}: no rows", fill=(0, 0, 0))
        img.save(path, quality=92)
        return
    header_h = 48
    sheet_rows = math.ceil(len(rows) / cols)
    sheet = Image.new("RGB", (thumb_w * cols, header_h + sheet_rows * thumb_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((14, 14), title, fill=(0, 0, 0))
    for idx, row in enumerate(rows):
        col = idx % cols
        rr = idx // cols
        x = col * thumb_w
        y = header_h + rr * thumb_h
        tile = Image.new("RGB", (thumb_w, thumb_h), "white")
        src = Path(str(row.get("image_path", "")))
        if src.exists():
            with Image.open(src) as raw:
                im = draw_boxes_on_image(raw, row, pair_lookup)
                im.thumbnail((thumb_w, thumb_h - 76), Image.Resampling.LANCZOS)
                tile.paste(im, ((thumb_w - im.width) // 2, 0))
        tdraw = ImageDraw.Draw(tile)
        label_id = f"A{idx + 1:03d}" if anonymize else row.get("pair_id", "")
        tdraw.text((6, thumb_h - 72), f"#{idx + 1} {label_id}"[:72], fill=(0, 0, 0))
        tdraw.text((6, thumb_h - 54), f"pool={row.get('pool','')} gate={row.get('first_failed_gate','')}"[:72], fill=(0, 0, 0))
        tdraw.text((6, thumb_h - 36), f"T {row.get('target_object_category','')}/{row.get('target_verbs','')} C {row.get('contrast_object_category','')}"[:72], fill=(0, 0, 0))
        tdraw.text((6, thumb_h - 18), f"d={row.get('activity_signature_distance','')} axes={row.get('nontrivial_contrast_axis_count','')} score={row.get('score','')}"[:72], fill=(0, 0, 0))
        sheet.paste(tile, (x, y))
    sheet.save(path, quality=92)


def rank_pool(rows: list[dict[str, Any]], pool: str) -> list[dict[str, Any]]:
    selected = [dict(r) for r in rows if r["pool"] == pool]
    selected.sort(key=lambda r: (-float(r["score"]), r["image_key"], r["pair_id"]))
    for rank, row in enumerate(selected, start=1):
        row["rank"] = rank
    return selected


def unique_by_image(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    out = []
    used = set()
    for row in rows:
        if row["image_key"] in used:
            continue
        out.append(row)
        used.add(row["image_key"])
        if len(out) >= limit:
            break
    return out


def rows_for_ids(rows: list[dict[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    out = [dict(r) for r in rows if r["pair_id"] in ids and r["pool"] == "main"]
    out.sort(key=lambda r: (-float(r["score"]), r["image_key"], r["pair_id"]))
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


def reclassify_current_ten(v2_rows: list[dict[str, str]], balanced_rows_by_pair: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in v2_rows:
        pair_id = row.get("pair_id", "")
        classified = balanced_rows_by_pair.get(pair_id)
        if classified:
            payload = dict(classified)
            payload["rank"] = row.get("top10_id", "")
            payload["current_top10_status"] = classified["pool"]
            payload["current_top10_old_rank"] = row.get("final_rank", "")
            payload["current_top10_old_note"] = row.get("selection_note", "")
        else:
            payload = {
                "rank": row.get("top10_id", ""),
                "pool": "not_found_in_balanced_classification",
                "first_failed_gate": "missing_pair_id_in_v4",
                "pair_id": pair_id,
                "candidate_id": row.get("candidate_id", ""),
                "image_key": pair_id.rsplit(":p", 1)[0] if ":p" in pair_id else "",
                "file_name": Path(row.get("original_path", "")).name,
                "image_path": str(PROJECT_ROOT / row.get("original_path", "")) if row.get("original_path") else "",
                "target_object_category": row.get("object_class", ""),
                "target_verbs": row.get("target_gt_verbs", ""),
                "contrast_object_category": "",
                "activity_signature_distance": "",
                "nontrivial_contrast_axis_count": "",
                "score": "",
            }
        out.append(payload)
    return out


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "contact_sheets").mkdir(parents=True, exist_ok=True)

    action_names = read_action_names(args.actions)
    pairs = pd.read_parquet(args.input_dir / "pairs.parquet")
    instances = pd.read_parquet(args.input_dir / "instances.parquet")
    summaries = pd.read_parquet(args.input_dir / "image_summary.parquet")
    pairs = add_pair_families(pairs, action_names)

    global_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pairs.itertuples(index=False):
        for verb in positive_verbs(row):
            global_counts[str(row.object_category_name)][verb] += 1
    global_confusion = {obj: [verb for verb, _ in counts.most_common()] for obj, counts in global_counts.items()}
    summaries_by_key = {str(row.image_key): row for row in summaries.itertuples(index=False)}
    instances_by_image = {str(key): df for key, df in instances.groupby("image_key", sort=False)}
    pair_lookup = {str(row.pair_id): row for row in pairs.itertuples(index=False)}

    all_results: dict[str, list[dict[str, Any]]] = {}
    summaries_out: dict[str, Any] = {}
    coverage_out: dict[str, Any] = {}
    current_ten_rows: list[dict[str, Any]] = []

    for cfg in CONFIGS:
        image_contexts: dict[str, dict[str, Any]] = {}
        for image_key, image_pairs in pairs.groupby("image_key", sort=False):
            key = str(image_key)
            summary = summaries_by_key.get(key)
            if summary is None:
                continue
            image_contexts[key] = build_image_context(key, image_pairs, instances_by_image.get(key, instances.iloc[0:0]), summary, cfg)

        rows = []
        for row in pairs.itertuples(index=False):
            context = image_contexts.get(str(row.image_key))
            if context is None:
                continue
            rows.append(classify_pair(row, context, cfg, global_confusion))
        rows.sort(key=lambda r: (r["pool"] != "main", r["pool"] != "single_pair_context_exception", -float(r["score"]), r["image_key"], r["pair_id"]))
        main_rows = rank_pool(rows, "main")
        exception_rows = rank_pool(rows, "single_pair_context_exception")
        reject_rows = [r for r in rows if r["pool"] == "reject"]
        for row in reject_rows:
            row["rank"] = 0

        all_results[cfg.name] = rows
        write_csv(args.output_dir / f"candidate_classification_{cfg.name}.csv", rows, MAIN_COLUMNS)
        write_csv(args.output_dir / f"main_pool_ranked_{cfg.name}.csv", main_rows, MAIN_COLUMNS)
        write_csv(args.output_dir / f"single_pair_context_exception_ranked_{cfg.name}.csv", exception_rows, MAIN_COLUMNS)
        write_csv(args.output_dir / f"rejects_ranked_{cfg.name}.csv", reject_rows, MAIN_COLUMNS)

        single_leaks = sum(1 for r in main_rows if int(r["effective_human_count"]) < 2)
        homogeneous_mis = sum(1 for r in main_rows if str(r["is_homogeneous_multi_reject"]) == "True")
        summaries_out[cfg.name] = {
            "main_pool_candidate_count": len(main_rows),
            "single_pair_exception_count": len(exception_rows),
            "single_person_main_pool_leak_count": single_leaks,
            "homogeneous_multi_person_misselection_count": homogeneous_mis,
            "first_failed_gate_counts": dict(Counter(r["first_failed_gate"] for r in rows)),
        }
        coverage_out[cfg.name] = {
            "main_pool": coverage(main_rows),
            "single_pair_context_exception": coverage(exception_rows),
        }
        if cfg.name == "balanced":
            by_pair = {r["pair_id"]: r for r in rows}
            current_ten_rows = reclassify_current_ten(read_csv(args.v2_top10), by_pair)

    strict_ids = {r["pair_id"] for r in all_results["strict"] if r["pool"] == "main"}
    balanced_ids = {r["pair_id"] for r in all_results["balanced"] if r["pool"] == "main"}
    recall_ids = {r["pair_id"] for r in all_results["recall"] if r["pool"] == "main"}
    intersection_specs = {
        "strict_balanced_recall_intersection": (strict_ids & balanced_ids & recall_ids, "intersection_strict_balanced_recall.csv", "balanced"),
        "strict_balanced_intersection": (strict_ids & balanced_ids, "intersection_strict_balanced.csv", "balanced"),
        "balanced_recall_intersection": (balanced_ids & recall_ids, "intersection_balanced_recall.csv", "balanced"),
        "balanced_new_candidates_vs_strict": (balanced_ids - strict_ids, "balanced_new_candidates_vs_strict.csv", "balanced"),
        "recall_only_false_positive_candidates": (recall_ids - balanced_ids, "recall_only_false_positive_candidates.csv", "recall"),
    }
    intersections = {}
    for name, (ids, filename, source_config) in intersection_specs.items():
        write_csv(args.output_dir / filename, rows_for_ids(all_results[source_config], ids), MAIN_COLUMNS)
        intersections[name] = {
            "count": len(ids),
            "csv": str((args.output_dir / filename).relative_to(PROJECT_ROOT)),
        }

    balanced_rows = all_results["balanced"]
    balanced_main = rank_pool(balanced_rows, "main")
    balanced_exception = rank_pool(balanced_rows, "single_pair_context_exception")
    false_multi = [r for r in balanced_rows if r["pool"] == "reject" and str(r["is_false_multi_background_reject"]) == "True"]
    homogeneous = [r for r in balanced_rows if r["pool"] == "reject" and str(r["is_homogeneous_multi_reject"]) == "True"]
    single_single = [r for r in balanced_rows if r["pool"] == "reject" and str(r["is_single_person_single_object_reject"]) == "True"]
    false_multi.sort(key=lambda r: (-float(r["score"]), r["image_key"]))
    homogeneous.sort(key=lambda r: (-float(r["score"]), r["image_key"]))
    single_single.sort(key=lambda r: (-float(r["score"]), r["image_key"]))

    shortlist_main = unique_by_image(balanced_main, 10)
    shortlist_exception = unique_by_image(balanced_exception, 2)
    # The policy is recorded without merging scores: main and exception rows keep
    # their separate source ranks and pool labels.
    selected_policy = shortlist_main[:8] + shortlist_exception[:2] if len(shortlist_main) >= 8 else shortlist_main
    write_csv(args.output_dir / "shortlist_main_balanced.csv", shortlist_main, MAIN_COLUMNS)
    write_csv(args.output_dir / "shortlist_single_pair_context_exception_balanced.csv", shortlist_exception, MAIN_COLUMNS)
    write_csv(args.output_dir / "shortlist_policy_balanced_main_first_max_20pct_exception.csv", selected_policy, MAIN_COLUMNS)
    write_csv(args.output_dir / "current_ten_reclassification_balanced.csv", current_ten_rows, MAIN_COLUMNS + ["current_top10_status", "current_top10_old_rank", "current_top10_old_note"])

    draw_contact_sheet(balanced_main, args.output_dir / "contact_sheets/multi_person_heterogeneous_candidates_balanced.jpg", "Balanced multi-person heterogeneous main candidates", pair_lookup, args.sheet_cols, args.thumb_width, args.thumb_height)
    draw_contact_sheet(balanced_exception, args.output_dir / "contact_sheets/single_pair_context_exceptions_balanced.jpg", "Balanced single-pair context exceptions", pair_lookup, args.sheet_cols, args.thumb_width, args.thumb_height)
    draw_contact_sheet(false_multi, args.output_dir / "contact_sheets/false_multi_background_person_rejects_balanced.jpg", "False multi/background person rejects", pair_lookup, args.sheet_cols, args.thumb_width, args.thumb_height)
    draw_contact_sheet(homogeneous, args.output_dir / "contact_sheets/homogeneous_multi_activity_rejects_balanced.jpg", "Homogeneous multi-person activity rejects", pair_lookup, args.sheet_cols, args.thumb_width, args.thumb_height)
    draw_contact_sheet(single_single, args.output_dir / "contact_sheets/single_person_single_object_rejects_balanced.jpg", "Single-person single-object rejects", pair_lookup, args.sheet_cols, args.thumb_width, args.thumb_height)
    draw_contact_sheet(current_ten_rows, args.output_dir / "contact_sheets/current_ten_reclassified_balanced.jpg", "Current ten images reclassified by balanced gates", pair_lookup, args.sheet_cols, args.thumb_width, args.thumb_height)
    draw_contact_sheet(selected_policy, args.output_dir / "contact_sheets/anonymous_shortlist_balanced.jpg", "Anonymous shortlist: main first, exception max 20 percent", pair_lookup, args.sheet_cols, args.thumb_width, args.thumb_height, anonymize=True)

    with (args.output_dir / "threshold_comparison_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": "figure1_search_v4_pair_gate.summary",
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "output_dir": str(args.output_dir.relative_to(PROJECT_ROOT)),
            "configs": [cfg.__dict__ for cfg in CONFIGS],
            "counts": summaries_out,
            "intersections": intersections,
            "coverage": coverage_out,
            "shortlist_policy": {
                "main_source": "shortlist_main_balanced.csv",
                "exception_source": "shortlist_single_pair_context_exception_balanced.csv",
                "selected_policy_rows": "shortlist_policy_balanced_main_first_max_20pct_exception.csv",
                "top10_rule": "At least 80% main-pool if 10 rows are selected; single-pair exceptions are capped at 20%; single-pair exceptions are not used to fill a missing main quota.",
                "selected_policy_count": len(selected_policy),
                "selected_policy_main_count": sum(1 for r in selected_policy if r["pool"] == "main"),
                "selected_policy_exception_count": sum(1 for r in selected_policy if r["pool"] == "single_pair_context_exception"),
            },
            "contact_sheets": sorted(str(p.relative_to(PROJECT_ROOT)) for p in (args.output_dir / "contact_sheets").glob("*.jpg")),
        }, handle, ensure_ascii=False, indent=2)

    readme = [
        "# Figure 1 Search v4 Pair Gate",
        "",
        "This directory contains threshold comparison, classification contact sheets, and a re-screened shortlist only. No final Figure 1 or paper text was produced.",
        "",
        "## Core Definition",
        "",
        "`activity_signature = (verb_family, object_function_group, evidence_template)`. Bicycle and motorcycle ride cases share the same two-wheel vehicle ride evidence template and therefore do not count as heterogeneous activity by object category alone.",
        "",
        "## Counts",
        "",
    ]
    for name in ["recall", "balanced", "strict"]:
        info = summaries_out[name]
        readme.append(f"- {name}: main={info['main_pool_candidate_count']}; single_pair_exception={info['single_pair_exception_count']}; single_person_main_leaks={info['single_person_main_pool_leak_count']}; homogeneous_multi_misselections={info['homogeneous_multi_person_misselection_count']}.")
    readme.extend([
        "",
        "## Key Files",
        "",
        "- `threshold_comparison_summary.json`: counts, first-failed gates, intersections, and object/verb/scene coverage.",
        "- `candidate_classification_{recall,balanced,strict}.csv`: every evaluated candidate with `first_failed_gate`.",
        "- `main_pool_ranked_*.csv` and `single_pair_context_exception_ranked_*.csv`: separate ranking pools, not a combined leaderboard.",
        "- `shortlist_main_balanced.csv`: main-pool shortlist, prioritized for any final selection.",
        "- `shortlist_single_pair_context_exception_balanced.csv`: exception shortlist, capped separately.",
        "- `contact_sheets/*.jpg`: requested classification sheets.",
    ])
    (args.output_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(args.output_dir),
        "counts": summaries_out,
        "selected_policy_count": len(selected_policy),
        "contact_sheets": sorted(str(p) for p in (args.output_dir / "contact_sheets").glob("*.jpg")),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
