#!/usr/bin/env python3
"""Stage 3 full-index HICO-DET context-counterfactual search.

The search unit is (target pair, GT verb, competitor, c+, c-). It starts from
the complete canonical HICO-DET train/test pair index, not from old Top-K lists.
All model-like judgments are deterministic [AUTO] or [PROTOTYPE] proxies; no
EvidenceHOI attention, probability, or causal effect is claimed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from feature_views import (
    Box,
    PROJECT_ROOT,
    SOURCE_AUTO,
    SOURCE_GT,
    SOURCE_PROTO,
    box_from_instance,
    box_from_pair,
    build_object_verb_prototypes,
    compute_geometry,
    dilate_box,
    intersection_area,
    json_dumps,
    load_json_list,
    outside_fraction,
    positive_verbs,
    read_action_names,
    rel,
    select_competitors,
    true_pair_occupancy_area,
    union_rect,
    verb_family,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v5_context_counterfactual"
DEFAULT_STAGE2_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifest_stage2.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "outputs/figure1_search_v2/index"
DEFAULT_ACTIONS = PROJECT_ROOT / "hico_20160224_det/hico_list_vb.txt"

PRIORITY_OBJECTS = {"pizza", "suitcase", "tv", "laptop", "cake", "bottle", "cup", "wine glass", "bicycle", "book"}

POSITIVE_CONTEXT_PREFS: dict[str, list[str]] = {
    "pizza": ["oven", "paddle", "plate", "pizza", "dining table", "knife", "fork", "person", "chair"],
    "suitcase": ["car", "trunk", "clothes", "bed", "bench", "backpack", "handbag", "person"],
    "tv": ["remote", "couch", "repair tools", "indoor", "outdoor", "chair", "laptop", "cell phone", "person"],
    "laptop": ["mouse", "repair tools", "book", "bag", "keyboard", "backpack", "suitcase", "chair", "dining table", "cell phone"],
    "cake": ["knife", "candles", "plate", "oven", "birthday context", "fork", "dining table", "cup", "bottle", "person", "chair"],
    "bottle": ["sink", "cup", "wine glass", "dining table", "food", "bowl", "pizza", "cake", "sandwich", "person"],
    "cup": ["sink", "bottle", "wine glass", "dining table", "food", "bowl", "pizza", "cake", "sandwich", "person"],
    "wine glass": ["bottle", "cup", "dining table", "bowl", "pizza", "cake", "sandwich", "person"],
    "bicycle": ["repair tools", "helmet", "bicycle", "road", "backpack", "motorcycle", "car", "truck", "person"],
    "book": ["dining table", "couch", "person", "chair", "laptop", "backpack"],
}

THRESHOLDS = {
    "recall": {
        "occupancy_min": 0.16,
        "occupancy_max": 0.70,
        "human_projected_height": 110.0,
        "object_projected_sqrt_area": 35.0,
        "context_outside_fraction": 0.35,
        "external_context_projected_size": 16.0,
        "pair_union_ambiguity": 2.0,
    },
    "balanced": {
        "occupancy_min": 0.20,
        "occupancy_max": 0.62,
        "human_projected_height": 130.0,
        "object_projected_sqrt_area": 45.0,
        "context_outside_fraction": 0.50,
        "external_context_projected_size": 22.0,
        "pair_union_ambiguity": 3.0,
    },
    "strict": {
        "occupancy_min": 0.24,
        "occupancy_max": 0.55,
        "human_projected_height": 150.0,
        "object_projected_sqrt_area": 55.0,
        "context_outside_fraction": 0.65,
        "external_context_projected_size": 28.0,
        "pair_union_ambiguity": 3.0,
    },
}

GATE_ORDER = ["Gate S", "Gate A", "Gate P", "Gate N", "Gate I"]


@dataclass(frozen=True)
class ContextCandidate:
    context_id: str
    category: str
    box: Box
    source: str
    supported_verb: str
    relation_endpoints: str
    provenance: str
    priority_rank: int


@dataclass(frozen=True)
class PairAmbiguity:
    plausible_verbs: list[str]
    pair_only_ambiguity: float
    union_only_ambiguity: float
    internal_decisive: bool
    internal_decisive_reason: str


@dataclass(frozen=True)
class ImageContextIndex:
    pairs: list[Any]
    inst_map: dict[str, Any]
    same_human_object: dict[tuple[str, str], Any]
    first_object_pair: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage2-manifest", type=Path, default=DEFAULT_STAGE2_MANIFEST)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--review-width", type=int, default=900)
    parser.add_argument("--outside-dilation-ratio", type=float, default=0.03)
    parser.add_argument("--max-cpos-per-target", type=int, default=3)
    parser.add_argument("--max-cneg-per-competitor", type=int, default=3)
    parser.add_argument("--max-tuples-per-target-competitor", type=int, default=4)
    parser.add_argument("--overwrite-stage3", action="store_true")
    return parser.parse_args()


def mask_iou(a: Box, b: Box) -> float:
    inter = intersection_area(a, b)
    union = a.area + b.area - inter
    return 0.0 if union <= 0 else inter / union


def projected_size(box: Box, scale: float) -> float:
    return math.sqrt(max(0.0, box.area)) * scale


def get_pair_row_dict(row: Any) -> dict[str, Any]:
    return {name: getattr(row, name) for name in row._fields}


def pair_positive_verbs_from_series(row: pd.Series) -> list[str]:
    return [str(v) for v in load_json_list(row.get("verb_names_json")) if str(v) != "no_interaction"]


def object_counts_from_pairs(valid_pairs: pd.DataFrame) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in valid_pairs.itertuples(index=False):
        for verb in positive_verbs(row):
            counts[str(row.object_category_name)][verb] += 1
    return counts


def compute_ambiguity(target_obj: str, gt_verb: str, competitors: list[str]) -> PairAmbiguity:
    plausible = [gt_verb] + [v for v in competitors if v != gt_verb]
    families = {verb_family(v) for v in plausible}
    pair_score = min(4.0, 1.0 + 0.75 * max(0, len(plausible) - 1) + 0.45 * max(0, len(families) - 1))
    union_score = pair_score
    internal_decisive = False
    reason = ""

    if target_obj in {"bicycle", "dog"} and gt_verb in {"walk", "run"}:
        internal_decisive = True
        reason = "pair_internal_cues_already_decisive: walking/running is usually resolved by target body pose, leash/contact, and pair geometry inside the union"
        union_score = min(union_score, 1.5)
    elif target_obj in {"skateboard", "snowboard", "skis", "surfboard"} and gt_verb in {"ride", "stand_on", "jump"}:
        internal_decisive = True
        reason = "pair_internal_cues_already_decisive: sport-board action is usually resolved by target body/support relation inside the union"
        union_score = min(union_score, 1.75)
    elif target_obj in {"chair", "bench", "couch", "bed"} and gt_verb in {"sit_on", "lie_on", "stand_on"} and len(competitors) < 3:
        internal_decisive = True
        reason = "pair_internal_cues_already_decisive: furniture posture may be resolved by body pose inside the union"
        union_score = min(union_score, 2.0)

    return PairAmbiguity(plausible, round(pair_score, 3), round(union_score, 3), internal_decisive, reason)


def relation_for_context(image_pairs: pd.DataFrame, context_id: str, target_human_id: str) -> tuple[str, str, list[str]]:
    same_human = image_pairs[
        (image_pairs["canonical_human_id"].astype(str) == target_human_id)
        & (image_pairs["canonical_object_id"].astype(str) == context_id)
    ]
    if not same_human.empty:
        row = same_human.iloc[0]
        return f"{row['canonical_human_id']}->{row['canonical_object_id']}", f"{SOURCE_GT} same target-human auxiliary pair {row['pair_id']}", pair_positive_verbs_from_series(row)
    any_pair = image_pairs[image_pairs["canonical_object_id"].astype(str) == context_id]
    if not any_pair.empty:
        row = any_pair.iloc[0]
        return f"{row['canonical_human_id']}->{row['canonical_object_id']}", f"{SOURCE_GT} HICO external pair {row['pair_id']}", pair_positive_verbs_from_series(row)
    return f"context_instance_only:{context_id}", f"{SOURCE_AUTO} canonical instance without direct target-human relation", []


def build_image_context_indexes(valid_pairs: pd.DataFrame, instances: pd.DataFrame) -> dict[str, ImageContextIndex]:
    instances_by_image = {
        str(image_key): {str(row.canonical_instance_id): row for row in df.itertuples(index=False)}
        for image_key, df in instances.groupby("image_key", sort=False)
    }
    indexes: dict[str, ImageContextIndex] = {}
    for image_key, df in valid_pairs.groupby("image_key", sort=False):
        pair_rows = list(df.itertuples(index=False))
        same_human_object: dict[tuple[str, str], Any] = {}
        first_object_pair: dict[str, Any] = {}
        for row in pair_rows:
            human_id = str(row.canonical_human_id)
            object_id = str(row.canonical_object_id)
            same_human_object.setdefault((human_id, object_id), row)
            first_object_pair.setdefault(object_id, row)
        indexes[str(image_key)] = ImageContextIndex(
            pairs=pair_rows,
            inst_map=instances_by_image.get(str(image_key), {}),
            same_human_object=same_human_object,
            first_object_pair=first_object_pair,
        )
    return indexes


def relation_for_context_fast(index: ImageContextIndex, context_id: str, target_human_id: str) -> tuple[str, str, list[str]]:
    row = index.same_human_object.get((target_human_id, context_id))
    if row is not None:
        return f"{row.canonical_human_id}->{row.canonical_object_id}", f"{SOURCE_GT} same target-human auxiliary pair {row.pair_id}", positive_verbs(row)
    row = index.first_object_pair.get(context_id)
    if row is not None:
        return f"{row.canonical_human_id}->{row.canonical_object_id}", f"{SOURCE_GT} HICO external pair {row.pair_id}", positive_verbs(row)
    return f"context_instance_only:{context_id}", f"{SOURCE_AUTO} canonical instance without direct target-human relation", []


def build_positive_contexts_fast(
    target_row: Any,
    gt_verb: str,
    index: ImageContextIndex,
    target_endpoint_ids: set[str],
) -> list[ContextCandidate]:
    target_obj = str(target_row.object_category_name)
    prefs = POSITIVE_CONTEXT_PREFS.get(target_obj, [])
    pref_rank = {name: idx for idx, name in enumerate(prefs)}
    contexts: list[ContextCandidate] = []
    for inst_id, inst in index.inst_map.items():
        if inst_id in target_endpoint_ids:
            continue
        category = str(getattr(inst, "category_name", ""))
        rel_endpoints, provenance, rel_verbs = relation_for_context_fast(index, inst_id, str(target_row.canonical_human_id))
        category_pref = category in pref_rank
        same_human_aux = provenance.startswith(f"{SOURCE_GT} same target-human")
        supports_gt_family = any(verb_family(v) == verb_family(gt_verb) for v in rel_verbs)
        if not (category_pref or same_human_aux or supports_gt_family):
            continue
        source_parts = [SOURCE_AUTO]
        if category_pref:
            source_parts.append(f"{SOURCE_PROTO} preferred {target_obj} context category")
        if rel_verbs:
            source_parts.append(f"{SOURCE_GT} relation verbs={json_dumps(rel_verbs)}")
        contexts.append(
            ContextCandidate(
                context_id=inst_id,
                category=category,
                box=box_from_instance(inst),
                source="; ".join(source_parts),
                supported_verb=gt_verb,
                relation_endpoints=rel_endpoints,
                provenance=provenance,
                priority_rank=pref_rank.get(category, 100) - (20 if same_human_aux else 0) - (10 if supports_gt_family else 0),
            )
        )
    contexts.sort(key=lambda c: (c.priority_rank, c.context_id))
    return contexts


def build_negative_contexts_fast(
    target_row: Any,
    competitor: str,
    index: ImageContextIndex,
    target_endpoint_ids: set[str],
    cpos_id: str,
) -> list[ContextCandidate]:
    target_obj = str(target_row.object_category_name)
    competitor_family = verb_family(competitor)
    contexts: list[ContextCandidate] = []
    seen: set[str] = set()

    for row in index.pairs:
        obj_id = str(row.canonical_object_id)
        if obj_id in target_endpoint_ids or obj_id == cpos_id or obj_id not in index.inst_map or obj_id in seen:
            continue
        verbs = positive_verbs(row)
        if not verbs:
            continue
        exact = competitor in verbs
        same_family = any(verb_family(v) == competitor_family for v in verbs)
        wrong_scene_prior = str(row.object_category_name) in POSITIVE_CONTEXT_PREFS.get(target_obj, []) and str(row.canonical_human_id) != str(target_row.canonical_human_id)
        same_or_neighbor_object = str(row.object_category_name) == target_obj
        if not (exact or same_family or wrong_scene_prior or same_or_neighbor_object):
            continue
        seen.add(obj_id)
        inst = index.inst_map[obj_id]
        if exact:
            supported = competitor
            rank = 0
            reason = "supports exact competitor"
        elif same_family:
            supported = next((v for v in verbs if verb_family(v) == competitor_family), verbs[0])
            rank = 1
            reason = "supports competitor verb family"
        elif same_or_neighbor_object:
            supported = competitor
            rank = 2
            reason = "same-object neighborhood may be misselected"
        else:
            supported = competitor
            rank = 3
            reason = "wrong scene prior from preferred context category"
        contexts.append(
            ContextCandidate(
                context_id=obj_id,
                category=str(getattr(inst, "category_name", row.object_category_name)),
                box=box_from_instance(inst),
                source=f"{SOURCE_GT} external pair; {SOURCE_AUTO} {reason}",
                supported_verb=supported,
                relation_endpoints=f"{row.canonical_human_id}->{row.canonical_object_id}",
                provenance=f"{SOURCE_GT} pair {row.pair_id}; second pair optional, not a qualification gate",
                priority_rank=rank,
            )
        )

    if not contexts:
        for inst_id, inst in index.inst_map.items():
            if inst_id in target_endpoint_ids or inst_id == cpos_id:
                continue
            category = str(getattr(inst, "category_name", ""))
            if category not in POSITIVE_CONTEXT_PREFS.get(target_obj, []) and category != target_obj:
                continue
            contexts.append(
                ContextCandidate(
                    context_id=inst_id,
                    category=category,
                    box=box_from_instance(inst),
                    source=f"{SOURCE_AUTO} scene-prior fallback; {SOURCE_PROTO} may support {competitor}",
                    supported_verb=competitor,
                    relation_endpoints=f"context_instance_only:{inst_id}",
                    provenance=f"{SOURCE_AUTO} instance-level wrong scene prior fallback",
                    priority_rank=4,
                )
            )
    contexts.sort(key=lambda c: (c.priority_rank, c.context_id))
    return contexts


def build_positive_contexts(
    target_row: Any,
    gt_verb: str,
    image_pairs: pd.DataFrame,
    inst_map: dict[str, Any],
    target_endpoint_ids: set[str],
) -> list[ContextCandidate]:
    target_obj = str(target_row.object_category_name)
    prefs = POSITIVE_CONTEXT_PREFS.get(target_obj, [])
    pref_rank = {name: idx for idx, name in enumerate(prefs)}
    contexts: list[ContextCandidate] = []
    for inst_id, inst in inst_map.items():
        if inst_id in target_endpoint_ids:
            continue
        category = str(getattr(inst, "category_name", ""))
        rel_endpoints, provenance, rel_verbs = relation_for_context(image_pairs, inst_id, str(target_row.canonical_human_id))
        category_pref = category in pref_rank
        same_human_aux = provenance.startswith(f"{SOURCE_GT} same target-human")
        supports_gt_family = any(verb_family(v) == verb_family(gt_verb) for v in rel_verbs)
        if not (category_pref or same_human_aux or supports_gt_family):
            continue
        source_parts = [SOURCE_AUTO]
        if category_pref:
            source_parts.append(f"{SOURCE_PROTO} preferred {target_obj} context category")
        if rel_verbs:
            source_parts.append(f"{SOURCE_GT} relation verbs={json_dumps(rel_verbs)}")
        contexts.append(
            ContextCandidate(
                context_id=inst_id,
                category=category,
                box=box_from_instance(inst),
                source="; ".join(source_parts),
                supported_verb=gt_verb,
                relation_endpoints=rel_endpoints,
                provenance=provenance,
                priority_rank=pref_rank.get(category, 100) - (20 if same_human_aux else 0) - (10 if supports_gt_family else 0),
            )
        )
    contexts.sort(key=lambda c: (c.priority_rank, c.context_id))
    return contexts


def build_negative_contexts(
    target_row: Any,
    competitor: str,
    image_pairs: pd.DataFrame,
    inst_map: dict[str, Any],
    target_endpoint_ids: set[str],
    cpos_id: str,
) -> list[ContextCandidate]:
    target_obj = str(target_row.object_category_name)
    competitor_family = verb_family(competitor)
    contexts: list[ContextCandidate] = []
    seen: set[str] = set()

    for row in image_pairs.itertuples(index=False):
        obj_id = str(row.canonical_object_id)
        if obj_id in target_endpoint_ids or obj_id == cpos_id or obj_id not in inst_map or obj_id in seen:
            continue
        verbs = positive_verbs(row)
        if not verbs:
            continue
        exact = competitor in verbs
        same_family = any(verb_family(v) == competitor_family for v in verbs)
        wrong_scene_prior = str(row.object_category_name) in POSITIVE_CONTEXT_PREFS.get(target_obj, []) and str(row.canonical_human_id) != str(target_row.canonical_human_id)
        same_or_neighbor_object = str(row.object_category_name) == target_obj
        if not (exact or same_family or wrong_scene_prior or same_or_neighbor_object):
            continue
        seen.add(obj_id)
        inst = inst_map[obj_id]
        if exact:
            supported = competitor
            rank = 0
            reason = "supports exact competitor"
        elif same_family:
            supported = next((v for v in verbs if verb_family(v) == competitor_family), verbs[0])
            rank = 1
            reason = "supports competitor verb family"
        elif same_or_neighbor_object:
            supported = competitor
            rank = 2
            reason = "same-object neighborhood may be misselected"
        else:
            supported = competitor
            rank = 3
            reason = "wrong scene prior from preferred context category"
        contexts.append(
            ContextCandidate(
                context_id=obj_id,
                category=str(getattr(inst, "category_name", row.object_category_name)),
                box=box_from_instance(inst),
                source=f"{SOURCE_GT} external pair; {SOURCE_AUTO} {reason}",
                supported_verb=supported,
                relation_endpoints=f"{row.canonical_human_id}->{row.canonical_object_id}",
                provenance=f"{SOURCE_GT} pair {row.pair_id}; second pair optional, not a qualification gate",
                priority_rank=rank,
            )
        )

    if not contexts:
        for inst_id, inst in inst_map.items():
            if inst_id in target_endpoint_ids or inst_id == cpos_id:
                continue
            category = str(getattr(inst, "category_name", ""))
            if category not in POSITIVE_CONTEXT_PREFS.get(target_obj, []) and category != target_obj:
                continue
            contexts.append(
                ContextCandidate(
                    context_id=inst_id,
                    category=category,
                    box=box_from_instance(inst),
                    source=f"{SOURCE_AUTO} scene-prior fallback; {SOURCE_PROTO} may support {competitor}",
                    supported_verb=competitor,
                    relation_endpoints=f"context_instance_only:{inst_id}",
                    provenance=f"{SOURCE_AUTO} instance-level wrong scene prior fallback",
                    priority_rank=4,
                )
            )
    contexts.sort(key=lambda c: (c.priority_rank, c.context_id))
    return contexts


def evidence_crop_box(human_box: Box, object_box: Box, cpos_box: Box, cneg_box: Box, image_width: int, image_height: int) -> Box:
    base = union_rect([human_box, object_box, cpos_box, cneg_box])
    pair_area = true_pair_occupancy_area(human_box, object_box)
    for ratio in [0.08, 0.14, 0.20, 0.28, 0.36, 0.48, 0.60]:
        pad = ratio * max(base.width, base.height, 1.0)
        candidate = Box(base.x1 - pad, base.y1 - pad, base.x2 + pad, base.y2 + pad).clamp(image_width, image_height)
        if candidate.area <= 0:
            continue
        if pair_area / candidate.area <= 0.62 or candidate.area >= image_width * image_height * 0.98:
            return candidate
    return base.clamp(image_width, image_height)


def gate_tuple(
    thresholds: dict[str, float],
    geom: Any,
    ambiguity: PairAmbiguity,
    cpos_outside: float,
    cneg_outside: float,
    cpos_projected_size: float,
    cneg_projected_size: float,
    cpos_cneg_iou: float,
    cpos_id: str,
    cneg_id: str,
) -> tuple[bool, str, dict[str, bool]]:
    object_size_ok = geom.target_object_projected_sqrt_area >= thresholds["object_projected_sqrt_area"]
    gate_s = (
        thresholds["occupancy_min"] <= geom.target_pair_occupancy <= thresholds["occupancy_max"]
        and geom.target_human_projected_height >= thresholds["human_projected_height"]
        and object_size_ok
    )
    gate_a = (
        len(ambiguity.plausible_verbs) >= 2
        and ambiguity.pair_only_ambiguity >= thresholds["pair_union_ambiguity"]
        and ambiguity.union_only_ambiguity >= thresholds["pair_union_ambiguity"]
        and not ambiguity.internal_decisive
    )
    gate_p = cpos_outside >= thresholds["context_outside_fraction"] and cpos_projected_size >= thresholds["external_context_projected_size"]
    gate_n = cneg_outside >= thresholds["context_outside_fraction"] and cneg_projected_size >= thresholds["external_context_projected_size"]
    gate_i = cpos_id != cneg_id and cpos_cneg_iou < 0.2
    gates = {"Gate S": gate_s, "Gate A": gate_a, "Gate P": gate_p, "Gate N": gate_n, "Gate I": gate_i}
    for gate in GATE_ORDER:
        if not gates[gate]:
            if gate == "Gate A" and ambiguity.internal_decisive:
                return False, "pair_internal_cues_already_decisive", gates
            return False, gate, gates
    return True, "pass", gates


def append_rejection(
    rows: list[dict[str, Any]],
    config: str,
    first_failed_gate: str,
    target_row: Any,
    gt_verb: str,
    competitor: str,
    reason: str,
    cpos: Optional[ContextCandidate] = None,
    cneg: Optional[ContextCandidate] = None,
) -> None:
    rows.append({
        "config": config,
        "pair_id": str(target_row.pair_id),
        "image_key": str(target_row.image_key),
        "target_object_category": str(target_row.object_category_name),
        "gt_verb": gt_verb,
        "competitor": competitor,
        "cpos_id": "" if cpos is None else cpos.context_id,
        "cneg_id": "" if cneg is None else cneg.context_id,
        "first_failed_gate": first_failed_gate,
        "reason": reason,
    })


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Optional[list[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    stage3_dir = args.output_root / "stage3_search"
    if stage3_dir.exists():
        if not args.overwrite_stage3:
            raise SystemExit(f"Refusing to overwrite existing stage3 directory: {stage3_dir}")
        shutil.rmtree(stage3_dir)
    stage3_dir.mkdir(parents=True, exist_ok=True)

    if not args.stage2_manifest.exists():
        raise FileNotFoundError(args.stage2_manifest)
    stage2_manifest = json.loads(args.stage2_manifest.read_text(encoding="utf-8"))

    actions = read_action_names(args.actions)
    pairs = pd.read_parquet(args.index_dir / "pairs.parquet")
    instances = pd.read_parquet(args.index_dir / "instances.parquet")
    images = pd.read_parquet(args.index_dir / "image_summary.parquet")
    valid_pairs = pairs[(pairs["valid"] == True) & (pairs["invisible"] == False)].copy()
    _prototype_rows, object_counts = build_object_verb_prototypes(valid_pairs, actions)

    images_by_key = {str(row.image_key): row for row in images.itertuples(index=False)}
    context_indexes = build_image_context_indexes(valid_pairs, instances)

    candidate_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    sweep_counts = {name: Counter() for name in THRESHOLDS}
    evaluated_target_pairs: set[str] = set()
    evaluated_tuple_count = 0

    for target in valid_pairs.itertuples(index=False):
        image_key = str(target.image_key)
        image_info = images_by_key.get(image_key)
        if image_info is None:
            continue
        context_index = context_indexes.get(image_key)
        if context_index is None:
            continue
        inst_map = context_index.inst_map
        target_human_id = str(target.canonical_human_id)
        target_object_id = str(target.canonical_object_id)
        target_endpoint_ids = {target_human_id, target_object_id}
        target_obj = str(target.object_category_name)
        human_box = box_from_pair(target, "human")
        object_box = box_from_pair(target, "object")
        pair_region = union_rect([human_box, object_box])
        dilated_pair = dilate_box(pair_region, float(image_info.width), float(image_info.height), args.outside_dilation_ratio)
        target_gt_verbs = positive_verbs(target)
        if not target_gt_verbs:
            continue
        competitors = select_competitors(target_obj, target_gt_verbs, object_counts, limit=5)
        evaluated_target_pairs.add(str(target.pair_id))

        if len(competitors) < 1:
            for cfg in THRESHOLDS:
                sweep_counts[cfg]["Gate A"] += 1
                append_rejection(rejection_rows, cfg, "Gate A", target, target_gt_verbs[0], "", "pair_only_plausible_verbs_lt_2")
            continue

        for gt_verb in target_gt_verbs:
            ambiguity = compute_ambiguity(target_obj, gt_verb, competitors)
            for competitor in competitors:
                if ambiguity.internal_decisive:
                    for cfg in THRESHOLDS:
                        sweep_counts[cfg]["pair_internal_cues_already_decisive"] += 1
                        append_rejection(rejection_rows, cfg, "pair_internal_cues_already_decisive", target, gt_verb, competitor, ambiguity.internal_decisive_reason)
                    continue

                cpos_list = build_positive_contexts_fast(target, gt_verb, context_index, target_endpoint_ids)[: args.max_cpos_per_target]
                if not cpos_list:
                    for cfg in THRESHOLDS:
                        sweep_counts[cfg]["Gate P"] += 1
                        append_rejection(rejection_rows, cfg, "Gate P", target, gt_verb, competitor, "no_external_positive_context_proxy_found")
                    continue

                tuple_for_competitor = 0
                for cpos in cpos_list:
                    cneg_list = build_negative_contexts_fast(target, competitor, context_index, target_endpoint_ids, cpos.context_id)[: args.max_cneg_per_competitor]
                    if not cneg_list:
                        for cfg in THRESHOLDS:
                            sweep_counts[cfg]["Gate N"] += 1
                            append_rejection(rejection_rows, cfg, "Gate N", target, gt_verb, competitor, "no_independent_misleading_context_proxy_found", cpos=cpos)
                        continue
                    for cneg in cneg_list:
                        if tuple_for_competitor >= args.max_tuples_per_target_competitor:
                            break
                        evaluated_tuple_count += 1
                        crop = evidence_crop_box(human_box, object_box, cpos.box, cneg.box, int(image_info.width), int(image_info.height))
                        geom = compute_geometry(human_box, object_box, crop, args.review_width)
                        cpos_out = outside_fraction(cpos.box, dilated_pair)
                        cneg_out = outside_fraction(cneg.box, dilated_pair)
                        cpos_proj = projected_size(cpos.box, geom.scale)
                        cneg_proj = projected_size(cneg.box, geom.scale)
                        cpos_cneg_iou = mask_iou(cpos.box, cneg.box)
                        config_results: dict[str, Any] = {}
                        any_pass = False
                        for cfg, thresholds in THRESHOLDS.items():
                            passed, first_failed, gates = gate_tuple(
                                thresholds,
                                geom,
                                ambiguity,
                                cpos_out,
                                cneg_out,
                                cpos_proj,
                                cneg_proj,
                                cpos_cneg_iou,
                                cpos.context_id,
                                cneg.context_id,
                            )
                            config_results[f"{cfg}_pass"] = passed
                            config_results[f"{cfg}_first_failed_gate"] = first_failed
                            for gate, ok in gates.items():
                                config_results[f"{cfg}_{gate.replace(' ', '_')}_pass"] = ok
                            if passed:
                                any_pass = True
                                sweep_counts[cfg]["pass"] += 1
                            else:
                                sweep_counts[cfg][first_failed] += 1
                                append_rejection(rejection_rows, cfg, first_failed, target, gt_verb, competitor, first_failed, cpos=cpos, cneg=cneg)
                        if any_pass:
                            tuple_for_competitor += 1
                            candidate_rows.append({
                                "tuple_id": f"{target.pair_id}|{gt_verb}|{competitor}|{cpos.context_id}|{cneg.context_id}",
                                "source": f"{SOURCE_AUTO} full v2 canonical HICO index; not old Top-K",
                                "pair_id": str(target.pair_id),
                                "image_key": image_key,
                                "image_id": str(target.image_id),
                                "split": str(target.split),
                                "file_name": str(target.file_name),
                                "image_path": str(getattr(image_info, "image_path", "")),
                                "target_human_id": target_human_id,
                                "target_object_id": target_object_id,
                                "target_object_category": target_obj,
                                "target_human_bbox": json_dumps(human_box.as_list()),
                                "target_object_bbox": json_dumps(object_box.as_list()),
                                "target_union_bbox": json_dumps(pair_region.as_list()),
                                "gt_verb_source": f"{SOURCE_GT} HICO-DET canonical annotation",
                                "gt_verb": gt_verb,
                                "competitor_source": f"{SOURCE_GT} same-object mapping + {SOURCE_PROTO} family merge",
                                "competitor": competitor,
                                "pair_only_plausible_verbs": len(ambiguity.plausible_verbs),
                                "pair_only_plausible_verb_list": json_dumps(ambiguity.plausible_verbs),
                                "pair_only_ambiguity": ambiguity.pair_only_ambiguity,
                                "union_only_ambiguity": ambiguity.union_only_ambiguity,
                                "pair_internal_decisive_proxy": ambiguity.internal_decisive,
                                "pair_internal_decisive_reason": ambiguity.internal_decisive_reason,
                                "cpos_id": cpos.context_id,
                                "cpos_category": cpos.category,
                                "cpos_bbox": json_dumps(cpos.box.as_list()),
                                "cpos_source": cpos.source,
                                "cpos_supported_verb": cpos.supported_verb,
                                "cpos_outside_fraction": round(cpos_out, 6),
                                "cpos_projected_size": round(cpos_proj, 3),
                                "cpos_relation_endpoints": cpos.relation_endpoints,
                                "cpos_provenance": cpos.provenance,
                                "cneg_id": cneg.context_id,
                                "cneg_category": cneg.category,
                                "cneg_bbox": json_dumps(cneg.box.as_list()),
                                "cneg_source": cneg.source,
                                "cneg_supported_verb": cneg.supported_verb,
                                "cneg_outside_fraction": round(cneg_out, 6),
                                "cneg_projected_size": round(cneg_proj, 3),
                                "cneg_relation_endpoints": cneg.relation_endpoints,
                                "cneg_provenance": cneg.provenance,
                                "cpos_cneg_mask_iou": round(cpos_cneg_iou, 6),
                                "evidence_crop_bbox": json_dumps(crop.as_list()),
                                "target_pair_occupancy": round(geom.target_pair_occupancy, 6),
                                "target_human_projected_height": round(geom.target_human_projected_height, 3),
                                "target_object_projected_sqrt_area": round(geom.target_object_projected_sqrt_area, 3),
                                "target_object_projected_short_side": round(geom.target_object_projected_short_side, 3),
                                "second_pair_required": False,
                                **config_results,
                            })

    raw_df = pd.DataFrame(candidate_rows)
    raw_path = stage3_dir / "raw_candidate_tuples.parquet"
    raw_df.to_parquet(raw_path, index=False)

    if raw_df.empty:
        candidate_pair_rows: list[dict[str, Any]] = []
    else:
        candidate_pair_rows = []
        for pair_id, group in raw_df.groupby("pair_id", sort=False):
            best = group.iloc[0]
            candidate_pair_rows.append({
                "pair_id": pair_id,
                "image_key": best["image_key"],
                "split": best["split"],
                "target_object_category": best["target_object_category"],
                "gt_verbs": json_dumps(sorted(set(group["gt_verb"]))),
                "competitors": json_dumps(sorted(set(group["competitor"]))),
                "tuple_count": len(group),
                "recall_tuple_count": int(group["recall_pass"].sum()),
                "balanced_tuple_count": int(group["balanced_pass"].sum()),
                "strict_tuple_count": int(group["strict_pass"].sum()),
                "second_pair_required": False,
            })
    write_csv(stage3_dir / "candidate_pairs.csv", candidate_pair_rows)
    write_csv(stage3_dir / "rejection_log.csv", rejection_rows)

    threshold_rows: list[dict[str, Any]] = []
    for cfg, params in THRESHOLDS.items():
        total = sum(sweep_counts[cfg].values())
        row = {"config": cfg, **params, "evaluated_outcomes": total}
        for key in ["pass", "Gate S", "Gate A", "Gate P", "Gate N", "Gate I", "pair_internal_cues_already_decisive"]:
            row[key] = sweep_counts[cfg].get(key, 0)
        threshold_rows.append(row)
    write_csv(stage3_dir / "threshold_sweep.csv", threshold_rows)

    if raw_df.empty:
        coverage_obj = []
        coverage_comp = []
        context_stats = []
        review_rows = []
    else:
        coverage_obj = []
        for obj, group in raw_df.groupby("target_object_category", sort=True):
            coverage_obj.append({
                "target_object_category": obj,
                "tuple_count": len(group),
                "pair_count": group["pair_id"].nunique(),
                "balanced_tuple_count": int(group["balanced_pass"].sum()),
                "strict_tuple_count": int(group["strict_pass"].sum()),
            })
        coverage_comp = []
        for (obj, comp), group in raw_df.groupby(["target_object_category", "competitor"], sort=True):
            coverage_comp.append({
                "target_object_category": obj,
                "competitor": comp,
                "tuple_count": len(group),
                "balanced_tuple_count": int(group["balanced_pass"].sum()),
                "strict_tuple_count": int(group["strict_pass"].sum()),
            })
        context_stats = []
        for role, src_col, cat_col in [("cpos", "cpos_source", "cpos_category"), ("cneg", "cneg_source", "cneg_category")]:
            for (source, category), group in raw_df.groupby([src_col, cat_col], sort=True):
                context_stats.append({"role": role, "context_source": source, "context_category": category, "tuple_count": len(group)})
        review_df = raw_df[raw_df["balanced_pass"] == True].copy()
        if not review_df.empty:
            review_df["priority_object"] = review_df["target_object_category"].isin(PRIORITY_OBJECTS)
            review_df = review_df.sort_values(
                ["strict_pass", "priority_object", "target_object_category", "pair_id", "competitor", "cpos_cneg_mask_iou"],
                ascending=[False, False, True, True, True, True],
            )
        review_rows = review_df.to_dict("records")

    write_csv(stage3_dir / "coverage_by_object.csv", coverage_obj)
    write_csv(stage3_dir / "coverage_by_competition.csv", coverage_comp)
    write_csv(stage3_dir / "context_source_statistics.csv", context_stats)
    write_csv(stage3_dir / "review_pool_balanced.csv", review_rows)

    report = f"""# Stage 3 Full HICO-DET Search Report

## Scope

- Input source: complete v2 canonical HICO-DET train/test index under `{rel(args.index_dir)}`.
- Old Top-K/shortlists used as seeds: no.
- Candidate tuple unit: `(target pair, GT verb, competitor, c+, c-)`.
- Same target pair can produce multiple tuples: yes.
- Second pair required: no. It can instantiate c+ or c-, but never qualifies a candidate by itself.

## Thresholds

| Parameter | Recall | Balanced | Strict |
|---|---:|---:|---:|
| occupancy lower | 0.16 | 0.20 | 0.24 |
| occupancy upper | 0.70 | 0.62 | 0.55 |
| human projected height | 110 | 130 | 150 |
| object sqrt-area | 35 | 45 | 55 |
| context outside fraction | 0.35 | 0.50 | 0.65 |
| external context projected size | 16 | 22 | 28 |
| pair/union ambiguity | 2/4 | 3/4 | 3/4 |

Balanced is the review-pool threshold. Strict is intended for final Top-10 filtering.

## Counts

- Legal target pairs scanned: {len(valid_pairs)}
- Target pairs evaluated: {len(evaluated_target_pairs)}
- Tuple attempts with concrete c+/c-: {evaluated_tuple_count}
- Candidate tuples passing at least one tier: {len(raw_df)}
- Balanced review tuples: {0 if raw_df.empty else int(raw_df['balanced_pass'].sum())}
- Strict tuples: {0 if raw_df.empty else int(raw_df['strict_pass'].sum())}
- Rejection rows: {len(rejection_rows)}

## Hard Gates

No comprehensive score compensates for a missing hard gate. `first_failed_gate` is recorded for every rejection row. `pair_internal_cues_already_decisive` is treated as a Gate A failure and excluded from the main pool.

## Source Discipline

- GT verbs and object categories are read from HICO-DET canonical mappings.
- Competitors are `[GT]` same-object mappings with `[PROTOTYPE]` verb-family merging.
- c+/c- roles are `[AUTO]`/`[PROTOTYPE]` structural proxies unless tied to a benchmark pair, in which case only the pair/category/verb identity is `[GT]`.
- No EvidenceHOI attention, probability, or causal result is claimed.
"""
    (stage3_dir / "search_report.md").write_text(report, encoding="utf-8")

    output_files = [
        raw_path,
        stage3_dir / "candidate_pairs.csv",
        stage3_dir / "rejection_log.csv",
        stage3_dir / "threshold_sweep.csv",
        stage3_dir / "coverage_by_object.csv",
        stage3_dir / "coverage_by_competition.csv",
        stage3_dir / "context_source_statistics.csv",
        stage3_dir / "review_pool_balanced.csv",
        stage3_dir / "search_report.md",
        Path(__file__),
    ]
    manifest = {
        "stage": 3,
        "task": "full HICO-DET search for v5 context-counterfactual tuples",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": rel(args.output_root),
        "stage3_dir": rel(stage3_dir),
        "stopped_after_manifest": True,
        "full_hico_index_source": True,
        "old_topk_seeded": False,
        "legal_target_pair_count": len(valid_pairs),
        "eligible_for_final_filter_used": False,
        "evaluated_target_pair_count": len(evaluated_target_pairs),
        "tuple_attempt_count": evaluated_tuple_count,
        "candidate_tuple_count": len(raw_df),
        "balanced_review_tuple_count": 0 if raw_df.empty else int(raw_df["balanced_pass"].sum()),
        "strict_tuple_count": 0 if raw_df.empty else int(raw_df["strict_pass"].sum()),
        "rejection_row_count": len(rejection_rows),
        "second_pair_required": False,
        "stage2_manifest": stage2_manifest,
        "outputs": [],
    }
    for path in output_files:
        manifest["outputs"].append({"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest_path = args.output_root / "manifest_stage3.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest_path), "candidate_tuple_count": len(raw_df), "balanced": manifest["balanced_review_tuple_count"], "strict": manifest["strict_tuple_count"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
