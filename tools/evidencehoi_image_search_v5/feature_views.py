#!/usr/bin/env python3
"""Stage 2 feature extraction and counterfactual view generation.

This module implements geometry-only and mapping-only features for the v5
EvidenceHOI Figure 1 search. It does not run CLIP/VLM models, does not infer
EvidenceHOI attention, and does not paste new objects into images.
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
from PIL import Image, ImageDraw


PROJECT_ROOT = Path("/wzj/zijian/OSSM-HOI-main")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/figure1_search_v5_context_counterfactual"
DEFAULT_STAGE1_MANIFEST = DEFAULT_OUTPUT_ROOT / "manifest_stage1.json"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "outputs/figure1_search_v2/index"
DEFAULT_V4_CLASSIFICATION = PROJECT_ROOT / "outputs/figure1_search_v4_pair_gate/candidate_classification_balanced.csv"
DEFAULT_ACTIONS = PROJECT_ROOT / "hico_20160224_det/hico_list_vb.txt"

SOURCE_AUTO = "[AUTO]"
SOURCE_GT = "[GT]"
SOURCE_PROTO = "[PROTOTYPE]"
SOURCE_AUTO_PROXY_UNAVAILABLE = "[AUTO proxy unavailable]"

VIEW_NAMES = [
    "pair_only_exact",
    "union_only_exact",
    "full_evidence_crop",
    "union_plus_cpos",
    "union_plus_cneg",
    "full_minus_cpos",
    "full_minus_cneg",
    "role_overlay",
]


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def clamp(self, width: float, height: float) -> "Box":
        return Box(
            max(0.0, min(float(width), self.x1)),
            max(0.0, min(float(height), self.y1)),
            max(0.0, min(float(width), self.x2)),
            max(0.0, min(float(height), self.y2)),
        ).normalized()

    def normalized(self) -> "Box":
        return Box(min(self.x1, self.x2), min(self.y1, self.y2), max(self.x1, self.x2), max(self.y1, self.y2))

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def short_side(self) -> float:
        return min(self.width, self.height)

    @property
    def sqrt_area(self) -> float:
        return math.sqrt(max(0.0, self.area))

    def as_list(self) -> list[float]:
        return [round(self.x1, 3), round(self.y1, 3), round(self.x2, 3), round(self.y2, 3)]

    def to_int_tuple(self) -> tuple[int, int, int, int]:
        return (int(math.floor(self.x1)), int(math.floor(self.y1)), int(math.ceil(self.x2)), int(math.ceil(self.y2)))


@dataclass(frozen=True)
class ContextRole:
    role: str
    context_id: str
    category: str
    box: Box
    source: str
    supported_verb: str
    relation_endpoints: str
    provenance: str


@dataclass(frozen=True)
class CandidateGeometry:
    crop_box: Box
    output_size: tuple[int, int]
    scale: float
    target_pair_occupancy: float
    target_human_projected_height: float
    target_object_projected_sqrt_area: float
    target_object_projected_short_side: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage1-manifest", type=Path, default=DEFAULT_STAGE1_MANIFEST)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--v4-classification", type=Path, default=DEFAULT_V4_CLASSIFICATION)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--sample-limit", type=int, default=30)
    parser.add_argument("--review-width", type=int, default=900)
    parser.add_argument("--outside-dilation-ratio", type=float, default=0.03)
    parser.add_argument("--overwrite-stage2", action="store_true")
    return parser.parse_args()


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


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


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


def box_from_instance(row: Any) -> Box:
    return Box(float(row.bbox_x1), float(row.bbox_y1), float(row.bbox_x2), float(row.bbox_y2)).normalized()


def box_from_pair(row: Any, prefix: str) -> Box:
    return Box(
        float(getattr(row, f"{prefix}_bbox_x1")),
        float(getattr(row, f"{prefix}_bbox_y1")),
        float(getattr(row, f"{prefix}_bbox_x2")),
        float(getattr(row, f"{prefix}_bbox_y2")),
    ).normalized()


def union_rect(boxes: Iterable[Box]) -> Box:
    items = list(boxes)
    return Box(min(b.x1 for b in items), min(b.y1 for b in items), max(b.x2 for b in items), max(b.y2 for b in items)).normalized()


def intersection_area(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def true_pair_occupancy_area(human_box: Box, object_box: Box) -> float:
    """Area of human/object box union as a set, not their enclosing rectangle."""
    return human_box.area + object_box.area - intersection_area(human_box, object_box)


def dilate_box(box: Box, image_width: float, image_height: float, ratio: float = 0.03) -> Box:
    pad = ratio * min(float(image_width), float(image_height))
    return Box(box.x1 - pad, box.y1 - pad, box.x2 + pad, box.y2 + pad).clamp(image_width, image_height)


def outside_fraction(context_box: Box, dilated_pair_region: Box) -> float:
    if context_box.area <= 0:
        return 0.0
    inside = intersection_area(context_box, dilated_pair_region)
    return max(0.0, min(1.0, (context_box.area - inside) / context_box.area))


def parse_context_cue(value: Any) -> Optional[tuple[str, str]]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value)
    if not text or text == "nan" or text.startswith("manual_or_vlm:"):
        return None
    if ":" not in text:
        return None
    category, instance_id = text.split(":", 1)
    if not category or not instance_id:
        return None
    return category, instance_id


def positive_verbs(row: Any) -> list[str]:
    return [str(v) for v in load_json_list(getattr(row, "verb_names_json")) if str(v) != "no_interaction"]


def select_competitors(object_name: str, target_verbs: Sequence[str], object_counts: dict[str, Counter[str]], limit: int = 5) -> list[str]:
    target = set(target_verbs)
    target_families = {verb_family(v) for v in target}
    selected: list[str] = []
    selected_families: set[str] = set()
    for verb, _ in object_counts.get(object_name, Counter()).most_common():
        family = verb_family(str(verb))
        if verb == "no_interaction" or verb in target or family in target_families or family in selected_families:
            continue
        selected.append(str(verb))
        selected_families.add(family)
        if len(selected) >= limit:
            break
    return selected


def build_object_verb_prototypes(pairs: pd.DataFrame, actions: dict[int, str]) -> tuple[list[dict[str, Any]], dict[str, Counter[str]]]:
    object_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in pairs.itertuples(index=False):
        obj = str(row.object_category_name)
        for verb in positive_verbs(row):
            object_counts[obj][verb] += 1

    rows: list[dict[str, Any]] = []
    for obj in sorted(object_counts):
        selected: list[str] = []
        selected_families: set[str] = set()
        observed = [verb for verb, _ in object_counts[obj].most_common() if verb != "no_interaction"]
        for verb in observed:
            family = verb_family(verb)
            if family in selected_families:
                continue
            selected.append(verb)
            selected_families.add(family)
            if len(selected) >= 5:
                break
        rows.append({
            "object_category": obj,
            "source": f"{SOURCE_GT} HICO-DET object/action co-occurrence; {SOURCE_PROTO} family merge",
            "observed_positive_verbs": json_dumps(observed),
            "prototype_verbs_2_to_5": json_dumps(selected[:5]),
            "prototype_verb_families": json_dumps([verb_family(v) for v in selected[:5]]),
            "observed_positive_verb_count": len(observed),
            "all_action_mapping_source": str(actions and "hico_20160224_det/hico_list_vb.txt"),
        })
    return rows, object_counts


def find_pair_by_id(pairs_by_id: dict[str, Any], pair_id: str) -> Optional[Any]:
    return pairs_by_id.get(str(pair_id))


def find_relation_for_context(image_pairs: pd.DataFrame, context_id: str, target_human_id: str) -> tuple[str, str]:
    same_human = image_pairs[
        (image_pairs["canonical_human_id"].astype(str) == target_human_id)
        & (image_pairs["canonical_object_id"].astype(str) == context_id)
    ]
    if not same_human.empty:
        row = same_human.iloc[0]
        return f"{row['canonical_human_id']}->{row['canonical_object_id']}", f"{SOURCE_GT} same target human auxiliary pair {row['pair_id']}"

    any_pair = image_pairs[image_pairs["canonical_object_id"].astype(str) == context_id]
    if not any_pair.empty:
        row = any_pair.iloc[0]
        return f"{row['canonical_human_id']}->{row['canonical_object_id']}", f"{SOURCE_GT} HICO pair {row['pair_id']}"
    return f"context_instance_only:{context_id}", f"{SOURCE_AUTO} canonical instance with no direct target-human relation"


def choose_negative_context(
    v4_row: Any,
    target_row: Any,
    image_pairs: pd.DataFrame,
    instances_by_id: dict[str, Any],
    cpos_id: str,
    target_endpoint_ids: set[str],
    competitors: Sequence[str],
) -> Optional[ContextRole]:
    contrast_object_id = str(getattr(v4_row, "contrast_object_id", ""))
    contrast_pair_id = str(getattr(v4_row, "contrast_pair_id", ""))
    contrast_verbs = [str(v) for v in load_json_list(getattr(v4_row, "contrast_verbs", "[]")) if str(v) != "no_interaction"]
    if contrast_object_id and contrast_object_id not in target_endpoint_ids and contrast_object_id != cpos_id and contrast_object_id in instances_by_id:
        inst = instances_by_id[contrast_object_id]
        endpoints = ""
        provenance = f"{SOURCE_GT} v4 contrast pair {contrast_pair_id}"
        matched = image_pairs[image_pairs["pair_id"].astype(str) == contrast_pair_id]
        if not matched.empty:
            r = matched.iloc[0]
            endpoints = f"{r['canonical_human_id']}->{r['canonical_object_id']}"
        supported = contrast_verbs[0] if contrast_verbs else (competitors[0] if competitors else "unknown_competitor")
        return ContextRole("cneg", contrast_object_id, str(getattr(inst, "category_name", "")), box_from_instance(inst), f"{SOURCE_GT}/v4_contrast", supported, endpoints, provenance)

    target_families = {verb_family(v) for v in positive_verbs(target_row)}
    competitor_set = set(competitors)
    for row in image_pairs.sort_values("pair_id").itertuples(index=False):
        obj_id = str(row.canonical_object_id)
        if obj_id in target_endpoint_ids or obj_id == cpos_id or obj_id not in instances_by_id:
            continue
        verbs = positive_verbs(row)
        if not verbs:
            continue
        different_family = any(verb_family(v) not in target_families for v in verbs)
        supports_competitor = any(v in competitor_set for v in verbs)
        if not (different_family or supports_competitor):
            continue
        inst = instances_by_id[obj_id]
        supported = next((v for v in verbs if v in competitor_set), verbs[0])
        return ContextRole(
            "cneg",
            obj_id,
            str(getattr(inst, "category_name", row.object_category_name)),
            box_from_instance(inst),
            f"{SOURCE_GT}/fallback_external_pair",
            supported,
            f"{row.canonical_human_id}->{row.canonical_object_id}",
            f"{SOURCE_GT} fallback pair {row.pair_id}; second pair allowed but not required",
        )
    return None


def evidence_crop_box(human_box: Box, object_box: Box, cpos_box: Box, cneg_box: Box, image_width: int, image_height: int) -> Box:
    crop = union_rect([human_box, object_box, cpos_box, cneg_box])
    # Adaptively pad tight crops so target occupancy does not exceed the balanced upper bound when possible.
    pair_area = true_pair_occupancy_area(human_box, object_box)
    for ratio in [0.08, 0.14, 0.20, 0.28, 0.36, 0.48, 0.60]:
        pad = ratio * max(crop.width, crop.height, 1.0)
        candidate = Box(crop.x1 - pad, crop.y1 - pad, crop.x2 + pad, crop.y2 + pad).clamp(image_width, image_height)
        if candidate.area <= 0:
            continue
        if pair_area / candidate.area <= 0.62 or candidate.area >= image_width * image_height * 0.98:
            return candidate
    return crop.clamp(image_width, image_height)


def compute_geometry(human_box: Box, object_box: Box, crop_box: Box, review_width: int) -> CandidateGeometry:
    scale = float(review_width) / max(1.0, crop_box.width)
    output_height = max(1, int(round(crop_box.height * scale)))
    occupancy = true_pair_occupancy_area(human_box, object_box) / max(1.0, crop_box.area)
    return CandidateGeometry(
        crop_box=crop_box,
        output_size=(int(review_width), output_height),
        scale=scale,
        target_pair_occupancy=occupancy,
        target_human_projected_height=human_box.height * scale,
        target_object_projected_sqrt_area=object_box.sqrt_area * scale,
        target_object_projected_short_side=object_box.short_side * scale,
    )


def passes_balanced_geometry(geom: CandidateGeometry) -> bool:
    object_ok = geom.target_object_projected_short_side >= 28.0 or geom.target_object_projected_sqrt_area >= 45.0
    return (
        0.20 <= geom.target_pair_occupancy <= 0.62
        and geom.target_human_projected_height >= 130.0
        and geom.target_object_projected_sqrt_area >= 45.0
        and object_ok
    )


def project_box(box: Box, crop_box: Box, scale: float) -> tuple[int, int, int, int]:
    return (
        int(round((box.x1 - crop_box.x1) * scale)),
        int(round((box.y1 - crop_box.y1) * scale)),
        int(round((box.x2 - crop_box.x1) * scale)),
        int(round((box.y2 - crop_box.y1) * scale)),
    )


def crop_image(image: Image.Image, geom: CandidateGeometry) -> Image.Image:
    crop = image.crop(geom.crop_box.to_int_tuple()).convert("RGB")
    return crop.resize(geom.output_size, Image.Resampling.LANCZOS)


def mask_for_boxes(size: tuple[int, int], boxes: Sequence[tuple[int, int, int, int]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for box in boxes:
        draw.rectangle(box, fill=255)
    return mask


def retained_view(base: Image.Image, retain_boxes: Sequence[tuple[int, int, int, int]]) -> Image.Image:
    out = Image.new("RGB", base.size, (128, 128, 128))
    out.paste(base, (0, 0), mask_for_boxes(base.size, retain_boxes))
    return out


def minus_view(base: Image.Image, remove_box: tuple[int, int, int, int]) -> Image.Image:
    out = base.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(remove_box, fill=(128, 128, 128))
    return out


def draw_overlay(base: Image.Image, labelled_boxes: Sequence[tuple[tuple[int, int, int, int], str, tuple[int, int, int]]]) -> Image.Image:
    out = base.copy()
    draw = ImageDraw.Draw(out)
    for box, label, color in labelled_boxes:
        draw.rectangle(box, outline=color, width=4)
        draw.text((box[0] + 4, max(0, box[1] - 14)), label, fill=color)
    return out


def save_context_mask(path: Path, size: tuple[int, int], box: tuple[int, int, int, int]) -> None:
    mask = mask_for_boxes(size, [box])
    mask.save(path)


def generate_views(
    image_path: Path,
    output_dir: Path,
    human_box: Box,
    object_box: Box,
    cpos: ContextRole,
    cneg: ContextRole,
    geom: CandidateGeometry,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as raw:
        base = crop_image(raw.convert("RGB"), geom)

    h_proj = project_box(human_box, geom.crop_box, geom.scale)
    o_proj = project_box(object_box, geom.crop_box, geom.scale)
    union_proj = project_box(union_rect([human_box, object_box]), geom.crop_box, geom.scale)
    cpos_proj = project_box(cpos.box, geom.crop_box, geom.scale)
    cneg_proj = project_box(cneg.box, geom.crop_box, geom.scale)

    views = {
        "pair_only_exact": retained_view(base, [h_proj, o_proj]),
        "union_only_exact": retained_view(base, [union_proj]),
        "full_evidence_crop": base,
        "union_plus_cpos": retained_view(base, [union_proj, cpos_proj]),
        "union_plus_cneg": retained_view(base, [union_proj, cneg_proj]),
        "full_minus_cpos": minus_view(base, cpos_proj),
        "full_minus_cneg": minus_view(base, cneg_proj),
        "role_overlay": draw_overlay(
            base,
            [
                (h_proj, "target human", (0, 180, 0)),
                (o_proj, "target object", (220, 0, 0)),
                (cpos_proj, "c+", (0, 80, 255)),
                (cneg_proj, "c-", (255, 150, 0)),
            ],
        ),
    }
    paths: dict[str, str] = {}
    for name, view in views.items():
        path = output_dir / f"{name}.jpg"
        view.save(path, quality=92)
        paths[f"{name}_path"] = rel(path)
    cpos_mask = output_dir / "cpos_mask.png"
    cneg_mask = output_dir / "cneg_mask.png"
    save_context_mask(cpos_mask, base.size, cpos_proj)
    save_context_mask(cneg_mask, base.size, cneg_proj)
    paths["cpos_mask_path"] = rel(cpos_mask)
    paths["cneg_mask_path"] = rel(cneg_mask)
    return paths


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def context_visible_pair_only(context_box: Box, human_box: Box, object_box: Box) -> bool:
    return intersection_area(context_box, human_box) > 0 or intersection_area(context_box, object_box) > 0


def build_feature_row(
    v4_row: Any,
    target_row: Any,
    image_path: Path,
    human_box: Box,
    object_box: Box,
    cpos: ContextRole,
    cneg: ContextRole,
    geom: CandidateGeometry,
    competitors: list[str],
    outside_pair_region: Box,
    view_paths: dict[str, str],
) -> dict[str, Any]:
    gt_verbs = positive_verbs(target_row)
    union = union_rect([human_box, object_box])
    cpos_out = outside_fraction(cpos.box, outside_pair_region)
    cneg_out = outside_fraction(cneg.box, outside_pair_region)
    return {
        "source_stage": "stage2_features",
        "candidate_id": str(getattr(v4_row, "candidate_id")),
        "pair_id": str(target_row.pair_id),
        "image_key": str(target_row.image_key),
        "image_path": rel(image_path),
        "target_human_id": str(target_row.canonical_human_id),
        "target_object_id": str(target_row.canonical_object_id),
        "target_object_category": str(target_row.object_category_name),
        "gt_verb_source": f"{SOURCE_GT} HICO-DET canonical annotation",
        "gt_verbs": json_dumps(gt_verbs),
        "competition_source": f"{SOURCE_GT} same-object local mapping + {SOURCE_PROTO} family merge",
        "competing_verbs_2_to_5": json_dumps(competitors),
        "competing_verb_families": json_dumps([verb_family(v) for v in competitors]),
        "human_bbox": json_dumps(human_box.as_list()),
        "object_bbox": json_dumps(object_box.as_list()),
        "union_bbox": json_dumps(union.as_list()),
        "evidence_crop_bbox": json_dumps(geom.crop_box.as_list()),
        "review_width_px": geom.output_size[0],
        "review_height_px": geom.output_size[1],
        "target_pair_occupancy_source": f"{SOURCE_AUTO} true area(human_box union object_box) / crop_area",
        "target_pair_occupancy": round(geom.target_pair_occupancy, 6),
        "target_human_projected_height_px": round(geom.target_human_projected_height, 3),
        "target_object_projected_sqrt_area_px": round(geom.target_object_projected_sqrt_area, 3),
        "target_object_projected_short_side_px": round(geom.target_object_projected_short_side, 3),
        "balanced_size_pass": passes_balanced_geometry(geom),
        "cpos_id": cpos.context_id,
        "cpos_category": cpos.category,
        "cpos_bbox": json_dumps(cpos.box.as_list()),
        "cpos_source": cpos.source,
        "cpos_supported_verb": cpos.supported_verb,
        "cpos_outside_fraction": round(cpos_out, 6),
        "cpos_visible_in_pair_only": context_visible_pair_only(cpos.box, human_box, object_box),
        "cpos_visible_in_union_only": intersection_area(cpos.box, union) > 0,
        "cpos_relation_endpoints": cpos.relation_endpoints,
        "cpos_provenance": cpos.provenance,
        "cneg_id": cneg.context_id,
        "cneg_category": cneg.category,
        "cneg_bbox": json_dumps(cneg.box.as_list()),
        "cneg_source": cneg.source,
        "cneg_supported_verb": cneg.supported_verb,
        "cneg_outside_fraction": round(cneg_out, 6),
        "cneg_visible_in_pair_only": context_visible_pair_only(cneg.box, human_box, object_box),
        "cneg_visible_in_union_only": intersection_area(cneg.box, union) > 0,
        "cneg_relation_endpoints": cneg.relation_endpoints,
        "cneg_provenance": cneg.provenance,
        "cpos_cneg_independent": cpos.context_id != cneg.context_id,
        "M_pair": "",
        "M_pair_source": SOURCE_AUTO_PROXY_UNAVAILABLE,
        "M_union": "",
        "M_union_source": SOURCE_AUTO_PROXY_UNAVAILABLE,
        "G_pos": "",
        "G_pos_source": SOURCE_AUTO_PROXY_UNAVAILABLE,
        "G_neg": "",
        "G_neg_source": SOURCE_AUTO_PROXY_UNAVAILABLE,
        "manual_pair_only_ambiguity_0_4": "",
        "manual_union_only_ambiguity_0_4": "",
        "manual_positive_context_gain_0_6": "",
        "manual_misleading_plausibility_0_4": "",
        "manual_key_distractor_selectivity_0_5": "",
        **view_paths,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_context_schema(path: Path) -> None:
    schema = {
        "schema_name": "EvidenceHOI Figure1 v5 context role schema",
        "source_labels": {
            "[GT]": "benchmark annotation only",
            "[AUTO]": "geometry/code proxy, deterministic measurement, generated masks/views",
            "[PROTOTYPE]": "object-verb mapping or family merge",
            "[AUTO proxy unavailable]": "reserved CLIP/VLM margin field not computed in this environment",
        },
        "context_roles": {
            "cpos": "positive external context supporting the target GT verb",
            "cneg": "independent misleading external context supporting a competitor or wrong prior",
        },
        "required_context_fields": [
            "context_id", "bbox", "mask_path", "source", "supported_verb", "outside_fraction",
            "visible_in_pair_only", "visible_in_union_only", "relation_endpoints", "provenance",
        ],
        "externality_rule": {
            "dilated_R_pair": "human-object union rectangle dilated by outside_dilation_ratio of image short side",
            "balanced_outside_fraction_min": 0.50,
        },
        "counterfactual_views": VIEW_NAMES,
        "view_policy": "All views for a candidate share the exact same crop box, scale, and output resolution; only masking/restoring original pixels is allowed.",
    }
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_spot_checks(feature_rows: list[dict[str, Any]], output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in feature_rows:
        view_sizes = {}
        for view in VIEW_NAMES:
            with Image.open(PROJECT_ROOT / row[f"{view}_path"]) as img:
                view_sizes[view] = img.size
        same_canvas = len(set(view_sizes.values())) == 1
        checks = {
            "target_human_not_cpos": row["target_human_id"] != row["cpos_id"],
            "target_object_not_cpos": row["target_object_id"] != row["cpos_id"],
            "target_human_not_cneg": row["target_human_id"] != row["cneg_id"],
            "target_object_not_cneg": row["target_object_id"] != row["cneg_id"],
            "cpos_cneg_distinct": row["cpos_id"] != row["cneg_id"],
            "pair_only_no_external_context_intersection": not bool(row["cpos_visible_in_pair_only"]) and not bool(row["cneg_visible_in_pair_only"]),
            "union_only_no_union_external_context_intersection": not bool(row["cpos_visible_in_union_only"]) and not bool(row["cneg_visible_in_union_only"]),
            "second_pair_optional": True,
            "auto_labels_present": "[AUTO]" in str(row["target_pair_occupancy_source"]),
            "same_canvas_all_views": same_canvas,
        }
        rows.append({
            "pair_id": row["pair_id"],
            "image_key": row["image_key"],
            "all_checks_pass": all(checks.values()),
            **checks,
            "view_size": json_dumps(list(next(iter(view_sizes.values())))),
            "role_overlay_path": row["role_overlay_path"],
        })
    summary = {
        "spot_check_count": len(rows),
        "all_pass_count": sum(1 for r in rows if r["all_checks_pass"]),
        "failed_pair_ids": [r["pair_id"] for r in rows if not r["all_checks_pass"]],
    }
    write_csv(output_dir / "spot_check_results.csv", rows)
    (output_dir / "spot_check_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, summary


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(output_root: Path, stage2_dir: Path, output_files: Sequence[Path], feature_count: int, spot_summary: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "stage": 2,
        "task": "implement v5 retrieval features and standardized counterfactual views",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": rel(output_root),
        "stage2_dir": rel(stage2_dir),
        "stopped_after_manifest": True,
        "full_retrieval_performed": False,
        "clip_vlm_run": False,
        "feature_sample_count": feature_count,
        "spot_check_summary": spot_summary,
        "outputs": [],
    }
    for path in output_files:
        manifest["outputs"].append({"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return manifest


def main() -> int:
    args = parse_args()
    output_root = args.output_root
    stage2_dir = output_root / "stage2_features"
    if stage2_dir.exists():
        if not args.overwrite_stage2:
            raise SystemExit(f"Refusing to overwrite existing stage2 directory: {stage2_dir}")
        shutil.rmtree(stage2_dir)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    views_dir = stage2_dir / "counterfactual_views"
    spot_dir = stage2_dir / "mask_spot_checks"
    views_dir.mkdir(parents=True, exist_ok=True)
    spot_dir.mkdir(parents=True, exist_ok=True)

    if not args.stage1_manifest.exists():
        raise FileNotFoundError(args.stage1_manifest)
    stage1_manifest = json.loads(args.stage1_manifest.read_text(encoding="utf-8"))

    actions = read_action_names(args.actions)
    pairs = pd.read_parquet(args.index_dir / "pairs.parquet")
    instances = pd.read_parquet(args.index_dir / "instances.parquet")
    images = pd.read_parquet(args.index_dir / "image_summary.parquet")
    v4 = pd.read_csv(args.v4_classification, low_memory=False)

    valid_pairs = pairs[(pairs["valid"] == True) & (pairs["eligible_for_final"] == True) & (pairs["invisible"] == False)].copy()
    prototype_rows, object_counts = build_object_verb_prototypes(valid_pairs, actions)
    write_csv(stage2_dir / "verb_prototypes.csv", prototype_rows)
    write_context_schema(stage2_dir / "context_schema.json")

    pairs_by_id = {str(row.pair_id): row for row in valid_pairs.itertuples(index=False)}
    images_by_key = {str(row.image_key): row for row in images.itertuples(index=False)}
    instances_by_image = {
        str(image_key): {str(row.canonical_instance_id): row for row in df.itertuples(index=False)}
        for image_key, df in instances.groupby("image_key", sort=False)
    }
    pairs_by_image = {str(image_key): df for image_key, df in valid_pairs.groupby("image_key", sort=False)}

    v4_candidates = v4[v4["concrete_context_cue"].notna()].copy()
    v4_candidates = v4_candidates.sort_values(["pool", "score"], ascending=[True, False])

    feature_rows: list[dict[str, Any]] = []
    skipped = Counter()
    for v4_row in v4_candidates.itertuples(index=False):
        if len(feature_rows) >= args.sample_limit:
            break
        target_row = find_pair_by_id(pairs_by_id, str(v4_row.pair_id))
        if target_row is None:
            skipped["missing_target_pair"] += 1
            continue
        gt_verbs = positive_verbs(target_row)
        if not gt_verbs:
            skipped["missing_gt_verb"] += 1
            continue
        competitors = select_competitors(str(target_row.object_category_name), gt_verbs, object_counts, limit=5)
        if len(competitors) < 2:
            skipped["competition_lt_2"] += 1
            continue
        cue = parse_context_cue(getattr(v4_row, "concrete_context_cue"))
        if cue is None:
            skipped["no_concrete_cpos_box"] += 1
            continue
        cpos_category, cpos_id = cue
        target_endpoint_ids = {str(target_row.canonical_human_id), str(target_row.canonical_object_id)}
        if cpos_id in target_endpoint_ids:
            skipped["cpos_is_target_endpoint"] += 1
            continue
        image_key = str(target_row.image_key)
        image_info = images_by_key.get(image_key)
        if image_info is None:
            skipped["missing_image_info"] += 1
            continue
        image_path = Path(str(image_info.image_path))
        if not image_path.exists():
            skipped["missing_image_file"] += 1
            continue
        inst_map = instances_by_image.get(image_key, {})
        if cpos_id not in inst_map:
            skipped["missing_cpos_instance"] += 1
            continue
        image_pairs = pairs_by_image.get(image_key, valid_pairs.iloc[0:0])
        cpos_rel, cpos_provenance = find_relation_for_context(image_pairs, cpos_id, str(target_row.canonical_human_id))
        cpos = ContextRole(
            "cpos",
            cpos_id,
            cpos_category,
            box_from_instance(inst_map[cpos_id]),
            f"{SOURCE_GT}/v4_concrete_context_cue",
            gt_verbs[0],
            cpos_rel,
            cpos_provenance,
        )
        cneg = choose_negative_context(v4_row, target_row, image_pairs, inst_map, cpos_id, target_endpoint_ids, competitors)
        if cneg is None:
            skipped["missing_independent_cneg"] += 1
            continue
        if cneg.context_id == cpos.context_id or cneg.context_id in target_endpoint_ids:
            skipped["invalid_cneg_identity"] += 1
            continue

        human_box = box_from_pair(target_row, "human")
        object_box = box_from_pair(target_row, "object")
        pair_union = union_rect([human_box, object_box])
        dilated_pair = dilate_box(pair_union, float(image_info.width), float(image_info.height), args.outside_dilation_ratio)
        if outside_fraction(cpos.box, dilated_pair) < 0.50 or outside_fraction(cneg.box, dilated_pair) < 0.50:
            skipped["outside_fraction_below_balanced"] += 1
            continue
        if context_visible_pair_only(cpos.box, human_box, object_box) or context_visible_pair_only(cneg.box, human_box, object_box):
            skipped["pair_only_context_intersection"] += 1
            continue
        if intersection_area(cpos.box, pair_union) > 0 or intersection_area(cneg.box, pair_union) > 0:
            skipped["union_only_context_intersection"] += 1
            continue

        crop = evidence_crop_box(human_box, object_box, cpos.box, cneg.box, int(image_info.width), int(image_info.height))
        geom = compute_geometry(human_box, object_box, crop, args.review_width)
        if not passes_balanced_geometry(geom):
            skipped["balanced_size_fail"] += 1
            continue

        candidate_dir = views_dir / f"{len(feature_rows) + 1:03d}_{str(target_row.split)}_{str(target_row.pair_id).replace(':', '_')}"
        paths = generate_views(image_path, candidate_dir, human_box, object_box, cpos, cneg, geom)
        feature_rows.append(build_feature_row(v4_row, target_row, image_path, human_box, object_box, cpos, cneg, geom, competitors, dilated_pair, paths))

    if len(feature_rows) < args.sample_limit:
        raise RuntimeError(f"Only generated {len(feature_rows)} feature rows; skipped={dict(skipped)}")

    feature_sample_path = stage2_dir / "feature_sample.csv"
    write_csv(feature_sample_path, feature_rows)
    spot_rows, spot_summary = run_spot_checks(feature_rows[:30], spot_dir)
    if spot_summary["all_pass_count"] != 30:
        raise RuntimeError(f"Mask spot checks failed: {spot_summary}")

    report = f"""# Stage 2 Implementation Report

## Inputs

- Stage 1 manifest: `{rel(args.stage1_manifest)}`
- Pair index: `{rel(args.index_dir / 'pairs.parquet')}`
- Instance index: `{rel(args.index_dir / 'instances.parquet')}`
- Image summary: `{rel(args.index_dir / 'image_summary.parquet')}`
- Candidate proxy source: `{rel(args.v4_classification)}`
- Action mapping: `{rel(args.actions)}`

## Implemented Features

- Target pair occupancy uses `area(human_box union object_box as sets) / crop_area`, not the human-object enclosing rectangle.
- Competition prototypes are read from the local HICO action list and v2 canonical object/action mapping, excluding `no_interaction` and merging visual verb families.
- Generated views: `{', '.join(VIEW_NAMES)}`.
- All views for a candidate share the same crop box, scale, and output resolution.
- Context externality uses outside fraction against a human-object union rectangle dilated by `{args.outside_dilation_ratio:.3f}` of image short side.
- CLIP/VLM margins are not computed in this environment and are labelled `{SOURCE_AUTO_PROXY_UNAVAILABLE}`.
- Manual scoring fields are present and blank for later image-capable review.

## Generated Sample

- Feature rows: {len(feature_rows)}
- Spot-checked rows: {len(spot_rows)}
- Spot-check all-pass rows: {spot_summary['all_pass_count']}

## Skip Counters

```json
{json.dumps(dict(skipped), ensure_ascii=False, indent=2)}
```

## Source Discipline

No automatic or generated context is written as `[GT]` unless it comes directly from the benchmark annotation/index. No EvidenceHOI attention, probability, or causal effect is claimed.
"""
    implementation_report_path = stage2_dir / "implementation_report.md"
    implementation_report_path.write_text(report, encoding="utf-8")

    test_report = f"""Stage 2 spot-check report
=========================
feature_rows={len(feature_rows)}
spot_check_count={spot_summary['spot_check_count']}
all_pass_count={spot_summary['all_pass_count']}
failed_pair_ids={json_dumps(spot_summary['failed_pair_ids'])}
pytest_status=run separately by tests/test_evidencehoi_v5_features.py
clip_vlm_status={SOURCE_AUTO_PROXY_UNAVAILABLE}
all_auto_results_labelled=True
"""
    test_report_path = stage2_dir / "test_report.txt"
    test_report_path.write_text(test_report, encoding="utf-8")

    output_files = [
        stage2_dir / "verb_prototypes.csv",
        stage2_dir / "context_schema.json",
        feature_sample_path,
        spot_dir / "spot_check_results.csv",
        spot_dir / "spot_check_summary.json",
        implementation_report_path,
        test_report_path,
    ]
    manifest = build_manifest(output_root, stage2_dir, output_files, len(feature_rows), spot_summary)
    manifest["stage1_manifest"] = stage1_manifest
    manifest_path = output_root / "manifest_stage2.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": rel(manifest_path), "feature_rows": len(feature_rows), "spot_summary": spot_summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
