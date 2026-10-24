#!/usr/bin/env python3
"""Rank real EvidenceHOI traces for supplementary mechanism figures.

The score is only a mining heuristic. It combines correct-GT filtering,
source-image readability, context quality, and the legibility of learned soft
path-role correspondence. It never treats attention or evidence weights as
ground-truth or causal evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image, ImageFilter, ImageStat

import rank_correct_context_evidence as base_ranker
from render_evidence_shortlist import load_name_map, load_trace, resolve_image


FINAL_ROLE_LABELS = ["H", "O", "A", "K", "S", "C"]
EPS = 1e-12


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalized_entropy(values: torch.Tensor) -> float:
    values = values.detach().float().flatten().clamp(min=0)
    total = values.sum()
    if values.numel() <= 1 or float(total.item()) <= EPS:
        return 0.0
    probs = values / total
    entropy = -(probs * probs.clamp(min=EPS).log()).sum()
    return float((entropy / math.log(values.numel())).item())


def mean(values: Sequence[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def tensor_shape(value: Any) -> Tuple[int, ...]:
    return tuple(int(v) for v in value.shape) if isinstance(value, torch.Tensor) else ()


def trace_profile(record: Mapping[str, Any]) -> Dict[str, Any]:
    graph = record.get("graph") or {}
    selected_context = graph.get("selected_context_indices")
    selected_hoi = graph.get("selected_hoi_mask")
    slot_mask = graph.get("prototype_slot_mask")
    layer_idx = int(graph.get("layer_idx", -1))
    return {
        "m": int(selected_context.shape[-1]) if isinstance(selected_context, torch.Tensor) else -1,
        "k": int(selected_hoi.shape[-1]) if isinstance(selected_hoi, torch.Tensor) else -1,
        "roles": int(slot_mask.shape[-1]) if isinstance(slot_mask, torch.Tensor) else -1,
        "layers": layer_idx + 1 if layer_idx >= 0 else -1,
        "stage": str(graph.get("stage", "")),
        "raw_role_names": [str(value) for value in graph.get("prototype_slot_names", [])],
        "has_layerwise_trace": bool(record.get("graph_layers")),
    }


def profile_compatible(
    profile: Mapping[str, Any],
    *,
    expected_m: int,
    expected_k: int,
    expected_roles: int,
    expected_layers: int,
) -> bool:
    checks = [
        int(profile.get("m", -1)) == expected_m,
        int(profile.get("k", -1)) == expected_k,
        int(profile.get("roles", -1)) == expected_roles,
    ]
    observed_layers = int(profile.get("layers", -1))
    if observed_layers >= 0 and expected_layers > 0:
        checks.append(observed_layers == expected_layers)
    return all(checks)


def mechanism_metrics(
    record: Mapping[str, Any],
    pair_idx: int,
    prototype_row: int,
) -> Dict[str, Any]:
    graph = record.get("graph") or {}
    required = [
        "candidate_evidence_mask",
        "prototype_slot_mask",
        "sync_evidence_weights",
        "sync_prototype_weights",
        "sync_prototype_coverage",
    ]
    missing = [name for name in required if not isinstance(graph.get(name), torch.Tensor)]
    if missing:
        return {"mechanism_valid": False, "mechanism_error": "missing:" + ",".join(missing)}

    visual_to_text = graph.get("sync_visual_to_text_attn")
    text_to_visual = graph.get("sync_text_to_visual_attn")
    if not isinstance(visual_to_text, torch.Tensor) and isinstance(text_to_visual, torch.Tensor):
        visual_to_text = text_to_visual.transpose(-1, -2)
    if not isinstance(text_to_visual, torch.Tensor) and isinstance(visual_to_text, torch.Tensor):
        text_to_visual = visual_to_text.transpose(-1, -2)
    if not isinstance(visual_to_text, torch.Tensor) or not isinstance(text_to_visual, torch.Tensor):
        return {"mechanism_valid": False, "mechanism_error": "missing correspondence tensors"}

    v2t = visual_to_text[pair_idx, prototype_row].float()
    t2v = text_to_visual[pair_idx, prototype_row].float()
    visual_mask = graph["candidate_evidence_mask"][pair_idx].bool()
    role_mask = graph["prototype_slot_mask"][pair_idx, prototype_row].bool()
    alpha = graph["sync_evidence_weights"][pair_idx, prototype_row].float()
    pi = graph["sync_prototype_weights"][pair_idx].float()
    coverage = graph["sync_prototype_coverage"][pair_idx, prototype_row].float()
    if v2t.ndim != 2 or t2v.ndim != 2:
        return {"mechanism_valid": False, "mechanism_error": "correspondence tensor must be rank 2"}
    if v2t.shape != (visual_mask.numel(), role_mask.numel()):
        return {
            "mechanism_valid": False,
            "mechanism_error": (
                f"shape mismatch v2t={tuple(v2t.shape)} visual={visual_mask.numel()} roles={role_mask.numel()}"
            ),
        }

    valid_visual_ids = torch.nonzero(visual_mask, as_tuple=False).flatten().tolist()
    valid_role_ids = torch.nonzero(role_mask, as_tuple=False).flatten().tolist()
    row_concentrations = [
        1.0 - normalized_entropy(v2t[row, role_mask])
        for row in valid_visual_ids
    ]
    role_concentrations = [
        1.0 - normalized_entropy(t2v[role, visual_mask])
        for role in valid_role_ids
    ]
    correspondence_concentration = 0.5 * (
        mean(row_concentrations) + mean(role_concentrations)
    )

    top_units = [
        int(torch.argmax(t2v[role].masked_fill(~visual_mask, -1e4)).item())
        for role in valid_role_ids
    ]
    role_top_unit_diversity = len(set(top_units)) / max(len(top_units), 1)
    expected_visual_for_role = {0: 1, 1: 2, 2: 0}
    endpoint_values = [
        float(t2v[role, visual].item())
        for role, visual in expected_visual_for_role.items()
        if role < role_mask.numel()
        and visual < visual_mask.numel()
        and bool(role_mask[role])
        and bool(visual_mask[visual])
    ]
    visual_unit_baseline = 1.0 / max(len(valid_visual_ids), 1)
    endpoint_role_support_raw = mean(endpoint_values)
    endpoint_role_support = clamp01(
        (endpoint_role_support_raw - visual_unit_baseline) / max(1.0 - visual_unit_baseline, EPS)
    )

    context_mask = visual_mask.clone()
    context_mask[: min(3, context_mask.numel())] = False
    context_role_values: Dict[str, float] = {}
    for role_idx, label in zip((3, 4, 5), ("key", "scene", "context_object")):
        if role_idx < role_mask.numel() and bool(role_mask[role_idx]) and bool(context_mask.any()):
            context_role_values[label] = float(t2v[role_idx, context_mask].sum().item())
        else:
            context_role_values[label] = 0.0
    context_baseline = float(context_mask.sum().item()) / max(float(visual_mask.sum().item()), 1.0)
    context_role_excess = {
        key: clamp01((value - context_baseline) / max(1.0 - context_baseline, EPS))
        for key, value in context_role_values.items()
    }
    context_role_support_raw = mean(list(context_role_values.values()))
    context_role_support = mean(list(context_role_excess.values()))

    valid_alpha = alpha[visual_mask]
    alpha_context = float(alpha[context_mask].sum().item()) if bool(context_mask.any()) else 0.0
    alpha_total = float(valid_alpha.sum().item())
    valid_coverage = coverage[role_mask]
    coverage_values = [float(value) for value in valid_coverage.tolist()]
    coverage_mean_raw = mean(coverage_values)
    coverage_mean = clamp01(
        (coverage_mean_raw - visual_unit_baseline) / max(1.0 - visual_unit_baseline, EPS)
    )
    valid_pi = pi[pi >= 0]
    sorted_pi = torch.sort(valid_pi, descending=True).values
    pi_margin = (
        float((sorted_pi[0] - sorted_pi[1]).item())
        if sorted_pi.numel() > 1 else float(sorted_pi[0].item())
    )
    anchor_delta = graph.get("sync_anchor_delta")
    anchor_delta_norm = (
        float(anchor_delta[pair_idx].float().norm().item())
        if isinstance(anchor_delta, torch.Tensor) else 0.0
    )
    mechanism_score = (
        0.24 * endpoint_role_support
        + 0.28 * context_role_support
        + 0.18 * correspondence_concentration
        + 0.15 * role_top_unit_diversity * correspondence_concentration
        + 0.15 * coverage_mean
    )
    return {
        "mechanism_valid": True,
        "mechanism_error": "",
        "visual_unit_count": int(visual_mask.sum().item()),
        "role_count": int(role_mask.sum().item()),
        "correspondence_concentration": correspondence_concentration,
        "role_top_unit_diversity": role_top_unit_diversity,
        "visual_unit_uniform_baseline": visual_unit_baseline,
        "endpoint_role_support_raw": endpoint_role_support_raw,
        "endpoint_role_support": endpoint_role_support,
        "key_context_support": context_role_values["key"],
        "scene_context_support": context_role_values["scene"],
        "context_object_support": context_role_values["context_object"],
        "key_context_excess": context_role_excess["key"],
        "scene_context_excess": context_role_excess["scene"],
        "context_object_excess": context_role_excess["context_object"],
        "context_role_uniform_baseline": context_baseline,
        "context_role_support_raw": context_role_support_raw,
        "context_role_support": context_role_support,
        "coverage_mean_raw": coverage_mean_raw,
        "coverage_mean": coverage_mean,
        "coverage_min_raw": min(coverage_values, default=0.0),
        "alpha_context_sum": alpha_context,
        "alpha_context_percent": 100.0 * alpha_context / max(alpha_total, EPS),
        "predicted_prototype_pi": float(pi[prototype_row].item()),
        "prototype_pi_margin": pi_margin,
        "prototype_pi_entropy": normalized_entropy(valid_pi),
        "anchor_delta_norm": anchor_delta_norm,
        "mechanism_score": clamp01(mechanism_score),
    }


def photometric_metrics(image_path: Optional[Path]) -> Dict[str, Any]:
    if image_path is None:
        return {
            "source_image_found": False,
            "photometric_score": 0.0,
            "image_width": 0,
            "image_height": 0,
        }
    with Image.open(image_path) as source:
        image = source.convert("L")
        width, height = image.size
        image.thumbnail((320, 320), Image.Resampling.LANCZOS)
        stat = ImageStat.Stat(image)
        mean_luma = float(stat.mean[0]) / 255.0
        contrast = clamp01(float(stat.stddev[0]) / 64.0)
        histogram = image.histogram()
        pixels = max(sum(histogram), 1)
        clipped = (sum(histogram[:8]) + sum(histogram[248:])) / pixels
        exposure = clamp01(1.0 - 2.5 * clipped)
        edges = image.filter(ImageFilter.FIND_EDGES)
        sharpness = clamp01(float(ImageStat.Stat(edges).mean[0]) / 32.0)
        resolution = clamp01(min(width, height) / 480.0)
        score = 0.30 * exposure + 0.25 * contrast + 0.25 * sharpness + 0.20 * resolution
        return {
            "source_image_found": True,
            "source_image_path": str(image_path),
            "photometric_score": score,
            "image_width": width,
            "image_height": height,
            "mean_luminance": mean_luma,
            "contrast_score": contrast,
            "exposure_score": exposure,
            "sharpness_proxy": sharpness,
            "resolution_score": resolution,
        }


def competitor_metrics(record: Mapping[str, Any], pair_idx: int, gt_ids: Sequence[int]) -> Dict[str, Any]:
    pairs = record["pairs"]
    final = pairs["action_probs"][pair_idx].float()
    coarse = pairs["coarse_action_probs"][pair_idx].float()
    allowed = torch.ones_like(final, dtype=torch.bool)
    for action_id in gt_ids:
        if 0 <= int(action_id) < allowed.numel():
            allowed[int(action_id)] = False
    competitor_id = int(final.masked_fill(~allowed, -1).argmax().item())
    best_gt_id = max(gt_ids, key=lambda idx: float(final[int(idx)].item()))
    return {
        "best_gt_action_id": int(best_gt_id),
        "best_gt_coarse_probability": float(coarse[int(best_gt_id)].item()),
        "best_gt_final_probability": float(final[int(best_gt_id)].item()),
        "best_gt_refinement_gain": float((final[int(best_gt_id)] - coarse[int(best_gt_id)]).item()),
        "competitor_action_id": competitor_id,
        "competitor_coarse_probability": float(coarse[competitor_id].item()),
        "competitor_final_probability": float(final[competitor_id].item()),
        "correct_competitor_margin": float((final[int(best_gt_id)] - final[competitor_id]).item()),
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_diverse(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int,
    per_action: int,
    per_object: int,
) -> List[Dict[str, Any]]:
    action_counts: Counter = Counter()
    object_counts: Counter = Counter()
    used_images = set()
    selected: List[Dict[str, Any]] = []
    for row in rows:
        image_key = str(row["file_name"])
        action_key = str(row["top_action_name"])
        object_key = str(row["object_name"])
        if image_key in used_images:
            continue
        if action_counts[action_key] >= per_action or object_counts[object_key] >= per_object:
            continue
        selected.append(dict(row))
        used_images.add(image_key)
        action_counts[action_key] += 1
        object_counts[object_key] += 1
        if len(selected) >= limit:
            break
    for rank, row in enumerate(selected, start=1):
        row["shortlist_rank"] = rank
    return selected


def enrich_rows(
    rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    action_names: Mapping[int, str],
) -> List[Dict[str, Any]]:
    image_root = Path(args.image_root) if args.image_root else None
    enriched: List[Dict[str, Any]] = []
    for index, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        record = load_trace(Path(row["record_path"]))
        pair_idx = int(row["pair_idx"])
        prototype_row = int(row["top_action_prototype_row"])
        profile = trace_profile(record)
        compatible = profile_compatible(
            profile,
            expected_m=args.expected_m,
            expected_k=args.expected_k,
            expected_roles=args.expected_roles,
            expected_layers=args.expected_layers,
        )
        row.update({f"trace_{key}": value for key, value in profile.items() if key != "raw_role_names"})
        row["trace_raw_role_names"] = ";".join(profile["raw_role_names"])
        row["config_compatible"] = compatible
        row.update(mechanism_metrics(record, pair_idx, prototype_row))
        gt_ids = [int(value) for value in str(row["gt_action_ids"]).split(";") if value != ""]
        pred = competitor_metrics(record, pair_idx, gt_ids)
        pred["competitor_action_name"] = action_names.get(
            int(pred["competitor_action_id"]), str(pred["competitor_action_id"])
        )
        row.update(pred)

        image_path = (
            resolve_image(record, image_root, args.image_pattern)
            if image_root is not None else None
        )
        photo = photometric_metrics(image_path)
        row.update(photo)
        source_quality = (
            float(photo["photometric_score"])
            if photo["source_image_found"] else float(row["visual_quality"])
        )
        gain_score = clamp01(0.5 + 2.5 * float(row["best_gt_refinement_gain"]))
        margin_score = clamp01(0.5 + float(row["correct_competitor_margin"]))
        prediction_score = (
            0.55 * clamp01(float(row["best_gt_final_probability"]))
            + 0.20 * gain_score
            + 0.25 * margin_score
        )
        row["prediction_legibility_score"] = prediction_score
        row["figure_score"] = (
            0.30 * float(row["paper_score"])
            + 0.40 * float(row.get("mechanism_score", 0.0))
            + 0.20 * source_quality
            + 0.10 * prediction_score
        )
        row["scientifically_eligible"] = bool(
            row.get("eligible", False)
            and row.get("mechanism_valid", False)
            and float(row.get("mechanism_score", 0.0)) >= args.min_mechanism_score
            and (compatible or not args.strict_config)
        )
        row["candidate_id"] = (
            f"{Path(str(row['file_name'])).stem}__p{pair_idx}"
            f"__a{int(row['top_action_id'])}"
        )
        enriched.append(row)
        if index % 50 == 0 or index == len(rows):
            print(f"Enriched {index}/{len(rows)} correct-pair candidates")
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--records-glob", default="records/*.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action-names-json", required=True)
    parser.add_argument("--object-names-json", required=True)
    parser.add_argument("--node-names-json", required=True)
    parser.add_argument("--image-root")
    parser.add_argument("--image-pattern")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-endpoint-iou", type=float, default=0.65)
    parser.add_argument("--min-informative-context-count", type=int, default=1)
    parser.add_argument("--min-informative-context-score", type=float, default=0.15)
    parser.add_argument("--min-informative-context-weight", type=float, default=0.0)
    parser.add_argument("--require-informative-two-hop", action="store_true")
    parser.add_argument("--min-mechanism-score", type=float, default=0.05)
    parser.add_argument("--no-require-gt-pair-matched", dest="require_gt_pair_matched", action="store_false")
    parser.set_defaults(require_gt_pair_matched=True)
    parser.add_argument("--expected-m", type=int, default=4)
    parser.add_argument("--expected-k", type=int, default=5)
    parser.add_argument("--expected-roles", type=int, default=6)
    parser.add_argument("--expected-layers", type=int, default=2)
    parser.add_argument("--strict-config", action="store_true")
    parser.add_argument("--shortlist-size", type=int, default=12)
    parser.add_argument("--per-action", type=int, default=1)
    parser.add_argument("--per-object", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    action_names = load_name_map(args.action_names_json)
    object_names = load_name_map(args.object_names_json)
    node_names = load_name_map(args.node_names_json)
    record_paths = sorted(Path(args.trace_dir).glob(args.records_glob))
    if args.max_records > 0:
        record_paths = record_paths[: args.max_records]

    skip_counts: Counter = Counter()
    base_rows: List[Dict[str, Any]] = []
    profile_counts: Counter = Counter()
    for index, path in enumerate(record_paths, start=1):
        record = load_trace(path)
        profile = trace_profile(record)
        profile_counts[
            f"M={profile['m']},K={profile['k']},R={profile['roles']},L={profile['layers']}"
        ] += 1
        base_rows.extend(
            base_ranker.scan_record(
                record,
                path,
                args,
                action_names,
                object_names,
                node_names,
                skip_counts,
            )
        )
        if index % 100 == 0 or index == len(record_paths):
            print(f"Scanned {index}/{len(record_paths)} records; correct pairs={len(base_rows)}")

    base_ranker.add_ranking_scores(base_rows)
    enriched = enrich_rows(base_rows, args, action_names)
    ranked = sorted(
        [row for row in enriched if row["scientifically_eligible"]],
        key=lambda row: float(row["figure_score"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["global_rank"] = rank
    shortlist = select_diverse(
        ranked,
        limit=max(args.shortlist_size, 1),
        per_action=max(args.per_action, 1),
        per_object=max(args.per_object, 1),
    )
    write_csv(output_dir / "all_correct_pairs_scored.csv", enriched)
    write_csv(output_dir / "supplementary_candidates.csv", ranked)
    write_csv(output_dir / "supplementary_shortlist.csv", shortlist)

    mismatch_count = sum(not bool(row["config_compatible"]) for row in enriched)
    summary = {
        "record_count": len(record_paths),
        "correct_pair_count": len(base_rows),
        "eligible_candidate_count": len(ranked),
        "shortlist_count": len(shortlist),
        "observed_trace_profiles": dict(profile_counts),
        "expected_profile": {
            "M": args.expected_m,
            "K": args.expected_k,
            "roles": args.expected_roles,
            "L": args.expected_layers,
        },
        "strict_config": args.strict_config,
        "min_mechanism_score": args.min_mechanism_score,
        "profile_mismatch_candidate_count": mismatch_count,
        "skip_counts": dict(sorted(skip_counts.items())),
        "score_notice": (
            "figure_score is a mining heuristic, not a model metric or evidence-faithfulness claim"
        ),
        "warnings": (
            ["Observed Trace profile differs from the final-paper profile; use only for script smoke testing."]
            if mismatch_count else []
        ),
    }
    (output_dir / "supplementary_ranking_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
