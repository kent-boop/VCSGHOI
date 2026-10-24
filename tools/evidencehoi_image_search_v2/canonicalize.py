"""Traceable, conservative per-image instance canonicalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .geometry import BBox, exact, high_iou_merge_compatible, iou


@dataclass(frozen=True)
class SourceInstance:
    source_instance_id: int
    role: str
    category_id: int
    category_name: str
    bbox: BBox
    provenance: dict[str, Any]


@dataclass
class CanonicalInstance:
    canonical_id: str
    role: str
    category_id: int
    category_name: str
    bbox: BBox
    members: list[SourceInstance]
    merge_method: str
    merge_confidence: float
    min_pairwise_iou: float


def _compatible(cluster: list[SourceInstance], candidate: SourceInstance, threshold: float) -> bool:
    return all(high_iou_merge_compatible(member.bbox, candidate.bbox, threshold) for member in cluster)


def _cluster_confidence(members: list[SourceInstance]) -> tuple[str, float, float]:
    if len(members) == 1:
        return "singleton", 1.0, 1.0
    pairwise = [
        iou(members[left].bbox, members[right].bbox)
        for left in range(len(members))
        for right in range(left + 1, len(members))
    ]
    minimum = min(pairwise)
    if all(exact(members[0].bbox, member.bbox) for member in members[1:]):
        return "exact_bbox", 1.0, 1.0
    return "high_iou_complete_link", minimum, minimum


def canonicalize_instances(
    image_key: str,
    sources: list[SourceInstance],
    high_iou_threshold: float = 0.92,
) -> tuple[list[CanonicalInstance], dict[int, str]]:
    """Canonicalize only duplicate representations, never generic neighbors."""
    groups: dict[tuple[str, int], list[SourceInstance]] = {}
    for source in sources:
        category_key = source.category_id if source.role == "object" else 1
        groups.setdefault((source.role, category_key), []).append(source)

    canonical: list[CanonicalInstance] = []
    source_to_canonical: dict[int, str] = {}
    role_counters = {"human": 0, "object": 0}
    for group_key in sorted(groups):
        clusters: list[list[SourceInstance]] = []
        for source in sorted(groups[group_key], key=lambda item: item.source_instance_id):
            exact_cluster = next(
                (cluster for cluster in clusters if exact(cluster[0].bbox, source.bbox)), None
            )
            if exact_cluster is not None:
                exact_cluster.append(source)
                continue
            compatible_cluster = next(
                (cluster for cluster in clusters if _compatible(cluster, source, high_iou_threshold)), None
            )
            if compatible_cluster is not None:
                compatible_cluster.append(source)
            else:
                clusters.append([source])

        for members in clusters:
            role = members[0].role
            index = role_counters[role]
            role_counters[role] += 1
            prefix = "h" if role == "human" else "o"
            canonical_id = f"{image_key}:{prefix}{index}"
            method, confidence, minimum_iou = _cluster_confidence(members)
            instance = CanonicalInstance(
                canonical_id=canonical_id,
                role=role,
                category_id=members[0].category_id,
                category_name=members[0].category_name,
                bbox=members[0].bbox,
                members=members,
                merge_method=method,
                merge_confidence=confidence,
                min_pairwise_iou=minimum_iou,
            )
            canonical.append(instance)
            for member in members:
                source_to_canonical[member.source_instance_id] = canonical_id
    return canonical, source_to_canonical
