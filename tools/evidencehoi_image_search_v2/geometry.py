"""Canonical bounding-box geometry for HICO-DET v2 indexing."""

from __future__ import annotations

from typing import Iterable

BBox = tuple[float, float, float, float]


def trainval_json_to_half_open(box: Iterable[float]) -> BBox:
    x1, y1, x2, y2 = (float(value) for value in box)
    return x1, y1, x2 + 1.0, y2 + 1.0


def test_json_to_half_open(box: Iterable[float]) -> BBox:
    x1, y1, x2, y2 = (float(value) for value in box)
    return x1 - 1.0, y1 - 1.0, x2, y2


def mat_to_half_open(box: Iterable[float]) -> BBox:
    x1, y1, x2, y2 = (float(value) for value in box)
    return x1 - 1.0, y1 - 1.0, x2, y2


def area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(left: BBox, right: BBox) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area(left) + area(right) - intersection
    return 0.0 if union <= 0.0 else intersection / union


def exact(left: BBox, right: BBox, tolerance: float = 1e-9) -> bool:
    return all(abs(a - b) <= tolerance for a, b in zip(left, right))


def center(box: BBox) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def high_iou_merge_compatible(left: BBox, right: BBox, threshold: float) -> bool:
    """Conservative duplicate check, not a generic proximity check."""
    overlap = iou(left, right)
    if overlap < threshold:
        return False
    left_area, right_area = area(left), area(right)
    if min(left_area, right_area) <= 0:
        return False
    area_ratio = min(left_area, right_area) / max(left_area, right_area)
    if area_ratio < threshold:
        return False
    left_center, right_center = center(left), center(right)
    scale = max(
        left[2] - left[0], left[3] - left[1],
        right[2] - right[0], right[3] - right[1], 1.0,
    )
    center_delta = max(
        abs(left_center[0] - right_center[0]),
        abs(left_center[1] - right_center[1]),
    ) / scale
    return center_delta <= (1.0 - threshold)
