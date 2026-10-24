#!/usr/bin/env python3
"""Shared dataclasses for normalized HOI annotation records."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class InstanceRecord:
    instance_id: int
    bbox_xyxy: tuple[float, float, float, float]
    category_id: int
    category_name: str
    is_person: bool


@dataclass
class HOIRelationRecord:
    subject_id: int
    object_id: int
    verb_id: int
    verb_name: str
    hoi_id: int | None
    hoi_name: str | None


@dataclass
class ImageRecord:
    image_id: str
    file_name: str
    image_path: str
    width: int
    height: int
    split: str
    dataset: str
    instances: list[InstanceRecord]
    relations: list[HOIRelationRecord]


def record_to_dict(record: ImageRecord) -> dict[str, Any]:
    return asdict(record)


def instance_from_dict(data: dict[str, Any]) -> InstanceRecord:
    return InstanceRecord(
        instance_id=int(data["instance_id"]),
        bbox_xyxy=tuple(float(x) for x in data["bbox_xyxy"]),
        category_id=int(data["category_id"]),
        category_name=str(data["category_name"]),
        is_person=bool(data["is_person"]),
    )


def relation_from_dict(data: dict[str, Any]) -> HOIRelationRecord:
    return HOIRelationRecord(
        subject_id=int(data["subject_id"]),
        object_id=int(data["object_id"]),
        verb_id=int(data["verb_id"]),
        verb_name=str(data["verb_name"]),
        hoi_id=None if data.get("hoi_id") is None else int(data["hoi_id"]),
        hoi_name=None if data.get("hoi_name") is None else str(data["hoi_name"]),
    )


def record_from_dict(data: dict[str, Any]) -> ImageRecord:
    return ImageRecord(
        image_id=str(data["image_id"]),
        file_name=str(data["file_name"]),
        image_path=str(data["image_path"]),
        width=int(data["width"]),
        height=int(data["height"]),
        split=str(data["split"]),
        dataset=str(data["dataset"]),
        instances=[instance_from_dict(item) for item in data.get("instances", [])],
        relations=[relation_from_dict(item) for item in data.get("relations", [])],
    )
