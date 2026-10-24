"""Strict local HICO object/action/HOI mapping bundle."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io as sio


def _assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                continue
    return values


def normalize_name(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ")


@dataclass(frozen=True)
class HOIMapping:
    hoi_id: int
    verb_id: int
    verb_name: str
    object_sparse_id: int
    object_contiguous_id: int
    object_name: str
    hoi_text: str


class MappingBundle:
    def __init__(
        self,
        object_meta_path: Path,
        action_path: Path,
        text_label_path: Path,
        compatibility_path: Path,
        mat_path: Path,
    ) -> None:
        categories = _assignments(object_meta_path).get("COCO_CATEGORIES")
        if not isinstance(categories, list):
            raise RuntimeError("COCO_CATEGORIES mapping is unavailable")
        things = [entry for entry in categories if int(entry.get("isthing", 0)) == 1]
        if len(things) != 80:
            raise RuntimeError(f"Expected 80 object categories, got {len(things)}")
        self.object_by_sparse = {int(entry["id"]): str(entry["name"]) for entry in things}
        self.sparse_to_contiguous = {
            int(entry["id"]): index for index, entry in enumerate(things)
        }
        self.object_sparse_by_name = {
            normalize_name(entry["name"]): int(entry["id"]) for entry in things
        }

        self.verb_by_id: dict[int, str] = {}
        for line in action_path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.strip().split()
            if len(fields) < 2:
                continue
            try:
                self.verb_by_id[int(fields[0])] = fields[1]
            except ValueError:
                continue
        if set(self.verb_by_id) != set(range(1, 118)):
            raise RuntimeError("Action mapping is not exactly 1..117")
        verb_id_by_name = {name: verb_id for verb_id, name in self.verb_by_id.items()}

        text_labels = _assignments(text_label_path).get("hico_text_label")
        if not isinstance(text_labels, dict) or len(text_labels) != 600:
            raise RuntimeError("Expected 600 HICO text labels")
        compatibility = np.load(compatibility_path)
        if compatibility.shape != (117, 80) or int(compatibility.sum()) != 600:
            raise RuntimeError("Invalid HICO compatibility matrix")
        official = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)["list_action"]

        self.hoi_by_id: dict[int, HOIMapping] = {}
        self.hoi_by_key: dict[tuple[int, int], HOIMapping] = {}
        for hoi_id, action in enumerate(official, start=1):
            object_sparse_id = self.object_sparse_by_name[normalize_name(action.nname)]
            object_contiguous_id = self.sparse_to_contiguous[object_sparse_id]
            verb_name = str(action.vname).strip()
            verb_id = verb_id_by_name[verb_name]
            key = (verb_id - 1, object_contiguous_id)
            if key not in text_labels or compatibility[key] != 1:
                raise RuntimeError(f"Unresolved official HOI mapping: {hoi_id}")
            mapping = HOIMapping(
                hoi_id=hoi_id,
                verb_id=verb_id,
                verb_name=verb_name,
                object_sparse_id=object_sparse_id,
                object_contiguous_id=object_contiguous_id,
                object_name=self.object_by_sparse[object_sparse_id],
                hoi_text=str(text_labels[key]),
            )
            if key in self.hoi_by_key:
                raise RuntimeError(f"Duplicate official HOI key: {key}")
            self.hoi_by_id[hoi_id] = mapping
            self.hoi_by_key[key] = mapping
        if len(self.hoi_by_id) != 600:
            raise RuntimeError("Official HOI mapping is incomplete")

    def for_hoi_id(self, hoi_id: int) -> HOIMapping:
        return self.hoi_by_id[hoi_id]

    def for_verb_object(self, verb_id: int, object_sparse_id: int) -> HOIMapping:
        contiguous = self.sparse_to_contiguous[object_sparse_id]
        return self.hoi_by_key[(verb_id - 1, contiguous)]
