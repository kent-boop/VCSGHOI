#!/usr/bin/env python3
"""Stage-1 schema inspection for HOI annotation files.

This script intentionally does not implement any image-ranking or filtering logic.
It discovers likely HICO-DET annotation / metadata files and samples only a few
records from each candidate so large annotation files are not fully traversed.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
import reprlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image
except Exception:  # pragma: no cover - PIL is expected but optional for reporting.
    Image = None


SKIP_DIRS = {
    ".git",
    ".kilo",
    ".kilocode",
    ".opencode",
    "__pycache__",
    "checkpoints",
    "images",
    "imgs",
    "outputs",
    "visual",
    "visual2",
}

SUPPORTED_SUFFIXES = {".json", ".pkl", ".pickle", ".txt", ".py", ".npy", ".mat"}

KNOWN_HICO_METADATA_FILES = {
    "hico_text_label.py",
    "hico_text_descriptions.json",
    "hico_lib.py",
    "hico_lib.txt",
    "list_action.txt",
    "hico_list_vb.txt",
}

IMAGE_NAME_KEYS = ("file_name", "filename", "image", "image_name", "img", "img_name", "path", "image_path")
WIDTH_KEYS = ("width", "w", "img_width")
HEIGHT_KEYS = ("height", "h", "img_height")
BBOX_KEYS = ("bbox", "bboxes", "boxes", "box", "bboxhuman", "bboxobject", "pair_bbox", "pair_boxes")
OBJECT_CATEGORY_KEYS = ("category_id", "object_category_id", "obj_category_id", "class_id", "label", "labels", "categories")
HOI_KEYS = ("hoi_annotation", "hoi", "hois", "relations", "interaction", "interactions", "role_object_id")
SUBJECT_KEYS = ("subject_id", "sub_id", "human_id", "person_id", "agent_id")
OBJECT_KEYS = ("object_id", "obj_id", "target_id", "role_object_id")
VERB_KEYS = ("verb_id", "action_id", "category_id", "hoi_category_id", "action_name", "vname")


@dataclass
class Inspection:
    path: Path
    role: str
    size_bytes: int = 0
    top_type: str = "unknown"
    image_record_keys: list[str] = field(default_factory=list)
    bbox_fields: list[str] = field(default_factory=list)
    object_category_fields: list[str] = field(default_factory=list)
    hoi_relation_fields: list[str] = field(default_factory=list)
    subject_fields: list[str] = field(default_factory=list)
    object_fields: list[str] = field(default_factory=list)
    verb_fields: list[str] = field(default_factory=list)
    image_file_fields: list[str] = field(default_factory=list)
    width_fields: list[str] = field(default_factory=list)
    height_fields: list[str] = field(default_factory=list)
    pil_size_probe: str = "not attempted"
    samples: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def rel(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def first_non_ws_char(path: Path, limit: int = 8192) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            data = f.read(limit)
    except Exception:
        return None
    for ch in data:
        if not ch.isspace():
            return ch
    return None


def safe_repr(value: Any, max_chars: int = 900) -> str:
    printer = reprlib.Repr()
    printer.maxstring = 180
    printer.maxother = max_chars
    printer.maxlist = 5
    printer.maxdict = 8
    text = printer.repr(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def read_more(f, buffer: str, chunk_size: int = 65536) -> tuple[str, bool]:
    chunk = f.read(chunk_size)
    if not chunk:
        return buffer, False
    return buffer + chunk, True


def skip_ws(buffer: str, pos: int) -> int:
    while pos < len(buffer) and buffer[pos].isspace():
        pos += 1
    return pos


def sample_json_array(path: Path, count: int, warnings: list[str]) -> list[Any]:
    """Decode only the first `count` elements from a top-level JSON array."""
    samples: list[Any] = []
    decoder = json.JSONDecoder()
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            buffer = ""
            pos = 0
            while "[" not in buffer:
                buffer, ok = read_more(f, buffer)
                if not ok:
                    warnings.append("Could not find top-level '[' while sampling JSON array.")
                    return samples
            pos = buffer.index("[") + 1

            while len(samples) < count:
                pos = skip_ws(buffer, pos)
                while pos >= len(buffer):
                    buffer, ok = read_more(f, buffer)
                    if not ok:
                        return samples
                    pos = skip_ws(buffer, pos)

                if buffer[pos] == "]":
                    return samples
                if buffer[pos] == ",":
                    pos += 1
                    continue

                while True:
                    try:
                        item, end = decoder.raw_decode(buffer, pos)
                        samples.append(item)
                        pos = end
                        break
                    except json.JSONDecodeError:
                        old_len = len(buffer)
                        buffer, ok = read_more(f, buffer)
                        if not ok or len(buffer) == old_len:
                            warnings.append("Failed to decode next JSON array element from available prefix.")
                            return samples
    except Exception as exc:
        warnings.append(f"JSON array sampling failed: {exc}")
    return samples


def sample_json_dict(path: Path, count: int, warnings: list[str]) -> tuple[dict[str, Any] | None, list[Any]]:
    """Inspect small dict JSON files; avoid full loading very large dicts."""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size > 8 * 1024 * 1024:
        warnings.append("Large top-level JSON dict was not fully loaded; only top-level type is reported.")
        return None, []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            obj = json.load(f)
    except Exception as exc:
        warnings.append(f"JSON dict load failed: {exc}")
        return None, []
    if not isinstance(obj, dict):
        return None, []
    samples: list[Any] = []
    for value in obj.values():
        if isinstance(value, list):
            samples.extend(value[:count])
        elif isinstance(value, dict):
            samples.append(dict(list(value.items())[:count]))
        else:
            samples.append(value)
        if len(samples) >= count:
            break
    return obj, samples[:count]


def sample_pickle(path: Path, count: int, warnings: list[str]) -> tuple[str, list[Any]]:
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
    except Exception as exc:
        warnings.append(f"Pickle load failed: {exc}")
        return "pkl object", []
    top_type = type(obj).__name__
    if isinstance(obj, list):
        return top_type, obj[:count]
    if isinstance(obj, tuple):
        return top_type, list(obj[:count])
    if isinstance(obj, dict):
        return top_type, [dict(list(obj.items())[:count])]
    return top_type, [obj]


def sample_text(path: Path, count: int, warnings: list[str]) -> list[str]:
    lines: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                text = line.rstrip("\n")
                if text.strip():
                    lines.append(text)
                if len(lines) >= count:
                    break
    except Exception as exc:
        warnings.append(f"Text sampling failed: {exc}")
    return lines


def sample_python_assignment(path: Path, count: int, warnings: list[str]) -> list[Any]:
    """Sample simple literal assignments from metadata Python files without importing them."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        warnings.append(f"Python file read failed: {exc}")
        return []
    try:
        tree = ast.parse(text)
    except Exception as exc:
        warnings.append(f"Python AST parse failed: {exc}")
        return sample_text(path, count, warnings)
    samples: list[Any] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets:
            target = node.targets[0]
            name = getattr(target, "id", None)
            try:
                value = ast.literal_eval(node.value)
            except Exception:
                samples.append({"assignment": name or "<unknown>", "value": "<non-literal or too complex>"})
                continue
            if isinstance(value, dict):
                samples.append({"assignment": name, "type": "dict", "items": dict(list(value.items())[:count])})
            elif isinstance(value, list):
                samples.append({"assignment": name, "type": "list", "items": value[:count]})
            else:
                samples.append({"assignment": name, "type": type(value).__name__, "value": value})
            if len(samples) >= count:
                break
    if not samples:
        warnings.append("No literal assignments found in Python metadata file.")
        return sample_text(path, count, warnings)
    return samples


def walk_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value[:10]:
            yield from walk_dicts(child)


def collect_matching_keys(samples: list[Any], candidates: tuple[str, ...]) -> list[str]:
    found: set[str] = set()
    candidate_set = set(candidates)
    candidate_lower_set = {candidate.lower() for candidate in candidates}
    for sample in samples:
        for obj in walk_dicts(sample):
            for key in obj.keys():
                key_text = str(key)
                key_lower = key_text.lower()
                if key_text in candidate_set or key_lower in candidate_lower_set:
                    found.add(key_text)
                    continue
                if any(len(token) >= 4 and token in key_lower for token in candidate_lower_set):
                    found.add(key_text)
    return sorted(found)


def likely_image_records(samples: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sample in samples:
        if isinstance(sample, dict):
            records.append(sample)
        elif isinstance(sample, list):
            records.extend([x for x in sample[:5] if isinstance(x, dict)])
    return records


def infer_split_from_file_name(file_name: str) -> str | None:
    if "train2015" in file_name or "train" in file_name:
        return "train2015"
    if "test2015" in file_name or "test" in file_name:
        return "test2015"
    return None


def probe_pil_size(dataset_root: Path, samples: list[Any], warnings: list[str]) -> str:
    if Image is None:
        return "not attempted: PIL unavailable"
    for record in likely_image_records(samples):
        file_name = None
        for key in IMAGE_NAME_KEYS:
            value = record.get(key)
            if isinstance(value, str):
                file_name = value
                break
        if not file_name:
            continue
        direct = Path(file_name)
        candidates: list[Path] = []
        if direct.is_absolute():
            candidates.append(direct)
        else:
            candidates.append(dataset_root / file_name)
            split = infer_split_from_file_name(file_name)
            if split:
                candidates.append(dataset_root / "images" / split / file_name)
            candidates.append(dataset_root / "images" / file_name)
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    with Image.open(candidate) as img:
                        width, height = img.size
                    return f"yes: opened `{candidate}` -> width={width}, height={height}"
                except Exception as exc:
                    warnings.append(f"PIL failed for {candidate}: {exc}")
        return f"needed but image path not resolved for sample file `{file_name}`"
    return "not needed or no image file field found in sampled records"


def classify_role(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in {".npy", ".mat"}:
        return "binary metadata / original annotation (listed only)"
    if "trainval_hico" in name:
        return "hico trainval annotation"
    if "test_hico" in name:
        return "hico test annotation"
    if "hico_text_label" in name or "hico_text" in name or name in {"list_action.txt", "hico_list_vb.txt", "hico_lib.txt"}:
        return "hico label/name metadata"
    if "corre" in name:
        return "compatibility/correlation metadata"
    if name.startswith("anno"):
        return "annotation candidate"
    return "candidate"


def should_include(path: Path, hico_only: bool) -> bool:
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        return False
    name = path.name.lower()
    text = str(path).lower()
    if hico_only and "vcoco" in text:
        return False

    if hico_only:
        if name in KNOWN_HICO_METADATA_FILES:
            return True
        if suffix == ".json":
            return (
                name == "anno_list.json"
                or name == "test_hico.json"
                or name.startswith("trainval_hico")
                or name.startswith("instances_hico")
                or name.startswith("corre_hico")
            )
        if suffix in {".pkl", ".pickle", ".npy"}:
            return name.startswith("corre_hico") or name.startswith("corre_")
        if suffix == ".mat":
            return name.startswith("anno")
        return False

    return any(token in name for token in ("anno", "hico", "vcoco", "instances_", "corre", "verb", "object", "list_action"))


def discover_candidates(roots: list[Path], hico_only: bool) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = []
            stack = [root]
            while stack:
                current = stack.pop()
                try:
                    for child in current.iterdir():
                        if child.is_dir():
                            if child.name in SKIP_DIRS:
                                continue
                            stack.append(child)
                        else:
                            candidates.append(child)
                except PermissionError:
                    continue
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            if resolved in seen:
                continue
            if should_include(candidate, hico_only=hico_only):
                seen.add(resolved)
                result.append(candidate)
    return sorted(result, key=lambda p: str(p))


def inspect_file(path: Path, dataset_root: Path, sample_count: int) -> Inspection:
    item = Inspection(path=path, role=classify_role(path))
    try:
        item.size_bytes = path.stat().st_size
    except OSError:
        item.warnings.append("Could not stat file size.")

    suffix = path.suffix.lower()
    if suffix == ".json":
        ch = first_non_ws_char(path)
        if ch == "[":
            item.top_type = "list"
            item.samples = sample_json_array(path, sample_count, item.warnings)
        elif ch == "{":
            item.top_type = "dict"
            obj, samples = sample_json_dict(path, sample_count, item.warnings)
            item.samples = samples
            if obj is not None and not item.samples:
                item.samples = [dict(list(obj.items())[:sample_count])]
        else:
            item.top_type = "json unknown"
            item.warnings.append("Could not determine JSON top-level type from prefix.")
    elif suffix in {".pkl", ".pickle"}:
        item.top_type, item.samples = sample_pickle(path, sample_count, item.warnings)
    elif suffix == ".py":
        item.top_type = "python metadata"
        item.samples = sample_python_assignment(path, sample_count, item.warnings)
    elif suffix == ".txt":
        item.top_type = "text"
        item.samples = sample_text(path, sample_count, item.warnings)
    elif suffix == ".npy":
        item.top_type = "npy binary"
        item.warnings.append("Not parsed: numpy is intentionally not required for Stage 1 inspection.")
    elif suffix == ".mat":
        item.top_type = "mat binary"
        item.warnings.append("Not parsed: scipy is intentionally not required for Stage 1 inspection.")
    else:
        item.top_type = "unsupported"
        item.warnings.append(f"Unsupported suffix `{suffix}`.")

    records = likely_image_records(item.samples)
    if records:
        keys: set[str] = set()
        for record in records[:sample_count]:
            keys.update(str(k) for k in record.keys())
        item.image_record_keys = sorted(keys)

    item.bbox_fields = collect_matching_keys(item.samples, BBOX_KEYS)
    item.object_category_fields = collect_matching_keys(item.samples, OBJECT_CATEGORY_KEYS)
    item.hoi_relation_fields = collect_matching_keys(item.samples, HOI_KEYS)
    item.subject_fields = collect_matching_keys(item.samples, SUBJECT_KEYS)
    item.object_fields = collect_matching_keys(item.samples, OBJECT_KEYS)
    item.verb_fields = collect_matching_keys(item.samples, VERB_KEYS)
    item.image_file_fields = collect_matching_keys(item.samples, IMAGE_NAME_KEYS)
    item.width_fields = collect_matching_keys(item.samples, WIDTH_KEYS)
    item.height_fields = collect_matching_keys(item.samples, HEIGHT_KEYS)

    if item.image_record_keys or item.image_file_fields:
        has_width_height = bool(item.width_fields and item.height_fields)
        if has_width_height:
            item.pil_size_probe = "not required by sampled schema"
        else:
            item.pil_size_probe = probe_pil_size(dataset_root, item.samples, item.warnings)
    return item


def md_list(values: list[str]) -> str:
    return ", ".join(f"`{v}`" for v in values) if values else "not found in samples"


def write_report(
    output: Path,
    dataset_root: Path,
    project_root: Path,
    candidates: list[Path],
    inspections: list[Inspection],
    hico_only: bool,
) -> None:
    lines: list[str] = []
    lines.append("# HOI Annotation Schema Report")
    lines.append("")
    lines.append("Stage 1 only: candidate discovery and sampled schema inspection. No scoring, filtering, contact sheets, or visualization logic is implemented here.")
    lines.append("")
    lines.append("## Scan Configuration")
    lines.append("")
    lines.append(f"- Dataset root: `{dataset_root}`")
    lines.append(f"- Project root: `{project_root}`")
    lines.append(f"- HICO-only mode: `{hico_only}`")
    lines.append(f"- Candidate files discovered: `{len(candidates)}`")
    lines.append("")
    lines.append("## Discovered Candidate Files")
    lines.append("")
    for path in candidates:
        lines.append(f"- `{path}`")
    if not candidates:
        lines.append("- No candidates discovered.")
    lines.append("")
    lines.append("## Per-File Schema Inspection")
    lines.append("")

    for item in inspections:
        lines.append(f"### `{item.path}`")
        lines.append("")
        lines.append(f"- Role guess: {item.role}")
        lines.append(f"- Size: `{item.size_bytes}` bytes")
        lines.append(f"- Top-level type: `{item.top_type}`")
        lines.append(f"- Image record keys: {md_list(item.image_record_keys)}")
        lines.append(f"- BBox field names: {md_list(item.bbox_fields)}")
        lines.append(f"- Object category field names: {md_list(item.object_category_fields)}")
        lines.append(f"- HOI relation field names: {md_list(item.hoi_relation_fields)}")
        lines.append(f"- Subject id field names: {md_list(item.subject_fields)}")
        lines.append(f"- Object id field names: {md_list(item.object_fields)}")
        lines.append(f"- Verb/action/category field names: {md_list(item.verb_fields)}")
        lines.append(f"- Image file name field names: {md_list(item.image_file_fields)}")
        lines.append(f"- Width field names: {md_list(item.width_fields)}")
        lines.append(f"- Height field names: {md_list(item.height_fields)}")
        lines.append(f"- Need PIL for width/height: `{item.pil_size_probe}`")
        if item.warnings:
            lines.append("- Warnings:")
            for warning in item.warnings:
                lines.append(f"  - {warning}")
        else:
            lines.append("- Warnings: none")
        if item.samples:
            lines.append("- Sample preview:")
            for idx, sample in enumerate(item.samples[:5], start=1):
                lines.append(f"  - sample {idx}: `{safe_repr(sample)}`")
        else:
            lines.append("- Sample preview: none")
        lines.append("")

    trainval = next((x.path for x in inspections if x.path.name == "trainval_hico.json"), None)
    test = next((x.path for x in inspections if x.path.name == "test_hico.json"), None)
    list_action = next((x.path for x in inspections if x.path.name == "list_action.txt"), None)
    hico_list_vb = next((x.path for x in inspections if x.path.name == "hico_list_vb.txt"), None)
    hico_text_label = next((x.path for x in inspections if x.path.name == "hico_text_label.py"), None)
    corre_hico = next((x.path for x in inspections if x.path.name.startswith("corre_hico")), None)

    lines.append("## Recommended Inputs for the Unified Loader")
    lines.append("")
    lines.append(f"- HICO trainval annotation: `{trainval}`" if trainval else "- HICO trainval annotation: not found")
    lines.append(f"- HICO test annotation: `{test}`" if test else "- HICO test annotation: not found")
    if list_action:
        lines.append(f"- HICO verb names: `{list_action}`")
    elif hico_list_vb:
        lines.append(f"- HICO verb names: `{hico_list_vb}`")
    else:
        lines.append("- HICO verb names: not found")
    lines.append(f"- Alternate HICO verb names: `{hico_list_vb}`" if hico_list_vb else "- Alternate HICO verb names: not found")
    lines.append(f"- HICO text label metadata: `{hico_text_label}`" if hico_text_label else "- HICO text label metadata: not found")
    lines.append(f"- HICO compatibility metadata: `{corre_hico}`" if corre_hico else "- HICO compatibility metadata: not found")
    lines.append("- V-COCO train/test inputs: intentionally not scanned in this run per current scope.")
    lines.append("")
    lines.append("## Fields Still Uncertain")
    lines.append("")
    lines.append("- `trainval_hico.json` includes `hoi_category_id`; `test_hico.json` samples do not, so test HOI triplet labels may require verb id plus object category mapping.")
    lines.append("- JSON image records do not include width/height in sampled HICO records; geometry-aware scoring should open images with PIL when needed.")
    lines.append("- `.mat` and `.npy` metadata files are discovered but not parsed in this Stage 1 script because only standard library, pickle, json, pathlib, and PIL are used.")
    lines.append("- Fractional HICO annotation files appear to share the trainval schema but should not be used as the default full-data source unless explicitly requested.")
    lines.append("")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect sampled schemas of HOI annotation files.")
    parser.add_argument("--dataset_root", type=Path, default=Path("/share/share/Datasets/hico_20160224_det"))
    parser.add_argument("--project_root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("tools/evidencehoi_image_search/schema_report.md"))
    parser.add_argument("--sample_count", type=int, default=5)
    parser.add_argument("--include_vcoco", action="store_true", help="Disable HICO-only filtering and include V-COCO-like names.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    project_root = args.project_root.resolve()
    output = args.output
    if not output.is_absolute():
        output = project_root / output
    sample_count = max(2, min(args.sample_count, 5))
    hico_only = not args.include_vcoco

    roots = [dataset_root, project_root]
    candidates = discover_candidates(roots, hico_only=hico_only)
    inspections = [inspect_file(path, dataset_root=dataset_root, sample_count=sample_count) for path in candidates]
    write_report(output, dataset_root, project_root, candidates, inspections, hico_only=hico_only)

    print(f"Discovered {len(candidates)} candidate files.")
    print(f"Wrote schema report to {output}")
    for path in candidates:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
