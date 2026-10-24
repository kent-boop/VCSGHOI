#!/usr/bin/env python3
"""Draw GT HICO boxes and interaction labels for selected images."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


PERSON_CATEGORY_ID = 1

PAIR_COLORS = [
    (220, 50, 47),
    (38, 139, 210),
    (42, 161, 152),
    (181, 137, 0),
    (211, 54, 130),
    (108, 113, 196),
    (203, 75, 22),
    (88, 110, 117),
    (133, 153, 0),
    (147, 64, 191),
    (0, 128, 128),
    (180, 80, 80),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw GT HICO interaction overlays.")
    parser.add_argument("--dataset-root", default="/share/share/Datasets/hico_20160224_det")
    parser.add_argument("--annotation", default="annotations/trainval_hico.json")
    parser.add_argument("--image-root", default="images/train2015")
    parser.add_argument(
        "--image-dir",
        default="outputs/figure1_candidate_search/selected_gt_many_images",
        help="Directory containing selected image files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figure1_candidate_search/selected_gt_many_images/gt_interaction_overlays",
    )
    parser.add_argument("--max-side", type=int, default=1200)
    return parser.parse_args()


def resolve(path_text: str, base: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else base / path


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    return ImageFont.load_default()


FONT_SMALL = get_font(14)
FONT_MED = get_font(17)
FONT_BOLD = get_font(18, bold=True)


def load_action_names(dataset_root: Path) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for path in [dataset_root / "list_action.txt", dataset_root / "hico_list_vb.txt"]:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].isdigit():
                    mapping[int(parts[0])] = parts[1]
        if mapping:
            break
    return mapping


def load_object_names() -> Dict[int, str]:
    repo_root = Path(__file__).resolve().parents[1]
    meta_path = repo_root / "src" / "data" / "datasets" / "builtin_meta.py"
    if not meta_path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_overlay_builtin_meta", meta_path)
    if spec is None or spec.loader is None:
        return {}
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    out: Dict[int, str] = {}
    for item in getattr(module, "COCO_CATEGORIES", []):
        if isinstance(item, dict) and isinstance(item.get("id"), int) and isinstance(item.get("name"), str):
            if item["name"] != "N/A":
                out[item["id"]] = item["name"]
    return out


def bbox(raw: Any) -> Optional[List[float]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        return [float(x) for x in raw]
    except Exception:
        return None


def scale_box(box: Sequence[float], scale: float) -> List[float]:
    return [float(x) * scale for x in box]


def center(box: Sequence[float]) -> Tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], color: Tuple[int, int, int], width: int = 3) -> None:
    for offset in range(width):
        draw.rectangle(
            (box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset),
            outline=color,
        )


def text_bg(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    color: Tuple[int, int, int],
    font: ImageFont.ImageFont = FONT_SMALL,
) -> None:
    x, y = xy
    box = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle((box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad), fill=color)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        cand = word if not current else current + " " + word
        if draw.textbbox((0, 0), cand, font=font)[2] <= max_width:
            current = cand
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def build_pair_groups(sample: Dict[str, Any], action_names: Dict[int, str], object_names: Dict[int, str]) -> List[Dict[str, Any]]:
    annotations = sample.get("annotations") if isinstance(sample.get("annotations"), list) else []
    instance_by_idx: Dict[int, Dict[str, Any]] = {}
    for idx, raw in enumerate(annotations):
        if not isinstance(raw, dict):
            continue
        cat = raw.get("category_id")
        instance_by_idx[idx] = {
            "index": idx,
            "bbox": bbox(raw.get("bbox")),
            "category_id": cat,
            "category_name": object_names.get(cat, str(cat)),
        }

    grouped: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for raw in sample.get("hoi_annotation") if isinstance(sample.get("hoi_annotation"), list) else []:
        if not isinstance(raw, dict):
            continue
        h = raw.get("subject_id")
        o = raw.get("object_id")
        verb_id = raw.get("category_id")
        if not isinstance(h, int) or not isinstance(o, int):
            continue
        key = (h, o)
        item = grouped.setdefault(
            key,
            {
                "human_index": h,
                "object_index": o,
                "human": instance_by_idx.get(h),
                "object": instance_by_idx.get(o),
                "verbs": [],
            },
        )
        item["verbs"].append(
            {
                "verb_id": verb_id,
                "verb_name": action_names.get(verb_id, str(verb_id)),
                "hoi_category_id": raw.get("hoi_category_id"),
            }
        )
    return list(grouped.values())


def draw_overlay(
    image_path: Path,
    sample: Dict[str, Any],
    out_path: Path,
    action_names: Dict[int, str],
    object_names: Dict[int, str],
    max_side: int,
) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    scale = min(1.0, max_side / max(img.size))
    if scale < 1.0:
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)

    annotations = sample.get("annotations") if isinstance(sample.get("annotations"), list) else []
    pair_groups = build_pair_groups(sample, action_names, object_names)

    legend_lines = [f"{sample.get('file_name')} | GT boxes and HOI labels"]
    for i, group in enumerate(pair_groups):
        obj = group.get("object") or {}
        verb_text = "/".join(v["verb_name"] for v in group["verbs"])
        legend_lines.append(
            f"P{i + 1}: human {group['human_index']} -> {obj.get('category_name', 'object')} {group['object_index']}: {verb_text}"
        )

    # Keep detailed labels below the image. Height is dynamic.
    dummy = Image.new("RGB", (10, 10))
    dummy_draw = ImageDraw.Draw(dummy)
    wrapped_legend: List[Tuple[str, Optional[Tuple[int, int, int]]]] = []
    wrapped_legend.append((legend_lines[0], None))
    max_text_width = max(200, img.width - 28)
    for i, line in enumerate(legend_lines[1:]):
        color = PAIR_COLORS[i % len(PAIR_COLORS)]
        for wrapped in wrap_text(dummy_draw, line, FONT_MED, max_text_width - 30):
            wrapped_legend.append((wrapped, color))
    line_h = 23
    legend_h = 18 + line_h * len(wrapped_legend)
    canvas = Image.new("RGB", (img.width, img.height + legend_h), (245, 245, 245))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Draw all GT boxes first: humans in white, objects in gray.
    for idx, raw in enumerate(annotations):
        if not isinstance(raw, dict):
            continue
        box = bbox(raw.get("bbox"))
        if box is None:
            continue
        box = scale_box(box, scale)
        cat = raw.get("category_id")
        if cat == PERSON_CATEGORY_ID:
            color = (255, 255, 255)
            label = f"H{idx} person"
        else:
            color = (160, 160, 160)
            label = f"O{idx} {object_names.get(cat, cat)}"
        draw_box(draw, box, color, width=2)
        text_bg(draw, (int(max(2, box[0])), int(max(2, box[1] - 18))), label, color)

    # Draw pair-specific colored boxes and anchor lines on top.
    for i, group in enumerate(pair_groups):
        color = PAIR_COLORS[i % len(PAIR_COLORS)]
        human = group.get("human") or {}
        obj = group.get("object") or {}
        hbox = human.get("bbox")
        obox = obj.get("bbox")
        if hbox is None or obox is None:
            continue
        hbox = scale_box(hbox, scale)
        obox = scale_box(obox, scale)
        draw_box(draw, hbox, color, width=4)
        draw_box(draw, obox, color, width=4)
        hc = center(hbox)
        oc = center(obox)
        draw.line([hc, oc], fill=color, width=4)
        r = 5
        draw.ellipse((hc[0] - r, hc[1] - r, hc[0] + r, hc[1] + r), fill=color)
        draw.ellipse((oc[0] - r, oc[1] - r, oc[0] + r, oc[1] + r), fill=color)
        verb_text = "/".join(v["verb_name"] for v in group["verbs"])
        text_bg(draw, (int(max(2, min(hbox[0], img.width - 80))), int(max(2, hbox[1] + 4))), f"P{i + 1}", color, FONT_SMALL)
        text_bg(draw, (int(max(2, min(obox[0], img.width - 100))), int(max(2, obox[1] + 4))), f"P{i + 1}: {verb_text}", color, FONT_SMALL)

    # Legend below image.
    y = img.height + 9
    for j, (line, color) in enumerate(wrapped_legend):
        if j == 0:
            draw.text((10, y), line, fill=(20, 20, 20), font=FONT_BOLD)
        else:
            assert color is not None
            draw.rectangle((10, y + 5, 24, y + 18), fill=color)
            draw.text((32, y), line, fill=(20, 20, 20), font=FONT_MED)
        y += line_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return {
        "filename": sample.get("file_name"),
        "source_image": str(image_path),
        "overlay_path": str(out_path),
        "pair_count": len(pair_groups),
        "pairs": pair_groups,
    }


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    annotation_path = resolve(args.annotation, dataset_root).resolve()
    image_root = resolve(args.image_root, dataset_root).resolve()
    image_dir = Path(args.image_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    action_names = load_action_names(dataset_root)
    object_names = load_object_names()
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_filename = {item.get("file_name"): item for item in annotations if isinstance(item, dict)}

    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    manifest: List[Dict[str, Any]] = []
    for local_image in image_files:
        sample = by_filename.get(local_image.name)
        if sample is None:
            print(f"Skip missing annotation: {local_image.name}")
            continue
        source = image_root / local_image.name
        if not source.exists():
            source = local_image
        out_path = output_dir / f"{local_image.stem}_gt_overlay.jpg"
        manifest.append(draw_overlay(source, sample, out_path, action_names, object_names, args.max_side))

    manifest_path = output_dir / "gt_overlay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(manifest)} GT overlay images in {output_dir}")
    print(f"Manifest: {manifest_path}")
    for item in manifest:
        print(f"- {item['overlay_path']} ({item['pair_count']} GT pair groups)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
