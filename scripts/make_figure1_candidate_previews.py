#!/usr/bin/env python3
"""Create visual previews for top Figure 1 candidates.

The previews are for manual inspection only. They overlay a small number of
representative HOI pairs with colored boxes, anchor lines, and a legend placed
below the image to avoid covering entities.
"""

from __future__ import annotations

import argparse
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps


COLORS = [
    (220, 50, 47),
    (38, 139, 210),
    (42, 161, 152),
    (181, 137, 0),
    (211, 54, 130),
]

COLOR_NAMES = ["red", "blue", "green", "yellow", "magenta"]

CONTEXT_NAMES = {
    "dining table",
    "chair",
    "cup",
    "bottle",
    "sports ball",
    "bicycle",
    "motorcycle",
    "laptop",
    "keyboard",
    "mouse",
    "cell phone",
    "remote",
    "backpack",
    "handbag",
    "suitcase",
    "bench",
    "frisbee",
    "skis",
    "snowboard",
    "skateboard",
    "surfboard",
    "tennis racket",
    "baseball bat",
    "baseball glove",
    "kite",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "book",
    "bed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Figure 1 candidate preview overlays.")
    parser.add_argument(
        "--ranked-json",
        default="outputs/figure1_candidate_search/ranked_candidates.json",
        help="Input ranked_candidates.json.",
    )
    parser.add_argument(
        "--image-root",
        default=None,
        help="Original image root. Defaults to metadata.image_root in ranked JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/figure1_candidate_search/previews",
        help="Preview output directory.",
    )
    parser.add_argument("--top-k", type=int, default=30, help="Number of previews to generate.")
    parser.add_argument("--notes-top-k", type=int, default=10, help="Number of candidates to describe in notes.")
    return parser.parse_args()


def load_ranked(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("metadata", {}), payload.get("ranked_candidates", [])


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


FONT_SMALL = font(14)
FONT_MED = font(18)
FONT_MED_BOLD = font(18, bold=True)
FONT_BIG_BOLD = font(24, bold=True)


def bbox_center(box: Optional[Sequence[float]]) -> Optional[Tuple[float, float]]:
    if not box or len(box) != 4:
        return None
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def draw_box(draw: ImageDraw.ImageDraw, box: Sequence[float], color: Tuple[int, int, int], width: int) -> None:
    xy = tuple(float(x) for x in box)
    for offset in range(width):
        draw.rectangle(
            [xy[0] - offset, xy[1] - offset, xy[2] + offset, xy[3] + offset],
            outline=color,
        )


def draw_text_with_bg(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[int, int],
    text: str,
    fill: Tuple[int, int, int] = (255, 255, 255),
    bg: Tuple[int, int, int] = (0, 0, 0),
    text_font: ImageFont.ImageFont = FONT_SMALL,
) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=text_font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=bg)
    draw.text((x, y), text, fill=fill, font=text_font)


def pair_key(pair: Dict[str, Any]) -> Tuple[Any, Any, Any]:
    return (pair.get("human_index"), pair.get("object_index"), pair.get("verb_id"))


def select_representative_pairs(candidate: Dict[str, Any], min_pairs: int = 2, max_pairs: int = 5) -> List[Dict[str, Any]]:
    pairs = [p for p in candidate.get("hoi_pairs", []) if p.get("human_bbox") and p.get("object_bbox")]
    if not pairs:
        return []

    selected: List[Dict[str, Any]] = []
    used_keys = set()
    used_humans = set()
    used_objects = set()
    used_verbs = set()
    used_categories = set()

    def add(pair: Dict[str, Any]) -> bool:
        key = pair_key(pair)
        if key in used_keys:
            return False
        selected.append(pair)
        used_keys.add(key)
        used_humans.add(pair.get("human_index"))
        used_objects.add(pair.get("object_index"))
        used_verbs.add(pair.get("verb_id"))
        used_categories.add(pair.get("object_category_id"))
        return True

    # 1. Prefer diverse people and diverse object categories.
    for pair in sorted(
        pairs,
        key=lambda p: (
            p.get("object_category_name") in CONTEXT_NAMES,
            p.get("verb_name") not in {"no_interaction", None},
        ),
        reverse=True,
    ):
        if len(selected) >= max_pairs:
            break
        new_human = pair.get("human_index") not in used_humans
        new_category = pair.get("object_category_id") not in used_categories
        new_verb = pair.get("verb_id") not in used_verbs
        if new_human or new_category or new_verb:
            add(pair)

    # 2. Add contrastive pairs: same object category with different verbs.
    category_groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    verb_groups: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        category_groups[pair.get("object_category_id")].append(pair)
        verb_groups[pair.get("verb_id")].append(pair)
    for group in list(category_groups.values()) + list(verb_groups.values()):
        if len(selected) >= max_pairs:
            break
        if len(group) < 2:
            continue
        for pair in group:
            if len(selected) >= max_pairs:
                break
            if pair_key(pair) not in used_keys:
                add(pair)

    # 3. Fill to minimum with original high-ranked order.
    for pair in pairs:
        if len(selected) >= max_pairs:
            break
        add(pair)
        if len(selected) >= min_pairs:
            break

    return selected[:max_pairs]


def entity_label(prefix: str, index: Any) -> str:
    return f"{prefix}_{index}" if index is not None else prefix


def pair_summary(pair: Dict[str, Any], color_name: str) -> str:
    human = entity_label("person", pair.get("human_index"))
    obj = pair.get("object_category_name") or f"object_{pair.get('object_index')}"
    verb = pair.get("verb_name") or f"verb_{pair.get('verb_id')}"
    return f"{color_name}: {human} - {obj}: {verb}"


def resize_for_preview(img: Image.Image, max_side: int = 960) -> Tuple[Image.Image, float]:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale == 1.0:
        return img.copy(), 1.0
    return img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS), scale


def scale_box(box: Sequence[float], scale: float) -> List[float]:
    return [float(x) * scale for x in box]


def make_preview(candidate: Dict[str, Any], image_path: Path, out_path: Path) -> Dict[str, Any]:
    img = Image.open(image_path).convert("RGB")
    canvas_img, scale = resize_for_preview(img)
    selected = select_representative_pairs(candidate)

    legend_lines = [
        f"rank {candidate.get('rank')} | score {candidate.get('figure1_score'):.1f} | {candidate.get('filename')}",
    ]
    for idx, pair in enumerate(selected):
        legend_lines.append(pair_summary(pair, COLOR_NAMES[idx]))

    line_height = 23
    legend_h = 22 + line_height * len(legend_lines)
    canvas = Image.new("RGB", (canvas_img.width, canvas_img.height + legend_h), (245, 245, 245))
    canvas.paste(canvas_img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    for idx, pair in enumerate(selected):
        color = COLORS[idx]
        human_box = scale_box(pair["human_bbox"], scale)
        object_box = scale_box(pair["object_bbox"], scale)
        draw_box(draw, human_box, color, width=3)
        draw_box(draw, object_box, color, width=3)
        hc = bbox_center(human_box)
        oc = bbox_center(object_box)
        if hc and oc:
            draw.line([hc, oc], fill=color, width=3)
            r = 4
            draw.ellipse((hc[0] - r, hc[1] - r, hc[0] + r, hc[1] + r), fill=color)
            draw.ellipse((oc[0] - r, oc[1] - r, oc[0] + r, oc[1] + r), fill=color)

        # Small numeric labels near boxes only; detailed text stays in legend.
        hx = int(max(2, min(human_box[0], canvas_img.width - 45)))
        hy = int(max(2, human_box[1] - 19))
        ox = int(max(2, min(object_box[0], canvas_img.width - 45)))
        oy = int(max(2, object_box[1] - 19))
        draw_text_with_bg(draw, (hx, hy), f"P{idx + 1} H", bg=color, text_font=FONT_SMALL)
        draw_text_with_bg(draw, (ox, oy), f"P{idx + 1} O", bg=color, text_font=FONT_SMALL)

    y0 = canvas_img.height + 8
    draw.rectangle((0, canvas_img.height, canvas.width, canvas.height), fill=(245, 245, 245))
    draw.text((10, y0), legend_lines[0], fill=(20, 20, 20), font=FONT_MED_BOLD)
    y = y0 + line_height
    for idx, line in enumerate(legend_lines[1:]):
        color = COLORS[idx]
        draw.rectangle((10, y + 4, 24, y + 18), fill=color)
        draw.text((32, y), line, fill=(20, 20, 20), font=FONT_MED)
        y += line_height

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=92)
    return {
        "preview_path": str(out_path),
        "selected_pairs": selected,
        "legend_lines": legend_lines,
    }


def make_contact_sheet(candidates: Sequence[Dict[str, Any]], preview_paths: Sequence[Path], out_path: Path) -> None:
    cols = 5
    rows = math.ceil(len(preview_paths) / cols)
    thumb_w, thumb_h = 300, 245
    label_h = 48
    sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (235, 235, 235))
    draw = ImageDraw.Draw(sheet)
    for i, (candidate, preview_path) in enumerate(zip(candidates, preview_paths)):
        if not preview_path.exists():
            continue
        img = Image.open(preview_path).convert("RGB")
        # Crop away part of legend for thumbnail, preserving overlay readability.
        crop_h = int(img.height * 0.78)
        img = img.crop((0, 0, img.width, crop_h))
        img.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        col = i % cols
        row = i // cols
        x = col * thumb_w + (thumb_w - img.width) // 2
        y = row * (thumb_h + label_h) + 6
        sheet.paste(img, (x, y))
        label_y = row * (thumb_h + label_h) + thumb_h + 4
        label = f"#{candidate.get('rank')} score {candidate.get('figure1_score'):.1f} pairs {candidate.get('hoi_pair_count')}"
        draw.text((col * thumb_w + 8, label_y), label, fill=(0, 0, 0), font=FONT_MED_BOLD)
        draw.text((col * thumb_w + 8, label_y + 23), candidate.get("filename", ""), fill=(40, 40, 40), font=FONT_SMALL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=92)


def unique_pair_phrases(pairs: Sequence[Dict[str, Any]], max_items: int = 3) -> List[str]:
    out: List[str] = []
    seen = set()
    for pair in pairs:
        phrase = f"person_{pair.get('human_index')} - {pair.get('object_category_name') or pair.get('object_index')}: {pair.get('verb_name') or pair.get('verb_id')}"
        if phrase in seen:
            continue
        out.append(phrase)
        seen.add(phrase)
        if len(out) >= max_items:
            break
    return out


def context_entities(candidate: Dict[str, Any]) -> List[str]:
    names = []
    for obj in candidate.get("object_boxes", []):
        name = obj.get("category_name")
        if name in CONTEXT_NAMES and name not in names:
            names.append(name)
    return names[:6]


def local_group_entities(candidate: Dict[str, Any]) -> List[str]:
    cats = defaultdict(int)
    for pair in candidate.get("hoi_pairs", []):
        name = pair.get("object_category_name") or f"object_{pair.get('object_category_id')}"
        cats[name] += 1
    return [name for name, count in sorted(cats.items(), key=lambda x: (-x[1], x[0])) if count >= 2][:4]


def write_notes(path: Path, candidates: Sequence[Dict[str, Any]], preview_infos: Dict[int, Dict[str, Any]], top_k: int) -> None:
    lines: List[str] = []
    lines.append("# Top 10 Figure 1 Candidate Notes")
    lines.append("")
    for candidate in candidates[:top_k]:
        rank = candidate.get("rank")
        info = preview_infos.get(rank, {})
        selected = info.get("selected_pairs") or select_representative_pairs(candidate, max_pairs=3)
        show_pairs = unique_pair_phrases(selected, max_items=3)
        contexts = context_entities(candidate)
        local_groups = local_group_entities(candidate)
        score = candidate.get("figure1_score")
        filename = candidate.get("filename")
        preview_path = info.get("preview_path", "")

        if rank <= 3:
            suitability = "主图候选"
        elif rank <= 7:
            suitability = "强备用图"
        else:
            suitability = "备用图"

        lines.append(f"## Rank {rank}: `{filename}`")
        lines.append("")
        lines.append(f"- Score: `{score:.2f}`")
        lines.append(f"- Preview: `{preview_path}`")
        lines.append(
            f"- 为什么适合 Figure 1：{candidate.get('final_reason', '该图包含多个候选 HOI pair，具备 pair-specific evidence 展示潜力。')}"
        )
        lines.append("- 建议展示的 2-3 个 candidate pairs：" + ("；".join(show_pairs) if show_pairs else "无可用 pair"))
        chain_suggestions = []
        for pair in selected[:3]:
            verb = pair.get("verb_name") or f"verb_{pair.get('verb_id')}"
            obj = pair.get("object_category_name") or f"object_{pair.get('object_index')}"
            chain_suggestions.append(
                f"person_{pair.get('human_index')} - {obj}: 可画 human pose/接触区域 -> {obj} 局部外观 -> `{verb}` 语义证据"
            )
        lines.append("- 每个 pair 可能的 evidence chain：" + ("；".join(chain_suggestions) if chain_suggestions else "无"))
        lines.append("- 适合画成 generic context 的实体：" + ("、".join(contexts) if contexts else "无明显上下文实体"))
        lines.append("- 适合画成 local group 的实体：" + ("、".join(local_groups) if local_groups else "无明显 local group"))
        lines.append(f"- 是否适合主图或备用图：{suitability}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def resolve_image_path(candidate: Dict[str, Any], image_root: Path) -> Path:
    filename = candidate.get("filename")
    if isinstance(filename, str):
        path = image_root / filename
        if path.exists():
            return path
    raw = candidate.get("image_path")
    if isinstance(raw, str):
        return Path(raw)
    return image_root / str(filename or "")


def main() -> int:
    args = parse_args()
    ranked_path = Path(args.ranked_json).resolve()
    metadata, ranked = load_ranked(ranked_path)
    image_root_text = args.image_root or metadata.get("image_root")
    if not image_root_text:
        raise ValueError("image root missing; pass --image-root or include metadata.image_root")
    image_root = Path(image_root_text).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    top_candidates = ranked[: args.top_k]
    preview_paths: List[Path] = []
    preview_infos: Dict[int, Dict[str, Any]] = {}
    manifest: List[Dict[str, Any]] = []
    for candidate in top_candidates:
        rank = int(candidate.get("rank"))
        image_path = resolve_image_path(candidate, image_root)
        safe_name = Path(candidate.get("filename", f"rank_{rank}.jpg")).stem
        out_path = out_dir / f"rank_{rank:02d}_{safe_name}_preview.jpg"
        info = make_preview(candidate, image_path, out_path)
        preview_paths.append(out_path)
        preview_infos[rank] = info
        manifest.append(
            {
                "rank": rank,
                "filename": candidate.get("filename"),
                "score": candidate.get("figure1_score"),
                "image_path": str(image_path),
                "preview_path": str(out_path),
                "selected_pair_count": len(info.get("selected_pairs", [])),
                "legend_lines": info.get("legend_lines", []),
            }
        )

    contact_path = out_dir / "top30_contact_sheet.jpg"
    make_contact_sheet(top_candidates, preview_paths, contact_path)
    notes_path = out_dir / "top10_candidate_notes.md"
    write_notes(notes_path, top_candidates, preview_infos, args.notes_top_k)
    manifest_path = out_dir / "preview_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated previews: {len(preview_paths)} files under {out_dir}")
    print(f"Contact sheet: {contact_path}")
    print(f"Top 10 notes: {notes_path}")
    print(f"Manifest: {manifest_path}")
    print("Top 10 brief recommendation")
    for candidate in top_candidates[: args.notes_top_k]:
        rank = candidate.get("rank")
        filename = candidate.get("filename")
        score = candidate.get("figure1_score")
        selected = preview_infos.get(rank, {}).get("selected_pairs", [])
        pairs = "; ".join(unique_pair_phrases(selected, max_items=3))
        suitability = "主图候选" if rank <= 3 else "强备用图" if rank <= 7 else "备用图"
        print(f"- #{rank} {filename} score={score:.1f}: {suitability}; 建议 pairs: {pairs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
