import argparse
import csv
import importlib.util
import importlib.machinery
import json
import os
import random
import sys
import types
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if "wandb" not in sys.modules:
    wandb_stub = types.ModuleType("wandb")
    wandb_stub.__spec__ = importlib.machinery.ModuleSpec("wandb", loader=None)
    wandb_stub.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_stub
if "src.models" not in sys.modules:
    models_pkg = types.ModuleType("src.models")
    models_pkg.__path__ = [str(REPO_ROOT / "src" / "models")]
    models_pkg.__spec__ = importlib.machinery.ModuleSpec("src.models", loader=None, is_package=True)
    sys.modules["src.models"] = models_pkg

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageDraw
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler, Subset
from torchvision.ops import box_iou

from src.data.datasets import build_dataset
from src.models.backbone import build_backbone
from src.models.transformer import build_transformer
from src.util import box_ops
from src.util.misc import collate_fn


HICO_VALID_OBJ_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]


def build_parser():
    arg_parser_path = REPO_ROOT / "src" / "engine" / "arg_parser.py"
    spec = importlib.util.spec_from_file_location("stip_arg_parser", arg_parser_path)
    arg_parser_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arg_parser_module)
    parser = argparse.ArgumentParser(
        "Scan DETR score thresholds for STIP visual graph nodes",
        parents=[arg_parser_module.get_args_parser()],
        conflict_handler="resolve",
    )
    parser.add_argument("--detr_weights", default=None, type=str, help="Path to DETR checkpoint")
    parser.add_argument("--checkpoint", default="", type=str, help="Optional full STIP checkpoint")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_images", default=500, type=int, help="Number of images to scan; <=0 means all")
    parser.add_argument("--vis_images", default=16, type=int, help="Number of images to visualize")
    parser.add_argument("--vis_thresholds", default="0.05,0.10,0.15", type=str)
    parser.add_argument("--thresholds", default="0.01,0.03,0.05,0.07,0.10,0.15,0.20,0.30,0.50", type=str)
    parser.add_argument("--iou_thresh", default=0.5, type=float)
    parser.add_argument("--save_dir", default="threshold_scan", type=str)
    parser.add_argument("--score_topk", default=-1, type=int, help="Optional top-k cap after thresholding, -1 disables")
    parser.add_argument(
        "--simulate_forced_topk_pairs",
        default=32,
        type=int,
        help="Approximate forced proposal endpoints by top-k human-object candidate pairs; <=0 disables",
    )
    parser.add_argument("--human_thresh", default=0.5, type=float, help="Human threshold used for simulated pairs")
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--seed", default=3407, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--dataset_file", default="hico-det", type=str)
    parser.add_argument("--HOIDet", action="store_true", default=False)
    parser.add_argument("--no_aux_loss", dest="aux_loss", action="store_false", default=False)
    return parser


def parse_float_list(value):
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def complete_args(args):
    args.HOIDet = False
    args.STIP_relation_head = False
    args.use_visual_scene_graph = False
    args.aux_loss = False
    args.num_classes = 91
    args.num_actions = 117
    args.valid_obj_ids = HICO_VALID_OBJ_IDS
    args.correct_mat = torch.ones(args.num_actions, len(args.valid_obj_ids))
    args.invalid_ids = []
    args.valid_ids = list(range(args.num_actions))
    args.action_names = [str(i) for i in range(args.num_actions)]
    args.share_enc = getattr(args, "share_enc", False)
    args.pretrained_dec = getattr(args, "pretrained_dec", False)
    args.finetune_detr = getattr(args, "finetune_detr", False)
    args.train_detr = getattr(args, "train_detr", False)
    args.resume = args.checkpoint
    return args


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DETROnly(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries):
        super().__init__()
        hidden_dim = transformer.d_model
        self.backbone = backbone
        self.transformer = transformer
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)


def build_detr_only(args):
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    return DETROnly(
        backbone=backbone,
        transformer=transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
    )


def load_detr_weights(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if any(k.startswith("detr.") for k in state_dict):
        state_dict = {k.replace("detr.", "", 1): v for k, v in state_dict.items() if k.startswith("detr.")}
    state_dict = filter_compatible_state_dict(model.detr if hasattr(model, "detr") else model, state_dict)
    detr = model.detr if hasattr(model, "detr") else model
    missing, unexpected = detr.load_state_dict(state_dict, strict=False)
    print(f"Loaded DETR weights from {checkpoint_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def load_full_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if not hasattr(model, "detr") and any(k.startswith("detr.") for k in state_dict):
        state_dict = {k.replace("detr.", "", 1): v for k, v in state_dict.items() if k.startswith("detr.")}
    state_dict = filter_compatible_state_dict(model, state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded full checkpoint from {checkpoint_path}")
    print(f"Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")


def filter_compatible_state_dict(model, state_dict):
    model_state = model.state_dict()
    return {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}


@torch.no_grad()
def run_detr(model, samples):
    detr = model.detr if hasattr(model, "detr") else model
    features, pos = detr.backbone(samples)
    src, mask = features[-1].decompose()
    hs = detr.transformer(detr.input_proj(src), mask, detr.query_embed.weight, pos[-1])[0]
    outputs_class = detr.class_embed(hs)
    outputs_coord = detr.bbox_embed(hs).sigmoid()
    probs = outputs_class[-1].softmax(-1)
    scores, labels = probs[..., :-1].max(-1)
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(outputs_coord[-1]).clamp(0, 1)
    return scores, labels, boxes_xyxy


def normalized_gt_boxes(target, device):
    boxes = target["boxes"].to(device).float()
    orig_h, orig_w = target["orig_size"].to(device).float()
    scale = torch.stack([orig_w, orig_h, orig_w, orig_h])
    return (boxes / scale).clamp(0, 1)


def unique_hoi_node_indices(target):
    if "hois" not in target or target["hois"].numel() == 0:
        return torch.empty(0, dtype=torch.long)
    hois = target["hois"]
    if hois.dim() == 1:
        hois = hois.view(-1, 3)
    return torch.unique(hois[:, :2].reshape(-1).long())


def hoi_pair_indices(target):
    if "hois" not in target or target["hois"].numel() == 0:
        return torch.empty(0, 2, dtype=torch.long)
    hois = target["hois"]
    if hois.dim() == 1:
        hois = hois.view(-1, 3)
    return torch.unique(hois[:, :2].long(), dim=0)


def class_aware_match(pred_boxes, pred_labels, gt_boxes, gt_labels, iou_thresh):
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return (
            torch.zeros(pred_boxes.shape[0], dtype=torch.bool, device=pred_boxes.device),
            torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device),
            torch.full((gt_boxes.shape[0],), -1, dtype=torch.long, device=gt_boxes.device),
        )

    ious = box_iou(gt_boxes, pred_boxes)
    same_class = gt_labels[:, None] == pred_labels[None, :]
    ious = ious * same_class.float()
    best_iou, best_pred = ious.max(dim=1)
    gt_matched = best_iou >= iou_thresh
    pred_matched = torch.zeros(pred_boxes.shape[0], dtype=torch.bool, device=pred_boxes.device)
    if gt_matched.any():
        pred_matched[best_pred[gt_matched]] = True
    best_pred = best_pred.masked_fill(~gt_matched, -1)
    return pred_matched, gt_matched, best_pred


def build_threshold_mask(scores, threshold, score_topk):
    mask = scores > threshold
    if score_topk > 0 and int(mask.sum().item()) > score_topk:
        keep = scores.topk(score_topk).indices
        limited = torch.zeros_like(mask)
        limited[keep] = True
        mask = limited
    return mask


def simulate_forced_pair_mask(scores, labels, topk_pairs, human_thresh, dataset_file):
    mask = torch.zeros_like(scores, dtype=torch.bool)
    if topk_pairs <= 0:
        return mask, 0, 0
    human_ids = ((labels == 1) & (scores > human_thresh)).nonzero(as_tuple=False).flatten()
    if human_ids.numel() == 0:
        human_ids = scores.topk(min(1, scores.numel())).indices
    obj_mask = torch.ones_like(scores, dtype=torch.bool)
    if dataset_file != "vcoco":
        obj_mask[human_ids] = False
    obj_ids = obj_mask.nonzero(as_tuple=False).flatten()
    if obj_ids.numel() == 0:
        obj_ids = scores.topk(min(1, scores.numel())).indices

    pair_scores = (scores[human_ids][:, None] * scores[obj_ids][None, :]).flatten()
    pair_count = min(topk_pairs, pair_scores.numel())
    if pair_count == 0:
        return mask, 0, 0
    top_ids = pair_scores.topk(pair_count).indices
    h = human_ids[top_ids // obj_ids.numel()]
    o = obj_ids[top_ids % obj_ids.numel()]
    endpoints = torch.unique(torch.cat([h, o], dim=0))
    mask[endpoints] = True
    return mask, int(pair_count), int(endpoints.numel())


def update_average(stats, threshold, image_stats):
    entry = stats[threshold]
    if "images" not in entry:
        entry.update({
            "images": 0,
            "node_counts": [],
            "final_node_counts": [],
        })
    entry["images"] += 1
    for key, value in image_stats.items():
        if isinstance(value, (int, float)):
            entry[key] += value
        elif isinstance(value, list):
            entry[key].extend(value)


def summarize(stats):
    rows = []
    for threshold, values in sorted(stats.items()):
        images = max(values["images"], 1)
        gt_all = max(values["gt_all"], 1)
        gt_human = max(values["gt_human"], 1)
        gt_hoi_nodes = max(values["gt_hoi_nodes"], 1)
        gt_hoi_pairs = max(values["gt_hoi_pairs"], 1)
        pred_nodes = max(values["pred_nodes"], 1)
        final_nodes = max(values["final_nodes"], 1)
        row = {
            "threshold": threshold,
            "images": values["images"],
            "mean_nodes": values["pred_nodes"] / images,
            "p50_nodes": float(np.percentile(values["node_counts"], 50)) if values["node_counts"] else 0.0,
            "p90_nodes": float(np.percentile(values["node_counts"], 90)) if values["node_counts"] else 0.0,
            "mean_forced_nodes": values["forced_nodes"] / images,
            "mean_final_nodes": values["final_nodes"] / images,
            "p90_final_nodes": float(np.percentile(values["final_node_counts"], 90)) if values["final_node_counts"] else 0.0,
            "gt_obj_recall": values["matched_all"] / gt_all,
            "gt_human_recall": values["matched_human"] / gt_human,
            "hoi_endpoint_recall": values["matched_hoi_nodes"] / gt_hoi_nodes,
            "hoi_pair_endpoint_recall": values["matched_hoi_pairs"] / gt_hoi_pairs,
            "node_precision": values["matched_pred_nodes"] / pred_nodes,
            "final_node_precision": values["matched_final_nodes"] / final_nodes,
            "forced_node_ratio": values["forced_only_nodes"] / final_nodes,
            "mean_simulated_pairs": values["simulated_pairs"] / images,
        }
        rows.append(row)
    return rows


def draw_box(draw, box, color, text, width=3):
    draw.rectangle(box, outline=color, width=width)
    draw.text((box[0] + 2, max(0, box[1] - 12)), text, fill=color)


def scale_box(box, width, height):
    x1, y1, x2, y2 = box.detach().cpu().tolist()
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def save_visualization(raw_img, target, scores, labels, boxes, threshold, score_topk, forced_mask, iou_thresh, save_path):
    image = raw_img.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    device = boxes.device

    gt_boxes = normalized_gt_boxes(target, device)
    gt_labels = target["labels"].to(device).long()
    hoi_nodes = unique_hoi_node_indices(target).to(device)
    pred_mask = build_threshold_mask(scores, threshold, score_topk)
    final_mask = pred_mask | forced_mask

    pred_indices = pred_mask.nonzero(as_tuple=False).flatten()
    pred_matched, gt_matched, _ = class_aware_match(
        boxes[pred_mask], labels[pred_mask], gt_boxes, gt_labels, iou_thresh
    )
    final_pred_matched, final_gt_matched, _ = class_aware_match(
        boxes[final_mask], labels[final_mask], gt_boxes, gt_labels, iou_thresh
    )
    missed_hoi_nodes = []
    if hoi_nodes.numel() > 0:
        missed_hoi_nodes = hoi_nodes[~final_gt_matched[hoi_nodes]].detach().cpu().tolist()

    for gt_idx in hoi_nodes.detach().cpu().tolist():
        draw_box(draw, scale_box(gt_boxes[gt_idx], width, height), "lime", f"GT hoi cls={int(gt_labels[gt_idx])}", width=2)

    for local_idx, pred_idx in enumerate(pred_indices.detach().cpu().tolist()):
        color = "deepskyblue" if bool(pred_matched[local_idx]) else "red"
        label = f"q{pred_idx} c={int(labels[pred_idx])} s={float(scores[pred_idx]):.2f}"
        draw_box(draw, scale_box(boxes[pred_idx], width, height), color, label, width=2)

    forced_only = (forced_mask & ~pred_mask).nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    for pred_idx in forced_only[:80]:
        label = f"forced q{pred_idx} c={int(labels[pred_idx])} s={float(scores[pred_idx]):.2f}"
        draw_box(draw, scale_box(boxes[pred_idx], width, height), "orange", label, width=2)

    for gt_idx in missed_hoi_nodes:
        draw_box(draw, scale_box(gt_boxes[gt_idx], width, height), "yellow", f"MISS cls={int(gt_labels[gt_idx])}", width=4)

    title = (
        f"thr={threshold:.2f} raw_nodes={int(pred_mask.sum())} "
        f"forced_only={len(forced_only)} final_nodes={int(final_mask.sum())} "
        f"matched_gt={int(final_gt_matched.sum())}/{gt_boxes.shape[0]}"
    )
    draw.rectangle([0, 0, min(width, 850), 26], fill="black")
    draw.text((6, 6), title, fill="white")
    image.save(save_path)


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(rows, path):
    if not rows:
        return
    thresholds = [r["threshold"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(thresholds, [r["mean_nodes"] for r in rows], marker="o", label="mean raw nodes")
    ax1.plot(thresholds, [r["mean_final_nodes"] for r in rows], marker="o", label="mean final nodes")
    ax1.set_xlabel("score threshold")
    ax1.set_ylabel("nodes per image")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(thresholds, [r["hoi_endpoint_recall"] for r in rows], marker="s", color="green", label="HOI endpoint recall")
    ax2.plot(thresholds, [r["node_precision"] for r in rows], marker="s", color="red", label="node precision")
    ax2.set_ylabel("recall / precision")
    ax2.set_ylim(0, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = complete_args(build_parser().parse_args())
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    thresholds = parse_float_list(args.thresholds)
    vis_thresholds = parse_float_list(args.vis_thresholds)
    save_dir = Path(args.save_dir)
    vis_dir = save_dir / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    args.device = str(device)

    dataset = build_dataset(args.split, args)
    if args.max_images > 0:
        dataset = Subset(dataset, list(range(min(args.max_images, len(dataset)))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    model = build_detr_only(args)
    model.to(device)
    model.eval()
    load_detr_weights(model, args.detr_weights, device)
    load_full_checkpoint(model, args.checkpoint, device)

    stats = defaultdict(lambda: defaultdict(float))
    visualized = 0
    processed = 0

    for batch_idx, (samples, targets, raw_images) in enumerate(loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        scores_b, labels_b, boxes_b = run_detr(model, samples)

        for img_idx, target in enumerate(targets):
            scores = scores_b[img_idx]
            labels = labels_b[img_idx]
            boxes = boxes_b[img_idx]
            gt_boxes = normalized_gt_boxes(target, device)
            gt_labels = target["labels"].to(device).long()
            hoi_nodes = unique_hoi_node_indices(target).to(device)
            hoi_pairs = hoi_pair_indices(target).to(device)
            human_gt_mask = gt_labels == 1
            forced_mask, pair_count, forced_endpoint_count = simulate_forced_pair_mask(
                scores, labels, args.simulate_forced_topk_pairs, args.human_thresh, args.dataset_file
            )

            for threshold in thresholds:
                pred_mask = build_threshold_mask(scores, threshold, args.score_topk)
                final_mask = pred_mask | forced_mask
                pred_matched, gt_matched, _ = class_aware_match(
                    boxes[pred_mask], labels[pred_mask], gt_boxes, gt_labels, args.iou_thresh
                )
                final_pred_matched, final_gt_matched, _ = class_aware_match(
                    boxes[final_mask], labels[final_mask], gt_boxes, gt_labels, args.iou_thresh
                )

                matched_hoi_nodes = 0
                if hoi_nodes.numel() > 0:
                    matched_hoi_nodes = int(final_gt_matched[hoi_nodes].sum().item())

                matched_hoi_pairs = 0
                if hoi_pairs.numel() > 0:
                    pair_ok = final_gt_matched[hoi_pairs[:, 0]] & final_gt_matched[hoi_pairs[:, 1]]
                    matched_hoi_pairs = int(pair_ok.sum().item())

                forced_only = forced_mask & ~pred_mask
                image_stats = {
                    "gt_all": int(gt_boxes.shape[0]),
                    "gt_human": int(human_gt_mask.sum().item()),
                    "gt_hoi_nodes": int(hoi_nodes.numel()),
                    "gt_hoi_pairs": int(hoi_pairs.shape[0]),
                    "matched_all": int(final_gt_matched.sum().item()),
                    "matched_human": int(final_gt_matched[human_gt_mask].sum().item()) if human_gt_mask.any() else 0,
                    "matched_hoi_nodes": matched_hoi_nodes,
                    "matched_hoi_pairs": matched_hoi_pairs,
                    "pred_nodes": int(pred_mask.sum().item()),
                    "final_nodes": int(final_mask.sum().item()),
                    "forced_nodes": forced_endpoint_count,
                    "forced_only_nodes": int(forced_only.sum().item()),
                    "matched_pred_nodes": int(pred_matched.sum().item()),
                    "matched_final_nodes": int(final_pred_matched.sum().item()),
                    "simulated_pairs": pair_count,
                    "node_counts": [int(pred_mask.sum().item())],
                    "final_node_counts": [int(final_mask.sum().item())],
                }
                update_average(stats, threshold, image_stats)

                if visualized < args.vis_images and any(abs(threshold - vt) < 1e-9 for vt in vis_thresholds):
                    image_id = int(target["image_id"].flatten()[0].item()) if "image_id" in target else processed
                    save_path = vis_dir / f"img{processed:05d}_id{image_id}_thr{threshold:.2f}.jpg"
                    save_visualization(
                        raw_images[img_idx], target, scores, labels, boxes, threshold, args.score_topk,
                        forced_mask, args.iou_thresh, save_path
                    )
            processed += 1
            if visualized < args.vis_images:
                visualized += 1

        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {processed} images")

    rows = summarize(stats)
    save_csv(rows, save_dir / "summary.csv")
    with open(save_dir / "summary.json", "w") as f:
        json.dump(rows, f, indent=2)
    save_plot(rows, save_dir / "threshold_curves.png")

    print(f"Processed images: {processed}")
    print(f"Saved summary: {save_dir / 'summary.csv'}")
    print(f"Saved curves: {save_dir / 'threshold_curves.png'}")
    print(f"Saved visualizations: {vis_dir}")
    for row in rows:
        print(
            f"thr={row['threshold']:.2f} mean_nodes={row['mean_nodes']:.1f} "
            f"mean_final={row['mean_final_nodes']:.1f} "
            f"hoi_recall={row['hoi_endpoint_recall']:.3f} "
            f"precision={row['node_precision']:.3f}"
        )


if __name__ == "__main__":
    main()
