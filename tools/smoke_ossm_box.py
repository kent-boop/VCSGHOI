#!/usr/bin/env python3
"""One-batch scratch-training smoke test for the SAM-free OSSM ROI variant."""

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Avoid importing optional TensorFlow/Flax backends through transformers.  The
# project uses only PyTorch and some cluster images have incompatible TF libs.
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

# wandb is optional for this local smoke test; the repository imports it from
# generic training modules even when logging is disabled.
try:
    import wandb  # noqa: F401
except ModuleNotFoundError:
    import types

    wandb_stub = types.ModuleType("wandb")
    wandb_stub.log = lambda *args, **kwargs: None
    sys.modules["wandb"] = wandb_stub

from src.data.datasets import build_dataset
from src.models import build_model
from src.models.ossm_blip_roi import BLIP_MEAN, BLIP_STD
from src.util import misc as utils


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detr_checkpoint", required=True, help="Pretrained DETR checkpoint")
    parser.add_argument("--blip_checkpoint", required=True, help="Matching BLIP retrieval checkpoint")
    parser.add_argument("--text_library", required=True, help="Matching hierarchical text .npy")
    parser.add_argument("--data_path", required=True, help="HICO-DET root")
    parser.add_argument("--prior", default="obj_act_prior.npy", help="80x117 object-action prior")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--roundtrip_path", default="", help="Optional path for save/reload validation")
    parser.add_argument("--visualization_path", default="", help="Optional PNG showing human/object/union ROI boxes")
    return parser.parse_args()


def _set_default(namespace, name, value):
    if not hasattr(namespace, name):
        setattr(namespace, name, value)


def _migration_args(cli, checkpoint):
    if "args" not in checkpoint:
        raise KeyError("Real-data smoke test needs the DETR checkpoint's saved args")
    args = checkpoint["args"]
    args.data_path = cli.data_path
    args.device = cli.device
    args.HOIDet = True
    args.STIP_relation_head = True
    args.stip_model_variant = "ossm_box"
    args.model_variant = None
    args.ossm_blip_pretrained = cli.blip_checkpoint
    args.ossm_blip_vit = "large"
    args.ossm_text_library = cli.text_library
    args.ossm_obj_act_prior = cli.prior
    args.ossm_region_image_size = 384
    args.ossm_debug_regions = True
    args.num_workers = cli.num_workers
    args.train_detr = False
    args.resume = ""
    args.init_from_ossm = ""
    args.detr_weights = cli.detr_checkpoint
    # Preserve old first-comparison loss semantics explicitly.
    args.proposal_loss_coef = 5.0
    args.action_loss_coef = 1.0
    args.action_sampled_loss_coef = 0.0

    defaults = {
        "frozen_weights": None,
        "use_high_resolution_relation_feature_map": False,
        "pair_filter_threshold": 0.08,
        "use_pair_score_weighting": True,
        "pair_dedupe_iou_thr": 0.0,
        "pair_dedupe_ignore_obj_label": False,
        "zero_shot_type": "default",
        "finetune_detr_weight": 0.1,
        "use_moe_visual_router": True,
        "use_moe_aux_loss": True,
        "use_moe_noisy_gating": True,
    }
    for name, value in defaults.items():
        _set_default(args, name, value)
    return args


def _prepare_hico_metadata(args):
    train_dataset = build_dataset(image_set="train", args=args)
    val_dataset = build_dataset(image_set="test", args=args)
    args.num_classes = train_dataset.num_category()
    args.num_actions = train_dataset.num_action()
    args.action_names = train_dataset.get_actions()
    args.valid_obj_ids = train_dataset.get_valid_obj_ids()
    args.correct_mat = torch.tensor(val_dataset.correct_mat)
    return train_dataset


def _parameter_counts(model):
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "detr_trainable": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("detr.") and parameter.requires_grad
        ),
        "blip_trainable": sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if name.startswith("blip.") and parameter.requires_grad
        ),
    }


def _save_region_visualization(debug, output_path):
    image = debug["roi_image"].float()
    mean = image.new_tensor(BLIP_MEAN).view(3, 1, 1)
    std = image.new_tensor(BLIP_STD).view(3, 1, 1)
    rgb = (image * std + mean).clamp(0, 1)
    array = (rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    visualization = Image.fromarray(array)
    draw = ImageDraw.Draw(visualization)
    width, height = visualization.size
    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    for key, color in (
        ("human_boxes_xyxy", "lime"),
        ("object_boxes_xyxy", "red"),
        ("union_boxes_xyxy", "yellow"),
    ):
        for box in debug[key][:3].float() * scale:
            draw.rectangle(box.tolist(), outline=color, width=2)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    visualization.save(path)
    print(f"[OSSM smoke] region visualization: {path}")


def main():
    cli = _parse_args()
    raw_checkpoint = torch.load(cli.detr_checkpoint, map_location="cpu")
    args = _migration_args(cli, raw_checkpoint)
    dataset = _prepare_hico_metadata(args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cli.num_workers,
        collate_fn=utils.collate_fn,
    )

    model, criterion, postprocessors = build_model(args)
    model.detr.load_state_dict(raw_checkpoint["model"], strict=True)
    del raw_checkpoint
    print("[OSSM smoke] loaded pretrained DETR; OSSM ROI modules start from scratch")
    device = torch.device(cli.device)
    model.to(device)
    criterion.to(device)
    args.correct_mat = args.correct_mat.to(device)
    counts = _parameter_counts(model)
    print("[OSSM smoke] parameters", json.dumps(counts, sort_keys=True))
    if counts["detr_trainable"] or counts["blip_trainable"]:
        raise AssertionError("Frozen DETR/BLIP unexpectedly contains trainable parameters")

    samples, targets, images = next(iter(loader))
    samples = samples.to(device)
    targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr), weight_decay=float(args.weight_decay))

    model.train()
    criterion.train()
    outputs = model(samples, images, targets)
    loss_dict = criterion(outputs, targets)
    loss = sum(
        loss_dict[name] * criterion.weight_dict[name]
        for name in loss_dict
        if name in criterion.weight_dict
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite smoke loss: {loss_dict}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    finite_gradient_tensors = 0
    for parameter in trainable:
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("Non-finite gradient in OSSM smoke test")
            finite_gradient_tensors += 1
    if finite_gradient_tensors == 0:
        raise AssertionError("No trainable OSSM parameter received a gradient")
    optimizer.step()

    model.eval()
    with torch.no_grad():
        eval_outputs = model(samples, images, targets)
        original_sizes = torch.stack([target["orig_size"] for target in targets])
        results = postprocessors["hoi"](
            eval_outputs, original_sizes, dataset=args.dataset_file
        )
    if len(results) != 1 or "verb_scores" not in results[0]:
        raise AssertionError("HICO postprocessor output contract failed")

    if cli.visualization_path:
        if model.last_box_regions is None:
            raise AssertionError("OSSM region debug tensors were not retained")
        _save_region_visualization(model.last_box_regions, cli.visualization_path)

    if cli.roundtrip_path:
        roundtrip_path = Path(cli.roundtrip_path)
        roundtrip_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "args": args}, roundtrip_path)
        saved = torch.load(roundtrip_path, map_location="cpu")["model"]
        incompatible = model.load_state_dict(saved, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise AssertionError(f"Roundtrip state mismatch: {incompatible}")

    summary = {
        "loss": float(loss.detach().cpu()),
        "loss_keys": sorted(loss_dict),
        "train_proposals": len(outputs["pred_rel_pairs"][0]),
        "eval_proposals": len(eval_outputs["pred_rel_pairs"][0]),
        "postprocess_verb_shape": list(results[0]["verb_scores"].shape),
        "finite_gradient_tensors": finite_gradient_tensors,
        "peak_cuda_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2)
            if device.type == "cuda"
            else 0.0
        ),
        "sam_runtime": False,
        "evidence_graph_runtime": False,
        "old_ossm_initialized": False,
    }
    print("[OSSM smoke] PASS", json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
