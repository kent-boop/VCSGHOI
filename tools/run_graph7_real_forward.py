import argparse
import importlib.util
import json
import os
import random
import traceback
from collections import OrderedDict
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")
sys.modules.setdefault("wandb", types.SimpleNamespace(init=lambda *args, **kwargs: None, log=lambda *args, **kwargs: None))

import src.util.misc as utils
from src.data.datasets import build_dataset


def _load_get_args_parser():
    module_path = REPO_ROOT / "src" / "engine" / "arg_parser.py"
    spec = importlib.util.spec_from_file_location("graph7_real_forward_arg_parser", module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.get_args_parser


get_args_parser = _load_get_args_parser()


def _build_args():
    parser = argparse.ArgumentParser(
        "Graph7 reciprocal SETM real forward smoke test",
        parents=[get_args_parser()],
    )
    parser.add_argument("--detr_weights", default=None, type=str)
    parser.add_argument("--train_detr", action="store_true", default=False)
    parser.add_argument("--finetune_detr_weight", default=0.1, type=float)
    parser.add_argument("--lr_detr", default=1e-5, type=float)
    parser.add_argument("--reduce_lr_on_plateau_patience", default=2, type=int)
    parser.add_argument("--reduce_lr_on_plateau_factor", default=0.1, type=float)
    parser.add_argument("--proposal_focal_loss_alpha", default=0.75, type=float)
    parser.add_argument("--action_focal_loss_alpha", default=0.5, type=float)
    parser.add_argument("--proposal_focal_loss_gamma", default=2, type=float)
    parser.add_argument("--action_focal_loss_gamma", default=2, type=float)
    parser.add_argument("--proposal_loss_coef", default=1, type=float)
    parser.add_argument("--action_loss_coef", default=1, type=float)
    parser.add_argument("--action_sampled_loss_coef", default=1.0, type=float)
    parser.add_argument("--action_fore_loss_coef", default=1.0, type=float)
    parser.add_argument("--global_action_fore_loss_coef", default=1.0, type=float)
    parser.add_argument("--action_fea_loss_coef", default=1.0, type=float)
    parser.add_argument("--no_hard_mining_for_relation_discovery", dest="use_hard_mining_for_relation_discovery", action="store_false", default=True)
    parser.add_argument("--no_relation_dependency_encoding", dest="use_relation_dependency_encoding", action="store_false", default=True)
    parser.add_argument("--no_memory_layout_encoding", dest="use_memory_layout_encoding", action="store_false", default=True)
    parser.add_argument("--no_nms_on_detr", dest="apply_nms_on_detr", action="store_false", default=True)
    parser.add_argument("--no_tail_semantic_feature", dest="use_tail_semantic_feature", action="store_false", default=True)
    parser.add_argument("--no_spatial_feature", dest="use_spatial_feature", action="store_false", default=True)
    parser.add_argument("--no_interaction_decoder", action="store_true", default=False)
    parser.add_argument("--use_memory_union_mask", action="store_true", default=False)
    parser.add_argument("--use_union_feature", action="store_true", default=False)
    parser.add_argument("--adaptive_relation_query_num", action="store_true", default=False)
    parser.add_argument("--use_relation_tgt_mask", action="store_true", default=False)
    parser.add_argument("--use_relation_tgt_mask_attend_topk", default=10, type=int)
    parser.add_argument("--use_prior_verb_label_mask", action="store_true", default=False)
    parser.add_argument("--relation_feature_map_from", default="backbone")
    parser.add_argument("--use_query_fourier_encoding", action="store_true", default=False)
    parser.add_argument("--proj_blip", action="store_true", default=True)
    parser.add_argument("--merge_blip", action="store_true", default=True)
    parser.add_argument("--only_blip", action="store_true", default=False)
    parser.add_argument("--zero_shot_type", default="default", choices=["default", "RF", "NF"])
    parser.add_argument("--graph_text_export_each_layer", nargs="?", const=True, default=True, type=_str2bool)
    parser.add_argument("--run_train_forward", action="store_true", default=False)
    return parser.parse_args()


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _move_targets_to_device(targets, device):
    moved = []
    for target in targets:
        moved.append({k: v.to(device) if torch.is_tensor(v) else v for k, v in target.items()})
    return moved


def _load_detr_weights(model, path):
    if not path:
        return {"loaded": False, "path": ""}
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    if "hico_ft_q16.pth" in path:
        mapped = OrderedDict()
        for key, value in state.items():
            if key.startswith("detr."):
                mapped[key.replace("detr.", "")] = value
        msg = model.detr.load_state_dict(mapped)
    else:
        msg = model.detr.load_state_dict(state)
    return {
        "loaded": True,
        "path": path,
        "missing_keys": list(msg.missing_keys),
        "unexpected_keys": list(msg.unexpected_keys),
    }


def _summarize_debug(out):
    debug_list = out.get("graph_reasoner_debug", [])
    if not debug_list:
        raise RuntimeError("out['graph_reasoner_debug'] is empty")
    debug = debug_list[0]
    keys = [
        "reciprocal_setm_enabled",
        "reciprocal_setm_applied",
        "reciprocal_setm_layer_count",
        "reciprocal_valid_pair_count",
        "reciprocal_valid_prototype_count",
        "prototype_slot_valid_ratio",
        "reciprocal_path_context_valid_ratio",
        "reciprocal_text_state_updated_slot_count",
        "reciprocal_setm_has_nan",
        "text_dense_total_edge_update_count",
        "text_dense_total_node_update_count",
        "has_nan",
        "pair_feats_shape",
        "text_layer_count",
        "text_layerwise_enabled",
        "graph_text_dense_injection",
        "text_dynamic_retrieval",
        "reciprocal_write_visual",
        "reciprocal_write_visual_applied",
    ]
    return {key: debug.get(key) for key in keys}, debug


def _run_forward(model, batch, device, mode):
    samples, targets, images = batch
    samples = samples.to(device)
    targets = _move_targets_to_device(targets, device)
    if mode == "eval":
        model.eval()
    elif mode == "train":
        model.train()
    else:
        raise ValueError(mode)
    with torch.no_grad():
        out = model(samples, images, targets)
    summary, debug = _summarize_debug(out)
    graph_shape = summary.get("pair_feats_shape")
    pred_actions_shape = tuple(out["pred_actions"][0].shape) if out.get("pred_actions") else None
    rel_pairs_shape = tuple(out["pred_rel_pairs"][0].shape) if out.get("pred_rel_pairs") else None
    return {
        "mode": mode,
        "summary": summary,
        "pred_actions_shape": pred_actions_shape,
        "pred_rel_pairs_shape": rel_pairs_shape,
        "graph_pair_feats_shape": graph_shape,
        "full_debug_first_image": debug,
    }


def main():
    args = _build_args()
    args.STIP_relation_head = True
    args.distributed = False
    args.seed = 3407
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    import src.models.detr as detr_module
    import src.models.stip_graph7 as stip_graph7_module

    detr_module.STIP = stip_graph7_module.STIP
    detr_module.STIPCriterion = stip_graph7_module.STIPCriterion
    detr_module.STIPPostProcess = stip_graph7_module.STIPPostProcess

    device = torch.device(args.device)
    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)
    assert dataset_train.num_action() == dataset_val.num_action()
    args.num_classes = dataset_train.num_category()
    args.num_actions = dataset_train.num_action()
    args.action_names = dataset_train.get_actions()
    if args.dataset_file == "hico-det":
        args.valid_obj_ids = dataset_train.get_valid_obj_ids()
        args.correct_mat = torch.tensor(dataset_val.correct_mat).to(device)
    elif args.dataset_file == "vcoco":
        args.valid_ids = np.array(dataset_train.get_object_label_idx()).nonzero()[0]
        args.invalid_ids = np.argwhere(np.array(dataset_train.get_object_label_idx()) == 0).squeeze(1)
        args.human_actions = dataset_train.get_human_action()
        args.object_actions = dataset_train.get_object_action()
        args.num_human_act = dataset_train.num_human_act()

    from src.models import build_model

    model, _, _ = build_model(args)
    weight_info = _load_detr_weights(model, args.detr_weights)
    model.to(device)

    val_loader = DataLoader(dataset_val, batch_size=1, sampler=torch.utils.data.SequentialSampler(dataset_val), drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    train_loader = DataLoader(dataset_train, batch_size=1, sampler=torch.utils.data.SequentialSampler(dataset_train), drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    val_batch = next(iter(val_loader))
    results = [
        _run_forward(model, val_batch, device, "eval"),
    ]
    if args.run_train_forward:
        train_batch = next(iter(train_loader))
        results.append(_run_forward(model, train_batch, device, "train"))

    guard = {
        "using_graph7_class": model.__class__.__module__.endswith("stip_graph7"),
        "visual_reasoner_module": model.visual_scene_graph.__class__.__module__,
        "candidate_evidence_inducer_attr": hasattr(model.visual_scene_graph, "candidate_evidence_inducer"),
        "graph_enable_layerwise_sync_evidence_matching": bool(getattr(args, "graph_enable_layerwise_sync_evidence_matching", False)),
        "graph_enable_candidate_evidence": bool(getattr(args, "graph_enable_candidate_evidence", False)),
        "graph_enable_sync_evidence_matching": bool(getattr(args, "graph_enable_sync_evidence_matching", False)),
        "graph_text_dense_injection": bool(getattr(args, "graph_text_dense_injection", False)),
        "graph_reciprocal_write_visual": bool(getattr(args, "graph_reciprocal_write_visual", False)),
    }
    print(json.dumps({"status": "ok", "weights": weight_info, "guard": guard, "results": results}, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
