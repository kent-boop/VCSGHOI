"""Dedicated launcher for VET internal ablations.

Unlike the legacy entry, this file respects ``--seed`` and records the fully
resolved command-line configuration before training.
"""

import argparse
import hashlib
import subprocess
import json
from pathlib import Path

import STIP_main as legacy_main
from src.engine.arg_parser import get_args_parser
from src.util.vet_ablation_run import install_dataset_limits as _install_dataset_limits, write_run_manifest as _write_run_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "VET internal ablation training and evaluation",
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
    return parser


def _serializable_args(args) -> dict:
    resolved = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        try:
            json.dumps(value)
        except TypeError:
            value = repr(value)
        resolved[key] = value
    return resolved


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.STIP_relation_head = True
    if args.model_variant not in (None, "vet_internal_ablation"):
        raise ValueError("This entry only supports --model_variant vet_internal_ablation")
    if args.stip_model_variant not in ("default", "vet_internal_ablation"):
        raise ValueError("This entry only supports --stip_model_variant vet_internal_ablation")
    args.model_variant = "vet_internal_ablation"
    if args.vet_run_kind == "formal" and args.resume and not args.vet_allow_matching_resume:
        raise ValueError(
            "Formal VET ablations must start from the shared DETR initialization. "
            "Use --vet_allow_matching_resume only for an interrupted identical run."
        )
    if args.output_dir:
        args.output_dir += f"/{args.group_name}/{args.run_name}/"
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_args.json").write_text(
            json.dumps(_serializable_args(args), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        _write_run_manifest(args, output_dir)
    _install_dataset_limits(args)
    legacy_main.main(args)
    if args.output_dir:
        metric_lines = []
        for log_path in sorted(Path(args.output_dir).glob("*-log.txt")):
            metric_lines.extend(line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip())
        (Path(args.output_dir) / "metrics.jsonl").write_text("\n".join(metric_lines) + ("\n" if metric_lines else ""), encoding="utf-8")
        resource_metrics = {"peak_cuda_memory_bytes": int(legacy_main.torch.cuda.max_memory_allocated()) if legacy_main.torch.cuda.is_available() else None, "flops": None, "flops_status": "unavailable_without_optional_profiler"}
        (Path(args.output_dir) / "resource_metrics.json").write_text(json.dumps(resource_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

