"""Run metadata and smoke-dataset helpers for VET internal ablations."""

import hashlib
import json
import subprocess
from pathlib import Path

import STIP_main as legacy_main


def _serializable_args(args) -> dict:
    result = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        try:
            json.dumps(value)
        except TypeError:
            value = repr(value)
        result[key] = value
    return result


def install_dataset_limits(args) -> None:
    """Limit both train and validation splits without changing legacy main."""
    original_build_dataset = legacy_main.build_dataset

    def limited_build_dataset(image_set, args):
        dataset = original_build_dataset(image_set=image_set, args=args)
        limit = args.vet_train_max_samples if image_set == "train" else args.vet_val_max_samples
        if limit and int(limit) > 0:
            dataset = legacy_main.AttributeSubset(dataset, range(min(int(limit), len(dataset))))
        return dataset

    legacy_main.build_dataset = limited_build_dataset


def write_run_manifest(args, output_dir: Path) -> None:
    resolved = _serializable_args(args)
    excluded = {"graph_vet_ablation_mode", "seed", "output_dir", "run_name", "resume"}
    frozen = {key: value for key, value in resolved.items() if key not in excluded}
    config_hash = hashlib.sha256(
        json.dumps(frozen, sort_keys=True).encode("utf-8")
    ).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except Exception:
        commit, dirty = "unavailable", True
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset_file,
        "mode": args.graph_vet_ablation_mode,
        "seed": int(args.seed),
        "run_kind": args.vet_run_kind,
        "config_hash_excluding_mode_seed": config_hash,
        "git_commit": commit,
        "git_dirty": dirty,
        "detr_weights": args.detr_weights,
        "resume": args.resume,
        "graph_evidence_topk": int(args.graph_evidence_topk),
        "graph_num_layers": int(args.graph_num_layers),
        "expected_setm_calls": int(args.graph_num_layers),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

