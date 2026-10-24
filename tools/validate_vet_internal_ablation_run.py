"""Validate VET internal ablation manifests and frozen conditions."""

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Tuple


MODES = (
    "anchor_raw",
    "anchor_svrm",
    "endpoints",
    "context_uniform",
    "path_selection",
    "path_reweight",
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_run(run_dir: Path, require_metrics: bool = False) -> Tuple[dict, List[str]]:
    errors: List[str] = []
    manifest_path = run_dir / "run_manifest.json"
    args_path = run_dir / "resolved_args.json"
    if not manifest_path.exists():
        return {}, [f"missing {manifest_path}"]
    if not args_path.exists():
        return {}, [f"missing {args_path}"]
    manifest, args = load_json(manifest_path), load_json(args_path)
    mode = manifest.get("mode")
    if mode not in MODES:
        errors.append(f"invalid mode: {mode}")
    checks = {
        "model_variant": "vet_internal_ablation",
        "setm_ablation_stage": 5,
        "use_vet": True,
        "use_lpt": True,
        "use_setm_ablation": True,
        "graph_text_dynamic_retrieval": False,
        "graph_layerwise_sync_write_visual": True,
        "graph_layerwise_sync_write_text": True,
        "graph_layerwise_sync_final_matching": False,
        "graph_evidence_topk": 4,
        "graph_num_layers": 2,
    }
    for key, expected in checks.items():
        if args.get(key) != expected:
            errors.append(f"{key}: expected {expected!r}, got {args.get(key)!r}")
    if args.get("graph_vet_ablation_mode") != mode:
        errors.append("manifest mode differs from resolved args")
    if int(manifest.get("expected_setm_calls", -1)) != int(args.get("graph_num_layers", -2)):
        errors.append("expected SETM call count differs from graph_num_layers")
    if manifest.get("run_kind") == "formal" and manifest.get("resume"):
        if not args.get("vet_allow_matching_resume", False):
            errors.append("formal run resumed without identical-run authorization")
    metrics_path = run_dir / "metrics.jsonl"
    if require_metrics and (not metrics_path.exists() or not metrics_path.read_text(encoding="utf-8").strip()):
        errors.append("missing or empty formal metrics.jsonl")
    return manifest, errors


def validate_root(root: Path, require_metrics: bool = False) -> dict:
    manifests: List[dict] = []
    errors: Dict[str, List[str]] = {}
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        manifest, run_errors = validate_run(manifest_path.parent, require_metrics)
        manifests.append(manifest)
        if run_errors:
            errors[str(manifest_path.parent)] = run_errors
    if not manifests:
        errors[str(root)] = ["no run manifests found"]
    hashes: Dict[str, set] = {}
    for manifest in manifests:
        dataset = str(manifest.get("dataset"))
        hashes.setdefault(dataset, set()).add(manifest.get("config_hash_excluding_mode_seed"))
    inconsistent = {dataset: sorted(values) for dataset, values in hashes.items() if len(values) > 1}
    if inconsistent:
        errors["config_hash_consistency"] = [json.dumps(inconsistent, sort_keys=True)]
    return {
        "passed": not errors,
        "run_count": len(manifests),
        "datasets": sorted(hashes),
        "modes": sorted({row.get("mode") for row in manifests}),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--require-metrics", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_root(args.root, args.require_metrics)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

