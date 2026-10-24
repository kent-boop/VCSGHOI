"""Collect multi-seed VET ablation metrics without fabricating missing runs."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


MODE_ORDER = [
    ("anchor_raw", "V0 HO Anchor Only"),
    ("anchor_svrm", "V1 + Shared Visual Relation Memory"),
    ("endpoints", "V2 + Endpoint States"),
    ("context_uniform", "V3 + Context Paths"),
    ("path_selection", "V4 + Learned Path Selection"),
    ("path_reweight", "V5 + Path Reweighting"),
]
DELTAS = [
    ("Delta_SVRM", "anchor_svrm", "anchor_raw"),
    ("Delta_Endpoint", "endpoints", "anchor_svrm"),
    ("Delta_Context", "context_uniform", "endpoints"),
    ("Delta_Selection", "path_selection", "context_uniform"),
    ("Delta_Weight", "path_reweight", "path_selection"),
]
DATASET_METRICS = {
    "hico-det": [("Full", "mAP_full"), ("Rare", "mAP_rare"), ("Non-Rare", "mAP_non_rare")],
    "vcoco": [("AP_role#1", "Scenario1"), ("AP_role#2", "Scenario2")],
}
RESOURCE_METRICS = [("Params", "total_params"), ("FLOPs", "flops"), ("Peak Memory", "peak_cuda_memory_bytes"), ("Inference Time", "inference_time_ms")]



def read_latest_metrics(path: Path) -> dict:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-1] if rows else {}


def collect_runs(root: Path, include_smoke: bool) -> Tuple[List[dict], List[dict]]:
    valid, rejected = [], []
    for manifest_path in sorted(root.rglob("run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_kind") != "formal" and not include_smoke:
            rejected.append({"run_dir": str(run_dir), "reason": "non-formal run"})
            continue
        metrics = read_latest_metrics(run_dir / "metrics.jsonl")
        if not metrics:
            rejected.append({"run_dir": str(run_dir), "reason": "missing metrics"})
            continue
        resource_path = run_dir / "resource_metrics.json"
        resources = json.loads(resource_path.read_text(encoding="utf-8")) if resource_path.exists() else {}
        valid.append({**manifest, "metrics": metrics, "resources": resources, "run_dir": str(run_dir)})
    return valid, rejected


def mean_std(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    values = [float(value) for value in values]
    if not values:
        return None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize(runs: List[dict], allow_v5: bool) -> dict:
    by_key: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    hash_errors = []
    hashes: Dict[str, set] = defaultdict(set)
    for run in runs:
        by_key[(run["dataset"], run["mode"])].append(run)
        hashes[run["dataset"]].add(run.get("config_hash_excluding_mode_seed"))
    for dataset, values in hashes.items():
        if len(values) > 1:
            hash_errors.append({"dataset": dataset, "hashes": sorted(values)})

    tables, paired = {}, {}
    for dataset, metric_defs in DATASET_METRICS.items():
        rows = []
        seed_metrics: Dict[Tuple[str, int], dict] = {}
        for mode, label in MODE_ORDER:
            mode_runs = by_key.get((dataset, mode), [])
            row = {"mode": mode, "label": label, "seeds": sorted(int(r["seed"]) for r in mode_runs)}
            if mode == "path_reweight" and not allow_v5:
                row["status"] = "pending causal validation (--allow-v5 to include)"
            for display, key in metric_defs:
                values = [r["metrics"][key] for r in mode_runs if key in r["metrics"]]
                mean, std = mean_std(values)
                row[display] = {"mean": mean, "std": std, "n": len(values)}
            for display, key in RESOURCE_METRICS:
                values = [r.get("resources", {}).get(key) for r in mode_runs]
                values = [value for value in values if isinstance(value, (int, float))]
                mean, std = mean_std(values)
                row[display] = {"mean": mean, "std": std, "n": len(values)}
            for run in mode_runs:
                seed_metrics[(mode, int(run["seed"]))] = run["metrics"]
            rows.append(row)
        tables[dataset] = rows

        delta_rows = []
        for delta_name, high, low in DELTAS:
            if delta_name == "Delta_Weight" and not allow_v5:
                delta_rows.append({"delta": delta_name, "status": "pending causal validation"})
                continue
            shared_seeds = sorted(
                {seed for mode, seed in seed_metrics if mode == high}
                & {seed for mode, seed in seed_metrics if mode == low}
            )
            delta = {"delta": delta_name, "high": high, "low": low, "paired_seeds": shared_seeds}
            for display, key in metric_defs:
                values = [
                    float(seed_metrics[(high, seed)][key]) - float(seed_metrics[(low, seed)][key])
                    for seed in shared_seeds
                    if key in seed_metrics[(high, seed)] and key in seed_metrics[(low, seed)]
                ]
                mean, std = mean_std(values)
                delta[display] = {"mean": mean, "std": std, "n": len(values), "per_seed": values}
            delta_rows.append(delta)
        paired[dataset] = delta_rows
    return {"tables": tables, "paired_deltas": paired, "config_hash_errors": hash_errors}


def fmt(stat: dict) -> str:
    if not stat or stat.get("mean") is None:
        return "N/A"
    return "{:.3f} ± {:.3f}".format(stat["mean"], stat["std"])


def markdown_report(summary: dict, rejected: List[dict], allow_v5: bool) -> str:
    lines = ["# VET Internal Ablation Results", "", "Missing results remain N/A; no values are imputed.", ""]
    for dataset, rows in summary["tables"].items():
        metric_names = [name for name, _ in DATASET_METRICS[dataset]]
        table_names = metric_names + [name for name, _ in RESOURCE_METRICS]
        lines += [
            "## {}".format(dataset),
            "",
            "| Variant | " + " | ".join(table_names) + " | Seeds | Status |",
            "|---|" + "---|" * (len(table_names) + 2),
        ]
        for row in rows:
            values = [fmt(row[name]) for name in table_names]
            lines.append("| " + " | ".join([row["label"], *values, ",".join(map(str, row["seeds"])) or "N/A", row.get("status", "valid")]) + " |")
        lines += ["", "### Paired deltas", ""]
        for delta in summary["paired_deltas"][dataset]:
            if "status" in delta:
                lines.append("- {}: {}".format(delta["delta"], delta["status"]))
            else:
                values = ", ".join("{}={}".format(name, fmt(delta[name])) for name in metric_names)
                lines.append("- {} (seeds {}): {}".format(delta["delta"], delta["paired_seeds"], values))
        lines.append("")
    if summary["config_hash_errors"]:
        lines += ["## Configuration mismatches", "", "```json", json.dumps(summary["config_hash_errors"], indent=2), "```", ""]
    lines += ["## Rejected or incomplete runs", "", "```json", json.dumps(rejected, indent=2), "```", ""]
    if not allow_v5:
        lines.append("V5 is intentionally excluded from formal attribution until its causal audit is accepted.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vet_internal_ablation_summary"))
    parser.add_argument("--include-smoke", action="store_true")
    parser.add_argument("--allow-v5", action="store_true")
    args = parser.parse_args()
    runs, rejected = collect_runs(args.root, args.include_smoke)
    summary = summarize(runs, args.allow_v5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {**summary, "rejected_runs": rejected, "allow_v5": args.allow_v5}
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown_report(summary, rejected, args.allow_v5), encoding="utf-8")
    with (args.output_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["dataset", "mode", "seed", "run_kind", "run_dir"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run.get(key) for key in fieldnames})
    print(json.dumps({"valid_runs": len(runs), "rejected_runs": len(rejected), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
