#!/usr/bin/env python3
"""Aggregate paper-ready EvidenceHOI efficiency runs into CSV/JSON/Markdown/LaTeX."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.benchmark_utils import (
    BenchmarkError,
    atomic_write_json,
    compare_environments,
    load_and_validate_manifest,
    write_csv,
)


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper tables from immutable benchmark summaries.")
    parser.add_argument("--input-dir", default="outputs/efficiency")
    parser.add_argument("--output-dir", help="Defaults to --input-dir")
    parser.add_argument("--variant-manifest", default="configs/efficiency_variants.yaml")
    parser.add_argument("--variants", help="Comma-separated subset; default is enabled R50+ViT-B variants")
    parser.add_argument(
        "--allow-without-baseline",
        action="store_true",
        help="Allow a partial table without the manifest baseline; relative fields are reported as N/A",
    )
    parser.add_argument("--allow-mixed-environments", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise BenchmarkError(f"Required benchmark artifact missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise BenchmarkError(f"Expected JSON object in {path}")
    return payload


def _selected_variants(args, manifest) -> List[str]:
    if args.variants:
        names = [item.strip() for item in args.variants.split(",") if item.strip()]
    else:
        names = [
            name for name, variant in manifest["variants"].items()
            if variant["enabled"] and variant["backbone"] == "resnet50" and variant["visual_encoder"] == "vit_base"
        ]
    unknown = [name for name in names if name not in manifest["variants"]]
    if unknown:
        raise BenchmarkError(f"Unknown variants: {unknown}")
    if manifest["baseline_variant"] not in names and not args.allow_without_baseline:
        raise BenchmarkError("Selected runs must include the manifest baseline_variant")
    return names


def _extract_row(name: str, variant: Mapping[str, Any], summary: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = summary.get("metrics", {})
    params = metrics.get("params", {})
    latency = metrics.get("latency", {})
    memory = metrics.get("memory", {})
    flops = metrics.get("flops", {})
    paper_flops = flops.get("paper_flops") or {}
    required_values = {
        "total_params": params.get("total_params"),
        "trainable_params": params.get("trainable_params"),
        "frozen_params": params.get("frozen_params"),
        "latency_mean": latency.get("mean"),
        "latency_fps": latency.get("fps"),
        "peak_memory_max": memory.get("peak_allocated_bytes", {}).get("max"),
        "incremental_activation_mean": memory.get("incremental_activation_bytes", {}).get("mean"),
    }
    missing = sorted(key for key, value in required_values.items() if value is None)
    if missing:
        raise BenchmarkError(f"Paper-ready summary {name} is missing critical values: {missing}")
    if not summary.get("checkpoint_sha256"):
        raise BenchmarkError(f"Paper-ready summary {name} has no checkpoint hash")
    components = variant["paper_components"]
    return {
        "variant": name,
        "method": variant["display_name"],
        "backbone": variant["backbone"],
        "visual_encoder": variant["visual_encoder"],
        "VET": components["VET"],
        "KPT": components["KPT"],
        "SETM": components["SETM"],
        "total_params_m": params["total_params"] / 1e6,
        "trainable_params_m": params["trainable_params"] / 1e6,
        "frozen_params_m": params["frozen_params"] / 1e6,
        "flops_g": (paper_flops.get("mean") / 1e9) if paper_flops.get("mean") is not None else None,
        "peak_memory_gb": (
            memory.get("peak_allocated_bytes", {}).get("max") / (1024 ** 3)
            if memory.get("peak_allocated_bytes", {}).get("max") is not None else None
        ),
        "incremental_activation_gb": (
            memory.get("incremental_activation_bytes", {}).get("mean") / (1024 ** 3)
            if memory.get("incremental_activation_bytes", {}).get("mean") is not None else None
        ),
        "latency_ms": latency.get("mean"),
        "latency_median_ms": latency.get("median"),
        "latency_p90_ms": latency.get("p90"),
        "fps": latency.get("fps"),
        "checkpoint_sha256": summary.get("checkpoint_sha256"),
        "sample_ids_sha256": summary.get("sample_ids_sha256"),
        "manifest_sha256": summary.get("manifest_sha256"),
        "paper_ready": summary.get("paper_ready", False),
    }


def _fmt(value: Any, digits: int) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _configuration_label(row: Mapping[str, Any]) -> str:
    backbone = {"resnet50": "R50", "resnet101": "R101"}.get(
        str(row.get("backbone")), str(row.get("backbone"))
    )
    visual = {"vit_base": "ViT-B", "vit_large": "ViT-L"}.get(
        str(row.get("visual_encoder")), str(row.get("visual_encoder"))
    )
    return f"{backbone}+{visual}"


def _markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Method | VET | KPT | SETM | Total Params (M) | Trainable Params (M) | Extra Params (M) | FLOPs (G) | Peak Mem. (GiB) | Latency (ms/img) | FPS |",
        "|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        mark = lambda key: "✓" if row[key] else "–"
        lines.append(
            f"| {row['method']} | {mark('VET')} | {mark('KPT')} | {mark('SETM')} | "
            f"{_fmt(row['total_params_m'], 2)} | {_fmt(row['trainable_params_m'], 2)} | "
            f"{_fmt(row['extra_params_m'], 2)} | {_fmt(row['flops_g'], 2)} | "
            f"{_fmt(row['peak_memory_gb'], 2)} | {_fmt(row['latency_ms'], 1)} | {_fmt(row['fps'], 2)} |"
        )
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")


def _latex(rows: Sequence[Mapping[str, Any]], environment: Mapping[str, Any], run_config: Mapping[str, Any]) -> str:
    gpu = environment.get("gpu", {}).get("name", "unknown GPU")
    precision = run_config.get("precision", "unknown")
    count = run_config.get("num_images", "unknown")
    batch = run_config.get("batch_size", "unknown")
    configurations = list(dict.fromkeys(_configuration_label(row) for row in rows))
    if len(configurations) == 1:
        configuration_scope = f"the {configurations[0]} configuration"
    else:
        configuration_scope = "the " + " and ".join(configurations) + " configurations"
    caption = (
        f"\\textbf{{Computational efficiency on {configuration_scope}.}} "
        f"All reported rows use the same {count} HICO-DET test images, batch size {batch}, "
        f"and {str(precision).upper()} inference on a single {_latex_escape(str(gpu))}. "
        "Latency measures the official model forward after DataLoader preprocessing and GPU transfer; "
        "the current official forward includes its internal PIL-to-BLIP conversion. "
        "FLOPs are averaged over real images and are shown only when profiler coverage is complete."
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\label{tab:supp_efficiency}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.2pt}",
        "\\begin{tabular}{lccc|rrrrrrr}",
        "\\toprule",
        "Method & VET & KPT & SETM & Total & Train. & Extra & FLOPs & Mem. & Latency & FPS \\\\",
        "& & & & \\multicolumn{3}{c}{Params (M)} & (G) & (GiB) & (ms/img) & (img/s) \\\\",
        "\\midrule",
    ]
    for row in rows:
        mark = lambda key: "\\checkmark" if row[key] else "--"
        lines.append(
            f"{_latex_escape(row['method'])} & {mark('VET')} & {mark('KPT')} & {mark('SETM')} & "
            f"{_fmt(row['total_params_m'], 2)} & {_fmt(row['trainable_params_m'], 2)} & "
            f"{_fmt(row['extra_params_m'], 2)} & {_fmt(row['flops_g'], 2)} & "
            f"{_fmt(row['peak_memory_gb'], 2)} & {_fmt(row['latency_ms'], 1)} & {_fmt(row['fps'], 2)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}"])
    return "\n".join(lines) + "\n"



def _delta(current: Any, reference: Any) -> Any:
    if current is None or reference is None:
        return None
    return float(current) - float(reference)


def _module_overhead_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_name = {row["variant"]: row for row in rows}
    required = {
        "baseline_r50_vitb", "vet_r50_vitb", "kpt_r50_vitb",
        "cross_attention_r50_vitb", "evidencehoi_r50_vitb",
    }
    if not required.issubset(by_name):
        return []
    baseline = by_name["baseline_r50_vitb"]
    definitions = [
        ("VET", by_name["vet_r50_vitb"], baseline,
         "Node-relation reasoning and Top-M evidence-chain induction"),
        ("KPT", by_name["kpt_r50_vitb"], baseline,
         "Top-K knowledge prototype topology retrieval and reasoning"),
        ("SETM", by_name["evidencehoi_r50_vitb"], by_name["cross_attention_r50_vitb"],
         "Path-role correspondence and bidirectional evidence update"),
    ]
    output = []
    for component, current, reference, source in definitions:
        output.append({
            "component": component,
            "reference_variant": reference["variant"],
            "extra_params_m": _delta(current["total_params_m"], reference["total_params_m"]),
            "extra_flops_g": _delta(current["flops_g"], reference["flops_g"]),
            "incremental_peak_memory_gb": _delta(current["peak_memory_gb"], reference["peak_memory_gb"]),
            "latency_overhead_ms": _delta(current["latency_ms"], reference["latency_ms"]),
            "primary_source_of_cost": source,
        })
    return output


def _module_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "| Component | Reference | Extra Params (M) | Extra FLOPs (G) | Incremental Peak Mem. (GiB) | Latency Overhead (ms/img) | Primary Source of Cost |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['component']} | {row['reference_variant']} | {_fmt(row['extra_params_m'], 2)} | "
            f"{_fmt(row['extra_flops_g'], 2)} | {_fmt(row['incremental_peak_memory_gb'], 2)} | "
            f"{_fmt(row['latency_overhead_ms'], 1)} | {row['primary_source_of_cost']} |"
        )
    return "\n".join(lines) + "\n"


def _module_latex(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Incremental component overhead under the same efficiency protocol. VET and KPT are measured relative to Baseline; SETM is measured relative to Cross-Attention Fusion.}",
        "\\label{tab:supp_component_overhead}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Component & Extra Params (M) & Extra FLOPs (G) & Mem. (GiB) & Latency (ms/img) \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['component']} & {_fmt(row['extra_params_m'], 2)} & {_fmt(row['extra_flops_g'], 2)} & "
            f"{_fmt(row['incremental_peak_memory_gb'], 2)} & {_fmt(row['latency_overhead_ms'], 1)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines) + "\n"

def generate(args: argparse.Namespace) -> None:
    manifest = load_and_validate_manifest(args.variant_manifest)
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
    names = _selected_variants(args, manifest)
    summaries = []
    environments = []
    run_configs = []
    rows = []
    for name in names:
        run_dir = input_dir / name
        summary = _load_json(run_dir / "summary.json")
        if not summary.get("paper_ready", False):
            raise BenchmarkError(
                f"Refusing paper table: {name} is not paper_ready: {summary.get('paper_ready_failure_reasons', [])}"
            )
        summaries.append(summary)
        environments.append(_load_json(run_dir / "environment.json"))
        run_configs.append(_load_json(run_dir / "run_config.json"))
        rows.append(_extract_row(name, manifest["variants"][name], summary))

    configurations = {_configuration_label(row) for row in rows}
    if len(configurations) > 1:
        for row in rows:
            label = _configuration_label(row)
            if label not in row["method"]:
                row["method"] = f"{row['method']} ({label})"

    compare_environments(environments, args.allow_mixed_environments)
    sample_hashes = {row["sample_ids_sha256"] for row in rows}
    manifest_hashes = {row["manifest_sha256"] for row in rows}
    protocol_keys = ("dataset_file", "split", "batch_size", "precision", "num_images", "warmup_iters", "repeats_per_image")
    protocols = {tuple(config.get(key) for key in protocol_keys) for config in run_configs}
    if len(sample_hashes) != 1 or len(manifest_hashes) != 1 or len(protocols) != 1:
        raise BenchmarkError("Runs do not share sample IDs, manifest, and critical protocol settings")

    baseline_name = manifest["baseline_variant"]
    baseline = next((row for row in rows if row["variant"] == baseline_name), None)
    for row in rows:
        if baseline is None:
            row["extra_params_m"] = None
            row["latency_overhead_ms"] = None
            row["relative_fps"] = None
        else:
            row["extra_params_m"] = row["total_params_m"] - baseline["total_params_m"]
            row["latency_overhead_ms"] = (
                row["latency_ms"] - baseline["latency_ms"]
                if row["latency_ms"] is not None and baseline["latency_ms"] is not None else None
            )
            row["relative_fps"] = (
                row["fps"] / baseline["fps"] if row["fps"] is not None and baseline["fps"] else None
            )

    module_rows = _module_overhead_rows(rows)
    output_names = ["summary.csv", "summary.json", "summary.md", "summary_table.tex"]
    if module_rows:
        output_names.extend(["module_overhead.csv", "module_overhead.md", "module_overhead_table.tex"])
    targets = [output_dir / name for name in output_names]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        raise BenchmarkError(f"Refusing to overwrite {existing}; use --force")
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant", "method", "backbone", "visual_encoder", "VET", "KPT", "SETM", "total_params_m", "trainable_params_m",
        "frozen_params_m", "extra_params_m", "flops_g", "peak_memory_gb", "incremental_activation_gb",
        "latency_ms", "latency_median_ms", "latency_p90_ms", "latency_overhead_ms", "fps", "relative_fps",
        "checkpoint_sha256", "sample_ids_sha256", "manifest_sha256", "paper_ready",
    ]
    write_csv(output_dir / "summary.csv", rows, fieldnames)
    atomic_write_json(output_dir / "summary.json", {
        "baseline_variant": baseline_name if baseline is not None else None,
        "environment_signature": environments[0],
        "protocol": run_configs[0],
        "rows": rows,
    })
    (output_dir / "summary.md").write_text(_markdown(rows), encoding="utf-8")
    (output_dir / "summary_table.tex").write_text(_latex(rows, environments[0], run_configs[0]), encoding="utf-8")
    if module_rows:
        write_csv(output_dir / "module_overhead.csv", module_rows)
        (output_dir / "module_overhead.md").write_text(_module_markdown(module_rows), encoding="utf-8")
        (output_dir / "module_overhead_table.tex").write_text(_module_latex(module_rows), encoding="utf-8")


def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    try:
        generate(args)
        print(f"Wrote efficiency tables to {Path(args.output_dir or args.input_dir).resolve()}")
        return 0
    except BenchmarkError as exc:
        print(f"[aggregation error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
