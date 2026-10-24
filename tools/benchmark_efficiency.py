#!/usr/bin/env python3
"""Reproducible parameter, FLOPs, latency, memory, and module benchmark.

Formal latency/memory runs are intentionally checkpoint- and dataset-gated.
Architecture-only parameter counting is available without a trained checkpoint.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.benchmark_utils import (
    BenchmarkError,
    CheckpointError,
    CudaModuleProfiler,
    ManifestError,
    active_component_state,
    atomic_write_json,
    build_fixed_loader,
    build_model_from_variant,
    build_official_dataset,
    build_repository_args,
    cache_preprocessed_batches,
    collect_environment,
    count_parameters,
    dynamic_workload_rows,
    has_unresolved_environment,
    load_and_validate_manifest,
    load_checkpoint_strict,
    output_tensor_snapshot,
    compare_output_snapshots,
    inference_context,
    precision_context,
    prepare_official_batch,
    profiler_modules,
    read_sample_ids,
    require_cuda,
    resolve_variants,
    sample_identifier,
    sanitize_path,
    select_stratified_sample_ids,
    sha256_file,
    sha256_text,
    summarize_numbers,
    validate_component_state,
    validate_model_mode,
    validate_outputs,
    verify_call_semantics,
    write_csv,
    write_sample_ids,
)
from tools.profile_flops import profile_flops_on_real_batches


ALL_METRICS = ("params", "flops", "latency", "memory", "module")


def _run_config_hash(config: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in config.items() if key != "run_config_sha256"}
    return sha256_text(json.dumps(canonical, sort_keys=True))


def parse_args(argv: Sequence[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the official EvidenceHOI STIP implementation on a fixed real-image subset."
    )
    parser.add_argument("--variant-manifest", default="configs/efficiency_variants.yaml")
    selection = parser.add_mutually_exclusive_group(required=False)
    selection.add_argument("--variant")
    selection.add_argument("--all-enabled", action="store_true")
    parser.add_argument("--dataset-file", default="hico-det", choices=["hico-det", "vcoco"])
    parser.add_argument("--data-root", required=False)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--checkpoint", help="Single-variant override; otherwise use the manifest/environment.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--precision", default="fp32", choices=["fp32", "amp"])
    parser.add_argument("--warmup-iters", type=int, default=50)
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--flops-num-images", type=int, default=50)
    parser.add_argument("--flops-backend", default="auto", choices=["auto", "fvcore", "torch_profiler"],
                        help="Use auto fallback, strict fvcore, or runtime torch.profiler FLOPs.")
    parser.add_argument("--repeats-per-image", type=int, default=3)
    parser.add_argument("--metrics", default=",".join(ALL_METRICS))
    parser.add_argument("--output-dir", default="outputs/efficiency")
    parser.add_argument("--sample-ids", help="Reuse this sample_ids.txt instead of output-dir/sample_ids.txt")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--architecture-only", action="store_true")
    parser.add_argument("--accept-official-forward-scope", action="store_true",
                        help="Acknowledge that stip.py performs PIL-to-BLIP conversion inside forward.")
    parser.add_argument("--allow-mixed-environments", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--_worker", metavar="VARIANT", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.metrics = tuple(item.strip() for item in args.metrics.split(",") if item.strip())
    invalid = sorted(set(args.metrics) - set(ALL_METRICS))
    if invalid:
        parser.error(f"unknown metrics: {invalid}")
    if not args.metrics:
        parser.error("--metrics must not be empty")
    if args.architecture_only and set(args.metrics) != {"params"}:
        parser.error("--architecture-only permits only --metrics params")
    if args.checkpoint and (args.all_enabled or (not args.variant and not args._worker)):
        parser.error("--checkpoint can only override a single --variant")
    if args.batch_size != 1 and any(metric in args.metrics for metric in ("flops", "latency", "memory", "module")):
        parser.error("The paper protocol requires --batch-size 1 for dynamic per-image metrics")
    if args.warmup_iters < 0 or args.num_images <= 0 or args.repeats_per_image <= 0:
        parser.error("warmup >= 0, num-images > 0, and repeats > 0 are required")
    if not args._worker and not args.variant and not args.all_enabled:
        parser.error("select --variant or --all-enabled")
    return args


def _protocol_dict(args: argparse.Namespace, variant_name: str) -> Dict[str, Any]:
    return {
        "variant": variant_name,
        "dataset_file": args.dataset_file,
        "split": args.split,
        "device": args.device,
        "batch_size": args.batch_size,
        "precision": args.precision,
        "warmup_iters": args.warmup_iters,
        "num_images": args.num_images,
        "flops_num_images": args.flops_num_images,
        "flops_backend": args.flops_backend,
        "repeats_per_image": args.repeats_per_image,
        "metrics": list(args.metrics),
        "seed": args.seed,
        "num_workers": args.num_workers,
        "timing_scope": "official_forward",
        "architecture_only": args.architecture_only,
    }


def _safe_run_config(args, variant_name, variant, checkpoint, manifest) -> Dict[str, Any]:
    protocol = _protocol_dict(args, variant_name)
    protocol.update({
        "data_root": sanitize_path(args.data_root, "DATA_ROOT"),
        "checkpoint": sanitize_path(checkpoint, "CHECKPOINT"),
        "manifest_sha256": manifest["_manifest_sha256"],
        "paper_components": variant["paper_components"],
        "expected": variant["expected"],
        "backbone": variant["backbone"],
        "visual_encoder": variant["visual_encoder"],
        "model_file": variant["model_file"],
        "paper_configuration_verified": variant["paper_configuration_verified"],
        "config_args": variant["config_args"],
        "resources": {key: sanitize_path(str(value), "RESOURCE") for key, value in variant["resources"].items()},
        "resource_sha256": {
            key: sha256_file(value) if Path(value).is_file() else None
            for key, value in variant["resources"].items()
        },
        "code_sha256": {
            relative: sha256_file(REPO_ROOT / relative)
            for relative in (
                "tools/benchmark_efficiency.py",
                "tools/benchmark_utils.py",
                "tools/profile_flops.py",
                "tools/generate_efficiency_table.py",
                "src/models/stip.py",
                "src/models/visual_scene_graph_oov8.py",
                "src/models/candidate_evidence_topology.py",
                "src/models/text_prototype_slots.py",
                "src/models/synchronized_evidence_topology_matcher.py",
            )
        },
    })
    protocol["run_config_sha256"] = _run_config_hash(protocol)
    return protocol


def _checkpoint_for(args, variant) -> str:
    if args.architecture_only:
        return ""
    checkpoint = args.checkpoint or variant.get("checkpoint", "")
    if checkpoint is None:
        return ""
    return str(checkpoint)


def _sample_ids_path(args) -> Path:
    return Path(args.sample_ids).resolve() if args.sample_ids else (Path(args.output_dir).resolve() / "sample_ids.txt")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _warm_up(model, cached_batches, device, precision, iterations) -> None:
    if iterations <= 0:
        return
    with inference_context():
        for iteration in range(iterations):
            prepared = prepare_official_batch(cached_batches[iteration % len(cached_batches)], device)
            with precision_context(precision, device):
                outputs = model(*prepared)
            del outputs, prepared
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_latency(model, cached_batches, device, precision, repeats, variant_name):
    require_cuda(device, "latency")
    rows: List[Dict[str, Any]] = []
    workload: List[Dict[str, Any]] = []
    with inference_context():
        for batch in cached_batches:
            prepared = prepare_official_batch(batch, device)
            image_ids = [sample_identifier(target) for target in prepared[2]]
            for repeat in range(repeats):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                start.record()
                with precision_context(precision, device):
                    outputs = model(*prepared)
                end.record()
                torch.cuda.synchronize(device)
                batch_ms = float(start.elapsed_time(end))
                validate_outputs(outputs)
                rows.append({
                    "variant": variant_name,
                    "image_id": image_ids[0],
                    "repeat": repeat,
                    "latency_ms": batch_ms / len(image_ids),
                    "batch_size": len(image_ids),
                    "precision": precision,
                    "timing_scope": "official_forward",
                })
                if repeat == 0:
                    workload.extend(dynamic_workload_rows(outputs, variant_name, image_ids))
                del outputs
            del prepared
    values = [row["latency_ms"] for row in rows]
    stats = summarize_numbers(values)
    stats.update({
        "fps": 1000.0 / stats["mean"] if stats["mean"] else None,
        "images": len(cached_batches),
        "repeats_per_image": repeats,
        "timing_scope": "official_forward",
        "excludes": ["model construction", "checkpoint loading", "disk I/O", "JPEG decoding", "DataLoader wait"],
        "includes": ["DETR", "BLIP", "VET/KPT/SETM as enabled", "HOI decoder", "PIL-to-BLIP conversion inside stip.py forward"],
        "timing_api": "torch.cuda.Event + synchronize",
    })
    return rows, stats, workload


def benchmark_memory(model, cached_batches, device, precision, variant_name):
    require_cuda(device, "memory")
    rows: List[Dict[str, Any]] = []
    with inference_context():
        for batch in cached_batches:
            prepared = prepare_official_batch(batch, device)
            image_ids = [sample_identifier(target) for target in prepared[2]]
            torch.cuda.synchronize(device)
            before = int(torch.cuda.memory_allocated(device))
            torch.cuda.reset_peak_memory_stats(device)
            with precision_context(precision, device):
                outputs = model(*prepared)
            torch.cuda.synchronize(device)
            peak_allocated = int(torch.cuda.max_memory_allocated(device))
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            after = int(torch.cuda.memory_allocated(device))
            validate_outputs(outputs)
            rows.append({
                "variant": variant_name,
                "image_id": image_ids[0],
                "before_allocated_bytes": before,
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "after_allocated_bytes": after,
                "incremental_activation_bytes": max(0, peak_allocated - before),
                "precision": precision,
            })
            del outputs, prepared
            gc.collect()
            torch.cuda.empty_cache()
    peak_stats = summarize_numbers([row["peak_allocated_bytes"] for row in rows])
    incremental_stats = summarize_numbers([row["incremental_activation_bytes"] for row in rows])
    reserved_stats = summarize_numbers([row["peak_reserved_bytes"] for row in rows])
    return rows, {
        "peak_allocated_bytes": peak_stats,
        "incremental_activation_bytes": incremental_stats,
        "peak_reserved_bytes_diagnostic": reserved_stats,
        "main_metric": "torch.cuda.max_memory_allocated",
        "inference_only": True,
    }


def benchmark_modules(model, cached_batches, device, precision, variant_name, variant):
    require_cuda(device, "module profiling")
    aggregate_ms: Dict[str, float] = {}
    aggregate_calls: Dict[str, int] = {}
    rows: List[Dict[str, Any]] = []
    workload: List[Dict[str, Any]] = []
    groups = profiler_modules(model)
    with inference_context():
        for batch in cached_batches:
            prepared = prepare_official_batch(batch, device)
            image_ids = [sample_identifier(target) for target in prepared[2]]
            with CudaModuleProfiler(groups) as profiler:
                with precision_context(precision, device):
                    outputs = model(*prepared)
            elapsed, calls = profiler.results()
            validate_outputs(outputs)
            workload.extend(dynamic_workload_rows(outputs, variant_name, image_ids))
            for group, value in elapsed.items():
                aggregate_ms[group] = aggregate_ms.get(group, 0.0) + value
                rows.append({
                    "variant": variant_name,
                    "image_id": image_ids[0],
                    "module_group": group,
                    "cuda_time_ms": value,
                    "inclusive_or_overlapping": True,
                })
            for group, value in calls.items():
                aggregate_calls[group] = aggregate_calls.get(group, 0) + value
            del outputs, prepared
    image_count = len(cached_batches)
    summary = {
        group: {
            "total_cuda_time_ms": value,
            "mean_cuda_time_ms_per_image": value / image_count,
            "calls": aggregate_calls.get(group, 0),
        }
        for group, value in sorted(aggregate_ms.items())
    }
    call_report = {
        "counts": aggregate_calls,
        "image_count": image_count,
        "expected_L": variant["expected"]["L"],
        "retrieval_once_required": bool(variant["paper_components"]["KPT"]),
        "notes": "CUDA-event intervals are explanatory and may overlap; do not sum them as total latency.",
    }
    reasons = verify_call_semantics(aggregate_calls, variant, image_count)
    call_report["semantic_failures"] = reasons
    return rows, summary, call_report, workload


def _existing_resume_ok(summary_path: Path, run_config: Mapping[str, Any]) -> bool:
    if not summary_path.is_file():
        return False
    try:
        old = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return old.get("run_config_sha256") == run_config.get("run_config_sha256")


def run_worker(args: argparse.Namespace, variant_name: str) -> int:
    manifest = load_and_validate_manifest(args.variant_manifest)
    if variant_name not in manifest["variants"]:
        raise ManifestError(f"Unknown worker variant: {variant_name}")
    variant = manifest["variants"][variant_name]
    checkpoint = _checkpoint_for(args, variant)
    if not args.data_root:
        raise BenchmarkError("--data-root is required even for architecture-only mode to obtain dataset class metadata")
    if has_unresolved_environment(variant.get("resources", {})):
        raise BenchmarkError(f"Unresolved resource environment variable(s) for {variant_name}")
    if not args.architecture_only:
        if not checkpoint or has_unresolved_environment(checkpoint):
            raise BenchmarkError(f"A trained checkpoint is required for {variant_name}")
        if not variant["paper_configuration_verified"]:
            raise BenchmarkError(
                f"{variant_name} is not marked paper_configuration_verified; confirm its Table 4 flags first"
            )
    run_dir = Path(args.output_dir).resolve() / variant_name
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = _safe_run_config(args, variant_name, variant, checkpoint, manifest)
    if checkpoint and Path(checkpoint).is_file():
        run_config["checkpoint_sha256"] = sha256_file(checkpoint)
        run_config["run_config_sha256"] = _run_config_hash(run_config)
    sample_path = _sample_ids_path(args)
    if sample_path.is_file():
        run_config["sample_ids_sha256"] = sha256_file(sample_path)
        run_config["run_config_sha256"] = _run_config_hash(run_config)
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not (args.force or args.resume):
        raise BenchmarkError(f"Refusing to overwrite existing {summary_path}; use --resume or --force")
    if args.resume and _existing_resume_ok(summary_path, run_config):
        print(f"[resume] {variant_name}: matching completed run already exists")
        return 0
    if args.resume and summary_path.exists():
        raise BenchmarkError("Existing run does not match current manifest/protocol hash; use --force")

    _seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        require_cuda(device, "GPU benchmark")
        torch.cuda.set_device(device)
    environment = collect_environment(args.device, args.precision, Path(__file__))
    atomic_write_json(run_dir / "environment.json", environment)
    atomic_write_json(run_dir / "run_config.json", run_config)

    repository_args = build_repository_args(
        variant, args.dataset_file, args.data_root, args.device, args.batch_size, args.num_workers
    )
    repository_args.seed = args.seed
    dataset = build_official_dataset(repository_args, args.split)
    if sample_path.exists():
        sample_ids = read_sample_ids(sample_path)
        if len(sample_ids) != args.num_images:
            raise BenchmarkError(
                f"sample_ids count {len(sample_ids)} != --num-images {args.num_images}; protocol changes require a new output dir"
            )
    else:
        sample_ids = select_stratified_sample_ids(dataset, args.num_images, args.seed)
        write_sample_ids(sample_path, sample_ids)
    sample_hash = sha256_file(sample_path)
    run_config["sample_ids_sha256"] = sample_hash
    run_config["run_config_sha256"] = _run_config_hash(run_config)
    atomic_write_json(run_dir / "run_config.json", run_config)

    print(f"[{variant_name}] building official model from src/models/stip.py")
    model, _, _ = build_model_from_variant(repository_args, variant["model_file"])
    checkpoint_report = None
    if checkpoint:
        try:
            checkpoint_report = load_checkpoint_strict(
                model, checkpoint, checkpoint_sha256=run_config.get("checkpoint_sha256")
            )
        except CheckpointError as exc:
            if exc.report is not None:
                atomic_write_json(run_dir / "checkpoint_load_report.json", exc.report)
            raise
        atomic_write_json(run_dir / "checkpoint_load_report.json", checkpoint_report)
    model.to(device)
    model.eval()
    validate_model_mode(model)
    component_state = validate_component_state(model, variant)
    atomic_write_json(run_dir / "component_state.json", component_state)

    result: Dict[str, Any] = {
        "variant": variant_name,
        "display_name": variant["display_name"],
        "paper_components": variant["paper_components"],
        "backbone": variant["backbone"],
        "visual_encoder": variant["visual_encoder"],
        "run_config_sha256": run_config["run_config_sha256"],
        "manifest_sha256": manifest["_manifest_sha256"],
        "sample_ids_sha256": sample_hash,
        "checkpoint_sha256": checkpoint_report["sha256"] if checkpoint_report else None,
        "metrics_requested": list(args.metrics),
        "metrics": {},
        "paper_ready": False,
        "paper_ready_failure_reasons": [],
        "paper_ready_warnings": [],
    }

    if "params" in args.metrics:
        parameter_summary, module_rows = count_parameters(model)
        atomic_write_json(run_dir / "parameter_summary.json", parameter_summary)
        write_csv(run_dir / "module_parameters.csv", module_rows)
        result["metrics"]["params"] = parameter_summary

    if args.architecture_only:
        result["paper_ready_failure_reasons"] = [
            "architecture-only run has no trained checkpoint or measured runtime metrics"
        ]
        atomic_write_json(summary_path, result)
        print(f"[{variant_name}] architecture-only parameter count complete")
        return 0

    loader = build_fixed_loader(dataset, sample_ids, args.batch_size, args.num_workers)
    cached_batches, subset_stats = cache_preprocessed_batches(loader)
    atomic_write_json(Path(args.output_dir).resolve() / "subset_statistics.json", subset_stats)

    # Verify that the adapter is numerically identical to the evaluator's direct
    # (samples.to, target tensor move, model(samples, images, targets)) path.
    first_batch = cached_batches[0]
    _seed_all(args.seed)
    direct_samples = first_batch[0].to(device)
    direct_targets = [
        {key: (value.to(device) if hasattr(value, "to") else value) for key, value in target.items()}
        for target in first_batch[1]
    ]
    direct_args = (direct_samples, first_batch[2], direct_targets)
    with inference_context(), precision_context(args.precision, device):
        direct_outputs = model(*direct_args)
    validate_outputs(direct_outputs)
    direct_snapshot = output_tensor_snapshot(direct_outputs)

    _seed_all(args.seed)
    prepared = prepare_official_batch(first_batch, device)
    with inference_context(), precision_context(args.precision, device):
        sanity_outputs = model(*prepared)
    validate_outputs(sanity_outputs)
    adapter_snapshot = output_tensor_snapshot(sanity_outputs)
    compare_output_snapshots(direct_snapshot, adapter_snapshot)

    profiling_invariant = None
    if "module" in args.metrics:
        require_cuda(device, "module profiling sanity check")
        _seed_all(args.seed)
        with CudaModuleProfiler(profiler_modules(model)) as sanity_profiler:
            with inference_context(), precision_context(args.precision, device):
                profiled_outputs = model(*prepared)
        sanity_profiler.results()
        validate_outputs(profiled_outputs)
        compare_output_snapshots(adapter_snapshot, output_tensor_snapshot(profiled_outputs))
        profiling_invariant = True
        del profiled_outputs

    atomic_write_json(run_dir / "sanity_report.json", {
        "official_forward_ok": True,
        "adapter_matches_official_evaluator_call": True,
        "profiling_preserves_outputs": profiling_invariant,
        "finite_outputs": True,
        "model_eval": True,
        "component_state": component_state,
    })
    del direct_outputs, sanity_outputs, direct_snapshot, adapter_snapshot, direct_args, direct_samples, direct_targets, prepared
    _seed_all(args.seed)
    _warm_up(model, cached_batches, device, args.precision, args.warmup_iters)

    workload_rows: List[Dict[str, Any]] = []
    if "flops" in args.metrics:
        rows, flops_summary, diagnostics = profile_flops_on_real_batches(
            model, cached_batches, device, args.precision,
            min(args.flops_num_images, args.num_images),
            backend_preference=args.flops_backend,
        )
        for row in rows:
            row["variant"] = variant_name
        write_csv(run_dir / "flops_per_image.csv", rows)
        atomic_write_json(run_dir / "flops_summary.json", flops_summary)
        (run_dir / "unsupported_ops.txt").write_text("\n".join(diagnostics) + ("\n" if diagnostics else ""), encoding="utf-8")
        result["metrics"]["flops"] = flops_summary

    if "latency" in args.metrics:
        rows, latency_summary, workload = benchmark_latency(
            model, cached_batches, device, args.precision, args.repeats_per_image, variant_name
        )
        write_csv(run_dir / "raw_latency.csv", rows)
        atomic_write_json(run_dir / "latency_summary.json", latency_summary)
        result["metrics"]["latency"] = latency_summary
        workload_rows.extend(workload)

    if "memory" in args.metrics:
        rows, memory_summary = benchmark_memory(model, cached_batches, device, args.precision, variant_name)
        write_csv(run_dir / "raw_memory.csv", rows)
        atomic_write_json(run_dir / "memory_summary.json", memory_summary)
        result["metrics"]["memory"] = memory_summary

    if "module" in args.metrics:
        rows, module_summary, call_report, workload = benchmark_modules(
            model, cached_batches, device, args.precision, variant_name, variant
        )
        write_csv(run_dir / "module_profile.csv", rows)
        atomic_write_json(run_dir / "module_profile_summary.json", module_summary)
        atomic_write_json(run_dir / "module_call_counts.json", call_report)
        result["metrics"]["module"] = module_summary
        result["module_call_semantic_failures"] = call_report["semantic_failures"]
        workload_rows = workload  # One row/image from the explanatory profiler is sufficient.

    if workload_rows:
        write_csv(run_dir / "dynamic_workload.csv", workload_rows)

    reasons = result["paper_ready_failure_reasons"]
    missing_metrics = sorted(set(ALL_METRICS) - set(result["metrics"]))
    if missing_metrics:
        reasons.append(f"missing required metrics: {missing_metrics}")
    if not args.accept_official_forward_scope:
        reasons.append(
            "timing scope not acknowledged: stip.py performs PIL-to-BLIP conversion inside official forward"
        )
    flops = result["metrics"].get("flops")
    if flops and not flops.get("paper_value_available", False):
        result["paper_ready_warnings"].append(
            "FLOPs is N/A because profiler coverage is incomplete/unknown; see unsupported_ops.txt"
        )
    reasons.extend(result.get("module_call_semantic_failures", []))
    if not variant["paper_configuration_verified"]:
        reasons.append("variant flags are not verified against final Table 4(a)")
    result["paper_ready"] = not reasons
    atomic_write_json(summary_path, result)
    print(f"[{variant_name}] complete; paper_ready={result['paper_ready']}")
    return 0


def _worker_command(args: argparse.Namespace, variant_name: str) -> List[str]:
    command = [
        sys.executable, str(Path(__file__).resolve()),
        "--_worker", variant_name,
        "--variant-manifest", str(Path(args.variant_manifest).resolve()),
        "--dataset-file", args.dataset_file,
        "--data-root", args.data_root or "",
        "--split", args.split,
        "--device", args.device,
        "--batch-size", str(args.batch_size),
        "--precision", args.precision,
        "--warmup-iters", str(args.warmup_iters),
        "--num-images", str(args.num_images),
        "--flops-num-images", str(args.flops_num_images),
        "--flops-backend", args.flops_backend,
        "--repeats-per-image", str(args.repeats_per_image),
        "--metrics", ",".join(args.metrics),
        "--output-dir", str(Path(args.output_dir).resolve()),
        "--num-workers", str(args.num_workers),
        "--seed", str(args.seed),
    ]
    if args.sample_ids:
        command.extend(["--sample-ids", str(Path(args.sample_ids).resolve())])
    if args.checkpoint:
        command.extend(["--checkpoint", args.checkpoint])
    for flag, enabled in (
        ("--architecture-only", args.architecture_only),
        ("--accept-official-forward-scope", args.accept_official_forward_scope),
        ("--allow-mixed-environments", args.allow_mixed_environments),
        ("--resume", args.resume),
        ("--force", args.force),
    ):
        if enabled:
            command.append(flag)
    return command


def _redact_log_line(line: str, args: argparse.Namespace) -> str:
    redacted = line
    for value, token in ((args.data_root, "<DATA_ROOT>"), (args.checkpoint, "<CHECKPOINT>")):
        if value:
            redacted = redacted.replace(str(value), token)
    # The repository builder currently contains a legacy absolute BLIP path.
    # Preserve the filename for diagnosis without archiving a user/home prefix.
    redacted = re.sub(
        r"(?<![A-Za-z0-9_])/(?:home|root|wzj|share)/[^\s'\"]+",
        lambda match: "<PRIVATE_PATH>/" + Path(match.group(0).rstrip(",:;)")).name,
        redacted,
    )
    return redacted

def _run_clean_worker(args: argparse.Namespace, variant_name: str) -> int:
    run_dir = Path(args.output_dir).resolve() / variant_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    command = _worker_command(args, variant_name)
    print(f"[orchestrator] starting clean subprocess for {variant_name}")
    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(_redact_log_line(line, args))
            log.flush()
        return process.wait()


def _write_run_matrix_artifacts(
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    variants: Sequence[str],
    failures: Sequence[Tuple[str, int]],
) -> None:
    output_dir = Path(args.output_dir).resolve()
    failed_names = {name for name, _ in failures}
    records = []
    lines = [
        "# Efficiency benchmark run matrix",
        "",
        "| Variant | Process | Paper ready | Failure reasons | Checkpoint hash | Sample hash |",
        "|---|---|:---:|---|---|---|",
    ]
    for name in variants:
        summary_path = output_dir / name / "summary.json"
        summary = None
        if summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary = None
        process_status = "failed" if name in failed_names else "completed"
        ready = bool(summary and summary.get("paper_ready", False))
        reasons = summary.get("paper_ready_failure_reasons", []) if summary else ["summary.json missing"]
        record = {
            "variant": name,
            "process_status": process_status,
            "paper_ready": ready,
            "failure_reasons": reasons,
            "checkpoint_sha256": summary.get("checkpoint_sha256") if summary else None,
            "sample_ids_sha256": summary.get("sample_ids_sha256") if summary else None,
            "run_config_sha256": summary.get("run_config_sha256") if summary else None,
        }
        records.append(record)
        lines.append(
            f"| {name} | {process_status} | {'yes' if ready else 'no'} | "
            f"{'; '.join(str(item) for item in reasons) if reasons else '--'} | "
            f"{record['checkpoint_sha256'] or '--'} | {record['sample_ids_sha256'] or '--'} |"
        )
    atomic_write_json(output_dir / "benchmark_manifest.json", {
        "variant_manifest_sha256": manifest["_manifest_sha256"],
        "baseline_variant": manifest["baseline_variant"],
        "selected_variants": list(variants),
        "runs": records,
    })
    (output_dir / "run_matrix_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main(argv: Sequence[str] = None) -> int:
    args = parse_args(argv)
    try:
        if args._worker:
            return run_worker(args, args._worker)
        manifest = load_and_validate_manifest(args.variant_manifest)
        variants = resolve_variants(manifest, args.variant, args.all_enabled)
        failures = []
        for variant_name in variants:
            return_code = _run_clean_worker(args, variant_name)
            if return_code != 0:
                failures.append((variant_name, return_code))
        _write_run_matrix_artifacts(args, manifest, variants, failures)
        if failures:
            print(f"Benchmark failures: {failures}", file=sys.stderr)
            return 1
        print(f"Completed {len(variants)} clean variant subprocess(es).")
        return 0
    except (BenchmarkError, FileNotFoundError, AssertionError, ValueError) as exc:
        print(f"[benchmark error] {exc}", file=sys.stderr)
        return 2
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower():
            raise
        print(f"[benchmark OOM] {exc}; do not skip images—change the protocol and rerun all variants.", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
