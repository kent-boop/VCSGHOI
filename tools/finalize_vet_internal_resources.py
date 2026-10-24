"""Finalize resource metadata for a completed VET ablation run."""

import argparse
import json
import re
from pathlib import Path

import torch


TIME_PATTERN = re.compile(r"HOI Recognition Time \(avg\)\s*:\s*([0-9.]+)\s*ms")


def checkpoint_parameter_count(path: Path):
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("model", checkpoint)
    return int(sum(value.numel() for value in state.values() if torch.is_tensor(value)))


def inference_time_ms(log_path: Path):
    if not log_path.exists():
        return None
    values = [float(value) for value in TIME_PATTERN.findall(log_path.read_text(encoding="utf-8", errors="replace"))]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--total-params", type=int, help="Exact nn.Parameter count from a documented model construction")
    parser.add_argument("--flops", type=float, help="Optional value produced by an external, documented profiler")
    args = parser.parse_args()
    resource_path = args.run_dir / "resource_metrics.json"
    resource = json.loads(resource_path.read_text(encoding="utf-8")) if resource_path.exists() else {}
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = args.run_dir / "best.pth"
        if not checkpoint.exists():
            checkpoint = args.run_dir / "checkpoint.pth"
    resource.update({
        "model_state_numel": checkpoint_parameter_count(checkpoint),
        "model_state_numel_source": str(checkpoint) if checkpoint.exists() else "unavailable",
        "inference_time_ms": inference_time_ms(args.run_dir / "launcher.log"),
        "inference_time_source": "evaluator HOI Recognition Time (avg)",
    })
    if args.total_params is not None:
        resource["total_params"] = int(args.total_params)
        resource["total_params_status"] = "user-supplied exact nn.Parameter count"
    if args.flops is not None:
        resource["flops"] = args.flops
        resource["flops_status"] = "user-supplied external profiler result"
    else:
        resource.setdefault("flops", None)
        resource.setdefault("flops_status", "unavailable_without_optional_profiler")
    resource_path.write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(resource, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

