"""Smoke checks for VET result collection and paired deltas."""

import json
import sys
import tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tools.collect_vet_internal_ablation_results import collect_runs, summarize


def write_run(root: Path, mode: str, seed: int, full: float, run_kind="formal", config_hash="same"):
    run_dir = root / mode / str(seed)
    run_dir.mkdir(parents=True)
    manifest = {
        "dataset": "hico-det",
        "mode": mode,
        "seed": seed,
        "run_kind": run_kind,
        "config_hash_excluding_mode_seed": config_hash,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    metrics = {"mAP_full": full, "mAP_rare": full - 1, "mAP_non_rare": full + 1}
    (run_dir / "metrics.jsonl").write_text(json.dumps(metrics) + "\n", encoding="utf-8")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for seed, v0, v1 in [(3407, 30.0, 31.0), (3408, 32.0, 34.0)]:
            write_run(root, "anchor_raw", seed, v0)
            write_run(root, "anchor_svrm", seed, v1)
        write_run(root, "endpoints", 9999, 99.0, run_kind="smoke")
        runs, rejected = collect_runs(root, include_smoke=False)
        assert len(runs) == 4
        assert len(rejected) == 1 and rejected[0]["reason"] == "non-formal run"
        report = summarize(runs, allow_v5=False)
        v0 = report["tables"]["hico-det"][0]["Full"]
        assert v0["mean"] == 31.0 and v0["n"] == 2
        delta = report["paired_deltas"]["hico-det"][0]["Full"]
        assert delta["per_seed"] == [1.0, 2.0]
        assert delta["mean"] == 1.5
        assert report["paired_deltas"]["hico-det"][-1]["status"].startswith("pending")
        assert report["tables"]["vcoco"][0]["AP_role#1"]["mean"] is None
    print("VET result collector smoke passed")


if __name__ == "__main__":
    main()

