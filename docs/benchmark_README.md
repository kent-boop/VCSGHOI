# EvidenceHOI efficiency benchmark

## Delivered tools

- `tools/benchmark_efficiency.py`: clean-subprocess orchestrator and per-variant worker.
- `tools/benchmark_utils.py`: manifest validation, official builders/adapters,
  environment capture, strict checkpoint loading, parameter counting, fixed subset,
  CUDA timing/memory, module groups, and sanity validation.
- `tools/profile_flops.py`: real-image fvcore profiling with runtime fallback and
  explicit unsupported-operator handling.
- `tools/generate_efficiency_table.py`: strict cross-run validation and
  CSV/JSON/Markdown/LaTeX generation.
- `configs/efficiency_variants.yaml`: R50+ViT-B main variants plus disabled optional
  R101+ViT-L variants.

## Additional Python dependencies

`PyYAML` is required for the variant manifest. `fvcore` is recommended for the
primary FLOPs trace; without it, the runtime profiler fallback is diagnostic and
FLOPs remain N/A for the paper table. `pytest` is only needed to run the supplied
test file.

```bash
pip install pyyaml fvcore pytest
python -m pytest -q tests/test_efficiency_benchmark.py
```

## Protocol defaults

The main protocol is batch size 1, FP32, 50 warm-up forwards, 200 fixed real
HICO-DET test images, and three timed forwards per image. CUDA latency uses events
and explicit synchronization. Peak memory uses `max_memory_allocated`; reserved
memory is diagnostic only. Each variant runs in a fresh process.

The sample list is selected deterministically in low/medium/high annotation
complexity strata and written once to `outputs/efficiency/sample_ids.txt`.
All later variants must have the same image count and reuse that file.

## Architecture-only parameter smoke run

No variant checkpoint or prediction is needed, but the repository builder still
loads its BLIP-B pretrained file while constructing the model. HICO-DET metadata
is also required to instantiate the correct heads.

```bash
python tools/benchmark_efficiency.py \
  --variant evidencehoi_r50_vitb \
  --variant-manifest configs/efficiency_variants.yaml \
  --dataset-file hico-det \
  --data-root /path/to/hico_20160224_det \
  --architecture-only \
  --metrics params \
  --device cpu \
  --num-images 200 \
  --output-dir outputs/efficiency_params
```

This produces parameter artifacts but deliberately sets `paper_ready=false`.

## Full EvidenceHOI smoke run

The confirmed full-model checkpoint is already stored as a repository-relative path in the
manifest; `--checkpoint` can still override it for a single variant. This
command runs predictions on ten real images and therefore requires a CUDA GPU,
the trained checkpoint, the BLIP-B pretrained file used by the builder, and the
HICO-DET test split.

```bash
python tools/benchmark_efficiency.py \
  --variant evidencehoi_r50_vitb \
  --variant-manifest configs/efficiency_variants.yaml \
  --dataset-file hico-det \
  --data-root /path/to/hico_20160224_det \
  --device cuda:0 \
  --batch-size 1 \
  --precision fp32 \
  --warmup-iters 10 \
  --num-images 10 \
  --flops-num-images 10 \
  --repeats-per-image 1 \
  --metrics params,flops,latency,memory,module \
  --accept-official-forward-scope \
  --output-dir outputs/efficiency_smoke
```

Use a separate smoke output directory. The formal 200-image run must not reuse a
10-image `sample_ids.txt`.

## Formal full-model run

After the smoke run succeeds, use the default protocol:

```bash
python tools/benchmark_efficiency.py \
  --variant evidencehoi_r50_vitb \
  --variant-manifest configs/efficiency_variants.yaml \
  --dataset-file hico-det \
  --data-root /path/to/hico_20160224_det \
  --device cuda:0 \
  --batch-size 1 \
  --precision fp32 \
  --warmup-iters 50 \
  --num-images 200 \
  --flops-num-images 50 \
  --repeats-per-image 3 \
  --metrics params,flops,latency,memory,module \
  --accept-official-forward-scope \
  --output-dir outputs/efficiency
```

`--force` permits replacing a run. `--resume` only accepts an existing summary
whose critical run hash matches; it does not merge mismatched protocols.

## Five-row table

Before `--all-enabled`, assign all five checkpoint environment variables and
resolve the two blocked Table 4(a) mappings documented in `efficiency_audit.md`.
The formal runner refuses unverified variants. After all summaries are
paper-ready, generate the table:

```bash
python tools/generate_efficiency_table.py \
  --input-dir outputs/efficiency \
  --variant-manifest configs/efficiency_variants.yaml
```

The generator refuses incomplete runs, mixed GPUs/precision/software, different
sample hashes, or different critical protocols. It never fills missing values.
