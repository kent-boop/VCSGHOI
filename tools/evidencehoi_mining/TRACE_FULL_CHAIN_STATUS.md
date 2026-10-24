# Evidence Trace full-chain status

Updated: 2026-07-12

## Current status

The code path is connected end to end without replacing the training model:

```text
STIP_main.py
  -> src/models/detr.py
  -> --stip_model_variant evidence_trace
  -> src/models/stip_evidence_trace.py
  -> visual_scene_graph_oov8.py debug_info["evidence_trace"]
  -> outputs["evidence_traces"]
  -> evaluator_hico.py
  -> EvidenceTraceWriter
  -> records/*.pt + manifests + run metadata
  -> validate_evidence_trace.py
  -> rank_evidence_examples.py
  -> render_evidence_shortlist.py
```

The pre-integration source snapshot is archived in GitHub commit
`3cc628a1e4866e71fe7365821ac101204023980e`.

The original `src/models/stip.py`, `src/models/stip_graph6.py`, and all
checkpoints remain unchanged. The Trace variant adds no trainable layers or
checkpoint keys.

## Runtime configuration

There are two checkpoint-compatible profiles. Do not mix their M/R values.

### EvidenceHOI-Large

- checkpoint:
  `checkpoints/EvidenceHOI-Large/STIP_opensource/hico_run_graph6_base/best.pth`
- DETR backbone: ResNet-101
- BLIP width: 1024 (Large)
- contextual evidence count: `graph_evidence_topk=1`
- retrieved prototype count: `graph_text_topk=3`
- saved epoch: 14

### Graph6 M4/R5

- checkpoint:
  `checkpoints/hico-det_sync_vt_graph6/STIP_opensource/hico_run_graph6/best.pth`
- DETR backbone: ResNet-101
- BLIP width: 1024 (Large)
- contextual evidence count: `graph_evidence_topk=4`
- retrieved prototype count: `graph_text_topk=5`
- saved epoch: 17

Both scripts use float16 CPU Trace tensors, full action logits, and GT
matching. `graph_path_topk=3` is a separate path-context option.

The scripts run `check_trace_checkpoint_profile.py` before model startup.
A mismatched checkpoint now fails immediately with a concise profile error.

## BLIP requirement

The active BLIP line in `src/models/detr.py` must use
`model_large_retrieval_coco.pth` with `vit="large"`. Both scripts verify
this before model startup.

## Two-image smoke tests

For the preferred EvidenceHOI-Large checkpoint:

```bash
conda activate SPLHOI
cd /wzj/zijian/OSSM-HOI-main
bash run_evidence_trace_smoke_large_m1r3.sh
```

For the M4/R5 checkpoint:

```bash
conda activate SPLHOI
cd /wzj/zijian/OSSM-HOI-main
bash run_evidence_trace_smoke_graph6_m4r5.sh
```

Each wrapper enforces one GPU, two test images, and two Trace records. Success
ends with:

```text
record count: 2
graph available: True (2/2)
manifest rows: 2
TRACE VALIDATION PASSED
```

Both wrappers have now passed with the real HICO test loader, BLIP Large,
ResNet-101 DETR weights, and their matching checkpoints. EvidenceHOI-Large
produced one context per pair; Graph6 M4/R5 produced four contexts per pair
and valid two-hop context fields.

## Full HICO-DET test export

EvidenceHOI-Large M1/R3:

```bash
bash run_evidence_trace_large_m1r3_v2.sh
```

Graph6 M4/R5:

```bash
bash run_evidence_trace_graph6_m4r5_v2.sh
```

Both scripts automatically validate all generated records after evaluation.

The scripts derive the expected full-test record count from
`annotations/test_hico.json` (currently 9546). Multi-GPU evaluation padding
is deduplicated by image identity before validation. Duplicate `.pt` records
are retained under `duplicate_records/`, while `records/` and `manifest.jsonl`
contain only the unique active Trace set.

### Verified Graph6 M4/R5 full run

The complete HICO-DET test export finished successfully with the matching
M4/R5 checkpoint:

- evaluation time: `1:02:02`
- mAP: `48.7473`
- rare mAP: `51.4820`
- non-rare mAP: `47.9305`
- unique Trace records: `9546`
- graph Trace availability: `9546/9546`
- final validation: passed

## Ranking

```bash
TRACE_DIR=/path/to/run/traces
python tools/evidence_vis/rank_evidence_examples.py \
  --trace-dir "${TRACE_DIR}" \
  --output-dir "${TRACE_DIR}/ranking"
```

The default ranker requires at least two contexts. Therefore:

- Graph6 M4/R5 is the recommended profile for multi-evidence visualization.
- EvidenceHOI-Large M1/R3 must add `--min-context-count 1`; it cannot provide
  the same multi-context evidence display because the checkpoint was trained
  with only one contextual evidence slot.

The two-image M4/R5 smoke produced 200 pair-specific rows. Its combined hero
list was empty because the first two HICO test images had no GT-matched
positive pair in the exported predictions; this is a sample-size result, not
a Trace or ranking failure.

Outputs:

- `hypothesis_sensitive_candidates.csv`
- `pair_specific_candidates.csv`
- `combined_hero_candidates.csv`
- `ranking_summary.json`

The corrected full ranking computes combined matches before truncating either
source leaderboard. The Graph6 M4/R5 export produced:

- 64 raw hypothesis-sensitive candidates
- 1,190,568 raw pair-specific candidates
- 64 combined hero candidates

Pair-specific visual review additionally filters controlled endpoint IoU to
`<= 0.3`, reducing candidates caused by overlapping duplicate detections. The
current shortlist and audit report are stored in
`outputs/evidence_trace_graph6_m4r5_full/shortlist_v1/`.

The current hypothesis-sensitive pool is diagnostic rather than claim-ready:
only 1/64 candidates changes top context, maximum JS divergence is `0.0007102`,
and maximum heatmap distance is `0.0010254`. A second-pass evidence deletion test is required before using these heatmaps for a faithfulness claim.

### GT-grounded evidence-chain review

Manifest prefiltering found 93 images with at least two GT-matched pairs. The
GT-grounded ranking produced 343 controlled pair-specific combinations; 9
combinations have both pairs GT-matched and Top-1 correct.

Three evidence-chain panels are stored in
`outputs/evidence_trace_graph6_m4r5_full/evidence_chain_examples_v2/`:

- bird/feed: both pairs correct; primary example
- cow/herd: both pairs correct; distant-scene alternative
- hot-dog make/cook: one correct pair and one failure contrast

The selected context sets are pair-specific and mostly two-hop, but final
action-specific Context C1-C4 weight is only about 0.013-0.017%. Human receives
about 69-71% and Object about 27-29%. These panels demonstrate context
retrieval, while also documenting that the current checkpoint nearly suppresses
context tokens during final action matching.

## Rendering

HICO Trace records now contain the real test file name, so no numeric
`image-pattern` guess is needed.

```bash
python tools/evidence_vis/render_evidence_shortlist.py \
  --ranking-csv "${TRACE_DIR}/ranking/combined_hero_candidates.csv" \
  --image-root /share/share/Datasets/hico_20160224_det/images/test2015 \
  --output-dir "${TRACE_DIR}/preview_combined" \
  --limit 50
```

## Verification completed

- Python syntax checks passed in SPLHOI.
- All four Trace Bash scripts passed `bash -n`.
- Synthetic Trace record, GT mapping, Writer, metadata, manifest, validator,
  and ranker-consumption tests passed.
- Existing VET, SETM, and prototype-slot smoke tests passed.
- A simulated multi-rank padding case passed manifest deduplication, duplicate
  record archiving, and exact-count validation.
- Both real two-image model runs loaded their checkpoint without shape errors,
  wrote exactly two complete graph Trace records, and passed validation.
- The M4/R5 real Trace was consumed successfully by the ranking stage.
- Full ranking truncation was fixed and covered by a regression test; all 64
  hypothesis candidates now receive their best partner from the complete
  pair-specific pool.
- Pair-specific endpoint-IoU filtering and a 10-example audited shortlist were
  generated without rerunning the model.
- The Graph6 M4/R5 full HICO-DET run produced 9546 validated unique records.
