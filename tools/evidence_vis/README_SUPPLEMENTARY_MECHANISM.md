# EvidenceHOI Supplementary Mechanism Visualization

This workflow selects and renders real HICO-DET evaluation examples for
Supplementary Section D.1. It is trace-only after inference and does not modify
predictions or checkpoints.

## 1. Model equivalence check

The Trace implementation is the same parameterized model as `src/models/stip.py`.
Only opt-in evaluation serialization is added.

```bash
python tools/evidence_vis/check_stip_trace_equivalence.py \
  --base src/models/stip.py \
  --trace src/models/stip_graph6.py
```

A passing report requires:

- no trainable assignments added or removed;
- identical criterion, post-processing, and relation feature extractor;
- only `__init__` and `forward` changed for Trace wiring;
- exactly six Trace helper methods added.

The actual `--stip_model_variant evidence_trace` entry uses
`src/models/stip_evidence_trace.py`, which has the same model structure and
adds a GT-format fallback for trace metadata.

## 2. Export the final-model Trace

The wrapper is configured for the selected checkpoint: `M=4`, `K=5`, `L=2`,
and `k_oo=3`. Edit the user configuration block near the top of
`run_evidence_trace_stip_final.sh`:

```bash
RESUME_CHECKPOINT="/path/to/final/best.pth"
DETR_WEIGHTS="/path/to/detr.pth"
BLIP_PRETRAINED="/path/to/blip.pth"
BLIP_VIT="large"
BACKBONE="resnet101"
GPU_IDS="0"
EVAL_MAX_SAMPLES=64
```

Then run:

```bash
bash run_evidence_trace_stip_final.sh
```

Use `EVAL_MAX_SAMPLES=64` for the first smoke export. After the checkpoint
profile and smoke output pass, change it to `0` and optionally expand
`GPU_IDS` for the full test split.

The wrapper checks checkpoint dimensions before inference. Do not bypass a
profile mismatch for a paper figure.

`TRACE_FLOAT_DTYPE=float32` is the default because very small alpha values are
shown numerically. Set `TRACE_EACH_LAYER=True` only when layerwise panels or
Section C.3 statistics are needed; it increases storage.

## 3. Rank scientifically eligible examples

```bash
python tools/evidence_vis/rank_supplementary_examples.py \
  --trace-dir /path/to/final_trace_output/traces \
  --output-dir /path/to/final_trace_output/supplementary_ranking \
  --action-names-json /path/to/action_names.json \
  --object-names-json /path/to/object_names.json \
  --node-names-json /path/to/coco_node_names.json \
  --image-root /path/to/hico/images/test2015 \
  --expected-m 4 \
  --expected-k 5 \
  --expected-roles 6 \
  --expected-layers 2 \
  --strict-config
```

Main outputs:

- `all_correct_pairs_scored.csv`: complete audit table;
- `supplementary_candidates.csv`: all eligible candidates by figure score;
- `supplementary_shortlist.csv`: action/object/image-diverse review set;
- `supplementary_ranking_summary.json`: observed profile, filters, and warnings.

The figure score is only a mining heuristic. It combines:

- correct GT pair and correct final top-1 action;
- non-endpoint, non-duplicate, visible context;
- original-image exposure, contrast, resolution, and sharpness proxies;
- non-uniform learned path-role correspondence;
- role coverage and prototype/prediction readability.

Uniform correspondence is explicitly baseline-corrected and cannot receive a
high mechanism score merely because four context tokens exist.
The default `--min-mechanism-score 0.05` excludes nearly uniform matrices from
the formal shortlist. Set it to zero only for renderer smoke tests or audits;
do not lower it silently to obtain a paper figure.

## 4. Render the shortlist

```bash
python tools/evidence_vis/render_supplementary_mechanism.py \
  --ranking-csv /path/to/final_trace_output/supplementary_ranking/supplementary_shortlist.csv \
  --ranks 1-8 \
  --image-root /path/to/hico/images/test2015 \
  --action-names-json /path/to/action_names.json \
  --output-dir /path/to/final_trace_output/supplementary_figures \
  --expected-m 4 \
  --expected-k 5 \
  --expected-roles 6 \
  --expected-layers 2 \
  --strict-config
```

The default `--correspondence-view probability` prints the saved absolute
probabilities. For a checkpoint whose probabilities are near uniform, use
`--correspondence-view deviation` only as an explicitly labeled diagnostic:
cells then show `P(role | visual unit) - 1/(valid roles)` in units of `10^-3`,
while the subtitle reports the baseline and maximum absolute deviation. The
original probabilities remain unchanged in `figure_metadata.json`. This mode
improves numerical legibility but does not make weak correspondence strong.

Each figure contains:

1. target human/object and selected context boxes;
2. H-C and C-O path availability;
3. the `M+3` visual evidence chain;
4. retrieved Top-K interaction prototypes;
5. the `(M+3) x 6` visual-to-role soft correspondence matrix;
6. alpha evidence weights, pi prototype weights, and six-role coverage;
7. coarse and final scores for the correct and strongest competing action;
8. a fixed interpretation boundary.

The display roles are always the final-paper roles:
Human State (H), Object State (O), Action Cue (A), Key Cue (K), Scene Cue (S),
and Context Object (C). Older internal names are retained only in metadata for
audit and never shown as final terminology.

Additional outputs:

- `figure_metadata.json`: all plotted values and provenance;
- `caption_drafts.md`: conservative caption drafts;
- `render_report.json`: failures and strict-profile status;
- `supplementary_mechanism_contact.jpg`: visual review sheet.

## 5. Scientific interpretation boundary

The matrices, alpha, pi, coverage, and attention values are learned soft model
states. They may support discussion of correspondence, suppression, retrieval,
and refinement dynamics. They must not be described as human evidence labels,
ground-truth explanations, causal attributions, correct evidence, or minimal
sufficient evidence.

Existing `outputs/evidence_trace_graph6_m4r5_full` records have
`M=4, K=5, roles=6, L=2` and were exported from the selected checkpoint.
They can be reused directly for offline ranking and rendering.
