# EvidenceHOI Example-Mining Package

This package adds an **opt-in, evaluation-only** trace path for selecting compact qualitative examples from the final Graph6 / Stage 4-standard model.

## Included files

- `patched_in_place/visual_scene_graph_oov8.py` — patched original module; captures the last real layerwise VET/LPT/SETM trace.
- `patched_in_place/stip_graph6.py` — patched original STIP; adds image/node/pair/action/GT metadata to each trace.
- `visual_scene_graph_oov8_trace.py`, `stip_graph6_trace.py` — side-by-side variants that preserve the uploaded originals.
- `evidence_trace_model.patch` — unified patch against the uploaded original filenames.
- `evidence_trace_io.py` — distributed-safe per-image `.pt` writer and manifest merger.
- `rank_evidence_examples.py` — offline hard filtering and three leaderboards.
- `render_evidence_shortlist.py` — inspection panels with pair/context boxes and SETM heatmaps.
- `EvidenceHOI_Codex_Example_Mining_Prompt_Pack.md` — staged, copy-paste prompts for Codex.

## What is recorded

### Image and detector nodes

- image id, path (when supplied by the dataset target), original size;
- all DETR query boxes, detector scores/classes, node types, graph-valid mask.

### Candidate pairs and predictions

- pair endpoint query indices and relation-proposal score;
- raw final action logits/probabilities;
- the internal coarse action head (for mining only, **not** an independent baseline);
- pre-deduplication ranking scores consistent with the main HICO post-processing factors;
- detector-matched multi-label GT action targets.

### Evidence topology and matching

- VET selected context node indices and pair-level weights;
- whether each selected context has `h→k`, `k→o`, or a complete two-hop path;
- retrieved prototype IDs/action IDs/object IDs;
- six-slot validity mask;
- `text_to_visual_attn`, `visual_to_text_attn`;
- prototype-conditioned evidence weights, prototype scores/weights/coverage;
- SETM anchor update.

By default, D-dimensional feature vectors and raw match logits are **not** stored.

## Required parser options

The patched model reads options through `getattr`, so it remains backward compatible. Add these options to the repository's existing argument parser:

```python
parser.add_argument('--graph_export_evidence_trace', action='store_true')
parser.add_argument('--graph_export_trace_dir', default='')
parser.add_argument('--graph_export_trace_each_layer', action='store_true')
parser.add_argument('--graph_export_trace_to_cpu', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--graph_export_trace_float_dtype', choices=['float16', 'float32', 'bfloat16'], default='float16')
parser.add_argument('--graph_export_trace_include_match_logits', action='store_true')
parser.add_argument('--graph_export_trace_include_features', action='store_true')
parser.add_argument('--graph_export_trace_topk_actions', type=int, default=10)
parser.add_argument('--graph_export_trace_include_all_action_logits', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--graph_export_trace_match_ground_truth', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--graph_export_trace_overwrite', action='store_true')
```

Adapt underscore/hyphen spelling to the repository's established CLI style.

## Evaluation-loop integration

Create the writer once per rank, write before any cross-rank gather, and remove the trace from objects sent to the evaluator:

```python
from src.models.evidence_trace_io import EvidenceTraceWriter, merge_manifests
from src.util.misc import get_rank, is_dist_avail_and_initialized

trace_writer = None
if args.graph_export_evidence_trace:
    trace_writer = EvidenceTraceWriter(
        args.graph_export_trace_dir,
        rank=get_rank(),
        overwrite=args.graph_export_trace_overwrite,
    )

outputs = model(samples, images, targets)
if trace_writer is not None:
    trace_writer.write_outputs(outputs)

# Do not all_gather the large trace dictionaries.
evaluator_outputs = dict(outputs)
evaluator_outputs.pop('evidence_traces', None)
# ... normal post-processing/evaluation using evaluator_outputs ...
```

After all ranks finish and synchronize, rank 0 can call:

```python
merge_manifests(args.graph_export_trace_dir)
```

## Recommended first-pass settings

```text
graph_export_evidence_trace=True
graph_export_trace_each_layer=False
graph_export_trace_to_cpu=True
graph_export_trace_float_dtype=float16
graph_export_trace_include_match_logits=False
graph_export_trace_include_features=False
graph_export_trace_include_all_action_logits=True
graph_export_trace_match_ground_truth=True
```

This is enough for image mining and heatmaps while keeping storage moderate.

## Screening logic

The ranker first applies scientific/readability hard filters, then creates separate leaderboards.

### 1. Hard filters

- matched positive GT pair;
- at least two selected context nodes;
- at least one complete `h→k→o` selected path;
- context detector score and box-area thresholds;
- GT action is present among retrieved prototypes;
- competitor action is retrieved but is not any GT action;
- final GT probability exceeds competitor probability;
- pair-specific comparison preferably shares the same object node, human node, or object class.

### 2. Hypothesis-sensitive leaderboard

Ranks a fixed pair by:

- JS divergence between GT/competitor SETM evidence weights;
- distance between their six-slot `text_to_visual_attn` maps;
- ambiguity of the two action predictions;
- evidence concentration and top-context switch;
- prototype coverage;
- internal refinement gain from coarse to final logits.

### 3. Pair-specific leaderboard

Ranks two pairs in the same image by:

- set Jaccard distance between selected context nodes;
- weighted Jaccard distance using CETI context weights;
- controlled-comparison bonus;
- path and image readability.

### 4. Combined hero leaderboard

Joins a hypothesis-sensitive focal pair with its strongest pair-specific partner. The score is only a **mining heuristic**, never a paper metric.

## Commands

```bash
python rank_evidence_examples.py \
  --trace-dir /path/to/evidence_trace \
  --output-dir /path/to/evidence_ranking \
  --action-names-json /path/to/action_names.json \
  --object-names-json /path/to/object_names.json \
  --top-n 200
```

```bash
python render_evidence_shortlist.py \
  --ranking-csv /path/to/evidence_ranking/combined_hero_candidates.csv \
  --image-root /path/to/hico/images/test2015 \
  --output-dir /path/to/evidence_panels \
  --limit 50 \
  --action-names-json /path/to/action_names.json
```

## Final selection protocol

1. Inspect the top 30–50 rendered panels.
2. Keep 8–12 semantically clear candidates.
3. Run evidence deletion only for these candidates.
4. Select one hero image, one backup, and several supplementary examples.
5. In the paper, state that the same pair-specific VET evidence pool is **reweighted** by different prototypes; do not claim VET independently rebuilds a graph per action.
6. Use “decision-relevant” or “faithfulness diagnostic” for deletion results, not “causal proof.”

## Static checks performed here

All included Python files pass `python -m py_compile`. The ranker/writer/renderer were also exercised on a synthetic trace record.
