# VET Internal Ablation Experiments

This document records the isolated VET V0--V5 implementation. It does not
contain measured performance until formal runs have completed.

## Isolation boundary

The experiment uses `vet_internal_ablation`, which inherits the detector,
BLIP fusion, LPT, layerwise SETM, interaction decoder, classifier, criterion,
and post-process from the B5 implementation. It replaces only the VET
reasoner with files whose names end in `vet_ablation`.

All modes instantiate the same modules and parameters, emit `3+K` visual
evidence slots, and call a Stage-4 SETM matcher after every one of the fixed
visual scheduling layers. With the frozen default, `K=4` and there are two
SETM calls per image-level VET invocation.

## Modes

| ID | CLI mode | SVRM | Active evidence | Path selection | VET path weighting |
|---|---|---:|---|---|---|
| V0 | `anchor_raw` | identity | raw HO anchor | none | none |
| V1 | `anchor_svrm` | enabled | refined HO anchor | none | none |
| V2 | `endpoints` | enabled | anchor, human, object | none | none |
| V3 | `context_uniform` | enabled | V2 plus context | deterministic | uniform |
| V4 | `path_selection` | enabled | V2 plus context | learned hard top-K | uniform |
| V5 | `path_reweight` | enabled | same selection as V4 | learned hard top-K | learned relative context weights |

V0 keeps the same scheduling depth and SETM call count, but each visual
relation layer is an identity. SETM write-back from an earlier layer may be
seen by the next layer; shared visual relation message passing is the removed
component.

V2 tests explicit endpoint tokens. It does not claim that endpoints first
enter the model at V2: endpoints can already influence the relation anchor
and fixed pair-output path.

V3 selects paths without reading `path_score_mlp`. Full two-hop connectivity
is ranked before one-sided connectivity; edge scores and then lower node
index provide deterministic tie-breaking. `path_mlp` remains trainable.

V4 uses deterministic hard top-K in the forward pass. Because hard indices
alone would give the selection MLP zero gradient when selected paths are
uniform, the implementation uses a hard-forward/soft-backward
straight-through gate whose forward value is exactly one.

For V5 with `n` selected contexts, each of the three base tokens keeps weight
`1/(3+n)`. Contexts keep total mass `n/(3+n)` and distribute it using softmax
over selected path scores. Therefore V5 changes relative context allocation,
not total context mass. These weights affect `evidence_context`, then the VET
candidate anchor delta, which is added to SETM delta during dense HO edge
write-back.

## Frozen protocol

The machine-readable protocol is in
`configs/vet_internal_ablation_frozen.json`. Formal runs use seeds 3407,
3408, and 3409; the dedicated main respects the CLI seed rather than the
legacy hard-coded seed. Formal runs reject a non-empty resume checkpoint by
default. Interrupted-run resume requires explicit authorization and an
identical manifest.

HICO-DET reports Full, Rare, and Non-Rare. V-COCO reports role AP scenario 1
and scenario 2. Smoke runs are written under a separate directory and are not
accepted as formal results.

## Commands

```bash
python tests/smoke_vet_internal_ablation.py

MODE=anchor_raw SEED=3407 RUN_KIND=smoke \
  ./run_vet_internal_ablation_hico.sh

MODE=path_selection SEED=3407 RUN_KIND=formal \
  ./run_vet_internal_ablation_hico.sh

DATASET=hico-det DRY_RUN=1 \
  ./scripts/run_vet_internal_ablation_matrix.sh
```

The matrix launcher defaults to `DRY_RUN=1`. Set it to zero only when the
dataset, text assets, DETR weights, and GPU allocation have been verified.

## Attribution

Formal attribution uses paired seeds:

```text
Delta_SVRM      = V1 - V0
Delta_Endpoint  = V2 - V1
Delta_Context   = V3 - V2
Delta_Selection = V4 - V3
Delta_Weight    = V5 - V4
```

No monotonic improvement is assumed. V5 should remain supplementary until
the causal path-weight smoke and real-run gradient diagnostics are accepted.

