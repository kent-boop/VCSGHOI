# VET Internal Ablation Results

This is an empty results template. Use
`tools/collect_vet_internal_ablation_results.py` after formal runs; do not
replace missing values with zero.

## HICO-DET

| Variant | Full | Rare | Non-Rare | Seeds | Status |
|---|---:|---:|---:|---|---|
| V0 HO Anchor Only | N/A | N/A | N/A | N/A | pending |
| V1 + Shared Visual Relation Memory | N/A | N/A | N/A | N/A | pending |
| V2 + Endpoint States | N/A | N/A | N/A | N/A | pending |
| V3 + Context Paths | N/A | N/A | N/A | N/A | pending |
| V4 + Learned Path Selection | N/A | N/A | N/A | N/A | pending |

## V-COCO

| Variant | AP role #1 | AP role #2 | Seeds | Status |
|---|---:|---:|---|---|
| V0 HO Anchor Only | N/A | N/A | N/A | pending |
| V1 + Shared Visual Relation Memory | N/A | N/A | N/A | pending |
| V2 + Endpoint States | N/A | N/A | N/A | pending |
| V3 + Context Paths | N/A | N/A | N/A | pending |
| V4 + Learned Path Selection | N/A | N/A | N/A | pending |

## Supplementary V5

V5 remains pending until path weights pass causal validation and complete the
same formal seed set as V4.

## Interpretation rules

- If V2 is close to V1, endpoint information may already be encoded in the anchor.
- If V3 decreases, explicit context may be noisy.
- If V4 recovers from V3, learned selection may be filtering noisy paths.
- A V5 claim is valid only if learned VET weights demonstrably affect downstream computation.

