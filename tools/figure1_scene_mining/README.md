# EvidenceHOI Figure 1 Scene Mining

This directory performs pair-centric, reproducible mining of HICO-DET images for the EvidenceHOI Figure 1 base photograph.

## Data policy

- Uses the local HICO-DET copy at `/share/share/Datasets/hico_20160224_det` through the audited index in `outputs/figure1_search_v2/index/`.
- Does not alter dataset images or training/model code.
- Does not download datasets or web images.
- Automatic scores are search-order proxies. Semantic ambiguity and evidence roles are confirmed manually from image pixels.

## Reproduce

```bash
python tools/figure1_scene_mining/mine_candidates.py --max-per-scenario 250
python tools/figure1_scene_mining/render_contact_sheet.py --top-k 40 --per-page 20
python tools/figure1_scene_mining/score_candidates.py
python tools/figure1_scene_mining/annotate_candidates.py --top-k 3
```

The fixed random seed is `20260714`.

## Main outputs

- `outputs/raw_candidates.csv`: 1,500 structurally filtered pair candidates (250 per scenario).
- `outputs/contact_sheets/`: scenario-specific 5x4 review pages.
- `outputs/shortlist.csv`: manual 100-point shortlist with H/A/O/S/D/C mapping.
- `outputs/final_report.md`: Top-10, Top-3 narratives, and explicit final recommendation.
- `outputs/finalists/`: original, target-pair overlay, evidence-role overlay, and panel mockups for the Top-3.
