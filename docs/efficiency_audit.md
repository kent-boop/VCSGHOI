# EvidenceHOI efficiency benchmark source audit

## Scope and source of truth

The user confirmed that the final model implementation is `src/models/stip.py`.
That confirmation supersedes the older `stip_graph6.py` path named in the planning
document. The benchmark therefore builds the default STIP model through
`src.models.build_model` and then asserts that the resulting class was loaded
from `src/models/stip.py`. Historical `stip_graph*` and staged ablation model
files are not benchmark model entry points.

The final paper source/PDF was not present in the supplied workspace/attachment.
The planning document provides the final terminology and proposed values. The user-confirmed
R50+ViT-B checkpoint is the runtime source of truth for its architecture: `d=256, L=2,
k_oo=3, M=4, K=1, coarse_pool=20`. This resolves the plan/checkpoint conflict for the
full model; flags whose exact Table 4(a) semantics cannot be proved remain blocked.

## Official construction and inference chain

1. `src.engine.arg_parser.get_args_parser()` supplies shared defaults;
   `STIP_main.py` adds the STIP runtime defaults.
2. `src.data.datasets.build_dataset(image_set="test", args=args)` builds the
   official HICO-DET dataset and transforms.
3. Dataset metadata supplies `num_classes`, `num_actions`, `action_names`,
   `valid_obj_ids`, and `correct_mat` before model construction.
4. `src.models.build_model(args)` calls `src.models.detr.build(args)`.
5. With `HOIDet=True` and `STIP_relation_head=True`, the default branch constructs
   `src.models.stip.STIP`.
6. `src.util.misc.collate_fn` returns `(NestedTensor samples, targets, raw PIL images)`.
7. The official evaluator moves `samples` and tensor-valued target fields to the
   device, then calls `model(samples, images, targets)`.

The model output used for sanity checks contains `pred_logits`, `pred_boxes`,
`pred_rel_pairs`, `pred_actions`, and `pred_action_exists`. Graph-enabled variants
also expose `graph_input_debug` and `graph_reasoner_debug`, which permit workload
statistics without changing the forward implementation.

## Paper variant to code mapping

| Paper row | VET | KPT | SETM | Principal flags in `stip.py` / graph builder | Status |
|---|:---:|:---:|:---:|---|---|
| Baseline | no | no | no | `use_visual_scene_graph=False`; candidate/text/sync/layerwise sync all false | Executable |
| VET | yes | no | no | graph forward true; `graph_enable_candidate_evidence=True`; text/sync false | Executable |
| KPT | no | yes | no | text graph must be true, but `stip.py` only invokes it inside the visual graph forward | **Unresolved:** no verified KPT-only bypass of VET message passing |
| Cross-Attention Fusion | yes | yes | no | graph/candidate/text true; post-hoc cross-attention; both SETM switches false | Candidate mapping exists, but final Table 4(a) training flags require confirmation |
| EvidenceHOI | yes | yes | yes | graph/candidate/text/sync/layerwise sync true; dynamic retrieval false; duplicate final matching false | Executable full path |

The concrete candidate configuration and resource paths are stored in
`configs/efficiency_variants.yaml`. Unverified entries have
`paper_configuration_verified: false`; the orchestrator refuses a formal runtime
run for them.

## Verified final-path invariants

- The model width is taken from `hidden_dim=256`.
- Visual and text reasoning use `L=graph_num_layers=2`.
- Object-object topology uses `graph_k_oo=3`.
- Candidate visual evidence uses `graph_evidence_topk=4` (`M=4`).
- The confirmed full-model checkpoint records `graph_text_topk=1` (`K=1`) and
  `graph_text_pre_topk=20`. Its `graph_evidence_topk=4` gives `M=4`.
- Full layerwise SETM requires candidate evidence, KPT, `graph_text_sync_mode=layerwise`,
  and `graph_text_dynamic_retrieval=False`.
- Full SETM disables duplicate final matching and the old dense-injection path.

## Checkpoint behavior

`STIP_main.py` currently resumes with `strict=False`. That is not sufficient for
paper efficiency reporting. The benchmark accepts checkpoint roots containing
`model` or `state_dict`, records SHA256/epoch/missing/unexpected keys, and fails
if any key is missing or unexpected or any tensor shape is incompatible.

Every formal variant needs its own trained checkpoint. The manifest uses
environment placeholders and intentionally does not assign checkpoint files by
directory-name inference. Architecture-only parameter counting may run without
a variant checkpoint, but is marked not paper-ready.

## Important implementation facts affecting measurement

1. `STIP.__init__` constructs `visual_scene_graph` even when
   `use_visual_scene_graph=False`. Total Params therefore includes those owned
   parameters in the baseline, as required by the definition “all parameters in
   the instantiated model.” Module rows make inactive ownership visible. Removing
   those parameters only for counting would no longer describe the actual model.
2. `STIP.forward` accepts raw PIL images and performs PIL-to-BLIP conversion inside
   the official forward. Disk loading, JPEG decoding, DataLoader waiting, and the
   outer tensor transfer are excluded from CUDA-event timing, but the internal
   conversion cannot be excluded without changing the official model interface.
   Results are consequently labeled `official_forward`; formal export requires an
   explicit acknowledgement of this scope.
3. `src/models/detr.py` currently constructs BLIP-B from a hard-coded pretrained
   path. R50+ViT-B matches the main requested configuration, but even an
   architecture-only model build needs that BLIP file to exist. The optional
   ViT-L variants remain disabled because the builder does not expose a verified
   ViT-L selection argument.
4. FLOPs are dynamic. The profiler uses real fixed test images. It reports a paper
   value only if the fvcore trace has no unsupported operators for every profiled
   image; runtime profiler values with unknown coverage remain diagnostics.

## Unresolved items required before the five-row formal table

- Confirm the exact final-paper KPT-only implementation/flags in `stip.py`.
- Confirm that the proposed post-hoc cross-attention flags exactly match Table 4(a).
- The full R50+ViT-B checkpoint is confirmed as
  `checkpoints/EvidenceHOI-base/STIP_opensource/hico_run_graph6_base/best.pth`; assign
  the other four matching ablation checkpoints before generating the five-row table.
- The full checkpoint confirms `text_hybrid`, `graph_oo_lambda_text=0.5`, and `all`
  text-prior injection. Confirm the corresponding settings for the remaining Table 4(a) variants.
- Confirm acceptance of the `official_forward` latency scope, or refactor the
  official model interface once (and rerun every variant) so prebuilt BLIP tensors
  can be supplied without changing computation.
- Expose and validate the optional R101+ViT-L BLIP construction before enabling
  the two large variants.
