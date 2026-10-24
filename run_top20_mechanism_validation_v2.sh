#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV_BIN="${PYTHON_ENV_BIN:-/share/apps/anaconda3/envs/SPLHOI/bin}"
export PATH="${PYTHON_ENV_BIN}:${PATH}"

# 用户修改区
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/wzj/zijian/OSSM-HOI-main/checkpoints/hico-det_sync_vt_graph6/STIP_opensource/hico_run_graph6/best.pth}"
DETR_WEIGHTS="${DETR_WEIGHTS:-/share/share/HOI/pretrained_model/HICO/all/res101/checkpoint0105.pth}"
BLIP_PRETRAINED="${BLIP_PRETRAINED:-/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_large_retrieval_coco.pth}"
BLIP_VIT="${BLIP_VIT:-large}"
BACKBONE="${BACKBONE:-resnet101}"
GPU_IDS="${GPU_IDS:-2,3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GRAPH_EVIDENCE_TOPK="${GRAPH_EVIDENCE_TOPK:-4}"
GRAPH_TEXT_TOPK="${GRAPH_TEXT_TOPK:-5}"
TRACE_FLOAT_DTYPE="${TRACE_FLOAT_DTYPE:-float32}"
VARIANTS="${VARIANTS:-full no_anchor_writeback no_prototype_state_writeback no_setm endpoint_only_setm}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/outputs/evidence_trace_top20_mechanism_validation_v2_${RUN_TAG}}"
TOP20_CSV="${TOP20_CSV:-${SCRIPT_DIR}/outputs/evidence_trace_stip_final_20260731_190715/supplementary_top20_mechanism_relative/ranking/top20_mechanism_candidates.csv}"
OLD_TRACE_DIR="${OLD_TRACE_DIR:-${SCRIPT_DIR}/outputs/evidence_trace_top20_detailed_20260731_top20_vet_setm_detailed/traces}"
DATA_PATH="${DATA_PATH:-/share/share/Datasets/hico_20160224_det}"
TEST_ANNOTATION="${TEST_ANNOTATION:-${DATA_PATH}/annotations/test_hico.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${DATA_PATH}/images/test2015}"
IMAGE_IDS_FILE="${IMAGE_IDS_FILE:-${RUN_ROOT}/manifests/top20_image_ids.json}"
TEXT_GRAPH_DIR="${TEXT_GRAPH_DIR:-${SCRIPT_DIR}/hico_text_graph_blip256}"
TEXT_FEATURES_PATH="${TEXT_FEATURES_PATH:-${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_features_6x256.npy}"
TEXT_MASK_PATH="${TEXT_MASK_PATH:-${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_mask_6.npy}"
PROMPT_METADATA_JSON="${PROMPT_METADATA_JSON:-${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_prompt_metadata.json}"
ACTION_NAMES_JSON="${ACTION_NAMES_JSON:-${SCRIPT_DIR}/outputs/evidence_trace_graph6_m4r5_full/label_maps/action_names.json}"
OBJECT_NAMES_JSON="${OBJECT_NAMES_JSON:-${SCRIPT_DIR}/outputs/evidence_trace_graph6_m4r5_full/label_maps/object_names.json}"

allowed_variants=(full no_anchor_writeback no_prototype_state_writeback no_setm endpoint_only_setm)

is_allowed_variant() {
    local candidate="$1"
    local allowed
    for allowed in "${allowed_variants[@]}"; do
        [[ "$candidate" == "$allowed" ]] && return 0
    done
    return 1
}

mkdir -p \
    "$RUN_ROOT/config/variants" \
    "$RUN_ROOT/manifests" \
    "$RUN_ROOT/provenance/code_snapshot" \
    "$RUN_ROOT/reports" \
    "$RUN_ROOT/logs/variants" \
    "$RUN_ROOT/variants"

python - "$TOP20_CSV" "$IMAGE_IDS_FILE" <<'PY'
import csv
import json
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open("r", encoding="utf-8")))
image_ids = [int(row["image_id"]) for row in rows]
if len(rows) != 20 or len(set(image_ids)) != 20:
    raise SystemExit(f"Expected 20 unique Top-20 image IDs, got rows={len(rows)}, unique={len(set(image_ids))}")
output = Path(sys.argv[2])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps({"image_ids": image_ids}, indent=2) + "\n", encoding="utf-8")
print(f"Frozen {len(image_ids)} image IDs at {output}")
PY

if [[ ! -f "$RUN_ROOT/run_meta.json" ]]; then
    python "$SCRIPT_DIR/tools/evidence_vis/freeze_mechanism_validation_metadata.py" \
        --output-dir "$RUN_ROOT" \
        --repo-root "$SCRIPT_DIR" \
        --checkpoint "$RESUME_CHECKPOINT" \
        --detr-weights "$DETR_WEIGHTS" \
        --blip-pretrained "$BLIP_PRETRAINED" \
        --dataset-root "$DATA_PATH" \
        --annotation "$TEST_ANNOTATION" \
        --image-root "$IMAGE_ROOT" \
        --top20-csv "$TOP20_CSV" \
        --old-trace-dir "$OLD_TRACE_DIR" \
        --prototype-bank "$TEXT_FEATURES_PATH" \
        --prototype-mask "$TEXT_MASK_PATH" \
        --prototype-prompt-metadata "$PROMPT_METADATA_JSON" \
        --action-labels "$ACTION_NAMES_JSON" \
        --object-labels "$OBJECT_NAMES_JSON" \
        --gpu-ids "$GPU_IDS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --seed 3407 \
        --trace-float-dtype "$TRACE_FLOAT_DTYPE"
fi

for variant in $VARIANTS; do
    if ! is_allowed_variant "$variant"; then
        echo "[Error] Unsupported variant: $variant" >&2
        exit 2
    fi

    variant_root="$RUN_ROOT/variants/$variant"
    trace_dir="$variant_root/traces"
    validation_json="$RUN_ROOT/provenance/${variant}_raw_validation.json"
    mapping_csv="$RUN_ROOT/manifests/${variant}_pair_mapping.csv"
    mapping_json="$RUN_ROOT/manifests/${variant}_pair_mapping.json"
    complete_marker="$variant_root/.validation_complete"

    if [[ -f "$complete_marker" ]]; then
        python "$SCRIPT_DIR/tools/evidence_vis/validate_mechanism_validation_trace.py" \
            --trace-dir "$trace_dir" \
            --variant "$variant" \
            --expected-records 20 \
            --output-json "$validation_json"
        echo "[Resume] $variant already validated; skipping inference."
        continue
    fi

    mkdir -p "$variant_root"
    {
        echo
        echo "# variant=$variant"
        echo "RUN_ROOT=$RUN_ROOT"
        echo "GPU_IDS=$GPU_IDS"
        echo "GRAPH_MECHANISM_VALIDATION_VARIANT=$variant"
        echo "bash $SCRIPT_DIR/run_evidence_trace_stip_final.sh"
    } >> "$RUN_ROOT/provenance/commands.txt"

    RUN_TAG="${RUN_TAG}_${variant}" \
    OUTPUT_ROOT="$variant_root" \
    RESUME_CHECKPOINT="$RESUME_CHECKPOINT" \
    DETR_WEIGHTS="$DETR_WEIGHTS" \
    BLIP_PRETRAINED="$BLIP_PRETRAINED" \
    BLIP_VIT="$BLIP_VIT" \
    BACKBONE="$BACKBONE" \
    GPU_IDS="$GPU_IDS" \
    BATCH_SIZE="$BATCH_SIZE" \
    NUM_WORKERS="$NUM_WORKERS" \
    DATA_PATH="$DATA_PATH" \
    TEST_ANNOTATION="$TEST_ANNOTATION" \
    EVAL_IMAGE_IDS_FILE="$IMAGE_IDS_FILE" \
    EVAL_MAX_SAMPLES=0 \
    TRACE_EXPECTED_RECORDS=20 \
    TRACE_EACH_LAYER=True \
    TRACE_INCLUDE_MATCH_LOGITS=True \
    TRACE_INCLUDE_FEATURES=True \
    GRAPH_STORE_DENSE_DEBUG_TENSORS=True \
    TRACE_FLOAT_DTYPE="$TRACE_FLOAT_DTYPE" \
    TRACE_OVERWRITE=False \
    GRAPH_EVIDENCE_TOPK="$GRAPH_EVIDENCE_TOPK" \
    GRAPH_TEXT_TOPK="$GRAPH_TEXT_TOPK" \
    GRAPH_EXPORT_MECHANISM_VALIDATION_TRACE=True \
    GRAPH_MECHANISM_VALIDATION_VARIANT="$variant" \
    bash "$SCRIPT_DIR/run_evidence_trace_stip_final.sh" \
        2>&1 | tee "$RUN_ROOT/logs/variants/${variant}.log"

    python "$SCRIPT_DIR/tools/evidence_vis/validate_detailed_vet_setm_trace.py" \
        --trace-dir "$trace_dir" \
        --image-ids-file "$IMAGE_IDS_FILE" \
        --expected-layers 2 \
        --expected-m 4 \
        --expected-k 5 \
        --expected-roles 6 \
        --require-features \
        --require-dense

    python "$SCRIPT_DIR/tools/evidence_vis/validate_mechanism_validation_trace.py" \
        --trace-dir "$trace_dir" \
        --variant "$variant" \
        --expected-records 20 \
        --output-json "$validation_json"

    python "$SCRIPT_DIR/tools/evidence_vis/remap_ranking_pairs_to_trace.py" \
        --ranking-csv "$TOP20_CSV" \
        --trace-dir "$trace_dir" \
        --output-csv "$mapping_csv" \
        --report-json "$mapping_json" \
        --warning-iou 0.8

    cp "$trace_dir/run_metadata_rank0.json" "$RUN_ROOT/config/variants/${variant}_effective_args.json"
    touch "$complete_marker"
    echo "[Complete] $variant"
done

echo "Mechanism validation variants complete under: $RUN_ROOT"
