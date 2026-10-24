#!/bin/bash

set -euo pipefail

# Final-model HICO-DET evidence Trace export.
#
# The evidence_trace implementation has the same trainable model structure as
# src/models/stip.py and only adds opt-in evaluation serialization.
#
# 请直接修改下方“用户修改区”，然后运行本脚本。


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV_BIN="${PYTHON_ENV_BIN:-/share/apps/anaconda3/envs/SPLHOI/bin}"
if [[ ! -x "${PYTHON_ENV_BIN}/python" || ! -x "${PYTHON_ENV_BIN}/torchrun" ]]; then
    echo "[Error] Python environment is incomplete: ${PYTHON_ENV_BIN}" >&2
    exit 1
fi
export PATH="${PYTHON_ENV_BIN}:${PATH}"

# =========================================================
# 用户修改区：请在这里填写模型和运行配置
# EVAL_MAX_SAMPLES=64 用于 smoke；确认无误后改成 0 跑完整测试集
# =========================================================
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-/wzj/zijian/OSSM-HOI-main/checkpoints/hico-det_sync_vt_graph6/STIP_opensource/hico_run_graph6/best.pth}"
DETR_WEIGHTS="${DETR_WEIGHTS:-/share/share/HOI/pretrained_model/HICO/all/res101/checkpoint0105.pth}"
BLIP_PRETRAINED="${BLIP_PRETRAINED:-/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_large_retrieval_coco.pth}"
BLIP_VIT="${BLIP_VIT:-large}"
BACKBONE="${BACKBONE:-resnet101}"
GPU_IDS="${GPU_IDS:-4,5,6,7}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
EVAL_IMAGE_IDS_FILE="${EVAL_IMAGE_IDS_FILE:-}"
GRAPH_EVIDENCE_TOPK="${GRAPH_EVIDENCE_TOPK:-4}"
GRAPH_TEXT_TOPK="${GRAPH_TEXT_TOPK:-5}"
# =========================================================
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/evidence_trace_stip_final_${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_ROOT}/eval"
TRACE_DIR="${OUTPUT_ROOT}/traces"

BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
TRACE_EXPECTED_RECORDS="${TRACE_EXPECTED_RECORDS:-}"

DATA_PATH="${DATA_PATH:-/share/share/Datasets/hico_20160224_det}"
TEST_ANNOTATION="${TEST_ANNOTATION:-${DATA_PATH}/annotations/test_hico.json}"
BLIP_IMAGE_SIZE="${BLIP_IMAGE_SIZE:-384}"

GRAPH_EVIDENCE_TOPK="${GRAPH_EVIDENCE_TOPK:-4}"
GRAPH_TEXT_TOPK="${GRAPH_TEXT_TOPK:-5}"
GRAPH_NUM_LAYERS="${GRAPH_NUM_LAYERS:-2}"
GRAPH_K_OO="${GRAPH_K_OO:-3}"
GRAPH_TEXT_PRE_TOPK="${GRAPH_TEXT_PRE_TOPK:-20}"
TRACE_EACH_LAYER="${TRACE_EACH_LAYER:-False}"
TRACE_FLOAT_DTYPE="${TRACE_FLOAT_DTYPE:-float32}"
TRACE_INCLUDE_MATCH_LOGITS="${TRACE_INCLUDE_MATCH_LOGITS:-True}"
TRACE_INCLUDE_FEATURES="${TRACE_INCLUDE_FEATURES:-False}"
GRAPH_STORE_DENSE_DEBUG_TENSORS="${GRAPH_STORE_DENSE_DEBUG_TENSORS:-False}"
TRACE_OVERWRITE="${TRACE_OVERWRITE:-False}"
CHECK_CHECKPOINT_PROFILE="${CHECK_CHECKPOINT_PROFILE:-True}"
GRAPH_EXPORT_MECHANISM_VALIDATION_TRACE="${GRAPH_EXPORT_MECHANISM_VALIDATION_TRACE:-False}"
GRAPH_MECHANISM_VALIDATION_VARIANT="${GRAPH_MECHANISM_VALIDATION_VARIANT:-full}"

OO_PRIOR_PATH="${OO_PRIOR_PATH:-${SCRIPT_DIR}/oo_affordance_prior_80.npy}"
TEXT_GRAPH_DIR="${TEXT_GRAPH_DIR:-${SCRIPT_DIR}/hico_text_graph_blip256}"
TEXT_FEATURES_PATH="${TEXT_FEATURES_PATH:-${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_features_6x256.npy}"
TEXT_MASK_PATH="${TEXT_MASK_PATH:-${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_mask_6.npy}"
TEXT_HOI_IDS_PATH="${TEXT_HOI_IDS_PATH:-${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_hoi_ids.json}"
TEXT_OBJECT_TO_HOI_PATH="${TEXT_OBJECT_TO_HOI_PATH:-${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_object_to_hoi_indices.json}"
TEXT_FIELD_ORDER_PATH="${TEXT_FIELD_ORDER_PATH:-${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_field_order.json}"

required_paths=(
    "${DATA_PATH}"
    "${TEST_ANNOTATION}"
    "${DETR_WEIGHTS}"
    "${RESUME_CHECKPOINT}"
    "${BLIP_PRETRAINED}"
    "${OO_PRIOR_PATH}"
    "${TEXT_FEATURES_PATH}"
    "${TEXT_MASK_PATH}"
    "${TEXT_HOI_IDS_PATH}"
    "${TEXT_OBJECT_TO_HOI_PATH}"
    "${TEXT_FIELD_ORDER_PATH}"
)
if [[ -n "${EVAL_IMAGE_IDS_FILE}" ]]; then
    required_paths+=("${EVAL_IMAGE_IDS_FILE}")
fi
for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[Error] Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

if [[ -n "${EVAL_IMAGE_IDS_FILE}" && "${EVAL_MAX_SAMPLES}" -gt 0 ]]; then
    echo "[Error] EVAL_IMAGE_IDS_FILE and EVAL_MAX_SAMPLES cannot be used together." >&2
    exit 1
fi

if [[ -z "${TRACE_EXPECTED_RECORDS}" ]]; then
    if [[ -n "${EVAL_IMAGE_IDS_FILE}" ]]; then
        TRACE_EXPECTED_RECORDS="$(python -c 'import json,sys; x=json.load(open(sys.argv[1], encoding="utf-8")); x=x.get("image_ids", x) if isinstance(x, dict) else x; print(len(dict.fromkeys(map(int, x))))' "${EVAL_IMAGE_IDS_FILE}")"
    elif [[ "${EVAL_MAX_SAMPLES}" -gt 0 ]]; then
        TRACE_EXPECTED_RECORDS="${EVAL_MAX_SAMPLES}"
    else
        TRACE_EXPECTED_RECORDS="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "${TEST_ANNOTATION}")"
    fi
fi

if [[ "${BLIP_VIT}" == "large" ]]; then
    EXPECTED_BLIP_WIDTH=1024
else
    EXPECTED_BLIP_WIDTH=768
fi

if [[ "${CHECK_CHECKPOINT_PROFILE}" == "True" ]]; then
    python "${SCRIPT_DIR}/tools/evidence_vis/check_trace_checkpoint_profile.py" \
        --checkpoint "${RESUME_CHECKPOINT}" \
        --expected-evidence-topk "${GRAPH_EVIDENCE_TOPK}" \
        --expected-text-topk "${GRAPH_TEXT_TOPK}" \
        --expected-backbone "${BACKBONE}" \
        --expected-blip-width "${EXPECTED_BLIP_WIDTH}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
MASTER_PORT="${MASTER_PORT:-$((29000 + RANDOM % 1000))}"
mkdir -p "${OUTPUT_DIR}" "${TRACE_DIR}"

echo "========================================================="
echo "EvidenceHOI final STIP Trace export"
echo "Checkpoint       : ${RESUME_CHECKPOINT}"
echo "DETR / backbone  : ${DETR_WEIGHTS} / ${BACKBONE}"
echo "BLIP             : ${BLIP_VIT} / ${BLIP_PRETRAINED}"
echo "Checkpoint profile: M=${GRAPH_EVIDENCE_TOPK}, K=${GRAPH_TEXT_TOPK}, L=${GRAPH_NUM_LAYERS}, k_oo=${GRAPH_K_OO}"
echo "GPUs / batch     : ${GPU_IDS} / ${BATCH_SIZE} per GPU"
echo "Eval samples     : ${EVAL_MAX_SAMPLES} (0 means full test)"
echo "Exact image IDs  : ${EVAL_IMAGE_IDS_FILE:-<none>}"
echo "Trace dtype      : ${TRACE_FLOAT_DTYPE}"
echo "Each-layer Trace : ${TRACE_EACH_LAYER}"
echo "Trace features   : ${TRACE_INCLUDE_FEATURES}"
echo "Dense graph data : ${GRAPH_STORE_DENSE_DEBUG_TENSORS}"
echo "Mechanism Trace  : ${GRAPH_EXPORT_MECHANISM_VALIDATION_TRACE}"
echo "Mechanism variant: ${GRAPH_MECHANISM_VALIDATION_VARIANT}"
echo "Output           : ${OUTPUT_ROOT}"
echo "========================================================="

cd "${SCRIPT_DIR}"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_backend="c10d" \
    --master_port="${MASTER_PORT}" \
    STIP_main.py \
    --output_dir "${OUTPUT_DIR}" \
    --group_name "evidence_trace" \
    --run_name "stip_final_trace" \
    --stip_model_variant evidence_trace \
    --HOIDet \
    --no_aux_loss \
    --hoi_aux_loss \
    --dataset_file "hico-det" \
    --data_path "${DATA_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_max_samples "${EVAL_MAX_SAMPLES}" \
    --eval_image_ids_file "${EVAL_IMAGE_IDS_FILE}" \
    --resume "${RESUME_CHECKPOINT}" \
    --eval \
    --num_hoi_queries 32 \
    --backbone "${BACKBONE}" \
    --blip_pretrained "${BLIP_PRETRAINED}" \
    --blip_vit "${BLIP_VIT}" \
    --blip_image_size "${BLIP_IMAGE_SIZE}" \
    --pair_dedupe_iou_thr 0.55 \
    --pair_filter_threshold 0 \
    --use_visual_scene_graph \
    --graph_oo_mode text_hybrid \
    --graph_oo_lambda_geo 1.0 \
    --graph_oo_lambda_share 0.5 \
    --graph_oo_lambda_text 0.5 \
    --graph_oo_prior_path "${OO_PRIOR_PATH}" \
    --graph_oo_prior_normalize none \
    --graph_k_oo "${GRAPH_K_OO}" \
    --graph_enable_text_graph True \
    --graph_text_features_path "${TEXT_FEATURES_PATH}" \
    --graph_text_mask_path "${TEXT_MASK_PATH}" \
    --graph_text_hoi_ids_path "${TEXT_HOI_IDS_PATH}" \
    --graph_text_object_to_hoi_indices_path "${TEXT_OBJECT_TO_HOI_PATH}" \
    --graph_text_field_order_path "${TEXT_FIELD_ORDER_PATH}" \
    --graph_text_topk "${GRAPH_TEXT_TOPK}" \
    --graph_text_pre_topk "${GRAPH_TEXT_PRE_TOPK}" \
    --graph_text_temperature 0.07 \
    --graph_text_rerank_weight 0.5 \
    --graph_text_sync_mode layerwise \
    --graph_text_dense_injection True \
    --graph_text_final_pair_injection False \
    --graph_text_dynamic_retrieval False \
    --graph_text_sync_query_source base_pair \
    --graph_text_sync_layers_with_visual True \
    --graph_text_edge_inject_reverse True \
    --graph_text_node_scatter_reduce mean \
    --graph_text_prior_injection all \
    --graph_text_use_ho_edge_prior True \
    --graph_text_use_human_node_prior True \
    --graph_text_use_object_node_prior True \
    --graph_text_use_discriminative_prior True \
    --graph_text_use_scene_prior True \
    --graph_text_use_oo_prior True \
    --graph_enable_candidate_evidence True \
    --graph_evidence_topk "${GRAPH_EVIDENCE_TOPK}" \
    --graph_evidence_gate_init 0.0 \
    --graph_evidence_dropout 0.1 \
    --graph_enable_sync_evidence_matching True \
    --graph_sync_visual_gate_init 0.0 \
    --graph_sync_text_gate_init 0.0 \
    --graph_sync_anchor_gate_init 0.0 \
    --graph_sync_dropout 0.1 \
    --graph_enable_layerwise_sync_evidence_matching True \
    --graph_layerwise_sync_write_visual True \
    --graph_layerwise_sync_write_text True \
    --graph_layerwise_sync_final_matching False \
    --graph_layerwise_sync_disable_old_dense_injection True \
    --graph_layerwise_sync_write_reverse_edge False \
    --graph_layerwise_sync_detach_text_update False \
    --graph_num_layers "${GRAPH_NUM_LAYERS}" \
    --graph_num_heads 8 \
    --graph_store_dense_debug_tensors "${GRAPH_STORE_DENSE_DEBUG_TENSORS}" \
    --graph_enable_path_context \
    --graph_path_topk 3 \
    --graph_path_use_edge_scores \
    --graph_path_score_thresh 0.0 \
    --graph_node_score_thresh 0.05 \
    --graph_min_score_nodes 40 \
    --graph_export_evidence_trace True \
    --graph_export_trace_dir "${TRACE_DIR}" \
    --graph_export_trace_each_layer "${TRACE_EACH_LAYER}" \
    --graph_export_trace_to_cpu True \
    --graph_export_trace_float_dtype "${TRACE_FLOAT_DTYPE}" \
    --graph_export_trace_include_match_logits "${TRACE_INCLUDE_MATCH_LOGITS}" \
    --graph_export_trace_include_features "${TRACE_INCLUDE_FEATURES}" \
    --graph_export_trace_topk_actions 10 \
    --graph_export_trace_include_all_action_logits True \
    --graph_export_trace_match_ground_truth True \
    --graph_export_trace_overwrite "${TRACE_OVERWRITE}" \
    --graph_export_mechanism_validation_trace "${GRAPH_EXPORT_MECHANISM_VALIDATION_TRACE}" \
    --graph_mechanism_validation_variant "${GRAPH_MECHANISM_VALIDATION_VARIANT}" \
    --detr_weights "${DETR_WEIGHTS}" \
    --num_workers "${NUM_WORKERS}" \
    2>&1 | tee "${OUTPUT_ROOT}/eval_${RUN_TAG}.log"

python tools/evidence_vis/validate_evidence_trace.py \
    --trace-dir "${TRACE_DIR}" \
    --expected-records "${TRACE_EXPECTED_RECORDS}" \
    --require-graph

echo "Trace export complete: ${TRACE_DIR}"
