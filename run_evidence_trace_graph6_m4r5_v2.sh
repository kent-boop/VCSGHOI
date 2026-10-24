#!/bin/bash

set -euo pipefail

# HICO-DET full-test evidence trace export for the Graph6 M=4 / R=5 model.
#
# Before running:
#   1. conda activate SPLHOI
#   2. switch src/models/detr.py to BLIP Large (vit='large')
#   3. keep --stip_model_variant evidence_trace in this script
#
# Optional overrides:
#   GPU_IDS=0,1 BATCH_SIZE=2 OUTPUT_ROOT=/path/to/output bash "$0"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-22}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-0}"
TRACE_EXPECTED_RECORDS="${TRACE_EXPECTED_RECORDS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/evidencehoi_graph6_m4r5_${RUN_TAG}}"
OUTPUT_DIR="${OUTPUT_ROOT}/eval"
TRACE_DIR="${OUTPUT_ROOT}/traces"

DATA_PATH="${DATA_PATH:-/share/share/Datasets/hico_20160224_det}"
TEST_ANNOTATION="${DATA_PATH}/annotations/test_hico.json"
DETR_WEIGHTS="${DETR_WEIGHTS:-/share/share/HOI/pretrained_model/HICO/all/res101/checkpoint0105.pth}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${SCRIPT_DIR}/checkpoints/hico-det_sync_vt_graph6/STIP_opensource/hico_run_graph6/best.pth}"
BLIP_LARGE_WEIGHTS="${BLIP_LARGE_WEIGHTS:-/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_large_retrieval_coco.pth}"

OO_PRIOR_PATH="${SCRIPT_DIR}/oo_affordance_prior_80.npy"
TEXT_GRAPH_DIR="${SCRIPT_DIR}/hico_text_graph_blip256"
TEXT_FEATURES_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_features_6x256.npy"
TEXT_MASK_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_text_mask_6.npy"
TEXT_HOI_IDS_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_hoi_ids.json"
TEXT_OBJECT_TO_HOI_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_object_to_hoi_indices.json"
TEXT_FIELD_ORDER_PATH="${TEXT_GRAPH_DIR}/hico_hoi_graph_blip256_field_order.json"

if [[ "${CONDA_DEFAULT_ENV:-}" != "SPLHOI" ]]; then
    echo "[Warning] Expected conda environment SPLHOI; current environment: ${CONDA_DEFAULT_ENV:-none}" >&2
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

required_paths=(
    "${TEST_ANNOTATION}"
    "${DATA_PATH}"
    "${DETR_WEIGHTS}"
    "${RESUME_CHECKPOINT}"
    "${BLIP_LARGE_WEIGHTS}"
    "${OO_PRIOR_PATH}"
    "${TEXT_FEATURES_PATH}"
    "${TEXT_MASK_PATH}"
    "${TEXT_HOI_IDS_PATH}"
    "${TEXT_OBJECT_TO_HOI_PATH}"
    "${TEXT_FIELD_ORDER_PATH}"
)

for required_path in "${required_paths[@]}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[Error] Required path does not exist: ${required_path}" >&2
        exit 1
    fi
done

if [[ -z "${TRACE_EXPECTED_RECORDS}" ]]; then
    if [[ "${EVAL_MAX_SAMPLES}" -gt 0 ]]; then
        TRACE_EXPECTED_RECORDS="${EVAL_MAX_SAMPLES}"
    else
        TRACE_EXPECTED_RECORDS="$(python -c 'import json, sys; print(len(json.load(open(sys.argv[1], "r", encoding="utf-8"))))' "${TEST_ANNOTATION}")"
    fi
fi

if ! rg -q "^[[:space:]]*blip = blip_itm\\(pretrained=.*model_large_retrieval_coco\\.pth.*vit='large'\\)" "${SCRIPT_DIR}/src/models/detr.py"; then
    echo "[Error] src/models/detr.py is not configured with the active BLIP Large line." >&2
    echo "        Change the active blip_itm call to model_large_retrieval_coco.pth and vit='large'." >&2
    exit 1
fi

python "${SCRIPT_DIR}/tools/evidence_vis/check_trace_checkpoint_profile.py" \
    --checkpoint "${RESUME_CHECKPOINT}" \
    --expected-evidence-topk 4 \
    --expected-text-topk 5 \
    --expected-backbone resnet101 \
    --expected-blip-width 1024

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_NSOCKS_PERTRACK="${NCCL_NSOCKS_PERTRACK:-4}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-2}"
export NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
export NCCL_PROTO="${NCCL_PROTO:-LL}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTHONUNBUFFERED=1
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
MASTER_PORT="${MASTER_PORT:-$((29000 + RANDOM % 1000))}"

mkdir -p "${OUTPUT_DIR}" "${TRACE_DIR}"

echo "========================================================="
echo "EvidenceHOI Graph6 M=4/R=5 HICO-DET trace evaluation"
echo "Conda environment : ${CONDA_DEFAULT_ENV:-none}"
echo "Visible GPUs      : ${GPU_IDS}"
echo "GPU count         : ${NUM_GPUS}"
echo "Batch/GPU         : ${BATCH_SIZE}"
echo "Total batch       : $((BATCH_SIZE * NUM_GPUS))"
echo "Backbone          : ResNet-101"
echo "BLIP              : Large (verify src/models/detr.py before launch)"
echo "Checkpoint        : ${RESUME_CHECKPOINT}"
echo "Trace directory   : ${TRACE_DIR}"
echo "Output directory  : ${OUTPUT_DIR}"
echo "Eval sample limit : ${EVAL_MAX_SAMPLES} (0 means full test split)"
echo "Expected records  : ${TRACE_EXPECTED_RECORDS}"
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
    --run_name "hico_graph6_m4r5" \
    --stip_model_variant evidence_trace \
    --HOIDet \
    --no_aux_loss \
    --hoi_aux_loss \
    --dataset_file "hico-det" \
    --data_path "${DATA_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_max_samples "${EVAL_MAX_SAMPLES}" \
    --resume "${RESUME_CHECKPOINT}" \
    --eval \
    --num_hoi_queries 32 \
    --backbone "resnet101" \
    --pair_dedupe_iou_thr 0.55 \
    --pair_filter_threshold 0 \
    --use_visual_scene_graph \
    --graph_debug \
    --graph_oo_mode text_hybrid \
    --graph_oo_lambda_geo 1.0 \
    --graph_oo_lambda_share 0.5 \
    --graph_oo_lambda_text 0.5 \
    --graph_oo_prior_path "${OO_PRIOR_PATH}" \
    --graph_oo_prior_normalize none \
    --graph_enable_text_graph True \
    --graph_text_features_path "${TEXT_FEATURES_PATH}" \
    --graph_text_mask_path "${TEXT_MASK_PATH}" \
    --graph_text_hoi_ids_path "${TEXT_HOI_IDS_PATH}" \
    --graph_text_object_to_hoi_indices_path "${TEXT_OBJECT_TO_HOI_PATH}" \
    --graph_text_field_order_path "${TEXT_FIELD_ORDER_PATH}" \
    --graph_text_topk 5 \
    --graph_text_pre_topk 20 \
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
    --graph_text_gate_init 0.0 \
    --graph_text_edge_prior_gate_init 0.0 \
    --graph_text_node_prior_gate_init 0.0 \
    --graph_text_scene_prior_gate_init 0.0 \
    --graph_text_oo_prior_gate_init 0.0 \
    --graph_enable_candidate_evidence True \
    --graph_evidence_topk 4 \
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
    --graph_layerwise_sync_debug False \
    --graph_num_layers 2 \
    --graph_num_heads 8 \
    --graph_k_oo 3 \
    --graph_store_dense_debug_tensors False \
    --graph_enable_path_context \
    --graph_path_topk 3 \
    --graph_path_dropout 0.1 \
    --graph_path_gate_init 0.0 \
    --graph_path_use_edge_scores \
    --graph_path_score_thresh 0.0 \
    --graph_node_score_thresh 0.05 \
    --graph_min_score_nodes 40 \
    --graph_debug_interval 20 \
    --graph_debug_max_prints 20 \
    --graph_export_evidence_trace True \
    --graph_export_trace_dir "${TRACE_DIR}" \
    --graph_export_trace_each_layer False \
    --graph_export_trace_to_cpu True \
    --graph_export_trace_float_dtype float16 \
    --graph_export_trace_include_match_logits False \
    --graph_export_trace_include_features False \
    --graph_export_trace_topk_actions 10 \
    --graph_export_trace_include_all_action_logits True \
    --graph_export_trace_match_ground_truth True \
    --graph_export_trace_overwrite False \
    --detr_weights "${DETR_WEIGHTS}" \
    --num_workers "${NUM_WORKERS}" \
    2>&1 | tee "${OUTPUT_ROOT}/eval_${RUN_TAG}.log"

VALIDATION_ARGS=(
    --trace-dir "${TRACE_DIR}"
    --require-graph
)
if [[ "${TRACE_EXPECTED_RECORDS}" -gt 0 ]]; then
    VALIDATION_ARGS+=(--expected-records "${TRACE_EXPECTED_RECORDS}")
fi

python tools/evidence_vis/validate_evidence_trace.py "${VALIDATION_ARGS[@]}"

