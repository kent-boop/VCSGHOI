#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

DATASET="${DATASET:?Set DATASET to hico-det or vcoco}"
MODE="${MODE:-path_reweight}"
SEED="${SEED:-3407}"
RUN_KIND="${RUN_KIND:-smoke}"
GPU_IDS="${GPU_IDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EPOCHS="${EPOCHS:-100}"
VET_TRAIN_MAX_SAMPLES="${VET_TRAIN_MAX_SAMPLES:-0}"
VET_VAL_MAX_SAMPLES="${VET_VAL_MAX_SAMPLES:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

case "${MODE}" in
  anchor_raw|anchor_svrm|endpoints|context_uniform|path_selection|path_reweight) ;;
  *) echo "Invalid MODE=${MODE}" >&2; exit 2 ;;
esac
case "${RUN_KIND}" in
  smoke|formal) ;;
  *) echo "Invalid RUN_KIND=${RUN_KIND}" >&2; exit 2 ;;
esac
if [[ "${RUN_KIND}" == "smoke" ]]; then
  EPOCHS="${SMOKE_EPOCHS:-1}"
  if [[ "${VET_TRAIN_MAX_SAMPLES}" == "0" ]]; then VET_TRAIN_MAX_SAMPLES=64; fi
  if [[ "${VET_VAL_MAX_SAMPLES}" == "0" ]]; then VET_VAL_MAX_SAMPLES=64; fi
fi
if [[ "${RUN_KIND}" == "formal" && -n "${RESUME_CHECKPOINT}" && "${VET_ALLOW_MATCHING_RESUME:-0}" != "1" ]]; then
  echo "Formal runs may not resume a full/different model checkpoint." >&2
  exit 2
fi

if [[ "${DATASET}" == "hico-det" ]]; then
  DATA_PATH="${DATA_PATH:-/share/share/Datasets/hico_20160224_det}"
  DETR_WEIGHTS="${DETR_WEIGHTS:-/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/detr_coco/checkpoint0120.pth}"
  TEXT_GRAPH_DIR="${TEXT_GRAPH_DIR:-${PROJECT_ROOT}/hico_text_graph_blip256}"
  STRICT_SHAPE=True
  PREFIX=hico
elif [[ "${DATASET}" == "vcoco" ]]; then
  DATA_PATH="${DATA_PATH:-/share/share/Datasets/coco2014/images}"
  DETR_WEIGHTS="${DETR_WEIGHTS:-/share/share/HOI/pretrained_model/VCOCO/train_test/checkpoint0140.pth}"
  TEXT_GRAPH_DIR="${TEXT_GRAPH_DIR:-${PROJECT_ROOT}/vcoco_text_graph_blip256}"
  STRICT_SHAPE=False
  PREFIX=vcoco
else
  echo "Invalid DATASET=${DATASET}" >&2
  exit 2
fi

TEXT_FEATURES_PATH="${TEXT_FEATURES_PATH:-${TEXT_GRAPH_DIR}/${PREFIX}_hoi_graph_blip256_text_features_6x256.npy}"
TEXT_MASK_PATH="${TEXT_MASK_PATH:-${TEXT_GRAPH_DIR}/${PREFIX}_hoi_graph_blip256_text_mask_6.npy}"
TEXT_HOI_IDS_PATH="${TEXT_HOI_IDS_PATH:-${TEXT_GRAPH_DIR}/${PREFIX}_hoi_graph_blip256_hoi_ids.json}"
TEXT_OBJECT_TO_HOI_PATH="${TEXT_OBJECT_TO_HOI_PATH:-${TEXT_GRAPH_DIR}/${PREFIX}_hoi_graph_blip256_object_to_hoi_indices.json}"
TEXT_FIELD_ORDER_PATH="${TEXT_FIELD_ORDER_PATH:-${TEXT_GRAPH_DIR}/${PREFIX}_hoi_graph_blip256_field_order.json}"
BLIP_WEIGHTS="${BLIP_WEIGHTS:-/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/pth/blip/model_base_retrieval_coco.pth}"

REQUIRED_PATHS=(
  "${DATA_PATH}"
  "${DETR_WEIGHTS}"
  "${BLIP_WEIGHTS}"
  "${TEXT_FEATURES_PATH}"
  "${TEXT_MASK_PATH}"
  "${TEXT_HOI_IDS_PATH}"
  "${TEXT_OBJECT_TO_HOI_PATH}"
  "${TEXT_FIELD_ORDER_PATH}"
)
for required_path in "${REQUIRED_PATHS[@]}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required VET ablation resource: ${required_path}" >&2
    exit 2
  fi
done

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS="${#GPU_ARRAY[@]}"
MASTER_PORT="${MASTER_PORT:-$((29000 + RANDOM % 1000))}"
OUTPUT_ROOT="${OUTPUT_ROOT:-checkpoints/vet_internal_ablation}"
OUTPUT_DIR="${OUTPUT_ROOT}/${DATASET}/${RUN_KIND}/${MODE}/seed_${SEED}"
RUN_NAME="${DATASET}_${MODE}_seed${SEED}_${RUN_KIND}"
mkdir -p "${OUTPUT_DIR}"

RESUME_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  RESUME_ARGS=(--resume "${RESUME_CHECKPOINT}")
fi
MATCHING_RESUME_ARGS=()
if [[ "${VET_ALLOW_MATCHING_RESUME:-0}" == "1" ]]; then
  MATCHING_RESUME_ARGS=(--vet_allow_matching_resume True)
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "VET internal ablation: dataset=${DATASET} mode=${MODE} seed=${SEED} kind=${RUN_KIND} GPUs=${GPU_IDS}"
if [[ "${LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  echo "dry-run validated: output=${OUTPUT_DIR} text_features=${TEXT_FEATURES_PATH}"
  exit 0
fi

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  STIP_main_vet_internal_ablation.py \
  --output_dir "${OUTPUT_DIR}" \
  --group_name "STIP_vet_internal" \
  --run_name "${RUN_NAME}" \
  --model_variant vet_internal_ablation \
  --graph_vet_ablation_mode "${MODE}" \
  --vet_run_kind "${RUN_KIND}" \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --HOIDet --validate --no_aux_loss --hoi_aux_loss \
  --dataset_file "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr 5e-4 \
  --weight_decay 1e-4 \
  --num_hoi_queries 32 \
  --backbone resnet50 \
  --pair_dedupe_iou_thr 0.55 \
  --pair_filter_threshold 0 \
  --use_edge_blip_fusion True \
  --use_blip_global_memory True \
  --use_vet True --use_lpt True --use_setm_ablation True \
  --setm_ablation_stage 5 \
  --disable_simple_lpt_residual_for_setm True \
  --vet_gate_init 0.0 --lpt_gate_init 0.0 \
  --setm_visual_gate_init 0.0 --setm_text_gate_init 0.0 --setm_anchor_gate_init 0.0 \
  --graph_sync_dropout 0.1 \
  --lpt_query_source vet_pair --lpt_context_source slots --lpt_use_selected_score_weight True \
  --vet_node_score_thresh 0.05 --vet_min_score_nodes 0 --vet_max_extra_nodes 100 \
  --graph_num_layers 2 --graph_num_heads 8 --graph_k_oo 3 \
  --graph_oo_mode hybrid --graph_oo_lambda_geo 1.0 --graph_oo_lambda_share 0.5 \
  --graph_oo_min_score 0.0 --graph_enable_oo_edges --graph_oo_fallback_to_spatial \
  --graph_evidence_topk 4 --graph_evidence_gate_init 0.0 --graph_evidence_dropout 0.1 \
  --graph_evidence_allow_node_only_context False --graph_vet_selection_temperature 1.0 \
  --graph_text_features_path "${TEXT_FEATURES_PATH}" \
  --graph_text_mask_path "${TEXT_MASK_PATH}" \
  --graph_text_hoi_ids_path "${TEXT_HOI_IDS_PATH}" \
  --graph_text_object_to_hoi_indices_path "${TEXT_OBJECT_TO_HOI_PATH}" \
  --graph_text_field_order_path "${TEXT_FIELD_ORDER_PATH}" \
  --graph_text_strict_shape "${STRICT_SHAPE}" \
  --graph_text_topk 5 --graph_text_pre_topk 20 \
  --graph_text_temperature 0.07 --graph_text_rerank_weight 0.5 \
  --graph_text_use_discriminative_rerank True \
  --graph_text_num_layers 1 --graph_text_num_heads 8 --graph_text_dropout 0.1 \
  --graph_text_use_scene_token True --graph_text_use_oo_context_token True \
  --graph_text_prototype_slot_fallback_to_raw True \
  --graph_text_dynamic_retrieval False \
  --graph_layerwise_sync_write_visual True \
  --graph_layerwise_sync_write_text True \
  --graph_layerwise_sync_disable_old_dense_injection True \
  --graph_layerwise_sync_final_matching False \
  --graph_layerwise_sync_write_reverse_edge False \
  --graph_layerwise_sync_detach_text_update False \
  --vet_train_max_samples "${VET_TRAIN_MAX_SAMPLES}" \
  --vet_val_max_samples "${VET_VAL_MAX_SAMPLES}" \
  --detr_weights "${DETR_WEIGHTS}" \
  "${RESUME_ARGS[@]}" "${MATCHING_RESUME_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}/launcher.log"
