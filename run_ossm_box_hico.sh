#!/usr/bin/env bash
set -euo pipefail

# SAM-free OSSM ROI recipe (HICO-DET, scratch HOI training).
GPU_IDS="${GPU_IDS:-0}"
NPROC="${NPROC:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/hico-det_ossm_box}"
DATA_PATH="${DATA_PATH:-/share/share/Datasets/hico_20160224_det}"
OLD_ROOT="${OLD_ROOT:-/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3}"
BLIP_CHECKPOINT="${BLIP_CHECKPOINT:-${OLD_ROOT}/pth/blip/model_large_retrieval_coco.pth}"
TEXT_LIBRARY="${TEXT_LIBRARY:-${OLD_ROOT}/hico_L_embeds_256.npy}"
DETR_WEIGHTS="${DETR_WEIGHTS:-/share/share/HOI/pretrained_model/HICO/all/res101/checkpoint0105.pth}"
OBJECT_ACTION_PRIOR="${OBJECT_ACTION_PRIOR:-obj_act_prior.npy}"

for required in "${DATA_PATH}" "${BLIP_CHECKPOINT}" "${TEXT_LIBRARY}" "${DETR_WEIGHTS}" "${OBJECT_ACTION_PRIOR}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required OSSM-BOX asset: ${required}" >&2
        exit 2
    fi
done

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${OUTPUT_DIR}"
torchrun \
    --standalone \
    --nproc_per_node="${NPROC}" \
    tools/ossm_box_main.py \
    --output_dir "${OUTPUT_DIR}" \
    --group_name "OSSM_box" \
    --run_name "hico_blipl_roi_experts_scratch" \
    --HOIDet \
    --validate \
    --no_aux_loss \
    --hoi_aux_loss \
    --dataset_file "hico-det" \
    --data_path "${DATA_PATH}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --num_hoi_queries 32 \
    --backbone "resnet101" \
    --detr_weights "${DETR_WEIGHTS}" \
    --lr 4e-4 \
    --proposal_focal_loss_alpha 0.5 \
    --proposal_loss_coef 5 \
    --action_loss_coef 1 \
    --action_sampled_loss_coef 0 \
    --pair_filter_threshold 0.08 \
    --use_pair_score_weighting \
    --pair_dedupe_iou_thr 0.55 \
    --stip_model_variant "ossm_box" \
    --ossm_blip_pretrained "${BLIP_CHECKPOINT}" \
    --ossm_blip_vit "large" \
    --ossm_text_library "${TEXT_LIBRARY}" \
    --ossm_obj_act_prior "${OBJECT_ACTION_PRIOR}" \
    --ossm_region_image_size 384
