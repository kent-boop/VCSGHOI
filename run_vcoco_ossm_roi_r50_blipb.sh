#!/usr/bin/env bash
set -euo pipefail

# V-COCO training for the SAM-free OSSM ROI model with R50 + BLIP-Base.
# Dataset, detector, GPU, batch-size and optimizer defaults follow run2.sh.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-SPLHOI}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/share/apps/anaconda3/envs/${CONDA_ENV_NAME}}"
PYTHON_BIN="${CONDA_ENV_PREFIX}/bin/python"
TORCHRUN_BIN="${CONDA_ENV_PREFIX}/bin/torchrun"

GPU_IDS="${GPU_IDS:-0,2,3,4,7}"
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
NPROC="${NPROC:-${#GPU_ARRAY[@]}}"
BATCH_SIZE="${BATCH_SIZE:-22}"
NUM_WORKERS="${NUM_WORKERS:-4}"

DATA_PATH="${DATA_PATH:-/share/share/Datasets/coco2014/images}"
DETR_WEIGHTS="${DETR_WEIGHTS:-/share/share/HOI/pretrained_model/VCOCO/train_test/checkpoint0140.pth}"
OLD_MODEL_ROOT="${OLD_MODEL_ROOT:-/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3}"
BLIP_CHECKPOINT="${BLIP_CHECKPOINT:-${OLD_MODEL_ROOT}/pth/blip/model_base_retrieval_coco.pth}"
TEXT_LIBRARY="${TEXT_LIBRARY:-${PROJECT_ROOT}/vcoco_embeds_256.npy}"
OBJECT_ACTION_PRIOR="${OBJECT_ACTION_PRIOR:-${PROJECT_ROOT}/vcoco_obj_act_prior.npy}"

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/vcoco_NF_roi_r50_blipb}"
GROUP_NAME="${GROUP_NAME:-STIP_opensource}"
RUN_NAME="${RUN_NAME:-vcoco_ossm_roi_r50_blipb_scratch}"
# Leave empty for a fresh R50+Base experiment; set an absolute path to resume.
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((29000 + RANDOM % 1000))}"

for executable in "${PYTHON_BIN}" "${TORCHRUN_BIN}"; do
    if [[ ! -x "${executable}" ]]; then
        echo "Missing executable in Conda environment ${CONDA_ENV_NAME}: ${executable}" >&2
        exit 2
    fi
done

if [[ "${NPROC}" -lt 1 || "${NPROC}" -gt "${#GPU_ARRAY[@]}" ]]; then
    echo "NPROC=${NPROC} must be between 1 and the number of visible GPUs (${#GPU_ARRAY[@]})." >&2
    exit 2
fi

for required in \
    "${DATA_PATH}" \
    "${DETR_WEIGHTS}" \
    "${BLIP_CHECKPOINT}" \
    "${TEXT_LIBRARY}" \
    "${OBJECT_ACTION_PRIOR}"; do
    if [[ ! -e "${required}" ]]; then
        echo "Missing required training asset: ${required}" >&2
        exit 2
    fi
done

if [[ -n "${RESUME_CHECKPOINT}" && ! -f "${RESUME_CHECKPOINT}" ]]; then
    echo "Resume checkpoint does not exist: ${RESUME_CHECKPOINT}" >&2
    echo 'Set RESUME_CHECKPOINT="" to start from scratch.' >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-6}"
export PATH="${CONDA_ENV_PREFIX}/bin:${PATH}"
export CONDA_PREFIX="${CONDA_ENV_PREFIX}"
export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Preserve the old launcher's cluster defaults, but leave every value
# overridable. SPLHOI uses PyTorch 2.4 and supports expandable_segments.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ALGO="${NCCL_ALGO:-Ring}"
export NCCL_NSOCKS_PERTHREAD="${NCCL_NSOCKS_PERTHREAD:-4}"
export NCCL_SOCKET_NTHREADS="${NCCL_SOCKET_NTHREADS:-2}"
export NCCL_IGNORE_CPU_AFFINITY="${NCCL_IGNORE_CPU_AFFINITY:-1}"
export NCCL_PROTO="${NCCL_PROTO:-LL}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

mkdir -p "${OUTPUT_DIR}"

LAUNCH_COMMAND=(
    "${TORCHRUN_BIN}"
    --nnodes=1
    --nproc_per_node="${NPROC}"
    --rdzv_backend=c10d
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
    tools/ossm_box_main.py
    --output_dir "${OUTPUT_DIR}"
    --group_name "${GROUP_NAME}"
    --run_name "${RUN_NAME}"
    --HOIDet
    --validate
    --no_aux_loss
    --hoi_aux_loss
    --dataset_file vcoco
    --data_path "${DATA_PATH}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --epochs 100
    --reduce_lr_on_plateau_patience 2
    --lr 5e-4
    --proposal_focal_loss_alpha 0.75
    --proposal_focal_loss_gamma 2
    --action_focal_loss_alpha 0.5
    --action_focal_loss_gamma 2
    --proposal_loss_coef 1
    --action_loss_coef 1
    --action_sampled_loss_coef 1
    --pair_filter_threshold 0
    --pair_dedupe_iou_thr 0.55
    --num_hoi_queries 32
    --backbone resnet50
    --detr_weights "${DETR_WEIGHTS}"
    --stip_model_variant ossm_box
    --ossm_blip_pretrained "${BLIP_CHECKPOINT}"
    --ossm_blip_vit base
    --ossm_text_library "${TEXT_LIBRARY}"
    --ossm_obj_act_prior "${OBJECT_ACTION_PRIOR}"
    --ossm_region_image_size 384
)

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    LAUNCH_COMMAND+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "========================================================="
echo "V-COCO OSSM ROI training (ResNet-50 + BLIP-Base)"
echo "Conda environment: ${CONDA_ENV_NAME} / ${CONDA_ENV_PREFIX}"
echo "GPUs: ${GPU_IDS} (processes: ${NPROC})"
echo "Batch size: ${BATCH_SIZE} per GPU / $((BATCH_SIZE * NPROC)) total"
echo "DETR: ResNet-50 / ${DETR_WEIGHTS}"
echo "BLIP: Base / ${BLIP_CHECKPOINT}"
echo "Text library: ${TEXT_LIBRARY}"
echo "Object-action prior: ${OBJECT_ACTION_PRIOR}"
echo "Output: ${OUTPUT_DIR}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    echo "Resume: ${RESUME_CHECKPOINT}"
else
    echo "Resume: disabled (fresh training)"
fi
echo "========================================================="

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${LAUNCH_COMMAND[@]}"
    printf '\n'
    exit 0
fi

"${LAUNCH_COMMAND[@]}" 2>&1 | tee "${OUTPUT_DIR}/train_log_$(date +%Y%m%d_%H%M%S).txt"
