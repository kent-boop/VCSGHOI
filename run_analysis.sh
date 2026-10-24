#!/bin/bash
# =========================================================
#  HOI 性能瓶颈分析脚本 (支持多卡)
#  
#  分析 ResNet101 + BLIP + SAM 在 HICO-DET 上的性能瓶颈
#  使用已训练好的 checkpoint 进行推理分析
#
#  用法:
#    bash run_analysis.sh              # 运行全部分析 (所有可用GPU)
#    bash run_analysis.sh 3,4,5        # 只运行 Phase 3,4,5
#
#  Phase 说明:
#    Phase 1: 推理 (收集预测结果)
#    Phase 2: 标准 mAP 计算
#    Phase 3: 目标检测质量分析
#    Phase 4: 逐动作(Verb) AP 分析
#    Phase 5: 逐物体(Object) AP + 错误类型分析
# =========================================================

# Conda 环境
CONDA_ENV="SPLHOI"

# 路径配置
DATA_PATH="/share/share/Datasets/hico_20160224_det"
# DETR_WEIGHTS="/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/detr_coco/checkpoint0120.pth"
DETR_WEIGHTS="/share/share/HOI/pretrained_model/HICO/all/res101/checkpoint0105.pth"
MODEL_CHECKPOINT="/wzj/zijian/SAM-BLIP-HOI-FUSE/SAM-BLIP-HOI3/checkpoints/hico-det_101_vitl/STIP_opensource/hico_run1/best.pth"
OUTPUT_DIR="analysis_results"

# GPU 配置
GPU_ID="0,1,5,7"
NUM_GPUS=4
BATCH_SIZE=20
NUM_WORKERS=4

# Phase 选择 (从命令行参数读取，默认 all)
PHASES=${1:-all}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONUNBUFFERED=1

echo "========================================================="
echo "HOI 性能瓶颈分析 (多卡模式)"
echo "Backbone: ResNet101"
echo "Model Checkpoint: ${MODEL_CHECKPOINT}"
echo "GPUs: ${GPU_ID} (${NUM_GPUS} cards)"
echo "Phases: ${PHASES}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================================="

mkdir -p "${OUTPUT_DIR}"

conda run --no-capture-output -n ${CONDA_ENV} \
    torchrun --nproc_per_node=${NUM_GPUS} \
    analyze_hoi_bottleneck.py \
    --dataset_file "hico-det" \
    --backbone "resnet101" \
    --data_path "${DATA_PATH}" \
    --resume "${MODEL_CHECKPOINT}" \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --num_hoi_queries 32 \
    --HOIDet \
    --eval \
    --no_aux_loss \
    --hoi_aux_loss \
    --analysis_output_dir "${OUTPUT_DIR}" \
    --analysis_phases "${PHASES}" \
2>&1 | tee "${OUTPUT_DIR}/analysis_log_$(date +%Y%m%d_%H%M%S).txt"

echo ""
echo "✅ 分析完成！结果保存在: ${OUTPUT_DIR}/"
echo "   - bottleneck_analysis.json  (详细数据)"
echo "   - per_verb_ap.png           (逐动作 AP)"
echo "   - per_object_ap.png         (逐物体 AP)"
echo "   - error_distribution.png    (错误类型分布)"
echo "   - detection_recall_per_class.png (检测 recall)"
