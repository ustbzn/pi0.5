#!/bin/bash
#
# 全量 val 评测脚本 — 对比 0429 state-only baseline 和 0509 qwen3-aligned 模型
#
# 用法:
#   bash script/eval_full_compare.sh
#
# 说明:
#   - eval_size=4000 覆盖全部 3900 个 val 样本
#   - 两个模型分别评测，结果输出到日志
#   - 使用单 GPU 推理，无需 torchrun

set -e

export TZ='UTC-8'
export WANDB_DISABLED=true
export LOG_DIR="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_mogo/logs"

cd /pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_mogo

GPU=0
EVAL_SIZE=4000

# ================================================================
# 模型 1: 0429 state-only baseline (acc@0.05 训练中报告 0.712)
# ================================================================
CONFIG_1="config/train/DrivePi0/0429_qwen3_ablation_state_only_noHisTraj_1frame_4gpu.yaml"
CKPT_1="logs/train/drive-pi0/2026-04-29_14-46_mogo_naviv2_7dim_ablation_state_only_noHisTraj_1frame_4gpu/checkpoint/step24920.pt"

echo "============================================================"
echo "Evaluating Model 1: 0429 state-only baseline"
echo "Config: ${CONFIG_1}"
echo "Checkpoint: ${CKPT_1}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python script/eval_full.py \
    --config "${CONFIG_1}" \
    --checkpoint "${CKPT_1}" \
    --eval_size ${EVAL_SIZE} \
    --batch_size 24 \
    --gpu 0 \
    2>&1 | tee logs/eval_0429_state_only_full.log

echo ""
echo "============================================================"
echo "Model 1 evaluation complete. Log: logs/eval_0429_state_only_full.log"
echo "============================================================"
echo ""

# ================================================================
# 模型 2: 0509 Qwen3 完全适配 (SiLU + Qwen3RMSNorm)
# ================================================================
CONFIG_2="config/train/DrivePi0/0509_qwen3_aligned_planD_hardcase_bucketedEval_4gpu.yaml"
CKPT_2="logs/train/drive-pi0/2026-05-09_15-12_mogo_naviv2_7dim_noHisTraj_1frame_qwen3_aligned_planD_hardcase_4gpu/checkpoint/step18690.pt"

echo "============================================================"
echo "Evaluating Model 2: 0509 Qwen3-aligned"
echo "Config: ${CONFIG_2}"
echo "Checkpoint: ${CKPT_2}"
echo "============================================================"

CUDA_VISIBLE_DEVICES=${GPU} python script/eval_full.py \
    --config "${CONFIG_2}" \
    --checkpoint "${CKPT_2}" \
    --eval_size ${EVAL_SIZE} \
    --batch_size 24 \
    --gpu 0 \
    2>&1 | tee logs/eval_0509_qwen3_aligned_full.log

echo ""
echo "============================================================"
echo "Model 2 evaluation complete. Log: logs/eval_0509_qwen3_aligned_full.log"
echo "============================================================"
echo ""
echo "Both evaluations complete. Compare results:"
echo "  grep 'acc thres' logs/eval_0429_state_only_full.log"
echo "  grep 'acc thres' logs/eval_0509_qwen3_aligned_full.log"
