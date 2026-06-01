#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# B2D Action 开环评测：论文口径 Avg. L2
# ============================================================================
# 用法:
#   GPU_ID=0 bash script/evaluation/eval_b2d_action_avg_l2.sh state_only
#   GPU_ID=1 bash script/evaluation/eval_b2d_action_avg_l2.sh frozen_vlm
#   GPU_ID=2 bash script/evaluation/eval_b2d_action_avg_l2.sh both
#
# 说明:
#   1. eval.py 负责推理并保存反归一化后的 pred_trajectories.npy / gt_trajectories.npy。
#   2. analyzing_traj.py --mode error 负责计算论文口径 Avg. L2。
#   3. eval.py 日志中的 "l2 Loss" 是 MSE，不是论文 Avg. L2。
# ============================================================================

MODE="${1:-both}"
GPU_ID="${GPU_ID:-}"

if [[ -z "${GPU_ID}" ]]; then
  echo "请显式指定空闲 GPU，避免影响正在训练的任务，例如："
  echo "  GPU_ID=0 bash $0 state_only"
  exit 1
fi

case "${MODE}" in
  state_only|frozen_vlm|both) ;;
  *)
    echo "不支持的 MODE: ${MODE}"
    echo "可选: state_only | frozen_vlm | both"
    exit 1
    ;;
esac

PROJECT_DIR="/pcpt/pcpt/project/jiangjunyuan/code/VLA/Drive_pi"
DATASET_DIR="/pcpt/pcpt/data/VLA/MOGO/b2d_action_hisTraj"
STATISTICS_PATH="${DATASET_DIR}/b2d_statistics.json"
LOG_DIR="${PROJECT_DIR}/logs"

STATE_ONLY_CKPT="${PROJECT_DIR}/logs/train/drive-pi0/2026-05-13_17-44_fair_b2d_state_only_action_front_history_histraj/checkpoint/step3033.pt"
FROZEN_VLM_CKPT="${PROJECT_DIR}/logs/train/drive-pi0/2026-05-13_20-36_fair_b2d_action_front_history_histraj/checkpoint/step3033.pt"

export LOG_DIR
export HYDRA_FULL_ERROR=1
export WANDB_DISABLED=true

cd "${PROJECT_DIR}"

run_one() {
  local name="$1"
  local config_name="$2"
  local checkpoint_path="$3"
  local batch_size="$4"

  local eval_dir="${LOG_DIR}/eval/drive-pi0/${name}_step3033"
  local trajectory_dir="${eval_dir}/trajectories"
  local analysis_dir="${eval_dir}/analysis"

  echo "============================================================================"
  echo "开始评测: ${name}"
  echo "GPU_ID: ${GPU_ID}"
  echo "checkpoint: ${checkpoint_path}"
  echo "eval_dir: ${eval_dir}"
  echo "============================================================================"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" torchrun \
    script/run.py \
    --config-path=../config/eval/DrivePi0 \
    --config-name="${config_name}" \
    checkpoint_path="${checkpoint_path}" \
    log_dir="${eval_dir}" \
    device_batch_size="${batch_size}"

  echo "============================================================================"
  echo "计算论文口径 Avg. L2: ${name}"
  echo "trajectory_dir: ${trajectory_dir}"
  echo "============================================================================"

  mkdir -p "${analysis_dir}"

  python src/utils/analyzing_traj.py \
    --trajectory_dir "${trajectory_dir}" \
    --mode error \
    --statistic_path "${STATISTICS_PATH}" \
    --output_dir "${analysis_dir}" \
    | tee "${analysis_dir}/avg_l2_report.txt"

  echo "完成: ${name}"
  echo "报告: ${analysis_dir}/avg_l2_report.txt"
}

if [[ "${MODE}" == "state_only" || "${MODE}" == "both" ]]; then
  run_one \
    "fair_b2d_state_only_front_history_histraj" \
    "fair_b2d_state_only_front_history_histraj" \
    "${STATE_ONLY_CKPT}" \
    "32"
fi

if [[ "${MODE}" == "frozen_vlm" || "${MODE}" == "both" ]]; then
  run_one \
    "fair_b2d_front_history_histraj" \
    "fair_b2d_front_history_histraj" \
    "${FROZEN_VLM_CKPT}" \
    "16"
fi
