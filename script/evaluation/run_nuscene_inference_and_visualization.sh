#!/bin/bash
set -eu
(set -o pipefail) 2>/dev/null && set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# You can either edit the defaults below or override them at runtime.
# If your checkpoint was trained with Qwen3-VL, set:
#   CONFIG_PATH=config/inference/DrivePi0/base_qwen3.yaml
# Example:
#   CONFIG_PATH=config/inference/DrivePi0/base_qwen3.yaml \
#   OUTPUT_DIR=/tmp/nuscene_infer \
#   bash script/evaluation/run_nuscene_inference_and_visualization.sh

GPU_ID="${GPU_ID:-0}"
CONFIG_PATH="${CONFIG_PATH:-config/inference/DrivePi0/base_qwen3.yaml}"
VAL_DATA_DIR="${VAL_DATA_DIR:-datasets/nuscene-0111/val}"
STATISTIC_PATH="${STATISTIC_PATH:-datasets/nuscene-0111/statistics.json}"
OUTPUT_DIR="${OUTPUT_DIR:-datasets/nuscene-0111/inference/run1}"
DUAL_DIR="${DUAL_DIR:-datasets/nuscene-0111/visualizations/run1/dual}"
VIDEO_DIR="${VIDEO_DIR:-datasets/nuscene-0111/visualizations/run1/video}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FPS="${FPS:-6}"

echo "Repository root : ${REPO_ROOT}"
echo "PYTHONPATH      : ${PYTHONPATH}"
echo "GPU_ID          : ${GPU_ID}"
echo "CONFIG_PATH     : ${CONFIG_PATH}"
echo "VAL_DATA_DIR    : ${VAL_DATA_DIR}"
echo "STATISTIC_PATH  : ${STATISTIC_PATH}"
echo "OUTPUT_DIR      : ${OUTPUT_DIR}"
echo "DUAL_DIR        : ${DUAL_DIR}"
echo "VIDEO_DIR       : ${VIDEO_DIR}"
echo "NUM_WORKERS     : ${NUM_WORKERS}"

mkdir -p "${OUTPUT_DIR}" "${DUAL_DIR}" "${VIDEO_DIR}"

echo
echo "[1/3] Running nuScenes batch inference ..."
CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/agent/drivepi0/inference_nuscene_test.py \
  --mode batch \
  --val_data_dir "${VAL_DATA_DIR}" \
  --config_path "${CONFIG_PATH}" \
  --statistic_path "${STATISTIC_PATH}" \
  --output_dir "${OUTPUT_DIR}"

echo
echo "[2/3] Computing metrics and generating dual-view visualizations ..."
python src/utils/analyzing_traj.py \
  --trajectory_dir "${OUTPUT_DIR}" \
  --pkl_dir "${VAL_DATA_DIR}" \
  --mode dual \
  --output_dir "${DUAL_DIR}" \
  --statistic_path "${STATISTIC_PATH}" \
  --num_workers "${NUM_WORKERS}"

echo
echo "[3/3] Converting dual-view images to videos ..."
python src/utils/merge_images.py \
  --dual-dir "${DUAL_DIR}" \
  --out-dir "${VIDEO_DIR}" \
  --fps "${FPS}"

echo
echo "nuScenes inference and visualization finished."
echo "Pred/GT trajectories: ${OUTPUT_DIR}"
echo "Dual-view images    : ${DUAL_DIR}"
echo "Videos              : ${VIDEO_DIR}"
