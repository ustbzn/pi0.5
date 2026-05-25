#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export TZ="${TZ:-UTC-8}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export SWANLAB_API_KEY="${SWANLAB_API_KEY:-ZyfQt9lym45HXXQD2ynZR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_CONFIG="${BASE_CONFIG:-0421_qwen3_naviv2_9dim_smoothLoss_lora_vlm_1gpu_local.yaml}"
NUM_GPU=1

PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/drive_pi/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/root/miniconda3/envs/drive_pi/bin/torchrun}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN to the Python executable for the training environment." >&2
  exit 1
fi

if [[ ! -x "${TORCHRUN_BIN}" ]]; then
  echo "torchrun not found or not executable: ${TORCHRUN_BIN}" >&2
  echo "Set TORCHRUN_BIN to the torchrun executable for the training environment." >&2
  exit 1
fi

CONFIG_FILE="${REPO_DIR}/config/train/DrivePi0/${BASE_CONFIG}"
if [[ ! -f "${CONFIG_FILE}" ]]; then
  echo "Config file not found: ${CONFIG_FILE}" >&2
  exit 1
fi

find_free_port() {
  "${PYTHON_BIN}" -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()"
}

export MASTER_PORT="${MASTER_PORT:-$(find_free_port)}"
mkdir -p "${LOG_DIR}"

echo "REPO_DIR=${REPO_DIR}"
echo "BASE_CONFIG=${BASE_CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NUM_GPU=${NUM_GPU}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "LOG_DIR=${LOG_DIR}"
echo "SWANLAB_API_KEY=$(if [[ -n "${SWANLAB_API_KEY}" ]]; then echo '***set***'; else echo '***unset***'; fi)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
fi

cd "${REPO_DIR}"

HYDRA_FULL_ERROR=1 "${TORCHRUN_BIN}" \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPU}" \
  --rdzv_id="${RANDOM}" \
  --rdzv_backend=c10d \
  --max-restarts=0 \
  --standalone \
  script/run.py \
  --config-path=../config/train/DrivePi0 \
  --config-name="${BASE_CONFIG}"

echo "Training script finished."
echo "Logs are saved in ${LOG_DIR}"
