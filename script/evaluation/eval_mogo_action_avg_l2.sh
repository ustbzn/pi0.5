#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# MOGO Action 开环评测：真实坐标系 Avg. L2
# ============================================================================
#
# 默认用法:
#   bash script/evaluation/eval_mogo_action_avg_l2.sh
#
# 可选模式:
#   bash script/evaluation/eval_mogo_action_avg_l2.sh eval          # 默认：推理 + Avg. L2 分析
#   bash script/evaluation/eval_mogo_action_avg_l2.sh analyze_only  # 只分析已有轨迹，不重新推理
#
# 本脚本只做评测，不修改训练代码。评测链路如下:
#   1. 读取 CONFIG_PATH 对应的 0514 inference 配置。
#   2. 写入 CHECKPOINT_PATH、GPU_ID 等运行时字段，生成 RUN_DIR/config_eval.yaml。
#   3. 调用 inference_mogo_test.py 在 MOGO val 集上推理，保存反归一化轨迹:
#        ${TRAJECTORY_DIR}/{timestamp}_pred_trajectories.npy
#        ${TRAJECTORY_DIR}/{timestamp}_gt_trajectories.npy
#   4. 调用 analyzing_traj_mogo.py --mode error 计算真实坐标系 Avg. L2。
#
# 注意:
#   - GPU 和 checkpoint 默认写在下方“用户配置区”，不需要命令行传参。
#   - 当前默认 checkpoint 指向 0514 训练目录的 step18690.pt；如果你的目标
#     checkpoint 文件名不同，只需要改 CHECKPOINT_PATH 一行。
#   - CONFIG_PATH 来自 0514 训练配置生成的专用评测配置，不使用 LoRA。
#   - 如需临时覆盖任意变量，也可以用环境变量，例如:
#       GPU_ID=1 CHECKPOINT_PATH=/path/to/step.pt bash script/evaluation/eval_mogo_action_avg_l2.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ============================================================================
# 用户配置区：日常评测主要改这里
# ============================================================================

# 使用哪张 GPU。当前环境正在训练时，请改成空闲 GPU。
GPU_ID="${GPU_ID:-0}"

# 0514 训练产物 checkpoint。训练未保存到该 step 前，脚本会提前报错退出。
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_DIR}/logs/train/drive-pi0/2026-05-14_20-56_qwen3_hdmap_all_features_8gpu/checkpoint/step37400.pt}"

# 由 config/train/DrivePi0/0514_qwen3_hdmap_all_features_8gpu.yaml 提取出的评测配置。
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config/inference/DrivePi0/0514_qwen3_hdmap_all_features_8gpu.yaml}"

# MOGO naviv2 val 数据目录；inference_mogo_test.py 会遍历其中的 pkl。
VAL_DATA_DIR="${VAL_DATA_DIR:-/pcpt/pcpt/data/VLA/MOGO/mogo_dataset_worldsim/val}"

# 与 0514 训练一致的 statistics 文件，用于 state/trajectory 的归一化和反归一化。
# 注意：必须使用训练时同一份 statistics（训练数据集目录 mogo_dataset_navi_hdmap），
# 否则 fur_x/fur_y 的 minmax 不一致，反归一化会引入恒定坐标偏置。
STATISTIC_PATH="${STATISTIC_PATH:-/pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/statistics.json}"

# 评测输出根目录。每次运行会在下面创建一个带 checkpoint 名称的 RUN_DIR。
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_DIR}/logs/eval/drive-pi0/mogo_action_avg_l2}"

# 默认关闭 torch.compile，避免首次编译慢、额外占显存；需要时可改为 True。
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"

# 与 0514 训练一致，降低 Qwen3-VL 推理显存。
USE_BF16="${USE_BF16:-True}"

# 调试用：限制最多处理多少个 pkl。空字符串表示完整 val 集。
MAX_SAMPLES="${MAX_SAMPLES:-}"

# ============================================================================
# 脚本内部配置区：通常不需要改
# ============================================================================

MODE="${1:-eval}"
if [[ "${MODE}" != "eval" && "${MODE}" != "analyze_only" ]]; then
  echo "不支持的模式: ${MODE}"
  echo "可选模式: eval | analyze_only"
  exit 1
fi

cd "${PROJECT_DIR}"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PROJECT_DIR}/${CONFIG_PATH}"
fi
if [[ "${CHECKPOINT_PATH}" != /* ]]; then
  CHECKPOINT_PATH="${PROJECT_DIR}/${CHECKPOINT_PATH}"
fi
if [[ "${EVAL_ROOT}" != /* ]]; then
  EVAL_ROOT="${PROJECT_DIR}/${EVAL_ROOT}"
fi

CKPT_STEP="$(basename "${CHECKPOINT_PATH%.*}")"
CKPT_RUN="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"
RUN_NAME="${RUN_NAME:-${CKPT_RUN}_${CKPT_STEP}}"
RUN_DIR="${EVAL_ROOT}/${RUN_NAME}"
TRAJECTORY_DIR="${TRAJECTORY_DIR:-${RUN_DIR}/trajectories}"
ANALYSIS_DIR="${ANALYSIS_DIR:-${RUN_DIR}/analysis}"
TMP_CONFIG_PATH="${RUN_DIR}/config_eval.yaml"
INFERENCE_LOG="${RUN_DIR}/inference.log"
ANALYSIS_LOG="${ANALYSIS_DIR}/avg_l2_report.txt"

mkdir -p "${TRAJECTORY_DIR}" "${ANALYSIS_DIR}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "CONFIG_PATH 不存在: ${CONFIG_PATH}"
  exit 1
fi
if [[ "${MODE}" == "eval" && ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "CHECKPOINT_PATH 不存在: ${CHECKPOINT_PATH}"
  echo "如果 0514 训练还没保存到该 step，请先等待 checkpoint 生成，或修改脚本中的 CHECKPOINT_PATH。"
  exit 1
fi
if [[ "${MODE}" == "eval" && ! -d "${VAL_DATA_DIR}" ]]; then
  echo "VAL_DATA_DIR 不存在: ${VAL_DATA_DIR}"
  exit 1
fi
if [[ ! -f "${STATISTIC_PATH}" ]]; then
  echo "STATISTIC_PATH 不存在: ${STATISTIC_PATH}"
  exit 1
fi
if [[ "${MODE}" == "analyze_only" ]] && ! compgen -G "${TRAJECTORY_DIR}/*_gt_trajectories.npy" > /dev/null; then
  echo "analyze_only 模式下没有找到已有轨迹文件: ${TRAJECTORY_DIR}/*_gt_trajectories.npy"
  echo "请先运行 eval 模式，或通过 TRAJECTORY_DIR=/path/to/trajectories 指定已有轨迹目录。"
  exit 1
fi

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export CONFIG_PATH CHECKPOINT_PATH STATISTIC_PATH TMP_CONFIG_PATH USE_TORCH_COMPILE USE_BF16

# 从 inference config 中读取 norm_mode，确保 acc 评测落在与训练一致的归一化空间。
# 默认回退到 minmax，与当前 0514 训练配置保持一致。
NORM_MODE="${NORM_MODE:-$(python - <<PY
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load(os.environ["CONFIG_PATH"])
mode = cfg.get("norm_mode") or cfg.get("data", {}).get("norm_mode") or "minmax"
print(mode)
PY
)}"

echo "============================================================================"
echo "MOGO Avg. L2 评测配置"
echo "mode: ${MODE}"
echo "gpu_id: ${GPU_ID}"
echo "checkpoint: ${CHECKPOINT_PATH}"
echo "base config: ${CONFIG_PATH}"
echo "runtime config: ${TMP_CONFIG_PATH}"
echo "val data: ${VAL_DATA_DIR}"
echo "statistics: ${STATISTIC_PATH}"
echo "trajectory_dir: ${TRAJECTORY_DIR}"
echo "analysis_dir: ${ANALYSIS_DIR}"
echo "use_torch_compile: ${USE_TORCH_COMPILE}"
echo "use_bf16: ${USE_BF16}"
echo "norm_mode: ${NORM_MODE}"
echo "============================================================================"

if [[ "${MODE}" == "eval" ]]; then
  # 生成本次运行专用配置，避免直接改动 config/inference 下的基准配置。
  python - <<'PY'
import os
from omegaconf import OmegaConf

cfg = OmegaConf.load(os.environ["CONFIG_PATH"])
OmegaConf.update(cfg, "checkpoint_path", os.environ["CHECKPOINT_PATH"], merge=True)
OmegaConf.update(cfg, "gpu_id", 0, merge=True)
OmegaConf.update(cfg, "use_torch_compile", os.environ["USE_TORCH_COMPILE"].lower() == "true", merge=True)
OmegaConf.update(cfg, "use_bf16", os.environ["USE_BF16"].lower() == "true", merge=True)
OmegaConf.update(cfg, "data.statistics_path", os.environ["STATISTIC_PATH"], merge=True)
OmegaConf.save(config=cfg, f=os.environ["TMP_CONFIG_PATH"], resolve=False)
PY

  MAX_SAMPLES_ARGS=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    MAX_SAMPLES_ARGS=(--max_samples "${MAX_SAMPLES}")
  fi

  echo "开始推理并保存反归一化轨迹..."
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/agent/drivepi0/inference_mogo_test.py \
    --mode batch \
    --val_data_dir "${VAL_DATA_DIR}" \
    --config_path "${TMP_CONFIG_PATH}" \
    --statistic_path "${STATISTIC_PATH}" \
    --output_dir "${TRAJECTORY_DIR}" \
    "${MAX_SAMPLES_ARGS[@]}" \
    2>&1 | tee "${INFERENCE_LOG}"
else
  echo "跳过推理，仅分析已有轨迹。"
fi

echo "============================================================================"
echo "计算真实坐标系 Avg. L2..."
echo "============================================================================"

python src/utils/analyzing_traj_mogo.py \
  --trajectory_dir "${TRAJECTORY_DIR}" \
  --mode error \
  --statistic_path "${STATISTIC_PATH}" \
  --norm_mode "${NORM_MODE}" \
  --output_dir "${ANALYSIS_DIR}" \
  2>&1 | tee "${ANALYSIS_LOG}"

echo "完成 MOGO Avg. L2 评测"
echo "轨迹文件: ${TRAJECTORY_DIR}"
echo "报告文件: ${ANALYSIS_LOG}"
