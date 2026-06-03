#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# MOGO Avg. L2 AB 诊断评测：定位"自车不主动启动"问题的根因
# ============================================================================
#
# 目的：
#   在不重训的前提下，回答两个问题：
#     1. his_trajectory 在 proprio 里是否是"未来 ≈ 历史外推"的捷径？
#     2. VLM 在 action expert 里到底贡献了多少？
#
# 实验：
#   baseline       : 当前 ckpt 的默认行为，做对照
#   zero_his_traj  : 推理时把 his_trajectory 强制置零，其它历史信号保留
#   mask_vlm       : 推理时强制 mask_vlm_for_action=True，VLM 不被 attend
#   ablate_both    : 同时启用上面两个开关
#
# 用法：
#   bash script/evaluation/eval_mogo_ab_diagnostic.sh                   # 跑全部 4 组
#   bash script/evaluation/eval_mogo_ab_diagnostic.sh baseline          # 只跑指定组
#   EXP_LIST="baseline zero_his_traj" bash script/evaluation/eval_mogo_ab_diagnostic.sh
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ============================================================================
# 用户配置区：与 eval_mogo_action_avg_l2.sh 保持一致
# ============================================================================
GPU_ID="${GPU_ID:-0}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_DIR}/logs/train/drive-pi0/2026-05-14_20-56_qwen3_hdmap_all_features_8gpu/checkpoint/step37400.pt}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_DIR}/config/inference/DrivePi0/0514_qwen3_hdmap_all_features_8gpu.yaml}"
VAL_DATA_DIR="${VAL_DATA_DIR:-/pcpt/pcpt/data/VLA/MOGO/mogo_dataset_worldsim/val}"
STATISTIC_PATH="${STATISTIC_PATH:-/pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/statistics.json}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_DIR}/logs/eval/drive-pi0/mogo_action_avg_l2_ab}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-False}"
USE_BF16="${USE_BF16:-True}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# 默认跑全部 4 组；命令行第一个参数若为某个组名，则只跑那一组
ALL_EXP="baseline zero_his_traj mask_vlm ablate_both"
if [[ $# -ge 1 ]]; then
  EXP_LIST="$1"
elif [[ -n "${EXP_LIST:-}" ]]; then
  EXP_LIST="${EXP_LIST}"
else
  EXP_LIST="${ALL_EXP}"
fi

# ============================================================================
# 路径与前置检查
# ============================================================================
cd "${PROJECT_DIR}"

[[ "${CONFIG_PATH}" != /* ]]      && CONFIG_PATH="${PROJECT_DIR}/${CONFIG_PATH}"
[[ "${CHECKPOINT_PATH}" != /* ]]  && CHECKPOINT_PATH="${PROJECT_DIR}/${CHECKPOINT_PATH}"
[[ "${EVAL_ROOT}" != /* ]]        && EVAL_ROOT="${PROJECT_DIR}/${EVAL_ROOT}"

CKPT_STEP="$(basename "${CHECKPOINT_PATH%.*}")"
CKPT_RUN="$(basename "$(dirname "$(dirname "${CHECKPOINT_PATH}")")")"

[[ ! -f "${CONFIG_PATH}"     ]] && { echo "CONFIG_PATH 不存在: ${CONFIG_PATH}"; exit 1; }
[[ ! -f "${CHECKPOINT_PATH}" ]] && { echo "CHECKPOINT_PATH 不存在: ${CHECKPOINT_PATH}"; exit 1; }
[[ ! -d "${VAL_DATA_DIR}"    ]] && { echo "VAL_DATA_DIR 不存在: ${VAL_DATA_DIR}"; exit 1; }
[[ ! -f "${STATISTIC_PATH}"  ]] && { echo "STATISTIC_PATH 不存在: ${STATISTIC_PATH}"; exit 1; }

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export CONFIG_PATH CHECKPOINT_PATH STATISTIC_PATH USE_TORCH_COMPILE USE_BF16

NORM_MODE="${NORM_MODE:-$(python - <<PY
from omegaconf import OmegaConf
import os
cfg = OmegaConf.load(os.environ["CONFIG_PATH"])
mode = cfg.get("norm_mode") or cfg.get("data", {}).get("norm_mode") or "minmax"
print(mode)
PY
)}"

# ============================================================================
# 单组实验执行
# ============================================================================
run_one_exp() {
  local exp_name="$1"
  local extra_args=()
  case "${exp_name}" in
    baseline)        ;;
    zero_his_traj)   extra_args+=(--zero_his_traj) ;;
    mask_vlm)        extra_args+=(--mask_vlm) ;;
    ablate_both)     extra_args+=(--zero_his_traj --mask_vlm) ;;
    *) echo "未知实验组: ${exp_name}"; return 1 ;;
  esac

  local run_dir="${EVAL_ROOT}/${CKPT_RUN}_${CKPT_STEP}/${exp_name}"
  local trajectory_dir="${run_dir}/trajectories"
  local analysis_dir="${run_dir}/analysis"
  local tmp_config="${run_dir}/config_eval.yaml"
  local inference_log="${run_dir}/inference.log"
  local analysis_log="${analysis_dir}/avg_l2_report.txt"

  mkdir -p "${trajectory_dir}" "${analysis_dir}"

  echo "============================================================================"
  echo "[${exp_name}] 写入运行配置: ${tmp_config}"
  TMP_CONFIG_PATH="${tmp_config}" python - <<'PY'
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

  local max_samples_args=()
  if [[ -n "${MAX_SAMPLES}" ]]; then
    max_samples_args=(--max_samples "${MAX_SAMPLES}")
  fi

  echo "[${exp_name}] 开始推理。诊断开关: ${extra_args[*]:-<none>}"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/agent/drivepi0/inference_mogo_test.py \
    --mode batch \
    --val_data_dir "${VAL_DATA_DIR}" \
    --config_path "${tmp_config}" \
    --statistic_path "${STATISTIC_PATH}" \
    --output_dir "${trajectory_dir}" \
    "${extra_args[@]}" \
    "${max_samples_args[@]}" \
    2>&1 | tee "${inference_log}"

  echo "[${exp_name}] 计算 Avg. L2..."
  python src/utils/analyzing_traj_mogo.py \
    --trajectory_dir "${trajectory_dir}" \
    --mode error \
    --statistic_path "${STATISTIC_PATH}" \
    --norm_mode "${NORM_MODE}" \
    --output_dir "${analysis_dir}" \
    2>&1 | tee "${analysis_log}"

  echo "[${exp_name}] 完成 → ${analysis_log}"
}

# ============================================================================
# 批量执行 + 末尾对比汇总
# ============================================================================
echo "============================================================================"
echo "MOGO AB 诊断评测"
echo "checkpoint: ${CHECKPOINT_PATH}"
echo "experiments: ${EXP_LIST}"
echo "eval root:  ${EVAL_ROOT}/${CKPT_RUN}_${CKPT_STEP}"
echo "============================================================================"

for exp in ${EXP_LIST}; do
  run_one_exp "${exp}"
done

echo "============================================================================"
echo "AB 汇总（各组报告路径）"
echo "============================================================================"
for exp in ${EXP_LIST}; do
  rep="${EVAL_ROOT}/${CKPT_RUN}_${CKPT_STEP}/${exp}/analysis/avg_l2_report.txt"
  echo "  ${exp}: ${rep}"
done

# 提取并对比四组的关键指标
python - <<'PY' "${EVAL_ROOT}/${CKPT_RUN}_${CKPT_STEP}" ${EXP_LIST}
import os, re, sys
root, *exps = sys.argv[1], *sys.argv[2:]
print()
print("=" * 78)
print(f"{'experiment':<18} {'L1':>8} {'L2(MSE)':>10} {'acc@0.05':>10} {'acc@0.1':>10} {'final_t5':>10}")
print("-" * 78)
patterns = {
    "L1":       re.compile(r"真实坐标 L1 Loss:\s*([\d.]+)"),
    "L2":       re.compile(r"真实坐标 L2 Loss \(MSE\):\s*([\d.]+)"),
    "acc05":    re.compile(r"acc thres 0\.05:\s*([\d.]+)"),
    "acc10":    re.compile(r"acc thres 0\.1:\s*([\d.]+)"),
    "t5":       re.compile(r"最终时间步 \(t=5\) 误差:\s*\n\s*平均:\s*([\d.]+)"),
}
for exp in exps:
    rep = os.path.join(root, exp, "analysis", "avg_l2_report.txt")
    if not os.path.isfile(rep):
        print(f"{exp:<18} (missing)")
        continue
    txt = open(rep).read()
    vals = {}
    for k, p in patterns.items():
        m = p.search(txt)
        vals[k] = float(m.group(1)) if m else float("nan")
    print(f"{exp:<18} {vals['L1']:>8.4f} {vals['L2']:>10.4f} {vals['acc05']:>10.4f} {vals['acc10']:>10.4f} {vals['t5']:>10.4f}")
print("=" * 78)
PY
