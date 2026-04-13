#!/usr/bin/env bash

set -euo pipefail

SRC_ROOT="/pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_v2"
VAL_SRC_DIR="${SRC_ROOT}/0312-2"
DST_ROOT="/pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_v2"
VAL_DST_DIR="${DST_ROOT}/val"
TRAIN_DST_DIR="${DST_ROOT}/train"

mkdir -p "${VAL_DST_DIR}" "${TRAIN_DST_DIR}"

copy_flat() {
  local src_dir="$1"
  local dst_dir="$2"
  find "${src_dir}" -type f -name '*.pkl' -exec cp -f -t "${dst_dir}" {} +
}

copy_flat "${VAL_SRC_DIR}" "${VAL_DST_DIR}"

find "${SRC_ROOT}" -mindepth 1 -maxdepth 1 -type d ! -name '0312-2' -print0 | while IFS= read -r -d '' dir; do
  copy_flat "${dir}" "${TRAIN_DST_DIR}"
done

echo "val_count=$(find "${VAL_DST_DIR}" -maxdepth 1 -type f -name '*.pkl' | wc -l)"
echo "train_count=$(find "${TRAIN_DST_DIR}" -maxdepth 1 -type f -name '*.pkl' | wc -l)"
