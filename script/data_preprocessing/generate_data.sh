#!/bin/bash

DATASET_PATH="$DATA_DIR"
WORK_DIR="${PWD}"
CAM_ID_PATH="$CAMERA_LABEL_DIR"
TARGET_DIR="datasets/b2d"

python src/data_processing/generate_data.py \
    --dataset_path "$DATASET_PATH" \
    --cam_id_path "$CAM_ID_PATH" \
    --work_dir "$WORK_DIR" \
    --target_dir $TARGET_DIR