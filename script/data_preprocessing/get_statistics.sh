#!/bin/bash

# DATA_PATH="/D2/NXT/vlm/miaobinqi/DriveMoE/datasets/b2d/train"
DATA_PATH="datasets/nuscene-0111/train"
# DATA_PATH="/D2/NXT/vlm/dataset/MOGO/test_data/2O-bags/20251210141637/drivepi"
# OUTPUT_PATH="/D2/NXT/vlm/miaobinqi/DriveMoE/datasets/b2d/statistics.json"
OUTPUT_PATH="datasets/nuscene-0111/statistics.json"
# OUTPUT_PATH="/D2/NXT/vlm/dataset/MOGO/test_data/2O-bags/20251210141637/statistics.json"

python "src/data_processing/get_statistics.py" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" 