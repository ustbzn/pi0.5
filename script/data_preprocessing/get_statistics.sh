#!/bin/bash

DATA_PATH="/pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_v2/train"
OUTPUT_PATH="/pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_v2/statistics.json"

python "src/data_processing/get_statistics.py" \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH"
