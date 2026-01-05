#!/bin/bash

WORK_DIR="${PWD}"
# WINDOW_SIZE=5
# HORIZON=10

python "src/data_processing/window.py" \
    --work_dir "$WORK_DIR" \
    --window_size 4 \
    --horizon 6 \
    --stride 5 \
    --generate_fixed False \
    --dynamic_dir "datasets/b2d"