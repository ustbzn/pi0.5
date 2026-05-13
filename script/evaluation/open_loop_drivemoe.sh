#!/bin/bash

export PYTHONPATH="${PWD}"

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 torchrun \
    script/run.py \
    --config-name=open_loop \
    --config-path=../config/eval/DriveMoE \
    checkpoint_path="YOUR_CHECKPOINT_PATH"  # You need to set path