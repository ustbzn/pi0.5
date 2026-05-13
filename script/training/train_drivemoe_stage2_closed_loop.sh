#!/bin/bash

# GPU check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

export PYTHONPATH="${PWD}"
export WANDB_ENTITY="YOUR_WANDB_ENTITY" # You need to set wandb

HYDRA_FULL_ERROR=1 torchrun \
  --nproc_per_node=$NUM_GPU \
  --standalone \
  script/run.py \
  --config-path=../config/train/DriveMoE \
  --config-name=stage2_closed_loop \
  ckpt_path="YOUR_STAGE1_MODEL_CKPT"  # You need to set path