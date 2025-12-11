#!/bin/bash

#SBATCH --job-name=eval-pretrain
#SBATCH --output=logs/eval/%A.out
#SBATCH --error=logs/eval/%A.err
#SBATCH --time=5:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

export TZ='UTC-8'
# SwanLab 配置
export WANDB_DISABLED=true
export SWANLAB_API_KEY='WIuVGDGJJcxnKEAis5knC'
# 保存路径
export LOG_DIR="/D2/NXT/vlm/miaobinqi/DriveMoE/logs"

CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 torchrun \
    script/run.py \
    --config-name=base \
    --config-path=../config/eval/DrivePi0