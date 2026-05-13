#!/bin/bash

if [ -n "$PET_NNODES" ]; then
    NNODES=$PET_NNODES
elif [ -n "$NNODES" ]; then
    NNODES=$NNODES
else
    NNODES=${NNODES:-2}
fi

if [ -n "$PET_NPROC_PER_NODE" ]; then
    NPROC_PER_NODE=$PET_NPROC_PER_NODE
elif [ -n "$NPROC_PER_NODE" ]; then
    NPROC_PER_NODE=$NPROC_PER_NODE
else
    NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}
fi

if [ -n "$PET_NODE_RANK" ]; then
    NODE_RANK=$PET_NODE_RANK
elif [ -n "$NODE_RANK" ]; then
    NODE_RANK=$NODE_RANK
else
    NODE_RANK=${NODE_RANK:-0}
fi

if [ -n "$MASTER_ADDR" ]; then
    MASTER_ADDR=$MASTER_ADDR
else
    MASTER_ADDR=${MASTER_ADDR:-"localhost"}
fi

if [ -n "$MASTER_PORT" ]; then
    MASTER_PORT=$MASTER_PORT
else
    MASTER_PORT=${MASTER_PORT:-"29500"}
fi

echo "========== Distributed Training Config =========="
echo "NNODES=$NNODES"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "=================================================="

HYDRA_FULL_ERROR=1 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    script/run.py \
    --config-path=../config/train/DriveMoE \
    --config-name=stage1_closed_loop