#!/bin/bash

#SBATCH --job-name=pg-vla
#SBATCH --output=logs/%A.out
#SBATCH --error=logs/%A.err
#SBATCH --time=71:59:59
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=104
#SBATCH --mem=500G

export TZ='UTC-8'
# SwanLab 配置
export WANDB_DISABLED=true
export SWANLAB_API_KEY='MaEAs5BY7UrlqJkDxwQJF'
export PYTHONPATH="/pcpt/pcpt/project/atlb/vla/Drive_pi:$PYTHONPATH"
# 保存路径
export LOG_DIR="/pcpt/pcpt/project/atlb/vla/Drive_pi/logs"

# GPU check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
# NUM_GPU=1
echo "NUM_GPU=$NUM_GPU"
# 配置文件
BASE_CONFIG="qwen3.yaml"

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)

# run script with selected configuration using torchrun
HYDRA_FULL_ERROR=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPU \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --max-restarts=0 \
  --standalone \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  script/run.py \
  --config-path=../config/train/DrivePi0 \
  --config-name=${BASE_CONFIG}

echo "Training script finished."
echo "Logs are saved in $LOG_DIR"
python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all