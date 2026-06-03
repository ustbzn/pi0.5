#!/bin/bash

export TZ='UTC-8'
# SwanLab 配置
export WANDB_DISABLED=true
export SWANLAB_API_KEY='wMwCepr1MuBtmCchEgh9x'
export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU:$PYTHONPATH"
# 保存路径
export LOG_DIR="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU/logs"

# GPU check
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
NUM_GPU="$(nvidia-smi --list-gpus | wc -l)"
echo "NUM_GPU=$NUM_GPU"

# 配置文件：使用 one-hot 编码 + 加权采样的新配置
BASE_CONFIG="0603_qwen3_video_hdmap_A_vlm_8gpu.yaml"

# 自动寻找空闲端口
find_free_port() {
    python -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); port = s.getsockname()[1]; s.close(); print(port)"
}
export MASTER_PORT=$(find_free_port)

# 使用 torchrun 进行多卡训练（standalone 模式，无需 SLURM）
HYDRA_FULL_ERROR=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=$NUM_GPU \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --max-restarts=0 \
  --standalone \
  script/run.py \
  --config-path=../config/train/DrivePi0 \
  --config-name=${BASE_CONFIG}

echo "Training script finished."
echo "Logs are saved in $LOG_DIR"
python /pcpt/pcpt/project/jiangjunyuan/tools/simple_gpu_stress.py all