# Video 5帧全链路 debug：单卡 n_updates=1，验证 video 输入训练+eval 流水线打通
export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU:$PYTHONPATH"
export LOG_DIR="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU/logs"
export CUDA_VISIBLE_DEVICES=0
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

HYDRA_FULL_ERROR=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --max-restarts=0 \
  --standalone \
  script/run.py \
  --config-path=../config/train/DrivePi0 \
  --config-name=0529_qwen3_video_5frame_8gpu.yaml \
  use_torch_compile=False \
  per_device_batch_size=2 \
  num_workers=2 \
  eval_size=8 \
  ++eval_full_val=False \
  ++eval_seed=20260529 \
  ++eval_repeats=1 \
  ++eval_map_modes='[normal]' \
  eval_freq=1 \
  n_updates=1 \
  save_model_freq=999999 \
  action_lr=1e-8 \
  vlm_lr=1e-8 \
  action_lr_scheduler.first_cycle_steps=1000 \
  vlm_lr_scheduler.first_cycle_steps=1000 \
  action_lr_scheduler.min_lr=1e-8 \
  vlm_lr_scheduler.min_lr=1e-8
