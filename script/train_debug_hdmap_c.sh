export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU:$PYTHONPATH"
export LOG_DIR="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU/logs"
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")

HYDRA_FULL_ERROR=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=8 \
  --rdzv_id=$RANDOM \
  --rdzv_backend=c10d \
  --max-restarts=0 \
  --standalone \
  script/run.py \
  --config-path=../config/train/DrivePi0 \
  --config-name=0424_qwen3_hdmap_C_8gpu.yaml \
  finetune_checkpoint_path="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU/logs/train/drive-pi0/2026-04-24_09-43_mogo_hdmap_C_8gpu/checkpoint/step12480.pt" \
  eval_size=8192 \
  ++eval_full_val=False \
  ++eval_seed=20260428 \
  ++eval_repeats=3 \
  ++eval_map_modes='[normal,zero,shuffle]' \
  eval_freq=1 \
  n_updates=1 \
  save_model_freq=999999 \
  action_lr=1e-8 \
  vlm_lr=1e-8 \
  action_lr_scheduler.first_cycle_steps=1000 \
  vlm_lr_scheduler.first_cycle_steps=1000 \
  action_lr_scheduler.min_lr=1e-8 \
  vlm_lr_scheduler.min_lr=1e-8
