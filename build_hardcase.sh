python script/analysis/build_state_only_hardcase_weights.py \
      --checkpoint /pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU/logs/train/drive-pi0/2026-05-18_16-02_ablation_state_only_8gpu/checkpoint/step28050.pt \
      --data_dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/train \
      --statistics_path /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/statistics.json \
      --output /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/train_hardcase_weights.json \
      --batch_size 48 \
      --gpu_id 0