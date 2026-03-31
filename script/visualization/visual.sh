python src/utils/analyzing_traj_mogo.py \
    --trajectory_dir inference_out/mogo-2503/inference/no_VLM_0330 \
    --pkl_root_dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset/val \
    --mode dual \
    --bev_xlim -1 50 \
    --bev_ylim -10 10 \
    --output_dir visual_out/mogo-2503/dual_0330 \
    # --scene_names 20251210141637