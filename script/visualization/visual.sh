python src/utils/analyzing_traj.py \
    --trajectory_dir inference_out/nuscene-0111/inference/no_VLM \
    --pkl_dir datasets/nuscene-0111/val \
    --mode dual \
    --bev_xlim -1 50 \
    --bev_ylim -20 20 \
    --output_dir visual_out/nuscene-0111/dual_0323 \
    # --scene_names 20251210141637