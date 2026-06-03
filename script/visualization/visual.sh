export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_8GPU:$PYTHONPATH"

python src/utils/analyzing_traj_mogo.py \
    --trajectory_dir logs/eval/drive-pi0/mogo_action_avg_l2/0602_video_worldsim/trajectories \
    --pkl_root_dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_worldsim/val \
    --mode dual \
    --bev_xlim -25 50 \
    --bev_ylim -15 15 \
    --output_dir visual_out/mogo-2503/dual_0602_worldsim \
    --show_objects
    
    # --scene_names 20260312161041 #20260312160557 #20260312164028 --bev_xlim -1 50 \
