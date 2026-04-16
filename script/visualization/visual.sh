export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_mogo:$PYTHONPATH"

python src/utils/analyzing_traj_mogo.py \
    --trajectory_dir inference_out/mogo-2503/inference/no_VLM_0415 \
    --pkl_root_dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_v2/val \
    --mode dual \
    --bev_xlim -1 50 \
    --bev_ylim -10 10 \
    --output_dir visual_out/mogo-2503/dual_0415 \
    # --scene_names 20260312161041 #20260312164028 #20260312160557