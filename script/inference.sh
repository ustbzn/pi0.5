export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/Drive_pi:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 python src/agent/drivepi0/inference_nuscene_test.py \
    --mode batch \
    --val_data_dir datasets/nuscene-0111/val \
    --config_path config/inference/DrivePi0/base_mogo_havetrain.yaml \
    --statistic_path datasets/nuscene-0111/statistics.json \
    --output_dir inference_out/nuscene-0111/inference/no_VLM