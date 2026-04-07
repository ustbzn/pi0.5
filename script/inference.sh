export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/Drive_pi:$PYTHONPATH"
CUDA_VISIBLE_DEVICES=0 python src/agent/drivepi0/inference_mogo_test.py \
    --mode batch \
    --val_data_dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset/val \
    --config_path config/inference/DrivePi0/base_mogo_havetrain.yaml \
    --statistic_path /pcpt/pcpt/data/VLA/MOGO/mogo_dataset/statistics.json \
    --output_dir inference_out/mogo-2503/inference/no_VLM_0331