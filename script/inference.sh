CUDA_VISIBLE_DEVICES=0 python src/agent/drivepi0/inference_mogo_test.py --mode batch \
    --val_data_dir datasets/mogo-test/val \
    --config_path config/inference/DrivePi0/base_mogo_havetrain.yaml \
    --statistic_path datasets/mogo-test/statistics.json \
    --output_dir datasets/mogo-test/inference