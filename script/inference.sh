export PYTHONPATH="/pcpt/pcpt/project/jiangjunyuan/code/VLA/DrivePi_mogo:$PYTHONPATH"

# ============================================================
# 多卡并行推理：4 张 GPU 各自处理 1/4 的数据
# 每个进程通过 CUDA_VISIBLE_DEVICES 隔离 GPU，
# --rank 指定当前进程编号（0-indexed），--world_size 为总 GPU 数
# ============================================================

WORLD_SIZE=4
VAL_DATA_DIR=/pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_v2/val
CONFIG_PATH=config/inference/DrivePi0/qwen_mogo_havetrain.yaml
STATISTIC_PATH=/pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_v2/statistics_9dim.json
OUTPUT_DIR=inference_out/mogo-2503/inference/no_VLM_0415
LOG_DIR=inference_out/mogo-2503/logs/no_VLM_0415

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

for RANK in $(seq 0 $((WORLD_SIZE - 1))); do
    CUDA_VISIBLE_DEVICES=${RANK} python src/agent/drivepi0/inference_mogo_test.py \
        --mode batch \
        --val_data_dir ${VAL_DATA_DIR} \
        --config_path ${CONFIG_PATH} \
        --statistic_path ${STATISTIC_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --rank ${RANK} \
        --world_size ${WORLD_SIZE} \
        > ${LOG_DIR}/rank${RANK}.log 2>&1 &
    echo "Launched rank=${RANK} on GPU ${RANK} (PID $!)"
done

echo "All ${WORLD_SIZE} processes launched. Waiting for completion..."
wait
echo "All processes finished. Results saved to: ${OUTPUT_DIR}"
echo "Logs saved to: ${LOG_DIR}"

# CUDA_VISIBLE_DEVICES=0 python src/agent/drivepi0/inference_mogo_test.py \
#     --mode batch \
#     --val_data_dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_250/val \
#     --config_path config/inference/DrivePi0/base_mogo_havetrain.yaml \
#     --statistic_path /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_250/statistics.json \
#     --output_dir inference_out/mogo-2503/inference/naviv2_histraj