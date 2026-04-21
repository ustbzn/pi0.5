#!/bin/bash
# HDMap 数据集归一化统计脚本
# 默认只计算 speed/acceleration 等训练所需的归一化参数（~1-2分钟）
# 加 --compute_hdmap_stats 会额外加载所有 NPZ 计算通道分布（~1小时，仅诊断用）

python src/data_processing/get_statistics_hdmap.py \
    --data_path /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/train \
    --output_path /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/statistics.json
