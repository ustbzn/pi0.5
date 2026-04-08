#!/bin/bash
# 生成 val (0312-2)
python3 json_to_pkl_py3.py \
      --jsons-root /pcpt/pcpt/data/VLA/MOGO/jsons_new_v5 \
      --output-dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi/val \
      --dates 0312-2

# 生成 train (其余所有日期)
python3 json_to_pkl_py3.py \
      --jsons-root /pcpt/pcpt/data/VLA/MOGO/jsons_new_v5 \
      --output-dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi/train \
      --dates 0313-1 0316 0317 0318 0319
