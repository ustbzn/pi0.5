#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JSON文件转换为PKL文件的工具脚本（适配新数据目录结构）

输入目录结构 (jsons):
    jsons/
    |-- 0312-2/
    |   |-- 2026-03-12-15-48-50/
    |   |   |-- 20260312154919/
    |   |   |   |-- 20260312154919_step1.json
    |   |   |   |-- 20260312154919_step2.json

输出目录结构 (mogo_dataset):
    mogo_dataset/
    |-- train/
    |   |-- xxxx.pkl
    |-- val/
    |   |-- xxxx.pkl

使用方法:
    # 全部转换到 train
    python3 json_to_pkl_py3.py \\
        --jsons-root /pcpt/pcpt/data/VLA/MOGO/jsons \\
        --output-dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset/train

    # 指定日期转换到 val
    python3 json_to_pkl_py3.py \\
        --jsons-root /pcpt/pcpt/data/VLA/MOGO/jsons \\
        --output-dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset/val \\
        --dates 0312-2

    # 跳过已存在的 pkl
    python3 json_to_pkl_py3.py \\
        --jsons-root /pcpt/pcpt/data/VLA/MOGO/jsons \\
        --output-dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset/train \\
        --skip-existing
"""

import sys
import os
import json
import pickle
import base64
import argparse
import traceback
import numpy as np


# 导航指令映射：第一个路口的指令 -> 整数ID
NAVIGATION_COMMAND_MAP = {
    "straight": 0,
    "turn left": 1,
    "turn right": 2,
    "u turn": 3,
    "park": 4,
}


class StringWrapper:
    """
    包装字符串，使其具有 .numpy() 方法。
    序列化时使用 __reduce__ 返回 dict 类型，加载时不需要类定义。
    """
    def __init__(self, string_value):
        if isinstance(string_value, str):
            self._bytes_value = string_value.encode('utf-8')
        elif isinstance(string_value, bytes):
            self._bytes_value = string_value
        else:
            self._bytes_value = str(string_value).encode('utf-8')

    def numpy(self):
        return self._bytes_value

    def __reduce__(self):
        return (dict, ({'_bytes_value': self._bytes_value, '_string_wrapper_marker': True},))


def deserialize_from_json(obj):
    """从JSON格式反序列化为原始格式"""
    if isinstance(obj, dict):
        if '__numpy_array__' in obj:
            arr = np.array(obj['data'], dtype=obj['dtype'])
            if arr.shape != tuple(obj['shape']):
                arr = arr.reshape(obj['shape'])
            return arr
        elif '__bytes__' in obj:
            return base64.b64decode(obj['data'])
        else:
            return {key: deserialize_from_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [deserialize_from_json(item) for item in obj]
    else:
        return obj


def convert_json_to_pkl(json_path, pkl_path, overwrite=True):
    """将单个JSON文件转换为PKL文件"""
    if not overwrite and os.path.exists(pkl_path):
        return False

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        pkl_data = deserialize_from_json(json_data)

        image_path_fields = [
            'his_cam_front', 'his_cam_front_time',
            'his_cam_right', 'his_cam_right_time',
            'his_cam_left', 'his_cam_left_time',
            'his_cam_back', 'his_cam_back_time',
            'his_cam_left_back', 'his_cam_left_back_time',
            'his_cam_right_back', 'his_cam_right_back_time',
            'skill_cam',
        ]

        def wrap_image_path_fields(data):
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in image_path_fields:
                        if isinstance(value, (str, bytes)):
                            data[key] = StringWrapper(value)
                        elif isinstance(value, np.ndarray):
                            try:
                                if value.dtype == np.uint8:
                                    decoded_str = value.tobytes().decode('utf-8')
                                    data[key] = StringWrapper(decoded_str)
                            except Exception:
                                pass
                    else:
                        if isinstance(value, dict):
                            wrap_image_path_fields(value)

        if isinstance(pkl_data, dict):
            wrap_image_path_fields(pkl_data)

            # 从 navigation_command 提取第一个路口的指令，映射为 navigation_id
            nav_cmd = pkl_data.get('navigation_command')
            if nav_cmd and isinstance(nav_cmd, list) and len(nav_cmd) > 0:
                first_cmd = nav_cmd[0]  # 只使用第一个路口
                if isinstance(first_cmd, list) and len(first_cmd) > 0:
                    cmd_str = first_cmd[0]  # e.g. "straight", "turn left", "turn right"
                    nav_id = NAVIGATION_COMMAND_MAP.get(cmd_str, -1)
                    if nav_id == -1:
                        print(f"[WARNING] 未知导航指令: {cmd_str}, 文件: {json_path}")
                        nav_id = 0  # 回退到 straight
                    # 存为与 window 维度一致的数组 (shape=[5,])，方便训练加载
                    window_size = len(pkl_data.get('his_speed', {}).get('data', [0]*5)) \
                        if isinstance(pkl_data.get('his_speed'), dict) \
                        else (pkl_data['his_speed'].shape[0] if hasattr(pkl_data.get('his_speed'), 'shape') else 5)
                    pkl_data['navigation_id'] = np.array([nav_id] * window_size, dtype=np.float32)
                else:
                    pkl_data['navigation_id'] = np.array([0] * 5, dtype=np.float32)
            else:
                pkl_data['navigation_id'] = np.array([0] * 5, dtype=np.float32)

        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f, protocol=3)

        return True
    except Exception as e:
        print(f"[ERROR] 转换失败 {json_path}: {e}")
        return False


def discover_json_files(jsons_root, dates=None):
    """
    遍历jsons目录结构，收集所有 segment 目录下的 .json 文件。

    返回:
        list of (json_path, label_str)
    """
    json_files = []

    if dates is None:
        dates = sorted([
            d for d in os.listdir(jsons_root)
            if os.path.isdir(os.path.join(jsons_root, d))
        ])

    for date_dir in dates:
        date_path = os.path.join(jsons_root, date_dir)
        if not os.path.isdir(date_path):
            continue

        for route_name in sorted(os.listdir(date_path)):
            route_path = os.path.join(date_path, route_name)
            if not os.path.isdir(route_path):
                continue

            for segment_name in sorted(os.listdir(route_path)):
                segment_path = os.path.join(route_path, segment_name)
                if not os.path.isdir(segment_path):
                    continue

                # 直接在 segment 目录下查找 .json 文件
                for filename in sorted(os.listdir(segment_path)):
                    if filename.endswith('.json'):
                        json_path = os.path.join(segment_path, filename)
                        label = f"{date_dir}/{route_name}/{segment_name}"
                        json_files.append((json_path, label))

    return json_files


def main():
    parser = argparse.ArgumentParser(
        description="将 JSON 文件转换为 PKL 文件，输出到指定目录（适配新数据目录结构）")
    parser.add_argument(
        "--jsons-root",
        type=str,
        default="/pcpt/pcpt/data/VLA/MOGO/jsons",
        help="json根目录 (默认: /pcpt/pcpt/data/VLA/MOGO/jsons)")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="pkl输出目录 (如 /pcpt/pcpt/data/VLA/MOGO/mogo_dataset/train)")
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=None,
        help="指定要处理的日期文件夹 (默认: 全部)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果PKL文件已存在，跳过转换（默认会覆盖）")

    args = parser.parse_args()

    jsons_root = args.jsons_root
    output_dir = args.output_dir
    overwrite = not args.skip_existing

    print("[INFO] JSON转PKL工具 (新数据结构)")
    print(f"[INFO] Jsons根目录: {jsons_root}")
    print(f"[INFO] PKL输出目录: {output_dir}")

    if not os.path.isdir(jsons_root):
        print(f"[ERROR] 目录不存在：{jsons_root}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    json_files = discover_json_files(jsons_root, dates=args.dates)

    if not json_files:
        print("[ERROR] 未找到任何 JSON 文件")
        sys.exit(1)

    print(f"[INFO] 找到 {len(json_files)} 个 JSON 文件")

    total_converted = 0
    total_skipped = 0

    for json_path, label in json_files:
        filename = os.path.basename(json_path)
        pkl_filename = filename.replace('.json', '.pkl')
        pkl_path = os.path.join(output_dir, pkl_filename)

        if convert_json_to_pkl(json_path, pkl_path, overwrite=overwrite):
            total_converted += 1
        elif not overwrite and os.path.exists(pkl_path):
            total_skipped += 1

        if total_converted % 200 == 0 and total_converted > 0:
            print(f"[INFO] 已转换 {total_converted} 个文件...")

    print(f"\n{'='*80}")
    print(f"[INFO] 处理完成！共转换 {total_converted} 个文件, 跳过 {total_skipped} 个")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
