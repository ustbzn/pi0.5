#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将JSON文件转换为PKL文件的工具脚本

使用方法:
    python3 json_to_pkl.py <bags_root_dir>
    
功能:
    - 遍历bags_root_dir下的所有bag文件夹
    - 查找每个bag文件夹下的drivepi_json目录
    - 将JSON文件转换为PKL文件，保存到drivepi目录
"""

from __future__ import print_function
import sys
import os
import json
import pickle
import base64
import numpy as np
from tqdm import tqdm
import argparse


class StringWrapper:
    """
    包装字符串，使其具有 .numpy() 方法
    用于模拟 TensorFlow Tensor 的行为，兼容 inference_mogo.py 的期望格式
    
    序列化时使用 __reduce__ 返回 dict 类型，这样加载时不需要类定义
    """
    def __init__(self, string_value):
        if isinstance(string_value, str):
            # 将字符串编码为 bytes（与 TensorFlow Tensor 行为一致）
            self._bytes_value = string_value.encode('utf-8')
        elif isinstance(string_value, bytes):
            self._bytes_value = string_value
        else:
            # 如果是其他类型，先转为字符串
            self._bytes_value = str(string_value).encode('utf-8')
    
    def numpy(self):
        """返回 bytes 对象，兼容 .numpy().decode('utf-8') 的调用方式"""
        return self._bytes_value
    
    def __reduce__(self):
        """
        序列化为 dict 类型，避免依赖外部还原函数
        这样生成的 pkl 文件可以在任何脚本中加载
        加载时会得到 dict，inference_mogo.py 中已有兼容性处理
        """
        # 返回 dict 类型和标记字典
        # 加载时会得到 dict，不需要 StringWrapper 类定义
        return (dict, ({'_bytes_value': self._bytes_value, '_string_wrapper_marker': True},))

def deserialize_from_json(obj):
    """
    从JSON格式反序列化为原始格式
    处理numpy数组和bytes类型
    """
    if isinstance(obj, dict):
        # 检查是否是numpy数组
        if '__numpy_array__' in obj:
            arr = np.array(obj['data'], dtype=obj['dtype'])
            # 确保shape正确
            if arr.shape != tuple(obj['shape']):
                arr = arr.reshape(obj['shape'])
            return arr
        # 检查是否是bytes
        elif '__bytes__' in obj:
            return base64.b64decode(obj['data'])
        else:
            # 递归处理字典
            return {key: deserialize_from_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        # 递归处理列表
        return [deserialize_from_json(item) for item in obj]
    else:
        return obj

def convert_json_to_pkl(json_path, pkl_path):
    """
    将单个JSON文件转换为PKL文件
    
    参数:
        json_path: JSON文件路径
        pkl_path: 输出PKL文件路径
    """
    try:
        # 读取JSON文件
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 反序列化为原始格式
        pkl_data = deserialize_from_json(json_data)
        
        # 处理需要 .numpy() 方法的字符串字段（兼容 inference_mogo.py 的期望格式）
        # 这些字段在 inference_mogo.py 中会被调用 .numpy().decode('utf-8')
        image_path_fields = [
            'his_cam_front',
            'his_cam_front_time',
            'his_cam_back',
            'his_cam_back_time',
            'his_cam_front_left',
            'his_cam_front_right',
            'his_cam_back_left',
            'his_cam_back_right',
            'skill_cam',
        ]
        
        def wrap_image_path_fields(data):
            """递归包装图像路径字段"""
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in image_path_fields:
                        # 如果是字符串或 bytes，包装成 StringWrapper
                        if isinstance(value, (str, bytes)):
                            data[key] = StringWrapper(value)
                        # 如果已经是 StringWrapper，跳过（避免重复包装）
                        elif isinstance(value, StringWrapper):
                            pass
                        # 如果是 numpy 数组，尝试解码后包装
                        elif isinstance(value, np.ndarray):
                            try:
                                # 尝试解码为字符串
                                if value.dtype == np.uint8:
                                    decoded_str = value.tobytes().decode('utf-8')
                                    data[key] = StringWrapper(decoded_str)
                            except Exception:
                                # 如果解码失败，保持原样
                                pass
                    else:
                        # 递归处理嵌套字典
                        if isinstance(value, dict):
                            wrap_image_path_fields(value)
        
        if isinstance(pkl_data, dict):
            wrap_image_path_fields(pkl_data)
        
        # 保存为PKL文件（使用 Python 3 协议）
        with open(pkl_path, 'wb') as f:
            pickle.dump(pkl_data, f, protocol=3)  # protocol=3 是 Python 3 的标准协议
        
        return True
    except Exception as e:
        print("[ERROR] 转换失败 {}: {}".format(json_path, e))
        return False

def process_bag_folder(bag_dir):
    """
    处理单个bag文件夹
    
    参数:
        bag_dir: bag文件夹路径
    
    返回:
        成功转换的文件数量
    """
    json_dir = os.path.join(bag_dir, "drivepi_json")
    pkl_dir = os.path.join(bag_dir, "drivepi")
    
    # 检查JSON目录是否存在
    if not os.path.isdir(json_dir):
        print("[WARN] JSON目录不存在: {}".format(json_dir))
        return 0
    
    # 创建PKL目录
    if not os.path.exists(pkl_dir):
        try:
            os.makedirs(pkl_dir)
        except OSError:
            print("[ERROR] 无法创建PKL目录: {}".format(pkl_dir))
            return 0
    
    # 查找所有JSON文件
    json_files = []
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            json_path = os.path.join(json_dir, filename)
            json_files.append(json_path)
    
    if not json_files:
        print("[WARN] 未找到JSON文件: {}".format(json_dir))
        return 0
    
    json_files.sort()
    
    # 转换每个JSON文件
    success_count = 0
    for json_path in json_files:
        filename = os.path.basename(json_path)
        pkl_filename = filename.replace('.json', '.pkl')
        pkl_path = os.path.join(pkl_dir, pkl_filename)
        
        if convert_json_to_pkl(json_path, pkl_path):
            success_count += 1
    
    return success_count

def main():
    print("[INFO] JSON转PKL工具")
    print("[INFO] 使用方法: python3 json_to_pkl.py <bags_root_dir>")

    parser = argparse.ArgumentParser(description="将包含 drivepi_json 的 bag 目录转换成 PKL 文件")
    parser.add_argument(
        "bags_root_dir",
        help="bags 根目录路径（其下包含若干 bag 子目录，每个子目录下有 drivepi_json）"
    )
    args = parser.parse_args()
    bags_root_dir = args.bags_root_dir

    if not os.path.isdir(bags_root_dir):
        print("[ERROR] 目录不存在：{}".format(bags_root_dir))
        sys.exit(1)
    
    # 遍历所有bag文件夹
    bag_dirs = []
    for item in os.listdir(bags_root_dir):
        item_path = os.path.join(bags_root_dir, item)
        if os.path.isdir(item_path):
            # 检查是否包含drivepi_json目录
            json_dir = os.path.join(item_path, "drivepi_json")
            if os.path.isdir(json_dir):
                bag_dirs.append(item_path)
    
    if not bag_dirs:
        print("[ERROR] 未找到包含drivepi_json目录的文件夹")
        sys.exit(1)
    
    print("[INFO] 找到 {} 个bag文件夹".format(len(bag_dirs)))
    
    # 处理每个bag文件夹
    total_converted = 0
    for idx, bag_dir in enumerate(sorted(bag_dirs), 1):
        print("\n" + "="*80)
        print("[INFO] 处理第 {}/{} 个bag文件夹: {}".format(idx, len(bag_dirs), bag_dir))
        print("="*80)
        
        try:
            count = process_bag_folder(bag_dir)
            total_converted += count
            print("[INFO] 完成处理: {} (转换了 {} 个文件)".format(bag_dir, count))
        except Exception as e:
            print("[ERROR] 处理失败 {}: {}".format(bag_dir, e))
            import traceback
            traceback.print_exc()
            print("[WARN] 继续处理下一个bag文件夹...")
            continue
    
    print("\n" + "="*80)
    print("[INFO] 处理完成！共转换了 {} 个文件".format(total_converted))
    print("="*80)

if __name__ == '__main__':
    main()

