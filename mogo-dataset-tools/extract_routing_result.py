#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
提取并统一分发 routing_result.bag 文件（无交互版）

功能：
  在指定的根目录下，遍历所有包含 -rosmaster- 和 -rosslave- bag 文件的子文件夹，
  寻找第一个包含话题 /hadmap_engine/route_result_msg 的 master bag，提取该话题
  的所有消息生成 routing_result.bag，然后将该文件复制到其他缺失该文件的子文件夹中。

用法：
  python3 extract_routing_result.py [bags_root_dir]

  如果未提供 bags_root_dir，则使用默认路径 /media/lanxx/T7/VLA/luxian_1
"""

import os
import sys
import shutil
import rosbag

def find_bag_dirs(root_dir):
    """
    在 root_dir 下查找同时包含 -rosmaster- 和 -rosslave- bag 文件的子文件夹。
    返回符合条件的子文件夹路径列表。
    """
    bag_dirs = []
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if not os.path.isdir(item_path):
            continue
        has_master = False
        has_slave = False
        for filename in os.listdir(item_path):
            if not filename.endswith('.bag'):
                continue
            if '-rosmaster-' in filename:
                has_master = True
            elif '-rosslave-' in filename:
                has_slave = True
        if has_master and has_slave:
            bag_dirs.append(item_path)
    return bag_dirs

def extract_routing_result(bag_dir):
    """
    在给定的 bag_dir 中查找第一个包含目标话题的 master bag，
    提取话题 /hadmap_engine/route_result_msg 的所有消息，
    生成 routing_result.bag 文件。
    返回生成的 bag 文件完整路径，如果未找到目标话题则返回 None。
    """
    master_bags = [f for f in os.listdir(bag_dir) 
                   if f.endswith('.bag') and '-rosmaster-' in f]
    if not master_bags:
        return None

    for master_bag in master_bags:
        master_path = os.path.join(bag_dir, master_bag)
        try:
            with rosbag.Bag(master_path, 'r') as bag:
                topics = bag.get_type_and_topic_info().topics
                if '/hadmap_engine/route_result_msg' in topics:
                    out_path = os.path.join(bag_dir, 'routing_result.bag')
                    with rosbag.Bag(out_path, 'w') as outbag:
                        for topic, msg, t in bag.read_messages(
                                topics=['/hadmap_engine/route_result_msg']):
                            outbag.write(topic, msg, t)
                    print(f"[INFO] 已从 {master_bag} 提取话题，生成: {out_path}")
                    return out_path
        except Exception as e:
            print(f"[ERROR] 处理 {master_path} 时出错: {e}")
            continue
    return None

def main():
    print("[INFO] 脚本启动：提取并分发 routing_result.bag")
    print("[INFO] 用法: python3 extract_routing_result.py [bags_root_dir]")

    # 默认根目录
    default_root = "/media/lanxx/T7/VLA/luxian_1"

    # 获取根目录：如果有命令行参数则使用，否则使用默认值（无交互）
    if len(sys.argv) > 1:
        bags_root_dir = sys.argv[1]
    else:
        bags_root_dir = default_root

    print(f"[INFO] Bags根目录：{bags_root_dir}")

    # 检查目录是否存在
    if not os.path.isdir(bags_root_dir):
        print(f"[ERROR] 目录不存在：{bags_root_dir}")
        sys.exit(1)

    # 查找所有符合条件的子文件夹
    bag_dirs = find_bag_dirs(bags_root_dir)
    if not bag_dirs:
        print("[ERROR] 未找到包含 master 和 slave bag 的文件夹")
        sys.exit(1)

    print(f"[INFO] 找到 {len(bag_dirs)} 个符合条件的文件夹")

    # 遍历（排序）寻找第一个包含目标话题的目录
    source_path = None
    source_dir = None
    for bag_dir in sorted(bag_dirs):
        result = extract_routing_result(bag_dir)
        if result:
            source_path = result
            source_dir = bag_dir
            break

    if source_path is None:
        print("[ERROR] 所有文件夹的 master bag 中均未找到话题 /hadmap_engine/route_result_msg")
        sys.exit(1)

    # 复制到其他缺失该文件的文件夹
    print("\n[INFO] 开始复制到其他文件夹...")
    copied_count = 0
    for bag_dir in bag_dirs:
        if bag_dir == source_dir:
            continue  # 源文件夹已有
        target = os.path.join(bag_dir, "routing_result.bag")
        if os.path.exists(target):
            print(f"[INFO] {target} 已存在，跳过")
            continue
        try:
            shutil.copy2(source_path, target)
            print(f"[INFO] 已复制到 {target}")
            copied_count += 1
        except Exception as e:
            print(f"[ERROR] 复制到 {target} 失败: {e}")

    print(f"\n[INFO] 处理完成。共复制 {copied_count} 个文件。")

if __name__ == "__main__":
    main()