#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将所有 segment 中 drivepi 目录下的 pkl 文件复制到指定目标文件夹（适配新数据目录结构）

新数据目录结构:
    rosbags/
    |-- 0312-2/
    |   |-- 2026-03-12-15-48-50/
    |   |   |-- 20260312154919/
    |   |   |   |-- drivepi/  (pkl 文件)

使用示例:
    # 复制所有数据到 train 目录（排除某些日期作为 val）
    python3 move_pkl_files_py3.py \\
        --rosbags-root /pcpt/pcpt/data/VLA/MOGO/rosbags \\
        --target-dir datasets/mogo-new/train \\
        --exclude-dates 0312-2

    # 只复制某些日期到 val 目录
    python3 move_pkl_files_py3.py \\
        --rosbags-root /pcpt/pcpt/data/VLA/MOGO/rosbags \\
        --target-dir datasets/mogo-new/val \\
        --include-dates 0312-2

    # 复制所有数据
    python3 move_pkl_files_py3.py \\
        --rosbags-root /pcpt/pcpt/data/VLA/MOGO/rosbags \\
        --target-dir datasets/mogo-new/train
"""

import os
import shutil
import argparse
from pathlib import Path


def discover_pkl_files(rosbags_root, include_dates=None, exclude_dates=None):
    """
    遍历新目录结构，收集所有 drivepi 目录下的 pkl 文件。

    返回:
        list of (pkl_path, label_str)
    """
    pkl_files = []

    include_set = set(include_dates) if include_dates else None
    exclude_set = set(exclude_dates) if exclude_dates else set()

    rosbags_path = Path(rosbags_root)
    if not rosbags_path.exists():
        print(f"[ERROR] rosbag根目录不存在: {rosbags_root}")
        return []

    for date_dir in sorted(rosbags_path.iterdir()):
        if not date_dir.is_dir():
            continue

        date_name = date_dir.name

        # 日期过滤
        if include_set is not None and date_name not in include_set:
            continue
        if date_name in exclude_set:
            continue

        for route_dir in sorted(date_dir.iterdir()):
            if not route_dir.is_dir():
                continue

            for segment_dir in sorted(route_dir.iterdir()):
                if not segment_dir.is_dir():
                    continue

                drivepi_dir = segment_dir / "drivepi"
                if not drivepi_dir.exists() or not drivepi_dir.is_dir():
                    continue

                for pkl_file in sorted(drivepi_dir.glob("*.pkl")):
                    label = f"{date_name}/{route_dir.name}/{segment_dir.name}"
                    pkl_files.append((pkl_file, label))

    return pkl_files


def main():
    parser = argparse.ArgumentParser(
        description="将所有 segment 中 drivepi/ 下的 pkl 文件复制到指定目标文件夹（适配新数据目录结构）")
    parser.add_argument(
        "--rosbags-root",
        type=str,
        default="/pcpt/pcpt/data/VLA/MOGO/rosbags",
        help="rosbag根目录 (默认: /pcpt/pcpt/data/VLA/MOGO/rosbags)")
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="目标目录路径（pkl文件将被复制到这里）")
    parser.add_argument(
        "--include-dates",
        type=str,
        nargs="+",
        default=None,
        help="只包含这些日期文件夹（如果指定则只处理这些）")
    parser.add_argument(
        "--exclude-dates",
        type=str,
        nargs="+",
        default=[],
        help="排除这些日期文件夹")

    args = parser.parse_args()

    target_path = Path(args.target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Rosbags根目录: {args.rosbags_root}")
    print(f"[INFO] 目标目录: {args.target_dir}")
    if args.include_dates:
        print(f"[INFO] 只包含日期: {args.include_dates}")
    if args.exclude_dates:
        print(f"[INFO] 排除日期: {args.exclude_dates}")

    pkl_files = discover_pkl_files(
        args.rosbags_root,
        include_dates=args.include_dates,
        exclude_dates=args.exclude_dates)

    if not pkl_files:
        print("[WARN] 未找到任何pkl文件")
        return

    print(f"[INFO] 找到 {len(pkl_files)} 个pkl文件")

    copied_count = 0
    skipped_count = 0

    for pkl_file, label in pkl_files:
        target_file = target_path / pkl_file.name

        if target_file.exists():
            # 新数据中不同 segment 可能有同名文件，需要重命名
            # 用 segment 名称作为前缀避免冲突
            segment_name = pkl_file.parent.parent.name
            new_name = f"{segment_name}_{pkl_file.name}"
            target_file = target_path / new_name

            if target_file.exists():
                print(f"[WARN] 目标文件已存在，跳过: {new_name}")
                skipped_count += 1
                continue

        try:
            shutil.copy2(str(pkl_file), str(target_file))
            copied_count += 1
        except Exception as e:
            print(f"[ERROR] 复制文件失败 {pkl_file}: {e}")
            skipped_count += 1

    print(f"\n{'='*50}")
    print(f"成功复制: {copied_count} 个文件")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} 个文件")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
