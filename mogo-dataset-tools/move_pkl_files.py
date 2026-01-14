#!/usr/bin/env python3
"""
将所有子文件夹中drivepi目录下的pkl文件复制到指定目标文件夹
cd /D2/NXT/vlm/miaobinqi/Senna/mogo_test/mogo-dataset-tools
# 基本用法
python move_pkl_files.py \
    --source-dir /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags \
    --target-dir /path/to/target

# 只处理指定的子文件夹
python move_pkl_files.py \
    --source-dir /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags \
    --target-dir /D2/NXT/vlm/miaobinqi/DriveMoE/datasets/mogo-test/val \
    --include 20251210141637

# 排除单个子文件夹
python move_pkl_files.py \
    --source-dir /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags \
    --target-dir /D2/NXT/vlm/miaobinqi/DriveMoE/datasets/mogo-test/train \
    --exclude 20251210141637

# 排除多个子文件夹
python move_pkl_files.py \
    --source-dir /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags \
    --target-dir /path/to/target \
    --exclude 20251210141637 20251210144033 20251210144451
"""

import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def copy_pkl_files(source_dir, target_dir, include_dirs=None, exclude_dirs=None):
    """
    从源目录的所有子文件夹的drivepi目录中复制pkl文件到目标目录
    
    Args:
        source_dir: 源目录路径（包含所有子文件夹的目录）
        target_dir: 目标目录路径
        include_dirs: 要包含的子文件夹名列表（如果指定，则只处理这些文件夹）
        exclude_dirs: 要排除的子文件夹名列表
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # 检查源目录是否存在
    if not source_path.exists():
        print(f"错误: 源目录不存在: {source_dir}")
        return
    
    # 创建目标目录（如果不存在）
    target_path.mkdir(parents=True, exist_ok=True)
    print(f"目标目录: {target_dir}")
    
    # 处理包含列表
    if include_dirs is None:
        include_dirs = []
    include_set = set(include_dirs) if include_dirs else None
    
    # 处理排除列表
    if exclude_dirs is None:
        exclude_dirs = []
    exclude_set = set(exclude_dirs)
    
    # 收集所有pkl文件
    pkl_files = []
    subdirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    # 过滤子文件夹
    if include_set is not None:
        # 如果指定了包含列表，只处理这些文件夹
        filtered_subdirs = [d for d in subdirs if d.name in include_set]
        if exclude_set:
            # 同时应用排除列表
            filtered_subdirs = [d for d in filtered_subdirs if d.name not in exclude_set]
        if include_set:
            print(f"包含的子文件夹: {', '.join(include_set)}")
        if exclude_set:
            print(f"排除的子文件夹: {', '.join(exclude_set)}")
        print(f"\n扫描 {len(subdirs)} 个子文件夹（只处理指定的 {len(filtered_subdirs)} 个）...")
    else:
        # 如果没有指定包含列表，处理所有文件夹（除了被排除的）
        filtered_subdirs = [d for d in subdirs if d.name not in exclude_set]
        excluded_count = len(subdirs) - len(filtered_subdirs)
        if exclude_set:
            print(f"排除的子文件夹: {', '.join(exclude_set)}")
        if excluded_count > 0:
            print(f"\n扫描 {len(subdirs)} 个子文件夹（排除 {excluded_count} 个）...")
        else:
            print(f"\n扫描 {len(subdirs)} 个子文件夹...")
    
    for subdir in filtered_subdirs:
        drivepi_dir = subdir / "drivepi"
        if drivepi_dir.exists() and drivepi_dir.is_dir():
            for pkl_file in drivepi_dir.glob("*.pkl"):
                pkl_files.append((pkl_file, subdir.name))
    
    if not pkl_files:
        print("未找到任何pkl文件")
        return
    
    print(f"找到 {len(pkl_files)} 个pkl文件\n")
    
    # 复制文件
    copied_count = 0
    skipped_count = 0
    
    for pkl_file, subdir_name in tqdm(pkl_files, desc="复制文件"):
        target_file = target_path / pkl_file.name
        
        if target_file.exists():
            print(f"警告: 目标文件已存在，跳过: {pkl_file.name}")
            skipped_count += 1
            continue
        
        try:
            shutil.copy2(str(pkl_file), str(target_file))
            copied_count += 1
        except Exception as e:
            print(f"错误: 复制文件失败 {pkl_file}: {e}")
            skipped_count += 1
    
    # 输出统计信息
    print(f"\n{'='*50}")
    print(f"成功复制: {copied_count} 个文件")
    if skipped_count > 0:
        print(f"跳过: {skipped_count} 个文件（目标文件已存在）")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(
        description="将所有子文件夹中drivepi目录下的pkl文件复制到指定目标文件夹"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="源目录路径（包含所有子文件夹的目录）"
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="目标目录路径（pkl文件将被复制到这里）"
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="+",
        default=None,
        help="要包含的子文件夹名（可以指定多个，如果指定则只处理这些文件夹）"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=[],
        help="要排除的子文件夹名（可以指定多个）"
    )
    
    args = parser.parse_args()
    
    copy_pkl_files(args.source_dir, args.target_dir, include_dirs=args.include, exclude_dirs=args.exclude)


if __name__ == "__main__":
    main()

