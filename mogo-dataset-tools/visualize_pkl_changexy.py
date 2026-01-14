#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化pkl文件：左图显示原图，右图显示BEV视角下的轨迹点
python /D2/NXT/vlm/miaobinqi/Senna/mogo_test/mogo-dataset-tools/visualize_pkl_changexy.py \
    --pkl-path /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags/20251210141637/drivepi \
    --save-dir /D2/NXT/vlm/miaobinqi/Senna/mogo_test/mogo_bag_vis_changexy/20251210141637 \
    --x-range -1 15 --y-range -8 8
"""

from __future__ import print_function
import sys
import os
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
import argparse
from tqdm import tqdm

def load_pkl(pkl_path):
    """加载pkl文件"""
    try:
        with open(pkl_path, 'rb') as f:
            # 尝试使用默认方式加载（Python 3格式）
            try:
                data = pickle.load(f)
            except (UnicodeDecodeError, TypeError):
                # 如果失败，尝试使用latin1编码（兼容Python 2格式）
                f.seek(0)  # 重置文件指针
                data = pickle.load(f, encoding='latin1')
        return data
    except Exception as e:
        print("[ERROR] 加载pkl文件失败：{}".format(e))
        return None

def extract_image_path(obj):
    """
    提取图像路径，兼容 StringWrapper 序列化后的 dict、bytes、str、以及带 .numpy() 的对象
    """
    if obj is None:
        return None

    # 兼容 StringWrapper __reduce__ 为 dict 的情况
    if isinstance(obj, dict):
        if '_string_wrapper_marker' in obj and '_bytes_value' in obj:
            obj = obj['_bytes_value']
        else:
            return None

    # 兼容带 .numpy() 的对象
    if hasattr(obj, 'numpy'):
        try:
            obj = obj.numpy()
        except Exception:
            pass

    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except Exception:
            return None

    if isinstance(obj, str):
        return obj

    return None

def load_image(image_path):
    """加载图像"""
    image_path = extract_image_path(image_path)

    if not image_path or not isinstance(image_path, str):
        return None
    
    if not os.path.exists(image_path):
        return None
    
    try:
        img = cv2.imread(image_path)
        if img is not None:
            # BGR转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print("[WARN] 加载图像失败 {}: {}".format(image_path, e))
        return None

def draw_bev_trajectory(data, ax, x_range=[-20, 20], y_range=[-1, 30]):
    """
    在BEV图上绘制轨迹点
    
    参数:
        data: pkl数据字典
        ax: matplotlib axes对象
        x_range: x轴范围，列表格式 [min, max]，例如 [0, 50]
        y_range: y轴范围，列表格式 [min, max]，例如 [0, 50]
    """
    # 设置坐标轴范围
    if x_range is not None and len(x_range) == 2:
        ax.set_xlim(x_range[0], x_range[1])
    else:
        # 默认范围
        ax.set_xlim(-25, 25)
    
    if y_range is not None and len(y_range) == 2:
        ax.set_ylim(y_range[0], y_range[1])
    else:
        # 默认范围
        ax.set_ylim(-25, 50)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    # 绘制坐标轴标签
    ax.set_xlabel('Forward X (m)', fontsize=10)
    ax.set_ylabel('Left Y (m)', fontsize=10)
    ax.set_title('BEV Trajectory', fontsize=12, fontweight='bold')
    
    # 绘制当前车辆位置（原点）
    vehicle_size = 0.5  # 车辆大小（米）
    vehicle_rect = patches.Rectangle(
        (-vehicle_size/2, -vehicle_size/2), 
        vehicle_size, vehicle_size,
        angle=0, 
        color='blue', 
        alpha=0.6
    )
    ax.add_patch(vehicle_rect)
    
    # 绘制未来轨迹（fur_x, fur_y）——先交换 x/y 再可视化
    if 'fur_x' in data and 'fur_y' in data:
        fur_x = data['fur_x']
        fur_y = data['fur_y']
        
        # fur_x和fur_y的shape是(1, 7)
        if fur_x.shape == (1, 7) and fur_y.shape == (1, 7):
            # 交换：原始 (x,y) -> (y,x)
            x_points = fur_y[0]
            y_points = fur_x[0]
            
            # 绘制轨迹线
            ax.plot(x_points, y_points, 'o-', color='green', linewidth=2, 
                   markersize=6, alpha=0.8, label='Future Trajectory (swap xy)')
    
    # 添加图例
    ax.legend(loc='upper right', fontsize=8)
    
    # 添加信息文本
    info_text = []
    if 'status' in data:
        info_text.append("Status: {}".format(data['status']))
    if 'his_speed' in data and len(data['his_speed']) > 0:
        current_speed = data['his_speed'][-1]
        info_text.append("Speed: {:.2f} m/s".format(current_speed))
    
    if info_text:
        ax.text(0.02, 0.98, '\n'.join(info_text), 
               transform=ax.transAxes, 
               fontsize=9,
               verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def visualize_pkl(pkl_path, save_path=None, x_range=None, y_range=None):
    """
    可视化pkl文件
    
    参数:
        pkl_path: pkl文件路径
        save_path: 保存图像路径（可选）
        x_range: x轴范围，列表格式 [min, max]，例如 [0, 50]
        y_range: y轴范围，列表格式 [min, max]，例如 [0, 50]
    """
    # 加载pkl数据
    data = load_pkl(pkl_path)
    if data is None:
        return
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：显示原图（优先使用front相机）
    img = None
    if 'his_cam_front' in data:
        img_path = data['his_cam_front']
        img = load_image(img_path)
    
    if img is None:
        # 如果front图像不存在，尝试其他相机
        for cam_key in ['his_cam_front_left', 'his_cam_front_right', 'his_cam_back']:
            if cam_key in data:
                img_path = data[cam_key]
                img = load_image(img_path)
                if img is not None:
                    break
    
    if img is not None:
        ax1.imshow(img)
        ax1.set_title('Camera Image (Front)', fontsize=12, fontweight='bold')
        ax1.axis('off')
    else:
        ax1.text(0.5, 0.5, 'Image Not Found', 
                ha='center', va='center', fontsize=14)
        ax1.set_title('Camera Image', fontsize=12)
        ax1.axis('off')
    
    # 右图：显示BEV轨迹
    draw_bev_trajectory(data, ax2, x_range=x_range, y_range=y_range)
    
    # 设置总标题
    pkl_filename = os.path.basename(pkl_path)
    fig.suptitle('PKL Visualization: {}'.format(pkl_filename), fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        # print("[INFO] 图像已保存到: {}".format(save_path))
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='可视化pkl文件')
    parser.add_argument('--pkl-path', type=str, dest='pkl_path', required=True,
                       help='pkl文件路径或包含pkl文件的目录')
    parser.add_argument('--save-dir', type=str, default=None, 
                       help='保存可视化图像的目录（可选）')
    parser.add_argument('--index', type=int, default=None,
                       help='如果输入是目录，指定要可视化的文件索引（从0开始）')
    parser.add_argument('--x-range', type=float, nargs=2, default=None,
                       metavar=('MIN', 'MAX'),
                       help='x轴范围，例如: --x-range 0 50')
    parser.add_argument('--y-range', type=float, nargs=2, default=None,
                       metavar=('MIN', 'MAX'),
                       help='y轴范围，例如: --y-range 0 50')
    
    args = parser.parse_args()
    
    pkl_path = args.pkl_path
    x_range = list(args.x_range) if args.x_range else None
    y_range = list(args.y_range) if args.y_range else None
    
    if os.path.isfile(pkl_path):
        # 单个文件
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            pkl_name = os.path.basename(pkl_path)
            save_path = os.path.join(args.save_dir, pkl_name.replace('.pkl', '.png'))
        visualize_pkl(pkl_path, save_path, x_range=x_range, y_range=y_range)
    
    elif os.path.isdir(pkl_path):
        # 目录：查找所有pkl文件
        pkl_files = []
        for root, dirs, files in os.walk(pkl_path):
            for file in files:
                if file.endswith('.pkl'):
                    pkl_files.append(os.path.join(root, file))
        
        pkl_files.sort()
        
        if not pkl_files:
            print("[ERROR] 未找到pkl文件")
            return
        
        print("[INFO] 找到 {} 个pkl文件".format(len(pkl_files)))
        
        if args.index is not None:
            # 可视化指定索引的文件
            if 0 <= args.index < len(pkl_files):
                pkl_file = pkl_files[args.index]
                save_path = None
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    pkl_name = os.path.basename(pkl_file)
                    save_path = os.path.join(args.save_dir, pkl_name.replace('.pkl', '.png'))
                visualize_pkl(pkl_file, save_path, x_range=x_range, y_range=y_range)
            else:
                print("[ERROR] 索引超出范围，有效范围: 0-{}".format(len(pkl_files) - 1))
        else:
            # 可视化所有文件
            print("[INFO] 开始可视化所有 {} 个pkl文件...".format(len(pkl_files)))
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
            
            for pkl_file in tqdm(pkl_files, desc="处理pkl文件", unit="个"):
                save_path = None
                if args.save_dir:
                    pkl_name = os.path.basename(pkl_file)
                    save_path = os.path.join(args.save_dir, pkl_name.replace('.pkl', '.png'))
                visualize_pkl(pkl_file, save_path, x_range=x_range, y_range=y_range)
            
            print("[INFO] 所有文件处理完成！")
    else:
        print("[ERROR] 路径不存在: {}".format(pkl_path))

if __name__ == '__main__':
    main()

