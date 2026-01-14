#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
# 可视化单个bag文件夹（输出到bag目录）
python2.7 visualize_utm_trajectory.py /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags/20251210141637

# 指定输出文件夹
python2.7 visualize_utm_trajectory.py /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags/20251210141637 /D2/NXT/vlm/miaobinqi/Senna/mogo_test/mogo-dataset-tools/output
"""
from __future__ import print_function
import sys
import os
import numpy as np
import rosbag
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from proto.common.localization_pb2 import Localization

# 计算header时间戳（秒）
def calculate_header_timestamp(header):
    """Calculate ROS Header timestamp in seconds."""
    sec = float(header.stamp.sec)
    nsec = float(header.stamp.nsec) if hasattr(header.stamp, 'nsec') else 0.0
    return sec + nsec * 1.0e-9

def parse_bag_utm_coordinates(bag_dir):
    """
    从bag文件夹中解析UTM坐标
    
    参数:
        bag_dir: bag文件夹路径
    
    返回:
        utm_x_list: UTM X坐标列表（Easting，东向）
        utm_y_list: UTM Y坐标列表（Northing，北向）
        timestamps: 时间戳列表
    
    注意:
        UTM坐标系定义：
        - X轴 = 东向（Easting），向东为正
        - Y轴 = 北向（Northing），向北为正
    """
    # 查找bag文件
    bags_file_master = None
    for filename in os.listdir(bag_dir):
        if filename.endswith(".bag") and "-rosmaster-" in filename:
            bags_file_master = os.path.join(bag_dir, filename)
            break
    
    if bags_file_master is None:
        raise IOError("未找到master bag文件")
    
    print("[INFO] 读取bag文件: {}".format(bags_file_master))
    
    utm_x_list = []
    utm_y_list = []
    timestamps = []
    
    try:
        master_bag = rosbag.Bag(bags_file_master, "r")
        
        print("[INFO] 解析定位数据...")
        for topic, msg, t in master_bag.read_messages(topics=["/localization/global"]):
            try:
                loc_data = Localization()
                loc_data.ParseFromString(msg.data)
                ts = calculate_header_timestamp(loc_data.header)
                
                # 获取UTM坐标
                # UTM坐标系定义：X轴 = 东向（Easting），Y轴 = 北向（Northing）
                if hasattr(loc_data, 'position') and loc_data.position:
                    utm_x = loc_data.position.x  # UTM Easting (东向)
                    utm_y = loc_data.position.y  # UTM Northing (北向)
                # else:
                    # fallback: 使用longitude/latitude
                    # utm_x = loc_data.longitude if hasattr(loc_data, 'longitude') else 0.0
                    # utm_y = loc_data.latitude if hasattr(loc_data, 'latitude') else 0.0
                
                utm_x_list.append(utm_x)
                utm_y_list.append(utm_y)
                timestamps.append(ts)
                
            except Exception as e:
                print("[WARN] 解析定位消息失败：{}".format(e))
                continue
        
        master_bag.close()
        
        print("[INFO] 解析完成：共 {} 个定位点".format(len(utm_x_list)))
        
    except Exception as e:
        print("[ERROR] 读取bag文件失败：{}".format(e))
        raise
    
    return utm_x_list, utm_y_list, timestamps

def visualize_utm_trajectory(utm_x_list, utm_y_list, output_path=None):
    """
    可视化UTM坐标轨迹
    
    参数:
        utm_x_list: UTM X坐标列表
        utm_y_list: UTM Y坐标列表
        output_path: 输出图片路径，如果为None则显示
    """
    if len(utm_x_list) == 0 or len(utm_y_list) == 0:
        print("[WARN] 没有数据可可视化")
        return
    
    # 转换为numpy数组
    utm_x = np.array(utm_x_list)
    utm_y = np.array(utm_y_list)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制轨迹线
    ax.plot(utm_x, utm_y, 'b-', linewidth=1.5, alpha=0.7, label='Trajectory')
    
    # 绘制起点
    if len(utm_x) > 0:
        ax.plot(utm_x[0], utm_y[0], 'go', markersize=10, label='Start', zorder=5)
    
    # 绘制终点
    if len(utm_x) > 0:
        ax.plot(utm_x[-1], utm_y[-1], 'ro', markersize=10, label='End', zorder=5)
    
    # 每隔一定数量的点绘制一个标记点（避免点太密集）
    step = max(1, len(utm_x) // 50)  # 最多显示50个点
    if step > 1:
        ax.plot(utm_x[::step], utm_y[::step], 'b.', markersize=3, alpha=0.5, label='Sampled Points')
    
    # 设置坐标轴标签
    ax.set_xlabel('UTM X (Easting, m)', fontsize=12)
    ax.set_ylabel('UTM Y (Northing, m)', fontsize=12)
    ax.set_title('UTM Trajectory Visualization', fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例（右下角）
    ax.legend(loc='lower right')
    
    # 设置等比例坐标轴（保持轨迹形状）
    ax.set_aspect('equal', adjustable='box')
    
    # 添加统计信息
    if len(utm_x) > 0:
        total_distance = 0.0
        for i in range(1, len(utm_x)):
            dist = np.sqrt((utm_x[i] - utm_x[i-1])**2 + (utm_y[i] - utm_y[i-1])**2)
            total_distance += dist
        
        info_text = "Total Points: {}\nTotal Distance: {:.2f} m\nX Range: [{:.2f}, {:.2f}]\nY Range: [{:.2f}, {:.2f}]".format(
            len(utm_x),
            total_distance,
            np.min(utm_x), np.max(utm_x),
            np.min(utm_y), np.max(utm_y)
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print("[INFO] 图片已保存到: {}".format(output_path))
    else:
        plt.show()
    
    plt.close()

def sample_utm_2hz(utm_x_list, utm_y_list, timestamps):
    """
    对UTM坐标进行2Hz采样
    
    参数:
        utm_x_list: UTM X坐标列表
        utm_y_list: UTM Y坐标列表
        timestamps: 时间戳列表
    
    返回:
        sampled_x: 采样后的X坐标数组
        sampled_y: 采样后的Y坐标数组
    """
    if len(utm_x_list) == 0 or len(utm_y_list) == 0 or len(timestamps) == 0:
        return np.array([]), np.array([])
    
    # 转换为numpy数组
    utm_x = np.array(utm_x_list)
    utm_y = np.array(utm_y_list)
    ts = np.array(timestamps)
    
    # 2Hz采样：每0.5秒采样一次
    sample_interval = 0.5  # 秒
    start_time = ts[0]
    end_time = ts[-1]
    
    # 采样索引
    sampled_indices = []
    current_time = start_time
    
    while current_time <= end_time:
        # 找到最接近当前时间的索引
        idx = np.argmin(np.abs(ts - current_time))
        if idx not in sampled_indices:  # 避免重复
            sampled_indices.append(idx)
        current_time += sample_interval
    
    # 确保包含最后一个点
    if len(sampled_indices) == 0 or sampled_indices[-1] != len(utm_x) - 1:
        sampled_indices.append(len(utm_x) - 1)
    
    sampled_indices = sorted(list(set(sampled_indices)))  # 去重并排序
    
    # 获取采样后的数据
    sampled_x = utm_x[sampled_indices]
    sampled_y = utm_y[sampled_indices]
    
    return sampled_x, sampled_y

def visualize_utm_trajectory_2hz(utm_x_list, utm_y_list, timestamps, output_path=None):
    """
    可视化UTM坐标轨迹（2Hz采样）
    
    参数:
        utm_x_list: UTM X坐标列表
        utm_y_list: UTM Y坐标列表
        timestamps: 时间戳列表
        output_path: 输出图片路径，如果为None则显示
    """
    if len(utm_x_list) == 0 or len(utm_y_list) == 0 or len(timestamps) == 0:
        print("[WARN] 没有数据可可视化")
        return
    
    # 使用采样函数进行2Hz采样
    sampled_x, sampled_y = sample_utm_2hz(utm_x_list, utm_y_list, timestamps)
    
    print("[INFO] 2Hz采样：原始点数 {}，采样后点数 {}".format(len(utm_x_list), len(sampled_x)))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制轨迹线
    ax.plot(sampled_x, sampled_y, 'b-', linewidth=1.5, alpha=0.7, label='Trajectory (2Hz)')
    
    # 绘制起点
    if len(sampled_x) > 0:
        ax.plot(sampled_x[0], sampled_y[0], 'go', markersize=10, label='Start', zorder=5)
    
    # 绘制终点
    if len(sampled_x) > 0:
        ax.plot(sampled_x[-1], sampled_y[-1], 'ro', markersize=10, label='End', zorder=5)
    
    # 绘制采样点
    ax.plot(sampled_x, sampled_y, 'b.', markersize=4, alpha=0.6, label='Sampled Points (2Hz)')
    
    # 设置坐标轴标签
    ax.set_xlabel('UTM X (Easting, m)', fontsize=12)
    ax.set_ylabel('UTM Y (Northing, m)', fontsize=12)
    ax.set_title('UTM Trajectory Visualization (2Hz Sampling)', fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例（右下角）
    ax.legend(loc='lower right')
    
    # 设置等比例坐标轴（保持轨迹形状）
    ax.set_aspect('equal', adjustable='box')
    
    # 添加统计信息
    if len(sampled_x) > 0:
        total_distance = 0.0
        for i in range(1, len(sampled_x)):
            dist = np.sqrt((sampled_x[i] - sampled_x[i-1])**2 + (sampled_y[i] - sampled_y[i-1])**2)
            total_distance += dist
        
        info_text = "Total Points: {} (Original: {})\nTotal Distance: {:.2f} m\nX Range: [{:.2f}, {:.2f}]\nY Range: [{:.2f}, {:.2f}]".format(
            len(sampled_x),
            len(utm_x_list),
            total_distance,
            np.min(sampled_x), np.max(sampled_x),
            np.min(sampled_y), np.max(sampled_y)
        )
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print("[INFO] 2Hz采样图片已保存到: {}".format(output_path))
    else:
        plt.show()
    
    plt.close()

def main():
    print("[INFO] UTM坐标轨迹可视化工具")
    print("[INFO] 使用方法: python2.7 visualize_utm_trajectory.py <bag_dir> [output_dir]")
    
    if len(sys.argv) < 2:
        print("[ERROR] 请提供bag文件夹路径")
        sys.exit(1)
    
    bag_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.isdir(bag_dir):
        print("[ERROR] 目录不存在：{}".format(bag_dir))
        sys.exit(1)
    
    # 获取输入文件夹的最后一级目录名
    bag_name = os.path.basename(bag_dir.rstrip('/'))
    
    # 确定输出目录
    if output_dir is None:
        output_dir = bag_dir
    else:
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError:
                print("[ERROR] 无法创建输出目录：{}".format(output_dir))
                sys.exit(1)
    
    try:
        # 解析UTM坐标
        utm_x_list, utm_y_list, timestamps = parse_bag_utm_coordinates(bag_dir)
        
        if len(utm_x_list) == 0:
            print("[ERROR] 未找到定位数据")
            sys.exit(1)
        
        # 生成输出文件路径（使用输入文件夹的最后一级目录名作为前缀）
        utm_x_npy_path = os.path.join(output_dir, "{}_utm_x.npy".format(bag_name))
        utm_y_npy_path = os.path.join(output_dir, "{}_utm_y.npy".format(bag_name))
        output_path_10hz = os.path.join(output_dir, "{}_utm_trajectory_10hz.png".format(bag_name))
        output_path_2hz = os.path.join(output_dir, "{}_utm_trajectory_2hz.png".format(bag_name))
        
        # 对UTM坐标进行2Hz采样
        sampled_x, sampled_y = sample_utm_2hz(utm_x_list, utm_y_list, timestamps)
        print("[INFO] 2Hz采样：原始点数 {}，采样后点数 {}".format(len(utm_x_list), len(sampled_x)))
        
        # 保存2Hz采样后的数据为npy文件
        np.save(utm_x_npy_path, sampled_x)
        np.save(utm_y_npy_path, sampled_y)
        print("[INFO] UTM坐标（2Hz采样）已保存到npy文件:")
        print("[INFO]   X坐标: {}".format(utm_x_npy_path))
        print("[INFO]   Y坐标: {}".format(utm_y_npy_path))
        
        # 生成10Hz原始数据可视化
        visualize_utm_trajectory(utm_x_list, utm_y_list, output_path_10hz)
        
        # 生成2Hz采样数据可视化
        visualize_utm_trajectory_2hz(utm_x_list, utm_y_list, timestamps, output_path_2hz)
        
        print("[INFO] 可视化完成！")
        print("[INFO] 10Hz轨迹图: {}".format(output_path_10hz))
        print("[INFO] 2Hz轨迹图: {}".format(output_path_2hz))
        
    except Exception as e:
        print("[ERROR] 处理失败：{}".format(e))
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

