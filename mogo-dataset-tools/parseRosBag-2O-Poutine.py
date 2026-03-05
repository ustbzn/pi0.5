#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
"""
python2.7 /D2/NXT/vlm/miaobinqi/Senna/mogo_test/mogo-dataset-tools/parseRosBag-2O_py27.py /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags
"""
from __future__ import print_function
import sys
import os
import pickle
import json
import bisect
import re
import numpy as np
import rosbag
import base64
from proto.common.localization_pb2 import Localization
from proto.common.trajectory_pb2 import Trajectory, TrajectoryPoint
from proto.common.sensor_meta_pb2 import RegisteredSensor
from google.protobuf import text_format

# 计算header时间戳（秒）
def calculate_header_timestamp(header):
    """Calculate ROS Header timestamp in seconds."""
    sec = float(header.stamp.sec)
    nsec = float(header.stamp.nsec) if hasattr(header.stamp, 'nsec') else 0.0
    return sec + nsec * 1.0e-9

def serialize_for_json(obj):
    """
    将数据序列化为JSON兼容格式
    处理numpy数组和bytes类型
    """
    if isinstance(obj, np.ndarray):
        return {
            '__numpy_array__': True,
            'data': obj.tolist(),
            'dtype': str(obj.dtype),
            'shape': obj.shape
        }
    elif isinstance(obj, (bytes, bytearray)):
        return {
            '__bytes__': True,
            'data': base64.b64encode(obj).decode('utf-8')
        }
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, dict):
        return {key: serialize_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    else:
        return obj

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """将四元数转换为旋转矩阵"""
    # 归一化四元数
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm > 0:
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # 转换为旋转矩阵
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R

def build_rotation_matrix_from_yaw(yaw):
    """
    从yaw角度构建旋转矩阵（绕Z轴旋转）
    
    参数:
        yaw: 航向角（弧度），从北向（Y轴正方向）逆时针旋转的角度
    
    返回:
        3x3旋转矩阵
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    # 右手坐标系：X轴指向车头前方，Y轴指向车身左侧，Z轴垂直向上
    # 从世界坐标系（UTM：X东，Y北）到自车坐标系（X前，Y左）的旋转矩阵
    R = np.array([
        [sin_yaw,  cos_yaw, 0.0],  # X轴：前进方向
        [-cos_yaw, sin_yaw, 0.0],  # Y轴：左侧方向
        [0.0,      0.0,     1.0]   # Z轴：垂直向上
    ])
    
    return R

def transform_global_to_local(ego_global_position, global_points, ego_rotation_matrix=None):
    """
    将全局坐标系下的点转换为局部坐标系（以ego为中心）
    
    参数:
        ego_global_position: ego在全局坐标系下的位置 [x, y, z]
        global_points: 全局坐标系下的点列表，每个点为 [x, y, z] 或 (x, y, z)
        ego_rotation_matrix: ego的旋转矩阵 (3x3)，如果为None则假设无旋转
    
    返回:
        local_points: 局部坐标系下的点列表
    """
    ego_global_position = np.array(ego_global_position)
    
    # 如果ego_rotation_matrix为None，使用单位矩阵（无旋转）
    if ego_rotation_matrix is None:
        R = np.eye(3)
    else:
        R = np.array(ego_rotation_matrix)
        # 确保是3x3矩阵
        if R.shape != (3, 3):
            R = np.eye(3)
    
    # 计算旋转矩阵的转置（从全局到局部的旋转）
    R_T = R.T
    
    local_points = []
    for point in global_points:
        point_global = np.array(point)
        
        # 平移到以ego为原点
        point_relative = point_global - ego_global_position
        
        # 应用旋转（从全局坐标系旋转到局部坐标系）
        point_local = R_T.dot(point_relative)
        
        local_points.append(point_local.tolist())
    
    return local_points

class BagDataHandler:
    """
    负责加载 rosbag 文件，并提供解析相机与定位数据的接口。
    """

    def __init__(self, bag_dir, output_dir=None, work_dir=None):
        self.bag_dir = bag_dir
        self.work_dir = work_dir or os.getcwd()
        # 输出目录，如果不指定则使用bag_dir
        self.output_dir = output_dir or bag_dir

        # bag 文件路径
        self.bags_file_master = None
        self.bags_file_105 = None

        # 已打开的 bag
        self.master_bag = None
        self.slave_105_bag = None

        # 相机相关
        self.camera_cache = {}

        # 定位相关
        self.valid_timestamps = []
        self.valid_locations = []
        
        # 2Hz采样后的定位数据
        self.sampled_2hz_timestamps = []
        self.sampled_2hz_locations = []
        self.sampled_2hz_indices = []  # 原始索引，用于查找原始数据

        # 轨迹相关
        # self.trajectory_timestamps = []
        # self.trajectory_data = []  # 存储Trajectory对象

        # 注意：ego旋转矩阵将从定位数据的yaw角度动态构建，不在这里预加载

        # json文件夹，用于保存每个sample的json文件，命名为poutine_base_json
        self.json_samples_dir = os.path.join(self.output_dir, "poutine_base_json")
        if not os.path.exists(self.json_samples_dir):
            try:
                os.makedirs(self.json_samples_dir)
            except OSError:
                if not os.path.isdir(self.json_samples_dir):
                    raise

        self._check_and_find_bags()
        self._open_bags()

    # ---------------- internal helpers ----------------
    def _check_and_find_bags(self):
        """检查目录并查找 master/slave bag 文件。"""
        if not os.path.isdir(self.bag_dir):
            raise IOError("目录{}无效".format(self.bag_dir))

        for filename in os.listdir(self.bag_dir):
            if filename.endswith(".bag"):
                fp = os.path.join(self.bag_dir, filename)
                if "-rosmaster-" in filename:
                    self.bags_file_master = fp
                elif "-rosslave-" in filename:
                    self.bags_file_105 = fp

        if not self.bags_file_master or not self.bags_file_105:
            raise IOError("未找到指定Bag文件")

        print("[INFO] Master Bag路径：{}".format(self.bags_file_master))
        print("[INFO] Slave105 Bag路径：{}".format(self.bags_file_105))

    def _open_bags(self):
        """打开 master 与 slave bag。"""
        try:
            self.master_bag = rosbag.Bag(self.bags_file_master, "r")
        except Exception as e:
            raise RuntimeError("打开Master Bag失败：{}".format(e))

        try:
            self.slave_105_bag = rosbag.Bag(self.bags_file_105, "r")
        except Exception as e:
            if self.master_bag is not None:
                self.master_bag.close()
            raise RuntimeError("打开Slave105 Bag失败：{}".format(e))

    # ---------------- public APIs ----------------
    def load_existing_cameras(self):
        """
        从已存在的相机图像目录加载图像路径和时间戳。
        相机字段：back、front、left、left_back、right、right_back

        结果保存在：
            self.camera_cache
        """
        # 相机字段映射
        camera_keys = ["back", "front", "left", "left_back", "right", "right_back"]
        
        print("[INFO] 从已存在的相机目录加载图像...")
        for key in camera_keys:
            camera_dir = os.path.join(self.output_dir, key)
            if not os.path.exists(camera_dir):
                print("[WARN] 相机目录不存在：{}".format(camera_dir))
                continue
            
            # 扫描目录中的所有jpg文件
            image_files = []
            for filename in os.listdir(camera_dir):
                if filename.endswith('.jpg'):
                    filepath = os.path.join(camera_dir, filename)
                    # 从文件名提取时间戳（文件名格式：timestamp.jpg）
                    try:
                        timestamp = float(filename[:-4])  # 去掉.jpg后缀
                        image_files.append((timestamp, filepath))
                    except ValueError:
                        print("[WARN] 无法解析文件名时间戳：{}".format(filename))
                        continue
            
            # 按时间戳排序
            image_files.sort(key=lambda x: x[0])
            
            if image_files:
                self.camera_cache[key] = {
                    "times": [item[0] for item in image_files],
                    "paths": [item[1] for item in image_files]
                }
                print("[INFO] 相机 {}: 加载了 {} 张图像".format(key, len(image_files)))
            else:
                print("[WARN] 相机 {}: 未找到图像文件".format(key))
        
        print("[INFO] 相机图像加载完成：共{}路相机".format(len(self.camera_cache)))

    def parse_localization(self):
        """
        解析定位数据并预处理时间戳。

        结果保存在：
            self.valid_timestamps
            self.valid_locations
        """
        assert self.master_bag is not None

        self.valid_timestamps = []
        self.valid_locations = []

        print("[INFO] 解析定位数据并预处理时间戳...")
        for topic, msg, t in self.master_bag.read_messages(topics=["/localization/global"]):
            try:
                loc_data = Localization()
                loc_data.ParseFromString(msg.data)
                ts = calculate_header_timestamp(loc_data.header)
                self.valid_timestamps.append(ts)
                self.valid_locations.append(loc_data)
            except Exception as e:
                print("[WARN] 解析定位消息失败：{}".format(e))

        if not self.valid_timestamps:
            raise RuntimeError("定位数据中无有效时间戳信息")
        print("[INFO] 定位数据处理完成：有效数据{}条".format(len(self.valid_timestamps)))
    
    def sample_localization_2hz(self, sample_rate_hz=2.0):
        """
        对定位数据进行2Hz采样
        
        参数:
            sample_rate_hz: 采样频率，默认2Hz（每0.5秒一帧）
        
        结果保存在:
            self.sampled_2hz_timestamps
            self.sampled_2hz_locations
            self.sampled_2hz_indices
        """
        if not self.valid_timestamps:
            print("[WARN] 定位数据为空，无法进行2Hz采样")
            return
        
        self.sampled_2hz_timestamps = []
        self.sampled_2hz_locations = []
        self.sampled_2hz_indices = []
        
        # 2Hz采样：每0.5秒采样一次
        sample_interval = 1.0 / sample_rate_hz  # 0.5秒
        start_time = self.valid_timestamps[0]
        end_time = self.valid_timestamps[-1]
        
        current_time = start_time
        
        print("[INFO] 开始2Hz采样定位数据，采样频率: {}Hz，时间范围: {:.3f} - {:.3f}".format(
            sample_rate_hz, start_time, end_time))
        
        while current_time <= end_time:
            # 找到最接近当前时间的定位数据索引
            idx = bisect.bisect_left(self.valid_timestamps, current_time)
            candidates = []
            if idx == 0:
                candidates.append(0)
            elif idx == len(self.valid_timestamps):
                candidates.append(len(self.valid_timestamps) - 1)
            else:
                candidates.extend([idx - 1, idx])
            
            # 选择最接近的索引
            best_idx = None
            best_diff = float('inf')
            for ci in candidates:
                diff = abs(self.valid_timestamps[ci] - current_time)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = ci
            
            # 如果时间差在合理范围内（0.25秒内），添加到采样结果
            if best_idx is not None and best_diff <= 0.25:
                # 避免重复添加相同的索引
                if not self.sampled_2hz_indices or self.sampled_2hz_indices[-1] != best_idx:
                    self.sampled_2hz_timestamps.append(self.valid_timestamps[best_idx])
                    self.sampled_2hz_locations.append(self.valid_locations[best_idx])
                    self.sampled_2hz_indices.append(best_idx)
            
            # 移动到下一个采样时间点
            current_time += sample_interval
        
        # 确保包含最后一个点
        # if self.sampled_2hz_indices and self.sampled_2hz_indices[-1] != len(self.valid_timestamps) - 1:
        #     last_idx = len(self.valid_timestamps) - 1
        #     self.sampled_2hz_timestamps.append(self.valid_timestamps[last_idx])
        #     self.sampled_2hz_locations.append(self.valid_locations[last_idx])
        #     self.sampled_2hz_indices.append(last_idx)
        
        print("[INFO] 2Hz采样完成：原始点数 {}，采样后点数 {}".format(
            len(self.valid_timestamps), len(self.sampled_2hz_timestamps)))

    def parse_trajectory(self):
        """
        解析轨迹数据。

        结果保存在：
            self.trajectory_timestamps
            self.trajectory_data
        """
        assert self.master_bag is not None

        self.trajectory_timestamps = []
        self.trajectory_data = []

        print("[INFO] 解析轨迹数据...")
        # 尝试常见的trajectory topic名称
        trajectory_topics = ["/planning/trajectory", "/trajectory", "/planning/trajectory_plan"]
        
        found_topic = None
        master_topics = self.master_bag.get_type_and_topic_info()[1].keys()
        for topic_name in trajectory_topics:
            if topic_name in master_topics:
                found_topic = topic_name
                break
        
        if found_topic is None:
            # 查找包含trajectory的topic
            for topic_name in master_topics:
                if "trajectory" in topic_name.lower():
                    found_topic = topic_name
                    break
        
        if found_topic is None:
            print("[WARN] 未找到轨迹topic，将使用空轨迹数据")
            return

        print("[INFO] 使用轨迹topic: {}".format(found_topic))
        for topic, msg, t in self.master_bag.read_messages(topics=[found_topic]):
            try:
                traj_data = Trajectory()
                traj_data.ParseFromString(msg.data)
                ts = calculate_header_timestamp(traj_data.header)
                self.trajectory_timestamps.append(ts)
                self.trajectory_data.append(traj_data)
            except Exception as e:
                print("[WARN] 解析轨迹消息失败：{}".format(e))

        print("[INFO] 轨迹数据处理完成：有效数据{}条".format(len(self.trajectory_timestamps)))

    def align_camera_cache_by_front(self, max_delta=0.05):
        """
        根据 front 时间戳对齐其他视角 camera 的图片路径。

        功能：
        - 以 front 相机的时间戳序列为基准；
        - 对每个 front 时间戳，在其他相机中寻找时间差 <= max_delta(秒) 的最近帧；
        - 将 front 及所有匹配到的其他视角组成一个 dict：{视角名: path}；
        - 存入新的容器：key 为 front 的时间戳，value 为该时间戳下所有视角的 path dict。

        返回：
            Dict[front_ts, {cam_key: path}]
        """
        if "front" not in self.camera_cache or not self.camera_cache["front"]["times"]:
            print("[WARN] align_camera_cache_by_front: front 相机数据为空")
            self.camera_group_by_front = {}
            return self.camera_group_by_front

        front_times = self.camera_cache["front"]["times"]
        front_paths = self.camera_cache["front"]["paths"]

        # 先确保每路相机的 times 与 paths 一一对应且按时间升序
        for key, data in self.camera_cache.items():
            times = data.get("times", [])
            paths = data.get("paths", [])
            if len(times) != len(paths):
                print("[WARN] 相机 {} 的 times 与 paths 长度不一致，跳过排序".format(key))
                continue
            if not times:
                continue
            # 如果不是升序，则按时间重新排序
            if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
                paired = sorted(zip(times, paths), key=lambda x: x[0])
                data["times"] = [p[0] for p in paired]
                data["paths"] = [p[1] for p in paired]

        grouped = {}

        # 遍历 front 的时间轴，对其它相机做 max_delta 内的最近匹配
        for t_front, path_front in zip(front_times, front_paths):
            # 第一个一定是 front 自己
            paths_at_t = {"front": path_front}

            for cam_key, data in self.camera_cache.items():
                if cam_key == "front":
                    continue
                times = data.get("times", [])
                paths = data.get("paths", [])
                if not times or len(times) != len(paths):
                    continue

                # 二分查找最接近 t_front 的索引
                idx = bisect.bisect_left(times, t_front)
                candidates = []
                if idx == 0:
                    candidates.append(0)
                elif idx == len(times):
                    candidates.append(len(times) - 1)
                else:
                    candidates.extend([idx - 1, idx])

                best_idx = None
                best_diff = float("inf")
                for ci in candidates:
                    diff = abs(times[ci] - t_front)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = ci

                if best_idx is not None and best_diff <= max_delta:
                    paths_at_t[cam_key] = paths[best_idx]

            grouped[t_front] = paths_at_t

        self.camera_group_by_front = grouped
        print("[INFO] 对齐相机帧完成：front 时间戳 {} 个".format(len(grouped)))
        return grouped

    def _find_nearest_loc_idx(self, target_ts, max_delta=0.1, use_2hz_sampled=True):
        """
        查找最接近目标时间戳的定位数据索引
        
        参数:
            target_ts: 目标时间戳
            max_delta: 最大时间差
            use_2hz_sampled: 是否使用2Hz采样后的数据（默认True）
        
        返回:
            (索引, 时间差) 或 (None, 时间差)
        """
        # 优先使用2Hz采样后的数据
        if use_2hz_sampled and self.sampled_2hz_timestamps:
            timestamps = self.sampled_2hz_timestamps
        else:
            timestamps = self.valid_timestamps
        
        if not timestamps:
            return None, float('inf')
        
        idx = bisect.bisect_left(timestamps, target_ts)
        candidates = []
        if idx == 0:
            candidates.append(0)
        elif idx == len(timestamps):
            candidates.append(len(timestamps) - 1)
        else:
            candidates.extend([idx - 1, idx])
        
        best_idx = None
        best_diff = float("inf")
        for ci in candidates:
            diff = abs(timestamps[ci] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = ci
        
        if best_idx is not None and best_diff <= max_delta:
            return best_idx, best_diff
        return None, best_diff

    def _find_nearest_traj_idx(self, target_ts, max_delta=0.1):
        """查找最接近目标时间戳的轨迹数据索引"""
        if not self.trajectory_timestamps:
            return None, float('inf')
        
        idx = bisect.bisect_left(self.trajectory_timestamps, target_ts)
        candidates = []
        if idx == 0:
            candidates.append(0)
        elif idx == len(self.trajectory_timestamps):
            candidates.append(len(self.trajectory_timestamps) - 1)
        else:
            candidates.extend([idx - 1, idx])
        
        best_idx = None
        best_diff = float("inf")
        for ci in candidates:
            diff = abs(self.trajectory_timestamps[ci] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = ci
        
        if best_idx is not None and best_diff <= max_delta:
            return best_idx, best_diff
        return None, best_diff

    def _get_camera_path(self, cam_key, target_ts, max_delta=0.05):
        """获取指定时间戳的相机图像路径"""
        if cam_key not in self.camera_cache:
            return None
        
        times = self.camera_cache[cam_key]["times"]
        paths = self.camera_cache[cam_key]["paths"]
        
        if not times:
            return None
        
        idx = bisect.bisect_left(times, target_ts)
        candidates = []
        if idx == 0:
            candidates.append(0)
        elif idx == len(times):
            candidates.append(len(times) - 1)
        else:
            candidates.extend([idx - 1, idx])
        
        best_idx = None
        best_diff = float("inf")
        for ci in candidates:
            diff = abs(times[ci] - target_ts)
            if diff < best_diff:
                best_diff = diff
                best_idx = ci
        
        if best_idx is not None and best_diff <= max_delta:
            return paths[best_idx]
        return None

    def generate_pkl_samples(self, sample_rate_hz=2.0):
        """
        按照2Hz采样频率生成pkl样本。
        
        参数:
            sample_rate_hz: 采样频率，默认2Hz（每0.5秒一帧）
        """
        if not hasattr(self, "camera_group_by_front") or not self.camera_group_by_front:
            print("[WARN] generate_pkl_samples: camera_group_by_front 为空")
            return

        if not self.valid_timestamps:
            print("[WARN] generate_pkl_samples: 定位数据为空")
            return

        # 获取scene_name（从bag_dir的basename获取）
        scene_name = os.path.basename(self.bag_dir)

        # 提取scene ID（用于status字段，保持原有逻辑）
        bag_name = os.path.basename(self.bags_file_master)
        if bag_name.endswith('.bag'):
            bag_name = bag_name[:-4]
        scene_id = "0001"  # 默认值
        if '-' in bag_name:
            parts = bag_name.split('-')
            first_part = parts[0]
            if first_part.isdigit():
                scene_id = first_part.zfill(4)
            elif 'scene' in first_part.lower():
                match = re.search(r'(\d+)', first_part)
                if match:
                    scene_id = match.group(1).zfill(4)

        # 使用2Hz采样后的定位数据时间戳作为基准
        if not self.sampled_2hz_timestamps:
            print("[WARN] 2Hz采样数据为空，无法生成PKL样本")
            return
        
        # 使用2Hz采样后的时间戳序列
        sampled_times = self.sampled_2hz_timestamps
        start_time = sampled_times[0]
        end_time = sampled_times[-1]
        print("[INFO] 开始生成JSON样本，基于2Hz采样数据，时间范围: {:.3f} - {:.3f}，共 {} 个采样点".format(
            start_time, end_time, len(sampled_times)))

        sample_idx = 0
        # 跳过前8帧，从第9帧开始生成数据
        for idx, sampled_ts in enumerate(sampled_times):
            if idx < 8: # 跳过前8帧
                continue
            if idx >= len(sampled_times) - 14:
                break

            # 找到最接近采样时间的front帧（用于获取图像）
            sorted_front_times = sorted(self.camera_group_by_front.keys())
            if not sorted_front_times:
                continue

            front_idx = bisect.bisect_left(sorted_front_times, sampled_ts)
            if front_idx >= len(sorted_front_times):
                front_idx = len(sorted_front_times) - 1

            # 选择最接近的front时间戳
            if front_idx > 0:
                if abs(sorted_front_times[front_idx - 1] - sampled_ts) < \
                   abs(sorted_front_times[front_idx] - sampled_ts):
                    front_idx -= 1

            front_ts = sorted_front_times[front_idx]

            # 生成该样本的数据（使用2Hz采样后的时间戳）
            sample_data = self._create_sample_data(sampled_ts, scene_id, sample_idx, front_ts_for_image=front_ts, scene_name=scene_name)

            if sample_data is not None:
                # 保存为单独的json文件，格式：{scene_name}_step{number}.json
                json_filename = os.path.join(self.json_samples_dir, "{}_step{}.json".format(scene_name, sample_idx + 1))
                try:
                    # 序列化为JSON兼容格式
                    json_data = serialize_for_json(sample_data)
                    with open(json_filename, 'w') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    sample_idx += 1
                    if sample_idx % 100 == 0:
                        print("[INFO] 已生成 {} 个样本".format(sample_idx))
                except Exception as e:
                    print("[ERROR] 保存样本失败：{}".format(e))

        print("[INFO] JSON样本生成完成：共 {} 个样本".format(sample_idx))

    def _create_sample_data(self, current_ts, scene_id, step_id, front_ts_for_image=None, scene_name=None):
        """
        创建单个样本的数据。

        参数:
            current_ts: 当前帧的时间戳（来自2Hz采样）
            scene_id: 场景ID
            step_id: 步骤ID
            front_ts_for_image: 用于获取图像的时间戳（如果为None则使用current_ts）
            scene_name: 场景名称（用于文件名）
        """
        # 使用2Hz采样后的定位数据
        loc_idx, loc_diff = self._find_nearest_loc_idx(current_ts, max_delta=0.1, use_2hz_sampled=True)
        if loc_idx is None:
            return None

        # 从2Hz采样数据中获取定位信息
        if self.sampled_2hz_locations:
            current_loc = self.sampled_2hz_locations[loc_idx]
            current_ts_loc = self.sampled_2hz_timestamps[loc_idx]
        else:
            # 回退到原始数据
            current_loc = self.valid_locations[loc_idx]
            current_ts_loc = self.valid_timestamps[loc_idx]

        if hasattr(current_loc, 'position') and current_loc.position:
            ego_utm_x = current_loc.position.x
            ego_utm_y = current_loc.position.y
            ego_utm_z = current_loc.position.z
        else:
            return None

        current_yaw = getattr(current_loc, "yaw", 0.0)

        # 用于转换到当前帧的ego坐标系
        ego_rotation_matrix = build_rotation_matrix_from_yaw(current_yaw)
        ego_global_position = [ego_utm_x, ego_utm_y, ego_utm_z]

        # 用于获取图像的时间戳
        image_ts = front_ts_for_image if front_ts_for_image is not None else current_ts

        # 收集历史8帧数据 (基于2Hz采样数据，每帧间隔0.5秒)
        history_window = 8
        history_interval = 0.5  # 2Hz采样，每帧间隔0.5秒

        his_timestamps = []
        his_locations = []

        # 使用2Hz采样后的数据查找历史帧
        for i in range(history_window):
            hist_ts = current_ts_loc - (history_window - i) * history_interval
            hist_idx, _ = self._find_nearest_loc_idx(hist_ts, max_delta=0.25, use_2hz_sampled=True)
            if hist_idx is not None:
                if self.sampled_2hz_timestamps:
                    his_timestamps.append(self.sampled_2hz_timestamps[hist_idx])
                    his_locations.append(self.sampled_2hz_locations[hist_idx])
                else:
                    his_timestamps.append(self.valid_timestamps[hist_idx])
                    his_locations.append(self.valid_locations[hist_idx])

        # 确保有8帧数据
        if len(his_locations) < history_window:
            return None

        his_trajectory_data = np.zeros((history_window, 2), dtype=np.float32)
        for i in range(history_window):
            # 获取位置（UTM坐标）
            loc = his_locations[i]
            if hasattr(loc, 'position') and loc.position:
                loc_utm_x = loc.position.x
                loc_utm_y = loc.position.y
                loc_utm_z = loc.position.z
            else:
                return None

            global_points = [[loc_utm_x, loc_utm_y, loc_utm_z]]
            local_points = transform_global_to_local(ego_global_position, global_points, ego_rotation_matrix)
            local_point_x = local_points[0][0]
            local_point_y = local_points[0][1]

            his_trajectory_data[i, 0] = local_point_x
            his_trajectory_data[i, 1] = local_point_y

        # 收集未来14帧数据 (基于2Hz采样数据，每帧间隔0.5秒)
        future_window = 14
        future_interval = 0.5  # 2Hz采样，每帧间隔0.5秒

        future_timestamps = []
        future_locations = []

        # 使用2Hz采样后的数据查找未来帧
        for i in range(future_window):
            future_ts = current_ts_loc + (i + 1) * future_interval
            future_idx, _ = self._find_nearest_loc_idx(future_ts, max_delta=0.25, use_2hz_sampled=True)
            if future_idx is not None:
                if self.sampled_2hz_timestamps:
                    future_timestamps.append(self.sampled_2hz_timestamps[future_idx])
                    future_locations.append(self.sampled_2hz_locations[future_idx])
                else:
                    future_timestamps.append(self.valid_timestamps[future_idx])
                    future_locations.append(self.valid_locations[future_idx])

        # 确保有14帧数据
        if len(future_locations) < future_window:
            return None

        future_trajectory_data = np.zeros((future_window, 2), dtype=np.float32)
        for i in range(future_locations):
            loc = future_locations[i]
            # 获取位置（UTM坐标）
            if hasattr(loc, 'position') and loc.position:
                loc_utm_x = loc.position.x
                loc_utm_y = loc.position.y
                loc_utm_z = loc.position.z
            else:
                return None

            global_points = [[loc_utm_x, loc_utm_y, loc_utm_z]]
            local_points = transform_global_to_local(ego_global_position, global_points, ego_rotation_matrix)
            local_point_x = local_points[0][0]
            local_point_y = local_points[0][1]

            future_trajectory_data[i, 0] = local_point_x
            future_trajectory_data[i, 1] = local_point_y

        # 获取3帧图像路径（当前帧和历史两帧）
        # 相机字段映射：back、front、left、left_back、right、right_back
        cam_paths_current = {}
        cam_paths_prev_1 = {}
        cam_paths_prev_2 = {}

        cam_keys = ["back", "front", "left", "left_back", "right", "right_back"]
        for key in cam_keys:
            # 当前帧（使用image_ts获取图像）
            path = self._get_camera_path(key, image_ts, max_delta=0.05)
            if path:
                cam_paths_current[key] = path
            
            # 前1帧（0.5秒前，基于2Hz采样）
            prev_ts_1 = image_ts - 0.5
            path_prev_1 = self._get_camera_path(key, prev_ts_1, max_delta=0.25)
            if path_prev_1:
                cam_paths_prev_1[key] = path_prev_1

            # 前2帧（0.5秒前，基于2Hz采样）
            prev_ts_2 = image_ts - 1.0
            path_prev_2 = self._get_camera_path(key, prev_ts_2, max_delta=0.25)
            if path_prev_2:
                cam_paths_prev_2[key] = path_prev_2

        # 构建图像路径字段（存储为bytes）
        sample_data = {}

        # 当前帧图像路径（存储为bytes字符串）
        # 使用 his_cam_ 前缀 + 原始相机字段名称：back、front、left、left_back、right、right_back
        def safe_encode(path):
            """安全编码路径，如果为空则返回空bytes"""
            return path.encode('utf-8') if path else b''

        # 当前帧图像路径，使用 cam_ 前缀 + 原始字段名
        sample_data['cam_front'] = safe_encode(cam_paths_current.get('front', ''))
        sample_data['cam_right'] = safe_encode(cam_paths_current.get('right', ''))
        sample_data['cam_left'] = safe_encode(cam_paths_current.get('left', ''))
        sample_data['cam_back'] = safe_encode(cam_paths_current.get('back', ''))
        sample_data['cam_left_back'] = safe_encode(cam_paths_current.get('left_back', ''))
        sample_data['cam_right_back'] = safe_encode(cam_paths_current.get('right_back', ''))

        # 向前一帧图像路径，使用 cam_ 前缀 + 原始字段名 + _prev_1 后缀表示
        sample_data['cam_front_prev_1'] = safe_encode(cam_paths_prev_1.get('front', ''))
        sample_data['cam_right_prev_1'] = safe_encode(cam_paths_prev_1.get('right', ''))
        sample_data['cam_left_prev_1'] = safe_encode(cam_paths_prev_1.get('left', ''))
        sample_data['cam_back_prev_1'] = safe_encode(cam_paths_prev_1.get('back', ''))
        sample_data['cam_left_back_prev_1'] = safe_encode(cam_paths_prev_1.get('left_back', ''))
        sample_data['cam_right_back_prev_1'] = safe_encode(cam_paths_prev_1.get('right_back', ''))

        # 向前两帧图像路径，使用 cam_ 前缀 + 原始字段名 + _prev_2 后缀表示
        sample_data['cam_front_prev_2'] = safe_encode(cam_paths_prev_2.get('front', ''))
        sample_data['cam_right_prev_2'] = safe_encode(cam_paths_prev_2.get('right', ''))
        sample_data['cam_left_prev_2'] = safe_encode(cam_paths_prev_2.get('left', ''))
        sample_data['cam_back_prev_2'] = safe_encode(cam_paths_prev_2.get('back', ''))
        sample_data['cam_left_back_prev_2'] = safe_encode(cam_paths_prev_2.get('left_back', ''))
        sample_data['cam_right_back_prev_2'] = safe_encode(cam_paths_prev_2.get('right_back', ''))

        # 历史运动状态
        sample_data['his_trajectory'] = his_trajectory_data
        sample_data['fur_trajectory'] = future_trajectory_data

        return sample_data

    def close(self):
        """关闭已打开的 bag"""
        if self.master_bag is not None:
            self.master_bag.close()
            self.master_bag = None
        if self.slave_105_bag is not None:
            self.slave_105_bag.close()
            self.slave_105_bag = None

    def run(self):
        # 主逻辑
        # 加载已存在的相机图像（不再从bag解析）
        self.load_existing_cameras()
        # 图片按时间戳拼帧
        self.align_camera_cache_by_front()
        # 解析定位topic
        self.parse_localization()
        # 对定位数据进行2Hz采样
        self.sample_localization_2hz(sample_rate_hz=2.0)
        # 解析轨迹topic
        # self.parse_trajectory()
        # 生成pkl样本（基于2Hz采样数据）
        self.generate_pkl_samples(sample_rate_hz=2.0)
        # 关闭 bag
        self.close()

if __name__ == '__main__':
    print("[INFO] 脚本启动")
    print("[INFO] 使用方法: python2.7 parseRosBag-2O_py27.py <bags_root_dir>")
    print("[INFO]   bags_root_dir: 包含多个bag文件夹的根目录（必需）")
    print("[INFO]   脚本会自动遍历该目录下的所有bag文件夹并处理")
    
    # 默认bags根目录
    default_bags_root = "/D2/NXT/vlm/dataset/MOGO/test_data/2O-bags"
    
    try:
        bags_root_dir = sys.argv[1] if len(sys.argv) > 1 else default_bags_root
        print("[INFO] Bags根目录：{}".format(bags_root_dir))
    except IndexError:
        bags_root_dir = raw_input("input bags root directory (default: {}): ".format(default_bags_root))
        if not bags_root_dir:
            bags_root_dir = default_bags_root
        print("[INFO] Bags根目录：{}".format(bags_root_dir))
    
    if not os.path.isdir(bags_root_dir):
        print("[ERROR] 目录不存在：{}".format(bags_root_dir))
        sys.exit(1)
    
    # 遍历所有bag文件夹
    bag_dirs = []
    for item in os.listdir(bags_root_dir):
        item_path = os.path.join(bags_root_dir, item)
        if os.path.isdir(item_path):
            # 检查是否包含bag文件
            has_master = False
            has_slave = False
            for filename in os.listdir(item_path):
                if filename.endswith('.bag'):
                    if '-rosmaster-' in filename:
                        has_master = True
                    elif '-rosslave-' in filename:
                        has_slave = True
            if has_master and has_slave:
                bag_dirs.append(item_path)
    
    if not bag_dirs:
        print("[ERROR] 未找到包含bag文件的文件夹")
        sys.exit(1)
    
    print("[INFO] 找到 {} 个bag文件夹".format(len(bag_dirs)))
    
    # 依次处理每个bag文件夹（暂时只处理第一个）
    for idx, bag_dir in enumerate(sorted(bag_dirs), 1):
        print("\n" + "="*80)
        print("[INFO] 处理第 {}/{} 个bag文件夹: {}".format(idx, len(bag_dirs), bag_dir))
        print("="*80)
        
        try:
            # 输出目录就是bag目录本身（图像已存在，只需生成pkl）
            handler = BagDataHandler(bag_dir, output_dir=bag_dir)
            handler.run()
            print("[INFO] 完成处理: {}".format(bag_dir))
        except Exception as e:
            print("[ERROR] 处理失败 {}: {}".format(bag_dir, e))
            import traceback
            traceback.print_exc()
            print("[WARN] 继续处理下一个bag文件夹...")
            continue
        
        # 暂时只处理第一个bag文件夹
        # print("[INFO] 暂时只处理第一个bag文件夹，退出")
        # break
    
    print("\n" + "="*80)
    print("[INFO] 处理完成！")
    print("="*80)
