#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python3 版本的 parseRosBag-2O 脚本，适配新的数据目录结构。

新数据目录结构：
  rosbags/
  |-- 0312-2/
  |   |-- 2026-03-12-15-48-50/          (路线名)
  |   |   |-- 2026-03-12-15-48-50.bag   (忽略，不是rosmaster/rosslave格式)
  |   |   |-- 20260312154919/           (segment, 包含 *rosmaster*.bag 和 *rosslave*.bag)
  |   |   |-- 20260312155025/
  |   |   ...

  images/
  |-- 0312-2/
  |   |-- 2026-03-12-15-48-50/
  |   |   |-- 20260312154919/
  |   |   |   |-- back/   (时间戳命名的jpg)
  |   |   |   |-- front/
  |   |   |   |-- left/
  |   |   |   |-- right/

  jsons/ (输出目录，结构同 rosbags)
  |-- 0312-2/
  |   |-- 2026-03-12-15-48-50/
  |   |   |-- 20260312154919/
  |   |   |   |-- 20260312154919_step1.json
  |   |   |   |-- 20260312154919_step2.json
  |   |   |   |-- ...

使用方法:
    python3 parseRosBag-2O_py3.py

    可选参数:
        --rosbags-root  rosbag根目录 (默认: /pcpt/pcpt/data/VLA/MOGO/rosbags)
        --images-root   images根目录 (默认: /pcpt/pcpt/data/VLA/MOGO/images)
        --jsons-root    json输出根目录 (默认: /pcpt/pcpt/data/VLA/MOGO/jsons)
        --dates         指定要处理的日期文件夹 (默认: 全部)
"""

import sys
import os
import json
import bisect
import re
import argparse
import traceback
import numpy as np
import base64

# proto 包内部使用平铺式导入，需要将 proto/common 加入 sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_proto_common_dir = os.path.join(_script_dir, "proto", "common")
if _proto_common_dir not in sys.path:
    sys.path.insert(0, _proto_common_dir)

# 使用 rosbags 替代 rosbag（pip install rosbags），支持纯 Python3 环境
from rosbags.rosbag1 import Reader as RosbagReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore
from proto.common.localization_pb2 import Localization
from proto.common.trajectory_pb2 import Trajectory, TrajectoryPoint
from proto.common.sensor_meta_pb2 import RegisteredSensor
from google.protobuf import text_format

# BinaryData 消息定义（autopilot_msgs/msg/BinaryData 是 protobuf 的 ROS 包装类型）
_BINARY_DATA_MSGDEF = (
    "Header header\nstring name\nuint32 size\nuint8[] data\n"
    "\n================================================================================\n"
    "MSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n"
)

def _make_typestore():
    """创建包含 autopilot_msgs/msg/BinaryData 定义的 typestore。"""
    add_types = get_types_from_msg(_BINARY_DATA_MSGDEF, "autopilot_msgs/msg/BinaryData")
    ts = get_typestore(Stores.ROS1_NOETIC)
    ts.register(add_types)
    return ts

_TYPESTORE = _make_typestore()


def _extract_proto_bytes(rawdata, msgtype):
    """从 ROS 序列化数据中提取 BinaryData.data 字段（protobuf bytes）。"""
    msg = _TYPESTORE.deserialize_ros1(rawdata, msgtype)
    return bytes(msg.data)


# ======================== 工具函数 ========================

def calculate_header_timestamp(header):
    """Calculate ROS Header timestamp in seconds."""
    sec = float(header.stamp.sec)
    nsec = float(header.stamp.nsec) if hasattr(header.stamp, 'nsec') else 0.0
    return sec + nsec * 1.0e-9


def serialize_for_json(obj):
    """将数据序列化为JSON兼容格式，处理numpy数组和bytes类型。"""
    if isinstance(obj, np.ndarray):
        return {
            '__numpy_array__': True,
            'data': obj.tolist(),
            'dtype': str(obj.dtype),
            'shape': list(obj.shape)
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
    norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm > 0:
        qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx*qx + qy*qy)]
    ])
    return R


def build_rotation_matrix_from_yaw(yaw):
    """从yaw角度构建旋转矩阵（绕Z轴旋转）"""
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([
        [sin_yaw,  cos_yaw, 0.0],
        [-cos_yaw, sin_yaw, 0.0],
        [0.0,      0.0,     1.0]
    ])
    return R


def transform_global_to_local(global_pose, global_points, ego_rotation_matrix=None):
    """将全局坐标系下的点转换为局部坐标系（以ego为中心）"""
    global_pose = np.array(global_pose)
    if ego_rotation_matrix is None:
        R = np.eye(3)
    else:
        R = np.array(ego_rotation_matrix)
        if R.shape != (3, 3):
            R = np.eye(3)
    R_T = R.T
    local_points = []
    for point in global_points:
        point_global = np.array(point)
        point_relative = point_global - global_pose
        point_local = R_T.dot(point_relative)
        local_points.append(point_local.tolist())
    return local_points


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


# ======================== 数据处理类 ========================

class BagDataHandler:
    """
    负责加载 rosbag 文件，并提供解析相机与定位数据的接口。
    适配新目录结构：rosbag 和 images 分开存放，只有4路相机。
    """

    # 新数据只有4路相机
    CAMERA_KEYS = ["back", "front", "left", "right"]

    def __init__(self, bag_dir, image_dir, output_dir=None):
        """
        参数:
            bag_dir:    包含 *rosmaster*.bag 和 *rosslave*.bag 的目录
            image_dir:  对应的图像目录（包含 back/front/left/right 子目录）
            output_dir: 输出目录，如果不指定则使用 bag_dir
        """
        self.bag_dir = bag_dir
        self.image_dir = image_dir
        self.output_dir = output_dir or bag_dir

        # bag 文件路径
        self.bags_file_master = None
        self.bags_file_slave = None

        # 已打开的 bag
        self.master_bag = None
        self.slave_bag = None

        # 相机相关
        self.camera_cache = {}

        # 定位相关
        self.valid_timestamps = []
        self.valid_locations = []

        # 2Hz采样后的定位数据
        self.sampled_2hz_timestamps = []
        self.sampled_2hz_locations = []
        self.sampled_2hz_indices = []

        # 轨迹相关
        self.trajectory_timestamps = []
        self.trajectory_data = []

        # json直接输出到 output_dir 下（即 segment 目录）
        self.json_samples_dir = self.output_dir
        os.makedirs(self.json_samples_dir, exist_ok=True)

        self._check_and_find_bags()
        self._open_bags()

    def _check_and_find_bags(self):
        """检查目录并查找 master/slave bag 文件。"""
        if not os.path.isdir(self.bag_dir):
            raise FileNotFoundError(f"目录{self.bag_dir}无效")

        for filename in os.listdir(self.bag_dir):
            if filename.endswith(".bag"):
                fp = os.path.join(self.bag_dir, filename)
                if "rosmaster" in filename:
                    self.bags_file_master = fp
                elif "rosslave" in filename:
                    self.bags_file_slave = fp

        if not self.bags_file_master or not self.bags_file_slave:
            raise FileNotFoundError(
                f"未找到 rosmaster/rosslave bag 文件，目录: {self.bag_dir}")

        print(f"[INFO] Master Bag路径：{self.bags_file_master}")
        print(f"[INFO] Slave Bag路径：{self.bags_file_slave}")

    def _open_bags(self):
        """打开 master 与 slave bag。"""
        try:
            self.master_bag = RosbagReader(self.bags_file_master)
            self.master_bag.open()
        except Exception as e:
            raise RuntimeError(f"打开Master Bag失败：{e}")

        try:
            self.slave_bag = RosbagReader(self.bags_file_slave)
            self.slave_bag.open()
        except Exception as e:
            if self.master_bag is not None:
                self.master_bag.close()
            raise RuntimeError(f"打开Slave Bag失败：{e}")

    def load_existing_cameras(self):
        """
        从已存在的相机图像目录加载图像路径和时间戳。
        只加载4路相机：back、front、left、right
        """
        print(f"[INFO] 从已存在的相机目录加载图像: {self.image_dir}")
        for key in self.CAMERA_KEYS:
            camera_dir = os.path.join(self.image_dir, key)
            if not os.path.exists(camera_dir):
                print(f"[WARN] 相机目录不存在：{camera_dir}")
                continue

            image_files = []
            for filename in os.listdir(camera_dir):
                if filename.endswith('.jpg'):
                    filepath = os.path.join(camera_dir, filename)
                    try:
                        timestamp = float(filename[:-4])
                        image_files.append((timestamp, filepath))
                    except ValueError:
                        print(f"[WARN] 无法解析文件名时间戳：{filename}")
                        continue

            image_files.sort(key=lambda x: x[0])

            if image_files:
                self.camera_cache[key] = {
                    "times": [item[0] for item in image_files],
                    "paths": [item[1] for item in image_files]
                }
                print(f"[INFO] 相机 {key}: 加载了 {len(image_files)} 张图像")
            else:
                print(f"[WARN] 相机 {key}: 未找到图像文件")

        print(f"[INFO] 相机图像加载完成：共{len(self.camera_cache)}路相机")

    def parse_localization(self):
        """解析定位数据并预处理时间戳。"""
        assert self.master_bag is not None

        self.valid_timestamps = []
        self.valid_locations = []

        print("[INFO] 解析定位数据并预处理时间戳...")
        loc_connections = [c for c in self.master_bag.connections
                           if c.topic == "/localization/global"]
        for conn, timestamp_ns, rawdata in self.master_bag.messages(connections=loc_connections):
            try:
                proto_bytes = _extract_proto_bytes(rawdata, conn.msgtype)
                loc_data = Localization()
                loc_data.ParseFromString(proto_bytes)
                ts = calculate_header_timestamp(loc_data.header)
                self.valid_timestamps.append(ts)
                self.valid_locations.append(loc_data)
            except Exception as e:
                print(f"[WARN] 解析定位消息失败：{e}")

        if not self.valid_timestamps:
            raise RuntimeError("定位数据中无有效时间戳信息")
        print(f"[INFO] 定位数据处理完成：有效数据{len(self.valid_timestamps)}条")

    def sample_localization_2hz(self, sample_rate_hz=2.0):
        """对定位数据进行2Hz采样"""
        if not self.valid_timestamps:
            print("[WARN] 定位数据为空，无法进行2Hz采样")
            return

        self.sampled_2hz_timestamps = []
        self.sampled_2hz_locations = []
        self.sampled_2hz_indices = []

        sample_interval = 1.0 / sample_rate_hz
        start_time = self.valid_timestamps[0]
        end_time = self.valid_timestamps[-1]

        current_time = start_time

        print(f"[INFO] 开始2Hz采样定位数据，采样频率: {sample_rate_hz}Hz，"
              f"时间范围: {start_time:.3f} - {end_time:.3f}")

        while current_time <= end_time:
            idx = bisect.bisect_left(self.valid_timestamps, current_time)
            candidates = []
            if idx == 0:
                candidates.append(0)
            elif idx == len(self.valid_timestamps):
                candidates.append(len(self.valid_timestamps) - 1)
            else:
                candidates.extend([idx - 1, idx])

            best_idx = None
            best_diff = float('inf')
            for ci in candidates:
                diff = abs(self.valid_timestamps[ci] - current_time)
                if diff < best_diff:
                    best_diff = diff
                    best_idx = ci

            if best_idx is not None and best_diff <= 0.25:
                if not self.sampled_2hz_indices or self.sampled_2hz_indices[-1] != best_idx:
                    self.sampled_2hz_timestamps.append(self.valid_timestamps[best_idx])
                    self.sampled_2hz_locations.append(self.valid_locations[best_idx])
                    self.sampled_2hz_indices.append(best_idx)

            current_time += sample_interval

        if self.sampled_2hz_indices and self.sampled_2hz_indices[-1] != len(self.valid_timestamps) - 1:
            last_idx = len(self.valid_timestamps) - 1
            self.sampled_2hz_timestamps.append(self.valid_timestamps[last_idx])
            self.sampled_2hz_locations.append(self.valid_locations[last_idx])
            self.sampled_2hz_indices.append(last_idx)

        print(f"[INFO] 2Hz采样完成：原始点数 {len(self.valid_timestamps)}，"
              f"采样后点数 {len(self.sampled_2hz_timestamps)}")

    def parse_trajectory(self):
        """解析轨迹数据。"""
        assert self.master_bag is not None

        self.trajectory_timestamps = []
        self.trajectory_data = []

        print("[INFO] 解析轨迹数据...")
        trajectory_topics = ["/planning/trajectory", "/trajectory", "/planning/trajectory_plan"]

        found_topic = None
        master_topics = [c.topic for c in self.master_bag.connections]
        for topic_name in trajectory_topics:
            if topic_name in master_topics:
                found_topic = topic_name
                break

        if found_topic is None:
            for topic_name in master_topics:
                if "trajectory" in topic_name.lower():
                    found_topic = topic_name
                    break

        if found_topic is None:
            print("[WARN] 未找到轨迹topic，将使用空轨迹数据")
            return

        print(f"[INFO] 使用轨迹topic: {found_topic}")
        traj_connections = [c for c in self.master_bag.connections if c.topic == found_topic]
        for conn, timestamp_ns, rawdata in self.master_bag.messages(connections=traj_connections):
            try:
                proto_bytes = _extract_proto_bytes(rawdata, conn.msgtype)
                traj_data = Trajectory()
                traj_data.ParseFromString(proto_bytes)
                ts = calculate_header_timestamp(traj_data.header)
                self.trajectory_timestamps.append(ts)
                self.trajectory_data.append(traj_data)
            except Exception as e:
                print(f"[WARN] 解析轨迹消息失败：{e}")

        print(f"[INFO] 轨迹数据处理完成：有效数据{len(self.trajectory_timestamps)}条")

    def align_camera_cache_by_front(self, max_delta=0.05):
        """根据 front 时间戳对齐其他视角 camera 的图片路径。"""
        if "front" not in self.camera_cache or not self.camera_cache["front"]["times"]:
            print("[WARN] align_camera_cache_by_front: front 相机数据为空")
            self.camera_group_by_front = {}
            return self.camera_group_by_front

        front_times = self.camera_cache["front"]["times"]
        front_paths = self.camera_cache["front"]["paths"]

        for key, data in self.camera_cache.items():
            times = data.get("times", [])
            paths = data.get("paths", [])
            if len(times) != len(paths):
                print(f"[WARN] 相机 {key} 的 times 与 paths 长度不一致，跳过排序")
                continue
            if not times:
                continue
            if any(times[i] > times[i + 1] for i in range(len(times) - 1)):
                paired = sorted(zip(times, paths), key=lambda x: x[0])
                data["times"] = [p[0] for p in paired]
                data["paths"] = [p[1] for p in paired]

        grouped = {}

        for t_front, path_front in zip(front_times, front_paths):
            paths_at_t = {"front": path_front}

            for cam_key, data in self.camera_cache.items():
                if cam_key == "front":
                    continue
                times = data.get("times", [])
                paths = data.get("paths", [])
                if not times or len(times) != len(paths):
                    continue

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
        print(f"[INFO] 对齐相机帧完成：front 时间戳 {len(grouped)} 个")
        return grouped

    def _find_nearest_loc_idx(self, target_ts, max_delta=0.1, use_2hz_sampled=True):
        """查找最接近目标时间戳的定位数据索引"""
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

    def generate_json_samples(self, sample_rate_hz=2.0):
        """按照2Hz采样频率生成json样本。"""
        if not hasattr(self, "camera_group_by_front") or not self.camera_group_by_front:
            print("[WARN] generate_json_samples: camera_group_by_front 为空")
            return

        if not self.valid_timestamps:
            print("[WARN] generate_json_samples: 定位数据为空")
            return

        scene_name = os.path.basename(self.bag_dir)

        bag_name = os.path.basename(self.bags_file_master)
        if bag_name.endswith('.bag'):
            bag_name = bag_name[:-4]
        scene_id = "0001"
        if '-' in bag_name:
            parts = bag_name.split('-')
            first_part = parts[0]
            if first_part.isdigit():
                scene_id = first_part.zfill(4)
            elif 'scene' in first_part.lower():
                match = re.search(r'(\d+)', first_part)
                if match:
                    scene_id = match.group(1).zfill(4)

        if not self.sampled_2hz_timestamps:
            print("[WARN] 2Hz采样数据为空，无法生成样本")
            return

        sampled_times = self.sampled_2hz_timestamps
        start_time = sampled_times[0]
        end_time = sampled_times[-1]

        print(f"[INFO] 开始生成JSON样本，基于2Hz采样数据，时间范围: {start_time:.3f} - {end_time:.3f}，"
              f"共 {len(sampled_times)} 个采样点")

        sample_idx = 0
        for idx, sampled_ts in enumerate(sampled_times):
            if idx == 0:
                continue

            sorted_front_times = sorted(self.camera_group_by_front.keys())
            if not sorted_front_times:
                continue

            front_idx = bisect.bisect_left(sorted_front_times, sampled_ts)
            if front_idx >= len(sorted_front_times):
                front_idx = len(sorted_front_times) - 1

            if front_idx > 0:
                if abs(sorted_front_times[front_idx - 1] - sampled_ts) < \
                   abs(sorted_front_times[front_idx] - sampled_ts):
                    front_idx -= 1

            front_ts = sorted_front_times[front_idx]

            sample_data = self._create_sample_data(
                sampled_ts, scene_id, sample_idx,
                front_ts_for_image=front_ts, scene_name=scene_name)

            if sample_data is not None:
                json_filename = os.path.join(
                    self.json_samples_dir,
                    f"{scene_name}_step{sample_idx + 1}.json")
                try:
                    json_data = serialize_for_json(sample_data)
                    with open(json_filename, 'w') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
                    sample_idx += 1
                    if sample_idx % 100 == 0:
                        print(f"[INFO] 已生成 {sample_idx} 个样本")
                except Exception as e:
                    print(f"[ERROR] 保存样本失败：{e}")

        print(f"[INFO] JSON样本生成完成：共 {sample_idx} 个样本")

    def _create_sample_data(self, current_ts, scene_id, step_id,
                            front_ts_for_image=None, scene_name=None):
        """创建单个样本的数据。"""
        loc_idx, loc_diff = self._find_nearest_loc_idx(
            current_ts, max_delta=0.1, use_2hz_sampled=True)
        if loc_idx is None:
            return None

        if self.sampled_2hz_locations:
            current_loc = self.sampled_2hz_locations[loc_idx]
            current_ts_loc = self.sampled_2hz_timestamps[loc_idx]
        else:
            current_loc = self.valid_locations[loc_idx]
            current_ts_loc = self.valid_timestamps[loc_idx]

        image_ts = front_ts_for_image if front_ts_for_image is not None else current_ts

        # 收集历史5帧数据
        history_window = 5
        history_interval = 0.5

        his_timestamps = []
        his_locations = []

        for i in range(history_window):
            hist_ts = current_ts_loc - (history_window - 1 - i) * history_interval
            hist_idx, _ = self._find_nearest_loc_idx(hist_ts, max_delta=0.25, use_2hz_sampled=True)
            if hist_idx is not None:
                if self.sampled_2hz_timestamps:
                    his_timestamps.append(self.sampled_2hz_timestamps[hist_idx])
                    his_locations.append(self.sampled_2hz_locations[hist_idx])
                else:
                    his_timestamps.append(self.valid_timestamps[hist_idx])
                    his_locations.append(self.valid_locations[hist_idx])
            else:
                his_timestamps.append(current_ts_loc)
                his_locations.append(current_loc)

        while len(his_locations) < history_window:
            his_locations.append(current_loc)
            his_timestamps.append(current_ts_loc)

        # 收集位置信息
        frame_data = []
        dt = history_interval

        if hasattr(current_loc, 'position') and current_loc.position:
            current_pos = np.array([current_loc.position.x, current_loc.position.y, current_loc.position.z])
        else:
            current_pos = np.array([
                getattr(current_loc, 'longitude', 0.0),
                getattr(current_loc, 'latitude', 0.0),
                getattr(current_loc, 'altitude', 0.0)
            ])
        current_yaw = getattr(current_loc, "yaw", 0.0)

        next_pos = None
        next_loc = None
        locs = self.sampled_2hz_locations if self.sampled_2hz_locations else self.valid_locations
        if loc_idx + 1 < len(locs):
            next_loc = locs[loc_idx + 1]
            if hasattr(next_loc, 'position') and next_loc.position:
                next_pos = np.array([next_loc.position.x, next_loc.position.y, next_loc.position.z])
            else:
                next_pos = np.array([
                    getattr(next_loc, 'longitude', 0.0),
                    getattr(next_loc, 'latitude', 0.0),
                    getattr(next_loc, 'altitude', 0.0)
                ])

        for i, loc in enumerate(his_locations):
            if hasattr(loc, 'position') and loc.position:
                pos = np.array([loc.position.x, loc.position.y, loc.position.z])
            else:
                pos = np.array([
                    getattr(loc, 'longitude', 0.0),
                    getattr(loc, 'latitude', 0.0),
                    getattr(loc, 'altitude', 0.0)
                ])

            yaw = getattr(loc, "yaw", 0.0)

            if i + 1 < len(his_locations):
                next_loc_frame = his_locations[i + 1]
                if hasattr(next_loc_frame, 'position') and next_loc_frame.position:
                    next_pos_frame = np.array([next_loc_frame.position.x, next_loc_frame.position.y, next_loc_frame.position.z])
                else:
                    next_pos_frame = np.array([
                        getattr(next_loc_frame, 'longitude', 0.0),
                        getattr(next_loc_frame, 'latitude', 0.0),
                        getattr(next_loc_frame, 'altitude', 0.0)
                    ])
                vel_global = (next_pos_frame - pos) / dt
            elif i == len(his_locations) - 1 and next_pos is not None:
                vel_global = (next_pos - pos) / dt
            else:
                vel_global = np.zeros(3)

            frame_data.append({
                'pos': pos,
                'yaw': yaw,
                'vel_global': vel_global
            })

        # 追加下一帧数据
        if next_pos is not None and next_loc is not None:
            next_next_pos = None
            if loc_idx + 2 < len(locs):
                next_next_loc = locs[loc_idx + 2]
                if hasattr(next_next_loc, 'position') and next_next_loc.position:
                    next_next_pos = np.array([next_next_loc.position.x, next_next_loc.position.y, next_next_loc.position.z])
                else:
                    next_next_pos = np.array([
                        getattr(next_next_loc, 'longitude', 0.0),
                        getattr(next_next_loc, 'latitude', 0.0),
                        getattr(next_next_loc, 'altitude', 0.0)
                    ])

            if next_next_pos is not None:
                vel_next = (next_next_pos - next_pos) / dt
            else:
                vel_next = (next_pos - current_pos) / dt

            next_yaw = getattr(next_loc, "yaw", current_yaw)
            frame_data.append({
                'pos': next_pos,
                'yaw': next_yaw,
                'vel_global': vel_next
            })

        # 计算历史特征
        ego_rotation_matrix = build_rotation_matrix_from_yaw(current_yaw)
        rot_mat_global_to_ego = ego_rotation_matrix.T

        his_speed = np.zeros(history_window, dtype=np.float32)
        his_acceleration = np.zeros((history_window, 3), dtype=np.float32)
        his_angular_velocity = np.zeros((history_window, 3), dtype=np.float32)
        his_theta = np.zeros(history_window, dtype=np.float32)

        for i in range(history_window):
            frame = frame_data[i]
            vel_global = frame['vel_global']

            vel_ego = np.dot(rot_mat_global_to_ego, vel_global)
            his_speed[i] = np.linalg.norm(vel_ego[:2])

            if i + 1 < len(frame_data):
                next_vel_global = frame_data[i + 1]['vel_global']
                acc_global = (next_vel_global - vel_global) / dt
                acc_ego = np.dot(rot_mat_global_to_ego, acc_global)
                his_acceleration[i, 0] = acc_ego[0]
                his_acceleration[i, 1] = acc_ego[1]
                his_acceleration[i, 2] = acc_ego[2]

            if i + 1 < len(frame_data):
                next_frame = frame_data[i + 1]
                dyaw = normalize_angle(next_frame['yaw'] - frame['yaw'])
                his_angular_velocity[i, 2] = dyaw / dt

            if i == history_window - 1:
                his_theta[i] = 0.0
            else:
                his_theta[i] = normalize_angle(frame['yaw'] - current_yaw)

        # 获取图像路径 (只有4路相机，没有 left_back 和 right_back)
        cam_paths_current = {}
        cam_paths_prev = {}

        for key in self.CAMERA_KEYS:
            path = self._get_camera_path(key, image_ts, max_delta=0.05)
            if path:
                cam_paths_current[key] = path

            prev_ts = image_ts - 0.5
            path_prev = self._get_camera_path(key, prev_ts, max_delta=0.25)
            if path_prev:
                cam_paths_prev[key] = path_prev

        def safe_encode(path):
            return path.encode('utf-8') if path else b''

        sample_data = {}

        # 当前帧图像路径 (4路相机)
        sample_data['his_cam_front'] = safe_encode(cam_paths_current.get('front', ''))
        sample_data['his_cam_right'] = safe_encode(cam_paths_current.get('right', ''))
        sample_data['his_cam_left'] = safe_encode(cam_paths_current.get('left', ''))
        sample_data['his_cam_back'] = safe_encode(cam_paths_current.get('back', ''))
        # left_back 和 right_back 设为空 (新数据没有这两路相机)
        sample_data['his_cam_left_back'] = b''
        sample_data['his_cam_right_back'] = b''

        # 前一帧图像路径
        sample_data['his_cam_front_time'] = safe_encode(cam_paths_prev.get('front', ''))
        sample_data['his_cam_right_time'] = safe_encode(cam_paths_prev.get('right', ''))
        sample_data['his_cam_left_time'] = safe_encode(cam_paths_prev.get('left', ''))
        sample_data['his_cam_back_time'] = safe_encode(cam_paths_prev.get('back', ''))
        sample_data['his_cam_left_back_time'] = b''
        sample_data['his_cam_right_back_time'] = b''

        # 历史运动状态
        sample_data['his_speed'] = his_speed
        sample_data['his_acceleration'] = his_acceleration
        sample_data['his_angular_velocity'] = his_angular_velocity
        sample_data['his_theta'] = his_theta

        # 导航目标点
        if hasattr(current_loc, 'position') and current_loc.position:
            ego_utm_x = current_loc.position.x
            ego_utm_y = current_loc.position.y
            ego_utm_z = current_loc.position.z
        else:
            ego_utm_x = getattr(current_loc, 'longitude', 0.0)
            ego_utm_y = getattr(current_loc, 'latitude', 0.0)
            ego_utm_z = getattr(current_loc, 'altitude', 0.0)

        navi_x_ego = 0.0
        navi_y_ego = 0.0
        navi_frame_offset = 6
        navi_target_time = current_ts_loc + navi_frame_offset * 0.5

        navi_loc_idx, _ = self._find_nearest_loc_idx(navi_target_time, max_delta=0.25, use_2hz_sampled=True)

        if navi_loc_idx is not None:
            navi_loc = locs[navi_loc_idx] if locs else self.valid_locations[navi_loc_idx]
            if hasattr(navi_loc, 'position') and navi_loc.position:
                navi_utm_x = navi_loc.position.x
                navi_utm_y = navi_loc.position.y
                navi_utm_z = navi_loc.position.z
            else:
                navi_utm_x = getattr(navi_loc, 'longitude', ego_utm_x)
                navi_utm_y = getattr(navi_loc, 'latitude', ego_utm_y)
                navi_utm_z = getattr(navi_loc, 'altitude', ego_utm_z)

            ego_rot = build_rotation_matrix_from_yaw(current_yaw)
            global_pose = [ego_utm_x, ego_utm_y, ego_utm_z]
            local_points = transform_global_to_local(
                global_pose, [[navi_utm_x, navi_utm_y, navi_utm_z]], ego_rot)
            navi_x_ego = local_points[0][0]
            navi_y_ego = local_points[0][1]

        sample_data['x_command_far'] = np.array([navi_y_ego] * 5, dtype=np.float32)
        sample_data['y_command_far'] = np.array([navi_x_ego] * 5, dtype=np.float32)

        # 未来轨迹
        future_horizon = 7
        future_interval = 0.5

        fur_x = np.zeros((1, future_horizon), dtype=np.float32)
        fur_y = np.zeros((1, future_horizon), dtype=np.float32)

        ego_rot = build_rotation_matrix_from_yaw(current_yaw)
        global_pose = [ego_utm_x, ego_utm_y, ego_utm_z]

        last_found_local_x = None
        last_found_local_y = None

        for i in range(1, future_horizon):
            target_time = current_ts_loc + i * future_interval
            target_loc_idx, _ = self._find_nearest_loc_idx(target_time, max_delta=0.25, use_2hz_sampled=True)

            if target_loc_idx is not None:
                future_loc = locs[target_loc_idx] if locs else self.valid_locations[target_loc_idx]

                if hasattr(future_loc, 'position') and future_loc.position:
                    future_utm_x = future_loc.position.x
                    future_utm_y = future_loc.position.y
                    future_utm_z = future_loc.position.z
                else:
                    future_utm_x = getattr(future_loc, 'longitude', ego_utm_x)
                    future_utm_y = getattr(future_loc, 'latitude', ego_utm_y)
                    future_utm_z = getattr(future_loc, 'altitude', ego_utm_z)

                local_pts = transform_global_to_local(
                    global_pose, [[future_utm_x, future_utm_y, future_utm_z]], ego_rot)
                future_x_ego = local_pts[0][0]
                future_y_ego = local_pts[0][1]

                last_found_local_x = future_x_ego
                last_found_local_y = future_y_ego

                fur_x[0, i] = future_y_ego
                fur_y[0, i] = future_x_ego
            else:
                if last_found_local_x is not None:
                    fur_x[0, i] = last_found_local_y
                    fur_y[0, i] = last_found_local_x

        sample_data['fur_x'] = fur_x
        sample_data['fur_y'] = fur_y
        sample_data['status'] = f"scene-{scene_id}_step{step_id + 1}"

        return sample_data

    def close(self):
        """关闭已打开的 bag。"""
        if self.master_bag is not None:
            self.master_bag.close()
            self.master_bag = None
        if self.slave_bag is not None:
            self.slave_bag.close()
            self.slave_bag = None

    def run(self):
        self.load_existing_cameras()
        self.align_camera_cache_by_front()
        self.parse_localization()
        self.sample_localization_2hz(sample_rate_hz=2.0)
        self.parse_trajectory()
        self.generate_json_samples(sample_rate_hz=2.0)
        self.close()


# ======================== 目录遍历逻辑 ========================

def discover_segments(rosbags_root, images_root, jsons_root, dates=None):
    """
    遍历新目录结构，发现所有的 segment（包含 rosmaster/rosslave bag 的目录）。

    返回:
        list of (bag_dir, image_dir, json_output_dir, label_str)
    """
    segments = []

    if dates is None:
        dates = sorted([
            d for d in os.listdir(rosbags_root)
            if os.path.isdir(os.path.join(rosbags_root, d))
        ])

    for date_dir in dates:
        date_bag_path = os.path.join(rosbags_root, date_dir)
        date_img_path = os.path.join(images_root, date_dir)
        date_json_path = os.path.join(jsons_root, date_dir)

        if not os.path.isdir(date_bag_path):
            print(f"[WARN] rosbag日期目录不存在: {date_bag_path}")
            continue

        # 遍历路线名
        for route_name in sorted(os.listdir(date_bag_path)):
            route_bag_path = os.path.join(date_bag_path, route_name)
            if not os.path.isdir(route_bag_path):
                continue

            # 遍历 segment
            for segment_name in sorted(os.listdir(route_bag_path)):
                segment_bag_path = os.path.join(route_bag_path, segment_name)
                if not os.path.isdir(segment_bag_path):
                    continue

                # 检查是否包含 rosmaster 和 rosslave bag
                has_master = False
                has_slave = False
                for filename in os.listdir(segment_bag_path):
                    if filename.endswith('.bag'):
                        if 'rosmaster' in filename:
                            has_master = True
                        elif 'rosslave' in filename:
                            has_slave = True

                if not (has_master and has_slave):
                    continue

                # 构建对应的图像目录和json输出目录
                segment_img_path = os.path.join(date_img_path, route_name, segment_name)
                segment_json_path = os.path.join(date_json_path, route_name, segment_name)

                label = f"{date_dir}/{route_name}/{segment_name}"
                segments.append((segment_bag_path, segment_img_path, segment_json_path, label))

    return segments


def main():
    parser = argparse.ArgumentParser(
        description="Python3 版 parseRosBag-2O，适配新数据目录结构")
    parser.add_argument(
        "--rosbags-root",
        type=str,
        default="/pcpt/pcpt/data/VLA/MOGO/rosbags",
        help="rosbag根目录 (默认: /pcpt/pcpt/data/VLA/MOGO/rosbags)")
    parser.add_argument(
        "--images-root",
        type=str,
        default="/pcpt/pcpt/data/VLA/MOGO/images",
        help="images根目录 (默认: /pcpt/pcpt/data/VLA/MOGO/images)")
    parser.add_argument(
        "--jsons-root",
        type=str,
        default="/pcpt/pcpt/data/VLA/MOGO/jsons",
        help="json输出根目录 (默认: /pcpt/pcpt/data/VLA/MOGO/jsons)")
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=None,
        help="指定要处理的日期文件夹 (默认: 全部)，例如: --dates 0312-2 0313-1 0316")

    args = parser.parse_args()

    rosbags_root = args.rosbags_root
    images_root = args.images_root
    jsons_root = args.jsons_root

    print("[INFO] 脚本启动 (Python3 版本)")
    print(f"[INFO] Rosbags根目录: {rosbags_root}")
    print(f"[INFO] Images根目录:  {images_root}")
    print(f"[INFO] Jsons输出目录: {jsons_root}")

    if not os.path.isdir(rosbags_root):
        print(f"[ERROR] rosbag根目录不存在：{rosbags_root}")
        sys.exit(1)

    if not os.path.isdir(images_root):
        print(f"[ERROR] images根目录不存在：{images_root}")
        sys.exit(1)

    # 发现所有 segment
    segments = discover_segments(rosbags_root, images_root, jsons_root, dates=args.dates)

    if not segments:
        print("[ERROR] 未找到包含 rosmaster/rosslave bag 文件的 segment 目录")
        sys.exit(1)

    print(f"[INFO] 找到 {len(segments)} 个 segment")
    for i, (bag_dir, img_dir, json_dir, label) in enumerate(segments, 1):
        print(f"  [{i}] {label}")
        print(f"      bag:  {bag_dir}")
        print(f"      img:  {img_dir}")
        print(f"      json: {json_dir}")

    # 依次处理每个 segment
    success_count = 0
    fail_count = 0
    for idx, (bag_dir, img_dir, json_dir, label) in enumerate(segments, 1):
        print(f"\n{'='*80}")
        print(f"[INFO] 处理第 {idx}/{len(segments)} 个 segment: {label}")
        print(f"{'='*80}")

        if not os.path.isdir(img_dir):
            print(f"[WARN] 图像目录不存在，跳过: {img_dir}")
            fail_count += 1
            continue

        try:
            handler = BagDataHandler(bag_dir, image_dir=img_dir, output_dir=json_dir)
            handler.run()
            print(f"[INFO] 完成处理: {label}")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] 处理失败 {label}: {e}")
            traceback.print_exc()
            print("[WARN] 继续处理下一个 segment...")
            fail_count += 1
            continue

    print(f"\n{'='*80}")
    print(f"[INFO] 处理完成！成功: {success_count}，失败: {fail_count}，总计: {len(segments)}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
