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
import math
import re
import numpy as np
import rosbag
import base64
import codecs
from proto.common.localization_pb2 import Localization
from proto.common.trajectory_pb2 import Trajectory, TrajectoryPoint
from proto.common.sensor_meta_pb2 import RegisteredSensor
from proto.common.routing_pb2 import  RoutingRequest, RoutingResult, RoutePath, RouteRoad, RouteLink, RouteSection, RouteLane, RouteRoadLink
from proto.common.hadmap_pb2 import LaneLink, LaneType
from proto.common.geometry_pb2 import Point
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
            'data': base64.b64encode(obj)
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


        # 存储解码后的RoutingResult对象
        self.route_result = None
        # 每个定位点的高阶意图列表（格式：[[动作, 距离, 车道起, 车道止], ...]）每个定位点包含2个高阶意图。如无则NONE
        self.high_order_intent = []

        # 轨迹相关
        # self.trajectory_timestamps = []
        # self.trajectory_data = []  # 存储Trajectory对象

        # 注意：ego旋转矩阵将从定位数据的yaw角度动态构建，不在这里预加载

        # action_sequence.txt 保存人工标注的 meta action 及每个 action 对应的时间区间
        # scenario_sequence.txt 保存人工标注的 meta action 及每个 action 对应的时间区间
        # action/scenario sequence 文件路径
        self.file_path_action_sequence = None
        self.file_path_scenario_sequence = None

        # action/scenario sequence 文件
        self.file_action_sequence = None
        self.file_scenario_sequence = None

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

        self._check_and_find_sequence_files()
        self._open_sequence_files()

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

    def _check_and_find_sequence_files(self):
        """检查目录并查找 action_sequence.txt 和 scenario_sequence.txt 文件"""
        if not os.path.isdir(self.bag_dir):
            raise IOError("目录{}无效".format(self.bag_dir))

        # 初始化文件路径为None
        self.file_path_action_sequence = None
        self.file_path_scenario_sequence = None

        # 遍历目录查找目标文件
        for filename in os.listdir(self.bag_dir):
            fp = os.path.join(self.bag_dir, filename)
            if filename == "action_sequence.txt":
                self.file_path_action_sequence = fp
            elif filename == "scenario_sequence.txt":
                self.file_path_scenario_sequence = fp

        # 打印查找结果
        if self.file_path_action_sequence:
            print("[INFO] action_sequence.txt 路径：{}".format(self.file_path_action_sequence))
        else:
            print("[WARN] 未找到 action_sequence.txt 文件")

        if self.file_path_scenario_sequence:
            print("[INFO] scenario_sequence.txt 路径：{}".format(self.file_path_scenario_sequence))
        else:
            print("[WARN] 未找到 scenario_sequence.txt 文件")

    def _open_sequence_files(self):
        """打开 action_sequence.txt 和 scenario_sequence.txt 文件"""
        # 初始化文件对象为None
        self.file_action_sequence = None
        self.file_scenario_sequence = None

        # 打开 action_sequence.txt
        if self.file_path_action_sequence:
            try:
                self.file_action_sequence = codecs.open(self.file_path_action_sequence, 'r', encoding='utf-8')
                print("[INFO] 成功打开 action_sequence.txt：{}".format(self.file_path_action_sequence))
            except Exception as e:
                # 打开失败时清理已打开的文件
                if self.file_action_sequence is not None:
                    self.file_action_sequence.close()
                    self.file_action_sequence = None
                raise RuntimeError("打开 action_sequence.txt 失败：{}".format(e))
        else:
            print("[WARN] action_sequence.txt 文件路径为空，跳过打开")

        # 打开 scenario_sequence.txt
        if self.file_path_scenario_sequence:
            try:
                self.file_scenario_sequence = codecs.open(self.file_path_scenario_sequence, 'r', encoding='utf-8')
                print("[INFO] 成功打开 scenario_sequence.txt：{}".format(self.file_path_scenario_sequence))
            except Exception as e:
                # 打开失败时清理已打开的文件
                if self.file_scenario_sequence is not None:
                    self.file_scenario_sequence.close()
                    self.file_scenario_sequence = None
                raise RuntimeError("打开 scenario_sequence.txt 失败：{}".format(e))
        else:
            print("[WARN] scenario_sequence.txt 文件路径为空，跳过打开")

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
        




    def parse_routing(self):
        """
        核心流程：解析路由→提取稀疏点→统计路口→匹配自车到路口的路径距离
        关键索引逻辑（正序）：
        - roadlink_idx 从小到大 → 实际行驶路径从近（自车）到远（前方）
        - 目标roadlink_idx > 当前roadlink_idx → 目标在自车前方（有效）
        """
        # 基础配置
        assert self.master_bag is not None, "master_bag未初始化！"
        MAX_PATH_DISTANCE = 400.0  # 路径距离阈值（米）
        MAX_MATCH_JUNCTION = 2     # 最多匹配2个路口

        # 路口动作映射
        ACTION_MAP = {
            LaneType.LANE_TYPE_JuncTurnLeft: "turn left",
            LaneType.LANE_TYPE_JuncStraight: "straight",
            LaneType.LANE_TYPE_JuncTurnRight: "turn right",
            LaneType.LANE_TYPE_Turn: "u turn"
        }
        target_lane_types = {
            LaneType.LANE_TYPE_JuncTurnLeft,
            LaneType.LANE_TYPE_JuncStraight,
            LaneType.LANE_TYPE_JuncTurnRight,
            LaneType.LANE_TYPE_Turn
        }

        # ========== 辅助函数 ==========
        def get_junction_unique_id(lane_link):
            """生成路口唯一标识（避免重复统计）"""
            from_road = lane_link.from_road_pkid if lane_link.HasField("from_road_pkid") else 0
            to_road = lane_link.to_road_pkid if lane_link.HasField("to_road_pkid") else 0
            from_junc = lane_link.from_junction if lane_link.HasField("from_junction") else False
            to_junc = lane_link.to_junction if lane_link.HasField("to_junction") else False
            return f"{from_road}_{to_road}_{from_junc}_{to_junc}"
 
        def calculate_path_distance(current_loc, tar_junction):
            """
            计算自车到目标roadlink的路径距离（正序索引逻辑）
            :param current_loc: 自车当前位置（x/y）
            :param tar_junction: 目标路口信息字典（含roadlink_idx/action等）
            :return: 路径距离（米），无效返回nan
            """
            target_roadlink_idx = tar_junction["roadlink_idx"]
            if not sparse_points or target_roadlink_idx not in roadlink_length_map:
                return math.nan
            
            has_direction = False
            dir_x, dir_y = 0.0, 0.0
            if len(self.sampled_2hz_locations) >= 2:
                prev_loc = self.sampled_2hz_locations[max(0, self.sampled_2hz_locations.index(current_loc)-1)]
                dir_x = current_loc.x - prev_loc.x
                dir_y = current_loc.y - prev_loc.y
                dir_norm = math.hypot(dir_x, dir_y)
                if dir_norm > 1e-6 and tar_junction["action"] != "u turn":
                    dir_x /= dir_norm
                    dir_y /= dir_norm
                    has_direction = True
            
            closest_roadlink_idx = None
            min_straight_dist = float('inf')
            closest_roadlink_type = ""
            for p in sparse_points:
                dx = p["x"] - current_loc.x
                dy = p["y"] - current_loc.y
                if has_direction:
                    dot_product = dir_x * dx + dir_y * dy   
                    if dot_product <= 1e-6:  # 后方/侧后方点 → 跳过
                        continue
                
                straight_dist = math.hypot(dx, dy)
                if straight_dist < min_straight_dist:
                    min_straight_dist = straight_dist
                    closest_roadlink_idx = p["roadlink_idx"]
                    closest_roadlink_type = p["type"]
            
            if closest_roadlink_idx is None:
                return math.nan
            
            if target_roadlink_idx < closest_roadlink_idx and closest_roadlink_type == "route_road":
                return math.nan
            
            if closest_roadlink_type == "route_link" and closest_roadlink_idx == target_roadlink_idx:
                return 0.1
            
            current_road_end_point = next(
                (p for p in sparse_points if p["type"] == "route_road" and p["roadlink_idx"] == closest_roadlink_idx),
                None
            )

            current_offset = 0.0
            if current_road_end_point:
                dx = current_road_end_point["x"] - current_loc.x
                dy = current_road_end_point["y"] - current_loc.y
                current_offset = math.hypot(dx, dy)  # 自车到当前road终点的距离
            
            path_dist_sum = 0.0
            for idx in range(closest_roadlink_idx + 1, target_roadlink_idx + 1):
                if idx in roadlink_length_map:
                    path_dist_sum += roadlink_length_map[idx]
            
            final_dist = current_offset + path_dist_sum
            final_dist = max(0.0, final_dist)
            if closest_roadlink_idx == target_roadlink_idx:
                final_dist = current_offset

            return final_dist

        # ========== 第一步：解析路由数据 ==========
        self.route_result = None
        for topic, msg in self.master_bag.read_messages(topics=["/hadmap_engine/route_result_msg"]):
            try:
                route_roads = RoutingResult()  
                route_roads.ParseFromString(msg.data)
                self.route_result = route_roads
                break 
            except Exception as e:
                print(f"[WARN] 解析路由消息失败：{str(e)}")

        if not self.route_result:
            print("[WARN] 未解析到有效路由数据")
            empty_junction = [["None", math.nan, math.nan, math.nan], ["None", math.nan, math.nan, math.nan]]
            self.high_order_intent = [empty_junction for _ in self.sampled_2hz_locations]
            return

        # ========== 第二步：正序提取稀疏点列 + 记录road长度 ==========
        lane_junction_map = {}
        sparse_points = []  # 局部变量（原self.sparse_points）
        roadlink_length_map = {}  # 局部变量（原self.roadlink_length_map）

        if not hasattr(self.route_result, "paths") or not self.route_result.paths:
            print("[WARN] 路由数据中无有效路径")
            empty_junction = [["None", math.nan, math.nan, math.nan], ["None", math.nan, math.nan, math.nan]]
            self.high_order_intent = [empty_junction for _ in self.sampled_2hz_locations]
            return

        first_path = self.route_result.paths[0]
        for roadlink_idx, roadlink in enumerate(first_path.route_roadlinks):
            if roadlink.route_links and len(roadlink.route_links) > 0:
                first_link = roadlink.route_links[0]
                if first_link.geom and len(first_link.geom) > 0:
                    link_last_point = first_link.geom[-1]
                    sparse_points.append({
                        "type": "route_link",
                        "roadlink_idx": roadlink_idx,  
                        "x": link_last_point.x,
                        "y": link_last_point.y
                    })

            if roadlink.HasField("route_road") and roadlink.route_road.geom and len(roadlink.route_road.geom) > 0:
                road_last_point = roadlink.route_road.geom[-1]
                sparse_points.append({
                    "type": "route_road",
                    "roadlink_idx": roadlink_idx,
                    "x": road_last_point.x,
                    "y": road_last_point.y
                })
                roadlink_length_map[roadlink_idx] = roadlink.route_road.length if roadlink.route_road.HasField("length") else 0.0

            for lane_link in roadlink.route_links:
                if lane_link.type not in target_lane_types:
                    continue

                lane_id = lane_link.from_lane_id if lane_link.HasField("from_lane_id") else math.nan
                junc_id = get_junction_unique_id(lane_link)

                if junc_id not in lane_junction_map:
                    first_geom = lane_link.geom[0] if lane_link.geom else Point()
                    lane_junction_map[junc_id] = {
                        "action": ACTION_MAP.get(lane_link.type, "unknown"),
                        "x": first_geom.x,
                        "y": first_geom.y,
                        "lane_start": lane_id,
                        "lane_end": lane_id,
                        "roadlink_idx": roadlink_idx
                    }
                else:
                    if not math.isnan(lane_id):
                        junc_data = lane_junction_map[junc_id]
                        junc_data["lane_start"] = min(junc_data["lane_start"], lane_id) if not math.isnan(junc_data["lane_start"]) else lane_id
                        junc_data["lane_end"] = max(junc_data["lane_end"], lane_id) if not math.isnan(junc_data["lane_end"]) else lane_id

        # ========== 第三步：封装路口信息（按正序索引排序） ==========
        junction_list = []
        for junc_data in lane_junction_map.values():
            junction_list.append({
                "action": junc_data["action"],
                "x": junc_data["x"],
                "y": junc_data["y"],
                "lane_start": junc_data["lane_start"],
                "lane_end": junc_data["lane_end"],
                "roadlink_idx": junc_data["roadlink_idx"] 
            })

        junction_list.sort(key=lambda x: x["roadlink_idx"])

        # ========== 第四步：匹配自车到路口的路径距离（正序逻辑） ==========
        self.high_order_intent = []
        for current_loc in self.sampled_2hz_locations:
            matched_junctions = []
            candidate_junctions = []

            for junc in junction_list:
                path_dist = calculate_path_distance(current_loc, junc)
                
                if math.isnan(path_dist) or path_dist > MAX_PATH_DISTANCE:
                    continue
                
                candidate_junctions.append({
                    "distance": round(path_dist, 1),
                    "action": junc["action"],
                    "lane_start": junc["lane_start"],
                    "lane_end": junc["lane_end"]
                })

            candidate_junctions.sort(key=lambda x: x["distance"])
            selected_junctions = candidate_junctions[:MAX_MATCH_JUNCTION]

            for junc in selected_junctions:
                matched_junctions.append([
                    junc["action"],
                    junc["distance"],
                    junc["lane_start"],
                    junc["lane_end"]
                ])

            empty_count = MAX_MATCH_JUNCTION - len(matched_junctions)
            for _ in range(empty_count):
                matched_junctions.append(["None", math.nan, math.nan, math.nan])

            self.high_order_intent.append(matched_junctions)





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

    def parse_action_sequence(self):
        """
        解析 action_sequence.txt 文件内容，返回动作类型与对应时间段的字典
        空时间段的动作会被直接跳过，不加入字典
        
        返回格式:
            {
                "left_turn": [[0.5, 1.0], [1.5, 2.0]],
                "straight": [[2.5, 3.0]]
            }
        
        异常处理:
            - 文件未打开/不存在时返回空字典
            - 格式错误的行跳过并打印警告
        """
        action_dict = {}
        
        # 检查文件是否已打开
        if self.file_action_sequence is None:
            print("[WARN] action_sequence.txt 文件未打开，无法解析")
            return action_dict
        
        try:
            # 重置文件指针到开头
            self.file_action_sequence.seek(0)
            # 读取所有行并处理
            lines = self.file_action_sequence.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # 去除空白字符和换行符
                line = line.strip()
                # 跳过空行和注释行
                if not line or line.startswith('#'):
                    continue
                
                # 去除行尾的逗号（处理示例中的逗号分隔格式）
                line = line.rstrip(',').strip()
                
                # 分割动作类型和时间段部分（按第一个冒号分割）
                if ':' not in line:
                    print(f"[WARN] 第{line_num}行格式错误，无冒号分隔: {line}")
                    continue
                
                action_type, time_str = line.split(':', 1)
                action_type = action_type.strip()
                time_str = time_str.strip()
                
                # 解析时间段列表
                try:
                    # 使用eval解析列表（安全场景下使用，因为是已知格式的标注文件）
                    time_intervals = eval(time_str)
                    
                    # 验证格式：必须是列表，且每个元素是包含2个数字的列表
                    valid_intervals = []
                    if isinstance(time_intervals, list):
                        for interval in time_intervals:
                            if (isinstance(interval, list) and len(interval) == 2 and
                                    all(isinstance(x, (int, float)) for x in interval)):
                                # 确保起始时间 <= 结束时间
                                start, end = sorted(interval)
                                valid_intervals.append([start, end])
                            else:
                                print(f"[WARN] 第{line_num}行{action_type}的时间段格式无效: {interval}")
                    
                    # 核心调整：仅当有效时间段非空时，才加入字典
                    if valid_intervals:
                        action_dict[action_type] = valid_intervals
                    # 空时间段则直接跳过，不加入字典
                        
                except Exception as e:
                    print(f"[WARN] 第{line_num}行解析失败: {line}, 错误: {str(e)}")
                    # 解析失败也跳过该动作，不加入字典
        
        except Exception as e:
            print(f"[ERROR] 解析action_sequence.txt失败: {str(e)}")
        
        return action_dict

    def match_action_by_timestamp(self, relative_ts, action_dict):
        """
        根据相对时间戳匹配对应的动作类型（action）
        匹配规则：
            1. 按action_dict的键遍历顺序（从前到后）检查每个动作的时间段
            2. 若任意一个时间段包含该时间戳（start ≤ ts ≤ end），立即返回该动作名称
            3. 未匹配到任何动作时返回默认值 "lane_keeping_and_centering"
        
        参数:
            relative_ts: 待匹配的相对时间戳（float/int）
            action_dict: 由parse_action_sequence生成的动作字典，格式如：
                {
                    "left_turn": [[0.5, 1.0], [1.5, 2.0]],
                    "straight": [[2.5, 3.0]]
                }
        
        返回:
            str: 匹配到的第一个动作名称；未匹配返回 "lane_keeping_and_centering"
        """
        
        if not isinstance(relative_ts, (int, float)):
            print("[WARN] 时间戳格式非法，必须是数字类型: {}".format(relative_ts))
            return "lane_keeping_and_centering"
        
        if not isinstance(action_dict, dict) or len(action_dict) == 0:
            print("[WARN] 动作字典为空或格式非法")
            return "lane_keeping_and_centering"
        
        # Python2.7 字典遍历：使用keys()保证顺序（需确保action_dict是有序插入的）
        for action_name in action_dict.keys():
            time_intervals = action_dict[action_name]
            # 遍历当前动作的所有时间段
            for interval in time_intervals:
                start = min(interval[0], interval[1])
                end = max(interval[0], interval[1])
                
                # 检查时间戳是否在当前时间段内
                if start <= relative_ts <= end:
                    # print("[INFO] 时间戳{}匹配到动作: {} (时间段: [{}, {}])".format(
                    #     relative_ts, action_name, start, end))
                    return action_name
        
        # 未匹配到任何动作时返回默认值
        print("[INFO] 时间戳{}未匹配到任何动作，返回默认值: lane_keeping_and_centering".format(relative_ts))
        return "lane_keeping_and_centering"

    def parse_scenario_sequence(self):
        """
        解析 scenario_sequence.txt 文件内容，返回场景类型与对应时间段的字典
        空时间段的场景会被直接跳过，不加入字典
        
        返回格式:
            {
                "scenario_1": [[1.0, 2.5], [3.0, 4.5]],
                "scenario_2": [[5.0, 6.0]]
            }
        
        异常处理:
            - 文件未打开/不存在时返回空字典
            - 格式错误的行跳过并打印警告
        """
        scenario_dict = {}
        
        # 检查文件是否已打开
        if self.file_scenario_sequence is None:
            print("[WARN] scenario_sequence.txt 文件未打开，无法解析")
            return scenario_dict
        
        try:
            # 重置文件指针到开头
            self.file_scenario_sequence.seek(0)
            # 读取所有行并处理
            lines = self.file_scenario_sequence.readlines()
            
            for line_num, line in enumerate(lines, 1):
                # 去除空白字符和换行符
                line = line.strip()
                # 跳过空行和注释行
                if not line or line.startswith('#'):
                    continue
                
                # 去除行尾的逗号（处理示例中的逗号分隔格式）
                line = line.rstrip(',').strip()
                
                # 分割场景类型和时间段部分（按第一个冒号分割）
                if ':' not in line:
                    print(f"[WARN] 第{line_num}行格式错误，无冒号分隔: {line}")
                    continue
                
                scenario_type, time_str = line.split(':', 1)
                scenario_type = scenario_type.strip()
                time_str = time_str.strip()
                
                # 解析时间段列表
                try:
                    # 使用eval解析列表（安全场景下使用，因为是已知格式的标注文件）
                    time_intervals = eval(time_str)
                    
                    # 验证格式：必须是列表，且每个元素是包含2个数字的列表
                    valid_intervals = []
                    if isinstance(time_intervals, list):
                        for interval in time_intervals:
                            if (isinstance(interval, list) and len(interval) == 2 and
                                    all(isinstance(x, (int, float)) for x in interval)):
                                # 确保起始时间 <= 结束时间
                                start, end = sorted(interval)
                                valid_intervals.append([start, end])
                            else:
                                print(f"[WARN] 第{line_num}行{scenario_type}的时间段格式无效: {interval}")
                    
                    # 核心规则：仅当有效时间段非空时，才加入字典
                    if valid_intervals:
                        scenario_dict[scenario_type] = valid_intervals
                    # 空时间段则直接跳过，不加入字典
                        
                except Exception as e:
                    print(f"[WARN] 第{line_num}行解析失败: {line}, 错误: {str(e)}")
                    # 解析失败也跳过该场景，不加入字典
        
        except Exception as e:
            print(f"[ERROR] 解析scenario_sequence.txt失败: {str(e)}")
        
        return scenario_dict

    def match_scenario_by_timestamp(self, relative_ts, scenario_dict):
        """
        根据相对时间戳匹配对应的场景类型（scenario）
        匹配规则：
            1. 按scenario_dict的键遍历顺序（从前到后）检查每个场景的时间段
            2. 若任意一个时间段包含该时间戳（start ≤ ts ≤ end），立即返回该场景名称
            3. 未匹配到任何场景时返回默认值 "lane_keep"
        
        参数:
            relative_ts: 待匹配的相对时间戳（float/int）
            scenario_dict: 由parse_scenario_sequence生成的场景字典，格式如：
                {
                    "scenario_1": [[0.5, 1.0], [1.5, 2.0]],
                    "scenario_2": [[2.5, 3.0]]
                }
        
        返回:
            str: 匹配到的第一个场景名称；未匹配返回 "lane_keep"
        """
        
        if not isinstance(relative_ts, (int, float)):
            print("[WARN] 时间戳格式非法，必须是数字类型: {}".format(relative_ts))
            return "lane_keep"
        
        if not isinstance(scenario_dict, dict) or len(scenario_dict) == 0:
            print("[WARN] 场景字典为空或格式非法")
            return "lane_keep"
        
        # Python2.7 字典遍历：使用keys()保证顺序（需确保scenario_dict是有序插入的）
        for scenario_name in scenario_dict.keys():
            time_intervals = scenario_dict[scenario_name]
            # 遍历当前场景的所有时间段
            for interval in time_intervals:
                start = min(interval[0], interval[1])
                end = max(interval[0], interval[1])
                
                # 检查时间戳是否在当前时间段内
                if start <= relative_ts <= end:
                    # print("[INFO] 时间戳{}匹配到场景: {} (时间段: [{}, {}])".format(
                    #     relative_ts, scenario_name, start, end))
                    return scenario_name
        
        # 未匹配到任何场景时返回默认值
        print("[INFO] 时间戳{}未匹配到任何场景，返回默认值: lane_keep".format(relative_ts))
        return "lane_keep"

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

        # 获取 meta_action/scenario_type 与其对应时间段的字典
        action_dict = self.parse_action_sequence()
        scenario_dict = self.parse_scenario_sequence()

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
            sample_data = self._create_sample_data(sampled_ts, action_dict, scenario_dict, front_ts_for_image=front_ts)

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

    def _create_sample_data(self, current_ts, action_dict, scenario_dict, front_ts_for_image=None):
        """
        创建单个样本的数据。

        参数:
            current_ts: 当前帧的时间戳（来自2Hz采样）
            action_dict: meta action 与持续时间段对应关系
            scenario_dict: 场景类型与持续时间段对应关系
            front_ts_for_image: 用于获取图像的时间戳（如果为None则使用current_ts）
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

        # 获取 meta action 和 scenario type
        relative_ts = current_ts - current_ts_loc[0]
        meta_action = self.match_action_by_timestamp(relative_ts, action_dict)
        scenario_type = self.match_scenario_by_timestamp(relative_ts, scenario_dict)

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

        sample_data['meta_action'] = meta_action
        sample_data['scenario_type'] = scenario_type

        return sample_data

    def close(self):
        """关闭已打开的 bag"""
        if self.master_bag is not None:
            self.master_bag.close()
            self.master_bag = None
        if self.slave_105_bag is not None:
            self.slave_105_bag.close()
            self.slave_105_bag = None

        # 关闭 sequence.txt 文件
        if self.file_action_sequence is not None:
            self.file_action_sequence.close()
            self.file_action_sequence = None
        if self.file_scenario_sequence is not None:
            self.file_scenario_sequence.close()
            self.file_scenario_sequence = None

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
        # 解析高阶意图导航信息
        self.parse_routing()
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
