#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python3 /D2/NXT/vlm/miaobinqi/Senna/mogo_test/mogo-dataset-tools/parseRosBag-2O_Poutine.py /D2/NXT/vlm/dataset/MOGO/test_data/2O-bags
"""
import sys
import os
import pickle
import json
import ast
import bisect
import math
import re
import argparse
import sqlite3
from datetime import datetime, timezone
import numpy as np
import base64
import codecs
import cv2
from typing import Any, Dict, List, Tuple, Optional

try:
    import pyproj
except ImportError:
    pyproj = None

try:
    from shapely import wkt as shapely_wkt
except ImportError:
    shapely_wkt = None

try:
    from map.lane_frenet import LaneFrenetMatcher
except ImportError:
    LaneFrenetMatcher = None

try:
    from hdmap_vector_generator import HDMapVectorGenerator
except ImportError:
    HDMapVectorGenerator = None

try:
    import rosbag
except ImportError:
    rosbag = None
    from types import SimpleNamespace
    from rosbags.rosbag1 import Reader as RosbagReader
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PROTO_COMMON_DIR = os.path.join(SCRIPT_DIR, "proto", "common")
if PROTO_COMMON_DIR not in sys.path:
    sys.path.insert(0, PROTO_COMMON_DIR)

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None
from proto.common.header_pb2 import Header
from proto.common.localization_pb2 import Localization
from proto.common.trajectory_pb2 import Trajectory, TrajectoryPoint
# from proto.common.sensor_meta_pb2 import RegisteredSensor
from proto.common.geometry_pb2 import Point
from proto.common.routing_pb2 import  RoutingRequest, RoutingResult, RoutePath, RouteRoad, RouteLink, RouteSection, RouteLane, RouteRoadLink, Waypoint
from proto.common.hadmap_pb2 import LaneLink, LaneType
from google.protobuf import text_format

if rosbag is None:
    _BINARY_DATA_MSGDEF = (
        "Header header\nstring name\nuint32 size\nuint8[] data\n"
        "\n================================================================================\n"
        "MSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n"
    )

    def _make_typestore():
        """为纯 Python3 环境注册 BinaryData 消息定义。"""
        typestore = get_typestore(Stores.ROS1_NOETIC)
        typestore.register(get_types_from_msg(_BINARY_DATA_MSGDEF, "autopilot_msgs/msg/BinaryData"))
        return typestore

    _TYPESTORE = _make_typestore()

    def _decode_rosbags_message(connection, rawdata):
        """将 rosbags 读取结果转换为脚本现有逻辑可消费的消息对象。"""
        if "BinaryData" in connection.msgtype:
            binary_msg = _TYPESTORE.deserialize_ros1(rawdata, "autopilot_msgs/msg/BinaryData")
            return SimpleNamespace(data=bytes(binary_msg.data))
        return _TYPESTORE.deserialize_ros1(rawdata, connection.msgtype)

    class _CompatBag(object):
        """为 rosbags 提供最小 rosbag.Bag 兼容接口。"""

        def __init__(self, bag_path, mode="r"):
            if mode != "r":
                raise ValueError("仅支持只读模式")
            self._reader = RosbagReader(bag_path)
            self._reader.open()

        def read_messages(self, topics=None):
            if topics is None:
                connections = list(self._reader.connections)
            else:
                if isinstance(topics, str):
                    topic_set = {topics}
                else:
                    topic_set = set(topics)
                connections = [conn for conn in self._reader.connections if conn.topic in topic_set]

            for connection, timestamp_ns, rawdata in self._reader.messages(connections=connections):
                yield connection.topic, _decode_rosbags_message(connection, rawdata), timestamp_ns

        def get_type_and_topic_info(self):
            topic_info = {}
            for connection in self._reader.connections:
                topic_info[connection.topic] = connection
            return (None, topic_info)

        def close(self):
            self._reader.close()

    class _RosbagCompat(object):
        Bag = _CompatBag

    rosbag = _RosbagCompat()
#实车
# DEFAULT_HDMAP_POINT_CLOUD_RANGE = [-81.6, -48.0, -1.0, 97.6, 48.0, 3.0]
# DEFAULT_HDMAP_VOXEL_SIZE = [0.1, 0.1, 0.2]
# DEFAULT_HDMAP_BEV_HEIGHT = 1792
# DEFAULT_HDMAP_BEV_WIDTH = 960
DEFAULT_HDMAP_POINT_CLOUD_RANGE = [-48.0, -81.6, -1.0, 48.0, 97.6, 3.0]
DEFAULT_HDMAP_VOXEL_SIZE = [0.1, 0.1, 0.2]
DEFAULT_HDMAP_BEV_HEIGHT = 960
DEFAULT_HDMAP_BEV_WIDTH = 1792
HDMAP_CHANNELS = {
    "lanes": 0,
    "lane_links": 1,
    "lane_boundaries": 2,
    "junctions": 3,
}
COMPLETION_MARKER_FILENAME = "parse_complete.json"
ACTION_MAP = {
    LaneType.Value("LANE_TYPE_JuncTurnLeft"): "turn left",
    LaneType.Value("LANE_TYPE_JuncStraight"): "straight",
    LaneType.Value("LANE_TYPE_JuncTurnRight"): "turn right",
    LaneType.Value("LANE_TYPE_Turn"): "u turn",
}
ACTION_MAP_SCENARIO_TYPES = set(ACTION_MAP.values())


def _require_hdmap_dependencies():
    if pyproj is None:
        raise ImportError(
            "生成 HDMap BEV 需要 `pyproj`，请先执行 `python -m pip install pyproj`。"
        )
    if shapely_wkt is None:
        raise ImportError(
            "生成 HDMap BEV 需要 `shapely`，请先执行 `python -m pip install shapely`。"
        )


def _load_spatialite_extension(conn):
    extension_candidates = []
    env_extension = os.getenv("SPATIALITE_EXTENSION")
    if env_extension:
        extension_candidates.append(env_extension)
    extension_candidates.extend(["mod_spatialite", "mod_spatialite.so"])

    errors = []
    for extension_name in extension_candidates:
        try:
            conn.load_extension(extension_name)
            return
        except sqlite3.OperationalError as exc:
            errors.append("{}: {}".format(extension_name, exc))

    raise RuntimeError(
        "加载 Spatialite 扩展失败，请安装 `libsqlite3-mod-spatialite` "
        "或设置环境变量 `SPATIALITE_EXTENSION`。尝试结果：{}".format(
            "; ".join(errors)
        )
    )


class HDMapBEVGenerator:
    def __init__(
        self,
        db_path,
        point_cloud_range=None,
        voxel_size=None,
        bev_height=DEFAULT_HDMAP_BEV_HEIGHT,
        bev_width=DEFAULT_HDMAP_BEV_WIDTH,
        utm_zone=50,
        utm_south=False,
    ):
        _require_hdmap_dependencies()
        self.db_path = db_path
        self.point_cloud_range = list(
            point_cloud_range or DEFAULT_HDMAP_POINT_CLOUD_RANGE
        )
        self.voxel_size = list(voxel_size or DEFAULT_HDMAP_VOXEL_SIZE)
        self.bev_height = int(bev_height)
        self.bev_width = int(bev_width)
        self.utm_zone = utm_zone
        self.utm_south = utm_south

        if len(self.point_cloud_range) != 6:
            raise ValueError("point_cloud_range 必须包含 6 个数值")
        if len(self.voxel_size) != 3:
            raise ValueError("voxel_size 必须包含 3 个数值")
        if self.bev_height <= 0 or self.bev_width <= 0:
            raise ValueError("bev_height 和 bev_width 必须为正整数")
        if not os.path.isfile(self.db_path):
            raise IOError("HDMap sqlite 文件不存在：{}".format(self.db_path))

        self.connection = self._open_database(self.db_path)
        self._init_transformers()

        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range
        width = x_max - x_min
        height = y_max - y_min
        radius_m = int(math.sqrt(width ** 2 + height ** 2) / 2)
        self.query_range = [-radius_m, -radius_m, radius_m, radius_m]

    @staticmethod
    def _open_database(db_path):
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        _load_spatialite_extension(conn)
        return conn

    def _init_transformers(self):
        proj4 = "+proj=utm +zone={} +datum=WGS84 +units=m +no_defs".format(
            self.utm_zone
        )
        if self.utm_south:
            proj4 += " +south"
        utm_crs = pyproj.CRS.from_proj4(proj4)
        self._wgs84_to_utm = pyproj.Transformer.from_crs(
            "EPSG:4326", utm_crs, always_xy=True
        )
        self._utm_to_wgs84 = pyproj.Transformer.from_crs(
            utm_crs, "EPSG:4326", always_xy=True
        )

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def build(self, ego_utm_x, ego_utm_y, ego_utm_z, ego_yaw):
        lon, lat = self._utm_to_wgs84.transform(ego_utm_x, ego_utm_y)
        hdmap_global = self._load_hdmap_around_ego(lon, lat)
        hdmap_local = self._transform_hdmap_to_local(
            hdmap_global,
            ego_position=np.array([ego_utm_x, ego_utm_y, ego_utm_z], dtype=np.float32),
            ego_yaw=ego_yaw,
        )
        hdmap_dense = self._densify_hdmap(hdmap_local)
        return self._rasterize_hdmap(hdmap_dense)

    def _compute_query_bounds(self, lon, lat):
        deg_per_meter_lon = 1.0 / 111320.0
        deg_per_meter_lat = 1.0 / 110540.0
        x_min, y_min, x_max, y_max = self.query_range
        lon_min = lon + x_min * deg_per_meter_lon
        lon_max = lon + x_max * deg_per_meter_lon
        lat_min = lat + y_min * deg_per_meter_lat
        lat_max = lat + y_max * deg_per_meter_lat
        return lon_min, lat_min, lon_max, lat_max

    def _query_geometry_rows(self, table_name, lon, lat):
        lon_min, lat_min, lon_max, lat_max = self._compute_query_bounds(lon, lat)
        sql = """
        SELECT AsText(geom)
        FROM {}
        WHERE ST_Intersects(
            geom,
            BuildMBR(?, ?, ?, ?, 4326)
        )
        """.format(table_name)
        cursor = self.connection.cursor()
        cursor.execute(sql, (lon_min, lat_min, lon_max, lat_max))
        return cursor.fetchall()

    def _geometry_to_numpy(self, row_geom, geom_type):
        if not row_geom:
            return []

        geom = shapely_wkt.loads(row_geom)
        if geom.is_empty:
            return []

        if geom_type == "polygon":
            if geom.geom_type == "Polygon":
                geoms = [geom]
            elif geom.geom_type == "MultiPolygon":
                geoms = list(geom.geoms)
            else:
                return []

            def coord_getter(item):
                return np.asarray(item.exterior.coords, dtype=np.float64)
        else:
            if geom.geom_type == "LineString":
                geoms = [geom]
            elif geom.geom_type == "MultiLineString":
                geoms = list(geom.geoms)
            else:
                return []

            def coord_getter(item):
                return np.asarray(item.coords, dtype=np.float64)

        arrays = []
        for item in geoms:
            coords = coord_getter(item)
            if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
                continue
            xs, ys = self._wgs84_to_utm.transform(coords[:, 0], coords[:, 1])
            points = np.stack(
                [np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)],
                axis=1,
            )
            arrays.append(points)
        return arrays

    def _load_hdmap_around_ego(self, lon, lat):
        hdmap = {
            "lanes": [],
            "lane_links": [],
            "lane_boundaries": [],
            "junctions": [],
        }
        table_specs = {
            "lanes": ("TX_LANE", "line"),
            "lane_links": ("TX_LANE_LINK", "line"),
            "lane_boundaries": ("TX_LANE_BOUNDARY", "line"),
            "junctions": ("TX_JUNCTION", "polygon"),
        }
        for key, (table_name, geom_type) in table_specs.items():
            rows = self._query_geometry_rows(table_name, lon, lat)
            for (row_geom,) in rows:
                hdmap[key].extend(self._geometry_to_numpy(row_geom, geom_type))
        return hdmap

    def _transform_hdmap_to_local(self, hdmap_global, ego_position, ego_yaw):
        rotation = build_rotation_matrix_from_yaw(ego_yaw)
        transformed = {}
        for key, geometries in hdmap_global.items():
            transformed[key] = []
            for points in geometries:
                points_3d = np.zeros((points.shape[0], 3), dtype=np.float32)
                points_3d[:, :2] = points[:, :2]
                points_local = (points_3d - ego_position) @ rotation
                transformed[key].append(points_local[:, :2].astype(np.float32))
        return transformed

    @staticmethod
    def _densify_polyline(points, step):
        if len(points) < 2:
            return points.astype(np.float32)

        dense_points = [points[0]]
        for idx in range(1, len(points)):
            point_start = points[idx - 1]
            point_end = points[idx]
            distance = np.linalg.norm(point_end - point_start)
            if distance < step:
                dense_points.append(point_end)
                continue
            samples = max(1, int(distance / step))
            for sample_idx in range(1, samples + 1):
                dense_points.append(
                    point_start + (point_end - point_start) * sample_idx / samples
                )
        return np.asarray(dense_points, dtype=np.float32)

    def _densify_hdmap(self, hdmap_local):
        line_keys = {"lanes", "lane_links", "lane_boundaries"}
        hdmap_dense = {}
        for key, geometries in hdmap_local.items():
            hdmap_dense[key] = []
            for points in geometries:
                if key in line_keys:
                    if len(points) >= 2:
                        hdmap_dense[key].append(
                            self._densify_polyline(points, step=self.voxel_size[0])
                        )
                elif len(points) >= 3:
                    hdmap_dense[key].append(points.astype(np.float32))
        return hdmap_dense

    def _points_to_bev_indices(self, points):
        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        x_max, y_max = self.point_cloud_range[3], self.point_cloud_range[4]
        vx, vy = self.voxel_size[0], self.voxel_size[1]

        valid_mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] < x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] < y_max)
        )
        points = points[valid_mask]
        if len(points) == 0:
            return None, None

        rows = np.clip(
            ((points[:, 0] - x_min) / vx).astype(np.int32),
            0,
            self.bev_height - 1,
        )
        cols = np.clip(
            ((points[:, 1] - y_min) / vy).astype(np.int32),
            0,
            self.bev_width - 1,
        )
        return rows, cols

    def _fill_polygon_bev(self, points, bev_feat, channel):
        if len(points) < 3:
            return

        x_min, y_min = self.point_cloud_range[0], self.point_cloud_range[1]
        vx, vy = self.voxel_size[0], self.voxel_size[1]

        rows = np.clip(
            np.round((points[:, 0] - x_min) / vx).astype(np.int32),
            0,
            self.bev_height - 1,
        )
        cols = np.clip(
            np.round((points[:, 1] - y_min) / vy).astype(np.int32),
            0,
            self.bev_width - 1,
        )
        polygon = np.stack([cols, rows], axis=1)
        if len(np.unique(polygon, axis=0)) < 3:
            return

        mask = np.zeros((self.bev_height, self.bev_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.reshape(-1, 1, 2)], 1)
        bev_feat[channel] = np.maximum(bev_feat[channel], mask.astype(np.float32))

    def _rasterize_hdmap(self, hdmap_dense):
        bev_feat = np.zeros(
            (len(HDMAP_CHANNELS), self.bev_height, self.bev_width), dtype=np.float32
        )

        for key, geometries in hdmap_dense.items():
            channel = HDMAP_CHANNELS[key]
            if key == "junctions":
                for polygon in geometries:
                    self._fill_polygon_bev(polygon, bev_feat, channel)
                continue

            for points in geometries:
                rows, cols = self._points_to_bev_indices(points)
                if rows is None:
                    continue
                bev_feat[channel, rows, cols] = 1.0

        return bev_feat

# 计算header时间戳（秒）
def calculate_header_timestamp(header):
    """Calculate ROS Header timestamp in seconds."""
    sec = float(header.stamp.sec)
    nsec = float(header.stamp.nsec) if hasattr(header.stamp, 'nsec') else 0.0
    return sec + nsec * 1.0e-9


DEFAULT_VEHICLE_ID = "JLBJA20073D"


def path_stem(path):
    """提取图像路径的文件名 stem，兼容 bytes/str。"""
    if isinstance(path, bytes):
        try:
            path = path.decode("utf-8")
        except Exception:
            path = path.decode("latin1", errors="ignore")
    if not isinstance(path, str) or not path:
        return ""
    return os.path.splitext(os.path.basename(path))[0]


def make_token(vehicle_id, image_path):
    stem = path_stem(image_path)
    if not stem:
        return ""
    return "{}_{}".format(vehicle_id, stem)


def read_vehicle_id_from_metadata_json(bag_dir):
    """优先从 clip 目录内元数据 JSON 读取 CAR_NUMBER。"""
    for filename in sorted(os.listdir(bag_dir)):
        if not filename.endswith(".json"):
            continue

        metadata_path = os.path.join(bag_dir, filename)
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            continue

        if isinstance(metadata, dict):
            car_number = metadata.get("CAR_NUMBER")
            if isinstance(car_number, str):
                car_number = car_number.strip()
                if car_number:
                    return car_number

    return None


def infer_vehicle_id(bag_dir, master_bag_path):
    vehicle_id = read_vehicle_id_from_metadata_json(bag_dir)
    if vehicle_id:
        return vehicle_id

    master_bag_name = os.path.basename(master_bag_path or "")
    if "-rosmaster-" in master_bag_name:
        suffix = master_bag_name.split("-rosmaster-", 1)[1]
        vehicle_id = suffix.rsplit(".bag", 1)[0].strip()
        if vehicle_id:
            return vehicle_id

    return DEFAULT_VEHICLE_ID


def build_scene_token(vehicle_id, clip_dir_path):
    normalized_path = os.path.normpath(clip_dir_path)
    parts = [part for part in normalized_path.split(os.sep) if part]
    if len(parts) >= 2:
        tail = os.path.join(parts[-2], parts[-1])
    elif parts:
        tail = parts[-1]
    else:
        tail = ""
    return "{}_{}".format(vehicle_id, tail) if tail else vehicle_id


def serialize_for_json(obj):
    """
    将数据序列化为JSON兼容格式
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
    """
    从yaw角度构建旋转矩阵（绕Z轴旋转） 
    
    参数:
        yaw: 航向角（弧度），从北向（Y轴正方向）逆时针旋转的角度
    
    返回:
        3x3旋转矩阵
    """
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    R = np.array([
        [sin_yaw,  cos_yaw, 0.0],  # X轴：前进方向
        [-cos_yaw, sin_yaw, 0.0],  # Y轴：左侧方向
        [0.0,      0.0,     1.0]   # Z轴：垂直向上
    ])
    
    return R


def override_scenario_type_from_navigation_command(
    scenario_type, navigation_command
):
    if not isinstance(navigation_command, list) or not navigation_command:
        return scenario_type

    first_navigation = navigation_command[0]
    if not isinstance(first_navigation, (list, tuple)) or not first_navigation:
        return scenario_type
    first_action=None
    if first_navigation[1]<15:
        first_action = first_navigation[0]
    if first_action in ACTION_MAP_SCENARIO_TYPES and scenario_type=="lane_keep":
        return first_action

    return scenario_type

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

    def __init__(
        self,
        bag_dir,
        image_dir=None,
        output_dir=None,
        label_txt_path=None,
        label_json_path=None,
        work_dir=None,
        write_json=False,
        json_output_dir=None,
        hdmap_sql_path=None,
        hdmap_point_cloud_range=None,
        hdmap_voxel_size=None,
        hdmap_bev_height=DEFAULT_HDMAP_BEV_HEIGHT,
        hdmap_bev_width=DEFAULT_HDMAP_BEV_WIDTH,
        hdmap_utm_zone=50,
        hdmap_utm_south=False,
        hdmap_vector_radius=None,
        route_result=None,
    ):
        self.bag_dir = bag_dir
        self.image_dir = image_dir or bag_dir
        self.work_dir = work_dir or os.getcwd()
        # 输出目录，如果不指定则使用bag_dir
        self.output_dir = output_dir or bag_dir
        self.write_json = write_json
        self.json_output_dir = json_output_dir
        self.label_txt_path = label_txt_path
        self.label_json_path = label_json_path
        self.segment_time_origin = None
        self.hdmap_sql_path = hdmap_sql_path
        self.hdmap_point_cloud_range = list(
            hdmap_point_cloud_range or DEFAULT_HDMAP_POINT_CLOUD_RANGE
        )
        self.hdmap_voxel_size = list(hdmap_voxel_size or DEFAULT_HDMAP_VOXEL_SIZE)
        self.hdmap_bev_height = int(hdmap_bev_height)
        self.hdmap_bev_width = int(hdmap_bev_width)
        self.hdmap_utm_zone = hdmap_utm_zone
        self.hdmap_utm_south = hdmap_utm_south
        # None 表示让 HDMapVectorGenerator 用其内部 DEFAULT_RADIUS_M (120m, PlanVOCA 默认)
        self.hdmap_vector_radius = hdmap_vector_radius

        # bag 文件路径
        self.bags_file_master = None
        self.bags_file_105 = None
        self.bags_file_routing = None

        # 已打开的 bag
        self.master_bag = None
        self.slave_105_bag = None
        self.routing_bag = None

        # 相机相关
        self.camera_topic_dict = {}
        self.camera_cache = {}

        # 定位相关
        self.valid_timestamps = []
        self.valid_locations = []
        
        # 2Hz采样后的定位数据
        self.sampled_2hz_timestamps = []
        self.sampled_2hz_locations = []
        self.sampled_2hz_indices = []  # 原始索引，用于查找原始数据
        self.sampled_2hz_p2lane = []
        self.sampled_2hz_p2lane_stable = []


        # 存储解码后的RoutingResult对象；同一路线的后续 bag 可复用首个 bag 的结果。
        self.route_result = route_result
        # 每个定位点的高阶意图列表（格式：[[动作, 距离, 车道起, 车道止], ...]）每个定位点包含2个高阶意图。如无则NONE
        self.high_order_intent = []
        # PlanVOCA-style on_route 判定使用：route 上的 lane / lane_link 唯一键集合
        self.route_lane_keys = set()
        self.route_link_keys = set()
        # vector 失败计数：首次失败立刻抛，避免整库 sweep 全是 None 才发现
        self._hdmap_vector_failed = False

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

        os.makedirs(self.output_dir, exist_ok=True)

        self.hdmap_bev_dir = None
        self.hdmap_generator = None
        self.hdmap_vector_generator = None
        self.lane_frenet_matcher = None
        if self.hdmap_sql_path:
            self.hdmap_bev_dir = os.path.join(self.output_dir, "hdmap_bev_npz")
            os.makedirs(self.hdmap_bev_dir, exist_ok=True)
            self.hdmap_generator = HDMapBEVGenerator(
                db_path=self.hdmap_sql_path,
                point_cloud_range=self.hdmap_point_cloud_range,
                voxel_size=self.hdmap_voxel_size,
                bev_height=self.hdmap_bev_height,
                bev_width=self.hdmap_bev_width,
                utm_zone=self.hdmap_utm_zone,
                utm_south=self.hdmap_utm_south,
            )
            if HDMapVectorGenerator is not None:
                try:
                    vector_kwargs = dict(
                        db_path=self.hdmap_sql_path,
                        utm_zone=self.hdmap_utm_zone,
                        utm_south=self.hdmap_utm_south,
                    )
                    if self.hdmap_vector_radius is not None:
                        vector_kwargs["radius"] = float(self.hdmap_vector_radius)
                    self.hdmap_vector_generator = HDMapVectorGenerator(**vector_kwargs)
                    print(
                        "[INFO] 启用 HDMap vector token 生成（PlanVOCA-style），radius={}m".format(
                            self.hdmap_vector_generator.radius
                        )
                    )
                except Exception as exc:
                    self.hdmap_vector_generator = None
                    print("[WARN] 初始化 HDMap vector generator 失败：{}".format(exc))
            else:
                print("[WARN] hdmap_vector_generator 模块不可用，跳过 vector token 生成")
            if LaneFrenetMatcher is not None:
                try:
                    self.lane_frenet_matcher = LaneFrenetMatcher(
                        db_path=self.hdmap_sql_path,
                        utm_zone=self.hdmap_utm_zone,
                        utm_south=self.hdmap_utm_south,
                    )
                    print("[INFO] 启用 sampled p2lane 计算")
                except Exception as exc:
                    self.lane_frenet_matcher = None
                    print("[WARN] 初始化 sampled p2lane 失败：{}".format(exc))
            else:
                print("[WARN] lane frenet helper 不可用，sampled p2lane 将为空")
            print("[INFO] 启用 HDMap BEV 生成：{}".format(self.hdmap_sql_path))
        else:
            print("[WARN] 未提供 HDMap sqlite，样本字段 hdmap_bev 将为空")

        # json文件夹，用于保存每个sample的json文件，命名为poutine_base_json
        self.json_samples_dir = None
        if self.write_json:
            if self.json_output_dir:
                self.json_samples_dir = self.json_output_dir
            else:
                self.json_samples_dir = os.path.join(
                    self.output_dir, "poutine_base_json"
                )
            if not os.path.exists(self.json_samples_dir):
                try:
                    os.makedirs(self.json_samples_dir)
                except OSError:
                    if not os.path.isdir(self.json_samples_dir):
                        raise

        self._check_and_find_bags()
        self.vehicle_id = infer_vehicle_id(self.bag_dir, self.bags_file_master)
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
                    # self.bags_file_routing = fp
                elif "-rosslave-" in filename:
                    self.bags_file_105 = fp
                elif "routing_result" in filename:
                    print("[INFO] Find Routing Bag!!!")
                    self.bags_file_routing = fp

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

        if self.bags_file_routing:
            try:
                self.routing_bag = rosbag.Bag(self.bags_file_routing, "r")
            except Exception as e:
                print("[WARN] 打开Routing Bag失败，将跳过导航解析：{}".format(e))
                self.routing_bag = None
        else:
            print("[WARN] 未找到Routing Bag，将跳过导航解析")


    def _check_and_find_sequence_files(self):
        """检查目录并查找 action_sequence.txt 和 scenario_sequence.txt 文件"""
        # 初始化文件路径为None
        self.file_path_action_sequence = None
        self.file_path_scenario_sequence = None

        if self.label_txt_path:
            self.file_path_scenario_sequence = self.label_txt_path
            print("[INFO] 标签场景文件路径：{}".format(self.file_path_scenario_sequence))
            if self.label_json_path:
                print("[INFO] 标签JSON路径：{}".format(self.label_json_path))

        if not os.path.isdir(self.bag_dir):
            raise IOError("目录{}无效".format(self.bag_dir))

        # 遍历目录查找目标文件
        for filename in os.listdir(self.bag_dir):
            fp = os.path.join(self.bag_dir, filename)
            if filename == "action_sequence.txt":
                self.file_path_action_sequence = fp
            elif filename == "scenario_sequence.txt" and not self.file_path_scenario_sequence:
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

    def _parse_sequence_file(self, file_obj, file_desc):
        """解析 sequence 文件，并合并同名标签的多个时间区间。"""
        sequence_dict = {}

        if file_obj is None:
            print("[WARN] {} 文件未打开，无法解析".format(file_desc))
            return sequence_dict

        try:
            file_obj.seek(0)
            lines = file_obj.readlines()

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                line = line.rstrip(',').strip()
                if ':' not in line:
                    print("[WARN] 第{}行格式错误，无冒号分隔: {}".format(line_num, line))
                    continue

                label_name, time_str = line.split(':', 1)
                label_name = label_name.strip()
                time_str = time_str.strip()

                try:
                    time_intervals = ast.literal_eval(time_str)
                    valid_intervals = []
                    if isinstance(time_intervals, list):
                        for interval in time_intervals:
                            if (
                                isinstance(interval, list)
                                and len(interval) == 2
                                and all(isinstance(x, (int, float)) for x in interval)
                            ):
                                start, end = sorted(interval)
                                valid_intervals.append([start, end])
                            else:
                                print(
                                    "[WARN] 第{}行{}的时间段格式无效: {}".format(
                                        line_num, label_name, interval
                                    )
                                )

                    if valid_intervals:
                        sequence_dict.setdefault(label_name, []).extend(valid_intervals)

                except Exception as e:
                    print("[WARN] 第{}行解析失败: {}, 错误: {}".format(line_num, line, str(e)))

        except Exception as e:
            print("[ERROR] 解析{}失败: {}".format(file_desc, str(e)))

        for label_name in sequence_dict:
            sequence_dict[label_name].sort(key=lambda interval: (interval[0], interval[1]))

        return sequence_dict

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
                try:
                    if self.file_action_sequence is not None:
                        self.file_action_sequence.close()
                finally:
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
                try:
                    if self.file_scenario_sequence is not None:
                        self.file_scenario_sequence.close()
                finally:
                    self.file_scenario_sequence = None
                raise RuntimeError("打开 scenario_sequence.txt 失败：{}".format(e))
        else:
            print("[WARN] scenario_sequence.txt 文件路径为空，跳过打开")

    # ---------------- public APIs ----------------
    def parse_cameras(self):
        """
        解析相机 topic，并将图片写入工作目录下的各相机子目录.
        """
        assert self.slave_105_bag is not None
        if CvBridge is None:
            raise RuntimeError("cv_bridge 未安装，无法从 bag 中直接解码相机图像")

        slave_105_topics = self.slave_105_bag.get_type_and_topic_info()[1].keys()
        print("[INFO] Slave105 Bag的topic：{}".format(slave_105_topics))
        for topic in slave_105_topics:
            if "/sensor/camera" in topic and "resize" in topic:
                print("[INFO] camera topic is: {}".format(topic))
                if "image_raw_120" in topic or "front" in topic:
                    self.camera_topic_dict["front"] = topic
                elif "front_left" in topic:
                    self.camera_topic_dict["front_left"] = topic
                elif "front_right" in topic:
                    self.camera_topic_dict["front_right"] = topic
                elif "image_raw_left_back" in topic:
                    self.camera_topic_dict["left_back"] = topic
                elif "image_raw_right_back" in topic:
                    self.camera_topic_dict["right_back"] = topic
                elif "image_raw_back" in topic or "back" in topic:
                    self.camera_topic_dict["back"] = topic
                elif "left" in topic:
                    self.camera_topic_dict["left"] = topic
                elif "right" in topic:
                    self.camera_topic_dict["right"] = topic
                
        bridge = CvBridge()
        for key in self.camera_topic_dict.keys():
            for topic, msg, t in self.slave_105_bag.read_messages(self.camera_topic_dict[key]):
                if topic.endswith("nvjpeg"):
                    cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
                else:
                    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                camera_dir = os.path.join(self.bag_dir, key)
                if not os.path.exists(camera_dir):
                    os.makedirs(camera_dir, exist_ok=True)
                ts = msg.header.stamp.to_sec()
                timestr = "%.9f" % ts                
                camera_path = os.path.join(camera_dir, timestr + ".jpg")
                cv2.imwrite(camera_path, cv_image)

        print("[INFO] 相机解析完成：共{len(self.camera_topic_dict)}路相机")


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
            camera_dir = os.path.join(self.image_dir, key)
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
        基于 front 图像时间轴做 2Hz 均匀采样，并在原始频率定位数据中
        为每个采样图像匹配最近的定位数据。
        
        参数:
            sample_rate_hz: 采样频率，默认2Hz（每0.5秒一帧）
        
        结果保存在:
            self.sampled_2hz_timestamps: 2Hz 采样后的 front 图像时间戳
            self.sampled_2hz_locations: 每个采样图像匹配到的定位数据
            self.sampled_2hz_indices: 匹配到的原始定位索引
        """
        if not self.valid_timestamps:
            print("[WARN] 定位数据为空，无法进行图像驱动的2Hz采样")
            return

        if not hasattr(self, "camera_group_by_front") or not self.camera_group_by_front:
            print("[WARN] front 图像为空，无法进行图像驱动的2Hz采样")
            return
        
        self.sampled_2hz_timestamps = []
        self.sampled_2hz_locations = []
        self.sampled_2hz_indices = []

        front_times = sorted(self.camera_group_by_front.keys())
        if not front_times:
            print("[WARN] front 图像时间轴为空，无法进行图像驱动的2Hz采样")
            return

        # 2Hz采样：每0.5秒采样一次
        sample_interval = 1.0 / sample_rate_hz  # 0.5秒
        start_time = front_times[0]
        end_time = front_times[-1]
        max_image_delta = min(0.25, sample_interval / 2.0 + 1e-6)
        max_loc_delta = 0.1

        current_time = start_time

        print("[INFO] 开始基于图像时间轴进行2Hz采样，采样频率: {}Hz，时间范围: {:.3f} - {:.3f}".format(
            sample_rate_hz, start_time, end_time))

        while current_time <= end_time:
            # 找到最接近当前时间的 front 图像
            idx = bisect.bisect_left(front_times, current_time)
            candidates = []
            if idx == 0:
                candidates.append(0)
            elif idx == len(front_times):
                candidates.append(len(front_times) - 1)
            else:
                candidates.extend([idx - 1, idx])
            
            # 选择最接近的索引
            best_front_idx = None
            best_diff = float('inf')
            for ci in candidates:
                diff = abs(front_times[ci] - current_time)
                if diff < best_diff:
                    best_diff = diff
                    best_front_idx = ci

            if best_front_idx is not None and best_diff <= max_image_delta:
                front_ts = front_times[best_front_idx]

                loc_idx, loc_diff = self._find_nearest_loc_idx(
                    front_ts, max_delta=max_loc_delta, use_2hz_sampled=False)

                if loc_idx is not None:
                    # 避免重复添加相同的图像时间戳或相同的定位索引
                    if self.sampled_2hz_timestamps and self.sampled_2hz_timestamps[-1] == front_ts:
                        current_time += sample_interval
                        continue
                    if self.sampled_2hz_indices and self.sampled_2hz_indices[-1] == loc_idx:
                        current_time += sample_interval
                        continue

                    self.sampled_2hz_timestamps.append(front_ts)
                    self.sampled_2hz_locations.append(self.valid_locations[loc_idx])
                    self.sampled_2hz_indices.append(loc_idx)
            
            # 移动到下一个采样时间点
            current_time += sample_interval

        # 尝试补齐最后一个 front 图像对应的样本
        last_front_ts = front_times[-1]
        if not self.sampled_2hz_timestamps or self.sampled_2hz_timestamps[-1] != last_front_ts:
            last_loc_idx, _ = self._find_nearest_loc_idx(
                last_front_ts, max_delta=max_loc_delta, use_2hz_sampled=False)
            if last_loc_idx is not None:
                self.sampled_2hz_timestamps.append(last_front_ts)
                self.sampled_2hz_locations.append(self.valid_locations[last_loc_idx])
                self.sampled_2hz_indices.append(last_loc_idx)

        print("[INFO] 图像驱动的2Hz采样完成：front原始帧数 {}，采样后样本数 {}，原始定位点数 {}".format(
            len(front_times), len(self.sampled_2hz_timestamps), len(self.valid_timestamps)))

    def build_sampled_p2lane(self):
        """
        基于 2Hz sampled localization 生成逐点的原始/稳定 p2lane 结果。
        """
        self.sampled_2hz_p2lane = []
        self.sampled_2hz_p2lane_stable = []

        if not self.sampled_2hz_locations:
            print("[WARN] sampled localization 为空，跳过 p2lane 计算")
            return

        default_items = [("", float("nan"), float("nan")) for _ in self.sampled_2hz_locations]
        default_stable_items = [
            ("", float("nan"), float("nan"), "") for _ in self.sampled_2hz_locations
        ]

        if not self.lane_frenet_matcher:
            print("[WARN] 未初始化 lane frenet matcher，sampled p2lane 将为空")
            self.sampled_2hz_p2lane = default_items
            self.sampled_2hz_p2lane_stable = default_stable_items
            return

        utm_points = []
        for loc in self.sampled_2hz_locations:
            if hasattr(loc, "position") and loc.position:
                utm_points.append((float(loc.position.x), float(loc.position.y)))
            else:
                utm_points.append((float("nan"), float("nan")))

        valid_points = [point for point in utm_points if math.isfinite(point[0]) and math.isfinite(point[1])]
        if not valid_points:
            print("[WARN] sampled localization 缺少有效 UTM 点，跳过 p2lane 计算")
            self.sampled_2hz_p2lane = default_items
            self.sampled_2hz_p2lane_stable = default_stable_items
            return

        try:
            matched_results = self.lane_frenet_matcher.match_utm_points_detailed(valid_points)
        except Exception as exc:
            print("[WARN] 计算 sampled p2lane 失败：{}".format(exc))
            self.sampled_2hz_p2lane = default_items
            self.sampled_2hz_p2lane_stable = default_stable_items
            return

        result_iter = iter(matched_results)
        filled_results = []
        filled_stable_results = []
        matched_count = 0
        stable_matched_count = 0
        for point in utm_points:
            if math.isfinite(point[0]) and math.isfinite(point[1]):
                item = next(result_iter, None)
                if item is None:
                    raw_item = ("", float("nan"), float("nan"))
                    stable_item = ("", float("nan"), float("nan"), "")
                else:
                    raw_item = item.as_raw_tuple()
                    stable_item = item.as_stable_tuple()
            else:
                raw_item = ("", float("nan"), float("nan"))
                stable_item = ("", float("nan"), float("nan"), "")
            if raw_item[0]:
                matched_count += 1
            if stable_item[0]:
                stable_matched_count += 1
            filled_results.append(raw_item)
            filled_stable_results.append(stable_item)

        self.sampled_2hz_p2lane = filled_results
        self.sampled_2hz_p2lane_stable = filled_stable_results
        print(
            "[INFO] sampled p2lane 计算完成：共{}个点，原始匹配{}个，stable匹配{}个".format(
                len(self.sampled_2hz_p2lane), matched_count, stable_matched_count
            )
        )

    def parse_routing(self):
        """
        核心流程：解析路由→提取稀疏点→统计路口→匹配自车到路口的路径距离
        关键索引逻辑（正序）：
        - roadlink_idx 从小到大 → 实际行驶路径从近（自车）到远（前方）
        - 目标roadlink_idx > 当前roadlink_idx → 目标在自车前方（有效）
        """
        # 基础配置
        MAX_PATH_DISTANCE = 400.0  # 路径距离阈值（米）
        MAX_MATCH_JUNCTION = 2     # 最多匹配2个路口

        # 路口动作映射
        #    ========== 修正后的路口动作映射 ==========
        # ========== 修正后的目标lane类型集合 ==========
        target_lane_types = set(ACTION_MAP.keys())

        # ========== 辅助函数 ==========
        def get_junction_unique_id(lane_link):
            """生成路口唯一标识（避免重复统计）"""
            from_road = lane_link.from_road_pkid if lane_link.HasField("from_road_pkid") else 0
            to_road = lane_link.to_road_pkid if lane_link.HasField("to_road_pkid") else 0
            from_junc = lane_link.from_junction if lane_link.HasField("from_junction") else False
            to_junc = lane_link.to_junction if lane_link.HasField("to_junction") else False
            return f"{from_road}_{to_road}_{from_junc}_{to_junc}"
        
        def is_point_behind_ego(ego_x: float, ego_y: float, ego_yaw: float, 
                        target_x: float, target_y: float) -> bool:
            """
            判断目标点是否在自车后方（适配：yaw=0正东，逆时针为正的ENU坐标系）
            :param ego_x: 自车定位点x坐标（东向）
            :param ego_y: 自车定位点y坐标（北向）
            :param ego_yaw: 自车朝向角（弧度，yaw=0正东，逆时针增大）
            :param target_x: 目标点x坐标（东向）
            :param target_y: 目标点y坐标（北向）
            :return: True=在后方，False=在前方/侧前方/重合
            """
            dx = target_x - ego_x
            dy = target_y - ego_y

            if math.hypot(dx, dy) < 1e-6:
                return False
            target_azimuth = math.atan2(dy, dx)

            delta_angle = target_azimuth - ego_yaw
            delta_angle = math.atan2(math.sin(delta_angle), math.cos(delta_angle))  # 归一化

            if abs(delta_angle) > math.pi / 2:
                return True  # 后方（左后方/右后方）
            else:
                return False  # 前方（正前方/左前方/右前方）
 
        def calculate_path_distance(index, current_location, tar_junction):
            """
            计算自车到目标roadlink的路径距离（正序索引逻辑）
            :param current_loc: 自车当前位置（x/y）
            :param tar_junction: 目标路口信息字典（含roadlink_idx/action等）
            :return: 路径距离（米），无效返回nan
            """
            current_loc = current_location.position
            target_roadlink_idx = tar_junction["roadlink_idx"]
            if not sparse_points or target_roadlink_idx not in roadlink_length_map:
                return math.nan
            
            has_direction = False
            if tar_junction["action"] != "u turn":
                has_direction = True
            
            closest_roadlink_idx = None
            min_straight_dist = float('inf')
            closest_roadlink_type = ""
            for p in sparse_points:
                dx = p["x"] - current_loc.x
                dy = p["y"] - current_loc.y
                if has_direction:
                    is_point_behind = is_point_behind_ego(current_loc.x, current_loc.y, current_location.yaw, p["x"], p["y"])
                    if is_point_behind:  # 后方/侧后方点 → 跳过
                        continue
                
                straight_dist = math.hypot(dx, dy)
                if straight_dist < min_straight_dist:
                    min_straight_dist = straight_dist
                    closest_roadlink_idx = p["roadlink_idx"]
                    closest_roadlink_type = p["type"]
            
            if closest_roadlink_idx is None:
                return math.nan
            
            if target_roadlink_idx < closest_roadlink_idx:
                return math.nan
            
            
            current_road_end_point = next(
                (p for p in sparse_points if p["type"] == closest_roadlink_type and p["roadlink_idx"] == closest_roadlink_idx),
                None
            )

            if closest_roadlink_idx == target_roadlink_idx and closest_roadlink_type == "route_link":
                current_road_end_point = next(
                (p for p in sparse_points if p["type"] == "route_road" and p["roadlink_idx"] == closest_roadlink_idx - 1),
                None) 
            elif closest_roadlink_type == "route_link":
                current_road_end_point = next(
                (p for p in sparse_points if p["type"] == "route_road" and p["roadlink_idx"] == closest_roadlink_idx + 1),
                None) 



            current_offset = 0.0
            if current_road_end_point:
                dx = current_road_end_point["x"] - current_loc.x
                dy = current_road_end_point["y"] - current_loc.y
                current_offset = math.hypot(dx, dy)
            
            path_dist_sum = 0.0
            for idx in range(closest_roadlink_idx + 1, target_roadlink_idx + 1):
                if idx in roadlink_length_map:
                    path_dist_sum += roadlink_length_map[idx]
            
            final_dist = current_offset + path_dist_sum
            final_dist = max(0.0, final_dist)
            if closest_roadlink_idx == target_roadlink_idx:
                final_dist = current_offset
            
            if closest_roadlink_type == "route_link" and closest_roadlink_idx == target_roadlink_idx:
                return -1 * final_dist

            return final_dist

        # ========== 第一步：解析路由数据 ==========
        # 优先解析本 bag 自带的 routing；解析不到再回退到外部传入的缓存 route_result。
        # 这样可以同时兼容“多切片共享同一路线”（依赖缓存）和
        # “每个 bag 一条独立路线”（必须用本 bag 自己的路由）两种数据形态。
        local_route_result = None
        ROUTE_TOPIC = "/planning/route_result_debug"
        # ROUTE_TOPIC = "/hadmap_engine/route_result_msg"

        # (1) 优先使用独立的 routing_result.bag
        if self.routing_bag is not None:
            for topic, msg, time_stamp in self.routing_bag.read_messages(topics=[ROUTE_TOPIC]):
                try:
                    route_roads = RoutingResult()
                    route_roads.ParseFromString(msg.data)
                    local_route_result = route_roads
                    break
                except Exception as e:
                    print(f"[WARN] 解析路由消息失败（routing_bag）：{str(e)}")

        # (2) 退而求其次：从 master bag 中读 routing topic（worldsim 等场景）
        if local_route_result is None and self.master_bag is not None:
            try:
                master_topics = list(self.master_bag.get_type_and_topic_info()[1].keys())
            except Exception as e:
                master_topics = []
                print(f"[WARN] 获取 master bag topic 列表失败：{str(e)}")
            if ROUTE_TOPIC in master_topics:
                print(f"[INFO] routing_bag 不存在，回退到 master bag 中的 {ROUTE_TOPIC}")
                for topic, msg, _ in self.master_bag.read_messages(topics=[ROUTE_TOPIC]):
                    try:
                        route_roads = RoutingResult()
                        route_roads.ParseFromString(msg.data)
                        local_route_result = route_roads
                        break
                    except Exception as e:
                        print(f"[WARN] 解析路由消息失败（master_bag）：{str(e)}")

        if local_route_result is not None:
            if self.route_result is not None and self.route_result is not local_route_result:
                print("[INFO] 当前切片有独立路由，覆盖外部缓存的路由")
            else:
                print("[INFO] 使用当前 bag 自带的路由数据")
            self.route_result = local_route_result
        elif self.route_result is not None:
            print("[INFO] 当前切片无独立路由，复用缓存的路由数据")
        else:
            print("[WARN] routing_bag 不存在且无缓存路由，使用空导航信息")

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

        for path in self.route_result.paths:
            for roadlink_idx, roadlink in enumerate(path.route_roadlinks):
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
                            junc_data["lane_start"] =  max(junc_data["lane_start"], lane_id) if not math.isnan(junc_data["lane_start"]) else lane_id
                            junc_data["lane_end"] = min(junc_data["lane_end"], lane_id) if not math.isnan(junc_data["lane_end"]) else lane_id

        # ========== 第三步：封装路口以及终点信息（按正序索引排序） ==========
        closest_roadlink_idx = None
        min_straight_dist = float('inf')
        end_point = self.route_result.end.pose
        dir_x = sparse_points[-1]["x"] - end_point.x
        dir_y = sparse_points[-1]["y"] - end_point.y
        for p in sparse_points:
            dx = p["x"] - end_point.x
            dy = p["y"] - end_point.y
            dot_product = dir_x * dx + dir_y * dy   
            if dot_product <= 1e-6:
                continue
                
            straight_dist = math.hypot(dx, dy)
            if straight_dist < min_straight_dist:
                min_straight_dist = straight_dist
                closest_roadlink_idx = p["roadlink_idx"]
        
        junction_list = []
        junction_list.append({
            "action": "park",
            "x": end_point.x,
            "y": end_point.y,
            "lane_start": -1,
            "lane_end": -1,
            "roadlink_idx": closest_roadlink_idx 
        })
        for junc_data in lane_junction_map.values():
            junction_list.append({
                "action": junc_data["action"],
                "x": junc_data["x"],
                "y": junc_data["y"],
                "lane_start": -1 * junc_data["lane_start"],
                "lane_end": -1 * junc_data["lane_end"],
                "roadlink_idx": junc_data["roadlink_idx"] 
            })

        junction_list.sort(key=lambda x: x["roadlink_idx"])

        # ========== 第四步：匹配自车到路口的路径距离（正序逻辑） ==========
        self.high_order_intent = []
        for index, current_loc in enumerate(self.sampled_2hz_locations):
            matched_junctions = []
            candidate_junctions = []

            for junc in junction_list:
                path_dist = calculate_path_distance(index, current_loc, junc)
                
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

        # ========== 第五步：收集 route 上的 lane / lane_link 三元组键 ==========
        # 用于 HDMapVectorGenerator 标记 polygon_on_route，与 PlanVOCA
        # route_mogo_roadblock_correction 等价：lane 用 (road_pkid, section_id, lane_id)，
        # lane_link 用 (from_road, from_section, from_lane, to_road, to_section, to_lane)。
        self.route_lane_keys = set()
        self.route_link_keys = set()
        for path in self.route_result.paths:
            for roadlink in path.route_roadlinks:
                if roadlink.HasField("route_road"):
                    road = roadlink.route_road
                    road_pkid = road.pkid if road.HasField("pkid") else 0
                    if hasattr(road, "sections"):
                        for section in road.sections:
                            sec_id = section.id if section.HasField("id") else 0
                            for lane in section.lanes:
                                lane_id = lane.lane_id if lane.HasField("lane_id") else 0
                                self.route_lane_keys.add((int(road_pkid), int(sec_id), int(lane_id)))
                for lane_link in roadlink.route_links:
                    self.route_link_keys.add((
                        int(lane_link.from_road_pkid) if lane_link.HasField("from_road_pkid") else 0,
                        int(lane_link.from_section_id) if lane_link.HasField("from_section_id") else 0,
                        int(lane_link.from_lane_id) if lane_link.HasField("from_lane_id") else 0,
                        int(lane_link.to_road_pkid) if lane_link.HasField("to_road_pkid") else 0,
                        int(lane_link.to_section_id) if lane_link.HasField("to_section_id") else 0,
                        int(lane_link.to_lane_id) if lane_link.HasField("to_lane_id") else 0,
                    ))



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
        master_topics = list(self.master_bag.get_type_and_topic_info()[1].keys())
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
        return self._parse_sequence_file(self.file_action_sequence, "action_sequence.txt")

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
            return "lane_keeping_and_centering"
        
        if not isinstance(action_dict, dict) or len(action_dict) == 0:
            return "lane_keeping_and_centering"
        
        for action_name in list(action_dict.keys()):
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
        return self._parse_sequence_file(self.file_scenario_sequence, "scenario_sequence.txt")

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
            return "lane_keep"
        
        if not isinstance(scenario_dict, dict) or len(scenario_dict) == 0:
            return "lane_keep"
        
        for scenario_name in list(scenario_dict.keys()):
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
        
        return "lane_keep"

    def align_camera_cache_by_front(self, max_delta=0.2):
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

    def _get_camera_path(self, cam_key, target_ts, max_delta=0.2):
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

    def _build_and_save_hdmap_bev(
        self, image_ts, ego_utm_x, ego_utm_y, ego_utm_z, ego_yaw
    ):
        if self.hdmap_generator is None or self.hdmap_bev_dir is None:
            return b""

        file_stem = "{:.9f}".format(image_ts)
        npz_filename = os.path.join(self.hdmap_bev_dir, file_stem + ".npz")

        if not os.path.exists(npz_filename):
            hdmap_bev = self.hdmap_generator.build(
                ego_utm_x=ego_utm_x,
                ego_utm_y=ego_utm_y,
                ego_utm_z=ego_utm_z,
                ego_yaw=ego_yaw,
            )
            np.savez_compressed(npz_filename, hdmap_bev=hdmap_bev)

        return npz_filename.encode("utf-8")

    def generate_pkl_samples(self, sample_rate_hz=2.0):
        """
        按照2Hz采样频率生成pkl样本。
        
        参数:
            sample_rate_hz: 采样频率，默认2Hz（每0.5秒一帧）
        """
        if not hasattr(self, "camera_group_by_front") or not self.camera_group_by_front:
            print("[WARN] generate_pkl_samples: camera_group_by_front 为空")
            return 0

        if not self.valid_timestamps:
            print("[WARN] generate_pkl_samples: 定位数据为空")
            return 0

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

        # 使用2Hz采样后的 front 图像时间戳作为基准
        if not self.sampled_2hz_timestamps:
            print("[WARN] 2Hz采样数据为空，无法生成PKL样本")
            return 0
        
        # 使用2Hz采样后的图像时间戳序列
        sampled_times = self.sampled_2hz_timestamps
        start_time = sampled_times[0]
        end_time = sampled_times[-1]
        if "front" in self.camera_cache and self.camera_cache["front"]["times"]:
            self.segment_time_origin = self.camera_cache["front"]["times"][0]
        else:
            self.segment_time_origin = start_time
        print("[INFO] 开始生成JSON样本，基于2Hz采样后的图像时间轴，时间范围: {:.3f} - {:.3f}，共 {} 个采样点".format(
            start_time, end_time, len(sampled_times)))

        # 获取 meta_action/scenario_type 与其对应时间段的字典
        action_dict = self.parse_action_sequence()
        scenario_dict = self.parse_scenario_sequence()

        sample_idx = 0
        # 跳过前4帧，从第5帧开始生成数据
        for idx, sampled_ts in enumerate(sampled_times):
            if idx < 5:  #跳过前4帧
                continue
            if idx >= len(sampled_times) - 6: #不要后6帧
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
                try:
                    sample_basename = "{}_step{}".format(scene_name, sample_idx + 1)
                    pkl_filename = os.path.join(self.output_dir, sample_basename + ".pkl")
                    with open(pkl_filename, "wb") as f:
                        pickle.dump(sample_data, f, protocol=3)

                    if self.write_json and self.json_samples_dir:
                        json_filename = os.path.join(self.json_samples_dir, sample_basename + ".json")
                        json_data = serialize_for_json(sample_data)
                        with open(json_filename, 'w') as f:
                            json.dump(json_data, f, indent=2, ensure_ascii=False)

                    sample_idx += 1
                    if sample_idx % 100 == 0:
                        print("[INFO] 已生成 {} 个样本".format(sample_idx))
                except Exception as e:
                    print("[ERROR] 保存样本失败：{}".format(e))

        print("[INFO] 样本生成完成：共 {} 个样本，输出目录 {}".format(sample_idx, self.output_dir))
        return sample_idx
    


    def generate_navigation_strings(self, nav_intent):
        """
        生成适合自然语言模型输入的核心字符串列表（车道格式：1st lane to 2nd (left to right) are passable）
        :param nav_intent: 原始导航意图列表
        :return: [str1, str2]
        """
        def get_ordinal(num):
            """将数字转为序数词（1→1st，2→2nd，3→3rd，4+→4th）"""
            num = int(num)
            if 10 <= num % 100 <= 20:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(num % 10, 'th')
            return f"{num}{suffix}"
        core_str_list = []
        for idx, nav_item in enumerate(nav_intent):
            action = nav_item[0] if len(nav_item) > 0 else "None"
            distance = nav_item[1] if len(nav_item) > 1 else math.nan
            lane_start = nav_item[2] if len(nav_item) > 2 else math.nan
            lane_end = nav_item[3] if len(nav_item) > 3 else math.nan

            action_str = f"Navigation action is {action}" if action != "None" else "Action is None"
            
            if isinstance(distance, float) and math.isnan(distance):
                distance_str = "Distance to junction (meters) is unavailable"
            else:
                distance_str = f"Distance to junction (meters) is {distance}"
            
            if isinstance(lane_start, float) and math.isnan(lane_start) or isinstance(lane_end, float) and math.isnan(lane_end):
                lane_str = "Lane information is unavailable"
            else:
                ord_start = get_ordinal(lane_start)
                ord_end = get_ordinal(lane_end)
                lane_str = f"{ord_start} lane to {ord_end} (left to right) are passable"
            
            intent_str = f"Intent {idx}: {action_str}, {distance_str}, {lane_str}"
            core_str_list.append(intent_str)
        
        return core_str_list

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

        current_navigation_str = []
        if self.high_order_intent:
            current_navigation_str = self.generate_navigation_strings(self.high_order_intent[loc_idx])


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
        history_window = 4 #收集历史4帧轨迹  
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
        future_window = 6 #收集未来6帧轨迹  
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
        for i in range(len(future_locations)):
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

        traj_p2lane = []
        traj_p2lane_stable = []
        if self.sampled_2hz_p2lane:
            history_start_idx = loc_idx - history_window
            future_end_idx = loc_idx + future_window
            for sampled_idx in range(history_start_idx, future_end_idx + 1):
                if 0 <= sampled_idx < len(self.sampled_2hz_p2lane):
                    traj_p2lane.append(self.sampled_2hz_p2lane[sampled_idx])
                else:
                    traj_p2lane.append(("", float("nan"), float("nan")))
                if 0 <= sampled_idx < len(self.sampled_2hz_p2lane_stable):
                    traj_p2lane_stable.append(
                        self.sampled_2hz_p2lane_stable[sampled_idx]
                    )
                else:
                    traj_p2lane_stable.append(("", float("nan"), float("nan"), ""))

        # 参考 parseRosBag-2O_py27_image2hz_rawkey_action_routing.py
        # 补充历史状态特征：4帧历史 + 当前帧
        state_history_window = 5
        state_his_locations = []

        for i in range(state_history_window):
            hist_ts = current_ts_loc - (state_history_window - 1 - i) * history_interval
            hist_idx, _ = self._find_nearest_loc_idx(
                hist_ts, max_delta=0.25, use_2hz_sampled=True
            )
            if hist_idx is not None:
                if self.sampled_2hz_locations:
                    state_his_locations.append(self.sampled_2hz_locations[hist_idx])
                else:
                    state_his_locations.append(self.valid_locations[hist_idx])
            else:
                state_his_locations.append(current_loc)

        while len(state_his_locations) < state_history_window:
            state_his_locations.append(current_loc)

        def extract_position(loc):
            if hasattr(loc, "position") and loc.position:
                return np.array([loc.position.x, loc.position.y, loc.position.z])
            return np.array([
                loc.longitude if hasattr(loc, "longitude") else 0.0,
                loc.latitude if hasattr(loc, "latitude") else 0.0,
                loc.altitude if hasattr(loc, "altitude") else 0.0,
            ])

        def normalize_angle(angle):
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            return angle

        frame_data = []
        dt = history_interval
        current_pos = extract_position(current_loc)
        loc_sequence = self.sampled_2hz_locations if self.sampled_2hz_locations else self.valid_locations

        next_pos = None
        next_loc = None
        if loc_idx + 1 < len(loc_sequence):
            next_loc = loc_sequence[loc_idx + 1]
            next_pos = extract_position(next_loc)

        for i, loc in enumerate(state_his_locations):
            pos = extract_position(loc)
            yaw = getattr(loc, "yaw", 0.0)

            if i + 1 < len(state_his_locations):
                next_pos_frame = extract_position(state_his_locations[i + 1])
                vel_global = (next_pos_frame - pos) / dt
            elif i == len(state_his_locations) - 1 and next_pos is not None:
                vel_global = (next_pos - pos) / dt
            else:
                vel_global = np.zeros(3)

            frame_data.append({
                "pos": pos,
                "yaw": yaw,
                "vel_global": vel_global,
            })

        if next_pos is not None and next_loc is not None:
            next_next_pos = None
            if loc_idx + 2 < len(loc_sequence):
                next_next_pos = extract_position(loc_sequence[loc_idx + 2])

            if next_next_pos is not None:
                vel_next = (next_next_pos - next_pos) / dt
            else:
                vel_next = (next_pos - current_pos) / dt

            frame_data.append({
                "pos": next_pos,
                "yaw": getattr(next_loc, "yaw", current_yaw),
                "vel_global": vel_next,
            })

        rot_mat_global_to_ego = ego_rotation_matrix.T
        his_speed = np.zeros((state_history_window, 3), dtype=np.float32)
        his_acceleration = np.zeros((state_history_window, 3), dtype=np.float32)
        his_angular_velocity = np.zeros((state_history_window, 3), dtype=np.float32)
        his_theta = np.zeros(state_history_window, dtype=np.float32)

        for i in range(state_history_window):
            frame = frame_data[i]
            vel_global = frame["vel_global"]
            vel_ego = np.dot(rot_mat_global_to_ego, vel_global)
            # his_speed[i] = np.linalg.norm(vel_ego[:2])
            his_speed[i, 0] = np.linalg.norm(vel_ego[0])
            his_speed[i, 1] = np.linalg.norm(vel_ego[1])
            # his_speed[i, 2] = np.linalg.norm(vel_ego[2])

            if i + 1 < len(frame_data):
                next_vel_global = frame_data[i + 1]["vel_global"]
                acc_global = (next_vel_global - vel_global) / dt
                acc_ego = np.dot(rot_mat_global_to_ego, acc_global)
                his_acceleration[i, 0] = acc_ego[0]
                his_acceleration[i, 1] = acc_ego[1]
                his_acceleration[i, 2] = acc_ego[2]

                next_frame = frame_data[i + 1]
                dyaw = normalize_angle(next_frame["yaw"] - frame["yaw"])
                his_angular_velocity[i, 2] = dyaw / dt

            if i == state_history_window - 1:
                his_theta[i] = 0.0
            else:
                his_theta[i] = normalize_angle(frame["yaw"] - current_yaw)

        # 获取3帧图像路径（当前帧和历史两帧）
        # 相机字段映射：back、front、left、left_back、right、right_back
        cam_paths_current = {}
        cam_paths_prev_1 = {}
        cam_paths_prev_2 = {}
        cam_paths_next = {}

        cam_keys = ["back", "front", "left", "left_back", "right", "right_back"]
        for key in cam_keys:
            # 当前帧（使用image_ts获取图像）
            path = self._get_camera_path(key, image_ts, max_delta=0.2)
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

            # 后1帧（0.5秒后，基于2Hz采样）
            next_ts = image_ts + 0.5
            path_next = self._get_camera_path(key, next_ts, max_delta=0.25)
            if path_next:
                cam_paths_next[key] = path_next

        # 获取 meta action 和 scenario type
        relative_ts = image_ts - (self.segment_time_origin or image_ts)
        meta_action = self.match_action_by_timestamp(relative_ts, action_dict)
        scenario_type = self.match_scenario_by_timestamp(relative_ts, scenario_dict)

        # 构建图像路径字段（存储为bytes）
        sample_data = {}

        # 当前帧图像路径（存储为bytes字符串）
        # 使用 his_cam_ 前缀 + 原始相机字段名称：back、front、left、left_back、right、right_back
        def safe_encode(path):
            """安全编码路径，如果为空则返回空bytes"""
            return path.encode('utf-8') if path else b''
            # return path if path else b''

        try:
            hdmap_bev_path = self._build_and_save_hdmap_bev(
                image_ts=image_ts,
                ego_utm_x=ego_utm_x,
                ego_utm_y=ego_utm_y,
                ego_utm_z=ego_utm_z,
                ego_yaw=current_yaw,
            )
        except Exception as exc:
            print(
                "[WARN] 生成 HDMap BEV 失败，跳过当前样本 {:.9f}: {}".format(
                    image_ts, exc
                )
            )
            return None

        # PlanVOCA-style 结构化 lane token：与 BEV 共用 sqlite，但保留几何/属性/route。
        # 失败必须抛，不能静默：之前 P0-1 (TX_LANE.pkid 不存在) 就是被吞掉的。
        hdmap_vector = None
        if self.hdmap_vector_generator is not None:
            try:
                hdmap_vector = self.hdmap_vector_generator.build(
                    ego_utm_x=ego_utm_x,
                    ego_utm_y=ego_utm_y,
                    ego_utm_z=ego_utm_z,
                    ego_yaw=current_yaw,
                    route_lane_keys=self.route_lane_keys,
                    route_link_keys=self.route_link_keys,
                )
            except Exception as exc:
                import traceback
                traceback.print_exc()
                print(
                    "[FATAL] 生成 HDMap vector token 失败 {:.9f}: {}".format(image_ts, exc)
                )
                self._hdmap_vector_failed = True
                raise

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

        sample_data['cam_front_next'] = safe_encode(cam_paths_next.get('front', ''))
        sample_data['hdmap_bev'] = hdmap_bev_path
        sample_data['token'] = make_token(self.vehicle_id, cam_paths_current.get('front', ''))
        sample_data['prev'] = make_token(self.vehicle_id, cam_paths_prev_1.get('front', ''))
        sample_data['next'] = make_token(self.vehicle_id, cam_paths_next.get('front', ''))
        sample_data['scene_token'] = build_scene_token(self.vehicle_id, self.bag_dir)

        # 历史运动状态
        sample_data['his_speed'] = his_speed
        sample_data['his_acceleration'] = his_acceleration
        sample_data['his_angular_velocity'] = his_angular_velocity
        sample_data['his_theta'] = his_theta
        sample_data['his_trajectory'] = his_trajectory_data
        sample_data['fur_trajectory'] = future_trajectory_data
        sample_data['traj_p2lane'] = traj_p2lane
        sample_data['traj_p2lane_stable'] = traj_p2lane_stable
        sample_data['ego_utm_position'] = np.array(
            [ego_utm_x, ego_utm_y, ego_utm_z], dtype=np.float32
        )
        sample_data['ego_yaw'] = float(current_yaw)

        navigation_command = self.high_order_intent[loc_idx]
        scenario_type = override_scenario_type_from_navigation_command(
            scenario_type, navigation_command
        )

        sample_data['meta_action'] = meta_action
        sample_data['scenario_type'] = scenario_type
        sample_data['sample_timestamp'] = float(image_ts)
        sample_data['relative_timestamp'] = float(relative_ts)
        sample_data['source_bag_dir'] = self.bag_dir
        sample_data['source_image_dir'] = self.image_dir

        # 导航信息
        sample_data['navigation_command'] = navigation_command
        sample_data['navigation_command_descreption'] = current_navigation_str

        # PlanVOCA-style 结构化 HDMap 字段，直接写入 pkl 顶层。下游
        # src/data/dataset.py 在 use_hdmap_vector=True 时读取。
        if hdmap_vector is not None:
            sample_data['hdmap_vector_point_position'] = hdmap_vector["point_position"]
            sample_data['hdmap_vector_point_vector'] = hdmap_vector["point_vector"]
            sample_data['hdmap_vector_point_orientation'] = hdmap_vector["point_orientation"]
            sample_data['hdmap_vector_polygon_center'] = hdmap_vector["polygon_center"]
            sample_data['hdmap_vector_polygon_type'] = hdmap_vector["polygon_type"]
            sample_data['hdmap_vector_polygon_on_route'] = hdmap_vector["polygon_on_route"]
            sample_data['hdmap_vector_polygon_tl_status'] = hdmap_vector["polygon_tl_status"]
            sample_data['hdmap_vector_polygon_has_speed_limit'] = hdmap_vector["polygon_has_speed_limit"]
            sample_data['hdmap_vector_polygon_speed_limit'] = hdmap_vector["polygon_speed_limit"]
            sample_data['hdmap_vector_point_valid_mask'] = hdmap_vector["point_valid_mask"]
            sample_data['hdmap_vector_valid_mask'] = hdmap_vector["valid_mask"]
            sample_data['hdmap_vector_num_lanes'] = int(hdmap_vector["num_lanes"])

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
        if self.hdmap_generator is not None:
            self.hdmap_generator.close()
            self.hdmap_generator = None
        if self.hdmap_vector_generator is not None:
            self.hdmap_vector_generator.close()
            self.hdmap_vector_generator = None
        if self.lane_frenet_matcher is not None:
            self.lane_frenet_matcher.close()
            self.lane_frenet_matcher = None

    def run(self):
        # 主逻辑
        self.load_existing_cameras()
        # 图片按时间戳拼帧
        self.align_camera_cache_by_front()
        # 解析定位topic
        self.parse_localization()
        # 基于图像时间轴进行2Hz采样，并回查原始定位
        self.sample_localization_2hz(sample_rate_hz=2.0)
        # 基于 sampled localization 生成逐点 p2lane
        self.build_sampled_p2lane()
        # 解析高阶意图导航信息
        self.parse_routing()
        # 解析轨迹topic
        # self.parse_trajectory()
        # 生成pkl样本（基于2Hz采样数据）
        sample_count = self.generate_pkl_samples(sample_rate_hz=2.0)
        # 关闭 bag
        self.close()
        return sample_count

def discover_segment_dirs(rosbags_root):
    bag_dirs = []
    for current_root, _, files in os.walk(rosbags_root):
        has_master = any(name.endswith(".bag") and "-rosmaster-" in name for name in files)
        has_slave = any(name.endswith(".bag") and "-rosslave-" in name for name in files)
        if has_master and has_slave:
            bag_dirs.append(current_root)
    return sorted(bag_dirs)


def build_route_key(rosbags_root, bag_dir):
    rel_path = os.path.relpath(bag_dir, rosbags_root)
    parent_rel_path = os.path.dirname(rel_path)
    if parent_rel_path in ("", os.curdir):
        return rel_path
    return parent_rel_path


def find_master_bag_path(bag_dir):
    if not os.path.isdir(bag_dir):
        return None
    for filename in sorted(os.listdir(bag_dir)):
        if filename.endswith(".bag") and "-rosmaster-" in filename:
            return os.path.join(bag_dir, filename)
    return None


def load_route_result_from_bag(bag_path):
    if not bag_path:
        return None

    bag = None
    try:
        bag = rosbag.Bag(bag_path, "r")
        # for topic, msg, time_stamp in bag.read_messages(topics=["/hadmap_engine/route_result_msg"]):
        for topic, msg, time_stamp in bag.read_messages(topics=["/planning/route_result_debug"]):
            try:
                route_roads = RoutingResult()
                route_roads.ParseFromString(msg.data)
                return route_roads
            except Exception as exc:
                print("[WARN] 解析路由消息失败 {}: {}".format(bag_path, exc))
                return None
    except Exception as exc:
        print("[WARN] 打开路由 bag 失败 {}: {}".format(bag_path, exc))
        return None
    finally:
        if bag is not None:
            bag.close()

    return None


def build_completion_marker_path(output_dir):
    return os.path.join(output_dir, COMPLETION_MARKER_FILENAME)


def find_existing_pkl(output_dir):
    if not os.path.isdir(output_dir):
        return None

    for filename in sorted(os.listdir(output_dir)):
        if filename.endswith(".pkl"):
            return os.path.join(output_dir, filename)
    return None


def should_skip_existing_output(output_dir):
    marker_path = build_completion_marker_path(output_dir)
    if os.path.isfile(marker_path):
        return True, "检测到完成标记文件 {}".format(marker_path)

    existing_pkl = find_existing_pkl(output_dir)
    if existing_pkl:
        return True, "检测到已存在的 PKL 文件 {}".format(existing_pkl)

    return False, ""


def write_completion_marker(output_dir, rel_path, sample_count):
    os.makedirs(output_dir, exist_ok=True)
    marker_path = build_completion_marker_path(output_dir)
    marker_payload = {
        "status": "completed",
        "segment_rel_path": rel_path,
        "sample_count": int(sample_count),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(marker_path, "w", encoding="utf-8") as f:
        json.dump(marker_payload, f, indent=2, ensure_ascii=False)


def build_label_paths(labels_root, rel_path):
    rel_parts = rel_path.split(os.sep)
    if len(rel_parts) != 3:
        return None, None
    scene_name, route_name, segment_name = rel_parts
    label_stem = "{}_{}_{}".format(scene_name, route_name, segment_name)
    txt_path = os.path.join(labels_root, label_stem + ".txt")
    json_path = os.path.join(labels_root, label_stem + ".json")
    if not os.path.isfile(txt_path):
        txt_path = None
    if not os.path.isfile(json_path):
        json_path = None
    return txt_path, json_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="解析 MOGO rosbags 与拆分图像，生成带动作信息的 PKL 样本"
    )
    parser.add_argument("--rosbags-root", required=True, help="原始 bag 根目录")
    parser.add_argument("--images-root", required=True, help="拆分图像根目录")
    parser.add_argument("--output-root", required=True, help="PKL 输出根目录")
    parser.add_argument("--labels-root", default=None, help="标签根目录")
    parser.add_argument(
        "--write-json",
        nargs="?",
        const="__DEFAULT__",
        default=None,
        help=(
            "输出 JSON 调试文件；不带值时默认写到每个切片输出目录下的 "
            "`poutine_base_json`，带值时写到指定 JSON 根目录"
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果切片输出目录下已存在 PKL 或完成标记，则跳过该切片且不覆盖已有结果",
    )
    parser.add_argument("--limit", type=int, default=0, help="仅处理前 N 个切片，0 表示全量")
    parser.add_argument(
        "--hdmap-sql",
        default=os.getenv("HDMAP_SQL_PATH"),
        help="HDMap Spatialite 文件路径；提供后会生成 hdmap_bev 的 .npz 并写回 pkl",
    )
    parser.add_argument(
        "--hdmap-point-cloud-range",
        type=float,
        nargs=6,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        default=list(DEFAULT_HDMAP_POINT_CLOUD_RANGE),
        help="HDMap BEV 栅格范围，默认参考 multimodal_bevfusion",
    )
    parser.add_argument(
        "--hdmap-voxel-size",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=list(DEFAULT_HDMAP_VOXEL_SIZE),
        help="HDMap BEV 体素大小，默认参考 multimodal_bevfusion",
    )
    parser.add_argument(
        "--hdmap-bev-height",
        type=int,
        default=DEFAULT_HDMAP_BEV_HEIGHT,
        help="HDMap BEV 高度，默认 1792",
    )
    parser.add_argument(
        "--hdmap-bev-width",
        type=int,
        default=DEFAULT_HDMAP_BEV_WIDTH,
        help="HDMap BEV 宽度，默认 960",
    )
    parser.add_argument(
        "--hdmap-utm-zone",
        type=int,
        default=50,
        help="定位 UTM 坐标对应的 zone，默认 50",
    )
    parser.add_argument(
        "--hdmap-utm-south",
        action="store_true",
        help="如果定位 UTM 坐标位于南半球则开启",
    )
    parser.add_argument(
        "--hdmap-vector-radius",
        type=float,
        default=None,
        help="HDMapVectorGenerator 查询半径（米）。默认 None = 用 generator 内部默认 120m (PlanVOCA)",
    )
    parser.add_argument(
        "--per-bag-route",
        action="store_true",
        help="每个 bag 独立路由，不在切片之间复用 routing 缓存（worldsim 等每包独立路线场景使用）",
    )
    return parser.parse_args()


if __name__ == '__main__':
    print("[INFO] 脚本启动")
    args = parse_args()

    if not os.path.isdir(args.rosbags_root):
        print("[ERROR] rosbags 根目录不存在：{}".format(args.rosbags_root))
        sys.exit(1)
    if not os.path.isdir(args.images_root):
        print("[ERROR] images 根目录不存在：{}".format(args.images_root))
        sys.exit(1)
    if args.labels_root and not os.path.isdir(args.labels_root):
        print("[ERROR] labels 根目录不存在：{}".format(args.labels_root))
        sys.exit(1)
    if args.write_json not in (None, "__DEFAULT__"):
        os.makedirs(args.write_json, exist_ok=True)
    if args.hdmap_sql and not os.path.isfile(args.hdmap_sql):
        print("[ERROR] HDMap sqlite 文件不存在：{}".format(args.hdmap_sql))
        sys.exit(1)

    bag_dirs = discover_segment_dirs(args.rosbags_root)
    if args.limit > 0:
        bag_dirs = bag_dirs[:args.limit]

    if not bag_dirs:
        print("[ERROR] 未找到包含 master/slave bag 的切片目录")
        sys.exit(1)

    print("[INFO] 找到 {} 个待处理切片目录".format(len(bag_dirs)))

    success_count = 0
    failed_count = 0
    skipped_count = 0
    route_result_cache = {}
    for idx, bag_dir in enumerate(bag_dirs, 1):
        rel_path = os.path.relpath(bag_dir, args.rosbags_root)
        image_dir = os.path.join(args.images_root, rel_path)
        output_dir = os.path.join(args.output_root, rel_path)
        route_key = build_route_key(args.rosbags_root, bag_dir)
        json_output_dir = None
        if args.write_json == "__DEFAULT__":
            json_output_dir = None
        elif args.write_json:
            json_output_dir = os.path.join(args.write_json, rel_path)
        label_txt_path = None
        label_json_path = None
        if args.labels_root:
            label_txt_path, label_json_path = build_label_paths(args.labels_root, rel_path)

        print("\n" + "=" * 80)
        print("[INFO] 处理第 {}/{} 个切片: {}".format(idx, len(bag_dirs), rel_path))
        print("[INFO] 图像目录: {}".format(image_dir))
        print("[INFO] 输出目录: {}".format(output_dir))
        print("=" * 80)

        if not os.path.isdir(image_dir):
            print("[WARN] 图像目录不存在，跳过: {}".format(image_dir))
            failed_count += 1
            continue

        if args.skip_existing:
            should_skip, skip_reason = should_skip_existing_output(output_dir)
            if should_skip:
                if not args.per_bag_route and route_key not in route_result_cache:
                    route_result_cache[route_key] = load_route_result_from_bag(
                        find_master_bag_path(bag_dir)
                    )
                    if route_result_cache[route_key] is not None:
                        print("[INFO] 已从跳过切片缓存路线 {} 的 routing 信息".format(route_key))
                skipped_count += 1
                print("[INFO] 跳过已存在输出的切片: {}".format(rel_path))
                print("[INFO] 跳过原因: {}".format(skip_reason))
                continue

        try:
            if args.per_bag_route:
                cached_route_result = None
            else:
                cached_route_result = route_result_cache.get(route_key)
                if route_key in route_result_cache and cached_route_result is not None:
                    print("[INFO] 当前切片复用路线 {} 的 routing 信息".format(route_key))
            handler = BagDataHandler(
                bag_dir=bag_dir,
                image_dir=image_dir,
                output_dir=output_dir,
                label_txt_path=label_txt_path,
                label_json_path=label_json_path,
                write_json=args.write_json is not None,
                json_output_dir=json_output_dir,
                hdmap_sql_path=args.hdmap_sql,
                hdmap_point_cloud_range=args.hdmap_point_cloud_range,
                hdmap_voxel_size=args.hdmap_voxel_size,
                hdmap_bev_height=args.hdmap_bev_height,
                hdmap_bev_width=args.hdmap_bev_width,
                hdmap_utm_zone=args.hdmap_utm_zone,
                hdmap_utm_south=args.hdmap_utm_south,
                hdmap_vector_radius=args.hdmap_vector_radius,
                route_result=cached_route_result,
            )
            sample_count = handler.run()
            if not args.per_bag_route and route_key not in route_result_cache:
                route_result_cache[route_key] = handler.route_result
                if handler.route_result is not None:
                    print("[INFO] 已缓存路线 {} 的 routing 信息".format(route_key))
            write_completion_marker(output_dir, rel_path, sample_count)
            success_count += 1
            print("[INFO] 完成处理: {}".format(rel_path))
        except Exception as e:
            failed_count += 1
            print("[ERROR] 处理失败 {}: {}".format(rel_path, e))
            import traceback
            traceback.print_exc()
            print("[WARN] 继续处理下一个切片...")
            continue

    print("\n" + "=" * 80)
    print(
        "[INFO] 处理完成：成功 {}，跳过 {}，失败 {}".format(
            success_count, skipped_count, failed_count
        )
    )
    print("=" * 80)
