#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS bag -> NAVSIM-compatible pkl exporter for ReCogDrive/RL.

This script is intentionally independent from the existing Poutine exporter, but
it follows the same ROS topic conventions wherever possible. The target output is
one NAVSIM-style log pkl:

    List[frame_dict]

The generated frame_dict is expected to be consumable by:

    navsim.common.dataloader.SceneLoader
    navsim.common.dataclasses.Scene / AgentInput
    navsim.planning.metric_caching.metric_cache_processor.MetricCacheProcessor

Current scope of this scaffold:
    1. Read one bag file, one log directory, or a raw root containing multiple log directories.
    2. Reuse the reference script's main topics:
       - /localization/global
       - /perception/fusion/obstacles
       - /hadmap_engine/map_msg
       - /hadmap_engine/route_result_msg, if present
       - /sensor/camera/sensing/*/resize/nvjpeg
    3. Use the front camera timeline as the default sampling timeline, so 10 Hz
       front images naturally produce 0.1 s frame_dicts.
    4. Extract compressed camera images to dataset/sensor_blobs/{split} when requested.
    5. Reserve explicit conversion points for obstacles, local map, route, and
       calibration details.

The semantic conversion from internal perception/map protos to nuPlan/NAVSIM map
and annotation structures is intentionally isolated in small methods below.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Type

import numpy as np

try:
    import rosbag  # type: ignore
except ImportError:
    rosbag = None
    from rosbags.rosbag1 import Reader as RosbagReader  # type: ignore
    from rosbags.typesys import Stores, get_types_from_msg, get_typestore  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
PROTO_COMMON_DIR = SCRIPT_DIR / "proto" / "common"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROTO_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_COMMON_DIR))

from google.protobuf.message import DecodeError  # noqa: E402
from proto.common.chassis2025_pb2 import ChassisStates  # noqa: E402
from proto.common.localization_pb2 import Localization  # noqa: E402
from proto.common.object_pb2 import TrackedObjects  # noqa: E402
from proto.common.routing_pb2 import RoutingResult  # noqa: E402


if rosbag is None:
    _BINARY_DATA_MSGDEF = (
        "Header header\nstring name\nuint32 size\nuint8[] data\n"
        "\n================================================================================\n"
        "MSG: std_msgs/Header\nuint32 seq\ntime stamp\nstring frame_id\n"
    )

    def _make_typestore():
        typestore = get_typestore(Stores.ROS1_NOETIC)
        typestore.register(get_types_from_msg(_BINARY_DATA_MSGDEF, "autopilot_msgs/msg/BinaryData"))
        return typestore

    _TYPESTORE = _make_typestore()

    def _decode_rosbags_message(connection, rawdata):
        if "BinaryData" in connection.msgtype:
            binary_msg = _TYPESTORE.deserialize_ros1(rawdata, "autopilot_msgs/msg/BinaryData")
            return SimpleNamespace(data=bytes(binary_msg.data), header=getattr(binary_msg, "header", None))
        return _TYPESTORE.deserialize_ros1(rawdata, connection.msgtype)

    class _CompatBag(object):
        def __init__(self, bag_path: str, mode: str = "r"):
            if mode != "r":
                raise ValueError("Only read mode is supported")
            self._reader = RosbagReader(bag_path)
            self._reader.open()

        def read_messages(self, topics=None):
            if topics is None:
                connections = list(self._reader.connections)
            else:
                topic_set = {topics} if isinstance(topics, str) else set(topics)
                connections = [conn for conn in self._reader.connections if conn.topic in topic_set]

            for connection, timestamp_ns, rawdata in self._reader.messages(connections=connections):
                yield connection.topic, _decode_rosbags_message(connection, rawdata), timestamp_ns

        def get_type_and_topic_info(self):
            topic_info = {connection.topic: connection for connection in self._reader.connections}
            return (None, topic_info)

        def close(self):
            self._reader.close()

    class _RosbagCompat(object):
        Bag = _CompatBag

    rosbag = _RosbagCompat()


DEFAULT_LOCALIZATION_TOPIC = "/localization/global"
DEFAULT_CHASSIS_TOPIC = "/chassis/chassis2025"
DEFAULT_OBSTACLE_TOPIC = "/perception/fusion/obstacles"
DEFAULT_LOCAL_MAP_TOPIC = "/hadmap_engine/map_msg"
DEFAULT_ROUTE_TOPIC = "/hadmap_engine/route_result_msg"
DEFAULT_FRONT_CAMERA_TOPIC = "/sensor/camera/sensing/image_raw_120/resize/nvjpeg"
DEFAULT_VEHICLE_ID = "unknown_vehicle"

# The source bags provide camera/image and local-map messages at 10 Hz. We keep
# this raw cadence first; later NAVSIM scene sampling can decide whether to use
# every frame or stride to 0.5 s.
DEFAULT_SAMPLE_INTERVAL = 0.1
DEFAULT_HISTORY_FRAMES = 16
DEFAULT_FUTURE_FRAMES = 40
DEFAULT_DRIVING_COMMAND_DISTANCE = 30.0
DEFAULT_MAX_LATERAL_MATCH_DISTANCE = 2.0
DEFAULT_MAX_CAMERA_DELTA = 0.05
DEFAULT_MAX_OBSTACLE_DELTA = 0.10
DEFAULT_MAX_LOCAL_MAP_DELTA = 0.10

NAVSIM_CAMERA_NAMES = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")

# Keep the same camera naming logic as the reference script, then map available
# roles into NAVSIM camera slots. The remaining NAVSIM slots stay as empty camera
# entries so Scene/AgentInput loading remains structurally stable.
CAMERA_ROLE_TO_NAVSIM_KEY = {
    "front": "CAM_F0",
    "left": "CAM_L0",
    "left_back": "CAM_L1",
    "right": "CAM_R0",
    "right_back": "CAM_R1",
    "back": "CAM_B0",
}

BEVFUSION_OBJECT_TYPE_MAP = {
    0: "unknown",
    3: "pedestrian",
    4: "cyclist",
    5: "motor",
    6: "rider",
    7: "vehicle",
    8: "vehicle",
    9: "vehicle",
    10: "vehicle",
    20: "traffic_sign",
    30: "traffic_light",
    91: "unknown_small",
    92: "unknown_big",
    93: "static_obstacle",
    94: "dynamic_obstacle",
}

NAVSIM_OBJECT_TYPE_MAP = {
    "vehicle": "vehicle",
    "pedestrian": "pedestrian",
    "cyclist": "bicycle",
    "motor": "bicycle",
    "rider": "bicycle",
    "unknown_big": "generic_object",
    "static_obstacle": "generic_object",
    "dynamic_obstacle": "generic_object",
}

COMMAND_LEFT = 0
COMMAND_STRAIGHT = 1
COMMAND_RIGHT = 2
COMMAND_UNKNOWN = 3
COMMAND_UTURN = 4

# NAVSIM stores driving_command as a 4D vector. Keep existing one-hot codes
# unchanged and reserve [1, 1, 0, 0] for U-turn without changing the schema.
# Current automatic route-link mapping is intentionally conservative:
# Driving/JuncStraight -> straight, JuncTurnLeft -> left,
# JuncTurnRight -> right, Turn -> U-turn. Other LaneTypes stay pending.
# TODO(navsim): downstream prompt/status conversion still assumes one-hot
# left/straight/right/unknown; adapt it before consuming U-turn samples.
COMMAND_CODE_MAP = {
    COMMAND_LEFT: [1, 0, 0, 0],
    COMMAND_STRAIGHT: [0, 1, 0, 0],
    COMMAND_RIGHT: [0, 0, 1, 0],
    COMMAND_UNKNOWN: [0, 0, 0, 1],
    COMMAND_UTURN: [1, 1, 0, 0],
}

LANE_TYPE_DRIVING = 1
LANE_TYPE_TURN_LEFT_WAITING = 21
LANE_TYPE_UTURN_WAITING = 23
LANE_TYPE_TURN_RIGHT_WAITING = 24
LANE_TYPE_JUNC_TURN_LEFT = 25
LANE_TYPE_JUNC_STRAIGHT = 26
LANE_TYPE_JUNC_TURN_RIGHT = 27
LANE_TYPE_TURN = 28
LANE_TYPE_LEFT_OR_UTURN_WAITING = 34

LEFT_LINK_TYPES = {LANE_TYPE_JUNC_TURN_LEFT}
STRAIGHT_LINK_TYPES = {LANE_TYPE_DRIVING, LANE_TYPE_JUNC_STRAIGHT}
RIGHT_LINK_TYPES = {LANE_TYPE_JUNC_TURN_RIGHT}
UTURN_LINK_TYPES = {LANE_TYPE_TURN}


@dataclass
class TopicRecord:
    timestamp: float
    message: Any
    raw_msg: Any = None
    topic: str = ""


@dataclass
class ConverterConfig:
    input_path: Path
    output_path: Path
    sensor_blobs_root: Optional[Path] = None
    route_source_path: Optional[Path] = None
    log_name: Optional[str] = None
    vehicle_id: Optional[str] = None
    map_name: str = "las_vegas"
    localization_topic: str = DEFAULT_LOCALIZATION_TOPIC
    chassis_topic: str = DEFAULT_CHASSIS_TOPIC
    obstacle_topic: str = DEFAULT_OBSTACLE_TOPIC
    local_map_topic: str = DEFAULT_LOCAL_MAP_TOPIC
    route_topic: str = DEFAULT_ROUTE_TOPIC
    front_camera_topic: str = DEFAULT_FRONT_CAMERA_TOPIC
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL
    num_history_frames: int = DEFAULT_HISTORY_FRAMES
    num_future_frames: int = DEFAULT_FUTURE_FRAMES
    max_localization_delta: float = 0.05
    max_chassis_delta: float = 0.05
    max_camera_delta: float = DEFAULT_MAX_CAMERA_DELTA
    max_obstacle_delta: float = DEFAULT_MAX_OBSTACLE_DELTA
    max_local_map_delta: float = DEFAULT_MAX_LOCAL_MAP_DELTA
    driving_command_distance: float = DEFAULT_DRIVING_COMMAND_DISTANCE
    max_lateral_match_distance: float = DEFAULT_MAX_LATERAL_MATCH_DISTANCE
    route_result: Optional[RoutingResult] = None
    route_roadblock_ids: List[str] = field(default_factory=list)
    extract_images: bool = True
    write_debug_json: bool = False

    @property
    def bag_paths(self) -> List[Path]:
        if self.input_path.is_file():
            return [self.input_path]
        if self.input_path.is_dir():
            return sorted(self.input_path.glob("*.bag"))
        return []

    @property
    def resolved_vehicle_id(self) -> str:
        return infer_vehicle_id_from_bag_names(self.bag_paths) or self.vehicle_id or DEFAULT_VEHICLE_ID

    @property
    def base_log_name(self) -> str:
        return self.log_name or self.input_path.stem

    @property
    def resolved_log_name(self) -> str:
        return "{}_{}".format(self.resolved_vehicle_id, self.base_log_name)

    @property
    def num_frames(self) -> int:
        return self.num_history_frames + self.num_future_frames

    @property
    def current_frame_offset(self) -> int:
        return self.num_history_frames - 1


@dataclass
class CameraCache:
    role_to_topic: Dict[str, str] = field(default_factory=dict)
    role_to_records: Dict[str, List[TopicRecord]] = field(default_factory=dict)

    def role_times(self, role: str) -> List[float]:
        return [record.timestamp for record in self.role_to_records.get(role, [])]


class MultiBagReader:
    """Small adapter that reads topics from one or more ROS1 bag files."""

    def __init__(self, bag_paths: Sequence[Path]):
        self.bag_paths = list(bag_paths)
        self.bags: List[Any] = []

    def open(self) -> None:
        self.bags = [rosbag.Bag(str(path), "r") for path in self.bag_paths]

    def close(self) -> None:
        for bag in self.bags:
            bag.close()
        self.bags = []

    def topic_names(self) -> List[str]:
        topics = set()
        for bag in self.bags:
            topics.update(bag.get_type_and_topic_info()[1].keys())
        return sorted(topics)

    def read_messages(self, topics: Optional[Iterable[str]] = None):
        topic_list = None if topics is None else list(topics)
        for bag in self.bags:
            available_topics = set(bag.get_type_and_topic_info()[1].keys())
            if topic_list is None:
                selected_topics = None
            else:
                selected_topics = [topic for topic in topic_list if topic in available_topics]
                if not selected_topics:
                    continue
            for topic, msg, timestamp in bag.read_messages(topics=selected_topics):
                yield topic, msg, timestamp


def ros_timestamp_to_sec(timestamp: Any) -> float:
    if hasattr(timestamp, "to_sec"):
        return float(timestamp.to_sec())
    return float(timestamp) * 1e-9


def choose_topic_timestamp(header_timestamp: float, bag_timestamp: Any) -> float:
    """Choose an alignment timestamp with protection for coarse integer headers.

    The reference exporter uses message headers. Some current localization bags,
    however, carry only integer-second header timestamps while the bag timestamp
    keeps the sub-second cadence. In that case, using the bag timestamp preserves
    the 50/100 Hz localization sequence and keeps it alignable with camera time.
    """
    bag_time = ros_timestamp_to_sec(bag_timestamp)
    if header_timestamp <= 0.0:
        return bag_time

    header_is_integer_second = abs(header_timestamp - round(header_timestamp)) < 1e-9
    bag_has_subsecond = abs(bag_time - round(bag_time)) > 1e-6
    same_epoch = abs(header_timestamp - bag_time) < 1.0
    if header_is_integer_second and bag_has_subsecond and same_epoch:
        return bag_time

    return header_timestamp


def message_timestamp_to_sec(msg: Any, bag_timestamp: Any) -> float:
    """Prefer message header time, with coarse-header fallback to bag time."""
    header_time = calculate_header_timestamp(getattr(msg, "header", None))
    return choose_topic_timestamp(header_time, bag_timestamp)


def calculate_header_timestamp(header: Any) -> float:
    """Return timestamp in seconds from a protobuf or ROS header-like object."""
    if header is None:
        return 0.0

    if hasattr(header, "timestamp_sec"):
        return float(header.timestamp_sec)
    if hasattr(header, "time_stamp"):
        return float(header.time_stamp)
    if hasattr(header, "timestamp"):
        return float(header.timestamp)

    stamp = getattr(header, "stamp", None)
    if stamp is not None:
        secs = float(getattr(stamp, "secs", getattr(stamp, "sec", 0.0)))
        nsecs = float(getattr(stamp, "nsecs", getattr(stamp, "nanosec", 0.0)))
        return secs + nsecs * 1e-9

    return 0.0


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_to_quaternion(yaw: float) -> List[float]:
    """Return NAVSIM-compatible quaternion order used by pyquaternion: [w, x, y, z]."""
    half_yaw = yaw * 0.5
    return [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]


def infer_vehicle_id_from_bag_names(bag_paths: Sequence[Path]) -> Optional[str]:
    for bag_path in bag_paths:
        bag_name = bag_path.name
        if "-rosmaster-" in bag_name:
            suffix = bag_name.split("-rosmaster-", 1)[1]
            vehicle_id = suffix.rsplit(".bag", 1)[0].strip()
            if vehicle_id:
                return vehicle_id
    return None


def make_token(vehicle_id: str, base_log_name: str, timestamp: float) -> str:
    key = "{}:{}:{:.6f}".format(vehicle_id, base_log_name, timestamp).encode("utf-8")
    return hashlib.md5(key).hexdigest()


def nearest_index(times: Sequence[float], target_time: float, max_delta: Optional[float] = None) -> Optional[int]:
    if not times:
        return None
    idx = bisect.bisect_left(times, target_time)
    candidates: List[int] = []
    if idx == 0:
        candidates.append(0)
    elif idx == len(times):
        candidates.append(len(times) - 1)
    else:
        candidates.extend([idx - 1, idx])

    best_idx = min(candidates, key=lambda candidate: abs(times[candidate] - target_time))
    if max_delta is not None and abs(times[best_idx] - target_time) > max_delta:
        return None
    return best_idx


def parse_binary_proto(msg: Any, proto_cls: Type[Any]) -> Any:
    proto = proto_cls()
    payload = getattr(msg, "data", None)
    if payload is None:
        raise ValueError("ROS message does not contain a .data payload for protobuf parsing")
    proto.ParseFromString(bytes(payload))
    return proto


def get_float_attr(obj: Any, name: str) -> Optional[float]:
    if obj is None or not hasattr(obj, name):
        return None
    return float(getattr(obj, name))


def first_valid_float(*values: Optional[float], default: float = 0.0) -> float:
    for value in values:
        if value is not None:
            return value
    return default


def make_driving_command(command_label: int) -> np.ndarray:
    if command_label not in COMMAND_CODE_MAP:
        command_label = COMMAND_UNKNOWN
    return np.asarray(COMMAND_CODE_MAP[command_label], dtype=np.int64)


def points_to_xy(points: Sequence[Any]) -> List[Tuple[float, float]]:
    return [(float(point.x), float(point.y)) for point in points]


def polyline_length(points: Sequence[Any]) -> float:
    xy_points = points_to_xy(points)
    return polyline_xy_length(xy_points)


def polyline_xy_length(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    return float(sum(
        math.hypot(points[idx][0] - points[idx - 1][0], points[idx][1] - points[idx - 1][1])
        for idx in range(1, len(points))
    ))


def project_point_to_polyline(
    point_xy: Tuple[float, float],
    polyline_xy: Sequence[Tuple[float, float]],
) -> Tuple[float, float, float]:
    if len(polyline_xy) < 2:
        return math.inf, 0.0, 0.0

    px, py = point_xy
    best_distance = math.inf
    best_progress = 0.0
    total_length = 0.0

    for idx in range(1, len(polyline_xy)):
        ax, ay = polyline_xy[idx - 1]
        bx, by = polyline_xy[idx]
        vx = bx - ax
        vy = by - ay
        segment_length_sq = vx * vx + vy * vy
        if segment_length_sq <= 1e-12:
            continue

        segment_length = math.sqrt(segment_length_sq)
        ratio = ((px - ax) * vx + (py - ay) * vy) / segment_length_sq
        ratio = min(1.0, max(0.0, ratio))
        proj_x = ax + ratio * vx
        proj_y = ay + ratio * vy
        distance = math.hypot(px - proj_x, py - proj_y)
        progress = total_length + ratio * segment_length
        if distance < best_distance:
            best_distance = distance
            best_progress = progress
        total_length += segment_length

    return best_distance, best_progress, total_length


def link_type_to_command_index(link_type: int) -> Optional[int]:
    if link_type in UTURN_LINK_TYPES:
        return COMMAND_UTURN
    if link_type in LEFT_LINK_TYPES:
        return COMMAND_LEFT
    if link_type in STRAIGHT_LINK_TYPES:
        return COMMAND_STRAIGHT
    if link_type in RIGHT_LINK_TYPES:
        return COMMAND_RIGHT
    return None


def route_bag_paths_from_source(route_source_path: Path) -> List[Path]:
    if route_source_path.is_file():
        return [route_source_path]
    if route_source_path.is_dir():
        return sorted(route_source_path.glob("*.bag"))
    return []


def extract_route_road_entries(route_result: RoutingResult) -> List[Dict[str, Any]]:
    route_entries: List[Dict[str, Any]] = []
    previous_path_last_id: Optional[str] = None

    for path_idx, path in enumerate(route_result.paths):
        path_entries: List[Dict[str, Any]] = []
        for roadlink_idx, route_roadlink in enumerate(path.route_roadlinks):
            if not route_roadlink.HasField("route_road"):
                continue
            route_road = route_roadlink.route_road
            if route_road.HasField("pkid") and route_road.pkid:
                road_pkid = int(route_road.pkid)
                path_entries.append({
                    "path_idx": path_idx,
                    "roadlink_idx": roadlink_idx,
                    "road_pkid": road_pkid,
                    "roadblock_id": "roadblock_{}".format(road_pkid),
                })

        if not path_entries:
            continue

        # Keep route order and loop duplicates, but remove the stitching duplicate
        # between adjacent paths when prev_path[-1] == next_path[0].
        entries_to_append = path_entries
        if previous_path_last_id is not None and path_entries[0]["roadblock_id"] == previous_path_last_id:
            entries_to_append = path_entries[1:]

        for entry in entries_to_append:
            entry = dict(entry)
            entry["global_idx"] = len(route_entries)
            route_entries.append(entry)

        previous_path_last_id = path_entries[-1]["roadblock_id"]

    return route_entries


def extract_route_roadblock_ids(route_result: RoutingResult) -> List[str]:
    return [entry["roadblock_id"] for entry in extract_route_road_entries(route_result)]


def load_valid_route_result(route_source_path: Path, route_topic: str) -> Tuple[RoutingResult, List[str]]:
    bag_paths = route_bag_paths_from_source(route_source_path)
    if not bag_paths:
        raise RuntimeError("no routing bag files found under {}".format(route_source_path))

    reader = MultiBagReader(bag_paths)
    reader.open()
    try:
        route_message_count = 0
        for _, msg, _ in reader.read_messages(topics=[route_topic]):
            route_message_count += 1
            try:
                route_result = parse_binary_proto(msg, RoutingResult)
            except (DecodeError, ValueError):
                continue
            roadblock_ids = extract_route_roadblock_ids(route_result)
            if roadblock_ids:
                return route_result, roadblock_ids
    finally:
        reader.close()

    raise RuntimeError(
        "no valid RoutingResult with route_roadlinks.route_road.pkid found in {} on topic {}".format(
            route_source_path, route_topic
        )
    )


def discover_route_source_path(raw_data_root: Path) -> Optional[Path]:
    routing_bags = sorted(raw_data_root.glob("*routing_results*.bag"))
    if routing_bags:
        return routing_bags[0]
    return None


def infer_camera_role(topic: str) -> Optional[str]:
    """Infer camera role with the same priority used by the reference exporter."""
    if "/sensor/camera" not in topic:
        return None
    if "image_raw_120" in topic or "front" in topic:
        return "front"
    if "image_raw_60" in topic:
        return "front_narrow"
    if "image_raw_left_back" in topic:
        return "left_back"
    if "image_raw_right_back" in topic:
        return "right_back"
    if "image_raw_back" in topic or "back" in topic:
        return "back"
    if "left" in topic:
        return "left"
    if "right" in topic:
        return "right"
    return None


class RosbagToRecogDrivePklConverter:
    def __init__(self, config: ConverterConfig):
        self.config = config
        self.reader = MultiBagReader(config.bag_paths)
        self.localization_records: List[TopicRecord] = []
        self.chassis_records: List[TopicRecord] = []
        self.obstacle_records: List[TopicRecord] = []
        self.local_map_records: List[TopicRecord] = []
        self.route_records: List[TopicRecord] = []
        self.route_result: Optional[RoutingResult] = None
        self.route_roadblock_ids: List[str] = []
        self.route_road_entries: List[Dict[str, Any]] = []
        self.route_entry_by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.loop_duplicate_road_pkids: set = set()
        self.route_index_bounds: Optional[Tuple[int, int]] = None
        self.route_match_distances: List[float] = []
        self.driving_command_labels: List[int] = []
        self.camera_cache = CameraCache()

    def open(self) -> None:
        self.reader.open()

    def close(self) -> None:
        self.reader.close()

    def topic_names(self) -> List[str]:
        return self.reader.topic_names()

    def run(self) -> int:
        self.open()
        try:
            self.parse_localization()
            self.parse_chassis()
            self.parse_cameras()
            self.parse_obstacles()
            self.parse_local_map()
            self.parse_route()
            frame_dicts = self.build_navsim_frame_dicts()
            self.report_route_match_diagnostics(frame_dicts)
            self.dump_navsim_log(frame_dicts)
            return len(frame_dicts)
        finally:
            self.close()

    def _read_topic_as_proto(self, topic: str, proto_cls: Type[Any]) -> List[TopicRecord]:
        records: List[TopicRecord] = []
        for topic_name, msg, bag_timestamp in self.reader.read_messages(topics=[topic]):
            try:
                proto = parse_binary_proto(msg, proto_cls)
                header_timestamp = calculate_header_timestamp(getattr(proto, "header", None))
                timestamp = choose_topic_timestamp(header_timestamp, bag_timestamp)
                records.append(TopicRecord(timestamp=timestamp, message=proto, raw_msg=msg, topic=topic_name))
            except (DecodeError, ValueError) as exc:
                print("[WARN] failed to parse topic {}: {}".format(topic, exc))
        records.sort(key=lambda record: record.timestamp)
        return records

    def _read_topic_raw(self, topic: str) -> List[TopicRecord]:
        records: List[TopicRecord] = []
        for topic_name, msg, timestamp in self.reader.read_messages(topics=[topic]):
            records.append(TopicRecord(timestamp=message_timestamp_to_sec(msg, timestamp), message=msg, raw_msg=msg, topic=topic_name))
        records.sort(key=lambda record: record.timestamp)
        return records

    def parse_localization(self) -> None:
        self.localization_records = self._read_topic_as_proto(self.config.localization_topic, Localization)
        if not self.localization_records:
            raise RuntimeError("No localization records parsed from {}".format(self.config.localization_topic))
        print("[INFO] localization records: {} from {}".format(len(self.localization_records), self.config.localization_topic))

    def parse_chassis(self) -> None:
        self.chassis_records = self._read_topic_as_proto(self.config.chassis_topic, ChassisStates)
        if self.chassis_records:
            print("[INFO] chassis records: {} from {}".format(len(self.chassis_records), self.config.chassis_topic))
        else:
            print("[WARN] chassis topic not found or empty: {}; can_bus will use localization only".format(self.config.chassis_topic))

    def parse_cameras(self) -> None:
        role_to_topic: Dict[str, str] = {}
        for topic in self.topic_names():
            if "/sensor/camera" not in topic or "resize" not in topic:
                continue
            role = infer_camera_role(topic)
            if role is None:
                continue
            # Prefer the first discovered topic per role, matching the reference script behavior.
            role_to_topic.setdefault(role, topic)

        if self.config.front_camera_topic in self.topic_names():
            role_to_topic["front"] = self.config.front_camera_topic

        self.camera_cache.role_to_topic = role_to_topic
        for role, topic in sorted(role_to_topic.items()):
            self.camera_cache.role_to_records[role] = self._read_topic_raw(topic)
            print("[INFO] camera role {:>12s}: {:4d} records from {}".format(
                role,
                len(self.camera_cache.role_to_records[role]),
                topic,
            ))

        if "front" not in self.camera_cache.role_to_records:
            print("[WARN] no front camera records found; sample timeline will fall back to localization")

    def parse_obstacles(self) -> None:
        self.obstacle_records = self._read_topic_as_proto(self.config.obstacle_topic, TrackedObjects)
        print("[INFO] obstacle records: {} from {}".format(len(self.obstacle_records), self.config.obstacle_topic))

    def parse_local_map(self) -> None:
        # Local-map schema is project-specific. Store raw records now and isolate
        # roadblock/traffic-light conversion in convert_map_* methods.
        self.local_map_records = self._read_topic_raw(self.config.local_map_topic)
        print("[INFO] local map records: {} from {}".format(len(self.local_map_records), self.config.local_map_topic))

    def parse_route(self) -> None:
        if self.config.route_result is not None and self.config.route_roadblock_ids:
            self.route_result = self.config.route_result
            self.route_roadblock_ids = list(self.config.route_roadblock_ids)
            self.prepare_route_index_context()
            print("[INFO] route roadblock ids: {} from shared RoutingResult".format(len(self.route_roadblock_ids)))
            return

        route_source_path = self.config.route_source_path or self.config.input_path
        self.route_result, self.route_roadblock_ids = load_valid_route_result(route_source_path, self.config.route_topic)
        self.prepare_route_index_context()
        print("[INFO] route roadblock ids: {} from {}".format(len(self.route_roadblock_ids), route_source_path))

    def prepare_route_index_context(self) -> None:
        self.route_road_entries = extract_route_road_entries(self.route_result)
        self.route_entry_by_pair = {
            (int(entry["path_idx"]), int(entry["roadlink_idx"])): entry
            for entry in self.route_road_entries
        }

        road_counts: Dict[int, int] = {}
        for entry in self.route_road_entries:
            road_pkid = int(entry["road_pkid"])
            road_counts[road_pkid] = road_counts.get(road_pkid, 0) + 1
        self.loop_duplicate_road_pkids = {road_pkid for road_pkid, count in road_counts.items() if count > 1}
        self.route_index_bounds = None
        if not self.loop_duplicate_road_pkids:
            return

        start_match = self.find_route_lane_for_localization_record(self.localization_records[0])
        end_match = self.find_route_lane_for_localization_record(self.localization_records[-1])
        if start_match is None or end_match is None:
            print("[WARN] loop route disambiguation skipped for log {}: cannot match start/end localization".format(
                self.config.resolved_log_name
            ))
            return

        start_in_loop = int(start_match["road_pkid"]) in self.loop_duplicate_road_pkids
        end_in_loop = int(end_match["road_pkid"]) in self.loop_duplicate_road_pkids
        if start_in_loop and end_in_loop:
            raise RuntimeError(
                "ambiguous loop route for log {}: start and end localization both match duplicated route roads".format(
                    self.config.resolved_log_name
                )
            )

        min_idx = 0 if start_in_loop else int(start_match["route_global_idx"])
        max_idx = len(self.route_road_entries) - 1 if end_in_loop else int(end_match["route_global_idx"])
        if min_idx > max_idx:
            print("[WARN] loop route disambiguation skipped for log {}: start route index {} > end route index {}".format(
                self.config.resolved_log_name, min_idx, max_idx
            ))
            return

        self.route_index_bounds = (min_idx, max_idx)
        print("[INFO] loop route disambiguation for log {}: duplicate roads={}, route index bounds=[{}, {}]".format(
            self.config.resolved_log_name,
            len(self.loop_duplicate_road_pkids),
            min_idx,
            max_idx,
        ))

    def find_route_lane_for_localization_record(self, record: TopicRecord) -> Optional[Dict[str, Any]]:
        loc_msg = record.message
        ego_xy = (float(loc_msg.position.x), float(loc_msg.position.y))
        return self.find_nearest_route_lane(ego_xy, use_loop_disambiguation=False)

    def build_sample_times(self) -> List[float]:
        front_records = self.camera_cache.role_to_records.get("front", [])
        if front_records:
            return [record.timestamp for record in front_records]

        start = self.localization_records[0].timestamp
        end = self.localization_records[-1].timestamp
        times = []
        current = start
        while current <= end + 1e-9:
            times.append(current)
            current += self.config.sample_interval
        return times

    def build_navsim_frame_dicts(self) -> List[Dict[str, Any]]:
        sample_times = self.build_sample_times()
        if not sample_times:
            print("[WARN] no valid sample times")
            return []

        frame_dicts = []
        for timestamp in sample_times:
            loc = self.get_nearest_record(self.localization_records, timestamp, self.config.max_localization_delta)
            if loc is None:
                print("[WARN] skip frame {:.3f}: no localization within {:.3f}s".format(
                    timestamp,
                    self.config.max_localization_delta,
                ))
                continue
            chassis = self.get_nearest_record(self.chassis_records, timestamp, self.config.max_chassis_delta)
            chassis_msg = None if chassis is None else chassis.message
            frame_dicts.append(self.build_frame_dict(timestamp, loc.message, chassis_msg))
        return frame_dicts

    def get_nearest_record(
        self,
        records: Sequence[TopicRecord],
        timestamp: float,
        max_delta: Optional[float],
    ) -> Optional[TopicRecord]:
        times = [record.timestamp for record in records]
        idx = nearest_index(times, timestamp, max_delta)
        return None if idx is None else records[idx]

    def build_frame_dict(self, timestamp: float, loc_msg: Localization, chassis_msg: Optional[ChassisStates]) -> Dict[str, Any]:
        vehicle_id = self.config.resolved_vehicle_id
        log_name = self.config.resolved_log_name
        token = make_token(vehicle_id, self.config.base_log_name, timestamp)

        ego_translation, ego_yaw, ego_dynamic_state, can_bus = self.convert_localization(loc_msg, chassis_msg)
        driving_command = self.convert_route_to_driving_command(ego_translation)
        annotations = self.convert_obstacles_to_navsim_anns(timestamp)
        roadblock_ids = self.convert_map_to_roadblock_ids(timestamp)
        traffic_lights = self.convert_map_to_traffic_lights(timestamp)
        cams = self.convert_images_to_navsim_cams(timestamp)

        return {
            "token": token,
            "log_name": log_name,
            "scene_token": log_name,
            "timestamp": int(timestamp * 1e6),
            "map_location": self.config.map_name,
            "ego2global_translation": ego_translation,
            "ego2global_rotation": yaw_to_quaternion(ego_yaw),
            "ego_dynamic_state": ego_dynamic_state,
            "driving_command": driving_command,
            "can_bus": can_bus,
            "cams": cams,
            "lidar_path": "",
            "anns": annotations,
            "roadblock_ids": roadblock_ids,
            "traffic_lights": traffic_lights,
        }

    def convert_localization(
        self,
        loc_msg: Localization,
        chassis_msg: Optional[ChassisStates] = None,
    ) -> Tuple[List[float], float, List[float], List[float]]:
        if not hasattr(loc_msg, "position"):
            raise ValueError("Localization message has no position field")

        x = float(loc_msg.position.x)
        y = float(loc_msg.position.y)
        z = float(getattr(loc_msg.position, "z", 0.0))
        yaw = normalize_angle(float(getattr(loc_msg, "yaw", 0.0)))

        # Localization is the primary source for ego motion. ChassisStates is a
        # fallback for fields that may be absent in future localization schemas.
        motion_states = getattr(chassis_msg, "vehicle_motion_states", None)
        drive_states = getattr(chassis_msg, "drive_system_states", None)
        steer_states = getattr(chassis_msg, "steer_system_states", None)
        vx = first_valid_float(
            get_float_attr(loc_msg, "longitudinal_v"),
            get_float_attr(motion_states, "speed"),
            get_float_attr(drive_states, "vcu_speed"),
            default=0.0,
        )
        vy = first_valid_float(get_float_attr(loc_msg, "lateral_v"), default=0.0)
        ax = first_valid_float(
            get_float_attr(loc_msg, "longitudinal_a"),
            get_float_attr(motion_states, "longitude_acceleration"),
            get_float_attr(motion_states, "acceleration"),
            default=0.0,
        )
        ay = first_valid_float(
            get_float_attr(loc_msg, "lateral_a"),
            get_float_attr(motion_states, "lateral_acceleration"),
            default=0.0,
        )
        yaw_rate = first_valid_float(
            get_float_attr(loc_msg, "yaw_v"),
            get_float_attr(motion_states, "angular_speed"),
            default=0.0,
        )
        # Chassis2025 stores steering wheel state in degrees / degrees per second.
        # Convert to radians so downstream ego status features stay numerically compact.
        steering_wheel_angle = math.radians(first_valid_float(
            get_float_attr(steer_states, "steer_wheel_angle"),
            default=0.0,
        ))
        steering_wheel_rate = math.radians(first_valid_float(
            get_float_attr(steer_states, "steer_wheel_speed"),
            default=0.0,
        ))

        can_bus = [0.0] * 18
        can_bus[0] = x
        can_bus[1] = y
        can_bus[2] = z
        can_bus[3:7] = yaw_to_quaternion(yaw)
        can_bus[7] = vx
        can_bus[8] = vy
        can_bus[10] = ax
        can_bus[11] = ay
        can_bus[15] = yaw_rate
        can_bus[16] = steering_wheel_angle
        can_bus[17] = steering_wheel_rate

        return [x, y, z], yaw, [vx, vy, ax, ay], can_bus

    def convert_route_to_driving_command(self, ego_translation: Sequence[float]) -> np.ndarray:
        command_index = COMMAND_UNKNOWN
        if self.route_result is None:
            self.driving_command_labels.append(command_index)
            return make_driving_command(command_index)

        lane_match = self.find_nearest_route_lane((float(ego_translation[0]), float(ego_translation[1])))
        if lane_match is None:
            self.driving_command_labels.append(command_index)
            return make_driving_command(command_index)

        link_match = self.find_nearest_route_link_around_lane(
            (float(ego_translation[0]), float(ego_translation[1])), lane_match
        )
        nearest_distance = float(lane_match["distance_to_lane"])
        if link_match is not None and link_match["distance_to_link"] <= nearest_distance:
            nearest_distance = float(link_match["distance_to_link"])
            link_command_index = link_type_to_command_index(int(link_match["link_type"]))
            if nearest_distance <= self.config.max_lateral_match_distance and link_command_index is not None:
                command_index = link_command_index
                self.route_match_distances.append(nearest_distance)
                self.driving_command_labels.append(command_index)
                return make_driving_command(command_index)

        self.route_match_distances.append(nearest_distance)
        if nearest_distance > self.config.max_lateral_match_distance:
            self.driving_command_labels.append(command_index)
            return make_driving_command(command_index)

        remaining_distance = self.config.driving_command_distance - lane_match["distance_to_lane_end"]
        if remaining_distance <= 0.0:
            command_index = COMMAND_STRAIGHT
        else:
            command_index = self.trace_route_command_from_lane(lane_match, remaining_distance)

        self.driving_command_labels.append(command_index)
        return make_driving_command(command_index)

    def report_route_match_diagnostics(self, frame_dicts: Sequence[Dict[str, Any]]) -> None:
        if not frame_dicts or not self.route_match_distances:
            return

        distances = np.asarray(self.route_match_distances, dtype=np.float64)
        threshold = float(self.config.max_lateral_match_distance)
        exceed_ratio = float(np.mean(distances > threshold))
        unknown_ratio = 0.0
        if self.driving_command_labels:
            unknown_ratio = float(np.mean(np.asarray(self.driving_command_labels) == COMMAND_UNKNOWN))

        if exceed_ratio <= 0.5 and unknown_ratio <= 0.5:
            return

        print(
            "[WARN] route/ego spatial mismatch for log {}: "
            "{:.1f}% frames farther than {:.2f}m from route lane; "
            "distance min/median/max = {:.2f}/{:.2f}/{:.2f}m; "
            "unknown command ratio = {:.1f}%".format(
                self.config.resolved_log_name,
                exceed_ratio * 100.0,
                threshold,
                float(np.min(distances)),
                float(np.median(distances)),
                float(np.max(distances)),
                unknown_ratio * 100.0,
            )
        )

    def find_nearest_route_lane(
        self,
        ego_xy: Tuple[float, float],
        use_loop_disambiguation: bool = True,
    ) -> Optional[Dict[str, Any]]:
        best_match: Optional[Dict[str, Any]] = None
        for path_idx, route_path in enumerate(self.route_result.paths):
            for roadlink_idx, route_roadlink in enumerate(route_path.route_roadlinks):
                route_entry = self.route_entry_by_pair.get((path_idx, roadlink_idx))
                if route_entry is None:
                    continue
                if not route_roadlink.HasField("route_road"):
                    continue
                road = route_roadlink.route_road
                road_pkid = int(route_entry["road_pkid"])
                route_global_idx = int(route_entry["global_idx"])
                if (
                    use_loop_disambiguation
                    and road_pkid in self.loop_duplicate_road_pkids
                    and self.route_index_bounds is not None
                ):
                    min_idx, max_idx = self.route_index_bounds
                    if route_global_idx < min_idx or route_global_idx > max_idx:
                        continue

                for section in road.sections:
                    lanes = sorted(section.lanes, key=lambda lane: lane.lane_id)
                    for lane_order, lane in enumerate(lanes):
                        lane_points = points_to_xy(lane.point)
                        if len(lane_points) < 2:
                            continue
                        distance_to_lane, lane_progress, lane_length = project_point_to_polyline(ego_xy, lane_points)
                        if math.isinf(distance_to_lane):
                            continue
                        match = {
                            "path_idx": path_idx,
                            "roadlink_idx": roadlink_idx,
                            "route_global_idx": route_global_idx,
                            "road_pkid": road_pkid,
                            "section_id": int(lane.section_id),
                            "lane_id": int(lane.lane_id),
                            "is_edge_lane": lane_order == 0 or lane_order == len(lanes) - 1,
                            "distance_to_lane": distance_to_lane,
                            "lane_progress": lane_progress,
                            "lane_length": lane_length,
                            "distance_to_lane_end": max(0.0, lane_length - lane_progress),
                        }
                        if best_match is None or distance_to_lane < best_match["distance_to_lane"]:
                            best_match = match

        return best_match

    def find_nearest_route_link_around_lane(
        self,
        ego_xy: Tuple[float, float],
        lane_match: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        candidate_links: List[Tuple[Any, int]] = []
        current_path_idx = int(lane_match["path_idx"])
        current_roadlink_idx = int(lane_match["roadlink_idx"])
        current_route_path = self.route_result.paths[current_path_idx]
        current_road_pkid = int(lane_match["road_pkid"])

        for idx in [current_roadlink_idx - 1, current_roadlink_idx, current_roadlink_idx + 1]:
            if idx < 0 or idx >= len(current_route_path.route_roadlinks):
                continue
            route_roadlink = current_route_path.route_roadlinks[idx]
            route_entry = self.route_entry_by_pair.get((current_path_idx, idx))
            if route_entry is None:
                continue
            if (
                int(route_entry["road_pkid"]) in self.loop_duplicate_road_pkids
                and self.route_index_bounds is not None
            ):
                min_idx, max_idx = self.route_index_bounds
                route_global_idx = int(route_entry["global_idx"])
                if route_global_idx < min_idx or route_global_idx > max_idx:
                    continue

            for link in route_roadlink.route_links:
                if len(link.geom) == 0:
                    continue
                is_predecessor = int(link.to_road_pkid) == current_road_pkid
                is_successor = int(link.from_road_pkid) == current_road_pkid
                if is_predecessor or is_successor:
                    candidate_links.append((link, idx))

        best_match: Optional[Dict[str, Any]] = None
        for link, roadlink_idx in candidate_links:
            link_points = points_to_xy(link.geom)
            if len(link_points) < 2:
                continue
            distance_to_link, link_progress, link_length = project_point_to_polyline(ego_xy, link_points)
            if math.isinf(distance_to_link):
                continue
            match = {
                "path_idx": current_path_idx,
                "roadlink_idx": roadlink_idx,
                "distance_to_link": distance_to_link,
                "link_progress": link_progress,
                "link_length": link_length,
                "link_type": int(link.type),
                "from_road_pkid": int(link.from_road_pkid),
                "to_road_pkid": int(link.to_road_pkid),
            }
            if best_match is None or distance_to_link < best_match["distance_to_link"]:
                best_match = match

        return best_match

    def trace_route_command_from_lane(self, lane_match: Dict[str, Any], remaining_distance: float) -> int:
        route_path = self.route_result.paths[lane_match["path_idx"]]
        roadlink_idx = int(lane_match["roadlink_idx"])
        current_road_pkid = int(lane_match["road_pkid"])
        current_section_id = int(lane_match["section_id"])
        current_lane_id = int(lane_match["lane_id"])

        while remaining_distance > 0.0:
            if roadlink_idx >= len(route_path.route_roadlinks):
                return COMMAND_STRAIGHT

            route_roadlink = route_path.route_roadlinks[roadlink_idx]
            link = self.find_successor_link(route_roadlink.route_links, current_road_pkid, current_section_id, current_lane_id)
            if link is None:
                if roadlink_idx >= len(route_path.route_roadlinks) - 1:
                    return COMMAND_STRAIGHT
                return COMMAND_UNKNOWN

            if len(link.geom) > 0:
                command_index = link_type_to_command_index(int(link.type))
                if command_index is not None:
                    return command_index
                remaining_distance -= polyline_length(link.geom)

            current_road_pkid = int(link.to_road_pkid)
            current_section_id = int(link.to_section_id)
            current_lane_id = int(link.to_lane_id)
            next_roadlink_idx = self.find_next_roadlink_idx(route_path, roadlink_idx, current_road_pkid)
            if next_roadlink_idx is None:
                return COMMAND_STRAIGHT

            roadlink_idx = next_roadlink_idx
            road_length = self.get_route_road_length(route_path.route_roadlinks[roadlink_idx])
            if remaining_distance <= road_length:
                return COMMAND_STRAIGHT
            remaining_distance -= road_length

        return COMMAND_STRAIGHT

    def find_successor_link(self, route_links: Sequence[Any], road_pkid: int, section_id: int, lane_id: int) -> Optional[Any]:
        candidates = [
            link for link in route_links
            if int(link.from_road_pkid) == road_pkid
            and int(link.from_section_id) == section_id
            and int(link.from_lane_id) == lane_id
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda link: (len(link.geom) == 0, abs(int(link.to_lane_id) - lane_id)))
        return candidates[0]

    def find_next_roadlink_idx(self, route_path: Any, current_idx: int, road_pkid: int) -> Optional[int]:
        for idx in range(current_idx + 1, len(route_path.route_roadlinks)):
            route_roadlink = route_path.route_roadlinks[idx]
            if route_roadlink.HasField("route_road") and int(route_roadlink.route_road.pkid) == road_pkid:
                return idx
        return None

    def get_route_road_length(self, route_roadlink: Any) -> float:
        if not route_roadlink.HasField("route_road"):
            return 0.0
        road = route_roadlink.route_road
        if road.HasField("length") and road.length > 0.0:
            return float(road.length)
        return polyline_length(road.geom)

    def convert_obstacles_to_navsim_anns(self, timestamp: float) -> Dict[str, Any]:
        record = self.get_nearest_record(self.obstacle_records, timestamp, self.config.max_obstacle_delta)
        if record is None:
            return self.make_empty_annotations()

        boxes: List[List[float]] = []
        names: List[str] = []
        velocities: List[List[float]] = []
        instance_tokens: List[str] = []
        track_tokens: List[str] = []

        timestamp_us = int(timestamp * 1e6)
        log_name = self.config.resolved_log_name
        for tracked_object in record.message.objs:
            obj = tracked_object.obj
            navsim_name = self.convert_object_type_to_navsim_name(int(obj.type))
            if navsim_name is None:
                continue

            box = [
                float(obj.center.x),
                float(obj.center.y),
                float(obj.center.z),
                float(obj.size.x),
                float(obj.size.y),
                float(obj.size.z),
                normalize_angle(float(tracked_object.yaw)),
            ]
            velocity = [
                float(tracked_object.velocity.x),
                float(tracked_object.velocity.y),
                float(tracked_object.velocity.z),
            ]

            object_id = int(obj.id)
            boxes.append(box)
            names.append(navsim_name)
            velocities.append(velocity)
            track_tokens.append(str(object_id))
            instance_tokens.append("{}_{}_{}".format(log_name, timestamp_us, object_id))

        if not boxes:
            return self.make_empty_annotations()

        return {
            "gt_boxes": np.asarray(boxes, dtype=np.float32),
            "gt_names": names,
            "gt_velocity_3d": np.asarray(velocities, dtype=np.float32),
            "instance_tokens": instance_tokens,
            "track_tokens": track_tokens,
        }

    def make_empty_annotations(self) -> Dict[str, Any]:
        return {
            "gt_boxes": np.zeros((0, 7), dtype=np.float32),
            "gt_names": [],
            "gt_velocity_3d": np.zeros((0, 3), dtype=np.float32),
            "instance_tokens": [],
            "track_tokens": [],
        }

    def convert_object_type_to_navsim_name(self, object_type: int) -> Optional[str]:
        bevfusion_name = BEVFUSION_OBJECT_TYPE_MAP.get(object_type, "unknown")
        return NAVSIM_OBJECT_TYPE_MAP.get(bevfusion_name)

    def convert_map_to_roadblock_ids(self, timestamp: float) -> List[str]:
        # Route ids are scenario-level information. Keep order and duplicates from RoutingResult.
        _ = timestamp
        return list(self.route_roadblock_ids)

    def convert_map_to_traffic_lights(self, timestamp: float) -> List[Tuple[str, bool]]:
        # TODO: convert local-map/perception traffic-light states if available.
        _ = self.get_nearest_record(self.local_map_records, timestamp, self.config.max_local_map_delta)
        return []

    def make_empty_camera(self) -> Dict[str, Any]:
        return {
            "data_path": "",
            "sensor2lidar_rotation": np.eye(3, dtype=np.float32),
            "sensor2lidar_translation": np.zeros(3, dtype=np.float32),
            "cam_intrinsic": np.eye(3, dtype=np.float32),
            "distortion": np.zeros(5, dtype=np.float32),
        }

    def convert_images_to_navsim_cams(self, timestamp: float) -> Dict[str, Dict[str, Any]]:
        cameras = {camera_name: self.make_empty_camera() for camera_name in NAVSIM_CAMERA_NAMES}
        for role, navsim_key in CAMERA_ROLE_TO_NAVSIM_KEY.items():
            records = self.camera_cache.role_to_records.get(role, [])
            record = self.get_nearest_record(records, timestamp, self.config.max_camera_delta)
            if record is None:
                continue
            camera_dict = self.make_empty_camera()
            camera_dict["data_path"] = self.extract_camera_image(role, record)
            cameras[navsim_key] = camera_dict
        return cameras

    def extract_camera_image(self, role: str, record: TopicRecord) -> str:
        if not self.config.extract_images or self.config.sensor_blobs_root is None:
            return ""

        payload = getattr(record.message, "data", None)
        if payload is None:
            return ""

        log_name = self.config.resolved_log_name
        timestamp_us = int(record.timestamp * 1e6)
        rel_path = Path(log_name) / role / "{}.jpg".format(timestamp_us)
        abs_path = self.config.sensor_blobs_root / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)

        if not abs_path.exists():
            # The reference bags publish /resize/nvjpeg as CompressedImage. The data
            # payload is already encoded image bytes, so writing it directly avoids
            # a cv_bridge dependency in this data-preparation scaffold.
            abs_path.write_bytes(bytes(payload))
        return str(rel_path)

    def dump_navsim_log(self, frame_dicts: List[Dict[str, Any]]) -> None:
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.output_path.open("wb") as f:
            pickle.dump(frame_dicts, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("[INFO] wrote {} frames to {}".format(len(frame_dicts), self.config.output_path))

        if self.config.write_debug_json:
            debug_path = self.config.output_path.with_suffix(".debug.json")
            debug = {
                "input_path": str(self.config.input_path),
                "bag_paths": [str(path) for path in self.config.bag_paths],
                "vehicle_id": self.config.resolved_vehicle_id,
                "base_log_name": self.config.base_log_name,
                "log_name": self.config.resolved_log_name,
                "output_path": str(self.config.output_path),
                "sensor_blobs_root": str(self.config.sensor_blobs_root) if self.config.sensor_blobs_root else None,
                "route_source_path": str(self.config.route_source_path) if self.config.route_source_path else None,
                "route_roadblock_count": len(self.route_roadblock_ids),
                "frame_count": len(frame_dicts),
                "first_token": frame_dicts[0]["token"] if frame_dicts else None,
                "last_token": frame_dicts[-1]["token"] if frame_dicts else None,
                "camera_topics": self.camera_cache.role_to_topic,
                "topics": {
                    "localization": self.config.localization_topic,
                    "chassis": self.config.chassis_topic,
                    "obstacle": self.config.obstacle_topic,
                    "local_map": self.config.local_map_topic,
                    "route": self.config.route_topic,
                },
            }
            debug_path.write_text(json.dumps(debug, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ROS bag logs to NAVSIM-compatible pkl for ReCogDrive/RL")
    parser.add_argument("--input-path", default=None, help="Input .bag file or one log directory containing .bag files")
    parser.add_argument("--raw-data-root", default=None, help="Directory containing multiple log directories, e.g. .../data/2026-03-25-12-13-50")
    parser.add_argument("--dataset-root", default=None, help="Output dataset root. Logs/images are written under dataset_root/dataset")
    parser.add_argument("--split", default="trainval", choices=["trainval", "test"], help="Dataset split name for output folders")
    parser.add_argument("--route-source-path", default=None, help="Routing .bag file or directory shared by all logs")
    parser.add_argument("--output-path", default=None, help="Output NAVSIM-style log pkl path for single input mode")
    parser.add_argument("--sensor-blobs-root", default=None, help="Root for extracted camera images in single input mode")
    parser.add_argument("--log-name", default=None, help="Override base log name before vehicle_id prefix is added; only for single input mode")
    parser.add_argument("--vehicle-id", default=None, help="Fallback vehicle id when it cannot be inferred from bag filename")
    parser.add_argument("--map-name", default="las_vegas", help="NAVSIM/nuPlan map name to write into frame_dict")
    parser.add_argument("--localization-topic", default=DEFAULT_LOCALIZATION_TOPIC)
    parser.add_argument("--chassis-topic", default=DEFAULT_CHASSIS_TOPIC)
    parser.add_argument("--obstacle-topic", default=DEFAULT_OBSTACLE_TOPIC)
    parser.add_argument("--local-map-topic", default=DEFAULT_LOCAL_MAP_TOPIC)
    parser.add_argument("--route-topic", default=DEFAULT_ROUTE_TOPIC)
    parser.add_argument("--front-camera-topic", default=DEFAULT_FRONT_CAMERA_TOPIC)
    parser.add_argument("--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--num-history-frames", type=int, default=DEFAULT_HISTORY_FRAMES)
    parser.add_argument("--num-future-frames", type=int, default=DEFAULT_FUTURE_FRAMES)
    parser.add_argument("--max-localization-delta", type=float, default=0.05)
    parser.add_argument("--max-chassis-delta", type=float, default=0.05)
    parser.add_argument("--max-camera-delta", type=float, default=DEFAULT_MAX_CAMERA_DELTA)
    parser.add_argument("--max-obstacle-delta", type=float, default=DEFAULT_MAX_OBSTACLE_DELTA)
    parser.add_argument("--max-local-map-delta", type=float, default=DEFAULT_MAX_LOCAL_MAP_DELTA)
    parser.add_argument("--max-sensor-delta", type=float, default=None, help="Deprecated: override camera/obstacle/local-map deltas with one value")
    parser.add_argument("--driving-command-distance", type=float, default=DEFAULT_DRIVING_COMMAND_DISTANCE)
    parser.add_argument("--max-lateral-match-distance", type=float, default=DEFAULT_MAX_LATERAL_MATCH_DISTANCE)
    parser.add_argument("--no-extract-images", action="store_true", help="Do not write camera payloads to sensor_blobs_root")
    parser.add_argument("--write-debug-json", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only list planned logs/topics and exit")
    return parser.parse_args()


def discover_log_dirs(raw_data_root: Path) -> List[Path]:
    """Find child directories that look like one ROS bag log segment."""
    log_dirs = []
    for child in sorted(raw_data_root.iterdir()):
        if not child.is_dir():
            continue
        if list(child.glob("*.bag")):
            log_dirs.append(child)
    return log_dirs


def make_converter_config(
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    sensor_blobs_root: Optional[Path],
) -> ConverterConfig:
    return ConverterConfig(
        input_path=input_path,
        output_path=output_path,
        sensor_blobs_root=sensor_blobs_root,
        route_source_path=getattr(args, "resolved_route_source_path", None),
        log_name=args.log_name if args.raw_data_root is None else None,
        vehicle_id=args.vehicle_id,
        map_name=args.map_name,
        localization_topic=args.localization_topic,
        chassis_topic=args.chassis_topic,
        obstacle_topic=args.obstacle_topic,
        local_map_topic=args.local_map_topic,
        route_topic=args.route_topic,
        front_camera_topic=args.front_camera_topic,
        sample_interval=args.sample_interval,
        num_history_frames=args.num_history_frames,
        num_future_frames=args.num_future_frames,
        max_localization_delta=args.max_localization_delta,
        max_chassis_delta=args.max_chassis_delta,
        max_camera_delta=args.max_sensor_delta if args.max_sensor_delta is not None else args.max_camera_delta,
        max_obstacle_delta=args.max_sensor_delta if args.max_sensor_delta is not None else args.max_obstacle_delta,
        max_local_map_delta=args.max_sensor_delta if args.max_sensor_delta is not None else args.max_local_map_delta,
        driving_command_distance=args.driving_command_distance,
        max_lateral_match_distance=args.max_lateral_match_distance,
        extract_images=not args.no_extract_images,
        write_debug_json=args.write_debug_json,
    )


def ensure_navsim_dataset_layout(dataset_root: Path) -> None:
    required_dirs = [
        dataset_root / "dataset" / "maps",
        dataset_root / "dataset" / "navsim_logs" / "test",
        dataset_root / "dataset" / "navsim_logs" / "trainval",
        dataset_root / "dataset" / "sensor_blobs" / "test",
        dataset_root / "dataset" / "sensor_blobs" / "trainval",
        dataset_root / "exp",
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def build_config_for_input(args: argparse.Namespace, input_path: Path) -> ConverterConfig:
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        logs_root = dataset_root / "dataset" / "navsim_logs" / args.split
        sensor_root = dataset_root / "dataset" / "sensor_blobs" / args.split
        temp_config = make_converter_config(args, input_path, logs_root / "__placeholder__.pkl", sensor_root)
        output_path = logs_root / "{}.pkl".format(temp_config.resolved_log_name)
        temp_config.output_path = output_path
        return temp_config

    if args.output_path is None:
        raise ValueError("--output-path is required when --dataset-root is not provided")
    sensor_root = Path(args.sensor_blobs_root) if args.sensor_blobs_root else None
    return make_converter_config(args, input_path, Path(args.output_path), sensor_root)


def validate_config(config: ConverterConfig) -> Optional[str]:
    if not config.input_path.exists():
        return "input path does not exist: {}".format(config.input_path)
    if not config.bag_paths:
        return "no .bag files found under input path: {}".format(config.input_path)
    if config.route_source_path is not None and not config.route_source_path.exists():
        return "route source path does not exist: {}".format(config.route_source_path)
    if config.extract_images and config.sensor_blobs_root is None:
        print("[WARN] image extraction requested but sensor_blobs_root is empty; camera data_path will be empty")
    return None


def collect_input_paths(args: argparse.Namespace) -> List[Path]:
    if args.raw_data_root is not None:
        raw_data_root = Path(args.raw_data_root)
        if not raw_data_root.is_dir():
            raise ValueError("--raw-data-root must be a directory: {}".format(raw_data_root))
        log_dirs = discover_log_dirs(raw_data_root)
        if not log_dirs:
            raise ValueError("no log directories with .bag files found under {}".format(raw_data_root))
        return log_dirs

    if args.input_path is None:
        raise ValueError("Either --input-path or --raw-data-root is required")
    return [Path(args.input_path)]


def print_dry_run(configs: Sequence[ConverterConfig]) -> int:
    for config in configs:
        error = validate_config(config)
        if error is not None:
            print("[ERROR] {}".format(error))
            return 1

        converter = RosbagToRecogDrivePklConverter(config)
        converter.open()
        try:
            topics = converter.topic_names()
            print("[INFO] input: {}".format(config.input_path))
            print("[INFO] output pkl: {}".format(config.output_path))
            print("[INFO] sensor blobs root: {}".format(config.sensor_blobs_root))
            print("[INFO] vehicle_id: {}".format(config.resolved_vehicle_id))
            print("[INFO] log_name: {}".format(config.resolved_log_name))
            print("[INFO] bag files:")
            for path in config.bag_paths:
                print("  {}".format(path))
            print("[INFO] topics:")
            for topic in topics:
                print("  {}".format(topic))
            print("[INFO] discovered camera roles:")
            for topic in topics:
                role = infer_camera_role(topic)
                if role is not None:
                    print("  {:>12s}: {}".format(role, topic))
        finally:
            converter.close()
    return 0


def run_one_config(config: ConverterConfig) -> Tuple[Path, int]:
    error = validate_config(config)
    if error is not None:
        raise RuntimeError(error)

    converter = RosbagToRecogDrivePklConverter(config)
    frame_count = converter.run()
    if frame_count <= 0:
        print("[WARN] no frame_dict generated for {}".format(config.input_path))
    return config.output_path, frame_count


def main() -> int:
    args = parse_args()
    try:
        route_source_path = Path(args.route_source_path) if args.route_source_path else None
        if route_source_path is None and args.raw_data_root is not None:
            route_source_path = discover_route_source_path(Path(args.raw_data_root))
        args.resolved_route_source_path = route_source_path

        input_paths = collect_input_paths(args)
        configs = [build_config_for_input(args, input_path) for input_path in input_paths]
    except ValueError as exc:
        print("[ERROR] {}".format(exc))
        return 1

    if args.dry_run:
        return print_dry_run(configs)

    if args.dataset_root:
        ensure_navsim_dataset_layout(Path(args.dataset_root))

    try:
        route_source_path = configs[0].route_source_path or configs[0].input_path
        shared_route_result, shared_roadblock_ids = load_valid_route_result(route_source_path, configs[0].route_topic)
        print("[INFO] shared route source: {}".format(route_source_path))
        print("[INFO] shared route roadblock ids: {}".format(len(shared_roadblock_ids)))
        for config in configs:
            config.route_result = shared_route_result
            config.route_roadblock_ids = list(shared_roadblock_ids)
    except Exception as exc:
        print("[ERROR] failed to load valid routing result: {}".format(exc))
        return 1

    total_frames = 0
    failed = 0
    for idx, config in enumerate(configs, start=1):
        print("[INFO] converting log {}/{}: {}".format(idx, len(configs), config.input_path))
        try:
            output_path, frame_count = run_one_config(config)
            total_frames += frame_count
            print("[INFO] finished {} frames -> {}".format(frame_count, output_path))
        except Exception as exc:
            failed += 1
            print("[ERROR] failed to convert {}: {}".format(config.input_path, exc))

    print("[INFO] conversion summary: logs={}, failed={}, frames={}".format(len(configs), failed, total_frames))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
