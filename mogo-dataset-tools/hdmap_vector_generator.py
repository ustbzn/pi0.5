#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HDMapVectorGenerator: 从同源 sqlite 离线生成 PlanVOCA-style 结构化 HDMap token。

与 HDMapBEVGenerator 共用：
  - sqlite 连接 + spatialite 扩展
  - WGS84 / UTM 转换
  - ego-local 坐标变换
  - 几何采样

差异：
  - 查询时保留 fid / 属性 / 几何类型，而不是直接栅格化为 0/1 mask。
  - 输出 PlanVOCA MogoFeatureBuilder._get_map_features 等价字段。
"""

import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pyproj
except ImportError:
    pyproj = None

try:
    from shapely import wkt as shapely_wkt
    from shapely.geometry import LineString, Point as ShapelyPoint
except ImportError:
    shapely_wkt = None
    LineString = None
    ShapelyPoint = None


# PlanVOCA polygon_type 取值约定：0=LANE，1=LANE_CONNECTOR
POLYGON_TYPE_LANE = 0
POLYGON_TYPE_LANE_CONNECTOR = 1

# PlanVOCA traffic light enum: GREEN=0, YELLOW=1, RED=2, UNKNOWN=3
TL_UNKNOWN = 3

DEFAULT_SAMPLE_POINTS = 20  # 每条 lane 采样的中心线段数 (P)，对应 21 个采样点
# PlanVOCA 默认 120m 视野；同时确保 ≥ DrivePi0 BEV 覆盖范围（约 97.6m）。
DEFAULT_RADIUS_M = 120.0
DEFAULT_SPEED_LIMIT_KMH = 80.0  # PlanVOCA 默认限速值（km/h）

# PlanVOCA 内部以 m/s 存储限速，从 km/h 除以 3.6
DEFAULT_SPEED_LIMIT_MPS = DEFAULT_SPEED_LIMIT_KMH / 3.6


def _require_dependencies():
    if pyproj is None:
        raise ImportError("HDMapVectorGenerator 需要 pyproj，请先 pip install pyproj")
    if shapely_wkt is None:
        raise ImportError("HDMapVectorGenerator 需要 shapely，请先 pip install shapely")


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
        "加载 Spatialite 扩展失败：{}".format("; ".join(errors))
    )


def _build_rotation_matrix_2d(yaw: float) -> np.ndarray:
    """与 parseRosBag.build_rotation_matrix_from_yaw 对齐的 2D 旋转矩阵。

    parseRosBag 中 rotation = [[sin, cos, 0], [-cos, sin, 0], [0, 0, 1]]
    对应的 transform_global_to_local 用的是 R.T。这里只取前 2x2 并直接给出 R.T。
    """
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    R = np.array([
        [sin_yaw, cos_yaw],
        [-cos_yaw, sin_yaw],
    ], dtype=np.float64)
    # 与 transform_global_to_local 一致：local = R.T @ (global - ego)
    return R.T


def _resample_polyline(coords: np.ndarray, num_points: int) -> Optional[np.ndarray]:
    """将任意 polyline 等弧长重采样到 num_points 个点。"""
    if coords is None or len(coords) < 2:
        return None
    coords = np.asarray(coords, dtype=np.float64)
    diffs = np.diff(coords, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 1e-6:
        return np.tile(coords[0:1], (num_points, 1))
    targets = np.linspace(0.0, total, num_points)
    out = np.zeros((num_points, 2), dtype=np.float64)
    j = 0
    for i, t in enumerate(targets):
        while j < len(cum) - 2 and cum[j + 1] < t:
            j += 1
        seg_total = cum[j + 1] - cum[j]
        if seg_total <= 1e-6:
            out[i] = coords[j]
        else:
            ratio = (t - cum[j]) / seg_total
            out[i] = coords[j] + ratio * (coords[j + 1] - coords[j])
    return out


class HDMapVectorGenerator:
    """从 mogo HDMap sqlite 直接生成 PlanVOCA-style 结构化 lane token。

    用法：
        gen = HDMapVectorGenerator(db_path=..., utm_zone=50)
        result = gen.build(ego_utm_x, ego_utm_y, ego_utm_z, ego_yaw, route_lane_ids=...)
        result["point_position"], result["point_vector"], ...
    """

    def __init__(
        self,
        db_path: str,
        radius: float = DEFAULT_RADIUS_M,
        sample_points: int = DEFAULT_SAMPLE_POINTS,
        utm_zone: int = 50,
        utm_south: bool = False,
        max_lanes: int = 256,
    ):
        _require_dependencies()
        if not os.path.isfile(db_path):
            raise IOError("HDMap sqlite 文件不存在：{}".format(db_path))
        self.db_path = db_path
        self.radius = float(radius)
        self.sample_points = int(sample_points)
        self.utm_zone = int(utm_zone)
        self.utm_south = bool(utm_south)
        self.max_lanes = int(max_lanes)

        self.connection = sqlite3.connect(self.db_path)
        self.connection.enable_load_extension(True)
        _load_spatialite_extension(self.connection)
        self._init_transformers()

    def _init_transformers(self):
        proj4 = "+proj=utm +zone={} +datum=WGS84 +units=m +no_defs".format(self.utm_zone)
        if self.utm_south:
            proj4 += " +south"
        utm_crs = pyproj.CRS.from_proj4(proj4)
        self._wgs84_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        self._utm_to_wgs84 = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    def close(self):
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _compute_query_bounds(self, lon: float, lat: float) -> Tuple[float, float, float, float]:
        deg_per_meter_lon = 1.0 / 111320.0
        deg_per_meter_lat = 1.0 / 110540.0
        r = self.radius
        return (
            lon - r * deg_per_meter_lon,
            lat - r * deg_per_meter_lat,
            lon + r * deg_per_meter_lon,
            lat + r * deg_per_meter_lat,
        )

    def _query_lanes(self, lon: float, lat: float) -> List[Dict[str, Any]]:
        # mogo TX_LANE 没有 pkid 列，唯一键是 (road_pkid, section_id, lane_id) 三元组
        lon_min, lat_min, lon_max, lat_max = self._compute_query_bounds(lon, lat)
        sql = """
        SELECT road_pkid, section_id, lane_id, lane_type, speed_limit, AsText(geom)
        FROM TX_LANE
        WHERE ST_Intersects(geom, BuildMBR(?, ?, ?, ?, 4326))
        """
        cur = self.connection.cursor()
        cur.execute(sql, (lon_min, lat_min, lon_max, lat_max))
        rows = []
        for road_pkid, section_id, lane_id, lane_type, speed_limit, wkt in cur.fetchall():
            rows.append({
                "type": POLYGON_TYPE_LANE,
                "road_pkid": int(road_pkid) if road_pkid is not None else -1,
                "section_id": int(section_id) if section_id is not None else -1,
                "lane_id": int(lane_id) if lane_id is not None else -1,
                "lane_type": int(lane_type) if lane_type is not None else -1,
                "speed_limit": float(speed_limit) if speed_limit is not None else 0.0,
                "wkt": wkt,
            })
        return rows

    def _query_lane_links(self, lon: float, lat: float) -> List[Dict[str, Any]]:
        lon_min, lat_min, lon_max, lat_max = self._compute_query_bounds(lon, lat)
        sql = """
        SELECT pkid, from_road_pkid, from_section_id, from_lane_id,
               to_road_pkid, to_section_id, to_lane_id, AsText(geom)
        FROM TX_LANE_LINK
        WHERE ST_Intersects(geom, BuildMBR(?, ?, ?, ?, 4326))
        """
        cur = self.connection.cursor()
        cur.execute(sql, (lon_min, lat_min, lon_max, lat_max))
        rows = []
        for (pkid, from_rd, from_sec, from_ln, to_rd, to_sec, to_ln, wkt) in cur.fetchall():
            rows.append({
                "type": POLYGON_TYPE_LANE_CONNECTOR,
                "pkid": int(pkid) if pkid is not None else -1,
                "from_road_pkid": int(from_rd) if from_rd is not None else -1,
                "from_section_id": int(from_sec) if from_sec is not None else -1,
                "from_lane_id": int(from_ln) if from_ln is not None else -1,
                "to_road_pkid": int(to_rd) if to_rd is not None else -1,
                "to_section_id": int(to_sec) if to_sec is not None else -1,
                "to_lane_id": int(to_ln) if to_ln is not None else -1,
                "speed_limit": 0.0,
                "wkt": wkt,
            })
        return rows

    def _wkt_line_to_utm(self, wkt: str) -> Optional[np.ndarray]:
        """从 WKT 线串解析为 UTM 平面下的 polyline。"""
        if not wkt:
            return None
        geom = shapely_wkt.loads(wkt)
        if geom.is_empty:
            return None
        if geom.geom_type == "LineString":
            coords = np.asarray(geom.coords, dtype=np.float64)
        elif geom.geom_type == "MultiLineString":
            sub = max(geom.geoms, key=lambda g: g.length, default=None)
            if sub is None or sub.is_empty:
                return None
            coords = np.asarray(sub.coords, dtype=np.float64)
        else:
            return None
        if coords.shape[0] < 2 or coords.shape[1] < 2:
            return None
        xs, ys = self._wgs84_to_utm.transform(coords[:, 0], coords[:, 1])
        return np.stack([np.asarray(xs), np.asarray(ys)], axis=1)

    def build(
        self,
        ego_utm_x: float,
        ego_utm_y: float,
        ego_utm_z: float,
        ego_yaw: float,
        route_lane_keys: Optional[Sequence[Tuple[int, int, int]]] = None,
        route_link_keys: Optional[Sequence[Tuple[int, int, int, int, int, int]]] = None,
    ) -> Dict[str, Any]:
        """生成 PlanVOCA-style map token。

        Args:
            ego_utm_x/y/z: ego UTM 坐标。
            ego_yaw:       ego 朝向 (rad)。
            route_lane_keys:  在 route 上的 lane 三元组集合 (road_pkid, section_id, lane_id)。
            route_link_keys:  在 route 上的 lane_link 六元组集合
                              (from_road_pkid, from_section_id, from_lane_id,
                               to_road_pkid, to_section_id, to_lane_id)。

        Returns:
            dict 包含 PlanVOCA 字段，shape 见 docs/planvoca_style_structured_map_token_design.md §8.2。
            坐标系：ego-local，与 his_trajectory / fur_trajectory 对齐。
        """
        lon, lat = self._utm_to_wgs84.transform(ego_utm_x, ego_utm_y)
        lane_rows = self._query_lanes(lon, lat)
        link_rows = self._query_lane_links(lon, lat)
        rows = lane_rows + link_rows

        R_T = _build_rotation_matrix_2d(ego_yaw)
        ego_xy = np.array([ego_utm_x, ego_utm_y], dtype=np.float64)

        route_lane_set = set(tuple(int(v) for v in k) for k in (route_lane_keys or []))
        route_link_set = set(tuple(int(v) for v in k) for k in (route_link_keys or []))

        P = self.sample_points
        # 先无上限收集，最后按距离截断到 max_lanes
        lane_records: List[Dict[str, Any]] = []
        for row in rows:
            poly_global = self._wkt_line_to_utm(row["wkt"])
            if poly_global is None:
                continue
            # local = R^T @ (p - ego)，与 parseRosBag.transform_global_to_local 一致
            poly_local = np.einsum("ij,kj->ki", R_T, poly_global - ego_xy)

            centerline = _resample_polyline(poly_local, P + 1)
            if centerline is None:
                continue
            # 距离 ego 的最近点
            min_dist = float(np.min(np.linalg.norm(centerline, axis=1)))
            if min_dist > self.radius:
                continue

            point_position = np.zeros((3, P, 2), dtype=np.float32)
            point_vector = np.zeros((3, P, 2), dtype=np.float32)
            point_orientation = np.zeros((3, P), dtype=np.float32)
            valid_pts = np.zeros((P,), dtype=bool)

            seg_diff = centerline[1:] - centerline[:-1]  # [P, 2]
            seg_lens = np.linalg.norm(seg_diff, axis=1)
            valid_pts = seg_lens > 1e-6
            point_position[0] = centerline[:-1].astype(np.float32)
            point_vector[0] = seg_diff.astype(np.float32)
            point_orientation[0] = np.arctan2(seg_diff[:, 1], seg_diff[:, 0]).astype(np.float32)
            # 第一版只用 centerline，左右边界保持 0；保留维度以兼容 PlanVOCA 字段。

            half = P // 2
            polygon_center = np.array(
                [centerline[half, 0], centerline[half, 1], point_orientation[0, half]],
                dtype=np.float32,
            )

            poly_type = int(row["type"])
            if poly_type == POLYGON_TYPE_LANE:
                lane_key = (int(row["road_pkid"]), int(row["section_id"]), int(row["lane_id"]))
                on_route = lane_key in route_lane_set
                speed_limit_kmh = row.get("speed_limit", 0.0) or 0.0
                if speed_limit_kmh <= 0.0:
                    speed_limit_kmh = DEFAULT_SPEED_LIMIT_KMH
                speed_limit_mps = speed_limit_kmh / 3.6
                has_speed = True
            else:
                link_key = (
                    int(row["from_road_pkid"]), int(row["from_section_id"]), int(row["from_lane_id"]),
                    int(row["to_road_pkid"]), int(row["to_section_id"]), int(row["to_lane_id"]),
                )
                on_route = link_key in route_link_set
                speed_limit_mps = DEFAULT_SPEED_LIMIT_MPS
                has_speed = False

            lane_records.append({
                "min_dist": min_dist,
                "point_position": point_position,
                "point_vector": point_vector,
                "point_orientation": point_orientation,
                "valid_mask_pts": valid_pts.astype(np.float32),
                "polygon_center": polygon_center,
                "polygon_type": poly_type,
                "polygon_on_route": int(bool(on_route)),
                "polygon_tl_status": TL_UNKNOWN,
                "polygon_has_speed_limit": bool(has_speed),
                "polygon_speed_limit": float(speed_limit_mps),
            })

        # 截断到 max_lanes，按到 ego 的距离从近到远排序
        lane_records.sort(key=lambda r: r["min_dist"])
        lane_records = lane_records[: self.max_lanes]
        M = len(lane_records)

        out_point_position = np.zeros((M, 3, P, 2), dtype=np.float32)
        out_point_vector = np.zeros((M, 3, P, 2), dtype=np.float32)
        out_point_orientation = np.zeros((M, 3, P), dtype=np.float32)
        out_valid_mask_pts = np.zeros((M, P), dtype=np.float32)
        out_polygon_center = np.zeros((M, 3), dtype=np.float32)
        out_polygon_type = np.zeros((M,), dtype=np.int8)
        out_polygon_on_route = np.zeros((M,), dtype=np.int8)
        out_polygon_tl_status = np.full((M,), TL_UNKNOWN, dtype=np.int8)
        out_polygon_has_speed_limit = np.zeros((M,), dtype=bool)
        out_polygon_speed_limit = np.zeros((M,), dtype=np.float32)

        for i, r in enumerate(lane_records):
            out_point_position[i] = r["point_position"]
            out_point_vector[i] = r["point_vector"]
            out_point_orientation[i] = r["point_orientation"]
            out_valid_mask_pts[i] = r["valid_mask_pts"]
            out_polygon_center[i] = r["polygon_center"]
            out_polygon_type[i] = r["polygon_type"]
            out_polygon_on_route[i] = r["polygon_on_route"]
            out_polygon_tl_status[i] = r["polygon_tl_status"]
            out_polygon_has_speed_limit[i] = r["polygon_has_speed_limit"]
            out_polygon_speed_limit[i] = r["polygon_speed_limit"]

        return {
            "point_position": out_point_position,
            "point_vector": out_point_vector,
            "point_orientation": out_point_orientation,
            "polygon_center": out_polygon_center,
            "polygon_type": out_polygon_type,
            "polygon_on_route": out_polygon_on_route,
            "polygon_tl_status": out_polygon_tl_status,
            "polygon_has_speed_limit": out_polygon_has_speed_limit,
            "polygon_speed_limit": out_polygon_speed_limit,
            "point_valid_mask": out_valid_mask_pts,
            "valid_mask": (out_valid_mask_pts.sum(-1) > 0).astype(np.float32),
            "num_lanes": np.int32(M),
        }
