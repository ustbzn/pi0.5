#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pyproj
except ImportError:
    pyproj = None

try:
    from shapely import wkt as shapely_wkt
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import linemerge
except ImportError:
    shapely_wkt = None
    LineString = None
    Point = None
    Polygon = None
    linemerge = None


def _require_dependencies() -> None:
    if pyproj is None:
        raise ImportError(
            "lane frenet 计算需要 `pyproj`，请先执行 `python -m pip install pyproj`。"
        )
    if shapely_wkt is None or LineString is None or Point is None or Polygon is None:
        raise ImportError(
            "lane frenet 计算需要 `shapely`，请先执行 `python -m pip install shapely`。"
        )


def _load_spatialite_extension(conn: sqlite3.Connection) -> None:
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
        "或设置环境变量 `SPATIALITE_EXTENSION`。尝试结果：{}".format("; ".join(errors))
    )


def _compute_heading(linestring: LineString, arc_length: float, delta: float = 0.5) -> float:
    line_length = float(linestring.length)
    if line_length <= 0.0:
        return 0.0

    state1 = linestring.interpolate(arc_length)
    arc2 = min(line_length, arc_length + delta)
    state2 = linestring.interpolate(arc2)

    if state1.equals(state2):
        arc2 = max(0.0, arc_length - delta)
        state2 = linestring.interpolate(arc2)
        if state1.equals(state2):
            return 0.0
        return math.atan2(state1.y - state2.y, state1.x - state2.x)

    return math.atan2(state2.y - state1.y, state2.x - state1.x)


def _normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _angle_diff_abs(angle1: float, angle2: float) -> float:
    return abs(_normalize_angle(angle1 - angle2))


def _bounds_contains(bounds: Tuple[float, float, float, float], x: float, y: float, padding: float = 0.0) -> bool:
    min_x, min_y, max_x, max_y = bounds
    return (min_x - padding) <= x <= (max_x + padding) and (min_y - padding) <= y <= (max_y + padding)


def _half_width_from_boundaries(left_boundary: LineString, right_boundary: LineString, use_end: bool) -> float:
    if left_boundary is None or right_boundary is None:
        return 1.5

    if use_end:
        left_point = Point(left_boundary.coords[-1])
        right_point = Point(right_boundary.coords[-1])
    else:
        left_point = Point(left_boundary.coords[0])
        right_point = Point(right_boundary.coords[0])

    width = left_point.distance(right_point) / 2.0
    return float(width) if width > 0 else 1.5


def _offset_point(point: Point, heading: float, distance: float) -> Point:
    return Point(
        point.x + distance * math.cos(heading),
        point.y + distance * math.sin(heading),
    )


def _compute_direction(points: Sequence[Tuple[float, float]]) -> float:
    if len(points) < 2:
        return 0.0
    start = points[0]
    end = points[1]
    return math.atan2(end[1] - start[1], end[0] - start[0])


def _normalize_line_from_wkt(
    geom_wkt: Optional[str],
    transformer: pyproj.Transformer,
) -> Optional[LineString]:
    if not geom_wkt:
        return None

    geom = shapely_wkt.loads(geom_wkt)
    if geom.is_empty:
        return None

    if geom.geom_type == "LineString":
        line = geom
    elif geom.geom_type == "MultiLineString":
        merged = linemerge(geom)
        if merged.geom_type == "LineString":
            line = merged
        elif merged.geom_type == "MultiLineString":
            line = max(merged.geoms, key=lambda item: item.length)
        else:
            return None
    else:
        return None

    coords = np.asarray(line.coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] < 2 or coords.shape[1] < 2:
        return None

    xs, ys = transformer.transform(coords[:, 0], coords[:, 1])
    utm_coords = np.stack([np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)], axis=1)
    return LineString(utm_coords)


def _make_lane_polygon(
    centerline: LineString,
    left_boundary: Optional[LineString],
    right_boundary: Optional[LineString],
) -> Polygon:
    if left_boundary is not None and right_boundary is not None:
        left_coords = list(left_boundary.coords)
        right_coords = list(right_boundary.coords)
        right_coords.reverse()
        polygon = Polygon(left_coords + right_coords)
        if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
            return polygon
        buffered = polygon.buffer(0)
        if buffered.geom_type == "Polygon" and not buffered.is_empty and buffered.area > 0:
            return buffered

    fallback_width = 1.75
    if left_boundary is not None and right_boundary is not None:
        start_width = _half_width_from_boundaries(left_boundary, right_boundary, use_end=False)
        end_width = _half_width_from_boundaries(left_boundary, right_boundary, use_end=True)
        fallback_width = max(start_width, end_width, 1.5)

    return centerline.buffer(fallback_width, cap_style=2, join_style=2)


def _make_lane_link_polygon(
    centerline: LineString,
    from_lane: Optional["MapLaneObject"],
    to_lane: Optional["MapLaneObject"],
) -> Polygon:
    start_width = 1.5
    end_width = 1.5
    start_left = None
    start_right = None
    end_left = None
    end_right = None

    if from_lane and from_lane.left_boundary is not None and from_lane.right_boundary is not None:
        start_width = _half_width_from_boundaries(from_lane.left_boundary, from_lane.right_boundary, use_end=True)
        start_left = Point(from_lane.left_boundary.coords[-1])
        start_right = Point(from_lane.right_boundary.coords[-1])

    if to_lane and to_lane.left_boundary is not None and to_lane.right_boundary is not None:
        end_width = _half_width_from_boundaries(to_lane.left_boundary, to_lane.right_boundary, use_end=False)
        end_left = Point(to_lane.left_boundary.coords[0])
        end_right = Point(to_lane.right_boundary.coords[0])

    baseline_points = list(centerline.coords)
    if len(baseline_points) < 2:
        return centerline.buffer(max(start_width, end_width, 1.5), cap_style=2, join_style=2)

    left_boundary_points: List[Point] = []
    right_boundary_points: List[Point] = []

    if start_left is not None and start_right is not None:
        left_boundary_points.append(start_left)
        right_boundary_points.append(start_right)

    point_count = max(1, len(baseline_points) - 1)
    for idx, coord in enumerate(baseline_points):
        progress = float(idx) / float(point_count)
        current_width = start_width + (end_width - start_width) * progress

        if idx < len(baseline_points) - 1:
            direction = _compute_direction([baseline_points[idx], baseline_points[idx + 1]])
        else:
            direction = _compute_direction([baseline_points[idx - 1], baseline_points[idx]])

        current_point = Point(coord)
        left_boundary_points.append(_offset_point(current_point, direction + math.pi / 2.0, current_width))
        right_boundary_points.append(_offset_point(current_point, direction - math.pi / 2.0, current_width))

    if end_left is not None and end_right is not None:
        left_boundary_points.append(end_left)
        right_boundary_points.append(end_right)

    polygon_coords = [(pt.x, pt.y) for pt in left_boundary_points]
    polygon_coords.extend((pt.x, pt.y) for pt in reversed(right_boundary_points))
    polygon = Polygon(polygon_coords)
    if polygon.is_valid and not polygon.is_empty and polygon.area > 0:
        return polygon

    buffered = polygon.buffer(0)
    if buffered.geom_type == "Polygon" and not buffered.is_empty and buffered.area > 0:
        return buffered

    return centerline.buffer(max(start_width, end_width, 1.5), cap_style=2, join_style=2)


def _make_lane_link_centerline(
    from_lane: Optional["MapLaneObject"],
    to_lane: Optional["MapLaneObject"],
) -> Optional[LineString]:
    if from_lane is None or to_lane is None:
        return None

    start_point = Point(from_lane.centerline.coords[-1])
    end_point = Point(to_lane.centerline.coords[0])
    if not start_point.equals(end_point):
        return LineString([start_point.coords[0], end_point.coords[0]])

    sample_delta = 0.5
    from_probe = from_lane.centerline.interpolate(
        max(0.0, float(from_lane.centerline.length) - sample_delta)
    )
    to_probe = to_lane.centerline.interpolate(
        min(float(to_lane.centerline.length), sample_delta)
    )

    coords: List[Tuple[float, float]] = []
    for point in [from_probe, start_point, to_probe]:
        coord = (float(point.x), float(point.y))
        if not coords or coord != coords[-1]:
            coords.append(coord)

    if len(coords) >= 2:
        return LineString(coords)

    heading = _compute_heading(to_lane.centerline, 0.0)
    if abs(heading) < 1e-6:
        heading = _compute_heading(
            from_lane.centerline, float(from_lane.centerline.length)
        )
    epsilon = 0.1
    return LineString(
        [
            (
                float(start_point.x - epsilon * math.cos(heading)),
                float(start_point.y - epsilon * math.sin(heading)),
            ),
            (
                float(start_point.x + epsilon * math.cos(heading)),
                float(start_point.y + epsilon * math.sin(heading)),
            ),
        ]
    )


@dataclass
class MapLaneObject:
    key: str
    matched_lane_id: str
    object_type: str
    centerline: LineString
    polygon: Polygon
    outgoing_ids: List[str] = field(default_factory=list)
    left_boundary: Optional[LineString] = None
    right_boundary: Optional[LineString] = None
    road_pkid: Optional[int] = None
    section_id: Optional[int] = None
    lane_id: Optional[int] = None
    from_lane_key: str = ""
    to_lane_key: str = ""
    stable_lane_id: str = ""
    stable_s_offset: float = 0.0

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        return self.polygon.bounds


@dataclass(frozen=True)
class LaneMatchResult:
    matched_lane_id: str
    s: float
    d: float
    stable_lane_id: str = ""
    stable_s: float = field(default_factory=lambda: float("nan"))
    stable_d: float = field(default_factory=lambda: float("nan"))
    lane_relation: str = ""

    def as_raw_tuple(self) -> Tuple[str, float, float]:
        return self.matched_lane_id, self.s, self.d

    def as_stable_tuple(self) -> Tuple[str, float, float, str]:
        return self.stable_lane_id, self.stable_s, self.stable_d, self.lane_relation


class LaneFrenetMatcher:
    """
    根据 sampled localization 列表，输出逐点的 matched lane/lane_link 与 Frenet (s, d)。
    """

    def __init__(
        self,
        db_path: str,
        utm_zone: int = 50,
        utm_south: bool = False,
        query_margin_m: float = 120.0,
        max_match_distance_m: float = 8.0,
    ) -> None:
        _require_dependencies()
        if not os.path.isfile(db_path):
            raise IOError("地图 sqlite 文件不存在：{}".format(db_path))

        self.db_path = db_path
        self.utm_zone = utm_zone
        self.utm_south = utm_south
        self.query_margin_m = float(query_margin_m)
        self.max_match_distance_m = float(max_match_distance_m)
        self.stable_endpoint_distance_m = 2.0
        self.stable_heading_diff_rad = math.radians(15.0)
        self.lane_change_min_previous_d_m = 0.9

        self.connection = sqlite3.connect(self.db_path)
        self.connection.enable_load_extension(True)
        _load_spatialite_extension(self.connection)
        self._init_transformers()

        self.objects_by_id: Dict[str, MapLaneObject] = {}
        self.lane_objects: List[MapLaneObject] = []
        self.lane_link_objects: List[MapLaneObject] = []
        self.stable_lane_id_by_lane_key: Dict[str, str] = {}

    def close(self) -> None:
        if self.connection is not None:
            self.connection.close()
            self.connection = None

    def _init_transformers(self) -> None:
        proj4 = "+proj=utm +zone={} +datum=WGS84 +units=m +no_defs".format(self.utm_zone)
        if self.utm_south:
            proj4 += " +south"
        utm_crs = pyproj.CRS.from_proj4(proj4)
        self._wgs84_to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        self._utm_to_wgs84 = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    def _query_all(self, sql: str, params: Sequence[object]) -> List[sqlite3.Row]:
        cursor = self.connection.cursor()
        cursor.execute(sql, params)
        return cursor.fetchall()

    def _chunked_in_query(self, table_name: str, pk_field: str, ids: Sequence[int]) -> Dict[int, str]:
        if not ids:
            return {}

        result: Dict[int, str] = {}
        cursor = self.connection.cursor()
        unique_ids = sorted(set(int(item) for item in ids))
        chunk_size = 500
        for start in range(0, len(unique_ids), chunk_size):
            chunk = unique_ids[start : start + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            sql = "SELECT {}, AsText(geom) FROM {} WHERE {} IN ({})".format(
                pk_field, table_name, pk_field, placeholders
            )
            cursor.execute(sql, chunk)
            for pkid, geom_wkt in cursor.fetchall():
                result[int(pkid)] = geom_wkt
        return result

    def _utm_bounds_to_wgs84(self, utm_points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
        xs = [point[0] for point in utm_points]
        ys = [point[1] for point in utm_points]
        min_x = min(xs) - self.query_margin_m
        max_x = max(xs) + self.query_margin_m
        min_y = min(ys) - self.query_margin_m
        max_y = max(ys) + self.query_margin_m

        corners = [
            (min_x, min_y),
            (min_x, max_y),
            (max_x, min_y),
            (max_x, max_y),
        ]
        lon_lat = [self._utm_to_wgs84.transform(x, y) for x, y in corners]
        lons = [item[0] for item in lon_lat]
        lats = [item[1] for item in lon_lat]
        return min(lons), min(lats), max(lons), max(lats)

    def load_map_for_utm_points(self, utm_points: Sequence[Tuple[float, float]]) -> None:
        if not utm_points:
            self.objects_by_id = {}
            self.lane_objects = []
            self.lane_link_objects = []
            self.stable_lane_id_by_lane_key = {}
            return

        lon_min, lat_min, lon_max, lat_max = self._utm_bounds_to_wgs84(utm_points)

        lane_rows = self._query_all(
            """
            SELECT road_pkid, section_id, lane_id, left_boundary, right_boundary, AsText(geom)
            FROM TX_LANE
            WHERE ST_Intersects(geom, BuildMBR(?, ?, ?, ?, 4326))
            """,
            (lon_min, lat_min, lon_max, lat_max),
        )

        road_ids = sorted({int(row[0]) for row in lane_rows})

        lane_link_rows = self._query_all(
            """
            SELECT pkid, from_road_pkid, from_section_id, from_lane_id,
                   to_road_pkid, to_section_id, to_lane_id, AsText(geom)
            FROM TX_LANE_LINK
            WHERE ST_Intersects(geom, BuildMBR(?, ?, ?, ?, 4326))
            """,
            (lon_min, lat_min, lon_max, lat_max),
        )

        if road_ids:
            placeholders = ",".join("?" for _ in road_ids)
            link_by_road_rows = self._query_all(
                """
                SELECT pkid, from_road_pkid, from_section_id, from_lane_id,
                       to_road_pkid, to_section_id, to_lane_id, AsText(geom)
                FROM TX_LANE_LINK
                WHERE from_road_pkid IN ({}) OR to_road_pkid IN ({})
                """.format(placeholders, placeholders),
                tuple(road_ids) + tuple(road_ids),
            )
        else:
            link_by_road_rows = []

        lane_link_map: Dict[int, Tuple] = {}
        for row in list(lane_link_rows) + list(link_by_road_rows):
            lane_link_map[int(row[0])] = row

        boundary_ids: List[int] = []
        for row in lane_rows:
            if row[3] is not None:
                boundary_ids.append(int(row[3]))
            if row[4] is not None:
                boundary_ids.append(int(row[4]))

        boundary_wkts = self._chunked_in_query("TX_LANE_BOUNDARY", "pkid", boundary_ids)

        lane_objects_by_key: Dict[str, MapLaneObject] = {}
        lane_outgoing_map: Dict[str, List[str]] = {}
        lane_outgoing_lane_map: Dict[str, List[str]] = {}
        lane_incoming_lane_map: Dict[str, List[str]] = {}
        lane_link_pending: List[Tuple] = []

        for row in lane_rows:
            road_pkid, section_id, lane_id, left_boundary_id, right_boundary_id, geom_wkt = row
            centerline = _normalize_line_from_wkt(geom_wkt, self._wgs84_to_utm)
            if centerline is None or centerline.length <= 0.0:
                continue

            left_boundary = _normalize_line_from_wkt(boundary_wkts.get(int(left_boundary_id)), self._wgs84_to_utm)
            right_boundary = _normalize_line_from_wkt(boundary_wkts.get(int(right_boundary_id)), self._wgs84_to_utm)
            polygon = _make_lane_polygon(centerline, left_boundary, right_boundary)

            lane_key = "{}_{}_{}".format(int(road_pkid), int(section_id), int(lane_id))
            lane_obj = MapLaneObject(
                key=lane_key,
                matched_lane_id=lane_key,
                object_type="lane",
                centerline=centerline,
                polygon=polygon,
                outgoing_ids=[],
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                road_pkid=int(road_pkid),
                section_id=int(section_id),
                lane_id=int(lane_id),
                stable_lane_id=lane_key,
            )
            lane_objects_by_key[lane_key] = lane_obj
            lane_outgoing_map[lane_key] = []
            lane_outgoing_lane_map.setdefault(lane_key, [])
            lane_incoming_lane_map.setdefault(lane_key, [])

        for row in lane_link_map.values():
            lane_link_pending.append(row)
            _, from_road_pkid, from_section_id, from_lane_id, to_road_pkid, to_section_id, to_lane_id, _ = row
            from_lane_key = "{}_{}_{}".format(int(from_road_pkid), int(from_section_id), int(from_lane_id))
            to_lane_key = "{}_{}_{}".format(int(to_road_pkid), int(to_section_id), int(to_lane_id))
            lane_link_key = "{}_{}_{}_{}_{}_{}".format(
                int(from_road_pkid),
                int(from_section_id),
                int(from_lane_id),
                int(to_road_pkid),
                int(to_section_id),
                int(to_lane_id),
            )
            lane_outgoing_map.setdefault(from_lane_key, []).append(lane_link_key)
            lane_outgoing_lane_map.setdefault(from_lane_key, []).append(to_lane_key)
            lane_incoming_lane_map.setdefault(to_lane_key, []).append(from_lane_key)

        lane_link_objects_by_key: Dict[str, MapLaneObject] = {}
        for row in lane_link_pending:
            _, from_road_pkid, from_section_id, from_lane_id, to_road_pkid, to_section_id, to_lane_id, geom_wkt = row
            from_lane_key = "{}_{}_{}".format(int(from_road_pkid), int(from_section_id), int(from_lane_id))
            to_lane_key = "{}_{}_{}".format(int(to_road_pkid), int(to_section_id), int(to_lane_id))
            lane_link_key = "{}_{}_{}_{}_{}_{}".format(
                int(from_road_pkid),
                int(from_section_id),
                int(from_lane_id),
                int(to_road_pkid),
                int(to_section_id),
                int(to_lane_id),
            )

            centerline = _normalize_line_from_wkt(geom_wkt, self._wgs84_to_utm)
            from_lane = lane_objects_by_key.get(from_lane_key)
            to_lane = lane_objects_by_key.get(to_lane_key)

            if centerline is None:
                centerline = _make_lane_link_centerline(from_lane, to_lane)
                if centerline is None:
                    continue

            if centerline.length <= 0.0:
                continue

            polygon = _make_lane_link_polygon(centerline, from_lane, to_lane)
            lane_link_objects_by_key[lane_link_key] = MapLaneObject(
                key=lane_link_key,
                matched_lane_id=lane_link_key,
                object_type="lane_link",
                centerline=centerline,
                polygon=polygon,
                outgoing_ids=[to_lane_key],
                from_lane_key=from_lane_key,
                to_lane_key=to_lane_key,
            )

        for lane_key, lane_obj in lane_objects_by_key.items():
            lane_obj.outgoing_ids = [
                object_key
                for object_key in lane_outgoing_map.get(lane_key, [])
                if object_key in lane_link_objects_by_key
            ]

        self._assign_stable_lane_ids(
            lane_objects_by_key=lane_objects_by_key,
            lane_link_objects_by_key=lane_link_objects_by_key,
            lane_outgoing_lane_map=lane_outgoing_lane_map,
            lane_incoming_lane_map=lane_incoming_lane_map,
        )

        self.objects_by_id = {}
        self.objects_by_id.update(lane_objects_by_key)
        self.objects_by_id.update(lane_link_objects_by_key)
        self.lane_objects = list(lane_objects_by_key.values())
        self.lane_link_objects = list(lane_link_objects_by_key.values())
        self.stable_lane_id_by_lane_key = {
            lane_key: lane_obj.stable_lane_id for lane_key, lane_obj in lane_objects_by_key.items()
        }

    @staticmethod
    def _find_parent(parents: Dict[str, str], key: str) -> str:
        parent = parents[key]
        if parent != key:
            parents[key] = LaneFrenetMatcher._find_parent(parents, parent)
        return parents[key]

    @staticmethod
    def _union_parent(parents: Dict[str, str], key1: str, key2: str) -> None:
        root1 = LaneFrenetMatcher._find_parent(parents, key1)
        root2 = LaneFrenetMatcher._find_parent(parents, key2)
        if root1 == root2:
            return
        if root1 < root2:
            parents[root2] = root1
        else:
            parents[root1] = root2

    def _should_merge_stable_lane(
        self,
        from_lane: MapLaneObject,
        to_lane: MapLaneObject,
        lane_outgoing_lane_map: Dict[str, List[str]],
        lane_incoming_lane_map: Dict[str, List[str]],
    ) -> bool:
        if from_lane.key == to_lane.key:
            return False
        if from_lane.lane_id is None or to_lane.lane_id is None:
            return False
        if int(from_lane.lane_id) != int(to_lane.lane_id):
            return False

        from_outgoing = set(lane_outgoing_lane_map.get(from_lane.key, []))
        to_incoming = set(lane_incoming_lane_map.get(to_lane.key, []))
        if from_outgoing != {to_lane.key}:
            return False
        if to_incoming != {from_lane.key}:
            return False

        from_end = Point(from_lane.centerline.coords[-1])
        to_start = Point(to_lane.centerline.coords[0])
        if from_end.distance(to_start) > self.stable_endpoint_distance_m:
            return False

        from_heading = _compute_heading(from_lane.centerline, float(from_lane.centerline.length))
        to_heading = _compute_heading(to_lane.centerline, 0.0)
        if _angle_diff_abs(from_heading, to_heading) > self.stable_heading_diff_rad:
            return False

        return True

    def _assign_stable_lane_ids(
        self,
        lane_objects_by_key: Dict[str, MapLaneObject],
        lane_link_objects_by_key: Dict[str, MapLaneObject],
        lane_outgoing_lane_map: Dict[str, List[str]],
        lane_incoming_lane_map: Dict[str, List[str]],
    ) -> None:
        parents = {lane_key: lane_key for lane_key in lane_objects_by_key}
        merged_lane_pairs: List[Tuple[str, str]] = []

        for lane_link_obj in lane_link_objects_by_key.values():
            from_lane = lane_objects_by_key.get(lane_link_obj.from_lane_key)
            to_lane = lane_objects_by_key.get(lane_link_obj.to_lane_key)
            if from_lane is None or to_lane is None:
                continue
            if not self._should_merge_stable_lane(
                from_lane=from_lane,
                to_lane=to_lane,
                lane_outgoing_lane_map=lane_outgoing_lane_map,
                lane_incoming_lane_map=lane_incoming_lane_map,
            ):
                continue
            self._union_parent(parents, from_lane.key, to_lane.key)
            merged_lane_pairs.append((from_lane.key, to_lane.key))

        groups: Dict[str, List[str]] = {}
        for lane_key in lane_objects_by_key:
            root_key = self._find_parent(parents, lane_key)
            groups.setdefault(root_key, []).append(lane_key)

        merged_successor_by_lane: Dict[str, str] = {}
        merged_predecessor_by_lane: Dict[str, str] = {}
        for from_lane_key, to_lane_key in merged_lane_pairs:
            merged_successor_by_lane[from_lane_key] = to_lane_key
            merged_predecessor_by_lane[to_lane_key] = from_lane_key

        for lane_keys in groups.values():
            group_key_set = set(lane_keys)
            start_lane_keys = sorted(
                lane_key
                for lane_key in lane_keys
                if merged_predecessor_by_lane.get(lane_key) not in group_key_set
            )

            if start_lane_keys:
                current_lane_key = start_lane_keys[0]
            else:
                current_lane_key = min(lane_keys)

            ordered_lane_keys: List[str] = []
            visited_lane_keys = set()
            while current_lane_key in group_key_set and current_lane_key not in visited_lane_keys:
                ordered_lane_keys.append(current_lane_key)
                visited_lane_keys.add(current_lane_key)
                current_lane_key = merged_successor_by_lane.get(current_lane_key, "")

            for lane_key in sorted(group_key_set - visited_lane_keys):
                ordered_lane_keys.append(lane_key)

            stable_lane_id = ordered_lane_keys[0]
            for lane_key in lane_keys:
                lane_objects_by_key[lane_key].stable_lane_id = stable_lane_id

            stable_s_offset = 0.0
            for lane_key in ordered_lane_keys:
                lane_obj = lane_objects_by_key[lane_key]
                lane_obj.stable_s_offset = stable_s_offset
                stable_s_offset += float(lane_obj.centerline.length)

        for lane_link_obj in lane_link_objects_by_key.values():
            from_lane = lane_objects_by_key.get(lane_link_obj.from_lane_key)
            to_lane = lane_objects_by_key.get(lane_link_obj.to_lane_key)
            if from_lane is not None and to_lane is not None:
                if from_lane.stable_lane_id == to_lane.stable_lane_id:
                    lane_link_obj.stable_lane_id = from_lane.stable_lane_id
                else:
                    lane_link_obj.stable_lane_id = to_lane.stable_lane_id or from_lane.stable_lane_id
            elif to_lane is not None:
                lane_link_obj.stable_lane_id = to_lane.stable_lane_id
            elif from_lane is not None:
                lane_link_obj.stable_lane_id = from_lane.stable_lane_id

    def _candidate_containing(self, point: Point, objects: Iterable[MapLaneObject]) -> List[MapLaneObject]:
        point_x = point.x
        point_y = point.y
        candidates = []
        for obj in objects:
            if not _bounds_contains(obj.bounds, point_x, point_y):
                continue
            if obj.polygon.distance(point) <= 1e-6:
                candidates.append(obj)
        return candidates

    def _nearest_object(
        self,
        point: Point,
        objects: Iterable[MapLaneObject],
        max_distance_m: Optional[float] = None,
    ) -> Optional[MapLaneObject]:
        point_x = point.x
        point_y = point.y
        best_obj = None
        best_dist = float("inf")
        padding = max_distance_m if max_distance_m is not None else self.max_match_distance_m
        for obj in objects:
            if not _bounds_contains(obj.bounds, point_x, point_y, padding=padding):
                continue
            distance = obj.centerline.distance(point)
            if distance < best_dist:
                best_dist = distance
                best_obj = obj

        if best_obj is None:
            return None
        if max_distance_m is not None and best_dist > max_distance_m:
            return None
        if best_dist > self.max_match_distance_m:
            return None
        return best_obj

    def _choose_best(self, point: Point, objects: Sequence[MapLaneObject]) -> Optional[MapLaneObject]:
        if not objects:
            return None
        return min(objects, key=lambda obj: (obj.centerline.distance(point), 0 if obj.object_type == "lane" else 1))

    def _match_with_previous(self, point: Point, previous_obj: Optional[MapLaneObject]) -> Optional[MapLaneObject]:
        if previous_obj is None:
            return None

        if previous_obj.polygon.distance(point) <= 1e-6:
            return previous_obj

        outgoing_candidates = [
            self.objects_by_id[obj_id]
            for obj_id in previous_obj.outgoing_ids
            if obj_id in self.objects_by_id
        ]

        downstream_lane_candidates = []
        for obj in outgoing_candidates:
            if obj.object_type != "lane_link" or not obj.to_lane_key:
                continue
            to_lane = self.objects_by_id.get(obj.to_lane_key)
            if to_lane is not None:
                downstream_lane_candidates.append(to_lane)

        transition_candidates = outgoing_candidates + downstream_lane_candidates
        containing = self._candidate_containing(point, transition_candidates)
        if containing:
            return self._choose_best(point, containing)

        fallback_candidates = [previous_obj] + transition_candidates
        return self._nearest_object(point, fallback_candidates, max_distance_m=4.0)

    def _match_globally(self, point: Point) -> Optional[MapLaneObject]:
        lane_candidates = self._candidate_containing(point, self.lane_objects)
        if lane_candidates:
            return self._choose_best(point, lane_candidates)

        lane_link_candidates = self._candidate_containing(point, self.lane_link_objects)
        if lane_link_candidates:
            return self._choose_best(point, lane_link_candidates)

        nearest_lane = self._nearest_object(point, self.lane_objects, max_distance_m=6.0)
        nearest_lane_link = self._nearest_object(point, self.lane_link_objects, max_distance_m=6.0)
        if nearest_lane is None:
            return nearest_lane_link
        if nearest_lane_link is None:
            return nearest_lane
        return self._choose_best(point, [nearest_lane, nearest_lane_link])

    def _compute_frenet(self, obj: MapLaneObject, point: Point) -> Tuple[str, float, float]:
        s = float(obj.centerline.project(point))
        ref_point = obj.centerline.interpolate(s)
        heading = _compute_heading(obj.centerline, s)
        dx = point.x - ref_point.x
        dy = point.y - ref_point.y
        d = -math.sin(heading) * dx + math.cos(heading) * dy
        return obj.matched_lane_id, s, float(d)

    def _compute_stable_frenet(
        self,
        stable_lane_obj: MapLaneObject,
        point: Point,
    ) -> Tuple[str, float, float]:
        _, local_s, local_d = self._compute_frenet(stable_lane_obj, point)
        stable_s = float(stable_lane_obj.stable_s_offset + local_s)
        return stable_lane_obj.stable_lane_id, stable_s, local_d

    def _resolve_stable_lane_object(
        self,
        matched_obj: MapLaneObject,
        point: Point,
    ) -> Optional[MapLaneObject]:
        if matched_obj.object_type == "lane":
            return matched_obj

        from_lane = self.objects_by_id.get(matched_obj.from_lane_key)
        to_lane = self.objects_by_id.get(matched_obj.to_lane_key)
        if from_lane is None:
            return to_lane
        if to_lane is None:
            return from_lane

        if from_lane.stable_lane_id == to_lane.stable_lane_id:
            return min(
                [from_lane, to_lane],
                key=lambda lane_obj: lane_obj.centerline.distance(point),
            )

        return to_lane

    def _compute_lane_relation(
        self,
        point: Point,
        stable_lane_obj: Optional[MapLaneObject],
        previous_stable_lane_obj: Optional[MapLaneObject],
    ) -> str:
        if stable_lane_obj is None or not stable_lane_obj.stable_lane_id:
            return ""
        if previous_stable_lane_obj is None or not previous_stable_lane_obj.stable_lane_id:
            return "keep"
        if stable_lane_obj.stable_lane_id == previous_stable_lane_obj.stable_lane_id:
            return "keep"

        _, _, previous_d = self._compute_frenet(previous_stable_lane_obj, point)
        if not math.isfinite(previous_d):
            return "keep"
        if abs(previous_d) <= self.lane_change_min_previous_d_m:
            return "keep"
        return "toleft" if previous_d >= 0.0 else "toright"

    def _has_lane_link_to(self, from_obj: MapLaneObject, to_lane_key: str) -> bool:
        if from_obj.object_type == "lane_link":
            return from_obj.to_lane_key == to_lane_key

        for obj_id in from_obj.outgoing_ids:
            obj = self.objects_by_id.get(obj_id)
            if obj is not None and obj.object_type == "lane_link" and obj.to_lane_key == to_lane_key:
                return True
        return False

    def _is_observed_same_lane_continuation(
        self,
        previous_obj: Optional[MapLaneObject],
        previous_stable_lane_obj: Optional[MapLaneObject],
        stable_lane_obj: Optional[MapLaneObject],
    ) -> bool:
        if previous_obj is None or previous_stable_lane_obj is None or stable_lane_obj is None:
            return False
        if previous_stable_lane_obj.object_type != "lane" or stable_lane_obj.object_type != "lane":
            return False
        if previous_stable_lane_obj.stable_lane_id == stable_lane_obj.stable_lane_id:
            return False
        if previous_stable_lane_obj.lane_id is None or stable_lane_obj.lane_id is None:
            return False
        if int(previous_stable_lane_obj.lane_id) != int(stable_lane_obj.lane_id):
            return False
        if not self._has_lane_link_to(previous_obj, stable_lane_obj.key):
            return False

        previous_end = Point(previous_stable_lane_obj.centerline.coords[-1])
        current_start = Point(stable_lane_obj.centerline.coords[0])
        if previous_end.distance(current_start) > self.stable_endpoint_distance_m:
            return False

        previous_heading = _compute_heading(
            previous_stable_lane_obj.centerline,
            float(previous_stable_lane_obj.centerline.length),
        )
        current_heading = _compute_heading(stable_lane_obj.centerline, 0.0)
        return _angle_diff_abs(previous_heading, current_heading) <= self.stable_heading_diff_rad

    def match_utm_points_detailed(self, utm_points: Sequence[Tuple[float, float]]) -> List[LaneMatchResult]:
        self.load_map_for_utm_points(utm_points)
        results: List[LaneMatchResult] = []
        previous_obj = None
        previous_stable_lane_obj = None
        previous_dynamic_stable_lane_id = ""
        previous_dynamic_stable_delta = 0.0
        dynamic_stable_aliases: Dict[str, Tuple[str, float]] = {}

        for x, y in utm_points:
            point = Point(float(x), float(y))

            matched_obj = self._match_with_previous(point, previous_obj)
            if matched_obj is None:
                matched_obj = self._match_globally(point)

            if matched_obj is None:
                results.append(LaneMatchResult("", float("nan"), float("nan")))
                continue

            matched_lane_id, s, d = self._compute_frenet(matched_obj, point)
            stable_lane_obj = self._resolve_stable_lane_object(matched_obj, point)
            if stable_lane_obj is not None and stable_lane_obj.stable_lane_id:
                stable_lane_id, stable_s, stable_d = self._compute_stable_frenet(
                    stable_lane_obj, point
                )
                if (
                    stable_lane_id not in dynamic_stable_aliases
                    and previous_dynamic_stable_lane_id
                    and self._is_observed_same_lane_continuation(
                        previous_obj=previous_obj,
                        previous_stable_lane_obj=previous_stable_lane_obj,
                        stable_lane_obj=stable_lane_obj,
                    )
                ):
                    previous_group_end_s = (
                        previous_stable_lane_obj.stable_s_offset
                        + float(previous_stable_lane_obj.centerline.length)
                        + previous_dynamic_stable_delta
                    )
                    dynamic_stable_aliases[stable_lane_id] = (
                        previous_dynamic_stable_lane_id,
                        previous_group_end_s - stable_lane_obj.stable_s_offset,
                    )

                dynamic_stable_lane_id, dynamic_stable_delta = dynamic_stable_aliases.get(
                    stable_lane_id,
                    (stable_lane_id, 0.0),
                )
                stable_lane_id = dynamic_stable_lane_id
                stable_s = float(stable_s + dynamic_stable_delta)
                lane_relation = (
                    "keep"
                    if not previous_dynamic_stable_lane_id
                    or stable_lane_id == previous_dynamic_stable_lane_id
                    else self._compute_lane_relation(
                        point=point,
                        stable_lane_obj=stable_lane_obj,
                        previous_stable_lane_obj=previous_stable_lane_obj,
                    )
                )
            else:
                stable_lane_id = ""
                stable_s = float("nan")
                stable_d = float("nan")
                lane_relation = ""
                dynamic_stable_delta = 0.0

            results.append(
                LaneMatchResult(
                    matched_lane_id=matched_lane_id,
                    s=s,
                    d=d,
                    stable_lane_id=stable_lane_id,
                    stable_s=stable_s,
                    stable_d=stable_d,
                    lane_relation=lane_relation,
                )
            )
            previous_obj = matched_obj
            if stable_lane_obj is not None and stable_lane_obj.stable_lane_id:
                previous_stable_lane_obj = stable_lane_obj
                previous_dynamic_stable_lane_id = stable_lane_id
                previous_dynamic_stable_delta = dynamic_stable_delta

        return results

    def match_utm_points(self, utm_points: Sequence[Tuple[float, float]]) -> List[Tuple[str, float, float]]:
        return [item.as_raw_tuple() for item in self.match_utm_points_detailed(utm_points)]

    def match_utm_points_stable(
        self, utm_points: Sequence[Tuple[float, float]]
    ) -> List[Tuple[str, float, float, str]]:
        return [item.as_stable_tuple() for item in self.match_utm_points_detailed(utm_points)]
