"""HDMap vector token 加载工具。

从 pkl dict 读取 PlanVOCA-style 字段，按 ego 距离排序、截断/pad 到固定 max_map_tokens，
输出 torch tensor。与 src/model/DrivePi0/map_encoder.py 配套使用。
"""

from typing import Any, Dict, Optional

import numpy as np
import torch


# pkl 顶层字段命名（与 mogo-dataset-tools/parseRosBag 写入端对齐）
HDMAP_VECTOR_KEYS = (
    "hdmap_vector_point_position",
    "hdmap_vector_point_vector",
    "hdmap_vector_point_orientation",
    "hdmap_vector_polygon_center",
    "hdmap_vector_polygon_type",
    "hdmap_vector_polygon_on_route",
    "hdmap_vector_polygon_tl_status",
    "hdmap_vector_polygon_has_speed_limit",
    "hdmap_vector_polygon_speed_limit",
    "hdmap_vector_point_valid_mask",
    "hdmap_vector_valid_mask",
)


def has_hdmap_vector(all_data: Dict[str, Any]) -> bool:
    return all(k in all_data for k in HDMAP_VECTOR_KEYS)


def _zero_outputs(max_map_tokens: int, sample_points: int) -> Dict[str, torch.Tensor]:
    M = int(max_map_tokens)
    P = int(sample_points)
    return {
        "map_point_position": torch.zeros((M, 3, P, 2), dtype=torch.float32),
        "map_point_vector": torch.zeros((M, 3, P, 2), dtype=torch.float32),
        "map_point_orientation": torch.zeros((M, 3, P), dtype=torch.float32),
        "map_polygon_center": torch.zeros((M, 3), dtype=torch.float32),
        "map_polygon_type": torch.zeros((M,), dtype=torch.long),
        "map_polygon_on_route": torch.zeros((M,), dtype=torch.long),
        "map_polygon_tl_status": torch.full((M,), 3, dtype=torch.long),  # UNKNOWN=3
        "map_polygon_has_speed_limit": torch.zeros((M,), dtype=torch.bool),
        "map_polygon_speed_limit": torch.zeros((M,), dtype=torch.float32),
        "map_point_valid_mask": torch.zeros((M, P), dtype=torch.float32),
        "map_valid_mask": torch.zeros((M,), dtype=torch.float32),
    }


def load_hdmap_vector_tokens(
    all_data: Dict[str, Any],
    max_map_tokens: int = 128,
    sample_points: int = 20,
    strict: bool = True,
) -> Dict[str, torch.Tensor]:
    """从 pkl dict 读取 hdmap_vector_* 字段，按距离排序后截断/pad。

    Args:
        all_data:        pkl 解出的字典。
        max_map_tokens:  目标 token 上限 M。
        sample_points:   每个 lane 的 centerline 采样段数 P。
        strict:          True 时缺字段 / shape 错位直接 raise；False 时回退 zero
                         tensor（仅用于离线可视化 / debug，**绝不要**在训练里用）。

    Returns:
        dict[str, torch.Tensor]，键名与 src/model/DrivePi0/map_encoder.py.forward 对齐：
            map_point_position:        [M, 3, P, 2] float32
            map_point_vector:          [M, 3, P, 2] float32
            map_point_orientation:     [M, 3, P]    float32
            map_polygon_center:        [M, 3]       float32
            map_polygon_type:          [M]          long  (0=lane, 1=lane_connector)
            map_polygon_on_route:      [M]          long  (0/1)
            map_polygon_tl_status:     [M]          long  (0=GREEN..3=UNKNOWN)
            map_polygon_has_speed_limit: [M]        bool
            map_polygon_speed_limit:   [M]          float32 (m/s)
            map_point_valid_mask:      [M, P]       float32
            map_valid_mask:            [M]          float32
    """
    if not has_hdmap_vector(all_data):
        if strict:
            missing = [k for k in HDMAP_VECTOR_KEYS if k not in all_data]
            raise RuntimeError(
                "use_hdmap_vector=True 但 pkl 缺失 hdmap_vector_* 字段：{}。"
                " 请重新跑 mogo-dataset-tools/parseRosBag-...p2lane.py 生成数据，"
                " 或改 strict=False 仅用于 debug。".format(missing)
            )
        return _zero_outputs(max_map_tokens, sample_points)

    point_position = np.asarray(all_data["hdmap_vector_point_position"], dtype=np.float32)
    if point_position.size == 0 or point_position.shape[0] == 0:
        # 无 lane 命中是合法的（旷野等），返回 zero + valid_mask=0
        return _zero_outputs(max_map_tokens, sample_points)

    M_in = point_position.shape[0]
    P_in = point_position.shape[2] if point_position.ndim >= 3 else sample_points
    if P_in != sample_points:
        if strict:
            raise RuntimeError(
                "hdmap_vector_point_position 的 P={} 与 map_sample_points={} 不一致；"
                "请检查数据生成端的 DEFAULT_SAMPLE_POINTS / sample_points 配置。"
                .format(P_in, sample_points)
            )
        return _zero_outputs(max_map_tokens, sample_points)

    point_vector = np.asarray(all_data["hdmap_vector_point_vector"], dtype=np.float32)
    point_orientation = np.asarray(all_data["hdmap_vector_point_orientation"], dtype=np.float32)
    polygon_center = np.asarray(all_data["hdmap_vector_polygon_center"], dtype=np.float32)
    polygon_type = np.asarray(all_data["hdmap_vector_polygon_type"]).astype(np.int64)
    polygon_on_route = np.asarray(all_data["hdmap_vector_polygon_on_route"]).astype(np.int64)
    polygon_tl_status = np.asarray(all_data["hdmap_vector_polygon_tl_status"]).astype(np.int64)
    polygon_has_speed_limit = np.asarray(all_data["hdmap_vector_polygon_has_speed_limit"]).astype(bool)
    polygon_speed_limit = np.asarray(all_data["hdmap_vector_polygon_speed_limit"], dtype=np.float32)
    point_valid_mask = np.asarray(all_data["hdmap_vector_point_valid_mask"], dtype=np.float32)
    valid_mask = np.asarray(all_data["hdmap_vector_valid_mask"], dtype=np.float32)

    # 距离 ego 的最近点：用 polygon_center 的 (x, y) 即可。
    if polygon_center.shape[0] != M_in:
        if strict:
            raise RuntimeError(
                "hdmap_vector_polygon_center.M={} 与 point_position.M={} 不一致".format(
                    polygon_center.shape[0], M_in
                )
            )
        return _zero_outputs(max_map_tokens, sample_points)
    dists = np.linalg.norm(polygon_center[:, :2], axis=1)
    # 把 invalid lane（valid_mask == 0）排到最后
    sort_keys = np.where(valid_mask > 0, dists, dists + 1e9)
    order = np.argsort(sort_keys)

    M = int(max_map_tokens)
    P = int(sample_points)

    keep = order[:M]

    out = _zero_outputs(M, P)
    n_keep = len(keep)
    if n_keep == 0:
        return out

    out["map_point_position"][:n_keep] = torch.from_numpy(point_position[keep])
    out["map_point_vector"][:n_keep] = torch.from_numpy(point_vector[keep])
    out["map_point_orientation"][:n_keep] = torch.from_numpy(point_orientation[keep])
    out["map_polygon_center"][:n_keep] = torch.from_numpy(polygon_center[keep])
    out["map_polygon_type"][:n_keep] = torch.from_numpy(polygon_type[keep])
    out["map_polygon_on_route"][:n_keep] = torch.from_numpy(polygon_on_route[keep])
    out["map_polygon_tl_status"][:n_keep] = torch.from_numpy(polygon_tl_status[keep])
    out["map_polygon_has_speed_limit"][:n_keep] = torch.from_numpy(polygon_has_speed_limit[keep])
    out["map_polygon_speed_limit"][:n_keep] = torch.from_numpy(polygon_speed_limit[keep])
    out["map_point_valid_mask"][:n_keep] = torch.from_numpy(point_valid_mask[keep])
    out["map_valid_mask"][:n_keep] = torch.from_numpy(valid_mask[keep])

    return out
