"""
HDMap BEV 工具模块：

- npz_to_png_path: NPZ 路径 → PNG 路径转换
- extract_hdmap_features: 从 NPZ 提取 8 维标量特征（方案 B）
- load_hdmap_image: 加载 HDMap BEV 图像并 resize（方案 A）
"""

import os
from typing import Optional, Tuple

import numpy as np

# ---- BEV 坐标系常量 ----
# shape: (4, 960, 1792)
# dim-1 (960): 横向 (left-right), 48m each side, 0.1m/px, ego at row 480
# dim-2 (1792): 纵向 (rear-front), 89.6m each side, 0.1m/px, ego at col 896
_EGO_ROW = 480
_EGO_COL = 896

# 通道索引
_CH_LANES = 0
_CH_LANE_LINKS = 1
_CH_LANE_BOUNDARIES = 2
_CH_JUNCTIONS = 3

# ---- 特征提取区域定义 (单位: pixel, 1px = 0.1m) ----
# 近前方: ego 前方 0~30m, 左右 ±10m
_NEAR_FRONT = (
    _EGO_ROW - 100, _EGO_ROW + 100,  # rows: ±10m
    _EGO_COL, _EGO_COL + 300,         # cols: 0~30m ahead
)
# 远前方: ego 前方 30~60m, 左右 ±10m
_FAR_FRONT = (
    _EGO_ROW - 100, _EGO_ROW + 100,
    _EGO_COL + 300, _EGO_COL + 600,
)
# 左侧: 横向 10~20m 左, 纵向 ±10m
_LEFT_SIDE = (
    _EGO_ROW - 200, _EGO_ROW - 100,
    _EGO_COL - 100, _EGO_COL + 100,
)
# 右侧: 横向 10~20m 右, 纵向 ±10m
_RIGHT_SIDE = (
    _EGO_ROW + 100, _EGO_ROW + 200,
    _EGO_COL - 100, _EGO_COL + 100,
)


def npz_to_png_path(npz_path) -> str:
    """将 hdmap_bev NPZ 路径转为对应的 PNG 路径。"""
    if isinstance(npz_path, bytes):
        npz_path = npz_path.decode("utf-8")
    return npz_path.replace("hdmap_bev_npz", "hdmap_bev_img").replace(".npz", ".png")


def _region_mean(bev_channel: np.ndarray, region: tuple) -> float:
    """计算 BEV 通道在指定区域的平均激活率。"""
    r0, r1, c0, c1 = region
    # 边界裁剪
    r0 = max(r0, 0)
    r1 = min(r1, bev_channel.shape[0])
    c0 = max(c0, 0)
    c1 = min(c1, bev_channel.shape[1])
    if r1 <= r0 or c1 <= c0:
        return 0.0
    return float(np.mean(bev_channel[r0:r1, c0:c1]))


def extract_hdmap_features(npz_path) -> np.ndarray:
    """从 HDMap BEV NPZ 提取 8 维标量特征。

    特征定义 (每个值 ∈ [0, 1]):
      0: near_front_lane_density     - 近前方 (0~30m) 车道密度
      1: far_front_lane_density      - 远前方 (30~60m) 车道密度
      2: left_lane_density           - 左侧车道密度
      3: right_lane_density          - 右侧车道密度
      4: near_front_junction         - 近前方路口密度
      5: far_front_junction          - 远前方路口密度
      6: near_front_lane_link        - 近前方换道区域密度
      7: near_front_boundary         - 近前方车道线密度

    Args:
        npz_path: NPZ 文件路径 (str 或 bytes)

    Returns:
        features: (8,) float32 array, 值域 [0, 1]
    """
    if isinstance(npz_path, bytes):
        npz_path = npz_path.decode("utf-8")
    if not npz_path or not os.path.exists(npz_path):
        return np.zeros(8, dtype=np.float32)

    bev = np.load(npz_path)["hdmap_bev"]  # (4, 960, 1792)

    features = np.array([
        _region_mean(bev[_CH_LANES], _NEAR_FRONT),
        _region_mean(bev[_CH_LANES], _FAR_FRONT),
        _region_mean(bev[_CH_LANES], _LEFT_SIDE),
        _region_mean(bev[_CH_LANES], _RIGHT_SIDE),
        _region_mean(bev[_CH_JUNCTIONS], _NEAR_FRONT),
        _region_mean(bev[_CH_JUNCTIONS], _FAR_FRONT),
        _region_mean(bev[_CH_LANE_LINKS], _NEAR_FRONT),
        _region_mean(bev[_CH_LANE_BOUNDARIES], _NEAR_FRONT),
    ], dtype=np.float32)

    return features


HDMAP_PROPRIO_DIM = 8  # 方案 B 新增的 proprio 维度数


def load_hdmap_image(npz_path, size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    """加载 HDMap BEV PNG 图像并 resize (方案 A)。

    Args:
        npz_path: hdmap_bev 字段值 (NPZ 路径, str 或 bytes)
        size: 目标尺寸 (H, W)

    Returns:
        image: (H, W, 3) uint8 array, 或 None (文件不存在时)
    """
    from PIL import Image

    png_path = npz_to_png_path(npz_path)
    if not os.path.exists(png_path):
        return None

    img = Image.open(png_path).convert("RGB")
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)
