"""
统计 mogo_dataset_navi_hdmap 数据集的归一化参数和 hdmap 特征分布。

用法:
python src/data_processing/get_statistics_hdmap.py \
    --data_path /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/train \
    --output_path /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_hdmap/statistics_hdmap.json
"""

import argparse
import json
import os
import pickle

import numpy as np
from tqdm import tqdm


def calculate(data):
    percentile_99 = np.percentile(data, 99)
    percentile_01 = np.percentile(data, 1)
    return percentile_01, percentile_99


def calc_minmax(data):
    return np.min(data), np.max(data)


def fmt(x):
    return float(x)


def flatten(data):
    return np.concatenate([arr.flatten() for arr in data])


def split_future_xy(all_data):
    if "fur_trajectory" in all_data:
        fur = np.asarray(all_data["fur_trajectory"], dtype=np.float32)
        # fur_trajectory 存储顺序为 [y, x]，所以 [:,0]=y, [:,1]=x
        return fur[:, 1], fur[:, 0]
    return np.asarray(all_data["fur_x"]), np.asarray(all_data["fur_y"])


def split_history_xy(all_data):
    his = all_data.get("his_trajectory")
    if his is None:
        return None, None
    his = np.asarray(his, dtype=np.float32)
    if his.ndim != 2 or his.shape[-1] != 2:
        raise ValueError(f"Unexpected his_trajectory shape: {his.shape}")
    # his_trajectory 存储顺序为 [y, x]，所以 [:,0]=y, [:,1]=x
    return his[:, 1], his[:, 0]


NAVIGATION_COMMAND_TO_ID = {
    "straight": 0,
    "turn left": 1,
    "turn right": 2,
    "u turn": 3,
    "park": 4,
    "none": 0,
}


def infer_navigation_id(all_data):
    nav = all_data.get("navigation_id")
    if nav is not None:
        nav = np.asarray(nav)
        if nav.size > 0:
            return int(nav.reshape(-1)[0])

    cmd = all_data.get("navigation_command")
    if cmd is None:
        return 0
    cmd = np.asarray(cmd)
    if cmd.size == 0:
        return 0
    token = str(cmd.reshape(-1, cmd.shape[-1])[0][0]).strip().lower() if cmd.ndim >= 2 else str(cmd.reshape(-1)[0]).strip().lower()
    return NAVIGATION_COMMAND_TO_ID.get(token, 0)


def compute_hdmap_channel_stats(npz_path):
    """计算单个 hdmap npz 文件的通道级统计。

    返回每个通道的激活率 (1的比例)。
    通道定义: 0=lanes, 1=lane_links, 2=lane_boundaries, 3=junctions
    """
    if isinstance(npz_path, bytes):
        npz_path = npz_path.decode("utf-8")
    if not npz_path or not os.path.exists(npz_path):
        return None
    data = np.load(npz_path)
    bev = data["hdmap_bev"]  # (4, 960, 1792)
    total_pixels = bev.shape[1] * bev.shape[2]
    channel_density = []
    for ch in range(bev.shape[0]):
        density = np.sum(bev[ch]) / total_pixels
        channel_density.append(float(density))
    return channel_density


def generate_data(data_path, output_path="statistics_hdmap.json", compute_hdmap_stats=False):
    list_data_dict = [f for f in os.listdir(data_path) if f.endswith(".pkl")]
    his_speed = []
    his_acceleration = []
    his_angular_velocity = []
    his_theta = []
    his_x = []
    his_y = []
    fur_x = []
    fur_y = []
    navigation_id = []

    # hdmap 统计
    hdmap_channel_densities = [[] for _ in range(4)]  # 4 channels
    hdmap_valid_count = 0
    hdmap_missing_count = 0
    channel_names = ["lanes", "lane_links", "lane_boundaries", "junctions"]

    print("calculating percentile_01 and percentile_99 ...")
    for i in tqdm(range(len(list_data_dict))):
        pkl_data_path = os.path.join(data_path, list_data_dict[i])
        with open(pkl_data_path, "rb") as f:
            all_data = pickle.load(f)

            # --- 传统状态量 ---
            speed = np.asarray(all_data["his_speed"])
            # 兼容 (N,) 和 (N,3) 两种格式
            if speed.ndim == 1:
                his_speed.append(speed)
            else:
                his_speed.append(speed[:, :2])

            his_theta.append(np.asarray(all_data["his_theta"]))
            # angular_velocity: 仅取 dim2 (yaw)
            his_angular_velocity.append(np.asarray(all_data["his_angular_velocity"])[:, 2:3])
            # acceleration: 取前2维 (y, x)
            his_acceleration.append(np.asarray(all_data["his_acceleration"])[:, :2])

            hx, hy = split_history_xy(all_data)
            if hx is not None:
                his_x.append(hx)
                his_y.append(hy)

            fx, fy = split_future_xy(all_data)
            fur_x.append(fx)
            fur_y.append(fy)

            nav_id = infer_navigation_id(all_data)
            navigation_id.append(np.asarray([nav_id], dtype=np.float32))

            # --- HDMap BEV 统计 (可选，加载 NPZ 较慢) ---
            if compute_hdmap_stats:
                hdmap_bev_path = all_data.get("hdmap_bev")
                if hdmap_bev_path is not None:
                    channel_stats = compute_hdmap_channel_stats(hdmap_bev_path)
                    if channel_stats is not None:
                        hdmap_valid_count += 1
                        for ch_idx, density in enumerate(channel_stats):
                            hdmap_channel_densities[ch_idx].append(density)
                    else:
                        hdmap_missing_count += 1
                else:
                    hdmap_missing_count += 1

    # --- 计算传统状态量统计 ---
    his_speed = flatten(his_speed)
    his_acceleration = flatten(his_acceleration)
    his_angular_velocity = flatten(his_angular_velocity)
    his_theta = flatten(his_theta)
    fur_x = flatten(fur_x)
    fur_y = flatten(fur_y)
    has_his_traj = len(his_x) > 0
    if has_his_traj:
        his_x = flatten(his_x)
        his_y = flatten(his_y)
    navigation_id = flatten(navigation_id)

    speed_01, speed_99 = calculate(his_speed)
    acceleration_01, acceleration_99 = calculate(his_acceleration)
    angular_velocity_01, angular_velocity_99 = calculate(his_angular_velocity)
    theta_01, theta_99 = calculate(his_theta)
    fur_x_01, fur_x_99 = calculate(fur_x)
    fur_y_01, fur_y_99 = calculate(fur_y)
    if has_his_traj:
        his_x_01, his_x_99 = calculate(his_x)
        his_y_01, his_y_99 = calculate(his_y)
    navigation_id_01, navigation_id_99 = calculate(navigation_id)

    speed_min, speed_max = calc_minmax(his_speed)
    acceleration_min, acceleration_max = calc_minmax(his_acceleration)
    angular_velocity_min, angular_velocity_max = calc_minmax(his_angular_velocity)
    theta_min, theta_max = calc_minmax(his_theta)
    fur_x_min, fur_x_max = calc_minmax(fur_x)
    fur_y_min, fur_y_max = calc_minmax(fur_y)
    if has_his_traj:
        his_x_min, his_x_max = calc_minmax(his_x)
        his_y_min, his_y_max = calc_minmax(his_y)
    navigation_id_min, navigation_id_max = calc_minmax(navigation_id)

    print("=== Percentile (p01, p99) ===")
    print("speed:", speed_01, speed_99)
    print("acceleration:", acceleration_01, acceleration_99)
    print("angular_velocity:", angular_velocity_01, angular_velocity_99)
    print("theta:", theta_01, theta_99)
    if has_his_traj:
        print("his_x:", his_x_01, his_x_99)
        print("his_y:", his_y_01, his_y_99)
    print("fur_x:", fur_x_01, fur_x_99)
    print("fur_y:", fur_y_01, fur_y_99)
    print("navigation_id:", navigation_id_01, navigation_id_99)

    print("\n=== MinMax (min, max) ===")
    print("speed:", speed_min, speed_max)
    print("acceleration:", acceleration_min, acceleration_max)
    print("angular_velocity:", angular_velocity_min, angular_velocity_max)
    print("theta:", theta_min, theta_max)
    if has_his_traj:
        print("his_x:", his_x_min, his_x_max)
        print("his_y:", his_y_min, his_y_max)
    print("fur_x:", fur_x_min, fur_x_max)
    print("fur_y:", fur_y_min, fur_y_max)
    print("navigation_id:", navigation_id_min, navigation_id_max)

    # --- HDMap 统计输出 ---
    hdmap_section = {}
    if compute_hdmap_stats:
        print(f"\n=== HDMap BEV Statistics ===")
        print(f"Valid hdmap samples: {hdmap_valid_count}, Missing: {hdmap_missing_count}")
        hdmap_stats = {}
        for ch_idx, ch_name in enumerate(channel_names):
            if hdmap_channel_densities[ch_idx]:
                ch_arr = np.array(hdmap_channel_densities[ch_idx])
                ch_mean = float(np.mean(ch_arr))
                ch_std = float(np.std(ch_arr))
                ch_min = float(np.min(ch_arr))
                ch_max = float(np.max(ch_arr))
                ch_median = float(np.median(ch_arr))
                print(f"  {ch_name}: mean_density={ch_mean:.6f}, std={ch_std:.6f}, min={ch_min:.6f}, max={ch_max:.6f}, median={ch_median:.6f}")
                hdmap_stats[ch_name] = {
                    "mean_density": ch_mean,
                    "std_density": ch_std,
                    "min_density": ch_min,
                    "max_density": ch_max,
                    "median_density": ch_median,
                }
        hdmap_section = {
            "valid_count": hdmap_valid_count,
            "missing_count": hdmap_missing_count,
            "total_count": hdmap_valid_count + hdmap_missing_count,
            "channel_stats": hdmap_stats,
            "bev_shape": [4, 960, 1792],
            "channel_names": channel_names,
            "spatial_range": {
                "lateral_m": 96.0,
                "longitudinal_m": 179.2,
                "resolution_m_per_pixel": 0.1,
            },
        }

    # --- 组装输出 JSON ---
    data = {
        "percentile": {
            "speed": [fmt(speed_01), fmt(speed_99)],
            "acceleration": [fmt(acceleration_01), fmt(acceleration_99)],
            "angular_velocity": [fmt(angular_velocity_01), fmt(angular_velocity_99)],
            "theta": [fmt(theta_01), fmt(theta_99)],
            "fur_x": [fmt(fur_x_01), fmt(fur_x_99)],
            "fur_y": [fmt(fur_y_01), fmt(fur_y_99)],
            **({"his_x": [fmt(his_x_01), fmt(his_x_99)], "his_y": [fmt(his_y_01), fmt(his_y_99)]} if has_his_traj else {}),
            "navigation_id": [fmt(navigation_id_01), fmt(navigation_id_99)],
        },
        "minmax": {
            "speed": [fmt(speed_min), fmt(speed_max)],
            "acceleration": [fmt(acceleration_min), fmt(acceleration_max)],
            "angular_velocity": [fmt(angular_velocity_min), fmt(angular_velocity_max)],
            "theta": [fmt(theta_min), fmt(theta_max)],
            "fur_x": [fmt(fur_x_min), fmt(fur_x_max)],
            "fur_y": [fmt(fur_y_min), fmt(fur_y_max)],
            **({"his_x": [fmt(his_x_min), fmt(his_x_max)], "his_y": [fmt(his_y_min), fmt(his_y_max)]} if has_his_traj else {}),
            "navigation_id": [fmt(navigation_id_min), fmt(navigation_id_max)],
        },
    }
    if hdmap_section:
        data["hdmap"] = hdmap_section

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"\nStatistics saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--output_path",
        type=str,
        default="statistics_hdmap.json",
        help="Output path for statistics JSON file",
    )
    parser.add_argument(
        "--compute_hdmap_stats",
        action="store_true",
        help="是否加载 NPZ 计算 hdmap 通道分布统计（很慢，仅诊断用，训练不需要）",
    )
    args = parser.parse_args()
    generate_data(args.data_path, args.output_path, compute_hdmap_stats=args.compute_hdmap_stats)
