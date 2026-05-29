#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


# DEFAULT_POINT_CLOUD_RANGE = [-81.6, -48.0, -1.0, 97.6, 48.0, 3.0]
DEFAULT_POINT_CLOUD_RANGE = [-48.0, -81.6, -1.0, 48.0, 97.6, 3.0]
DEFAULT_VOXEL_SIZE = [0.1, 0.1, 0.2]

CHANNEL_NAMES = {
    0: "lanes",
    1: "lane_links",
    2: "lane_boundaries",
    3: "junctions",
}


def decode_path(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, str):
        return value
    raise TypeError("不支持的路径类型: {}".format(type(value)))


def load_pkl_data(pkl_path):
    with open(pkl_path, "rb") as file_obj:
        return pickle.load(file_obj)


def load_hdmap_bev_from_pkl(pkl_path, key="hdmap_bev"):
    pkl_data = load_pkl_data(pkl_path)
    if key not in pkl_data:
        raise KeyError("pkl 中缺少字段 `{}`: {}".format(key, pkl_path))

    npz_path = decode_path(pkl_data[key])
    if not os.path.isfile(npz_path):
        raise FileNotFoundError("pkl 指向的 hdmap npz 不存在: {}".format(npz_path))

    npz_data = np.load(npz_path)
    if "hdmap_bev" in npz_data:
        bev = npz_data["hdmap_bev"]
    elif len(npz_data.files) == 1:
        bev = npz_data[npz_data.files[0]]
    else:
        raise KeyError(
            "未在 {} 中找到 key `hdmap_bev`，可用 keys: {}".format(
                npz_path, npz_data.files
            )
        )

    if bev.ndim != 3 or bev.shape[0] != 4:
        raise ValueError(
            "期望 HDMap BEV shape 为 (4, H, W)，实际得到 {}".format(bev.shape)
        )
    return pkl_data, npz_path, bev.astype(np.float32)


def build_rgb_overlay(bev):
    rgb = np.zeros((bev.shape[1], bev.shape[2], 3), dtype=np.float32)

    # 参考 multimodal_bevfusion:
    # R -> junctions, G -> lane_links, B -> lane_boundaries
    rgb[..., 0] = bev[3]
    rgb[..., 1] = bev[1]
    rgb[..., 2] = bev[2]

    # lanes 作为白色叠加到 RGB 三个通道
    rgb += bev[0][..., None]
    return np.clip(rgb, 0.0, 1.0)


def infer_bev_layout(bev, point_cloud_range, voxel_size):
    x_bins = int(round((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]))
    y_bins = int(round((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]))

    if bev.shape[1:] == (x_bins, y_bins):
        return "xy"
    if bev.shape[1:] == (y_bins, x_bins):
        return "yx"

    raise ValueError(
        "无法根据 point_cloud_range/voxel_size 推断 bev 布局，"
        "bev.shape={}, 期望 spatial 为 ({}, {}) 或 ({}, {})".format(
            bev.shape, x_bins, y_bins, y_bins, x_bins
        )
    )


def to_display_image(image, bev_layout):
    if bev_layout == "xy":
        return image.transpose(1, 0, 2)
    if bev_layout == "yx":
        return image
    raise ValueError("未知 bev_layout: {}".format(bev_layout))


def ego_to_display_xy(x_ego, y_ego, point_cloud_range, voxel_size):
    x_min, y_min = point_cloud_range[0], point_cloud_range[1]
    vx, vy = voxel_size[0], voxel_size[1]
    # imshow 显示时，像素中心位于整数坐标处。
    # 参考生成侧 row/col -> ego 坐标时使用的是 cell center:
    # x = x_min + (row + 0.5) * vx, y = y_min + (col + 0.5) * vy
    # 因此这里做逆映射时减去 0.5，保证连续坐标和显示中心对齐。
    x_disp = (x_ego - x_min) / vx - 0.5
    y_disp = (y_ego - y_min) / vy - 0.5
    return x_disp, y_disp


def draw_ego_pose(ax, point_cloud_range, voxel_size, arrow_length_m):
    ego_x_disp, ego_y_disp = ego_to_display_xy(
        0.0, 0.0, point_cloud_range, voxel_size
    )
    # 当前 pkl/hdmap 坐标系:
    # x 正方向 -> 自车右侧
    # y 正方向 -> 自车前方
    arrow_dx = 0.0
    arrow_dy = arrow_length_m / voxel_size[1]

    ax.scatter(
        [ego_x_disp],
        [ego_y_disp],
        c="yellow",
        s=120,
        edgecolors="black",
        linewidths=1.0,
        zorder=5,
        label="ego",
    )
    ax.arrow(
        ego_x_disp,
        ego_y_disp,
        arrow_dx,
        arrow_dy,
        width=3.0,
        head_width=24.0,
        head_length=28.0,
        length_includes_head=True,
        color="yellow",
        ec="black",
        zorder=6,
    )
    ax.text(
        ego_x_disp + 10.0,
        ego_y_disp + arrow_dy + 12.0,
        "ego forward",
        color="yellow",
        fontsize=9,
        weight="bold",
        zorder=7,
    )


def draw_trajectory(
    ax,
    trajectory,
    point_cloud_range,
    voxel_size,
    color,
    label,
    linestyle="-",
    prepend_ego=False,
    append_ego=False,
):
    if trajectory is None:
        return

    traj = np.asarray(trajectory, dtype=np.float32)
    if traj.ndim != 2 or traj.shape[1] != 2 or traj.shape[0] == 0:
        return

    if prepend_ego:
        traj = np.vstack([np.zeros((1, 2), dtype=np.float32), traj])
    if append_ego:
        traj = np.vstack([traj, np.zeros((1, 2), dtype=np.float32)])

    x_disp, y_disp = ego_to_display_xy(
        traj[:, 0], traj[:, 1], point_cloud_range, voxel_size
    )
    ax.plot(
        x_disp,
        y_disp,
        linestyle=linestyle,
        linewidth=2.5,
        color=color,
        marker="o",
        markersize=4.5,
        alpha=0.95,
        label=label,
        zorder=4,
    )


def save_rgb_png(
    bev,
    save_path,
    pkl_data,
    point_cloud_range,
    voxel_size,
    arrow_length_m,
    title=None,
):
    bev_layout = infer_bev_layout(bev, point_cloud_range, voxel_size)
    rgb = build_rgb_overlay(bev)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(to_display_image(rgb, bev_layout), origin="lower")
    draw_ego_pose(
        ax=ax,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        arrow_length_m=arrow_length_m,
    )
    draw_trajectory(
        ax=ax,
        trajectory=pkl_data.get("his_trajectory"),
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        color="#00d1ff",
        label="history trajectory",
        linestyle="--",
        append_ego=True,
    )
    draw_trajectory(
        ax=ax,
        trajectory=pkl_data.get("fur_trajectory"),
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        color="#ff8c00",
        label="future trajectory",
        linestyle="-",
        prepend_ego=True,
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel("right")
    ax.set_ylabel("forward")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_channel_png(bev, channel, save_path, point_cloud_range, voxel_size):
    channel_name = CHANNEL_NAMES.get(channel, "channel_{}".format(channel))
    bev_layout = infer_bev_layout(bev, point_cloud_range, voxel_size)
    channel_img = bev[channel]
    if bev_layout == "xy":
        channel_img = channel_img.T
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(channel_img, origin="lower", cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_title(channel_name)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def default_output_dir(pkl_path):
    return os.path.join(os.path.dirname(pkl_path), "hdmap_vis")


def parse_args():
    parser = argparse.ArgumentParser(
        description="从单个 pkl 读取 HDMap 路径并可视化，同时绘制自车位置和朝向"
    )
    parser.add_argument("--pkl-path", required=True, help="输入的 pkl 路径")
    parser.add_argument(
        "--key",
        default="hdmap_bev",
        help="pkl 中保存 hdmap npz 路径的字段名，默认 `hdmap_bev`",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录，默认写到 pkl 同目录下的 `hdmap_vis/`",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="输出文件名前缀，默认使用 pkl 文件名（不含后缀）",
    )
    parser.add_argument(
        "--point-cloud-range",
        type=float,
        nargs=6,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
        default=list(DEFAULT_POINT_CLOUD_RANGE),
        help="BEV 对应的 point_cloud_range，默认与生成脚本一致",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        nargs=3,
        metavar=("VX", "VY", "VZ"),
        default=list(DEFAULT_VOXEL_SIZE),
        help="BEV 对应的 voxel_size，默认与生成脚本一致",
    )
    parser.add_argument(
        "--arrow-length-m",
        type=float,
        default=6.0,
        help="自车朝向箭头长度，单位米，默认 6.0",
    )
    parser.add_argument(
        "--rgb-only",
        action="store_true",
        help="仅保存 RGB 叠加图，不额外保存单通道图",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.pkl_path):
        raise FileNotFoundError("pkl 文件不存在: {}".format(args.pkl_path))

    pkl_data, npz_path, bev = load_hdmap_bev_from_pkl(args.pkl_path, key=args.key)
    output_dir = args.output_dir or default_output_dir(args.pkl_path)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.pkl_path))[0]

    os.makedirs(output_dir, exist_ok=True)

    title = "sample_ts={:.3f}".format(float(pkl_data["sample_timestamp"]))
    rgb_path = os.path.join(output_dir, "{}_rgb.png".format(prefix))
    save_rgb_png(
        bev=bev,
        save_path=rgb_path,
        pkl_data=pkl_data,
        point_cloud_range=args.point_cloud_range,
        voxel_size=args.voxel_size,
        arrow_length_m=args.arrow_length_m,
        title=title,
    )
    print("[INFO] 输入 pkl: {}".format(args.pkl_path))
    print("[INFO] pkl 指向的 hdmap npz: {}".format(npz_path))
    print("[INFO] 保存 RGB 可视化: {}".format(rgb_path))

    if not args.rgb_only:
        for channel, channel_name in CHANNEL_NAMES.items():
            channel_path = os.path.join(
                output_dir, "{}_{}.png".format(prefix, channel_name)
            )
            save_channel_png(
                bev,
                channel,
                channel_path,
                args.point_cloud_range,
                args.voxel_size,
            )
            print("[INFO] 保存单通道可视化: {}".format(channel_path))

    nonzero_counts = [int(bev[idx].sum()) for idx in range(bev.shape[0])]
    print(
        "[INFO] HDMap shape={}, dtype={}, nonzero_per_channel={}".format(
            bev.shape, bev.dtype, nonzero_counts
        )
    )


if __name__ == "__main__":
    main()
