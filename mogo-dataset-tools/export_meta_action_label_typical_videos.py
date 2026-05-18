#!/usr/bin/env python3
"""Export representative failure videos for each mismatched raw meta_action label."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
from export_meta_action_failure_videos import (
    DEFAULT_METRICS_CSV,
    natural_sort_key,
    parse_bool,
    safe_name,
)


DEFAULT_OUTPUT_DIR = Path(
    "/pcpt/pcpt/data/VLA/MOGO/meta_action_test_pipeline/"
    "20260511_after_processed_ali/label_typical_videos"
)


def load_records(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def scene_key_from_rel_path(rel_path: str) -> str:
    return str(Path(rel_path).parent).replace("\\", "/")


def decode_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return "" if value is None else str(value)


def step_sort_key(record: dict[str, str]) -> list[object]:
    return natural_sort_key(record["rel_path"])


def group_scene_records(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        by_scene[scene_key_from_rel_path(record["rel_path"])].append(record)
    return {
        scene_key: sorted(scene_records, key=step_sort_key)
        for scene_key, scene_records in sorted(
            by_scene.items(),
            key=lambda item: natural_sort_key(item[0]),
        )
    }


def build_label_scene_stats(
    records: list[dict[str, str]],
) -> dict[str, list[dict[str, object]]]:
    mismatch_counter: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        if parse_bool(record["match"]):
            continue
        label = record["raw_meta_action"]
        scene_key = scene_key_from_rel_path(record["rel_path"])
        mismatch_counter[label][scene_key] += 1

    label_scene_stats: dict[str, list[dict[str, object]]] = {}
    for label, scene_counts in mismatch_counter.items():
        scene_stats = [
            {
                "label": label,
                "scene_key": scene_key,
                "mismatch_count": mismatch_count,
            }
            for scene_key, mismatch_count in scene_counts.items()
        ]
        scene_stats.sort(
            key=lambda item: (
                natural_sort_key(str(item["scene_key"])),
                -int(item["mismatch_count"]),
            )
        )
        label_scene_stats[label] = scene_stats
    return dict(
        sorted(label_scene_stats.items(), key=lambda item: natural_sort_key(item[0]))
    )


def select_spread_representatives(
    scene_stats: list[dict[str, object]],
    videos_per_label: int,
) -> list[dict[str, object]]:
    if len(scene_stats) <= videos_per_label:
        return scene_stats

    selected: list[dict[str, object]] = []
    for bucket_index in range(videos_per_label):
        start = bucket_index * len(scene_stats) // videos_per_label
        end = (bucket_index + 1) * len(scene_stats) // videos_per_label
        bucket = scene_stats[start:end]
        if not bucket:
            continue
        best = max(
            bucket,
            key=lambda item: (
                int(item["mismatch_count"]),
                -len(str(item["scene_key"])),
            ),
        )
        selected.append(best)
    return selected


@lru_cache(maxsize=8192)
def load_pkl_data(pkl_path: str) -> dict[str, Any] | None:
    path = Path(pkl_path)
    if not path.is_file():
        return None
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def valid_traj_p2lane_point(point: Any) -> bool:
    if not isinstance(point, (list, tuple)) or len(point) < 4:
        return False
    try:
        _ = float(point[1])
        d = float(point[2])
    except (TypeError, ValueError):
        return False
    return math.isfinite(d)


def draw_text_block(
    frame: cv2.typing.MatLike,
    lines: list[str],
    *,
    origin_x: int = 16,
    origin_y: int = 20,
    line_gap: int = 10,
    font_scale: float = 0.75,
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    max_width = 0
    total_height = 0
    text_sizes: list[tuple[tuple[int, int], int]] = []

    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )
        text_sizes.append(((text_width, text_height), baseline))
        max_width = max(max_width, text_width)
        total_height += text_height + baseline
    total_height += max(0, len(lines) - 1) * line_gap

    x1 = origin_x - 10
    y1 = origin_y - 14
    x2 = origin_x + max_width + 16
    y2 = origin_y + total_height + 12

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    cursor_y = origin_y
    for line, ((_, text_height), baseline) in zip(lines, text_sizes):
        cv2.putText(
            frame,
            line,
            (origin_x, cursor_y + text_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cursor_y += text_height + baseline + line_gap


def draw_traj_p2lane_panel(
    frame: cv2.typing.MatLike,
    traj_p2lane_stable: Any,
) -> None:
    if not isinstance(traj_p2lane_stable, (list, tuple)) or not traj_p2lane_stable:
        return

    width = frame.shape[1]
    height = frame.shape[0]
    panel_width = min(520, int(width * 0.38))
    panel_height = min(290, int(height * 0.34))
    panel_x1 = width - panel_width - 20
    panel_y1 = 18
    panel_x2 = panel_x1 + panel_width
    panel_y2 = panel_y1 + panel_height

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), (180, 180, 180), 1)

    title = "traj_p2lane_stable  centerline(d=0) vs. window points"
    cv2.putText(
        frame,
        title,
        (panel_x1 + 10, panel_y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    plot_x1 = panel_x1 + 18
    plot_y1 = panel_y1 + 42
    plot_x2 = panel_x2 - 18
    plot_y2 = panel_y2 - 42
    cv2.rectangle(frame, (plot_x1, plot_y1), (plot_x2, plot_y2), (120, 120, 120), 1)

    valid_points = [
        (idx, point)
        for idx, point in enumerate(traj_p2lane_stable)
        if valid_traj_p2lane_point(point)
    ]
    if not valid_points:
        cv2.putText(
            frame,
            "no valid traj_p2lane_stable points",
            (plot_x1 + 10, (plot_y1 + plot_y2) // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        return

    ds = [float(point[2]) for _, point in valid_points]
    scale_abs_d = max(1.5, min(6.0, max(abs(value) for value in ds) * 1.2))
    center_y = (plot_y1 + plot_y2) // 2
    cv2.line(frame, (plot_x1, center_y), (plot_x2, center_y), (80, 220, 80), 2)

    current_idx = min(4, len(traj_p2lane_stable) - 1)
    total_points = max(len(traj_p2lane_stable), 2)

    def point_xy(point_index: int, d_value: float) -> tuple[int, int]:
        x = int(
            plot_x1
            + (plot_x2 - plot_x1) * point_index / max(total_points - 1, 1)
        )
        y = int(
            center_y
            - (d_value / scale_abs_d) * ((plot_y2 - plot_y1) * 0.44)
        )
        return x, y

    prev_xy: tuple[int, int] | None = None
    for point_index, point in enumerate(traj_p2lane_stable):
        x = int(
            plot_x1
            + (plot_x2 - plot_x1) * point_index / max(total_points - 1, 1)
        )
        if valid_traj_p2lane_point(point):
            d_value = float(point[2])
            xy = point_xy(point_index, d_value)
            if prev_xy is not None:
                cv2.line(frame, prev_xy, xy, (255, 255, 255), 2)
            color = (0, 200, 255) if point_index == current_idx else (255, 255, 255)
            radius = 7 if point_index == current_idx else 4
            cv2.circle(frame, xy, radius, color, -1)
            relation = decode_value(point[3]) if len(point) >= 4 else ""
            if relation:
                cv2.putText(
                    frame,
                    relation,
                    (xy[0] - 14, xy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    color,
                    1,
                    cv2.LINE_AA,
                )
            prev_xy = xy
        else:
            cv2.line(frame, (x, center_y - 6), (x, center_y + 6), (140, 140, 140), 1)
            prev_xy = None

    current_x = int(
        plot_x1 + (plot_x2 - plot_x1) * current_idx / max(total_points - 1, 1)
    )
    cv2.line(frame, (current_x, plot_y1), (current_x, plot_y2), (0, 200, 255), 1)

    labels = [
        ("left +", plot_x1 + 6, plot_y1 + 18),
        ("right -", plot_x1 + 6, plot_y2 - 10),
        (f"scale +/- {scale_abs_d:.2f} m", plot_x2 - 160, plot_y2 - 10),
        ("centerline", plot_x2 - 100, center_y - 8),
        ("current", current_x - 22, plot_y2 + 22),
    ]
    for text, x, y in labels:
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )


def build_frame(
    image_path: Path,
    record: dict[str, str],
    scene_key: str,
    frame_index: int,
    total_frames: int,
    pkl_root: Path,
) -> cv2.typing.MatLike | None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None

    pkl_data = load_pkl_data(str((pkl_root / record["rel_path"]).resolve()))
    temp_meta_action = ""
    traj_p2lane_stable: Any = None
    if pkl_data:
        temp_meta_action = decode_value(pkl_data.get("temp_meta_action"))
        traj_p2lane_stable = pkl_data.get("traj_p2lane_stable")

    match = parse_bool(record["match"])
    status = "MATCH" if match else "MISMATCH"
    status_color = (60, 180, 75) if match else (50, 50, 230)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 8), status_color, -1)

    lines = [
        f"scene: {scene_key}",
        f"frame: {frame_index}/{total_frames}",
        f"raw_meta_action: {record['raw_meta_action']}",
    ]
    if temp_meta_action:
        lines.append(f"temp_meta_action: {temp_meta_action}")
    lines.extend(
        [
            f"pred_meta_action: {record['final_meta_action']}",
            f"match: {status}",
            f"pkl: {record['rel_path']}",
        ]
    )
    draw_text_block(frame, lines)
    draw_traj_p2lane_panel(frame, traj_p2lane_stable)
    return frame


def infer_video_shape(
    records: list[dict[str, str]],
) -> tuple[int, int]:
    for record in records:
        frame = cv2.imread(str(record["cam_front"]))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        return width, height
    raise RuntimeError("unable to infer video shape: no readable cam_front images")


def write_video(
    records: list[dict[str, str]],
    output_path: Path,
    scene_key: str,
    fps: int,
    codec: str,
    pkl_root: Path,
) -> tuple[int, int]:
    width, height = infer_video_shape(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {output_path}")

    written = 0
    missing = 0
    total_frames = len(records)
    try:
        for frame_index, record in enumerate(records, start=1):
            frame = build_frame(
                Path(record["cam_front"]),
                record,
                scene_key,
                frame_index,
                total_frames,
                pkl_root,
            )
            if frame is None:
                missing += 1
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            written += 1
    finally:
        writer.release()
    return written, missing


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "sample_rank",
        "scene_key",
        "video_path",
        "mismatch_count_for_label",
        "scene_total_frames",
        "scene_total_mismatch_frames",
        "first_rel_path",
        "last_rel_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each mismatched raw meta_action label, export representative scene "
            "videos from spread-out parts of the dataset."
        )
    )
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--pkl-root",
        type=Path,
        default=None,
        help="Root directory containing the final pkls. Defaults to <metrics_csv_dir>/pkl_after_pipeline.",
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--videos-per-label",
        type=int,
        default=2,
        help="How many representative videos to export for each label.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Only export the specified raw meta_action label. Repeatable.",
    )
    parser.add_argument(
        "--limit-labels",
        type=int,
        default=None,
        help="Only process the first N labels after sorting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_csv = args.metrics_csv.resolve()
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"metrics csv not found: {metrics_csv}")

    pkl_root = (
        args.pkl_root.resolve()
        if args.pkl_root is not None
        else (metrics_csv.parent / "pkl_after_pipeline").resolve()
    )
    if not pkl_root.is_dir():
        raise FileNotFoundError(f"pkl root not found: {pkl_root}")

    records = load_records(metrics_csv)
    scene_records_by_key = group_scene_records(records)
    label_scene_stats = build_label_scene_stats(records)

    if args.label:
        requested_labels = set(args.label)
        label_scene_stats = {
            label: stats
            for label, stats in label_scene_stats.items()
            if label in requested_labels
        }

    if args.limit_labels is not None:
        label_scene_stats = dict(list(label_scene_stats.items())[: args.limit_labels])

    if not label_scene_stats:
        raise ValueError("no mismatched labels found for the given filters")

    output_dir = args.output_dir.resolve()
    manifest_rows: list[dict[str, object]] = []
    total_labels = len(label_scene_stats)

    for label_index, (label, scene_stats) in enumerate(label_scene_stats.items(), start=1):
        print(
            f"[INFO] exporting label {label_index}/{total_labels}: {label} "
            f"(candidate_scenes={len(scene_stats)})",
            flush=True,
        )
        selected_scenes = select_spread_representatives(
            scene_stats=scene_stats,
            videos_per_label=args.videos_per_label,
        )
        label_dir = output_dir / safe_name(label)
        for sample_rank, scene_stat in enumerate(selected_scenes, start=1):
            scene_key = str(scene_stat["scene_key"])
            scene_records = scene_records_by_key[scene_key]
            scene_total_mismatch_frames = sum(
                1 for record in scene_records if not parse_bool(record["match"])
            )
            output_path = (
                label_dir
                / f"{safe_name(label)}__sample{sample_rank:02d}__"
                f"{safe_name(scene_key)}.mp4"
            )
            overlay_scene_key = f"{scene_key} | focus_label={label}"
            written, missing = write_video(
                scene_records,
                output_path,
                overlay_scene_key,
                args.fps,
                args.codec,
                pkl_root,
            )
            if missing:
                print(
                    f"[WARN] missing images in {scene_key}: {missing}",
                    flush=True,
                )
            manifest_rows.append(
                {
                    "label": label,
                    "sample_rank": sample_rank,
                    "scene_key": scene_key,
                    "video_path": str(output_path),
                    "mismatch_count_for_label": int(scene_stat["mismatch_count"]),
                    "scene_total_frames": len(scene_records),
                    "scene_total_mismatch_frames": scene_total_mismatch_frames,
                    "first_rel_path": scene_records[0]["rel_path"],
                    "last_rel_path": scene_records[-1]["rel_path"],
                }
            )
            print(
                f"[INFO] saved {output_path.name} frames={written} "
                f"label_mismatch={scene_stat['mismatch_count']}",
                flush=True,
            )

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"[DONE] videos saved under: {output_dir}", flush=True)
    print(f"[DONE] manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
