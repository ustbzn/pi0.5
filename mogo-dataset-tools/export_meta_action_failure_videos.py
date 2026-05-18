#!/usr/bin/env python3
"""Export per-scene failure videos for meta_action mismatches."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2


DEFAULT_METRICS_CSV = Path(
    "/pcpt/pcpt/data/VLA/MOGO/meta_action_test_pipeline/"
    "20260511_after_processed_ali/metrics_all.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "/pcpt/pcpt/data/VLA/MOGO/meta_action_test_pipeline/"
    "20260511_after_processed_ali/failure_scene_videos"
)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def natural_sort_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def scene_key_from_rel_path(rel_path: str) -> str:
    return str(Path(rel_path).parent).replace("\\", "/")


def safe_name(value: str) -> str:
    return value.replace("\\", "__").replace("/", "__")


def step_sort_key(record: dict[str, str]) -> list[object]:
    return natural_sort_key(record["rel_path"])


def load_records(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_scene_groups(
    records: list[dict[str, str]],
    mismatch_only_scene: bool,
) -> dict[str, list[dict[str, str]]]:
    by_scene: dict[str, list[dict[str, str]]] = defaultdict(list)
    mismatch_scenes: set[str] = set()

    for record in records:
        scene_key = scene_key_from_rel_path(record["rel_path"])
        by_scene[scene_key].append(record)
        if not parse_bool(record["match"]):
            mismatch_scenes.add(scene_key)

    if mismatch_only_scene:
        by_scene = {
            scene_key: sorted(scene_records, key=step_sort_key)
            for scene_key, scene_records in by_scene.items()
            if scene_key in mismatch_scenes
        }
    else:
        by_scene = {
            scene_key: sorted(scene_records, key=step_sort_key)
            for scene_key, scene_records in by_scene.items()
        }
    return dict(sorted(by_scene.items(), key=lambda item: natural_sort_key(item[0])))


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


def build_frame(
    image_path: Path,
    record: dict[str, str],
    scene_key: str,
    frame_index: int,
    total_frames: int,
) -> cv2.typing.MatLike | None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None

    match = parse_bool(record["match"])
    status = "MATCH" if match else "MISMATCH"
    status_color = (60, 180, 75) if match else (50, 50, 230)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 8), status_color, -1)

    lines = [
        f"scene: {scene_key}",
        f"frame: {frame_index}/{total_frames}",
        f"raw_meta_action: {record['raw_meta_action']}",
        f"pred_meta_action: {record['final_meta_action']}",
        f"match: {status}",
        f"pkl: {record['rel_path']}",
    ]
    draw_text_block(frame, lines)
    return frame


def infer_video_shape(
    records: list[dict[str, str]],
) -> tuple[int, int]:
    for record in records:
        image_path = Path(record["cam_front"])
        frame = cv2.imread(str(image_path))
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
            image_path = Path(record["cam_front"])
            frame = build_frame(image_path, record, scene_key, frame_index, total_frames)
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


def export_scene_videos(
    scene_key: str,
    scene_records: list[dict[str, str]],
    output_dir: Path,
    fps: int,
    codec: str,
) -> list[dict[str, object]]:
    mismatch_records = [record for record in scene_records if not parse_bool(record["match"])]
    if not mismatch_records:
        return []

    scene_name = safe_name(scene_key)
    outputs = [
        ("full", scene_records),
        ("mismatch_only", mismatch_records),
    ]
    manifests: list[dict[str, object]] = []
    for video_kind, records in outputs:
        output_path = output_dir / f"{scene_name}__{video_kind}.mp4"
        written, missing = write_video(records, output_path, scene_key, fps, codec)
        manifests.append(
            {
                "scene_key": scene_key,
                "video_kind": video_kind,
                "video_path": str(output_path),
                "frame_count_requested": len(records),
                "frame_count_written": written,
                "missing_image_count": missing,
                "first_rel_path": records[0]["rel_path"],
                "last_rel_path": records[-1]["rel_path"],
            }
        )
    return manifests


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_key",
        "video_kind",
        "video_path",
        "frame_count_requested",
        "frame_count_written",
        "missing_image_count",
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
            "Export two videos for each scene that contains meta_action mismatches: "
            "one full-scene video and one mismatch-only video."
        )
    )
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Specific scene key to export, repeatable. Example: 0319/.../20260319145244",
    )
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument(
        "--limit-scenes",
        type=int,
        default=None,
        help="Only export the first N mismatch scenes after sorting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_csv = args.metrics_csv.resolve()
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"metrics csv not found: {metrics_csv}")

    records = load_records(metrics_csv)
    scene_groups = load_scene_groups(records, mismatch_only_scene=True)

    if args.scene:
        requested = set(args.scene)
        scene_groups = {
            scene_key: scene_records
            for scene_key, scene_records in scene_groups.items()
            if scene_key in requested
        }

    if args.limit_scenes is not None:
        items = list(scene_groups.items())[: args.limit_scenes]
        scene_groups = dict(items)

    if not scene_groups:
        raise ValueError("no mismatch scenes found for the given filters")

    output_dir = args.output_dir.resolve()
    manifest_rows: list[dict[str, object]] = []
    total = len(scene_groups)
    for index, (scene_key, scene_records) in enumerate(scene_groups.items(), start=1):
        print(f"[INFO] exporting scene {index}/{total}: {scene_key}", flush=True)
        manifest_rows.extend(
            export_scene_videos(
                scene_key=scene_key,
                scene_records=scene_records,
                output_dir=output_dir,
                fps=args.fps,
                codec=args.codec,
            )
        )

    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"[DONE] videos saved under: {output_dir}", flush=True)
    print(f"[DONE] manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
