#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refine MOGO pkl meta_action labels with traj_p2lane_stable.

The script runs the p2lane label decision on every sample. Every written sample
receives raw_meta_action with the pre-refine value.

Examples:
    python3 mogo-dataset-tools/update_meta_action_by_p2lane.py \
        --input-dir /pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane \
        --dry-run

    python3 mogo-dataset-tools/update_meta_action_by_p2lane.py \
        --input-dir /pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane \
        --inplace --backup

    python3 mogo-dataset-tools/update_meta_action_by_p2lane.py \
        --input-dir /pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane \
        --output-dir /pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane_refined
"""

import argparse
import csv
import json
import math
import pickle
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


KEEP = "lane_keeping_and_centering"
LEFT_NUDGE = "in_lane_left_nudge"
RIGHT_NUDGE = "in_lane_right_nudge"
LEFT_OUT = "out_of_lane_left"
RIGHT_OUT = "out_of_lane_right"
LEFT_OUT_NUDGE = "out_of_lane_left_nudge"
RIGHT_OUT_NUDGE = "out_of_lane_right_nudge"
LEFT_LANE_CHANGE = "left_lane_change"
RIGHT_LANE_CHANGE = "right_lane_change"
RIGHT_LANE_CHAGE = "right_lane_chage"

TARGET_META_ACTIONS = {
    LEFT_OUT_NUDGE,
    RIGHT_OUT_NUDGE,
    LEFT_NUDGE,
    RIGHT_NUDGE,
    LEFT_LANE_CHANGE,
    RIGHT_LANE_CHANGE,
    RIGHT_LANE_CHAGE,
}

NUDGE_BY_DIRECTION = {"left": LEFT_NUDGE, "right": RIGHT_NUDGE}
OUT_BY_DIRECTION = {"left": LEFT_OUT, "right": RIGHT_OUT}
SIDE_ACTIONS = {
    LEFT_NUDGE,
    RIGHT_NUDGE,
    LEFT_OUT,
    RIGHT_OUT,
    LEFT_OUT_NUDGE,
    RIGHT_OUT_NUDGE,
    LEFT_LANE_CHANGE,
    RIGHT_LANE_CHANGE,
    RIGHT_LANE_CHAGE,
}
OUT_ACTIONS = {LEFT_OUT, RIGHT_OUT, LEFT_OUT_NUDGE, RIGHT_OUT_NUDGE}

STEP_RE = re.compile(r"(?:^|[_-])step(\d+)(?:\D|$)", re.IGNORECASE)
CURRENT_POINT_INDEX = 4
OUT_WINDOW_START_INDEX = CURRENT_POINT_INDEX - 1


@dataclass
class P2LaneStats:
    total_points: int = 0
    valid_points: int = 0
    median_d: float = 0.0
    current_d: float = 0.0
    max_abs_d: float = 0.0
    over_out_count: int = 0
    relations: Tuple[str, ...] = ()
    explicit_relations: Tuple[str, ...] = ()
    reason: str = ""


@dataclass
class SampleRecord:
    path: Path
    rel_path: Path
    data: Optional[Dict[str, Any]]
    load_error: str = ""
    original_action: str = ""
    single_action: str = KEEP
    final_action: str = KEEP
    confidence: float = 0.0
    reason: str = ""
    stats: P2LaneStats = field(default_factory=P2LaneStats)
    temporal_changed: bool = False
    temporal_reason: str = ""
    missing_p2lane: bool = False
    insufficient_p2lane: bool = False
    skipped_non_keep: bool = False
    skipped_by_navigation_rule: bool = False


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def relation_to_direction(relation: str) -> Optional[str]:
    if relation == "toleft":
        return "left"
    if relation == "toright":
        return "right"
    return None


def action_direction(action: str) -> Optional[str]:
    if action in (LEFT_NUDGE, LEFT_OUT, LEFT_OUT_NUDGE, LEFT_LANE_CHANGE):
        return "left"
    if action in (
        RIGHT_NUDGE,
        RIGHT_OUT,
        RIGHT_OUT_NUDGE,
        RIGHT_LANE_CHANGE,
        RIGHT_LANE_CHAGE,
    ):
        return "right"
    return None


def matching_out_action(direction: str) -> str:
    return OUT_BY_DIRECTION[direction]


def matching_nudge_action(direction: str) -> str:
    return NUDGE_BY_DIRECTION[direction]


def valid_p2lane_points(points: Any) -> Tuple[List[Tuple[Any, float, float, str]], int]:
    if not isinstance(points, (list, tuple)):
        return [], 0

    valid = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 4:
            continue
        stable_s = point[1]
        stable_d = point[2]
        if not is_finite_number(stable_d):
            continue
        relation = point[3] if isinstance(point[3], str) else ""
        valid.append(
            (point[0], safe_float(stable_s, float("nan")), float(stable_d), relation)
        )
    return valid, len(points)


def should_preserve_meta_action_by_navigation_command(sample: Dict[str, Any]) -> bool:
    navigation_command = sample.get("navigation_command")
    if not isinstance(navigation_command, (list, tuple)) or not navigation_command:
        return False

    first_item = navigation_command[0]
    if not isinstance(first_item, (list, tuple)) or len(first_item) < 2:
        return False
    if "None" in first_item:
        return False

    return safe_float(first_item[1], float("inf")) < 15.0


def compute_p2lane_stats(
    points: Any,
    min_valid_points: int,
    out_threshold: float,
) -> P2LaneStats:
    valid, total_points = valid_p2lane_points(points)
    stats = P2LaneStats(total_points=total_points, valid_points=len(valid))
    if len(valid) < min_valid_points:
        stats.reason = "insufficient_p2lane"
        return stats

    ds = [point[2] for point in valid]
    relations = tuple(point[3] for point in valid)
    explicit_relations = tuple(
        relation for relation in relations if relation in ("toleft", "toright")
    )

    stats.median_d = float(median(ds))
    stats.max_abs_d = max(abs(d) for d in ds)
    stats.over_out_count = sum(abs(d) >= out_threshold for d in ds)
    stats.relations = relations
    stats.explicit_relations = explicit_relations

    raw_points = points if isinstance(points, (list, tuple)) else []
    if len(raw_points) >= 5 and isinstance(raw_points[4], (list, tuple)):
        candidate_d = raw_points[4][2] if len(raw_points[4]) >= 3 else None
        stats.current_d = (
            float(candidate_d) if is_finite_number(candidate_d) else stats.median_d
        )
    else:
        stats.current_d = stats.median_d

    return stats


def count_same_sign(values: Iterable[float], direction: str, threshold: float) -> int:
    if direction == "left":
        return sum(value > threshold for value in values)
    return sum(value < -threshold for value in values)


def direction_from_d(value: float, threshold: float) -> Optional[str]:
    if value > threshold:
        return "left"
    if value < -threshold:
        return "right"
    return None


def infer_sample_direction(
    values: Sequence[float],
    relations: Sequence[str],
    current_d: float,
    median_d: float,
    center_threshold: float,
) -> Optional[str]:
    explicit_directions = [
        direction
        for direction in (relation_to_direction(relation) for relation in relations)
        if direction is not None
    ]
    if explicit_directions:
        counts = Counter(explicit_directions)
        direction, count = counts.most_common(1)[0]
        if len(counts) == 1 or count > len(explicit_directions) / 2:
            return direction

    current_direction = direction_from_d(current_d, center_threshold)
    median_direction = direction_from_d(median_d, center_threshold)
    if current_direction is not None and current_direction == median_direction:
        return current_direction

    majority = len(values) // 2 + 1
    left_votes = count_same_sign(values, "left", center_threshold)
    right_votes = count_same_sign(values, "right", center_threshold)
    if left_votes >= majority and left_votes > right_votes:
        return "left"
    if right_votes >= majority and right_votes > left_votes:
        return "right"
    return current_direction or median_direction


def classify_single_sample(
    sample: Dict[str, Any],
    center_threshold: float,
    out_threshold: float,
    min_valid_points: int,
) -> Tuple[str, float, str, P2LaneStats]:
    points = sample.get("traj_p2lane_stable")
    stats = compute_p2lane_stats(
        points,
        min_valid_points=min_valid_points,
        out_threshold=out_threshold,
    )
    if stats.valid_points < min_valid_points:
        return KEEP, 0.0, stats.reason, stats

    valid, _ = valid_p2lane_points(points)
    ds = [point[2] for point in valid]
    relations = [point[3] for point in valid]
    majority = len(ds) // 2 + 1

    raw_points = points if isinstance(points, (list, tuple)) else []
    context_points = raw_points[OUT_WINDOW_START_INDEX:]
    context_valid, _ = valid_p2lane_points(context_points)
    context_ds = [point[2] for point in context_valid]
    context_relations = [point[3] for point in context_valid]
    context_directions = {
        direction
        for direction in (
            relation_to_direction(relation) for relation in context_relations
        )
        if direction is not None
    }
    future_points = raw_points[CURRENT_POINT_INDEX:]
    future_valid, _ = valid_p2lane_points(future_points)
    future_ds = [point[2] for point in future_valid]
    future_relations = [point[3] for point in future_valid]

    valid_ratio = stats.valid_points / max(stats.total_points, 1)
    max_abs_d = stats.max_abs_d
    future_max_abs_d = max((abs(d) for d in future_ds), default=max_abs_d)
    strong_out_threshold = out_threshold + 0.3
    current_out_threshold = out_threshold + 0.1
    recovery_threshold = max(center_threshold + 0.55, out_threshold - 0.15)
    keep_reentry_threshold = out_threshold + 0.15

    centered_count = sum(abs(d) < center_threshold for d in ds)
    if centered_count >= majority and max_abs_d < out_threshold:
        confidence = min(
            1.0,
            0.35 + 0.45 * (centered_count / len(ds)) + 0.20 * valid_ratio,
        )
        return KEEP, confidence, "single_centered", stats

    direction = infer_sample_direction(
        ds,
        relations,
        current_d=stats.current_d,
        median_d=stats.median_d,
        center_threshold=center_threshold,
    )
    if direction is None:
        confidence = min(1.0, 0.25 + 0.35 * valid_ratio)
        return KEEP, confidence, "single_no_stable_direction", stats

    same_direction_points = count_same_sign(ds, direction, center_threshold)
    future_same_direction_points = count_same_sign(
        future_ds or ds,
        direction,
        center_threshold,
    )
    future_out_count = count_same_sign(future_ds or ds, direction, out_threshold)
    future_strong_out_count = count_same_sign(
        future_ds or ds,
        direction,
        strong_out_threshold,
    )
    future_majority = len(future_ds or ds) // 2 + 1
    current_abs_d = abs(stats.current_d)
    median_abs_d = abs(stats.median_d)
    future_last_abs_d = abs(future_ds[-1]) if future_ds else current_abs_d
    relation_supports_direction = (
        len(context_directions) == 1 and direction in context_directions
    ) or (
        len({d for d in (relation_to_direction(r) for r in future_relations) if d})
        == 1
    )
    consistent_direction = (
        same_direction_points >= majority
        and future_same_direction_points >= future_majority
    )
    recovering_to_lane = (
        bool(future_ds)
        and future_last_abs_d <= recovery_threshold
        and future_strong_out_count == 0
        and future_out_count <= len(future_ds) // 2
    )
    persistent_out = future_out_count >= future_majority
    severe_offset = (
        future_max_abs_d >= strong_out_threshold
        or current_abs_d >= current_out_threshold
        or (
            relation_supports_direction
            and max((abs(d) for d in context_ds), default=0.0) >= out_threshold
        )
    )

    if consistent_direction and persistent_out and severe_offset and not recovering_to_lane:
        confidence = min(
            1.0,
            0.45
            + 0.20 * valid_ratio
            + 0.20 * (future_out_count / max(len(future_ds or ds), 1))
            + 0.10 * (same_direction_points / len(ds))
            + 0.15 * min(future_max_abs_d / max(strong_out_threshold, 1e-6), 1.2),
        )
        return (
            matching_out_action(direction),
            confidence,
            "single_out_of_lane_persistent_offset",
            stats,
        )

    if recovering_to_lane and current_abs_d <= keep_reentry_threshold:
        confidence = min(
            1.0,
            0.30
            + 0.25 * valid_ratio
            + 0.20 * (same_direction_points / len(ds))
            + 0.25
            * min(
                (keep_reentry_threshold - future_last_abs_d)
                / max(keep_reentry_threshold, 1e-6),
                1.0,
            ),
        )
        return KEEP, confidence, "single_recovering_to_lane", stats

    if same_direction_points >= majority:
        confidence = min(
            1.0,
            0.32
            + 0.28 * valid_ratio
            + 0.25 * (same_direction_points / len(ds))
            + 0.15 * min(max(current_abs_d, median_abs_d) / max(out_threshold, 1e-6), 1.0),
        )
        return (
            matching_nudge_action(direction),
            confidence,
            "single_directional_nudge",
            stats,
        )

    return KEEP, 0.20 * valid_ratio, "single_unstable_relation_or_offset", stats


def discover_pkl_files(input_dir: Path) -> List[Path]:
    return sorted(path for path in input_dir.rglob("*.pkl") if path.is_file())


def load_pickle(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        return None, str(exc)
    if not isinstance(data, dict):
        return None, f"unexpected pickle type: {type(data).__name__}"
    return data, ""


def parse_step_from_name(path: Path) -> Optional[int]:
    match = STEP_RE.search(path.stem)
    return int(match.group(1)) if match else None


def sort_key(record: SampleRecord) -> Tuple[int, float, float, str]:
    step = parse_step_from_name(record.path)
    data = record.data or {}
    sample_ts = safe_float(data.get("sample_timestamp"), 0.0)
    relative_ts = safe_float(data.get("relative_timestamp"), 0.0)
    return (
        step if step is not None else 10**12,
        sample_ts,
        relative_ts,
        record.path.name,
    )


def group_records(records: Sequence[SampleRecord]) -> List[List[SampleRecord]]:
    grouped_by_leaf: Dict[Path, List[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped_by_leaf[record.path.parent].append(record)

    groups = []
    for leaf_records in grouped_by_leaf.values():
        scene_groups: Dict[str, List[SampleRecord]] = defaultdict(list)
        for record in leaf_records:
            scene_token = ""
            if record.data:
                scene_token = str(record.data.get("scene_token") or "")
            scene_groups[scene_token].append(record)
        groups.extend(
            sorted(items, key=sort_key) for items in scene_groups.values()
        )
    return groups


def side_action_same_direction(left: str, right: str) -> bool:
    return (
        action_direction(left) is not None
        and action_direction(left) == action_direction(right)
    )


def has_future_out(
    records: Sequence[SampleRecord],
    index: int,
    direction: str,
    context_k: int,
) -> bool:
    target = matching_out_action(direction)
    end = min(len(records), index + context_k + 1)
    return any(
        records[pos].final_action == target for pos in range(index + 1, end)
    )


def has_neighbor_out(
    records: Sequence[SampleRecord],
    index: int,
    direction: str,
    context_k: int,
) -> bool:
    target = matching_out_action(direction)
    begin = max(0, index - context_k)
    end = min(len(records), index + context_k + 1)
    return any(
        pos != index and records[pos].final_action == target
        for pos in range(begin, end)
    )


def mark_temporal(
    record: SampleRecord,
    action: str,
    reason: str,
    counters: Counter,
) -> None:
    if record.final_action == action:
        return
    record.final_action = action
    record.temporal_changed = True
    record.temporal_reason = reason
    counters["temporal_smoothing_changed"] += 1
    counters[reason] += 1


def apply_temporal_smoothing(
    groups: Sequence[Sequence[SampleRecord]],
    context_k: int,
    center_threshold: float,
    out_threshold: float,
) -> Counter:
    counters = Counter()
    for records in groups:
        for record in records:
            record.final_action = record.single_action

        # Fill one-frame holes between equal-direction side labels.
        for index in range(1, len(records) - 1):
            record = records[index]
            if record.skipped_non_keep or record.final_action != KEEP:
                continue
            prev_action = records[index - 1].final_action
            next_action = records[index + 1].final_action
            if prev_action in SIDE_ACTIONS and side_action_same_direction(
                prev_action, next_action
            ):
                mark_temporal(record, prev_action, "short_hole_filled", counters)

        # Remove one-frame side pulses surrounded by keep.
        for index in range(1, len(records) - 1):
            record = records[index]
            if record.skipped_non_keep or record.final_action not in SIDE_ACTIONS:
                continue
            if (
                records[index - 1].final_action == KEEP
                and records[index + 1].final_action == KEEP
                and record.confidence < 0.80
            ):
                mark_temporal(record, KEEP, "short_pulse_removed", counters)

        # Upgrade near-boundary nudges immediately before same-side out_of_lane.
        for index, record in enumerate(records):
            if (
                record.skipped_non_keep
                or record.final_action not in (LEFT_NUDGE, RIGHT_NUDGE)
            ):
                continue
            direction = action_direction(record.final_action)
            if (
                direction
                and record.stats.max_abs_d >= max(center_threshold, out_threshold - 0.1)
                and has_future_out(records, index, direction, context_k)
            ):
                mark_temporal(
                    record,
                    matching_out_action(direction),
                    "nudge_upgraded_to_out",
                    counters,
                )

        # Downgrade isolated out_of_lane labels supported by one threshold hit.
        for index, record in enumerate(records):
            if record.skipped_non_keep or record.final_action not in OUT_ACTIONS:
                continue
            direction = action_direction(record.final_action)
            if (
                direction
                and not has_neighbor_out(records, index, direction, context_k)
                and record.stats.over_out_count <= 1
            ):
                mark_temporal(
                    record,
                    matching_nudge_action(direction),
                    "out_downgraded_to_nudge",
                    counters,
                )

        # Resolve low-confidence left/right flips in a local three-frame window.
        for index in range(1, len(records) - 1):
            record = records[index]
            if (
                record.skipped_non_keep
                or record.final_action not in (LEFT_NUDGE, RIGHT_NUDGE)
            ):
                continue
            prev_dir = action_direction(records[index - 1].final_action)
            cur_dir = action_direction(record.final_action)
            next_dir = action_direction(records[index + 1].final_action)
            has_conflict = (
                prev_dir is not None
                and next_dir is not None
                and prev_dir != next_dir
            )
            low_confidence = (
                record.confidence < 0.55 and abs(record.stats.median_d) <= 0.5
            )
            if has_conflict and low_confidence:
                mark_temporal(record, KEEP, "left_right_conflict_to_keep", counters)
            elif (
                prev_dir is not None
                and next_dir is not None
                and cur_dir not in (prev_dir, next_dir)
            ):
                if low_confidence:
                    mark_temporal(record, KEEP, "left_right_conflict_to_keep", counters)

    return counters


def output_path_for(record: SampleRecord, input_dir: Path, output_dir: Path) -> Path:
    return output_dir / record.path.relative_to(input_dir)


def write_record(
    record: SampleRecord,
    input_dir: Path,
    output_dir: Optional[Path],
    inplace: bool,
    backup: bool,
    protocol: int,
) -> bool:
    if record.data is None:
        return False

    data = record.data
    if not record.skipped_non_keep:
        data["raw_meta_action"] = record.original_action
        data["meta_action"] = record.final_action

    if inplace:
        target_path = record.path
        if backup:
            backup_path = record.path.with_suffix(record.path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(str(record.path), str(backup_path))
    elif output_dir is not None:
        target_path = output_path_for(record, input_dir, output_dir)
        target_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        return False

    with target_path.open("wb") as f:
        pickle.dump(data, f, protocol=protocol)
    return True


def write_reports(
    report_prefix: Path,
    records: Sequence[SampleRecord],
    summary: Dict[str, Any],
    input_dir: Path,
) -> None:
    report_prefix.parent.mkdir(parents=True, exist_ok=True)

    with (report_prefix.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    csv_path = report_prefix.with_name(report_prefix.name + "_changes.csv")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pkl_path",
                "old_meta_action",
                "new_meta_action",
                "single_frame_action",
                "confidence",
                "reason",
                "temporal_reason",
                "max_abs_d",
                "median_d",
                "current_d",
                "valid_points",
                "total_points",
                "relations",
            ],
        )
        writer.writeheader()
        for record in records:
            if record.data is None:
                continue
            unchanged = (
                record.original_action == record.final_action
                and record.single_action == record.final_action
            )
            if unchanged:
                continue
            writer.writerow(
                {
                    "pkl_path": str(record.path.relative_to(input_dir)),
                    "old_meta_action": record.original_action,
                    "new_meta_action": record.final_action,
                    "single_frame_action": record.single_action,
                    "confidence": f"{record.confidence:.4f}",
                    "reason": record.reason,
                    "temporal_reason": record.temporal_reason,
                    "max_abs_d": f"{record.stats.max_abs_d:.4f}",
                    "median_d": f"{record.stats.median_d:.4f}",
                    "current_d": f"{record.stats.current_d:.4f}",
                    "valid_points": record.stats.valid_points,
                    "total_points": record.stats.total_points,
                    "relations": "|".join(record.stats.relations),
                }
            )


def build_summary(
    records: Sequence[SampleRecord],
    temporal_counters: Counter,
) -> Dict[str, Any]:
    summary = Counter()
    final_actions = Counter()
    single_actions = Counter()

    for record in records:
        summary["total_pkl"] += 1
        if record.load_error:
            summary["load_error"] += 1
            continue
        if record.missing_p2lane:
            summary["missing_traj_p2lane_stable"] += 1
        if record.insufficient_p2lane:
            summary["insufficient_p2lane"] += 1
        if record.original_action in TARGET_META_ACTIONS:
            summary["original_target_meta_action"] += 1
        if record.skipped_non_keep:
            summary["skipped_non_target_meta_action"] += 1
        if record.skipped_by_navigation_rule:
            summary["preserved_by_navigation_command"] += 1
        changed_keep_sample = (
            record.original_action != record.final_action
            and not record.skipped_non_keep
            and not record.skipped_by_navigation_rule
        )
        if changed_keep_sample:
            summary["meta_action_changed"] += 1
        final_actions[record.final_action] += 1
        single_actions[record.single_action] += 1

    result = dict(summary)
    result["single_action_counts"] = dict(single_actions)
    result["final_action_counts"] = dict(final_actions)
    result["temporal_counts"] = dict(temporal_counters)
    return result


def print_summary(summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print("[INFO] meta_action refinement summary")
    for key in [
        "total_pkl",
        "load_error",
        "missing_traj_p2lane_stable",
        "insufficient_p2lane",
        "original_target_meta_action",
        "skipped_non_target_meta_action",
        "preserved_by_navigation_command",
        "meta_action_changed",
    ]:
        print(f"{key}: {summary.get(key, 0)}")

    print("\n[INFO] final_action_counts")
    for action, count in sorted(summary.get("final_action_counts", {}).items()):
        print(f"  {action}: {count}")

    print("\n[INFO] temporal_counts")
    for key, count in sorted(summary.get("temporal_counts", {}).items()):
        print(f"  {key}: {count}")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine MOGO pkl meta_action labels using traj_p2lane_stable."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input pkl root directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output root directory. Mutually exclusive with --inplace.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input pkl files in place.",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create .pkl.bak files before inplace overwrite.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute statistics and reports; do not write pkl files.",
    )
    parser.add_argument(
        "--context-k",
        type=int,
        default=2,
        help="Temporal context size on each side. Default: 2.",
    )
    parser.add_argument(
        "--min-valid-points",
        type=int,
        default=5,
        help="Minimum valid traj_p2lane_stable points. Default: 5.",
    )
    parser.add_argument(
        "--center-threshold",
        type=float,
        default=0.4,
        help="Centering threshold for abs(stable_d). Default: 0.4.",
    )
    parser.add_argument(
        "--out-threshold",
        type=float,
        default=1.1,
        help="Out-of-lane threshold for abs(stable_d). Default: 1.1.",
    )
    parser.add_argument(
        "--write-report",
        type=str,
        default=None,
        help="Report prefix. Writes <prefix>.json and <prefix>_changes.csv.",
    )
    parser.add_argument(
        "--pickle-protocol",
        type=int,
        default=3,
        help="Pickle protocol for written files. Default: 3.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Tuple[Path, Optional[Path]]:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    if not input_dir.is_dir():
        print(f"[ERROR] input-dir does not exist or is not a directory: {input_dir}")
        sys.exit(1)

    if args.inplace and output_dir is not None:
        print("[ERROR] --inplace and --output-dir are mutually exclusive")
        sys.exit(1)

    if args.backup and not args.inplace:
        print("[WARN] --backup only applies to --inplace; ignoring backup")

    if not args.dry_run and not args.inplace and output_dir is None:
        print("[ERROR] choose --dry-run, --inplace, or --output-dir")
        sys.exit(1)

    if output_dir is not None and output_dir == input_dir:
        print("[ERROR] --output-dir must be different from --input-dir")
        sys.exit(1)

    return input_dir, output_dir


def main() -> None:
    args = parse_args()
    input_dir, output_dir = validate_args(args)

    print(f"[INFO] input_dir: {input_dir}")
    if output_dir:
        print(f"[INFO] output_dir: {output_dir}")
    print(f"[INFO] dry_run: {args.dry_run}")

    pkl_files = discover_pkl_files(input_dir)
    if not pkl_files:
        print("[WARN] no pkl files found")
        return
    print(f"[INFO] found {len(pkl_files)} pkl files")

    records: List[SampleRecord] = []
    for index, path in enumerate(pkl_files, start=1):
        data, load_error = load_pickle(path)
        record = SampleRecord(
            path=path,
            rel_path=path.relative_to(input_dir),
            data=data,
            load_error=load_error,
        )

        if data is not None:
            original_action = str(data.get("meta_action") or "")
            record.original_action = original_action
            record.final_action = original_action or KEEP

            if "traj_p2lane_stable" not in data:
                record.missing_p2lane = True

            if should_preserve_meta_action_by_navigation_command(data):
                record.skipped_by_navigation_rule = True
                record.single_action = original_action
                record.final_action = original_action
                record.reason = "preserve_by_navigation_command"
            else:
                action, confidence, reason, stats = classify_single_sample(
                    data,
                    center_threshold=args.center_threshold,
                    out_threshold=args.out_threshold,
                    min_valid_points=args.min_valid_points,
                )
                record.single_action = action
                record.final_action = action
                record.confidence = confidence
                record.reason = reason
                record.stats = stats
                record.insufficient_p2lane = stats.valid_points < args.min_valid_points

        records.append(record)

        if index % 5000 == 0:
            print(f"[INFO] loaded and classified {index}/{len(pkl_files)} files")

    groups = group_records([record for record in records if record.data is not None])
    temporal_counters = apply_temporal_smoothing(
        groups,
        context_k=max(args.context_k, 0),
        center_threshold=args.center_threshold,
        out_threshold=args.out_threshold,
    )

    summary = build_summary(records, temporal_counters)
    print_summary(summary)

    if args.write_report:
        write_reports(Path(args.write_report).resolve(), records, summary, input_dir)
        print(f"[INFO] wrote report prefix: {Path(args.write_report).resolve()}")

    if args.dry_run:
        print("[INFO] dry-run complete; no pkl files were written")
        return

    written = 0
    for index, record in enumerate(records, start=1):
        if record.data is None:
            continue
        if write_record(
            record,
            input_dir=input_dir,
            output_dir=output_dir,
            inplace=args.inplace,
            backup=args.backup,
            protocol=args.pickle_protocol,
        ):
            written += 1
        if index % 5000 == 0:
            print(f"[INFO] written {written} files")

    print(f"[INFO] done, written {written} pkl files")


if __name__ == "__main__":
    main()
