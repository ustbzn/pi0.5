#!/usr/bin/env python3
"""Build merged confusion matrices from stage-1 meta_action outputs.

This script reads pkls from an existing meta_action test pipeline run directory.
It reconstructs the stage-1 prediction by preferring ``temp_meta_action`` and
falling back to ``meta_action`` when the sample never went through the LLM
refine stage. Both truth and prediction are then merged with the requested
label rules before exporting metrics and confusion-matrix images.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_RUN_DIR = Path(
    "/pcpt/pcpt/data/VLA/MOGO/meta_action_test_pipeline/"
    "20260513_after_processed_ali"
)
TARGET_META_ACTIONS = {
    "out_of_lane_left_nudge",
    "out_of_lane_right_nudge",
    "in_lane_left_nudge",
    "in_lane_right_nudge",
    "left_lane_change",
    "right_lane_change",
}
MERGED_LABEL_MAP = {
    "in_lane_left_nudge": "lane_keeping_and_centering",
    "in_lane_right_nudge": "lane_keeping_and_centering",
    "lane_keeping_and_centering": "lane_keeping_and_centering",
    "out_of_lane_left_nudge": "out_of_lane_left",
    "left_lane_change": "out_of_lane_left",
    "out_of_lane_left": "out_of_lane_left",
    "out_of_lane_right_nudge": "out_of_lane_right",
    "right_lane_change": "out_of_lane_right",
    "out_of_lane_right": "out_of_lane_right",
}


def load_pickle(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def normalize_label(value: Any) -> str:
    return str(value or "").strip()


def merge_label(label: str) -> str:
    return MERGED_LABEL_MAP.get(label, label)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def rel_pkl_paths(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*.pkl"))


def build_records(pkl_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel_path in rel_pkl_paths(pkl_root):
        data = load_pickle(pkl_root / rel_path)
        if data is None:
            continue

        raw_meta_action = normalize_label(data.get("raw_meta_action"))
        stage1_meta_action = normalize_label(
            data.get("temp_meta_action") or data.get("meta_action")
        )
        if not raw_meta_action or not stage1_meta_action:
            continue

        merged_raw = merge_label(raw_meta_action)
        merged_stage1 = merge_label(stage1_meta_action)
        records.append(
            {
                "rel_path": str(rel_path),
                "raw_meta_action": raw_meta_action,
                "stage1_meta_action": stage1_meta_action,
                "merged_raw_meta_action": merged_raw,
                "merged_stage1_meta_action": merged_stage1,
                "is_original_target": raw_meta_action in TARGET_META_ACTIONS,
                "match": merged_raw == merged_stage1,
            }
        )
    return records


def filter_records(
    records: list[dict[str, Any]],
    subset: str,
) -> list[dict[str, Any]]:
    if subset == "all":
        return records
    if subset == "original_target":
        return [record for record in records if parse_bool(record["is_original_target"])]
    raise ValueError(f"unsupported subset: {subset}")


def build_labels(records: list[dict[str, Any]]) -> list[str]:
    labels = {
        str(record["merged_raw_meta_action"])
        for record in records
        if str(record["merged_raw_meta_action"])
    }
    labels.update(
        {
            str(record["merged_stage1_meta_action"])
            for record in records
            if str(record["merged_stage1_meta_action"])
        }
    )
    return sorted(labels)


def build_count_matrix(
    records: list[dict[str, Any]],
    labels: list[str],
) -> list[list[int]]:
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for record in records:
        raw_action = str(record["merged_raw_meta_action"])
        pred_action = str(record["merged_stage1_meta_action"])
        matrix[index[raw_action]][index[pred_action]] += 1
    return matrix


def row_normalize(matrix: list[list[int]]) -> list[list[float]]:
    percent_matrix: list[list[float]] = []
    for row in matrix:
        row_total = sum(row)
        if row_total == 0:
            percent_matrix.append([0.0 for _ in row])
            continue
        percent_matrix.append([(value / row_total) * 100.0 for value in row])
    return percent_matrix


def save_confusion_png(
    labels: list[str],
    matrix: list[list[int]],
    output_path: Path,
    title: str,
) -> None:
    size = max(8, min(22, 0.55 * len(labels) + 5))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("pred stage1 merged meta_action")
    ax.set_ylabel("raw merged meta_action")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    max_value = max((max(row) for row in matrix), default=0)
    threshold = max_value * 0.55
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if value <= 0:
                continue
            color = "white" if value >= threshold else "black"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_confusion_percent_png(
    labels: list[str],
    matrix: list[list[float]],
    output_path: Path,
    title: str,
) -> None:
    size = max(8, min(22, 0.55 * len(labels) + 5))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_title(title)
    ax.set_xlabel("pred stage1 merged meta_action")
    ax.set_ylabel("raw merged meta_action")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if value <= 0:
                continue
            color = "white" if value >= 55.0 else "black"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Percentage (%)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rel_path",
        "raw_meta_action",
        "stage1_meta_action",
        "merged_raw_meta_action",
        "merged_stage1_meta_action",
        "is_original_target",
        "match",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(
    path: Path,
    records: list[dict[str, Any]],
    labels: list[str],
    count_matrix: list[list[int]],
    subset: str,
) -> None:
    raw_counts = Counter(record["merged_raw_meta_action"] for record in records)
    pred_counts = Counter(record["merged_stage1_meta_action"] for record in records)
    accuracy = (
        sum(1 for record in records if record["match"]) / len(records) if records else 0.0
    )
    summary = {
        "subset": subset,
        "total": len(records),
        "accuracy": accuracy,
        "labels": labels,
        "raw_counts": dict(raw_counts),
        "stage1_counts": dict(pred_counts),
        "count_matrix": count_matrix,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)


def infer_output_prefix(run_dir: Path, subset: str) -> Path:
    return run_dir / f"stage1_merged_confusion_{subset}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate confusion matrices from stage-1 meta_action outputs with "
            "custom merged labels."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Existing run directory that contains pkl_after_pipeline.",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "original_target"],
        default="all",
        help="Which sample subset to include in the confusion matrix.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=None,
        help=(
            "Output path prefix. The script will append .csv, .json, .png and "
            "_percent.png."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    pkl_root = run_dir / "pkl_after_pipeline"
    if not pkl_root.is_dir():
        raise FileNotFoundError(f"pkl root not found: {pkl_root}")

    output_prefix = (
        args.output_prefix.resolve()
        if args.output_prefix is not None
        else infer_output_prefix(run_dir, args.subset)
    )

    records = filter_records(build_records(pkl_root), args.subset)
    if not records:
        raise ValueError(f"no records found for subset={args.subset}")

    labels = build_labels(records)
    count_matrix = build_count_matrix(records, labels)
    percent_matrix = row_normalize(count_matrix)

    write_records_csv(output_prefix.with_suffix(".csv"), records)
    write_summary(
        output_prefix.with_suffix(".json"),
        records,
        labels,
        count_matrix,
        args.subset,
    )
    save_confusion_png(
        labels,
        count_matrix,
        output_prefix.with_suffix(".png"),
        f"Stage1 Merged Confusion ({args.subset})",
    )
    save_confusion_percent_png(
        labels,
        percent_matrix,
        output_prefix.parent / f"{output_prefix.name}_percent.png",
        f"Stage1 Merged Confusion ({args.subset}, %)",
    )

    print(f"[DONE] records csv: {output_prefix.with_suffix('.csv')}")
    print(f"[DONE] summary json: {output_prefix.with_suffix('.json')}")
    print(f"[DONE] count confusion: {output_prefix.with_suffix('.png')}")
    print(f"[DONE] percent confusion: {output_prefix.parent / f'{output_prefix.name}_percent.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
