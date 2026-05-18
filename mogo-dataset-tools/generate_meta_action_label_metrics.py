#!/usr/bin/env python3
"""Generate per-label meta_action precision/recall from metrics_all.csv."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_METRICS_CSV = Path(
    "/pcpt/pcpt/data/VLA/MOGO/meta_action_test_pipeline/"
    "20260511_after_processed_ali/metrics_all.csv"
)


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_records(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_records(
    records: list[dict[str, str]],
    subset: str,
) -> list[dict[str, str]]:
    if subset == "all":
        return records
    if subset == "original_target":
        return [
            record
            for record in records
            if parse_bool(record.get("is_original_target", "False"))
        ]
    raise ValueError(f"unsupported subset: {subset}")


def build_labels(records: list[dict[str, str]]) -> list[str]:
    labels = {
        str(record.get("raw_meta_action", "")).strip()
        for record in records
        if str(record.get("raw_meta_action", "")).strip()
    }
    labels.update(
        {
            str(record.get("final_meta_action", "")).strip()
            for record in records
            if str(record.get("final_meta_action", "")).strip()
        }
    )
    return sorted(labels)


def compute_label_metrics(records: list[dict[str, str]]) -> list[dict[str, object]]:
    labels = build_labels(records)
    truth_counts = Counter(
        str(record.get("raw_meta_action", "")).strip()
        for record in records
        if str(record.get("raw_meta_action", "")).strip()
    )
    pred_counts = Counter(
        str(record.get("final_meta_action", "")).strip()
        for record in records
        if str(record.get("final_meta_action", "")).strip()
    )
    true_positive_counts = Counter()

    for record in records:
        truth = str(record.get("raw_meta_action", "")).strip()
        pred = str(record.get("final_meta_action", "")).strip()
        if truth and pred and truth == pred:
            true_positive_counts[truth] += 1

    rows: list[dict[str, object]] = []
    for label in labels:
        truth_count = truth_counts[label]
        pred_count = pred_counts[label]
        true_positive = true_positive_counts[label]
        recall = true_positive / truth_count if truth_count else 0.0
        precision = true_positive / pred_count if pred_count else 0.0
        rows.append(
            {
                "meta_action": label,
                "truth_count": truth_count,
                "pred_count": pred_count,
                "true_positive": true_positive,
                "recall": recall,
                "precision": precision,
            }
        )
    return rows


def infer_default_output_path(metrics_csv: Path, subset: str) -> Path:
    suffix = "all" if subset == "all" else "original_target"
    return metrics_csv.parent / f"label_metrics_{suffix}.csv"


def write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "meta_action",
        "truth_count",
        "pred_count",
        "true_positive",
        "recall",
        "precision",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "meta_action": row["meta_action"],
                    "truth_count": row["truth_count"],
                    "pred_count": row["pred_count"],
                    "true_positive": row["true_positive"],
                    "recall": f"{float(row['recall']):.6f}",
                    "precision": f"{float(row['precision']):.6f}",
                }
            )


def print_table(rows: list[dict[str, object]], subset: str) -> None:
    headers = [
        "meta_action",
        "truth_count",
        "pred_count",
        "true_positive",
        "recall",
        "precision",
    ]
    display_rows = [
        [
            str(row["meta_action"]),
            str(row["truth_count"]),
            str(row["pred_count"]),
            str(row["true_positive"]),
            f"{float(row['recall']):.6f}",
            f"{float(row['precision']):.6f}",
        ]
        for row in rows
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in display_rows))
        for index, header in enumerate(headers)
    ]

    print(f"[INFO] subset={subset}")
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in display_rows:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-label recall and precision from meta_action metrics."
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=DEFAULT_METRICS_CSV,
        help="Existing metrics_all.csv generated by run_meta_action_test_pipeline.py.",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "original_target"],
        default="all",
        help="Which subset of metrics_all.csv to evaluate.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV path. Defaults to label_metrics_<subset>.csv next to the metrics CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics_csv = args.metrics_csv.resolve()
    if not metrics_csv.is_file():
        raise FileNotFoundError(f"metrics csv not found: {metrics_csv}")

    records = filter_records(load_records(metrics_csv), args.subset)
    if not records:
        raise ValueError(f"no records available for subset={args.subset}")

    rows = compute_label_metrics(records)
    output_csv = (
        args.output_csv.resolve()
        if args.output_csv is not None
        else infer_default_output_path(metrics_csv, args.subset)
    )
    write_csv_rows(output_csv, rows)
    print_table(rows, args.subset)
    print(f"[DONE] saved label metrics to: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
