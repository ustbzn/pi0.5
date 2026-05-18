#!/usr/bin/env python3
"""Run p2lane + Qwen meta_action refinement and evaluate against raw labels."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = Path("/pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane_test")
DEFAULT_OUTPUT_ROOT = Path("/pcpt/pcpt/data/VLA/MOGO/meta_action_test_pipeline")
P2LANE_SCRIPT = TOOL_DIR / "update_meta_action_by_p2lane_test.py"
ALI_SCRIPT = TOOL_DIR / "ali_api_test.py"

TARGET_META_ACTIONS = {
    "out_of_lane_left_nudge",
    "out_of_lane_right_nudge",
    "in_lane_left_nudge",
    "in_lane_right_nudge",
    "left_lane_change",
    "right_lane_change",
}

ALI_REFINE_META_ACTIONS = {
    "in_lane_left_nudge",
    "in_lane_right_nudge",
    "out_of_lane_right",
    "out_of_lane_left",
}

P2LANE_ARTIFACTS = (
    "p2lane_report.json",
    "p2lane_report_changes.csv",
)
FINAL_ARTIFACTS = (
    "metrics_all.csv",
    "confusion_all.png",
    "confusion_all_percent.png",
    "confusion_original_target.png",
    "confusion_original_target_percent.png",
    "summary.json",
    "report.md",
    "mismatches/mismatches.csv",
)
RESETTABLE_OUTPUT_PATHS = (
    "p2lane_report.json",
    "p2lane_report_changes.csv",
    "ali_refine_log.jsonl",
    "ali_vis",
    "metrics_all.csv",
    "confusion_all.png",
    "confusion_all_percent.png",
    "confusion_original_target.png",
    "confusion_original_target_percent.png",
    "summary.json",
    "report.md",
    "mismatches",
)


def load_pickle(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def decode_path_value(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tobytes"):
        try:
            return value.tobytes().decode("utf-8")
        except Exception:
            return str(value)
    return "" if value is None else str(value)


def import_ali_api_test():
    spec = importlib.util.spec_from_file_location("ali_api_test", ALI_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to import {ALI_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.TARGET_ACTIONS = set(ALI_REFINE_META_ACTIONS)
    return module


def rel_pkl_paths(root: Path) -> list[Path]:
    return sorted(path.relative_to(root) for path in root.rglob("*.pkl"))


def count_pkls(root: Path) -> int:
    if not root.is_dir():
        return 0
    return sum(1 for path in root.rglob("*.pkl") if path.is_file())


def missing_artifacts(output_root: Path, rel_paths: tuple[str, ...]) -> list[str]:
    return [rel_path for rel_path in rel_paths if not (output_root / rel_path).exists()]


def cleanup_output_paths(output_root: Path, rel_paths: tuple[str, ...]) -> None:
    for rel_path in rel_paths:
        path = output_root / rel_path
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def inspect_existing_run(input_dir: Path, output_root: Path) -> dict[str, Any]:
    work_dir = output_root / "pkl_after_pipeline"
    input_pkl_count = count_pkls(input_dir)
    work_pkl_count = count_pkls(work_dir)
    p2lane_missing = missing_artifacts(output_root, P2LANE_ARTIFACTS)
    final_missing = missing_artifacts(output_root, FINAL_ARTIFACTS)
    has_existing_outputs = output_root.exists() and any(output_root.iterdir())
    return {
        "work_dir": work_dir,
        "output_root_exists": output_root.exists(),
        "has_existing_outputs": has_existing_outputs,
        "work_dir_exists": work_dir.is_dir(),
        "input_pkl_count": input_pkl_count,
        "work_pkl_count": work_pkl_count,
        "work_dir_complete": work_dir.is_dir() and work_pkl_count == input_pkl_count,
        "p2lane_missing": p2lane_missing,
        "p2lane_complete": work_dir.is_dir()
        and work_pkl_count == input_pkl_count
        and not p2lane_missing,
        "final_missing": final_missing,
        "final_complete": not final_missing,
        "has_ali_log": (output_root / "ali_refine_log.jsonl").is_file(),
    }


def resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    return (args.output_root / args.run_name).resolve()


def copy_input_tree(input_dir: Path, work_dir: Path, overwrite: bool) -> None:
    if work_dir.exists():
        if not overwrite:
            raise FileExistsError(f"work pkl dir already exists: {work_dir}")
        shutil.rmtree(work_dir)
    shutil.copytree(input_dir, work_dir)


def run_p2lane(work_dir: Path, report_prefix: Path, p2lane_args: list[str]) -> None:
    cmd = [
        sys.executable,
        str(P2LANE_SCRIPT),
        "--input-dir",
        str(work_dir),
        "--inplace",
        "--write-report",
        str(report_prefix),
        *p2lane_args,
    ]
    print("[INFO] running p2lane:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def load_completed_ali_results(log_path: Path) -> tuple[set[str], Counter]:
    completed: set[str] = set()
    counters: Counter = Counter()
    if not log_path.is_file():
        return completed, counters

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                counters["bad_log_line"] += 1
                continue
            status = result.get("status", "unknown")
            rel_path = result.get("rel_path")
            counters[f"existing_{status}"] += 1
            if rel_path and status in {"updated", "kept", "skip"}:
                completed.add(str(rel_path))
    return completed, counters


def original_target_paths(input_dir: Path) -> list[Path]:
    targets: list[Path] = []
    for rel_path in rel_pkl_paths(input_dir):
        data = load_pickle(input_dir / rel_path)
        if data and data.get("meta_action") in TARGET_META_ACTIONS:
            targets.append(rel_path)
    return targets


def ali_target_paths_after_processing(work_dir: Path) -> list[Path]:
    targets: list[Path] = []
    for rel_path in rel_pkl_paths(work_dir):
        data = load_pickle(work_dir / rel_path)
        if data and data.get("meta_action") in ALI_REFINE_META_ACTIONS:
            targets.append(rel_path)
    return targets


def run_ali_refine(
    work_dir: Path,
    target_paths: list[Path],
    output_root: Path,
    api_key: str,
    model: str,
    timeout: int,
    max_retries: int,
    limit: int | None,
    dry_run: bool,
    resume: bool,
) -> tuple[Path, Counter]:
    ali_api = import_ali_api_test()
    client = ali_api.DashScopeVLClient(
        api_key=api_key,
        model=model,
        timeout=timeout,
        max_retries=max_retries,
    )

    log_path = output_root / "ali_refine_log.jsonl"
    vis_dir = output_root / "ali_vis"
    counters: Counter = Counter()
    completed: set[str] = set()
    if resume:
        completed, existing_counters = load_completed_ali_results(log_path)
        counters.update(existing_counters)
        print(
            f"[INFO] resume enabled: skip {len(completed)} completed ali records",
            flush=True,
        )

    paths = [path for path in target_paths if str(path) not in completed]
    if limit is not None:
        paths = paths[:limit]

    log_mode = "a" if resume else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        for index, rel_path in enumerate(paths, start=1):
            pkl_path = work_dir / rel_path
            data = load_pickle(pkl_path)
            current_action = data.get("meta_action") if data else None
            if current_action not in ali_api.TARGET_ACTIONS:
                result = {
                    "status": "skip",
                    "reason": "not_ali_target_after_p2lane",
                    "pkl": str(pkl_path),
                    "current_meta_action": current_action,
                }
                counters["skipped_not_ali_target"] += 1
            else:
                try:
                    result = ali_api.refine_one_pkl(
                        pkl_path=pkl_path,
                        client=client,
                        pkl_root=work_dir,
                        vis_dir=vis_dir,
                        dry_run=dry_run,
                    )
                    counters[result.get("status", "unknown")] += 1
                except Exception as exc:
                    result = {
                        "status": "error",
                        "pkl": str(pkl_path),
                        "error": str(exc),
                        "current_meta_action": current_action,
                    }
                    counters["error"] += 1

            result["rel_path"] = str(rel_path)
            log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            log_file.flush()
            if index % 20 == 0 or result["status"] == "error":
                print(
                    f"[INFO] ali processed {index}/{len(paths)} "
                    f"status={result['status']} rel={rel_path}",
                    flush=True,
                )

    counters["requested"] = len(paths)
    counters["skipped_completed_by_resume"] = len(completed)
    counters["original_target_total"] = len(target_paths)
    return log_path, counters


def build_records(input_dir: Path, work_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for rel_path in rel_pkl_paths(input_dir):
        raw = load_pickle(input_dir / rel_path)
        final = load_pickle(work_dir / rel_path)
        if raw is None or final is None:
            continue
        raw_action = str(raw.get("meta_action") or "")
        final_action = str(final.get("meta_action") or "")
        p2lane_action = str(final.get("temp_meta_action") or final_action)
        image_path = decode_path_value(final.get("cam_front") or raw.get("cam_front"))
        records.append(
            {
                "rel_path": str(rel_path),
                "raw_meta_action": raw_action,
                "p2lane_or_pre_ali_meta_action": p2lane_action,
                "final_meta_action": final_action,
                "is_original_target": raw_action in TARGET_META_ACTIONS,
                "match": raw_action == final_action,
                "cam_front": image_path,
            }
        )
    return records


def confusion_counts(records: list[dict[str, Any]]) -> tuple[list[str], list[list[int]]]:
    labels = sorted(
        {
            str(record["raw_meta_action"])
            for record in records
        }
        | {str(record["final_meta_action"]) for record in records}
    )
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for record in records:
        matrix[index[record["raw_meta_action"]]][index[record["final_meta_action"]]] += 1
    return labels, matrix


def confusion_percentages(
    records: list[dict[str, Any]],
) -> tuple[list[str], list[list[float]]]:
    labels, matrix = confusion_counts(records)
    percent_matrix: list[list[float]] = []
    for row in matrix:
        row_total = sum(row)
        if row_total == 0:
            percent_matrix.append([0.0 for _ in row])
            continue
        percent_matrix.append([(value / row_total) * 100.0 for value in row])
    return labels, percent_matrix


def save_confusion_png(records: list[dict[str, Any]], output_path: Path, title: str) -> None:
    labels, matrix = confusion_counts(records)
    if not labels:
        return

    size = max(8, min(22, 0.55 * len(labels) + 5))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("pred final meta_action")
    ax.set_ylabel("raw meta_action")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    max_value = max((max(row) for row in matrix), default=0)
    threshold = max_value * 0.55
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            if value:
                color = "white" if value >= threshold else "black"
                ax.text(col_idx, row_idx, str(value), ha="center", va="center", color=color)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_confusion_percent_png(
    records: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    labels, matrix = confusion_percentages(records)
    if not labels:
        return

    size = max(8, min(22, 0.55 * len(labels) + 5))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=100.0)
    ax.set_title(title)
    ax.set_xlabel("pred final meta_action")
    ax.set_ylabel("raw meta_action")
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
            )

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Percentage (%)")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rel_path",
        "raw_meta_action",
        "p2lane_or_pre_ali_meta_action",
        "final_meta_action",
        "is_original_target",
        "match",
        "cam_front",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def export_mismatches(
    records: list[dict[str, Any]],
    input_dir: Path,
    work_dir: Path,
    output_root: Path,
) -> Path:
    mismatch_dir = output_root / "mismatches"
    image_dir = mismatch_dir / "front_images"
    pkl_dir = mismatch_dir / "final_pkls"
    mismatches = [record for record in records if not record["match"]]
    image_dir.mkdir(parents=True, exist_ok=True)
    pkl_dir.mkdir(parents=True, exist_ok=True)
    write_csv(mismatch_dir / "mismatches.csv", mismatches)

    missing_images: list[dict[str, Any]] = []
    for index, record in enumerate(mismatches, start=1):
        rel_path = Path(record["rel_path"])
        stem = "__".join(rel_path.with_suffix("").parts)
        image_path = Path(record["cam_front"])
        if image_path.is_file():
            suffix = image_path.suffix or ".jpg"
            shutil.copy2(image_path, image_dir / f"{stem}{suffix}")
        else:
            missing_images.append(
                {
                    "rel_path": record["rel_path"],
                    "cam_front": str(image_path),
                    "reason": "source image not found",
                }
            )
        final_pkl = work_dir / rel_path
        if final_pkl.is_file():
            target_pkl = pkl_dir / rel_path
            target_pkl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_pkl, target_pkl)
        raw_pkl = input_dir / rel_path
        if raw_pkl.is_file():
            target_pkl = mismatch_dir / "raw_pkls" / rel_path
            target_pkl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw_pkl, target_pkl)
        if index % 1000 == 0:
            print(
                f"[INFO] exported mismatches {index}/{len(mismatches)}",
                flush=True,
            )

    if missing_images:
        with (mismatch_dir / "missing_front_images.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=["rel_path", "cam_front", "reason"])
            writer.writeheader()
            writer.writerows(missing_images)
    return mismatch_dir


def accuracy(records: list[dict[str, Any]]) -> float:
    if not records:
        return 0.0
    return sum(1 for record in records if record["match"]) / len(records)


def write_report(
    output_root: Path,
    input_dir: Path,
    work_dir: Path,
    records: list[dict[str, Any]],
    ali_counters: Counter,
    p2lane_summary: dict[str, Any],
) -> None:
    target_records = [record for record in records if record["is_original_target"]]
    mismatches = [record for record in records if not record["match"]]
    target_mismatches = [record for record in target_records if not record["match"]]
    final_counts = Counter(record["final_meta_action"] for record in records)
    raw_counts = Counter(record["raw_meta_action"] for record in records)

    summary = {
        "input_dir": str(input_dir),
        "work_dir": str(work_dir),
        "total_pkl": len(records),
        "original_target_total": len(target_records),
        "accuracy_all": accuracy(records),
        "accuracy_original_target": accuracy(target_records),
        "mismatch_all": len(mismatches),
        "mismatch_original_target": len(target_mismatches),
        "raw_meta_action_counts": dict(raw_counts),
        "final_meta_action_counts": dict(final_counts),
        "ali_counts": dict(ali_counters),
        "p2lane_summary": p2lane_summary,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)

    report = [
        "# Meta Action Test Pipeline Report",
        "",
        f"- input_dir: `{input_dir}`",
        f"- processed_pkl_dir: `{work_dir}`",
        f"- total_pkl: {len(records)}",
        f"- original_target_total: {len(target_records)}",
        f"- accuracy_all: {summary['accuracy_all']:.6f}",
        f"- accuracy_original_target: {summary['accuracy_original_target']:.6f}",
        f"- mismatch_all: {len(mismatches)}",
        f"- mismatch_original_target: {len(target_mismatches)}",
        "",
        "## Artifacts",
        "",
        "- `p2lane_report.json` / `p2lane_report_changes.csv`",
        "- `ali_refine_log.jsonl`",
        "- `metrics_all.csv`",
        "- `confusion_all.png`",
        "- `confusion_all_percent.png`",
        "- `confusion_original_target.png`",
        "- `confusion_original_target_percent.png`",
        "- `mismatches/`",
    ]
    (output_root / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run p2lane, Qwen reset, and meta_action evaluation on MOGO test pkls."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Direct pipeline output directory. When set, this overrides "
            "--output-root/--run-name and enables resume checks on that directory."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Subdirectory name under --output-root. Ignored when --output-dir is set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Compatibility flag. Existing output directories are now inspected "
            "automatically and completed ali records are skipped when possible."
        ),
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help=(
            "Do not call p2lane or ali; only rebuild metrics, confusion matrices, "
            "mismatch exports, and reports from an existing run directory."
        ),
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen3-vl-plus")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--ali-limit", type=int, default=None)
    parser.add_argument("--ali-dry-run", action="store_true")
    parser.add_argument(
        "--p2lane-arg",
        action="append",
        default=[],
        help="Extra argument passed to update_meta_action_by_p2lane_test.py. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_root = resolve_output_root(args)
    work_dir = output_root / "pkl_after_pipeline"

    if not input_dir.is_dir():
        print(f"[ERROR] input dir does not exist: {input_dir}", file=sys.stderr)
        return 2

    api_key = args.api_key or os.getenv("DASHSCOPE_API_KEY")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] output_root: {output_root}", flush=True)
    inspect = inspect_existing_run(input_dir, output_root)
    print(
        "[INFO] existing outputs: "
        f"work_pkl={inspect['work_pkl_count']}/{inspect['input_pkl_count']} "
        f"p2lane_complete={inspect['p2lane_complete']} "
        f"final_complete={inspect['final_complete']} "
        f"ali_log={inspect['has_ali_log']}",
        flush=True,
    )
    if inspect["p2lane_missing"]:
        print(
            f"[INFO] missing p2lane artifacts: {', '.join(inspect['p2lane_missing'])}",
            flush=True,
        )
    if inspect["final_missing"]:
        print(
            f"[INFO] missing final artifacts: {', '.join(inspect['final_missing'])}",
            flush=True,
        )

    if args.report_only:
        if not inspect["work_dir_complete"]:
            print(
                "[ERROR] report-only requires a complete existing work dir matching "
                f"the input dataset: {work_dir}",
                file=sys.stderr,
            )
            return 2
        print("[INFO] report-only enabled: skip copy, p2lane, and ali", flush=True)
    else:
        rebuild_from_p2lane = args.overwrite or not inspect["p2lane_complete"]
        if rebuild_from_p2lane:
            if args.overwrite:
                print(
                    "[INFO] overwrite enabled: rebuild work dir and downstream outputs",
                    flush=True,
                )
            elif inspect["has_existing_outputs"]:
                print(
                    "[INFO] detected incomplete existing run: rebuild from p2lane "
                    "boundary and clean stale downstream outputs",
                    flush=True,
                )
            cleanup_output_paths(output_root, RESETTABLE_OUTPUT_PATHS)
            copy_input_tree(input_dir, work_dir, overwrite=work_dir.exists())
            run_p2lane(work_dir, output_root / "p2lane_report", args.p2lane_arg)
        else:
            print(
                f"[INFO] auto resume: reuse existing work dir and p2lane outputs: {work_dir}",
                flush=True,
            )

    p2lane_report_prefix = output_root / "p2lane_report"
    if args.report_only:
        pass
    elif not p2lane_report_prefix.with_suffix(".json").is_file():
        print("[ERROR] missing p2lane report after preparation", file=sys.stderr)
        return 2

    p2lane_summary_path = p2lane_report_prefix.with_suffix(".json")
    p2lane_summary = {}
    if p2lane_summary_path.is_file():
        p2lane_summary = json.loads(p2lane_summary_path.read_text(encoding="utf-8"))

    original_targets = original_target_paths(input_dir)
    print(
        f"[INFO] original_target_meta_action count: {len(original_targets)}",
        flush=True,
    )
    target_paths = ali_target_paths_after_processing(work_dir)
    print(
        f"[INFO] ali_target_after_processing count: {len(target_paths)}",
        flush=True,
    )
    completed_ali, existing_ali_counters = load_completed_ali_results(
        output_root / "ali_refine_log.jsonl"
    )
    pending_ali = [path for path in target_paths if str(path) not in completed_ali]
    print(
        f"[INFO] ali completed_in_log: {len(completed_ali)} pending: {len(pending_ali)}",
        flush=True,
    )

    if (
        not args.report_only
        and not args.overwrite
        and inspect["p2lane_complete"]
        and not pending_ali
        and inspect["final_complete"]
    ):
        print("[DONE] output directory already contains all expected artifacts", flush=True)
        return 0

    if args.report_only:
        ali_counters = existing_ali_counters
        ali_counters["report_only"] = 1
    elif not pending_ali:
        print("[INFO] auto resume: ali stage already complete", flush=True)
        ali_counters = existing_ali_counters
        ali_counters["requested"] = 0
        ali_counters["skipped_completed_by_resume"] = len(completed_ali)
        ali_counters["original_target_total"] = len(target_paths)
        ali_counters["auto_resume_no_pending"] = 1
    else:
        if not api_key:
            print("[ERROR] DASHSCOPE_API_KEY or --api-key is required", file=sys.stderr)
            return 2
        _, ali_counters = run_ali_refine(
            work_dir=work_dir,
            target_paths=target_paths,
            output_root=output_root,
            api_key=api_key,
            model=args.model,
            timeout=args.timeout,
            max_retries=args.max_retries,
            limit=args.ali_limit,
            dry_run=args.ali_dry_run,
            resume=(output_root / "ali_refine_log.jsonl").is_file(),
        )

    records = build_records(input_dir, work_dir)
    write_csv(output_root / "metrics_all.csv", records)
    save_confusion_png(records, output_root / "confusion_all.png", "All Samples")
    save_confusion_percent_png(
        records,
        output_root / "confusion_all_percent.png",
        "All Samples (%)",
    )
    target_records = [record for record in records if record["is_original_target"]]
    save_confusion_png(
        target_records,
        output_root / "confusion_original_target.png",
        "Original Target Samples",
    )
    save_confusion_percent_png(
        target_records,
        output_root / "confusion_original_target_percent.png",
        "Original Target Samples (%)",
    )
    export_mismatches(records, input_dir, work_dir, output_root)
    write_report(output_root, input_dir, work_dir, records, ali_counters, p2lane_summary)

    print(f"[DONE] report: {output_root / 'report.md'}", flush=True)
    print(f"[DONE] summary: {output_root / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
