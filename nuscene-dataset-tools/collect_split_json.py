import argparse
import shutil
from pathlib import Path
from typing import List
from tqdm import tqdm
"""
python /D2/NXT/vlm/miaobinqi/Senna/data_tools_pi0/collect_split_json.py \
    --base-dir /D2/NXT/vlm/miaobinqi/datasets/nuscene-train-pi0-all-0109-1 \
    --out-dir  /D2/NXT/vlm/miaobinqi/DriveMoE/datasets/nuscene-0109
"""

def gather_files(base_dir: Path, exclude_names: List[str]):
    """Return list of json file paths under numbered subfolders."""
    json_files = []
    for sub in sorted(base_dir.iterdir()):
        if not sub.is_dir():
            continue
        # only consider subfolders named with digits (1..10)
        if not sub.name.isdigit():
            continue
        for fp in sub.glob("*.json"):
            if fp.name in exclude_names:
                continue
            json_files.append(fp)
    return json_files


def move_files(json_files: List[Path], out_train_dir: Path, out_val_dir: Path):
    """Copy files by suffix into train/val folders."""
    train_cnt = 0
    val_cnt = 0
    out_train_dir.mkdir(parents=True, exist_ok=True)
    out_val_dir.mkdir(parents=True, exist_ok=True)

    for fp in tqdm(json_files, desc="Copying json files", unit="file"):
        lower_name = fp.name.lower()
        if lower_name.endswith("_train.json"):
            dst_name = fp.name[:-len("_train.json")] + ".json"
            dst = out_train_dir / dst_name
            shutil.copy2(str(fp), dst)
            train_cnt += 1
        elif lower_name.endswith("_val.json"):
            dst_name = fp.name[:-len("_val.json")] + ".json"
            dst = out_val_dir / dst_name
            shutil.copy2(str(fp), dst)
            val_cnt += 1
        else:
            continue
    return train_cnt, val_cnt


def main():
    parser = argparse.ArgumentParser(description="Move per-sample json into train/val folders.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Root directory containing numbered subfolders (1..10).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to store train/val subfolders.",
    )
    args = parser.parse_args()

    exclude_names = {"drivepi_nusc_train.json", "drivepi_nusc_val.json"}

    json_files = gather_files(args.base_dir, exclude_names)
    if not json_files:
        raise RuntimeError(f"No json files found under {args.base_dir}")

    # Ensure output root exists
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_train_dir = args.out_dir / "train_json"
    out_val_dir = args.out_dir / "val_json"
    train_cnt, val_cnt = move_files(json_files, out_train_dir, out_val_dir)

    print(f"Moved train samples: {train_cnt} -> {out_train_dir}")
    print(f"Moved val samples: {val_cnt} -> {out_val_dir}")


if __name__ == "__main__":
    main()

