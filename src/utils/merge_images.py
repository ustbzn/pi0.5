"""
Batch convert scene image folders into mp4 videos or gifs.

Default behavior:
    - Scan `visualizations/dual` for scene folders
    - For each scene, merge all .png/.jpg images (natural-sorted) into a video/gif
    - Output files into `visualizations` with filenames `<scene>.mp4` or `<scene>.gif`

Usage examples:
    # convert all scenes under visualizations/dual
    python merge_images.py

    # convert specific scenes
    python merge_images.py --scene Accident_Town05_Route218_Weather10 --scene SceneB

    # customize fps and output directory
    python merge_images.py --fps 8 --out-dir /tmp/videos

    # generate gifs instead of videos
    python merge_images.py --format gif --fps 5

    # generate both mp4 and gif
    python merge_images.py --format both --fps 5
"""

import argparse
import re
from pathlib import Path

import cv2
from PIL import Image
from tqdm import tqdm


DEFAULT_DUAL_DIR = Path("/D2/NXT/vlm/miaobinqi/DriveMoE/visualizations/dual")
DEFAULT_OUT_DIR = Path("/D2/NXT/vlm/miaobinqi/DriveMoE/visualizations/videos")


def natural_sort_key(text: str):
    # Split digits to achieve natural ordering: img2 < img10
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", text)]


def collect_images(scene_dir: Path):
    exts = {".png", ".jpg", ".jpeg"}
    imgs = [p for p in scene_dir.iterdir() if p.suffix.lower() in exts and p.is_file()]
    imgs.sort(key=lambda p: natural_sort_key(p.name))
    return imgs


def scene_to_video(scene_dir: Path, out_dir: Path, fps: int = 10, codec: str = "mp4v"):
    images = collect_images(scene_dir)
    if not images:
        print(f"[skip] No images found in {scene_dir}")
        return

    first = cv2.imread(str(images[0]))
    if first is None:
        print(f"[skip] Unable to read first image in {scene_dir}")
        return
    height, width = first.shape[:2]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_dir.name}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))

    for img_path in tqdm(images, desc=f"{scene_dir.name}", unit="frame"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[warn] failed to read {img_path}, skipping")
            continue
        if frame.shape[0] != height or frame.shape[1] != width:
            frame = cv2.resize(frame, (width, height))
        writer.write(frame)

    writer.release()
    print(f"[ok] {scene_dir.name} -> {out_path}")


def scene_to_gif(scene_dir: Path, out_dir: Path, fps: int = 10):
    images = collect_images(scene_dir)
    if not images:
        print(f"[skip] No images found in {scene_dir}")
        return

    first = cv2.imread(str(images[0]))
    if first is None:
        print(f"[skip] Unable to read first image in {scene_dir}")
        return
    height, width = first.shape[:2]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_dir.name}.gif"
    duration_ms = max(1, round(1000 / fps))
    palette = getattr(getattr(Image, "Palette", Image), "ADAPTIVE")
    dither = getattr(getattr(Image, "Dither", Image), "FLOYDSTEINBERG")

    frames = []
    for img_path in tqdm(images, desc=f"{scene_dir.name}", unit="frame"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"[warn] failed to read {img_path}, skipping")
            continue
        # if frame.shape[0] != height or frame.shape[1] != width:
        frame = cv2.resize(frame, (960, 540))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(
            Image.fromarray(rgb).convert(
                "P",
                palette=palette,
                colors=256,
                dither=dither,
            )
        )

    if not frames:
        print(f"[skip] No readable images found in {scene_dir}")
        return

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"[ok] {scene_dir.name} -> {out_path}")


def find_scenes(dual_dir: Path):
    return [p for p in dual_dir.iterdir() if p.is_dir()]


def parse_args():
    parser = argparse.ArgumentParser(description="Merge scene images into videos or gifs.")
    parser.add_argument(
        "--dual-dir",
        type=Path,
        default=DEFAULT_DUAL_DIR,
        help=f"Scene images root dir (default: {DEFAULT_DUAL_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for generated files (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Scene name to process; repeatable. If omitted, process all scenes.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=6,
        help="Frames per second for output files (default: 6)",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec (default: mp4v). Common alternatives: avc1, H264",
    )
    parser.add_argument(
        "--format",
        choices=["mp4", "gif", "both"],
        default="mp4",
        help="Output format: mp4, gif, or both (default: mp4)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")

    dual_dir: Path = args.dual_dir
    out_dir: Path = args.out_dir

    if args.scene:
        targets = []
        for name in args.scene:
            path = dual_dir / name
            if path.is_dir():
                targets.append(path)
            else:
                print(f"[warn] scene not found: {path}")
    else:
        targets = find_scenes(dual_dir)

    if not targets:
        print(f"[error] No scenes to process under {dual_dir}")
        return

    for scene_dir in tqdm(targets, desc="Processing scenes", unit="scene"):
        if args.format in {"mp4", "both"}:
            scene_to_video(scene_dir, out_dir, fps=args.fps, codec=args.codec)
        if args.format in {"gif", "both"}:
            scene_to_gif(scene_dir, out_dir, fps=args.fps)


if __name__ == "__main__":
    main()