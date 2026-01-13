import argparse
import os
import cv2
from tqdm import tqdm
"""
python /D2/NXT/vlm/miaobinqi/Senna/mogo_test/merge_images.py \
    --img-dir /D2/NXT/vlm/miaobinqi/Senna/mogo_test/mogo_vis/2O-bags/20251210141637_test_nusc/pic \
    --out /D2/NXT/vlm/miaobinqi/Senna/mogo_test/mogo_vis/2O-bags/20251210141637_test_nusc/output.mp4 \
    --fps 8
"""
def main():
    parser = argparse.ArgumentParser(description="将图片序列按顺序合并成视频")
    parser.add_argument(
        "--img-dir",
        default="/D2/NXT/vlm/miaobinqi/output_eval/senna/eval_nuscene_all_fft_0915_1/images",
        help="待合成图片所在目录",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="输出视频路径（默认与图片目录同级、名为 output.mp4）",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="仅使用前 N 张图片进行合并；不填则使用全部图片",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="输出视频帧率（默认 10）",
    )
    args = parser.parse_args()

    img_dir = args.img_dir
    out_dir = os.path.dirname(img_dir)
    out_path = args.out or os.path.join(out_dir, "output.mp4")

    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
    img_files.sort()

    if not img_files:
        raise ValueError("目录里没有找到图片！")

    if args.num_images is not None:
        if args.num_images <= 0:
            raise ValueError("--num-images 必须大于 0")
        img_files = img_files[: args.num_images]

    first_frame = cv2.imread(os.path.join(img_dir, img_files[0]))
    h, w, _ = first_frame.shape

    # 创建输出目录
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    print(f"开始合并 {len(img_files)} 张图片为视频...")
    for file in tqdm(img_files, desc="处理图片", unit="张"):
        frame = cv2.imread(os.path.join(img_dir, file))
        if frame is None:
            continue
        out.write(frame)

    out.release()
    print(f"视频已保存: {out_path}")


if __name__ == "__main__":
    main()