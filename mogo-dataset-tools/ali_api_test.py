#!/usr/bin/env python3
"""Refine MOGO meta_action labels with DashScope Qwen-VL.

The script filters pkl files whose ``meta_action`` indicates a nudge or
out-of-lane maneuver, asks Qwen-VL whether the front camera image contains an
obstacle affecting ego-vehicle passage, stores the original label in
``temp_meta_action``, and updates ``meta_action`` according to the label rules.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pickle
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


DEFAULT_PKL_ROOT = Path(
    "/pcpt/pcpt/data/VLA/MOGO/pkl_rout_action_scene_hdmap_lane"
)
DEFAULT_MODEL = "qwen3-vl-plus"
DEFAULT_VIS_DIR = Path("ali_api_refine_images")
TARGET_ACTIONS = {
    "in_lane_left_nudge",
    "in_lane_right_nudge",
    "out_of_lane_right",
    "out_of_lane_left",
}
ACTION_WITH_OBSTACLE = {
    "in_lane_left_nudge": "in_lane_left_nudge",
    "in_lane_right_nudge": "in_lane_right_nudge",
    "out_of_lane_left": "out_of_lane_left_nudge",
    "out_of_lane_right": "out_of_lane_right_nudge",
}
ACTION_WITHOUT_OBSTACLE = {
    "in_lane_left_nudge": "lane_keeping_and_centering",
    "in_lane_right_nudge": "lane_keeping_and_centering",
    "out_of_lane_left": "left_lane_change",
    "out_of_lane_right": "right_lane_change",
}


PROMPT_TEMPLATE = """你是自动驾驶数据标注质检员。当前 pkl 的原始 meta_action 是：{meta_action}。

请只根据输入的当前帧前置相机图片，判断该标签是否应该按“存在障碍物影响通行/需要避让”处理。

判断为有障碍物的情况包括：施工围挡、锥桶、水马、静止或低速占道车辆、行人/骑行者/三轮车、路面大件物体、明显占据自车车道并阻碍通行的临时设施。
特别规则：如果原始 meta_action 是 in_lane_left_nudge 或 in_lane_right_nudge，且图像中自车相邻车道存在行人、骑行者、三轮车等可能影响自车横向安全距离的动态障碍物，即使这些障碍物没有完全占据自车当前车道，也应判断为有障碍物，以便保留原有 nudge 标签。
判断为无障碍物的情况包括：正常车道线、阴影、路沿、远处未占据自车车道的车辆、相邻车道车辆、可正常跟车通过的前车、图片不清但无法确认占道障碍。

输出必须是严格 JSON，不要输出解释，不要使用 Markdown：
{{"obstacle_affects_passage": true 或 false, "confidence": 0到1之间的小数}}
"""


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


def get_image_format(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext == ".png":
        return "png"
    if ext in {".jpg", ".jpeg"}:
        return "jpeg"
    if ext == ".gif":
        return "gif"
    return "jpeg"


def encode_image_to_base64(image_path: Path) -> str:
    with image_path.open("rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/{get_image_format(image_path)};base64,{encoded}"


def parse_bool_response(text: str) -> tuple[bool, float | None]:
    cleaned = text.strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    data = json.loads(cleaned)
    obstacle = data.get("obstacle_affects_passage")
    if not isinstance(obstacle, bool):
        raise ValueError(f"invalid obstacle_affects_passage: {obstacle!r}")
    confidence = data.get("confidence")
    if confidence is not None:
        confidence = float(confidence)
    return obstacle, confidence


def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        for char in paragraph:
            candidate = current + char
            bbox = font.getbbox(candidate)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = char
        lines.append(current)
    return lines


def save_result_image(
    image_path: Path,
    output_path: Path,
    pkl_path: Path,
    old_action: str,
    new_action: str,
    obstacle: bool,
    confidence: float | None,
    raw_answer: str,
) -> None:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        font = ImageFont.load_default()
        padding = 12
        line_gap = 4
        text = "\n".join(
            [
                f"pkl: {pkl_path}",
                f"image: {image_path}",
                f"meta_action: {old_action} -> {new_action}",
                f"obstacle_affects_passage: {obstacle}",
                f"confidence: {confidence}",
                f"qwen_raw: {raw_answer}",
            ]
        )
        max_text_width = image.width - padding * 2
        lines = wrap_text(text, font, max_text_width)
        line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
        panel_height = padding * 2 + len(lines) * (line_height + line_gap)
        canvas = Image.new("RGB", (image.width, image.height + panel_height), "white")
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle(
            [(0, image.height), (image.width, image.height + panel_height)],
            fill=(255, 255, 255),
        )
        y = image.height + padding
        for line in lines:
            draw.text((padding, y), line, fill=(0, 0, 0), font=font)
            y += line_height + line_gap
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path, quality=95)


class DashScopeVLClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.max_retries = max_retries

    def infer_obstacle(
        self, image_path: Path, meta_action: str
    ) -> tuple[bool, float | None, str]:
        prompt = PROMPT_TEMPLATE.format(meta_action=meta_action)
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": encode_image_to_base64(image_path)},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0,
            "stream": False,
            "extra_body": {"enable_thinking": False},
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            request = urllib.request.Request(
                self.url, data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    result = json.loads(response.read().decode("utf-8"))
                answer = result["choices"][0]["message"]["content"]
                obstacle, confidence = parse_bool_response(answer)
                return obstacle, confidence, answer
            except (urllib.error.URLError, KeyError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"DashScope inference failed for {image_path}: {last_error}")


def iter_target_pkls(root: Path) -> list[Path]:
    paths: list[Path] = []
    for pkl_path in root.rglob("*.pkl"):
        try:
            with pkl_path.open("rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            print(f"[WARN] skip unreadable pkl {pkl_path}: {exc}", file=sys.stderr)
            continue
        if isinstance(data, dict) and data.get("meta_action") in TARGET_ACTIONS:
            paths.append(pkl_path)
    return sorted(paths)


def refine_one_pkl(
    pkl_path: Path,
    client: DashScopeVLClient,
    pkl_root: Path,
    vis_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    with pkl_path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        return {"status": "skip", "reason": "not_dict", "pkl": str(pkl_path)}

    old_action = data.get("meta_action")
    if old_action not in TARGET_ACTIONS:
        return {"status": "skip", "reason": "not_target", "pkl": str(pkl_path)}

    image_path = Path(decode_path_value(data.get("cam_front")))
    if not image_path.is_file():
        raise FileNotFoundError(f"cam_front image not found: {image_path}")

    obstacle, confidence, raw_answer = client.infer_obstacle(image_path, old_action)
    new_action = (
        ACTION_WITH_OBSTACLE[old_action]
        if obstacle
        else ACTION_WITHOUT_OBSTACLE[old_action]
    )

    vis_path = None
    if vis_dir is not None:
        try:
            relative = pkl_path.relative_to(pkl_root)
        except ValueError:
            relative = pkl_path
        vis_path = vis_dir / relative.with_suffix(".jpg")
        save_result_image(
            image_path=image_path,
            output_path=vis_path,
            pkl_path=pkl_path,
            old_action=old_action,
            new_action=new_action,
            obstacle=obstacle,
            confidence=confidence,
            raw_answer=raw_answer,
        )

    if not dry_run:
        data["temp_meta_action"] = old_action
        data["meta_action"] = new_action
        with pkl_path.open("wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "status": "updated" if old_action != new_action else "kept",
        "pkl": str(pkl_path),
        "image": str(image_path),
        "old_meta_action": old_action,
        "new_meta_action": new_action,
        "obstacle_affects_passage": obstacle,
        "confidence": confidence,
        "raw_answer": raw_answer,
        "result_image": str(vis_path) if vis_path else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Qwen-VL to refine MOGO pkl meta_action labels."
    )
    parser.add_argument("--pkl-root", type=Path, default=DEFAULT_PKL_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--log", type=Path, default=Path("ali_api_refine_log.jsonl"))
    parser.add_argument("--vis-dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("DASHSCOPE_API_KEY is required", file=sys.stderr)
        return 2

    target_pkls = iter_target_pkls(args.pkl_root)
    if args.limit is not None:
        target_pkls = target_pkls[: args.limit]
    print(f"[INFO] target pkl count: {len(target_pkls)}")

    client = DashScopeVLClient(
        api_key=api_key,
        model=args.model,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    args.log.parent.mkdir(parents=True, exist_ok=True)
    ok = 0
    failed = 0
    with args.log.open("a", encoding="utf-8") as log_file:
        for idx, pkl_path in enumerate(target_pkls, 1):
            try:
                result = refine_one_pkl(
                    pkl_path,
                    client=client,
                    pkl_root=args.pkl_root,
                    vis_dir=None if args.no_vis else args.vis_dir,
                    dry_run=args.dry_run,
                )
                ok += 1
                print(
                    "[{}/{}] {} {} -> {} obstacle={} conf={} vis={}".format(
                        idx,
                        len(target_pkls),
                        result["pkl"],
                        result.get("old_meta_action"),
                        result.get("new_meta_action"),
                        result.get("obstacle_affects_passage"),
                        result.get("confidence"),
                        result.get("result_image"),
                    ),
                    flush=True,
                )
            except Exception as exc:
                failed += 1
                result = {"status": "error", "pkl": str(pkl_path), "error": str(exc)}
                print(f"[ERROR] {pkl_path}: {exc}", file=sys.stderr, flush=True)
            log_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            log_file.flush()

    print(f"[DONE] processed={ok} failed={failed} log={args.log}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
