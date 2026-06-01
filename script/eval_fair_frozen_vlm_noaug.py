import argparse
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import einops
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.agent.dataset import Bench2DriveDataset
from src.model.DrivePi0.drivepi0 import DrivePiZero
from src.model.DrivePi0.processing import VLAProcessor
from src.utils.metric import get_action_accuracy, get_bucketed_action_accuracy


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


CAM_NAMES = ["front", "right", "left", "back"]
CAM_LABELS = [
    "current front view",
    "current right view",
    "current left view",
    "current rear view",
    "previous front view",
    "previous right view",
    "previous left view",
    "previous rear view",
]
BUCKET_ORDER = ["straight", "turn", "static", "high_curvature"]
BUCKET_WEIGHTS = {
    "straight": 1.0,
    "turn": 3.0,
    "static": 2.0,
    "high_curvature": 4.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a fair_frozen_vlm checkpoint with val image augmentation disabled."
    )
    parser.add_argument(
        "--config",
        default="config/train/DrivePi0/fair_frozen_vlm.yaml",
        help="Training config used to build the model and val dataset.",
    )
    parser.add_argument(
        "--checkpoint",
        default="logs/train/drive-pi0/2026-05-11_15-33_fair_frozen_vlm/checkpoint/step18700.pt",
        help="Checkpoint path to evaluate.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--image-augment",
        action="store_true",
        help="Enable val image augmentation. Default is disabled.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional debug limit. Omit for full validation.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for evaluation.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model: DrivePiZero, checkpoint_path: str):
    data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state_dict = {
        key.replace("_orig_mod.", ""): value
        for key, value in data["model"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    log.info("Loaded checkpoint: %s", checkpoint_path)


def build_images(batch):
    image_list = []
    for cam in CAM_NAMES:
        img = einops.rearrange(batch[f"image_{cam}"], "B H W C -> B C H W")
        image_list.append(img.unsqueeze(1))
    for cam in CAM_NAMES:
        img = einops.rearrange(batch[f"image_{cam}_time"], "B H W C -> B C H W")
        image_list.append(img.unsqueeze(1))
    return torch.cat(image_list, dim=1)


def preprocess_batch(batch, cfg, model, processor, dtype, device):
    state = batch["state"]
    trajectory = batch["trajectory"].squeeze(1)
    texts = batch["language_instruction"]
    images = build_images(batch)

    model_inputs = processor(text=texts, images=images, cam_labels=CAM_LABELS)
    causal_mask, vlm_position_ids, state_position_ids, action_position_ids = (
        model.build_causal_mask_and_position_ids(model_inputs["attention_mask"], dtype)
    )
    image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
        causal_mask
    )

    inputs = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"].to(dtype),
        "image_text_proprio_mask": image_text_proprio_mask,
        "action_mask": action_mask,
        "vlm_position_ids": vlm_position_ids,
        "proprio_position_ids": state_position_ids,
        "action_position_ids": action_position_ids,
        "proprios": state.to(dtype),
    }
    inputs = {key: value.to(device) for key, value in inputs.items()}
    gt_trajectory = trajectory.to(device=device, dtype=dtype)
    bucket_ids = batch.get("trajectory_bucket")
    if bucket_ids is not None:
        bucket_ids = bucket_ids.long().to(device)
    return inputs, gt_trajectory, bucket_ids


def format_results(
    eval_l1_loss,
    eval_accuracy,
    eval_thresholds,
    bucket_l1_sums,
    bucket_acc_sums,
    bucket_counts,
):
    log_msg = f"Eval | l1 Loss: {eval_l1_loss:.3f}"
    for threshold, accuracy in zip(eval_thresholds, eval_accuracy):
        log_msg += f" | acc thres {threshold}: {accuracy:.3f}"

    for bucket_name in BUCKET_ORDER:
        count = bucket_counts.get(bucket_name, 0)
        if count <= 0:
            continue
        bucket_l1 = bucket_l1_sums[bucket_name] / count
        log_msg += f" | {bucket_name}_l1: {bucket_l1:.3f}"
        for acc_key in sorted(bucket_acc_sums.get(bucket_name, {})):
            bucket_acc = bucket_acc_sums[bucket_name][acc_key] / count
            log_msg += f" | {bucket_name}_{acc_key}: {bucket_acc:.3f}"

    weighted_l1_num = 0.0
    weighted_total = 0.0
    weighted_acc_nums = defaultdict(float)
    for bucket_name, weight in BUCKET_WEIGHTS.items():
        count = bucket_counts.get(bucket_name, 0)
        if count <= 0:
            continue
        weighted_l1_num += bucket_l1_sums[bucket_name] * weight
        weighted_total += count * weight
        for acc_key, value in bucket_acc_sums.get(bucket_name, {}).items():
            weighted_acc_nums[acc_key] += value * weight

    if weighted_total > 0:
        log_msg += f" | weighted_l1: {weighted_l1_num / weighted_total:.3f}"
        for acc_key in sorted(weighted_acc_nums):
            log_msg += (
                f" | weighted_{acc_key}: "
                f"{weighted_acc_nums[acc_key] / weighted_total:.3f}"
            )
    return log_msg


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    seed = cfg.get("seed", 42) if args.seed is None else args.seed
    set_seed(seed)

    device = torch.device(args.device)
    dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
    batch_size = args.batch_size or cfg.per_device_batch_size
    num_workers = cfg.num_workers if args.num_workers is None else args.num_workers

    cfg.data.val.image_augment = bool(args.image_augment)
    cfg.data.val.state_only = False

    log.info("Config: %s", args.config)
    log.info("Checkpoint: %s", args.checkpoint)
    log.info("Val image_augment: %s", cfg.data.val.image_augment)
    log.info("Batch size: %s, num_workers: %s", batch_size, num_workers)
    log.info("Device: %s, dtype: %s, seed: %s", device, dtype, seed)

    model = DrivePiZero(cfg, use_ddp=False)
    load_checkpoint(model, args.checkpoint)
    model.freeze_all_weights()
    model.to(dtype=dtype)
    model.to(device)
    model.eval()
    if args.compile:
        model.vision_tower.forward = torch._dynamo.disable(model.vision_tower.forward)
        model = torch.compile(model, mode="default")
        log.info("Compiled model for evaluation")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_path,
        padding_side="right",
    )
    processor = VLAProcessor(
        tokenizer,
        num_image_tokens=cfg.vision.config.num_image_tokens,
        max_seq_len=cfg.max_seq_len,
        tokenizer_padding=cfg.tokenizer_padding,
    )

    val_dataset = Bench2DriveDataset(cfg.data.val).dataset
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    log.info("Val samples: %s", len(val_dataset))

    eval_thresholds = cfg.eval_thresholds
    eval_accuracy = torch.zeros(len(eval_thresholds), device=device)
    eval_l1_loss = torch.tensor(0.0, device=device)
    bucket_l1_sums = defaultdict(float)
    bucket_acc_sums = defaultdict(lambda: defaultdict(float))
    bucket_counts = defaultdict(int)
    actual_batches = 0

    with torch.inference_mode():
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            if args.max_batches is not None and actual_batches >= args.max_batches:
                break
            actual_batches += 1
            inputs, gt_trajectory, bucket_ids = preprocess_batch(
                batch,
                cfg,
                model,
                processor,
                dtype,
                device,
            )
            preds = model.infer_action(**inputs)
            eval_accuracy += get_action_accuracy(
                gt_trajectory,
                preds,
                eval_thresholds,
            )
            eval_l1_loss += torch.nn.functional.l1_loss(preds, gt_trajectory)

            if bucket_ids is not None:
                bucket_results = get_bucketed_action_accuracy(
                    gt_trajectory,
                    preds,
                    bucket_ids,
                    thresholds=[0.05, 0.1],
                )
                for bucket_name, metrics in bucket_results.items():
                    count = metrics["count"]
                    if count <= 0:
                        continue
                    bucket_l1_sums[bucket_name] += metrics["l1"] * count
                    bucket_counts[bucket_name] += count
                    for key, value in metrics.items():
                        if key.startswith("acc@"):
                            bucket_acc_sums[bucket_name][key] += value * count

    actual_batches = max(actual_batches, 1)
    eval_accuracy = (eval_accuracy / actual_batches).detach().float().cpu().tolist()
    eval_l1_loss = float((eval_l1_loss / actual_batches).detach().float().cpu())
    log.info("Evaluated batches: %s", actual_batches)
    log.info(
        format_results(
            eval_l1_loss,
            eval_accuracy,
            eval_thresholds,
            bucket_l1_sums,
            bucket_acc_sums,
            bucket_counts,
        )
    )


if __name__ == "__main__":
    main()
