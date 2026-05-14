#!/usr/bin/env python3
"""
独立全量评测脚本 — 对已训练的 checkpoint 在完整 val 集上评测。

用法:
    python script/eval_full.py \
        --config config/train/DrivePi0/0429_qwen3_ablation_state_only_noHisTraj_1frame_4gpu.yaml \
        --checkpoint logs/train/drive-pi0/.../checkpoint/step24920.pt \
        --eval_size 4000 \
        --batch_size 12 \
        --gpu 0

说明:
    - 直接复用训练配置 (yaml)，无需单独维护 eval config
    - 支持 state-only 和 VLM 两种模式
    - eval_size=4000 保证覆盖全部 val 样本 (共 3900 个)
"""

import argparse
import logging
import math
import os
import sys
import time
from collections import defaultdict

import einops
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen3VLProcessor

sys.path.insert(0, os.getcwd())

from src.agent.dataset import Bench2DriveDataset
from src.model.DrivePi0.drivepi0 import DrivePiZero
from src.utils.metric import get_action_accuracy

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil)
OmegaConf.register_new_resolver("round_down", math.floor)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def detect_norm_style(checkpoint_path):
    """检测 checkpoint 中的 RMSNorm 风格 (Gemma vs Qwen3)。

    GemmaRMSNorm: weight 初始化为 0, 训练后通常在 0 附近
    Qwen3RMSNorm: weight 初始化为 1, 训练后通常在 1 附近
    """
    data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    for key, val in data["model"].items():
        k = key.replace("_orig_mod.", "")
        if "mixtures.action.layers.0.input_layernorm.weight" in k:
            mean_val = val.mean().item()
            if abs(mean_val) < 0.5:
                return "gemma"
            else:
                return "qwen3"
    return "unknown"


def patch_mixture_to_gemma():
    """临时将 mixture.py 的组件切换回 Gemma 风格，用于加载旧 checkpoint"""
    from src.model.DrivePi0 import mixture
    from src.model.paligemma.modules import GemmaMLP, GemmaRMSNorm
    mixture._original_MLP = None
    mixture._original_RMSNorm = None

    # 检查当前是否已经在用 Qwen3 组件
    test_layer = mixture.MixtureDecoderLayer.__init__
    import inspect
    src = inspect.getsource(test_layer)
    if "Qwen3MLP" in src or "Qwen3RMSNorm" in src:
        # 需要 monkey-patch: 替换类中引用的组件
        log.info("Patching Mixture to use Gemma-style components for old checkpoint...")

        # 保存原始引用
        mixture._original_Qwen3MLP = mixture.Qwen3MLP
        mixture._original_Qwen3RMSNorm = mixture.Qwen3RMSNorm

        # 替换为 Gemma 版本
        mixture.Qwen3MLP = GemmaMLP
        mixture.Qwen3RMSNorm = GemmaRMSNorm
        return True
    return False


def unpatch_mixture():
    """恢复 mixture.py 的组件为 Qwen3 风格"""
    from src.model.DrivePi0 import mixture
    if hasattr(mixture, '_original_Qwen3MLP') and mixture._original_Qwen3MLP is not None:
        mixture.Qwen3MLP = mixture._original_Qwen3MLP
        mixture.Qwen3RMSNorm = mixture._original_Qwen3RMSNorm
        log.info("Unpatched Mixture back to Qwen3-style components.")


def load_model_from_checkpoint(cfg, checkpoint_path, device, dtype):
    """加载模型并从 checkpoint 恢复权重，自动检测 Gemma/Qwen3 组件风格"""
    norm_style = detect_norm_style(checkpoint_path)
    log.info(f"Detected checkpoint norm style: {norm_style}")

    patched = False
    if norm_style == "gemma":
        patched = patch_mixture_to_gemma()

    model = DrivePiZero(cfg, use_ddp=False, load_pretrained=True)
    model.tie_action_proprio_weights()

    if cfg.get("lora", False):
        model.inject_lora(cfg)

    data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    state_dict = {
        k.replace("_orig_mod.", ""): v for k, v in data["model"].items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    proprio_missing = [k for k in missing if "mixtures.proprio" not in k]
    if proprio_missing:
        log.warning(f"Missing keys (excl. proprio): {proprio_missing[:10]}")
    if unexpected:
        log.warning(f"Unexpected keys: {unexpected[:10]}")
    log.info(f"Loaded checkpoint from {checkpoint_path} (step={data.get('cnt_update', '?')})")

    if patched:
        unpatch_mixture()

    model.freeze_all_weights()
    model.to(dtype)
    model.to(device)
    model.eval()
    return model


def build_processor(cfg):
    """构建 Qwen3VL Processor"""
    processor = Qwen3VLProcessor.from_pretrained(
        cfg.pretrained_model_path, trust_remote_code=True,
    )
    return processor


def preprocess_batch_vlm(batch, model, processor, cfg, device, dtype):
    """预处理 batch (VLM 模式)"""
    state = batch["state"]
    trajectory = batch["trajectory"].squeeze(1)
    texts = batch["language_instruction"]

    cam_names = ["front", "right", "left", "back"]
    cam_labels = [
        "current front view", "current right view",
        "current left view", "current rear view",
        "previous front view", "previous right view",
        "previous left view", "previous rear view",
    ]

    image_list = []
    for cam in cam_names:
        img = batch[f"image_{cam}"]
        img = einops.rearrange(img, "B H W C -> B C H W")
        image_list.append(img.unsqueeze(1))
    for cam in cam_names:
        img = batch[f"image_{cam}_time"]
        img = einops.rearrange(img, "B H W C -> B C H W")
        image_list.append(img.unsqueeze(1))

    images = torch.cat(image_list, dim=1)  # [B, 8, C, H, W]
    batch_size = images.shape[0]
    num_images_per_sample = images.shape[1]

    pil_images = []
    for i in range(batch_size):
        for j in range(num_images_per_sample):
            img = Image.fromarray(
                images[i, j].cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            )
            pil_images.append(img)

    chat_texts = []
    for i, text in enumerate(texts):
        content = []
        for label in cam_labels:
            content.append({"type": "text", "text": f"{label}:"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        chat_texts.append(chat_text)

    model_inputs = processor(
        text=chat_texts,
        images=pil_images,
        return_tensors="pt",
        padding=cfg.tokenizer_padding,
        max_length=cfg.max_seq_len,
        truncation=True,
    ).to(device)

    causal_mask, _, state_position_ids, action_position_ids = (
        model.build_causal_mask_and_position_ids(model_inputs["attention_mask"], dtype)
    )
    # P0-A: 使用官方 3D M-RoPE position_ids 替代 1D arange
    vlm_position_ids = model.build_qwen3_vl_position_ids(
        model_inputs["input_ids"],
        model_inputs["image_grid_thw"],
        model_inputs["attention_mask"],
    )
    image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)

    inputs = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"].to(dtype),
        "image_grid_thw": model_inputs["image_grid_thw"],
        "image_text_proprio_mask": image_text_proprio_mask.to(device),
        "action_mask": action_mask.to(device),
        "vlm_position_ids": vlm_position_ids.to(device),
        "proprio_position_ids": state_position_ids.to(device),
        "action_position_ids": action_position_ids.to(device),
        "proprios": state.to(dtype).to(device),
    }
    gt = trajectory.to(dtype).to(device)
    return inputs, gt


def preprocess_batch_state_only(batch, model, device, dtype):
    """预处理 batch (state-only 模式)"""
    state = batch["state"]
    trajectory = batch["trajectory"].squeeze(1)
    batch_size = state.shape[0]

    causal_mask, vlm_position_ids, state_position_ids, action_position_ids = (
        model.build_state_only_causal_mask_and_position_ids(
            batch_size=batch_size, device=device, dtype=dtype,
        )
    )
    state_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)

    inputs = {
        "input_ids": torch.empty((batch_size, 0), dtype=torch.long).to(device),
        "pixel_values": torch.empty((0,), dtype=dtype).to(device),
        "image_grid_thw": torch.empty((0, 3), dtype=torch.long).to(device),
        "image_text_proprio_mask": state_mask,
        "action_mask": action_mask,
        "vlm_position_ids": vlm_position_ids,
        "proprio_position_ids": state_position_ids,
        "action_position_ids": action_position_ids,
        "proprios": state.to(dtype).to(device),
    }
    gt = trajectory.to(dtype).to(device)
    return inputs, gt


def main():
    parser = argparse.ArgumentParser(description="Full validation evaluation")
    parser.add_argument("--config", required=True, help="Training config yaml path")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint .pt path")
    parser.add_argument("--eval_size", type=int, default=4000, help="Max eval samples (4000 covers full val)")
    parser.add_argument("--batch_size", type=int, default=None, help="Eval batch size (default: from config)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    # 训练 config 中有 ${oc.env:LOG_DIR}、${now:...} 等解析器，评测时不需要。
    # 先把依赖这些解析器的字段设为固定值，避免 resolve 失败。
    with open_dict(cfg):
        cfg.log_dir = "/tmp/eval_dummy"
        cfg.run_name = "eval"
        if "swanlab" in cfg:
            cfg.swanlab.experiment_name = "eval"

    OmegaConf.resolve(cfg)

    device = torch.device(f"cuda:{args.gpu}")
    dtype = torch.bfloat16 if cfg.get("use_bf16", False) else torch.float32
    batch_size = args.batch_size or cfg.per_device_batch_size
    state_only = cfg.get("state_only_for_action", False)
    eval_thresholds = cfg.eval_thresholds

    log.info(f"Config: {args.config}")
    log.info(f"Checkpoint: {args.checkpoint}")
    log.info(f"Mode: {'state-only' if state_only else 'VLM'}")
    log.info(f"Batch size: {batch_size}, Eval size: {args.eval_size}, GPU: {args.gpu}")

    # 加载模型
    with open_dict(cfg):
        cfg.gpu_id = args.gpu
        cfg.multi_gpu = False
    model = load_model_from_checkpoint(cfg, args.checkpoint, device, dtype)

    if not args.no_compile and cfg.get("use_torch_compile", False):
        log.info("Compiling model with torch.compile (first batch will be slow)...")
        model = torch.compile(model, mode="default")

    # 构建 val dataloader
    val_dataset = Bench2DriveDataset(cfg.data.val).dataset
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=cfg.num_workers,
        shuffle=False,
    )
    max_batches = args.eval_size // batch_size
    actual_max = min(max_batches, len(val_dataloader))
    log.info(f"Val dataset size: {len(val_dataset)}, Will eval {actual_max} batches x {batch_size} = ~{actual_max * batch_size} samples")

    # 构建 processor (VLM 模式)
    processor = None
    if not state_only:
        processor = build_processor(cfg)

    # 评测
    eval_accuracy = torch.zeros(len(eval_thresholds), device=device)
    eval_l1_loss = torch.tensor(0.0, device=device)
    cnt = 0

    log.info("Starting evaluation...")
    t0 = time.time()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_dataloader, total=actual_max)):
            if batch_idx >= actual_max:
                break

            if state_only:
                inputs, gt_trajectory = preprocess_batch_state_only(batch, model, device, dtype)
            else:
                inputs, gt_trajectory = preprocess_batch_vlm(batch, model, processor, cfg, device, dtype)

            preds = model.infer_action(**inputs)
            eval_accuracy += get_action_accuracy(gt_trajectory, preds, eval_thresholds)
            eval_l1_loss += torch.nn.functional.l1_loss(preds, gt_trajectory)
            cnt += 1

    elapsed = time.time() - t0
    eval_accuracy /= cnt
    eval_l1_loss /= cnt

    # 输出结果
    log.info(f"Evaluation completed in {elapsed:.1f}s ({cnt} batches, {cnt * batch_size} samples)")
    log.info(f"Config: {os.path.basename(args.config)}")
    log.info(f"Checkpoint: {args.checkpoint}")
    result_msg = f"Eval | l1 Loss: {eval_l1_loss.item():.4f} | "
    result_msg += " | ".join(
        [f"acc thres {t}: {a.item():.4f}" for t, a in zip(eval_thresholds, eval_accuracy)]
    )
    log.info(result_msg)


if __name__ == "__main__":
    main()
