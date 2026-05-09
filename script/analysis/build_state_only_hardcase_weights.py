#!/usr/bin/env python3
"""
方案 D: 构建 state-only hard case 权重

功能：
  1. 加载 state-only 训练好的 checkpoint
  2. 在训练集上逐样本跑推理，计算每个 pkl 的 L1 error
  3. 输出 JSON 文件：{filename.pkl: l1_error, ...}
  4. 该 JSON 可被训练配置中的 trajectory_weighted_sampling.hard_case_weight_path 引用

使用方法：
  python script/analysis/build_state_only_hardcase_weights.py \
      --checkpoint /path/to/state_only/checkpoint/stepXXXX.pt \
      --data_dir /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_v2/train \
      --statistics_path /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_v2/statistics_9dim.json \
      --output /pcpt/pcpt/data/VLA/MOGO/mogo_dataset_navi_v2/train_hardcase_weights.json \
      --batch_size 48 \
      --gpu_id 0

注意：
  - 这个脚本应该用你最好的 state-only checkpoint 来跑
    例如：0429_qwen3_ablation_state_only_noHisTraj_1frame_4gpu 对应的 checkpoint
  - 跑一次即可，输出的 JSON 可以复用于多轮训练实验
  - 预计耗时约 30-60 分钟（视数据集大小和 GPU）
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import einops
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

# 将项目根目录加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import TorchDataset
from src.model.DrivePi0.drivepi0 import DrivePiZero
from src.data.utils.normalization import Normalize


def build_state_only_model(checkpoint_path: str, cfg_overrides: dict, device: torch.device, dtype: torch.dtype):
    """构建一个 state-only 推理模型并加载 checkpoint。"""
    from omegaconf import OmegaConf

    # 使用最小化配置构建模型
    # 从 checkpoint 推断训练配置
    base_cfg_path = PROJECT_ROOT / "config" / "train" / "DrivePi0" / "0429_qwen3_ablation_state_only_noHisTraj_1frame_4gpu.yaml"

    if base_cfg_path.exists():
        base_cfg = OmegaConf.load(str(base_cfg_path))
    else:
        # 如果找不到对应配置，使用默认值
        base_cfg = OmegaConf.create({
            "vocab_size": 151936,
            "pad_token_id": 0,
            "image_token_index": 151655,
            "max_image_text_tokens": 600,
            "cond_steps": 1,
            "horizon_steps": 6,
            "action_dim": 2,
            "proprio_dim": 7,
            "num_inference_steps": 10,
            "final_action_clip_value": 1.0,
            "flow_sig_min": 0.001,
            "mask_vlm_for_action": False,
            "state_only_for_action": True,
            "action_expert_adaptive_mode": None,
            "use_lm_head": False,
            "quantize": False,
            "lora": False,
            "pretrained_model_path": "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct",
            "mixture": {
                "vlm": {"hidden_size": 2048, "intermediate_size": 6144, "use_final_norm": False, "cache": True, "use_quantize": False, "use_lora": False, "adaptive_mode": None},
                "proprio": {"hidden_size": 1024, "intermediate_size": 4096, "use_final_norm": True, "cache": True, "use_quantize": False, "use_lora": False, "adaptive_mode": None},
                "action": {"hidden_size": 1024, "intermediate_size": 4096, "use_final_norm": True, "cache": False, "use_quantize": False, "use_lora": False, "adaptive_mode": None},
            },
            "time_hidden_size": 256,
            "trajectory_loss": {"lambda_smooth": 0.0, "lambda_accel": 0.0, "endpoint_weight_mode": "none", "endpoint_weight_max": 3.0},
            "joint": {
                "config": {
                    "pretrained_model_path": "/pcpt/pcpt/project/jiangjunyuan/code/VLA/Qwen3-VL-2B-Instruct",
                    "action_expert_adaptive_mode": None,
                    "time_hidden_size": 256,
                    "num_hidden_layers": 28,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "hidden_act": "gelu_pytorch_tanh",
                    "max_position_embeddings": 262144,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 5000000,
                    "attention_bias": False,
                    "attention_dropout": 0.0,
                    "pad_token_id": 0,
                    "rope_scaling": {"mrope_interleaved": True, "mrope_section": [24, 20, 20], "rope_type": "default"},
                    "lora": {"r": 32, "dropout": 0.0},
                },
            },
        })

    # 合并 overrides
    cfg = OmegaConf.merge(base_cfg, OmegaConf.create(cfg_overrides))

    # 确保 state_only_for_action = True
    cfg.state_only_for_action = True

    # 构建模型（不加载预训练 VLM 权重，由 checkpoint 提供所有权重）
    model = DrivePiZero(cfg, use_ddp=False, load_pretrained=False)
    model.tie_action_proprio_weights()

    # 加载 checkpoint
    data = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    ckpt = {k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}

    model_state = model.state_dict()
    filtered = {}
    for k, v in ckpt.items():
        if k in model_state and model_state[k].shape == v.shape:
            filtered[k] = v
    model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)}/{len(model_state)} weights from checkpoint")

    model.freeze_all_weights()
    model.to(dtype)
    model.to(device)
    model.eval()
    return model


def infer_state_only_batch(model, states, device, dtype):
    """对一个 batch 的 state 做 state-only 推理，返回预测轨迹。"""
    batch_size = states.shape[0]

    causal_mask, vlm_position_ids, state_position_ids, action_position_ids = (
        model.build_state_only_causal_mask_and_position_ids(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
    )

    image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(causal_mask)

    inputs = {
        "input_ids": torch.empty((batch_size, 0), dtype=torch.long).to(device),
        "pixel_values": torch.empty((0,), dtype=dtype).to(device),
        "image_grid_thw": torch.empty((0, 3), dtype=torch.long).to(device),
        "vlm_position_ids": vlm_position_ids.to(device),
        "proprio_position_ids": state_position_ids.to(device),
        "action_position_ids": action_position_ids.to(device),
        "proprios": states.to(dtype).to(device),
        "image_text_proprio_mask": image_text_proprio_mask.to(device),
        "action_mask": action_mask.to(device),
    }
    with torch.no_grad():
        preds = model.infer_action(**inputs)
    return preds


def main():
    parser = argparse.ArgumentParser(description="方案D: 生成 state-only hard case 权重")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="state-only checkpoint 路径")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="训练集 pkl 目录 (例如 mogo_dataset_navi_v2/train)")
    parser.add_argument("--statistics_path", type=str, required=True,
                        help="统计文件路径 (例如 statistics_9dim.json)")
    parser.add_argument("--output", type=str, required=True,
                        help="输出 JSON 路径")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--norm_mode", type=str, default="minmax")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}")
    dtype = torch.bfloat16

    print(f"[1/4] 构建 state-only 模型...")
    model = build_state_only_model(
        checkpoint_path=args.checkpoint,
        cfg_overrides={},
        device=device,
        dtype=dtype,
    )

    print(f"[2/4] 加载训练集数据集: {args.data_dir}")
    dataset = TorchDataset(
        data_path=args.data_dir,
        statistics_path=args.statistics_path,
        use_fixed_images=False,
        num_of_action_experts=13,
        norm_mode=args.norm_mode,
        use_his_trajectory=False,
        state_frame_index=0,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"[3/4] 逐 batch 推理，计算逐样本 L1 error...")
    print(f"       共 {len(dataset)} 个样本，batch_size={args.batch_size}")

    # 因为 DataLoader shuffle=False，样本顺序与 dataset.list_data_dict 一致
    all_errors = []
    sample_idx = 0

    for batch in tqdm(dataloader, desc="推理中"):
        states = batch["state"]  # [B, window, state_dim]
        trajectories = batch["trajectory"].squeeze(1)  # [B, horizon, action_dim]
        batch_size = states.shape[0]

        preds = infer_state_only_batch(model, states, device, dtype)
        # 计算逐样本 L1
        l1_per_sample = torch.mean(
            torch.abs(preds.cpu().float() - trajectories.float()),
            dim=(1, 2),
        )  # [B]

        all_errors.extend(l1_per_sample.tolist())
        sample_idx += batch_size

    # 构建 filename -> error 的映射
    assert len(all_errors) == len(dataset), \
        f"Error count mismatch: {len(all_errors)} vs {len(dataset)}"

    result = {}
    for fname, error in zip(dataset.list_data_dict, all_errors):
        result[fname] = round(error, 6)

    # 统计信息
    errors_np = np.array(all_errors)
    print(f"\n[4/4] 统计信息:")
    print(f"       样本数: {len(result)}")
    print(f"       L1 均值: {errors_np.mean():.4f}")
    print(f"       L1 中位数: {np.median(errors_np):.4f}")
    print(f"       L1 std: {errors_np.std():.4f}")
    print(f"       L1 P90: {np.percentile(errors_np, 90):.4f}")
    print(f"       L1 P99: {np.percentile(errors_np, 99):.4f}")
    print(f"       L1 max: {errors_np.max():.4f}")

    # 保存
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f)
    print(f"\n✅ 已保存 hard case 权重到: {args.output}")
    print(f"   可在训练配置中设置:")
    print(f"   trajectory_weighted_sampling:")
    print(f"     hard_case_weight_path: {args.output}")
    print(f"     hard_case_alpha: 1.0")


if __name__ == "__main__":
    main()
