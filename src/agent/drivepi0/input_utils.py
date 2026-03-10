from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoTokenizer

from src.model.DrivePi0.processing import VLAProcessor
from src.model.DrivePi0.qwen3_processing import DriveQwen3Processor


def build_processor_from_config(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pretrained_model_path,
        padding_side="right",
    )
    if cfg.mixture.vlm.get("arch", "gemma") == "qwen3_vl":
        processor = DriveQwen3Processor(
            cfg.pretrained_model_path,
            max_seq_len=cfg.max_seq_len,
            padding=cfg.tokenizer_padding,
        )
    else:
        processor = VLAProcessor(
            tokenizer,
            num_image_tokens=cfg.vision.config.num_image_tokens,
            max_seq_len=cfg.max_seq_len,
            tokenizer_padding=cfg.tokenizer_padding,
        )
    return tokenizer, processor


def prepare_drivepi_inputs(
    *,
    model,
    processor,
    texts,
    images: torch.Tensor,
    proprios: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    actions: Optional[torch.Tensor] = None,
    split_mask: bool = True,
    t: Optional[torch.Tensor] = None,
):
    model_inputs = processor(text=texts, images=images)
    image_grid_thw = model_inputs.get("image_grid_thw")

    causal_mask, vlm_position_ids, proprio_position_ids, action_position_ids = (
        model.build_causal_mask_and_position_ids(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            dtype=dtype,
            image_grid_thw=image_grid_thw,
        )
    )

    inputs = {
        "input_ids": model_inputs["input_ids"],
        "pixel_values": model_inputs["pixel_values"].to(dtype),
        "vlm_position_ids": vlm_position_ids,
        "proprio_position_ids": proprio_position_ids,
        "action_position_ids": action_position_ids,
        "proprios": proprios.to(dtype),
    }
    if actions is not None:
        inputs["actions"] = actions.to(dtype)

    if split_mask:
        image_text_proprio_mask, action_mask = model.split_full_mask_into_submasks(
            causal_mask
        )
        inputs["image_text_proprio_mask"] = image_text_proprio_mask
        inputs["action_mask"] = action_mask
    else:
        inputs["causal_mask"] = causal_mask

    if t is not None:
        inputs["t"] = t.to(dtype)

    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["image_grid_thw"] = (
        image_grid_thw.to(device) if image_grid_thw is not None else None
    )
    return inputs
