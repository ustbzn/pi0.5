# src/model/DrivePi0/qwen3_vl_vision.py
from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

log = logging.getLogger(__name__)


@dataclass
class Qwen3VLVisionTowerOutput:
    vision_features: torch.Tensor
    deepstack_features: list[torch.Tensor]


class Qwen3VLVisionTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        # self.deepstack_visual_indexes = cfg_dict.pop("deepstack_visual_indexes", [])
        self.config = Qwen3VLVisionConfig(**cfg_dict)
        self.visual = Qwen3VLVisionModel(self.config)

    def forward(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor):
        assert pixel_values is not None, "pixel_values must not be None for Qwen3 vision."
        assert image_grid_thw is not None, "image_grid_thw must not be None for Qwen3 vision."
        assert pixel_values.ndim == 2, (
            f"Qwen3 vision expects flattened patches [N, C], got {tuple(pixel_values.shape)}"
        )
        assert image_grid_thw.ndim == 2 and image_grid_thw.shape[-1] == 3, (
            f"image_grid_thw must be [num_images, 3], got {tuple(image_grid_thw.shape)}"
        )

        # HF 文档里 Qwen3VLVisionModel.forward 返回的是 torch.Tensor。
        # 不同 transformers 版本也可能返回 tuple / ModelOutput，因此这里统一归一化。
        outputs = self.visual(pixel_values, grid_thw=image_grid_thw)
        '''
            BaseModelOutputWithDeepstackFeatures(
                last_hidden_state=hidden_states,
                pooler_output=merged_hidden_states,
                deepstack_features=deepstack_feature_lists,
            )
        '''
        vision_features = outputs.pooler_output
        deepstack_features = outputs.deepstack_features
        return Qwen3VLVisionTowerOutput(
            vision_features=vision_features,
            deepstack_features=deepstack_features,
        )
