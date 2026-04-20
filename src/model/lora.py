import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """LoRA adapter that wraps an existing nn.Linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 32,
        lora_alpha: int = None,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha if lora_alpha is not None else r
        self.scaling = self.lora_alpha / self.r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        if lora_dropout > 0:
            self.lora_dropout = nn.Dropout(lora_dropout)
        else:
            self.lora_dropout = nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return base_out + lora_out

    @property
    def weight(self):
        return self.base_layer.weight

    @property
    def bias(self):
        return self.base_layer.bias

    @property
    def in_features(self):
        return self.base_layer.in_features

    @property
    def out_features(self):
        return self.base_layer.out_features


def get_layer(
    quantize: bool = False,
    lora: bool = False,
    r: int = 32,
    dropout: float = 0.05,
):
    if quantize and lora:
        raise NotImplementedError("Quantized LoRA (QLoRA) is not yet supported")
    elif quantize and not lora:
        raise NotImplementedError("4-bit quantization is not yet supported")
    elif not quantize and lora:
        return lambda *args, **kwargs: LoRALinear(
            nn.Linear(*args, **kwargs), r=r, lora_dropout=dropout,
        )
    else:
        return nn.Linear
