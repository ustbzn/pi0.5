import logging
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def get_layer(
    quantize: bool = False,
    lora: bool = False,
    r: int = 32,
    dropout: float = 0.05,
):
    if quantize and lora:
        return lambda *args, **kwargs: LoRALinear4bit(
            r=r, lora_dropout=dropout, *args, **kwargs
        )
    elif quantize and not lora:
        return Linear4bit
    elif not quantize and lora:
        return lambda *args, **kwargs: LoRALinear(
            r=r, lora_dropout=dropout, *args, **kwargs
        )
    else:
        return nn.Linear