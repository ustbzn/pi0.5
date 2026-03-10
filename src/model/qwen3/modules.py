import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# class Qwen3RotaryEmbedding(nn.Module):
#     def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
#         super().__init__()
#         self.dim = dim
#         self.max_position_embeddings = max_position_embeddings
#         self.base = base

#         inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
#         self.register_buffer("inv_freq", inv_freq, persistent=False)

#         # 直接缓存所有位置，避免动态扩展和 graph break
#         t = torch.arange(max_position_embeddings).float()
#         freqs = torch.einsum("i,j->ij", t, inv_freq)
#         emb = torch.cat((freqs, freqs), dim=-1)
#         self.register_buffer("cos_cached", emb.cos(), persistent=False)
#         self.register_buffer("sin_cached", emb.sin(), persistent=False)

#     def forward(self, x, position_ids):
#         # position_ids: [B, Seq_Len]
#         # cos/sin: [B, Seq_Len, Head_Dim]
#         cos = self.cos_cached[position_ids].to(x.dtype)
#         sin = self.sin_cached[position_ids].to(x.dtype)
#         return cos, sin

class Qwen3RotaryEmbedding(nn.Module):
    """
    Fully aligned with Qwen3 style rotary embedding.
    dim must equal head_dim.
    """

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.LongTensor):
        """
        position_ids: [B, L]
        return:
            cos, sin: [B, 1, L, head_dim]
        """
        B, L = position_ids.shape

        freqs = torch.einsum(
            "bl,d->bld", position_ids.float(), self.inv_freq
        )  # [B, L, dim/2]

        emb = torch.cat([freqs, freqs], dim=-1)  # [B, L, dim]

        cos = emb.cos()
        sin = emb.sin()

        # expand head dimension
        # cos = cos.unsqueeze(1)  # [B, 1, L, dim]
        # sin = sin.unsqueeze(1)

        return cos, sin

class Qwen3MLP(nn.Module):
    def __init__(self, config, use_quantize=False, use_lora=False):
        super().__init__()
        self.config = config

        from src.model.lora import get_layer

        layer = get_layer(quantize=use_quantize, lora=use_lora, **config.lora if use_lora else {})

        self.gate_proj = layer(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = layer(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = layer(config.intermediate_size, config.hidden_size, bias=False)

        # 使用官方的activation函数
        if config.hidden_act == "silu":
            self.act_fn = nn.functional.silu
        elif config.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.functional.gelu
        else:
            self.act_fn = nn.functional.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))