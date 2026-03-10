# src/model/DrivePi0/qwen3_vl_mixture.py
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers.activations import ACT2FN
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    repeat_kv,
)

from src.model.lora import get_layer
from src.model.DrivePi0.modules import AdaptiveLayerscale, AdaptiveRMSNorm
from src.model.utils import apply_rotary_pos_emb


def _to_container(cfg):
    return OmegaConf.to_container(cfg, resolve=True) if not isinstance(cfg, dict) else cfg


def build_hf_qwen_text_config(cfg) -> Qwen3VLTextConfig:
    cfg = _to_container(cfg)
    hf_cfg = dict(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        max_position_embeddings=cfg["max_position_embeddings"],
        rope_theta=cfg["rope_theta"],
        attention_bias=cfg["attention_bias"],
        attention_dropout=cfg["attention_dropout"],
        rms_norm_eps=cfg["rms_norm_eps"],
        hidden_act=cfg.get("hidden_act", "silu"),
        vocab_size=cfg.get("vocab_size", 151936),
        pad_token_id=cfg.get("pad_token_id", 151643),
        rope_scaling=cfg.get(
            "rope_scaling",
            {
                "rope_type": "default",
                "mrope_interleaved": True,
                "mrope_section": [24, 20, 20],
            },
        ),
    )
    return Qwen3VLTextConfig(**hf_cfg)


class Qwen3VLTextMLPAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        layer = get_layer(
            config.use_quantize,
            config.use_lora,
            **config.lora if config.use_lora else {},
        )
        self.gate_proj = layer(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = layer(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = layer(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLAttentionAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim

        layer = get_layer(
            config.use_quantize,
            config.use_lora,
            **config.lora if config.use_lora else {},
        )

        self.q_proj = layer(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = layer(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = layer(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = layer(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        hf_text_config = build_hf_qwen_text_config(config)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(hf_text_config)

    def _format_rope_position_ids(self, position_ids: torch.LongTensor) -> torch.LongTensor:
        # vlm: [4, B, L] -> 只把后 3 路给 mRoPE
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            return position_ids[1:]

        # action/proprio: [B, L] -> 退化成 [3, B, L]
        if position_ids.ndim == 2:
            return position_ids.unsqueeze(0).expand(3, -1, -1)

        # 已经是 [3, B, L]
        if position_ids.ndim == 3 and position_ids.shape[0] == 3:
            return position_ids

        raise ValueError(f"Unsupported position_ids shape: {tuple(position_ids.shape)}")

    def forward_q_proj(self, x: torch.Tensor) -> torch.Tensor:
        bsz, q_len = x.shape[:2]
        q = self.q_proj(x).view(bsz, q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        return q.transpose(1, 2)

    def forward_k_proj(self, x: torch.Tensor) -> torch.Tensor:
        bsz, q_len = x.shape[:2]
        k = self.k_proj(x).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        k = self.k_norm(k)
        return k.transpose(1, 2)

    def forward_v_proj(self, x: torch.Tensor) -> torch.Tensor:
        bsz, q_len = x.shape[:2]
        v = self.v_proj(x).view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        return v.transpose(1, 2)

    def forward_o_proj(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj(x)

    def forward_rotary_emb(
        self,
        x: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rope_position_ids = self._format_rope_position_ids(position_ids)
        cos, sin = self.rotary_emb(x, rope_position_ids)
        cos, sin = self._normalize_rotary_cache_layout(x, cos, sin)
        return cos, sin

    def _normalize_rotary_cache_layout(
        self,
        states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize rotary outputs to [B, L, D] for our local attention layout.

        Our attention tensors use [B, H, L, D]. Depending on the installed
        transformers version, Qwen3-VL rotary embeddings may return [B, L, D]
        or [L, B, D]. We canonicalize here so the downstream application logic
        stays stable.
        """
        batch_size = states.shape[0]
        seq_len = states.shape[-2]

        if cos.ndim != 3 or sin.ndim != 3:
            raise ValueError(
                f"Unexpected rotary output rank: cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
            )

        if cos.shape[:2] == (batch_size, seq_len):
            return cos, sin

        if states.ndim == 4 and states.shape[1:3] == cos.shape[:2]:
            return cos, sin

        if cos.shape[:2] == (seq_len, batch_size):
            return cos.transpose(0, 1), sin.transpose(0, 1)

        raise ValueError(
            "Unsupported rotary output shape for Qwen3-VL attention: "
            f"states={tuple(states.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
        )

    def forward_apply_rotary_emb(
        self,
        states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        # Standard local layout: [B, H, L, D]
        if states.ndim == 4 and states.shape[0] == cos.shape[0] and states.shape[-2] == cos.shape[1]:
            return apply_rotary_pos_emb(states, cos, sin)

        # Some Qwen3-VL paths expose [X, B, L, D]; move batch to dim 0,
        # apply RoPE, then restore the original layout.
        if states.ndim == 4 and states.shape[1] == cos.shape[0] and states.shape[-2] == cos.shape[1]:
            states = states.transpose(0, 1)
            states = apply_rotary_pos_emb(states, cos, sin)
            return states.transpose(0, 1)

        raise ValueError(
            "Unsupported attention state shape for Qwen3-VL rotary application: "
            f"states={tuple(states.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
        )

    def repeat_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return key_states, value_states


class Qwen3VLDecoderLayerAdapter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = Qwen3VLAttentionAdapter(config)
        self.mlp = Qwen3VLTextMLPAdapter(config)

        self.adaptive_mode = config.get("adaptive_mode", None)
        if self.adaptive_mode:
            self.input_layernorm = AdaptiveRMSNorm(
                config.hidden_size, config.time_hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = AdaptiveRMSNorm(
                config.hidden_size, config.time_hidden_size, eps=config.rms_norm_eps
            )
            if self.adaptive_mode == "adaLN-Zero":
                self.post_adaptive_scale = AdaptiveLayerscale(
                    config.hidden_size, config.time_hidden_size
                )
                self.final_adaptive_scale = AdaptiveLayerscale(
                    config.hidden_size, config.time_hidden_size
                )
        else:
            self.input_layernorm = Qwen3VLTextRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_attention_layernorm = Qwen3VLTextRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward_norm(
        self,
        norm_name: str,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        args = [x] if self.adaptive_mode is None else [x, cond]
        return getattr(self, norm_name)(*args)

    def forward_adaptive_scale(
        self,
        stage: str,
        x: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.adaptive_mode == "adaLN-Zero":
            if stage == "post_attn":
                return self.post_adaptive_scale(x, cond)
            if stage == "final":
                return self.final_adaptive_scale(x, cond)
            raise ValueError(f"Invalid stage: {stage}")
        return x


class Qwen3VLMixture(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [Qwen3VLDecoderLayerAdapter(config) for _ in range(config.num_hidden_layers)]
        )
        self.scale_input_embeds = bool(config.get("scale_input_embeds", False))
        self.adaptive_mode = None

        if config.use_final_norm:
            self.adaptive_mode = config.get("adaptive_mode", None)
            if self.adaptive_mode:
                self.norm = AdaptiveRMSNorm(
                    config.hidden_size, config.time_hidden_size, eps=config.rms_norm_eps
                )
            else:
                self.norm = Qwen3VLTextRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )

    @property
    def head_dim(self) -> int:
        return self.layers[0].self_attn.head_dim

    def layer_func(self, method_name: str, layer_idx: int, *args):
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx], method_name)(*args)

    def attn_func(self, method_name: str, layer_idx: int, *args):
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx].self_attn, method_name)(*args)

    def forward_norm(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        if not hasattr(self, "norm"):
            return None
        args = [x] if self.adaptive_mode is None else [x, cond]
        return self.norm(*args)

    def apply_deepstack(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        if visual_pos_masks is None or visual_embeds is None:
            return hidden_states
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks] = hidden_states[visual_pos_masks] + visual_embeds
        return hidden_states
