"""
Individual mixture in Qwen3-aligned format

Additional support for AdaLN(-Zero) and (Q)LoRA

"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.model.lora import get_layer
from src.model.paligemma.modules import (
    GemmaRotaryEmbedding,
)
from src.model.qwen3.modules import (
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding)
from src.model.utils import apply_rotary_pos_emb, repeat_kv
from src.model.DrivePi0.modules import AdaptiveLayerscale, AdaptiveRMSNorm


class Mixture(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList(
            [MixtureDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

        self.adaptive_mode = None
        if config.use_final_norm:
            self.adaptive_mode = config.get("adaptive_mode", None)
            if self.adaptive_mode:
                self.norm = AdaptiveRMSNorm(
                    config.hidden_size,
                    config.time_hidden_size,
                    eps=config.rms_norm_eps,
                )
            else:
                self.norm = Qwen3RMSNorm(
                    config.hidden_size,
                    eps=config.rms_norm_eps,
                )

    @property
    def head_dim(self) -> int:
        return self.layers[0].self_attn.head_dim

    def layer_func(
        self,
        method_name: str,
        layer_idx: int,
        *args,
    ) -> torch.FloatTensor:
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx], method_name)(*args)

    def attn_func(
        self,
        method_name: str,
        layer_idx: int,
        *args,
    ) -> torch.FloatTensor:
        args = [arg for arg in args if arg is not None]
        return getattr(self.layers[layer_idx].self_attn, method_name)(*args)

    def forward_norm(
        self,
        x: torch.FloatTensor,
        cond: Optional[torch.FloatTensor] = None,
    ) -> Optional[torch.FloatTensor]:
        if hasattr(self, "norm"):
            args = [x] if self.adaptive_mode is None else [x, cond]
            return self.norm(*args)
        else:
            return None


class MixtureDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MixtureAttention(config)

        # ── MoE 切换 ─────────────────────────────────────────────────────
        # 当 config.moe.action_expert.enable=True 时，把这条 mixture 的 MLP 替换为
        # SharedExpertMoEMLP（1 shared + N routed top-k）。
        # shared/routed expert 的具体初始化（用当前 checkpoint 权重 + 扰动）由
        # DrivePiZero.__init__ 在替换之后统一处理，这里只负责构造模块结构。
        # 注意：routed expert 默认强制关闭 LoRA / 量化，避免参数爆炸；如需细化可在
        # SharedExpertMoEMLP 外层增加策略。
        moe_cfg = config.get("moe", None)
        action_moe_enable = bool(
            moe_cfg
            and moe_cfg.get("action_expert", None)
            and moe_cfg.action_expert.get("enable", False)
        )
        if action_moe_enable:
            from src.model.DrivePi0.moe_mlp import SharedExpertMoEMLP
            ae_cfg = moe_cfg.action_expert
            self.mlp = SharedExpertMoEMLP(
                mlp_class=Qwen3MLP,
                config=config,
                num_routed_experts=int(ae_cfg.get("num_routed_experts", 3)),
                top_k=int(ae_cfg.get("top_k", 1)),
                load_balance_coeff=float(ae_cfg.get("load_balance_coeff", 0.01)),
                mlp_kwargs={
                    "use_quantize": config.use_quantize,
                    # routed expert 不再叠加 LoRA，避免与 inject_lora 的 LoRA 路径重复
                    "use_lora": False,
                },
            )
            self.is_moe_mlp = True
        else:
            self.mlp = Qwen3MLP(
                config, use_quantize=config.use_quantize, use_lora=config.use_lora
            )
            self.is_moe_mlp = False
        # ────────────────────────────────────────────────────────────────

        self.adaptive_mode = config.get("adaptive_mode", None)
        if self.adaptive_mode:
            self.input_layernorm = AdaptiveRMSNorm(
                config.hidden_size,
                config.time_hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = AdaptiveRMSNorm(
                config.hidden_size,
                config.time_hidden_size,
                eps=config.rms_norm_eps,
            )
            if self.adaptive_mode == "adaLN-Zero":
                self.post_adaptive_scale = AdaptiveLayerscale(
                    config.hidden_size,
                    config.time_hidden_size,
                )
                self.final_adaptive_scale = AdaptiveLayerscale(
                    config.hidden_size,
                    config.time_hidden_size,
                )
        else:
            self.input_layernorm = Qwen3RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = Qwen3RMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

    def forward_norm(
        self,
        norm_name: str,
        x: torch.FloatTensor,
        cond: Optional[torch.FloatTensor] = None,
    ) -> Optional[torch.FloatTensor]:
        args = [x] if self.adaptive_mode is None else [x, cond]
        return getattr(self, norm_name)(*args)

    def forward_adaptive_scale(
        self,
        stage: str,
        x: torch.FloatTensor,
        cond: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        if self.adaptive_mode == "adaLN-Zero":
            if stage == "post_attn":
                return self.post_adaptive_scale(x, cond)
            elif stage == "final":
                return self.final_adaptive_scale(x, cond)
            else:
                raise ValueError(f"Invalid stage for adaptive scaling: {stage}!")
        return x


class MixtureAttention(nn.Module):
    """assume head_dim same for all blocks"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        assert config.hidden_size % self.num_heads == 0

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
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )

    def forward_q_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bsz, q_len = x.shape[:2]
        # [Batch_Size, Seq_Len, Num_Heads_Q * Head_Dim]
        query_states = self.q_proj(x)
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        return query_states

    def forward_k_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bsz, q_len = x.shape[:2]
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        key_states = self.k_proj(x)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        return key_states

    def forward_v_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        bsz, q_len = x.shape[:2]
        # [Batch_Size, Seq_Len, Num_Heads_KV * Head_Dim]
        value_states = self.v_proj(x)
        # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        return value_states

    def forward_o_proj(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.o_proj(x)

    def forward_rotary_emb(self, query_states, position_ids):
        """
        统一处理不同版本 transformers 的 Qwen3RotaryEmbedding 输出格式：
        - 新版：[max_pos, D]           (2D, 全缓存，需手动按 position_ids 索引)
        - 旧版：[1, S, D] 或 [B, S, D] (3D, 已部分处理)
        """
        cos, sin = self.rotary_emb(query_states, position_ids)
        bsz, seq_len = position_ids.shape

        if cos.dim() == 2:
            # [max_pos, D] → 用 position_ids 索引 → [B, S, D]
            # 注意：position_ids 的值必须 < max_pos（通常满足）
            cos = cos[position_ids]   # [B, S, D]
            sin = sin[position_ids]

        elif cos.dim() == 3:
            if cos.shape[0] == 1 and bsz != 1:
                # [1, S, D] → [B, S, D]（广播形式）
                cos = cos.expand(bsz, -1, -1)[:, :seq_len, :]
                sin = sin.expand(bsz, -1, -1)[:, :seq_len, :]
            elif cos.shape[0] != bsz:
                # 其他不匹配情况：尝试 squeeze 后重新索引
                cos = cos.reshape(-1, cos.shape[-1])[position_ids.reshape(-1)].reshape(bsz, seq_len, -1)
                sin = sin.reshape(-1, sin.shape[-1])[position_ids.reshape(-1)].reshape(bsz, seq_len, -1)
            else:
                # [B, S, D]，已正确，截取实际 seq_len
                cos = cos[:, :seq_len, :]
                sin = sin[:, :seq_len, :]
        # dim == 4 时（multimodal RoPE）：透传给 forward_apply_rotary_emb 处理

        return cos, sin
    
    # def forward_rotary_emb(
    #     self,
    #     x: torch.FloatTensor,
    #     position_ids: torch.LongTensor,
    # ):
    #     """
    #     x: [B, H, L, D]
    #     position_ids: [B, L]
    #     """

    #     B, H, L, D = x.shape

    #     # 只根据 position_ids 生成 cos/sin
    #     cos, sin = self.rotary_emb(position_ids)

    #     # rotary_emb 返回 [B, L, D]
    #     assert cos.shape == (B, L, D)
    #     assert sin.shape == (B, L, D)

    #     return cos, sin

    def forward_apply_rotary_emb(
        self,
        states: torch.FloatTensor,
        cos: torch.FloatTensor,
        sin: torch.FloatTensor,
    ) -> torch.FloatTensor:
        # [Batch_Size, Num_Heads_Q / Num_Heads_KV, Seq_Len, Head_Dim]
        bsz, num_heads, seq_len, head_dim = states.shape
        if cos.dim() == 2:
            # 漏网之鱼：[max_pos, D] → 截取 → [S, D] → [1, 1, S, D]
            cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            # [B, S, D] → [B, 1, S, D]
            if cos.shape[0] != bsz:
                B = cos.shape[0]
                H = states.shape[0] // B
                # cos = cos.repeat_interleave(H, dim=0)
                # sin = sin.repeat_interleave(H, dim=0)
                raise ValueError(
                    f"forward_apply_rotary_emb: cos batch {cos.shape[0]} != states batch {bsz}. "
                    f"states={states.shape}, cos={cos.shape}. "
                    f"Check forward_rotary_emb is returning [B, S, D]."
                    f"B = {B}, H = {H} (batch size // num_key_value_heads)"
                )
            cos = cos.unsqueeze(1)   # [B, 1, S, D]
            sin = sin.unsqueeze(1)
        elif cos.dim() == 4 and cos.shape[0] != bsz:
            # 异常 4D（非 head-expanded），修正
            cos = cos[:, :seq_len, :].unsqueeze(0) if cos.shape[0] == 1 else cos
            sin = sin[:, :seq_len, :].unsqueeze(0) if sin.shape[0] == 1 else sin
        states = apply_rotary_pos_emb(states, cos, sin)
        return states

    def repeat_kv(
        self, key_states: torch.FloatTensor, value_states: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor]:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return key_states, value_states