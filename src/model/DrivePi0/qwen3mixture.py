import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

class VLMMixtureAdapter(nn.Module):
    """
    将 Qwen3VLTextModel 包装成与 Mixture 相同的 layer_func/attn_func 接口，
    使其能参与 forward_mixture_layers 的联合注意力计算。
    """
    def __init__(self, vlm_text_model: nn.Module):
        super().__init__()
        self.model = vlm_text_model
        cfg = vlm_text_model.config
        self._head_dim     = cfg.hidden_size // cfg.num_attention_heads
        self._num_heads    = cfg.num_attention_heads
        self._num_kv_heads = cfg.num_key_value_heads

    # ── 供 forward_mixture_attn 读取 ──────────────────────────────────
    @property
    def head_dim(self) -> int:
        return self._head_dim

    # ── 供 forward_mixture_layers 调用 ───────────────────────────────
    def layer_func(self, func_name: str, layer_idx: int, *args):
        layer = self.model.layers[layer_idx]

        if func_name == "forward_norm":
            # args: (norm_attr_name: str, hidden_states, time_cond)
            norm_name, hidden_states = args[0], args[1]
            if hidden_states is None:
                return None
            out = getattr(layer, norm_name)(hidden_states)
            # ── LayerNorm 内部可能提升为 float32，强制转回 ──────────
            return out.to(hidden_states.dtype)

        elif func_name == "forward_adaptive_scale":
            # args: (scale_name: str, hidden_states, time_cond)
            # VLM 没有 adaptive scale，直接返回输入
            hidden_states = args[1]
            return hidden_states

        elif func_name == "mlp":
            # args: (hidden_states,)
            hidden_states = args[0]
            if hidden_states is None:
                return None
            src_dtype = hidden_states.dtype
            out = layer.mlp(hidden_states)
            # ── MLP 内部 SiLU/gate 可能提升 dtype，强制转回 ─────────
            return out.to(src_dtype)

        else:
            raise ValueError(f"VLMMixtureAdapter.layer_func: 未知 func_name='{func_name}'")

    def attn_func(self, func_name: str, layer_idx: int, *args):
        layer = self.model.layers[layer_idx]
        attn  = layer.self_attn

        if func_name == "forward_q_proj":
            hidden_states = args[0]
            src_dtype = hidden_states.dtype
            bsz, q_len, _ = hidden_states.shape
            q = attn.q_proj(hidden_states)
            q = q.view(bsz, q_len, self._num_heads, self._head_dim).transpose(1, 2)
            if hasattr(attn, "q_norm"):
                q = attn.q_norm(q)
            # ── Linear + norm 可能提升，强制转回 ────────────────────
            return q.to(src_dtype)

        elif func_name == "forward_k_proj":
            hidden_states = args[0]
            src_dtype = hidden_states.dtype
            bsz, seq_len, _ = hidden_states.shape
            k = attn.k_proj(hidden_states)
            k = k.view(bsz, seq_len, self._num_kv_heads, self._head_dim).transpose(1, 2)
            if hasattr(attn, "k_norm"):
                k = attn.k_norm(k)
            return k.to(src_dtype)

        elif func_name == "forward_v_proj":
            hidden_states = args[0]
            src_dtype = hidden_states.dtype
            bsz, seq_len, _ = hidden_states.shape
            v = attn.v_proj(hidden_states)
            v = v.view(bsz, seq_len, self._num_kv_heads, self._head_dim).transpose(1, 2)
            return v.to(src_dtype)

        elif func_name == "forward_rotary_emb":
            query_states, position_ids = args[0], args[1]
            cos, sin = self.model.rotary_emb(query_states, position_ids)
            # if cos.dim() == 2:
            #     # [max_seq_len, head_dim] → 用 position_ids 索引 → [B, S, head_dim]
            #     cos = cos[position_ids]   # position_ids: [B, S] → [B, S, head_dim]
            #     sin = sin[position_ids]
            # elif cos.dim() == 3 and cos.shape[0] != position_ids.shape[0]:
            #     # [1, S, head_dim] 或其他 batch 不匹配的情况 → 直接索引
            #     # 先展平再恢复 batch 维
            #     cos = cos.squeeze(0)[position_ids]   # → [B, S, head_dim]
            #     sin = sin.squeeze(0)[position_ids]
            cos = cos.to(query_states.dtype)
            sin = sin.to(query_states.dtype)
            # print(f"VLMMixtureAdapter.attn_func: forward_rotary_emb 生成 cos/sin 形状 {cos.shape}，dtype {cos.dtype}")
            return cos, sin

        elif func_name == "forward_apply_rotary_emb":
            states, cos, sin = args[0], args[1], args[2]
            src_dtype = states.dtype
            cos = cos.to(src_dtype)
            sin = sin.to(src_dtype)
            states = self._apply_rotary_vlm(states, cos, sin)
            return states.to(src_dtype)

        elif func_name == "repeat_kv":
            key_states, value_states = args[0], args[1]
            n_rep = self._num_heads // self._num_kv_heads
            if n_rep == 1:
                return key_states, value_states
            bsz, nkv, slen, hdim = key_states.shape
            key_states   = key_states[:, :, None].expand(bsz, nkv, n_rep, slen, hdim).reshape(bsz, nkv * n_rep, slen, hdim)
            value_states = value_states[:, :, None].expand(bsz, nkv, n_rep, slen, hdim).reshape(bsz, nkv * n_rep, slen, hdim)
            return key_states, value_states

        elif func_name == "forward_o_proj":
            hidden_states = args[0]
            src_dtype = hidden_states.dtype
            out = attn.o_proj(hidden_states)
            # ── o_proj 可能提升，强制转回 ────────────────────────────
            return out.to(src_dtype)

        else:
            raise ValueError(f"VLMMixtureAdapter.attn_func: 未知 func_name='{func_name}'")

    def forward_norm(self, hidden_states, time_cond=None):
        """对应 Mixture.forward_norm，用于最终 layer norm。"""
        if hidden_states is None:
            return None
        src_dtype = hidden_states.dtype
        out = self.model.norm(hidden_states)
        return out.to(src_dtype)

    # ── 内部工具 ──────────────────────────────────────────────────────
    @staticmethod
    def _apply_rotary(states: torch.Tensor, cos: torch.Tensor,
                      sin: torch.Tensor, attn) -> torch.Tensor:
        """
        优先使用 Qwen3VL 的 multimodal RoPE，否则降级到标准 RoPE。
        """
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                apply_multimodal_rotary_pos_emb,
            )
            mrope_section = getattr(attn.rotary_emb, "mrope_section", None)
            if mrope_section is not None:
                # apply_multimodal_rotary_pos_emb 同时处理 q 和 k，这里只取 q
                out, _ = apply_multimodal_rotary_pos_emb(
                    states, states, cos, sin, mrope_section
                )
                return out
        except (ImportError, AttributeError):
            pass

        # 标准 rotate_half RoPE
        def rotate_half(x):
            x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        cos = cos.unsqueeze(1) if cos.dim() == 3 else cos
        sin = sin.unsqueeze(1) if sin.dim() == 3 else sin
        return states * cos + rotate_half(states) * sin
    
    def _apply_rotary_vlm(
        self,
        states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Qwen3VL 的 cos/sin 可能是 multimodal 形式 shape=[3, B, S, D]，
        也可能是普通形式 shape=[B, S, D] 或 [S, D]。
        统一处理为标准 rotate_half RoPE。
        """
        src_dtype = states.dtype
        cos = cos.to(src_dtype)
        sin = sin.to(src_dtype)
        
        # ── 2D 全缓存（新版 transformers）──────────────────────────────
        # 此分支在 forward_apply_rotary_emb 调用前 cos/sin 已通过 .to() 转型
        # 但 VLMMixtureAdapter 的 forward_rotary_emb 没有 position_ids 索引的机会
        # 所以需要在这里检查并处理
        if cos.dim() == 2:
            # [max_seq_len, head_dim]，用位置0开始截取实际seq_len
            seq_len = states.shape[-2]
            cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, S, D]
            sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
            return (states * cos + rotate_half(states) * sin).to(src_dtype)
        # multimodal RoPE: cos shape = [3, B, S, D] → 取第0维拼接的标准形式
        if cos.dim() == 4:
            # 尝试调用官方 apply_multimodal_rotary_pos_emb
            try:
                from transformers.models.qwen2_vl.modeling_qwen2_vl import (
                    apply_multimodal_rotary_pos_emb,
                )
                mrope_section = getattr(
                    self.model.rotary_emb, "mrope_section", None
                )
                if mrope_section is not None:
                    out, _ = apply_multimodal_rotary_pos_emb(
                        states, states, cos, sin, mrope_section
                    )
                    return out
            except (ImportError, AttributeError):
                pass
            # fallback：取第一个切片
            cos = cos[0]
            sin = sin[0]

        # 标准 rotate_half
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # 对齐维度 [B, S, D] → [B, 1, S, D] 或保持 [B, H, S, D]
        if cos.dim() == 3:          # [B, S, D]
            cos = cos.unsqueeze(1)  # → [B, 1, S, D]
            sin = sin.unsqueeze(1)

        return states * cos + rotate_half(states) * sin