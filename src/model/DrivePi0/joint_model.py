"""
Wrapper around the mixtures (without encoder / decoder)

Agnostic to the mixture setup

KV caches --- There are a few different modes depending on the setting:
    - text generation, only vlm active, use vlm cache --- append active (mode="append")
    - action naive inference, all active, use vlm and proprio cache --- no new tokens for the active mixture (mode="no_append")
    - action inference, no cache during vlm and proprio forward, then use vlm and proprio cache --- append, non-active (mode="append_non_active")
    - action flow matching training, all active, no cache (mode does not matter)
"""

import math
from typing import List, Optional, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.model.kv_cache import KVCache
from src.model.DrivePi0.mixture import Mixture
from src.model.DrivePi0.qwen3mixture import VLMMixtureAdapter


def forward_mixture_layers(
    mixtures: nn.ModuleList,
    attention_mask: torch.Tensor,
    position_ids_all: Dict[torch.LongTensor, torch.LongTensor],
    embeds_all: Dict[torch.FloatTensor, torch.FloatTensor],
    layer_idx: Union[int, Dict[str, int]],
    post_attn_skip_names: List[str] = ["vlm"],
    kv_caches: Dict[KVCache, KVCache] = {},
    cache_mode: str = "append_non_active",
    time_cond: Optional[torch.FloatTensor] = None,
    position_embeddings_all: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {},
) -> Dict[torch.FloatTensor, torch.FloatTensor]:
    """the usual norm + attn + res + norm + mlp + res"""
    # ── 辅助：按 mixture 名称取各自的 layer_idx ──────────────────────
    def get_lidx(name: str) -> int:
        if isinstance(layer_idx, dict):
            return layer_idx[name]
        return layer_idx
    
    active_mixture_names = list(embeds_all.keys())
    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_attn = embeds_all
    
    hidden_states_input_norm = {}
    for name in active_mixture_names:
        hidden_states_input_norm[name] = mixtures[name].layer_func(
            "forward_norm",
            get_lidx(name),
            "input_layernorm",
            embeds_all[name],
            time_cond,
        )  # a bit convoluted
    hidden_states_pre_attn = hidden_states_input_norm

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_attn = forward_mixture_attn(
        mixtures,
        hidden_states_all=hidden_states_pre_attn,
        attention_mask=attention_mask,
        position_ids_all=position_ids_all,
        layer_idx=layer_idx,
        post_attn_skip_names=post_attn_skip_names,
        kv_caches=kv_caches,
        cache_mode=cache_mode,
        position_embeddings_all=position_embeddings_all,
    )
    hidden_states_pre_res = hidden_states_post_attn

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_res = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            # hidden_states_post_res[name] = None
            # ── 关键修复：保留原始 hidden states，不置 None ──────────
            hidden_states_post_res[name] = residuals_pre_attn[name]
        else:
            hidden_states_pre_res[name] = mixtures[name].layer_func(
                "forward_adaptive_scale",
                get_lidx(name),
                "post_attn",
                hidden_states_pre_res[name],
                time_cond,
            )
            hidden_states_post_res[name] = (
                residuals_pre_attn[name] + hidden_states_pre_res[name]
            )
    hidden_states_pre_post_attn = hidden_states_post_res

    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_post_attn = hidden_states_pre_post_attn
    
    hidden_states_post_post_attn = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_post_post_attn[name] = None
        else:
            hidden_states_post_post_attn[name] = mixtures[name].layer_func(
                "forward_norm",
                get_lidx(name),
                "post_attention_layernorm",
                hidden_states_pre_post_attn[name],
                time_cond,
            )
    hidden_states_pre_mlp = hidden_states_post_post_attn
    
    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_pos_mlp = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_pos_mlp[name] = None
        else:
            hidden_states_pos_mlp[name] = mixtures[name].layer_func(
                "mlp",
                get_lidx(name),
                hidden_states_pre_mlp[name],
            )
    hidden_states_pre_final_res = hidden_states_pos_mlp

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_final = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            # hidden_states_final[name] = None
            # ── 同上：保留 VLM 的 hidden states ──────────────────────
            hidden_states_final[name] = residuals_pre_attn[name]
        else:
            hidden_states_pre_final_res[name] = mixtures[name].layer_func(
                "forward_adaptive_scale",
                get_lidx(name),
                "final",
                hidden_states_pre_final_res[name],
                time_cond,
            )
            hidden_states_final[name] = (
                residuals_pre_post_attn[name] + hidden_states_pre_final_res[name]
            )
    return hidden_states_final


def forward_mixture_attn(
    mixtures: nn.ModuleList,
    attention_mask: torch.Tensor,
    position_ids_all: Dict[torch.LongTensor, torch.LongTensor],
    hidden_states_all: Dict[torch.FloatTensor, torch.FloatTensor],
    layer_idx: Union[int, Dict[str, int]],
    post_attn_skip_names: List[str] = ["vlm"],
    kv_caches: Dict[KVCache, KVCache] = {},
    cache_mode: str = "append_non_active",
    attn_softclamp: float = 50.0,
    attention_dropout: float = 0.0,
    position_embeddings_all: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {},
) -> Dict[torch.FloatTensor, torch.FloatTensor]:
    """Assume all mixtures have the same head dim"""
    def get_lidx(name: str) -> int:
        if isinstance(layer_idx, dict):
            if name not in layer_idx:
                raise KeyError(
                    f"layer_idx dict missing key '{name}'. "
                    f"Available keys: {list(layer_idx.keys())}. "
                    f"Make sure JointModel.forward passes a complete layer_idx_dict "
                    f"covering all kv_cache names."
                )
            return layer_idx[name]
        return layer_idx
    
    assert cache_mode in [
        "no_append",
        "append",
        "append_non_active",
    ], f"Invalid cache mode: {cache_mode}"
    # bsz = len(attention_mask)
    q_lens = [hidden_states.size(1) for hidden_states in hidden_states_all.values()]
    active_mixture_names = list(hidden_states_all.keys())
    bsz = next(iter(hidden_states_all.values())).shape[0]
    # ── 核心修复：从实际 embeddings 推导 bsz，而非 len(attention_mask) ──
    # attention_mask 的第一维不一定等于真实 batch size（例如全局 mask 缓存）
    # if active_mixture_names:
    #     assert active_mixture_names, (
    #         "forward_mixture_attn called with empty hidden_states_all. "
    #         "JointModel.forward 应在 active_this_step 为空时 continue 跳过。"
    #     )
    #     bsz = next(iter(hidden_states_all.values())).shape[0]
    # else:
    #     # 无 active mixture 时（只取 kv cache），从缓存推导
    #     bsz = attention_mask.shape[0]
    # ───────────────────────────────────────────────────────────────────

    # always re-compute queries
    # ── 1. 计算所有 mixture 的 Q ─────────────────────────────────────
    query_states_all = {}
    for name in active_mixture_names:
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        position_ids = position_ids_all[name]
        # 同时传 hidden_states 和 position_ids，支持内部含 RoPE 的实现
        try:
            query_states = mixtures[name].attn_func(
                "forward_q_proj", get_lidx(name),
                hidden_states_all[name], position_ids
            )
        except TypeError:
            # 兼容不接受 position_ids 的旧实现
            query_states = mixtures[name].attn_func(
                "forward_q_proj", get_lidx(name), hidden_states_all[name]
            )
        query_states_all[name] = query_states

    # ── 2. 从非 active mixture 的 kv cache 中取 K/V ──────────────────
    # use kv caches from non-active mixtures
    key_states_all = {}
    value_states_all = {}
    if cache_mode == "append_non_active":
        for name, kv_cache in kv_caches.items():
            if name not in active_mixture_names:
                key_states_all[name], value_states_all[name] = kv_cache.get(get_lidx(name))

    # ── 3. 为 active mixture 计算 RoPE + K/V ─────────────────────────
    for name in active_mixture_names:
        query_states = query_states_all[name]
        lidx = get_lidx(name)
        position_ids = position_ids_all[name]

        # 如果有预计算的 position_embeddings（VLM），直接使用
        if name in position_embeddings_all:
            rope_cos, rope_sin = position_embeddings_all[name]
        else:
            rope_cos, rope_sin = mixtures[name].attn_func(
                "forward_rotary_emb", lidx, query_states, position_ids
            )

        # always use kv cache if it has the current layer
        flag_cached_mixture = name in kv_caches and kv_caches[name].has_item(lidx)
        if flag_cached_mixture:
            key_states_cached, value_states_cached = kv_caches[name].get(
                lidx
            )  # note: rope already applied before they were cached

        # always add to cache in append mode, or kv cache does not have the layer yet (in no_append mode)
        flag_to_cache_mixture = (
            name in kv_caches and not kv_caches[name].has_item(lidx)
        ) or cache_mode == "append"

        # calculate kv for new tokens if in append mode or this layer is not cached
        key_states_new, value_states_new = None, None
        # flag_calc_new_kv = not flag_cached_mixture or cache_mode == "append"
        # ── 核心修复：已缓存则不重算，防止 final layer 重复 action_layer_idx ──
        flag_calc_new_kv = not flag_cached_mixture
        # ────────────────────────────────────────────────────────────────────
        assert (
            flag_cached_mixture or flag_calc_new_kv
        ), "Cannot skip new kv calculation while also not using cache!"
        if flag_calc_new_kv:
            hidden_states = hidden_states_all[name]
            
            # ── 传入 position_ids 修复 forward_k_proj 内部 RoPE ──────
            try:
                key_states_new = mixtures[name].attn_func(
                    "forward_k_proj", lidx, hidden_states, position_ids
                )
                value_states_new = mixtures[name].attn_func(
                    "forward_v_proj", lidx, hidden_states, position_ids
                )
                # k/v 内部已含 RoPE，不需要再次调用 forward_apply_rotary_emb
                flag_k_rope_applied = True
            except TypeError:
                # 兼容旧的不含 RoPE 的接口
                key_states_new = mixtures[name].attn_func(
                    "forward_k_proj", lidx, hidden_states
                )
                value_states_new = mixtures[name].attn_func(
                    "forward_v_proj", lidx, hidden_states
                )
                flag_k_rope_applied = False
            
            
            # print(f"forward_mixture_attn: calculating new K/V for mixture '{name}' at layer_idx {lidx} with hidden_states shape {hidden_states.shape}")

            # print(f"forward_mixture_attn: mixture '{name}', layer_idx {lidx}, new key_states shape {key_states_new.shape}, new value_states shape {value_states_new.shape}")

            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            if not flag_k_rope_applied:
                key_states_new = mixtures[name].attn_func(
                    "forward_apply_rotary_emb", lidx,
                    key_states_new, rope_cos, rope_sin
                )

            if flag_to_cache_mixture:
                kv_caches[name].update(
                    key_states_new,
                    value_states_new,
                    lidx,
                )

        # always apply rope to Q
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        # print(f"forward_mixture_attn: applying RoPE to query_states of mixture '{name}' with shape {query_states.shape}")
        query_states = mixtures[name].attn_func(
            "forward_apply_rotary_emb", lidx, query_states, rope_cos, rope_sin
        )
        query_states_all[name] = query_states

        # 合并 cached + new K/V
        # assign K and V carefully for this active mixture
        if flag_cached_mixture:
            key_states = key_states_cached
            value_states = value_states_cached
            if key_states_new is not None:
                key_states = torch.cat((key_states, key_states_new), dim=-2)
            if value_states_new is not None:
                value_states = torch.cat((value_states, value_states_new), dim=-2)
        else:
            key_states = key_states_new
            value_states = value_states_new
        key_states_all[name] = key_states
        value_states_all[name] = value_states

    # ── 4. repeat KV heads ───────────────────────────────────────────
    # Repeat the key and values to match the number of heads of the query
    for name in key_states_all:
        key_states, value_states = mixtures[name].attn_func(
            "repeat_kv",
            get_lidx(name),
            key_states_all[name],
            value_states_all[name],
        )
        key_states_all[name] = key_states
        value_states_all[name] = value_states

    # ── 5. 拼接所有 mixture 的 Q/K/V ─────────────────────────────────
    # Concatenate all the blocks along sequence
    # [Batch_Size, Num_Heads_Q / Num_Heads_KV, Full_Seq_Len, Head_Dim]
    query_states = torch.cat(tuple(query_states_all.values()), dim=-2)
    key_states = torch.cat(tuple(key_states_all.values()), dim=-2)
    value_states = torch.cat(tuple(value_states_all.values()), dim=2)

    # ── 6. 标准注意力计算 ─────────────────────────────────────────────
    first_name = active_mixture_names[0]
    head_dim = mixtures[first_name].head_dim
    # Perform the calculation as usual, Q * K^T / sqrt(head_dim)
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)
                                ) / math.sqrt(head_dim)

    # Soft capping Qwen3可以不用
    # attn_weights = attn_weights / attn_softclamp
    # attn_weights = torch.tanh(attn_weights)
    # attn_weights = attn_weights * attn_softclamp
    
    # ── 动态裁剪 mask 到实际 Q/KV 序列长度 ──────────────────────────
    # 原因：Q 可能因 skip_action 变短，KV 可能因非 active 缓存变长，
    # 传入的 attention_mask 是按最大序列构建的，需要裁剪到实际尺寸
    actual_q_len  = attn_weights.shape[-2]
    actual_kv_len = attn_weights.shape[-1]
    if attention_mask.shape[-2] != actual_q_len or \
       attention_mask.shape[-1] != actual_kv_len:
        attention_mask = attention_mask[
            :, :, :actual_q_len, :actual_kv_len
        ]
    # ────────────────────────────────────────────────────────────────
    
    # ── 保险：强制 mask 与 attn_weights dtype 一致 ──────────────────
    # 防止 mask 为 float32 而 attn_weights 为 bfloat16 导致反向传播崩溃
    if attention_mask.dtype != attn_weights.dtype:
        attention_mask = attention_mask.to(attn_weights.dtype)
    # ────────────────────────────────────────────────────────────────

    # Apply the softmax / dropout
    attn_weights = attn_weights + attention_mask
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
        query_states.dtype
    )
    attn_weights = nn.functional.dropout(
        attn_weights,
        p=attention_dropout,
        training=mixtures[active_mixture_names[0]].training,
    )
    # Multiply by the values. [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len] x [Batch_Size, Num_Heads_KV, Full_Seq_Len, Head_Dim] -> [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim]
    attn_output = torch.matmul(attn_weights, value_states)

    # Make sure the sequence length is the second dimension. # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim]
    attn_output = attn_output.transpose(1, 2).contiguous()
    # Concatenate all the heads together. [Batch_Size, Full_Seq_Len, Num_Heads_Q, Head_Dim] -> [Batch_Size, Full_Seq_Len, Num_Heads_Q * Head_Dim]
    attn_output = attn_output.view(bsz, sum(q_lens), -1)

    # ── 7. 按 mixture 切分输出 ────────────────────────────────────────
    # Split into the different mixtures
    attn_outputs = torch.split(attn_output, q_lens, dim=1)
    attn_outputs = {
        key: value for key, value in zip(active_mixture_names, attn_outputs)
    }
    # attn_outputs = dict(zip(
    #     active_mixture_names,
    #     torch.split(attn_output, q_lens, dim=1)
    # ))

    # ── 8. W_o 投影 ───────────────────────────────────────────────────
    # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
    attn_outputs_final = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            attn_outputs_final[name] = None
        else:
            attn_outputs_final[name] = mixtures[name].attn_func(
                "forward_o_proj", get_lidx(name), attn_outputs[name]
            )
    return attn_outputs_final


class JointModel(nn.Module):
    def __init__(self, config, pretrained_language_model=None):
        """
        Args:
            pretrained_language_model: 由 DrivePiZero._build_qwen3_modules 构建的
                Qwen3 language model 实例。必须传入，不再内部加载。
        """
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers  # 28

        # action expert 层数为 14（从 VLM 每 2 层获取一次）
        self.num_action_layers = config.num_hidden_layers // 2  # 14

        self.num_mixture = len(config.mixture)
        self.cache_names = [
            name for name in config.mixture if config.mixture[name].cache
        ]

        # Action / Proprio Mixtures（从 config 构建）
        self.mixtures = nn.ModuleDict()
        config_dict = OmegaConf.to_container(config, resolve=True)
        for mixture_name, mixture_config in config.mixture.items():
            mixture_dict = OmegaConf.to_container(mixture_config, resolve=True)
            merged = {**config_dict, **mixture_dict}
            mixture_config = OmegaConf.create(merged)
            if mixture_name in ["action", "proprio"]:
                mixture_config.num_hidden_layers = self.num_action_layers
            self.mixtures[mixture_name] = Mixture(mixture_config)

        # VLM Mixture（接收外部构建好的 language_model）
        assert pretrained_language_model is not None, (
            "pretrained_language_model must be provided by DrivePiZero._build_qwen3_modules"
        )
        self.mixtures["vlm"] = VLMMixtureAdapter(pretrained_language_model)
        self.mixture_names = list(config.mixture.keys())

    def build_mixture_caches(self):
        return {name: KVCache() for name in self.cache_names}

    def forward(
        self,
        attention_mask: torch.Tensor,
        # vlm_attention_mask: torch.Tensor,
        position_ids_all: Dict[torch.LongTensor, torch.LongTensor],
        embeds_all: Dict[torch.FloatTensor, torch.FloatTensor],
        time_cond: Optional[torch.FloatTensor] = None,
        final_layer_post_attn_skip_names: List[str] = ["vlm"],
        kv_caches: Dict[KVCache, KVCache] = {},
        cache_mode: str = "append_non_active",
        return_caches: bool = False,
        deepstack_data: Optional[Tuple] = None,
    ) -> Dict[torch.FloatTensor, torch.FloatTensor]:
        """
        Assume attention_mask is in the right block attention form

        embeds_all and position_ids_all need to be in the correct order, e.g., {"vlm": ..., "proprio": ..., "action": ...}
        """
        active_mixture_names = list(embeds_all.keys())

        # 预计算 VLM 的 position_embeddings (cos, sin)，所有层共享
        # 对齐官方 Qwen3VLTextModel: position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_all = {}
        if "vlm" in active_mixture_names and "vlm" in position_ids_all:
            vlm_pos_ids = position_ids_all["vlm"]
            vlm_embeds = embeds_all["vlm"]
            vlm_cos, vlm_sin = self.mixtures["vlm"].model.rotary_emb(
                vlm_embeds, vlm_pos_ids
            )
            position_embeddings_all["vlm"] = (vlm_cos, vlm_sin)

        # layers
        for vlm_layer_idx in range(self.num_hidden_layers):   # 0-27
            is_final_layer   = vlm_layer_idx == self.num_hidden_layers - 1
            action_layer_idx = vlm_layer_idx // 2             # 0-13          

            # ── 构建 per-mixture layer_idx 字典 ──────────────────────────
            # layer_idx_dict: Dict[str, int] = {}
            # for name in active_mixture_names:
            #     if name in ("action", "proprio"):
            #         layer_idx_dict[name] = action_layer_idx
            #     else:                          # vlm
            #         layer_idx_dict[name] = vlm_layer_idx

            # ── action/proprio 每 2 个 VLM 层才前进一次 ──────────────────
            # 奇数 VLM 层时，跳过 action/proprio 的 MLP 更新（只做一次）
            # 最简单的做法：仅在偶数 VLM 层（或最后一层）执行 action/proprio
            skip_action = (vlm_layer_idx % 2 == 1) and not is_final_layer
            current_mask = attention_mask
            active_this_step = [
                n for n in active_mixture_names
                if not (skip_action and n in ("action", "proprio"))
            ]
            # ── 核心修复：active_this_step 为空时直接跳过 ────────────────
            # 场景：phase 2 action-only 推理的奇数 VLM 层
            # action 是唯一 active mixture 却因 skip_action 被排除
            # 此时没有任何 mixture 需要更新，强行调用 forward 会导致
            # bsz 从空 dict 或 attention_mask.shape[0] 推导出错误值
            if not active_this_step:
                continue
            # ────────────────────────────────────────────────────────────
            all_known_names = set(active_mixture_names) | set(kv_caches.keys())
            layer_idx_dict = {
                n: (action_layer_idx if n in ("action", "proprio") else vlm_layer_idx)
                for n in all_known_names
            }
            embeds_this_step = {n: embeds_all[n] for n in active_this_step}
            pos_ids_this_step = {n: position_ids_all[n] for n in active_this_step
                             if n in position_ids_all}
            # layer_idx_this_step = {n: layer_idx_dict[n] for n in active_this_step}

            embeds_updated = forward_mixture_layers(
                self.mixtures,
                current_mask,
                pos_ids_this_step,
                embeds_this_step,
                layer_idx=layer_idx_dict,
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode=cache_mode,
                post_attn_skip_names=final_layer_post_attn_skip_names
                    if is_final_layer else [],
                position_embeddings_all=position_embeddings_all,
            )
            # 合并回 embeds_all
            embeds_all.update(embeds_updated)

            # P0-D: DeepStack 注入 — VLM layer 0/1/2 完成后分别注入
            if (deepstack_data is not None
                    and "vlm" in active_this_step
                    and "vlm" in embeds_all):
                visual_pos_masks, deepstack_features = deepstack_data
                if vlm_layer_idx < len(deepstack_features):
                    self.mixtures["vlm"].deepstack_inject(
                        embeds_all["vlm"], visual_pos_masks,
                        deepstack_features[vlm_layer_idx],
                    )

        hidden_states_all = {}
        for name in active_mixture_names:
            hidden_states_all[name] = self.mixtures[name].forward_norm(
                embeds_all[name], time_cond
            )
        if return_caches:
            return hidden_states_all, kv_caches
        return hidden_states_all


def safe_forward_mixture_attn(
    mixtures,
    attention_mask,
    position_ids_all,
    hidden_states_all,
    layer_idx,
    kv_caches=None,
    cache_mode="append_non_active",
    attention_dropout=0.0,
):

    assert cache_mode in ["append", "no_append", "append_non_active"]

    if kv_caches is None:
        kv_caches = {}

    B = attention_mask.shape[0]
    active_names = list(hidden_states_all.keys())

    query_all = {}
    key_all = {}
    value_all = {}
    q_lens = []

    # --------------------------------------------------
    # 1️⃣ 先处理 active mixture（顺序固定）
    # --------------------------------------------------

    for name in active_names:

        hidden = hidden_states_all[name]
        B2, L, _ = hidden.shape
        assert B2 == B, f"{name} batch mismatch"

        q_lens.append(L)

        # ---- Q ----
        q = mixtures[name].attn_func(
            "forward_q_proj", layer_idx, hidden
        )

        cos, sin = mixtures[name].attn_func(
            "forward_rotary_emb",
            layer_idx,
            q,
            position_ids_all[name],
        )

        q = mixtures[name].attn_func(
            "forward_apply_rotary_emb",
            layer_idx,
            q,
            cos,
            sin,
        )

        # ---- KV ----
        has_cache = (
            name in kv_caches and kv_caches[name].has_item(layer_idx)
        )

        if cache_mode == "no_append" and has_cache:
            k, v = kv_caches[name].get(layer_idx)
        else:
            k = mixtures[name].attn_func(
                "forward_k_proj", layer_idx, hidden
            )
            v = mixtures[name].attn_func(
                "forward_v_proj", layer_idx, hidden
            )

            k = mixtures[name].attn_func(
                "forward_apply_rotary_emb", layer_idx, k, cos, sin
            )

            if name in kv_caches and cache_mode in ["append", "append_non_active"]:
                kv_caches[name].update(k, v, layer_idx)

        k, v = mixtures[name].attn_func(
            "repeat_kv", layer_idx, k, v
        )

        query_all[name] = q
        key_all[name] = k
        value_all[name] = v

    # --------------------------------------------------
    # 2️⃣ 处理 non-active（只追加 KV，不影响 Q 顺序）
    # --------------------------------------------------

    if cache_mode == "append_non_active":

        for name in kv_caches:

            if name in active_names:
                continue

            if kv_caches[name].has_item(layer_idx):

                k, v = kv_caches[name].get(layer_idx)

                k, v = mixtures[name].attn_func(
                    "repeat_kv", layer_idx, k, v
                )

                key_all[name] = k
                value_all[name] = v

    # --------------------------------------------------
    # 3️⃣ 拼接（顺序严格控制）
    # --------------------------------------------------

    # Q 只来自 active
    Q = torch.cat([query_all[n] for n in active_names], dim=-2)

    # K/V：active 按 active 顺序 + non-active 按缓存顺序
    kv_order = list(active_names) + [
        n for n in key_all.keys() if n not in active_names
    ]

    K = torch.cat([key_all[n] for n in kv_order], dim=-2)
    V = torch.cat([value_all[n] for n in kv_order], dim=-2)

    head_dim = mixtures[active_names[0]].head_dim

    assert Q.shape[-1] == head_dim
    assert K.shape == V.shape
    assert attention_mask.shape[-1] == K.shape[-2]

    # --------------------------------------------------
    # 4️⃣ attention
    # --------------------------------------------------

    attn = torch.matmul(Q, K.transpose(-2, -1))
    attn = attn / math.sqrt(head_dim)

    attn = attn + attention_mask
    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(
        attn,
        p=attention_dropout,
        training=mixtures[active_names[0]].training,
    )

    out = torch.matmul(attn, V)

    # --------------------------------------------------
    # 5️⃣ split 回 active（顺序 100% 对齐）
    # --------------------------------------------------

    out = out.transpose(1, 2).contiguous()
    out = out.view(B, sum(q_lens), -1)

    splits = torch.split(out, q_lens, dim=1)

    outputs = {}

    for name, chunk in zip(active_names, splits):

        outputs[name] = mixtures[name].attn_func(
            "forward_o_proj", layer_idx, chunk
        )

    return outputs

def safe_forward_mixture_layers(
    mixtures,
    attention_mask,
    position_ids_all,
    embeds_all,
    layer_idx,
    post_attn_skip_names=["vlm"],
    kv_caches=None,
    cache_mode="append_non_active",
    time_cond=None,
):

    active_mixture_names = list(embeds_all.keys())

    # 🔥 关键：copy，避免引用传递
    residuals_pre_attn = {
        k: v for k, v in embeds_all.items()
    }

    hidden_states_input_norm = {}

    for name in active_mixture_names:
        hidden_states_input_norm[name] = mixtures[name].layer_func(
            "forward_norm",
            layer_idx,
            "input_layernorm",
            embeds_all[name],
            time_cond,
        )

    hidden_states_pre_attn = hidden_states_input_norm

    hidden_states_post_attn = safe_forward_mixture_attn(
        mixtures,
        attention_mask=attention_mask,
        position_ids_all=position_ids_all,
        hidden_states_all=hidden_states_pre_attn,
        layer_idx=layer_idx,
        kv_caches=kv_caches,
        cache_mode=cache_mode,
    )

    hidden_states_pre_res = hidden_states_post_attn

    hidden_states_post_res = {}

    for name in active_mixture_names:

        if name in post_attn_skip_names:
            hidden_states_post_res[name] = None
            continue

        scaled = mixtures[name].layer_func(
            "forward_adaptive_scale",
            layer_idx,
            "post_attn",
            hidden_states_pre_res[name],
            time_cond,
        )

        assert residuals_pre_attn[name].shape == scaled.shape, (
            name,
            residuals_pre_attn[name].shape,
            scaled.shape,
        )

        hidden_states_post_res[name] = (
            residuals_pre_attn[name] + scaled
        )

    return hidden_states_post_res

if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("config/train/DrivePi0/base.yaml")
    model = JointModel(cfg.joint.config)

    # dummy inputs
    dummy_num_image_tokens = 7
    q_lens = [
        dummy_num_image_tokens,
        cfg.cond_steps,
        cfg.horizon_steps,
    ]  # not considering text (padding)
    total_len = sum(q_lens)
    inputs_embeds = torch.randn(
        1,
        dummy_num_image_tokens,
        cfg.mixture.vlm.hidden_size,
    )  # no history
    proprio_embeds = torch.randn(
        1,
        cfg.cond_steps,
        cfg.mixture.proprio.hidden_size,
    )
    action_embeds = torch.randn(
        1,
        cfg.horizon_steps,
        cfg.mixture.action.hidden_size,
    )
    time_cond = None
    if cfg.action_expert_adaptive_mode:
        time_cond = torch.randn(1, cfg.time_hidden_size)

    kv_caches = model.build_mixture_caches()
    position_ids_all = {
        "vlm": torch.arange(dummy_num_image_tokens)[None],
        "proprio": torch.arange(cfg.cond_steps)[None],
        "action": torch.arange(cfg.horizon_steps)[None],
    }  # add batch dim

    # block attention
    proprio_start = dummy_num_image_tokens
    proprio_end = dummy_num_image_tokens + 1
    action_start = proprio_end
    causal_mask = torch.full(
        (1, total_len, total_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )  # smallest value, avoid using inf for softmax nan issues with padding
    causal_mask[:, :dummy_num_image_tokens, :dummy_num_image_tokens] = (
        0  # image/text attend to itself
    )
    causal_mask[:, proprio_start:proprio_end, :dummy_num_image_tokens] = (
        0  # proprio attend to image/text
    )
    causal_mask[:, action_start:, :dummy_num_image_tokens] = (
        0  # action attend to image/text
    )
    causal_mask[:, proprio_start:proprio_end, proprio_start:proprio_end] = (
        0  # proprio attend to itself
    )
    causal_mask[:, action_start:, proprio_start:] = (
        0  # action attend to itself and proprio
    )

    # Add the head dimension
    # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
    causal_mask = causal_mask.unsqueeze(1)

    # dummy denoising - naive action inference
    print("Initial action embeds", action_embeds)
    num_step = 3
    for _step in range(num_step):
        print("running dummy denoising step", _step)
        action_embeds = model(
            attention_mask=causal_mask,
            position_ids_all=position_ids_all,
            embeds_all={
                "vlm": inputs_embeds,
                "proprio": proprio_embeds,
                "action": action_embeds,
            },
            kv_caches=kv_caches,
            time_cond=time_cond,
            cache_mode="no_append",
        )["action"]
        print("Updated action embeds", action_embeds)