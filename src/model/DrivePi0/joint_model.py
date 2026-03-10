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
from typing import List, Optional, Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from src.model.kv_cache import KVCache
from src.model.DrivePi0.mixture import Mixture as GemmaMixture
from src.model.DrivePi0.qwen3_vl_mixture import Qwen3VLMixture

MIXTURE_REGISTRY = {
    "gemma": GemmaMixture,
    "qwen3_vl": Qwen3VLMixture,
}


def forward_mixture_layers(
    mixtures: nn.ModuleList,
    attention_mask: torch.Tensor,
    position_ids_all: Dict[torch.LongTensor, torch.LongTensor],
    embeds_all: Dict[torch.FloatTensor, torch.FloatTensor],
    layer_idx: int,
    post_attn_skip_names: List[str] = ["vlm"],
    kv_caches: Dict[KVCache, KVCache] = {},
    cache_mode: str = "append_non_active",
    time_cond: Optional[torch.FloatTensor] = None,
    attn_softclamp: float = 50.0,  # default in gemma
) -> Dict[torch.FloatTensor, torch.FloatTensor]:
    """Run one decoder layer for every active mixture.

    The per-mixture flow is:
    input_layernorm -> joint attention across all active mixtures
    -> post-attn residual -> post_attention_layernorm
    -> mlp -> final residual.

    `post_attn_skip_names` is only meant for the final layer optimization.
    Mixtures listed there still participate in attention as query/key/value so
    they can affect other mixtures, but their own states stop after attention
    and this function returns `None` for them.
    """
    active_mixture_names = list(embeds_all.keys())

    # [Batch_Size, Seq_Len, Hidden_Size]
    residuals_pre_attn = embeds_all
    hidden_states_input_norm = {}
    for name in active_mixture_names:
        hidden_states_input_norm[name] = mixtures[name].layer_func(
            "forward_norm",
            layer_idx,
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
        attn_softclamp=attn_softclamp,
    )
    hidden_states_pre_res = hidden_states_post_attn

    # Mixtures in post_attn_skip_names intentionally stop here and become None.
    # This is used when we only need them to contribute to attention, but no
    # longer need their own hidden states afterwards.
    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_post_res = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_post_res[name] = None
        else:
            hidden_states_pre_res[name] = mixtures[name].layer_func(
                "forward_adaptive_scale",
                layer_idx,
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
                layer_idx,
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
                layer_idx,
                hidden_states_pre_mlp[name],
            )
    hidden_states_pre_final_res = hidden_states_pos_mlp

    # [Batch_Size, Seq_Len, Hidden_Size]
    hidden_states_final = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            hidden_states_final[name] = None
        else:
            hidden_states_pre_final_res[name] = mixtures[name].layer_func(
                "forward_adaptive_scale",
                layer_idx,
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
    layer_idx: int,
    post_attn_skip_names: List[str] = ["vlm"],
    kv_caches: Dict[KVCache, KVCache] = {},
    cache_mode: str = "append_non_active",
    attn_softclamp: float = 50.0,  # default in gemma
    attention_dropout: float = 0.0,
) -> Dict[torch.FloatTensor, torch.FloatTensor]:
    """Run one shared attention step over the concatenated mixture sequences.

    Each active mixture computes its own Q/K/V projections, then all sequences
    are concatenated along the token dimension to form one joint attention.
    The output is split back to per-mixture tensors afterwards.

    If a mixture name appears in `post_attn_skip_names`, attention is still
    computed for it so it can influence other mixtures, but its projected
    output is returned as `None`.
    """
    assert cache_mode in [
        "no_append",
        "append",
        "append_non_active",
    ], f"Invalid cache mode: {cache_mode}"
    bsz = len(attention_mask)
    q_lens = [hidden_states.size(1) for hidden_states in hidden_states_all.values()]
    active_mixture_names = list(hidden_states_all.keys())

    # always re-compute queries
    query_states_all = {}
    for name in active_mixture_names:
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = mixtures[name].attn_func(
            "forward_q_proj", layer_idx, hidden_states_all[name]
        )
        query_states_all[name] = query_states

    # use kv caches from non-active mixtures
    key_states_all = {}
    value_states_all = {}
    if cache_mode == "append_non_active":
        for name, kv_cache in kv_caches.items():
            if name not in active_mixture_names:
                key_states_all[name], value_states_all[name] = kv_cache.get(layer_idx)

    # the caching logic below can be much simplified if we ignore the "no_append" mode, which is only used in the naive action inference mode
    for name in active_mixture_names:
        # prepare rope
        query_states = query_states_all[name]
        rope_cos, rope_sin = mixtures[name].attn_func(
            "forward_rotary_emb", layer_idx, query_states, position_ids_all[name]
        )

        # always use kv cache if it has the current layer
        flag_cached_mixture = name in kv_caches and kv_caches[name].has_item(layer_idx)
        if flag_cached_mixture:
            key_states_cached, value_states_cached = kv_caches[name].get(
                layer_idx
            )  # note: rope already applied before they were cached

        # always add to cache in append mode, or kv cache does not have the layer yet (in no_append mode)
        flag_to_cache_mixture = (
            name in kv_caches and not kv_caches[name].has_item(layer_idx)
        ) or cache_mode == "append"

        # calculate kv for new tokens if in append mode or this layer is not cached
        key_states_new, value_states_new = None, None
        flag_calc_new_kv = not flag_cached_mixture or cache_mode == "append"
        assert (
            flag_cached_mixture or flag_calc_new_kv
        ), "Cannot skip new kv calculation while also not using cache!"
        if flag_calc_new_kv:
            hidden_states = hidden_states_all[name]
            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            key_states_new = mixtures[name].attn_func(
                "forward_k_proj", layer_idx, hidden_states
            )
            value_states_new = mixtures[name].attn_func(
                "forward_v_proj", layer_idx, hidden_states
            )

            # [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            key_states_new = mixtures[name].attn_func(
                "forward_apply_rotary_emb",
                layer_idx,
                key_states_new,
                rope_cos,
                rope_sin,
            )

            if flag_to_cache_mixture:
                kv_caches[name].update(
                    key_states_new,
                    value_states_new,
                    layer_idx,
                )

        # always apply rope to Q
        # [Batch_Size, Num_Heads_Q, Seq_Len, Head_Dim]
        query_states = mixtures[name].attn_func(
            "forward_apply_rotary_emb", layer_idx, query_states, rope_cos, rope_sin
        )
        query_states_all[name] = query_states

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

    # Repeat the key and values to match the number of heads of the query
    for name in key_states_all:
        key_states, value_states = mixtures[name].attn_func(
            "repeat_kv",
            layer_idx,
            key_states_all[name],
            value_states_all[name],
        )
        key_states_all[name] = key_states
        value_states_all[name] = value_states

    # Concatenate all the blocks along sequence
    # [Batch_Size, Num_Heads_Q / Num_Heads_KV, Full_Seq_Len, Head_Dim]
    query_states = torch.cat(tuple(query_states_all.values()), dim=-2)
    key_states = torch.cat(tuple(key_states_all.values()), dim=-2)
    value_states = torch.cat(tuple(value_states_all.values()), dim=2)

    # Perform the calculation as usual, Q * K^T / sqrt(head_dim)
    # [Batch_Size, Num_Heads_Q, Full_Seq_Len, Full_Seq_Len]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(
        mixtures[active_mixture_names[0]].head_dim
    )

    # Soft capping
    if attn_softclamp is not None:
        attn_weights = attn_weights / attn_softclamp
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * attn_softclamp

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

    # Split into the different mixtures
    attn_outputs = torch.split(attn_output, q_lens, dim=1)
    attn_outputs = {
        key: value for key, value in zip(active_mixture_names, attn_outputs)
    }

    # Multiply by W_o. [Batch_Size, Seq_Len_Q, Hidden_Size]
    attn_outputs_final = {}
    for name in active_mixture_names:
        if name in post_attn_skip_names:
            attn_outputs_final[name] = None
        else:
            attn_outputs_final[name] = mixtures[name].attn_func(
                "forward_o_proj", layer_idx, attn_outputs[name]
            )
    return attn_outputs_final


class JointModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_hidden_layers = config.num_hidden_layers
        self.num_mixture = len(config.mixture)
        self.cache_names = [
            name for name in config.mixture if config.mixture[name].cache
        ]  # name of the mixtures that use cache during generation; no cache during training

        # Mixtures
        self.mixtures = nn.ModuleDict()
        for mixture_name, mixture_config in config.mixture.items():
            mixture_config = OmegaConf.merge(config, mixture_config)
            arch = mixture_config.get("arch", "gemma")
            self.mixtures[mixture_name] = MIXTURE_REGISTRY[arch](mixture_config)
        self.mixture_names = list(config.mixture.keys())

    def build_mixture_caches(self):
        return {name: KVCache() for name in self.cache_names}

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids_all: Dict[torch.LongTensor, torch.LongTensor],
        embeds_all: Dict[torch.FloatTensor, torch.FloatTensor],
        time_cond: Optional[torch.FloatTensor] = None,
        final_layer_post_attn_skip_names: List[str] = ["vlm"],
        kv_caches: Dict[KVCache, KVCache] = {},
        cache_mode: str = "append_non_active",
        return_caches: bool = False,
        deepstack_visual_embeds: torch.Tensor=None,
        visual_pos_masks: List[torch.Tensor]=None,
    ) -> Dict[torch.FloatTensor, torch.FloatTensor]:
        """
        Assume attention_mask is already in the desired block-attention form.

        `embeds_all` and `position_ids_all` must share the same order, e.g.
        {"vlm": ..., "proprio": ..., "action": ...}.

        `final_layer_post_attn_skip_names` only applies to the last layer.
        Mixtures listed there still participate in the last-layer attention,
        but their returned hidden state becomes `None` after that layer.
        """
        active_mixture_names = list(embeds_all.keys())
        # normalization
        # [Batch_Size, Seq_Len, Hidden_Size]
        for name in active_mixture_names:
            if getattr(self.mixtures[name], "scale_input_embeds", True):
                hidden_size = embeds_all[name].shape[-1]
                normalizer = torch.tensor(
                    hidden_size ** 0.5,
                    dtype=embeds_all[name].dtype,
                    device=embeds_all[name].device,
                )
                embeds_all[name] = embeds_all[name] * normalizer
        # layers
        for layer_idx in range(self.num_hidden_layers):
            is_final_layer = layer_idx == self.num_hidden_layers - 1
            embeds_all = forward_mixture_layers(
                self.mixtures,
                attention_mask,
                position_ids_all,
                embeds_all,
                layer_idx=layer_idx,
                time_cond=time_cond,
                kv_caches=kv_caches,
                cache_mode=cache_mode,
                post_attn_skip_names=final_layer_post_attn_skip_names
                if is_final_layer
                else [],
                attn_softclamp=self.config.get("attn_softclamp", None),
            )
            if (
                "vlm" in embeds_all
                and embeds_all["vlm"] is not None
                and deepstack_visual_embeds is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                embeds_all["vlm"] = self.mixtures["vlm"].apply_deepstack(
                    embeds_all["vlm"],
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states_all = {}
        for name in active_mixture_names:
            hidden_states = embeds_all[name]
            hidden_states_all[name] = (
                None
                if hidden_states is None
                else self.mixtures[name].forward_norm(hidden_states, time_cond)
            )
        if return_caches:
            return hidden_states_all, kv_caches
        return hidden_states_all


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
