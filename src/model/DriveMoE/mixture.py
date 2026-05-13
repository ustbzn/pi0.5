import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from src.model.lora import get_layer
from src.model.utils import ScaleGradient
from src.model.paligemma.modules import GemmaRMSNorm, GemmaMLP
from src.model.DrivePi0.mixture import MixtureAttention
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
                self.norm = GemmaRMSNorm(
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


class MoEMixture(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.layers = nn.ModuleList([
            MoEDecoderLayer(config) if i >= (config.num_hidden_layers - config.num_moe_layers) else MixtureDecoderLayer(config)
            for i in range(config.num_hidden_layers)
        ])
        
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
                self.norm = GemmaRMSNorm(
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
    ) -> Union[torch.FloatTensor, None]:
        if hasattr(self, "norm"):
            args = [x] if self.adaptive_mode is None else [x, cond]
            return self.norm(*args)
        else:
            return None
        
class MixtureDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MixtureAttention(config)

        self.mlp = NormalGemmaMLP(
            config, use_quantize=config.use_quantize, use_lora=config.use_lora
        )

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
            self.input_layernorm = GemmaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = GemmaRMSNorm(
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


class MoEDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = MixtureAttention(config)

        self.mlp = MoEGemmaMLP(
            config, use_quantize=config.use_quantize, use_lora=config.use_lora
        )

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
            self.input_layernorm = GemmaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )
            self.post_attention_layernorm = GemmaRMSNorm(
                config.hidden_size,
                eps=config.rms_norm_eps,
            )

    def forward_norm(
        self,
        norm_name: str,
        x: torch.FloatTensor,
        cond: Optional[torch.FloatTensor] = None,
    ) -> Union[torch.FloatTensor, None]:
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
    

class MoEGemmaMLP(nn.Module):
    def __init__(self, config, use_quantize=False, use_lora=False):
        super().__init__()
        self.config = config
        self.num_shared_experts = config.num_shared_experts
        self.num_skill_experts = config.num_skill_experts
        self.hidden_size = config.hidden_size
        
        self.top_k = getattr(config, "num_experts_per_tok", 2)
        self.num_moe_layers = config.num_moe_layers

        self.shared_experts = nn.ModuleList(
            [
                GemmaMLP(
                    config, use_quantize=use_quantize, use_lora=use_lora
                ) for _ in range(self.num_shared_experts)
            ]
        )

        self.skill_experts = nn.ModuleList(
            [
                GemmaMLP(
                    config, use_quantize=use_quantize, use_lora=use_lora
                ) for _ in range(self.num_skill_experts)
            ]
        )

    def forward(self, x, router_logits): 
        # x: [batch_size, seq_len, hidden_size]
        # router_logits: [batch_size, num_skill_experts]

        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        # routing_weights: [batch_size, top_k]
        # selected_experts: [batch_size, top_k]
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(x.dtype)
        
        scale_factor = 1.0 / self.num_moe_layers
        routing_weights = ScaleGradient.apply(routing_weights, scale_factor)

        final_skill_outputs = torch.zeros_like(x) # [batch_size, seq_len, hidden_size]

        for expert_idx in range(self.num_skill_experts):
            expert_layer = self.skill_experts[expert_idx]

            batch_indices, routing_indices = torch.where(selected_experts == expert_idx)
            if batch_indices.shape[0] == 0:
                continue

            # current_seqs: [num_selected_batches, seq_len, hidden_size]
            current_seqs = x[batch_indices]

            expert_output = expert_layer(current_seqs)

            expert_weights = routing_weights[batch_indices, routing_indices]
            
            expert_weights = expert_weights.unsqueeze(1).unsqueeze(2)

            weighted_output = expert_output * expert_weights

            final_skill_outputs.index_add_(0, batch_indices, weighted_output)

        if self.num_shared_experts > 0:
            share_outputs = sum([expert(x) for expert in self.shared_experts])
            output = final_skill_outputs + share_outputs
        else:
            output = final_skill_outputs

        return output
    
class NormalGemmaMLP(nn.Module):
    def __init__(self, config, use_quantize=False, use_lora=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        layer = get_layer(
            use_quantize,
            use_lora,
            **config.lora if use_lora else {},
        )
        self.gate_proj = layer(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = layer(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = layer(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x, logits=None):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )