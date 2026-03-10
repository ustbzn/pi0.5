import torch
import torch.nn as nn
from src.model.lora import get_layer


class Qwen3VisionModel(nn.Module):
    def __init__(self, config, use_quantize=False, use_lora=False):
        super().__init__()
        self.config = config

        # 基于 Qwen3-VL 的视觉编码器结构（官方实现）
        from src.model.lora import get_layer

        layer = get_layer(quantize=use_quantize, lora=use_lora, **config.lora if hasattr(config, 'lora') else {})

        # Patch embedding（官方实现）
        self.patch_embed = nn.Conv2d(
            config.in_channels,
            config.hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )

        # Position embeddings（官方实现）
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.num_position_embeddings, config.hidden_size)
        )
        
        if config.hidden_act == "silu":
            activation = nn.functional.silu
        elif config.hidden_act == "gelu_pytorch_tanh":
            activation = nn.functional.gelu
        else:
            activation = nn.functional.silu

        # Transformer layers（官方实现）
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_attention_heads,
                dim_feedforward=config.intermediate_size,
                dropout=0.0,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )
            for _ in range(config.depth)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # DeepStack: 融合多级ViT特征（官方技术）
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_layers = nn.ModuleList([
            nn.Linear(config.hidden_size, config.out_hidden_size)
            for _ in range(len(self.deepstack_visual_indexes))
        ])

    def forward(self, pixel_values):
        # pixel_values: [Batch_Size, Channels, Height, Width]
        batch_size = pixel_values.shape[0]

        # Patch embedding
        patches = self.patch_embed(pixel_values)  # [B, hidden_size, H/patch_size, W/patch_size]
        patches = patches.flatten(2).transpose(1, 2)  # [B, num_patches, hidden_size]

        # Add position embeddings
        patches = patches + self.position_embeddings[:, :patches.size(1), :]

        # Pass through transformer layers并收集DeepStack特征
        deepstack_features = []
        for i, layer in enumerate(self.layers):
            patches = layer(patches)
            if i in self.deepstack_visual_indexes:
                deepstack_features.append(patches)

        # Apply final norm
        patches = self.norm(patches)

        # DeepStack: 融合多级特征
        if len(deepstack_features) > 0:
            stacked_features = []
            for i, features in enumerate(deepstack_features):
                stacked_features.append(self.deepstack_layers[i](features))
            # 取最后一层的特征作为主要输出
            patches = stacked_features[-1]

        return patches


class Qwen3MultiModalProjector(nn.Module):
    def __init__(self, config, use_quantize=False, use_lora=False):
        super().__init__()
        self.config = config

        from src.model.lora import get_layer

        layer = get_layer(quantize=use_quantize, lora=use_lora, **config.lora if use_lora else {})

        # Linear projection from vision hidden size to text hidden size（官方实现）
        self.linear = layer(
            config.vision_config.hidden_size,
            config.vision_config.projection_dim,
            bias=False,  # 官方实现中不使用bias
        )

    def forward(self, image_features):
        # image_features: [Batch_Size, Num_Patches, Vision_Hidden_Size]
        # output: [Batch_Size, Num_Patches, Projection_Dim]
        return self.linear(image_features)