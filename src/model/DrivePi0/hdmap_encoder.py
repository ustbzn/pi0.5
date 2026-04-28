"""
方案 C: 轻量 BEV CNN 编码器

将下采样后的原始 HDMap BEV (4, H, W) 编码为紧凑特征向量，
保留空间结构和通道间交互，替代方案 B 的手工标量特征。

数据流:
    BEV (4, 48, 90)
      → Conv(4→32, k3, s2) → BN → GELU   → (32, 24, 45)
      → Conv(32→64, k3, s2) → BN → GELU  → (64, 12, 23)
      → Conv(64→128, k3, s2) → BN → GELU → (128, 6, 12)
      → AdaptiveAvgPool2d(1)               → (128, 1, 1)
      → Flatten → Linear(128, output_dim)  → (output_dim,)
"""

import torch
import torch.nn as nn


class HDMapCNNEncoder(nn.Module):

    def __init__(self, in_channels: int = 4, output_dim: int = 32):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 4, H, W) float tensor, 值域 [0, 1]
        Returns:
            (B, output_dim) float tensor
        """
        feat = self.backbone(x)       # (B, 128, 1, 1)
        feat = feat.flatten(1)        # (B, 128)
        return self.head(feat)        # (B, output_dim)
