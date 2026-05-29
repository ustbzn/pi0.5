"""PlanVOCA-style MapEncoder for DrivePi0.

复刻 ai-pnc-master/nuplan/.../planTF/modules/map_encoder.py 的实际行为：
  - 只用 centerline (point_position[:, 0], point_vector[:, 0], point_orientation[:, 0])
  - 6 维点级特征：相对 polygon_center 的 (dx, dy) + (vec_x, vec_y) + (cos, sin)
  - 通过 PointsEncoder 池化 -> [B, M, dim]
  - 加上 type / on_route / traffic_light / speed_limit embedding

输出维度 dim 在 DrivePi0 中通常设为 cfg.mixture.proprio.hidden_size，
直接拼到 proprio embeds 后面，与现有 proprio mixture 共用 attention。
"""

from typing import Optional

import torch
import torch.nn as nn


class PointsEncoder(nn.Module):
    """复刻 PlanVOCA PointsEncoder：first MLP -> max pool -> concat -> second MLP -> max pool。"""

    def __init__(self, feat_channel: int, encoder_channel: int):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_mlp = nn.Sequential(
            nn.Linear(feat_channel, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
        )
        self.second_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, encoder_channel),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [B, P, C]; mask: [B, P]; returns [B, encoder_channel]."""
        x = self.first_mlp(x)
        x = x * mask.unsqueeze(-1)
        pooled = x.max(dim=1)[0]
        x = torch.cat([x, pooled.unsqueeze(1).expand(-1, x.size(1), -1)], dim=-1)
        x = self.second_mlp(x)
        x = x * mask.unsqueeze(-1)
        return x.max(dim=1)[0]


class MapEncoder(nn.Module):
    """PlanVOCA-style 结构化 lane token encoder。

    Args:
        hidden_size:        输出维度，建议设为 proprio mixture hidden_size（DrivePi0 中为 1024）。
        polygon_channel:    点级特征维度，复刻 PlanVOCA 默认 6（centerline only）。
        num_polygon_types:  polygon_type embedding 类别数，0=lane,1=lane_connector,(2 预留 crosswalk)。
        num_tl_states:      traffic light 状态数，PlanVOCA = 4 (GREEN/YELLOW/RED/UNKNOWN)。
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        polygon_channel: int = 6,
        num_polygon_types: int = 3,
        num_tl_states: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.polygon_encoder = PointsEncoder(polygon_channel, hidden_size)
        self.speed_limit_emb = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
        )
        self.type_emb = nn.Embedding(num_polygon_types, hidden_size)
        self.on_route_emb = nn.Embedding(2, hidden_size)
        self.traffic_light_emb = nn.Embedding(num_tl_states, hidden_size)
        self.unknown_speed_emb = nn.Embedding(1, hidden_size)
        nn.init.normal_(self.type_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.on_route_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.traffic_light_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.unknown_speed_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        polygon_center: torch.Tensor,         # [B, M, 3]
        polygon_type: torch.Tensor,           # [B, M] long
        polygon_on_route: torch.Tensor,       # [B, M] long
        polygon_tl_status: torch.Tensor,      # [B, M] long
        polygon_has_speed_limit: torch.Tensor,  # [B, M] bool
        polygon_speed_limit: torch.Tensor,    # [B, M] float
        point_position: torch.Tensor,         # [B, M, 3, P, 2]
        point_vector: torch.Tensor,           # [B, M, 3, P, 2]
        point_orientation: torch.Tensor,      # [B, M, 3, P]
        point_valid_mask: torch.Tensor,       # [B, M, P] float
        valid_mask: torch.Tensor,             # [B, M] float
    ) -> torch.Tensor:
        """returns [B, M, hidden_size]，padding token 也会有非零值，调用方需自行用 valid_mask 屏蔽 attention。"""
        B, M = polygon_type.shape

        # centerline-only 的 6 维点级特征：
        # (dx, dy) = point_position[:,:,0] - polygon_center.xy
        # (vx, vy) = point_vector[:,:,0]
        # (cos, sin) = orientation
        center_xy = polygon_center[:, :, :2].unsqueeze(2)  # [B, M, 1, 2]
        rel_pos = point_position[:, :, 0, :, :] - center_xy  # [B, M, P, 2]
        vec = point_vector[:, :, 0, :, :]                    # [B, M, P, 2]
        ori = point_orientation[:, :, 0, :]                  # [B, M, P]
        cos_ori = ori.cos().unsqueeze(-1)
        sin_ori = ori.sin().unsqueeze(-1)
        polygon_feature = torch.cat([rel_pos, vec, cos_ori, sin_ori], dim=-1)  # [B, M, P, 6]

        P = polygon_feature.shape[2]
        feat = polygon_feature.reshape(B * M, P, -1)
        pt_mask = point_valid_mask.reshape(B * M, P)
        # 空 mask 的 token：补一个全 1 mask 防止 PointsEncoder 内 max 退化为 -inf
        empty_token = (pt_mask.sum(-1) <= 0)
        if empty_token.any():
            pt_mask = pt_mask.clone()
            pt_mask[empty_token] = 1.0
        x_polygon = self.polygon_encoder(feat, pt_mask).view(B, M, self.hidden_size)

        x_type = self.type_emb(polygon_type.clamp(min=0, max=self.type_emb.num_embeddings - 1))
        x_on_route = self.on_route_emb(polygon_on_route.clamp(min=0, max=1))
        x_tl = self.traffic_light_emb(polygon_tl_status.clamp(min=0, max=self.traffic_light_emb.num_embeddings - 1))

        has_speed = polygon_has_speed_limit.bool()
        x_speed = self.unknown_speed_emb.weight.expand(B, M, -1).clone()
        if has_speed.any():
            speed_vals = polygon_speed_limit[has_speed].unsqueeze(-1)
            x_speed[has_speed] = self.speed_limit_emb(speed_vals)

        x = x_polygon + x_type + x_on_route + x_tl + x_speed
        # 用 valid_mask 把 padding token 的 embedding 置 0：attention 时既靠
        # mask 屏蔽，也靠这里降低数值幅度，防 norm 层产生异常梯度。
        x = x * valid_mask.unsqueeze(-1)
        return x
