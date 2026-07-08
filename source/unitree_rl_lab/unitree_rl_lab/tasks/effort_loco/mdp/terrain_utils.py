from __future__ import annotations

import torch
from typing import Literal


def terrain_height_from_ray_hits(
    ray_hits_w: torch.Tensor,
    query_pos_w: torch.Tensor,
    env_origin_z: torch.Tensor,
    rectangle_length: float,
    rectangle_width: float,
    yaw_quat_w: torch.Tensor | None = None,
    height_statistic: Literal["mean", "quantile"] = "quantile",
    height_quantile: float = 0.5,
) -> torch.Tensor:
    """Estimate terrain height in yaw-aligned rectangles around query positions."""
    squeeze_query_dim = query_pos_w.ndim == 2
    if squeeze_query_dim:
        query_pos_w = query_pos_w.unsqueeze(1)

    ray_hit_heights = ray_hits_w[..., 2]
    valid_hits = torch.isfinite(ray_hit_heights)

    delta_xy = ray_hits_w[:, None, :, :2] - query_pos_w[:, :, None, :2]
    if yaw_quat_w is None:
        local_x = delta_xy[..., 0]
        local_y = delta_xy[..., 1]
    else:
        q_w = yaw_quat_w[:, 0].view(-1, 1, 1)
        q_x = yaw_quat_w[:, 1].view(-1, 1, 1)
        q_y = yaw_quat_w[:, 2].view(-1, 1, 1)
        q_z = yaw_quat_w[:, 3].view(-1, 1, 1)
        yaw = torch.atan2(2.0 * (q_w * q_z + q_x * q_y), 1.0 - 2.0 * (q_y * q_y + q_z * q_z))
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        local_x = cos_yaw * delta_xy[..., 0] + sin_yaw * delta_xy[..., 1]
        local_y = -sin_yaw * delta_xy[..., 0] + cos_yaw * delta_xy[..., 1]

    distances = torch.square(local_x) + torch.square(local_y)
    distances = torch.where(valid_hits.unsqueeze(1), distances, torch.inf)

    nearby_hits = (torch.abs(local_x) <= 0.5 * rectangle_length) & (torch.abs(local_y) <= 0.5 * rectangle_width)
    nearby_hits = nearby_hits & valid_hits.unsqueeze(1)
    nearby_heights = torch.where(nearby_hits, ray_hit_heights.unsqueeze(1), torch.nan)
    if height_statistic == "mean":
        terrain_height = torch.nanmean(nearby_heights, dim=-1)
    elif height_statistic == "quantile":
        terrain_height = torch.nanquantile(nearby_heights, height_quantile, dim=-1)
    else:
        raise ValueError(f"Unsupported height_statistic: {height_statistic}")

    nearest_ray_ids = torch.argmin(distances, dim=-1)
    nearest_terrain_height = torch.gather(ray_hit_heights, 1, nearest_ray_ids)
    terrain_height = torch.where(torch.isfinite(terrain_height), terrain_height, nearest_terrain_height)
    terrain_height = torch.where(torch.isfinite(terrain_height), terrain_height, env_origin_z.unsqueeze(-1))

    if squeeze_query_dim:
        terrain_height = terrain_height.squeeze(1)
    return terrain_height
