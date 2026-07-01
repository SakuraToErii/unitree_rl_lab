from __future__ import annotations

import torch


def terrain_height_from_ray_hits(
    ray_hits_w: torch.Tensor,
    query_pos_w: torch.Tensor,
    env_origin_z: torch.Tensor,
    radius: float,
) -> torch.Tensor:
    """Estimate terrain height around query positions from nearby ray hits."""
    squeeze_query_dim = query_pos_w.ndim == 2
    if squeeze_query_dim:
        query_pos_w = query_pos_w.unsqueeze(1)

    ray_hit_heights = ray_hits_w[..., 2]
    valid_hits = torch.isfinite(ray_hit_heights)
    distances = torch.sum(torch.square(query_pos_w[..., :2].unsqueeze(2) - ray_hits_w[..., :2].unsqueeze(1)), dim=-1)
    distances = torch.where(valid_hits.unsqueeze(1), distances, torch.inf)

    nearby_hits = distances <= radius**2
    nearby_heights = torch.where(nearby_hits, ray_hit_heights.unsqueeze(1), torch.nan)
    terrain_height = torch.nanmedian(nearby_heights, dim=-1).values

    nearest_ray_ids = torch.argmin(distances, dim=-1)
    nearest_terrain_height = torch.gather(ray_hit_heights, 1, nearest_ray_ids)
    terrain_height = torch.where(torch.isfinite(terrain_height), terrain_height, nearest_terrain_height)
    terrain_height = torch.where(torch.isfinite(terrain_height), terrain_height, env_origin_z.unsqueeze(-1))

    if squeeze_query_dim:
        terrain_height = terrain_height.squeeze(1)
    return terrain_height
