from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

from .terrain_utils import terrain_height_from_ray_hits

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum_relative(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    terrain_height_length: float = 0.6,
    terrain_height_width: float = 0.5,
    terrain_height_statistic: str = "mean",
    terrain_height_quantile: float = 0.5,
) -> torch.Tensor:
    """Terminate when the root height is below a terrain-relative threshold."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]

    terrain_height = terrain_height_from_ray_hits(
        ray_hits_w=sensor.data.ray_hits_w,
        query_pos_w=asset.data.root_pos_w,
        env_origin_z=env.scene.env_origins[:, 2],
        rectangle_length=terrain_height_length,
        rectangle_width=terrain_height_width,
        yaw_quat_w=asset.data.root_quat_w,
        height_statistic=terrain_height_statistic,
        height_quantile=terrain_height_quantile,
    )
    root_height = asset.data.root_pos_w[:, 2] - terrain_height
    return root_height < minimum_height
