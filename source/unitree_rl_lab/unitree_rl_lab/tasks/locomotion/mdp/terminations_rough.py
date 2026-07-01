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
    terrain_height_radius: float = 0.5,
) -> torch.Tensor:
    """Terminate when the root height is below a terrain-relative threshold."""
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]

    terrain_height = terrain_height_from_ray_hits(
        sensor.data.ray_hits_w,
        asset.data.root_pos_w,
        env.scene.env_origins[:, 2],
        terrain_height_radius,
    )
    root_height = asset.data.root_pos_w[:, 2] - terrain_height
    return root_height < minimum_height
