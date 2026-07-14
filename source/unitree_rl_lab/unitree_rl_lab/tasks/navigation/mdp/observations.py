from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from isaaclab.envs.mdp.observations import height_scan as _height_scan
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def last_high_level_command(env: ManagerBasedRLEnv, action_name: str = "pre_trained_policy_action") -> torch.Tensor:
    """Previous high-level velocity command after command clipping."""
    # 提示：高层策略需要观察真正传给低层策略的速度指令，而不是裁剪前的原始动作。
    # 可通过 action_manager 按名称取得 action term，并读取其 processed_actions。
    # >>> HOMEWORK_TODO_1_START
    return env.action_manager.get_term(action_name).processed_actions
    # <<< HOMEWORK_TODO_1_END


def low_level_last_action(env: ManagerBasedRLEnv, action_name: str = "pre_trained_policy_action") -> torch.Tensor:
    """Last 29-DoF joint action emitted by the frozen low-level policy."""
    return env.action_manager.get_term(action_name).low_level_actions


def command_distance(env: ManagerBasedRLEnv, command_name: str = "pose_command") -> torch.Tensor:
    """2D distance to the active pose command."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1, keepdim=True)


def _height_scan_grid_shape(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> tuple[int, int]:
    """Return (ny, nx) ray-grid shape matching Isaac Lab ``grid_pattern`` with ordering ``xy``."""
    sensor = env.scene.sensors[sensor_cfg.name]
    pattern = sensor.cfg.pattern_cfg
    device = env.device
    x = torch.arange(start=-pattern.size[0] / 2, end=pattern.size[0] / 2 + 1.0e-9, step=pattern.resolution, device=device)
    y = torch.arange(start=-pattern.size[1] / 2, end=pattern.size[1] / 2 + 1.0e-9, step=pattern.resolution, device=device)
    return y.numel(), x.numel()


def height_scan_pooled(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    pool_size: int = 2,
) -> torch.Tensor:
    """Max-pooled height scan: 2D grid from the scanner, then ``pool_size`` max-pool, flattened."""
    # 提示：先调用 _height_scan 得到扁平射线高度，再依据 (ny, nx) 恢复二维网格。
    # 为适配 F.max_pool2d，需要添加通道维；池化后再展平为 (num_envs, -1)。
    # >>> HOMEWORK_TODO_2_START
    # 1. 获取高度扫描扁平射线，形状为（N, B），N 为传感器总数，B为每个传感器的射线数量
    height_scan_flat = _height_scan(env, sensor_cfg, offset=offset)
    # 2. 获取网格形状 (ny, nx)，恢复为 (num_envs, ny, nx)
    ny, nx = _height_scan_grid_shape(env, sensor_cfg)
    height_scan_grid = height_scan_flat.view(-1, ny, nx)
    # 3. 添加通道维度 (num_envs, 1, ny, nx)
    height_scan_grid = height_scan_grid.unsqueeze(1)
    # 4. 最大池化 (num_envs, 1, ny//pool_size, nx//pool_size)
    pooled_height_scan = F.max_pool2d(height_scan_grid, kernel_size=pool_size)
    # 5. 展平为 (num_envs, -1)，便于 MLP 输入
    pooled_height_scan_flat = pooled_height_scan.view(pooled_height_scan.size(0), -1)
    return pooled_height_scan_flat
    # <<< HOMEWORK_TODO_2_END
