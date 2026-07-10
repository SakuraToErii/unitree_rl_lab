"""Effort-control actor-critic variants with residual-centered initialization."""

from __future__ import annotations

import torch.nn as nn

from rsl_rl.modules import ActorCritic, ActorCriticMHA


def _initialize_actor_output(layer: nn.Module, gain: float) -> None:
    """Initialize an actor output layer close to zero while preserving gradient flow."""
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected the actor output layer to be nn.Linear, received {type(layer).__name__}.")
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


class EffortActorCritic(ActorCritic):
    """ActorCritic whose initial mean action stays close to the residual-torque origin."""

    def __init__(self, *args, actor_output_gain: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        _initialize_actor_output(self.actor[-1], actor_output_gain)


class EffortActorCriticMHA(ActorCriticMHA):
    """MHA actor-critic with the same residual-centered output initialization."""

    def __init__(self, *args, actor_output_gain: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        _initialize_actor_output(self.actor.trunk[-1], actor_output_gain)
