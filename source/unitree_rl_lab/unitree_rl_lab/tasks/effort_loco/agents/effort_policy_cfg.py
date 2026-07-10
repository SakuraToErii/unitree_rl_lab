"""RSL-RL policy configs local to the effort-control tasks."""

from __future__ import annotations

import rsl_rl
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoActorCriticMhaCfg

from .effort_actor_critic import EffortActorCritic, EffortActorCriticMHA


# OnPolicyRunner resolves policy_cfg.class_name with eval() in a module that already
# imports ``rsl_rl``. Registering the effort-only classes on that package keeps the
# shared runner untouched while giving eval() stable qualified names.
rsl_rl.EffortActorCritic = EffortActorCritic
rsl_rl.EffortActorCriticMHA = EffortActorCriticMHA


@configclass
class RslRlEffortActorCriticCfg(RslRlPpoActorCriticCfg):
    """PPO actor-critic config for residual effort control."""

    class_name: str = "rsl_rl.EffortActorCritic"
    actor_output_gain: float = 0.01


@configclass
class RslRlEffortActorCriticMhaCfg(RslRlPpoActorCriticMhaCfg):
    """MHA PPO actor-critic config for residual effort control."""

    class_name: str = "rsl_rl.EffortActorCriticMHA"
    actor_output_gain: float = 0.01
