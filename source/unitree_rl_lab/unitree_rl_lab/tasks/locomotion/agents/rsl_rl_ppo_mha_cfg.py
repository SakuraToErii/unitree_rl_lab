# MHA-history PPO runner cfg for G1-29dof velocity.
# Parallel to BasePPORunnerCfg; does NOT modify the original rsl_rl_ppo_cfg.py.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoActorCriticMhaCfg

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class BasePPOMhaRunnerCfg(BasePPORunnerCfg):
    """BasePPORunnerCfg with an MHA history encoder on the actor (+ critic here).

    term_dims are fixed for G1-29dof velocity, matching the env obs groups
    (velocity_env_cfg.py: ObservationsCfg) and their history_length=5:

      policy 96-dim single-step = [base_ang_vel, projected_gravity,
        velocity_commands, joint_pos_rel, joint_vel_rel, last_action]
        -> [3, 3, 3, 29, 29, 29], x5 = 480
      critic 99-dim single-step = policy + base_lin_vel
        -> [3, 3, 3, 3, 29, 29, 29], x5 = 495

    Coupling: if the env's obs terms (order or count) change, update the two
    term_dims lists here so to_time_major stays a correct block transpose.
    """

    policy = RslRlPpoActorCriticMhaCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        n_history=5,
        nheads=8,
        is_learnable_pos_embedding=True,
        use_critic_mha=True,
        actor_term_dims=[3, 3, 3, 29, 29, 29],
        critic_term_dims=[3, 3, 3, 3, 29, 29, 29],
    )