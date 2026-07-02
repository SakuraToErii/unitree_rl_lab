# MHA-history PPO runner cfg for G1-29dof velocity.
# Parallel to BasePPORunnerCfg; does NOT modify the original rsl_rl_ppo_cfg.py.

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticMhaCfg, RslRlPpoAlgorithmCfg


@configclass
class BasePPOMhaRunnerCfg(RslRlOnPolicyRunnerCfg):
    """BasePPORunnerCfg with an MHA history encoder on the actor.

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
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = ""  # same as task name
    empirical_normalization = False
    policy = RslRlPpoActorCriticMhaCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        n_history=5,
        nheads=4,
        encoder_hidden_dim=256,
        is_learnable_pos_embedding=True,
        use_critic_mha=False,
        actor_term_dims=[3, 3, 3, 29, 29, 29],
        critic_term_dims=[3, 3, 3, 3, 29, 29, 29],
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BasePPOMhaPomdp1RunnerCfg(BasePPOMhaRunnerCfg):
    """MHA PPO runner for G1-29dof Velocity-POMDP1.

    POMDP1 removes the actor's IMU terms and keeps:
      policy 90-dim single-step = [velocity_commands, joint_pos_rel, joint_vel_rel, last_action]
        -> [3, 29, 29, 29], x5 = 450
      critic 99-dim single-step = [base_lin_vel, base_ang_vel, projected_gravity,
        velocity_commands, joint_pos_rel, joint_vel_rel, last_action]
        -> [3, 3, 3, 3, 29, 29, 29], x5 = 495
    """

    policy = RslRlPpoActorCriticMhaCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        n_history=5,
        nheads=4,
        encoder_hidden_dim=256,
        is_learnable_pos_embedding=True,
        use_critic_mha=False,
        actor_term_dims=[3, 29, 29, 29],
        critic_term_dims=[3, 3, 3, 3, 29, 29, 29],
    )
