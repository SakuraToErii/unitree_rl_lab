import gymnasium as gym

gym.register(
    id="Unitree-G1-29dof-Effort",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.effort_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.effort_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.effort_loco.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Effort-POMDP1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.effort_env_cfg:Pomdp1RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.effort_env_cfg:Pomdp1RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.effort_loco.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Effort-POMDP2",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.effort_env_cfg:Pomdp2RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.effort_env_cfg:Pomdp2RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.effort_loco.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Effort-POMDP1-MHA",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.effort_env_cfg:Pomdp1RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.effort_env_cfg:Pomdp1RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.effort_loco.agents.rsl_rl_ppo_mha_cfg:BasePPOMhaPomdp1RunnerCfg",
    },
)

gym.register(
    id="Unitree-G1-29dof-Effort-POMDP2-MHA",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.effort_env_cfg:Pomdp2RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.effort_env_cfg:Pomdp2RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.effort_loco.agents.rsl_rl_ppo_mha_cfg:BasePPOMhaPomdp2RunnerCfg",
    },
)
