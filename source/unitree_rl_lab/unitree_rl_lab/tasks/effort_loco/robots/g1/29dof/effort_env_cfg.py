"""G1 29DOF velocity task variant with pure torque actions.

Compared with ``velocity_env_cfg.py`` this configuration keeps the same scene,
observations, rewards, commands and timing, but swaps in:

* ``UNITREE_G1_29DOF_EFFORT_CFG``: explicit Unitree actuator models.
* ``JointEffortActionCfg``: policy actions are feed-forward joint torques.

The explicit actuators have zero PD gains, so actions are not converted through a
position controller.  The Unitree actuator model still applies its motor
torque-speed and friction limits before writing forces to PhysX.
"""

from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.effort_loco import mdp

from .effort_robot_cfg import UNITREE_G1_29DOF_EFFORT_CFG
from .velocity_env_cfg import RobotEnvCfg as BaseRobotEnvCfg
from .velocity_env_cfg import RobotPlayEnvCfg as BaseRobotPlayEnvCfg
from .velocity_env_cfg import RobotSceneCfg as BaseRobotSceneCfg


# Action scale maps normalized policy output to nominal joint torque commands in N·m.
# Values use the explicit Unitree actuator same-direction peak torque (Y1) so most
# policy outputs stay within the model's low-speed torque envelope; the actuator
# still performs velocity-dependent clipping internally.
EFFORT_ACTION_SCALE = {
    ".*_hip_pitch_joint": 71.0,
    ".*_hip_yaw_joint": 71.0,
    "waist_yaw_joint": 71.0,
    ".*_hip_roll_joint": 111.0,
    ".*_knee_joint": 111.0,
    ".*_shoulder_.*": 24.8,
    ".*_elbow_joint": 24.8,
    ".*_wrist_roll.*": 24.8,
    ".*_ankle_.*": 24.8,
    "waist_roll_joint": 24.8,
    "waist_pitch_joint": 24.8,
    ".*_wrist_pitch.*": 4.8,
    ".*_wrist_yaw.*": 4.8,
}
EFFORT_ACTION_CLIP = {joint_expr: (-limit, limit) for joint_expr, limit in EFFORT_ACTION_SCALE.items()}


@configclass
class RobotSceneCfg(BaseRobotSceneCfg):
    """Scene with the G1 29DOF explicit Unitree torque-control robot."""

    robot: ArticulationCfg = UNITREE_G1_29DOF_EFFORT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    """Pure torque action specification for the MDP."""

    JointEffortAction = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=EFFORT_ACTION_SCALE,
        offset=0.0,
        clip=EFFORT_ACTION_CLIP,
    )


@configclass
class Pomdp1ObservationsCfg:
    """Observation specifications for the torque-control POMDP1 task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class Pomdp2ObservationsCfg:
    """Observation specifications for the torque-control POMDP2 task."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 5

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RobotEnvCfg(BaseRobotEnvCfg):
    """G1 29DOF locomotion environment using torque commands."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    """Play/evaluation variant of the torque-control environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=32, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()


@configclass
class Pomdp1RobotEnvCfg(RobotEnvCfg):
    """Torque-control POMDP1 variant: hide actor IMU terms, keep joint velocity."""

    observations: Pomdp1ObservationsCfg = Pomdp1ObservationsCfg()


@configclass
class Pomdp1RobotPlayEnvCfg(RobotPlayEnvCfg):
    """Play/evaluation variant of the torque-control POMDP1 environment."""

    observations: Pomdp1ObservationsCfg = Pomdp1ObservationsCfg()


@configclass
class Pomdp2RobotEnvCfg(RobotEnvCfg):
    """Torque-control POMDP2 variant: hide actor IMU terms and joint velocity."""

    observations: Pomdp2ObservationsCfg = Pomdp2ObservationsCfg()


@configclass
class Pomdp2RobotPlayEnvCfg(RobotPlayEnvCfg):
    """Play/evaluation variant of the torque-control POMDP2 environment."""

    observations: Pomdp2ObservationsCfg = Pomdp2ObservationsCfg()
