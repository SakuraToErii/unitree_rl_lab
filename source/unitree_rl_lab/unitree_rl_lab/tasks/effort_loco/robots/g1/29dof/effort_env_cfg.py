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


# Per-joint same-direction peak torque (Y1) of the explicit Unitree actuator.
# Used as the hard action clip (motor limit); the actuator's torque-speed curve
# still clips the applied torque on top of this.
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

# Residual action band: policy outputs are scaled into this fraction of each
# joint's Y1 peak and added to the standing-torque baseline (NOMINAL_TORQUE).
# 0.4 is sized for standing/balance; raise toward 0.5-0.7 for dynamic gaits where
# the residual must also cover the gravity-compensation error vs the default pose.
RESIDUAL_ACTION_SCALE = {joint_expr: 0.4 * limit for joint_expr, limit in EFFORT_ACTION_SCALE.items()}

# Approximate implicit-PD torque at the paired standing pose in
# EFFORT_STANDING_JOINT_POSITION. Collected from model_9400.pt over a continuous
# 1 s steady window at zero velocity command. This is a residual-action bias;
# the actor supplies the state-dependent feedback required for balance.
NOMINAL_TORQUE: dict[str, float] = {
    "left_hip_pitch_joint": -1.171482,
    "right_hip_pitch_joint": -0.392699,
    "waist_yaw_joint": -0.049220,
    "left_hip_roll_joint": 11.628835,
    "right_hip_roll_joint": -9.975743,
    "waist_roll_joint": 0.391920,
    "left_hip_yaw_joint": -3.657360,
    "right_hip_yaw_joint": 3.118077,
    "waist_pitch_joint": 1.178703,
    "left_knee_joint": -1.801823,
    "right_knee_joint": -0.455979,
    "left_shoulder_pitch_joint": 0.630705,
    "right_shoulder_pitch_joint": 0.784696,
    "left_ankle_pitch_joint": 4.369744,
    "right_ankle_pitch_joint": 3.314344,
    "left_shoulder_roll_joint": 1.602750,
    "right_shoulder_roll_joint": -1.674056,
    "left_ankle_roll_joint": 0.662338,
    "right_ankle_roll_joint": 0.517337,
    "left_shoulder_yaw_joint": 0.344647,
    "right_shoulder_yaw_joint": -0.339807,
    "left_elbow_joint": -0.398630,
    "right_elbow_joint": -0.325242,
    "left_wrist_roll_joint": -0.002183,
    "right_wrist_roll_joint": 0.002247,
    "left_wrist_pitch_joint": -0.106291,
    "right_wrist_pitch_joint": -0.086652,
    "left_wrist_yaw_joint": 0.057935,
    "right_wrist_yaw_joint": -0.055295,
}


@configclass
class RobotSceneCfg(BaseRobotSceneCfg):
    """Scene with the G1 29DOF explicit Unitree torque-control robot."""

    robot: ArticulationCfg = UNITREE_G1_29DOF_EFFORT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    """Constant-baseline residual torque action: tau = tau0 + residual * scale.

    The Unitree actuator keeps zero PD gains, so the commanded torque is pure
    feed-forward (standing-torque baseline + policy residual), motor-limited.
    With NOMINAL_TORQUE empty this reduces to pure residual (tau = action * scale).
    """

    JointEffortAction = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=RESIDUAL_ACTION_SCALE,
        clip=EFFORT_ACTION_CLIP,
        offset=NOMINAL_TORQUE,
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
    """POMDP2 observations: IMU attitude cues with hidden joint velocity."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
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
    """Torque-control POMDP2 variant: hide actor joint velocity."""

    observations: Pomdp2ObservationsCfg = Pomdp2ObservationsCfg()


@configclass
class Pomdp2RobotPlayEnvCfg(RobotPlayEnvCfg):
    """Play/evaluation variant of the torque-control POMDP2 environment."""

    observations: Pomdp2ObservationsCfg = Pomdp2ObservationsCfg()
