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
from isaaclab.utils import configclass

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
class RobotEnvCfg(BaseRobotEnvCfg):
    """G1 29DOF locomotion environment using torque commands."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()


@configclass
class RobotPlayEnvCfg(BaseRobotPlayEnvCfg):
    """Play/evaluation variant of the torque-control environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=32, env_spacing=2.5)
    actions: ActionsCfg = ActionsCfg()
