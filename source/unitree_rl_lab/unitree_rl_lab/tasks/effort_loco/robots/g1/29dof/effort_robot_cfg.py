"""G1 29DOF explicit Unitree actuator configuration for pure torque control.

This module intentionally lives under ``tasks/effort_loco`` so the torque-control
experiment does not mutate the shared robot asset configuration used by the
position-control locomotion tasks.
"""

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG


# Actual standing pose paired with NOMINAL_TORQUE in effort_env_cfg.py.
# Collected over a continuous 1 s steady window from model_9400.pt at zero
# velocity command.  The effort task uses this as its default/reset joint pose.
EFFORT_STANDING_JOINT_POSITION: dict[str, float] = {
    "left_hip_pitch_joint": -0.059750,
    "right_hip_pitch_joint": -0.028836,
    "waist_yaw_joint": -0.002169,
    "left_hip_roll_joint": 0.001778,
    "right_hip_roll_joint": -0.003393,
    "waist_roll_joint": -0.001681,
    "left_hip_yaw_joint": 0.011778,
    "right_hip_yaw_joint": -0.003191,
    "waist_pitch_joint": -0.000188,
    "left_knee_joint": 0.175060,
    "right_knee_joint": 0.122524,
    "left_shoulder_pitch_joint": 0.312136,
    "right_shoulder_pitch_joint": 0.324009,
    "left_ankle_pitch_joint": -0.120162,
    "right_ankle_pitch_joint": -0.098493,
    "left_shoulder_roll_joint": 0.257946,
    "right_shoulder_roll_joint": -0.246549,
    "left_ankle_roll_joint": 0.005809,
    "right_ankle_roll_joint": 0.007654,
    "left_shoulder_yaw_joint": 0.003801,
    "right_shoulder_yaw_joint": -0.019234,
    "left_elbow_joint": 0.929968,
    "right_elbow_joint": 0.969214,
    "left_wrist_roll_joint": 0.152960,
    "right_wrist_roll_joint": -0.152853,
    "left_wrist_pitch_joint": -0.025037,
    "right_wrist_pitch_joint": -0.013271,
    "left_wrist_yaw_joint": 0.013372,
    "right_wrist_yaw_joint": 0.007627,
}
EFFORT_STANDING_ROOT_HEIGHT = 0.789733


# Explicit actuator model + zero PD gains:
#   JointEffortAction -> tau_ff
#   UnitreeActuator   -> torque-speed clipping / friction / delay model
#   PhysX             -> set_dof_actuation_forces(applied_tau)
#
# With stiffness=damping=0, q_des and qd_des do not contribute to torque.
UNITREE_G1_29DOF_EFFORT_CFG = UNITREE_G1_29DOF_CFG.replace(
    actuators={
        "N7520-14.3": unitree_actuators.UnitreeActuatorCfg_N7520_14p3(
            joint_names_expr=[".*_hip_pitch_.*", ".*_hip_yaw_.*", "waist_yaw_joint"],
            stiffness=0.0,
            damping=0.0,
        ),
        "N7520-22.5": unitree_actuators.UnitreeActuatorCfg_N7520_22p5(
            joint_names_expr=[".*_hip_roll_.*", ".*_knee_.*"],
            stiffness=0.0,
            damping=0.0,
        ),
        "N5020-16": unitree_actuators.UnitreeActuatorCfg_N5020_16(
            joint_names_expr=[
                ".*_shoulder_.*",
                ".*_elbow_.*",
                ".*_wrist_roll.*",
            ],
            stiffness=0.0,
            damping=0.0,
        ),
        "N5020-16-parallel": unitree_actuators.UnitreeActuatorCfg_N5020_16(
            joint_names_expr=[
                ".*_ankle_.*",
                "waist_roll_joint",
                "waist_pitch_joint",
            ],
            stiffness=0.0,
            damping=0.0,
        ),
        "W4010-25": unitree_actuators.UnitreeActuatorCfg_W4010_25(
            joint_names_expr=[".*_wrist_pitch.*", ".*_wrist_yaw.*"],
            stiffness=0.0,
            damping=0.0,
        ),
    }
)
UNITREE_G1_29DOF_EFFORT_CFG.init_state.pos = (0.0, 0.0, EFFORT_STANDING_ROOT_HEIGHT)
UNITREE_G1_29DOF_EFFORT_CFG.init_state.joint_pos = EFFORT_STANDING_JOINT_POSITION.copy()
UNITREE_G1_29DOF_EFFORT_CFG.init_state.joint_vel = {".*": 0.0}
