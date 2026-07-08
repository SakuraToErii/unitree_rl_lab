"""G1 29DOF explicit Unitree actuator configuration for pure torque control.

This module intentionally lives under ``tasks/effort_loco`` so the torque-control
experiment does not mutate the shared robot asset configuration used by the
position-control locomotion tasks.
"""

from unitree_rl_lab.assets.robots import unitree_actuators
from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG


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
