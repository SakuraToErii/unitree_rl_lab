from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
G1_29DOF = ROOT / "robots" / "g1" / "29dof"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class G1EffortTorqueCfgStaticTest(unittest.TestCase):
    def test_g1_effort_robot_cfg_uses_unitree_explicit_actuators_with_zero_pd(self):
        text = _read(G1_29DOF / "effort_robot_cfg.py")

        self.assertIn("UNITREE_G1_29DOF_EFFORT_CFG", text)
        self.assertIn("UNITREE_G1_29DOF_CFG.replace", text)

        expected_groups = {
            '"N7520-14.3"': "UnitreeActuatorCfg_N7520_14p3",
            '"N7520-22.5"': "UnitreeActuatorCfg_N7520_22p5",
            '"N5020-16"': "UnitreeActuatorCfg_N5020_16",
            '"N5020-16-parallel"': "UnitreeActuatorCfg_N5020_16",
            '"W4010-25"': "UnitreeActuatorCfg_W4010_25",
        }
        for group_name, cfg_class in expected_groups.items():
            self.assertIn(group_name, text)
            self.assertIn(cfg_class, text)

        # Pure torque control: all explicit PD gains must be zero; tau_ff is the only command source.
        self.assertEqual(text.count("stiffness=0.0"), len(expected_groups))
        self.assertEqual(text.count("damping=0.0"), len(expected_groups))

        # Keep ankle and waist roll/pitch in a separate N5020 group as requested.
        self.assertIn('".*_ankle_.*"', text)
        self.assertIn('"waist_roll_joint"', text)
        self.assertIn('"waist_pitch_joint"', text)

    def test_g1_effort_env_uses_joint_effort_action_and_explicit_robot_cfg(self):
        text = _read(G1_29DOF / "effort_env_cfg.py")

        self.assertIn("UNITREE_G1_29DOF_EFFORT_CFG", text)
        self.assertIn("JointEffortActionCfg", text)
        self.assertNotIn("JointPositionActionCfg", text)
        self.assertIn("JointEffortAction", text)

        # Normalize policy outputs into joint torque commands and bound processed efforts.
        self.assertIn("EFFORT_ACTION_SCALE", text)
        self.assertIn("EFFORT_ACTION_CLIP", text)
        self.assertIn("clip=EFFORT_ACTION_CLIP", text)

    def test_g1_effort_task_is_registered(self):
        text = _read(G1_29DOF / "__init__.py")

        self.assertIn('id="Unitree-G1-29dof-Effort"', text)
        self.assertIn("effort_env_cfg:RobotEnvCfg", text)
        self.assertIn("effort_env_cfg:RobotPlayEnvCfg", text)


if __name__ == "__main__":
    unittest.main()
