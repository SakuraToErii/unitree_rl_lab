from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
G1_29DOF = ROOT / "robots" / "g1" / "29dof"
EXPORT_DEPLOY_CFG = ROOT.parents[1] / "utils" / "export_deploy_cfg.py"


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
        # JointEffortAction inherits a default base JointAction offset, but pure torque cfg should not set one.
        self.assertNotIn("offset=0.0", text)

    def test_g1_effort_env_defines_pomdp_observations_locally_for_torque_tasks(self):
        text = _read(G1_29DOF / "effort_env_cfg.py")

        self.assertNotIn("pomdp_velocity_env_cfg", text)
        self.assertNotIn("pomdp2_velocity_env_cfg", text)
        self.assertIn("class Pomdp1ObservationsCfg", text)
        self.assertIn("class Pomdp2ObservationsCfg", text)
        self.assertEqual(text.count("class PolicyCfg(ObsGroup)"), 2)
        self.assertEqual(text.count("class CriticCfg(ObsGroup)"), 2)
        self.assertEqual(text.count("policy: PolicyCfg = PolicyCfg()"), 2)
        self.assertEqual(text.count("critic: CriticCfg = CriticCfg()"), 2)
        self.assertNotIn("class Pomdp1PolicyObservationsCfg", text)
        self.assertNotIn("class Pomdp2PolicyObservationsCfg", text)
        self.assertNotIn("class PrivilegedCriticObservationsCfg", text)
        self.assertIn("class Pomdp1RobotEnvCfg(RobotEnvCfg)", text)
        self.assertIn("class Pomdp1RobotPlayEnvCfg(RobotPlayEnvCfg)", text)
        self.assertIn("class Pomdp2RobotEnvCfg(RobotEnvCfg)", text)
        self.assertIn("class Pomdp2RobotPlayEnvCfg(RobotPlayEnvCfg)", text)
        self.assertIn("observations: Pomdp1ObservationsCfg = Pomdp1ObservationsCfg()", text)
        self.assertIn("observations: Pomdp2ObservationsCfg = Pomdp2ObservationsCfg()", text)

    def test_export_deploy_cfg_handles_effort_action_without_default_offset_flag(self):
        text = _read(EXPORT_DEPLOY_CFG)

        self.assertNotIn(
            'for _ in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:\n'
            "            del term_cfg[_]",
            text,
        )
        self.assertIn("term_cfg.pop(_, None)", text)
        self.assertIn('term_cfg.pop("offset", None)', text)
        self.assertIn("isinstance(term_cfg.offset, (float, int))", text)

    def test_g1_effort_task_is_registered(self):
        text = _read(G1_29DOF / "__init__.py")

        expected_entries = {
            'id="Unitree-G1-29dof-Effort"': [
                "effort_env_cfg:RobotEnvCfg",
                "effort_env_cfg:RobotPlayEnvCfg",
            ],
            'id="Unitree-G1-29dof-Effort-POMDP1"': [
                "effort_env_cfg:Pomdp1RobotEnvCfg",
                "effort_env_cfg:Pomdp1RobotPlayEnvCfg",
            ],
            'id="Unitree-G1-29dof-Effort-POMDP2"': [
                "effort_env_cfg:Pomdp2RobotEnvCfg",
                "effort_env_cfg:Pomdp2RobotPlayEnvCfg",
            ],
            'id="Unitree-G1-29dof-Effort-POMDP1-MHA"': [
                "effort_env_cfg:Pomdp1RobotEnvCfg",
                "effort_env_cfg:Pomdp1RobotPlayEnvCfg",
                "effort_loco.agents.rsl_rl_ppo_mha_cfg:BasePPOMhaPomdp1RunnerCfg",
            ],
            'id="Unitree-G1-29dof-Effort-POMDP2-MHA"': [
                "effort_env_cfg:Pomdp2RobotEnvCfg",
                "effort_env_cfg:Pomdp2RobotPlayEnvCfg",
                "effort_loco.agents.rsl_rl_ppo_mha_cfg:BasePPOMhaPomdp2RunnerCfg",
            ],
        }
        for task_id, entry_points in expected_entries.items():
            self.assertIn(task_id, text)
            for entry_point in entry_points:
                self.assertIn(entry_point, text)
        self.assertNotIn('id="Unitree-G1-29dof-Velocity-POMDP1-MHA"', text)
        self.assertNotIn('id="Unitree-G1-29dof-Velocity-POMDP2-MHA"', text)


if __name__ == "__main__":
    unittest.main()
