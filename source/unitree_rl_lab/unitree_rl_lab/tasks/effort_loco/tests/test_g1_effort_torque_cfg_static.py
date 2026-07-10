from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
G1_29DOF = ROOT / "robots" / "g1" / "29dof"
AGENTS = ROOT / "agents"
MDP = ROOT / "mdp"
EXPORT_DEPLOY_CFG = ROOT.parents[1] / "utils" / "export_deploy_cfg.py"
ZERO_ACTION_PLAY = ROOT.parents[4] / "scripts" / "rsl_rl" / "play_zero_action.py"
COLLECT_NOMINAL_TORQUE = ROOT.parents[4] / "scripts" / "rsl_rl" / "collect_nominal_torque.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


class G1EffortTorqueCfgStaticTest(unittest.TestCase):
    def test_g1_effort_robot_cfg_uses_unitree_explicit_actuators_with_zero_pd(self):
        text = _read(G1_29DOF / "effort_robot_cfg.py")

        self.assertIn("UNITREE_G1_29DOF_EFFORT_CFG", text)
        self.assertIn("UNITREE_G1_29DOF_CFG.replace", text)
        self.assertIn("EFFORT_STANDING_JOINT_POSITION", text)
        self.assertIn("EFFORT_STANDING_ROOT_HEIGHT", text)
        self.assertIn("init_state.joint_pos = EFFORT_STANDING_JOINT_POSITION.copy()", text)

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

        pomdp2_policy = text.split("class Pomdp2ObservationsCfg", 1)[1].split("class CriticCfg", 1)[0]
        self.assertIn("base_ang_vel = ObsTerm", pomdp2_policy)
        self.assertIn("projected_gravity = ObsTerm", pomdp2_policy)
        self.assertNotIn("joint_vel_rel = ObsTerm", pomdp2_policy)

    def test_effort_task_uses_local_commands_reset_and_torque_rate_reward(self):
        env_text = _read(G1_29DOF / "velocity_env_cfg.py")
        rewards_text = _read(MDP / "rewards.py")

        self.assertIn("from unitree_rl_lab.tasks.effort_loco import mdp", env_text)
        self.assertIn('"velocity_range": (-0.2, 0.2)', env_text)
        self.assertIn("lin_vel_x=(-0.3, 0.5)", env_text)
        self.assertIn("lin_vel_y=(-0.2, 0.2)", env_text)
        self.assertIn("func=mdp.effort_action_rate_l2", env_text)
        self.assertIn('params={"action_term_name": "JointEffortAction"}', env_text)
        self.assertIn("def effort_action_rate_l2", rewards_text)
        self.assertIn("current_effort - previous_effort", rewards_text)
        self.assertIn("effort_normalizer.clamp_min", rewards_text)

    def test_effort_policy_configs_use_centered_output_and_lower_entropy_pressure(self):
        ppo_text = _read(AGENTS / "rsl_rl_ppo_cfg.py")
        mha_text = _read(AGENTS / "rsl_rl_ppo_mha_cfg.py")
        policy_text = _read(AGENTS / "effort_actor_critic.py")

        for text in (ppo_text, mha_text):
            self.assertIn("actor_output_gain=0.01", text)
            self.assertIn('noise_std_type="log"', text)
            self.assertIn("entropy_coef=0.001", text)
        self.assertIn("nn.init.orthogonal_(layer.weight, gain=gain)", policy_text)
        self.assertIn("nn.init.zeros_(layer.bias)", policy_text)
        self.assertIn("_initialize_actor_output(self.actor[-1]", policy_text)
        self.assertIn("_initialize_actor_output(self.actor.trunk[-1]", policy_text)
        self.assertIn("actor_term_dims=[3, 3, 3, 29, 29]", mha_text)

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

    def test_zero_action_play_script_uses_tau_ref_without_a_policy(self):
        text = _read(ZERO_ACTION_PLAY)

        self.assertIn('default="Unitree-G1-29dof-Effort-POMDP2"', text)
        self.assertIn("zero_action = torch.zeros", text)
        self.assertIn("action_term.processed_actions[0], tau_ref", text)
        self.assertIn("actuator.stiffness", text)
        self.assertIn("actuator.damping", text)
        self.assertIn('command_cfg.ranges.lin_vel_x = (0.0, 0.0)', text)
        self.assertIn('env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)', text)
        self.assertNotIn("OnPolicyRunner", text)
        self.assertNotIn("get_checkpoint_path", text)
        self.assertNotIn("torch.load", text)
        self.assertNotIn("runner.load", text)

    def test_nominal_torque_collection_uses_a_continuous_paired_steady_window(self):
        text = _read(COLLECT_NOMINAL_TORQUE)

        self.assertIn('parser.add_argument("--collect_s", type=float, default=1.0', text)
        self.assertIn('choices=("zero_action", "checkpoint")', text)
        self.assertIn('default="checkpoint"', text)
        self.assertIn("env_cfg.observations.policy.enable_corruption = False", text)
        self.assertIn('env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)', text)
        self.assertIn('contact_sensor.find_bodies(".*ankle_roll.*")', text)
        self.assertIn("streak[restart] = 0", text)
        self.assertIn("newly_collected = accepted & (streak >= window_steps)", text)
        self.assertIn('_format_dict("JOINT_POSITION_REFERENCE"', text)
        self.assertIn('_format_dict("NOMINAL_TORQUE"', text)

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
