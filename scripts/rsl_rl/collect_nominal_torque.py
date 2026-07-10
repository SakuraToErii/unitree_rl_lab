# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect a paired standing pose and torque reference from a position policy.

A converged residual-position policy runs deterministically on a nominal plane
with a zero velocity command.  A zero-action diagnostic mode is also available
to test whether the position controller can hold ``q_target = q_default`` by
itself.  Each environment must satisfy the standing criteria continuously for
the complete collection window (one second by default).  The script then selects
one representative environment and saves its time-averaged pair::

    JOINT_POSITION_REFERENCE = q_ref
    NOMINAL_TORQUE = tau_ref

Keeping the pair together matters because the implicit position actuator reports
the approximate PD command at the policy's actual pose and contact state::

    tau_ref ~= Kp * (q_target - q_ref) - Kd * qd

The saved file can be loaded directly by ``play_zero_action.py`` for a matched
zero-action check.

Example::

    source ~/projects/IsaacLab/.venv/bin/activate
    python scripts/rsl_rl/collect_nominal_torque.py \
        --task Unitree-G1-29dof-Velocity \
        --checkpoint /home/ordis/projects/unitree_rl_lab/models/flat/model_9400.pt \
        --num_envs 32 --headless

Zero-action diagnostic::

    python scripts/rsl_rl/collect_nominal_torque.py \
        --task Unitree-G1-29dof-Velocity \
        --action_source zero_action --num_envs 1 --headless
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Collect a paired standing pose and torque reference.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of deterministic collection environments.")
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-G1-29dof-Velocity",
    help="Standing-capable position-control locomotion task.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--action_source",
    choices=("zero_action", "checkpoint"),
    default="checkpoint",
    help="Action source. The checkpoint policy is used by default.",
)
parser.add_argument("--warmup_s", type=float, default=2.0, help="Policy warmup duration before steady-state search.")
parser.add_argument("--collect_s", type=float, default=1.0, help="Required continuous steady-state collection window.")
parser.add_argument("--max_wait_s", type=float, default=20.0, help="Maximum steady-state search duration after warmup.")
parser.add_argument(
    "--min_episode_age_s",
    type=float,
    default=2.0,
    help="Minimum time since the latest reset before a sample can enter the window.",
)
parser.add_argument("--max_tilt_deg", type=float, default=3.0, help="Maximum absolute roll or pitch.")
parser.add_argument("--max_base_lin_speed", type=float, default=0.05, help="Maximum base linear speed in m/s.")
parser.add_argument("--max_base_ang_speed", type=float, default=0.10, help="Maximum base angular speed in rad/s.")
parser.add_argument("--max_joint_speed", type=float, default=0.35, help="Maximum absolute joint speed in rad/s.")
parser.add_argument("--min_root_height", type=float, default=0.65, help="Minimum root height above the terrain origin.")
parser.add_argument("--min_foot_force", type=float, default=20.0, help="Minimum contact-force norm on each foot in N.")
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Output Python file. Defaults to nominal_torque.py beside the checkpoint.",
)
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim is available."""

import math
import os

import gymnasium as gym
import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils.string import resolve_matching_names
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _validate_args() -> None:
    positive_values = {
        "warmup_s": args_cli.warmup_s,
        "collect_s": args_cli.collect_s,
        "max_wait_s": args_cli.max_wait_s,
        "min_episode_age_s": args_cli.min_episode_age_s,
        "max_tilt_deg": args_cli.max_tilt_deg,
        "max_base_lin_speed": args_cli.max_base_lin_speed,
        "max_base_ang_speed": args_cli.max_base_ang_speed,
        "max_joint_speed": args_cli.max_joint_speed,
        "min_root_height": args_cli.min_root_height,
        "min_foot_force": args_cli.min_foot_force,
    }
    invalid = {name: value for name, value in positive_values.items() if value <= 0.0}
    if invalid:
        raise ValueError(f"Collection arguments must be positive, received {invalid}.")


def _configure_deterministic_standing(env_cfg) -> None:
    """Build one repeatable nominal standing experiment."""
    command_cfg = env_cfg.commands.base_velocity
    command_cfg.rel_standing_envs = 1.0
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.debug_vis = False

    for event_name in ("physics_material", "add_base_mass", "base_external_force_torque", "push_robot"):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    env_cfg.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    env_cfg.events.reset_base.params["pose_range"] = {
        axis: (0.0, 0.0) for axis in env_cfg.events.reset_base.params["pose_range"]
    }
    env_cfg.events.reset_base.params["velocity_range"] = {
        axis: (0.0, 0.0) for axis in env_cfg.events.reset_base.params["velocity_range"]
    }

    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False

    if getattr(env_cfg, "curriculum", None) is not None:
        for term_name in ("terrain_levels", "lin_vel_cmd_levels", "ang_vel_cmd_levels"):
            if hasattr(env_cfg.curriculum, term_name):
                setattr(env_cfg.curriculum, term_name, None)

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = 0
    required_episode_s = args_cli.warmup_s + args_cli.max_wait_s + args_cli.collect_s + 1.0
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, required_episode_s)


def _resolve_checkpoint(agent_cfg) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            raise RuntimeError(f"A published checkpoint is unavailable for task '{args_cli.task}'.")
        return resume_path
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _zero_rows(mask: torch.Tensor, *buffers: torch.Tensor) -> None:
    if mask.any():
        for buffer in buffers:
            buffer[mask] = 0


def _format_dict(name: str, joint_names: list[str], values: np.ndarray) -> str:
    lines = [f"{name}: dict[str, float] = {{"]
    lines.extend(f'    "{joint_name}": {values[i]:.6f},' for i, joint_name in enumerate(joint_names))
    lines.append("}")
    return "\n".join(lines)


def main() -> None:
    """Run the source policy and save one paired steady standing reference."""
    _validate_args()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.seed = agent_cfg.seed
    _configure_deterministic_standing(env_cfg)
    action_source = args_cli.action_source
    resume_path = _resolve_checkpoint(agent_cfg) if action_source == "checkpoint" else None
    log_dir = os.path.dirname(resume_path) if resume_path else os.path.abspath(os.path.join("models", "flat"))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    try:
        unwrapped = env.unwrapped
        asset = unwrapped.scene["robot"]
        contact_sensor = unwrapped.scene["contact_forces"]
        foot_ids, foot_names = contact_sensor.find_bodies(".*ankle_roll.*")
        if len(foot_ids) != 2:
            raise RuntimeError(f"Expected two ankle-roll contact bodies, resolved {foot_names}.")

        action_term_name = "JointPositionAction"
        if action_term_name not in unwrapped.action_manager.active_terms:
            raise RuntimeError(
                f"Expected action term '{action_term_name}', received {unwrapped.action_manager.active_terms}."
            )
        action_term = unwrapped.action_manager.get_term(action_term_name)

        joint_names = list(asset.data.joint_names)
        action_joint_names = list(action_term._joint_names)
        if action_joint_names != joint_names:
            raise RuntimeError(
                "Position action order must match articulation order. "
                f"action={action_joint_names}, articulation={joint_names}"
            )

        device = unwrapped.device
        num_envs = unwrapped.num_envs
        num_dof = asset.data.applied_torque.shape[1]
        zero_actions = torch.zeros(
            (num_envs, unwrapped.action_manager.total_action_dim),
            device=device,
        )

        policy = None
        if action_source == "checkpoint":
            print(f"[INFO] Loading model checkpoint from: {resume_path}")
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(resume_path)
            policy = runner.get_inference_policy(device=device)
            source_description = f"checkpoint policy: {resume_path}"
        else:
            source_description = "zero residual-position action: q_target = q_default"
        print(f"[INFO] Action source: {source_description}")

        observations = env.get_observations()
        if version("rsl-rl-lib").startswith("2.3."):
            observations, _ = env.get_observations()

        def get_actions(current_observations):
            if policy is None:
                return zero_actions
            return policy(current_observations)

        dt = unwrapped.step_dt
        warmup_steps = max(1, math.ceil(args_cli.warmup_s / dt))
        window_steps = max(1, math.ceil(args_cli.collect_s / dt))
        max_wait_steps = max(window_steps, math.ceil(args_cli.max_wait_s / dt))
        min_episode_age_steps = math.ceil(args_cli.min_episode_age_s / dt)
        report_interval_steps = max(1, round(1.0 / dt))

        print(
            f"[INFO] dt={dt:.4f}s, envs={num_envs}, warmup={warmup_steps * dt:.2f}s, "
            f"continuous_window={window_steps * dt:.2f}s, max_wait={max_wait_steps * dt:.2f}s"
        )
        print(
            "[INFO] Stability criteria: "
            f"tilt<={args_cli.max_tilt_deg:.2f}deg, "
            f"|v_base|<={args_cli.max_base_lin_speed:.3f}m/s, "
            f"|w_base|<={args_cli.max_base_ang_speed:.3f}rad/s, "
            f"max|qd|<={args_cli.max_joint_speed:.3f}rad/s, "
            f"height>={args_cli.min_root_height:.3f}m, "
            f"each_foot_force>={args_cli.min_foot_force:.1f}N"
        )
        print(f"[INFO] Foot contact bodies: {foot_names}")

        for _ in range(warmup_steps):
            with torch.inference_mode():
                actions = get_actions(observations)
                observations, _, _, _ = env.step(actions)

        shape = (num_envs, num_dof)
        streak = torch.zeros(num_envs, dtype=torch.long, device=device)
        best_streak = torch.zeros_like(streak)
        collected = torch.zeros(num_envs, dtype=torch.bool, device=device)

        q_sum = torch.zeros(shape, device=device)
        q_sq_sum = torch.zeros(shape, device=device)
        q_target_sum = torch.zeros(shape, device=device)
        tau_sum = torch.zeros(shape, device=device)
        tau_sq_sum = torch.zeros(shape, device=device)
        grav_sum = torch.zeros(shape, device=device)
        root_pos_sum = torch.zeros(num_envs, 3, device=device)
        root_quat_sum = torch.zeros(num_envs, 4, device=device)
        foot_force_sum = torch.zeros(num_envs, 2, device=device)

        q_samples = torch.zeros(shape, device=device)
        q_std_samples = torch.zeros(shape, device=device)
        q_target_samples = torch.zeros(shape, device=device)
        tau_samples = torch.zeros(shape, device=device)
        tau_std_samples = torch.zeros(shape, device=device)
        grav_samples = torch.zeros(shape, device=device)
        root_pos_samples = torch.zeros(num_envs, 3, device=device)
        root_quat_samples = torch.zeros(num_envs, 4, device=device)
        foot_force_samples = torch.zeros(num_envs, 2, device=device)

        sum_buffers = (
            q_sum,
            q_sq_sum,
            q_target_sum,
            tau_sum,
            tau_sq_sum,
            grav_sum,
            root_pos_sum,
            root_quat_sum,
            foot_force_sum,
        )

        last_metrics: dict[str, torch.Tensor] = {}
        for search_step in range(max_wait_steps):
            with torch.inference_mode():
                actions = get_actions(observations)
                observations, _, dones, _ = env.step(actions)

                q = asset.data.joint_pos.clone()
                qd = asset.data.joint_vel.clone()
                q_target = action_term.processed_actions.clone()
                tau = asset.data.applied_torque.clone()
                grav_raw = asset.root_physx_view.get_gravity_compensation_forces()
                grav = grav_raw[:, grav_raw.shape[1] - num_dof :].clone()
                root_pos = asset.data.root_pos_w - unwrapped.scene.env_origins
                root_quat = asset.data.root_quat_w.clone()
                root_quat = torch.where(root_quat[:, :1] < 0.0, -root_quat, root_quat)
                roll, pitch, _ = euler_xyz_from_quat(root_quat)
                tilt = torch.maximum(roll.abs(), pitch.abs())
                base_lin_speed = asset.data.root_lin_vel_b.norm(dim=1)
                base_ang_speed = asset.data.root_ang_vel_b.norm(dim=1)
                max_joint_speed = qd.abs().amax(dim=1)
                foot_force = contact_sensor.data.net_forces_w[:, foot_ids, :].norm(dim=-1)
                min_foot_force = foot_force.amin(dim=1)
                root_height = root_pos[:, 2]
                finite = (
                    torch.isfinite(q).all(dim=1)
                    & torch.isfinite(q_target).all(dim=1)
                    & torch.isfinite(tau).all(dim=1)
                )
                stable = (
                    (unwrapped.episode_length_buf >= min_episode_age_steps)
                    & (tilt <= math.radians(args_cli.max_tilt_deg))
                    & (base_lin_speed <= args_cli.max_base_lin_speed)
                    & (base_ang_speed <= args_cli.max_base_ang_speed)
                    & (max_joint_speed <= args_cli.max_joint_speed)
                    & (root_height >= args_cli.min_root_height)
                    & (min_foot_force >= args_cli.min_foot_force)
                    & ~dones.bool()
                    & finite
                )

            active = ~collected
            restart = active & ~stable
            _zero_rows(restart, *sum_buffers)
            streak[restart] = 0

            accepted = active & stable
            q_sum[accepted] += q[accepted]
            q_sq_sum[accepted] += q[accepted].square()
            q_target_sum[accepted] += q_target[accepted]
            tau_sum[accepted] += tau[accepted]
            tau_sq_sum[accepted] += tau[accepted].square()
            grav_sum[accepted] += grav[accepted]
            root_pos_sum[accepted] += root_pos[accepted]
            root_quat_sum[accepted] += root_quat[accepted]
            foot_force_sum[accepted] += foot_force[accepted]
            streak[accepted] += 1
            best_streak = torch.maximum(best_streak, streak)

            newly_collected = accepted & (streak >= window_steps)
            if newly_collected.any():
                count = float(window_steps)
                q_mean = q_sum[newly_collected] / count
                tau_mean = tau_sum[newly_collected] / count
                q_samples[newly_collected] = q_mean
                q_std_samples[newly_collected] = torch.sqrt(
                    torch.clamp(q_sq_sum[newly_collected] / count - q_mean.square(), min=0.0)
                )
                q_target_samples[newly_collected] = q_target_sum[newly_collected] / count
                tau_samples[newly_collected] = tau_mean
                tau_std_samples[newly_collected] = torch.sqrt(
                    torch.clamp(tau_sq_sum[newly_collected] / count - tau_mean.square(), min=0.0)
                )
                grav_samples[newly_collected] = grav_sum[newly_collected] / count
                root_pos_samples[newly_collected] = root_pos_sum[newly_collected] / count
                quat_mean = root_quat_sum[newly_collected] / count
                root_quat_samples[newly_collected] = quat_mean / quat_mean.norm(dim=1, keepdim=True).clamp_min(1e-8)
                foot_force_samples[newly_collected] = foot_force_sum[newly_collected] / count
                collected[newly_collected] = True

            last_metrics = {
                "tilt": tilt,
                "base_lin_speed": base_lin_speed,
                "base_ang_speed": base_ang_speed,
                "max_joint_speed": max_joint_speed,
                "root_height": root_height,
                "min_foot_force": min_foot_force,
                "episode_age": unwrapped.episode_length_buf.float() * dt,
            }
            if search_step == 0 or (search_step + 1) % report_interval_steps == 0:
                diagnostic_env = int(best_streak.argmax().item())
                fastest_joint_id = int(qd[diagnostic_env].abs().argmax().item())
                print(
                    f"[search t={(search_step + 1) * dt:5.2f}s] "
                    f"stable_now={int(stable.sum())}/{num_envs}, "
                    f"collected={int(collected.sum())}/{num_envs}, "
                    f"best_window={best_streak.max().item() * dt:.2f}s; "
                    f"env={diagnostic_env}: "
                    f"tilt={math.degrees(tilt[diagnostic_env].item()):.2f}deg, "
                    f"v={base_lin_speed[diagnostic_env].item():.4f}m/s, "
                    f"w={base_ang_speed[diagnostic_env].item():.4f}rad/s, "
                    f"max_qd={max_joint_speed[diagnostic_env].item():.4f}rad/s"
                    f"({joint_names[fastest_joint_id]}), "
                    f"z={root_height[diagnostic_env].item():.4f}m, "
                    f"foot_min={min_foot_force[diagnostic_env].item():.2f}N, "
                    f"age={unwrapped.episode_length_buf[diagnostic_env].item() * dt:.2f}s",
                    flush=True,
                )
            if bool(collected.all()):
                break

        collected_ids = collected.nonzero(as_tuple=False).flatten()
        if collected_ids.numel() == 0:
            best_env = int(best_streak.argmax().item())
            metrics = ", ".join(
                f"{name}={value[best_env].item():.5f}" for name, value in last_metrics.items()
            )
            raise RuntimeError(
                f"No environment completed a {window_steps * dt:.2f}s stable window. "
                f"Best streak={best_streak[best_env].item() * dt:.2f}s in env {best_env}; {metrics}."
            )

        q_candidates = q_samples[collected_ids]
        q_center = q_candidates.median(dim=0).values
        representative_local_id = (q_candidates - q_center).abs().mean(dim=1).argmin()
        representative_env_id = int(collected_ids[representative_local_id].item())

        q_ref = q_samples[representative_env_id]
        q_std = q_std_samples[representative_env_id]
        q_target_ref = q_target_samples[representative_env_id]
        tau_ref = tau_samples[representative_env_id]
        tau_std = tau_std_samples[representative_env_id]
        grav_ref = grav_samples[representative_env_id]
        root_pos_ref = root_pos_samples[representative_env_id]
        root_quat_ref = root_quat_samples[representative_env_id]
        foot_force_ref = foot_force_samples[representative_env_id]

        print(
            f"[INFO] Completed windows: {collected_ids.numel()}/{num_envs}; "
            f"selected representative env={representative_env_id}"
        )
        print(
            f"[INFO] Window quality: max joint-position std={q_std.max().item():.6f}rad, "
            f"max torque std={tau_std.max().item():.6f}N·m, "
            f"foot forces={foot_force_ref.detach().cpu().tolist()}N"
        )

        joint_sdk_names = list(env_cfg.scene.robot.joint_sdk_names)
        joint_ids_map, _ = resolve_matching_names(joint_names, joint_sdk_names, preserve_order=True)

        q_np = q_ref.detach().cpu().numpy()
        q_std_np = q_std.detach().cpu().numpy()
        q_target_np = q_target_ref.detach().cpu().numpy()
        tau_np = tau_ref.detach().cpu().numpy()
        tau_std_np = tau_std.detach().cpu().numpy()
        grav_np = grav_ref.detach().cpu().numpy()
        q_sdk = np.zeros(num_dof)
        tau_sdk = np.zeros(num_dof)
        q_sdk[joint_ids_map] = q_np
        tau_sdk[joint_ids_map] = tau_np

        print("\n=== Paired standing reference (sim/action order) ===")
        print(
            f"{'idx':>3}  {'joint':28s}  {'q_ref':>10}  {'q_target':>10}  "
            f"{'pd_tau':>10}  {'tau_std':>9}  {'gravity*':>10}"
        )
        for i, name in enumerate(joint_names):
            print(
                f"{i:>3}  {name:28s}  {q_np[i]:10.5f}  {q_target_np[i]:10.5f}  "
                f"{tau_np[i]:10.4f}  {tau_std_np[i]:9.4f}  {grav_np[i]:10.4f}"
            )
        print("* PhysX generalized gravity term is diagnostic data for the floating-base articulation.")

        q_dict_str = _format_dict("JOINT_POSITION_REFERENCE", joint_names, q_np)
        q_target_dict_str = _format_dict("JOINT_POSITION_TARGET_REFERENCE", joint_names, q_target_np)
        tau_dict_str = _format_dict("NOMINAL_TORQUE", joint_names, tau_np)
        print("\n=== Paired reference ===")
        print(q_dict_str)
        print()
        print(tau_dict_str)

        out_path = args_cli.output or os.path.join(log_dir, "nominal_torque.py")
        out_path = os.path.abspath(os.path.expanduser(out_path))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        root_pos_values = root_pos_ref.detach().cpu().tolist()
        root_quat_values = root_quat_ref.detach().cpu().tolist()
        foot_force_values = foot_force_ref.detach().cpu().tolist()
        with open(out_path, "w", encoding="utf-8") as output_file:
            output_file.write("# Paired steady-standing reference collected from an implicit-PD position policy.\n")
            output_file.write("# Keep JOINT_POSITION_REFERENCE and NOMINAL_TORQUE together.\n")
            output_file.write(f"# Action source: {source_description}\n")
            output_file.write(f"# Continuous window: {window_steps * dt:.4f} s\n")
            output_file.write(f"# Representative environment: {representative_env_id}\n\n")
            output_file.write(q_dict_str + "\n\n")
            output_file.write(q_target_dict_str + "\n\n")
            output_file.write(tau_dict_str + "\n\n")
            output_file.write(
                "ROOT_POSITION_REFERENCE = (" + ", ".join(f"{value:.8f}" for value in root_pos_values) + ")\n"
            )
            output_file.write(
                "ROOT_QUATERNION_REFERENCE = ("
                + ", ".join(f"{value:.8f}" for value in root_quat_values)
                + ")\n"
            )
            output_file.write(
                "MEAN_FOOT_CONTACT_FORCE = ("
                + ", ".join(f"{value:.6f}" for value in foot_force_values)
                + ")\n\n"
            )
            output_file.write("# SDK/deploy order:\n")
            output_file.write("q_ref_sdk = [" + ", ".join(f"{value:.6f}" for value in q_sdk) + "]\n")
            output_file.write("tau_sdk = [" + ", ".join(f"{value:.6f}" for value in tau_sdk) + "]\n")
            output_file.write("q_std_sim = [" + ", ".join(f"{value:.8f}" for value in q_std_np) + "]\n")
            output_file.write("tau_std_sim = [" + ", ".join(f"{value:.8f}" for value in tau_std_np) + "]\n")
        print(f"\n[INFO] Saved paired standing reference to: {out_path}")
    except Exception as error:
        print(f"[ERROR] {type(error).__name__}: {error}", flush=True)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
