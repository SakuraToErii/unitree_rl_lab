# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render the effort task under its current standing-torque reference.

The script loads an effort environment directly, verifies that every actuator has
zero stiffness and damping, and repeatedly sends a zero policy action. With the
residual effort action configuration this produces::

    tau_command = tau_ref + action_scale * 0 = tau_ref

No policy, runner, checkpoint, or exported model is loaded.

Usage::

    source ~/projects/IsaacLab/.venv/bin/activate
    python scripts/rsl_rl/play_zero_action.py

Use a paired reference produced by ``collect_nominal_torque.py``::

    python scripts/rsl_rl/play_zero_action.py \
        --reference-file models/flat/nominal_torque.py

The viewer runs in real time by default. Use ``--duration 30`` to extend the
test, or ``--headless --no-real-time`` for a fast diagnostic run.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Render zero-action standing under the current tau_ref.")
parser.add_argument(
    "--task",
    type=str,
    default="Unitree-G1-29dof-Effort-POMDP2",
    help="Effort-control task whose environment configuration supplies tau_ref.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of rendered environments.")
parser.add_argument("--duration", type=float, default=20.0, help="Standing-test duration in seconds.")
parser.add_argument("--print_interval", type=float, default=1.0, help="Diagnostic print interval in seconds.")
parser.add_argument(
    "--reference-file",
    type=str,
    default=None,
    help="Paired q_ref/tau_ref Python file produced by collect_nominal_torque.py.",
)
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Pace simulation at wall-clock speed. Use --no-real-time for maximum speed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim is available."""

import math
import os
import runpy
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.math import euler_xyz_from_quat

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def _load_paired_reference(env_cfg, path: str) -> None:
    """Apply a collected q_ref/tau_ref pair before the environment is created."""
    reference_path = os.path.abspath(os.path.expanduser(path))
    values = runpy.run_path(reference_path)
    required_names = ("JOINT_POSITION_REFERENCE", "NOMINAL_TORQUE")
    missing_names = [name for name in required_names if name not in values]
    if missing_names:
        raise ValueError(f"Reference file '{reference_path}' is missing {missing_names}.")

    q_ref = values["JOINT_POSITION_REFERENCE"]
    tau_ref = values["NOMINAL_TORQUE"]
    if not isinstance(q_ref, dict) or not isinstance(tau_ref, dict):
        raise TypeError("JOINT_POSITION_REFERENCE and NOMINAL_TORQUE must both be dictionaries.")

    expected_joint_names = set(env_cfg.scene.robot.joint_sdk_names)
    q_joint_names = set(q_ref)
    tau_joint_names = set(tau_ref)
    if q_joint_names != expected_joint_names or tau_joint_names != expected_joint_names:
        raise ValueError(
            "Reference joint names must match the effort robot. "
            f"q_missing={sorted(expected_joint_names - q_joint_names)}, "
            f"q_extra={sorted(q_joint_names - expected_joint_names)}, "
            f"tau_missing={sorted(expected_joint_names - tau_joint_names)}, "
            f"tau_extra={sorted(tau_joint_names - expected_joint_names)}"
        )

    env_cfg.scene.robot.init_state.joint_pos = {name: float(value) for name, value in q_ref.items()}
    env_cfg.scene.robot.init_state.joint_vel = {".*": 0.0}
    env_cfg.actions.JointEffortAction.offset = {name: float(value) for name, value in tau_ref.items()}

    if "ROOT_POSITION_REFERENCE" in values:
        root_position = tuple(float(value) for value in values["ROOT_POSITION_REFERENCE"])
        if len(root_position) != 3:
            raise ValueError(f"ROOT_POSITION_REFERENCE must contain three values, received {root_position}.")
        env_cfg.scene.robot.init_state.pos = root_position
    if "ROOT_QUATERNION_REFERENCE" in values:
        root_quaternion = tuple(float(value) for value in values["ROOT_QUATERNION_REFERENCE"])
        if len(root_quaternion) != 4:
            raise ValueError(f"ROOT_QUATERNION_REFERENCE must contain four values, received {root_quaternion}.")
        env_cfg.scene.robot.init_state.rot = root_quaternion

    print(f"[setup] Loaded paired q_ref/tau_ref from: {reference_path}")


def _disable_disturbances(env_cfg) -> None:
    """Turn the task into a deterministic nominal standing check."""
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

    pose_range = env_cfg.events.reset_base.params["pose_range"]
    for axis in pose_range:
        pose_range[axis] = (0.0, 0.0)
    velocity_range = env_cfg.events.reset_base.params["velocity_range"]
    for axis in velocity_range:
        velocity_range[axis] = (0.0, 0.0)

    if getattr(env_cfg, "curriculum", None) is not None:
        for term_name in ("terrain_levels", "lin_vel_cmd_levels", "ang_vel_cmd_levels"):
            if hasattr(env_cfg.curriculum, term_name):
                setattr(env_cfg.curriculum, term_name, None)

    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.scene.terrain.max_init_terrain_level = 0


def _tensor_abs_max(value) -> float:
    tensor = torch.as_tensor(value)
    return float(tensor.abs().max().item())


def _verify_pure_effort_control(env) -> tuple[object, torch.Tensor]:
    """Check the action and actuator chain, then return the action term and tau_ref."""
    unwrapped = env.unwrapped
    robot = unwrapped.scene["robot"]
    action_term_name = "JointEffortAction"
    if action_term_name not in unwrapped.action_manager.active_terms:
        raise RuntimeError(
            f"Expected action term '{action_term_name}', received {unwrapped.action_manager.active_terms}."
        )

    max_kp = 0.0
    max_kd = 0.0
    for actuator_name, actuator in robot.actuators.items():
        actuator_kp = _tensor_abs_max(actuator.stiffness)
        actuator_kd = _tensor_abs_max(actuator.damping)
        max_kp = max(max_kp, actuator_kp)
        max_kd = max(max_kd, actuator_kd)
        print(f"[actuator] {actuator_name}: max|kp|={actuator_kp:.6g}, max|kd|={actuator_kd:.6g}")

    if max_kp > 1.0e-8 or max_kd > 1.0e-8:
        raise RuntimeError(f"Pure effort check failed: max|kp|={max_kp}, max|kd|={max_kd}.")

    action_term = unwrapped.action_manager.get_term(action_term_name)
    offset = action_term._offset
    if isinstance(offset, torch.Tensor):
        tau_ref = offset[0].detach().clone()
    else:
        tau_ref = torch.full(
            (action_term.action_dim,),
            float(offset),
            device=unwrapped.device,
            dtype=robot.data.joint_pos.dtype,
        )

    print(
        f"[setup] action_dim={action_term.action_dim}, step_dt={unwrapped.step_dt:.4f}s, "
        f"tau_ref_norm={tau_ref.norm().item():.4f} N·m, "
        f"tau_ref_range=[{tau_ref.min().item():.4f}, {tau_ref.max().item():.4f}] N·m"
    )
    return action_term, tau_ref


def _print_state(env, action_term, elapsed_s: float) -> None:
    robot = env.unwrapped.scene["robot"]
    data = robot.data
    roll, pitch, _ = euler_xyz_from_quat(data.root_quat_w)
    root_height = data.root_pos_w[:, 2] - env.unwrapped.scene.env_origins[:, 2]
    joint_error = data.joint_pos - data.default_joint_pos
    command_torque = action_term.processed_actions

    print(
        f"[state t={elapsed_s:6.2f}s] "
        f"z={root_height[0].item():.3f}m, "
        f"roll={math.degrees(roll[0].item()):+.2f}deg, "
        f"pitch={math.degrees(pitch[0].item()):+.2f}deg, "
        f"|v_base|={data.root_lin_vel_b[0].norm().item():.3f}m/s, "
        f"|w_base|={data.root_ang_vel_b[0].norm().item():.3f}rad/s, "
        f"max|q-q0|={joint_error[0].abs().max().item():.3f}rad, "
        f"max|qd|={data.joint_vel[0].abs().max().item():.3f}rad/s, "
        f"|tau_cmd|={command_torque[0].norm().item():.3f}N·m, "
        f"|tau_applied|={data.applied_torque[0].norm().item():.3f}N·m"
    )


def _termination_reasons(env, env_id: int) -> list[str]:
    manager = env.unwrapped.termination_manager
    return [name for name in manager.active_terms if bool(manager.get_term(name)[env_id].item())]


def main() -> None:
    if "Effort" not in args_cli.task:
        raise ValueError(f"This script requires an Effort task, received '{args_cli.task}'.")
    if args_cli.duration <= 0.0:
        raise ValueError(f"--duration must be positive, received {args_cli.duration}.")
    if args_cli.print_interval <= 0.0:
        raise ValueError(f"--print_interval must be positive, received {args_cli.print_interval}.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=True,
        entry_point_key="play_env_cfg_entry_point",
    )
    if args_cli.reference_file is not None:
        _load_paired_reference(env_cfg, args_cli.reference_file)
    _disable_disturbances(env_cfg)

    control_dt = env_cfg.sim.dt * env_cfg.decimation
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args_cli.duration + control_dt)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    try:
        action_term, tau_ref = _verify_pure_effort_control(env)
        observation, _ = env.reset()
        del observation

        num_steps = math.ceil(args_cli.duration / env.unwrapped.step_dt)
        print_interval_steps = max(1, round(args_cli.print_interval / env.unwrapped.step_dt))
        zero_action = torch.zeros(
            (env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
        )

        print(
            f"[run] Sending zero action for {args_cli.duration:.2f}s ({num_steps} control steps). "
            f"Expected command torque is tau_ref."
        )
        completed_steps = 0
        failed = False

        for step in range(num_steps):
            if not simulation_app.is_running():
                break
            step_start = time.time()

            with torch.inference_mode():
                _, _, terminated, truncated, _ = env.step(zero_action)

            completed_steps = step + 1
            elapsed_s = completed_steps * env.unwrapped.step_dt
            if step == 0:
                torch.testing.assert_close(action_term.processed_actions[0], tau_ref, atol=1.0e-5, rtol=1.0e-5)
            if step == 0 or completed_steps % print_interval_steps == 0:
                _print_state(env, action_term, elapsed_s)

            failed_env_ids = terminated.nonzero(as_tuple=False).flatten().tolist()
            if failed_env_ids:
                failed = True
                for env_id in failed_env_ids:
                    print(
                        f"[result] env={env_id} terminated at t={elapsed_s:.3f}s, "
                        f"reasons={_termination_reasons(env, env_id)}"
                    )
                break

            timed_out_env_ids = truncated.nonzero(as_tuple=False).flatten().tolist()
            if timed_out_env_ids:
                print(f"[result] timeout at t={elapsed_s:.3f}s for envs={timed_out_env_ids}")
                break

            sleep_time = env.unwrapped.step_dt - (time.time() - step_start)
            if args_cli.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)

        elapsed_s = completed_steps * env.unwrapped.step_dt
        if failed:
            print(f"[result] FAIL: tau_ref zero-action standing lasted {elapsed_s:.3f}s.")
        elif completed_steps == num_steps:
            _print_state(env, action_term, elapsed_s)
            print(f"[result] PASS: tau_ref zero-action standing completed {elapsed_s:.3f}s.")
        else:
            print(f"[result] STOPPED: viewer closed after {elapsed_s:.3f}s.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
