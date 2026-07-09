# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect the standing-torque baseline (tau0) for the residual-torque effort task.

Based on ``play.py``: loads a converged standing policy, forces a zero velocity
command so the robot just stands, lets it settle, and averages the applied joint
torque over a steady window.  The result is the per-joint standing torque
``G(q_default)`` that you paste into ``NOMINAL_TORQUE`` in ``effort_env_cfg.py``.

Run it with a standing-capable *position-control* policy (the effort task has no
converged policy yet; the position PD actuator is what holds the robot against
gravity, and its applied torque at steady state equals ``G(q)``)::

    python scripts/rsl_rl/collect_nominal_torque.py \
        --task Unitree-G1-29dof-Velocity \
        --checkpoint logs/rsl_rl/unitree_g1_29dof_velocity/<run>/model_*.pt \
        --num_envs 32 --headless

Joint order: torque is read in the sim/action order (``asset.data.joint_names``),
which is exactly the order ``JointEffortActionCfg(offset=NOMINAL_TORQUE)`` resolves
the dict against.  Output is a name-keyed dict (order-independent, paste-ready)
plus the SDK-ordered vector for reference.  The analytic gravity-compensation force
is printed alongside as a cross-check (at steady state it must match applied_torque).
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect standing-torque baseline for the residual effort task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate (more = less noise).")
parser.add_argument("--task", type=str, default=None, help="Name of the task (a standing-capable locomotion task).")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--warmup_steps", type=int, default=100, help="Steps to let the robot settle before collecting.")
parser.add_argument("--collect_steps", type=int, default=200, help="Steps to average the torque over.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os

import numpy as np
import torch

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import isaaclab.terrains as terrain_gen
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab.utils.string import resolve_matching_names

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def main():
    """Play a standing policy at zero command and collect the standing torque."""
    # parse configuration (use the *play* env cfg, same as play.py)
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # ------------------------------------------------------------------
    # Overrides so the measurement is the nominal robot's level-ground
    # standing torque G(q_default), not a perturbed/averaged one.
    # ------------------------------------------------------------------
    # 1) Force a zero velocity command -> pure standing (all envs are "standing").
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "base_velocity"):
        cmd = env_cfg.commands.base_velocity
        cmd.rel_standing_envs = 1.0
        cmd.ranges.lin_vel_x = (0.0, 0.0)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.debug_vis = False

    # 2) Disable per-env disturbances (EventManager skips None terms).
    if hasattr(env_cfg, "events"):
        env_cfg.events.add_base_mass = None           # keep nominal mass
        env_cfg.events.push_robot = None              # no velocity kicks
        env_cfg.events.base_external_force_torque = None

    # 3) Flat terrain only -> standing torque is the level-ground G(q_default).
    env_cfg.scene.terrain.terrain_generator.sub_terrains = {
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)
    }
    env_cfg.scene.terrain.max_init_terrain_level = 0

    # locate the checkpoint (same logic as play.py)
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    asset = env.unwrapped.scene["robot"]
    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    num_dof = asset.data.applied_torque.shape[1]
    dt = env.unwrapped.step_dt

    print(
        f"[INFO] Collecting standing torque: warmup={args_cli.warmup_steps} "
        f"({args_cli.warmup_steps * dt:.2f}s), collect={args_cli.collect_steps} "
        f"({args_cli.collect_steps * dt:.2f}s), {num_envs} envs, dt={dt:.4f}s"
    )

    # warmup: settle to a standing pose under the zero command
    for _ in range(args_cli.warmup_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    # collect: average applied_torque and the analytic gravity-comp force over the
    # steady window.  Skip envs that just reset (episode_length_buf <= 10) so we never
    # mix in a fall/re-settle transient.
    tau_sum = torch.zeros(num_envs, num_dof, device=device)
    grav_sum = torch.zeros(num_envs, num_dof, device=device)
    cnt = torch.zeros(num_envs, device=device)
    for _ in range(args_cli.collect_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
            tau = asset.data.applied_torque.clone()
            # physx gravity-comp returns generalized forces over ALL dofs:
            # [floating-base root dofs] + [joints in asset.data.joint_names order].
            # Slice off the root dofs so grav aligns with applied_torque (29 joints).
            grav_raw = asset.root_physx_view.get_gravity_compensation_forces()
            grav = grav_raw[:, grav_raw.shape[1] - num_dof:].clone()
            standing = env.unwrapped.episode_length_buf > 10
        tau_sum[standing] += tau[standing]
        grav_sum[standing] += grav[standing]
        cnt[standing] += 1.0

    cnt_clamped = cnt.clamp(min=1.0).unsqueeze(1)
    used = cnt > 0
    tau_mean = (tau_sum / cnt_clamped)[used].mean(dim=0)        # (num_dof,) sim order
    grav_mean = (grav_sum / cnt_clamped)[used].mean(dim=0)     # (num_dof,) sim order
    print(f"[INFO] Contributing envs: {int(used.sum())}/{num_envs}")

    # ------------------------------------------------------------------
    # Output (joint-order aware)
    # ------------------------------------------------------------------
    joint_names = list(asset.data.joint_names)                 # sim/action order (29)
    joint_sdk_names = list(env_cfg.scene.robot.joint_sdk_names)  # sdk/deploy order (29)
    # joint_ids_map[i] = sdk index of the i-th sim joint  (see export_deploy_cfg.py)
    joint_ids_map, _ = resolve_matching_names(joint_names, joint_sdk_names, preserve_order=True)

    tau_np = tau_mean.detach().cpu().numpy()
    grav_np = grav_mean.detach().cpu().numpy()
    tau_sdk = np.zeros(num_dof)
    tau_sdk[joint_ids_map] = tau_np                            # reorder sim -> sdk

    # per-joint table (sim/action order)
    print("\n=== Standing torque per joint (sim/action order) ===")
    print(f"{'idx':>3}  {'joint':28s}  {'applied_tau':>11}  {'gravity_comp':>12}  {'diff':>8}")
    for i, name in enumerate(joint_names):
        print(
            f"{i:>3}  {name:28s}  {tau_np[i]:11.4f}  {grav_np[i]:12.4f}  {tau_np[i] - grav_np[i]:8.4f}"
        )

    # paste-ready dict (name-keyed -> NOMINAL_TORQUE in effort_env_cfg.py)
    dict_lines = ["NOMINAL_TORQUE: dict[str, float] = {"]
    for i, name in enumerate(joint_names):
        dict_lines.append(f'    "{name}": {tau_np[i]:.4f},')
    dict_lines.append("}")
    dict_str = "\n".join(dict_lines)
    print("\n=== Paste into NOMINAL_TORQUE (effort_env_cfg.py) ===")
    print(dict_str)

    print("\n=== SDK-ordered torque vector (deploy order, reference) ===")
    print("[" + ", ".join(f"{v:.4f}" for v in tau_sdk) + "]")

    # save a paste-ready file next to the checkpoint
    out_path = os.path.join(log_dir, "nominal_torque.py")
    with open(out_path, "w") as f:
        f.write("# Standing-torque baseline tau0 = G(q_default), collected by\n")
        f.write("# collect_nominal_torque.py from a converged standing policy at zero command.\n")
        f.write("# Paste this dict into NOMINAL_TORQUE in effort_env_cfg.py.\n")
        f.write(dict_str + "\n")
        f.write("\n# SDK-ordered vector (deploy order):\n")
        f.write("tau_sdk = [" + ", ".join(f"{v:.4f}" for v in tau_sdk) + "]\n")
    print(f"\n[INFO] Saved paste-ready baseline to: {out_path}")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()