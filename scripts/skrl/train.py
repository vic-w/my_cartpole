# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
import warnings
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument(
    "--camera_snapshot_interval",
    type=int,
    default=0,
    help=(
        "Save a still image from the task camera every N environment steps during training. "
        "Disabled when set to 0."
    ),
)
parser.add_argument(
    "--camera_snapshot_env_index",
    type=int,
    default=0,
    help="Environment index to use when saving camera snapshots.",
)
parser.add_argument(
    "--camera_snapshot_dir",
    type=str,
    default=None,
    help="Optional output directory for camera snapshots (defaults to the run's log directory).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
if args_cli.camera_snapshot_interval > 0:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import random
from datetime import datetime

import omni
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import numpy as np

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None


class CameraSnapshotWrapper(gym.Wrapper):
    """Gym wrapper that periodically saves RGB images from a named scene camera."""

    def __init__(
        self,
        env: gym.Env,
        sensor_name: str,
        env_index: int,
        interval: int,
        output_dir: Path,
    ) -> None:
        super().__init__(env)
        if interval <= 0:
            raise ValueError("Snapshot interval must be positive when using CameraSnapshotWrapper.")
        self._base_env = self.unwrapped
        self._sensor_name = sensor_name
        self._env_index = env_index
        self._interval = interval
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._step_counter = 0
        self._warned_imageio = False

        scene = getattr(self._base_env, "scene", None)
        sensors = getattr(scene, "sensors", {}) if scene is not None else {}
        if sensor_name not in sensors:
            raise ValueError(
                f"Camera sensor '{sensor_name}' is not available in the environment. "
                "Verify that the camera resource is registered in the scene configuration."
            )
        self._camera = sensors[sensor_name]

        # Validate environment index bounds if possible.
        num_envs = getattr(scene, "num_envs", None)
        if num_envs is not None and not (0 <= env_index < num_envs):
            raise ValueError(
                f"camera_snapshot_env_index={env_index} is out of bounds for {num_envs} environments."
            )

    def reset(self, **kwargs):  # noqa: D401 - gymnasium signature
        results = super().reset(**kwargs)
        self._step_counter = 0
        self._maybe_capture(step_is_reset=True)
        return results

    def step(self, action):  # noqa: D401 - gymnasium signature
        results = super().step(action)
        self._step_counter += 1
        if self._step_counter % self._interval == 0:
            self._maybe_capture()
        return results

    def _maybe_capture(self, step_is_reset: bool = False) -> None:
        if imageio is None:
            if not self._warned_imageio:
                warnings.warn(
                    "imageio is not installed; camera snapshots will be skipped.",
                    RuntimeWarning,
                )
                self._warned_imageio = True
            return

        output = self._camera.data.output
        rgb_frame = None
        if isinstance(output, dict):
            rgb_frame = output.get("rgb")
        else:
            getter = getattr(output, "get", None)
            if callable(getter):
                rgb_frame = getter("rgb")
            else:
                rgb_frame = getattr(output, "rgb", None)
        if rgb_frame is None:
            raise RuntimeError(
                f"Camera sensor '{self._sensor_name}' does not provide 'rgb' data. "
                "Ensure that 'rgb' is present in data_types for the sensor configuration."
            )

        frame_source = rgb_frame
        ndim = getattr(frame_source, "ndim", None)
        if ndim == 3:
            if self._env_index != 0:
                raise ValueError(
                    "Camera snapshot env index must be 0 when the sensor does not provide batched outputs."
                )
            frame = frame_source
        else:
            frame = frame_source[self._env_index]
        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        else:
            frame = np.asarray(frame)

        if frame.dtype in (np.float32, np.float64):
            frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        if frame.shape[-1] == 4:
            frame = frame[..., :3]

        step_tag = "reset" if step_is_reset else f"step_{self._step_counter:06d}"
        filename = self._output_dir / f"{step_tag}.png"
        imageio.imwrite(filename, frame)


import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import my_cartpole.tasks  # noqa: F401

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = os.path.join(log_root_path, log_dir)
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # determine render mode: enable RGB output when video recording or snapshots are requested
    render_mode = "rgb_array" if args_cli.video or args_cli.camera_snapshot_interval > 0 else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # optionally save camera snapshots during training
    if args_cli.camera_snapshot_interval > 0:
        snapshot_dir = Path(args_cli.camera_snapshot_dir) if args_cli.camera_snapshot_dir else Path(log_dir) / "snapshots"
        print(
            "[INFO] Saving camera snapshots to"
            f" '{snapshot_dir}' every {args_cli.camera_snapshot_interval} environment steps (env index"
            f" {args_cli.camera_snapshot_env_index})."
        )
        env = CameraSnapshotWrapper(
            env,
            sensor_name="camera",
            env_index=args_cli.camera_snapshot_env_index,
            interval=args_cli.camera_snapshot_interval,
            output_dir=snapshot_dir,
        )

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
