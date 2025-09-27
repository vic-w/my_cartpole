"""Utility helpers for camera-based logging in skrl scripts."""

from __future__ import annotations

from pathlib import Path
import warnings

import gymnasium as gym
import numpy as np

try:  # pragma: no cover - optional dependency
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


__all__ = ["CameraSnapshotWrapper"]
