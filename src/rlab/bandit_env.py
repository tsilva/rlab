from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AutoresetMode


class BanditVectorEnv(gym.vector.VectorEnv):
    """Deterministic two-arm native vector environment for rlab smoke tests."""

    metadata = {
        "autoreset_mode": AutoresetMode.DISABLED,
        "render_modes": [],
    }

    def __init__(
        self,
        game: str,
        num_envs: int,
        *,
        autoreset_mode: AutoresetMode,
    ) -> None:
        if game != "Bandit-v0":
            raise ValueError(f"unknown rlab environment {game!r}; expected 'Bandit-v0'")
        if int(num_envs) < 1:
            raise ValueError("num_envs must be >= 1")
        if autoreset_mode is not AutoresetMode.DISABLED:
            raise ValueError("Bandit-v0 requires autoreset_mode=DISABLED")

        self.num_envs = int(num_envs)
        self.autoreset_mode = autoreset_mode
        self.single_observation_space = gym.spaces.Box(
            low=0.0,
            high=0.0,
            shape=(1,),
            dtype=np.float32,
        )
        self.single_action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.vector.utils.batch_space(
            self.single_observation_space,
            self.num_envs,
        )
        self.action_space = gym.vector.utils.batch_space(
            self.single_action_space,
            self.num_envs,
        )
        self._observations = np.zeros((self.num_envs, 1), dtype=np.float32)
        self._needs_reset = np.ones(self.num_envs, dtype=np.bool_)
        self.closed = False

    def _reset_mask(self, options: Mapping[str, Any] | None) -> np.ndarray:
        reset_options = dict(options or {})
        raw_mask = reset_options.pop("reset_mask", None)
        if reset_options:
            unknown = ", ".join(sorted(str(key) for key in reset_options))
            raise ValueError(f"unsupported Bandit-v0 reset option(s): {unknown}")
        if raw_mask is None:
            return np.ones(self.num_envs, dtype=np.bool_)
        if not isinstance(raw_mask, np.ndarray):
            raise TypeError("options['reset_mask'] must be a numpy array")
        if raw_mask.shape != (self.num_envs,):
            raise ValueError(
                f"options['reset_mask'] must have shape ({self.num_envs},), "
                f"got {raw_mask.shape}"
            )
        if raw_mask.dtype != np.bool_:
            raise TypeError("options['reset_mask'] must have dtype=bool")
        if not np.any(raw_mask):
            raise ValueError("options['reset_mask'] must select at least one lane")
        return raw_mask

    def _validate_seed(self, seed: int | Sequence[int | None] | None) -> None:
        if seed is None or isinstance(seed, int):
            return
        if isinstance(seed, Sequence) and not isinstance(seed, str | bytes):
            if len(seed) != self.num_envs:
                raise ValueError(
                    f"seed sequence must contain {self.num_envs} values, got {len(seed)}"
                )
            if all(value is None or isinstance(value, int) for value in seed):
                return
        raise TypeError("seed must be an int, None, or one int/None value per lane")

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        if self.closed:
            raise RuntimeError("Bandit-v0 is closed")
        self._validate_seed(seed)
        mask = self._reset_mask(options)
        self._observations[mask] = 0.0
        self._needs_reset[mask] = False
        return self._observations, {}

    def step(
        self,
        actions: Any,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if self.closed:
            raise RuntimeError("Bandit-v0 is closed")
        if np.any(self._needs_reset):
            lanes = np.flatnonzero(self._needs_reset).tolist()
            raise RuntimeError(f"Bandit-v0 lanes require reset before step: {lanes}")

        action_batch = np.asarray(actions)
        if action_batch.shape != (self.num_envs,):
            raise ValueError(
                f"actions must have shape ({self.num_envs},), got {action_batch.shape}"
            )
        if not np.issubdtype(action_batch.dtype, np.integer):
            raise TypeError("Bandit-v0 actions must be integers")
        if np.any((action_batch < 0) | (action_batch >= 2)):
            raise ValueError("Bandit-v0 actions must be 0 or 1")

        chosen_arm = action_batch.astype(np.int64, copy=True)
        rewards = chosen_arm.astype(np.float32)
        terminated = np.ones(self.num_envs, dtype=np.bool_)
        truncated = np.zeros(self.num_envs, dtype=np.bool_)
        self._needs_reset[:] = True
        infos = {
            "chosen_arm": chosen_arm,
            "optimal_arm": np.ones(self.num_envs, dtype=np.int64),
            "is_optimal": chosen_arm == 1,
        }
        return self._observations, rewards, terminated, truncated, infos

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True
