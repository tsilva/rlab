from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv


def _array_item(value: Any, index: int) -> Any:
    if isinstance(value, np.ndarray):
        item = value[index]
        if isinstance(item, np.generic):
            return item.item()
        return item
    if isinstance(value, (list, tuple)) and len(value) > index:
        return value[index]
    return value


class VectorInfoView:
    """Lazy lane access for Gymnasium vector info mappings."""

    def __init__(self, infos: Any, num_envs: int):
        self.infos = infos
        self.num_envs = int(num_envs)
        self._masks: dict[str, np.ndarray] | None = None

    @property
    def is_mapping(self) -> bool:
        return isinstance(self.infos, Mapping)

    def _mask_map(self) -> dict[str, np.ndarray]:
        if self._masks is not None:
            return self._masks
        masks: dict[str, np.ndarray] = {}
        if isinstance(self.infos, Mapping):
            for key, value in self.infos.items():
                if isinstance(key, str) and key.startswith("_"):
                    masks[key[1:]] = np.asarray(value, dtype=bool)
        self._masks = masks
        return masks

    def has_lane(self, key: str, index: int) -> bool:
        if isinstance(self.infos, (list, tuple)):
            return index < len(self.infos) and isinstance(self.infos[index], Mapping) and key in self.infos[index]
        if not isinstance(self.infos, Mapping) or key not in self.infos:
            return False
        mask = self._mask_map().get(key)
        if mask is not None and index < len(mask) and not bool(mask[index]):
            return False
        return True

    def value(self, key: str, index: int, default: Any = None) -> Any:
        if isinstance(self.infos, (list, tuple)):
            if index >= len(self.infos) or not isinstance(self.infos[index], Mapping):
                return default
            value = self.infos[index].get(key, default)
            if isinstance(value, np.generic):
                return value.item()
            return value
        if not self.has_lane(key, index):
            return default
        if not isinstance(self.infos, Mapping):
            return default
        value = self.infos[key]
        if isinstance(value, Mapping):
            return VectorInfoView(value, self.num_envs).lane(index)
        return _array_item(value, index)

    def lane_mapping(self, key: str, index: int) -> dict[str, Any]:
        value = self.value(key, index, {})
        return dict(value) if isinstance(value, Mapping) else {}

    def array(self, key: str, default: Any = 0) -> np.ndarray:
        if not isinstance(self.infos, Mapping) or key not in self.infos:
            return np.full((self.num_envs,), default)
        value = self.infos[key]
        if isinstance(value, Mapping):
            return np.asarray([VectorInfoView(value, self.num_envs).lane(index) for index in range(self.num_envs)])
        return np.asarray(value)

    def lane(self, index: int) -> dict[str, Any]:
        if isinstance(self.infos, (list, tuple)):
            if index < len(self.infos) and isinstance(self.infos[index], Mapping):
                return dict(self.infos[index])
            return {}
        if not isinstance(self.infos, Mapping):
            return {}
        result: dict[str, Any] = {}
        for key in self.infos:
            if not isinstance(key, str) or key.startswith("_"):
                continue
            if self.has_lane(key, index):
                result[key] = self.value(key, index)
        return result

    def reset_info(self, index: int) -> dict[str, Any]:
        info = self.lane(index)
        info.pop("final_info", None)
        info.pop("final_obs", None)
        return info

    def terminal_info(self, index: int, *, truncated: bool) -> dict[str, Any]:
        info = self.lane_mapping("final_info", index)
        final_obs = self.value("final_obs", index)
        if final_obs is not None:
            info["terminal_observation"] = final_obs
        info["reset_info"] = self.reset_info(index)
        info["TimeLimit.truncated"] = bool(truncated)
        return info


@dataclass
class HookStep:
    rewards: np.ndarray
    infos: dict[int, dict[str, Any]]


class FusedVectorHooks:
    """Provider/target semantics plugged into a generic fused vector pipeline."""

    def __init__(self, config: Any, native_env: gym.vector.VectorEnv):
        self.config = config
        self.native_env = native_env

    @property
    def action_space(self) -> gym.Space:
        return getattr(self.native_env, "single_action_space", self.native_env.action_space)

    def map_actions(self, actions: Any) -> Any:
        return actions

    def on_reset(self, infos: VectorInfoView) -> None:
        del infos

    def shape_step(
        self,
        rewards: np.ndarray,
        terminations: np.ndarray,
        truncations: np.ndarray,
        infos: VectorInfoView,
    ) -> HookStep:
        del terminations, truncations, infos
        return HookStep(np.asarray(rewards, dtype=np.float32), {})


class IdentityFusedHooks(FusedVectorHooks):
    pass


@dataclass
class FusedStep:
    obs: Any
    rewards: np.ndarray
    terminations: np.ndarray
    truncations: np.ndarray
    dones: np.ndarray
    infos: list[dict[str, Any]]


class FusedGymVectorPipeline:
    """Trainer-agnostic fast path for Gymnasium VectorEnv stepping."""

    HOOK_WITH_TERMINAL_NATIVE = "hook_with_terminal_native"
    INFO_MODE_ALIASES = {
        "terminal": HOOK_WITH_TERMINAL_NATIVE,
        "events": HOOK_WITH_TERMINAL_NATIVE,
    }
    INFO_MODES = frozenset({HOOK_WITH_TERMINAL_NATIVE, "full", *INFO_MODE_ALIASES})

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        hooks: FusedVectorHooks,
        *,
        info_mode: str = HOOK_WITH_TERMINAL_NATIVE,
    ):
        info_mode = self.INFO_MODE_ALIASES.get(info_mode, info_mode)
        if info_mode not in self.INFO_MODES:
            raise ValueError(
                "info_mode must be one of: hook_with_terminal_native, full "
                "(legacy aliases: terminal, events)"
            )
        self.env = env
        self.hooks = hooks
        self.info_mode = info_mode
        self.num_envs = int(env.num_envs)
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_count = 0
        self.started_at = time.time()
        self.native_step_seconds_total = 0.0
        self.native_step_calls_total = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs, infos = self.env.reset(seed=seed, options=options)
        info_view = VectorInfoView(infos, self.num_envs)
        self.reset_infos = [info_view.lane(index) for index in range(self.num_envs)]
        self.hooks.on_reset(info_view)
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)
        return obs

    def step(self, actions: Any) -> FusedStep:
        native_actions = self.hooks.map_actions(actions)
        step_started_at = time.perf_counter()
        obs, rewards, terminations, truncations, infos = self.env.step(native_actions)
        self.native_step_seconds_total += time.perf_counter() - step_started_at
        self.native_step_calls_total += 1

        terminations = np.asarray(terminations, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool)
        dones = np.logical_or(terminations, truncations)
        info_view = VectorInfoView(infos, self.num_envs)
        hook_step = self.hooks.shape_step(
            np.asarray(rewards, dtype=np.float32),
            terminations,
            truncations,
            info_view,
        )

        self.episode_returns += hook_step.rewards
        self.episode_lengths += 1
        sb3_infos = self._sb3_infos(info_view, hook_step.infos, dones, truncations)
        return FusedStep(obs, hook_step.rewards, terminations, truncations, dones, sb3_infos)

    def _sb3_infos(
        self,
        info_view: VectorInfoView,
        hook_infos: dict[int, dict[str, Any]],
        dones: np.ndarray,
        truncations: np.ndarray,
    ) -> list[dict[str, Any]]:
        infos: list[dict[str, Any]] = []
        for index in range(self.num_envs):
            if self.info_mode == "full":
                info = info_view.lane(index)
            else:
                info = dict(hook_infos.get(index, {}))
            if bool(dones[index]):
                terminal = info_view.terminal_info(index, truncated=bool(truncations[index]))
                terminal.update(info)
                terminal["episode"] = {
                    "r": self.episode_returns[index].copy(),
                    "l": int(self.episode_lengths[index]),
                    "t": round(time.time() - self.started_at, 6),
                }
                self.episode_count += 1
                self.episode_returns[index] = 0.0
                self.episode_lengths[index] = 0
                self.reset_infos[index] = dict(terminal.get("reset_info", {}))
                info = terminal
            infos.append(info)
        return infos

    def close(self):
        return self.env.close()

    def native_step_stats(self) -> dict[str, float | int]:
        return {
            "seconds_total": self.native_step_seconds_total,
            "calls_total": self.native_step_calls_total,
            "num_envs": self.num_envs,
        }


class Sb3FusedVecEnv(VecEnv):
    """SB3 VecEnv adapter for a fused Gymnasium vector pipeline."""

    def __init__(self, pipeline: FusedGymVectorPipeline):
        self.pipeline = pipeline
        self.env = pipeline.env
        self.waiting = False
        self._actions = None
        super().__init__(
            pipeline.num_envs,
            getattr(self.env, "single_observation_space", self.env.observation_space),
            pipeline.hooks.action_space,
        )

    def __getattr__(self, name: str) -> Any:
        if name in {"env", "pipeline"}:
            raise AttributeError(name)
        return getattr(self.env, name)

    @staticmethod
    def _pending_seed(seeds: list[int | None]) -> int | None:
        return next((int(seed) for seed in seeds if seed is not None), None)

    @staticmethod
    def _pending_options(options: list[dict[str, Any]]) -> dict[str, Any] | None:
        non_empty = [option for option in options if option]
        if not non_empty:
            return None
        if all(option == non_empty[0] for option in non_empty):
            return dict(non_empty[0])
        return {"options": list(options)}

    @property
    def reset_infos(self):
        return self.pipeline.reset_infos

    @reset_infos.setter
    def reset_infos(self, value):
        self.pipeline.reset_infos = value

    def reset(self):
        seed = self._pending_seed(getattr(self, "_seeds", []))
        options = self._pending_options(getattr(self, "_options", []))
        obs = self.pipeline.reset(seed=seed, options=options)
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions):
        self._actions = np.asarray(actions)
        self.waiting = True

    def step_wait(self):
        step = self.pipeline.step(self._actions)
        self._actions = None
        self.waiting = False
        return step.obs, step.rewards, step.dones, step.infos

    def native_step_stats(self) -> dict[str, float | int]:
        return self.pipeline.native_step_stats()

    def close(self):
        return self.pipeline.close()

    def get_images(self):
        if hasattr(self.env, "get_images"):
            return self.env.get_images()
        frame = self.env.render()
        if frame is None:
            return [None for _ in range(self.num_envs)]
        return [frame]

    def render(self, mode: str | None = None):
        if mode is not None:
            return self.env.render()
        return self.env.render()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        value = getattr(self.env, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        setattr(self.env, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        method = getattr(self.env, method_name)
        return [
            method(*method_args, **method_kwargs) for _ in self._get_indices(indices)
        ]

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]
