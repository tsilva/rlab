from __future__ import annotations

# ruff: noqa: E402

import inspect
import os
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import gymnasium as gym
import numpy as np
import stable_retro as retro
from stable_retro import RetroVecEnv
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, VecMonitor, VecTransposeImage

from rlab.env_registry import (
    STABLE_RETRO_TURBO_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER,
    qualify_env_id,
    resolve_env_provider,
)
from rlab.env_wrappers import resolve_configured_env_wrappers, with_default_env_wrapper_specs
from rlab.targets import GenericRetroTarget, target_for_game

GAME = os.environ.get("RETRO_GAME", "")
DEFAULT_STATE = os.environ.get("RETRO_STATE", "")
DEFAULT_OBS_RESIZE_ALGORITHM = "area"
DEFAULT_HUD_CROP_TOP = GenericRetroTarget.default_hud_crop_top
FRAME_STACK_CHANNELS = {1, 3, 4}
DoneOnInfoRule = tuple[str | tuple[str, ...], str]
DoneOnInfoRules = dict[str, DoneOnInfoRule]
NativeDoneOnRule = DoneOnInfoRule | None
NativeDoneOnRules = dict[str, NativeDoneOnRule]
InfoEventRule = DoneOnInfoRule
InfoEventRules = DoneOnInfoRules


def _super_mario_bros_nes_turbo_vec_env_type():
    try:
        from supermariobrosnes_turbo import SuperMarioBrosNesTurboVecEnv
    except ImportError as exc:
        raise ImportError(
            "supermariobrosnes-turbo provider requires "
            "supermariobrosnes-turbo==0.2.10",
        ) from exc
    return SuperMarioBrosNesTurboVecEnv


def _provider_vec_env_type(config: EnvConfig | None = None):
    provider_id = (config or EnvConfig()).env_provider
    provider = resolve_env_provider(provider_id)
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        return RetroVecEnv
    if provider.provider_id == SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        return _super_mario_bros_nes_turbo_vec_env_type()
    raise ValueError(f"unsupported environment provider {provider.provider_id!r}")


def native_vec_env_supports_done_on(config: EnvConfig | None = None) -> bool:
    try:
        signature = inspect.signature(_provider_vec_env_type(config).__init__)
    except (OSError, TypeError):
        return False
    return "done_on" in signature.parameters


def native_vec_env_supports_named_done_on(config: EnvConfig | None = None) -> bool:
    provider_id = (config or EnvConfig()).env_provider
    provider = resolve_env_provider(provider_id)
    if provider.provider_id == SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        return True
    return callable(getattr(_provider_vec_env_type(config), "resolve_info_event_rules", None))


def native_vec_env_supports_rgb_render(config: EnvConfig | None = None) -> bool:
    provider_id = (config or EnvConfig()).env_provider
    provider = resolve_env_provider(provider_id)
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        return True
    try:
        env_type = _provider_vec_env_type(config)
    except (ImportError, ValueError):
        return False
    metadata = getattr(env_type, "metadata", {})
    render_modes = metadata.get("render_modes", ()) if isinstance(metadata, Mapping) else ()
    return "rgb_array" in render_modes


def action_names_for_set(action_set: str, game: str = GAME) -> tuple[str, ...]:
    return target_for_game(game).action_names_for_set(action_set)


@dataclass(frozen=True)
class EnvConfig:
    env_provider: str = STABLE_RETRO_TURBO_PROVIDER.provider_id
    game: str = GAME
    state: str = DEFAULT_STATE
    states: tuple[str, ...] = ()
    state_probs: tuple[float, ...] = ()
    task_conditioning: bool = False
    task_conditioning_info_vars: tuple[str, ...] = ()
    task_conditioning_info_values: tuple[tuple[int | str, ...], ...] = ()
    frame_skip: int = 4
    max_pool_frames: bool = True
    sticky_action_prob: float = 0.0
    max_episode_steps: int = 4500
    observation_size: int = 84
    hud_crop_top: int = -1
    obs_crop: tuple[int, int, int, int] | None = None
    obs_crop_mode: str = "remove"
    obs_crop_fill: int = 0
    obs_resize_algorithm: str = DEFAULT_OBS_RESIZE_ALGORITHM
    use_retro_reward: bool = False
    clip_rewards: bool = False
    reward_mode: str = "auto"
    progress_reward_cap: float = 30.0
    progress_reward_scale: float = 1.0
    terminal_reward: float = 50.0
    reward_scale: float = 10.0
    time_penalty: float = 0.0
    death_penalty: float = 25.0
    completion_reward: float = 0.0
    score_progress_clipped: bool = False
    env_wrappers: tuple[dict[str, Any], ...] = ()
    no_progress_timeout_steps: int = 0
    no_progress_min_delta: int = 0
    info_events: InfoEventRules = field(default_factory=dict)
    done_on_events: tuple[str, ...] = ()
    action_set: str = "auto"
    env_threads: int = 0


def normalize_event_config(config: EnvConfig) -> EnvConfig:
    done_on_events = tuple(dict.fromkeys(str(item) for item in config.done_on_events))
    if done_on_events != config.done_on_events:
        return replace(config, done_on_events=done_on_events)
    return config


def validate_obs_crop(value: tuple[int, int, int, int] | list[int] | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError("obs_crop must be [top, right, bottom, left]")
    result: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ValueError(f"obs_crop[{index}] must be a non-negative integer")
        result.append(int(item))
    return tuple(result)  # type: ignore[return-value]


def validate_obs_crop_mode(value: str) -> str:
    if value not in {"remove", "mask"}:
        raise ValueError("obs_crop_mode must be 'remove' or 'mask'")
    return value


def validate_obs_crop_fill(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 255:
        raise ValueError("obs_crop_fill must be an integer in [0, 255]")
    return int(value)


def native_obs_crop(config: EnvConfig) -> tuple[int, int, int, int] | None:
    obs_crop = validate_obs_crop(config.obs_crop)
    if obs_crop is not None:
        return obs_crop if any(obs_crop) else None
    if config.hud_crop_top > 0:
        return (config.hud_crop_top, 0, 0, 0)
    return None


def rendered_preprocess_hud_crop_top(config: EnvConfig) -> int:
    obs_crop = validate_obs_crop(config.obs_crop)
    if obs_crop is None:
        return config.hud_crop_top
    top, right, bottom, left = obs_crop
    if right or bottom or left:
        raise ValueError(
            "rendered replay preprocessing only supports obs_crop top cropping; "
            f"got obs_crop={list(obs_crop)}",
        )
    return top


def resolve_env_config(config: EnvConfig) -> EnvConfig:
    if not config.game:
        raise ValueError("game is required; pass --game or set RETRO_GAME")
    qualify_env_id(config.env_provider, config.game)
    _validate_sticky_action_prob(config.sticky_action_prob)
    validate_obs_crop_mode(config.obs_crop_mode)
    validate_obs_crop_fill(config.obs_crop_fill)
    if config.task_conditioning_info_vars and not config.task_conditioning:
        raise ValueError("--task-conditioning-info-vars requires --task-conditioning")
    if config.task_conditioning_info_values and not config.task_conditioning_info_vars:
        raise ValueError(
            "--task-conditioning-info-values requires --task-conditioning-info-vars",
        )
    for value in config.task_conditioning_info_values:
        if len(value) != len(config.task_conditioning_info_vars):
            raise ValueError(
                "--task-conditioning-info-values row length must match "
                "--task-conditioning-info-vars",
            )
    target = target_for_game(config.game)
    updates: dict[str, Any] = {}
    if not config.state and target.default_state:
        updates["state"] = target.default_state
    if config.action_set == "auto":
        updates["action_set"] = target.default_action_set
    elif config.action_set not in target.action_sets and not target.action_sets:
        updates["action_set"] = target.default_action_set
    if config.reward_mode == "auto":
        updates["reward_mode"] = target.default_reward_mode
    if config.obs_crop is None and config.hud_crop_top < 0:
        updates["hud_crop_top"] = target.default_hud_crop_top
    if target.default_env_wrappers:
        updates["env_wrappers"] = with_default_env_wrapper_specs(
            updates.get("env_wrappers", config.env_wrappers),
            target.default_env_wrappers,
        )
    config = replace(config, **updates) if updates else config
    config = resolve_configured_env_wrappers(config)
    return normalize_event_config(config)


def _validate_state_names(game: str, states: tuple[str, ...]) -> None:
    if any(not state for state in states):
        raise ValueError("--states must not contain empty state names")
    valid_states = set(retro.data.list_states(game))
    unknown = [state for state in states if state not in valid_states]
    if unknown:
        valid_preview = ", ".join(sorted(valid_states)[:12])
        raise ValueError(
            "unknown stable-retro state(s) for "
            f"{game}: {', '.join(unknown)}. Known examples: {valid_preview}",
        )


def resolve_mixed_state_config(config: EnvConfig, n_envs: int) -> EnvConfig:
    config = resolve_env_config(config)
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    if not config.states:
        if config.state_probs:
            raise ValueError("--state-probs requires --states")
        return config

    _validate_state_names(config.game, config.states)
    if config.state_probs:
        if len(config.state_probs) != len(config.states):
            raise ValueError(
                "--state-probs count must match --states count "
                f"({len(config.state_probs)} != {len(config.states)})",
            )
        probs = np.asarray(config.state_probs, dtype=np.float64)
        if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
            raise ValueError(
                "--state-probs values must be non-negative finite numbers",
            )
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("--state-probs must have a positive finite sum")
        return config

    if len(config.states) != n_envs:
        raise ValueError(
            "--states without --state-probs must provide exactly one state per env slot: "
            f"got {len(config.states)} states for n_envs={n_envs}",
        )
    return config


def state_distribution_metadata(config: EnvConfig) -> list[dict[str, float | str]]:
    if not config.states:
        return []
    if config.state_probs:
        distribution: dict[str, float] = {}
        for state, prob in zip(config.states, config.state_probs, strict=True):
            distribution[state] = distribution.get(state, 0.0) + float(prob)
        total = float(sum(distribution.values()))
        return [
            {"state": state, "probability": probability / total}
            for state, probability in distribution.items()
        ]
    probability = 1.0 / len(config.states)
    return [{"state": state, "probability": probability} for state in config.states]


def state_weight_mapping(config: EnvConfig) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not config.states or not config.state_probs:
        return weights
    for state, weight in zip(config.states, config.state_probs, strict=True):
        weights[state] = weights.get(state, 0.0) + float(weight)
    return weights


def state_name_candidates_from_level_id(level_id: str) -> tuple[str, ...]:
    """Return possible Stable Retro state names for a target level_id annotation."""

    candidates = [f"Level{level_id}"]
    parts = level_id.split("-", 1)
    if len(parts) == 2:
        try:
            world = int(parts[0]) + 1
            stage = int(parts[1]) + 1
        except ValueError:
            pass
        else:
            candidates.append(f"Level{world}-{stage}")
    return tuple(dict.fromkeys(candidates))


def info_value_from_state_name(
    state_name: str,
    info_vars: tuple[str, ...],
) -> tuple[int | str, ...] | None:
    if tuple(info_vars) == ("levelHi", "levelLo") and state_name.startswith("Level"):
        level = state_name.removeprefix("Level").split("-", 2)
        if len(level) >= 2:
            try:
                return (int(level[0]) - 1, int(level[1]) - 1)
            except ValueError:
                return None
    return None


def task_conditioning_info_values(config: EnvConfig) -> tuple[tuple[int | str, ...], ...]:
    if not config.task_conditioning_info_vars:
        return ()
    if config.task_conditioning_info_values:
        return config.task_conditioning_info_values
    values: list[tuple[int | str, ...]] = []
    for state_name in dict.fromkeys(config.states or ((config.state,) if config.state else ())):
        value = info_value_from_state_name(state_name, config.task_conditioning_info_vars)
        if value is None:
            continue
        values.append(value)
    return tuple(values)


def retro_make_kwargs(config: EnvConfig) -> dict[str, Any]:
    return {"state": config.state} if config.state else {}


def _require_stable_retro_turbo_provider(config: EnvConfig):
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id != STABLE_RETRO_TURBO_PROVIDER.provider_id:
        raise ValueError(
            f"unsupported environment provider {provider.provider_id!r}; "
            f"supported providers: {STABLE_RETRO_TURBO_PROVIDER.provider_id}",
        )
    return provider


def _stable_retro_turbo_make_env(
    config: EnvConfig,
    *,
    render_mode: str,
    fast_observation_path: bool = False,
) -> gym.Env:
    _require_stable_retro_turbo_provider(config)
    kwargs: dict[str, Any] = {
                "render_mode": render_mode,
                **retro_make_kwargs(config),
    }
    if fast_observation_path:
        kwargs.update(
            {
                "obs_resize": (config.observation_size, config.observation_size),
                "obs_crop": native_obs_crop(config),
                "obs_grayscale": True,
                "obs_resize_algorithm": config.obs_resize_algorithm,
                "frame_skip": config.frame_skip,
                "frame_stack": 4,
                "maxpool_last_two": config.max_pool_frames,
            }
        )
    return retro.make(config.game, **kwargs)


class SingleLaneVecEnvAdapter(gym.Env):
    """Adapt a one-lane vector env to the Gymnasium API used by replay display."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, vec_env):
        super().__init__()
        if getattr(vec_env, "num_envs", 1) != 1:
            raise ValueError("SingleLaneVecEnvAdapter requires num_envs=1")
        self.vec_env = vec_env
        self.action_space = getattr(vec_env, "single_action_space", vec_env.action_space)
        self.observation_space = getattr(
            vec_env,
            "single_observation_space",
            vec_env.observation_space,
        )
        self.render_mode = getattr(vec_env, "render_mode", None)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if isinstance(self.vec_env, gym.vector.VectorEnv):
            obs, infos = self.vec_env.reset(seed=seed, options=options)
            info_list = vector_infos_to_list(infos, self.vec_env.num_envs)
            return np.asarray(obs)[0], dict(info_list[0]) if info_list else {}
        if seed is not None:
            self.vec_env.seed(seed)
        if options is not None and hasattr(self.vec_env, "set_options"):
            self.vec_env.set_options(options)
        obs = self.vec_env.reset()
        reset_infos = getattr(self.vec_env, "reset_infos", [{}])
        info = dict(reset_infos[0]) if reset_infos else {}
        return np.asarray(obs)[0], info

    def step(self, action: Any):
        action_batch = np.asarray(action)
        if action_batch.shape == getattr(self.action_space, "shape", None):
            action_batch = action_batch.reshape((1, *action_batch.shape))
        elif action_batch.ndim == 0:
            action_batch = action_batch.reshape((1,))
        if isinstance(self.vec_env, gym.vector.VectorEnv):
            obs, rewards, terminations, truncations, infos = self.vec_env.step(action_batch)
            info_list = vector_infos_to_list(infos, self.vec_env.num_envs)
            info = dict(info_list[0]) if info_list else {}
            terminated = bool(np.asarray(terminations)[0])
            truncated = bool(np.asarray(truncations)[0])
            step_obs = np.asarray(obs)[0]
            if terminated or truncated:
                final_info = info.pop("final_info", {})
                final_obs = info.pop("final_obs", None)
                reset_info = dict(info)
                info = dict(final_info) if isinstance(final_info, Mapping) else {}
                if reset_info:
                    info["reset_info"] = reset_info
                if final_obs is not None:
                    step_obs = final_obs
            return step_obs, float(np.asarray(rewards)[0]), terminated, truncated, info
        obs, rewards, dones, infos = self.vec_env.step(action_batch)
        info = dict(infos[0]) if infos else {}
        done = bool(np.asarray(dones)[0])
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = done and not truncated
        return np.asarray(obs)[0], float(np.asarray(rewards)[0]), terminated, truncated, info

    def render(self):
        return self.vec_env.render()

    def close(self):
        return self.vec_env.close()


def _visual_state_for_config(config: EnvConfig) -> Any:
    if config.state:
        return config.state
    if config.states:
        return config.states[0]
    return None


def _super_mario_bros_nes_turbo_make_env(
    config: EnvConfig,
    *,
    render_mode: str,
    fast_observation_path: bool = False,
) -> gym.Env:
    if fast_observation_path:
        raise ValueError("supermariobrosnes-turbo single-env display does not use fast_observation_path")
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id != SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        raise ValueError(
            f"unsupported environment provider {provider.provider_id!r}; "
            f"expected {SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id}",
        )
    if render_mode != "rgb_array":
        raise ValueError("supermariobrosnes-turbo visual replay requires render_mode='rgb_array'")
    kwargs: dict[str, Any] = {
        "num_envs": 1,
        "num_threads": 1,
        "render_mode": "rgb_array",
        "use_restricted_actions": "ALL",
        "frame_skip": 1,
        "frame_stack": 1,
        "maxpool_last_two": False,
        "sticky_action_prob": 0.0,
        "obs_grayscale": False,
        "obs_layout": "hwc",
    }
    state = _visual_state_for_config(config)
    if state is not None:
        kwargs["state"] = state
    return SingleLaneVecEnvAdapter(
        _super_mario_bros_nes_turbo_vec_env_type()(config.game, **kwargs)
    )


def make_provider_env(
    config: EnvConfig,
    *,
    render_mode: str,
    fast_observation_path: bool = False,
) -> gym.Env:
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        return _stable_retro_turbo_make_env(
            config,
            render_mode=render_mode,
            fast_observation_path=fast_observation_path,
        )
    if provider.provider_id == SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        return _super_mario_bros_nes_turbo_make_env(
            config,
            render_mode=render_mode,
            fast_observation_path=fast_observation_path,
        )
    raise ValueError(f"unsupported environment provider {provider.provider_id!r}")



class DiscreteRetroActions(gym.ActionWrapper):
    """Map a target-specific discrete action set to stable-retro controls."""

    def __init__(self, env: gym.Env, config: EnvConfig):
        super().__init__(env)
        target = target_for_game(config.game)
        self.action_names = target.action_names_for_set(config.action_set)
        self.actions = target.action_masks_for_set(config.action_set)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action: int) -> np.ndarray:
        return self.actions[int(action)].copy()


class VecDiscreteRetroActions(VecEnvWrapper):
    """Map target-specific discrete SB3 actions to stable-retro controls."""

    def __init__(self, venv, config: EnvConfig):
        target = target_for_game(config.game)
        self.action_names = target.action_names_for_set(config.action_set)
        self.actions = np.stack(target.action_masks_for_set(config.action_set)).astype(np.int8)
        super().__init__(
            venv,
            observation_space=venv.observation_space,
            action_space=gym.spaces.Discrete(len(self.actions)),
        )

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        action_indices = np.asarray(actions, dtype=np.int64).reshape(-1)
        self.venv.step_async(self.actions[action_indices])

    def step_wait(self):
        return self.venv.step_wait()


def _array_item(value: Any, index: int) -> Any:
    if isinstance(value, np.ndarray):
        item = value[index]
        if isinstance(item, np.generic):
            return item.item()
        return item
    if isinstance(value, (list, tuple)) and len(value) > index:
        return value[index]
    return value


def vector_infos_to_list(infos: Any, num_envs: int) -> list[dict[str, Any]]:
    """Convert Gymnasium vector info dictionaries to lane-wise dictionaries."""

    if isinstance(infos, (list, tuple)):
        return [dict(info or {}) for info in infos]
    if not isinstance(infos, Mapping):
        return [{} for _ in range(num_envs)]

    result = [{} for _ in range(num_envs)]
    masks: dict[str, np.ndarray] = {}
    for key, value in infos.items():
        if isinstance(key, str) and key.startswith("_"):
            masks[key[1:]] = np.asarray(value, dtype=bool)

    for key, value in infos.items():
        if not isinstance(key, str) or key.startswith("_"):
            continue
        mask = masks.get(key)
        if isinstance(value, Mapping):
            values = vector_infos_to_list(value, num_envs)
        else:
            values = [_array_item(value, index) for index in range(num_envs)]
        for index in range(num_envs):
            if mask is not None and index < len(mask) and not bool(mask[index]):
                continue
            result[index][key] = values[index]
    return result


class GymVectorEnvToSb3VecEnv(VecEnv):
    """Adapt Gymnasium VectorEnv output to the SB3 VecEnv contract used by PPO."""

    def __init__(self, env: gym.vector.VectorEnv):
        self.env = env
        self.waiting = False
        self._actions = None
        super().__init__(
            int(env.num_envs),
            getattr(env, "single_observation_space", env.observation_space),
            getattr(env, "single_action_space", env.action_space),
        )

    def __getattr__(self, name: str) -> Any:
        if name == "env":
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

    def reset(self):
        seed = self._pending_seed(getattr(self, "_seeds", []))
        options = self._pending_options(getattr(self, "_options", []))
        obs, infos = self.env.reset(seed=seed, options=options)
        self.reset_infos = vector_infos_to_list(infos, self.num_envs)
        self._reset_seeds()
        self._reset_options()
        return obs

    def step_async(self, actions):
        self._actions = actions
        self.waiting = True

    def step_wait(self):
        obs, rewards, terminations, truncations, infos = self.env.step(self._actions)
        self._actions = None
        self.waiting = False
        terminations = np.asarray(terminations, dtype=bool)
        truncations = np.asarray(truncations, dtype=bool)
        dones = np.logical_or(terminations, truncations)
        lane_infos = vector_infos_to_list(infos, self.num_envs)
        sb3_infos: list[dict[str, Any]] = []
        for index, raw_info in enumerate(lane_infos):
            info = dict(raw_info)
            if bool(dones[index]):
                final_info = info.pop("final_info", {})
                final_obs = info.pop("final_obs", None)
                reset_info = dict(info)
                info = dict(final_info) if isinstance(final_info, Mapping) else {}
                if final_obs is not None:
                    info["terminal_observation"] = final_obs
                info["reset_info"] = reset_info
                info["TimeLimit.truncated"] = bool(truncations[index])
                self.reset_infos[index] = reset_info
            sb3_infos.append(info)
        return (
            obs,
            np.asarray(rewards, dtype=np.float32),
            dones,
            sb3_infos,
        )

    def close(self):
        return self.env.close()

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


def adapt_provider_vec_env_for_sb3(vec_env):
    if isinstance(vec_env, VecEnv):
        return vec_env
    if isinstance(vec_env, gym.vector.VectorEnv):
        return GymVectorEnvToSb3VecEnv(vec_env)
    return vec_env


def _validate_sticky_action_prob(sticky_action_prob: float) -> float:
    sticky_action_prob = float(sticky_action_prob)
    if not 0.0 <= sticky_action_prob <= 1.0:
        raise ValueError("sticky_action_prob must be between 0.0 and 1.0")
    return sticky_action_prob


def _copy_action(action: Any) -> Any:
    if isinstance(action, np.ndarray):
        return action.copy()
    return action


class StickyAction(gym.Wrapper):
    """Repeat the previous high-level action with a fixed probability."""

    def __init__(self, env: gym.Env, sticky_action_prob: float):
        super().__init__(env)
        self.sticky_action_prob = _validate_sticky_action_prob(sticky_action_prob)
        self.rng = np.random.default_rng()
        self.last_action: Any | None = None

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.last_action = None
        return self.env.reset(**kwargs)

    def step(self, action: Any):
        if (
            self.last_action is not None
            and self.sticky_action_prob > 0.0
            and self.rng.random() < self.sticky_action_prob
        ):
            effective_action = _copy_action(self.last_action)
        else:
            effective_action = _copy_action(action)
        self.last_action = _copy_action(effective_action)
        return self.env.step(effective_action)


def _find_vec_attr(venv, attr_name: str) -> Any:
    current = venv
    while current is not None:
        if attr_name in vars(current) or hasattr(type(current), attr_name):
            return getattr(current, attr_name)
        current = getattr(current, "venv", None)
    raise AttributeError(attr_name)


class VecTaskConditioning(VecEnvWrapper):
    """Expose active task as a one-hot vector in dict observations."""

    def __init__(self, venv, config: EnvConfig | None = None):
        config = config or EnvConfig()
        try:
            initial_state_names = _find_vec_attr(venv, "initial_state_names")
            active_state_indices = _find_vec_attr(venv, "active_state_indices")
        except AttributeError as exc:
            raise ValueError(
                "task conditioning requires stable-retro-turbo active-state support "
                "(initial_state_names and active_state_indices), available in post19+",
            ) from exc
        self._initial_state_names = tuple(initial_state_names)
        if not self._initial_state_names:
            raise ValueError("task conditioning requires at least one native initial state")
        self._info_vars = tuple(config.task_conditioning_info_vars)
        self._info_values = task_conditioning_info_values(config)
        if self._info_vars and not self._info_values:
            raise ValueError(
                "info-var task conditioning requires --task-conditioning-info-values or "
                "state names that can derive those values",
            )
        self._active_state_indices = active_state_indices()
        slot_to_task: list[int] = []
        if self._info_vars:
            info_value_to_task = {value: index for index, value in enumerate(self._info_values)}
            for state_name in self._initial_state_names:
                info_value = info_value_from_state_name(state_name, self._info_vars)
                if info_value is None or info_value not in info_value_to_task:
                    raise ValueError(
                        f"initial state {state_name!r} cannot map to configured "
                        f"task-conditioning info values {self._info_values!r}",
                    )
                slot_to_task.append(info_value_to_task[info_value])
            self.task_state_names = tuple(
                ",".join(str(part) for part in value) for value in self._info_values
            )
            self._task_index_by_state_name: dict[str, int] = {}
            self._task_index_by_info_value = info_value_to_task
        else:
            task_index_by_name: dict[str, int] = {}
            for state_name in self._initial_state_names:
                slot_to_task.append(
                    task_index_by_name.setdefault(state_name, len(task_index_by_name))
                )
            self.task_state_names = tuple(task_index_by_name)
            self._task_index_by_state_name = task_index_by_name
            self._task_index_by_info_value = {}
        self._slot_to_task = np.asarray(slot_to_task, dtype=np.int64)
        self._active_task_indices = self._slot_to_task[self._active_state_indices]
        self._task_eye = np.eye(len(self.task_state_names), dtype=np.float32)
        observation_space = gym.spaces.Dict(
            {
                "image": venv.observation_space,
                "task": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(len(self.task_state_names),),
                    dtype=np.float32,
                ),
            }
        )
        super().__init__(
            venv,
            observation_space=observation_space,
            action_space=venv.action_space,
        )

    @property
    def initial_state_names(self) -> tuple[str, ...]:
        return self._initial_state_names

    def active_state_indices(self) -> np.ndarray:
        return self._active_state_indices

    def _task_indices(self, active_indices: np.ndarray | None = None) -> np.ndarray:
        if active_indices is None:
            return self._active_task_indices
        return self._slot_to_task[np.asarray(active_indices, dtype=np.int64)]

    def _task_vectors(self, active_indices: np.ndarray | None = None) -> np.ndarray:
        return self._task_eye[self._task_indices(active_indices)]

    def _task_vector_from_task_index(self, task_index: int) -> np.ndarray:
        return self._task_eye[int(task_index)]

    def _task_index_from_info(self, info: dict[str, Any]) -> int | None:
        if self._info_vars:
            try:
                value = tuple(info[var] for var in self._info_vars)
            except KeyError:
                return None
            return self._task_index_by_info_value.get(value)

        level_id = info.get("level_id")
        if not isinstance(level_id, str) or not level_id:
            return None
        for state_name in state_name_candidates_from_level_id(level_id):
            task_index = self._task_index_by_state_name.get(state_name)
            if task_index is not None:
                return task_index
        return None

    def _observation(self, image_obs, active_indices: np.ndarray | None = None) -> dict[str, np.ndarray]:
        return {
            "image": image_obs,
            "task": self._task_vectors(active_indices),
        }

    def reset(self):
        obs = self.venv.reset()
        self._active_task_indices = self._slot_to_task[self._active_state_indices]
        return self._observation(obs)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        previous_task_indices = self._active_task_indices.copy()
        obs, rewards, dones, infos = self.venv.step_wait()
        reset_task_indices = self._slot_to_task[self._active_state_indices]
        for index, done in enumerate(dones):
            if done:
                if "terminal_observation" in infos[index]:
                    infos[index]["terminal_observation"] = {
                        "image": infos[index]["terminal_observation"],
                        "task": self._task_vector_from_task_index(previous_task_indices[index]),
                    }
                self._active_task_indices[index] = reset_task_indices[index]
                continue
            next_task_index = self._task_index_from_info(infos[index])
            self._active_task_indices[index] = (
                reset_task_indices[index] if next_task_index is None else next_task_index
            )
        return self._observation(obs), rewards, dones, infos


class VecRetroProgressInfo(VecEnvWrapper):
    """Vectorized target reward shaping and progress metrics.

    Image preprocessing, frame skip, frame stacking, and max-pooling stay inside
    RetroVecEnv. This wrapper only rewrites rewards and annotates info.
    """

    def __init__(self, venv, config: EnvConfig):
        super().__init__(venv)
        self.config = config
        target = target_for_game(config.game)
        tracker_config = replace(
            config,
            max_episode_steps=0,
            no_progress_timeout_steps=0,
        )
        self.trackers = [target.create_tracker(tracker_config) for _ in range(self.num_envs)]
        self.previous_event_values: list[dict[str, Any]] = [
            {} for _ in range(self.num_envs)
        ]

    def reset(self):
        obs = self.venv.reset()
        self._reset_tracking(range(self.num_envs), getattr(self.venv, "reset_infos", None))
        return obs

    def _reset_tracking(self, indices, infos=None) -> None:
        infos = infos or [{} for _ in range(self.num_envs)]
        for index in indices:
            info = infos[index] if index < len(infos) else {}
            self.trackers[index].reset(info)
            self.previous_event_values[index] = self.event_values(info)

    def event_values(self, info: Mapping[str, Any]) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, rule in self.config.info_events.items():
            key_or_keys, _op = rule
            value = self.info_value_for_keys(info, key_or_keys)
            if value is not None:
                values[name] = value
        return values

    @staticmethod
    def info_value_for_keys(
        info: Mapping[str, Any],
        keys: str | tuple[str, ...],
    ) -> Any | None:
        if isinstance(keys, str):
            return info.get(keys)
        values = []
        for key in keys:
            if key not in info:
                return None
            values.append(info[key])
        return tuple(values)

    @staticmethod
    def event_rule_fired(previous: Any, current: Any, op: str) -> bool:
        if previous is None or current is None:
            return False
        if op == "change":
            return current != previous
        if op == "increase":
            return current > previous
        if op == "decrease":
            return current < previous
        return False

    @staticmethod
    def native_event_payloads(info: Mapping[str, Any]) -> dict[str, Any]:
        done_on_info = info.get("done_on_info")
        if isinstance(done_on_info, dict):
            return {str(name): payload for name, payload in done_on_info.items() if str(name)}
        if isinstance(done_on_info, (list, tuple, set)):
            return {str(name): {} for name in done_on_info if str(name)}
        if isinstance(done_on_info, str) and done_on_info:
            return {done_on_info: {}}
        return {}

    def annotate_info_events(self, index: int, info: dict[str, Any]) -> None:
        event_payloads = self.native_event_payloads(info)
        previous_values = self.previous_event_values[index]
        current_values = self.event_values(info)

        for name, rule in self.config.info_events.items():
            if name in event_payloads:
                continue
            if name not in previous_values or name not in current_values:
                continue
            key_or_keys, op = rule
            previous = previous_values[name]
            current = current_values[name]
            if not self.event_rule_fired(previous, current, op):
                continue
            event_payloads[name] = {
                "op": op,
                "keys": key_or_keys,
                "prev": previous,
                "next": current,
            }

        if event_payloads:
            info["info_events"] = event_payloads
        self.previous_event_values[index].update(current_values)

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)
        infos = [dict(info) for info in infos]
        shaped_rewards = np.zeros(self.num_envs, dtype=np.float32)

        for index, info in enumerate(infos):
            self.annotate_info_events(index, info)
            progress = self.trackers[index].step(rewards[index], info, dones[index])
            shaped_rewards[index] = progress.reward

        if self.config.clip_rewards:
            shaped_rewards = np.sign(shaped_rewards).astype(np.float32)

        native_done_indices = [idx for idx, done in enumerate(dones) if done]
        if native_done_indices:
            reset_infos = [{} for _ in range(self.num_envs)]
            for idx in native_done_indices:
                reset_info = infos[idx].get("reset_info")
                if isinstance(reset_info, dict):
                    reset_infos[idx] = reset_info
            self._reset_tracking(native_done_indices, reset_infos)

        return obs, shaped_rewards, dones, infos


class FrameSkip(gym.Wrapper):
    """Repeat one action for several emulator frames and sum reward."""

    def __init__(self, env: gym.Env, skip: int, max_pool: bool = False):
        super().__init__(env)
        if skip < 1:
            raise ValueError("frame_skip must be >= 1")
        self.skip = skip
        self.max_pool = max_pool

    def step(self, action: Any):
        total_reward = 0.0
        final_obs = None
        pooled_obs: list[np.ndarray] = []
        final_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for step_idx in range(self.skip):
            final_obs, reward, terminated, truncated, final_info = self.env.step(action)
            total_reward += float(reward)
            if self.max_pool and step_idx >= self.skip - 2 and final_obs is not None:
                pooled_obs.append(final_obs)
            if terminated or truncated:
                break
        if self.max_pool and pooled_obs:
            final_obs = np.maximum.reduce(pooled_obs)
        return final_obs, total_reward, terminated, truncated, final_info


class RetroProgressInfo(gym.Wrapper):
    """Apply target-specific reward shaping and progress metrics."""

    def __init__(self, env: gym.Env, config: EnvConfig):
        super().__init__(env)
        if config.reward_mode not in {"baseline", "bounded", "additive", "score", "native"}:
            raise ValueError(
                "reward_mode must be 'baseline', 'bounded', 'additive', 'score', or 'native'"
            )
        if config.progress_reward_cap < 0:
            raise ValueError("progress_reward_cap must be >= 0")
        if config.terminal_reward < 0:
            raise ValueError("terminal_reward must be >= 0")
        if config.reward_scale < 0:
            raise ValueError("reward_scale must be >= 0")
        if config.no_progress_timeout_steps < 0:
            raise ValueError("no_progress_timeout_steps must be >= 0")
        if config.no_progress_min_delta < 0:
            raise ValueError("no_progress_min_delta must be >= 0")
        self.config = replace(config, max_episode_steps=0)
        self.tracker = target_for_game(config.game).create_tracker(self.config)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.tracker.reset(info)
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        progress = self.tracker.step(reward, info, terminated or truncated)
        terminated = terminated or progress.terminal
        truncated = truncated or progress.truncated
        return obs, progress.reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    def reward(self, reward: float) -> float:
        return float(np.sign(reward))


class RetroPreprocess(gym.ObservationWrapper):
    """Crop optional HUD rows, then convert RGB frames to grayscale observations."""

    def __init__(self, env: gym.Env, size: int = 84, hud_crop_top: int = 0):
        super().__init__(env)
        if hud_crop_top < 0:
            raise ValueError("hud_crop_top must be >= 0")
        self.size = size
        self.hud_crop_top = hud_crop_top
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(size, size, 1),
            dtype=np.uint8,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        import cv2

        if self.hud_crop_top >= observation.shape[0]:
            raise ValueError(
                f"hud_crop_top={self.hud_crop_top} must be less than frame height {observation.shape[0]}",
            )
        frame = observation[self.hud_crop_top :, :, :]
        gray = np.dot(frame[..., :3], np.array([0.299, 0.587, 0.114])).astype(np.uint8)
        resized = cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return resized[..., None]


def make_retro_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = resolve_env_config(config or EnvConfig())
    env = make_provider_env(config, render_mode="rgb_array")
    return wrap_retro_env(env, config=config, seed=seed)


def make_fast_retro_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = resolve_env_config(config or EnvConfig())
    target = target_for_game(config.game)
    env = make_provider_env(
        config,
        render_mode="rgb_array",
        fast_observation_path=True,
    )
    if target.uses_discrete_actions(config.action_set):
        env = DiscreteRetroActions(env, config=config)
    if config.sticky_action_prob > 0.0:
        env = StickyAction(env, config.sticky_action_prob)
    env = RetroProgressInfo(env, config=config)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    if config.clip_rewards:
        env = ClipRewardEnv(env)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_visual_replay_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    config = resolve_env_config(config or EnvConfig())
    target = target_for_game(config.game)
    env = make_provider_env(config, render_mode="rgb_array")
    if target.uses_discrete_actions(config.action_set):
        env = DiscreteRetroActions(env, config=config)
    env = FrameSkip(env, config.frame_skip, max_pool=False)
    if config.sticky_action_prob > 0.0:
        env = StickyAction(env, config.sticky_action_prob)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def wrap_retro_env(env: gym.Env, config: EnvConfig, seed: int | None = None) -> gym.Env:
    config = resolve_env_config(config)
    target = target_for_game(config.game)
    if target.uses_discrete_actions(config.action_set):
        env = DiscreteRetroActions(env, config=config)
    env = FrameSkip(env, config.frame_skip, max_pool=config.max_pool_frames)
    if config.sticky_action_prob > 0.0:
        env = StickyAction(env, config.sticky_action_prob)
    env = RetroProgressInfo(env, config=config)
    env = RetroPreprocess(
        env,
        config.observation_size,
        hud_crop_top=rendered_preprocess_hud_crop_top(config),
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=config.max_episode_steps)
    if config.clip_rewards:
        env = ClipRewardEnv(env)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def needs_vec_transpose_image(observation_space: gym.Space) -> bool:
    """Return whether SB3 needs VecTransposeImage to receive channel-first images."""

    if isinstance(observation_space, gym.spaces.Dict):
        transpose = False
        for key, space in observation_space.spaces.items():
            if key == "image":
                transpose = transpose or needs_vec_transpose_image(space)
        return transpose

    shape = getattr(observation_space, "shape", None)
    if not isinstance(observation_space, gym.spaces.Box) or shape is None or len(shape) != 3:
        raise ValueError(
            "expected image observation_space with shape (H, W, C) or (C, H, W), "
            f"got {observation_space!r}",
        )

    channels_first = (
        int(shape[0]) in FRAME_STACK_CHANNELS and int(shape[-1]) not in FRAME_STACK_CHANNELS
    )
    channels_last = (
        int(shape[-1]) in FRAME_STACK_CHANNELS and int(shape[0]) not in FRAME_STACK_CHANNELS
    )
    if channels_first:
        return False
    if channels_last:
        return True
    raise ValueError(
        "could not infer observation channel order from shape "
        f"{tuple(int(dim) for dim in shape)}; expected channel count in first or last axis",
    )


def maybe_transpose_vec_image(vec_env):
    if needs_vec_transpose_image(vec_env.observation_space):
        return VecTransposeImage(vec_env)
    return vec_env


def _native_done_on_rules(
    config: EnvConfig,
    *,
    done_on_supported: bool,
    named_done_on_supported: bool,
) -> NativeDoneOnRules:
    native_rules = {
        name: config.info_events.get(name)
        for name in config.done_on_events
    }
    if native_rules and not done_on_supported:
        raise RuntimeError(
            "configured done_on rules require stable-retro-turbo with native "
            "done_on support",
        )
    missing_rule_names = [name for name, rule in native_rules.items() if rule is None]
    if missing_rule_names and not named_done_on_supported:
        raise RuntimeError(
            "configured named done_on events require stable-retro-turbo with "
            "metadata-backed named event support; unresolved event(s): "
            f"{', '.join(missing_rule_names)}",
        )
    return native_rules


def _native_vec_kwargs(
    config: EnvConfig,
    *,
    n_envs: int,
    num_threads: int,
    native_done_on_rules: NativeDoneOnRules,
) -> dict[str, Any]:
    native_kwargs: dict[str, Any] = {
        "num_envs": n_envs,
        "num_threads": num_threads,
        "render_mode": "rgb_array",
        "obs_resize": (config.observation_size, config.observation_size),
        "obs_crop": native_obs_crop(config),
        "obs_grayscale": True,
        "obs_resize_algorithm": config.obs_resize_algorithm,
        "frame_skip": config.frame_skip,
        "frame_stack": 4,
        "maxpool_last_two": config.max_pool_frames,
        "sticky_action_prob": config.sticky_action_prob,
        "obs_copy": "safe_view",
        "obs_layout": "chw",
    }
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id == SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        native_kwargs.update(
            {
                "obs_crop_mode": config.obs_crop_mode,
                "obs_crop_fill": config.obs_crop_fill,
            }
        )
    if config.states:
        native_kwargs["state"] = (
            state_weight_mapping(config)
            if config.state_probs
            else list(config.states)
        )
    elif config.state:
        native_kwargs["state"] = config.state
    else:
        native_kwargs["state"] = None
    if native_done_on_rules:
        native_kwargs["done_on"] = native_done_on_rules
    return native_kwargs


def _stable_retro_turbo_make_vec_env(
    config: EnvConfig,
    *,
    native_kwargs: Mapping[str, Any],
):
    _require_stable_retro_turbo_provider(config)
    return RetroVecEnv(config.game, **dict(native_kwargs))


def _super_mario_bros_nes_turbo_make_vec_env(
    config: EnvConfig,
    *,
    native_kwargs: Mapping[str, Any],
):
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id != SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        raise ValueError(
            f"unsupported environment provider {provider.provider_id!r}; "
            f"expected {SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id}",
        )
    return _super_mario_bros_nes_turbo_vec_env_type()(config.game, **dict(native_kwargs))


def make_provider_vec_env(
    config: EnvConfig,
    *,
    native_kwargs: Mapping[str, Any],
):
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        return _stable_retro_turbo_make_vec_env(config, native_kwargs=native_kwargs)
    if provider.provider_id == SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        return _super_mario_bros_nes_turbo_make_vec_env(config, native_kwargs=native_kwargs)
    raise ValueError(f"unsupported environment provider {provider.provider_id!r}")


def make_vec_envs(config: EnvConfig, n_envs: int, seed: int, start_method: str = "fork"):
    os.environ.setdefault("STABLE_RETRO_DISABLE_AUDIO", "1")
    config = resolve_mixed_state_config(config, n_envs=n_envs)
    target = target_for_game(config.game)
    num_threads = config.env_threads if config.env_threads > 0 else min(max(n_envs, 1), 16)
    native_done_on_rules = _native_done_on_rules(
        config,
        done_on_supported=native_vec_env_supports_done_on(config),
        named_done_on_supported=native_vec_env_supports_named_done_on(config),
    )
    native_kwargs = _native_vec_kwargs(
        config,
        n_envs=n_envs,
        num_threads=num_threads,
        native_done_on_rules=native_done_on_rules,
    )
    vec_env = make_provider_vec_env(config, native_kwargs=native_kwargs)
    vec_env = adapt_provider_vec_env_for_sb3(vec_env)
    vec_env.seed(seed)
    if target.uses_discrete_actions(config.action_set):
        vec_env = VecDiscreteRetroActions(vec_env, config=config)
    vec_env = VecRetroProgressInfo(vec_env, config=config)
    vec_env = VecMonitor(vec_env)
    if config.task_conditioning:
        vec_env = VecTaskConditioning(vec_env, config=config)
    return maybe_transpose_vec_image(vec_env)


def make_training_vec_env(config: EnvConfig, n_envs: int, seed: int, start_method: str = "fork"):
    return make_vec_envs(config=config, n_envs=n_envs, seed=seed, start_method=start_method)


def make_eval_vec_env(config: EnvConfig, n_envs: int, seed: int, start_method: str = "fork"):
    eval_config = resolve_env_config(config)
    return make_vec_envs(config=eval_config, n_envs=n_envs, seed=seed, start_method=start_method)


def make_rendered_replay_env(config: EnvConfig | None = None, seed: int | None = None) -> gym.Env:
    eval_config = resolve_env_config(config or EnvConfig())
    return make_retro_env(config=eval_config, seed=seed)


def assert_rom_imported(game: str = GAME) -> str:
    if not game:
        raise ValueError("game is required; pass --game or set RETRO_GAME")
    try:
        return retro.data.get_romfile_path(game)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{game} is not imported in this rlab runtime. "
            f"Run: rlab import-roms ~/Desktop/roms --game {game}",
        ) from exc




def default_run_dir(run_name: str, runs_dir: str = "runs") -> str:
    return os.path.join(runs_dir, run_name)
