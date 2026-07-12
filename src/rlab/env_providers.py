from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import gymnasium as gym

from gymnasium.vector import AutoresetMode
from stable_retro import RetroVecEnv as DEFAULT_RETRO_VEC_ENV

from rlab.batch_runtime import ProviderDescriptor, SignalSpec
from rlab.env_registry import (
    ALE_PY_PROVIDER,
    GYMNASIUM_PROVIDER,
    STABLE_RETRO_TURBO_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER,
    resolve_env_provider,
)


def _is_disabled_autoreset_mode(value: Any) -> bool:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name == "DISABLED":
        return True
    raw_value = getattr(value, "value", value)
    normalized = "".join(char for char in str(raw_value).lower() if char.isalnum())
    return normalized == "disabled"


def _disabled_autoreset_kwargs(native_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = dict(native_kwargs)
    kwargs["autoreset_mode"] = AutoresetMode.DISABLED
    return kwargs


def _require_disabled_autoreset_mode(env: Any, provider_id: str):
    mode = getattr(env, "autoreset_mode", None)
    metadata = getattr(env, "metadata", None)
    if mode is None and isinstance(metadata, Mapping):
        mode = metadata.get("autoreset_mode")
    if mode is None:
        raise RuntimeError(
            f"{provider_id} vector env does not advertise disabled autoreset; "
            "install a provider release with autoreset_mode=DISABLED and masked reset",
        )
    if not _is_disabled_autoreset_mode(mode):
        raise RuntimeError(
            f"{provider_id} vector env must support disabled autoreset and masked reset; "
            f"got autoreset_mode={mode!r}",
        )
    return env


class _StartInfoAdapter:
    """Add canonical reset start identities from a provider's native tracker."""

    def __init__(self, env: Any):
        self.env = env

    def __getattr__(self, name: str) -> Any:
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        native_options = dict(options or {})
        start_ids = native_options.pop("start_ids", None)
        if start_ids is not None:
            values = getattr(self.env, "initial_state_names", ())
            values = values() if callable(values) else values
            catalog = tuple(str(value) for value in values or ())
            catalog_indices = {name: index for index, name in enumerate(catalog)}
            start_indices = np.full(self.env.num_envs, -1, dtype=np.int32)
            for lane, start_id in enumerate(np.asarray(start_ids, dtype=object)):
                if start_id is None:
                    continue
                try:
                    start_indices[lane] = catalog_indices[str(start_id)]
                except KeyError as exc:
                    raise ValueError(
                        f"unknown provider start id {start_id!r}; expected one of {catalog}"
                    ) from exc
            native_options["start_indices"] = start_indices
        observations, infos = self.env.reset(seed=seed, options=native_options)
        if not isinstance(infos, Mapping):
            return observations, infos
        active_states = getattr(self.env, "active_states", None)
        if not callable(active_states):
            return observations, infos
        mask = np.ones(self.env.num_envs, dtype=np.bool_)
        if native_options.get("reset_mask") is not None:
            mask = np.asarray(native_options["reset_mask"], dtype=np.bool_)
        result = dict(infos)
        result["start_id"] = np.asarray(active_states(), dtype=object)
        result["_start_id"] = mask.copy()
        return observations, result

    def step(self, actions):
        return self.env.step(actions)

    def get_images(self):
        native = getattr(self.env, "native", None)
        get_screen = getattr(native, "get_screen", None)
        if callable(get_screen):
            return [np.asarray(get_screen(lane)) for lane in range(self.env.num_envs)]
        get_images = getattr(self.env, "get_images", None)
        if callable(get_images):
            return get_images()
        return self.env.render()

    def close(self):
        return self.env.close()


def super_mario_bros_nes_turbo_vec_env_type():
    try:
        from supermariobrosnes_turbo import SuperMarioBrosNesTurboVecEnv
    except ImportError as exc:
        raise ImportError(
            "supermariobrosnes-turbo provider requires supermariobrosnes-turbo",
        ) from exc
    return SuperMarioBrosNesTurboVecEnv


def ale_py_atari_vector_env_type():
    try:
        from ale_py.vector_env import AtariVectorEnv
    except ImportError as exc:
        raise ImportError(
            "ale-py provider requires ale-py with native vector env support",
        ) from exc
    return AtariVectorEnv


def provider_native_vec_kwargs(
    config: Any,
    *,
    n_envs: int,
    native_obs_crop: Callable[[Any], tuple[int, int, int, int] | None],
    state_weight_mapping: Callable[[Any], dict[str, float]],
) -> dict[str, Any]:
    """Compile provider mechanics without task events or termination rules."""
    native_kwargs = dict(config.env_args or {})
    done_on = native_kwargs.pop("done_on", None)
    if done_on not in (None, (), [], {}):
        raise ValueError(
            "provider task detectors are unsupported; configure task events and termination"
        )
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id == GYMNASIUM_PROVIDER.provider_id:
        if config.state or config.states or config.state_probs:
            raise ValueError(
                f"{provider.provider_id} provider does not support state, states, or state_probs"
            )
        native_kwargs.setdefault("num_envs", n_envs)
        return native_kwargs
    if provider.provider_id == ALE_PY_PROVIDER.provider_id:
        if config.state or config.states or config.state_probs:
            raise ValueError("ale-py provider does not support state, states, or state_probs")
        obs_crop = native_obs_crop(config)
        if obs_crop is not None and config.obs_crop_mode != "mask":
            raise ValueError("ale-py provider only supports obs_crop_mode='mask'")
        defaults = {
            "num_envs": n_envs,
            "max_num_frames_per_episode": 108_000,
            "repeat_action_probability": config.sticky_action_prob,
            "img_height": config.observation_size,
            "img_width": config.observation_size,
            "grayscale": True,
            "stack_num": 4,
            "frameskip": config.frame_skip,
            "maxpool": config.max_pool_frames,
            "episodic_life": False,
            "reward_clipping": False,
        }
        defaults.update(native_kwargs)
        return defaults

    defaults = {
        "num_envs": n_envs,
        "render_mode": "rgb_array",
        "obs_resize": (config.observation_size, config.observation_size),
        "obs_crop": native_obs_crop(config),
        "obs_crop_mode": config.obs_crop_mode,
        "obs_crop_fill": config.obs_crop_fill,
        "obs_grayscale": True,
        "obs_resize_algorithm": config.obs_resize_algorithm,
        "frame_skip": config.frame_skip,
        "frame_stack": 4,
        "maxpool_last_two": config.max_pool_frames,
        "sticky_action_prob": config.sticky_action_prob,
        "obs_copy": "safe_view",
        "obs_layout": "chw",
    }
    if config.states:
        defaults["state"] = (
            state_weight_mapping(config) if config.state_probs else list(config.states)
        )
    else:
        defaults["state"] = config.state or None
    defaults.update(native_kwargs)
    task = config.task if isinstance(getattr(config, "task", None), Mapping) else {}
    if task.get("id") == "mario":
        signals = task.get("signals", {})
        required_info_keys = {"time"}
        if isinstance(signals, Mapping):
            for source in signals.values():
                required_info_keys.update((source,) if isinstance(source, str) else source)
        configured_filter = defaults.get("info_filter")
        if configured_filter is None:
            defaults["info_filter"] = {
                "mode": "all",
                "keys": tuple(sorted(str(key) for key in required_info_keys)),
            }
        elif isinstance(configured_filter, Mapping):
            mode = str(configured_filter.get("mode", "all"))
            keys = configured_filter.get("keys")
            configured_keys = set(str(key) for key in keys) if keys is not None else set()
            missing = required_info_keys - configured_keys if keys is not None else set()
            if mode != "all" or missing:
                raise ValueError(
                    "Mario task signals require info_filter mode='all' and keys "
                    f"{sorted(required_info_keys)}"
                )
        elif str(configured_filter) != "all":
            raise ValueError("Mario task signals require info_filter='all'")
    return defaults


def provider_descriptor(
    config: Any,
    native_env: Any,
    *,
    state_weight_mapping: Callable[[Any], dict[str, float]],
) -> ProviderDescriptor:
    """Describe the provider-facing contract after native construction."""

    provider = resolve_env_provider(config.env_provider)
    observation_space = getattr(
        native_env,
        "single_observation_space",
        native_env.observation_space,
    )
    action_space = getattr(native_env, "single_action_space", native_env.action_space)
    signal_schema: dict[str, SignalSpec] = {}
    raw_signal_schema = getattr(native_env, "signal_schema", {})
    if isinstance(raw_signal_schema, Mapping):
        for name, raw_spec in raw_signal_schema.items():
            if isinstance(raw_spec, Mapping):
                dtype = raw_spec.get("dtype", np.float32)
                shape = raw_spec.get("shape", ())
                available_on_reset = bool(raw_spec.get("available_on_reset", True))
                available_on_step = bool(raw_spec.get("available_on_step", True))
            else:
                dtype = getattr(raw_spec, "dtype", raw_spec)
                shape = getattr(raw_spec, "shape", ())
                available_on_reset = bool(getattr(raw_spec, "available_on_reset", True))
                available_on_step = bool(getattr(raw_spec, "available_on_step", True))
            signal_schema[str(name)] = (
                raw_spec
                if isinstance(raw_spec, SignalSpec)
                else SignalSpec(
                    name=str(name),
                    dtype=dtype,
                    shape=shape,
                    available_on_reset=available_on_reset,
                    available_on_step=available_on_step,
                )
            )

    task = config.task if isinstance(getattr(config, "task", None), Mapping) else {}
    configured_signals = task.get("signals", {}) if isinstance(task, Mapping) else {}
    if not signal_schema and isinstance(configured_signals, Mapping) and configured_signals:
        reset_mask = np.ones(native_env.num_envs, dtype=np.bool_)
        reset_options: dict[str, Any] = {"reset_mask": reset_mask}
        values = getattr(native_env, "initial_state_names", ())
        values = values() if callable(values) else values
        start_catalog = tuple(str(value) for value in values or ())
        if config.state and config.state in start_catalog:
            reset_options["start_ids"] = np.asarray(
                [config.state for _ in range(native_env.num_envs)],
                dtype=object,
            )
        _observations, reset_infos = native_env.reset(
            seed=[lane for lane in range(native_env.num_envs)],
            options=reset_options,
        )
        if not isinstance(reset_infos, Mapping):
            raise TypeError("native provider reset infos must be a columnar mapping")
        for name, values in reset_infos.items():
            if not isinstance(name, str) or name.startswith("_"):
                continue
            column = np.asarray(values)
            if column.shape[:1] != (native_env.num_envs,):
                continue
            signal_schema[name] = SignalSpec(
                name=name,
                dtype=column.dtype,
                shape=column.shape[1:],
            )

    if task.get("id") == "mario":
        if provider.provider_id not in {
            STABLE_RETRO_TURBO_PROVIDER.provider_id,
            SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id,
        }:
            raise ValueError(f"provider {provider.provider_id!r} does not implement the Mario task")

    values = getattr(native_env, "initial_state_names", ())
    values = values() if callable(values) else values
    start_catalog = tuple(str(value) for value in values) if values else ()
    start_probabilities: tuple[float, ...] = ()
    lane_start_ids: tuple[str, ...] = ()
    if config.states and config.state_probs:
        weights = state_weight_mapping(config)
        start_probabilities = tuple(weights.get(name, 0.0) for name in start_catalog)
    elif config.states:
        lane_start_ids = tuple(config.states)
    elif config.state and start_catalog:
        lane_start_ids = tuple(config.state for _ in range(int(native_env.num_envs)))

    metadata = getattr(native_env, "metadata", {})
    render_modes = metadata.get("render_modes", ()) if isinstance(metadata, Mapping) else ()
    return ProviderDescriptor(
        provider_id=provider.provider_id,
        native_observation_space=observation_space,
        native_action_space=action_space,
        signal_schema=signal_schema,
        start_catalog=start_catalog,
        start_probabilities=start_probabilities,
        lane_start_ids=lane_start_ids,
        render_support=tuple(str(mode) for mode in render_modes),
    )


def _registered_native_gymnasium_vec_env(config: Any, native_kwargs: Mapping[str, Any]):
    provider = resolve_env_provider(config.env_provider)
    kwargs = dict(native_kwargs)
    num_envs = int(kwargs.pop("num_envs"))
    spec = gym.spec(config.game)
    if spec.vector_entry_point is None:
        raise RuntimeError(
            f"{provider.provider_id}:{config.game} has no native Gymnasium vector entry point; "
            "sync and async synthesized vectorization are unsupported"
        )
    kwargs["autoreset_mode"] = AutoresetMode.DISABLED
    env = gym.make_vec(
        config.game,
        num_envs=num_envs,
        vectorization_mode="vector_entry_point",
        **kwargs,
    )
    return _require_disabled_autoreset_mode(env, provider.provider_id)


def _require_provider(config: Any, expected_provider_id: str):
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id != expected_provider_id:
        raise ValueError(
            f"unsupported environment provider {provider.provider_id!r}; "
            f"expected {expected_provider_id}",
        )
    return provider


class _AleManualResetAdapter:
    """Normalize ALE's compact masked-reset seeds to the rlab provider contract."""

    def __init__(self, env: Any):
        self.env = env
        self.autoreset_mode = AutoresetMode.DISABLED
        self.metadata = dict(getattr(env, "metadata", {}))
        self.metadata["autoreset_mode"] = AutoresetMode.DISABLED
        self._pending_reset = np.zeros(env.num_envs, dtype=np.bool_)
        self._observations: np.ndarray | None = None

    def __getattr__(self, name: str) -> Any:
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        reset_options = dict(options or {})
        mask = reset_options.get("reset_mask")
        if mask is None:
            mask = np.ones(self.env.num_envs, dtype=np.bool_)
            reset_options["reset_mask"] = mask
        mask = np.asarray(mask, dtype=np.bool_)
        compact_seed = seed
        if isinstance(seed, (list, tuple)):
            compact_seed = np.asarray(
                [-1 if seed[index] is None else int(seed[index]) for index in np.flatnonzero(mask)],
                dtype=np.int64,
            )
        result = self.env.reset(seed=compact_seed, options=reset_options)
        self._observations = np.asarray(result[0])
        self._pending_reset[mask] = False
        return result

    def step(self, actions):
        if np.any(self._pending_reset):
            raise RuntimeError("ALE done lanes must be explicitly reset before the next step")
        result = self.env.step(actions)
        self._observations = np.asarray(result[0])
        self._pending_reset |= np.asarray(result[2], dtype=np.bool_) | np.asarray(
            result[3], dtype=np.bool_
        )
        return result

    def get_images(self):
        if self._observations is None:
            return []
        observations = self._observations
        if observations.ndim == 4:
            frames = observations[:, -1]
        elif observations.ndim == 5 and observations.shape[-1] in (1, 3, 4):
            frames = observations[:, -1]
        else:
            raise ValueError(f"unsupported ALE observation shape for rendering: {observations.shape}")
        if frames.ndim == 3:
            frames = np.repeat(frames[..., None], 3, axis=-1)
        elif frames.ndim == 4 and frames.shape[-1] == 1:
            frames = np.repeat(frames, 3, axis=-1)
        return [np.asarray(frame) for frame in frames]

    def close(self):
        return self.env.close()


def _stable_retro_turbo_make_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    retro_vec_env_type=DEFAULT_RETRO_VEC_ENV,
):
    _require_provider(config, STABLE_RETRO_TURBO_PROVIDER.provider_id)
    env_type = retro_vec_env_type
    env = env_type(
        config.game,
        **_disabled_autoreset_kwargs(native_kwargs),
    )
    env = _require_disabled_autoreset_mode(env, STABLE_RETRO_TURBO_PROVIDER.provider_id)
    return _StartInfoAdapter(env)


def _super_mario_bros_nes_turbo_make_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
):
    _require_provider(config, SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id)
    env_type = super_mario_vec_env_type()
    env = env_type(
        config.game,
        **_disabled_autoreset_kwargs(native_kwargs),
    )
    env = _require_disabled_autoreset_mode(env, SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id)
    return _StartInfoAdapter(env)


def _ale_py_make_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    ale_py_vec_env_type=ale_py_atari_vector_env_type,
):
    _require_provider(config, ALE_PY_PROVIDER.provider_id)
    env_type = ale_py_vec_env_type()
    kwargs = dict(native_kwargs)
    kwargs["autoreset_mode"] = AutoresetMode.NEXT_STEP
    env = env_type(
        config.game,
        **kwargs,
    )
    return _AleManualResetAdapter(env)


def make_provider_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    retro_vec_env_type=DEFAULT_RETRO_VEC_ENV,
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
    ale_py_vec_env_type=ale_py_atari_vector_env_type,
):
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id:
        return _stable_retro_turbo_make_vec_env(
            config,
            native_kwargs=native_kwargs,
            retro_vec_env_type=retro_vec_env_type,
        )
    if provider.provider_id == SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id:
        return _super_mario_bros_nes_turbo_make_vec_env(
            config,
            native_kwargs=native_kwargs,
            super_mario_vec_env_type=super_mario_vec_env_type,
        )
    if provider.provider_id == ALE_PY_PROVIDER.provider_id:
        return _ale_py_make_vec_env(
            config,
            native_kwargs=native_kwargs,
            ale_py_vec_env_type=ale_py_vec_env_type,
        )
    if provider.provider_id == GYMNASIUM_PROVIDER.provider_id:
        return _registered_native_gymnasium_vec_env(config, native_kwargs)
    raise ValueError(f"unsupported environment provider {provider.provider_id!r}")
