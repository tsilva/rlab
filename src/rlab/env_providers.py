from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
import stable_retro as retro

from gymnasium.vector import AutoresetMode
from stable_retro import RetroVecEnv as DEFAULT_RETRO_VEC_ENV

from rlab.bandit_env import BanditVectorEnv
from rlab.action_contract import (
    BUILTIN_ACTION_MODES,
    declared_action_contract,
    normalize_action_configuration,
)
from rlab.batch_runtime import ProviderDescriptor, SignalSpec
from rlab.env_registry import (
    ALE_PY_PROVIDER,
    BREAKOUT_TURBO_ENV_PROVIDER,
    GYMNASIUM_PROVIDER,
    RLAB_PROVIDER,
    STABLE_RETRO_TURBO_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER,
    is_stable_retro_atari_env,
    resolve_env_provider,
)


def _is_disabled_autoreset_mode(value: Any) -> bool:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name == "DISABLED":
        return True
    raw_value = getattr(value, "value", value)
    normalized = "".join(char for char in str(raw_value).lower() if char.isalnum())
    return normalized == "disabled"


def _declared_autoreset_mode(provider_id: str) -> AutoresetMode:
    provider = resolve_env_provider(provider_id)
    contract = provider.constructor_contract
    if contract is None or "autoreset_mode" not in contract.required_values:
        raise RuntimeError(f"provider {provider_id!r} has no declared constructor autoreset mode")
    value = str(contract.required_values["autoreset_mode"]).strip().upper()
    try:
        return AutoresetMode[value]
    except KeyError as exc:
        raise RuntimeError(
            f"provider {provider_id!r} declares unknown autoreset mode {value!r}"
        ) from exc


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


def _native_start_catalog(env: Any) -> tuple[str, ...]:
    values = getattr(env, "state_catalog", None)
    if values is None:
        values = getattr(env, "initial_state_names", ())
    values = values() if callable(values) else values
    return tuple(str(value) for value in values or ())


class _StartInfoAdapter:
    """Translate provider-generic start IDs to native catalog indices."""

    def __init__(self, env: Any):
        self.env = env

    def __getattr__(self, name: str) -> Any:
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        native_options = dict(options or {})
        start_ids = native_options.pop("start_ids", None)
        uses_state_catalog = hasattr(self.env, "state_catalog")
        values = getattr(
            self.env,
            "state_catalog" if uses_state_catalog else "initial_state_names",
            (),
        )
        values = values() if callable(values) else values
        catalog = tuple(str(value) for value in values or ())
        mask = np.ones(self.env.num_envs, dtype=np.bool_)
        if native_options.get("reset_mask") is not None:
            mask = np.asarray(native_options["reset_mask"], dtype=np.bool_)
        if start_ids is not None:
            start_ids = np.asarray(start_ids, dtype=object)
            if start_ids.shape != (self.env.num_envs,):
                raise ValueError(
                    f"start_ids must have shape ({self.env.num_envs},), got {start_ids.shape}"
                )
            catalog_indices = {name: index for index, name in enumerate(catalog)}
            state_indices = np.full(self.env.num_envs, -1, dtype=np.int32)
            for lane, start_id in enumerate(start_ids):
                if not bool(mask[lane]):
                    continue
                if start_id is None:
                    raise ValueError(f"selected lane {lane} has no start id")
                try:
                    state_indices[lane] = catalog_indices[str(start_id)]
                except KeyError as exc:
                    raise ValueError(
                        f"unknown provider start id {start_id!r}; expected one of {catalog}"
                    ) from exc
            native_options[
                "state_indices" if uses_state_catalog else "start_indices"
            ] = state_indices
        observations, infos = self.env.reset(seed=seed, options=native_options)
        if not isinstance(infos, Mapping):
            return observations, infos
        state_indices = infos.get("state_index")
        if state_indices is None:
            active_state_indices = getattr(self.env, "active_state_indices", None)
            if callable(active_state_indices):
                state_indices = active_state_indices()
        if state_indices is None:
            active_states = getattr(self.env, "active_states", None)
            if callable(active_states):
                result = dict(infos)
                result["start_id"] = np.asarray(active_states(), dtype=object)
                result["_start_id"] = mask.copy()
                return observations, result
        if state_indices is None:
            return observations, infos
        state_indices = np.asarray(state_indices, dtype=np.int32)
        if state_indices.shape != (self.env.num_envs,):
            raise ValueError("provider state_index must contain one value per lane")
        active_starts = np.empty(self.env.num_envs, dtype=object)
        for lane, state_index in enumerate(state_indices):
            active_starts[lane] = (
                catalog[int(state_index)] if 0 <= int(state_index) < len(catalog) else None
            )
        result = dict(infos)
        result["start_id"] = active_starts
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


class _CanonicalBreakoutAdapter:
    """Expose one provider-neutral Atari Breakout action and start contract."""

    def __init__(self, env: Any):
        self.env = env
        self.num_envs = int(env.num_envs)
        self.single_action_space = gym.spaces.Discrete(4)
        self.action_space = gym.spaces.MultiDiscrete(
            np.full(self.num_envs, 4, dtype=np.int64)
        )
        raw_catalog = _native_start_catalog(env)
        self._canonical_to_native: dict[str, str] = {}
        canonical_catalog: list[str] = []
        for native_name in raw_catalog:
            canonical_name = "Start" if native_name == "full" else native_name
            self._canonical_to_native.setdefault(canonical_name, native_name)
            if canonical_name not in canonical_catalog:
                canonical_catalog.append(canonical_name)
        self.state_catalog = tuple(canonical_catalog)
        self.initial_state_names = self.state_catalog
        self._uses_button_actions = isinstance(
            getattr(env, "single_action_space", None), gym.spaces.MultiBinary
        )
        self._button_actions = np.zeros((self.num_envs, 8), dtype=np.int8)

    def __getattr__(self, name: str) -> Any:
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def reset(self, *, seed=None, options=None):
        native_options = dict(options or {})
        start_ids = native_options.get("start_ids")
        if start_ids is not None:
            native_ids = np.asarray(start_ids, dtype=object).copy()
            if native_ids.shape != (self.num_envs,):
                raise ValueError(
                    f"start_ids must have shape ({self.num_envs},), got {native_ids.shape}"
                )
            for lane, value in enumerate(native_ids):
                if value is None:
                    continue
                canonical = "Start" if str(value) == "full" else str(value)
                try:
                    native_ids[lane] = self._canonical_to_native[canonical]
                except KeyError as exc:
                    raise ValueError(
                        f"unknown provider start id {value!r}; expected one of {self.state_catalog}"
                    ) from exc
            native_options["start_ids"] = native_ids
        observations, infos = self.env.reset(seed=seed, options=native_options)
        return observations, self._canonical_infos(infos)

    def step(self, actions):
        action_ids = np.asarray(actions, dtype=np.int64)
        if action_ids.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        if np.any((action_ids < 0) | (action_ids > 3)):
            raise ValueError("Breakout actions must be in [0, 3]")
        native_actions: Any = action_ids
        if self._uses_button_actions:
            self._button_actions.fill(0)
            self._button_actions[:, 0] = action_ids == 1
            self._button_actions[:, 7] = action_ids == 2
            self._button_actions[:, 6] = action_ids == 3
            native_actions = self._button_actions
        observations, rewards, terminated, truncated, infos = self.env.step(native_actions)
        return observations, rewards, terminated, truncated, self._canonical_infos(infos)

    @staticmethod
    def _canonical_infos(infos: Any) -> Any:
        if not isinstance(infos, Mapping):
            return infos
        result = dict(infos)
        for key in ("start_id", "state", "start_state"):
            values = result.get(key)
            if values is None:
                continue
            canonical = np.asarray(values, dtype=object).copy()
            canonical[canonical == "full"] = "Start"
            result[key] = canonical
        return result

    def get_images(self):
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


def breakout_turbo_vec_env_type():
    try:
        from breakout_turbo_env import BreakoutVecEnv
    except ImportError as exc:
        raise ImportError(
            "breakout-turbo-env provider requires breakout-turbo-env",
        ) from exc
    return BreakoutVecEnv


def _stable_retro_packaged_data_path(game: str, filename: str) -> Path:
    path = Path(retro.__file__).resolve().parent / "data" / "stable" / game / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"stable-retro-turbo does not provide packaged data for {game}: {filename}"
        )
    return path


def provider_native_vec_kwargs(
    config: Any,
    *,
    n_envs: int,
    native_obs_crop: Callable[[Any], tuple[int, int, int, int] | None],
    state_weight_mapping: Callable[[Any], dict[str, float]],
    runtime_rom_path: str | None = None,
) -> dict[str, Any]:
    """Compile provider mechanics without task events or termination rules."""
    normalized_args, _normalized_task = normalize_action_configuration(
        provider_id=config.env_provider,
        game=config.game,
        env_args=config.env_args,
        task=getattr(config, "task", None),
    )
    native_kwargs = dict(normalized_args)
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id == RLAB_PROVIDER.provider_id:
        if config.state or config.states or config.state_probs:
            raise ValueError("rlab provider does not support state, states, or state_probs")
        native_kwargs.setdefault("num_envs", n_envs)
        return native_kwargs
    if provider.provider_id in {
        STABLE_RETRO_TURBO_PROVIDER.provider_id,
        SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id,
    }:
        if runtime_rom_path:
            native_kwargs["rom_path"] = runtime_rom_path
        enum_args = {
            "use_restricted_actions": ("Actions",),
            "inttype": ("data", "Integrations"),
            "obs_type": ("Observations",),
        }
        for key, attribute_path in enum_args.items():
            value = native_kwargs.get(key)
            if not isinstance(value, str):
                continue
            if key == "use_restricted_actions" and value.strip().casefold() not in BUILTIN_ACTION_MODES:
                continue
            enum_type: Any = retro
            for attribute in attribute_path:
                enum_type = getattr(enum_type, attribute)
            try:
                native_kwargs[key] = enum_type[value.strip().upper()]
            except KeyError as exc:
                choices = ", ".join(member.name.lower() for member in enum_type)
                raise ValueError(f"env_args.{key} must be one of {choices}") from exc
        if (
            provider.provider_id == STABLE_RETRO_TURBO_PROVIDER.provider_id
            and native_kwargs.get("info") == "data"
        ):
            native_kwargs["info"] = str(
                _stable_retro_packaged_data_path(config.game, "data.json")
            )
    if provider.provider_id == GYMNASIUM_PROVIDER.provider_id:
        if config.state or config.states or config.state_probs:
            raise ValueError(
                f"{provider.provider_id} provider does not support state, states, or state_probs"
            )
        native_kwargs.setdefault("num_envs", n_envs)
        return native_kwargs
    if provider.provider_id == BREAKOUT_TURBO_ENV_PROVIDER.provider_id:
        if config.max_pool_frames:
            raise ValueError("breakout-turbo-env does not support max_pool_frames=true")
        if config.sticky_action_prob != 0.0:
            raise ValueError("breakout-turbo-env requires sticky_action_prob=0.0")
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
            "maxpool_last_two": False,
            "obs_copy": "safe_view",
            "obs_layout": "chw",
            "info_filter": "all",
        }
        defaults.update(native_kwargs)
        return defaults
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
        defaults["state_catalog"] = tuple(dict.fromkeys(config.states))
    else:
        defaults["state"] = config.state or None
    defaults.update(native_kwargs)
    if is_stable_retro_atari_env(config.env_provider, config.game):
        defaults.setdefault("use_fire_reset", False)
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
    configured_source_names = (
        {
            str(name)
            for source in configured_signals.values()
            for name in ((source,) if isinstance(source, str) else source)
        }
        if isinstance(configured_signals, Mapping)
        else set()
    )
    missing_source_names = configured_source_names - set(signal_schema)
    if missing_source_names:
        reset_mask = np.ones(native_env.num_envs, dtype=np.bool_)
        reset_options: dict[str, Any] = {"reset_mask": reset_mask}
        start_catalog = _native_start_catalog(native_env)
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
            if name not in missing_source_names:
                continue
            column = np.asarray(values)
            if column.shape[:1] != (native_env.num_envs,):
                continue
            signal_schema[name] = SignalSpec(
                name=name,
                dtype=column.dtype,
                shape=column.shape[1:],
            )
        missing_source_names -= set(signal_schema)
        step = getattr(native_env, "step", None)
        if missing_source_names and callable(step):
            single_action_space = getattr(native_env, "single_action_space", None)
            if single_action_space is None or not hasattr(single_action_space, "shape"):
                raise ValueError("cannot probe provider step-only task signals")
            action_shape = tuple(int(value) for value in single_action_space.shape)
            action_dtype = getattr(single_action_space, "dtype", np.int64)
            actions = np.zeros((native_env.num_envs, *action_shape), dtype=action_dtype)
            try:
                _step_obs, _rewards, _terminated, _truncated, step_infos = step(actions)
                if not isinstance(step_infos, Mapping):
                    raise TypeError("native provider step infos must be a columnar mapping")
                for name, values in step_infos.items():
                    if name not in missing_source_names:
                        continue
                    column = np.asarray(values)
                    if column.shape[:1] != (native_env.num_envs,):
                        continue
                    signal_schema[name] = SignalSpec(
                        name=name,
                        dtype=column.dtype,
                        shape=column.shape[1:],
                        available_on_reset=False,
                        available_on_step=True,
                    )
            finally:
                native_env.reset(
                    seed=[lane for lane in range(native_env.num_envs)],
                    options=reset_options,
                )

    if task.get("id") == "mario":
        if provider.provider_id not in {
            STABLE_RETRO_TURBO_PROVIDER.provider_id,
            SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id,
        }:
            raise ValueError(f"provider {provider.provider_id!r} does not implement the Mario task")

    start_catalog = _native_start_catalog(native_env)
    start_probabilities: tuple[float, ...] = ()
    lane_start_ids: tuple[str, ...] = ()
    if config.states and config.state_probs:
        weights = state_weight_mapping(config)
        start_probabilities = tuple(weights.get(name, 0.0) for name in start_catalog)
    elif config.states:
        lane_start_ids = tuple(config.states)
    elif config.state and start_catalog:
        configured_start = (
            "Start"
            if provider.provider_id == BREAKOUT_TURBO_ENV_PROVIDER.provider_id
            and config.state == "full"
            else config.state
        )
        lane_start_ids = tuple(configured_start for _ in range(int(native_env.num_envs)))

    metadata = getattr(native_env, "metadata", {})
    render_modes = metadata.get("render_modes", ()) if isinstance(metadata, Mapping) else ()
    obs_copy = getattr(native_env, "obs_copy", None)
    if obs_copy is None:
        native_kwargs = getattr(native_env, "kwargs", {})
        if isinstance(native_kwargs, Mapping):
            obs_copy = native_kwargs.get("obs_copy")
    supports_safe_views = provider.provider_id in {
        BREAKOUT_TURBO_ENV_PROVIDER.provider_id,
        STABLE_RETRO_TURBO_PROVIDER.provider_id,
        SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id,
    }
    declared_action = declared_action_contract(config)
    action_mode = getattr(native_env, "action_mode", None)
    action_preset = getattr(native_env, "action_preset", None)
    action_table = getattr(native_env, "action_table", None)
    action_meanings = getattr(native_env, "action_meanings", None)
    action_table_hash = getattr(native_env, "action_table_hash", None)
    if (
        declared_action is not None
        and declared_action["table_hash"] is not None
        and hasattr(native_env, "action_table_hash")
    ):
        if action_table_hash != declared_action["table_hash"]:
            raise ValueError(
                f"provider {provider.provider_id!r} resolved action table hash "
                f"{action_table_hash!r}; expected {declared_action['table_hash']!r}"
            )
    return ProviderDescriptor(
        provider_id=provider.provider_id,
        native_observation_space=observation_space,
        native_action_space=action_space,
        signal_schema=signal_schema,
        start_catalog=start_catalog,
        start_probabilities=start_probabilities,
        lane_start_ids=lane_start_ids,
        render_support=tuple(str(mode) for mode in render_modes),
        observation_buffer_depth=2 if supports_safe_views and obs_copy == "safe_view" else 1,
        action_mode=str(action_mode) if action_mode is not None else None,
        action_preset=str(action_preset) if action_preset is not None else None,
        action_table=tuple(action_table) if action_table is not None else None,
        action_meanings=(
            tuple(str(value) for value in action_meanings)
            if action_meanings is not None
            else None
        ),
        action_table_hash=(
            str(action_table_hash) if action_table_hash is not None else None
        ),
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


def _rlab_make_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    bandit_vec_env_type=BanditVectorEnv,
):
    _require_provider(config, RLAB_PROVIDER.provider_id)
    kwargs = dict(native_kwargs)
    kwargs["autoreset_mode"] = _declared_autoreset_mode(RLAB_PROVIDER.provider_id)
    env = bandit_vec_env_type(config.game, **kwargs)
    return _require_disabled_autoreset_mode(env, RLAB_PROVIDER.provider_id)


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
    kwargs = dict(native_kwargs)
    env = env_type(config.game, **kwargs)
    env = _require_disabled_autoreset_mode(env, STABLE_RETRO_TURBO_PROVIDER.provider_id)
    env = _StartInfoAdapter(env)
    if config.game == "Breakout-Atari2600-v0":
        return _CanonicalBreakoutAdapter(env)
    return env


def _super_mario_bros_nes_turbo_make_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
):
    _require_provider(config, SUPERMARIOBROS_NES_TURBO_PROVIDER.provider_id)
    env_type = super_mario_vec_env_type()
    kwargs = dict(native_kwargs)
    env = env_type(
        config.game,
        **kwargs,
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
    kwargs["autoreset_mode"] = _declared_autoreset_mode(ALE_PY_PROVIDER.provider_id)
    env = env_type(
        config.game,
        **kwargs,
    )
    return _AleManualResetAdapter(env)


def _breakout_turbo_make_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    breakout_vec_env_type=breakout_turbo_vec_env_type,
):
    _require_provider(config, BREAKOUT_TURBO_ENV_PROVIDER.provider_id)
    env_type = breakout_vec_env_type()
    kwargs = dict(native_kwargs)
    if "game" in inspect.signature(env_type).parameters:
        env = env_type(config.game, state=config.state or "Start", **kwargs)
    else:
        legacy_unsupported = {
            "info",
            "inttype",
            "noop_reset_max",
            "obs_type",
            "players",
            "record",
            "reward_clip",
            "rom_path",
            "scenario",
            "use_fire_reset",
            "use_restricted_actions",
        }
        env = env_type(**{key: value for key, value in kwargs.items() if key not in legacy_unsupported})
    env = _require_disabled_autoreset_mode(env, BREAKOUT_TURBO_ENV_PROVIDER.provider_id)
    return _CanonicalBreakoutAdapter(env)


def make_provider_vec_env(
    config: Any,
    *,
    native_kwargs: Mapping[str, Any],
    retro_vec_env_type=DEFAULT_RETRO_VEC_ENV,
    super_mario_vec_env_type=super_mario_bros_nes_turbo_vec_env_type,
    ale_py_vec_env_type=ale_py_atari_vector_env_type,
    breakout_vec_env_type=breakout_turbo_vec_env_type,
):
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id == RLAB_PROVIDER.provider_id:
        return _rlab_make_vec_env(config, native_kwargs=native_kwargs)
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
    if provider.provider_id == BREAKOUT_TURBO_ENV_PROVIDER.provider_id:
        return _breakout_turbo_make_vec_env(
            config,
            native_kwargs=native_kwargs,
            breakout_vec_env_type=breakout_vec_env_type,
        )
    if provider.provider_id == GYMNASIUM_PROVIDER.provider_id:
        return _registered_native_gymnasium_vec_env(config, native_kwargs)
    raise ValueError(f"unsupported environment provider {provider.provider_id!r}")
