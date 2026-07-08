from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rlab.env_registry import (
    ALE_PY_PROVIDER,
    SUPERMARIOBROS_NES_TURBO_PROVIDER,
    resolve_env_provider,
)

DoneOnInfoRule = tuple[str | tuple[str, ...], str]
NativeDoneOnRule = DoneOnInfoRule | None
NativeDoneOnRules = dict[str, NativeDoneOnRule]


def native_done_on_rules(
    config: Any,
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
    config: Any,
    *,
    n_envs: int,
    num_threads: int,
    native_done_on_rules: NativeDoneOnRules,
    native_obs_crop: Callable[[Any], tuple[int, int, int, int] | None],
    state_weight_mapping: Callable[[Any], dict[str, float]],
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


def _ale_py_native_vec_kwargs(
    config: Any,
    *,
    n_envs: int,
    num_threads: int,
    native_done_on_rules: NativeDoneOnRules,
    native_obs_crop: Callable[[Any], tuple[int, int, int, int] | None],
) -> dict[str, Any]:
    if config.state or config.states or config.state_probs:
        raise ValueError("ale-py provider does not support state, states, or state_probs")
    if native_done_on_rules:
        raise ValueError("ale-py provider does not support done_on_events")
    obs_crop = native_obs_crop(config)
    if obs_crop is not None and config.obs_crop_mode != "mask":
        raise ValueError("ale-py provider only supports obs_crop_mode='mask'")
    max_num_frames_per_episode = 108_000
    if config.max_episode_steps > 0:
        max_num_frames_per_episode = config.max_episode_steps * config.frame_skip
    return {
        "num_envs": n_envs,
        "num_threads": num_threads,
        "max_num_frames_per_episode": max_num_frames_per_episode,
        "repeat_action_probability": config.sticky_action_prob,
        "img_height": config.observation_size,
        "img_width": config.observation_size,
        "grayscale": True,
        "stack_num": 4,
        "frameskip": config.frame_skip,
        "maxpool": config.max_pool_frames,
        "reward_clipping": config.clip_rewards,
    }


def provider_native_vec_kwargs(
    config: Any,
    *,
    n_envs: int,
    num_threads: int,
    native_done_on_rules: NativeDoneOnRules,
    native_obs_crop: Callable[[Any], tuple[int, int, int, int] | None],
    state_weight_mapping: Callable[[Any], dict[str, float]],
) -> dict[str, Any]:
    provider = resolve_env_provider(config.env_provider)
    if provider.provider_id == ALE_PY_PROVIDER.provider_id:
        return _ale_py_native_vec_kwargs(
            config,
            n_envs=n_envs,
            num_threads=num_threads,
            native_done_on_rules=native_done_on_rules,
            native_obs_crop=native_obs_crop,
        )
    return _native_vec_kwargs(
        config,
        n_envs=n_envs,
        num_threads=num_threads,
        native_done_on_rules=native_done_on_rules,
        native_obs_crop=native_obs_crop,
        state_weight_mapping=state_weight_mapping,
    )
