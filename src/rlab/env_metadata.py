from __future__ import annotations

import importlib.metadata
from dataclasses import asdict
from typing import Any

from rlab.env import EnvConfig, native_obs_crop, state_distribution_metadata, validate_obs_crop
from rlab.env_config import parse_event_names, parse_info_events
from rlab.env_identity import environment_hash, environment_identity_from_train_config


PLAYBACK_ENV_ARG_KEYS = {
    "env_provider": ("env_provider",),
    "game": ("game",),
    "state": ("state",),
    "states": ("states",),
    "state_probs": ("state_probs",),
    "task_conditioning": ("task_conditioning",),
    "task_conditioning_info_vars": ("task_conditioning_info_vars",),
    "task_conditioning_info_values": ("task_conditioning_info_values",),
    "frame_skip": ("frame_skip",),
    "max_pool_frames": ("max_pool_frames",),
    "sticky_action_prob": ("sticky_action_prob",),
    "max_steps": ("max_steps", "max_episode_steps"),
    "observation_size": ("observation_size",),
    "hud_crop_top": ("hud_crop_top",),
    "obs_crop": ("obs_crop",),
    "obs_crop_mode": ("obs_crop_mode",),
    "obs_crop_fill": ("obs_crop_fill",),
    "obs_resize_algorithm": ("obs_resize_algorithm",),
    "use_retro_reward": ("use_retro_reward",),
    "clip_rewards": ("clip_rewards",),
    "reward_mode": ("reward_mode",),
    "progress_reward_cap": ("progress_reward_cap",),
    "progress_reward_scale": ("progress_reward_scale",),
    "terminal_reward": ("terminal_reward",),
    "reward_scale": ("reward_scale",),
    "time_penalty": ("time_penalty",),
    "death_penalty": ("death_penalty",),
    "completion_reward": ("completion_reward",),
    "score_progress_clipped": ("score_progress_clipped",),
    "env_wrappers": ("env_wrappers",),
    "no_progress_timeout_steps": ("no_progress_timeout_steps",),
    "no_progress_min_delta": ("no_progress_min_delta",),
    "info_events_json": ("info_events",),
    "done_on_events": ("done_on_events",),
    "action_set": ("action_set",),
    "env_threads": ("env_threads",),
}


def env_config_metadata(config: EnvConfig) -> dict[str, Any]:
    metadata = asdict(config)
    metadata["states"] = list(config.states)
    metadata["state_probs"] = list(config.state_probs)
    metadata["task_conditioning_info_vars"] = list(config.task_conditioning_info_vars)
    metadata["task_conditioning_info_values"] = [
        list(value) for value in config.task_conditioning_info_values
    ]
    metadata["env_wrappers"] = [
        {
            "id": str(spec.get("id", "")),
            "kwargs": dict(spec.get("kwargs", {})) if isinstance(spec.get("kwargs"), dict) else {},
        }
        for spec in config.env_wrappers
    ]
    if config.state_probs:
        metadata["state_sampling_mode"] = "weighted"
    elif config.states:
        metadata["state_sampling_mode"] = "fixed_per_env"
    else:
        metadata["state_sampling_mode"] = "single"
    metadata["state_distribution"] = state_distribution_metadata(config)
    return metadata


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def training_preprocessing_metadata(config: EnvConfig) -> dict[str, Any]:
    pipeline = (
        "stable_retro_native_vec_env"
        if config.env_provider == "stable-retro-turbo"
        else f"{config.env_provider.replace('-', '_')}_native_vec_env"
    )
    return {
        "pipeline": pipeline,
        "obs_resize": [config.observation_size, config.observation_size],
        "obs_crop": list(native_obs_crop(config) or ()) or None,
        "obs_crop_mode": config.obs_crop_mode,
        "obs_crop_fill": config.obs_crop_fill,
        "obs_grayscale": True,
        "obs_resize_algorithm": config.obs_resize_algorithm,
        "frame_skip": config.frame_skip,
        "frame_stack": 4,
        "maxpool_last_two": config.max_pool_frames,
        "sticky_action_prob": config.sticky_action_prob,
        "obs_copy": "safe_view",
        "policy_observation_layout": (
            "dict_image_task" if config.task_conditioning else "channel_first"
        ),
    }


def training_metadata(config: EnvConfig) -> dict[str, Any]:
    env_config = env_config_metadata(config)
    preprocessing = training_preprocessing_metadata(config)
    environment = environment_identity_from_train_config(env_config)
    environment.setdefault("preprocessing", {}).update(preprocessing)
    return {
        "env_config": env_config,
        "environment": environment,
        "environment_hash": environment_hash(environment),
        "preprocessing": preprocessing,
        "versions": {
            "stable_retro_turbo": _package_version("stable-retro-turbo"),
            "supermariobrosnes_turbo": _package_version("supermariobrosnes-turbo"),
            "stable_baselines3": _package_version("stable-baselines3"),
        },
    }


def env_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    training = metadata.get("training_metadata")
    if isinstance(training, dict):
        env_config = training.get("env_config", {})
        if isinstance(env_config, dict) and env_config:
            return sanitize_env_config_metadata(env_config)
    env_config = metadata.get("env_config", {})
    return sanitize_env_config_metadata(env_config) if isinstance(env_config, dict) else {}


def sanitize_env_config_metadata(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(config)
    had_legacy_done_on_info = "done_on_info" in cleaned
    cleaned.pop("done_on_info", None)
    if had_legacy_done_on_info and not cleaned.get("info_events"):
        cleaned.pop("done_on_events", None)
    return cleaned


def env_config_from_config_dict(
    config: dict[str, Any],
    fallback: EnvConfig | None = None,
) -> EnvConfig | None:
    field_names = set(EnvConfig.__dataclass_fields__) - {"info_events", "done_on_events"}
    config_values = asdict(fallback) if fallback is not None else {}
    matched = False

    for field_name in field_names:
        if field_name in config and config[field_name] is not None:
            config_values[field_name] = (
                validate_obs_crop(config[field_name])
                if field_name == "obs_crop"
                else config[field_name]
            )
            matched = True

    if "info_events" in config and config.get("info_events") is not None:
        config_values["info_events"] = parse_info_events(config["info_events"])
        matched = True
    if "done_on_events" in config and config.get("done_on_events") is not None:
        config_values["done_on_events"] = parse_event_names(config["done_on_events"])
        matched = True
    if "max_steps" in config and config.get("max_steps") is not None:
        config_values["max_episode_steps"] = config["max_steps"]
        matched = True

    if "states" in config and config.get("states") is not None:
        states = config["states"]
        config_values["states"] = tuple(states) if isinstance(states, list) else states
        matched = True
    if "state_probs" in config and config.get("state_probs") is not None:
        state_probs = config["state_probs"]
        config_values["state_probs"] = (
            tuple(state_probs) if isinstance(state_probs, list) else state_probs
        )
        matched = True
    if (
        "task_conditioning_info_vars" in config
        and config.get("task_conditioning_info_vars") is not None
    ):
        info_vars = config["task_conditioning_info_vars"]
        config_values["task_conditioning_info_vars"] = (
            tuple(info_vars) if isinstance(info_vars, list) else info_vars
        )
        matched = True
    if (
        "task_conditioning_info_values" in config
        and config.get("task_conditioning_info_values") is not None
    ):
        info_values = config["task_conditioning_info_values"]
        config_values["task_conditioning_info_values"] = (
            tuple(tuple(row) for row in info_values)
            if isinstance(info_values, list)
            else info_values
        )
        matched = True
    if "env_wrappers" in config and config.get("env_wrappers") is not None:
        env_wrappers = config["env_wrappers"]
        config_values["env_wrappers"] = (
            tuple(env_wrappers) if isinstance(env_wrappers, list) else env_wrappers
        )
        matched = True

    if not matched and fallback is None:
        return None
    return EnvConfig(**config_values)
