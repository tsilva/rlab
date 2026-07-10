from __future__ import annotations

import importlib.metadata
from dataclasses import asdict
from typing import Any

from rlab.env import EnvConfig, native_obs_crop, state_distribution_metadata, validate_obs_crop
from rlab.env_identity import (
    environment_hash,
    environment_identity_from_train_config,
    task_config_from_train_config,
)


PLAYBACK_ENV_ARG_KEYS = {
    "env_provider": ("env_provider",),
    "game": ("game",),
    "env_args": ("env_args",),
    "state": ("state",),
    "states": ("states",),
    "state_probs": ("state_probs",),
    "frame_skip": ("frame_skip",),
    "max_pool_frames": ("max_pool_frames",),
    "sticky_action_prob": ("sticky_action_prob",),
    "observation_size": ("observation_size",),
    "hud_crop_top": ("hud_crop_top",),
    "obs_crop": ("obs_crop",),
    "obs_crop_mode": ("obs_crop_mode",),
    "obs_crop_fill": ("obs_crop_fill",),
    "obs_resize_algorithm": ("obs_resize_algorithm",),
    "task": ("task",),
}

ENV_CONFIG_METADATA_KEYS = frozenset(
    {
        *EnvConfig.__dataclass_fields__,
        "state_distribution",
        "state_sampling_mode",
    }
)


def env_config_metadata(config: EnvConfig) -> dict[str, Any]:
    raw_metadata = asdict(config)
    metadata = dict(raw_metadata)
    metadata["task"] = task_config_from_train_config(
        raw_metadata,
        task=config.task if config.task else None,
    )
    metadata["states"] = list(config.states)
    metadata["state_probs"] = list(config.state_probs)
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
            "dict_image_task"
            if bool(config.task.get("conditioning", {}).get("enabled"))
            else "channel_first"
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
    if not cleaned:
        return {}
    unexpected = sorted(set(cleaned) - ENV_CONFIG_METADATA_KEYS)
    if unexpected:
        raise ValueError(f"artifact environment config has unexpected keys: {unexpected}")
    existing_task = cleaned.get("task")
    cleaned["task"] = task_config_from_train_config(
        cleaned,
        task=existing_task if isinstance(existing_task, dict) else None,
    )
    return cleaned


def env_config_from_config_dict(
    config: dict[str, Any],
) -> EnvConfig | None:
    config = sanitize_env_config_metadata(config)
    task = config.get("task")
    field_names = set(EnvConfig.__dataclass_fields__)
    config_values: dict[str, Any] = {}
    if isinstance(task, dict):
        config_values["task"] = task
    matched = isinstance(task, dict)

    for field_name in field_names:
        if field_name in config and config[field_name] is not None:
            config_values[field_name] = (
                validate_obs_crop(config[field_name])
                if field_name == "obs_crop"
                else config[field_name]
            )
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
    if not matched:
        return None
    return EnvConfig(**config_values)
