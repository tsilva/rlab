from __future__ import annotations

import importlib.metadata
from dataclasses import asdict
from collections.abc import Mapping
from typing import Any

from rlab.env import EnvConfig, state_distribution_metadata, validate_obs_crop
from rlab.action_contract import declared_action_contract, normalize_action_configuration
from rlab.env_identity import (
    environment_hash,
    environment_identity_from_train_config,
    task_config_from_train_config,
)
from rlab.preprocessing import preprocessing_contract
from rlab.train_config import playback_env_arg_keys


PLAYBACK_ENV_ARG_KEYS = playback_env_arg_keys()

RUNTIME_VERSION_PACKAGES = {
    "stable_retro_turbo": "stable-retro-turbo",
    "supermariobrosnes_turbo": "supermariobrosnes-turbo",
    "breakout_turbo_env": "breakout-turbo-env",
    "stable_baselines3": "stable-baselines3",
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
    return preprocessing_contract(config)


def training_metadata(
    config: EnvConfig,
    *,
    rom_asset_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    env_config = env_config_metadata(config)
    preprocessing = training_preprocessing_metadata(config)
    identity_source = dict(env_config)
    if rom_asset_manifest is not None:
        identity_source["rom_asset_manifest"] = dict(rom_asset_manifest)
    environment = environment_identity_from_train_config(identity_source)
    environment.setdefault("preprocessing", {}).update(preprocessing)
    return {
        "env_config": env_config,
        "environment": environment,
        "environment_hash": environment_hash(environment),
        "preprocessing": preprocessing,
        "action": declared_action_contract(config),
        "versions": {
            "stable_retro_turbo": _package_version("stable-retro-turbo"),
            "supermariobrosnes_turbo": _package_version("supermariobrosnes-turbo"),
            "breakout_turbo_env": _package_version("breakout-turbo-env"),
            "stable_baselines3": _package_version("stable-baselines3"),
        },
    }


def assert_metadata_runtime_versions(metadata: dict[str, Any]) -> None:
    """Fail closed when an artifact's recorded runtime differs from playback."""
    training = metadata.get("training_metadata")
    versions = training.get("versions") if isinstance(training, dict) else None
    if not isinstance(versions, dict):
        return
    mismatches: list[str] = []
    for metadata_key, package in RUNTIME_VERSION_PACKAGES.items():
        expected = versions.get(metadata_key)
        if not isinstance(expected, str) or not expected:
            continue
        actual = _package_version(package)
        if actual != expected:
            mismatches.append(f"{package} expected {expected}, installed {actual or 'missing'}")
    if mismatches:
        raise SystemExit(
            "Artifact runtime version mismatch: "
            + "; ".join(mismatches)
            + ". Sync the checked-in environment with `uv sync --frozen` or reinstall the "
            "rlab uv tool before playback."
        )


def env_config_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Read canonical v3 metadata while retaining v2/top-level playback compatibility."""
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
    env_args = cleaned.get("env_args")
    if isinstance(env_args, dict):
        env_args = dict(env_args)
        legacy_game = env_args.pop("game", None)
        env_args.pop("num_envs", None)
        if legacy_game is not None:
            cleaned.setdefault("game", legacy_game)
        cleaned["env_args"] = env_args
    normalized_args, normalized_task = normalize_action_configuration(
        provider_id=str(cleaned.get("env_provider") or "stable-retro-turbo"),
        game=str(cleaned.get("game") or ""),
        env_args=cleaned.get("env_args"),
        task=cleaned.get("task"),
    )
    cleaned["env_args"] = normalized_args
    if normalized_task:
        cleaned["task"] = normalized_task
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
