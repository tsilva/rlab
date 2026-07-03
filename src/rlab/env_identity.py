from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from rlab.env_config_aliases import normalize_provider_env_config_aliases
from rlab.env_registry import qualify_env_id, resolve_env_id


ENVIRONMENT_HASH_ALGORITHM = "rlab.environment.v1"

STATE_KEYS = ("state", "states", "state_probs")
ACTION_KEYS = ("action_set",)
PREPROCESSING_KEYS = (
    "frame_skip",
    "max_pool_frames",
    "sticky_action_prob",
    "observation_size",
    "obs_resize",
    "hud_crop_top",
    "obs_crop",
    "obs_resize_algorithm",
)
TASK_CONDITIONING_KEYS = (
    "task_conditioning",
    "task_conditioning_info_vars",
    "task_conditioning_info_values",
)
TERMINATION_KEYS = (
    "max_episode_steps",
    "no_progress_timeout_steps",
    "no_progress_min_delta",
    "info_events_json",
    "info_events",
    "done_on_events",
)
REWARD_KEYS = (
    "use_retro_reward",
    "clip_rewards",
    "reward_mode",
    "progress_reward_cap",
    "progress_reward_scale",
    "terminal_reward",
    "reward_scale",
    "time_penalty",
    "death_penalty",
    "completion_reward",
    "score_progress_clipped",
    "env_wrappers",
)


def _normalize_preprocessing(identity: dict[str, Any]) -> None:
    preprocessing = identity.setdefault("preprocessing", {})
    if not isinstance(preprocessing, dict):
        return
    env_id = identity.get("env_id")
    provider_id = str(env_id).split(":", 1)[0] if isinstance(env_id, str) and ":" in env_id else ""
    pipeline = (
        "stable_retro_native_vec_env"
        if provider_id in ("", "stable-retro-turbo")
        else f"{provider_id.replace('-', '_')}_native_vec_env"
    )
    if "obs_copy" not in preprocessing and "copy_observations" in preprocessing:
        preprocessing["obs_copy"] = "copy" if preprocessing["copy_observations"] else "safe_view"
    preprocessing.pop("copy_observations", None)
    if "max_pool_frames" not in preprocessing and "maxpool_last_two" in preprocessing:
        preprocessing["max_pool_frames"] = preprocessing["maxpool_last_two"]
    if "max_pool_frames" not in preprocessing and "frame_maxpool" in preprocessing:
        preprocessing["max_pool_frames"] = preprocessing["frame_maxpool"]
    if "sticky_action_prob" not in preprocessing and "action_sticky_prob" in preprocessing:
        preprocessing["sticky_action_prob"] = preprocessing["action_sticky_prob"]
    preprocessing.pop("frame_maxpool", None)
    preprocessing.pop("action_sticky_prob", None)
    preprocessing.setdefault("pipeline", pipeline)
    preprocessing.setdefault("frame_skip", 4)
    preprocessing.setdefault("frame_stack", 4)
    preprocessing.setdefault("max_pool_frames", True)
    preprocessing.setdefault("sticky_action_prob", 0.0)
    preprocessing.setdefault("obs_grayscale", True)
    preprocessing.setdefault("obs_resize_algorithm", "area")
    preprocessing.setdefault("obs_copy", "safe_view")
    if "obs_resize" not in preprocessing:
        observation_size = preprocessing.get("observation_size", 84)
        preprocessing["obs_resize"] = [observation_size, observation_size]
    preprocessing.pop("observation_size", None)
    if "obs_crop" not in preprocessing:
        hud_crop_top = preprocessing.get("hud_crop_top")
        preprocessing["obs_crop"] = [hud_crop_top, 0, 0, 0] if hud_crop_top else None
    preprocessing.pop("hud_crop_top", None)
    task_conditioning = identity.get("task_conditioning")
    if isinstance(task_conditioning, Mapping) and task_conditioning.get("task_conditioning"):
        layout = "dict_image_task"
    else:
        layout = "channel_first"
    preprocessing.setdefault("policy_observation_layout", layout)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def environment_hash(environment: Mapping[str, Any]) -> str:
    payload = f"{ENVIRONMENT_HASH_ALGORITHM}\n{canonical_json(environment)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _copy_present(source: Mapping[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: deepcopy(source[key]) for key in keys if key in source and source[key] is not None}


def _setdefault_section(
    environment: dict[str, Any],
    section: str,
    values: Mapping[str, Any],
) -> None:
    if not values:
        return
    existing = environment.get(section)
    if not isinstance(existing, dict):
        environment[section] = dict(values)
        return
    for key, value in values.items():
        existing.setdefault(key, value)


def _setdefault_top_level(environment: dict[str, Any], values: Mapping[str, Any]) -> None:
    for key, value in values.items():
        environment.setdefault(key, value)


def _normalize_state_identity(identity: dict[str, Any]) -> None:
    state_value = identity.get("state")
    if not isinstance(state_value, Mapping):
        return
    state_section = dict(state_value)
    identity.pop("state", None)
    for key in STATE_KEYS:
        if key in state_section and state_section[key] is not None:
            identity.setdefault(key, deepcopy(state_section[key]))


def environment_identity_from_train_config(
    train_config: Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical, hashable environment identity from launch config.

    The identity intentionally excludes optimizer, vectorization, scheduling, and
    logging knobs. It captures the interface and transition/reward semantics the
    policy actually acts within.
    """

    identity = deepcopy(dict(environment or {}))
    identity.pop("env_config", None)
    identity.setdefault("schema_version", 1)
    legacy_provider = identity.get("env_provider", identity.get("provider"))
    legacy_provider_env_id = identity.get("provider_env_id")
    identity.pop("provider", None)
    identity.pop("env_provider", None)
    identity.pop("provider_env_id", None)
    provider = train_config.get("env_provider", train_config.get("provider", legacy_provider))
    if "env_id" not in identity and train_config.get("game") is not None:
        identity["env_id"] = qualify_env_id(
            str(provider or "stable-retro-turbo"),
            str(train_config["game"]),
        )
    elif "env_id" not in identity and legacy_provider_env_id is not None:
        identity["env_id"] = qualify_env_id(
            str(legacy_provider or "stable-retro-turbo"),
            str(legacy_provider_env_id),
        )
    elif isinstance(identity.get("env_id"), str):
        env_id = str(identity["env_id"])
        if ":" not in env_id:
            identity["env_id"] = qualify_env_id(
                str(legacy_provider or "stable-retro-turbo"),
                env_id,
            )
        else:
            identity["env_id"] = resolve_env_id(env_id).qualified_id

    _normalize_state_identity(identity)
    _setdefault_top_level(identity, _copy_present(train_config, STATE_KEYS))
    _setdefault_section(identity, "action", _copy_present(train_config, ACTION_KEYS))
    _setdefault_section(
        identity,
        "preprocessing",
        _copy_present(train_config, PREPROCESSING_KEYS),
    )
    _setdefault_section(
        identity,
        "task_conditioning",
        _copy_present(train_config, TASK_CONDITIONING_KEYS),
    )
    _setdefault_section(
        identity,
        "termination",
        _copy_present(train_config, TERMINATION_KEYS),
    )
    _setdefault_section(identity, "reward", _copy_present(train_config, REWARD_KEYS))
    _normalize_preprocessing(identity)
    return identity


def _obs_crop_from_value(obs_crop: Any) -> list[int] | None:
    if obs_crop is None:
        return None
    if not isinstance(obs_crop, list | tuple) or len(obs_crop) != 4:
        raise ValueError("environment.preprocessing.obs_crop must be [top, right, bottom, left]")
    result: list[int] = []
    for index, value in enumerate(obs_crop):
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"environment.preprocessing.obs_crop[{index}] must be a non-negative integer"
            )
        result.append(int(value))
    return result


def _observation_size_from_obs_resize(obs_resize: Any) -> int:
    if not isinstance(obs_resize, list | tuple) or len(obs_resize) != 2:
        raise ValueError("environment.preprocessing.obs_resize must be [height, width]")
    height, width = obs_resize
    if (
        not isinstance(height, int)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or isinstance(width, bool)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("environment.preprocessing.obs_resize values must be positive integers")
    if height != width:
        raise ValueError(
            "environment.preprocessing.obs_resize cannot be materialized into the current "
            "runtime unless height and width match",
        )
    return height


def train_config_from_environment(environment: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(environment, Mapping):
        return {}
    train_config: dict[str, Any] = {}
    env_config = environment.get("env_config")
    if isinstance(env_config, Mapping):
        train_config.update(
            normalize_provider_env_config_aliases(
                env_config,
                label="environment.env_config",
            )
        )
    env_id = environment.get("env_id", environment.get("provider_env_id"))
    if env_id is not None:
        env_id_text = str(env_id)
        if ":" in env_id_text:
            resolved = resolve_env_id(env_id_text)
            train_config["env_provider"] = resolved.provider_id
            train_config["game"] = resolved.provider_env_id
        else:
            train_config["game"] = deepcopy(env_id)
    state_value = environment.get("state")
    if isinstance(state_value, Mapping):
        train_config.update(
            deepcopy({key: state_value[key] for key in STATE_KEYS if key in state_value})
        )
    elif state_value is not None:
        train_config["state"] = deepcopy(state_value)
    if "states" in environment:
        train_config["states"] = deepcopy(environment["states"])
    if "state_probs" in environment:
        train_config["state_probs"] = deepcopy(environment["state_probs"])
    for section in (
        "action",
        "preprocessing",
        "task_conditioning",
        "termination",
        "reward",
    ):
        value = environment.get(section)
        if isinstance(value, Mapping):
            train_config.update(deepcopy(dict(value)))
    if "obs_crop" in train_config:
        train_config["obs_crop"] = _obs_crop_from_value(train_config["obs_crop"])
    if "obs_resize" in train_config and "observation_size" not in train_config:
        train_config["observation_size"] = _observation_size_from_obs_resize(
            train_config["obs_resize"],
        )
    train_config.pop("obs_resize", None)
    if "maxpool_last_two" in train_config and "max_pool_frames" not in train_config:
        train_config["max_pool_frames"] = deepcopy(train_config["maxpool_last_two"])
    train_config.pop("maxpool_last_two", None)
    if "frame_maxpool" in train_config and "max_pool_frames" not in train_config:
        train_config["max_pool_frames"] = deepcopy(train_config["frame_maxpool"])
    train_config.pop("frame_maxpool", None)
    if "action_sticky_prob" in train_config and "sticky_action_prob" not in train_config:
        train_config["sticky_action_prob"] = deepcopy(train_config["action_sticky_prob"])
    train_config.pop("action_sticky_prob", None)
    if "info_events" in train_config and "info_events_json" not in train_config:
        train_config["info_events_json"] = deepcopy(train_config["info_events"])
    return train_config


def attach_environment_identity(document: Mapping[str, Any]) -> dict[str, Any]:
    materialized = deepcopy(dict(document))
    train_config = materialized.get("train_config")
    if not isinstance(train_config, Mapping):
        return materialized
    environment = environment_identity_from_train_config(
        train_config,
        environment=materialized.get("environment")
        if isinstance(materialized.get("environment"), Mapping)
        else None,
    )
    materialized["environment"] = environment
    materialized["environment_hash"] = environment_hash(environment)
    return materialized
