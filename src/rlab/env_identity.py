from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from rlab.env_registry import resolve_env_id
from rlab.provider_config import provider_env_id, provider_game, semantic_provider_args
from rlab.preprocessing import preprocessing_contract
from rlab.task_kernels import default_task_document
from rlab.validation import normalize_obs_crop


ENVIRONMENT_HASH_ALGORITHM = "rlab.environment.v2"
ENVIRONMENT_SCHEMA_VERSION = 2

STATE_KEYS = ("state", "states", "state_probs")
PREPROCESSING_KEYS = (
    "frame_skip",
    "max_pool_frames",
    "sticky_action_prob",
    "observation_size",
    "hud_crop_top",
    "obs_crop",
    "obs_crop_mode",
    "obs_crop_fill",
    "obs_resize_algorithm",
)
IDENTITY_REWARD_KEYS = frozenset({"reward_mode"})
MARIO_REWARD_KEYS = frozenset(
    {
        "reward_mode",
        "use_native_reward",
        "clip_rewards",
        "progress_reward_cap",
        "progress_reward_scale",
        "terminal_reward",
        "reward_scale",
        "time_penalty",
        "death_penalty",
        "completion_reward",
        "score_progress_clipped",
    }
)


def _normalize_preprocessing(identity: dict[str, Any]) -> None:
    preprocessing = identity.setdefault("preprocessing", {})
    if not isinstance(preprocessing, dict):
        return
    env_id = identity.get("env_id")
    provider_id = str(env_id).split(":", 1)[0] if isinstance(env_id, str) and ":" in env_id else ""
    task = identity.get("task")
    canonical = preprocessing_contract(
        preprocessing,
        provider_id=provider_id,
        task=task if isinstance(task, Mapping) else None,
    )
    preprocessing.clear()
    preprocessing.update(canonical)


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


def _task_id(provider_id: str, game: str) -> str:
    if game == "SuperMarioBros-Nes-v0" and provider_id in {
        "",
        "stable-retro-turbo",
        "supermariobrosnes-turbo",
    }:
        return "mario"
    return "identity"


def task_config_from_train_config(
    train_config: Mapping[str, Any],
    *,
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return an explicit canonical task or the registered default for an environment."""

    if "provider" in train_config:
        raise ValueError("train config has unexpected key 'provider'; use 'env_provider'")
    provider_id = str(train_config.get("env_provider") or "")
    game = str(provider_game(train_config) or train_config.get("game") or "")
    inferred_id = _task_id(provider_id, game)
    canonical = default_task_document(inferred_id)

    embedded_task = train_config.get("task")
    if isinstance(embedded_task, Mapping) and embedded_task:
        canonical = deepcopy(dict(embedded_task))
    if isinstance(task, Mapping) and task:
        canonical = deepcopy(dict(task))
    validate_task_config(canonical)
    return canonical


def validate_task_config(task: Mapping[str, Any], *, label: str = "task") -> None:
    allowed = {"id", "action", "signals", "events", "termination", "reward", "conditioning"}
    extra = sorted(set(task) - allowed)
    if extra:
        raise ValueError(f"{label} has unexpected keys: {extra}")
    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"{label}.id must be a non-empty string")
    if task_id not in {"identity", "mario"}:
        raise ValueError(f"{label}.id has no registered task kernel: {task_id!r}")
    for section in ("action", "signals", "events", "termination", "reward"):
        if not isinstance(task.get(section), Mapping):
            raise ValueError(f"{label}.{section} must be an object")
    action = task["action"]
    action_set = action.get("set")
    if not isinstance(action_set, str) or not action_set.strip():
        raise ValueError(f"{label}.action.set must be a non-empty string")
    extra_action_keys = sorted(set(action) - {"set", "codec"})
    if extra_action_keys:
        raise ValueError(f"{label}.action has unexpected keys: {extra_action_keys}")
    codec = action.get("codec")
    if codec is not None:
        if task_id != "identity":
            raise ValueError(f"{label}.action.codec is only supported by the identity task")
        if not isinstance(codec, Mapping):
            raise ValueError(f"{label}.action.codec must be an object")
        extra_codec_keys = sorted(set(codec) - {"type", "values"})
        if extra_codec_keys:
            raise ValueError(
                f"{label}.action.codec has unexpected keys: {extra_codec_keys}"
            )
        if codec.get("type") != "discrete_lookup":
            raise ValueError(
                f"{label}.action.codec.type must be 'discrete_lookup'"
            )
        values = codec.get("values")
        if not isinstance(values, list | tuple) or not values:
            raise ValueError(f"{label}.action.codec.values must be a non-empty list")
    signals = task["signals"]
    for name, source in signals.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.signals keys must be non-empty strings")
        if isinstance(source, str) and source.strip():
            continue
        if (
            isinstance(source, list | tuple)
            and source
            and all(isinstance(item, str) and item.strip() for item in source)
        ):
            continue
        raise ValueError(f"{label}.signals.{name} must be a signal name or non-empty list")
    events = task["events"]
    for name, raw_rule in events.items():
        if not isinstance(raw_rule, Mapping):
            raise ValueError(f"{label}.events.{name} must be an object")
        signal = raw_rule.get("signal")
        if signal not in signals:
            raise ValueError(f"{label}.events.{name}.signal references unknown signal {signal!r}")
        operation = raw_rule.get("operation")
        if operation not in {"change", "decrease", "increase", "unchanged_for"}:
            raise ValueError(f"{label}.events.{name}.operation is unsupported: {operation!r}")
        if operation == "unchanged_for":
            steps = raw_rule.get("steps")
            if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 0:
                raise ValueError(f"{label}.events.{name}.steps must be a positive integer")
    if task_id == "identity" and events:
        raise ValueError(f"{label}.events must be empty for the identity task")
    if task_id == "mario":
        expected_events = {
            "life_loss": ("lives", "decrease"),
            "level_change": ("level", "change"),
            "stalled": ("x", "unchanged_for"),
        }
        unknown_events = sorted(set(events) - set(expected_events))
        if unknown_events:
            raise ValueError(
                f"{label} has unsupported Mario events: {', '.join(unknown_events)}"
            )
        for name, rule in events.items():
            expected_signal, expected_operation = expected_events[name]
            if (rule.get("signal"), rule.get("operation")) != (
                expected_signal,
                expected_operation,
            ):
                raise ValueError(
                    f"{label}.events.{name} requires signal={expected_signal!r} "
                    f"and operation={expected_operation!r}"
                )
    termination = task["termination"]
    for outcome in ("success", "failure", "timeout", "neutral"):
        if outcome not in termination:
            continue
        names = termination[outcome]
        if not isinstance(names, list | tuple):
            raise ValueError(f"{label}.termination.{outcome} must be a list")
        missing = sorted({str(name) for name in names} - set(events))
        if missing:
            raise ValueError(
                f"{label}.termination.{outcome} references unknown events: {', '.join(missing)}"
            )
    for key in ("max_episode_steps", "no_progress_min_delta"):
        if key not in termination:
            continue
        value = termination[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{label}.termination.{key} must be a non-negative integer")
    reward = task["reward"]
    allowed_reward_keys = (
        IDENTITY_REWARD_KEYS if task_id == "identity" else MARIO_REWARD_KEYS
    )
    extra_reward_keys = sorted(set(reward) - allowed_reward_keys)
    if extra_reward_keys:
        raise ValueError(f"{label}.reward has unexpected keys: {extra_reward_keys}")
    reward_mode = reward.get("reward_mode")
    if task_id == "identity" and reward_mode not in {None, "native"}:
        raise ValueError(f"{label}.reward.reward_mode must be 'native' for the identity task")
    if task_id == "mario" and reward_mode not in {
        None,
        "native",
        "bounded",
        "baseline",
        "score",
        "additive",
    }:
        raise ValueError(f"{label}.reward.reward_mode is unsupported: {reward_mode!r}")
    conditioning = task.get("conditioning")
    if conditioning is not None:
        if not isinstance(conditioning, Mapping):
            raise ValueError(f"{label}.conditioning must be an object")
        if conditioning.get("enabled"):
            signal = conditioning.get("signal")
            if signal not in signals:
                raise ValueError(
                    f"{label}.conditioning.signal references unknown signal {signal!r}"
                )


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
    identity["schema_version"] = ENVIRONMENT_SCHEMA_VERSION
    identity.pop("env_provider", None)
    if "env_id" not in identity:
        resolved_env_id = provider_env_id(train_config)
        if resolved_env_id is not None:
            identity["env_id"] = resolved_env_id
    elif isinstance(identity.get("env_id"), str):
        identity["env_id"] = resolve_env_id(str(identity["env_id"])).qualified_id

    _normalize_state_identity(identity)
    _setdefault_top_level(identity, _copy_present(train_config, STATE_KEYS))
    _setdefault_section(
        identity,
        "preprocessing",
        _copy_present(train_config, PREPROCESSING_KEYS),
    )
    existing_task = identity.pop("task", None)
    identity["task"] = task_config_from_train_config(
        train_config,
        task=existing_task if isinstance(existing_task, Mapping) else None,
    )
    provider_args = semantic_provider_args(train_config)
    if provider_args:
        identity.setdefault("provider_args", deepcopy(provider_args))
    _normalize_preprocessing(identity)
    return identity


def _obs_crop_from_value(obs_crop: Any) -> list[int] | None:
    normalized = normalize_obs_crop(
        obs_crop,
        label="environment.preprocessing.obs_crop",
    )
    return list(normalized) if normalized is not None else None


def train_config_from_source_environment(
    environment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(environment, Mapping):
        return {}
    unexpected = sorted(
        set(environment) - {"env_provider", "env_config", "preprocessing", "task"}
    )
    if unexpected:
        raise ValueError(
            "source environment has unexpected field(s): " + ", ".join(unexpected)
        )
    env_config = environment.get("env_config")
    if not isinstance(env_config, Mapping):
        raise ValueError("source environment.env_config must be an object")
    train_config = deepcopy(dict(env_config))
    env_args = train_config.get("env_args")
    if isinstance(env_args, Mapping):
        aliases = sorted(set(env_args) & {"game", "num_envs"})
        if aliases:
            raise ValueError(
                "source environment.env_config.env_args uses canonical field(s) "
                f"{aliases}; put game and n_envs directly in env_config"
            )
    provider = environment.get("env_provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError("source environment.env_provider must be a non-empty string")
    train_config["env_provider"] = provider.strip()
    preprocessing = environment.get("preprocessing")
    if preprocessing is not None:
        if not isinstance(preprocessing, Mapping):
            raise ValueError("source environment.preprocessing must be an object")
        train_config.update(deepcopy(dict(preprocessing)))
    game = provider_game(train_config)
    if game is not None:
        train_config.setdefault("game", game)
    task = environment.get("task")
    train_config["task"] = task_config_from_train_config(
        train_config,
        task=task if isinstance(task, Mapping) else None,
    )
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
