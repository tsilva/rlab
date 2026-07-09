from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlab.env_registry import qualify_env_id, resolve_env_id


DEFAULT_TRAIN_N_ENVS = 8
NON_SEMANTIC_ENV_ARG_KEYS = frozenset(
    {
        "batch_size",
        "game",
        "num_envs",
        "num_threads",
        "thread_affinity_offset",
    }
)


def _get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def provider_env_args(config: Any) -> Mapping[str, Any]:
    env_args = _get(config, "env_args", {})
    return env_args if isinstance(env_args, Mapping) else {}


def provider_game(config: Any) -> str | None:
    game = _get(config, "game")
    if game:
        return str(game)
    env_args = provider_env_args(config)
    arg_game = env_args.get("game")
    return str(arg_game) if arg_game else None


def provider_env_id(config: Any) -> str | None:
    env_id = _get(config, "env_id")
    if isinstance(env_id, str) and env_id.strip():
        return resolve_env_id(env_id).qualified_id
    game = provider_game(config)
    if not game:
        return None
    provider = str(_get(config, "env_provider", "stable-retro-turbo") or "stable-retro-turbo")
    return qualify_env_id(provider, game)


def provider_num_envs(
    config: Any,
    *,
    explicit_n_envs: Any = None,
    default_n_envs: int = DEFAULT_TRAIN_N_ENVS,
) -> int:
    env_args = provider_env_args(config)
    provider_n_envs = env_args.get("num_envs")
    if explicit_n_envs is not None:
        n_envs = int(explicit_n_envs)
        if provider_n_envs is not None and int(provider_n_envs) != n_envs:
            raise ValueError(
                "env_args.num_envs must match requested n_envs: "
                f"{provider_n_envs!r} != {n_envs!r}",
            )
        return n_envs
    if provider_n_envs is not None:
        return int(provider_n_envs)
    return int(default_n_envs)


def semantic_provider_args(config: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in provider_env_args(config).items()
        if key not in NON_SEMANTIC_ENV_ARG_KEYS
    }
