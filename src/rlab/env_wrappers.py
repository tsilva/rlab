from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from typing import Any


WRAPPER_SPEC_ID_KEYS = ("id", "wrapper", "class", "name", "type")
WRAPPER_SPEC_CONTROL_KEYS = frozenset((*WRAPPER_SPEC_ID_KEYS, "kwargs"))
REWARD_MODE_CHOICES = frozenset({"auto", "baseline", "bounded", "additive", "score", "native"})


class SuperMarioBrosNesRewardEnvWrapper:
    wrapper_id = "SuperMarioBrosNesRewardEnvWrapper"
    aliases = frozenset(
        {
            wrapper_id,
            "rlab.env_wrappers.SuperMarioBrosNesRewardEnvWrapper",
            "super_mario_bros_nes_reward",
        }
    )
    game = "SuperMarioBros-Nes-v0"
    config_keys = frozenset(
        {
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
        }
    )
    nonnegative_keys = frozenset(
        {
            "progress_reward_cap",
            "terminal_reward",
            "reward_scale",
            "death_penalty",
        }
    )
    number_keys = frozenset(
        {
            "progress_reward_cap",
            "progress_reward_scale",
            "terminal_reward",
            "reward_scale",
            "time_penalty",
            "death_penalty",
            "completion_reward",
        }
    )
    bool_keys = frozenset({"use_retro_reward", "clip_rewards", "score_progress_clipped"})

    @classmethod
    def validate_kwargs(cls, kwargs: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        extra = sorted(set(kwargs) - cls.config_keys)
        if extra:
            raise ValueError(f"{label}.kwargs has unsupported key(s): {extra}")

        cleaned = deepcopy(dict(kwargs))
        if "reward_mode" in cleaned:
            reward_mode = cleaned["reward_mode"]
            if not isinstance(reward_mode, str) or reward_mode not in REWARD_MODE_CHOICES:
                choices = ", ".join(sorted(REWARD_MODE_CHOICES))
                raise ValueError(f"{label}.kwargs.reward_mode must be one of: {choices}")
        for key in cls.bool_keys:
            if key in cleaned and not isinstance(cleaned[key], bool):
                raise ValueError(f"{label}.kwargs.{key} must be a boolean")
        for key in cls.number_keys:
            if key not in cleaned:
                continue
            value = cleaned[key]
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"{label}.kwargs.{key} must be a number")
            cleaned[key] = float(value)
        for key in cls.nonnegative_keys:
            if key in cleaned and cleaned[key] < 0:
                raise ValueError(f"{label}.kwargs.{key} must be >= 0")
        return cleaned

    @classmethod
    def apply(cls, config: Any, kwargs: Mapping[str, Any], *, label: str) -> Any:
        if getattr(config, "game", None) != cls.game:
            raise ValueError(f"{label} supports only game {cls.game}")
        return replace(config, **cls.validate_kwargs(kwargs, label=label))


ENV_WRAPPER_REGISTRY = {
    alias: SuperMarioBrosNesRewardEnvWrapper
    for alias in SuperMarioBrosNesRewardEnvWrapper.aliases
}


def _wrapper_id(spec: Mapping[str, Any], *, label: str) -> str:
    for key in WRAPPER_SPEC_ID_KEYS:
        value = spec.get(key)
        if value is not None:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.{key} must be a non-empty string")
            return value.strip()
    raise ValueError(f"{label} must include one of: {', '.join(WRAPPER_SPEC_ID_KEYS)}")


def _wrapper_kwargs(spec: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    kwargs = spec.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, Mapping):
        raise ValueError(f"{label}.kwargs must be an object")
    inline_kwargs = {
        key: value
        for key, value in spec.items()
        if key not in WRAPPER_SPEC_CONTROL_KEYS
    }
    merged = deepcopy(dict(kwargs))
    merged.update(deepcopy(inline_kwargs))
    return merged


def normalize_env_wrapper_specs(
    value: Any,
    *,
    label: str = "env_wrappers",
) -> tuple[dict[str, Any], ...]:
    if value in (None, "", ()):
        return ()
    if isinstance(value, str):
        value = [{"id": value}]
    elif isinstance(value, Mapping):
        value = [value]
    elif not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a list of wrapper specs")

    specs: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        item_label = f"{label}[{index}]"
        if isinstance(item, str):
            spec = {"id": item}
        elif isinstance(item, Mapping):
            spec = deepcopy(dict(item))
        else:
            raise ValueError(f"{item_label} must be an object or wrapper id string")
        wrapper_id = _wrapper_id(spec, label=item_label)
        wrapper_cls = ENV_WRAPPER_REGISTRY.get(wrapper_id)
        if wrapper_cls is None:
            known = ", ".join(sorted(ENV_WRAPPER_REGISTRY))
            raise ValueError(f"{item_label}.id is unknown: {wrapper_id!r}; known wrappers: {known}")
        kwargs = wrapper_cls.validate_kwargs(_wrapper_kwargs(spec, label=item_label), label=item_label)
        specs.append({"id": wrapper_cls.wrapper_id, "kwargs": kwargs})
    return tuple(specs)


def resolve_configured_env_wrappers(config: Any) -> Any:
    specs = normalize_env_wrapper_specs(getattr(config, "env_wrappers", ()))
    if not specs:
        return config
    resolved = replace(config, env_wrappers=specs)
    for index, spec in enumerate(specs):
        wrapper_cls = ENV_WRAPPER_REGISTRY[spec["id"]]
        resolved = wrapper_cls.apply(
            resolved,
            spec["kwargs"],
            label=f"env_wrappers[{index}]",
        )
    return resolved
