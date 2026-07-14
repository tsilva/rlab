from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from typing import Any

from rlab.env import EnvConfig, validate_obs_crop
from rlab.train_config import env_config_arg_fields


def parse_states(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        states = tuple(str(state).strip() for state in value)
        if any(not state for state in states):
            raise ValueError("--states must not contain empty state names")
        return states
    states = tuple(state.strip() for state in value.split(","))
    if any(not state for state in states):
        raise ValueError("--states must not contain empty state names")
    return states


def parse_state_probs(value: str | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        probs = tuple(float(prob) for prob in value)
        if any(not math.isfinite(prob) or prob < 0.0 for prob in probs) or not any(
            prob > 0.0 for prob in probs
        ):
            raise ValueError(
                "--state-probs values must be non-negative finite numbers with "
                "at least one positive value",
            )
        return probs
    probs: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--state-probs must not contain empty values")
        try:
            prob = float(item)
        except ValueError as exc:
            raise ValueError(f"--state-probs contains a non-numeric value: {item!r}") from exc
        if not math.isfinite(prob) or prob < 0.0:
            raise ValueError(
                "--state-probs values must be non-negative finite numbers with "
                "at least one positive value",
            )
        probs.append(prob)
    if not any(prob > 0.0 for prob in probs):
        raise ValueError(
            "--state-probs values must be non-negative finite numbers with "
            "at least one positive value",
        )
    return tuple(probs)


def parse_obs_crop(
    value: str | list[int] | tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if value is None or value == "":
        return None
    raw: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("["):
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"--obs-crop contains invalid JSON: {exc.msg}") from exc
        else:
            raw = [int(item.strip()) for item in text.split(",")]
    return validate_obs_crop(raw)


def env_config_from_args(
    args: argparse.Namespace,
    *,
    include_states: bool = False,
) -> EnvConfig:
    defaults = EnvConfig()

    def value(name: str, default: Any = None) -> Any:
        return getattr(args, name, getattr(defaults, name, default))

    config_kwargs: dict[str, Any] = {}
    for field in env_config_arg_fields():
        if field.dest in {"states", "state_probs"} and not include_states:
            continue
        key = field.dest
        raw_value = value(field.dest)
        if field.dest == "states":
            config_kwargs[key] = parse_states(raw_value)
        elif field.dest == "state_probs":
            config_kwargs[key] = parse_state_probs(raw_value)
        elif field.dest == "obs_crop":
            config_kwargs[key] = parse_obs_crop(raw_value)
        else:
            config_kwargs[key] = raw_value
    return EnvConfig(**config_kwargs)


def env_config_from_mapping(config: Mapping[str, Any]) -> EnvConfig:
    config_kwargs: dict[str, Any] = {}
    for field in env_config_arg_fields():
        if field.dest not in config:
            continue
        value = config[field.dest]
        if field.dest == "states":
            value = parse_states(value)
        elif field.dest == "state_probs":
            value = parse_state_probs(value)
        elif field.dest == "obs_crop":
            value = parse_obs_crop(value)
        config_kwargs[field.dest] = value
    return EnvConfig(**config_kwargs)
