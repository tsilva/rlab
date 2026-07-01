from __future__ import annotations

import argparse
import json
import math
from typing import Any

from rlab.env import EnvConfig, InfoEventRules


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


def parse_task_conditioning_info_vars(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return parse_states(value)


def parse_task_conditioning_info_values(
    value: str | list[list[int | str]] | list[tuple[int | str, ...]] | tuple[tuple[int | str, ...], ...],
) -> tuple[tuple[int | str, ...], ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(tuple(item) for item in value)
    rows: list[tuple[int | str, ...]] = []
    for row in value.split(";"):
        row = row.strip()
        if not row:
            raise ValueError("--task-conditioning-info-values must not contain empty rows")
        values: list[int | str] = []
        for item in row.split(","):
            item = item.strip()
            if not item:
                raise ValueError("--task-conditioning-info-values must not contain empty values")
            try:
                values.append(int(item))
            except ValueError:
                values.append(item)
        rows.append(tuple(values))
    return tuple(rows)


def parse_state_probs(value: str | list[float] | tuple[float, ...]) -> tuple[float, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        probs = tuple(float(prob) for prob in value)
        if any(not math.isfinite(prob) or prob <= 0.0 for prob in probs):
            raise ValueError("--state-probs values must be positive finite numbers")
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
        if not math.isfinite(prob) or prob <= 0.0:
            raise ValueError("--state-probs values must be positive finite numbers")
        probs.append(prob)
    return tuple(probs)


def parse_info_event_rules(
    value: str | dict[str, Any] | None,
    *,
    option_name: str,
) -> InfoEventRules:
    if not value:
        return {}
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{option_name} contains invalid JSON: {exc.msg}") from exc
    else:
        raw = value
    if not isinstance(raw, dict):
        raise ValueError(f"{option_name} must be a JSON object")

    rules: InfoEventRules = {}
    for name, rule in raw.items():
        rule_name = str(name).strip()
        if not rule_name:
            raise ValueError(f"{option_name} rule names must not be empty")
        if not isinstance(rule, (list, tuple)) or len(rule) != 2:
            raise ValueError(f"info event rule {rule_name!r} must be [key_or_keys, op]")
        key_or_keys, op = rule
        if isinstance(key_or_keys, str):
            key_text = key_or_keys.strip()
            if not key_text:
                raise ValueError(f"info event rule {rule_name!r} key must not be empty")
            key: str | tuple[str, ...] = key_text
        elif isinstance(key_or_keys, (list, tuple)):
            keys = tuple(str(item).strip() for item in key_or_keys)
            if not keys or any(not item for item in keys):
                raise ValueError(f"info event rule {rule_name!r} has invalid keys")
            key = keys
        else:
            raise ValueError(f"info event rule {rule_name!r} key must be a string or list")
        op_text = str(op).strip()
        if not op_text:
            raise ValueError(f"info event rule {rule_name!r} op must not be empty")
        rules[rule_name] = (key, op_text)
    return rules


def parse_info_events(value: str | dict[str, Any] | None) -> InfoEventRules:
    return parse_info_event_rules(value, option_name="--info-events-json")


def parse_event_names(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, (list, tuple)):
        names = tuple(str(name).strip() for name in value)
    else:
        names = tuple(name.strip() for name in value.split(","))
    if any(not name for name in names):
        raise ValueError("event name lists must not contain empty values")
    return tuple(dict.fromkeys(names))


def env_config_from_args(
    args: argparse.Namespace,
    *,
    max_episode_steps_attr: str = "max_episode_steps",
    include_states: bool = False,
    include_env_threads: bool = False,
) -> EnvConfig:
    defaults = EnvConfig()

    def value(name: str, default: Any = None) -> Any:
        return getattr(args, name, getattr(defaults, name, default))

    max_episode_steps = value(max_episode_steps_attr, defaults.max_episode_steps)
    config_kwargs: dict[str, Any] = {
        "env_provider": value("env_provider"),
        "game": value("game"),
        "state": value("state"),
        "task_conditioning": value("task_conditioning"),
        "task_conditioning_info_vars": parse_task_conditioning_info_vars(
            value("task_conditioning_info_vars", ""),
        ),
        "task_conditioning_info_values": parse_task_conditioning_info_values(
            value("task_conditioning_info_values", ""),
        ),
        "frame_skip": value("frame_skip"),
        "max_pool_frames": value("max_pool_frames"),
        "sticky_action_prob": value("sticky_action_prob"),
        "max_episode_steps": max_episode_steps,
        "hud_crop_top": value("hud_crop_top"),
        "use_retro_reward": value("use_retro_reward"),
        "clip_rewards": value("clip_rewards"),
        "reward_mode": value("reward_mode"),
        "progress_reward_cap": value("progress_reward_cap"),
        "progress_reward_scale": value("progress_reward_scale"),
        "terminal_reward": value("terminal_reward"),
        "reward_scale": value("reward_scale"),
        "time_penalty": value("time_penalty"),
        "death_penalty": value("death_penalty"),
        "completion_reward": value("completion_reward"),
        "score_progress_clipped": value("score_progress_clipped"),
        "no_progress_timeout_steps": value("no_progress_timeout_steps"),
        "no_progress_min_delta": value("no_progress_min_delta"),
        "info_events": parse_info_events(
            value("info_events_json", value("info_events", "")),
        ),
        "done_on_events": parse_event_names(value("done_on_events", "")),
        "action_set": value("action_set"),
    }
    if include_states:
        config_kwargs["states"] = parse_states(value("states", ""))
        config_kwargs["state_probs"] = parse_state_probs(value("state_probs", ""))
    if include_env_threads:
        config_kwargs["env_threads"] = value("env_threads")
    return EnvConfig(**config_kwargs)
