from __future__ import annotations

import argparse
from typing import Any

from mario_ppo.env import EnvConfig


def parse_states(value: str) -> tuple[str, ...]:
    return tuple(state.strip() for state in value.split(",") if state.strip())


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
        "game": value("game"),
        "state": value("state"),
        "frame_skip": value("frame_skip"),
        "max_pool_frames": value("max_pool_frames"),
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
        "completion_x_threshold": value("completion_x_threshold"),
        "terminate_on_life_loss": not value("no_terminate_on_life_loss", False),
        "terminate_on_level_change": value("terminate_on_level_change"),
        "terminate_on_completion": value("terminate_on_completion"),
        "action_set": value("action_set"),
    }
    if include_states:
        config_kwargs["states"] = parse_states(value("states", ""))
    if include_env_threads:
        config_kwargs["env_threads"] = value("env_threads")
    return EnvConfig(**config_kwargs)
