from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any


STABLE_RETRO_TURBO_ENV_CONFIG_KEYS = frozenset(
    {
        "num_envs",
        "num_threads",
        "rom_path",
        "obs_resize",
        "obs_crop",
        "obs_grayscale",
        "obs_resize_algorithm",
        "obs_layout",
        "obs_copy",
        "frame_skip",
        "frame_stack",
        "frame_maxpool",
        "reset_noops",
        "action_sticky_prob",
        "reward_clip",
        "info_filter",
        "done_on",
    }
)


def _square_size_from_obs_resize(value: Any, *, label: str) -> int:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"{label}.obs_resize must be [width, height]")
    width, height = value
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"{label}.obs_resize values must be positive integers")
    if width != height:
        raise ValueError(
            f"{label}.obs_resize cannot map to current EnvConfig unless width and height match"
        )
    return int(width)


def _hud_crop_top_from_obs_crop(value: Any, *, label: str) -> int:
    if value is None:
        return 0
    if not isinstance(value, list | tuple) or len(value) != 4:
        raise ValueError(f"{label}.obs_crop must be [top, right, bottom, left]")
    top, right, bottom, left = value
    if any(item not in (0, None) for item in (right, bottom, left)):
        raise ValueError(
            f"{label}.obs_crop cannot map to current EnvConfig unless right, bottom, and left are 0"
        )
    if not isinstance(top, int) or isinstance(top, bool) or top < 0:
        raise ValueError(f"{label}.obs_crop[0] must be a non-negative integer")
    return int(top)


def normalize_provider_env_config_aliases(
    config: Mapping[str, Any],
    *,
    label: str = "env_config",
    drop_provider_aliases: bool = True,
) -> dict[str, Any]:
    """Translate stable-retro-turbo parameter names into current EnvConfig names."""

    result = deepcopy(dict(config))
    if "obs_resize" in result and "observation_size" not in result:
        result["observation_size"] = _square_size_from_obs_resize(
            result["obs_resize"],
            label=label,
        )
    if "obs_crop" in result and "hud_crop_top" not in result:
        result["hud_crop_top"] = _hud_crop_top_from_obs_crop(result["obs_crop"], label=label)
    if "frame_maxpool" in result and "max_pool_frames" not in result:
        result["max_pool_frames"] = result["frame_maxpool"]
    if "action_sticky_prob" in result and "sticky_action_prob" not in result:
        result["sticky_action_prob"] = result["action_sticky_prob"]
    if "reward_clip" in result and "clip_rewards" not in result:
        result["clip_rewards"] = result["reward_clip"]
    if "num_threads" in result and "env_threads" not in result:
        result["env_threads"] = result["num_threads"]
    done_on = result.get("done_on")
    if isinstance(done_on, Mapping):
        done_on_rules = {
            str(name): deepcopy(rule)
            for name, rule in done_on.items()
            if rule is not None
        }
        if "info_events" not in result:
            result["info_events"] = done_on_rules
        if "done_on_events" not in result:
            result["done_on_events"] = list(done_on_rules)

    if drop_provider_aliases:
        for key in STABLE_RETRO_TURBO_ENV_CONFIG_KEYS:
            result.pop(key, None)
    return result
