from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlab.validation import normalize_obs_crop
from rlab.targets import target_for_game


def _value(source: Mapping[str, Any] | Any, key: str, default: Any) -> Any:
    if isinstance(source, Mapping):
        value = source.get(key, default)
    else:
        value = getattr(source, key, default)
    return default if value is None else value


def preprocessing_contract(
    source: Mapping[str, Any] | Any,
    *,
    provider_id: str | None = None,
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the canonical policy-facing preprocessing contract."""

    provider = str(provider_id or _value(source, "env_provider", "stable-retro-turbo"))
    pipeline = (
        "stable_retro_native_vec_env"
        if provider in {"", "stable-retro-turbo"}
        else f"{provider.replace('-', '_')}_native_vec_env"
    )
    existing_resize = source.get("obs_resize") if isinstance(source, Mapping) else None
    observation_size = int(
        _value(
            source,
            "observation_size",
            existing_resize[0]
            if isinstance(existing_resize, list | tuple) and existing_resize
            else 84,
        )
    )
    raw_crop = _value(source, "obs_crop", None)
    if raw_crop is None:
        hud_crop_top = int(_value(source, "hud_crop_top", 0))
        if hud_crop_top < 0:
            game = str(_value(source, "game", ""))
            hud_crop_top = target_for_game(game).default_hud_crop_top
        raw_crop = (hud_crop_top, 0, 0, 0) if hud_crop_top else None
    crop = normalize_obs_crop(raw_crop, label="environment.preprocessing.obs_crop")
    task_config = task or _value(source, "task", {})
    conditioning = (
        task_config.get("conditioning", {}) if isinstance(task_config, Mapping) else {}
    )
    max_pool_frames = _value(
        source,
        "max_pool_frames",
        _value(source, "maxpool_last_two", True),
    )
    return {
        "pipeline": pipeline,
        "obs_resize": [observation_size, observation_size],
        "obs_crop": list(crop) if crop is not None else None,
        "obs_crop_mode": str(_value(source, "obs_crop_mode", "remove")),
        "obs_crop_fill": int(_value(source, "obs_crop_fill", 0)),
        "obs_grayscale": True,
        "obs_resize_algorithm": str(_value(source, "obs_resize_algorithm", "area")),
        "frame_skip": int(_value(source, "frame_skip", 4)),
        "frame_stack": int(_value(source, "frame_stack", 4)),
        "max_pool_frames": bool(max_pool_frames),
        "sticky_action_prob": float(_value(source, "sticky_action_prob", 0.0)),
        "obs_copy": str(_value(source, "obs_copy", "safe_view")),
        "policy_observation_layout": (
            "dict_image_task" if bool(conditioning.get("enabled")) else "channel_first"
        ),
    }
