from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from rlab.event_payloads import info_event_payloads
from rlab.wrapper_specs import normalize_wrapper_spec_sequence, wrapper_spec_id


WRAPPER_SPEC_ID_KEYS = ("id", "wrapper", "class", "name", "type")
WRAPPER_SPEC_CONTROL_KEYS = frozenset((*WRAPPER_SPEC_ID_KEYS, "kwargs"))
REWARD_MODE_CHOICES = frozenset({"auto", "baseline", "bounded", "additive", "score", "native"})


class SuperMarioBrosNesRewardEnvWrapper:
    wrapper_id = "SuperMarioBrosNesRewardEnvWrapper"
    kind = "config"
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


class SuperMarioBros3NesRewardEnvWrapper(SuperMarioBrosNesRewardEnvWrapper):
    wrapper_id = "SuperMarioBros3NesRewardEnvWrapper"
    aliases = frozenset(
        {
            wrapper_id,
            "rlab.env_wrappers.SuperMarioBros3NesRewardEnvWrapper",
            "super_mario_bros3_nes_reward",
        }
    )
    game = "SuperMarioBros3-Nes-v0"


class SuperMarioBrosNesProgressInfoWrapper:
    wrapper_id = "SuperMarioBrosNesProgressInfoWrapper"
    kind = "progress_info"
    aliases = frozenset(
        {
            wrapper_id,
            "rlab.env_wrappers.SuperMarioBrosNesProgressInfoWrapper",
            "super_mario_bros_nes_progress_info",
        }
    )
    game = "SuperMarioBros-Nes-v0"
    default_keys = {
        "xscroll_hi_key": "xscrollHi",
        "xscroll_lo_key": "xscrollLo",
        "lives_key": "lives",
        "level_hi_key": "levelHi",
        "level_lo_key": "levelLo",
        "score_key": "score",
        "life_loss_event": "life_loss",
        "level_change_event": "level_change",
    }
    config_keys = frozenset(default_keys)

    def __init__(self, kwargs: Mapping[str, Any] | None = None):
        values = dict(self.default_keys)
        values.update(dict(kwargs or {}))
        self.keys = values
        self.level_x_pos = 0
        self.level_max_x_pos = 0
        self.completed_level_base = 0
        self.max_global_x_pos = 0
        self.curr_score = 0
        self.prev_lives: int | None = None
        self.initial_level: tuple[int, int] | None = None
        self.current_level: tuple[int, int] | None = None
        self.completed_level_count = 0
        self.current_level_completion_awarded = False
        self.completed = False

    @classmethod
    def validate_kwargs(cls, kwargs: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        extra = sorted(set(kwargs) - cls.config_keys)
        if extra:
            raise ValueError(f"{label}.kwargs has unsupported key(s): {extra}")
        cleaned = deepcopy(dict(kwargs))
        for key, value in cleaned.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.kwargs.{key} must be a non-empty string")
            cleaned[key] = value.strip()
        return cleaned

    @classmethod
    def apply(cls, config: Any, kwargs: Mapping[str, Any], *, label: str) -> Any:
        if getattr(config, "game", None) != cls.game:
            raise ValueError(f"{label} supports only game {cls.game}")
        cls.validate_kwargs(kwargs, label=label)
        return config

    def reset(self, info: dict[str, Any] | None = None) -> None:
        info = info or {}
        self.level_x_pos = 0
        self.level_max_x_pos = 0
        self.completed_level_base = 0
        self.max_global_x_pos = 0
        self.curr_score = int(info.get(self.keys["score_key"], 0))
        lives = info.get(self.keys["lives_key"])
        self.prev_lives = int(lives) if lives is not None else None
        if self.keys["level_hi_key"] in info or self.keys["level_lo_key"] in info:
            level = (
                int(info.get(self.keys["level_hi_key"], 0)),
                int(info.get(self.keys["level_lo_key"], 0)),
            )
            self.initial_level = level
            self.current_level = level
        else:
            self.initial_level = None
            self.current_level = None
        self.completed_level_count = 0
        self.current_level_completion_awarded = False
        self.completed = False

    def annotate(self, info: dict[str, Any], *, native_reward: float, done: bool) -> None:
        del native_reward, done
        x_pos = int(info.get(self.keys["xscroll_hi_key"], 0)) * 256 + int(
            info.get(self.keys["xscroll_lo_key"], 0)
        )
        lives = info.get(self.keys["lives_key"])
        level = (
            int(info.get(self.keys["level_hi_key"], 0)),
            int(info.get(self.keys["level_lo_key"], 0)),
        )
        if self.initial_level is None:
            self.initial_level = level
        if self.current_level is None:
            self.current_level = level

        info_events = info_event_payloads(info)
        life_loss_event = self.keys["life_loss_event"]
        level_change_event = self.keys["level_change_event"]
        died = life_loss_event in info_events or bool(info.get(life_loss_event, False))
        if self.prev_lives is not None and lives is not None and int(lives) < self.prev_lives:
            died = True
        if lives is not None:
            self.prev_lives = int(lives)

        native_level_changed = level_change_event in info_events
        level_changed = native_level_changed or level != self.current_level
        level_completion_event = False
        if level_changed and not died:
            self.completed_level_base += self.level_max_x_pos
            self.completed_level_count += 1
            level_completion_event = not self.current_level_completion_awarded
            self.current_level = level
            self.level_max_x_pos = 0
            self.current_level_completion_awarded = False

        effective_x_pos = 0 if level_changed else x_pos
        self.level_x_pos = effective_x_pos
        self.level_max_x_pos = max(self.level_max_x_pos, effective_x_pos)
        global_x_pos = self.completed_level_base + self.level_x_pos
        global_max_x_pos = self.completed_level_base + self.level_max_x_pos
        progress_delta = max(0, global_max_x_pos - self.max_global_x_pos)
        self.max_global_x_pos = max(self.max_global_x_pos, global_max_x_pos)

        completion_event = level_completion_event
        if completion_event:
            self.completed = True

        score = int(info.get(self.keys["score_key"], 0))
        score_delta = max(0, score - self.curr_score)
        self.curr_score = score

        info["x_pos"] = int(global_x_pos)
        info["max_x_pos"] = int(self.max_global_x_pos)
        info["level_x_pos"] = int(self.level_x_pos)
        info["level_max_x_pos"] = int(self.level_max_x_pos)
        info["completed_level_base"] = int(self.completed_level_base)
        info["global_x_pos"] = int(global_x_pos)
        info["global_max_x_pos"] = int(self.max_global_x_pos)
        info["progress_delta"] = int(progress_delta)
        info["level_id"] = f"{level[0]}-{level[1]}"
        info["level_changed"] = level_changed
        info["completed_level_count"] = int(self.completed_level_count)
        info["level_complete"] = bool(completion_event)
        info["completion_event"] = bool(completion_event)
        info["score_delta"] = int(score_delta)
        info["died"] = died
        if died:
            info["death_x_pos"] = int(self.max_global_x_pos)
            info["death_level_x_pos"] = int(self.level_max_x_pos)


class SuperMarioBros3NesProgressInfoWrapper:
    wrapper_id = "SuperMarioBros3NesProgressInfoWrapper"
    kind = "progress_info"
    aliases = frozenset(
        {
            wrapper_id,
            "rlab.env_wrappers.SuperMarioBros3NesProgressInfoWrapper",
            "super_mario_bros3_nes_progress_info",
        }
    )
    game = "SuperMarioBros3-Nes-v0"
    default_keys = {
        "hpos_key": "hpos",
        "lives_key": "lives",
        "score_key": "score",
        "life_loss_event": "life_loss",
    }
    config_keys = frozenset(default_keys)

    def __init__(self, kwargs: Mapping[str, Any] | None = None):
        values = dict(self.default_keys)
        values.update(dict(kwargs or {}))
        self.keys = values
        self.x_pos = 0
        self.max_x_pos = 0
        self.curr_score = 0
        self.prev_lives: int | None = None

    @classmethod
    def validate_kwargs(cls, kwargs: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        extra = sorted(set(kwargs) - cls.config_keys)
        if extra:
            raise ValueError(f"{label}.kwargs has unsupported key(s): {extra}")
        cleaned = deepcopy(dict(kwargs))
        for key, value in cleaned.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{label}.kwargs.{key} must be a non-empty string")
            cleaned[key] = value.strip()
        return cleaned

    @classmethod
    def apply(cls, config: Any, kwargs: Mapping[str, Any], *, label: str) -> Any:
        if getattr(config, "game", None) != cls.game:
            raise ValueError(f"{label} supports only game {cls.game}")
        cls.validate_kwargs(kwargs, label=label)
        return config

    def reset(self, info: dict[str, Any] | None = None) -> None:
        info = info or {}
        self.x_pos = int(info.get(self.keys["hpos_key"], 0))
        self.max_x_pos = self.x_pos
        self.curr_score = int(info.get(self.keys["score_key"], 0))
        lives = info.get(self.keys["lives_key"])
        self.prev_lives = int(lives) if lives is not None else None

    def annotate(self, info: dict[str, Any], *, native_reward: float, done: bool) -> None:
        del native_reward
        x_pos = int(info.get(self.keys["hpos_key"], 0))
        lives = info.get(self.keys["lives_key"])
        info_events = info_event_payloads(info)
        life_loss_event = self.keys["life_loss_event"]
        died = life_loss_event in info_events or bool(info.get(life_loss_event, False))
        if self.prev_lives is not None and lives is not None and int(lives) < self.prev_lives:
            died = True
        if lives is not None and int(lives) < 0:
            died = True
        if done and lives is not None and int(lives) < 0:
            died = True
        if lives is not None:
            self.prev_lives = int(lives)

        self.x_pos = x_pos
        previous_max_x_pos = self.max_x_pos
        self.max_x_pos = max(self.max_x_pos, x_pos)
        progress_delta = max(0, self.max_x_pos - previous_max_x_pos)

        score = int(info.get(self.keys["score_key"], 0))
        score_delta = max(0, score - self.curr_score)
        self.curr_score = score

        info["x_pos"] = int(self.x_pos)
        info["max_x_pos"] = int(self.max_x_pos)
        info["level_x_pos"] = int(self.x_pos)
        info["level_max_x_pos"] = int(self.max_x_pos)
        info["global_x_pos"] = int(self.x_pos)
        info["global_max_x_pos"] = int(self.max_x_pos)
        info["progress_delta"] = int(progress_delta)
        info["level_id"] = "1-1"
        info["level_changed"] = False
        info["completed_level_count"] = 0
        info["level_complete"] = False
        info["completion_event"] = False
        info["score_delta"] = int(score_delta)
        info["died"] = died
        if died:
            info["death_x_pos"] = int(self.max_x_pos)
            info["death_level_x_pos"] = int(self.max_x_pos)


ENV_WRAPPER_TYPES = (
    SuperMarioBrosNesRewardEnvWrapper,
    SuperMarioBros3NesRewardEnvWrapper,
    SuperMarioBrosNesProgressInfoWrapper,
    SuperMarioBros3NesProgressInfoWrapper,
)
ENV_WRAPPER_REGISTRY = {
    alias: wrapper_cls
    for wrapper_cls in ENV_WRAPPER_TYPES
    for alias in wrapper_cls.aliases
}


def _wrapper_id(spec: Mapping[str, Any], *, label: str) -> str:
    return wrapper_spec_id(spec, label=label, id_keys=WRAPPER_SPEC_ID_KEYS)


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
    specs: list[dict[str, Any]] = []
    for index, spec in enumerate(
        normalize_wrapper_spec_sequence(
            value,
            label=label,
            id_keys=WRAPPER_SPEC_ID_KEYS,
            item_kind="wrapper",
        )
    ):
        item_label = f"{label}[{index}]"
        wrapper_id = _wrapper_id(spec, label=item_label)
        wrapper_cls = ENV_WRAPPER_REGISTRY.get(wrapper_id)
        if wrapper_cls is None:
            known = ", ".join(sorted(ENV_WRAPPER_REGISTRY))
            raise ValueError(f"{item_label}.id is unknown: {wrapper_id!r}; known wrappers: {known}")
        kwargs = wrapper_cls.validate_kwargs(_wrapper_kwargs(spec, label=item_label), label=item_label)
        specs.append({"id": wrapper_cls.wrapper_id, "kwargs": kwargs})
    return tuple(specs)


def with_default_env_wrapper_specs(
    value: Any,
    defaults: Any,
    *,
    label: str = "env_wrappers",
) -> tuple[dict[str, Any], ...]:
    specs = normalize_env_wrapper_specs(value, label=label)
    default_specs = normalize_env_wrapper_specs(defaults, label=f"{label}.defaults")
    existing_ids = {spec["id"] for spec in specs}
    return tuple(spec for spec in default_specs if spec["id"] not in existing_ids) + specs


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


def progress_info_wrappers_for_config(config: Any) -> tuple[Any, ...]:
    wrappers: list[Any] = []
    for spec in normalize_env_wrapper_specs(getattr(config, "env_wrappers", ())):
        wrapper_cls = ENV_WRAPPER_REGISTRY[spec["id"]]
        if getattr(wrapper_cls, "kind", "") != "progress_info":
            continue
        wrappers.append(wrapper_cls(spec["kwargs"]))
    return tuple(wrappers)
