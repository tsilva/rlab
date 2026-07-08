from __future__ import annotations

import re
from copy import copy
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from rlab.env_wrappers import progress_info_wrappers_for_config


def target_class_name_for_game(game: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", game)
    return "".join(part[:1].upper() + part[1:] for part in parts) + "Target"


def _button_mask(size: int, *buttons: int) -> np.ndarray:
    mask = np.zeros(size, dtype=np.int8)
    for button in buttons:
        mask[button] = 1
    return mask


@dataclass
class ProgressStep:
    reward: float
    done: bool = False
    terminal: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class EvalProgressField:
    info_key: str
    result_key: str
    rank: bool = False


@dataclass(frozen=True)
class EvalSemantics:
    completion_reason: str | None = None
    completion_info_keys: tuple[str, ...] = ()
    completion_fallback_info_key: str | None = None
    completion_blocking_info_keys: tuple[str, ...] = ()
    progress_fields: tuple[EvalProgressField, ...] = ()
    death_flag_key: str | None = None
    death_position_key: str | None = None
    best_episode_rank: tuple[str, ...] = ("reward",)


class RetroProgressTracker:
    def __init__(self, target: type[RetroTarget], config: Any):
        self.target = target
        self.config = config
        self.episode_steps = 0
        self.last_progress_step = 0
        self.progress_info_wrappers = progress_info_wrappers_for_config(config)

    def reset(self, info: dict[str, Any] | None = None) -> None:
        self.episode_steps = 0
        self.last_progress_step = 0
        for wrapper in self.progress_info_wrappers:
            wrapper.reset(info)

    @staticmethod
    def _ensure_progress_defaults(info: dict[str, Any]) -> None:
        info.setdefault("x_pos", 0)
        info.setdefault("max_x_pos", 0)
        info.setdefault("level_x_pos", 0)
        info.setdefault("level_max_x_pos", 0)
        info.setdefault("progress_delta", 0)
        info.setdefault("level_changed", False)
        info.setdefault("level_complete", False)
        info.setdefault("completion_event", False)
        info.setdefault("completed_level_count", 0)
        info.setdefault("died", False)
        info.setdefault("score_delta", 0)

    def annotate_progress_info(self, native_reward: float, info: dict[str, Any], done: bool) -> None:
        self._ensure_progress_defaults(info)
        for wrapper in self.progress_info_wrappers:
            wrapper.annotate(info, native_reward=native_reward, done=done)
        self._ensure_progress_defaults(info)

    def step(self, native_reward: float, info: dict[str, Any], done: bool) -> ProgressStep:
        config = self.config
        if done:
            info["_native_done"] = True

        self.annotate_progress_info(native_reward, info, done)

        progress_delta = int(info.get("progress_delta", 0))
        if progress_delta > config.no_progress_min_delta:
            self.last_progress_step = self.episode_steps

        custom_done = False
        custom_truncated = False

        progress_reward = min(float(progress_delta), config.progress_reward_cap)
        score_delta = int(info.get("score_delta", 0))
        completion_event = bool(info.get("completion_event", info.get("level_complete", False)))
        died = bool(info.get("died", False))

        native_reward_component = float(native_reward) if config.use_retro_reward else 0.0
        progress_component = float(progress_reward)
        progress_reward_component = 0.0
        score_reward_component = 0.0
        completion_reward_component = 0.0
        death_penalty_component = 0.0

        if config.reward_mode == "native":
            raw_reward = float(native_reward)
            shaped_reward = raw_reward
            native_reward_component = float(native_reward)
        elif config.reward_mode == "bounded":
            raw_reward = progress_reward
            if completion_event:
                raw_reward = config.terminal_reward
            if died:
                raw_reward = -config.terminal_reward
            shaped_reward = raw_reward / config.reward_scale if config.reward_scale else raw_reward
            reward_divisor = config.reward_scale if config.reward_scale else 1.0
            if completion_event:
                completion_reward_component = config.terminal_reward / reward_divisor
            elif died:
                death_penalty_component = -config.terminal_reward / reward_divisor
            else:
                progress_reward_component = float(progress_reward) / reward_divisor
        elif config.reward_mode == "baseline":
            raw_reward = float(native_reward) + float(score_delta) / 40.0
            if completion_event:
                raw_reward += config.terminal_reward
            elif died or done:
                raw_reward -= config.terminal_reward
            shaped_reward = raw_reward / config.reward_scale if config.reward_scale else raw_reward
            reward_divisor = config.reward_scale if config.reward_scale else 1.0
            native_reward_component = float(native_reward) / reward_divisor
            score_reward_component = (float(score_delta) / 40.0) / reward_divisor
            if completion_event:
                completion_reward_component = config.terminal_reward / reward_divisor
            elif died or done:
                death_penalty_component = -config.terminal_reward / reward_divisor
        elif config.reward_mode == "score":
            progress_component = (
                progress_reward if config.score_progress_clipped else float(progress_delta)
            )
            progress_reward_component = config.progress_reward_scale * progress_component
            score_reward_component = 0.01 * float(score_delta)
            shaped_reward = (
                native_reward_component
                + progress_reward_component
                + score_reward_component
            )
            if completion_event:
                completion_reward_component = config.completion_reward
                shaped_reward += completion_reward_component
            if died:
                death_penalty_component = -config.death_penalty
                shaped_reward += death_penalty_component
            raw_reward = shaped_reward
        else:
            progress_component = float(progress_delta)
            progress_reward_component = config.progress_reward_scale * progress_component
            shaped_reward = (
                native_reward_component
            ) + progress_reward_component
            if completion_event:
                completion_reward_component = config.completion_reward
                shaped_reward += completion_reward_component
            if died:
                death_penalty_component = -config.death_penalty
                shaped_reward += death_penalty_component
            raw_reward = shaped_reward

        shaped_reward -= config.time_penalty
        time_penalty_component = -config.time_penalty
        self.episode_steps += 1
        if config.max_episode_steps > 0 and self.episode_steps >= config.max_episode_steps:
            custom_done = True
            custom_truncated = True
        if (
            config.no_progress_timeout_steps > 0
            and not custom_done
            and self.episode_steps - self.last_progress_step >= config.no_progress_timeout_steps
        ):
            custom_done = True
            custom_truncated = True

        info["completion_bonus"] = config.completion_reward if completion_event else 0.0
        info["reward_mode"] = config.reward_mode
        info["progress_reward"] = float(progress_reward)
        info["progress_component"] = float(progress_component)
        info["native_reward_component"] = float(native_reward_component)
        info["progress_reward_component"] = float(progress_reward_component)
        info["score_progress_clipped"] = config.score_progress_clipped
        info["score_delta"] = int(score_delta)
        info["score_reward_component"] = float(score_reward_component)
        info["completion_reward_component"] = float(completion_reward_component)
        info["death_penalty_component"] = float(death_penalty_component)
        info["time_penalty_component"] = float(time_penalty_component)
        info["terminal_reward"] = (
            -config.terminal_reward if died else config.terminal_reward if completion_event else 0.0
        )
        info["raw_reward"] = float(raw_reward)
        info["clipped_reward"] = float(raw_reward)
        info["reward_scale"] = config.reward_scale
        info["time_penalty"] = config.time_penalty
        info["shaped_reward"] = float(shaped_reward)
        info["no_progress_truncated"] = bool(
            custom_truncated
            and config.no_progress_timeout_steps > 0
            and self.episode_steps - self.last_progress_step >= config.no_progress_timeout_steps
        )

        return ProgressStep(
            reward=float(shaped_reward),
            done=custom_done,
            truncated=custom_truncated,
        )


class RetroTarget:
    game: ClassVar[str] = ""
    default_state: ClassVar[str] = ""
    default_hud_crop_top: ClassVar[int] = 0
    default_action_set: ClassVar[str] = "native"
    default_reward_mode: ClassVar[str] = "native"
    native_life_variable: ClassVar[str | None] = None
    native_level_variables: ClassVar[tuple[str, ...]] = ()
    action_library: ClassVar[dict[str, np.ndarray]] = {}
    action_sets: ClassVar[dict[str, tuple[str, ...]]] = {}
    default_env_wrappers: ClassVar[tuple[dict[str, Any], ...]] = ()
    tracker_cls: ClassVar[type[RetroProgressTracker]] = RetroProgressTracker
    eval_semantics: ClassVar[EvalSemantics] = EvalSemantics()

    @classmethod
    def action_names_for_set(cls, action_set: str) -> tuple[str, ...]:
        if action_set == "native":
            return ()
        if action_set not in cls.action_sets:
            valid = ", ".join(sorted(cls.action_sets)) or "native"
            raise ValueError(f"unknown action_set {action_set!r} for {cls.game}; valid values: {valid}")
        return cls.action_sets[action_set]

    @classmethod
    def action_masks_for_set(cls, action_set: str) -> tuple[np.ndarray, ...]:
        return tuple(cls.action_library[name] for name in cls.action_names_for_set(action_set))

    @classmethod
    def uses_discrete_actions(cls, action_set: str) -> bool:
        return bool(cls.action_masks_for_set(action_set))

    @classmethod
    def create_tracker(cls, config: Any) -> RetroProgressTracker:
        if cls.default_env_wrappers and not getattr(config, "env_wrappers", ()):
            config = copy(config)
            setattr(config, "env_wrappers", cls.default_env_wrappers)
        return cls.tracker_cls(cls, config)


class GenericRetroTarget(RetroTarget):
    default_action_set = "native"
    tracker_cls = RetroProgressTracker


class SuperMarioBrosNesV0Target(RetroTarget):
    game = "SuperMarioBros-Nes-v0"
    default_state = "Level1-1"
    default_hud_crop_top = 32
    default_action_set = "simple"
    default_reward_mode = "baseline"
    native_life_variable = "lives"
    native_level_variables = ("levelHi", "levelLo")
    default_env_wrappers = ({"id": "SuperMarioBrosNesProgressInfoWrapper"},)
    tracker_cls = RetroProgressTracker
    eval_semantics = EvalSemantics(
        completion_reason="level_change",
        completion_info_keys=("completion_event", "level_complete"),
        completion_fallback_info_key="level_changed",
        completion_blocking_info_keys=("died", "life_loss"),
        progress_fields=(
            EvalProgressField("max_x_pos", "max_x_pos", rank=True),
            EvalProgressField("level_max_x_pos", "max_level_x_pos"),
        ),
        death_flag_key="died",
        death_position_key="death_x_pos",
        best_episode_rank=("completion", "progress", "reward"),
    )

    # stable-retro button order for NES:
    # ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']
    action_library = {
        "noop": _button_mask(9),
        "right": _button_mask(9, 7),
        "right_b": _button_mask(9, 7, 0),
        "right_a": _button_mask(9, 7, 8),
        "right_a_b": _button_mask(9, 7, 8, 0),
        "a": _button_mask(9, 8),
        "left": _button_mask(9, 6),
    }
    action_sets = {
        "simple": ("noop", "right", "right_b", "right_a", "right_a_b", "a", "left"),
        "right": ("right", "right_b", "right_a", "right_a_b"),
    }


class SuperMarioBros3NesV0Target(RetroTarget):
    game = "SuperMarioBros3-Nes-v0"
    default_state = "1Player.World1.Level1"
    default_action_set = "native"
    default_reward_mode = "native"
    native_life_variable = "lives"
    default_env_wrappers = ({"id": "SuperMarioBros3NesProgressInfoWrapper"},)
    tracker_cls = RetroProgressTracker


TARGETS: dict[str, type[RetroTarget]] = {
    SuperMarioBrosNesV0Target.game: SuperMarioBrosNesV0Target,
    SuperMarioBros3NesV0Target.game: SuperMarioBros3NesV0Target,
}


def target_for_game(game: str) -> type[RetroTarget]:
    if game in TARGETS:
        return TARGETS[game]
    class_name = target_class_name_for_game(game)
    target = type(
        class_name,
        (GenericRetroTarget,),
        {
            "game": game,
            "__module__": __name__,
        },
    )
    TARGETS[game] = target
    return target
