from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


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


class RetroProgressTracker:
    def __init__(self, target: type[RetroTarget], config: Any):
        self.target = target
        self.config = config
        self.episode_steps = 0

    def reset(self, info: dict[str, Any] | None = None) -> None:
        self.episode_steps = 0

    def step(self, native_reward: float, info: dict[str, Any], done: bool) -> ProgressStep:
        self.episode_steps += 1
        if done:
            info["_native_done"] = True
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
        info["reward_mode"] = self.config.reward_mode
        info["raw_reward"] = float(native_reward)
        info["shaped_reward"] = float(native_reward)
        info["time_penalty"] = self.config.time_penalty

        reward = float(native_reward) if self.config.use_retro_reward else float(native_reward)
        reward -= self.config.time_penalty
        custom_done = False
        custom_truncated = False
        if self.config.max_episode_steps > 0 and self.episode_steps >= self.config.max_episode_steps:
            custom_done = True
            custom_truncated = True
        return ProgressStep(reward=reward, done=custom_done, truncated=custom_truncated)


class SuperMarioBrosNesV0ProgressTracker(RetroProgressTracker):
    def __init__(self, target: type[RetroTarget], config: Any):
        super().__init__(target, config)
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
        self.last_progress_step = 0

    def reset(self, info: dict[str, Any] | None = None) -> None:
        super().reset(info)
        info = info or {}
        self.level_x_pos = 0
        self.level_max_x_pos = 0
        self.completed_level_base = 0
        self.max_global_x_pos = 0
        self.curr_score = int(info.get("score", 0))
        lives = info.get("lives")
        self.prev_lives = int(lives) if lives is not None else None
        if "levelHi" in info or "levelLo" in info:
            level = (int(info.get("levelHi", 0)), int(info.get("levelLo", 0)))
            self.initial_level = level
            self.current_level = level
        else:
            self.initial_level = None
            self.current_level = None
        self.completed_level_count = 0
        self.current_level_completion_awarded = False
        self.completed = False
        self.last_progress_step = 0

    def step(self, native_reward: float, info: dict[str, Any], done: bool) -> ProgressStep:
        config = self.config
        x_pos = int(info.get("xscrollHi", 0)) * 256 + int(info.get("xscrollLo", 0))
        lives = info.get("lives")
        level = (int(info.get("levelHi", 0)), int(info.get("levelLo", 0)))
        if self.initial_level is None:
            self.initial_level = level
        if self.current_level is None:
            self.current_level = level

        info_events = info.get("info_events") or info.get("done_on_info") or {}
        died = "life_loss" in info_events or bool(info.get("life_loss", False))
        if self.prev_lives is not None and lives is not None and int(lives) < self.prev_lives:
            died = True
        if lives is not None:
            self.prev_lives = int(lives)

        native_level_changed = "level_change" in info_events
        level_changed = native_level_changed or level != self.current_level
        level_completion_event = False
        if level_changed and not died:
            self.completed_level_base += self.level_max_x_pos
            self.completed_level_count += 1
            level_completion_event = not self.current_level_completion_awarded
            self.current_level = level
            self.level_max_x_pos = 0
            self.current_level_completion_awarded = False

        self.level_x_pos = x_pos
        self.level_max_x_pos = max(self.level_max_x_pos, x_pos)
        global_x_pos = self.completed_level_base + self.level_x_pos
        global_max_x_pos = self.completed_level_base + self.level_max_x_pos
        progress_delta = max(0, global_max_x_pos - self.max_global_x_pos)
        self.max_global_x_pos = max(self.max_global_x_pos, global_max_x_pos)
        if progress_delta > config.no_progress_min_delta:
            self.last_progress_step = self.episode_steps

        completion_event = level_completion_event
        if completion_event:
            self.completed = True

        custom_done = False
        custom_truncated = False

        progress_reward = min(float(progress_delta), config.progress_reward_cap)
        score = int(info.get("score", 0))
        score_delta = max(0, score - self.curr_score)
        self.curr_score = score

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

        if done:
            info["_native_done"] = True
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
        info["died"] = died
        if died:
            info["death_x_pos"] = int(self.max_global_x_pos)
            info["death_level_x_pos"] = int(self.level_max_x_pos)

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
    tracker_cls: ClassVar[type[RetroProgressTracker]] = RetroProgressTracker

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
    tracker_cls = SuperMarioBrosNesV0ProgressTracker

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


TARGETS: dict[str, type[RetroTarget]] = {
    SuperMarioBrosNesV0Target.game: SuperMarioBrosNesV0Target,
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
