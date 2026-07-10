from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

import numpy as np


def target_class_name_for_game(game: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", game)
    return "".join(part[:1].upper() + part[1:] for part in parts) + "Target"


def _button_mask(size: int, *buttons: int) -> np.ndarray:
    mask = np.zeros(size, dtype=np.int8)
    for button in buttons:
        mask[button] = 1
    return mask


@dataclass(frozen=True)
class EvalProgressField:
    info_key: str
    result_key: str
    rank: bool = False


@dataclass(frozen=True)
class EvalSemantics:
    completion_reason: str | None = None
    completion_info_keys: tuple[str, ...] = ()
    completion_blocking_info_keys: tuple[str, ...] = ()
    progress_fields: tuple[EvalProgressField, ...] = ()
    death_flag_key: str | None = None
    death_position_key: str | None = None
    best_episode_rank: tuple[str, ...] = ("reward",)


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
    eval_semantics: ClassVar[EvalSemantics] = EvalSemantics()

    @classmethod
    def action_names_for_set(cls, action_set: str) -> tuple[str, ...]:
        if action_set == "native":
            return ()
        if action_set not in cls.action_sets:
            valid = ", ".join(sorted(cls.action_sets)) or "native"
            raise ValueError(f"unknown action_set {action_set!r} for {cls.game}; valid: {valid}")
        return cls.action_sets[action_set]

    @classmethod
    def action_masks_for_set(cls, action_set: str) -> tuple[np.ndarray, ...]:
        return tuple(cls.action_library[name] for name in cls.action_names_for_set(action_set))

    @classmethod
    def uses_discrete_actions(cls, action_set: str) -> bool:
        return bool(cls.action_masks_for_set(action_set))


class GenericRetroTarget(RetroTarget):
    pass


class SuperMarioBrosNesV0Target(RetroTarget):
    game = "SuperMarioBros-Nes-v0"
    default_state = "Level1-1"
    default_hud_crop_top = 32
    default_action_set = "simple"
    default_reward_mode = "baseline"
    native_life_variable = "lives"
    native_level_variables = ("levelHi", "levelLo")
    eval_semantics = EvalSemantics(
        completion_reason="level_change",
        completion_info_keys=("completion_event", "level_complete"),
        completion_blocking_info_keys=("died", "life_loss"),
        progress_fields=(
            EvalProgressField("max_x_pos", "max_x_pos", rank=True),
            EvalProgressField("level_max_x_pos", "max_level_x_pos"),
        ),
        death_flag_key="died",
        death_position_key="death_x_pos",
        best_episode_rank=("completion", "progress", "reward"),
    )

    # Stable Retro NES button order: B, -, SELECT, START, UP, DOWN, LEFT, RIGHT, A.
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


TARGETS: dict[str, type[RetroTarget]] = {
    SuperMarioBrosNesV0Target.game: SuperMarioBrosNesV0Target,
    SuperMarioBros3NesV0Target.game: SuperMarioBros3NesV0Target,
}


def target_for_game(game: str) -> type[RetroTarget]:
    if game not in TARGETS:
        TARGETS[game] = type(
            target_class_name_for_game(game),
            (GenericRetroTarget,),
            {"game": game, "__module__": __name__},
        )
    return TARGETS[game]
