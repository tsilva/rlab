"""Normalize legacy and provider-owned action configuration."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import importlib.resources
import json
from pathlib import Path
from typing import Any


MARIO_PROVIDERS = frozenset(
    {"stable-retro-turbo", "supermariobrosnes-turbo"}
)
BUILTIN_ACTION_MODES = frozenset(
    {"all", "filtered", "discrete", "multi_discrete"}
)


def _mode_name(value: Any) -> str:
    name = getattr(value, "name", value)
    return str(name).split(".")[-1].strip().casefold()


def _set_native_task_action(task: dict[str, Any]) -> None:
    action = task.get("action")
    if isinstance(action, Mapping):
        normalized = dict(action)
        normalized["set"] = "native"
        task["action"] = normalized


def normalize_action_configuration(
    *,
    provider_id: str,
    game: str,
    env_args: Mapping[str, Any] | None,
    task: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Translate historical rlab action codecs to provider-owned action tables."""
    normalized_args = deepcopy(dict(env_args or {}))
    normalized_task = deepcopy(dict(task or {}))
    legacy_action_set = normalized_args.pop("action_set", None)
    action = normalized_task.get("action")
    task_action_set = (
        str(action.get("set", "native")).strip()
        if isinstance(action, Mapping)
        else "native"
    )

    requested_preset = None
    if legacy_action_set is not None:
        requested_preset = str(legacy_action_set).strip()
    elif (
        game == "SuperMarioBros-Nes-v0"
        and provider_id in MARIO_PROVIDERS
        and task_action_set != "native"
    ):
        requested_preset = task_action_set

    if requested_preset:
        existing = normalized_args.get("use_restricted_actions")
        existing_name = _mode_name(existing) if existing is not None else ""
        if existing is not None and existing_name not in {"", "all", requested_preset.casefold()}:
            raise ValueError(
                "legacy action set conflicts with env_args.use_restricted_actions: "
                f"{requested_preset!r} != {existing!r}"
            )
        normalized_args["use_restricted_actions"] = requested_preset
        _set_native_task_action(normalized_task)

    return normalized_args, normalized_task


def _packaged_action_sets(provider_id: str, game: str) -> Mapping[str, Any]:
    if provider_id == "stable-retro-turbo":
        import stable_retro

        path = Path(stable_retro.__file__).resolve().parent / "data" / "stable" / game / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
    else:
        package = {
            "supermariobrosnes-turbo": "supermariobrosnes_turbo",
            "breakout-turbo-env": "breakout_turbo_env",
        }.get(provider_id)
        if package is None:
            return {}
        path = importlib.resources.files(package).joinpath("data", game, "metadata.json")
        metadata = json.loads(path.read_text(encoding="utf-8"))
    action_sets = metadata.get("action_sets", {})
    return action_sets if isinstance(action_sets, Mapping) else {}


def _provider_buttons(provider_id: str, game: str) -> tuple[str | None, ...]:
    if provider_id == "supermariobrosnes-turbo":
        from supermariobrosnes_turbo import NES_BUTTONS

        return tuple(NES_BUTTONS)
    if provider_id == "breakout-turbo-env":
        from breakout_turbo_env import BUTTONS

        return tuple(BUTTONS)
    if provider_id == "stable-retro-turbo":
        import stable_retro

        parts = game.rsplit("-", 2)
        if len(parts) != 3:
            raise ValueError(f"cannot infer Stable Retro system from game id {game!r}")
        return tuple(stable_retro.get_system_info(parts[-2])["buttons"])
    return ()


def declared_action_contract(config: Any) -> dict[str, Any] | None:
    """Resolve a config's provider-owned exact action table for provenance checks."""
    provider_id = str(
        config.get("env_provider", "stable-retro-turbo")
        if isinstance(config, Mapping)
        else getattr(config, "env_provider", "stable-retro-turbo")
    )
    game = str(
        config.get("game", "") if isinstance(config, Mapping) else getattr(config, "game", "")
    )
    env_args = (
        config.get("env_args", {})
        if isinstance(config, Mapping)
        else getattr(config, "env_args", {})
    )
    task = config.get("task", {}) if isinstance(config, Mapping) else getattr(config, "task", {})
    env_args, _task = normalize_action_configuration(
        provider_id=provider_id,
        game=game,
        env_args=env_args if isinstance(env_args, Mapping) else {},
        task=task if isinstance(task, Mapping) else {},
    )
    request = env_args.get("use_restricted_actions")
    if request is None:
        return None
    request_name = _mode_name(request)
    if request_name in BUILTIN_ACTION_MODES:
        return {
            "mode": request_name,
            "preset": None,
            "table": None,
            "meanings": None,
            "table_hash": None,
        }
    preset = None
    table = request
    if isinstance(request, str):
        action_sets = _packaged_action_sets(provider_id, game)
        matches = {str(name).casefold(): (str(name), value) for name, value in action_sets.items()}
        try:
            preset, table = matches[request_name]
        except KeyError as exc:
            raise ValueError(
                f"provider {provider_id!r} has no action preset {request!r} for {game!r}"
            ) from exc
    if isinstance(table, (str, bytes, bytearray)) or not isinstance(table, list | tuple) or not table:
        raise ValueError("custom use_restricted_actions must be a non-empty action table")
    buttons = _provider_buttons(provider_id, game)
    button_to_index = {name: index for index, name in enumerate(buttons) if name is not None}
    players = int(env_args.get("players", 1))
    if players <= 0:
        raise ValueError("env_args.players must be positive")
    normalized: list[Any] = []
    meanings: list[str] = []
    masks: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()

    def player_action(raw_player_action: Any) -> tuple[list[str], int, str]:
        if isinstance(raw_player_action, (str, bytes, bytearray)) or not isinstance(
            raw_player_action, list | tuple
        ):
            raise ValueError("custom action entries must be button-label lists")
        if not all(isinstance(label, str) for label in raw_player_action):
            raise ValueError("custom action-table button labels must be strings")
        labels = list(raw_player_action)
        if len(set(labels)) != len(labels):
            raise ValueError("custom action table entries cannot repeat a button")
        try:
            mask = sum(1 << button_to_index[label] for label in labels)
        except KeyError as exc:
            raise ValueError(f"unknown action-table button {exc.args[0]!r}") from exc
        meaning = "noop" if not labels else "_".join(label.lower() for label in labels)
        return labels, mask, meaning

    for raw_action in table:
        if players == 1:
            labels, mask, meaning = player_action(raw_action)
            public_action: Any = labels
            action_masks = (mask,)
        else:
            if isinstance(raw_action, (str, bytes, bytearray)) or not isinstance(
                raw_action, list | tuple
            ):
                raise ValueError(
                    "multiplayer action entries must contain one action per player"
                )
            if len(raw_action) != players:
                raise ValueError(
                    f"multiplayer action entries must contain exactly {players} player actions"
                )
            public_players: list[list[str]] = []
            player_masks: list[int] = []
            player_meanings: list[str] = []
            for index, raw_player_action in enumerate(raw_action):
                labels, mask, player_meaning = player_action(raw_player_action)
                public_players.append(labels)
                player_masks.append(mask)
                player_meanings.append(f"p{index + 1}_{player_meaning}")
            public_action = public_players
            action_masks = tuple(player_masks)
            meaning = "__".join(player_meanings)
        if action_masks in seen:
            raise ValueError("custom action table contains duplicate controller actions")
        normalized.append(public_action)
        meanings.append(meaning)
        masks.append(list(action_masks))
        seen.add(action_masks)
    payload = json.dumps(masks, separators=(",", ":"), ensure_ascii=True)
    return {
        "mode": "custom_discrete",
        "preset": preset,
        "table": normalized,
        "meanings": meanings,
        "table_hash": hashlib.sha256(payload.encode("ascii")).hexdigest(),
    }


def configured_action_name(config: Any) -> str:
    contract = declared_action_contract(config)
    if contract is not None and contract.get("preset"):
        return str(contract["preset"])
    task = config.get("task", {}) if isinstance(config, Mapping) else getattr(config, "task", {})
    action = task.get("action", {}) if isinstance(task, Mapping) else {}
    return str(action.get("set", "native")) if isinstance(action, Mapping) else "native"


def configured_action_meanings(config: Any) -> tuple[str, ...]:
    contract = declared_action_contract(config)
    if contract is not None and contract.get("meanings") is not None:
        return tuple(str(value) for value in contract["meanings"])
    game = str(config.get("game", "") if isinstance(config, Mapping) else getattr(config, "game", ""))
    from rlab.targets import target_for_game

    return target_for_game(game).action_names_for_set(configured_action_name(config))


__all__ = [
    "BUILTIN_ACTION_MODES",
    "MARIO_PROVIDERS",
    "configured_action_meanings",
    "configured_action_name",
    "declared_action_contract",
    "normalize_action_configuration",
]
