from __future__ import annotations

import unittest

from rlab.env_identity import environment_identity_from_train_config
from rlab.env_registry import (
    env_supports_states,
    registered_env_ids,
    resolve_env_id,
    resolve_env_provider,
)


def test_resolves_registered_stable_retro_turbo_env_id() -> None:
    env_id = "stable-retro-turbo:SuperMarioBros-Nes-v0"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "stable-retro-turbo"
    assert resolved.provider_env_id == "SuperMarioBros-Nes-v0"
    assert resolved.import_name == "stable_retro"


def test_resolves_registered_stable_retro_turbo_smb3_env_id() -> None:
    env_id = "stable-retro-turbo:SuperMarioBros3-Nes-v0"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "stable-retro-turbo"
    assert resolved.provider_env_id == "SuperMarioBros3-Nes-v0"
    assert resolved.import_name == "stable_retro"


def test_resolves_registered_stable_retro_turbo_atari_env_ids() -> None:
    for game in ("Breakout-Atari2600-v0", "MsPacman-Atari2600-v0"):
        env_id = f"stable-retro-turbo:{game}"

        resolved = resolve_env_id(env_id)

        assert env_id in registered_env_ids()
        assert resolved.provider_id == "stable-retro-turbo"
        assert resolved.provider_env_id == game
        assert resolved.import_name == "stable_retro"
        assert env_supports_states("stable-retro-turbo", game)


def test_resolves_registered_breakout_turbo_env_id() -> None:
    env_id = "breakout-turbo-env:BreakoutTurbo-v0"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "breakout-turbo-env"
    assert resolved.provider_env_id == "BreakoutTurbo-v0"
    assert resolved.import_name == "breakout_turbo_env"
    assert env_supports_states("breakout-turbo-env", "BreakoutTurbo-v0")


def test_resolves_registered_supermariobrosnes_turbo_env_id() -> None:
    env_id = "supermariobrosnes-turbo:SuperMarioBros-Nes-v0"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "supermariobrosnes-turbo"
    assert resolved.provider_env_id == "SuperMarioBros-Nes-v0"
    assert resolved.import_name == "supermariobrosnes_turbo"
    assert resolve_env_provider(resolved.provider_id).uses_stable_retro_roms


def test_resolves_registered_ale_py_env_id() -> None:
    env_id = "ale-py:breakout"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "ale-py"
    assert resolved.provider_env_id == "breakout"
    assert resolved.import_name == "ale_py"


def test_resolves_registered_ale_py_ms_pacman_env_id() -> None:
    env_id = "ale-py:ms_pacman"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "ale-py"
    assert resolved.provider_env_id == "ms_pacman"
    assert resolved.import_name == "ale_py"


def test_rejects_unregistered_env_id() -> None:
    with unittest.TestCase().assertRaisesRegex(ValueError, "does not register environment"):
        resolve_env_id("stable-retro-turbo:UnknownGame-v0")


def test_dynamic_native_provider_ids_are_explicit_but_not_hardcoded() -> None:
    gym_id = resolve_env_id("gymnasium:CustomNativeVector-v0")

    assert gym_id.provider_id == "gymnasium"
    assert gym_id.provider_env_id == "CustomNativeVector-v0"


def test_environment_identity_normalizes_bare_stable_retro_game() -> None:
    identity = environment_identity_from_train_config({"game": "SuperMarioBros-Nes-v0"})

    assert identity["env_id"] == "stable-retro-turbo:SuperMarioBros-Nes-v0"


def test_rejects_unknown_provider_alias() -> None:
    with unittest.TestCase().assertRaisesRegex(ValueError, "unknown environment provider"):
        environment_identity_from_train_config(
            {
                "env_provider": "stable-retro",
                "game": "SuperMarioBros-Nes-v0",
            }
        )
