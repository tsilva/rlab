from __future__ import annotations

import unittest

from rlab.env_identity import environment_identity_from_train_config, train_config_from_environment_identity
from rlab.env_registry import registered_env_ids, resolve_env_id


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


def test_resolves_registered_supermariobrosnes_turbo_env_id() -> None:
    env_id = "supermariobrosnes-turbo:SuperMarioBros-Nes-v0"

    resolved = resolve_env_id(env_id)

    assert env_id in registered_env_ids()
    assert resolved.qualified_id == env_id
    assert resolved.provider_id == "supermariobrosnes-turbo"
    assert resolved.provider_env_id == "SuperMarioBros-Nes-v0"
    assert resolved.import_name == "supermariobrosnes_turbo"


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


def test_train_config_materializes_provider_local_game_id() -> None:
    train_config = train_config_from_environment_identity(
        {"env_id": "stable-retro-turbo:SuperMarioBros-Nes-v0"}
    )

    assert train_config["game"] == "SuperMarioBros-Nes-v0"
    assert train_config["env_provider"] == "stable-retro-turbo"


def test_flat_state_materializes_train_config_state() -> None:
    train_config = train_config_from_environment_identity(
        {
            "env_id": "stable-retro-turbo:SuperMarioBros-Nes-v0",
            "state": "Level1-1",
        }
    )

    assert train_config["state"] == "Level1-1"


def test_flat_states_materializes_train_config_states() -> None:
    train_config = train_config_from_environment_identity(
        {
            "env_id": "stable-retro-turbo:SuperMarioBros-Nes-v0",
            "states": ["Level1-1", "Level1-2"],
        }
    )

    assert train_config["states"] == ["Level1-1", "Level1-2"]
