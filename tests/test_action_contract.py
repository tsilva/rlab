from types import SimpleNamespace

import pytest

from rlab.action_contract import (
    configured_action_meanings,
    configured_action_name,
    declared_action_contract,
    normalize_action_configuration,
)


def test_legacy_mario_action_set_moves_to_provider_contract():
    env_args, task = normalize_action_configuration(
        provider_id="supermariobrosnes-turbo",
        game="SuperMarioBros-Nes-v0",
        env_args={"action_set": "simple", "use_restricted_actions": "all"},
        task={"id": "mario", "action": {"set": "simple"}},
    )

    assert env_args == {"use_restricted_actions": "simple"}
    assert task["action"] == {"set": "native"}


def test_legacy_task_action_set_moves_to_stable_retro_provider():
    env_args, task = normalize_action_configuration(
        provider_id="stable-retro-turbo",
        game="SuperMarioBros-Nes-v0",
        env_args={"use_restricted_actions": "all"},
        task={"id": "mario", "action": {"set": "right"}},
    )

    assert env_args["use_restricted_actions"] == "right"
    assert task["action"]["set"] == "native"


def test_conflicting_legacy_and_provider_action_contracts_fail():
    with pytest.raises(ValueError, match="conflicts"):
        normalize_action_configuration(
            provider_id="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            env_args={"action_set": "simple", "use_restricted_actions": "right"},
            task={"id": "mario", "action": {"set": "simple"}},
        )


@pytest.mark.parametrize(
    ("provider", "game", "expected_hash"),
    [
        (
            "stable-retro-turbo",
            "SuperMarioBros-Nes-v0",
            "2eaa8ce13795d654097e6fbeb16460de8ae78f0af39b7f88259bc51604504134",
        ),
        (
            "supermariobrosnes-turbo",
            "SuperMarioBros-Nes-v0",
            "2eaa8ce13795d654097e6fbeb16460de8ae78f0af39b7f88259bc51604504134",
        ),
        (
            "breakout-turbo-env",
            "Breakout-Atari2600-v0",
            "ae2fea9e05910b0db9ba3980c162573a8ad9ad562e077babfeb5f6144d94a091",
        ),
    ],
)
def test_provider_metadata_resolves_shared_semantic_hash(provider, game, expected_hash):
    config = SimpleNamespace(
        env_provider=provider,
        game=game,
        env_args={"use_restricted_actions": "simple"},
        task={"action": {"set": "native"}},
    )
    contract = declared_action_contract(config)

    assert contract["preset"] == "simple"
    assert contract["table_hash"] == expected_hash
    assert configured_action_name(config) == "simple"
    assert configured_action_meanings(config) == tuple(contract["meanings"])


def test_multiplayer_inline_table_is_joint_not_cartesian_and_order_stable():
    base = SimpleNamespace(
        env_provider="stable-retro-turbo",
        game="SuperMarioBros-Nes-v0",
        env_args={
            "players": 2,
            "use_restricted_actions": [
                [[], []],
                [["RIGHT", "A"], ["LEFT"]],
            ],
        },
        task={"action": {"set": "native"}},
    )
    reordered = SimpleNamespace(
        **{
            **vars(base),
            "env_args": {
                "players": 2,
                "use_restricted_actions": [
                    [[], []],
                    [["A", "RIGHT"], ["LEFT"]],
                ],
            },
        }
    )

    contract = declared_action_contract(base)
    reordered_contract = declared_action_contract(reordered)

    assert contract["table"] == [[[], []], [["RIGHT", "A"], ["LEFT"]]]
    assert contract["meanings"] == ["p1_noop__p2_noop", "p1_right_a__p2_left"]
    assert contract["table_hash"] == reordered_contract["table_hash"]
