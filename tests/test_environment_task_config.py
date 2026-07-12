from __future__ import annotations

import unittest

from rlab.env_identity import (
    ENVIRONMENT_HASH_ALGORITHM,
    environment_hash,
    environment_identity_from_train_config,
    validate_task_config,
)
from rlab.env_metadata import sanitize_env_config_metadata


class EnvironmentTaskConfigTests(unittest.TestCase):
    def test_environment_identity_uses_v2_canonical_task(self) -> None:
        identity = environment_identity_from_train_config(
            {
                "env_provider": "supermariobrosnes-turbo",
                "game": "SuperMarioBros-Nes-v0",
                "task": {
                    "id": "mario",
                    "action": {"set": "simple"},
                    "signals": {
                        "x": ["xscrollHi", "xscrollLo"],
                        "score": "score",
                        "lives": "lives",
                        "level": ["levelHi", "levelLo"],
                    },
                    "events": {
                        "life_loss": {"signal": "lives", "operation": "decrease"},
                        "level_change": {"signal": "level", "operation": "change"},
                    },
                    "termination": {
                        "failure": ["life_loss"],
                        "success": ["level_change"],
                    },
                    "reward": {"reward_mode": "score"},
                },
            }
        )

        self.assertEqual(ENVIRONMENT_HASH_ALGORITHM, "rlab.environment.v2")
        self.assertEqual(identity["schema_version"], 2)
        self.assertEqual(identity["task"]["id"], "mario")
        self.assertEqual(identity["task"]["termination"]["failure"], ["life_loss"])
        self.assertEqual(identity["task"]["termination"]["success"], ["level_change"])
        self.assertTrue(environment_hash(identity).startswith("sha256:"))
        self.assertNotIn("reward", identity)

    def test_unknown_artifact_environment_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected keys"):
            sanitize_env_config_metadata({
                "env_provider": "supermariobrosnes-turbo",
                "game": "SuperMarioBros-Nes-v0",
                "unknown_runtime_key": True,
            })

    def test_task_validation_rejects_unknown_termination_event(self) -> None:
        with self.assertRaisesRegex(ValueError, "references unknown events"):
            validate_task_config(
                {
                    "id": "identity",
                    "action": {"set": "native"},
                    "signals": {},
                    "events": {},
                    "termination": {"success": ["missing"]},
                    "reward": {},
                }
            )

    def test_task_validation_rejects_unknown_reward_keys_by_task(self) -> None:
        identity_task = {
            "id": "identity",
            "action": {"set": "native"},
            "signals": {},
            "events": {},
            "termination": {},
            "reward": {"reward_mode": "native", "death_penalty": 1.0},
        }
        with self.assertRaisesRegex(ValueError, "reward has unexpected keys"):
            validate_task_config(identity_task)

        mario_task = {
            "id": "mario",
            "action": {"set": "simple"},
            "signals": {},
            "events": {},
            "termination": {},
            "reward": {"reward_mode": "score", "use_retro_reward": False},
        }
        with self.assertRaisesRegex(ValueError, "reward has unexpected keys"):
            validate_task_config(mario_task)

    def test_environment_identity_rejects_provider_alias(self) -> None:
        with self.assertRaisesRegex(ValueError, "use 'env_provider'"):
            environment_identity_from_train_config(
                {
                    "provider": "supermariobrosnes-turbo",
                    "game": "SuperMarioBros-Nes-v0",
                }
            )

    def test_identity_task_accepts_a_generic_discrete_action_lookup_codec(self) -> None:
        task = {
            "id": "identity",
            "action": {
                "set": "discrete",
                "codec": {
                    "type": "discrete_lookup",
                    "values": [
                        {"move": [0, 0], "attack": 0},
                        {"move": [1, 0], "attack": 1},
                    ],
                },
            },
            "signals": {},
            "events": {},
            "termination": {"max_episode_steps": 1000},
            "reward": {"reward_mode": "native"},
        }

        validate_task_config(task)
        identity = environment_identity_from_train_config(
            {
                "env_provider": "gymnasium",
                "game": "CustomNativeVector-v0",
                "task": task,
            }
        )
        self.assertEqual(
            identity["task"]["action"]["codec"]["type"],
            "discrete_lookup",
        )

        invalid = {**task, "action": {"set": "discrete", "codec": {"values": [0]}}}
        with self.assertRaisesRegex(ValueError, "codec.type"):
            validate_task_config(invalid)

        task_auto_fire = {
            **task,
            "action": {**task["action"], "auto_fire": {"action": 1}},
        }
        with self.assertRaisesRegex(ValueError, "action has unexpected keys"):
            validate_task_config(task_auto_fire)

    def test_task_validation_rejects_unimplemented_kernel_and_mario_event_semantics(self) -> None:
        with self.assertRaisesRegex(ValueError, "no registered task kernel"):
            validate_task_config(
                {
                    "id": "custom",
                    "action": {"set": "native"},
                    "signals": {},
                    "events": {},
                    "termination": {},
                    "reward": {"reward_mode": "native"},
                }
            )
        with self.assertRaisesRegex(ValueError, "requires signal='lives'"):
            validate_task_config(
                {
                    "id": "mario",
                    "action": {"set": "simple"},
                    "signals": {"x": "x", "lives": "lives"},
                    "events": {
                        "life_loss": {"signal": "x", "operation": "decrease"},
                    },
                    "termination": {"failure": ["life_loss"]},
                    "reward": {"reward_mode": "native"},
                }
            )


if __name__ == "__main__":
    unittest.main()
