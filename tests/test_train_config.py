from __future__ import annotations

import unittest

from rlab.train_config import (
    build_train_command_from_fields,
    train_config_field_for_key,
    validate_train_config_fields,
    validate_train_config_value,
)


class TrainConfigFieldSchemaTests(unittest.TestCase):
    def test_env_config_aliases_resolve_to_train_config_fields(self) -> None:
        field = train_config_field_for_key("info_events")

        self.assertIsNotNone(field)
        self.assertEqual(field.dest, "info_events_json")
        self.assertEqual(field.env_config_key, "info_events")

    def test_build_train_command_accepts_env_config_aliases(self) -> None:
        command = build_train_command_from_fields(
            {
                "info_events": {"life_loss": ["lives", "decrease"]},
                "done_on_events": ["life_loss"],
            }
        )

        self.assertEqual(command[1:3], ["-m", "rlab.train"])
        self.assertIn("--info-events-json", command)
        self.assertIn('{"life_loss":["lives","decrease"]}', command)
        self.assertIn("--done-on-events", command)
        self.assertIn("life_loss", command)

    def test_field_validation_uses_choices_and_numeric_bounds(self) -> None:
        validate_train_config_fields(
            {"wandb_mode": "online", "frame_skip": 1},
            keys=("wandb_mode", "frame_skip"),
        )

        with self.assertRaisesRegex(ValueError, "wandb_mode.*one of online, offline, disabled"):
            validate_train_config_value("wandb_mode", "maybe")
        with self.assertRaisesRegex(ValueError, "frame_skip.*>= 1"):
            validate_train_config_value("frame_skip", 0)

    def test_field_validation_uses_sequence_and_crop_shapes(self) -> None:
        validate_train_config_fields(
            {
                "states": ["Level1-1", "Level1-2"],
                "done_on_events": [],
                "obs_crop": [32, 0, 0, 0],
                "task_conditioning_info_values": [[0, 0], [0, 1]],
            },
            keys=("states", "done_on_events", "obs_crop", "task_conditioning_info_values"),
        )

        with self.assertRaisesRegex(ValueError, "states must be a list"):
            validate_train_config_value("states", "Level1-1")
        with self.assertRaisesRegex(ValueError, r"obs_crop\[0\].*non-negative integer"):
            validate_train_config_value("obs_crop", [-1, 0, 0, 0])
        with self.assertRaisesRegex(ValueError, r"task_conditioning_info_values\[0\]"):
            validate_train_config_value("task_conditioning_info_values", ["0,0"])


if __name__ == "__main__":
    unittest.main()
