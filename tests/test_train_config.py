from __future__ import annotations

import argparse
import json
import unittest

from rlab.env import EnvConfig
from rlab.train_config import (
    add_env_config_args,
    add_train_config_args,
    build_train_command_from_fields,
    train_config_field_for_key,
    validate_train_config_fields,
    validate_train_config_value,
)


class TrainConfigFieldSchemaTests(unittest.TestCase):
    def test_train_and_eval_parsers_share_env_field_behavior(self) -> None:
        train_parser = argparse.ArgumentParser()
        eval_parser = argparse.ArgumentParser()
        parser_kwargs = {
            "parse_json_value": json.loads,
            "parse_obs_crop": lambda value: tuple(int(item) for item in value.split(",")),
        }
        add_train_config_args(train_parser, env_defaults=EnvConfig(), **parser_kwargs)
        add_env_config_args(
            eval_parser,
            max_steps_default=987,
            defaults=EnvConfig(),
            **parser_kwargs,
        )

        args = [
            "--env-provider",
            "ale-py",
            "--env-args",
            '{"game":"breakout"}',
            "--no-episodic-life",
            "--obs-crop",
            "1,2,3,4",
        ]
        train_args = train_parser.parse_args(args)
        eval_args = eval_parser.parse_args([*args, "--max-steps", "123"])

        for dest in ("env_provider", "env_args", "episodic_life", "obs_crop"):
            self.assertEqual(getattr(train_args, dest), getattr(eval_args, dest))
        self.assertEqual(eval_args.max_steps, 123)
        self.assertFalse(hasattr(eval_args, "max_episode_steps"))

    def test_env_config_aliases_resolve_to_train_config_fields(self) -> None:
        field = train_config_field_for_key("info_events")

        self.assertIsNotNone(field)
        self.assertEqual(field.dest, "info_events_json")
        self.assertEqual(field.env_config_key, "info_events")

    def test_legacy_checkpoint_eval_n_envs_alias_resolves_to_canonical_field(self) -> None:
        field = train_config_field_for_key("post_train_eval_n_envs")

        self.assertIsNotNone(field)
        self.assertEqual(field.dest, "checkpoint_eval_n_envs")

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

    def test_build_train_command_emits_checkpoint_eval_n_envs_for_legacy_alias(self) -> None:
        command = build_train_command_from_fields({"post_train_eval_n_envs": 4})

        self.assertIn("--checkpoint-eval-n-envs", command)
        self.assertIn("4", command)
        self.assertNotIn("--post-train-eval-n-envs", command)

    def test_build_train_command_emits_checkpoint_eval_stages_json(self) -> None:
        stages = [
            {
                "name": "screen",
                "episodes": 10,
                "n_envs": 2,
                "pass": [
                    {
                        "metric": "eval/info/level_complete/rate/min",
                        "operator": ">=",
                        "threshold": 1.0,
                    }
                ],
            }
        ]
        command = build_train_command_from_fields({"checkpoint_eval_stages": stages})

        self.assertIn("--checkpoint-eval-stages", command)
        rendered = command[command.index("--checkpoint-eval-stages") + 1]
        self.assertIn('"name":"screen"', rendered)
        self.assertIn('"episodes":10', rendered)
        self.assertIn('"metric":"eval/info/level_complete/rate/min"', rendered)

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
