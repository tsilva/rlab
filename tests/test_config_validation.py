from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlab.config_validation import (
    load_goal_contract,
    main as validate_main,
    validate_experiment_tree,
    validate_goal_contract,
)
from rlab.main import COMMANDS


class ConfigValidationTests(unittest.TestCase):
    def test_checked_in_experiment_tree_validates(self) -> None:
        report = validate_experiment_tree(Path("."))

        self.assertEqual(report.issues, ())
        self.assertEqual(report.counts["json_files"], 0)
        self.assertGreaterEqual(report.counts["yaml_files"], 15)
        self.assertGreaterEqual(report.counts["train_specs"], 1)
        self.assertGreaterEqual(report.counts["goals"], 1)
        self.assertGreaterEqual(report.counts["env_configs"], 1)
        self.assertGreaterEqual(report.counts["benchmark_profiles"], 7)

    def test_goal_validator_accepts_goal_without_default_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "bad"
            goal_dir.mkdir(parents=True)
            goal_path = goal_dir / "goal.yaml"
            goal_path.write_text(
                """
goal_id: bad
title: Bad Goal
objective:
  states: [Level1-1]
  rank:
  - train/info/level_complete/rate/min/last
train:
  early_stop:
  - metric: train/info/level_complete/rate/min/last
    operator: '>'
    threshold: 0.99
  environment:
    env_config:
      env_provider: stable-retro-turbo
      game: SuperMarioBros-Nes-v0
      state: Level1-1
      action_set: simple
      frame_skip: 4
      max_pool_frames: false
      sticky_action_prob: 0.0
      observation_size: 84
      hud_crop_top: 32
      obs_resize_algorithm: area
      max_episode_steps: 4500
      info_events:
        life_loss: [lives, decrease]
        level_change: [[levelHi, levelLo], change]
      done_on_events: [life_loss, level_change]
eval:
  environment:
    env_config:
      env_provider: stable-retro-turbo
      game: SuperMarioBros-Nes-v0
      action_set: simple
      frame_skip: 4
      max_pool_frames: false
      sticky_action_prob: 0.0
      observation_size: 84
      hud_crop_top: 32
      obs_resize_algorithm: area
      max_episode_steps: 4500
      info_events:
        life_loss: [lives, decrease]
        level_change: [[levelHi, levelLo], change]
      max_episodes: 100
      done_on_events: [level_change]
""",
                encoding="utf-8",
            )

            validate_goal_contract(goal_path, root)

    def test_goal_validator_requires_slug_to_match_goal_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "real-goal"
            goal_dir.mkdir(parents=True)
            goal_path = goal_dir / "goal.yaml"
            goal_path.write_text(
                """
goal_id: stale-short-name
title: Bad Goal
objective: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "goal_id.*must match goal directory name: real-goal"):
                validate_goal_contract(goal_path, root)

    def test_goal_validator_rejects_objective_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "bad"
            goal_dir.mkdir(parents=True)
            goal_path = goal_dir / "goal.yaml"
            goal_path.write_text(
                """
goal_id: bad
title: Bad Goal
objective:
  success:
    metric: train/info/level_complete/rate/min/last
    operator: '>'
    threshold: 0.99
  rank:
  - train/info/level_complete/rate/min/last
train:
  environment:
    env_config:
      env_provider: stable-retro-turbo
      game: SuperMarioBros-Nes-v0
      state: Level1-1
      action_set: simple
      frame_skip: 4
      max_pool_frames: false
      sticky_action_prob: 0.0
      observation_size: 84
      hud_crop_top: 32
      obs_resize_algorithm: area
      max_episode_steps: 4500
      info_events:
        life_loss: [lives, decrease]
        level_change: [[levelHi, levelLo], change]
      done_on_events: [life_loss, level_change]
eval:
  environment:
    env_config:
      env_provider: stable-retro-turbo
      game: SuperMarioBros-Nes-v0
      action_set: simple
      frame_skip: 4
      max_pool_frames: false
      sticky_action_prob: 0.0
      observation_size: 84
      hud_crop_top: 32
      obs_resize_algorithm: area
      max_episode_steps: 4500
      info_events:
        life_loss: [lives, decrease]
        level_change: [[levelHi, levelLo], change]
      max_episodes: 100
      done_on_events: [level_change]
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "objective\\.success moved to train\\.early_stop"):
                validate_goal_contract(goal_path, root)

    def test_goal_validator_rejects_environment_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "bad"
            goal_dir.mkdir(parents=True)
            goal_path = goal_dir / "goal.yaml"
            goal_path.write_text(
                """
goal_id: bad
title: Bad Goal
objective:
  states: [Level1-1]
  rank:
  - train/info/level_complete/rate/min/last
train:
  early_stop:
  - metric: train/info/level_complete/rate/min/last
    operator: '>'
    threshold: 0.99
  environment:
    env_config:
      env_provider: stable-retro-turbo
      game: SuperMarioBros-Nes-v0
      state: Level1-1
      action_set: simple
      frame_skip: 4
      observation_size: 84
      hud_crop_top: 32
      max_episode_steps: 4500
environment_hash: sha256:deadbeef
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "environment_hash"):
                validate_goal_contract(goal_path, root)

    def test_load_goal_contract_returns_composed_document(self) -> None:
        document = load_goal_contract(Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml"))

        self.assertNotIn("extends", document)
        self.assertNotIn("schema_version", document)
        self.assertNotIn("status", document)
        self.assertEqual(document["goal_id"], "Level1-1")
        self.assertNotIn("seed_protocol", document)
        self.assertNotIn("historical_context", document)
        self.assertNotIn("updated_at", document)
        self.assertNotIn("notes", document)
        self.assertNotIn("runtime", document)
        self.assertNotIn("search_protocol", document)
        self.assertNotIn("batch_record_fields", document)
        self.assertNotIn("capacity_policy_file", document)
        self.assertNotIn("cap_policy", document)
        self.assertNotIn("constraints", document)
        self.assertNotIn("default_eval_profile", document)
        self.assertNotIn("default_train_profile", document)
        self.assertNotIn("environment_hash", document)
        self.assertNotIn("execution", document)
        self.assertNotIn("game", document["objective"])
        self.assertNotIn("algorithm", document["objective"])
        self.assertNotIn("states", document["objective"])
        self.assertNotIn("forbidden_stop_rules", document["objective"])
        self.assertNotIn("max_train_timesteps", document["objective"])
        self.assertNotIn("success", document["objective"])
        self.assertEqual(
            document["train"]["early_stop"],
            [
                {
                    "metric": "train/info/level_complete/rate/min/last",
                    "operator": ">",
                    "threshold": 0.99,
                }
            ],
        )
        self.assertEqual(
            document["objective"]["rank"],
            [
                "max(eval/done/level_change/from_rate/min)",
                "max(eval/done/level_change/from_rate/mean)",
                "min(leader/checkpoint/steps_to_completion_goal)",
                "max(eval/reward/mean)",
            ],
        )
        self.assertNotIn("selection_policy", document)
        self.assertNotIn("max_train_timesteps", document["train"])
        self.assertEqual(
            document["train"]["environment"]["env_provider"],
            "supermariobrosnes-turbo",
        )
        self.assertNotIn("env_provider", document["train"]["environment"]["env_config"])
        self.assertEqual(document["train"]["environment"]["env_config"]["game"], "SuperMarioBros-Nes-v0")
        self.assertEqual(document["train"]["environment"]["env_config"]["state"], "Level1-1")
        self.assertEqual(document["train"]["environment"]["env_config"]["num_envs"], 16)
        self.assertEqual(document["train"]["environment"]["env_config"]["num_threads"], 4)
        self.assertNotIn("reward_mode", document["train"]["environment"]["env_config"])
        self.assertEqual(document["train"]["environment"]["env_config"]["obs_crop"], [32, 0, 0, 0])
        self.assertEqual(document["train"]["environment"]["env_config"]["obs_resize"], [84, 84])
        self.assertEqual(document["train"]["environment"]["env_config"]["maxpool_last_two"], False)
        self.assertEqual(document["train"]["environment"]["env_config"]["sticky_action_prob"], 0.0)
        self.assertNotIn("policy", document["eval"])
        self.assertNotIn("schema_version", document["eval"])
        self.assertEqual(
            document["eval"]["environment"]["env_provider"],
            "supermariobrosnes-turbo",
        )
        self.assertNotIn("env_provider", document["eval"]["environment"]["env_config"])
        self.assertEqual(document["eval"]["environment"]["env_config"]["game"], "SuperMarioBros-Nes-v0")
        self.assertEqual(document["eval"]["environment"]["env_config"]["num_envs"], 16)
        self.assertEqual(document["eval"]["environment"]["env_config"]["num_threads"], 4)
        self.assertNotIn("reward_mode", document["eval"]["environment"]["env_config"])
        self.assertEqual(document["eval"]["environment"]["env_config"]["obs_crop"], [32, 0, 0, 0])
        self.assertEqual(document["eval"]["environment"]["env_config"]["obs_resize"], [84, 84])
        self.assertEqual(document["eval"]["environment"]["env_config"]["max_episodes"], 100)
        self.assertNotIn("episodes", document["eval"]["environment"]["env_config"])
        self.assertNotIn("seed", document["eval"]["environment"]["env_config"])
        self.assertNotIn("n_envs", document["eval"]["environment"]["env_config"])
        self.assertNotIn("max_steps", document["eval"]["environment"]["env_config"])
        self.assertEqual(
            document["eval"]["environment"]["env_config"]["done_on"],
            ["level_change"],
        )

    def test_validate_is_registered_on_unified_cli(self) -> None:
        self.assertIn("validate", COMMANDS)

    def test_validate_cli_success(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = validate_main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("YAML config validation passed", stdout.getvalue())

    def test_validate_cli_load_goal_emits_composed_json(self) -> None:
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            exit_code = validate_main(
                [
                    "--load-goal",
                    "experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        document = json.loads(stdout.getvalue())
        self.assertNotIn("extends", document)
        self.assertNotIn("schema_version", document)
        self.assertNotIn("status", document)
        self.assertEqual(document["goal_id"], "Level1-1")
        self.assertNotIn("goal_dir", document)
        self.assertNotIn("seed_protocol", document)
        self.assertNotIn("historical_context", document)
        self.assertNotIn("updated_at", document)
        self.assertEqual(document["train"]["environment"]["env_config"]["game"], "SuperMarioBros-Nes-v0")
        self.assertNotIn("execution", document)


if __name__ == "__main__":
    unittest.main()
