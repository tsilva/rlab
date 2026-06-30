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
        self.assertGreaterEqual(report.counts["yaml_files"], 196)
        self.assertGreaterEqual(report.counts["train_specs"], 179)
        self.assertGreaterEqual(report.counts["goals"], 5)
        self.assertGreaterEqual(report.counts["benchmark_profiles"], 7)

    def test_goal_validator_accepts_goal_without_default_spec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "bad"
            goal_dir.mkdir(parents=True)
            goal_path = goal_dir / "goal.yaml"
            goal_path.write_text(
                """
schema_version: 1
goal_id: bad
title: Bad Goal
status: draft
objective:
  states: [Level1-1]
  primary_metric: train/info/level_complete/rate/min/last
  success_threshold: 1.0
  success_window_attempts: 100
environment:
  max_train_timesteps: 5000000
  env_id: stable-retro-turbo:SuperMarioBros-Nes-v0
  state: Level1-1
  action:
    action_set: simple
  preprocessing:
    pipeline: stable_retro_native_vec_env
    frame_skip: 4
    frame_stack: 4
    max_pool_frames: false
    sticky_action_prob: 0.0
    obs_resize: [84, 84]
    obs_crop: [32, 0, 0, 0]
    obs_grayscale: true
    obs_resize_algorithm: area
    obs_copy: safe_view
    policy_observation_layout: channel_first
  termination:
    max_episode_steps: 4500
    completion_x_threshold: 0
    info_events_json:
      life_loss: [lives, decrease]
      level_change: [[levelHi, levelLo], change]
    done_on_events: [life_loss, level_change]
selection_policy:
  rank_order: [train/info/level_complete/rate/min/last]
eval_spec:
  schema_version: 1
  eval_config:
    episodes: 100
    seed: 10007
    n_envs: 20
    max_steps: 4500
    stochastic: true
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
schema_version: 1
goal_id: stale-short-name
title: Bad Goal
status: draft
objective: {}
selection_policy: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "goal_id.*must match goal directory name: real-goal"):
                validate_goal_contract(goal_path, root)

    def test_goal_validator_rejects_environment_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "bad"
            goal_dir.mkdir(parents=True)
            goal_path = goal_dir / "goal.yaml"
            goal_path.write_text(
                """
schema_version: 1
goal_id: bad
title: Bad Goal
status: draft
objective:
  states: [Level1-1]
  primary_metric: train/info/level_complete/rate/min/last
  success_threshold: 1.0
  success_window_attempts: 100
  max_train_timesteps: 5000000
environment:
  env_id: stable-retro-turbo:SuperMarioBros-Nes-v0
  state: Level1-1
  action:
    action_set: simple
  preprocessing:
    frame_skip: 4
    obs_resize: [84, 84]
    obs_crop: [32, 0, 0, 0]
  termination:
    max_episode_steps: 4500
environment_hash: sha256:deadbeef
selection_policy:
  rank_order: [train/info/level_complete/rate/min/last]
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "environment_hash"):
                validate_goal_contract(goal_path, root)

    def test_load_goal_contract_returns_composed_document(self) -> None:
        document = load_goal_contract(Path("experiments/goals/Level1-3/goal.yaml"))

        self.assertNotIn("extends", document)
        self.assertEqual(document["goal_id"], "Level1-3")
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
        self.assertEqual(document["environment"]["env_id"], "stable-retro-turbo:SuperMarioBros-Nes-v0")
        self.assertEqual(document["environment"]["max_train_timesteps"], 5000000)
        self.assertEqual(document["environment"]["state"], "Level1-3")
        self.assertNotIn("hud_crop_top", document["environment"]["preprocessing"])
        self.assertEqual(document["environment"]["preprocessing"]["obs_crop"], [32, 0, 0, 0])
        self.assertNotIn("observation_size", document["environment"]["preprocessing"])
        self.assertEqual(document["environment"]["preprocessing"]["obs_resize"], [84, 84])
        self.assertEqual(document["eval_spec"]["eval_config"]["episodes"], 100)
        self.assertEqual(document["eval_spec"]["eval_config"]["done_on_events"], ["level_change"])
        self.assertEqual(set(document["selection_policy"]), {"rank_order"})

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
                    "experiments/goals/Level1-3/goal.yaml",
                    "--json",
                ]
            )

        self.assertEqual(exit_code, 0)
        document = json.loads(stdout.getvalue())
        self.assertNotIn("extends", document)
        self.assertEqual(document["goal_id"], "Level1-3")
        self.assertNotIn("goal_dir", document)
        self.assertNotIn("seed_protocol", document)
        self.assertNotIn("historical_context", document)
        self.assertNotIn("updated_at", document)
        self.assertEqual(document["environment"]["env_id"], "stable-retro-turbo:SuperMarioBros-Nes-v0")
        self.assertNotIn("execution", document)


if __name__ == "__main__":
    unittest.main()
