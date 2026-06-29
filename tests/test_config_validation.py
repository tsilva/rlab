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
            (root / "experiments" / "policies").mkdir(parents=True)
            (root / "experiments" / "instances.yaml").write_text("instances: {}\n", encoding="utf-8")
            (root / "experiments" / "fleet.yaml").write_text("hosts: {}\n", encoding="utf-8")
            (root / "experiments" / "policies" / "capacity_policy.yaml").write_text(
                "schema_version: 1\nlanes: []\n",
                encoding="utf-8",
            )
            goal_path = goal_dir / "goal.yaml"
            goal_path.write_text(
                """
schema_version: 1
goal_slug: bad
title: Bad Goal
status: draft
objective:
  game: SuperMarioBros-Nes-v0
  states: [Level1-1]
  algorithm: PPO
  primary_metric: train/info/level_complete/rate/min/last
  success_threshold: 1.0
  success_window_attempts: 100
  max_train_timesteps: 5000000
environment:
  provider: stable_retro
  env_id: SuperMarioBros-Nes-v0
  state:
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
    copy_observations: false
    policy_observation_layout: channel_first
  termination:
    max_episode_steps: 4500
    completion_x_threshold: 0
    info_events_json:
      life_loss: [lives, decrease]
      level_change: [[levelHi, levelLo], change]
    done_on_events: [life_loss, level_change]
environment_hash: sha256:ce3af6d41b4ef1c0d953f1c5edcb1734c2846208b16cfada98a22ccefa46764f
selection_policy:
  rank_order: [train/info/level_complete/rate/min/last]
capacity_policy_file: experiments/policies/capacity_policy.yaml
execution:
  hardware_config_file: experiments/instances.yaml
  fleet_config_file: experiments/fleet.yaml
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
goal_slug: stale-short-name
title: Bad Goal
status: draft
objective: {}
selection_policy: {}
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "goal_slug.*must match goal directory name: real-goal"):
                validate_goal_contract(goal_path, root)

    def test_goal_validator_rejects_stale_environment_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "bad"
            goal_dir.mkdir(parents=True)
            (root / "experiments" / "policies").mkdir(parents=True)
            (root / "experiments" / "instances.yaml").write_text("instances: {}\n", encoding="utf-8")
            (root / "experiments" / "fleet.yaml").write_text("hosts: {}\n", encoding="utf-8")
            (root / "experiments" / "policies" / "capacity_policy.yaml").write_text(
                "schema_version: 1\nlanes: []\n",
                encoding="utf-8",
            )
            goal_path = goal_dir / "goal.yaml"
            goal_path.write_text(
                """
schema_version: 1
goal_slug: bad
title: Bad Goal
status: draft
objective:
  game: SuperMarioBros-Nes-v0
  states: [Level1-1]
  algorithm: PPO
  primary_metric: train/info/level_complete/rate/min/last
  success_threshold: 1.0
  success_window_attempts: 100
  max_train_timesteps: 5000000
environment:
  provider: stable_retro
  env_id: SuperMarioBros-Nes-v0
  state:
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
capacity_policy_file: experiments/policies/capacity_policy.yaml
execution:
  hardware_config_file: experiments/instances.yaml
  fleet_config_file: experiments/fleet.yaml
""",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "environment_hash"):
                validate_goal_contract(goal_path, root)

    def test_load_goal_contract_returns_composed_document(self) -> None:
        document = load_goal_contract(Path("experiments/goals/Level1-3/goal.yaml"))

        self.assertNotIn("extends", document)
        self.assertEqual(document["goal_slug"], "Level1-3")
        self.assertNotIn("seed_protocol", document)
        self.assertNotIn("historical_context", document)
        self.assertNotIn("updated_at", document)
        self.assertEqual(document["objective"]["game"], "SuperMarioBros-Nes-v0")
        self.assertNotIn("states", document["objective"])
        self.assertEqual(document["environment"]["provider"], "stable_retro")
        self.assertEqual(document["environment"]["state"]["state"], "Level1-3")
        self.assertNotIn("hud_crop_top", document["environment"]["preprocessing"])
        self.assertEqual(document["environment"]["preprocessing"]["obs_crop"], [32, 0, 0, 0])
        self.assertNotIn("observation_size", document["environment"]["preprocessing"])
        self.assertEqual(document["environment"]["preprocessing"]["obs_resize"], [84, 84])
        self.assertEqual(document["execution"]["primary_train_host"], "beast-3")

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
        self.assertEqual(document["goal_slug"], "Level1-3")
        self.assertNotIn("goal_dir", document)
        self.assertNotIn("seed_protocol", document)
        self.assertNotIn("historical_context", document)
        self.assertNotIn("updated_at", document)
        self.assertEqual(document["environment"]["env_id"], "SuperMarioBros-Nes-v0")
        self.assertEqual(document["execution"]["primary_train_target"], "rtx4090")


if __name__ == "__main__":
    unittest.main()
