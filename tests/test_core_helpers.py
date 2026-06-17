from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from mario_ppo.artifacts import checkpoint_step
from mario_ppo.cli import build_train_command
from mario_ppo.env_config import env_config_from_args, parse_states
from mario_ppo.eval_metrics import episode_rank
from mario_ppo.targets import SuperMarioBrosNesV0Target, target_for_game
from mario_ppo.wandb_artifacts import artifact_download_dir, model_artifact_ref, safe_artifact_stem


class EnvConfigFromArgsTests(unittest.TestCase):
    def test_parse_states_trims_empty_values(self) -> None:
        self.assertEqual(parse_states("A, B, ,C"), ("A", "B", "C"))

    def test_eval_max_steps_maps_to_env_max_episode_steps(self) -> None:
        args = argparse.Namespace(
            game="SuperMarioBros-Nes-v0",
            state="Level1-1",
            frame_skip=4,
            max_pool_frames=True,
            max_steps=123,
            hud_crop_top=32,
            reward_mode="baseline",
            progress_reward_cap=30.0,
            progress_reward_scale=1.0,
            terminal_reward=50.0,
            reward_scale=10.0,
            time_penalty=0.0,
            death_penalty=25.0,
            completion_reward=0.0,
            score_progress_clipped=False,
            no_progress_timeout_steps=0,
            no_progress_min_delta=0,
            completion_x_threshold=3160,
            no_terminate_on_life_loss=False,
            terminate_on_level_change=False,
            terminate_on_completion=False,
            action_set="right",
        )
        config = env_config_from_args(args, max_episode_steps_attr="max_steps")
        self.assertEqual(config.max_episode_steps, 123)
        self.assertEqual(config.action_set, "right")
        self.assertTrue(config.terminate_on_life_loss)


class TargetTests(unittest.TestCase):
    def test_known_mario_target_is_reused(self) -> None:
        self.assertIs(target_for_game("SuperMarioBros-Nes-v0"), SuperMarioBrosNesV0Target)

    def test_unknown_target_defaults_to_native(self) -> None:
        target = target_for_game("SonicTheHedgehog-Genesis")
        self.assertEqual(target.default_action_set, "native")
        self.assertEqual(target.action_names_for_set("native"), ())


class CommandAndArtifactTests(unittest.TestCase):
    def test_build_train_command_skips_empty_target_kl(self) -> None:
        cmd = build_train_command(
            {
                "run_name": "candidate",
                "target_kl": 0.0,
                "wandb": True,
                "normalize_advantage": False,
            }
        )
        self.assertIn("--run-name", cmd)
        self.assertNotIn("--target-kl", cmd)
        self.assertIn("--wandb", cmd)
        self.assertIn("--no-normalize-advantage", cmd)

    def test_checkpoint_step_from_sb3_checkpoint_name(self) -> None:
        self.assertEqual(checkpoint_step(Path("ppo_mario_123456_steps.zip")), 123456)
        self.assertIsNone(checkpoint_step(Path("final_model.zip")))

    def test_wandb_artifact_paths_are_stable(self) -> None:
        self.assertEqual(safe_artifact_stem("a/b:c"), "a-b-c")
        self.assertEqual(
            model_artifact_ref(
                project="entity/project",
                run_name="run",
                kind="best",
                version="latest",
            ),
            "entity/project/run-best:latest",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(
                artifact_download_dir(Path(tmp_dir), "entity/project/run-best:latest"),
                Path(tmp_dir) / "entity_project_run-best_latest",
            )


class EvalMetricTests(unittest.TestCase):
    def test_episode_rank_prefers_completion_then_progress_then_reward(self) -> None:
        incomplete = {"level_complete": False, "max_x_pos": 4000, "reward": 1000.0}
        complete = {"level_complete": True, "max_x_pos": 100, "reward": -10.0}
        better_progress = {"level_complete": False, "max_x_pos": 4500, "reward": 0.0}
        self.assertGreater(episode_rank(complete), episode_rank(incomplete))
        self.assertGreater(episode_rank(better_progress), episode_rank(incomplete))


if __name__ == "__main__":
    unittest.main()
