from __future__ import annotations

import argparse
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import gymnasium as gym
import numpy as np

from stable_retro_ppo.artifacts import build_s3_artifact_uri, checkpoint_step
from stable_retro_ppo.callbacks import ThroughputCallback
from stable_retro_ppo.cli import build_train_command
from stable_retro_ppo.env import EnvConfig, needs_vec_transpose_image
from stable_retro_ppo.env_config import env_config_from_args, parse_states
from stable_retro_ppo.eval_metrics import episode_rank
from stable_retro_ppo.eval_profiles import get_eval_profile
from stable_retro_ppo.eval_runner import evaluate_model_episodes
from stable_retro_ppo.targets import SuperMarioBrosNesV0Target, target_for_game
from stable_retro_ppo.wandb_artifacts import artifact_download_dir, model_artifact_ref, safe_artifact_stem
from scripts.eval_wandb_checkpoints import eval_seed_for_checkpoint


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
            completion_x_threshold=SuperMarioBrosNesV0Target.default_completion_x_threshold,
            terminate_on_life_loss=True,
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

    def test_mario_target_declares_native_life_variable(self) -> None:
        self.assertEqual(SuperMarioBrosNesV0Target.native_life_variable, "lives")

    def test_unknown_target_defaults_to_native(self) -> None:
        target = target_for_game("SonicTheHedgehog-Genesis")
        self.assertEqual(target.default_action_set, "native")
        self.assertEqual(target.action_names_for_set("native"), ())
        self.assertIsNone(target.native_life_variable)

    def test_native_life_loss_marks_death_without_python_termination(self) -> None:
        config = argparse.Namespace(
            reward_mode="baseline",
            no_progress_min_delta=0,
            completion_x_threshold=3160,
            terminate_on_level_change=False,
            terminate_on_completion=False,
            terminate_on_life_loss=False,
            max_episode_steps=0,
            no_progress_timeout_steps=0,
            progress_reward_cap=30.0,
            terminal_reward=50.0,
            reward_scale=10.0,
            time_penalty=0.0,
            completion_reward=0.0,
            death_penalty=25.0,
            score_progress_clipped=False,
            use_retro_reward=False,
        )
        tracker = SuperMarioBrosNesV0Target.create_tracker(config)
        tracker.reset({"lives": 3, "score": 0, "levelHi": 1, "levelLo": 1})
        info = {
            "lives": 3,
            "score": 0,
            "levelHi": 1,
            "levelLo": 1,
            "xscrollHi": 0,
            "xscrollLo": 0,
            "life_loss": True,
        }

        progress = tracker.step(0.0, info, done=True)

        self.assertFalse(progress.done)
        self.assertTrue(info["died"])
        self.assertEqual(info["raw_reward"], -50.0)


class VecImageShapeTests(unittest.TestCase):
    def test_channel_last_native_observations_need_transpose(self) -> None:
        space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        self.assertTrue(needs_vec_transpose_image(space))

    def test_channel_first_native_observations_skip_transpose(self) -> None:
        space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
        self.assertFalse(needs_vec_transpose_image(space))

    def test_unexpected_image_shape_fails_loudly(self) -> None:
        space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 8), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "could not infer"):
            needs_vec_transpose_image(space)


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
        self.assertEqual(checkpoint_step(Path("ppo_retro_123456_steps.zip")), 123456)
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

    def test_s3_artifact_uri_includes_wandb_rom_id_prefix(self) -> None:
        args = argparse.Namespace(game="TestGame-Platform", run_name="candidate/run")
        self.assertEqual(
            build_s3_artifact_uri("s3://wandb", args, Path("final_model.zip"), "final"),
            "s3://wandb/TestGame-Platform/candidate-run/final/final_model.zip",
        )
        self.assertEqual(
            build_s3_artifact_uri(
                "s3://wandb/TestGame-Platform",
                args,
                Path("ppo_test_100_steps.zip"),
                "checkpoint",
            ),
            "s3://wandb/TestGame-Platform/candidate-run/checkpoint/ppo_test_100_steps.zip",
        )


class EvalMetricTests(unittest.TestCase):
    def test_episode_rank_prefers_completion_then_progress_then_reward(self) -> None:
        incomplete = {"level_complete": False, "max_x_pos": 4000, "reward": 1000.0}
        complete = {"level_complete": True, "max_x_pos": 100, "reward": -10.0}
        better_progress = {"level_complete": False, "max_x_pos": 4500, "reward": 0.0}
        self.assertGreater(episode_rank(complete), episode_rank(incomplete))
        self.assertGreater(episode_rank(better_progress), episode_rank(incomplete))

    def test_checkpoint_eval_seed_defaults_to_paired_schedule(self) -> None:
        args = argparse.Namespace(seed=10007, seed_offset_by_checkpoint_step=False)

        self.assertEqual(eval_seed_for_checkpoint(args, 4_400_000), 10007)

    def test_checkpoint_eval_seed_legacy_step_offset(self) -> None:
        args = argparse.Namespace(seed=10007, seed_offset_by_checkpoint_step=True)

        self.assertEqual(eval_seed_for_checkpoint(args, 4_400_000), 4_410_007)

    def test_vector_eval_accumulates_completed_slots_independently(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.zeros(obs.shape[0], dtype=np.int64), None

        class FakeVecEnv:
            num_envs = 2

            def __init__(self) -> None:
                self.step_count = 0

            def reset(self):
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                obs = np.zeros((2, 4, 84, 84), dtype=np.uint8)
                if self.step_count == 1:
                    return (
                        obs,
                        np.array([1.0, 2.0], dtype=np.float32),
                        np.array([False, True]),
                        [
                            {"max_x_pos": 10, "level_max_x_pos": 10},
                            {
                                "max_x_pos": 20,
                                "level_max_x_pos": 20,
                                "died": True,
                                "death_x_pos": 20,
                                "score": 100,
                                "lives": 2,
                            },
                        ],
                    )
                return (
                    obs,
                    np.array([3.0, 4.0], dtype=np.float32),
                    np.array([True, False]),
                    [
                        {
                            "max_x_pos": 30,
                            "level_max_x_pos": 30,
                            "score": 200,
                            "lives": 3,
                        },
                        {"max_x_pos": 40, "level_max_x_pos": 40},
                    ],
                )

            def close(self) -> None:
                pass

        config = EnvConfig(game="SuperMarioBros-Nes-v0", completion_x_threshold=25)
        with patch("stable_retro_ppo.eval_runner.make_eval_vec_env", return_value=FakeVecEnv()):
            metrics, video_path = evaluate_model_episodes(
                model=FakeModel(),
                config=config,
                episodes=2,
                seed=7,
                max_steps=10,
                deterministic=True,
                completion_x_threshold=25,
                n_envs=2,
            )

        self.assertIsNone(video_path)
        self.assertEqual(metrics["eval_n_envs"], 2)
        self.assertEqual(metrics["episodes"], 2)
        self.assertEqual(metrics["reward_mean"], 3.0)
        self.assertEqual(metrics["completion_count"], 1)
        self.assertEqual(metrics["death_count"], 1)
        self.assertEqual(metrics["episode_results"][0]["env_index"], 1)
        self.assertEqual(metrics["episode_results"][0]["reward"], 2.0)
        self.assertEqual(metrics["episode_results"][1]["env_index"], 0)
        self.assertEqual(metrics["episode_results"][1]["reward"], 4.0)


class EvalProfileTests(unittest.TestCase):
    def test_mario_level1_profile_pins_eval_contract(self) -> None:
        profile = get_eval_profile("mario_level1_v1")
        config = profile.env_config()

        self.assertEqual(profile.n_envs, 1)
        self.assertFalse(profile.deterministic)
        self.assertEqual(profile.frame_stack, 4)
        self.assertTrue(profile.grayscale)
        self.assertEqual(config.game, "SuperMarioBros-Nes-v0")
        self.assertEqual(config.state, "Level1-1")
        self.assertEqual(config.action_set, "simple")
        self.assertEqual(config.hud_crop_top, 32)
        self.assertEqual(config.completion_x_threshold, 3160)
        self.assertTrue(config.terminate_on_life_loss)
        self.assertTrue(config.terminate_on_completion)

    def test_mario_level1_no_life_loss_profile_only_changes_life_loss_terminal(self) -> None:
        baseline = get_eval_profile("mario_level1_v1")
        profile = get_eval_profile("mario_level1_no_life_loss_v1")
        config = profile.env_config()

        baseline_metadata = baseline.metadata()
        profile_metadata = profile.metadata()
        for key in ("name", "terminate_on_life_loss"):
            baseline_metadata.pop(key)
            profile_metadata.pop(key)
        self.assertEqual(profile_metadata, baseline_metadata)
        self.assertFalse(config.terminate_on_life_loss)
        self.assertTrue(config.terminate_on_completion)

    def test_mario_level1_vector_profiles_only_change_env_count(self) -> None:
        baseline = get_eval_profile("mario_level1_v1")

        for profile_name, expected_n_envs in (
            ("mario_level1_vec8_v1", 8),
            ("mario_level1_vec20_v1", 20),
            ("mario_level1_vec24_v1", 24),
        ):
            with self.subTest(profile_name=profile_name):
                profile = get_eval_profile(profile_name)
                baseline_metadata = baseline.metadata()
                profile_metadata = profile.metadata()
                for key in ("name", "n_envs"):
                    baseline_metadata.pop(key)
                    profile_metadata.pop(key)
                self.assertEqual(profile_metadata, baseline_metadata)
                self.assertEqual(profile.n_envs, expected_n_envs)


class ThroughputCallbackTests(unittest.TestCase):
    def test_logs_rollout_fps_and_next_iteration_instant_fps(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: list[tuple[str, float]] = []

            def record(self, key: str, value: float) -> None:
                self.records.append((key, value))

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()

        times = iter([0.0, 2.0, 5.0, 7.0])
        callback = ThroughputCallback(clock=lambda: next(times))
        model = Model()
        callback.model = model  # type: ignore[assignment]

        callback.num_timesteps = 0
        callback._on_rollout_start()
        callback.num_timesteps = 100
        callback._on_rollout_end()

        callback.num_timesteps = 100
        callback._on_rollout_start()
        callback.num_timesteps = 220
        callback._on_rollout_end()

        self.assertEqual(
            model.logger.records,
            [
                ("time/rollout_fps", 50.0),
                ("time/rollout_fps", 60.0),
                ("time/fps_instant", 20.0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
