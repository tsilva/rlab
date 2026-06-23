from __future__ import annotations

import argparse
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import gymnasium as gym
import numpy as np

from scripts.play_wandb_artifact import (
    append_explicit_env_args,
    build_parser as build_wandb_play_parser,
)
from stable_retro_ppo.artifacts import (
    apply_model_config_defaults,
    apply_config_defaults,
    build_s3_artifact_uri,
    checkpoint_step,
    env_config_from_model_metadata,
    env_config_from_metadata,
    explicit_arg_dests,
    load_model_metadata,
    model_metadata_path,
    require_training_metadata,
    write_model_metadata,
)
from stable_retro_ppo.callbacks import (
    RolloutDiagnosticsCallback,
    ThroughputCallback,
    TrainingCompletionRateStopCallback,
)
from stable_retro_ppo.cli import build_train_command
from stable_retro_ppo.env import (
    EnvConfig,
    StickyAction,
    make_eval_vec_env,
    make_rendered_replay_env,
    make_training_vec_env,
    needs_vec_transpose_image,
    resolve_env_config,
    resolve_mixed_state_config,
)
from stable_retro_ppo.env_config import env_config_from_args, parse_state_probs, parse_states
from stable_retro_ppo.eval_metrics import episode_rank, is_level_complete
from stable_retro_ppo.eval_runner import evaluate_model_episodes
from stable_retro_ppo.metric_names import metric_path_segment
from stable_retro_ppo.play import build_parser as build_play_parser
from stable_retro_ppo.targets import SuperMarioBrosNesV0Target, target_for_game
from stable_retro_ppo.wandb_artifacts import (
    artifact_download_dir,
    model_artifact_ref,
    safe_artifact_stem,
)
from stable_retro_ppo.wandb_artifacts import metadata_from_wandb_artifact
from scripts.eval_wandb_checkpoints import eval_seed_for_checkpoint


class EnvConfigFromArgsTests(unittest.TestCase):
    def test_parse_states_trims_empty_values(self) -> None:
        self.assertEqual(parse_states("A, B,C"), ("A", "B", "C"))

    def test_parse_states_rejects_empty_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty state"):
            parse_states("A, ,C")

    def test_parse_state_probs_normalizes_later_but_validates_positive_finite(self) -> None:
        self.assertEqual(parse_state_probs("1, 3"), (1.0, 3.0))
        with self.assertRaisesRegex(ValueError, "positive finite"):
            parse_state_probs("1,0")

    def test_eval_max_steps_maps_to_env_max_episode_steps(self) -> None:
        args = argparse.Namespace(
            game="SuperMarioBros-Nes-v0",
            state="Level1-1",
            frame_skip=4,
            max_pool_frames=True,
            sticky_action_prob=0.25,
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
        self.assertEqual(config.sticky_action_prob, 0.25)
        self.assertTrue(config.terminate_on_life_loss)

    def test_sticky_action_probability_defaults_to_disabled(self) -> None:
        self.assertEqual(build_play_parser().parse_args([]).sticky_action_prob, 0.0)
        self.assertEqual(build_wandb_play_parser().parse_args(["run"]).sticky_action_prob, 0.0)

    def test_playback_eval_termination_defaults_to_disabled(self) -> None:
        play_args = build_play_parser().parse_args([])
        wandb_args = build_wandb_play_parser().parse_args(["run"])

        for args in (play_args, wandb_args):
            self.assertFalse(args.terminate_on_life_loss)
            self.assertFalse(args.terminate_on_level_change)
            self.assertFalse(args.terminate_on_completion)

    def test_sticky_action_probability_parses_for_playback(self) -> None:
        self.assertEqual(
            build_play_parser().parse_args(["--sticky-action-prob", "0.25"]).sticky_action_prob,
            0.25,
        )
        self.assertEqual(
            build_wandb_play_parser()
            .parse_args(["run", "--sticky-action-prob", "0.25"])
            .sticky_action_prob,
            0.25,
        )

    def test_invalid_sticky_action_probability_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "sticky_action_prob"):
            resolve_env_config(EnvConfig(game="SuperMarioBros-Nes-v0", sticky_action_prob=-0.1))

    def test_eval_vec_env_forces_life_loss_termination_off(self) -> None:
        sentinel = object()
        config = EnvConfig(game="SuperMarioBros-Nes-v0", terminate_on_life_loss=True)

        with patch("stable_retro_ppo.env.make_vec_envs", return_value=sentinel) as make_vec_envs:
            env = make_eval_vec_env(config=config, n_envs=2, seed=7)

        self.assertIs(env, sentinel)
        passed_config = make_vec_envs.call_args.kwargs["config"]
        self.assertFalse(passed_config.terminate_on_life_loss)
        self.assertFalse(passed_config.terminate_on_level_change)
        self.assertFalse(passed_config.terminate_on_completion)

    def test_eval_vec_env_disables_target_default_life_loss_termination(self) -> None:
        sentinel = object()
        config = EnvConfig(game="SuperMarioBros-Nes-v0", terminate_on_life_loss=None)

        with patch("stable_retro_ppo.env.make_vec_envs", return_value=sentinel) as make_vec_envs:
            make_eval_vec_env(config=config, n_envs=1, seed=7)

        passed_config = make_vec_envs.call_args.kwargs["config"]
        self.assertFalse(passed_config.terminate_on_life_loss)
        self.assertFalse(passed_config.terminate_on_level_change)
        self.assertFalse(passed_config.terminate_on_completion)

    def test_training_vec_env_preserves_requested_life_loss_termination(self) -> None:
        sentinel = object()
        config = EnvConfig(game="SuperMarioBros-Nes-v0", terminate_on_life_loss=True)

        with patch("stable_retro_ppo.env.make_vec_envs", return_value=sentinel) as make_vec_envs:
            env = make_training_vec_env(config=config, n_envs=2, seed=7)

        self.assertIs(env, sentinel)
        passed_config = make_vec_envs.call_args.kwargs["config"]
        self.assertTrue(passed_config.terminate_on_life_loss)

    def test_rendered_eval_replay_forces_life_loss_termination_off(self) -> None:
        sentinel = object()
        config = EnvConfig(game="SuperMarioBros-Nes-v0", terminate_on_life_loss=True)

        with patch("stable_retro_ppo.env.make_retro_env", return_value=sentinel) as make_retro_env:
            env = make_rendered_replay_env(config=config, seed=7)

        self.assertIs(env, sentinel)
        passed_config = make_retro_env.call_args.kwargs["config"]
        self.assertFalse(passed_config.terminate_on_life_loss)
        self.assertFalse(passed_config.terminate_on_level_change)
        self.assertFalse(passed_config.terminate_on_completion)

    def test_short_states_requires_one_state_per_env_slot(self) -> None:
        with patch(
            "stable_retro_ppo.env.retro.data.list_states",
            return_value=["Level1-1", "Level1-2"],
        ):
            with self.assertRaisesRegex(ValueError, "exactly one state per env slot"):
                resolve_mixed_state_config(
                    EnvConfig(
                        game="SuperMarioBros-Nes-v0",
                        states=("Level1-1", "Level1-2"),
                    ),
                    n_envs=3,
                )

    def test_state_probs_are_normalized_and_count_checked(self) -> None:
        with patch(
            "stable_retro_ppo.env.retro.data.list_states",
            return_value=["Level1-1", "Level1-2"],
        ):
            config = resolve_mixed_state_config(
                EnvConfig(
                    game="SuperMarioBros-Nes-v0",
                    states=("Level1-1", "Level1-2"),
                    state_probs=(1.0, 3.0),
                ),
                n_envs=8,
            )
            self.assertEqual(config.state_probs, (0.25, 0.75))

            with self.assertRaisesRegex(ValueError, "count must match"):
                resolve_mixed_state_config(
                    EnvConfig(
                        game="SuperMarioBros-Nes-v0",
                        states=("Level1-1", "Level1-2"),
                        state_probs=(1.0,),
                    ),
                    n_envs=8,
                )

    def test_state_probs_require_states(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --states"):
            resolve_mixed_state_config(
                EnvConfig(game="SuperMarioBros-Nes-v0", state_probs=(1.0,)),
                n_envs=1,
            )

    def test_unknown_mixed_state_fails_loudly(self) -> None:
        with patch("stable_retro_ppo.env.retro.data.list_states", return_value=["Level1-1"]):
            with self.assertRaisesRegex(ValueError, "unknown stable-retro state"):
                resolve_mixed_state_config(
                    EnvConfig(game="SuperMarioBros-Nes-v0", states=("Level9-9",)),
                    n_envs=1,
                )


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
            completion_x_threshold=0,
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

    def test_mario_completion_uses_level_change_not_x_threshold(self) -> None:
        config = argparse.Namespace(
            reward_mode="score",
            no_progress_min_delta=0,
            completion_x_threshold=25,
            terminate_on_level_change=False,
            terminate_on_completion=False,
            terminate_on_life_loss=False,
            max_episode_steps=0,
            no_progress_timeout_steps=0,
            progress_reward_cap=30.0,
            progress_reward_scale=1.0,
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

        same_level_info = {
            "lives": 3,
            "score": 0,
            "levelHi": 1,
            "levelLo": 1,
            "xscrollHi": 1,
            "xscrollLo": 0,
        }
        tracker.step(0.0, same_level_info, done=False)

        self.assertFalse(same_level_info["level_complete"])
        self.assertFalse(same_level_info["threshold_complete"])

        next_level_info = {
            "lives": 3,
            "score": 0,
            "levelHi": 1,
            "levelLo": 2,
            "xscrollHi": 0,
            "xscrollLo": 0,
        }
        tracker.step(0.0, next_level_info, done=False)

        self.assertTrue(next_level_info["level_changed"])
        self.assertTrue(next_level_info["level_complete"])


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


class StickyActionTests(unittest.TestCase):
    def test_probability_one_reuses_previous_high_level_action(self) -> None:
        class FakeEnv(gym.Env):
            action_space = gym.spaces.Discrete(4)
            observation_space = gym.spaces.Box(low=0, high=255, shape=(1,), dtype=np.uint8)

            def __init__(self) -> None:
                self.actions: list[int] = []

            def reset(self, **kwargs):
                return np.zeros((1,), dtype=np.uint8), {}

            def step(self, action):
                self.actions.append(int(action))
                return np.zeros((1,), dtype=np.uint8), 0.0, False, False, {}

        env = StickyAction(FakeEnv(), sticky_action_prob=1.0)
        env.reset(seed=7)
        env.step(1)
        env.step(2)
        env.step(3)

        self.assertEqual(env.unwrapped.actions, [1, 1, 1])


class NativeMixedStateVecEnvTests(unittest.TestCase):
    def test_training_vec_env_passes_mixed_states_to_native_vec_env(self) -> None:
        created: list[dict[str, object]] = []

        class FakeNative:
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)

            def __init__(self, game, **kwargs):
                self.game = game
                self.kwargs = kwargs
                created.append(kwargs)

            def seed(self, seed):
                return [seed]

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            terminate_on_life_loss=False,
            states=("Level1-1", "Level1-2"),
            state_probs=(0.5, 0.5),
        )
        with (
            patch("stable_retro_ppo.env.StableRetroNativeVecEnv", FakeNative),
            patch("stable_retro_ppo.env.VecRetroProgressInfo", side_effect=lambda env, config: env),
            patch("stable_retro_ppo.env.VecMonitor", side_effect=lambda env: env),
            patch("stable_retro_ppo.env.maybe_transpose_vec_image", side_effect=lambda env: env),
            patch(
                "stable_retro_ppo.env.retro.data.list_states",
                return_value=["Level1-1", "Level1-2"],
            ),
        ):
            env = make_training_vec_env(config, n_envs=16, seed=7)

        self.assertIsInstance(env, FakeNative)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["num_envs"], 16)
        self.assertIsNone(created[0]["state"])
        self.assertEqual(created[0]["states"], ["Level1-1", "Level1-2"])
        self.assertEqual(created[0]["state_probs"], [0.5, 0.5])


class CommandAndArtifactTests(unittest.TestCase):
    def test_build_train_command_skips_empty_target_kl(self) -> None:
        cmd = build_train_command(
            {
                "run_name": "candidate",
                "states": "Level1-1,Level1-2",
                "state_probs": "1,3",
                "target_kl": 0.0,
                "wandb": True,
                "normalize_advantage": False,
            }
        )
        self.assertIn("--run-name", cmd)
        self.assertIn("--states", cmd)
        self.assertIn("--state-probs", cmd)
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

    def test_model_metadata_sidecar_records_env_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            args = argparse.Namespace(
                run_name="run",
                run_description="description",
            )
            config = EnvConfig(
                game="SuperMarioBros-Nes-v0",
                state="Level1-1",
                states=("Level1-1", "Level1-2"),
                state_probs=(0.25, 0.75),
                max_pool_frames=False,
                observation_size=96,
                hud_crop_top=32,
                action_set="simple",
            )

            path = write_model_metadata(model_path, args, config, kind="checkpoint")

            self.assertEqual(path, model_metadata_path(model_path))
            metadata = load_model_metadata(model_path)
            self.assertEqual(metadata["checkpoint_step"], 100)
            self.assertEqual(metadata["env_config"]["max_pool_frames"], False)
            self.assertEqual(metadata["env_config"]["observation_size"], 96)
            self.assertEqual(metadata["env_config"]["hud_crop_top"], 32)
            self.assertIn("training_metadata", metadata)
            self.assertIn("training_metadata_hash", metadata)
            self.assertEqual(
                metadata["training_metadata"]["preprocessing"]["frame_stack"],
                4,
            )
            self.assertTrue(metadata["training_metadata"]["preprocessing"]["obs_grayscale"])
            self.assertEqual(metadata["env_config"]["state_sampling_mode"], "probability")
            self.assertEqual(metadata["env_config"]["state_probs"], [0.25, 0.75])
            self.assertEqual(
                metadata["env_config"]["state_distribution"],
                [
                    {"state": "Level1-1", "probability": 0.25},
                    {"state": "Level1-2", "probability": 0.75},
                ],
            )
            self.assertEqual(
                require_training_metadata(model_path)["env_config"]["observation_size"],
                96,
            )

    def test_saved_playback_config_applies_unless_cli_overrides(self) -> None:
        parser = build_play_parser()
        parser_defaults = vars(parser.parse_args([]))
        metadata = {
            "env_config": {
                "game": "SuperMarioBros-Nes-v0",
                "max_pool_frames": False,
                "observation_size": 96,
                "hud_crop_top": 32,
            }
        }

        args = parser.parse_args(["--model", "model.zip"])
        apply_config_defaults(args, env_config_from_metadata(metadata), parser_defaults, set())
        self.assertFalse(args.max_pool_frames)
        self.assertEqual(args.observation_size, 96)
        self.assertEqual(args.hud_crop_top, 32)

        argv = ["--model", "model.zip", "--max-pool-frames"]
        args = parser.parse_args(argv)
        explicit_dests = explicit_arg_dests(parser, argv)
        apply_config_defaults(
            args, env_config_from_metadata(metadata), parser_defaults, explicit_dests
        )
        self.assertTrue(args.max_pool_frames)

    def test_model_metadata_defaults_apply_to_env_parser_args(self) -> None:
        parser = build_play_parser()
        parser_defaults = vars(parser.parse_args([]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            write_model_metadata(
                model_path,
                argparse.Namespace(run_name="run", run_description="description"),
                EnvConfig(
                    game="SuperMarioBros-Nes-v0",
                    state="Level2-1",
                    max_pool_frames=False,
                    observation_size=96,
                    score_progress_clipped=True,
                    terminate_on_completion=True,
                ),
                kind="checkpoint",
            )

            argv = ["--model", str(model_path), "--no-terminate-on-completion"]
            args = parser.parse_args(argv)
            explicit_dests = explicit_arg_dests(parser, argv)

            self.assertTrue(
                apply_model_config_defaults(args, model_path, parser_defaults, explicit_dests)
            )
            self.assertEqual(args.state, "Level2-1")
            self.assertFalse(args.max_pool_frames)
            self.assertEqual(args.observation_size, 96)
            self.assertTrue(args.score_progress_clipped)
            self.assertFalse(args.terminate_on_completion)

            config = env_config_from_model_metadata(
                model_path, fallback=EnvConfig(state="fallback")
            )
            self.assertIsNotNone(config)
            assert config is not None
            self.assertEqual(config.state, "Level2-1")
            self.assertEqual(config.observation_size, 96)

    def test_playback_eval_defaults_block_training_termination_metadata(self) -> None:
        parser = build_play_parser()
        parser_defaults = vars(parser.parse_args([]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            write_model_metadata(
                model_path,
                argparse.Namespace(run_name="run", run_description="description"),
                EnvConfig(
                    game="SuperMarioBros-Nes-v0",
                    state="Level1-1",
                    terminate_on_life_loss=True,
                    terminate_on_level_change=True,
                    terminate_on_completion=True,
                ),
                kind="checkpoint",
            )

            args = parser.parse_args(["--model", str(model_path)])
            explicit_dests = explicit_arg_dests(parser, [])
            explicit_dests.update(
                {
                    "terminate_on_life_loss",
                    "terminate_on_level_change",
                    "terminate_on_completion",
                }
            )

            self.assertTrue(
                apply_model_config_defaults(args, model_path, parser_defaults, explicit_dests)
            )
            self.assertFalse(args.terminate_on_life_loss)
            self.assertFalse(args.terminate_on_level_change)
            self.assertFalse(args.terminate_on_completion)

    def test_wandb_play_forwards_only_explicit_env_overrides(self) -> None:
        parser = build_wandb_play_parser()
        argv = [
            "run",
            "--state",
            "Level2-1",
            "--no-max-pool-frames",
            "--no-terminate-on-completion",
        ]
        args = parser.parse_args(argv)
        explicit_dests = explicit_arg_dests(parser, argv)
        cmd: list[str] = []

        append_explicit_env_args(cmd, parser, args, explicit_dests)

        self.assertIn("--state", cmd)
        self.assertIn("Level2-1", cmd)
        self.assertIn("--no-max-pool-frames", cmd)
        self.assertIn("--no-terminate-on-completion", cmd)
        self.assertNotIn("--game", cmd)
        self.assertNotIn("--reward-mode", cmd)

    def test_wandb_artifact_metadata_requires_artifact_training_metadata(self) -> None:
        class FakeRun:
            id = "abc123"
            name = "run-name"
            path = ["entity", "project", "abc123"]
            notes = "run notes"
            config = {
                "run_name": "train-run",
                "run_description": "description",
                "game": "SuperMarioBros-Nes-v0",
                "state": "Level2-1",
                "max_pool_frames": False,
                "max_episode_steps": 1234,
                "observation_size": 96,
                "action_set": "simple",
            }

        class FakeArtifact:
            metadata = {"kind": "checkpoint"}

            def logged_by(self):
                return FakeRun()

        metadata = metadata_from_wandb_artifact(
            FakeArtifact(),
            Path("ppo_test_100_steps.zip"),
        )

        self.assertEqual(metadata["kind"], "checkpoint")
        self.assertNotIn("env_config", metadata)
        self.assertNotIn("training_metadata", metadata)


class EvalMetricTests(unittest.TestCase):
    def test_metric_path_segment_preserves_retro_state_names(self) -> None:
        self.assertEqual(metric_path_segment("Level1-2"), "Level1-2")
        self.assertEqual(metric_path_segment("Level 1/2"), "Level_1_2")

    def test_episode_rank_prefers_completion_then_progress_then_reward(self) -> None:
        incomplete = {"level_complete": False, "max_x_pos": 4000, "reward": 1000.0}
        complete = {"level_complete": True, "max_x_pos": 100, "reward": -10.0}
        better_progress = {"level_complete": False, "max_x_pos": 4500, "reward": 0.0}
        self.assertGreater(episode_rank(complete), episode_rank(incomplete))
        self.assertGreater(episode_rank(better_progress), episode_rank(incomplete))

    def test_level_complete_ignores_x_threshold_fallback(self) -> None:
        self.assertFalse(
            is_level_complete(
                {"level_complete": False, "level_changed": False, "level_max_x_pos": 5000},
                max_x_pos=5000,
                completion_x_threshold=25,
            )
        )
        self.assertTrue(
            is_level_complete(
                {"level_complete": False, "level_changed": True, "level_max_x_pos": 0},
                max_x_pos=0,
                completion_x_threshold=0,
            )
        )

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
                                "start_state": "Level1-2",
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
                            "start_state": "Level1-1",
                            "max_x_pos": 30,
                            "level_max_x_pos": 30,
                            "level_changed": True,
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
        self.assertEqual(metrics["eval/Level1-1/episodes"], 1)
        self.assertEqual(metrics["eval/Level1-1/completion_rate"], 1.0)
        self.assertEqual(metrics["eval/Level1-2/episodes"], 1)
        self.assertEqual(metrics["eval/Level1-2/completion_rate"], 0.0)
        self.assertEqual(metrics["episode_results"][0]["env_index"], 1)
        self.assertEqual(metrics["episode_results"][0]["start_state"], "Level1-2")
        self.assertEqual(metrics["episode_results"][0]["reward"], 2.0)
        self.assertEqual(metrics["episode_results"][1]["env_index"], 0)
        self.assertEqual(metrics["episode_results"][1]["start_state"], "Level1-1")
        self.assertEqual(metrics["episode_results"][1]["reward"], 4.0)


class TrainingCompletionRateStopCallbackTests(unittest.TestCase):
    def test_records_per_state_completion_windows(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.records: dict[str, int | float] = {}

            def record(self, key: str, value: int | float) -> None:
                self.records[key] = value

        class FakeModel:
            def __init__(self) -> None:
                self.logger = FakeLogger()

        with tempfile.TemporaryDirectory() as tmp_dir:
            callback = TrainingCompletionRateStopCallback(
                episode_window=2,
                rate_threshold=1.0,
                run_dir=tmp_dir,
                default_state="Level1-1",
            )
            model = FakeModel()
            callback.model = model  # type: ignore[assignment]
            callback.num_timesteps = 10
            callback.locals = {
                "dones": [True, True, True],
                "infos": [
                    {"start_state": "Level1-1", "completion_event": True},
                    {"start_state": "Level1-2", "completion_event": False},
                    {"start_state": "Level1-2", "completion_event": True},
                ],
            }

            self.assertTrue(callback._on_step())

        self.assertEqual(model.logger.records["train/completion_episode_rate"], 0.5)
        self.assertEqual(
            model.logger.records["train/Level1-1/completion_episode_rate"],
            1.0,
        )
        self.assertEqual(
            model.logger.records["train/Level1-1/completion_episode_window_size"],
            1,
        )
        self.assertEqual(
            model.logger.records["train/Level1-2/completion_episode_rate"],
            0.5,
        )
        self.assertEqual(
            model.logger.records["train/Level1-2/completion_episode_window_size"],
            2,
        )


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


class RolloutDiagnosticsCallbackTests(unittest.TestCase):
    def test_logs_value_prediction_and_advantage_stats(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: list[tuple[str, float]] = []

            def record(self, key: str, value: float) -> None:
                self.records.append((key, value))

        class RolloutBuffer:
            values = np.array([[1.0, 2.0], [3.0, 4.0]])
            advantages = np.array([[-1.0, 0.0], [1.0, 2.0]])

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()
                self.rollout_buffer = RolloutBuffer()

        model = Model()
        callback = RolloutDiagnosticsCallback(log_histograms=False)
        callback.model = model  # type: ignore[assignment]

        callback._on_rollout_end()

        records = dict(model.logger.records)
        self.assertEqual(records["train/value_pred_mean"], 2.5)
        self.assertAlmostEqual(records["train/value_pred_std"], float(np.std([1.0, 2.0, 3.0, 4.0])))
        self.assertEqual(records["train/value_pred_min"], 1.0)
        self.assertEqual(records["train/value_pred_max"], 4.0)
        self.assertEqual(records["train/value_pred_abs_mean"], 2.5)
        self.assertEqual(records["train/advantage_mean"], 0.5)
        self.assertAlmostEqual(records["train/advantage_std"], float(np.std([-1.0, 0.0, 1.0, 2.0])))
        self.assertEqual(records["train/advantage_min"], -1.0)
        self.assertEqual(records["train/advantage_max"], 2.0)
        self.assertEqual(records["train/advantage_abs_mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
