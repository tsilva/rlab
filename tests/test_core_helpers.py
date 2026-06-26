from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import gymnasium as gym
import numpy as np

from rlab.artifacts import (
    apply_model_config_defaults,
    apply_config_defaults,
    build_s3_artifact_uri,
    checkpoint_step,
    env_config_from_model_metadata,
    env_config_from_metadata,
    explicit_arg_dests,
    init_wandb,
    load_model_metadata,
    model_metadata_path,
    require_training_metadata,
    write_model_metadata,
)
from rlab.callbacks import (
    DoneCounterCallback,
    OutcomeCounterCallback,
    RewardComponentDiagnosticsCallback,
    RolloutDiagnosticsCallback,
    ThroughputCallback,
)
from rlab.cli import build_parser as build_train_parser
from rlab.cli import build_train_command
from rlab.cli import parse_train_args
from rlab.env import (
    EnvConfig,
    StickyAction,
    VecRetroProgressInfo,
    VecTaskConditioning,
    make_eval_vec_env,
    make_rendered_replay_env,
    make_training_vec_env,
    native_vec_env_supports_done_on_info,
    needs_vec_transpose_image,
    resolve_env_config,
    resolve_mixed_state_config,
    state_name_candidates_from_level_id,
)
from rlab.env_config import (
    env_config_from_args,
    parse_done_on_info,
    parse_info_events,
    parse_state_probs,
    parse_states,
)
from rlab.eval_metrics import episode_rank, is_level_complete
from rlab.eval_metrics import RetroEvalCallback
from rlab.eval_runner import evaluate_model_episodes
from rlab.metric_names import (
    TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MEAN,
    TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
    metric_path_segment,
)
from rlab.play import build_parser as build_play_parser
from rlab.play import model_observation
from rlab.play import playback_artifact_ref
from rlab.play import playback_should_end_episode
from rlab.play import task_conditioning_change_message
from rlab.play import task_conditioning_start_message
from rlab.eval import build_parser as build_eval_parser
from rlab.eval import eval_seed_for_checkpoint
from rlab.eval import score as eval_checkpoint_score
from rlab.task_advantage import normalize_advantages_by_task
from rlab.targets import SuperMarioBrosNesV0Target, target_for_game
from rlab.train import disable_sb3_human_output_truncation
from rlab.wandb_artifacts import (
    artifact_download_dir,
    model_artifact_ref,
    safe_artifact_stem,
)
from rlab.wandb_artifacts import metadata_from_wandb_artifact


class Sb3LoggerTests(unittest.TestCase):
    def test_human_output_truncation_is_disabled_for_long_outcome_metrics(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat

        key_values = {
            "train/outcome/level_change/from/0-1/attempts": 1,
            "train/outcome/level_change/from/0-1/fires": 0,
        }
        key_excluded = {key: () for key in key_values}

        with self.assertRaisesRegex(ValueError, "truncated"):
            HumanOutputFormat(io.StringIO()).write(key_values, key_excluded)

        output_format = HumanOutputFormat(io.StringIO())

        class FakeLogger:
            output_formats = [output_format]

        class FakeModel:
            logger = FakeLogger()

        disable_sb3_human_output_truncation(FakeModel())

        output_format.write(key_values, key_excluded)
        self.assertEqual(output_format.max_length, 512)


class EnvConfigFromArgsTests(unittest.TestCase):
    def test_parse_states_trims_empty_values(self) -> None:
        self.assertEqual(parse_states("A, B,C"), ("A", "B", "C"))

    def test_parse_states_accepts_metadata_sequence(self) -> None:
        self.assertEqual(parse_states(["A", " B "]), ("A", "B"))

    def test_parse_states_rejects_empty_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty state"):
            parse_states("A, ,C")

    def test_parse_state_probs_normalizes_later_but_validates_positive_finite(self) -> None:
        self.assertEqual(parse_state_probs("1, 3"), (1.0, 3.0))
        self.assertEqual(parse_state_probs([0.5, 1]), (0.5, 1.0))
        with self.assertRaisesRegex(ValueError, "positive finite"):
            parse_state_probs("1,0")

    def test_init_wandb_uses_global_step_as_metric_step(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.metric_defs: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def define_metric(self, *args: object, **kwargs: object) -> None:
                self.metric_defs.append((args, kwargs))

        class FakeWandb:
            def __init__(self) -> None:
                self.run = FakeRun()
                self.init_kwargs: dict[str, object] | None = None

            def init(self, **kwargs: object) -> FakeRun:
                self.init_kwargs = kwargs
                return self.run

        fake_wandb = FakeWandb()
        args = argparse.Namespace(
            wandb=True,
            wandb_project="SuperMarioBros-NES",
            wandb_entity="tsilva",
            wandb_group="group",
            wandb_tags="ppo, sample-efficiency",
            wandb_mode="offline",
            run_name="run",
            run_description="description",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("rlab.artifacts.load_wandb_env"), patch.dict(
                sys.modules, {"wandb": fake_wandb}
            ):
                run = init_wandb(args, tmp_dir, EnvConfig())

        self.assertIs(run, fake_wandb.run)
        self.assertEqual(
            fake_wandb.run.metric_defs,
            [
                (("global_step",), {}),
                (("*",), {"step_metric": "global_step"}),
            ],
        )

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
            done_on_info_json='{"life_loss":["lives","decrease"]}',
            action_set="right",
        )
        config = env_config_from_args(args, max_episode_steps_attr="max_steps")
        self.assertEqual(config.max_episode_steps, 123)
        self.assertEqual(config.action_set, "right")
        self.assertEqual(config.sticky_action_prob, 0.25)
        self.assertEqual(config.done_on_info, {"life_loss": ("lives", "decrease")})

    def test_parse_done_on_info_accepts_single_and_multi_key_rules(self) -> None:
        self.assertEqual(
            parse_done_on_info(
                '{"life_loss":["lives","decrease"],'
                '"level_change":[["levelHi","levelLo"],"change"]}',
            ),
            {
                "life_loss": ("lives", "decrease"),
                "level_change": (("levelHi", "levelLo"), "change"),
            },
        )

    def test_parse_info_events_accepts_observed_nonterminal_rules(self) -> None:
        self.assertEqual(
            parse_info_events('{"level_change":[["levelHi","levelLo"],"change"]}'),
            {"level_change": (("levelHi", "levelLo"), "change")},
        )

    def test_resolve_env_config_normalizes_legacy_done_on_info_to_events(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                game="SuperMarioBros-Nes-v0",
                done_on_info={
                    "life_loss": ("lives", "decrease"),
                    "level_change": (("levelHi", "levelLo"), "change"),
                },
            )
        )

        self.assertEqual(
            config.info_events,
            {
                "life_loss": ("lives", "decrease"),
                "level_change": (("levelHi", "levelLo"), "change"),
            },
        )
        self.assertEqual(config.done_on_events, ("life_loss", "level_change"))

    def test_train_config_json_applies_defaults_and_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps(
                    {
                        "game": "SuperMarioBros-Nes-v0",
                        "state": "Level1-2",
                        "timesteps": 1024,
                        "states": ["Level1-1", "Level1-2"],
                        "wandb_tags": ["from-json", "config-file"],
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            args = parse_train_args(
                [
                    "--train-config-json",
                    str(path),
                    "--timesteps",
                    "2048",
                ],
            )

        self.assertEqual(args.game, "SuperMarioBros-Nes-v0")
        self.assertEqual(args.state, "Level1-2")
        self.assertEqual(args.timesteps, 2048)
        self.assertEqual(args.states, ["Level1-1", "Level1-2"])
        self.assertEqual(args.wandb_tags, "from-json,config-file")

    def test_training_loop_eval_defaults_to_stochastic(self) -> None:
        parser = build_train_parser()

        self.assertTrue(parser.parse_args([]).eval_stochastic)
        self.assertTrue(parser.parse_args(["--eval-stochastic"]).eval_stochastic)
        self.assertFalse(parser.parse_args(["--no-eval-stochastic"]).eval_stochastic)

    def test_parse_done_on_info_rejects_invalid_shapes(self) -> None:
        invalid_values = [
            "{",
            "[]",
            '{"":["lives","decrease"]}',
            '{"life_loss":["lives"]}',
            '{"life_loss":[[],"change"]}',
            '{"life_loss":[" ","decrease"]}',
            '{"life_loss":["lives",""]}',
        ]

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_done_on_info(value)

    def test_sticky_action_probability_defaults_to_disabled(self) -> None:
        self.assertEqual(build_play_parser().parse_args([]).sticky_action_prob, 0.0)
        self.assertEqual(
            build_play_parser()
            .parse_args(["tsilva/SuperMarioBros-NES/run-checkpoint:latest"])
            .sticky_action_prob,
            0.0,
        )

    def test_sticky_action_probability_parses_for_playback(self) -> None:
        self.assertEqual(
            build_play_parser().parse_args(["--sticky-action-prob", "0.25"]).sticky_action_prob,
            0.25,
        )
        self.assertEqual(
            build_play_parser()
            .parse_args(
                [
                    "tsilva/SuperMarioBros-NES/run-checkpoint:latest",
                    "--sticky-action-prob",
                    "0.25",
                ]
            )
            .sticky_action_prob,
            0.25,
        )

    def test_task_conditioning_parses_for_playback(self) -> None:
        args = build_play_parser().parse_args(
            [
                "--states",
                "Level1-1,Level1-2",
                "--state-probs",
                "0.5,0.5",
                "--task-conditioning",
                "--task-conditioning-info-vars",
                "levelHi,levelLo",
            ]
        )
        config = env_config_from_args(
            args,
            max_episode_steps_attr="max_steps",
            include_states=True,
        )

        self.assertEqual(config.states, ("Level1-1", "Level1-2"))
        self.assertEqual(config.state_probs, (0.5, 0.5))
        self.assertTrue(config.task_conditioning)
        self.assertEqual(config.task_conditioning_info_vars, ("levelHi", "levelLo"))

    def test_task_conditioning_info_values_validate_arity(self) -> None:
        with self.assertRaisesRegex(ValueError, "row length"):
            resolve_env_config(
                EnvConfig(
                    game="SuperMarioBros-Nes-v0",
                    task_conditioning=True,
                    task_conditioning_info_vars=("levelHi", "levelLo"),
                    task_conditioning_info_values=((0,),),
                )
            )

    def test_invalid_sticky_action_probability_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "sticky_action_prob"):
            resolve_env_config(EnvConfig(game="SuperMarioBros-Nes-v0", sticky_action_prob=-0.1))

    def test_eval_vec_env_clears_done_on_info_rules(self) -> None:
        sentinel = object()
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            done_on_info={"life_loss": ("lives", "decrease")},
        )

        with patch("rlab.env.make_vec_envs", return_value=sentinel) as make_vec_envs:
            env = make_eval_vec_env(config=config, n_envs=2, seed=7)

        self.assertIs(env, sentinel)
        passed_config = make_vec_envs.call_args.kwargs["config"]
        self.assertEqual(passed_config.done_on_info, {})

    def test_training_vec_env_preserves_requested_done_on_info_rules(self) -> None:
        sentinel = object()
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            done_on_info={"life_loss": ("lives", "decrease")},
        )

        with patch("rlab.env.make_vec_envs", return_value=sentinel) as make_vec_envs:
            env = make_training_vec_env(config=config, n_envs=2, seed=7)

        self.assertIs(env, sentinel)
        passed_config = make_vec_envs.call_args.kwargs["config"]
        self.assertEqual(passed_config.done_on_info, {"life_loss": ("lives", "decrease")})

    def test_rendered_eval_replay_clears_done_on_info_rules(self) -> None:
        sentinel = object()
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            done_on_info={"life_loss": ("lives", "decrease")},
        )

        with patch("rlab.env.make_retro_env", return_value=sentinel) as make_retro_env:
            env = make_rendered_replay_env(config=config, seed=7)

        self.assertIs(env, sentinel)
        passed_config = make_retro_env.call_args.kwargs["config"]
        self.assertEqual(passed_config.done_on_info, {})

    def test_short_states_requires_one_state_per_env_slot(self) -> None:
        with patch(
            "rlab.env.retro.data.list_states",
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
            "rlab.env.retro.data.list_states",
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
        with patch("rlab.env.retro.data.list_states", return_value=["Level1-1"]):
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
        self.assertEqual(same_level_info["progress_component"], 256.0)
        self.assertEqual(same_level_info["progress_reward_component"], 256.0)
        self.assertEqual(same_level_info["score_reward_component"], 0.0)
        self.assertEqual(same_level_info["completion_reward_component"], 0.0)
        self.assertEqual(same_level_info["death_penalty_component"], 0.0)
        self.assertEqual(same_level_info["time_penalty_component"], 0.0)

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

    def test_mario_death_level_change_does_not_count_as_completion(self) -> None:
        config = argparse.Namespace(
            reward_mode="score",
            no_progress_min_delta=0,
            completion_x_threshold=0,
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
        tracker.reset({"lives": 3, "score": 0, "levelHi": 0, "levelLo": 1})

        death_level_change_info = {
            "lives": 2,
            "score": 0,
            "levelHi": 0,
            "levelLo": 0,
            "xscrollHi": 0,
            "xscrollLo": 64,
            "life_loss": True,
        }
        progress = tracker.step(0.0, death_level_change_info, done=True)

        self.assertTrue(death_level_change_info["level_changed"])
        self.assertTrue(death_level_change_info["died"])
        self.assertFalse(death_level_change_info["level_complete"])
        self.assertFalse(death_level_change_info["completion_event"])
        self.assertEqual(death_level_change_info["completed_level_count"], 0)
        self.assertEqual(death_level_change_info["terminal_reward"], -50.0)
        self.assertFalse(progress.done)

    def test_mario_done_on_info_death_level_change_does_not_count_as_completion(self) -> None:
        config = argparse.Namespace(
            reward_mode="score",
            no_progress_min_delta=0,
            completion_x_threshold=0,
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

        death_level_change_info = {
            "lives": 3,
            "score": 0,
            "levelHi": 2,
            "levelLo": 1,
            "xscrollHi": 0,
            "xscrollLo": 64,
            "done_on_info": {
                "life_loss": {"op": "decrease", "keys": ("lives",)},
                "level_change": {"op": "change", "keys": ("levelHi", "levelLo")},
            },
        }
        progress = tracker.step(0.0, death_level_change_info, done=True)

        self.assertTrue(death_level_change_info["level_changed"])
        self.assertTrue(death_level_change_info["died"])
        self.assertFalse(death_level_change_info["level_complete"])
        self.assertFalse(death_level_change_info["completion_event"])
        self.assertEqual(death_level_change_info["completed_level_count"], 0)
        self.assertFalse(progress.done)


class VecImageShapeTests(unittest.TestCase):
    def test_channel_last_native_observations_need_transpose(self) -> None:
        space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
        self.assertTrue(needs_vec_transpose_image(space))


class VecRetroProgressInfoEventTests(unittest.TestCase):
    def test_emits_nonterminal_info_events_for_configured_level_change(self) -> None:
        class FakeVecEnv:
            num_envs = 1
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)

            def __init__(self) -> None:
                self.reset_infos = [
                    {"lives": 3, "score": 0, "levelHi": 0, "levelLo": 0},
                ]

            def reset(self):
                return np.zeros((1, 4, 84, 84), dtype=np.uint8)

            def step_async(self, actions) -> None:
                self.actions = actions

            def step_wait(self):
                return (
                    np.zeros((1, 4, 84, 84), dtype=np.uint8),
                    np.array([0.0], dtype=np.float32),
                    np.array([False]),
                    [
                        {
                            "lives": 3,
                            "score": 0,
                            "levelHi": 0,
                            "levelLo": 1,
                            "xscrollHi": 0,
                            "xscrollLo": 0,
                        },
                    ],
                )

        config = resolve_env_config(
            EnvConfig(
                game="SuperMarioBros-Nes-v0",
                reward_mode="score",
                info_events={"level_change": (("levelHi", "levelLo"), "change")},
            )
        )
        env = VecRetroProgressInfo(FakeVecEnv(), config)

        env.reset()
        env.step_async(np.array([0], dtype=np.int64))
        _obs, _rewards, dones, infos = env.step_wait()

        self.assertFalse(dones[0])
        self.assertEqual(
            infos[0]["info_events"]["level_change"],
            {
                "op": "change",
                "keys": ("levelHi", "levelLo"),
                "prev": (0, 0),
                "next": (0, 1),
            },
        )
        self.assertTrue(infos[0]["level_complete"])

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
    def test_training_vec_env_passes_weighted_states_as_native_state_dict(self) -> None:
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
            states=("Level1-1", "Level1-2"),
            state_probs=(0.5, 0.5),
        )
        with (
            patch("rlab.env.StableRetroNativeVecEnv", FakeNative),
            patch("rlab.env.VecRetroProgressInfo", side_effect=lambda env, config: env),
            patch("rlab.env.VecMonitor", side_effect=lambda env: env),
            patch("rlab.env.maybe_transpose_vec_image", side_effect=lambda env: env),
            patch(
                "rlab.env.retro.data.list_states",
                return_value=["Level1-1", "Level1-2"],
            ),
        ):
            env = make_training_vec_env(config, n_envs=16, seed=7)

        self.assertIsInstance(env, FakeNative)
        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["num_envs"], 16)
        self.assertEqual(created[0]["state"], {"Level1-1": 0.5, "Level1-2": 0.5})
        self.assertNotIn("states", created[0])
        self.assertNotIn("state_probs", created[0])

    def test_training_vec_env_passes_fixed_lane_states_as_native_state_list(self) -> None:
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
                created.append(kwargs)

            def seed(self, seed):
                return [seed]

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            states=("Level1-1", "Level1-2", "Level1-1", "Level1-2"),
        )
        with (
            patch("rlab.env.StableRetroNativeVecEnv", FakeNative),
            patch("rlab.env.VecRetroProgressInfo", side_effect=lambda env, config: env),
            patch("rlab.env.VecMonitor", side_effect=lambda env: env),
            patch("rlab.env.maybe_transpose_vec_image", side_effect=lambda env: env),
            patch(
                "rlab.env.retro.data.list_states",
                return_value=["Level1-1", "Level1-2"],
            ),
        ):
            env = make_training_vec_env(config, n_envs=4, seed=7)

        self.assertIsInstance(env, FakeNative)
        self.assertEqual(created[0]["state"], ["Level1-1", "Level1-2", "Level1-1", "Level1-2"])
        self.assertNotIn("states", created[0])
        self.assertNotIn("state_probs", created[0])

    def test_training_vec_env_passes_configured_native_done_on_info_rules(self) -> None:
        created: list[dict[str, object]] = []
        progress_configs: list[EnvConfig] = []

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
                created.append(kwargs)

            def seed(self, seed):
                return [seed]

        def fake_progress(env, config):
            progress_configs.append(config)
            return env

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            done_on_info={
                "life_loss": ("lives", "decrease"),
                "level_change": (("levelHi", "levelLo"), "change"),
            },
            states=("Level1-1", "Level1-2"),
            state_probs=(0.5, 0.5),
        )
        with (
            patch("rlab.env.StableRetroNativeVecEnv", FakeNative),
            patch("rlab.env.native_vec_env_supports_done_on_info", return_value=True),
            patch("rlab.env.VecRetroProgressInfo", side_effect=fake_progress),
            patch("rlab.env.VecMonitor", side_effect=lambda env: env),
            patch("rlab.env.maybe_transpose_vec_image", side_effect=lambda env: env),
            patch(
                "rlab.env.retro.data.list_states",
                return_value=["Level1-1", "Level1-2"],
            ),
        ):
            env = make_training_vec_env(config, n_envs=16, seed=7)

        self.assertIsInstance(env, FakeNative)
        self.assertEqual(
            created[0]["done_on_info"],
            {
                "life_loss": ("lives", "decrease"),
                "level_change": (("levelHi", "levelLo"), "change"),
            },
        )
        self.assertNotIn("terminate_on_life_loss", created[0])
        self.assertNotIn("life_variable", created[0])
        self.assertEqual(progress_configs[0].done_on_info, config.done_on_info)
        self.assertEqual(progress_configs[0].info_events, config.done_on_info)
        self.assertEqual(progress_configs[0].done_on_events, ("life_loss", "level_change"))

    def test_training_vec_env_passes_only_done_events_to_native_done_on_info(self) -> None:
        created: list[dict[str, object]] = []
        progress_configs: list[EnvConfig] = []

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
                created.append(kwargs)

            def seed(self, seed):
                return [seed]

        def fake_progress(env, config):
            progress_configs.append(config)
            return env

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            info_events={
                "life_loss": ("lives", "decrease"),
                "level_change": (("levelHi", "levelLo"), "change"),
            },
            done_on_events=("life_loss",),
            states=("Level1-1", "Level1-2"),
            state_probs=(0.5, 0.5),
        )
        with (
            patch("rlab.env.StableRetroNativeVecEnv", FakeNative),
            patch("rlab.env.native_vec_env_supports_done_on_info", return_value=True),
            patch("rlab.env.VecRetroProgressInfo", side_effect=fake_progress),
            patch("rlab.env.VecMonitor", side_effect=lambda env: env),
            patch("rlab.env.maybe_transpose_vec_image", side_effect=lambda env: env),
            patch(
                "rlab.env.retro.data.list_states",
                return_value=["Level1-1", "Level1-2"],
            ),
        ):
            env = make_training_vec_env(config, n_envs=16, seed=7)

        self.assertIsInstance(env, FakeNative)
        self.assertEqual(created[0]["done_on_info"], {"life_loss": ("lives", "decrease")})

    def test_training_vec_env_requires_native_done_on_info_support_when_rules_requested(self) -> None:
        class FakeNative:
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)

            def __init__(self, game, **kwargs):
                pass

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            done_on_info={"life_loss": ("lives", "decrease")},
        )
        with (
            patch("rlab.env.StableRetroNativeVecEnv", FakeNative),
            patch("rlab.env.native_vec_env_supports_done_on_info", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "done_on_info support"):
                make_training_vec_env(config, n_envs=1, seed=7)

    def test_done_on_info_support_detection(self) -> None:
        class FakeNative:
            def __init__(self, game, *, done_on_info=None):
                self.game = game
                self.done_on_info = done_on_info

        with patch("rlab.env.StableRetroNativeVecEnv", FakeNative):
            self.assertTrue(native_vec_env_supports_done_on_info())

    def test_task_conditioning_wraps_native_active_state_as_one_hot(self) -> None:
        class FakeNative:
            num_envs = 4
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)
            initial_state_names = ("Level1-1", "Level1-2")

            def __init__(self, game, **kwargs):
                self.game = game
                self.kwargs = kwargs
                self._indices = np.asarray([0, 1, 0, 1], dtype=np.int32)

            def seed(self, seed):
                return [seed]

            def reset(self):
                return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)

            def active_state_indices(self):
                return self._indices

            def step_async(self, actions):
                self.actions = actions

            def step_wait(self):
                self._indices[:] = [1, 1, 0, 0]
                return (
                    np.ones((self.num_envs, 4, 84, 84), dtype=np.uint8),
                    np.zeros(self.num_envs, dtype=np.float32),
                    np.asarray([True, False, False, False]),
                    [{"terminal_observation": np.zeros((4, 84, 84), dtype=np.uint8)}, {}, {}, {}],
                )

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            states=("Level1-1", "Level1-2"),
            state_probs=(0.5, 0.5),
            task_conditioning=True,
        )
        with (
            patch("rlab.env.StableRetroNativeVecEnv", FakeNative),
            patch("rlab.env.VecRetroProgressInfo", side_effect=lambda env, config: env),
            patch("rlab.env.VecMonitor", side_effect=lambda env: env),
            patch("rlab.env.maybe_transpose_vec_image", side_effect=lambda env: env),
            patch(
                "rlab.env.retro.data.list_states",
                return_value=["Level1-1", "Level1-2"],
            ),
        ):
            env = make_training_vec_env(config, n_envs=4, seed=7)
            reset_obs = env.reset()
            env.step_async(np.zeros((4, 2), dtype=np.uint8))
            step_obs, _, dones, infos = env.step_wait()

        self.assertIsInstance(env, VecTaskConditioning)
        self.assertEqual(env.task_state_names, ("Level1-1", "Level1-2"))
        self.assertEqual(reset_obs["image"].shape, (4, 4, 84, 84))
        np.testing.assert_array_equal(
            reset_obs["task"],
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_array_equal(
            step_obs["task"],
            np.asarray(
                [
                    [0.0, 1.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        )
        self.assertTrue(dones[0])
        self.assertEqual(set(infos[0]["terminal_observation"]), {"image", "task"})
        np.testing.assert_array_equal(
            infos[0]["terminal_observation"]["task"],
            np.asarray([1.0, 0.0], dtype=np.float32),
        )

    def test_task_conditioning_collapses_duplicate_state_names(self) -> None:
        class FakeVec:
            num_envs = 4
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)
            initial_state_names = ("Level1-1", "Level1-2", "Level1-1", "Level1-2")

            def __init__(self) -> None:
                self._indices = np.asarray([0, 1, 2, 3], dtype=np.int32)

            def active_state_indices(self):
                return self._indices

            def reset(self):
                return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)

            def step_async(self, actions):
                pass

            def step_wait(self):
                return self.reset(), np.zeros(4, dtype=np.float32), np.zeros(4, dtype=bool), [{}, {}, {}, {}]

        env = VecTaskConditioning(FakeVec())
        obs = env.reset()

        self.assertEqual(env.task_state_names, ("Level1-1", "Level1-2"))
        np.testing.assert_array_equal(
            obs["task"],
            np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        )

    def test_task_conditioning_follows_non_terminal_level_id(self) -> None:
        class FakeVec:
            num_envs = 2
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)
            initial_state_names = ("Level1-1", "Level1-2")

            def __init__(self) -> None:
                self._indices = np.asarray([0, 0], dtype=np.int32)

            def active_state_indices(self):
                return self._indices

            def reset(self):
                return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)

            def step_async(self, actions):
                pass

            def step_wait(self):
                return (
                    self.reset(),
                    np.zeros(self.num_envs, dtype=np.float32),
                    np.zeros(self.num_envs, dtype=bool),
                    [{"level_id": "1-2"}, {"level_id": "1-1"}],
                )

        env = VecTaskConditioning(FakeVec())
        reset_obs = env.reset()
        env.step_async(np.zeros((2, 2), dtype=np.uint8))
        step_obs, _, _, _ = env.step_wait()

        np.testing.assert_array_equal(
            reset_obs["task"],
            np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            step_obs["task"],
            np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        )

    def test_task_conditioning_maps_zero_indexed_mario_level_id(self) -> None:
        class FakeVec:
            num_envs = 1
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)
            initial_state_names = ("Level1-1", "Level1-2")

            def __init__(self) -> None:
                self._indices = np.asarray([0], dtype=np.int32)

            def active_state_indices(self):
                return self._indices

            def reset(self):
                return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)

            def step_async(self, actions):
                pass

            def step_wait(self):
                return self.reset(), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=bool), [
                    {"level_id": "0-1"}
                ]

        env = VecTaskConditioning(FakeVec())
        env.reset()
        env.step_async(np.zeros((1, 2), dtype=np.uint8))
        step_obs, _, _, _ = env.step_wait()

        np.testing.assert_array_equal(
            step_obs["task"],
            np.asarray([[0.0, 1.0]], dtype=np.float32),
        )

    def test_task_conditioning_follows_configured_info_vars(self) -> None:
        class FakeVec:
            num_envs = 1
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)
            initial_state_names = ("Level1-1", "Level1-2")

            def __init__(self) -> None:
                self._indices = np.asarray([0], dtype=np.int32)

            def active_state_indices(self):
                return self._indices

            def reset(self):
                return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)

            def step_async(self, actions):
                pass

            def step_wait(self):
                return self.reset(), np.zeros(1, dtype=np.float32), np.zeros(1, dtype=bool), [
                    {"levelHi": 0, "levelLo": 1, "level_id": "not-used"}
                ]

        env = VecTaskConditioning(
            FakeVec(),
            config=EnvConfig(
                game="SuperMarioBros-Nes-v0",
                states=("Level1-1", "Level1-2"),
                task_conditioning=True,
                task_conditioning_info_vars=("levelHi", "levelLo"),
            ),
        )
        reset_obs = env.reset()
        env.step_async(np.zeros((1, 2), dtype=np.uint8))
        step_obs, _, _, _ = env.step_wait()

        np.testing.assert_array_equal(reset_obs["task"], np.asarray([[1.0, 0.0]], dtype=np.float32))
        np.testing.assert_array_equal(step_obs["task"], np.asarray([[0.0, 1.0]], dtype=np.float32))


class CommandAndArtifactTests(unittest.TestCase):
    def test_build_train_command_skips_empty_target_kl(self) -> None:
        cmd = build_train_command(
            {
                "run_name": "candidate",
                "states": "Level1-1,Level1-2",
                "state_probs": "1,3",
                "target_kl": 0.0,
                "clip_range_vf": 0.2,
                "task_conditioning_info_vars": ("levelHi", "levelLo"),
                "policy_net_arch": "128",
                "value_net_arch": "512,512",
                "advantage_normalization": "per-task",
                "task_conditioning": True,
                "wandb": True,
                "normalize_advantage": False,
                "info_events_json": {
                    "life_loss": ["lives", "decrease"],
                    "level_change": [["levelHi", "levelLo"], "change"],
                },
                "done_on_events": "life_loss,level_change",
            }
        )
        self.assertIn("--run-name", cmd)
        self.assertIn("--states", cmd)
        self.assertIn("--state-probs", cmd)
        self.assertIn("--task-conditioning", cmd)
        self.assertIn("--task-conditioning-info-vars", cmd)
        self.assertIn("levelHi,levelLo", cmd)
        self.assertNotIn("True", cmd)
        self.assertNotIn("--target-kl", cmd)
        self.assertIn("--clip-range-vf", cmd)
        self.assertIn("0.2", cmd)
        self.assertIn("--policy-net-arch", cmd)
        self.assertIn("128", cmd)
        self.assertIn("--value-net-arch", cmd)
        self.assertIn("512,512", cmd)
        self.assertIn("--advantage-normalization", cmd)
        self.assertIn("per-task", cmd)
        self.assertIn("--info-events-json", cmd)
        self.assertIn(
            '{"life_loss":["lives","decrease"],"level_change":[["levelHi","levelLo"],"change"]}',
            cmd,
        )
        self.assertIn("--done-on-events", cmd)
        self.assertIn("life_loss,level_change", cmd)
        self.assertIn("--wandb", cmd)
        self.assertIn("--no-normalize-advantage", cmd)

    def test_build_train_command_can_request_deterministic_eval(self) -> None:
        cmd = build_train_command({"eval_stochastic": False})

        self.assertIn("--no-eval-stochastic", cmd)

    def test_train_parser_accepts_task_conditioning_and_done_on_info_flags(self) -> None:
        args = build_train_parser().parse_args(
            [
                "--game",
                "SuperMarioBros-Nes-v0",
                "--states",
                "Level1-1,Level1-2",
                "--task-conditioning",
                "--task-conditioning-info-vars",
                "levelHi,levelLo",
                "--task-conditioning-info-values",
                "0,0;0,1",
                "--done-on-info-json",
                '{"level_change":[["levelHi","levelLo"],"change"]}',
                "--info-events-json",
                '{"life_loss":["lives","decrease"]}',
                "--done-on-events",
                "life_loss",
            ]
        )

        self.assertEqual(args.task_conditioning_info_vars, "levelHi,levelLo")
        self.assertEqual(args.task_conditioning_info_values, "0,0;0,1")
        config = env_config_from_args(args, include_states=True)
        self.assertEqual(
            config.done_on_info,
            {"level_change": (("levelHi", "levelLo"), "change")},
        )
        self.assertEqual(config.info_events, {"life_loss": ("lives", "decrease")})
        self.assertEqual(config.done_on_events, ("life_loss",))

    def test_train_parser_deletes_completion_stop_flags(self) -> None:
        args = build_train_parser().parse_args([])

        self.assertFalse(hasattr(args, "stop_completion_episode_window"))
        self.assertFalse(hasattr(args, "stop_completion_rate_threshold"))
        self.assertFalse(hasattr(args, "stop_state_min_completion_rate_threshold"))
        self.assertFalse(hasattr(args, "stop_completion_rolling_window"))
        self.assertFalse(hasattr(args, "stop_completion_rolling_threshold"))

    def test_normalize_advantages_by_task_updates_rollout_in_place(self) -> None:
        advantages = np.asarray(
            [
                [1.0, 10.0],
                [3.0, 12.0],
                [5.0, 14.0],
            ],
            dtype=np.float32,
        )
        observations = {
            "task": np.asarray(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ],
                dtype=np.float32,
            )
        }

        stats = normalize_advantages_by_task(advantages, observations)

        self.assertEqual(stats[0]["count"], 3.0)
        self.assertEqual(stats[1]["count"], 3.0)
        np.testing.assert_allclose(advantages[:, 0].mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(advantages[:, 1].mean(), 0.0, atol=1e-6)
        np.testing.assert_allclose(advantages[:, 0].std(), 1.0, atol=1e-6)
        np.testing.assert_allclose(advantages[:, 1].std(), 1.0, atol=1e-6)

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
                "states": ["Level1-1", "Level1-2"],
                "state_probs": [0.5, 0.5],
                "task_conditioning": True,
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
        self.assertEqual(args.states, ["Level1-1", "Level1-2"])
        self.assertEqual(args.state_probs, [0.5, 0.5])
        self.assertTrue(args.task_conditioning)

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
                    done_on_info={"level_change": (("levelHi", "levelLo"), "change")},
                ),
                kind="checkpoint",
            )

            argv = ["--model", str(model_path), "--done-on-info-json", "{}"]
            args = parser.parse_args(argv)
            explicit_dests = explicit_arg_dests(parser, argv)

            self.assertTrue(
                apply_model_config_defaults(args, model_path, parser_defaults, explicit_dests)
            )
            self.assertEqual(args.state, "Level2-1")
            self.assertFalse(args.max_pool_frames)
            self.assertEqual(args.observation_size, 96)
            self.assertTrue(args.score_progress_clipped)
            self.assertEqual(args.done_on_info_json, "{}")

            config = env_config_from_model_metadata(
                model_path, fallback=EnvConfig(state="fallback")
            )
            self.assertIsNotNone(config)
            assert config is not None
            self.assertEqual(config.state, "Level2-1")
            self.assertEqual(config.observation_size, 96)

    def test_model_observation_wraps_task_conditioned_policy_input(self) -> None:
        class FakeModel:
            observation_space = gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(4, 84, 84),
                        dtype=np.uint8,
                    ),
                    "task": gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                }
            )

        image_obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        obs = model_observation(
            FakeModel(),
            image_obs,
            EnvConfig(
                game="SuperMarioBros-Nes-v0",
                state="Level1-2",
                states=("Level1-1", "Level1-2"),
                task_conditioning=True,
            ),
        )

        self.assertIs(obs["image"], image_obs)
        np.testing.assert_array_equal(obs["task"], np.array([[0.0, 1.0]], dtype=np.float32))

    def test_model_observation_can_override_active_task_state(self) -> None:
        class FakeModel:
            observation_space = gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(4, 84, 84),
                        dtype=np.uint8,
                    ),
                    "task": gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                }
            )

        image_obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        obs = model_observation(
            FakeModel(),
            image_obs,
            EnvConfig(
                game="SuperMarioBros-Nes-v0",
                state="Level1-1",
                states=("Level1-1", "Level1-2"),
                task_conditioning=True,
            ),
            active_task_state="Level1-2",
        )

        self.assertIs(obs["image"], image_obs)
        np.testing.assert_array_equal(obs["task"], np.array([[0.0, 1.0]], dtype=np.float32))

    def test_model_observation_can_use_active_info_value(self) -> None:
        class FakeModel:
            observation_space = gym.spaces.Dict(
                {
                    "image": gym.spaces.Box(
                        low=0,
                        high=255,
                        shape=(4, 84, 84),
                        dtype=np.uint8,
                    ),
                    "task": gym.spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
                }
            )

        image_obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
        obs = model_observation(
            FakeModel(),
            image_obs,
            EnvConfig(
                game="SuperMarioBros-Nes-v0",
                state="Level1-1",
                states=("Level1-1", "Level1-2"),
                task_conditioning=True,
                task_conditioning_info_vars=("levelHi", "levelLo"),
            ),
            active_info_value=(0, 1),
        )

        self.assertIs(obs["image"], image_obs)
        np.testing.assert_array_equal(obs["task"], np.array([[0.0, 1.0]], dtype=np.float32))

    def test_task_state_from_info_maps_zero_indexed_mario_level_id(self) -> None:
        self.assertEqual(
            state_name_candidates_from_level_id("0-1"),
            ("Level0-1", "Level1-2"),
        )

    def test_task_conditioning_change_message_includes_one_hot(self) -> None:
        self.assertEqual(
            task_conditioning_change_message(
                episode=1,
                step=481,
                old_task=(0, 0),
                new_task=(0, 1),
                task_index=1,
                task_count=2,
            ),
            "task_conditioning_change episode=1 step=481 old=(0, 0) "
            "new=(0, 1) index=1 one_hot=[0, 1]",
        )

    def test_task_conditioning_start_message_includes_one_hot(self) -> None:
        self.assertEqual(
            task_conditioning_start_message(
                episode=1,
                step=0,
                task=(0, 0),
                task_index=0,
                task_count=3,
            ),
            "task_conditioning_start episode=1 step=0 task=(0, 0) "
            "index=0 one_hot=[1, 0, 0]",
        )

    def test_playback_eval_defaults_block_training_done_on_info_metadata(self) -> None:
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
                    done_on_info={
                        "life_loss": ("lives", "decrease"),
                        "level_change": (("levelHi", "levelLo"), "change"),
                    },
                ),
                kind="checkpoint",
            )

            args = parser.parse_args(["--model", str(model_path)])
            explicit_dests = explicit_arg_dests(parser, [])
            explicit_dests.add("done_on_info_json")

            self.assertTrue(
                apply_model_config_defaults(args, model_path, parser_defaults, explicit_dests)
            )
            self.assertEqual(args.done_on_info_json, "")

    def test_gui_playback_defaults_to_stochastic(self) -> None:
        parser = build_play_parser()

        self.assertFalse(parser.parse_args([]).deterministic)
        self.assertTrue(parser.parse_args(["--deterministic"]).deterministic)
        help_text = parser.format_help()
        self.assertIn("--deterministic", help_text)
        self.assertNotIn("--stochastic", help_text)
        self.assertNotIn("--no-stochastic", help_text)

    def test_eval_defaults_to_stochastic(self) -> None:
        parser = build_eval_parser()

        self.assertFalse(parser.parse_args([]).deterministic)
        self.assertTrue(parser.parse_args(["--deterministic"]).deterministic)
        help_text = parser.format_help()
        self.assertIn("--deterministic", help_text)
        self.assertNotIn("--stochastic", help_text)
        self.assertNotIn("--no-stochastic", help_text)

    def test_gui_playback_does_not_end_on_completion_without_env_done(self) -> None:
        self.assertFalse(
            playback_should_end_episode(
                terminated=False,
                truncated=False,
                completed=True,
            )
        )
        self.assertTrue(
            playback_should_end_episode(
                terminated=True,
                truncated=False,
                completed=True,
            )
        )

    def test_artifact_gui_playback_defaults_to_stochastic(self) -> None:
        parser = build_play_parser()

        ref = "tsilva/SuperMarioBros-NES/run-checkpoint:latest"

        self.assertFalse(parser.parse_args([ref]).deterministic)
        self.assertTrue(parser.parse_args([ref, "--deterministic"]).deterministic)

    def test_artifact_playback_ref_uses_positional_artifact_ref(self) -> None:
        parser = build_play_parser()
        args = parser.parse_args(["tsilva/SuperMarioBros-NES/run-checkpoint:latest"])

        self.assertEqual(
            playback_artifact_ref(args),
            "tsilva/SuperMarioBros-NES/run-checkpoint:latest",
        )

    def test_artifact_playback_ref_rejects_positional_run_name(self) -> None:
        parser = build_play_parser()

        with patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                parser.parse_args(["run"])

    def test_artifact_playback_ref_uses_explicit_run_kind_and_version(self) -> None:
        parser = build_play_parser()
        argv = ["--artifact-run", "run", "--artifact-kind", "best", "--artifact-version", "v8"]
        args = parser.parse_args(argv)

        self.assertEqual(
            playback_artifact_ref(args),
            "tsilva/SuperMarioBros-NES/run-best:v8",
        )

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
    def test_checkpoint_score_prefers_min_completion_rate_when_available(self) -> None:
        metrics = {
            "completion_rate": 0.95,
            "eval/done/level_change/from_rate/min": 0.80,
            "max_x_max": 3200,
            "reward_mean": 1200.0,
        }

        self.assertEqual(eval_checkpoint_score(metrics), (0.8, 3200, 1200.0))

    def test_eval_wandb_payload_includes_global_step(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.payload: dict[str, object] | None = None
                self.step: int | None = None

            def log(self, payload: dict[str, object], *, step: int) -> None:
                self.payload = payload
                self.step = step

        callback = RetroEvalCallback(
            config=EnvConfig(),
            run_dir=".",
            best_model_save_path=".",
            eval_freq=1,
            n_eval_episodes=1,
            deterministic=True,
            seed=0,
            completion_x_threshold=0,
            wandb_run=FakeRun(),
            record_video=False,
        )
        callback.num_timesteps = 12345
        metrics = {
            "reward_mean": 1.0,
            "reward_std": 0.0,
            "reward_max": 2.0,
            "max_x_mean": 3.0,
            "max_x_max": 4,
            "max_level_x_mean": 5.0,
            "max_level_x_max": 6,
            "completion_count": 1,
            "completion_rate": 1.0,
            "death_count": 0,
            "death_rate": 0.0,
            "terminated_count": 1,
            "terminated_rate": 1.0,
            "truncated_count": 0,
            "truncated_rate": 0.0,
            "eval/done/all": 1,
            "eval/done/level_change": 1,
            "eval/done/level_change/rate": 1.0,
            "eval/done/max_steps": 0,
            "eval/done/max_steps/rate": 0.0,
            "eval/done/unclassified": 0,
            "eval/done/unclassified/rate": 0.0,
            "best_episode": {"reward": 2.0, "max_x_pos": 4},
        }

        with patch.dict(sys.modules, {"wandb": object()}):
            callback.log_wandb(metrics, death_x_positions=[], video_path=None)

        assert callback.wandb_run.payload is not None
        self.assertEqual(callback.wandb_run.payload["global_step"], 12345)
        self.assertEqual(callback.wandb_run.payload["eval/done/all"], 1)
        self.assertEqual(callback.wandb_run.payload["eval/done/level_change"], 1)
        self.assertEqual(callback.wandb_run.payload["eval/done/level_change/rate"], 1.0)
        self.assertEqual(callback.wandb_run.payload["eval/done/max_steps"], 0)
        self.assertNotIn("eval/outcome/terminated", callback.wandb_run.payload)
        self.assertNotIn("eval/outcome/truncated", callback.wandb_run.payload)
        self.assertEqual(callback.wandb_run.step, 12345)

    def test_metric_path_segment_preserves_retro_state_names(self) -> None:
        self.assertEqual(metric_path_segment("Level1-2"), "Level1-2")
        self.assertEqual(metric_path_segment("Level 1/2"), "Level_1_2")

    def test_episode_rank_prefers_completion_then_progress_then_reward(self) -> None:
        incomplete = {"level_complete": False, "max_x_pos": 4000, "reward": 1000.0}
        complete = {"level_complete": True, "max_x_pos": 100, "reward": -10.0}
        better_progress = {"level_complete": False, "max_x_pos": 4500, "reward": 0.0}
        self.assertGreater(episode_rank(complete), episode_rank(incomplete))
        self.assertGreater(episode_rank(better_progress), episode_rank(incomplete))

    def test_level_complete_uses_explicit_completion_flag(self) -> None:
        self.assertFalse(
            is_level_complete(
                {"level_complete": False, "level_changed": False, "level_max_x_pos": 5000},
                max_x_pos=5000,
                completion_x_threshold=25,
            )
        )
        self.assertFalse(
            is_level_complete(
                {"level_complete": False, "level_changed": True},
                max_x_pos=0,
                completion_x_threshold=0,
            )
        )
        self.assertTrue(
            is_level_complete(
                {"level_complete": True, "level_changed": True},
                max_x_pos=0,
                completion_x_threshold=0,
            )
        )

    def test_level_complete_fallback_rejects_death_level_change(self) -> None:
        self.assertFalse(
            is_level_complete(
                {"level_changed": True, "died": True},
                max_x_pos=0,
                completion_x_threshold=0,
            )
        )
        self.assertTrue(
            is_level_complete(
                {"level_changed": True, "died": False},
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
                                "TimeLimit.truncated": True,
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
        with patch("rlab.eval_runner.make_eval_vec_env", return_value=FakeVecEnv()):
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
        self.assertEqual(metrics["terminated_count"], 1)
        self.assertEqual(metrics["terminated_rate"], 0.5)
        self.assertEqual(metrics["truncated_count"], 1)
        self.assertEqual(metrics["truncated_rate"], 0.5)
        self.assertEqual(metrics["eval/done/all"], 2)
        self.assertEqual(metrics["eval/done/level_change"], 1)
        self.assertEqual(metrics["eval/done/level_change/rate"], 0.5)
        self.assertEqual(metrics["eval/done/max_steps"], 1)
        self.assertEqual(metrics["eval/done/max_steps/rate"], 0.5)
        self.assertEqual(metrics["eval/done/unclassified"], 0)
        self.assertEqual(metrics["eval/done/unclassified/rate"], 0.0)
        self.assertEqual(metrics["eval/done/all/from/Level1-1"], 1)
        self.assertEqual(metrics["eval/done/level_change/from/Level1-1"], 1)
        self.assertEqual(metrics["eval/done/level_change/from/Level1-1/rate"], 1.0)
        self.assertEqual(metrics["eval/done/max_steps/from/Level1-1"], 0)
        self.assertEqual(metrics["eval/done/max_steps/from/Level1-1/rate"], 0.0)
        self.assertEqual(metrics["eval/done/all/from/Level1-2"], 1)
        self.assertEqual(metrics["eval/done/level_change/from/Level1-2"], 0)
        self.assertEqual(metrics["eval/done/level_change/from/Level1-2/rate"], 0.0)
        self.assertEqual(metrics["eval/done/max_steps/from/Level1-2"], 1)
        self.assertEqual(metrics["eval/done/max_steps/from/Level1-2/rate"], 1.0)
        self.assertEqual(metrics["eval/done/level_change/from_rate/min"], 0.0)
        self.assertEqual(metrics["eval/done/level_change/from_rate/mean"], 0.5)
        self.assertEqual(metrics["episode_results"][0]["env_index"], 1)
        self.assertEqual(metrics["episode_results"][0]["start_state"], "Level1-2")
        self.assertEqual(metrics["episode_results"][0]["reward"], 2.0)
        self.assertEqual(metrics["episode_results"][1]["env_index"], 0)
        self.assertEqual(metrics["episode_results"][1]["start_state"], "Level1-1")
        self.assertEqual(metrics["episode_results"][1]["reward"], 4.0)


class DoneCounterCallbackTests(unittest.TestCase):
    def test_records_life_loss_level_change_max_steps_and_unclassified(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.records: dict[str, int | float] = {}

            def record(self, key: str, value: int | float) -> None:
                self.records[key] = value

        class FakeModel:
            def __init__(self) -> None:
                self.logger = FakeLogger()

        model = FakeModel()
        callback = DoneCounterCallback(default_state="Level1-1")
        callback.model = model  # type: ignore[assignment]
        callback.num_timesteps = 10
        callback.locals = {
            "dones": [True, True, True, True, False, True],
            "infos": [
                {
                    "start_state": "Level1-1",
                    "done_on_info": {
                        "life_loss": {
                            "op": "decrease",
                            "keys": ("lives",),
                            "prev": 3,
                            "next": 2,
                        },
                    },
                },
                {
                    "start_state": "Level1-2",
                    "done_on_info": {
                        "level_change": {
                            "op": "change",
                            "keys": ("levelHi", "levelLo"),
                            "prev": [0, 0],
                            "next": [0, 1],
                        },
                    },
                },
                {"start_state": "Level1-2", "TimeLimit.truncated": True},
                {"start_state": "Level1-1"},
                {"start_state": "Level1-1", "done_on_info": {"life_loss": {}}},
                {"global_reset": True, "done_on_info": {"life_loss": {}}},
            ],
        }

        self.assertTrue(callback._on_step())

        self.assertEqual(model.logger.records["train/done/all"], 4)
        self.assertEqual(model.logger.records["train/done/life_loss"], 1)
        self.assertEqual(model.logger.records["train/done/level_change"], 1)
        self.assertEqual(model.logger.records["train/done/max_steps"], 1)
        self.assertEqual(model.logger.records["train/done/unclassified"], 1)
        self.assertEqual(model.logger.records["train/done/life_loss/from/3"], 1)
        self.assertEqual(model.logger.records["train/done/level_change/from/0-0"], 1)
        self.assertNotIn("train/done/life_loss/from/3/ep_window/rate", model.logger.records)
        self.assertNotIn(
            "train/done/level_change/from/0-0/ep_window/rate",
            model.logger.records,
        )
        self.assertFalse(any("/to/" in key for key in model.logger.records))
        self.assertFalse(any(key.startswith("train/state/") for key in model.logger.records))
        self.assertNotIn("train/outcome/pooled_rate", model.logger.records)

    def test_multiple_done_reasons_share_one_all_count(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.records: dict[str, int | float] = {}

            def record(self, key: str, value: int | float) -> None:
                self.records[key] = value

        class FakeModel:
            def __init__(self) -> None:
                self.logger = FakeLogger()

        model = FakeModel()
        callback = DoneCounterCallback(default_state="Level1-1")
        callback.model = model  # type: ignore[assignment]
        callback.num_timesteps = 20
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "start_state": "Level1-1",
                    "done_on_info": {
                        "life_loss": {"op": "decrease", "prev": [3], "next": [2]},
                        "level_change": {"op": "change", "prev": (0, 0), "next": (0, 1)},
                    },
                    "TimeLimit.truncated": True,
                },
            ],
        }

        self.assertTrue(callback._on_step())

        self.assertEqual(model.logger.records["train/done/all"], 1)
        self.assertEqual(model.logger.records["train/done/life_loss"], 1)
        self.assertEqual(model.logger.records["train/done/level_change"], 1)
        self.assertEqual(model.logger.records["train/done/max_steps"], 1)
        self.assertEqual(model.logger.records["train/done/unclassified"], 0)
        self.assertEqual(model.logger.records["train/done/life_loss/from/3"], 1)
        self.assertEqual(model.logger.records["train/done/level_change/from/0-0"], 1)
        self.assertNotIn("train/done/life_loss/from/3/ep_window/rate", model.logger.records)
        self.assertNotIn(
            "train/done/level_change/from/0-0/ep_window/rate",
            model.logger.records,
        )

    def test_done_from_ep_window_rate_uses_100_matching_source_terminal_episode_window(
        self,
    ) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.records: dict[str, int | float] = {}

            def record(self, key: str, value: int | float) -> None:
                self.records[key] = value

        class FakeModel:
            def __init__(self) -> None:
                self.logger = FakeLogger()

        model = FakeModel()
        callback = DoneCounterCallback(
            default_state="Level1-1",
            done_on_info={"level_change": (("levelHi", "levelLo"), "change")},
        )
        callback.model = model  # type: ignore[assignment]

        for index in range(50):
            callback.num_timesteps = index
            callback.locals = {
                "dones": [True],
                "infos": [
                    {
                        "levelHi": 0,
                        "levelLo": 0,
                        "done_on_info": {
                            "level_change": {
                                "op": "change",
                                "prev": [0, 0],
                                "next": [0, 1],
                            },
                        },
                    },
                ],
            }
            self.assertTrue(callback._on_step())

        for index in range(50, 100):
            callback.num_timesteps = index
            callback.locals = {
                "dones": [True],
                "infos": [
                    {
                        "levelHi": 0,
                        "levelLo": 1,
                        "done_on_info": {
                            "level_change": {
                                "op": "change",
                                "prev": [0, 1],
                                "next": [0, 2],
                            },
                        },
                    },
                ],
            }
            self.assertTrue(callback._on_step())

        self.assertNotIn(
            "train/done/level_change/from/0-0/ep_window/rate",
            model.logger.records,
        )
        self.assertNotIn(
            "train/done/level_change/from/0-1/ep_window/rate",
            model.logger.records,
        )

        for index in range(100, 150):
            callback.num_timesteps = index
            callback.locals = {
                "dones": [True],
                "infos": [
                    {
                        "levelHi": 0,
                        "levelLo": 0,
                        "done_on_info": {"life_loss": {"op": "decrease", "prev": 2, "next": 1}},
                    },
                ],
            }
            self.assertTrue(callback._on_step())

        self.assertEqual(
            model.logger.records["train/done/level_change/from/0-0/ep_window/rate"],
            0.5,
        )
        self.assertNotIn(
            "train/done/level_change/from/0-1/ep_window/rate",
            model.logger.records,
        )
        self.assertNotIn(TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MIN, model.logger.records)

        for index in range(150, 225):
            callback.num_timesteps = index
            callback.locals = {
                "dones": [True],
                "infos": [
                    {
                        "levelHi": 0,
                        "levelLo": 1,
                        "done_on_info": {"life_loss": {"op": "decrease", "prev": 2, "next": 1}},
                    },
                ],
            }
            self.assertTrue(callback._on_step())

        self.assertEqual(
            model.logger.records["train/done/level_change/from/0-0/ep_window/rate"],
            0.5,
        )
        self.assertEqual(
            model.logger.records["train/done/level_change/from/0-1/ep_window/rate"],
            0.25,
        )
        self.assertEqual(model.logger.records[TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MIN], 0.25)
        self.assertEqual(model.logger.records[TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MEAN], 0.375)

        callback.num_timesteps = 225
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "levelHi": 0,
                    "levelLo": 1,
                    "done_on_info": {
                        "level_change": {
                            "op": "change",
                            "prev": [0, 1],
                            "next": [0, 2],
                        },
                    },
                },
            ],
        }
        self.assertTrue(callback._on_step())

        self.assertEqual(
            model.logger.records["train/done/level_change/from/0-0/ep_window/rate"],
            0.5,
        )
        self.assertAlmostEqual(
            model.logger.records["train/done/level_change/from/0-1/ep_window/rate"],
            0.25,
        )
        self.assertEqual(model.logger.records[TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MIN], 0.25)

    def test_logs_done_metrics_to_wandb(self) -> None:
        class FakeLogger:
            def __init__(self) -> None:
                self.records: dict[str, int | float] = {}

            def record(self, key: str, value: int | float) -> None:
                self.records[key] = value

        class FakeModel:
            def __init__(self) -> None:
                self.logger = FakeLogger()

        class FakeRun:
            def __init__(self) -> None:
                self.payloads: list[tuple[dict[str, object], int]] = []

            def log(self, payload: dict[str, object], *, step: int) -> None:
                self.payloads.append((payload, step))

        model = FakeModel()
        run = FakeRun()
        callback = DoneCounterCallback(wandb_run=run, default_state="Level1-1")
        callback.model = model  # type: ignore[assignment]
        callback.num_timesteps = 30
        callback.locals = {"dones": [True], "infos": [{"done_on_info": "life_loss"}]}

        self.assertTrue(callback._on_step())

        self.assertEqual(run.payloads[0][1], 30)
        self.assertEqual(run.payloads[0][0]["global_step"], 30)
        self.assertEqual(run.payloads[0][0]["train/done/all"], 1)
        self.assertEqual(run.payloads[0][0]["train/done/life_loss"], 1)


class OutcomeCounterCallbackTests(unittest.TestCase):
    class FakeLogger:
        def __init__(self) -> None:
            self.records: dict[str, int | float] = {}

        def record(self, key: str, value: int | float) -> None:
            self.records[key] = value

    class FakeModel:
        def __init__(self) -> None:
            self.logger = OutcomeCounterCallbackTests.FakeLogger()

    def make_callback(self) -> tuple[OutcomeCounterCallback, FakeModel]:
        model = self.FakeModel()
        callback = OutcomeCounterCallback(
            info_events={"level_change": (("levelHi", "levelLo"), "change")},
        )
        callback.model = model  # type: ignore[assignment]
        return callback, model

    def test_records_nonterminal_level_change_attempt_fires(self) -> None:
        callback, model = self.make_callback()

        callback.num_timesteps = 1
        callback.locals = {
            "dones": [False],
            "infos": [
                {
                    "levelHi": 0,
                    "levelLo": 1,
                    "info_events": {
                        "level_change": {
                            "op": "change",
                            "keys": ("levelHi", "levelLo"),
                            "prev": (0, 0),
                            "next": (0, 1),
                        },
                    },
                },
            ],
        }
        self.assertTrue(callback._on_step())

        callback.num_timesteps = 2
        callback.locals = {
            "dones": [False],
            "infos": [
                {
                    "levelHi": 0,
                    "levelLo": 2,
                    "info_events": {
                        "level_change": {
                            "op": "change",
                            "keys": ("levelHi", "levelLo"),
                            "prev": (0, 1),
                            "next": (0, 2),
                        },
                    },
                },
            ],
        }
        self.assertTrue(callback._on_step())

        self.assertEqual(model.logger.records["train/event/level_change"], 2)
        self.assertEqual(model.logger.records["train/event/level_change/from/0-0"], 1)
        self.assertEqual(model.logger.records["train/event/level_change/from/0-1"], 1)
        self.assertEqual(
            model.logger.records["train/outcome/level_change/from/0-0/attempts"],
            1,
        )
        self.assertEqual(
            model.logger.records["train/outcome/level_change/from/0-0/fires"],
            1,
        )
        self.assertEqual(
            model.logger.records["train/outcome/level_change/from/0-1/attempts"],
            1,
        )
        self.assertEqual(
            model.logger.records["train/outcome/level_change/from/0-1/fires"],
            1,
        )

    def test_records_current_source_failure_on_life_loss(self) -> None:
        callback, model = self.make_callback()
        callback.num_timesteps = 1
        callback.locals = {
            "dones": [False],
            "infos": [{"levelHi": 0, "levelLo": 1}],
        }
        self.assertTrue(callback._on_step())

        callback.num_timesteps = 2
        callback.locals = {
            "dones": [False],
            "infos": [
                {
                    "levelHi": 0,
                    "levelLo": 1,
                    "died": True,
                    "info_events": {
                        "life_loss": {
                            "op": "decrease",
                            "keys": ("lives",),
                            "prev": 3,
                            "next": 2,
                        },
                    },
                },
            ],
        }
        self.assertTrue(callback._on_step())

        self.assertEqual(
            model.logger.records["train/outcome/level_change/from/0-1/attempts"],
            1,
        )
        self.assertEqual(
            model.logger.records["train/outcome/level_change/from/0-1/fires"],
            0,
        )

    def test_attempt_window_rates_cover_terminal_and_nonterminal_events(self) -> None:
        callback, model = self.make_callback()
        callback.ep_window_size = 2

        for step, done in enumerate((False, True), start=1):
            callback.num_timesteps = step
            callback.locals = {
                "dones": [done],
                "infos": [
                    {
                        "levelHi": 0,
                        "levelLo": 1,
                        "info_events": {
                            "level_change": {
                                "op": "change",
                                "keys": ("levelHi", "levelLo"),
                                "prev": (0, 0),
                                "next": (0, 1),
                            },
                        },
                    },
                ],
            }
            self.assertTrue(callback._on_step())

        self.assertEqual(
            model.logger.records[
                "train/outcome/level_change/from/0-0/attempt_window/rate"
            ],
            1.0,
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
                ("throughput/rollout_fps", 50.0),
                ("throughput/rollout_fps", 60.0),
                ("throughput/loop_fps", 20.0),
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
        self.assertEqual(records["rollout/value_pred/mean"], 2.5)
        self.assertAlmostEqual(records["rollout/value_pred/std"], float(np.std([1.0, 2.0, 3.0, 4.0])))
        self.assertEqual(records["rollout/value_pred/min"], 1.0)
        self.assertEqual(records["rollout/value_pred/max"], 4.0)
        self.assertEqual(records["rollout/value_pred/abs_mean"], 2.5)
        self.assertEqual(records["rollout/advantage/mean"], 0.5)
        self.assertAlmostEqual(records["rollout/advantage/std"], float(np.std([-1.0, 0.0, 1.0, 2.0])))
        self.assertEqual(records["rollout/advantage/min"], -1.0)
        self.assertEqual(records["rollout/advantage/max"], 2.0)
        self.assertEqual(records["rollout/advantage/abs_mean"], 1.0)


class RewardComponentDiagnosticsCallbackTests(unittest.TestCase):
    def test_logs_rollout_reward_component_stats(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: list[tuple[str, float]] = []

            def record(self, key: str, value: float) -> None:
                self.records.append((key, value))

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()

        callback = RewardComponentDiagnosticsCallback()
        callback.model = Model()  # type: ignore[assignment]
        callback.locals = {
            "infos": [
                {
                    "shaped_reward": 10.0,
                    "progress_reward_component": 8.0,
                    "score_reward_component": 2.0,
                    "death_penalty_component": 0.0,
                },
                {
                    "shaped_reward": -25.0,
                    "progress_reward_component": 0.0,
                    "score_reward_component": 0.0,
                    "death_penalty_component": -25.0,
                },
            ],
        }

        self.assertTrue(callback._on_step())
        callback._on_rollout_end()

        records = dict(callback.model.logger.records)
        self.assertEqual(records["train/reward/shaped/mean"], -7.5)
        self.assertEqual(records["train/reward/shaped/min"], -25.0)
        self.assertEqual(records["train/reward/shaped/max"], 10.0)
        self.assertEqual(records["train/reward/prog_x/mean"], 4.0)
        self.assertEqual(records["train/reward/prog_x/nonzero_rate"], 0.5)
        self.assertEqual(records["train/reward/score/nonzero_rate"], 0.5)
        self.assertEqual(records["train/reward/death/abs_mean"], 12.5)
        self.assertAlmostEqual(records["train/reward_share/prog_x"], 8.0 / 35.0)
        self.assertAlmostEqual(records["train/reward_share/score"], 2.0 / 35.0)
        self.assertAlmostEqual(records["train/reward_share/death"], 25.0 / 35.0)
        self.assertEqual(records["train/reward_share/done"], 0.0)
        self.assertEqual(records["train/reward_share/time"], 0.0)
        self.assertEqual(records["train/reward_share/native"], 0.0)

    def test_logs_rollout_reward_component_shares_with_negative_components(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: dict[str, float] = {}

            def record(self, key: str, value: float) -> None:
                self.records[key] = value

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()

        callback = RewardComponentDiagnosticsCallback()
        callback.model = Model()  # type: ignore[assignment]
        callback.locals = {
            "infos": [
                {
                    "progress_reward_component": 3.0,
                    "score_reward_component": 2.0,
                    "death_penalty_component": -4.0,
                    "completion_reward_component": 5.0,
                    "time_penalty_component": -1.0,
                    "native_reward_component": -5.0,
                },
            ],
        }

        self.assertTrue(callback._on_step())
        callback._on_rollout_end()

        records = callback.model.logger.records
        self.assertAlmostEqual(records["train/reward_share/prog_x"], 3.0 / 20.0)
        self.assertAlmostEqual(records["train/reward_share/score"], 2.0 / 20.0)
        self.assertAlmostEqual(records["train/reward_share/death"], 4.0 / 20.0)
        self.assertAlmostEqual(records["train/reward_share/done"], 5.0 / 20.0)
        self.assertAlmostEqual(records["train/reward_share/time"], 1.0 / 20.0)
        self.assertAlmostEqual(records["train/reward_share/native"], 5.0 / 20.0)

    def test_logs_zero_reward_component_shares_when_rollout_has_no_component_magnitude(
        self,
    ) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: dict[str, float] = {}

            def record(self, key: str, value: float) -> None:
                self.records[key] = value

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()

        callback = RewardComponentDiagnosticsCallback()
        callback.model = Model()  # type: ignore[assignment]

        callback._on_rollout_end()

        records = callback.model.logger.records
        for component in ("prog_x", "score", "death", "done", "time", "native"):
            self.assertEqual(records[f"train/reward_share/{component}"], 0.0)


if __name__ == "__main__":
    unittest.main()
