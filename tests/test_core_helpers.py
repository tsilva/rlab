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

import rlab.metric_names as metric_names
from rlab.artifacts import (
    apply_model_config_defaults,
    apply_config_defaults,
    build_s3_artifact_uri,
    checkpoint_step,
    env_config_from_config_dict,
    env_config_from_model_metadata,
    env_config_from_metadata,
    explicit_arg_dests,
    init_wandb,
    load_model_metadata,
    log_wandb_model_artifact,
    model_metadata_path,
    require_training_metadata,
    write_model_metadata,
)
from rlab.callbacks import (
    DoneCounterCallback,
    LevelCompleteInfoCallback,
    MetricThresholdStopCallback,
    RewardComponentDiagnosticsCallback,
    RolloutDiagnosticsCallback,
    ThroughputCallback,
    TimeElapsedCallback,
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
    make_retro_env,
    make_rendered_replay_env,
    make_training_vec_env,
    make_vec_envs,
    native_vec_env_supports_done_on,
    needs_vec_transpose_image,
    resolve_env_config,
    resolve_mixed_state_config,
    state_name_candidates_from_level_id,
)
from rlab.env_config import (
    env_config_from_args,
    parse_info_events,
    parse_state_probs,
    parse_states,
)
from rlab.env_config_aliases import normalize_provider_env_config_aliases
from rlab.eval_metrics import episode_rank, is_level_complete, run_eval_episode
from rlab.eval_runner import evaluate_model_episodes
from rlab.metric_names import (
    TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MEAN,
    TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
    TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
    metric_path_segment,
)
from rlab.model_sources import (
    ResolvedModelSource,
    apply_model_source_defaults,
    model_artifact_checkpoint_step,
    single_model_artifact_ref,
)
from rlab.play import build_parser as build_play_parser
from rlab.play import model_observation
from rlab.play import playback_should_end_episode
from rlab.play import task_conditioning_change_message
from rlab.play import task_conditioning_start_message
from rlab.eval import build_parser as build_eval_parser
from rlab.eval import already_evaluated_artifact
from rlab.eval import eval_seed_for_checkpoint
from rlab.eval import evaluate_checkpoint
from rlab.eval import log_wandb_eval
from rlab.eval import main as eval_main
from rlab.eval import score as eval_checkpoint_score
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.task_advantage import normalize_advantages_by_task
from rlab.targets import SuperMarioBrosNesV0Target, target_for_game
from rlab.train import (
    Sb3HumanOutputFormatCallback,
    disable_sb3_human_output_truncation,
    log_checkpoint_eval_metrics,
)
from rlab.wandb_artifacts import (
    artifact_download_dir,
    model_artifact_ref,
    safe_artifact_stem,
)
from rlab.wandb_artifacts import metadata_from_wandb_artifact


class EnvConfigAliasTests(unittest.TestCase):
    def test_stable_retro_turbo_101_provider_keywords_map_to_env_config(self) -> None:
        config = normalize_provider_env_config_aliases(
            {
                "num_envs": 16,
                "num_threads": 4,
                "maxpool_last_two": False,
                "sticky_action_prob": 0.25,
                "noop_reset_max": 3,
            }
        )

        self.assertEqual(config["n_envs"], 16)
        self.assertEqual(config["env_threads"], 4)
        self.assertEqual(config["max_pool_frames"], False)
        self.assertEqual(config["sticky_action_prob"], 0.25)
        self.assertNotIn("num_envs", config)
        self.assertNotIn("num_threads", config)
        self.assertNotIn("maxpool_last_two", config)
        self.assertNotIn("noop_reset_max", config)

    def test_legacy_stable_retro_turbo_provider_keywords_still_map_to_env_config(self) -> None:
        config = normalize_provider_env_config_aliases(
            {
                "frame_maxpool": False,
                "action_sticky_prob": 0.25,
                "reset_noops": 3,
            }
        )

        self.assertEqual(config["max_pool_frames"], False)
        self.assertEqual(config["sticky_action_prob"], 0.25)
        self.assertNotIn("frame_maxpool", config)
        self.assertNotIn("action_sticky_prob", config)
        self.assertNotIn("reset_noops", config)

    def test_reward_env_wrapper_resolves_to_mario_reward_config(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                game="SuperMarioBros-Nes-v0",
                env_wrappers=(
                    {
                        "id": "SuperMarioBrosNesRewardEnvWrapper",
                        "kwargs": {
                            "reward_mode": "score",
                            "progress_reward_scale": 1.5,
                            "death_penalty": 17,
                            "score_progress_clipped": True,
                        },
                    },
                ),
            )
        )

        self.assertEqual(config.reward_mode, "score")
        self.assertEqual(config.progress_reward_scale, 1.5)
        self.assertEqual(config.death_penalty, 17.0)
        self.assertTrue(config.score_progress_clipped)
        self.assertEqual(
            config.env_wrappers,
            (
                {
                    "id": "SuperMarioBrosNesProgressInfoWrapper",
                    "kwargs": {},
                },
                {
                    "id": "SuperMarioBrosNesRewardEnvWrapper",
                    "kwargs": {
                        "reward_mode": "score",
                        "progress_reward_scale": 1.5,
                        "death_penalty": 17.0,
                        "score_progress_clipped": True,
                    },
                },
            ),
        )

    def test_unknown_env_wrapper_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown"):
            resolve_env_config(
                EnvConfig(
                    game="SuperMarioBros-Nes-v0",
                    env_wrappers=({"id": "MissingRewardWrapper", "kwargs": {}},),
                )
            )

    def test_metadata_config_dict_preserves_env_wrappers(self) -> None:
        config = env_config_from_config_dict(
            {
                "game": "SuperMarioBros-Nes-v0",
                "env_wrappers": [
                    {
                        "id": "SuperMarioBrosNesRewardEnvWrapper",
                        "kwargs": {"reward_mode": "score", "terminal_reward": 30},
                    }
                ],
            }
        )

        self.assertIsNotNone(config)
        resolved = resolve_env_config(config)
        self.assertEqual(resolved.reward_mode, "score")
        self.assertEqual(resolved.terminal_reward, 30.0)


class Sb3LoggerTests(unittest.TestCase):
    def test_human_output_truncation_is_disabled_for_long_level_complete_metrics(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat

        key_values = {
            "train/info/level_complete/from/Level1-2_bonus_room_checkpoint/count": 1,
            "train/info/level_complete/from/Level1-2_bonus_room_checkpoint/rate": 0.0,
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

    def test_uninitialized_sb3_logger_is_ignored(self) -> None:
        class FakeSb3Model:
            @property
            def logger(self):
                raise AttributeError("'FakeSb3Model' object has no attribute '_logger'")

        disable_sb3_human_output_truncation(FakeSb3Model())

    def test_callback_updates_logger_after_training_starts(self) -> None:
        from stable_baselines3.common.logger import HumanOutputFormat

        output_format = HumanOutputFormat(io.StringIO())

        class FakeLogger:
            output_formats = [output_format]

        class FakeModel:
            _logger = FakeLogger()

        callback = Sb3HumanOutputFormatCallback(max_length=256)
        callback.model = FakeModel()
        callback._on_training_start()

        self.assertEqual(output_format.max_length, 256)


class MetricsDocumentationTests(unittest.TestCase):
    def test_metrics_reference_mentions_metric_name_constants_and_core_templates(self) -> None:
        metrics_doc = Path(__file__).resolve().parents[1] / "METRICS.md"
        content = metrics_doc.read_text(encoding="utf-8")

        constant_values = sorted(
            value
            for name, value in vars(metric_names).items()
            if name.isupper() and isinstance(value, str)
        )
        missing_constants = [value for value in constant_values if value not in content]
        self.assertEqual(missing_constants, [])

        required_templates = [
            "train/done/<reason>/from/<prev>",
            "train/done/<reason>/from/<prev>/ep_window/rate",
            "train/done/<reason>/from_rate/min",
            "train/done/<reason>/from_rate/mean",
            "train/info/level_complete/from/<prev>/count",
            "train/info/level_complete/from/<prev>/rate",
            "train/info/level_complete/rate/min/last",
            "train/info/level_complete/rate/mean/last",
            "train/reward/<component>/<stat>",
            "train/reward_share/<component>",
            "eval/done/<reason>/from/<start>",
            "eval/info/level_complete/rate/min/last",
            "eval/info/level_complete/rate/mean/last",
        ]
        missing_templates = [template for template in required_templates if template not in content]
        self.assertEqual(missing_templates, [])


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
            with (
                patch("rlab.artifacts.load_wandb_env"),
                patch.dict(sys.modules, {"wandb": fake_wandb}),
            ):
                run = init_wandb(args, tmp_dir, EnvConfig(game="SuperMarioBros-Nes-v0"))

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
            info_events_json='{"life_loss":["lives","decrease"]}',
            done_on_events="life_loss",
            action_set="right",
        )
        config = env_config_from_args(args, max_episode_steps_attr="max_steps")
        self.assertEqual(config.max_episode_steps, 123)
        self.assertEqual(config.action_set, "right")
        self.assertEqual(config.sticky_action_prob, 0.25)
        self.assertEqual(config.info_events, {"life_loss": ("lives", "decrease")})
        self.assertEqual(config.done_on_events, ("life_loss",))

    def test_parse_info_events_accepts_single_and_multi_key_rules(self) -> None:
        self.assertEqual(
            parse_info_events(
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

    def test_resolve_env_config_allows_named_done_events_without_local_rules(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                game="SuperMarioBros-Nes-v0",
                done_on_events=("life_loss",),
            )
        )

        self.assertEqual(config.info_events, {})
        self.assertEqual(config.done_on_events, ("life_loss",))

    def test_resolve_env_config_preserves_explicit_info_events(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                game="SuperMarioBros-Nes-v0",
                info_events={
                    "life_loss": ("lives", "decrease"),
                    "level_change": (("levelHi", "levelLo"), "change"),
                },
                done_on_events=("life_loss", "level_change"),
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

    def test_train_config_json_accepts_env_wrappers(self) -> None:
        env_wrappers = [
            {
                "id": "SuperMarioBrosNesRewardEnvWrapper",
                "kwargs": {"reward_mode": "score", "death_penalty": 12},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps(
                    {
                        "game": "SuperMarioBros-Nes-v0",
                        "env_wrappers": env_wrappers,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            args = parse_train_args(["--train-config-json", str(path)])
            config = resolve_env_config(env_config_from_args(args))

        self.assertEqual(args.env_wrappers, env_wrappers)
        self.assertEqual(config.reward_mode, "score")
        self.assertEqual(config.death_penalty, 12.0)

    def test_train_config_json_accepts_metric_early_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps(
                    {
                        "early_stop_metric": TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                        "early_stop_threshold": 0.99,
                        "early_stop_operator": ">",
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            args = parse_train_args(["--train-config-json", str(path)])

        self.assertEqual(args.early_stop_metric, TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST)
        self.assertEqual(args.early_stop_threshold, 0.99)
        self.assertEqual(args.early_stop_operator, ">")

    def test_train_config_json_accepts_structured_early_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            early_stop = [
                {
                    "metric": TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                    "operator": ">",
                    "threshold": 0.99,
                },
                {
                    "metric": "rollout/ep_rew_mean",
                    "operator": ">=",
                    "threshold": 1000,
                },
            ]
            path.write_text(json.dumps({"early_stop": early_stop}) + "\n", encoding="utf-8")

            args = parse_train_args(["--train-config-json", str(path)])

        self.assertEqual(args.early_stop, early_stop)
        self.assertEqual(args.early_stop_metric, "")
        self.assertIsNone(args.early_stop_threshold)

    def test_train_config_json_rejects_incomplete_metric_early_stop(self) -> None:
        with self.assertRaisesRegex(ValueError, "early-stop-metric"):
            parse_train_args(["--early-stop-metric", TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST])

        with self.assertRaisesRegex(ValueError, "early-stop-metric"):
            parse_train_args(["--early-stop-threshold", "0.99"])

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            parse_train_args(
                [
                    "--early-stop",
                    json.dumps(
                        {
                            "metric": TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                            "operator": ">",
                            "threshold": 0.99,
                        }
                    ),
                    "--early-stop-metric",
                    TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                    "--early-stop-threshold",
                    "0.99",
                ]
            )

    def test_build_train_command_includes_metric_early_stop_flags(self) -> None:
        command = build_train_command(
            {
                "early_stop_metric": TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                "early_stop_threshold": 0.99,
                "early_stop_operator": ">",
            }
        )

        self.assertIn("--early-stop-metric", command)
        self.assertIn(TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST, command)
        self.assertIn("--early-stop-threshold", command)
        self.assertIn("0.99", command)
        self.assertIn("--early-stop-operator", command)
        self.assertIn(">", command)

    def test_build_train_command_includes_structured_early_stop(self) -> None:
        early_stop = [
            {
                "metric": TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                "operator": ">",
                "threshold": 0.99,
            }
        ]
        command = build_train_command({"early_stop": early_stop})

        self.assertIn("--early-stop", command)
        self.assertIn(
            '[{"metric":"train/info/level_complete/rate/min/last","operator":">","threshold":0.99}]',
            command,
        )

    def test_build_train_command_serializes_env_wrappers(self) -> None:
        command = build_train_command(
            {
                "env_wrappers": [
                    {
                        "id": "SuperMarioBrosNesRewardEnvWrapper",
                        "kwargs": {"reward_mode": "score"},
                    }
                ],
            }
        )

        self.assertIn("--env-wrappers", command)
        self.assertIn(
            '[{"id":"SuperMarioBrosNesRewardEnvWrapper","kwargs":{"reward_mode":"score"}}]',
            command,
        )

    def test_training_loop_eval_settings_must_stay_disabled(self) -> None:
        args = parse_train_args(["--eval-freq", "0", "--eval-episodes", "0"])
        self.assertEqual(args.eval_freq, 0)
        self.assertEqual(args.eval_episodes, 0)

        with self.assertRaisesRegex(ValueError, "training-loop eval is disabled"):
            parse_train_args(["--eval-freq", "1"])

        with self.assertRaisesRegex(ValueError, "training-loop eval is disabled"):
            parse_train_args(["--eval-episodes", "1"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(json.dumps({"eval_freq": 1}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training-loop eval is disabled"):
                parse_train_args(["--train-config-json", str(path)])

    def test_train_parser_defaults_to_sparse_checkpoint_artifacts(self) -> None:
        args = build_train_parser().parse_args([])

        self.assertEqual(args.checkpoint_freq, 500_000)

    def test_train_parser_rejects_eval_reserved_seed_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved for eval"):
            parse_train_args(["--seed", "10000"])

        with self.assertRaisesRegex(ValueError, "training env slot"):
            parse_train_args(["--seed", "9999", "--n-envs", "2"])

        self.assertEqual(parse_train_args(["--seed", "9999", "--n-envs", "1"]).seed, 9999)

    def test_train_config_json_rejects_eval_reserved_seed_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(json.dumps({"seed": DEFAULT_EVAL_SEED}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reserved for eval"):
                parse_train_args(["--train-config-json", str(path)])

    def test_parse_info_events_rejects_invalid_shapes(self) -> None:
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
                    parse_info_events(value)

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

    def test_unknown_env_provider_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown environment provider"):
            resolve_env_config(
                EnvConfig(env_provider="stable-retro", game="SuperMarioBros-Nes-v0")
            )

    def test_make_retro_env_uses_provider_factory(self) -> None:
        sentinel = object()
        wrapped = object()
        config = EnvConfig(game="SuperMarioBros-Nes-v0")

        with (
            patch("rlab.env.make_provider_env", return_value=sentinel) as make_provider_env,
            patch("rlab.env.wrap_retro_env", return_value=wrapped) as wrap_retro_env,
        ):
            env = make_retro_env(config=config, seed=7)

        self.assertIs(env, wrapped)
        make_provider_env.assert_called_once()
        self.assertEqual(make_provider_env.call_args.kwargs["render_mode"], "rgb_array")
        wrap_retro_env.assert_called_once_with(sentinel, config=resolve_env_config(config), seed=7)

    def test_make_vec_envs_uses_provider_factory(self) -> None:
        class FakeNative:
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)

            def seed(self, seed):
                self.seed_value = seed
                return [seed]

        fake_native = FakeNative()
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
        )
        with (
            patch("rlab.env.make_provider_vec_env", return_value=fake_native) as make_provider_vec_env,
            patch("rlab.env.VecRetroProgressInfo", side_effect=lambda env, config: env),
            patch("rlab.env.VecMonitor", side_effect=lambda env: env),
            patch("rlab.env.maybe_transpose_vec_image", side_effect=lambda env: env),
        ):
            env = make_vec_envs(config=config, n_envs=2, seed=7)

        self.assertIs(env, fake_native)
        self.assertEqual(fake_native.seed_value, 7)
        make_provider_vec_env.assert_called_once()
        self.assertEqual(make_provider_vec_env.call_args.kwargs["native_kwargs"]["num_envs"], 2)

    def test_make_vec_envs_dispatches_to_supermariobrosnes_turbo_provider(self) -> None:
        class FakeSuperMarioNative:
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)
            calls = []

            def __init__(self, game, **kwargs):
                self.game = game
                self.kwargs = kwargs
                self.__class__.calls.append((game, kwargs))

            def seed(self, seed):
                self.seed_value = seed
                return [seed]

        FakeSuperMarioNative.calls = []
        config = EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            frame_skip=4,
            max_pool_frames=False,
            sticky_action_prob=0.0,
        )
        with (
            patch(
                "rlab.env._super_mario_bros_nes_turbo_vec_env_type",
                return_value=FakeSuperMarioNative,
            ),
            patch("rlab.env.VecRetroProgressInfo", side_effect=lambda env, config: env),
            patch("rlab.env.VecMonitor", side_effect=lambda env: env),
            patch("rlab.env.maybe_transpose_vec_image", side_effect=lambda env: env),
        ):
            env = make_vec_envs(config=config, n_envs=2, seed=7)

        self.assertIsInstance(env, FakeSuperMarioNative)
        self.assertEqual(env.seed_value, 7)
        self.assertEqual(FakeSuperMarioNative.calls[0][0], "SuperMarioBros-Nes-v0")
        native_kwargs = FakeSuperMarioNative.calls[0][1]
        self.assertEqual(native_kwargs["num_envs"], 2)
        self.assertEqual(native_kwargs["frame_skip"], 4)
        self.assertEqual(native_kwargs["obs_resize"], (84, 84))
        self.assertEqual(native_kwargs["maxpool_last_two"], False)

    def test_eval_vec_env_preserves_requested_terminal_info_events(self) -> None:
        sentinel = object()
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            info_events={
                "life_loss": ("lives", "decrease"),
                "level_change": (("levelHi", "levelLo"), "change"),
            },
            done_on_events=("life_loss", "level_change"),
        )

        with patch("rlab.env.make_vec_envs", return_value=sentinel) as make_vec_envs:
            env = make_eval_vec_env(config=config, n_envs=2, seed=7)

        self.assertIs(env, sentinel)
        passed_config = make_vec_envs.call_args.kwargs["config"]
        self.assertEqual(passed_config.info_events, config.info_events)
        self.assertEqual(passed_config.done_on_events, ("life_loss", "level_change"))

    def test_training_vec_env_preserves_requested_terminal_info_events(self) -> None:
        sentinel = object()
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            info_events={"life_loss": ("lives", "decrease")},
            done_on_events=("life_loss",),
        )

        with patch("rlab.env.make_vec_envs", return_value=sentinel) as make_vec_envs:
            env = make_training_vec_env(config=config, n_envs=2, seed=7)

        self.assertIs(env, sentinel)
        passed_config = make_vec_envs.call_args.kwargs["config"]
        self.assertEqual(passed_config.info_events, {"life_loss": ("lives", "decrease")})
        self.assertEqual(passed_config.done_on_events, ("life_loss",))

    def test_rendered_eval_replay_preserves_requested_terminal_info_events(self) -> None:
        sentinel = object()
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            info_events={
                "life_loss": ("lives", "decrease"),
                "level_change": (("levelHi", "levelLo"), "change"),
            },
            done_on_events=("life_loss", "level_change"),
        )

        with patch("rlab.env.make_retro_env", return_value=sentinel) as make_retro_env:
            env = make_rendered_replay_env(config=config, seed=7)

        self.assertIs(env, sentinel)
        passed_config = make_retro_env.call_args.kwargs["config"]
        self.assertEqual(passed_config.info_events, config.info_events)
        self.assertEqual(passed_config.done_on_events, ("life_loss", "level_change"))

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

    def test_mario_completion_uses_level_change(self) -> None:
        config = argparse.Namespace(
            reward_mode="score",
            no_progress_min_delta=0,
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

    def test_vector_progress_does_not_apply_python_truncation_or_global_reset(self) -> None:
        class FakeVecEnv:
            num_envs = 2
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.MultiBinary(2)

            def __init__(self) -> None:
                self.reset_count = 0
                self.reset_infos = [
                    {"lives": 3, "score": 0, "levelHi": 0, "levelLo": 0},
                    {"lives": 3, "score": 0, "levelHi": 0, "levelLo": 1},
                ]

            def reset(self):
                self.reset_count += 1
                return np.zeros((self.num_envs, 4, 84, 84), dtype=np.uint8)

            def step_async(self, actions) -> None:
                self.actions = actions

            def step_wait(self):
                return (
                    np.ones((self.num_envs, 4, 84, 84), dtype=np.uint8),
                    np.array([0.0, 0.0], dtype=np.float32),
                    np.array([False, False]),
                    [
                        {
                            "lives": 3,
                            "score": 0,
                            "levelHi": 0,
                            "levelLo": 0,
                            "xscrollHi": 0,
                            "xscrollLo": 0,
                        },
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

        fake_vec = FakeVecEnv()
        config = resolve_env_config(
            EnvConfig(
                game="SuperMarioBros-Nes-v0",
                reward_mode="score",
                max_episode_steps=1,
                no_progress_timeout_steps=1,
            )
        )
        env = VecRetroProgressInfo(fake_vec, config)

        env.reset()
        env.step_async(np.array([0, 0], dtype=np.int64))
        _obs, _rewards, dones, infos = env.step_wait()

        self.assertEqual(fake_vec.reset_count, 1)
        np.testing.assert_array_equal(dones, np.array([False, False]))
        self.assertFalse(any(info.get("global_reset") for info in infos))
        self.assertFalse(any(info.get("TimeLimit.truncated") for info in infos))

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
            patch("rlab.env.RetroVecEnv", FakeNative),
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
        self.assertEqual(created[0]["obs_layout"], "chw")
        self.assertEqual(created[0]["maxpool_last_two"], True)
        self.assertEqual(created[0]["obs_copy"], "safe_view")
        self.assertNotIn("frame_maxpool", created[0])
        self.assertNotIn("copy_observations", created[0])
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
            patch("rlab.env.RetroVecEnv", FakeNative),
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

    def test_training_vec_env_passes_sticky_action_prob_to_native_vec_env(self) -> None:
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
            sticky_action_prob=0.25,
        )
        with (
            patch("rlab.env.RetroVecEnv", FakeNative),
            patch("rlab.env.VecRetroProgressInfo", side_effect=lambda env, config: env),
            patch("rlab.env.VecMonitor", side_effect=lambda env: env),
            patch("rlab.env.maybe_transpose_vec_image", side_effect=lambda env: env),
        ):
            env = make_training_vec_env(config, n_envs=4, seed=7)

        self.assertIsInstance(env, FakeNative)
        self.assertEqual(created[0]["sticky_action_prob"], 0.25)
        self.assertNotIn("action_sticky_prob", created[0])

    def test_training_vec_env_passes_configured_native_done_on_rules(self) -> None:
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
            done_on_events=("life_loss", "level_change"),
            states=("Level1-1", "Level1-2"),
            state_probs=(0.5, 0.5),
        )
        with (
            patch("rlab.env.RetroVecEnv", FakeNative),
            patch("rlab.env.native_vec_env_supports_done_on", return_value=True),
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
            created[0]["done_on"],
            {
                "life_loss": ("lives", "decrease"),
                "level_change": (("levelHi", "levelLo"), "change"),
            },
        )
        self.assertNotIn("done_on_info", created[0])
        self.assertNotIn("terminate_on_life_loss", created[0])
        self.assertNotIn("life_variable", created[0])
        self.assertEqual(progress_configs[0].info_events, config.info_events)
        self.assertEqual(progress_configs[0].done_on_events, ("life_loss", "level_change"))

    def test_training_vec_env_passes_only_done_events_to_native_done_on(self) -> None:
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
            patch("rlab.env.RetroVecEnv", FakeNative),
            patch("rlab.env.native_vec_env_supports_done_on", return_value=True),
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
        self.assertEqual(created[0]["done_on"], {"life_loss": ("lives", "decrease")})

    def test_training_vec_env_passes_named_done_events_without_local_rules(self) -> None:
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
            done_on_events=("life_loss", "level_change"),
            states=("Level1-1", "Level1-2"),
            state_probs=(0.5, 0.5),
        )
        with (
            patch("rlab.env.RetroVecEnv", FakeNative),
            patch("rlab.env.native_vec_env_supports_done_on", return_value=True),
            patch("rlab.env.native_vec_env_supports_named_done_on", return_value=True),
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
        self.assertEqual(
            created[0]["done_on"],
            {"life_loss": None, "level_change": None},
        )

    def test_training_vec_env_requires_native_done_on_support_when_rules_requested(
        self,
    ) -> None:
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
            info_events={"life_loss": ("lives", "decrease")},
            done_on_events=("life_loss",),
        )
        with (
            patch("rlab.env.RetroVecEnv", FakeNative),
            patch("rlab.env.native_vec_env_supports_done_on", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "done_on support"):
                make_training_vec_env(config, n_envs=1, seed=7)

    def test_training_vec_env_requires_named_done_on_support_for_unresolved_events(
        self,
    ) -> None:
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            done_on_events=("life_loss",),
        )
        with (
            patch("rlab.env.native_vec_env_supports_done_on", return_value=True),
            patch("rlab.env.native_vec_env_supports_named_done_on", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "metadata-backed named event support"):
                make_training_vec_env(config, n_envs=1, seed=7)

    def test_done_on_support_detection(self) -> None:
        class FakeNative:
            def __init__(self, game, *, done_on=None):
                self.game = game
                self.done_on = done_on

        with patch("rlab.env.RetroVecEnv", FakeNative):
            self.assertTrue(native_vec_env_supports_done_on())

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
            patch("rlab.env.RetroVecEnv", FakeNative),
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
                return (
                    self.reset(),
                    np.zeros(4, dtype=np.float32),
                    np.zeros(4, dtype=bool),
                    [{}, {}, {}, {}],
                )

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
                return (
                    self.reset(),
                    np.zeros(1, dtype=np.float32),
                    np.zeros(1, dtype=bool),
                    [{"level_id": "0-1"}],
                )

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
                return (
                    self.reset(),
                    np.zeros(1, dtype=np.float32),
                    np.zeros(1, dtype=bool),
                    [{"levelHi": 0, "levelLo": 1, "level_id": "not-used"}],
                )

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

    def test_build_train_command_omits_training_loop_eval_toggles(self) -> None:
        cmd = build_train_command(
            {
                "eval_stochastic": False,
                "no_eval_videos": True,
                "eval_video_fps": 60,
                "eval_video_scale": 2,
            }
        )

        self.assertNotIn("--no-eval-stochastic", cmd)
        self.assertNotIn("--no-eval-videos", cmd)
        self.assertNotIn("--eval-video-fps", cmd)
        self.assertNotIn("--eval-video-scale", cmd)

    def test_train_parser_accepts_task_conditioning_and_info_event_flags(self) -> None:
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
                "--info-events-json",
                '{"life_loss":["lives","decrease"],"level_change":[["levelHi","levelLo"],"change"]}',
                "--done-on-events",
                "life_loss,level_change",
            ]
        )

        self.assertEqual(args.task_conditioning_info_vars, "levelHi,levelLo")
        self.assertEqual(args.task_conditioning_info_values, "0,0;0,1")
        config = env_config_from_args(args, include_states=True)
        self.assertEqual(
            config.info_events,
            {
                "life_loss": ("lives", "decrease"),
                "level_change": (("levelHi", "levelLo"), "change"),
            },
        )
        self.assertEqual(config.done_on_events, ("life_loss", "level_change"))

    def test_train_parser_rejects_done_on_info_flag(self) -> None:
        with patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit):
            build_train_parser().parse_args(["--done-on-info-json", "{}"])

        self.assertNotIn("--done-on-info-json", build_train_parser().format_help())

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

    def test_wandb_artifact_logging_reports_stall_timing_metrics(self) -> None:
        class FakeArtifact:
            def __init__(self, name: str, type: str, metadata: dict[str, object]) -> None:
                self.name = name
                self.type = type
                self.metadata = metadata
                self.references: list[tuple[str, str]] = []
                self.files: list[tuple[str, str]] = []

            def add_reference(self, uri: str, name: str) -> None:
                self.references.append((uri, name))

            def add_file(self, path: str, name: str) -> None:
                self.files.append((path, name))

        class FakeRun:
            id = "run-id"
            path = ("entity", "project", "runs", "run-id")

            def __init__(self) -> None:
                self.artifact_logs: list[tuple[FakeArtifact, list[str] | None]] = []
                self.metric_logs: list[tuple[dict[str, object], int | None]] = []

            def log_artifact(
                self, artifact: FakeArtifact, aliases: list[str] | None = None
            ) -> None:
                self.artifact_logs.append((artifact, aliases))

            def log(self, payload: dict[str, object], step: int | None = None) -> None:
                self.metric_logs.append((payload, step))

        class FakeWandb:
            def Artifact(self, name: str, type: str, metadata: dict[str, object]) -> FakeArtifact:
                return FakeArtifact(name, type, metadata)

        clock_values = iter([10.0, 10.0, 10.2, 10.2, 11.2, 11.2, 11.5, 11.5])
        uploads: list[tuple[Path, str]] = []
        args = argparse.Namespace(
            game="SuperMarioBros-Nes-v0",
            run_name="candidate/run",
            run_description="description",
            no_wandb_artifacts=False,
            wandb_artifact_storage_uri="s3://bucket/checkpoints",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            fake_run = FakeRun()

            with (
                patch.dict(sys.modules, {"wandb": FakeWandb()}),
                patch(
                    "rlab.artifacts.upload_s3_artifact",
                    side_effect=lambda path, uri: uploads.append((path, uri)),
                ),
            ):
                timing = log_wandb_model_artifact(
                    fake_run,
                    args,
                    EnvConfig(game="SuperMarioBros-Nes-v0"),
                    model_path,
                    kind="checkpoint",
                    aliases=["latest", "step-100"],
                    metric_step=100,
                    local_save_seconds=2.0,
                    stall_started_at=7.5,
                    clock=lambda: next(clock_values),
                )

        self.assertIsNotNone(timing)
        self.assertEqual(
            uploads,
            [
                (
                    model_path,
                    "s3://bucket/checkpoints/SuperMarioBros-Nes-v0/candidate-run/checkpoint/ppo_test_100_steps.zip",
                )
            ],
        )
        artifact, aliases = fake_run.artifact_logs[0]
        self.assertEqual(artifact.references[0][1], "ppo_test_100_steps.zip")
        self.assertEqual(aliases, ["latest", "step-100"])
        payload, step = fake_run.metric_logs[0]
        self.assertEqual(step, 100)
        self.assertEqual(payload[metric_names.GLOBAL_STEP], 100)
        self.assertAlmostEqual(payload[metric_names.TRAIN_ARTIFACT_METADATA_SECONDS], 0.2)
        self.assertAlmostEqual(payload[metric_names.TRAIN_ARTIFACT_STORAGE_UPLOAD_SECONDS], 1.0)
        self.assertAlmostEqual(payload[metric_names.TRAIN_ARTIFACT_WANDB_LOG_SECONDS], 0.3)
        self.assertAlmostEqual(payload[metric_names.TRAIN_ARTIFACT_LOG_SECONDS], 1.5)
        self.assertAlmostEqual(payload[metric_names.TRAIN_ARTIFACT_LOCAL_SAVE_SECONDS], 2.0)
        self.assertAlmostEqual(payload[metric_names.TRAIN_ARTIFACT_STALL_SECONDS], 4.0)

    def test_wandb_final_artifact_records_metric_step(self) -> None:
        class FakeArtifact:
            def __init__(self, name: str, type: str, metadata: dict[str, object]) -> None:
                self.name = name
                self.type = type
                self.metadata = metadata
                self.files: list[tuple[str, str]] = []

            def add_file(self, path: str, name: str) -> None:
                self.files.append((path, name))

        class FakeRun:
            id = "run-id"
            path = ("entity", "project", "runs", "run-id")

            def __init__(self) -> None:
                self.artifact_logs: list[tuple[FakeArtifact, list[str] | None]] = []
                self.metric_logs: list[tuple[dict[str, object], int | None]] = []

            def log_artifact(
                self, artifact: FakeArtifact, aliases: list[str] | None = None
            ) -> None:
                self.artifact_logs.append((artifact, aliases))

            def log(self, payload: dict[str, object], step: int | None = None) -> None:
                self.metric_logs.append((payload, step))

        class FakeWandb:
            def Artifact(self, name: str, type: str, metadata: dict[str, object]) -> FakeArtifact:
                return FakeArtifact(name, type, metadata)

        args = argparse.Namespace(
            game="SuperMarioBros-Nes-v0",
            run_name="candidate-run",
            run_description="description",
            no_wandb_artifacts=False,
            wandb_artifact_storage_uri="",
        )
        clock_values = iter([1.0, 1.0, 1.1, 1.1, 1.3, 1.3])

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "final_model.zip"
            model_path.write_bytes(b"zip")
            fake_run = FakeRun()

            with patch.dict(sys.modules, {"wandb": FakeWandb()}):
                timing = log_wandb_model_artifact(
                    fake_run,
                    args,
                    EnvConfig(game="SuperMarioBros-Nes-v0"),
                    model_path,
                    kind="final",
                    metric_step=2500000,
                    clock=lambda: next(clock_values),
                )

            artifact, _ = fake_run.artifact_logs[0]
            self.assertEqual(artifact.metadata["checkpoint_step"], 2500000)
            self.assertEqual(load_model_metadata(model_path)["checkpoint_step"], 2500000)
            self.assertIsNotNone(timing)
            assert timing is not None
            self.assertEqual(timing.checkpoint_step, 2500000)

    def test_final_artifact_step_falls_back_to_logged_run_global_step(self) -> None:
        class Summary:
            _json_dict = {"global_step": 2500000}

        class Run:
            summary = Summary()

        class Artifact:
            metadata = {"kind": "final", "checkpoint_step": None}
            qualified_name = "entity/project/candidate-final:v0"

            def logged_by(self) -> Run:
                return Run()

        self.assertEqual(model_artifact_checkpoint_step(Artifact()), 2500000)

    def test_final_artifact_eval_dedupes_by_artifact_not_step(self) -> None:
        self.assertFalse(
            already_evaluated_artifact(
                checkpoint_step=2500000,
                artifact_name="entity/project/run-final:v0",
                kind="final",
                evaluated_steps={2500000},
                evaluated_artifacts={"entity/project/run-checkpoint:v4"},
            )
        )
        self.assertTrue(
            already_evaluated_artifact(
                checkpoint_step=2500000,
                artifact_name="entity/project/run-checkpoint:v4",
                kind="checkpoint",
                evaluated_steps={2500000},
                evaluated_artifacts=set(),
            )
        )

    def test_wandb_artifact_logging_can_purge_uploaded_local_files(self) -> None:
        class FakeArtifact:
            def __init__(self, name: str, type: str, metadata: dict[str, object]) -> None:
                self.name = name
                self.type = type
                self.metadata = metadata
                self.files: list[tuple[str, str]] = []

            def add_file(self, path: str, name: str) -> None:
                self.files.append((path, name))

        class FakeLoggedArtifact:
            def __init__(self) -> None:
                self.wait_called = False

            def wait(self) -> None:
                self.wait_called = True

        class FakeRun:
            id = "run-id"
            path = ("entity", "project", "runs", "run-id")

            def __init__(self) -> None:
                self.logged = FakeLoggedArtifact()

            def log_artifact(
                self, artifact: FakeArtifact, aliases: list[str] | None = None
            ) -> FakeLoggedArtifact:
                return self.logged

            def log(self, payload: dict[str, object], step: int | None = None) -> None:
                pass

        class FakeWandb:
            def Artifact(self, name: str, type: str, metadata: dict[str, object]) -> FakeArtifact:
                return FakeArtifact(name, type, metadata)

        args = argparse.Namespace(
            game="TestGame-Platform",
            run_name="candidate-run",
            run_description="description",
            no_wandb_artifacts=False,
            wandb_artifact_storage_uri="",
            wandb_mode="online",
        )
        clock_values = iter([1.0, 1.0, 1.1, 1.1, 1.4, 1.4])

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            fake_run = FakeRun()

            with patch.dict(sys.modules, {"wandb": FakeWandb()}):
                log_wandb_model_artifact(
                    fake_run,
                    args,
                    EnvConfig(game="TestGame-Platform"),
                    model_path,
                    kind="checkpoint",
                    purge_after_upload=True,
                    clock=lambda: next(clock_values),
                )

            self.assertTrue(fake_run.logged.wait_called)
            self.assertFalse(model_path.exists())
            self.assertFalse(model_metadata_path(model_path).exists())

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
            self.assertEqual(metadata["environment"]["env_id"], "stable-retro-turbo:SuperMarioBros-Nes-v0")
            self.assertEqual(metadata["environment"]["preprocessing"]["frame_stack"], 4)
            self.assertEqual(
                metadata["training_metadata"]["preprocessing"]["maxpool_last_two"],
                False,
            )
            self.assertEqual(
                metadata["training_metadata"]["preprocessing"]["sticky_action_prob"],
                0.0,
            )
            self.assertEqual(metadata["training_metadata"]["preprocessing"]["obs_copy"], "safe_view")
            self.assertNotIn("frame_maxpool", metadata["training_metadata"]["preprocessing"])
            self.assertNotIn("action_sticky_prob", metadata["training_metadata"]["preprocessing"])
            self.assertNotIn("copy_observations", metadata["training_metadata"]["preprocessing"])
            self.assertIn("environment_hash", metadata)
            self.assertIn("training_metadata", metadata)
            self.assertIn("training_metadata_hash", metadata)
            self.assertEqual(
                metadata["training_metadata"]["environment_hash"],
                metadata["environment_hash"],
            )
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
                    info_events={"level_change": (("levelHi", "levelLo"), "change")},
                    done_on_events=("level_change",),
                ),
                kind="checkpoint",
            )

            argv = ["--model", str(model_path)]
            args = parser.parse_args(argv)
            explicit_dests = explicit_arg_dests(parser, argv)

            self.assertTrue(
                apply_model_config_defaults(args, model_path, parser_defaults, explicit_dests)
            )
            self.assertEqual(args.state, "Level2-1")
            self.assertFalse(args.max_pool_frames)
            self.assertEqual(args.observation_size, 96)
            self.assertTrue(args.score_progress_clipped)
            restored_args_config = env_config_from_args(
                args,
                max_episode_steps_attr="max_steps",
                include_states=True,
            )
            self.assertEqual(
                restored_args_config.info_events,
                {"level_change": (("levelHi", "levelLo"), "change")},
            )
            self.assertEqual(restored_args_config.done_on_events, ("level_change",))

            config = env_config_from_model_metadata(
                model_path, fallback=EnvConfig(state="fallback")
            )
            self.assertIsNotNone(config)
            assert config is not None
            self.assertEqual(config.state, "Level2-1")
            self.assertEqual(config.observation_size, 96)
            self.assertEqual(
                config.info_events,
                {"level_change": (("levelHi", "levelLo"), "change")},
            )
            self.assertEqual(config.done_on_events, ("level_change",))

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
            "task_conditioning_start episode=1 step=0 task=(0, 0) index=0 one_hot=[1, 0, 0]",
        )

    def test_playback_metadata_ignores_legacy_done_on_info(self) -> None:
        parser = build_play_parser()
        parser_defaults = vars(parser.parse_args([]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            model_metadata_path(model_path).write_text(
                json.dumps(
                    {
                        "env_config": {
                            "game": "SuperMarioBros-Nes-v0",
                            "state": "Level1-1",
                            "done_on_info": {
                                "life_loss": ["lives", "decrease"],
                                "level_change": [["levelHi", "levelLo"], "change"],
                            },
                            "done_on_events": ["life_loss", "level_change"],
                        },
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            args = parser.parse_args(["--model", str(model_path)])
            explicit_dests = explicit_arg_dests(parser, [])

            self.assertTrue(
                apply_model_config_defaults(args, model_path, parser_defaults, explicit_dests)
            )
            config = env_config_from_args(
                args,
                max_episode_steps_attr="max_steps",
                include_states=True,
            )
            self.assertEqual(config.state, "Level1-1")
            self.assertEqual(config.info_events, {})
            self.assertEqual(config.done_on_events, ())

    def test_artifact_run_config_ignores_legacy_done_on_info(self) -> None:
        parser = build_play_parser()
        parser_defaults = vars(parser.parse_args([]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            source = ResolvedModelSource(
                model_path=model_path,
                artifact_ref="entity/project/run-checkpoint:v0",
            )
            args = parser.parse_args(["--model", str(model_path)])
            legacy_run_config = {
                "game": "SuperMarioBros-Nes-v0",
                "state": "Level1-1",
                "done_on_info": {
                    "life_loss": ["lives", "decrease"],
                    "level_change": [["levelHi", "levelLo"], "change"],
                },
                "done_on_events": ["life_loss", "level_change"],
            }

            with (
                patch("rlab.model_sources.artifact_run_config", return_value=legacy_run_config),
                patch("sys.stdout", new=io.StringIO()),
            ):
                self.assertTrue(
                    apply_model_source_defaults(
                        args,
                        source,
                        parser,
                        parser_defaults,
                        set(),
                        infer_artifact_config=True,
                    )
                )

            config = env_config_from_args(
                args,
                max_episode_steps_attr="max_steps",
                include_states=True,
            )
            self.assertEqual(config.state, "Level1-1")
            self.assertEqual(config.info_events, {})
            self.assertEqual(config.done_on_events, ())

    def test_gui_playback_defaults_to_stochastic(self) -> None:
        parser = build_play_parser()

        self.assertFalse(parser.parse_args([]).deterministic)
        self.assertTrue(parser.parse_args(["--deterministic"]).deterministic)
        self.assertEqual(parser.parse_args([]).episodes, 0)
        self.assertEqual(parser.parse_args(["--episodes", "3"]).episodes, 3)
        self.assertEqual(parser.parse_args([]).seed, DEFAULT_EVAL_SEED)
        self.assertEqual(parser.parse_args(["--seed", "7"]).seed, 7)
        help_text = parser.format_help()
        self.assertIn("--deterministic", help_text)
        self.assertIn("--episodes", help_text)
        self.assertNotIn("--stochastic", help_text)
        self.assertNotIn("--no-stochastic", help_text)

    def test_eval_defaults_to_stochastic(self) -> None:
        with patch("rlab.eval.os.cpu_count", return_value=12):
            parser = build_eval_parser()

        self.assertFalse(parser.parse_args([]).deterministic)
        self.assertTrue(parser.parse_args(["--deterministic"]).deterministic)
        self.assertEqual(parser.parse_args([]).n_envs, 12)
        self.assertEqual(parser.parse_args([]).seed, DEFAULT_EVAL_SEED)
        self.assertEqual(parser.parse_args(["--n-envs", "5"]).n_envs, 5)
        help_text = parser.format_help()
        self.assertIn("--deterministic", help_text)
        self.assertIn("--n-envs", help_text)
        self.assertNotIn("--stochastic", help_text)
        self.assertNotIn("--no-stochastic", help_text)

    def test_eval_record_best_video_is_disabled_before_work(self) -> None:
        with patch.object(sys, "argv", ["rlab-eval", "--record-best-video"]):
            with self.assertRaisesRegex(SystemExit, "--record-best-video is temporarily disabled"):
                eval_main()

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

    def test_model_source_ref_uses_positional_artifact_ref(self) -> None:
        parser = build_play_parser()
        args = parser.parse_args(["tsilva/SuperMarioBros-NES/run-checkpoint:latest"])

        self.assertEqual(
            single_model_artifact_ref(args),
            "tsilva/SuperMarioBros-NES/run-checkpoint:latest",
        )

    def test_model_source_ref_uses_positional_run_name(self) -> None:
        parser = build_play_parser()
        args = parser.parse_args(["run"])

        self.assertEqual(
            single_model_artifact_ref(args),
            "tsilva/SuperMarioBros-NES/run-checkpoint:latest",
        )

    def test_model_source_ref_uses_positional_run_name_kind_and_version(self) -> None:
        parser = build_play_parser()
        args = parser.parse_args(["run", "--artifact-kind", "best", "--artifact-version", "v8"])

        self.assertEqual(
            single_model_artifact_ref(args),
            "tsilva/SuperMarioBros-NES/run-best:v8",
        )

    def test_model_source_ref_uses_explicit_run_kind_and_version(self) -> None:
        parser = build_play_parser()
        argv = ["--artifact-run", "run", "--artifact-kind", "best", "--artifact-version", "v8"]
        args = parser.parse_args(argv)

        self.assertEqual(
            single_model_artifact_ref(args),
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
            "eval/done/level_change/from_rate/mean": 0.90,
            "max_x_max": 3200,
            "reward_mean": 1200.0,
        }

        self.assertEqual(eval_checkpoint_score(metrics), (0.8, 0.9, 1200.0))

    def test_artifact_eval_wandb_log_does_not_force_retroactive_history_step(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.payload: dict[str, object] | None = None
                self.kwargs: dict[str, object] | None = None

            def log(self, payload: dict[str, object], **kwargs: object) -> None:
                self.payload = payload
                self.kwargs = kwargs

        metrics = {
            "reward_mean": 1.0,
            "reward_std": 0.0,
            "reward_max": 2.0,
            "max_x_mean": 3.0,
            "max_x_max": 4,
            "max_level_x_mean": 5.0,
            "max_level_x_max": 6,
            "death_count": 0,
            "death_rate": 0.0,
            "best_episode": {"reward": 2.0, "max_x_pos": 4},
            "checkpoint_step": 4100000,
            "checkpoint_artifact": "entity/project/run-checkpoint:step-4100000",
            "hud_crop_top": 32,
            "episode_results": [],
            "eval/done/all": 10,
            "eval/done/level_change": 9,
            "eval/done/level_change/rate": 0.9,
            "eval/done/level_change/from_rate/min": 0.9,
            "eval/info/level_complete/rate/min/last": 0.9,
            "eval/info/level_complete/rate/mean/last": 0.9,
        }
        run = FakeRun()

        with patch.dict(sys.modules, {"wandb": object()}):
            log_wandb_eval(run, metrics, video_path=None)

        assert run.payload is not None
        self.assertEqual(run.kwargs, {})
        self.assertEqual(run.payload["global_step"], 4100000)
        self.assertEqual(run.payload["eval/checkpoint/step"], 4100000)
        self.assertEqual(run.payload["eval/done/all"], 10)
        self.assertEqual(run.payload["eval/info/level_complete/rate/min/last"], 0.9)

    def test_post_train_checkpoint_eval_logs_checkpoint_step_as_metric(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.payload: dict[str, object] | None = None
                self.kwargs: dict[str, object] | None = None
                self.summary: dict[str, object] = {}

            def log(self, payload: dict[str, object], **kwargs: object) -> None:
                self.payload = payload
                self.kwargs = kwargs

        run = FakeRun()
        metrics = {
            "checkpoint_step": 120000,
            "checkpoint_artifact": "entity/project/run-checkpoint:step-120000",
            "reward_mean": 10.0,
            "reward_std": 1.0,
            "reward_max": 12.0,
            "max_x_mean": 300.0,
            "max_x_max": 400.0,
            "max_level_x_mean": 300.0,
            "max_level_x_max": 400.0,
            "death_count": 1,
            "death_rate": 0.1,
            "best_episode": {"reward": 12.0, "max_x_pos": 400.0},
            "eval/done/all": 10,
            "eval/done/level_change": 9,
            "eval/done/level_change/rate": 0.9,
            "eval/done/level_change/from_rate/min": 0.8,
            "eval/done/level_change/from_rate/mean": 0.9,
            "eval/reward/mean": 10.0,
        }

        log_checkpoint_eval_metrics(
            wandb_run=run,
            args=argparse.Namespace(hud_crop_top=32),
            metrics=metrics,
            checkpoint_path=Path("/tmp/model_120000_steps.zip"),
            checkpoint_step_value=120000,
            artifact_ref="entity/project/run-checkpoint:step-120000",
        )

        assert run.payload is not None
        self.assertEqual(run.kwargs, {})
        self.assertEqual(run.payload["global_step"], 120000)
        self.assertEqual(run.payload["eval/checkpoint/step"], 120000)
        self.assertEqual(run.summary["leader/checkpoint/eval_source"], "post_train_inline")
        self.assertEqual(run.summary["leader/checkpoint/completion_rate"], 0.8)
        self.assertEqual(run.summary["leader/checkpoint/completion_rate_mean"], 0.9)
        self.assertEqual(run.summary["leader/checkpoint/reward_mean"], 10.0)

    def test_checkpoint_eval_preserves_artifact_states_in_eval_config(self) -> None:
        parser = build_eval_parser()
        args = parser.parse_args(
            [
                "--game",
                "SuperMarioBros-Nes-v0",
                "--states",
                "Level1-1,Level1-2",
                "--state-probs",
                "0.5,0.5",
                "--task-conditioning",
                "--task-conditioning-info-vars",
                "levelHi,levelLo",
                "--episodes",
                "10",
                "--n-envs",
                "2",
            ]
        )
        args.eval_dir = "runs/local_evals"
        args.eval_run_name = "unit"
        args.record_best_video = False
        model_path = Path("model.zip")

        with (
            patch("rlab.eval.PPO.load", return_value=object()) as load_model,
            patch("rlab.eval.evaluate_model_episodes") as evaluate,
        ):
            evaluate.return_value = (
                {
                    "checkpoint_step": 4100000,
                    "episode_results": [],
                    "best_episode": {"reward": 0.0, "max_x_pos": 0},
                },
                None,
            )
            evaluate_checkpoint(args, model_path, 4100000, "artifact")

        load_model.assert_called_once()
        config = evaluate.call_args.kwargs["config"]
        self.assertEqual(config.states, ("Level1-1", "Level1-2"))
        self.assertEqual(config.state_probs, (0.5, 0.5))
        self.assertTrue(config.task_conditioning)
        self.assertEqual(config.task_conditioning_info_vars, ("levelHi", "levelLo"))

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
            )
        )
        self.assertFalse(
            is_level_complete(
                {"level_complete": False, "level_changed": True},
            )
        )
        self.assertTrue(
            is_level_complete(
                {"level_complete": True, "level_changed": True},
            )
        )

    def test_level_complete_fallback_rejects_death_level_change(self) -> None:
        self.assertFalse(
            is_level_complete(
                {"level_changed": True, "died": True},
            )
        )
        self.assertTrue(
            is_level_complete(
                {"level_changed": True, "died": False},
            )
        )

    def test_run_eval_episode_does_not_stop_on_completion(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.array([0], dtype=np.int64), None

        class FakeEnv:
            def __init__(self) -> None:
                self.step_count = 0

            def seed(self, seed: int) -> None:
                self.seed_value = seed

            def reset(self):
                self.step_count = 0
                return np.zeros((1, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                obs = np.zeros((1, 4, 84, 84), dtype=np.uint8)
                if self.step_count == 1:
                    return (
                        obs,
                        np.array([1.0], dtype=np.float32),
                        np.array([False]),
                        [
                            {
                                "start_state": "Level1-1",
                                "state": "Level1-1",
                                "max_x_pos": 100,
                                "level_max_x_pos": 100,
                                "level_changed": True,
                                "score": 10,
                                "lives": 3,
                                "time": 300,
                            }
                        ],
                    )
                return (
                    obs,
                    np.array([2.0], dtype=np.float32),
                    np.array([False]),
                    [
                        {
                            "state": "Level1-2",
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                            "score": 20,
                            "lives": 3,
                            "time": 299,
                        }
                    ],
                )

        result = run_eval_episode(
            FakeEnv(),
            FakeModel(),
            max_steps=2,
            deterministic=True,
            seed=7,
            default_start_state="Level1-1",
        )

        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["reward"], 3.0)
        self.assertEqual(result["max_x_pos"], 250)
        self.assertEqual(result["start_state"], "Level1-1")
        self.assertTrue(result["level_complete"])
        self.assertFalse(result["terminated"])
        self.assertTrue(result["truncated"])

    def test_checkpoint_eval_seed_defaults_to_paired_schedule(self) -> None:
        args = argparse.Namespace(seed=10007)

        self.assertEqual(eval_seed_for_checkpoint(args), 10007)

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

        config = EnvConfig(game="SuperMarioBros-Nes-v0")
        with patch("rlab.eval_runner.make_eval_vec_env", return_value=FakeVecEnv()):
            metrics, video_path = evaluate_model_episodes(
                model=FakeModel(),
                config=config,
                episodes=2,
                seed=7,
                max_steps=10,
                deterministic=True,
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
        self.assertEqual(metrics["eval/info/level_complete/rate/min/last"], 0.0)
        self.assertEqual(metrics["eval/info/level_complete/rate/mean/last"], 0.5)
        self.assertEqual(metrics["episode_results"][0]["env_index"], 1)
        self.assertEqual(metrics["episode_results"][0]["start_state"], "Level1-2")
        self.assertEqual(metrics["episode_results"][0]["reward"], 2.0)
        self.assertEqual(metrics["episode_results"][1]["env_index"], 0)
        self.assertEqual(metrics["episode_results"][1]["start_state"], "Level1-1")
        self.assertEqual(metrics["episode_results"][1]["reward"], 4.0)

    def test_vector_eval_does_not_stop_on_completion(self) -> None:
        class FakeModel:
            def predict(self, obs, deterministic):
                return np.zeros(obs.shape[0], dtype=np.int64), None

        class FakeVecEnv:
            num_envs = 2

            def __init__(self) -> None:
                self.step_count = 0

            def reset(self):
                self.step_count = 0
                return np.zeros((2, 4, 84, 84), dtype=np.uint8)

            def step(self, action):
                self.step_count += 1
                obs = np.zeros((2, 4, 84, 84), dtype=np.uint8)
                if self.step_count == 1:
                    return (
                        obs,
                        np.array([1.0, 10.0], dtype=np.float32),
                        np.array([False, False]),
                        [
                            {
                                "start_state": "Level1-1",
                                "state": "Level1-1",
                                "max_x_pos": 100,
                                "level_max_x_pos": 100,
                                "level_changed": True,
                            },
                            {
                                "start_state": "Level1-2",
                                "state": "Level1-2",
                                "max_x_pos": 110,
                                "level_max_x_pos": 110,
                                "level_changed": True,
                            },
                        ],
                    )
                return (
                    obs,
                    np.array([2.0, 20.0], dtype=np.float32),
                    np.array([False, False]),
                    [
                        {
                            "state": "Level1-2",
                            "max_x_pos": 250,
                            "level_max_x_pos": 150,
                        },
                        {
                            "state": "Level1-3",
                            "max_x_pos": 260,
                            "level_max_x_pos": 160,
                        },
                    ],
                )

            def close(self) -> None:
                pass

        fake_env = FakeVecEnv()
        config = EnvConfig(game="SuperMarioBros-Nes-v0")
        with patch("rlab.eval_runner.make_eval_vec_env", return_value=fake_env):
            metrics, video_path = evaluate_model_episodes(
                model=FakeModel(),
                config=config,
                episodes=1,
                seed=7,
                max_steps=2,
                deterministic=True,
                n_envs=2,
            )

        self.assertIsNone(video_path)
        self.assertEqual(fake_env.step_count, 2)
        self.assertEqual(metrics["episodes"], 1)
        self.assertEqual(metrics["completion_count"], 1)
        self.assertEqual(metrics["terminated_count"], 0)
        self.assertEqual(metrics["truncated_count"], 1)
        self.assertEqual(metrics["eval/done/level_change"], 1)
        self.assertEqual(metrics["eval/done/max_steps"], 1)
        self.assertEqual(metrics["episode_results"][0]["steps"], 2)
        self.assertEqual(metrics["episode_results"][0]["reward"], 3.0)
        self.assertEqual(metrics["episode_results"][0]["max_x_pos"], 250)
        self.assertEqual(metrics["episode_results"][0]["start_state"], "Level1-1")
        self.assertTrue(metrics["episode_results"][0]["level_complete"])
        self.assertFalse(metrics["episode_results"][0]["terminated"])
        self.assertTrue(metrics["episode_results"][0]["truncated"])

    def test_evaluate_model_episodes_updates_progress_bar(self) -> None:
        class FakeEnv:
            def close(self) -> None:
                pass

        class FakeProgressBar:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.updates: list[int] = []

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                pass

            def update(self, count: int) -> None:
                self.updates.append(count)

        progress_bars: list[FakeProgressBar] = []

        def fake_tqdm(**kwargs) -> FakeProgressBar:
            progress_bar = FakeProgressBar(**kwargs)
            progress_bars.append(progress_bar)
            return progress_bar

        def fake_run_eval_episode(*args, **kwargs) -> dict:
            return {
                "actions": [],
                "start_state": "Level1-1",
                "reward": 1.0,
                "max_x_pos": 10,
                "max_level_x_pos": 10,
                "score": 0,
                "lives": 3,
                "time": 399,
                "steps": 1,
                "terminated": True,
                "truncated": False,
                "level_complete": True,
                "died": False,
                "death_x_pos": None,
                "final_info": {"start_state": "Level1-1"},
            }

        with (
            patch("rlab.eval_runner.make_eval_vec_env", return_value=FakeEnv()),
            patch("rlab.eval_runner.run_eval_episode", side_effect=fake_run_eval_episode),
            patch("rlab.eval_runner.tqdm", side_effect=fake_tqdm),
        ):
            metrics, video_path = evaluate_model_episodes(
                model=object(),
                config=EnvConfig(game="SuperMarioBros-Nes-v0"),
                episodes=3,
                seed=7,
                max_steps=10,
                deterministic=True,
                progress=True,
                progress_description="eval checkpoint 4100000",
            )

        self.assertIsNone(video_path)
        self.assertEqual(metrics["episodes"], 3)
        self.assertEqual(len(progress_bars), 1)
        self.assertEqual(progress_bars[0].kwargs["total"], 3)
        self.assertEqual(progress_bars[0].kwargs["desc"], "eval checkpoint 4100000")
        self.assertEqual(progress_bars[0].kwargs["disable"], False)
        self.assertEqual(progress_bars[0].updates, [1, 1, 1])


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
            "dones": [True, True, True, True, False],
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
        self.assertFalse(any(key.startswith("train/info/") for key in model.logger.records))

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


class LevelCompleteInfoCallbackTests(unittest.TestCase):
    class FakeLogger:
        def __init__(self) -> None:
            self.records: dict[str, int | float] = {}

        def record(self, key: str, value: int | float) -> None:
            self.records[key] = value

    class FakeModel:
        def __init__(self) -> None:
            self.logger = LevelCompleteInfoCallbackTests.FakeLogger()

    def make_callback(self) -> tuple[LevelCompleteInfoCallback, FakeModel]:
        model = self.FakeModel()
        callback = LevelCompleteInfoCallback(
            info_events={"level_change": (("levelHi", "levelLo"), "change")},
        )
        callback.model = model  # type: ignore[assignment]
        return callback, model

    def assert_no_generic_info_metrics(self, records: dict[str, int | float]) -> None:
        self.assertFalse(any(key.startswith(("train/event/", "train/outcome/")) for key in records))

    def test_ignores_raw_level_change_without_completion(self) -> None:
        callback, model = self.make_callback()

        for step, source in enumerate(((0, 0), (0, 1)), start=1):
            callback.num_timesteps = step
            callback.locals = {
                "dones": [False],
                "infos": [
                    {
                        "levelHi": source[0],
                        "levelLo": source[1] + 1,
                        "info_events": {
                            "level_change": {
                                "op": "change",
                                "keys": ("levelHi", "levelLo"),
                                "prev": source,
                                "next": (source[0], source[1] + 1),
                            },
                        },
                    },
                ],
            }
            self.assertTrue(callback._on_step())

        self.assertEqual(model.logger.records, {})

    def test_records_level_complete_count_from_completion_event(self) -> None:
        callback, model = self.make_callback()
        callback.num_timesteps = 1
        callback.locals = {
            "dones": [False],
            "infos": [
                {
                    "levelHi": 0,
                    "levelLo": 1,
                    "completion_event": True,
                    "level_complete": True,
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

        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/count"], 1)
        self.assertNotIn("train/info/level_complete/from/0-0/rate", model.logger.records)
        self.assert_no_generic_info_metrics(model.logger.records)

    def test_death_level_change_does_not_count_as_level_complete(self) -> None:
        callback, model = self.make_callback()
        callback.ep_window_size = 1

        callback.num_timesteps = 1
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "levelHi": 0,
                    "levelLo": 1,
                    "died": True,
                    "life_loss": True,
                    "completion_event": False,
                    "level_complete": False,
                    "info_events": {
                        "level_change": {
                            "op": "change",
                            "keys": ("levelHi", "levelLo"),
                            "prev": (0, 0),
                            "next": (0, 1),
                        },
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

        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/count"], 0)
        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/rate"], 0.0)
        self.assert_no_generic_info_metrics(model.logger.records)

    def test_conflicting_completion_flag_and_life_loss_records_failure(self) -> None:
        callback, model = self.make_callback()
        callback.ep_window_size = 1

        callback.num_timesteps = 1
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "levelHi": 0,
                    "levelLo": 1,
                    "died": True,
                    "life_loss": True,
                    "completion_event": True,
                    "level_complete": True,
                    "info_events": {
                        "level_change": {
                            "op": "change",
                            "keys": ("levelHi", "levelLo"),
                            "prev": (0, 0),
                            "next": (0, 1),
                        },
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

        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/count"], 0)
        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/rate"], 0.0)
        self.assert_no_generic_info_metrics(model.logger.records)

    def test_conflicting_completion_flag_and_native_life_loss_records_failure(self) -> None:
        callback, model = self.make_callback()
        callback.ep_window_size = 1

        callback.num_timesteps = 1
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "levelHi": 0,
                    "levelLo": 1,
                    "completion_event": True,
                    "level_complete": True,
                    "done_on_info": {
                        "level_change": {
                            "op": "change",
                            "keys": ("levelHi", "levelLo"),
                            "prev": (0, 0),
                            "next": (0, 1),
                        },
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

        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/count"], 0)
        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/rate"], 0.0)
        self.assert_no_generic_info_metrics(model.logger.records)

    def test_records_current_source_failure_on_life_loss(self) -> None:
        callback, model = self.make_callback()
        callback.ep_window_size = 1
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
            model.logger.records["train/info/level_complete/from/0-1/count"],
            0,
        )
        self.assertEqual(
            model.logger.records["train/info/level_complete/from/0-1/rate"],
            0.0,
        )
        self.assert_no_generic_info_metrics(model.logger.records)

    def test_level_complete_rate_uses_rolling_attempt_window(self) -> None:
        callback, model = self.make_callback()
        callback.ep_window_size = 4

        completions = (True, False, True, False)
        for step, completed in enumerate(completions, start=1):
            info_events = {}
            info = {
                "levelHi": 0,
                "levelLo": 1 if completed else 0,
                "reset_info": {"levelHi": 0, "levelLo": 0},
            }
            if completed:
                info["completion_event"] = True
                info["level_complete"] = True
                info_events["level_change"] = {
                    "op": "change",
                    "keys": ("levelHi", "levelLo"),
                    "prev": (0, 0),
                    "next": (0, 1),
                }
            else:
                info["died"] = True
                info["life_loss"] = True
                info_events["life_loss"] = {
                    "op": "decrease",
                    "keys": ("lives",),
                    "prev": 3,
                    "next": 2,
                }
            info["info_events"] = info_events
            callback.num_timesteps = step
            callback.locals = {
                "dones": [True],
                "infos": [info],
            }
            self.assertTrue(callback._on_step())

        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/count"], 2)
        self.assertEqual(
            model.logger.records["train/info/level_complete/from/0-0/rate"],
            0.5,
        )
        self.assertEqual(model.logger.records["train/info/level_complete/rate/min/last"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/mean/last"], 0.5)
        self.assert_no_generic_info_metrics(model.logger.records)

    def test_rate_min_and_mean_last_use_latest_available_source_rates(self) -> None:
        callback, model = self.make_callback()
        callback.ep_window_size = 2

        def record_attempt(step: int, source: tuple[int, int], completed: bool) -> None:
            info_events: dict[str, object] = {}
            info = {
                "levelHi": source[0],
                "levelLo": source[1],
                "reset_info": {"levelHi": source[0], "levelLo": source[1]},
            }
            if completed:
                info["completion_event"] = True
                info["level_complete"] = True
                info["levelLo"] = source[1] + 1
                info_events["level_change"] = {
                    "op": "change",
                    "keys": ("levelHi", "levelLo"),
                    "prev": source,
                    "next": (source[0], source[1] + 1),
                }
            else:
                info["died"] = True
                info["life_loss"] = True
                info_events["life_loss"] = {
                    "op": "decrease",
                    "keys": ("lives",),
                    "prev": 3,
                    "next": 2,
                }
            info["info_events"] = info_events
            callback.num_timesteps = step
            callback.locals = {
                "dones": [True],
                "infos": [info],
            }
            self.assertTrue(callback._on_step())

        record_attempt(1, (0, 0), True)
        self.assertNotIn("train/info/level_complete/rate/min/last", model.logger.records)
        self.assertNotIn("train/info/level_complete/rate/mean/last", model.logger.records)

        record_attempt(2, (0, 0), True)
        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/rate"], 1.0)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/min/last"], 1.0)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/mean/last"], 1.0)

        record_attempt(3, (0, 0), False)
        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/rate"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/min/last"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/mean/last"], 0.5)

        record_attempt(4, (0, 1), True)
        self.assertEqual(model.logger.records["train/info/level_complete/from/0-0/rate"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/min/last"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/mean/last"], 0.5)

        record_attempt(5, (0, 1), False)
        self.assertEqual(model.logger.records["train/info/level_complete/from/0-1/rate"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/min/last"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/mean/last"], 0.5)

        record_attempt(6, (0, 1), True)
        self.assertEqual(model.logger.records["train/info/level_complete/from/0-1/rate"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/min/last"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/mean/last"], 0.5)

        record_attempt(7, (0, 1), True)
        self.assertEqual(model.logger.records["train/info/level_complete/from/0-1/rate"], 1.0)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/min/last"], 0.5)
        self.assertEqual(model.logger.records["train/info/level_complete/rate/mean/last"], 0.75)
        self.assert_no_generic_info_metrics(model.logger.records)


class MetricThresholdStopCallbackTests(unittest.TestCase):
    class FakeLogger:
        def __init__(self) -> None:
            self.records: dict[str, int | float] = {}

    class FakeModel:
        def __init__(self) -> None:
            self.logger = MetricThresholdStopCallbackTests.FakeLogger()

    def make_callback(self, marker_path: Path) -> tuple[MetricThresholdStopCallback, FakeModel]:
        model = self.FakeModel()
        callback = MetricThresholdStopCallback(
            metric_name=TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
            threshold=0.99,
            operator=">",
            marker_path=marker_path,
        )
        callback.model = model  # type: ignore[assignment]
        return callback, model

    def test_waits_until_metric_crosses_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker_path = Path(tmp) / "run" / "early_stop.txt"
            callback, model = self.make_callback(marker_path)
            callback.num_timesteps = 100

            self.assertTrue(callback._on_step())
            self.assertFalse(marker_path.exists())

            model.logger.records[TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST] = 0.99
            self.assertTrue(callback._on_step())
            self.assertFalse(marker_path.exists())

            model.logger.records[TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST] = 1.0
            callback.num_timesteps = 200
            self.assertFalse(callback._on_step())

            marker = marker_path.read_text(encoding="utf-8")
            self.assertIn("early_stop=metric_threshold", marker)
            self.assertIn(f"early_stop_metric={TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST}", marker)
            self.assertIn("early_stop_operator=>", marker)
            self.assertIn("early_stop_threshold=0.99", marker)
            self.assertIn("early_stop_value=1", marker)
            self.assertIn("timesteps=200", marker)

    def test_structured_detector_requires_all_metric_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker_path = Path(tmp) / "run" / "early_stop.txt"
            model = self.FakeModel()
            callback = MetricThresholdStopCallback(
                detector=[
                    {
                        "metric": TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                        "operator": ">",
                        "threshold": 0.99,
                    },
                    {
                        "metric": "rollout/ep_rew_mean",
                        "operator": ">=",
                        "threshold": 1000,
                    },
                ],
                marker_path=marker_path,
            )
            callback.model = model  # type: ignore[assignment]
            callback.num_timesteps = 100

            model.logger.records[TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST] = 1.0
            model.logger.records["rollout/ep_rew_mean"] = 999
            self.assertTrue(callback._on_step())
            self.assertFalse(marker_path.exists())

            model.logger.records["rollout/ep_rew_mean"] = 1000
            callback.num_timesteps = 200
            self.assertFalse(callback._on_step())

            marker = marker_path.read_text(encoding="utf-8")
            self.assertIn("early_stop_detector_json=", marker)
            self.assertIn("early_stop_value/rollout/ep_rew_mean=1000", marker)


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


class TimeElapsedCallbackTests(unittest.TestCase):
    def test_logs_elapsed_time_to_logger_and_wandb(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: list[tuple[str, float]] = []

            def record(self, key: str, value: float) -> None:
                self.records.append((key, value))

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()

        class FakeRun:
            def __init__(self) -> None:
                self.payloads: list[tuple[dict[str, object], int]] = []

            def log(self, payload: dict[str, object], *, step: int) -> None:
                self.payloads.append((payload, step))

        times = iter([10.0, 25.0])
        model = Model()
        run = FakeRun()
        callback = TimeElapsedCallback(wandb_run=run, clock=lambda: next(times))
        callback.model = model  # type: ignore[assignment]
        callback.num_timesteps = 8192

        callback._on_training_start()
        callback._on_rollout_end()

        self.assertEqual(model.logger.records, [(metric_names.TIME_TIME_ELAPSED, 15.0)])
        self.assertEqual(
            run.payloads,
            [({metric_names.GLOBAL_STEP: 8192, metric_names.TIME_TIME_ELAPSED: 15.0}, 8192)],
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
        self.assertAlmostEqual(
            records["rollout/value_pred/std"], float(np.std([1.0, 2.0, 3.0, 4.0]))
        )
        self.assertEqual(records["rollout/value_pred/min"], 1.0)
        self.assertEqual(records["rollout/value_pred/max"], 4.0)
        self.assertEqual(records["rollout/value_pred/abs_mean"], 2.5)
        self.assertEqual(records["rollout/advantage/mean"], 0.5)
        self.assertAlmostEqual(
            records["rollout/advantage/std"], float(np.std([-1.0, 0.0, 1.0, 2.0]))
        )
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
