from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager, redirect_stderr
from io import StringIO
from unittest.mock import patch
from pathlib import Path

import gymnasium as gym
import numpy as np

from rlab.artifacts import (
    env_config_from_config_dict,
    init_wandb,
)
from rlab.cli import build_parser as build_train_parser
from rlab.cli import build_train_command
from rlab.cli import effective_n_envs
from rlab.cli import parse_train_args
from rlab.env import (
    EnvConfig,
    GymVectorEnvToSb3VecEnv,
    StickyAction,
    VecDiscreteRetroActions,
    VecObservationMask,
    VecRetroProgressInfo,
    VecTaskConditioning,
    make_eval_vec_env,
    make_retro_env,
    make_rendered_replay_env,
    make_training_vec_env,
    make_visual_replay_env,
    make_vec_envs,
    native_vec_env_supports_done_on,
    native_vec_env_supports_rgb_render,
    needs_vec_transpose_image,
    native_obs_crop,
    resolve_env_config,
    resolve_mixed_state_config,
    state_weight_mapping,
    vector_infos_to_list,
)
from rlab.env_config import (
    env_config_from_args,
    parse_info_events,
    parse_obs_crop,
    parse_state_probs,
    parse_states,
)
from rlab.env_vec import provider_native_vec_kwargs
from rlab.env_providers import make_provider_vec_env as make_raw_provider_vec_env
from rlab.envs.super_mario_bros_nes import SuperMarioBrosNesFusedHooks
from rlab.fused_vec import (
    FusedGymVectorPipeline,
    IdentityFusedHooks,
    Sb3FusedVecEnv,
    VectorInfoView,
)
from rlab.metric_names import (
    TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
)
from rlab.play import build_parser as build_play_parser
from rlab.vec_wrappers import normalize_vec_wrapper_specs
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.targets import SuperMarioBros3NesV0Target, SuperMarioBrosNesV0Target, target_for_game
from rlab.checkpoint_eval_worker import (
    eval_checkpoint_artifact_ref,
)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


class EnvConfigTests(unittest.TestCase):
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

    def test_unknown_vec_wrapper_fails_loudly(self) -> None:
        with self.assertRaisesRegex(ValueError, "known vector wrappers"):
            resolve_env_config(
                EnvConfig(
                    game="SuperMarioBros-Nes-v0",
                    vec_wrappers=({"id": "missing_vec_wrapper"},),
                )
            )

    def test_vec_wrapper_specs_normalize_inline_fields(self) -> None:
        self.assertEqual(
            normalize_vec_wrapper_specs(
                [
                    {
                        "id": "sb3_fused",
                        "hooks": "super_mario_bros_nes",
                        "info_mode": "hook_with_terminal_native",
                    }
                ]
            ),
            (
                {
                    "id": "sb3_fused",
                    "hooks": "super_mario_bros_nes",
                    "info_mode": "hook_with_terminal_native",
                },
            ),
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


class EnvConfigFromArgsTests(unittest.TestCase):
    def test_parse_states_trims_empty_values(self) -> None:
        self.assertEqual(parse_states("A, B,C"), ("A", "B", "C"))

    def test_parse_states_accepts_metadata_sequence(self) -> None:
        self.assertEqual(parse_states(["A", " B "]), ("A", "B"))

    def test_parse_states_rejects_empty_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty state"):
            parse_states("A, ,C")

    def test_parse_state_probs_accepts_native_sampling_weights(self) -> None:
        self.assertEqual(parse_state_probs("1, 3"), (1.0, 3.0))
        self.assertEqual(parse_state_probs([0.5, 0]), (0.5, 0.0))
        with self.assertRaisesRegex(ValueError, "non-negative finite"):
            parse_state_probs("1,-1")
        with self.assertRaisesRegex(ValueError, "at least one positive"):
            parse_state_probs("0,0")

    def test_parse_obs_crop_accepts_bottom_crop(self) -> None:
        self.assertEqual(parse_obs_crop("0,0,32,0"), (0, 0, 32, 0))
        self.assertEqual(parse_obs_crop("[0,0,32,0]"), (0, 0, 32, 0))

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

    def test_init_wandb_defaults_project_to_env_id(self) -> None:
        class FakeRun:
            def define_metric(self, *args: object, **kwargs: object) -> None:
                pass

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
            wandb_project=None,
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
                init_wandb(args, tmp_dir, EnvConfig(game="SuperMarioBros3-Nes-v0"))

        self.assertEqual(fake_wandb.init_kwargs["project"], "SuperMarioBros3-Nes-v0")
        self.assertEqual(args.wandb_project, "SuperMarioBros3-Nes-v0")

    def test_eval_checkpoint_artifact_ref_defaults_project_to_env_id(self) -> None:
        args = argparse.Namespace(
            game="SuperMarioBros3-Nes-v0",
            no_wandb_artifacts=False,
            wandb_entity="tsilva",
            wandb_project=None,
            run_name="smb3-run",
        )

        self.assertEqual(
            eval_checkpoint_artifact_ref(args, Path("model.zip"), 500000),
            "tsilva/SuperMarioBros3-Nes-v0/smb3-run-checkpoint:step-500000",
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

    def test_smb3_reward_env_wrapper_resolves_to_mario_reward_config(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                game="SuperMarioBros3-Nes-v0",
                env_wrappers=(
                    {"id": "SuperMarioBros3NesProgressInfoWrapper"},
                    {
                        "id": "SuperMarioBros3NesRewardEnvWrapper",
                        "kwargs": {
                            "use_retro_reward": False,
                            "reward_mode": "score",
                            "progress_reward_scale": 1.0,
                            "death_penalty": 25,
                            "score_progress_clipped": False,
                        },
                    },
                ),
            )
        )

        self.assertEqual(config.reward_mode, "score")
        self.assertFalse(config.use_retro_reward)
        self.assertEqual(config.death_penalty, 25.0)
        self.assertEqual(
            config.env_wrappers,
            (
                {
                    "id": "SuperMarioBros3NesProgressInfoWrapper",
                    "kwargs": {},
                },
                {
                    "id": "SuperMarioBros3NesRewardEnvWrapper",
                    "kwargs": {
                        "use_retro_reward": False,
                        "reward_mode": "score",
                        "progress_reward_scale": 1.0,
                        "death_penalty": 25.0,
                        "score_progress_clipped": False,
                    },
                },
            ),
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
                        "obs_resize_algorithm": "area",
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
        self.assertEqual(args.obs_resize_algorithm, "area")
        self.assertEqual(args.wandb_tags, "from-json,config-file")

    def test_train_config_field_spec_covers_json_command_and_env_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps(
                    {
                        "obs_resize_algorithm": "nearest",
                        "env_args": {"game": "breakout", "frameskip": 4},
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            args = parse_train_args(["--train-config-json", str(path)])
            config = env_config_from_args(args)
            command = build_train_command(
                {
                    "obs_resize_algorithm": "nearest",
                    "env_args": {"game": "breakout", "frameskip": 4},
                }
            )

        self.assertEqual(args.obs_resize_algorithm, "nearest")
        self.assertEqual(args.env_args, {"game": "breakout", "frameskip": 4})
        self.assertEqual(config.obs_resize_algorithm, "nearest")
        self.assertEqual(config.env_args, {"game": "breakout", "frameskip": 4})
        self.assertIn("--obs-resize-algorithm", command)
        self.assertIn("nearest", command)
        self.assertIn("--env-args", command)
        self.assertIn('{"game":"breakout","frameskip":4}', command)

    def test_train_config_json_uses_provider_num_envs_for_seed_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps(
                    {
                        "env_provider": "ale-py",
                        "game": "breakout",
                        "env_args": {"game": "breakout", "num_envs": 16},
                        "seed": 9990,
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "reserved for eval"):
                parse_train_args(["--train-config-json", str(path)])

    def test_checkpoint_eval_n_envs_accepts_new_flag_and_legacy_json_key(self) -> None:
        args = parse_train_args(["--checkpoint-eval-n-envs", "3"])
        self.assertEqual(args.checkpoint_eval_n_envs, 3)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps({"post_train_eval_n_envs": 4}) + "\n",
                encoding="utf-8",
            )

            legacy_args = parse_train_args(["--train-config-json", str(path)])

        self.assertEqual(legacy_args.checkpoint_eval_n_envs, 4)
        self.assertIn("checkpoint_eval_n_envs", legacy_args._train_config_json_fields)
        self.assertNotIn("post_train_eval_n_envs", legacy_args._train_config_json_fields)

    def test_train_config_json_accepts_checkpoint_eval_stages(self) -> None:
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
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps({"checkpoint_eval_stages": stages}) + "\n",
                encoding="utf-8",
            )

            args = parse_train_args(["--train-config-json", str(path)])

        self.assertEqual(args.checkpoint_eval_stages[0]["name"], "screen")
        self.assertEqual(args.checkpoint_eval_stages[0]["episodes"], 10)
        self.assertEqual(args.checkpoint_eval_stages[0]["n_envs"], 2)
        self.assertIs(args.checkpoint_eval_stages[0]["candidate_stop"], False)

    def test_train_config_json_rejects_invalid_checkpoint_eval_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps({"checkpoint_eval_stages": [{"name": "screen", "episodes": 0}]}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "checkpoint-eval-stages.*episodes"):
                parse_train_args(["--train-config-json", str(path)])

    def test_train_config_json_rejects_retired_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(json.dumps({"env_threads": 4}) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown train config field"):
                parse_train_args(["--train-config-json", str(path)])

    def test_explicit_n_envs_must_match_provider_num_envs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps(
                    {
                        "env_provider": "ale-py",
                        "game": "breakout",
                        "env_args": {"game": "breakout", "num_envs": 16},
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError, "env_args.num_envs must match requested n_envs"
            ):
                parse_train_args(["--train-config-json", str(path), "--n-envs", "8"])

    def test_effective_n_envs_prefers_provider_env_args_when_cli_default_is_implicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps(
                    {
                        "env_provider": "ale-py",
                        "game": "breakout",
                        "env_args": {"game": "breakout", "num_envs": 16},
                    },
                )
                + "\n",
                encoding="utf-8",
            )

            args = parse_train_args(["--train-config-json", str(path)])

        self.assertEqual(args.n_envs, 8)
        self.assertEqual(effective_n_envs(args), 16)

    def test_resolve_env_config_accepts_provider_owned_game(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                env_provider="ale-py",
                env_args={"game": "breakout", "num_envs": 16},
                state="",
                action_set="native",
                reward_mode="native",
            )
        )

        self.assertEqual(config.game, "breakout")

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

    def test_train_config_json_accepts_vec_wrappers(self) -> None:
        vec_wrappers = [
            {
                "id": "sb3_fused",
                "hooks": "super_mario_bros_nes",
                "info_mode": "hook_with_terminal_native",
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(
                json.dumps(
                    {
                        "game": "SuperMarioBros-Nes-v0",
                        "vec_wrappers": vec_wrappers,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            args = parse_train_args(["--train-config-json", str(path)])
            config = resolve_env_config(env_config_from_args(args))

        self.assertEqual(args.vec_wrappers, vec_wrappers)
        self.assertEqual(config.vec_wrappers, tuple(vec_wrappers))

    def test_train_config_json_rejects_scalar_early_stop_fields(self) -> None:
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

            with self.assertRaisesRegex(ValueError, "unknown train config field"):
                parse_train_args(["--train-config-json", str(path)])

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

    def test_train_parser_rejects_scalar_early_stop_flags(self) -> None:
        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            parse_train_args(["--early-stop-metric", TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST])

        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            parse_train_args(["--early-stop-threshold", "0.99"])

        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            parse_train_args(["--early-stop-operator", ">"])

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

    def test_build_train_command_includes_obs_crop(self) -> None:
        command = build_train_command({"obs_crop": [0, 0, 32, 0]})

        self.assertIn("--obs-crop", command)
        self.assertIn("0,0,32,0", command)

    def test_build_train_command_includes_obs_resize_algorithm(self) -> None:
        command = build_train_command({"obs_resize_algorithm": "area"})

        self.assertIn("--obs-resize-algorithm", command)
        self.assertIn("area", command)

    def test_training_loop_eval_settings_are_not_train_options(self) -> None:
        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            parse_train_args(["--eval-freq", "0"])

        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            parse_train_args(["--eval-episodes", "0"])

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train_config.json"
            path.write_text(json.dumps({"eval_freq": 1}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown train config field"):
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

    def test_play_parser_rejects_environment_overrides(self) -> None:
        parser = build_play_parser()
        rejected_flags = (
            "--sticky-action-prob",
            "--states",
            "--task-conditioning",
            "--env-provider",
            "--done-on-events",
            "--policy-env",
        )

        for flag in rejected_flags:
            with self.subTest(flag=flag):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args([flag])

    def test_play_parser_exposes_only_metadata_safe_controls(self) -> None:
        help_text = build_play_parser().format_help()

        self.assertIn("--episodes", help_text)
        self.assertIn("--seed", help_text)
        self.assertIn("--fps", help_text)
        self.assertIn("--device", help_text)
        self.assertIn("--deterministic", help_text)
        self.assertIn("--step-over", help_text)
        self.assertIn("--show-obs", help_text)
        self.assertIn("--attribution", help_text)
        self.assertIn("--attribution-interval", help_text)
        self.assertIn("--attribution-opacity", help_text)
        self.assertNotIn("--show-obs-stack", help_text)
        self.assertNotIn("--sticky-action-prob", help_text)
        self.assertNotIn("--task-conditioning", help_text)
        self.assertNotIn("--policy-env", help_text)

    def test_play_parser_rejects_invalid_attribution_controls(self) -> None:
        parser = build_play_parser()
        invalid_args = (
            ["--attribution", "saliency"],
            ["--attribution-interval", "0"],
            ["--attribution-opacity", "-0.1"],
            ["--attribution-opacity", "1.1"],
        )

        for args in invalid_args:
            with self.subTest(args=args):
                with redirect_stderr(StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(args)

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
            resolve_env_config(EnvConfig(env_provider="stable-retro", game="SuperMarioBros-Nes-v0"))

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

    def test_visual_replay_env_avoids_preprocess_wrapper(self) -> None:
        class FakeSpace:
            def seed(self, seed):
                self.seed_value = seed

        class FakeEnv:
            action_space = FakeSpace()
            observation_space = FakeSpace()

        sentinel = FakeEnv()
        config = EnvConfig(game="SuperMarioBros-Nes-v0", action_set="native")

        with (
            patch("rlab.env.make_provider_env", return_value=sentinel) as make_provider_env,
            patch("rlab.env.FrameSkip", side_effect=lambda env, skip, max_pool: env) as frame_skip,
            patch("rlab.env.RetroPreprocess") as retro_preprocess,
        ):
            env = make_visual_replay_env(config=config, seed=7)

        self.assertIs(env, sentinel)
        make_provider_env.assert_called_once()
        self.assertEqual(make_provider_env.call_args.kwargs["render_mode"], "rgb_array")
        frame_skip.assert_called_once()
        retro_preprocess.assert_not_called()

    def test_visual_replay_env_supports_ale_py_provider(self) -> None:
        class FakeSpace:
            def seed(self, seed):
                self.seed_value = seed

        class FakeEnv:
            action_space = FakeSpace()
            observation_space = FakeSpace()

        class FakeAtariVectorEnv:
            call_args = None

            def __init__(self, *args, **kwargs):
                type(self).call_args = (args, kwargs)

        sentinel = FakeEnv()
        config = EnvConfig(
            env_provider="ale-py",
            game="breakout",
            action_set="native",
            frame_skip=4,
            sticky_action_prob=0.25,
            obs_crop=(34, 0, 0, 0),
            obs_crop_mode="mask",
        )

        with (
            patch("rlab.env._ale_py_atari_vector_env_type", return_value=FakeAtariVectorEnv),
            patch("rlab.env.SingleLaneVecEnvAdapter", return_value=sentinel),
            patch("rlab.env.FrameSkip") as frame_skip,
            patch("rlab.env.StickyAction") as sticky_action,
        ):
            env = make_visual_replay_env(config=config, seed=7)

        self.assertIs(env, sentinel)
        self.assertEqual(FakeAtariVectorEnv.call_args[0], ("breakout",))
        kwargs = FakeAtariVectorEnv.call_args[1]
        self.assertEqual(kwargs["num_envs"], 1)
        self.assertEqual(kwargs["frameskip"], 4)
        self.assertEqual(kwargs["repeat_action_probability"], 0.25)
        self.assertTrue(kwargs["use_fire_reset"])
        self.assertFalse(kwargs["grayscale"])
        self.assertEqual(kwargs["stack_num"], 1)
        frame_skip.assert_not_called()
        sticky_action.assert_not_called()

    def test_rendered_replay_preprocess_uses_top_obs_crop(self) -> None:
        class FakeSpace:
            def seed(self, seed):
                self.seed_value = seed

        class FakeEnv:
            action_space = FakeSpace()
            observation_space = FakeSpace()

        sentinel = FakeEnv()
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            obs_crop=(32, 0, 0, 0),
        )

        with (
            patch("rlab.env.make_provider_env", return_value=sentinel),
            patch("rlab.env.FrameSkip", side_effect=lambda env, skip, max_pool: env),
            patch("rlab.env.RetroProgressInfo", side_effect=lambda env, config: env),
            patch(
                "rlab.env.RetroPreprocess", side_effect=lambda env, size, hud_crop_top: env
            ) as preprocess,
            patch(
                "rlab.env.gym.wrappers.TimeLimit", side_effect=lambda env, max_episode_steps: env
            ),
        ):
            env = make_rendered_replay_env(config=config, seed=7)

        self.assertIs(env, sentinel)
        preprocess.assert_called_once_with(sentinel, 84, hud_crop_top=32)

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
            patch(
                "rlab.env.make_provider_vec_env", return_value=fake_native
            ) as make_provider_vec_env,
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
        self.assertNotIn("num_threads", native_kwargs)
        self.assertEqual(native_kwargs["frame_skip"], 4)
        self.assertEqual(native_kwargs["obs_resize"], (84, 84))
        self.assertEqual(native_kwargs["maxpool_last_two"], False)

    def test_provider_factory_forces_same_step_autoreset_when_supported(self) -> None:
        class FakeAtariVectorEnv:
            calls = []

            def __init__(
                self,
                game,
                *,
                num_envs,
                autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
            ):
                self.game = game
                self.num_envs = num_envs
                self.autoreset_mode = autoreset_mode
                self.metadata = {"autoreset_mode": autoreset_mode}
                self.__class__.calls.append((game, num_envs, autoreset_mode))

        config = EnvConfig(env_provider="ale-py", game="breakout")
        env = make_raw_provider_vec_env(
            config,
            native_kwargs={
                "num_envs": 2,
                "autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP,
            },
            ale_py_vec_env_type=lambda: FakeAtariVectorEnv,
        )

        self.assertIs(env.autoreset_mode, gym.vector.AutoresetMode.SAME_STEP)
        self.assertEqual(
            FakeAtariVectorEnv.calls,
            [("breakout", 2, gym.vector.AutoresetMode.SAME_STEP)],
        )

    def test_provider_factory_rejects_non_same_step_autoreset_mode(self) -> None:
        class FakeSuperMarioNative:
            metadata = {"autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP}

            def __init__(self, game, **kwargs):
                self.game = game
                self.kwargs = kwargs

        config = EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
        )

        with self.assertRaisesRegex(RuntimeError, "same-step autoreset mode"):
            make_raw_provider_vec_env(
                config,
                native_kwargs={"num_envs": 2},
                super_mario_vec_env_type=lambda: FakeSuperMarioNative,
            )

    def test_ale_py_native_vec_kwargs_map_env_config(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                env_provider="ale-py",
                game="breakout",
                state="",
                action_set="native",
                reward_mode="native",
                frame_skip=4,
                max_pool_frames=True,
                sticky_action_prob=0.25,
                max_episode_steps=27000,
                observation_size=84,
                clip_rewards=True,
                episodic_life=True,
            )
        )

        native_kwargs = provider_native_vec_kwargs(
            config,
            n_envs=16,
            native_done_on_rules={},
            native_obs_crop=native_obs_crop,
            state_weight_mapping=state_weight_mapping,
        )

        self.assertEqual(native_kwargs["num_envs"], 16)
        self.assertNotIn("num_threads", native_kwargs)
        self.assertEqual(native_kwargs["max_num_frames_per_episode"], 108000)
        self.assertEqual(native_kwargs["repeat_action_probability"], 0.25)
        self.assertEqual(native_kwargs["img_height"], 84)
        self.assertEqual(native_kwargs["img_width"], 84)
        self.assertEqual(native_kwargs["grayscale"], True)
        self.assertEqual(native_kwargs["stack_num"], 4)
        self.assertEqual(native_kwargs["frameskip"], 4)
        self.assertEqual(native_kwargs["maxpool"], True)
        self.assertEqual(native_kwargs["episodic_life"], True)
        self.assertEqual(native_kwargs["reward_clipping"], True)

    def test_ale_py_native_vec_kwargs_prefer_provider_env_args(self) -> None:
        env_args = {
            "game": "breakout",
            "num_envs": 16,
            "num_threads": 4,
            "thread_affinity_offset": -1,
            "max_num_frames_per_episode": 216000,
            "repeat_action_probability": 0.25,
            "img_height": 84,
            "img_width": 84,
            "grayscale": True,
            "stack_num": 4,
            "frameskip": 4,
            "maxpool": True,
            "reward_clipping": True,
        }
        config = resolve_env_config(
            EnvConfig(
                env_provider="ale-py",
                game="breakout",
                state="",
                action_set="native",
                reward_mode="native",
                env_args=env_args,
            )
        )

        native_kwargs = provider_native_vec_kwargs(
            config,
            n_envs=16,
            native_done_on_rules={},
            native_obs_crop=native_obs_crop,
            state_weight_mapping=state_weight_mapping,
        )

        self.assertEqual(
            native_kwargs,
            {key: value for key, value in env_args.items() if key != "game"},
        )

    def test_ale_py_native_vec_kwargs_do_not_synthesize_num_threads(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                env_provider="ale-py",
                game="breakout",
                state="",
                action_set="native",
                reward_mode="native",
                env_args={"game": "breakout", "num_envs": 16},
            )
        )

        native_kwargs = provider_native_vec_kwargs(
            config,
            n_envs=16,
            native_done_on_rules={},
            native_obs_crop=native_obs_crop,
            state_weight_mapping=state_weight_mapping,
        )

        self.assertEqual(native_kwargs["num_envs"], 16)
        self.assertNotIn("num_threads", native_kwargs)

    def test_provider_env_args_reject_conflicting_num_envs(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                env_provider="ale-py",
                game="breakout",
                state="",
                action_set="native",
                reward_mode="native",
                env_args={"game": "breakout", "num_envs": 16},
            )
        )

        with self.assertRaisesRegex(ValueError, "env_args.num_envs must match requested n_envs"):
            provider_native_vec_kwargs(
                config,
                n_envs=8,
                native_done_on_rules={},
                native_obs_crop=native_obs_crop,
                state_weight_mapping=state_weight_mapping,
            )

    def test_ale_py_native_vec_kwargs_keep_native_preprocessing_for_masked_crop(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                env_provider="ale-py",
                game="breakout",
                state="",
                action_set="native",
                reward_mode="native",
                frame_skip=4,
                max_pool_frames=True,
                sticky_action_prob=0.25,
                max_episode_steps=27000,
                observation_size=84,
                obs_crop=(34, 0, 0, 0),
                obs_crop_mode="mask",
                obs_crop_fill=0,
                clip_rewards=True,
            )
        )

        native_kwargs = provider_native_vec_kwargs(
            config,
            n_envs=16,
            native_done_on_rules={},
            native_obs_crop=native_obs_crop,
            state_weight_mapping=state_weight_mapping,
        )

        self.assertEqual(native_kwargs["img_height"], 84)
        self.assertEqual(native_kwargs["img_width"], 84)
        self.assertEqual(native_kwargs["grayscale"], True)
        self.assertEqual(native_kwargs["stack_num"], 4)
        self.assertEqual(native_kwargs["frameskip"], 4)
        self.assertEqual(native_kwargs["maxpool"], True)

    def test_observation_mask_masks_whole_chw_stack_in_scaled_raw_coordinates(self) -> None:
        class FakeVecEnv:
            num_envs = 2
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.Discrete(3)

            def reset(self):
                return np.full((2, 4, 84, 84), 9, dtype=np.uint8)

            def step_async(self, actions) -> None:
                self.actions = actions

            def step_wait(self):
                return (
                    np.full((2, 4, 84, 84), 7, dtype=np.uint8),
                    np.asarray([0.0, 1.0], dtype=np.float32),
                    np.asarray([False, True]),
                    [
                        {},
                        {
                            "terminal_observation": np.full((4, 84, 84), 5, dtype=np.uint8),
                        },
                    ],
                )

        config = EnvConfig(
            env_provider="ale-py",
            game="breakout",
            obs_crop=(34, 0, 0, 0),
            obs_crop_mode="mask",
            obs_crop_fill=0,
        )
        env = VecObservationMask(FakeVecEnv(), config, source_shape=(210, 160))

        obs = env.reset()
        self.assertEqual(obs.shape, (2, 4, 84, 84))
        np.testing.assert_array_equal(obs[:, :, :14, :], 0)
        np.testing.assert_array_equal(obs[:, :, 14:, :], 9)

        env.step_async(np.asarray([0, 1]))
        obs, _rewards, _dones, infos = env.step_wait()

        np.testing.assert_array_equal(obs[:, :, :14, :], 0)
        np.testing.assert_array_equal(obs[:, :, 14:, :], 7)
        terminal_obs = infos[1]["terminal_observation"]
        np.testing.assert_array_equal(terminal_obs[:, :14, :], 0)
        np.testing.assert_array_equal(terminal_obs[:, 14:, :], 5)

    def test_observation_mask_scales_bottom_raw_crop(self) -> None:
        class FakeVecEnv:
            num_envs = 1
            observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(4, 84, 84),
                dtype=np.uint8,
            )
            action_space = gym.spaces.Discrete(3)

            def reset(self):
                return np.full((1, 4, 84, 84), 11, dtype=np.uint8)

        config = EnvConfig(
            env_provider="ale-py",
            game="ms_pacman",
            obs_crop=(0, 0, 37, 0),
            obs_crop_mode="mask",
            obs_crop_fill=0,
        )
        env = VecObservationMask(FakeVecEnv(), config, source_shape=(210, 160))

        obs = env.reset()

        np.testing.assert_array_equal(obs[:, :, :-15, :], 11)
        np.testing.assert_array_equal(obs[:, :, -15:, :], 0)

    def test_ale_py_provider_rejects_state_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not support state"):
            resolve_mixed_state_config(
                EnvConfig(env_provider="ale-py", game="breakout", state="Level1-1"),
                n_envs=2,
            )

    def test_ale_py_identity_fused_native_config_builds_sb3_fused_env(self) -> None:
        class FakeAtariVectorEnv(gym.vector.VectorEnv):
            num_envs = 2

            def __init__(self, game, **kwargs):
                self.game = game
                self.kwargs = kwargs
                self.single_observation_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(4, 84, 84),
                    dtype=np.uint8,
                )
                self.single_action_space = gym.spaces.Discrete(4)
                self.observation_space = self.single_observation_space
                self.action_space = gym.spaces.MultiDiscrete([4, 4])

            def reset(self, *, seed=None, options=None):
                return np.zeros((2, 4, 84, 84), dtype=np.uint8), {}

            def step(self, actions):
                return (
                    np.ones((2, 4, 84, 84), dtype=np.uint8),
                    np.asarray([1.0, 0.0], dtype=np.float32),
                    np.asarray([False, False]),
                    np.asarray([False, False]),
                    {},
                )

        config = EnvConfig(
            env_provider="ale-py",
            game="breakout",
            action_set="native",
            reward_mode="native",
            obs_crop=None,
            vec_wrappers=({"id": "sb3_fused"},),
        )
        with patch("rlab.env._ale_py_atari_vector_env_type", return_value=FakeAtariVectorEnv):
            env = make_training_vec_env(config, n_envs=2, seed=7)

        self.assertIsInstance(env, Sb3FusedVecEnv)

    def test_ale_py_observation_mask_can_follow_requested_fused_env(self) -> None:
        class FakeAtariVectorEnv(gym.vector.VectorEnv):
            num_envs = 2

            def __init__(self, game, **kwargs):
                self.game = game
                self.kwargs = kwargs
                self.single_observation_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(4, 84, 84),
                    dtype=np.uint8,
                )
                self.single_action_space = gym.spaces.Discrete(4)
                self.observation_space = self.single_observation_space
                self.action_space = gym.spaces.MultiDiscrete([4, 4])

        config = EnvConfig(
            env_provider="ale-py",
            game="breakout",
            action_set="native",
            reward_mode="native",
            obs_crop=(34, 0, 0, 0),
            obs_crop_mode="mask",
            vec_wrappers=({"id": "sb3_fused"}, {"id": "observation_mask"}),
        )
        with patch("rlab.env._ale_py_atari_vector_env_type", return_value=FakeAtariVectorEnv):
            env = make_training_vec_env(config, n_envs=2, seed=7)

        self.assertIsInstance(env, VecObservationMask)

    def test_supermariobrosnes_turbo_rgb_render_support_uses_metadata(self) -> None:
        class FakeSuperMarioNative:
            metadata = {"render_modes": ["rgb_array"]}

        config = EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
        )
        with patch(
            "rlab.env._super_mario_bros_nes_turbo_vec_env_type",
            return_value=FakeSuperMarioNative,
        ):
            self.assertTrue(native_vec_env_supports_rgb_render(config))

    def test_supermariobrosnes_turbo_rgb_render_support_rejects_old_metadata(self) -> None:
        class FakeSuperMarioNative:
            metadata = {"render_modes": []}

        config = EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
        )
        with patch(
            "rlab.env._super_mario_bros_nes_turbo_vec_env_type",
            return_value=FakeSuperMarioNative,
        ):
            self.assertFalse(native_vec_env_supports_rgb_render(config))

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

    def test_state_probs_are_native_weights_and_count_checked(self) -> None:
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
            self.assertEqual(config.state_probs, (1.0, 3.0))

            with self.assertRaisesRegex(ValueError, "count must match"):
                resolve_mixed_state_config(
                    EnvConfig(
                        game="SuperMarioBros-Nes-v0",
                        states=("Level1-1", "Level1-2"),
                        state_probs=(1.0,),
                    ),
                    n_envs=8,
                )

            config = resolve_mixed_state_config(
                EnvConfig(
                    game="SuperMarioBros-Nes-v0",
                    states=("Level1-1", "Level1-2"),
                    state_probs=(1.0, 0.0),
                ),
                n_envs=8,
            )
            self.assertEqual(config.state_probs, (1.0, 0.0))

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

    def test_smb3_target_declares_native_life_variable(self) -> None:
        self.assertIs(target_for_game("SuperMarioBros3-Nes-v0"), SuperMarioBros3NesV0Target)
        self.assertEqual(SuperMarioBros3NesV0Target.native_life_variable, "lives")

    def test_smb3_score_reward_uses_hpos_score_and_death(self) -> None:
        config = resolve_env_config(
            EnvConfig(
                game="SuperMarioBros3-Nes-v0",
                env_wrappers=(
                    {"id": "SuperMarioBros3NesProgressInfoWrapper"},
                    {
                        "id": "SuperMarioBros3NesRewardEnvWrapper",
                        "kwargs": {
                            "use_retro_reward": False,
                            "reward_mode": "score",
                            "progress_reward_scale": 1.0,
                            "terminal_reward": 50,
                            "death_penalty": 25,
                            "completion_reward": 0.0,
                            "reward_scale": 10,
                            "score_progress_clipped": False,
                        },
                    },
                ),
            )
        )
        tracker = SuperMarioBros3NesV0Target.create_tracker(config)
        tracker.reset({"hpos": 0, "lives": 4, "score": 0})

        progress_info = {"hpos": 12, "lives": 4, "score": 100}
        progress = tracker.step(99.0, progress_info, done=False)

        self.assertEqual(progress.reward, 13.0)
        self.assertEqual(progress_info["progress_delta"], 12)
        self.assertEqual(progress_info["progress_reward_component"], 12.0)
        self.assertEqual(progress_info["score_delta"], 100)
        self.assertEqual(progress_info["score_reward_component"], 1.0)
        self.assertEqual(progress_info["native_reward_component"], 0.0)
        self.assertEqual(progress_info["death_penalty_component"], 0.0)

        death_info = {"hpos": 12, "lives": 3, "score": 100}
        death_progress = tracker.step(0.0, death_info, done=True)

        self.assertEqual(death_progress.reward, -25.0)
        self.assertTrue(death_info["died"])
        self.assertEqual(death_info["death_penalty_component"], -25.0)
        self.assertFalse(death_info["level_complete"])

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
    def test_vector_infos_to_list_preserves_masked_lane_values(self) -> None:
        previous_lives = np.empty((2,), dtype=object)
        previous_lives[:] = [[3], [0]]
        infos = {
            "state": np.asarray(["Level1-1", "Level1-2"], dtype=object),
            "_state": np.asarray([True, False]),
            "final_info": {
                "score": np.asarray([100, 200]),
                "done_on_info": {
                    "life_loss": {
                        "trigger": np.asarray(["lives_decrease", ""], dtype=object),
                        "_trigger": np.asarray([True, False]),
                        "prev": previous_lives,
                        "_prev": np.asarray([True, False]),
                    },
                    "_life_loss": np.asarray([True, False]),
                    "level_change": {
                        "trigger": np.asarray(["", "level_change"], dtype=object),
                        "_trigger": np.asarray([False, True]),
                    },
                    "_level_change": np.asarray([False, True]),
                },
            },
            "_final_info": np.asarray([True, True]),
        }

        lanes = vector_infos_to_list(infos, 2)

        self.assertEqual(lanes[0]["state"], "Level1-1")
        self.assertNotIn("state", lanes[1])
        self.assertEqual(lanes[0]["final_info"]["score"], 100)
        self.assertEqual(
            lanes[0]["final_info"]["done_on_info"],
            {"life_loss": {"trigger": "lives_decrease", "prev": [3]}},
        )
        self.assertEqual(lanes[1]["final_info"]["score"], 200)
        self.assertEqual(
            lanes[1]["final_info"]["done_on_info"],
            {"level_change": {"trigger": "level_change"}},
        )

    def test_gym_vector_adapter_emits_sb3_done_infos(self) -> None:
        class FakeGymVectorEnv(gym.vector.VectorEnv):
            metadata = {"autoreset_mode": "same-step"}

            def __init__(self) -> None:
                self.num_envs = 2
                self.single_observation_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(2,),
                    dtype=np.uint8,
                )
                self.single_action_space = gym.spaces.Discrete(3)
                self.observation_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(2, 2),
                    dtype=np.uint8,
                )
                self.action_space = gym.spaces.MultiDiscrete([3, 3])
                self.closed = False
                self.reset_seed = None

            def reset(self, *, seed=None, options=None):
                self.reset_seed = seed
                return (
                    np.zeros((2, 2), dtype=np.uint8),
                    {
                        "state": np.asarray(["Level1-1", "Level1-2"], dtype=object),
                        "_state": np.asarray([True, True]),
                    },
                )

            def step(self, actions):
                self.actions = actions
                return (
                    np.ones((2, 2), dtype=np.uint8),
                    np.asarray([1.0, 2.0], dtype=np.float32),
                    np.asarray([True, False]),
                    np.asarray([False, True]),
                    {
                        "state": np.asarray(["Level1-3", "Level1-4"], dtype=object),
                        "_state": np.asarray([True, True]),
                        "final_obs": np.asarray(
                            [
                                np.asarray([9, 9], dtype=np.uint8),
                                np.asarray([8, 8], dtype=np.uint8),
                            ],
                            dtype=object,
                        ),
                        "_final_obs": np.asarray([True, True]),
                        "final_info": {
                            "score": np.asarray([100, 200]),
                            "done_on_info": np.asarray(
                                [{"life_loss": {"prev": [3]}}, {"level_change": {}}],
                                dtype=object,
                            ),
                        },
                        "_final_info": np.asarray([True, True]),
                    },
                )

            def close(self):
                self.closed = True

        source_env = FakeGymVectorEnv()
        env = GymVectorEnvToSb3VecEnv(source_env)
        env.seed(123)

        obs = env.reset()
        self.assertEqual(source_env.reset_seed, 123)
        np.testing.assert_array_equal(obs, np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(
            env.reset_infos,
            [
                {"state": "Level1-1"},
                {"state": "Level1-2"},
            ],
        )

        env.step_async([0, 1])
        obs, rewards, dones, infos = env.step_wait()

        np.testing.assert_array_equal(obs, np.ones((2, 2), dtype=np.uint8))
        np.testing.assert_array_equal(rewards, np.asarray([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(dones, np.asarray([True, True]))
        self.assertEqual(infos[0]["score"], 100)
        self.assertEqual(infos[0]["done_on_info"], {"life_loss": {"prev": [3]}})
        np.testing.assert_array_equal(
            infos[0]["terminal_observation"],
            np.asarray([9, 9], dtype=np.uint8),
        )
        self.assertEqual(infos[0]["reset_info"], {"state": "Level1-3"})
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        self.assertEqual(infos[1]["score"], 200)
        self.assertEqual(infos[1]["done_on_info"], {"level_change": {}})
        self.assertEqual(infos[1]["reset_info"], {"state": "Level1-4"})
        self.assertTrue(infos[1]["TimeLimit.truncated"])
        native_step_stats = env.native_step_stats()
        self.assertGreater(native_step_stats["seconds_total"], 0.0)
        self.assertEqual(native_step_stats["calls_total"], 1)
        self.assertEqual(native_step_stats["num_envs"], 2)

    def test_records_provider_step_timing(self) -> None:
        class FakeGymVectorEnv(gym.vector.VectorEnv):
            num_envs = 2

            def __init__(self) -> None:
                self.single_observation_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(2, 2),
                    dtype=np.uint8,
                )
                self.single_action_space = gym.spaces.Discrete(3)
                self.observation_space = self.single_observation_space
                self.action_space = gym.spaces.MultiDiscrete([3, 3])

            def reset(self, *, seed=None, options=None):
                return np.zeros((2, 2), dtype=np.uint8), {}

            def step(self, actions):
                return (
                    np.ones((2, 2), dtype=np.uint8),
                    np.asarray([1.0, 2.0], dtype=np.float32),
                    np.asarray([False, False]),
                    np.asarray([False, False]),
                    {},
                )

        env = GymVectorEnvToSb3VecEnv(FakeGymVectorEnv())
        env.step_async([0, 1])
        with patch("rlab.env.time.perf_counter", side_effect=[10.0, 10.25]):
            env.step_wait()

        self.assertEqual(
            env.native_step_stats(),
            {
                "seconds_total": 0.25,
                "calls_total": 1,
                "num_envs": 2,
            },
        )

    def test_vector_info_view_lazily_reads_masked_terminal_info(self) -> None:
        final_obs = np.asarray(
            [np.asarray([9, 9], dtype=np.uint8), np.asarray([8, 8], dtype=np.uint8)],
            dtype=object,
        )
        view = VectorInfoView(
            {
                "state": np.asarray(["Level1-3", "Level1-4"], dtype=object),
                "_state": np.asarray([True, False]),
                "final_obs": final_obs,
                "_final_obs": np.asarray([True, True]),
                "final_info": {"score": np.asarray([100, 200])},
                "_final_info": np.asarray([True, True]),
            },
            2,
        )

        self.assertEqual(view.value("state", 0), "Level1-3")
        self.assertIsNone(view.value("state", 1))
        self.assertEqual(view.lane_mapping("final_info", 1), {"score": 200})
        terminal = view.terminal_info(0, truncated=False)
        self.assertEqual(terminal["score"], 100)
        self.assertEqual(terminal["reset_info"], {"state": "Level1-3"})
        np.testing.assert_array_equal(
            terminal["terminal_observation"],
            np.asarray([9, 9], dtype=np.uint8),
        )

    def test_fused_pipeline_uses_canonical_hook_info_mode(self) -> None:
        class FakeGymVectorEnv(gym.vector.VectorEnv):
            num_envs = 1
            single_observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(1,),
                dtype=np.uint8,
            )
            single_action_space = gym.spaces.Discrete(2)
            observation_space = single_observation_space
            action_space = single_action_space

            def reset(self, *, seed=None, options=None):
                return np.zeros((1, 1), dtype=np.uint8), {}

            def step(self, actions):
                return (
                    np.ones((1, 1), dtype=np.uint8),
                    np.asarray([0.0], dtype=np.float32),
                    np.asarray([False]),
                    np.asarray([False]),
                    {},
                )

        source_env = FakeGymVectorEnv()
        default_pipeline = FusedGymVectorPipeline(
            source_env,
            IdentityFusedHooks(None, source_env),
        )
        self.assertEqual(default_pipeline.info_mode, "hook_with_terminal_native")

        with self.assertRaisesRegex(ValueError, "hook_with_terminal_native, full"):
            FusedGymVectorPipeline(
                source_env,
                IdentityFusedHooks(None, source_env),
                info_mode="terminal",
            )

    def test_sb3_fused_vec_env_preserves_sb3_terminal_contract(self) -> None:
        class FakeGymVectorEnv(gym.vector.VectorEnv):
            num_envs = 2

            def __init__(self) -> None:
                self.single_observation_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(2,),
                    dtype=np.uint8,
                )
                self.single_action_space = gym.spaces.Discrete(3)
                self.observation_space = self.single_observation_space
                self.action_space = gym.spaces.MultiDiscrete([3, 3])
                self.reset_seed = None

            def reset(self, *, seed=None, options=None):
                self.reset_seed = seed
                return (
                    np.zeros((2, 2), dtype=np.uint8),
                    {
                        "state": np.asarray(["Level1-1", "Level1-2"], dtype=object),
                        "_state": np.asarray([True, True]),
                    },
                )

            def step(self, actions):
                self.actions = actions
                return (
                    np.ones((2, 2), dtype=np.uint8),
                    np.asarray([1.0, 2.0], dtype=np.float32),
                    np.asarray([True, False]),
                    np.asarray([False, True]),
                    {
                        "state": np.asarray(["Level1-3", "Level1-4"], dtype=object),
                        "_state": np.asarray([True, True]),
                        "final_obs": np.asarray(
                            [
                                np.asarray([9, 9], dtype=np.uint8),
                                np.asarray([8, 8], dtype=np.uint8),
                            ],
                            dtype=object,
                        ),
                        "_final_obs": np.asarray([True, True]),
                        "final_info": {"score": np.asarray([100, 200])},
                        "_final_info": np.asarray([True, True]),
                    },
                )

        source_env = FakeGymVectorEnv()
        pipeline = FusedGymVectorPipeline(source_env, IdentityFusedHooks(None, source_env))
        env = Sb3FusedVecEnv(pipeline)
        env.seed(123)

        obs = env.reset()
        self.assertEqual(source_env.reset_seed, 123)
        np.testing.assert_array_equal(obs, np.zeros((2, 2), dtype=np.uint8))
        self.assertEqual(env.reset_infos, [{"state": "Level1-1"}, {"state": "Level1-2"}])

        env.step_async([0, 1])
        obs, rewards, dones, infos = env.step_wait()

        np.testing.assert_array_equal(obs, np.ones((2, 2), dtype=np.uint8))
        np.testing.assert_array_equal(rewards, np.asarray([1.0, 2.0], dtype=np.float32))
        np.testing.assert_array_equal(dones, np.asarray([True, True]))
        self.assertEqual(infos[0]["score"], 100)
        self.assertEqual(infos[0]["reset_info"], {"state": "Level1-3"})
        self.assertEqual(infos[0]["episode"]["l"], 1)
        self.assertFalse(infos[0]["TimeLimit.truncated"])
        self.assertTrue(infos[1]["TimeLimit.truncated"])
        self.assertEqual(env.native_step_stats()["calls_total"], 1)

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

    def test_mario_fused_hooks_match_old_stack_for_level_change_reward(self) -> None:
        class FakeMarioVectorEnv(gym.vector.VectorEnv):
            num_envs = 1

            def __init__(self) -> None:
                self.single_observation_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(4, 84, 84),
                    dtype=np.uint8,
                )
                self.single_action_space = gym.spaces.MultiBinary(9)
                self.observation_space = self.single_observation_space
                self.action_space = gym.spaces.MultiBinary(9)
                self.actions = None
                self.step_count = 0

            def reset(self, *, seed=None, options=None):
                del seed, options
                return (
                    np.zeros((1, 4, 84, 84), dtype=np.uint8),
                    {
                        "lives": np.asarray([3]),
                        "score": np.asarray([0]),
                        "levelHi": np.asarray([0]),
                        "levelLo": np.asarray([0]),
                        "xscrollHi": np.asarray([0]),
                        "xscrollLo": np.asarray([0]),
                    },
                )

            def step(self, actions):
                self.actions = np.asarray(actions)
                self.step_count += 1
                level_lo = 0 if self.step_count == 1 else 1
                score = 0 if self.step_count == 1 else 40
                xscroll_lo = 0 if self.step_count == 1 else 10
                return (
                    np.ones((1, 4, 84, 84), dtype=np.uint8),
                    np.asarray([0.0], dtype=np.float32),
                    np.asarray([False]),
                    np.asarray([False]),
                    {
                        "lives": np.asarray([3]),
                        "score": np.asarray([score]),
                        "levelHi": np.asarray([0]),
                        "levelLo": np.asarray([level_lo]),
                        "xscrollHi": np.asarray([0]),
                        "xscrollLo": np.asarray([xscroll_lo]),
                    },
                )

            def close(self):
                pass

        config = resolve_env_config(
            EnvConfig(
                env_provider="supermariobrosnes-turbo",
                game="SuperMarioBros-Nes-v0",
                action_set="simple",
                reward_mode="baseline",
                max_episode_steps=0,
                info_events={"level_change": (("levelHi", "levelLo"), "change")},
            )
        )

        old_env = VecRetroProgressInfo(
            VecDiscreteRetroActions(GymVectorEnvToSb3VecEnv(FakeMarioVectorEnv()), config=config),
            config=config,
        )
        old_env.reset()
        old_env.step_async(np.asarray([0], dtype=np.int64))
        old_env.step_wait()
        old_env.step_async(np.asarray([0], dtype=np.int64))
        _old_obs, old_rewards, old_dones, old_infos = old_env.step_wait()

        native_env = FakeMarioVectorEnv()
        fused_env = Sb3FusedVecEnv(
            FusedGymVectorPipeline(
                native_env,
                SuperMarioBrosNesFusedHooks(config, native_env),
            )
        )
        fused_env.reset()
        fused_env.step_async(np.asarray([0], dtype=np.int64))
        fused_env.step_wait()
        fused_env.step_async(np.asarray([0], dtype=np.int64))
        _new_obs, new_rewards, new_dones, new_infos = fused_env.step_wait()

        np.testing.assert_allclose(new_rewards, old_rewards)
        np.testing.assert_array_equal(new_dones, old_dones)
        self.assertEqual(new_infos[0]["info_events"], old_infos[0]["info_events"])
        self.assertEqual(new_infos[0]["level_complete"], old_infos[0]["level_complete"])
        self.assertEqual(new_infos[0]["levelHi"], old_infos[0]["levelHi"])
        self.assertEqual(new_infos[0]["levelLo"], old_infos[0]["levelLo"])

    def test_mario_level_change_does_not_double_count_terminal_x_progress(self) -> None:
        class FakeMarioVectorEnv(gym.vector.VectorEnv):
            num_envs = 1

            def __init__(self) -> None:
                self.single_observation_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(4, 84, 84),
                    dtype=np.uint8,
                )
                self.single_action_space = gym.spaces.MultiBinary(9)
                self.observation_space = self.single_observation_space
                self.action_space = gym.spaces.MultiBinary(9)
                self.step_count = 0

            def reset(self, *, seed=None, options=None):
                del seed, options
                return (
                    np.zeros((1, 4, 84, 84), dtype=np.uint8),
                    {
                        "lives": np.asarray([3]),
                        "score": np.asarray([0]),
                        "levelHi": np.asarray([0]),
                        "levelLo": np.asarray([0]),
                        "xscrollHi": np.asarray([0]),
                        "xscrollLo": np.asarray([0]),
                    },
                )

            def step(self, actions):
                del actions
                self.step_count += 1
                level_lo = 0 if self.step_count == 1 else 1
                score = 0 if self.step_count == 1 else 40
                return (
                    np.ones((1, 4, 84, 84), dtype=np.uint8),
                    np.asarray([0.0], dtype=np.float32),
                    np.asarray([False]),
                    np.asarray([False]),
                    {
                        "lives": np.asarray([3]),
                        "score": np.asarray([score]),
                        "levelHi": np.asarray([0]),
                        "levelLo": np.asarray([level_lo]),
                        "xscrollHi": np.asarray([0]),
                        "xscrollLo": np.asarray([250]),
                    },
                )

            def close(self):
                pass

        config = resolve_env_config(
            EnvConfig(
                env_provider="supermariobrosnes-turbo",
                game="SuperMarioBros-Nes-v0",
                action_set="simple",
                reward_mode="score",
                max_episode_steps=0,
                info_events={"level_change": (("levelHi", "levelLo"), "change")},
            )
        )

        old_env = VecRetroProgressInfo(
            VecDiscreteRetroActions(GymVectorEnvToSb3VecEnv(FakeMarioVectorEnv()), config=config),
            config=config,
        )
        old_env.reset()
        old_env.step_async(np.asarray([0], dtype=np.int64))
        old_env.step_wait()
        old_env.step_async(np.asarray([0], dtype=np.int64))
        _old_obs, old_rewards, _old_dones, old_infos = old_env.step_wait()

        native_env = FakeMarioVectorEnv()
        fused_env = Sb3FusedVecEnv(
            FusedGymVectorPipeline(
                native_env,
                SuperMarioBrosNesFusedHooks(config, native_env),
            )
        )
        fused_env.reset()
        fused_env.step_async(np.asarray([0], dtype=np.int64))
        fused_env.step_wait()
        fused_env.step_async(np.asarray([0], dtype=np.int64))
        _new_obs, new_rewards, _new_dones, new_infos = fused_env.step_wait()

        np.testing.assert_allclose(old_rewards, np.asarray([0.4], dtype=np.float32))
        np.testing.assert_allclose(new_rewards, old_rewards)
        self.assertEqual(old_infos[0]["progress_delta"], 0)
        self.assertEqual(new_infos[0]["progress_delta"], 0)
        self.assertEqual(old_infos[0]["global_x_pos"], 250)
        self.assertEqual(new_infos[0]["global_x_pos"], 250)
        self.assertTrue(old_infos[0]["level_complete"])
        self.assertTrue(new_infos[0]["level_complete"])

    def test_mario_fused_hooks_emit_nonterminal_reward_component_info(self) -> None:
        class FakeMarioVectorEnv(gym.vector.VectorEnv):
            num_envs = 1

            def __init__(self) -> None:
                self.single_observation_space = gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(4, 84, 84),
                    dtype=np.uint8,
                )
                self.single_action_space = gym.spaces.MultiBinary(9)
                self.observation_space = self.single_observation_space
                self.action_space = gym.spaces.MultiBinary(9)
                self.step_count = 0

            def reset(self, *, seed=None, options=None):
                del seed, options
                return (
                    np.zeros((1, 4, 84, 84), dtype=np.uint8),
                    {},
                )

            def step(self, actions):
                del actions
                self.step_count += 1
                return (
                    np.ones((1, 4, 84, 84), dtype=np.uint8),
                    np.asarray([0.0], dtype=np.float32),
                    np.asarray([False]),
                    np.asarray([False]),
                    {
                        "lives": np.asarray([2]),
                        "score": np.asarray([0]),
                        "levelHi": np.asarray([0]),
                        "levelLo": np.asarray([0]),
                        "xscrollHi": np.asarray([0]),
                        "xscrollLo": np.asarray([self.step_count * 5]),
                    },
                )

        config = resolve_env_config(
            EnvConfig(
                env_provider="supermariobrosnes-turbo",
                game="SuperMarioBros-Nes-v0",
                action_set="simple",
                reward_mode="score",
                max_episode_steps=0,
            )
        )
        old_env = VecRetroProgressInfo(
            VecDiscreteRetroActions(GymVectorEnvToSb3VecEnv(FakeMarioVectorEnv()), config=config),
            config=config,
        )
        old_env.reset()
        old_env.step_async(np.asarray([0], dtype=np.int64))
        _old_obs, old_rewards, old_dones, old_infos = old_env.step_wait()

        native_env = FakeMarioVectorEnv()
        fused_env = Sb3FusedVecEnv(
            FusedGymVectorPipeline(
                native_env,
                SuperMarioBrosNesFusedHooks(config, native_env),
            )
        )
        fused_env.reset()
        fused_env.step_async(np.asarray([0], dtype=np.int64))
        _new_obs, new_rewards, new_dones, new_infos = fused_env.step_wait()

        np.testing.assert_allclose(new_rewards, old_rewards)
        np.testing.assert_array_equal(new_dones, old_dones)
        for key in (
            "levelHi",
            "levelLo",
            "progress_delta",
            "progress_component",
            "progress_reward_component",
            "death_penalty_component",
            "raw_reward",
            "shaped_reward",
        ):
            self.assertEqual(new_infos[0][key], old_infos[0][key])

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


def _recording_native_vec_env(created: list[dict[str, object]]) -> type:
    class RecordingNative:
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

    return RecordingNative


def _identity_vec_progress(env: object, config: EnvConfig) -> object:
    del config
    return env


@contextmanager
def _patched_native_vec_env(
    fake_native: type,
    *,
    listed_states: list[str] | None = None,
    progress_side_effect: Callable[[object, EnvConfig], object] | None = None,
    supports_done_on: bool | None = None,
    supports_named_done_on: bool | None = None,
) -> Iterator[None]:
    if progress_side_effect is None:
        progress_side_effect = _identity_vec_progress
    with ExitStack() as stack:
        stack.enter_context(patch("rlab.env.RetroVecEnv", fake_native))
        stack.enter_context(
            patch("rlab.env.VecRetroProgressInfo", side_effect=progress_side_effect)
        )
        stack.enter_context(patch("rlab.env.VecMonitor", side_effect=lambda env: env))
        stack.enter_context(
            patch("rlab.env.maybe_transpose_vec_image", side_effect=lambda env: env)
        )
        if listed_states is not None:
            stack.enter_context(
                patch("rlab.env.retro.data.list_states", return_value=listed_states)
            )
        if supports_done_on is not None:
            stack.enter_context(
                patch("rlab.env.native_vec_env_supports_done_on", return_value=supports_done_on)
            )
        if supports_named_done_on is not None:
            stack.enter_context(
                patch(
                    "rlab.env.native_vec_env_supports_named_done_on",
                    return_value=supports_named_done_on,
                )
            )
        yield


class NativeMixedStateVecEnvTests(unittest.TestCase):
    def test_training_vec_env_passes_weighted_states_as_native_state_dict(self) -> None:
        created: list[dict[str, object]] = []
        fake_native = _recording_native_vec_env(created)

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            states=("Level1-1", "Level1-2"),
            state_probs=(0.5, 0.5),
        )
        with _patched_native_vec_env(fake_native, listed_states=["Level1-1", "Level1-2"]):
            env = make_training_vec_env(config, n_envs=16, seed=7)

        self.assertIsInstance(env, fake_native)
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

    def test_training_vec_env_passes_raw_state_weights_to_native_vec_env(self) -> None:
        created: list[dict[str, object]] = []
        fake_native = _recording_native_vec_env(created)

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            states=("Level1-1", "Level1-2"),
            state_probs=(1.0, 3.0),
        )
        with _patched_native_vec_env(fake_native, listed_states=["Level1-1", "Level1-2"]):
            make_training_vec_env(config, n_envs=16, seed=7)

        self.assertEqual(created[0]["state"], {"Level1-1": 1.0, "Level1-2": 3.0})

    def test_training_vec_env_passes_fixed_lane_states_as_native_state_list(self) -> None:
        created: list[dict[str, object]] = []
        fake_native = _recording_native_vec_env(created)

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            states=("Level1-1", "Level1-2", "Level1-1", "Level1-2"),
        )
        with _patched_native_vec_env(fake_native, listed_states=["Level1-1", "Level1-2"]):
            env = make_training_vec_env(config, n_envs=4, seed=7)

        self.assertIsInstance(env, fake_native)
        self.assertEqual(created[0]["state"], ["Level1-1", "Level1-2", "Level1-1", "Level1-2"])
        self.assertNotIn("states", created[0])
        self.assertNotIn("state_probs", created[0])

    def test_training_vec_env_passes_sticky_action_prob_to_native_vec_env(self) -> None:
        created: list[dict[str, object]] = []
        fake_native = _recording_native_vec_env(created)

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            sticky_action_prob=0.25,
        )
        with _patched_native_vec_env(fake_native):
            env = make_training_vec_env(config, n_envs=4, seed=7)

        self.assertIsInstance(env, fake_native)
        self.assertEqual(created[0]["sticky_action_prob"], 0.25)
        self.assertNotIn("action_sticky_prob", created[0])

    def test_training_vec_env_passes_configured_native_done_on_rules(self) -> None:
        created: list[dict[str, object]] = []
        progress_configs: list[EnvConfig] = []
        fake_native = _recording_native_vec_env(created)

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
        with _patched_native_vec_env(
            fake_native,
            listed_states=["Level1-1", "Level1-2"],
            progress_side_effect=fake_progress,
            supports_done_on=True,
        ):
            env = make_training_vec_env(config, n_envs=16, seed=7)

        self.assertIsInstance(env, fake_native)
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
        fake_native = _recording_native_vec_env(created)

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
        with _patched_native_vec_env(
            fake_native,
            listed_states=["Level1-1", "Level1-2"],
            progress_side_effect=fake_progress,
            supports_done_on=True,
        ):
            env = make_training_vec_env(config, n_envs=16, seed=7)

        self.assertIsInstance(env, fake_native)
        self.assertEqual(created[0]["done_on"], {"life_loss": ("lives", "decrease")})

    def test_training_vec_env_passes_named_done_events_without_local_rules(self) -> None:
        created: list[dict[str, object]] = []
        fake_native = _recording_native_vec_env(created)

        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            done_on_events=("life_loss", "level_change"),
            states=("Level1-1", "Level1-2"),
            state_probs=(0.5, 0.5),
        )
        with _patched_native_vec_env(
            fake_native,
            listed_states=["Level1-1", "Level1-2"],
            supports_done_on=True,
            supports_named_done_on=True,
        ):
            env = make_training_vec_env(config, n_envs=16, seed=7)

        self.assertIsInstance(env, fake_native)
        self.assertEqual(
            created[0]["done_on"],
            {"life_loss": None, "level_change": None},
        )

    def test_training_vec_env_requires_native_done_on_support_when_rules_requested(
        self,
    ) -> None:
        fake_native = _recording_native_vec_env([])
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            action_set="native",
            reward_mode="native",
            info_events={"life_loss": ("lives", "decrease")},
            done_on_events=("life_loss",),
        )
        with _patched_native_vec_env(fake_native, supports_done_on=False):
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
        with _patched_native_vec_env(FakeNative, listed_states=["Level1-1", "Level1-2"]):
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
