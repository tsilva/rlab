from __future__ import annotations

# ruff: noqa: F401

import argparse
import io
import json
import re
import sys
import tempfile
import types
import unittest
from collections import deque
from contextlib import redirect_stderr
from io import StringIO
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
    GymVectorEnvToSb3VecEnv,
    StickyAction,
    VecDiscreteRetroActions,
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
    provider_native_vec_kwargs,
    resolve_env_config,
    resolve_mixed_state_config,
    state_name_candidates_from_level_id,
    vector_infos_to_list,
)
from rlab.env_config import (
    env_config_from_args,
    parse_info_events,
    parse_obs_crop,
    parse_state_probs,
    parse_states,
)
from rlab.envs.super_mario_bros_nes import SuperMarioBrosNesFusedHooks
from rlab.fused_vec import FusedGymVectorPipeline, IdentityFusedHooks, Sb3FusedVecEnv, VectorInfoView
from rlab.metric_names import (
    TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MEAN,
    TRAIN_DONE_LEVEL_CHANGE_FROM_RATE_MIN,
    TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
)
from rlab.model_sources import (
    ResolvedModelSource,
    download_huggingface_model_source,
    model_artifact_checkpoint_step,
    model_source_ref,
    parse_wandb_run_ref,
    parse_huggingface_model_ref,
    single_model_artifact_ref,
    single_huggingface_model_ref,
)
from rlab.play import build_parser as build_play_parser
from rlab.vec_wrappers import normalize_vec_wrapper_specs
from rlab.play import display_replay_config
from rlab.play import main as play_main
from rlab.play import metadata_playback_config
from rlab.play import model_observation
from rlab.play import playback_env_config
from rlab.play import playback_should_end_episode
from rlab.play import render_obs_stack
from rlab.play import resolved_play_launch_lines
from rlab.play import task_conditioning_change_message
from rlab.play import task_conditioning_start_message
from rlab.eval import build_parser as build_eval_parser
from rlab.eval import main as eval_main
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.task_advantage import normalize_advantages_by_task
from rlab.targets import SuperMarioBros3NesV0Target, SuperMarioBrosNesV0Target, target_for_game
from rlab.train import (
    Sb3HumanOutputFormatCallback,
    disable_sb3_human_output_truncation,
    eval_checkpoint_artifact_ref,
)
from rlab.wandb_artifacts import (
    artifact_download_dir,
    model_artifact_ref,
    safe_artifact_stem,
)
from rlab.wandb_artifacts import metadata_from_wandb_artifact
from rlab.wandb_utils import default_wandb_project_path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)

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

    def test_logs_native_env_step_throughput_when_available(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records: list[tuple[str, float]] = []

            def record(self, key: str, value: float) -> None:
                self.records.append((key, value))

        class NativeStatsEnv:
            def __init__(self) -> None:
                self.stats = [
                    {"seconds_total": 1.0, "calls_total": 10, "num_envs": 4},
                    {"seconds_total": 3.0, "calls_total": 35, "num_envs": 4},
                ]

            def native_step_stats(self):
                return self.stats.pop(0)

        class Wrapper:
            def __init__(self, env) -> None:
                self.venv = env

        class Model:
            def __init__(self) -> None:
                self.logger = Logger()
                self.env = Wrapper(NativeStatsEnv())

        times = iter([0.0, 5.0])
        callback = ThroughputCallback(clock=lambda: next(times))
        model = Model()
        callback.model = model  # type: ignore[assignment]

        callback.num_timesteps = 40
        callback._on_rollout_start()
        callback.num_timesteps = 140
        callback._on_rollout_end()

        self.assertEqual(
            model.logger.records,
            [
                ("throughput/rollout_fps", 20.0),
                ("throughput/native_env_step_seconds", 2.0),
                ("throughput/native_env_step_fps", 50.0),
                ("throughput/native_env_step_batch_fps", 12.5),
                ("throughput/native_env_step_fraction", 0.4),
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
