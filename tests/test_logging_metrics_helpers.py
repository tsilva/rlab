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
