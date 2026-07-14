from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tempfile
import types
import unittest
from collections import deque
from unittest.mock import patch
from pathlib import Path

import gymnasium as gym
import numpy as np

import rlab.metric_names as metric_names
from rlab.artifacts import (
    apply_config_defaults,
    build_s3_artifact_uri,
    checkpoint_step,
    load_playback_env_config,
    load_model_metadata,
    log_wandb_model_artifact,
    model_metadata_path,
    playback_env_config,
    write_model_metadata,
)
from rlab.cli_args import explicit_arg_dests
from rlab.env import (
    EnvConfig,
    state_name_candidates_from_level_id,
)
from rlab.env_config import (
    env_config_from_args,
)
from rlab.model_sources import (
    ResolvedModelSource,
    artifact_lookup_project_paths,
    download_huggingface_model_source,
    model_source_ref,
    parse_wandb_run_ref,
    parse_huggingface_model_ref,
    single_model_artifact_ref,
    single_huggingface_model_ref,
)
from rlab.play import build_parser as build_play_parser
from rlab.play import display_replay_config
from rlab.play import main as play_main
from rlab.play import model_observation
from rlab.play import ObsStackViewer
from rlab.play import playback_runtime_config
from rlab.play import playback_should_end_episode
from rlab.play import render_obs_stack
from rlab.play import resolved_play_launch_lines
from rlab.play import task_conditioning_change_message
from rlab.play import task_conditioning_start_message
from rlab.eval import build_parser as build_eval_parser
from rlab.eval import main as eval_main
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.task_advantage import normalize_advantages_by_task
from rlab.train import build_parser as build_train_parser
from rlab.train_config import build_train_command_from_fields as build_train_command
from rlab.wandb_artifacts import (
    artifact_download_dir,
    download_model_artifact,
    model_artifact_ref,
    safe_artifact_stem,
)
from rlab.wandb_artifacts import metadata_from_wandb_artifact
from rlab.wandb_utils import default_wandb_project_path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def mario_task(*, conditioning: bool = False) -> dict:
    task = {
        "id": "mario",
        "action": {"set": "simple"},
        "signals": {
            "x": ["xscrollHi", "xscrollLo"],
            "score": "score",
            "lives": "lives",
            "level": ["levelHi", "levelLo"],
        },
        "events": {
            "life_loss": {"signal": "lives", "operation": "decrease"},
            "level_change": {"signal": "level", "operation": "change"},
        },
        "termination": {
            "failure": ["life_loss"],
            "success": ["level_change"],
            "max_episode_steps": 2345,
        },
        "reward": {
            "reward_mode": "score",
            "reward_scale": 10.0,
            "score_progress_clipped": True,
        },
    }
    if conditioning:
        task["conditioning"] = {
            "enabled": True,
            "signal": "level",
            "values": [[0, 0], [0, 1]],
        }
    return task


class CommandAndArtifactTests(unittest.TestCase):
    def test_build_train_command_skips_empty_target_kl(self) -> None:
        cmd = build_train_command(
            {
                "run_name": "candidate",
                "states": "Level1-1,Level1-2",
                "state_probs": "1,3",
                "target_kl": 0.0,
                "clip_range_vf": 0.2,
                "task": mario_task(conditioning=True),
                "policy_net_arch": "128",
                "value_net_arch": "512,512",
                "advantage_normalization": "per-task",
                "wandb": True,
                "normalize_advantage": False,
            }
        )
        self.assertIn("--run-name", cmd)
        self.assertIn("--states", cmd)
        self.assertIn("--state-probs", cmd)
        self.assertIn("--task-json", cmd)
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
        self.assertIn("--wandb", cmd)
        self.assertIn("--no-normalize-advantage", cmd)

    def test_build_train_command_ignores_removed_training_loop_eval_toggles(self) -> None:
        cmd = build_train_command(
            {
                "eval_freq": 0,
                "eval_episodes": 0,
                "eval_stochastic": False,
                "no_eval_videos": True,
                "eval_video_fps": 60,
                "eval_video_scale": 2,
            }
        )

        self.assertNotIn("--eval-freq", cmd)
        self.assertNotIn("--eval-episodes", cmd)
        self.assertNotIn("--no-eval-stochastic", cmd)
        self.assertNotIn("--no-eval-videos", cmd)
        self.assertNotIn("--eval-video-fps", cmd)
        self.assertNotIn("--eval-video-scale", cmd)

    def test_train_parser_accepts_canonical_task(self) -> None:
        task = mario_task(conditioning=True)
        args = build_train_parser().parse_args(
            [
                "--game",
                "SuperMarioBros-Nes-v0",
                "--states",
                "Level1-1,Level1-2",
                "--task-json",
                json.dumps(task),
            ]
        )

        config = env_config_from_args(args, include_states=True)
        self.assertEqual(config.task, task)

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

    def test_wandb_alias_downloads_cache_by_resolved_immutable_version(self) -> None:
        requested_ref = "entity/project/run-checkpoint:latest"
        versions = [
            ("v14", "ppo_game_6500000_steps.zip"),
            ("v15", "ppo_game_8000000_steps.zip"),
        ]
        download_roots: list[Path] = []

        class FakeArtifact:
            def __init__(self, version: str, filename: str) -> None:
                self.version = version
                self.metadata = {"filename": filename}
                self.filename = filename

            def download(self, root: str) -> str:
                path = Path(root)
                download_roots.append(path)
                path.mkdir(parents=True, exist_ok=True)
                (path / self.filename).write_bytes(b"model")
                return str(path)

        class FakeApi:
            def artifact(self, ref: str, type: str | None = None):
                self.assertEqual(ref, requested_ref)
                self.assertEqual(type, "model")
                version, filename = versions.pop(0)
                return FakeArtifact(version, filename)

        fake_api = FakeApi()
        fake_api.assertEqual = self.assertEqual
        fake_wandb = types.SimpleNamespace(Api=lambda: fake_api)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            with (
                patch.dict(sys.modules, {"wandb": fake_wandb}),
                patch("rlab.wandb_artifacts.load_wandb_env"),
            ):
                first = download_model_artifact(requested_ref, root)
                second = download_model_artifact(requested_ref, root)

            self.assertEqual(
                first,
                root / "entity_project_run-checkpoint_v14" / "ppo_game_6500000_steps.zip",
            )
            self.assertEqual(
                second,
                root / "entity_project_run-checkpoint_v15" / "ppo_game_8000000_steps.zip",
            )
            self.assertEqual(
                download_roots,
                [
                    root / "entity_project_run-checkpoint_v14",
                    root / "entity_project_run-checkpoint_v15",
                ],
            )

    def test_s3_artifact_uri_includes_wandb_rom_id_prefix(self) -> None:
        args = argparse.Namespace(game="TestGame-Platform", run_name="candidate/run")
        self.assertEqual(
            build_s3_artifact_uri("s3://wandb", args, Path("final_model.zip"), "final"),
            "s3://wandb/TestGame-Platform/candidate-run-final/final_model.zip",
        )
        self.assertEqual(
            build_s3_artifact_uri(
                "s3://wandb/TestGame-Platform",
                args,
                Path("ppo_test_100_steps.zip"),
                "checkpoint",
            ),
            "s3://wandb/TestGame-Platform/candidate-run-checkpoint/ppo_test_100_steps.zip",
        )
        args.wandb_run_id = "rlab-immutable"
        self.assertEqual(
            build_s3_artifact_uri("s3://wandb", args, Path("final_model.zip"), "final"),
            "s3://wandb/TestGame-Platform/rlab-immutable-final/final_model.zip",
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
                    "s3://bucket/checkpoints/SuperMarioBros-Nes-v0/run-id-checkpoint/ppo_test_100_steps.zip",
                )
            ],
        )
        artifact, aliases = fake_run.artifact_logs[0]
        self.assertEqual(artifact.name, "run-id-checkpoint")
        self.assertEqual(artifact.references[0][1], "ppo_test_100_steps.zip")
        self.assertEqual(aliases, ["latest", "step-100"])
        payload, step = fake_run.metric_logs[0]
        self.assertIsNone(step)
        self.assertEqual(payload[metric_names.GLOBAL_STEP], 100)
        self.assertAlmostEqual(payload[metric_names.TRAIN_ARTIFACT_UPLOAD_SECONDS], 1.3)
        self.assertAlmostEqual(payload[metric_names.TRAIN_ARTIFACT_SAVE_SECONDS], 2.0)
        self.assertEqual(set(payload), {"global_step", "train/artifact/upload/seconds", "train/artifact/save/seconds"})

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

            with (
                patch.dict(sys.modules, {"wandb": FakeWandb()}),
                patch("rlab.artifacts.wandb_artifact_storage_uri", return_value=""),
            ):
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
            game="SuperMarioBros-Nes-v0",
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

            with (
                patch.dict(sys.modules, {"wandb": FakeWandb()}),
                patch("rlab.artifacts.wandb_artifact_storage_uri", return_value=""),
            ):
                log_wandb_model_artifact(
                    fake_run,
                    args,
                    EnvConfig(game="SuperMarioBros-Nes-v0"),
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
                obs_crop=(0, 0, 32, 0),
                obs_crop_mode="mask",
                obs_crop_fill=7,
                task=mario_task(),
            )

            path = write_model_metadata(model_path, args, config, kind="checkpoint")

            self.assertEqual(path, model_metadata_path(model_path))
            metadata = load_model_metadata(model_path)
            training = metadata["training_metadata"]
            env_config = training["env_config"]
            environment = training["environment"]
            self.assertEqual(metadata["checkpoint_step"], 100)
            self.assertEqual(env_config["max_pool_frames"], False)
            self.assertEqual(env_config["observation_size"], 96)
            self.assertEqual(env_config["obs_crop"], [0, 0, 32, 0])
            self.assertEqual(env_config["obs_crop_mode"], "mask")
            self.assertEqual(env_config["obs_crop_fill"], 7)
            self.assertEqual(env_config["task"]["id"], "mario")
            self.assertEqual(env_config["task"]["action"]["set"], "simple")
            self.assertEqual(
                environment["preprocessing"]["obs_crop"],
                [0, 0, 32, 0],
            )
            self.assertEqual(environment["preprocessing"]["obs_crop_mode"], "mask")
            self.assertEqual(environment["preprocessing"]["obs_crop_fill"], 7)
            self.assertEqual(environment["env_id"], "stable-retro-turbo:SuperMarioBros-Nes-v0")
            self.assertEqual(environment["preprocessing"]["frame_stack"], 4)
            self.assertEqual(
                metadata["training_metadata"]["preprocessing"]["max_pool_frames"],
                False,
            )
            self.assertEqual(
                metadata["training_metadata"]["preprocessing"]["sticky_action_prob"],
                0.0,
            )
            self.assertEqual(
                metadata["training_metadata"]["preprocessing"]["obs_crop_mode"],
                "mask",
            )
            self.assertEqual(metadata["training_metadata"]["preprocessing"]["obs_crop_fill"], 7)
            self.assertEqual(
                metadata["training_metadata"]["preprocessing"]["obs_copy"], "safe_view"
            )
            self.assertNotIn("frame_maxpool", metadata["training_metadata"]["preprocessing"])
            self.assertNotIn("action_sticky_prob", metadata["training_metadata"]["preprocessing"])
            self.assertNotIn("copy_observations", metadata["training_metadata"]["preprocessing"])
            self.assertNotIn("env_config", metadata)
            self.assertNotIn("environment", metadata)
            self.assertNotIn("environment_hash", metadata)
            self.assertIn("training_metadata", metadata)
            self.assertIn("training_metadata_hash", metadata)
            self.assertEqual(
                metadata["training_metadata"]["preprocessing"]["frame_stack"],
                4,
            )
            self.assertTrue(metadata["training_metadata"]["preprocessing"]["obs_grayscale"])
            self.assertEqual(env_config["state_sampling_mode"], "weighted")
            self.assertEqual(env_config["state_probs"], [0.25, 0.75])
            self.assertEqual(
                env_config["state_distribution"],
                [
                    {"state": "Level1-1", "probability": 0.25},
                    {"state": "Level1-2", "probability": 0.75},
                ],
            )
            self.assertEqual(
                load_model_metadata(model_path)["training_metadata"]["env_config"][
                    "observation_size"
                ],
                96,
            )

    def test_playback_reads_legacy_v2_top_level_environment_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "legacy.zip"
            model_path.write_bytes(b"zip")
            model_metadata_path(model_path).write_text(
                json.dumps(
                    {
                        "metadata_version": 2,
                        "env_config": {
                            "env_provider": "ale-py",
                            "env_args": {"game": "breakout", "num_envs": 16},
                        },
                    }
                ),
                encoding="utf-8",
            )

            config = load_playback_env_config(model_path)

            self.assertEqual(config.game, "breakout")
            self.assertNotIn("game", config.env_args)
            self.assertNotIn("num_envs", config.env_args)

    def test_playback_config_loads_from_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            write_model_metadata(
                model_path,
                argparse.Namespace(run_name="run", run_description="description"),
                EnvConfig(
                    env_provider="supermariobrosnes-turbo",
                    game="SuperMarioBros-Nes-v0",
                    state="Level2-1",
                    max_pool_frames=False,
                    task=mario_task(),
                    observation_size=96,
                    obs_crop=(32, 0, 0, 0),
                    obs_crop_mode="mask",
                    obs_crop_fill=7,
                ),
                kind="checkpoint",
            )

            config = load_playback_env_config(model_path)

            self.assertEqual(config.env_provider, "supermariobrosnes-turbo")
            self.assertEqual(config.state, "Level2-1")
            self.assertFalse(config.max_pool_frames)
            self.assertEqual(config.task["termination"]["max_episode_steps"], 0)
            self.assertEqual(config.observation_size, 96)
            self.assertEqual(config.obs_crop, (32, 0, 0, 0))
            self.assertEqual(config.obs_crop_mode, "mask")
            self.assertEqual(config.obs_crop_fill, 7)
            self.assertTrue(config.task["reward"]["score_progress_clipped"])
            self.assertEqual(config.task["termination"]["failure"], [])
            self.assertEqual(config.task["termination"]["success"], [])

    def test_playback_requires_model_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")

            with self.assertRaisesRegex(SystemExit, "missing playback metadata"):
                load_playback_env_config(model_path)

    def test_playback_rejects_artifact_runtime_version_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            write_model_metadata(
                model_path,
                argparse.Namespace(run_name="run", run_description="description"),
                EnvConfig(
                    env_provider="supermariobrosnes-turbo",
                    game="SuperMarioBros-Nes-v0",
                    state="Level1-1",
                ),
                kind="checkpoint",
            )
            metadata = load_model_metadata(model_path)
            metadata["training_metadata"]["versions"]["supermariobrosnes_turbo"] = "0.0.0"
            model_metadata_path(model_path).write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaisesRegex(SystemExit, "Artifact runtime version mismatch"):
                load_playback_env_config(model_path)

    def test_eval_model_metadata_defaults_apply_env_provider(self) -> None:
        parser = build_eval_parser()
        parser_defaults = vars(parser.parse_args([]))
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            write_model_metadata(
                model_path,
                argparse.Namespace(run_name="run", run_description="description"),
                EnvConfig(
                    env_provider="supermariobrosnes-turbo",
                    game="SuperMarioBros-Nes-v0",
                    state="Level1-1",
                    obs_crop_mode="mask",
                    obs_crop_fill=7,
                ),
                kind="checkpoint",
            )

            argv = ["--model", str(model_path)]
            args = parser.parse_args(argv)
            explicit_dests = explicit_arg_dests(parser, argv)

            saved_config = load_model_metadata(model_path)["training_metadata"]["env_config"]
            apply_config_defaults(args, saved_config, parser_defaults, explicit_dests)
            config = env_config_from_args(
                args,
                include_states=True,
            )
            self.assertEqual(config.env_provider, "supermariobrosnes-turbo")
            self.assertEqual(config.obs_crop_mode, "mask")
            self.assertEqual(config.obs_crop_fill, 7)

    def test_playback_uses_the_policy_provider_for_display(self) -> None:
        policy_config = EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            state="Level1-1",
        )

        config = display_replay_config(policy_config)

        self.assertEqual(config.env_provider, "supermariobrosnes-turbo")

    def test_stable_retro_atari_viewer_uses_policy_environment_config(self) -> None:
        policy_config = EnvConfig(
            env_provider="stable-retro-turbo",
            game="Breakout-Atari2600-v0",
            env_args={"num_threads": 2, "maxpool_last_two": True},
        )

        config = display_replay_config(policy_config)

        self.assertIs(config, policy_config)

    def test_ale_py_viewer_uses_policy_environment_config(self) -> None:
        policy_config = EnvConfig(
            env_provider="ale-py",
            game="breakout",
            env_args={"maxpool": True},
        )

        config = display_replay_config(policy_config)

        self.assertIs(config, policy_config)

    def test_resolved_play_launch_lines_summarize_repro_fields(self) -> None:
        args = argparse.Namespace(
            artifact_ref="run",
            model="model.zip",
            env_provider="supermariobrosnes-turbo",
            policy_env="fast",
            device="cpu",
            deterministic=False,
            seed=10000,
            episodes=1,
            max_steps=1200,
        )
        policy_config = EnvConfig(
            env_provider="supermariobrosnes-turbo",
            game="SuperMarioBros-Nes-v0",
            state="Level1-1",
            frame_skip=4,
            max_pool_frames=False,
            hud_crop_top=32,
            task=mario_task(),
        )
        display_config = display_replay_config(policy_config)

        lines = resolved_play_launch_lines(
            args,
            argv=["b82-b55reval-s-20260702T174216Z"],
            artifact_ref="entity/project/run-checkpoint:latest",
            policy_config=policy_config,
            display_config=display_config,
        )
        colored_text = "\n".join(lines)
        text = strip_ansi(colored_text)

        self.assertIn("▶ resolved play launch", text)
        self.assertIn("◇ artifact:", text)
        self.assertIn("● policy/eval env:", text)
        self.assertIn("○ viewer source:", text)
        self.assertIn("▤ preprocessing:", text)
        self.assertIn("⚙ action/reward:", text)
        self.assertIn("artifact: entity/project/run-checkpoint:latest", text)
        self.assertIn("model: model.zip", text)
        self.assertIn("policy/eval env: supermariobrosnes-turbo", text)
        self.assertIn("viewer source: supermariobrosnes-turbo", text)
        self.assertIn("shared_with_policy=True", text)
        self.assertIn("source of truth:", text)
        self.assertIn("frame_skip=4", text)
        self.assertIn("max_pool=False", text)
        self.assertIn("action_set=simple", text)
        self.assertIn("termination_events=life_loss,level_change", text)

    def test_playback_env_config_disables_task_termination(self) -> None:
        task = mario_task()
        task["events"]["stalled"] = {
            "signal": "x",
            "operation": "unchanged_for",
            "steps": 300,
        }
        config = EnvConfig(
            game="SuperMarioBros-Nes-v0",
            task=task,
        )

        playback_config = playback_env_config(config)

        self.assertEqual(playback_config.task["termination"]["failure"], [])
        self.assertEqual(playback_config.task["termination"]["success"], [])
        self.assertEqual(playback_config.task["termination"]["timeout"], [])
        self.assertEqual(playback_config.task["termination"]["max_episode_steps"], 0)
        self.assertNotIn("stalled", playback_config.task["events"])
        self.assertIn("level_change", playback_config.task["events"])
        self.assertIs(playback_runtime_config(playback_config), playback_config)
        self.assertEqual(config.task["termination"]["failure"], ["life_loss"])
        self.assertEqual(config.task["termination"]["max_episode_steps"], 2345)
        self.assertIn("stalled", config.task["events"])

        contract_config = playback_env_config(
            config,
            respect_task_termination=True,
        )
        self.assertEqual(contract_config.task["termination"]["failure"], ["life_loss"])
        self.assertEqual(contract_config.task["termination"]["success"], ["level_change"])

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
                task=mario_task(conditioning=True),
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
                task=mario_task(conditioning=True),
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
                task=mario_task(conditioning=True),
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

    def test_gui_playback_defaults_to_stochastic(self) -> None:
        parser = build_play_parser()

        self.assertFalse(hasattr(parser.parse_args([]), "deterministic"))
        self.assertFalse(parser.parse_args([]).step_over)
        self.assertTrue(parser.parse_args(["--step-over"]).step_over)
        self.assertFalse(parser.parse_args([]).respect_task_termination)
        self.assertTrue(parser.parse_args(["--respect-task-termination"]).respect_task_termination)
        self.assertFalse(parser.parse_args([]).no_progress)
        self.assertTrue(parser.parse_args(["--no-progress"]).no_progress)
        self.assertEqual(parser.parse_args([]).episodes, 0)
        self.assertEqual(parser.parse_args(["--episodes", "3"]).episodes, 3)
        self.assertEqual(parser.parse_args([]).seed, DEFAULT_EVAL_SEED)
        self.assertEqual(parser.parse_args(["--seed", "7"]).seed, 7)
        self.assertEqual(parser.parse_args([]).attribution, "none")
        self.assertEqual(parser.parse_args(["--attribution", "gradcam"]).attribution, "gradcam")
        self.assertIsNone(parser.parse_args([]).attribution_interval)
        self.assertEqual(
            parser.parse_args(["--attribution-interval", "12"]).attribution_interval, 12
        )
        self.assertEqual(parser.parse_args([]).attribution_opacity, 0.45)
        help_text = parser.format_help()
        self.assertNotIn("--deterministic", help_text)
        self.assertIn("--episodes", help_text)
        self.assertIn("--step-over", help_text)
        self.assertIn("--attribution", help_text)
        self.assertNotIn("--stochastic", help_text)
        self.assertNotIn("--no-stochastic", help_text)

    def test_play_main_checks_runtime_from_playback_metadata(self) -> None:
        class StopPlayback(Exception):
            pass

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            write_model_metadata(
                model_path,
                argparse.Namespace(run_name="run", run_description="description"),
                EnvConfig(
                    env_provider="supermariobrosnes-turbo",
                    game="SuperMarioBros-Nes-v0",
                    state="Level1-1",
                ),
                kind="checkpoint",
            )

            with (
                patch(
                    "rlab.play.assert_provider_runtime_available",
                    side_effect=StopPlayback,
                ) as assert_runtime,
                patch.object(sys, "stdout", io.StringIO()),
            ):
                with self.assertRaises(StopPlayback):
                    play_main(["--model", str(model_path)])

            assert_runtime.assert_called_once()
            runtime_config = assert_runtime.call_args.args[0]
            self.assertEqual(runtime_config.env_provider, "supermariobrosnes-turbo")
            self.assertEqual(runtime_config.game, "SuperMarioBros-Nes-v0")

    def test_play_main_constructs_one_environment_for_policy_and_viewer(self) -> None:
        class FakeEnv:
            reset_infos = [{}]

            def seed(self, seed):
                self.last_seed = seed

            def reset(self):
                return np.zeros((1, 4, 84, 84), dtype=np.uint8)

            def get_images(self):
                return [np.zeros((210, 160, 3), dtype=np.uint8)]

            def close(self):
                self.closed = True

        class FakeViewer:
            def __init__(self, *args, **kwargs):
                del args, kwargs

            def show(self, frame, overlay=None):
                del frame, overlay
                return False

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "ppo_test_100_steps.zip"
            model_path.write_bytes(b"zip")
            write_model_metadata(
                model_path,
                argparse.Namespace(run_name="run", run_description="description"),
                EnvConfig(
                    env_provider="stable-retro-turbo",
                    game="Breakout-Atari2600-v0",
                    state="Start",
                ),
                kind="checkpoint",
            )
            fake_env = FakeEnv()

            with (
                patch("rlab.play.assert_provider_runtime_available"),
                patch("rlab.play.make_eval_vec_env", return_value=fake_env) as make_env,
                patch("rlab.play.PygameViewer", FakeViewer),
                patch("stable_baselines3.PPO.load", return_value=object()),
                patch.object(sys, "stdout", io.StringIO()),
            ):
                self.assertEqual(play_main(["--model", str(model_path)]), 0)

            make_env.assert_called_once()
            self.assertTrue(fake_env.closed)

    def test_obs_stack_render_has_no_label_band(self) -> None:
        frames = deque(
            [np.full((3, 2, 1), value, dtype=np.uint8) for value in (10, 20, 30, 40)],
            maxlen=4,
        )

        image = render_obs_stack(frames, scale=2)

        self.assertEqual(image.shape, (6, 16, 3))
        self.assertTrue(np.all(image[:, 0:4, :] == 10))
        self.assertTrue(np.all(image[:, 4:8, :] == 20))
        self.assertTrue(np.all(image[:, 8:12, :] == 30))
        self.assertTrue(np.all(image[:, 12:16, :] == 40))

    def test_obs_stack_viewer_draws_with_pygame_window(self) -> None:
        rendered = []

        class FakeSurface:
            def __init__(self):
                self.blit_calls = []

            def blit(self, source, position):
                self.blit_calls.append((source, position))

        class FakeWindow:
            def __init__(self, title, **kwargs):
                self.title = title
                self.size = kwargs["size"]
                self.position = kwargs.get("position")
                self.surface = FakeSurface()
                self.flips = 0
                self.destroyed = False

            def get_surface(self):
                return self.surface

            def flip(self):
                self.flips += 1

            def destroy(self):
                self.destroyed = True

        fake_pygame = types.SimpleNamespace(
            Window=FakeWindow,
            surfarray=types.SimpleNamespace(
                make_surface=lambda image: rendered.append(image.copy()) or image
            ),
        )
        frames = deque(
            [np.full((3, 2, 1), value, dtype=np.uint8) for value in (10, 20, 30, 40)],
            maxlen=4,
        )

        with patch("rlab.play.import_pygame", return_value=fake_pygame):
            viewer = ObsStackViewer(scale=2, position=(40, 240))
            self.assertTrue(viewer.show(frames))
            window = viewer.window
            viewer.close()

        self.assertEqual(window.title, "rlab obs framestack")
        self.assertEqual(window.size, (16, 6))
        self.assertEqual(window.position, (40, 240))
        self.assertEqual(window.flips, 1)
        self.assertEqual(window.surface.blit_calls[0][1], (0, 0))
        self.assertEqual(rendered[0].shape, (16, 6, 3))
        self.assertTrue(window.destroyed)

    def test_eval_defaults_to_stochastic(self) -> None:
        with patch("rlab.eval.os.cpu_count", return_value=12):
            parser = build_eval_parser()

        self.assertFalse(hasattr(parser.parse_args([]), "deterministic"))
        self.assertEqual(parser.parse_args([]).n_envs, 12)
        self.assertEqual(parser.parse_args([]).seed, DEFAULT_EVAL_SEED)
        self.assertEqual(parser.parse_args(["--n-envs", "5"]).n_envs, 5)
        help_text = parser.format_help()
        self.assertNotIn("--deterministic", help_text)
        self.assertIn("--n-envs", help_text)
        self.assertNotIn("--record-best-video", help_text)
        self.assertNotIn("--stochastic", help_text)
        self.assertNotIn("--no-stochastic", help_text)

    def test_eval_parser_omits_checkpoint_artifact_workflow_flags(self) -> None:
        parser = build_eval_parser()
        help_text = parser.format_help()

        self.assertIn("--model", help_text)
        self.assertIn("--hf-file", help_text)
        self.assertNotIn("--artifact", help_text)
        self.assertNotIn("--artifact-run", help_text)
        self.assertNotIn("--checkpoint-series", help_text)
        self.assertNotIn("--eval-dir", help_text)
        self.assertNotIn("--force", help_text)
        self.assertNotIn("--wandb-run-id", help_text)
        self.assertNotIn("--no-promote-best", help_text)

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

        self.assertFalse(hasattr(parser.parse_args([ref]), "deterministic"))

    def test_model_source_ref_uses_positional_artifact_ref(self) -> None:
        parser = build_play_parser()
        args = parser.parse_args(["tsilva/SuperMarioBros-NES/run-checkpoint:latest"])

        self.assertEqual(
            single_model_artifact_ref(args),
            "tsilva/SuperMarioBros-NES/run-checkpoint:latest",
        )

    def test_model_source_ref_uses_positional_run_path_latest_checkpoint(self) -> None:
        parser = build_play_parser()
        args = parser.parse_args(
            ["tsilva/SuperMarioBros3-Nes-v0/1Player.World1.Level1_base_s123_20260703T171520Z"]
        )

        self.assertEqual(
            single_model_artifact_ref(args),
            "tsilva/SuperMarioBros3-Nes-v0/"
            "1Player.World1.Level1_base_s123_20260703T171520Z-checkpoint:latest",
        )

    def test_model_source_ref_resolves_unique_bare_run_across_projects(self) -> None:
        found_ref = (
            "tsilva/ms_pacman/alepy__mspacman_episodic-life_s126_20260709T102223Z-checkpoint:latest"
        )
        calls = []

        class FakeApi:
            def artifact(self, ref, type=None):
                calls.append((ref, type))
                if ref == found_ref:
                    return object()
                raise RuntimeError("not found")

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        parser = build_play_parser()
        args = parser.parse_args(["alepy__mspacman_episodic-life_s126_20260709T102223Z"])

        with (
            patch.dict(sys.modules, {"wandb": fake_wandb}),
            patch(
                "rlab.model_sources.artifact_lookup_project_paths",
                return_value=[
                    "tsilva/ms_pacman",
                    "tsilva/breakout",
                    "tsilva/SuperMarioBros-Nes-v0",
                ],
            ),
        ):
            self.assertEqual(single_model_artifact_ref(args), found_ref)

        self.assertIn((found_ref, "model"), calls)

    def test_model_source_ref_rejects_ambiguous_bare_run_across_projects(self) -> None:
        matches = {
            "tsilva/ms_pacman/shared-run-checkpoint:latest",
            "tsilva/breakout/shared-run-checkpoint:latest",
        }

        class FakeApi:
            def artifact(self, ref, type=None):
                if ref in matches:
                    return object()
                raise RuntimeError("not found")

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        parser = build_play_parser()
        args = parser.parse_args(["shared-run"])

        with (
            patch.dict(sys.modules, {"wandb": fake_wandb}),
            patch(
                "rlab.model_sources.artifact_lookup_project_paths",
                return_value=[
                    "tsilva/ms_pacman",
                    "tsilva/breakout",
                    "tsilva/SuperMarioBros-Nes-v0",
                ],
            ),
        ):
            with self.assertRaisesRegex(SystemExit, "ambiguous"):
                single_model_artifact_ref(args)

    def test_artifact_lookup_projects_infer_ale_project_from_run_prefix(self) -> None:
        projects = artifact_lookup_project_paths(
            "tsilva/SuperMarioBros-Nes-v0",
            "alepy__mspacman_episodic-life_s126_20260709T102223Z",
        )

        self.assertEqual(projects[0], "tsilva/ms_pacman")

    def test_model_source_ref_uses_wandb_run_url_latest_checkpoint(self) -> None:
        calls = []

        class FakeRun:
            id = "7gjw67kl"
            name = "renamed-in-wandb-ui"
            config = {"run_name": "b82-b55reval-s6-20260702T150934Z"}

        class FakeApi:
            def run(self, path):
                calls.append(path)
                return FakeRun()

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        parser = build_play_parser()
        args = parser.parse_args(["https://wandb.ai/tsilva/SuperMarioBros-Nes-v0/runs/7gjw67kl"])

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            self.assertEqual(
                single_model_artifact_ref(args),
                "tsilva/SuperMarioBros-Nes-v0/b82-b55reval-s6-20260702T150934Z-checkpoint:latest",
            )

        self.assertEqual(calls, ["tsilva/SuperMarioBros-Nes-v0/7gjw67kl"])

    def test_model_source_ref_uses_checkpoint_logged_by_wandb_run(self) -> None:
        class FakeArtifact:
            type = "model"
            qualified_name = "tsilva/SuperMarioBros-Nes-v0/rlab-run-id-checkpoint:v8"
            aliases = ["step-4500000", "latest"]

        class FakeRun:
            id = "rlab-run-id"
            name = "level1-1-base-s1"
            config = {"run_name": "level1-1-base-s1"}

            def logged_artifacts(self):
                return [FakeArtifact()]

        fake_wandb = types.SimpleNamespace(
            Api=lambda: types.SimpleNamespace(run=lambda _path: FakeRun())
        )
        parser = build_play_parser()
        args = parser.parse_args(["https://wandb.ai/tsilva/SuperMarioBros-Nes-v0/runs/rlab-run-id"])

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            self.assertEqual(
                single_model_artifact_ref(args),
                "tsilva/SuperMarioBros-Nes-v0/rlab-run-id-checkpoint:latest",
            )

    def test_model_source_ref_resolves_bare_run_to_logged_run_artifact(self) -> None:
        class FakeArtifact:
            type = "model"
            qualified_name = "tsilva/SuperMarioBros-Nes-v0/rlab-run-id-checkpoint:v8"
            aliases = ["latest"]

        class FakeRun:
            name = "level1-1-base-s1"
            config = {"run_name": "level1-1-base-s1"}

            def logged_artifacts(self):
                return [FakeArtifact()]

        class FakeApi:
            def artifact(self, _ref, type=None):
                raise RuntimeError("not found")

            def runs(self, _project, filters=None):
                self.filters = filters
                return [FakeRun()]

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        parser = build_play_parser()
        args = parser.parse_args(["level1-1-base-s1"])

        with (
            patch.dict(sys.modules, {"wandb": fake_wandb}),
            patch(
                "rlab.model_sources.artifact_lookup_project_paths",
                return_value=["tsilva/SuperMarioBros-Nes-v0"],
            ),
        ):
            self.assertEqual(
                single_model_artifact_ref(args),
                "tsilva/SuperMarioBros-Nes-v0/rlab-run-id-checkpoint:latest",
            )

    def test_bare_run_resolves_promoted_checkpoint_before_moving_latest(self) -> None:
        class FakeArtifact:
            type = "model"
            qualified_name = "tsilva/SuperMarioBros-Nes-v0/rlab-run-id-checkpoint:v13"
            aliases = ["step-6500000"]
            metadata = {"checkpoint_step": 6500000}

        class FakeRun:
            name = "level1-1-base-s1"
            config = {"run_name": "level1-1-base-s1"}
            summary = {"leader/checkpoint/step": 6500000}

            def logged_artifacts(self):
                return [FakeArtifact()]

        class FakeApi:
            def artifact(self, _ref, type=None):
                return object()

            def runs(self, _project, filters=None):
                return [FakeRun()]

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        parser = build_play_parser()
        args = parser.parse_args(["level1-1-base-s1"])

        with (
            patch.dict(sys.modules, {"wandb": fake_wandb}),
            patch(
                "rlab.model_sources.artifact_lookup_project_paths",
                return_value=["tsilva/SuperMarioBros-Nes-v0"],
            ),
        ):
            self.assertEqual(
                single_model_artifact_ref(args),
                "tsilva/SuperMarioBros-Nes-v0/rlab-run-id-checkpoint:step-6500000",
            )

    def test_bare_run_blocks_while_promoted_artifact_is_pending(self) -> None:
        class FakeArtifact:
            type = "model"
            qualified_name = "tsilva/SuperMarioBros-Nes-v0/rlab-run-id-checkpoint:v6"
            aliases = ["latest", "step-3000000"]
            metadata = {"checkpoint_step": 3000000}

        class FakeRun:
            name = "level1-1-base-s1"
            config = {"run_name": "level1-1-base-s1"}
            summary = {"leader/checkpoint/step": 6500000}

            def logged_artifacts(self):
                return [FakeArtifact()]

        class FakeApi:
            def artifact(self, _ref, type=None):
                return object()

            def runs(self, _project, filters=None):
                return [FakeRun()]

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        parser = build_play_parser()
        args = parser.parse_args(["level1-1-base-s1"])

        with (
            patch.dict(sys.modules, {"wandb": fake_wandb}),
            patch(
                "rlab.model_sources.artifact_lookup_project_paths",
                return_value=["tsilva/SuperMarioBros-Nes-v0"],
            ),
        ):
            with self.assertRaisesRegex(SystemExit, "retry after artifact projection"):
                single_model_artifact_ref(args)

    def test_model_source_ref_uses_wandb_run_url_with_query_latest_checkpoint(self) -> None:
        calls = []

        class FakeRun:
            id = "qxhbhcms"
            name = "renamed-in-wandb-ui"
            config = {"run_name": "alepy__mspacman_episodic-life_s126_20260709T102223Z"}

        class FakeApi:
            def run(self, path):
                calls.append(path)
                return FakeRun()

        fake_wandb = types.SimpleNamespace(Api=lambda: FakeApi())
        parser = build_play_parser()
        args = parser.parse_args(
            ["https://wandb.ai/tsilva/ms_pacman/runs/qxhbhcms?nw=nwusertsilva"]
        )

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            self.assertEqual(
                single_model_artifact_ref(args),
                "tsilva/ms_pacman/"
                "alepy__mspacman_episodic-life_s126_20260709T102223Z-checkpoint:latest",
            )

        self.assertEqual(calls, ["tsilva/ms_pacman/qxhbhcms"])

    def test_model_source_ref_uses_wandb_run_url_display_name_fallback(self) -> None:
        class FakeRun:
            id = "7gjw67kl"
            name = "Run With Spaces"
            config = {}

        fake_wandb = types.SimpleNamespace(
            Api=lambda: types.SimpleNamespace(run=lambda _path: FakeRun())
        )
        parser = build_play_parser()
        args = parser.parse_args(["https://wandb.ai/tsilva/SuperMarioBros-Nes-v0/runs/7gjw67kl"])

        with patch.dict(sys.modules, {"wandb": fake_wandb}):
            self.assertEqual(
                model_source_ref(args),
                "tsilva/SuperMarioBros-Nes-v0/Run-With-Spaces-checkpoint:latest",
            )

    def test_parse_wandb_run_ref_accepts_project_run_path(self) -> None:
        ref = parse_wandb_run_ref(
            "SuperMarioBros-Nes-v0/runs/7gjw67kl",
            default_project="tsilva/DefaultProject",
        )

        self.assertIsNotNone(ref)
        self.assertEqual(ref.project_path, "tsilva/SuperMarioBros-Nes-v0")
        self.assertEqual(ref.run_id, "7gjw67kl")

    def test_parse_wandb_run_ref_accepts_schemeless_url_with_query(self) -> None:
        ref = parse_wandb_run_ref("wandb.ai/tsilva/ms_pacman/runs/qxhbhcms?nw=nwusertsilva")

        self.assertIsNotNone(ref)
        self.assertEqual(ref.project_path, "tsilva/ms_pacman")
        self.assertEqual(ref.run_id, "qxhbhcms")

    def test_model_source_ref_uses_env_entity_for_project_run_path(self) -> None:
        with patch.dict("os.environ", {"WANDB_ENTITY": "env-entity"}):
            parser = build_play_parser()
            args = parser.parse_args(["SuperMarioBros3-Nes-v0/run"])

        self.assertEqual(
            single_model_artifact_ref(args),
            "env-entity/SuperMarioBros3-Nes-v0/run-checkpoint:latest",
        )

    def test_model_source_ref_accepts_positional_huggingface_ref(self) -> None:
        parser = build_play_parser()
        args = parser.parse_args(["hf://tsilva/SuperMarioBros-NES_Level1-2"])

        self.assertEqual(
            single_huggingface_model_ref(args),
            "hf://tsilva/SuperMarioBros-NES_Level1-2",
        )
        self.assertEqual(model_source_ref(args), "hf://tsilva/SuperMarioBros-NES_Level1-2")
        self.assertIsNone(single_model_artifact_ref(args))

    def test_eval_model_source_ref_accepts_positional_huggingface_ref(self) -> None:
        parser = build_eval_parser()
        args = parser.parse_args(["hf://tsilva/SuperMarioBros-NES_Level1-1"])

        self.assertEqual(
            single_huggingface_model_ref(args),
            "hf://tsilva/SuperMarioBros-NES_Level1-1",
        )
        self.assertEqual(model_source_ref(args), "hf://tsilva/SuperMarioBros-NES_Level1-1")
        self.assertIsNone(single_model_artifact_ref(args))

    def test_eval_main_resolves_huggingface_model_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.zip"
            model_path.write_bytes(b"model")

            def fake_resolve(args):
                self.assertEqual(args.model_ref, "hf://tsilva/SuperMarioBros-NES_Level1-1")
                return ResolvedModelSource(
                    model_path=model_path,
                    artifact_name="hf://tsilva/SuperMarioBros-NES_Level1-1/model.zip",
                )

            def fake_apply(args, source, parser, parser_defaults, explicit_dests, **_kwargs):
                self.assertEqual(source.model_path, model_path)
                apply_config_defaults(
                    args,
                    {"game": "SuperMarioBros-Nes-v0", "state": "Level1-1"},
                    parser_defaults,
                    explicit_dests,
                )
                return True

            def fake_evaluate_model_episodes(**kwargs):
                self.assertEqual(kwargs["model"], "ppo")
                self.assertEqual(kwargs["episodes"], 1)
                self.assertEqual(kwargs["extra"]["model"], str(model_path))
                return (
                    {
                        "completion_rate": 1.0,
                        "episode_results": [{"reward": 1.0}],
                    },
                    None,
                )

            output = io.StringIO()
            with (
                patch("rlab.eval.resolve_single_model_source", side_effect=fake_resolve),
                patch("rlab.eval.apply_model_source_defaults", side_effect=fake_apply),
                patch("rlab.eval.assert_provider_runtime_available") as assert_runtime,
                patch("rlab.eval.PPO.load", return_value="ppo"),
                patch(
                    "rlab.eval.evaluate_model_episodes", side_effect=fake_evaluate_model_episodes
                ),
                patch.object(sys, "stdout", output),
            ):
                eval_main(
                    [
                        "hf://tsilva/SuperMarioBros-NES_Level1-1",
                        "--episodes",
                        "1",
                        "--summary-only",
                        "--no-progress",
                    ]
                )

            assert_runtime.assert_called_once()
            runtime_config = assert_runtime.call_args.args[0]
            self.assertEqual(runtime_config.game, "SuperMarioBros-Nes-v0")
            text = output.getvalue()
            self.assertIn("Downloading hf://tsilva/SuperMarioBros-NES_Level1-1", text)
            self.assertIn(f"Downloaded model: {model_path}", text)
            self.assertIn('"completion_rate": 1.0', text)

    def test_huggingface_model_ref_parses_model_url_and_file_url(self) -> None:
        self.assertEqual(
            parse_huggingface_model_ref("hf://tsilva/SuperMarioBros-NES_Level1-2"),
            ("tsilva/SuperMarioBros-NES_Level1-2", None, None),
        )
        self.assertEqual(
            parse_huggingface_model_ref(
                "https://huggingface.co/tsilva/SuperMarioBros-NES_Level1-2/resolve/main/model.zip"
            ),
            ("tsilva/SuperMarioBros-NES_Level1-2", "model.zip", "main"),
        )

    def test_huggingface_model_source_downloads_checkpoint_and_metadata_sidecar(self) -> None:
        test_case = self

        class FakeApi:
            def list_repo_files(self, *, repo_id, repo_type, revision):
                test_case.assertEqual(repo_id, "tsilva/SuperMarioBros-NES_Level1-2")
                test_case.assertEqual(repo_type, "model")
                test_case.assertEqual(revision, "main")
                return ["README.md", "model_metadata.json", "ppo_test_500_steps.zip"]

        def fake_hf_hub_download(*, repo_id, repo_type, revision, filename, local_dir):
            test_case.assertEqual(repo_id, "tsilva/SuperMarioBros-NES_Level1-2")
            test_case.assertEqual(repo_type, "model")
            test_case.assertEqual(revision, "main")
            path = Path(local_dir) / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            if filename.endswith(".json"):
                path.write_text('{"env_config": {"state": "Level1-2"}}\n', encoding="utf-8")
            else:
                path.write_bytes(b"zip")
            return str(path)

        fake_hub = types.SimpleNamespace(
            HfApi=lambda: FakeApi(),
            hf_hub_download=fake_hf_hub_download,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.dict(sys.modules, {"huggingface_hub": fake_hub}):
                source = download_huggingface_model_source(
                    "hf://tsilva/SuperMarioBros-NES_Level1-2",
                    root=Path(tmp_dir),
                )

            self.assertEqual(source.model_path.name, "ppo_test_500_steps.zip")
            self.assertEqual(
                source.artifact_name,
                "hf://tsilva/SuperMarioBros-NES_Level1-2/ppo_test_500_steps.zip",
            )
            self.assertIsNone(source.artifact_ref)
            self.assertEqual(source.checkpoint_step, 500)
            self.assertTrue(source.model_path.with_suffix(".metadata.json").is_file())

    def test_model_source_ref_uses_positional_run_name(self) -> None:
        parser = build_play_parser()
        args = parser.parse_args(["run"])

        self.assertEqual(
            single_model_artifact_ref(args),
            "tsilva/SuperMarioBros-Nes-v0/run-checkpoint:latest",
        )

    def test_model_source_ref_uses_positional_full_artifact_ref(self) -> None:
        parser = build_play_parser()
        args = parser.parse_args(["tsilva/SuperMarioBros-Nes-v0/run-best:v8"])

        self.assertEqual(
            single_model_artifact_ref(args),
            "tsilva/SuperMarioBros-Nes-v0/run-best:v8",
        )

    def test_default_wandb_project_path_uses_env_entity(self) -> None:
        with patch.dict("os.environ", {"WANDB_ENTITY": "env-entity"}):
            self.assertEqual(
                default_wandb_project_path("SuperMarioBros3-Nes-v0"),
                "env-entity/SuperMarioBros3-Nes-v0",
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
