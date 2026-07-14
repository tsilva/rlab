from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlab.checkpoint_eval_worker import process_eval
from rlab.env import EnvConfig
from rlab.metric_names import (
    CHECKPOINT_EVAL_CANDIDATE_PASS,
    EVAL_FULL_DURATION_SECONDS,
    EVAL_FULL_SUCCESS_RATE_MIN,
)
from rlab.metric_store import MetricStore, metric_store_path
from rlab.wandb_publisher import process_upload


class FakeWandbRun:
    def __init__(self) -> None:
        self.finished = False
        self.logged: list[dict[str, object]] = []

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(dict(payload))

    def finish(self) -> None:
        self.finished = True


def worker_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "run_name": "run",
        "run_description": "test",
        "game": "SuperMarioBros-Nes-v0",
        "wandb": True,
        "wandb_run_id": "rlab-test-run",
        "no_wandb_artifacts": False,
        "wandb_entity": "entity",
        "wandb_project": "project",
        "wandb_mode": "offline",
        "wandb_artifact_storage_uri": "",
        "device": "cpu",
        "hud_crop_top": 32,
        "post_train_eval_episodes": 100,
        "post_train_eval_max_steps": 0,
        "max_episode_steps": 4500,
        "checkpoint_eval_n_envs": 1,
        "post_train_eval_stochastic": True,
        "selection_rank": [
            "max(eval/full/outcome/success/rate/min)",
            "max(eval/full/episode/return/mean)",
        ],
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def checkpoint_eval_stages() -> list[dict[str, object]]:
    return [
        {
            "name": "screen",
            "episodes": 10,
            "n_envs": 2,
            "pass": [
                {
                    "metric": EVAL_FULL_SUCCESS_RATE_MIN,
                    "operator": ">=",
                    "threshold": 1.0,
                }
            ],
        },
        {
            "name": "confirm",
            "episodes": 30,
            "n_envs": 4,
            "pass": [
                {
                    "metric": EVAL_FULL_SUCCESS_RATE_MIN,
                    "operator": ">=",
                    "threshold": 1.0,
                }
            ],
            "candidate_stop": True,
        },
    ]


def eval_metrics(*, episodes: int, completion: float) -> dict[str, object]:
    return {
        "episodes": episodes,
        "return_mean": 12.0,
        "return_std": 1.0,
        "return_median": 12.0,
        "eval/full/episode/return/best": 15.0,
        "episode_length_mean": 100.0,
        "best_episode": {"return": 15.0},
        "eval/full/episode/return/mean": 12.0,
        "eval/full/episode/return/std": 1.0,
        "eval/full/episode/return/median": 12.0,
        "eval/full/episode/length/mean": 100.0,
        "eval/full/episode/count": episodes,
        EVAL_FULL_SUCCESS_RATE_MIN: completion,
        "eval/full/outcome/success/rate/mean": completion,
        EVAL_FULL_DURATION_SECONDS: 12.5,
    }


class AsyncWorkerTests(unittest.TestCase):
    def test_publisher_uploads_to_object_storage_without_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint_path = run_dir / "checkpoints" / "model_100_steps.zip"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"checkpoint")
            store = MetricStore(metric_store_path(run_dir))
            store.init()
            checkpoint_id = store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=100,
                path=checkpoint_path,
                metadata_path=None,
                sha256="sha",
            )
            row = store.pending_artifact_uploads()[0]
            args = worker_args(
                wandb=False,
                wandb_artifact_storage_uri="s3://bucket/checkpoints",
            )

            with patch("rlab.artifacts.upload_s3_artifact") as upload:
                uploaded = process_upload(
                    store=store,
                    args=args,
                    config=EnvConfig(),
                    row=row,
                )

            expected_uri = (
                "s3://bucket/checkpoints/SuperMarioBros-Nes-v0/"
                "rlab-test-run-checkpoint/model_100_steps.zip"
            )
            self.assertTrue(uploaded)
            upload.assert_called_once_with(checkpoint_path, expected_uri)
            with store.connection() as conn:
                uploaded_row = conn.execute(
                    "SELECT status, artifact_ref, storage_uri FROM artifact_uploads "
                    "WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                ).fetchone()
            self.assertEqual(uploaded_row["status"], "uploaded")
            self.assertIsNone(uploaded_row["artifact_ref"])
            self.assertEqual(uploaded_row["storage_uri"], expected_uri)

    def test_artifact_upload_failure_becomes_retryable_worker_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint_path = run_dir / "checkpoints" / "model_100_steps.zip"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"checkpoint")
            store = MetricStore(metric_store_path(run_dir))
            store.init()
            checkpoint_id = store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=100,
                path=checkpoint_path,
                metadata_path=None,
                sha256="sha",
            )
            row = store.pending_artifact_uploads()[0]

            with (
                patch(
                    "rlab.wandb_publisher.log_wandb_model_artifact",
                    side_effect=RuntimeError("wandb 503"),
                ),
            ):
                process_upload(
                    store=store,
                    args=worker_args(),
                    config=EnvConfig(),
                    row=row,
                    wandb_run=FakeWandbRun(),
                )

            retry_rows = store.pending_artifact_uploads()
            self.assertEqual(retry_rows[0]["id"], checkpoint_id)
            self.assertEqual(retry_rows[0]["worker_status"], "failed_retryable")
            self.assertEqual(retry_rows[0]["attempts"], 1)
            self.assertIn("wandb 503", retry_rows[0]["last_error"])

    def test_eval_worker_writes_local_metric_store_and_mirrors_wandb_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint_path = run_dir / "checkpoints" / "model_200_steps.zip"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"checkpoint")
            store = MetricStore(metric_store_path(run_dir))
            store.init()
            checkpoint_id = store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=200,
                path=checkpoint_path,
                metadata_path=None,
                sha256="sha",
            )
            row = store.pending_evals()[0]
            metrics = {
                "episodes": 100,
                **eval_metrics(episodes=100, completion=1.0),
            }

            with (
                patch("rlab.checkpoint_eval_worker.PPO.load", return_value=object()),
                patch(
                    "rlab.checkpoint_eval_worker.evaluate_model_episodes",
                    return_value=(metrics, None),
                ),
            ):
                process_eval(
                    store=store,
                    args=worker_args(),
                    config=EnvConfig(task={"termination": {"success": ["level_change"]}}),
                    run_dir=run_dir,
                    row=row,
                )

            self.assertEqual(store.latest_metric(EVAL_FULL_SUCCESS_RATE_MIN), 1.0)
            self.assertEqual(store.latest_metric(EVAL_FULL_DURATION_SECONDS), 12.5)
            self.assertEqual(store.phase_counts()["evals:succeeded"], 1)
            frames = store.pending_metric_frames()
            self.assertEqual([frame["kind"] for frame in frames], ["checkpoint_eval"])
            self.assertEqual(frames[0]["step"], 200)

            with store.connection() as conn:
                row = conn.execute(
                    "SELECT episodes, metrics_json FROM eval_results WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                ).fetchone()
            self.assertEqual(row["episodes"], 100)
            self.assertIn(EVAL_FULL_SUCCESS_RATE_MIN, row["metrics_json"])

    def test_staged_eval_screen_fail_does_not_write_canonical_eval_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint_path = run_dir / "checkpoints" / "model_200_steps.zip"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"checkpoint")
            store = MetricStore(metric_store_path(run_dir))
            store.init()
            checkpoint_id = store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=200,
                path=checkpoint_path,
                metadata_path=None,
                sha256="sha",
            )
            stages = checkpoint_eval_stages()
            store.ensure_checkpoint_eval_stages(stages)
            row = store.pending_checkpoint_eval_stages()[0]
            with (
                patch("rlab.checkpoint_eval_worker.PPO.load", return_value=object()),
                patch(
                    "rlab.checkpoint_eval_worker.evaluate_model_episodes",
                    return_value=(eval_metrics(episodes=10, completion=0.9), None),
                ),
            ):
                process_eval(
                    store=store,
                    args=worker_args(checkpoint_eval_stages=stages),
                    config=EnvConfig(task={"termination": {"success": ["level_change"]}}),
                    run_dir=run_dir,
                    row=row,
                )

            self.assertIsNone(store.latest_metric(EVAL_FULL_SUCCESS_RATE_MIN))
            self.assertEqual(
                store.latest_metric("eval/screen/outcome/success/rate/min"),
                0.9,
            )
            self.assertEqual(store.latest_metric("eval/screen/candidate/pass"), 0.0)
            self.assertIsNone(store.latest_metric(CHECKPOINT_EVAL_CANDIDATE_PASS))
            self.assertEqual(store.phase_counts()["evals:non_candidate"], 1)
            self.assertEqual(store.phase_counts()["eval_stages:succeeded"], 1)
            self.assertEqual(store.pending_metric_frames()[0]["kind"], "history")
            with store.connection() as conn:
                result = conn.execute(
                    "SELECT metrics_json FROM eval_results WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                ).fetchone()
            self.assertNotIn(EVAL_FULL_SUCCESS_RATE_MIN, result["metrics_json"])

    def test_staged_eval_confirm_pass_emits_candidate_stop_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint_path = run_dir / "checkpoints" / "model_300_steps.zip"
            checkpoint_path.parent.mkdir(parents=True)
            checkpoint_path.write_bytes(b"checkpoint")
            store = MetricStore(metric_store_path(run_dir))
            store.init()
            checkpoint_id = store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=300,
                path=checkpoint_path,
                metadata_path=None,
                sha256="sha",
            )
            stages = checkpoint_eval_stages()
            store.ensure_checkpoint_eval_stages(stages)
            screen_row = store.pending_checkpoint_eval_stages()[0]

            with (
                patch("rlab.checkpoint_eval_worker.PPO.load", return_value=object()),
                patch(
                    "rlab.checkpoint_eval_worker.evaluate_model_episodes",
                    return_value=(eval_metrics(episodes=10, completion=1.0), None),
                ),
            ):
                process_eval(
                    store=store,
                    args=worker_args(checkpoint_eval_stages=stages),
                    config=EnvConfig(task={"termination": {"success": ["level_change"]}}),
                    run_dir=run_dir,
                    row=screen_row,
                )

            self.assertIsNone(store.latest_metric(CHECKPOINT_EVAL_CANDIDATE_PASS))
            confirm_row = store.pending_checkpoint_eval_stages()[0]
            self.assertEqual(confirm_row["stage_name"], "confirm")

            with (
                patch("rlab.checkpoint_eval_worker.PPO.load", return_value=object()),
                patch(
                    "rlab.checkpoint_eval_worker.evaluate_model_episodes",
                    return_value=(eval_metrics(episodes=30, completion=1.0), None),
                ),
            ):
                process_eval(
                    store=store,
                    args=worker_args(checkpoint_eval_stages=stages),
                    config=EnvConfig(task={"termination": {"success": ["level_change"]}}),
                    run_dir=run_dir,
                    row=confirm_row,
                )

            self.assertEqual(store.latest_metric("eval/confirm/candidate/pass"), 1.0)
            self.assertEqual(store.latest_metric(CHECKPOINT_EVAL_CANDIDATE_PASS), 1.0)
            self.assertIsNone(store.latest_metric(EVAL_FULL_SUCCESS_RATE_MIN))
            self.assertEqual(store.phase_counts()["evals:candidate"], 1)
            with store.connection() as conn:
                result = conn.execute(
                    "SELECT episodes, metrics_json FROM eval_results WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                ).fetchone()
            self.assertEqual(result["episodes"], 30)
            self.assertIn(CHECKPOINT_EVAL_CANDIDATE_PASS, result["metrics_json"])
            self.assertNotIn(EVAL_FULL_SUCCESS_RATE_MIN, result["metrics_json"])


if __name__ == "__main__":
    unittest.main()
