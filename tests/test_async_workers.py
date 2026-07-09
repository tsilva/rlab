from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlab.artifact_worker import process_upload
from rlab.checkpoint_eval_worker import process_eval
from rlab.env import EnvConfig
from rlab.metric_names import EVAL_INFO_LEVEL_COMPLETE_RATE_MIN
from rlab.metric_store import MetricStore, metric_store_path


class FakeWandbRun:
    def __init__(self) -> None:
        self.finished = False

    def finish(self) -> None:
        self.finished = True


def worker_args(**overrides: object) -> argparse.Namespace:
    values: dict[str, object] = {
        "run_name": "run",
        "run_description": "test",
        "game": "SuperMarioBros-Nes-v0",
        "wandb": True,
        "no_wandb_artifacts": False,
        "wandb_entity": "entity",
        "wandb_project": "project",
        "wandb_mode": "offline",
        "device": "cpu",
        "hud_crop_top": 32,
        "post_train_eval_episodes": 100,
        "post_train_eval_max_steps": 0,
        "max_episode_steps": 4500,
        "post_train_eval_n_envs": 1,
        "post_train_eval_stochastic": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class AsyncWorkerTests(unittest.TestCase):
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
                patch("rlab.artifact_worker.resume_wandb_run", return_value=FakeWandbRun()),
                patch(
                    "rlab.artifact_worker.log_wandb_model_artifact",
                    side_effect=RuntimeError("wandb 503"),
                ),
            ):
                process_upload(
                    store=store,
                    args=worker_args(),
                    config=EnvConfig(),
                    run_dir=run_dir,
                    row=row,
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
                "reward_mean": 12.0,
                "reward_std": 1.0,
                "reward_max": 15.0,
                "best_episode": {"reward": 15.0},
                "eval/done/level_change/from_rate/min": 1.0,
                "eval/done/level_change/from_rate/mean": 1.0,
                EVAL_INFO_LEVEL_COMPLETE_RATE_MIN: 1.0,
                "eval/info/level_complete/rate/mean": 1.0,
            }

            with (
                patch("rlab.checkpoint_eval_worker.PPO.load", return_value=object()),
                patch("rlab.checkpoint_eval_worker.evaluate_model_episodes", return_value=(metrics, None)),
                patch("rlab.checkpoint_eval_worker.resume_wandb_run", return_value=FakeWandbRun()),
                patch("rlab.checkpoint_eval_worker.log_checkpoint_eval_metrics") as log_eval,
            ):
                process_eval(
                    store=store,
                    args=worker_args(),
                    config=EnvConfig(done_on_events=("level_change",)),
                    run_dir=run_dir,
                    row=row,
                )

            self.assertEqual(store.latest_metric(EVAL_INFO_LEVEL_COMPLETE_RATE_MIN), 1.0)
            self.assertEqual(store.phase_counts()["evals:succeeded"], 1)
            log_eval.assert_called_once()
            self.assertEqual(log_eval.call_args.kwargs["eval_source"], "async_worker")
            self.assertEqual(log_eval.call_args.kwargs["checkpoint_step_value"], 200)

            with store.connection() as conn:
                row = conn.execute(
                    "SELECT episodes, metrics_json FROM eval_results WHERE checkpoint_id = ?",
                    (checkpoint_id,),
                ).fetchone()
            self.assertEqual(row["episodes"], 100)
            self.assertIn(EVAL_INFO_LEVEL_COMPLETE_RATE_MIN, row["metrics_json"])


if __name__ == "__main__":
    unittest.main()
