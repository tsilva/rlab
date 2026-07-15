from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from rlab.metric_names import EVAL_FULL_SUCCESS_RATE_MIN, TRAIN_REWARD_ROOT
from rlab.metric_store import MetricStore, file_sha256, metric_store_path

STAGES = [
    {
        "name": "screen",
        "episodes": 10,
        "n_envs": 2,
        "pass": [{"metric": EVAL_FULL_SUCCESS_RATE_MIN, "operator": ">=", "threshold": 1.0}],
        "candidate_stop": False,
    },
    {
        "name": "confirm",
        "episodes": 30,
        "n_envs": 4,
        "pass": [{"metric": EVAL_FULL_SUCCESS_RATE_MIN, "operator": ">=", "threshold": 1.0}],
        "candidate_stop": True,
    },
]


class MetricStoreTests(unittest.TestCase):
    def test_result_projection_uses_structured_metrics_and_artifact_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            checkpoint_id = store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=128,
                path=Path(tmp) / "model_128_steps.zip",
                metadata_path=None,
            )
            store.append_metrics(
                {"global_step": 128, "train/throughput/loop_fps": 1.25},
                step=128,
                source="learner",
                publish=False,
            )
            store.mark_artifact_uploaded(
                checkpoint_id,
                artifact_ref="entity/project/rlab-run-checkpoint:step-128",
                storage_uri="s3://bucket/rlab-run-checkpoint/model.zip",
            )

            result = store.result_projection()
            pending_frames = store.pending_metric_frames()

        self.assertEqual(result["metrics_json"]["global_step"], 128)
        self.assertEqual(result["metrics_json"]["train/throughput/loop_fps"], 1.25)
        self.assertEqual(pending_frames, [])
        self.assertEqual(
            result["artifact_refs"],
            [
                {
                    "name": "rlab-run-checkpoint",
                    "location": "s3://bucket/rlab-run-checkpoint/model.zip",
                    "artifact_ref": "entity/project/rlab-run-checkpoint:step-128",
                }
            ],
        )

    def test_rejects_unknown_scalar_metric_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            with self.assertRaisesRegex(ValueError, "unknown metric"):
                store.append_metrics(
                    {"train/reward/typo": 1.0},
                    step=1,
                    source="train",
                )

    def test_closed_mailbox_outbox_rejects_late_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {"train/throughput/loop_fps": 1.0},
                step=1,
                source="learner",
            )

            self.assertEqual(store.close_metric_outbox(), 1)
            with self.assertRaisesRegex(RuntimeError, "outbox is closed"):
                store.append_metrics(
                    {"train/throughput/loop_fps": 2.0},
                    step=2,
                    source="late_callback",
                )

    def test_mailbox_ack_deletes_only_delivered_frames_and_advances_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            for step in (10, 20):
                store.append_metrics(
                    {"train/throughput/loop_fps": float(step)},
                    step=step,
                    source="learner",
                )
            frames = store.pending_mailbox_frames()

            store.mark_metric_frames_delivered(
                [int(frames[0]["id"])], batch_sequence=1
            )

            remaining = store.pending_mailbox_frames()
            self.assertEqual([int(row["step"]) for row in remaining], [20])
            self.assertEqual(store.next_mailbox_batch_sequence(), 2)
            self.assertEqual(store.metric_outbox_stats()["frames"], 1)

    def test_writer_waits_for_bounded_concurrent_lock_instead_of_failing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rlab.sqlite"
            store = MetricStore(path, timeout=1.0)
            store.init()
            blocker = store.connect()
            blocker.execute("BEGIN IMMEDIATE")
            errors: list[BaseException] = []

            def write() -> None:
                try:
                    store.append_metrics(
                        {f"{TRAIN_REWARD_ROOT}/shaped/mean": 1.0},
                        step=10,
                        source="train",
                    )
                except BaseException as exc:  # pragma: no cover - asserted below
                    errors.append(exc)

            writer = threading.Thread(target=write)
            writer.start()
            time.sleep(0.15)
            blocker.commit()
            blocker.close()
            writer.join(timeout=2.0)

            self.assertFalse(writer.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(store.latest_metric(f"{TRAIN_REWARD_ROOT}/shaped/mean"), 1.0)

    def test_init_persists_wal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()

            with store.connect() as conn:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

            self.assertEqual(mode.lower(), "wal")

    def test_init_migrates_legacy_metric_observations_to_latest_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rlab.sqlite"
            with sqlite3.connect(path) as conn:
                conn.executescript(
                    """
                    CREATE TABLE metric_observations (
                      id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT NOT NULL,
                      value REAL NOT NULL,
                      step INTEGER,
                      source TEXT NOT NULL,
                      checkpoint_step INTEGER,
                      created_at REAL NOT NULL
                    );
                    INSERT INTO metric_observations
                      (name, value, step, source, checkpoint_step, created_at)
                    VALUES
                      ('train/episode/return/shaped/mean', 1.0, 10, 'train', NULL, 1.0),
                      ('train/episode/return/shaped/mean', 2.5, 20, 'train', NULL, 2.0);
                    """
                )

            store = MetricStore(path)
            store.init()

            self.assertEqual(store.latest_metric("train/episode/return/shaped/mean"), 2.5)
            with store.connect() as conn:
                legacy_table = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'metric_observations'"
                ).fetchone()
            self.assertIsNone(legacy_table)

    def test_append_metrics_and_latest_lookup_keep_newest_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()

            self.assertEqual(
                store.append_metrics(
                    {
                        f"{TRAIN_REWARD_ROOT}/shaped/mean": 1.0,
                        "ignored/text": "nope",
                        "ignored/bool": True,
                    },
                    step=10,
                    source="train",
                    created_at=1.0,
                ),
                1,
            )
            store.append_metrics(
                {f"{TRAIN_REWARD_ROOT}/shaped/mean": 2.5},
                step=20,
                source="train",
                created_at=2.0,
            )

            self.assertEqual(store.latest_metric(f"{TRAIN_REWARD_ROOT}/shaped/mean"), 2.5)
            self.assertIsNone(store.latest_metric("missing"))

    def test_checkpoint_row_creates_pending_artifact_and_eval_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            model_path = run_dir / "checkpoints" / "model_100_steps.zip"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"checkpoint")
            metadata_path = model_path.with_suffix(".metadata.json")
            metadata_path.write_text(json.dumps({"step": 100}), encoding="utf-8")
            store = MetricStore(metric_store_path(run_dir))
            store.init()

            checkpoint_id = store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=100,
                path=model_path,
                metadata_path=metadata_path,
                sha256=file_sha256(model_path),
            )

            pending_artifacts = store.pending_artifact_uploads()
            pending_evals = store.pending_evals()
            self.assertEqual(pending_artifacts[0]["id"], checkpoint_id)
            self.assertEqual(pending_evals[0]["id"], checkpoint_id)
            self.assertEqual(store.phase_counts()["artifacts:pending"], 1)
            self.assertEqual(store.phase_counts()["evals:pending"], 1)

    def test_final_checkpoint_does_not_enqueue_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            model_path = run_dir / "final_model.zip"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"final")
            store = MetricStore(metric_store_path(run_dir))
            store.init()

            store.record_checkpoint(
                run_name="run",
                kind="final",
                step=500,
                path=model_path,
                metadata_path=None,
                sha256=file_sha256(model_path),
            )

            self.assertEqual(len(store.pending_artifact_uploads()), 1)
            self.assertEqual(store.pending_evals(), [])

    def test_no_eval_checkpoint_still_enqueues_artifact_without_eval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            model_path = run_dir / "checkpoints" / "model_100_steps.zip"
            model_path.parent.mkdir(parents=True)
            model_path.write_bytes(b"checkpoint")
            store = MetricStore(metric_store_path(run_dir))
            store.init()

            store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=100,
                path=model_path,
                metadata_path=None,
                eval_required=False,
            )

            self.assertEqual(len(store.pending_artifact_uploads()), 1)
            self.assertEqual(store.pending_evals(), [])

    def test_checkpoint_scoped_eval_metric_latest_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {EVAL_FULL_SUCCESS_RATE_MIN: 0.5},
                step=100,
                source="eval",
            )
            store.append_metrics(
                {EVAL_FULL_SUCCESS_RATE_MIN: 1.0},
                step=200,
                source="eval",
            )

            self.assertEqual(store.latest_metric(EVAL_FULL_SUCCESS_RATE_MIN), 1.0)

    def test_staged_eval_skips_old_screens_but_preserves_confirm_priority(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            store = MetricStore(metric_store_path(run_dir))
            store.init()
            for step in (100, 200, 300):
                path = checkpoint_dir / f"model_{step}_steps.zip"
                path.write_bytes(f"checkpoint-{step}".encode())
                store.record_checkpoint(
                    run_name="run",
                    kind="checkpoint",
                    step=step,
                    path=path,
                    metadata_path=None,
                    sha256="sha",
                )

            store.ensure_checkpoint_eval_stages(STAGES)
            rows = store.pending_checkpoint_eval_stages()
            self.assertEqual(rows[0]["step"], 300)
            self.assertEqual(rows[0]["stage_name"], "screen")

            skipped = store.skip_stale_initial_checkpoint_eval_stages(
                keep_checkpoint_id=int(rows[0]["id"]),
            )
            self.assertEqual(skipped, 2)
            store.mark_checkpoint_eval_stage_succeeded(
                int(rows[0]["eval_stage_id"]),
                episodes=10,
                n_envs=2,
                metrics={"eval/screen/candidate/pass": 1.0},
            )
            store.enqueue_checkpoint_eval_stage(
                int(rows[0]["id"]),
                STAGES[1],
                stage_index=1,
            )
            newer_path = checkpoint_dir / "model_400_steps.zip"
            newer_path.write_bytes(b"checkpoint-400")
            store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=400,
                path=newer_path,
                metadata_path=None,
                sha256="sha",
            )
            store.ensure_checkpoint_eval_stages(STAGES)

            rows = store.pending_checkpoint_eval_stages(limit=2)
            self.assertEqual(rows[0]["step"], 300)
            self.assertEqual(rows[0]["stage_name"], "confirm")
            self.assertEqual(rows[1]["step"], 400)
            self.assertEqual(rows[1]["stage_name"], "screen")
            self.assertEqual(store.phase_counts()["evals:skipped_stale"], 2)
            self.assertEqual(store.phase_counts()["eval_stages:skipped_stale"], 2)

    def test_modal_stage_decision_queues_live_wandb_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            checkpoint_id = store.record_checkpoint(
                run_name="run",
                kind="checkpoint",
                step=500000,
                path=Path(tmp) / "checkpoint.zip",
                metadata_path=None,
                sha256="sha",
            )
            metrics = {
                "global_step": 500000.0,
                "eval/screen/outcome/success/rate/min": 1.0,
                "eval/screen/candidate/pass": 1.0,
            }

            store.apply_modal_eval_decision(
                checkpoint_id,
                stage_name="screen",
                stage_index=0,
                episodes=10,
                n_envs=2,
                metrics=metrics,
                passed=True,
                candidate_stop=False,
                publish=True,
            )

            frames = store.pending_metric_frames()
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0]["kind"], "history")
            self.assertEqual(frames[0]["source"], "modal_checkpoint_eval")
            self.assertEqual(json.loads(frames[0]["payload_json"]), metrics)


if __name__ == "__main__":
    unittest.main()
