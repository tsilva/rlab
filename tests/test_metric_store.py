from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from rlab.metric_names import EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN
from rlab.metric_store import MetricStore, file_sha256, metric_store_path

STAGES = [
    {
        "name": "screen",
        "episodes": 10,
        "n_envs": 2,
        "pass": [
            {"metric": EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN, "operator": ">=", "threshold": 1.0}
        ],
        "candidate_stop": False,
    },
    {
        "name": "confirm",
        "episodes": 30,
        "n_envs": 4,
        "pass": [
            {"metric": EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN, "operator": ">=", "threshold": 1.0}
        ],
        "candidate_stop": True,
    },
]


class MetricStoreTests(unittest.TestCase):
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
                    store.append_metrics({"train/reward": 1.0}, step=10, source="train")
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
            self.assertEqual(store.latest_metric("train/reward"), 1.0)

    def test_init_persists_wal_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()

            with store.connect() as conn:
                mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

            self.assertEqual(mode.lower(), "wal")

    def test_append_metrics_and_latest_lookup_keep_newest_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()

            self.assertEqual(
                store.append_metrics(
                    {
                        "train/reward": 1.0,
                        "ignored/text": "nope",
                        "ignored/bool": True,
                    },
                    step=10,
                    source="train",
                    created_at=1.0,
                ),
                1,
            )
            store.append_metrics({"train/reward": 2.5}, step=20, source="train", created_at=2.0)

            self.assertEqual(store.latest_metric("train/reward"), 2.5)
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

    def test_checkpoint_scoped_eval_metric_latest_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN: 0.5},
                step=100,
                source="eval",
                checkpoint_step=100,
            )
            store.append_metrics(
                {EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN: 1.0},
                step=200,
                source="eval",
                checkpoint_step=200,
            )

            self.assertEqual(store.latest_metric(EVAL_DONE_LEVEL_CHANGE_FROM_RATE_MIN), 1.0)

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
                metrics={"checkpoint_eval/screen/pass": 1.0},
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


if __name__ == "__main__":
    unittest.main()
