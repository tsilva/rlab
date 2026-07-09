from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlab.metric_names import EVAL_INFO_LEVEL_COMPLETE_RATE_MIN
from rlab.metric_store import MetricStore, file_sha256, metric_store_path


class MetricStoreTests(unittest.TestCase):
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
                {EVAL_INFO_LEVEL_COMPLETE_RATE_MIN: 0.5},
                step=100,
                source="eval",
                checkpoint_step=100,
            )
            store.append_metrics(
                {EVAL_INFO_LEVEL_COMPLETE_RATE_MIN: 1.0},
                step=200,
                source="eval",
                checkpoint_step=200,
            )

            self.assertEqual(store.latest_metric(EVAL_INFO_LEVEL_COMPLETE_RATE_MIN), 1.0)


if __name__ == "__main__":
    unittest.main()
