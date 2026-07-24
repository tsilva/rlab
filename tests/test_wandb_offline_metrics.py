from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import wandb

from rlab.metric_names import LEADER_CHECKPOINT_ARTIFACT_REF
from rlab.metric_store import MetricStore
from rlab.wandb_publisher import (
    _publish_frame,
    publish_pending_frames,
    publish_promotion_summary,
)
from rlab.wandb_utils import configure_wandb_metrics


class WandbOfflineMetricIntegrationTests(unittest.TestCase):
    def test_replayed_outbox_event_reuses_the_same_wandb_step(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.calls: list[tuple[dict, int]] = []

            def log(self, payload, *, step):
                self.calls.append((dict(payload), int(step)))

        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {"train/episode/return/shaped/mean": 5.0},
                step=100,
                source="train:rollout",
            )
            row = store.pending_metric_frames(limit=1)[0]
            run = FakeRun()

            _publish_frame(
                run,
                row,
                args=SimpleNamespace(metrics_schema_version=6),
            )
            _publish_frame(
                run,
                row,
                args=SimpleNamespace(metrics_schema_version=6),
            )

        self.assertEqual([step for _payload, step in run.calls], [row["id"], row["id"]])
        self.assertEqual(
            {payload["orchestration/event_id"] for payload, _step in run.calls},
            {row["event_id"]},
        )

    def test_supervisor_publishes_eval_metrics_table_and_promotion_without_artifacts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {
                    "eval/full/episode/return/mean": 5.0,
                    "eval/full/episode/return/best": 5.0,
                    "eval/full/outcome/success/rate/min": 1.0,
                    "eval/full/outcome/success/rate/mean": 1.0,
                },
                step=100,
                source="eval:modal",
            )
            store.enqueue_event(
                kind="eval_by_start",
                payload={
                    "rows": [
                        ["Start", 1, 1, 1.0, 5.0, 0.0, 5.0, "", 0, 0.0]
                    ]
                },
                step=100,
                source="eval:modal",
            )
            run = configure_wandb_metrics(
                wandb.init(
                    project="rlab-metrics-schema-test",
                    dir=tmp,
                    mode="offline",
                    reinit="finish_previous",
                    settings=wandb.Settings(silent=True, disable_git=True),
                )
            )
            assert run is not None
            published = publish_pending_frames(
                store,
                run,
                args=SimpleNamespace(metrics_schema_version=6),
                config=None,
                limit=10,
            )
            publish_promotion_summary(
                run,
                checkpoint_step=100,
                checkpoint_url="https://models.example/model.zip",
                metrics={
                    "eval/full/episode/return/mean": 5.0,
                    "eval/full/episode/return/best": 5.0,
                    "eval/full/outcome/success/rate/min": 1.0,
                    "eval/full/outcome/success/rate/mean": 1.0,
                },
                updated_at="2026-07-24T00:00:00Z",
            )

            self.assertEqual(published, 2)
            self.assertEqual(
                run.summary[LEADER_CHECKPOINT_ARTIFACT_REF],
                "https://models.example/model.zip",
            )
            self.assertEqual(store.metric_outbox_stats()["frames"], 0)
            run.finish()


if __name__ == "__main__":
    unittest.main()
