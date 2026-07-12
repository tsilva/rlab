from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rlab.metric_store import MetricStore
from rlab.wandb_publisher import publish_pending_frames


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(dict(payload))


class WandbPublisherTests(unittest.TestCase):
    def test_publishes_batched_frame_without_overriding_wandb_internal_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {"rollout/ep_rew_mean": 12.5, "time/fps": 4000},
                step=2048,
                source="train",
            )
            run = FakeRun()

            count = publish_pending_frames(
                store,
                run,
                args=SimpleNamespace(),
                config=SimpleNamespace(),
                limit=100,
            )

            self.assertEqual(count, 1)
            self.assertEqual(len(run.logged), 1)
            self.assertEqual(run.logged[0]["global_step"], 2048.0)
            self.assertEqual(run.logged[0]["rollout/ep_rew_mean"], 12.5)
            self.assertEqual(store.phase_counts()["telemetry:published"], 1)
            self.assertEqual(store.telemetry_health()["published_step"], 2048)

    def test_interrupted_publish_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics({"time/fps": 1}, step=10, source="train")
            frame = store.pending_metric_frames()[0]
            self.assertTrue(store.claim_metric_frame(int(frame["id"])))

            self.assertEqual(store.reset_interrupted_metric_frames(), 1)
            self.assertEqual(store.pending_metric_frames()[0]["status"], "failed_retryable")


if __name__ == "__main__":
    unittest.main()
