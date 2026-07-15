from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab.fleet_wandb_publisher import (
    _cursor_mapping,
    _partition_batches,
    _summary_step_max,
)
from rlab.metric_store import MetricStore
from rlab.wandb_publisher import project_payload_to_run, publish_pending_frames
from rlab import wandb_publisher


class FakeRun:
    def __init__(self) -> None:
        self.logged: list[dict[str, object]] = []

    def log(self, payload: dict[str, object]) -> None:
        self.logged.append(dict(payload))


class FakeHtml:
    def __init__(self, data, **kwargs) -> None:
        self.data = data
        self.kwargs = kwargs


class WandbPublisherTests(unittest.TestCase):
    def test_resumed_projection_can_flush_without_finishing_active_run(self) -> None:
        captured: dict[str, object] = {}
        fake_run = SimpleNamespace(define_metric=lambda *_args, **_kwargs: None)

        def init(**kwargs):
            captured.update(kwargs)
            return fake_run

        fake_wandb = SimpleNamespace(
            init=init,
            Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        with (
            mock.patch.dict("sys.modules", {"wandb": fake_wandb}),
            mock.patch.object(wandb_publisher, "load_wandb_env"),
            mock.patch.object(
                wandb_publisher,
                "resolve_wandb_namespace",
                return_value=("entity", "project"),
            ),
            mock.patch.object(
                wandb_publisher,
                "configure_wandb_metrics",
                side_effect=lambda run: run,
            ),
        ):
            projector = wandb_publisher.WandbProjector.resume(
                {"wandb_run_id": "rlab-test", "game": "game"},
                update_finish_state=False,
            )

        self.assertIs(projector.run, fake_run)
        self.assertFalse(captured["settings"].x_update_finish_state)

    def test_wandb_nested_summary_mapping_is_accepted(self) -> None:
        class ItemsOnly:
            def items(self):
                return (("train-7", 3),)

        self.assertEqual(
            _cursor_mapping(ItemsOnly()),
            {"train-7": 3},
        )

    def test_wandb_nested_step_summary_mapping_is_accepted(self) -> None:
        class ItemsOnly:
            def items(self):
                return (("max", 4196496),)

        self.assertEqual(_summary_step_max(ItemsOnly()), 4196496.0)
        self.assertEqual(_summary_step_max({"max": 2000000}), 2000000.0)
        self.assertIsNone(_summary_step_max({}))

    def test_mailbox_batches_have_distinct_confirmed_submitted_and_new_states(self) -> None:
        batches = [
            {"id": 1, "stream_id": "train-7", "batch_sequence": 1, "submitted_sequence": 2},
            {"id": 2, "stream_id": "train-7", "batch_sequence": 2, "submitted_sequence": 2},
            {"id": 3, "stream_id": "train-7", "batch_sequence": 3, "submitted_sequence": 2},
        ]

        confirmed, awaiting, unpublished = _partition_batches(
            batches,
            {"train-7": 1},
        )

        self.assertEqual([row["id"] for row in confirmed], [1])
        self.assertEqual([row["id"] for row in awaiting], [2])
        self.assertEqual([row["id"] for row in unpublished], [3])

    def test_late_evaluations_keep_their_checkpoint_steps_without_internal_step(self) -> None:
        run = FakeRun()
        run.log({"global_step": 300, "train/throughput/loop_fps": 10.0})
        with mock.patch.dict("sys.modules", {"wandb": SimpleNamespace()}):
            for checkpoint_step in (100, 200):
                project_payload_to_run(
                    run,
                    {
                        "train_config": {"wandb_run_id": "rlab-test"},
                        "purpose": "screen",
                        "checkpoint_uri": f"s3://bucket/{checkpoint_step}/model.zip",
                        "checkpoint_step": checkpoint_step,
                        "decision": {
                            "metrics": {
                                "global_step": checkpoint_step,
                                "eval/screen/checkpoint/step": checkpoint_step,
                                "eval/screen/outcome/success/rate/min": 1.0,
                            }
                        },
                    },
                )

        self.assertEqual([row["global_step"] for row in run.logged], [300, 100, 200])

    def test_publishes_batched_frame_without_overriding_wandb_internal_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {"train/episode/return/shaped/mean": 12.5},
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
            self.assertEqual(run.logged[0]["train/episode/return/shaped/mean"], 12.5)
            self.assertEqual(store.phase_counts()["telemetry:published"], 1)
            self.assertEqual(store.telemetry_health()["published_step"], 2048)

    def test_interrupted_publish_is_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.append_metrics(
                {"train/throughput/loop_fps": 1},
                step=10,
                source="train",
            )
            frame = store.pending_metric_frames()[0]
            self.assertTrue(store.claim_metric_frame(int(frame["id"])))

            self.assertEqual(store.reset_interrupted_metric_frames(), 1)
            self.assertEqual(store.pending_metric_frames()[0]["status"], "failed_retryable")

    def test_preview_failure_never_blocks_scalars_and_is_abandoned_after_two_attempts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.enqueue_event(
                kind="checkpoint_preview",
                payload={
                    "url": "https://preview.example/checkpoint.mp4",
                    "checkpoint_step": 10,
                    "lane_count": 2,
                    "duration_seconds": 30,
                },
                step=10,
                source="modal_checkpoint_eval",
            )
            store.append_metrics(
                {"train/throughput/loop_fps": 7.0},
                step=11,
                source="train",
            )
            self.assertEqual(store.pending_metric_frames()[0]["kind"], "history")

            class FailingPreviewRun(FakeRun):
                def log(self, payload: dict[str, object]) -> None:
                    if "eval/screen/preview" in payload:
                        raise RuntimeError("HTML unavailable")
                    super().log(payload)

            run = FailingPreviewRun()
            fake_wandb = SimpleNamespace(Html=FakeHtml)
            with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
                self.assertEqual(
                    publish_pending_frames(
                        store,
                        run,
                        args=SimpleNamespace(),
                        config=SimpleNamespace(),
                        limit=100,
                    ),
                    1,
                )
                self.assertEqual(run.logged[0]["global_step"], 11.0)
                self.assertEqual(store.pending_metric_frames()[0]["kind"], "checkpoint_preview")
                self.assertEqual(
                    publish_pending_frames(
                        store,
                        run,
                        args=SimpleNamespace(),
                        config=SimpleNamespace(),
                        limit=100,
                    ),
                    0,
                )

            self.assertEqual(store.pending_metric_frames(), [])
            self.assertEqual(store.phase_counts()["telemetry:failed_terminal"], 1)

    def test_preview_logs_external_html_at_checkpoint_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MetricStore(Path(tmp) / "rlab.sqlite")
            store.init()
            store.enqueue_event(
                kind="checkpoint_preview",
                payload={
                    "url": "https://preview.example/checkpoint.mp4",
                    "checkpoint_step": 500000,
                    "passed": True,
                    "lane_count": 2,
                    "duration_seconds": 30,
                },
                step=500000,
                source="modal_checkpoint_eval",
            )
            run = FakeRun()
            with mock.patch.dict("sys.modules", {"wandb": SimpleNamespace(Html=FakeHtml)}):
                count = publish_pending_frames(
                    store,
                    run,
                    args=SimpleNamespace(),
                    config=SimpleNamespace(),
                    limit=100,
                )

        self.assertEqual(count, 1)
        self.assertEqual(run.logged[0]["global_step"], 500000)
        media = run.logged[0]["eval/screen/preview"]
        self.assertIn("https://preview.example/checkpoint.mp4", media.data)
        self.assertEqual(media.kwargs["inject"], False)
        self.assertEqual(media.kwargs["data_is_not_path"], True)


if __name__ == "__main__":
    unittest.main()
