from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab.fleet_wandb_publisher import (
    _canonical_goal_summary,
    _cursor_mapping,
    _partition_batches,
    _summary_step_max,
    drain_cycle_parallel,
    finalize_finishing_run,
    run_publisher_actor,
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
    def test_terminal_goal_summary_uses_the_promoted_acceptance_evidence(self) -> None:
        summary = _canonical_goal_summary(
            {
                "outcome": "accepted",
                "promotion_json": {
                    "checkpoint_step": 8_000_000,
                    "raw_metrics": {
                        "episodes": 100,
                        "_acceptance_duration_seconds": 33.75,
                        "_acceptance_aggregates": {
                            "episodes_planned": 100,
                            "episodes_completed": 100,
                            "failure_count": 0,
                        },
                        "eval/full/outcome/success/rate/min": 1.0,
                        "eval/full/outcome/success/rate/mean": 1.0,
                        "eval/full/episode/return/mean": 3144.17,
                    },
                },
            }
        )

        self.assertEqual(summary["eval/acceptance/pass"], 1.0)
        self.assertEqual(summary["eval/acceptance/episodes/planned"], 100)
        self.assertEqual(summary["eval/acceptance/episodes/completed"], 100)
        self.assertEqual(summary["eval/acceptance/failure/count"], 0)
        self.assertEqual(summary["eval/full/checkpoint/step"], 8_000_000)
        self.assertEqual(summary["eval/full/outcome/success/rate/min"], 1.0)

    def test_publisher_actor_survives_idle_gaps_until_remote_completion(self) -> None:
        conn = mock.MagicMock()
        with (
            mock.patch(
                "rlab.fleet_wandb_publisher.drain_once",
                side_effect=[2, 3],
            ) as drain,
            mock.patch(
                "rlab.fleet_wandb_publisher._publisher_actor_done",
                side_effect=[False, True],
            ),
            mock.patch("rlab.fleet_wandb_publisher.time.sleep") as sleep,
        ):
            published = run_publisher_actor(
                conn,
                41,
                limit=100,
                poll_seconds=2.0,
            )

        self.assertEqual(published, 5)
        self.assertEqual(drain.call_count, 2)
        self.assertEqual(
            drain.call_args_list[0].kwargs,
            {"limit": 100, "train_job_id": 41},
        )
        sleep.assert_called_once_with(2.0)
        statements = [
            call.args[0]
            for call in conn.cursor.return_value.__enter__.return_value.execute.call_args_list
        ]
        self.assertTrue(any("pg_advisory_lock" in statement for statement in statements))
        self.assertTrue(any("pg_advisory_unlock" in statement for statement in statements))

    def test_parallel_cycle_uses_one_isolated_process_per_run(self) -> None:
        with (
            mock.patch(
                "rlab.fleet_wandb_publisher.pending_metric_run_ids",
                return_value=[41, 42, 43],
            ),
            mock.patch(
                "rlab.fleet_wandb_publisher.subprocess.Popen",
            ) as popen,
        ):
            result = drain_cycle_parallel(
                mock.MagicMock(),
                repo_root=Path.cwd(),
                max_runs=3,
                limit=20,
            )

        self.assertEqual(
            result,
            {
                "runs_attempted": 3,
                "runs_started": 3,
                "batches_published": 0,
                "runs_failed": 0,
            },
        )
        self.assertEqual(popen.call_count, 3)
        train_ids = {
            call.args[0][call.args[0].index("--train-job-id") + 1] for call in popen.call_args_list
        }
        self.assertEqual(train_ids, {"41", "42", "43"})
        self.assertTrue(all(call.kwargs["start_new_session"] for call in popen.call_args_list))

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

    def test_finishing_run_completes_only_after_remote_cursors_metrics_and_artifact(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            {"acquired": True},
            {
                "id": 7,
                "train_config": {
                    "wandb_run_id": "run-7",
                    "game": "SuperMarioBros-Nes-v0",
                },
                "outcome": "accepted",
                "promotion_json": {"checkpoint_sha256": "a" * 64},
            },
        ]
        cursor.fetchall.return_value = [
            {
                "stream_id": "train-7",
                "final_sequence": 3,
                "published_sequence": 3,
            }
        ]
        cursor.rowcount = 1
        artifact = SimpleNamespace(
            aliases=["promoted"],
            metadata={"checkpoint_sha256": "a" * 64},
        )
        remote = SimpleNamespace(
            state="finished",
            summary={
                "_rlab_telemetry_cursors": {"train-7": 3},
                "eval/acceptance/pass": 1.0,
            },
            logged_artifacts=lambda: [artifact],
        )

        with mock.patch(
            "rlab.fleet_wandb_publisher._wandb_api_run",
            return_value=remote,
        ):
            self.assertTrue(finalize_finishing_run(conn, 7))

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        complete = [
            statement
            for statement in statements
            if "live_publication_status = 'complete'" in statement
        ]
        self.assertEqual(len(complete), 1)

    def test_wandb_failure_remains_retryable_finalization_work(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [
            {"acquired": True},
            {
                "id": 7,
                "train_config": {
                    "wandb_run_id": "run-7",
                    "game": "SuperMarioBros-Nes-v0",
                },
                "outcome": "accepted",
                "promotion_json": {"checkpoint_sha256": "a" * 64},
            },
        ]
        cursor.fetchall.return_value = [
            {
                "stream_id": "train-7",
                "final_sequence": 3,
                "published_sequence": 3,
            }
        ]

        with (
            mock.patch(
                "rlab.fleet_wandb_publisher._wandb_api_run",
                side_effect=RuntimeError("W&B unavailable"),
            ),
            self.assertRaisesRegex(RuntimeError, "W&B unavailable"),
        ):
            finalize_finishing_run(conn, 7)

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(
            any("live_publication_next_retry_at" in statement for statement in statements)
        )
        self.assertFalse(
            any("live_publication_status = 'complete'" in statement for statement in statements)
        )

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
