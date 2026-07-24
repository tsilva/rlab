from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

from rlab.metric_store import MetricStore, metric_store_path
from rlab.telemetry_mailbox import (
    MailboxProtocolError,
    WorkerMailbox,
    claim_run_metric_batches,
    commit_published_batches,
    decode_metric_batch,
    encode_metric_batch,
    mark_submitted_batches,
    pending_metric_run_ids,
    release_metric_batch_claims_by_owner,
    schedule_artifact_publications,
    discard_disabled_metric_batches,
)
from rlab.telemetry_relay import CommandRelay, main as relay_main


def frame(frame_id: int, *, step: int | None = None, payload: dict | None = None) -> dict:
    step = frame_id if step is None else step
    return {
        "id": frame_id,
        "event_id": f"event-{frame_id}",
        "step": step,
        "source": "learner",
        "kind": "history",
        "payload_json": json.dumps(
            payload
            or {
                "global_step": step,
                "train/throughput/loop_fps": float(frame_id),
            }
        ),
        "created_at": float(frame_id),
    }


class FakeMailbox:
    attempt_id = "train-7"

    def __init__(self) -> None:
        self.preflighted = False
        self.batches: list[dict] = []
        self.events: list[tuple[str, dict, str | None]] = []
        self.acknowledged_commands: list[str] = []

    def preflight(self) -> None:
        self.preflighted = True

    def submit_batch(self, *, batch_sequence, batch, final):
        self.batches.append(
            {
                "sequence": batch_sequence,
                "frames": decode_metric_batch(batch.payload),
                "final": final,
            }
        )
        return {"accepted_sequence": batch_sequence, "commands": []}

    def append_event(self, event_type, payload, *, event_id=None):
        self.events.append((event_type, dict(payload), event_id))
        return event_id or "event"

    def acknowledge_command(self, command_id):
        self.acknowledged_commands.append(command_id)
        return True


class TelemetryBatchTests(unittest.TestCase):
    @staticmethod
    def _artifact_candidate(*, wandb: bool = True) -> dict:
        return {
            "train_job_id": 7,
            "ledger_id": 3,
            "artifact_kind": "checkpoint",
            "checkpoint_step": 300,
            "checkpoint_sha256": "2" * 64,
            "checkpoint_uri": "s3://bucket/model.zip",
            "metadata_uri": "s3://bucket/metadata.json",
            "metadata_sha256": "3" * 64,
            "recipe_uri": "s3://bucket/recipe.json",
            "recipe_sha256": "4" * 64,
            "announcement_sha256": "1" * 64,
            "verified_at": datetime.now(UTC),
            "train_config": {"wandb": wandb, "game": "Bandit-v0"},
            "wandb_run_id": "rlab-7",
            "run_name": "run-7",
            "wandb_group": "batch",
            "wandb_tags": ["test"],
            "publication_role": "availability",
            "publication_revision": 0,
        }

    def test_verified_artifact_is_scheduled_as_a_durable_projection_stream(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.side_effect = [{"acquired": True}, {"id": 99}]
        cursor.fetchall.side_effect = [[self._artifact_candidate()], []]
        cursor.rowcount = 1

        result = schedule_artifact_publications(conn)

        self.assertEqual(result, {"scheduled": 1, "opted_out": 0})
        calls = cursor.execute.call_args_list
        stream_insert = next(call for call in calls if "INSERT INTO metric_streams" in call.args[0])
        self.assertEqual(
            stream_insert.args[1]["stream_id"],
            "artifact-v3-7-3-availability-r0",
        )
        batch_insert = next(call for call in calls if "INSERT INTO metric_batches" in call.args[0])
        frames = decode_metric_batch(bytes(batch_insert.args[1]["payload"].adapted))
        self.assertEqual(frames[0]["payload"]["artifact_publication_schema"], "v3")
        self.assertEqual(frames[0]["payload"]["content_mode"], "wandb_native_v1")
        self.assertEqual(frames[0]["payload"]["ledger_id"], 3)

    def test_legacy_null_wandb_column_preserves_immutable_config_run_id(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        candidate = self._artifact_candidate()
        candidate["wandb_run_id"] = None
        candidate["train_config"]["wandb_run_id"] = "rlab-config-id"
        cursor.fetchone.side_effect = [{"acquired": True}, {"id": 99}]
        cursor.fetchall.side_effect = [[candidate], []]

        self.assertEqual(
            schedule_artifact_publications(conn),
            {"scheduled": 1, "opted_out": 0},
        )

        batch_insert = next(
            call
            for call in cursor.execute.call_args_list
            if "INSERT INTO metric_batches" in call.args[0]
        )
        frames = decode_metric_batch(bytes(batch_insert.args[1]["payload"].adapted))
        self.assertEqual(
            frames[0]["payload"]["train_config"]["wandb_run_id"],
            "rlab-config-id",
        )

    def test_wandb_artifact_opt_out_creates_receipt_without_a_projection(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {"acquired": True}
        cursor.fetchall.side_effect = [[self._artifact_candidate(wandb=False)], []]
        cursor.rowcount = 1

        result = schedule_artifact_publications(conn)

        self.assertEqual(result, {"scheduled": 0, "opted_out": 1})
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(any("'opted_out'" in statement for statement in statements))
        self.assertFalse(any("INSERT INTO metric_batches" in statement for statement in statements))

    def test_terminal_publications_are_not_claimable(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = None

        self.assertIsNone(claim_run_metric_batches(conn, owner="publisher"))

        statement = cursor.execute.call_args.args[0]
        self.assertIn(
            "live_publication_status NOT IN ('complete', 'disabled', 'failed')",
            statement,
        )
        self.assertIn("'finalization_failed'", statement)
        self.assertIn("s.stream_id LIKE 'artifact-v2-%%'", statement)
        self.assertIn(
            "t.telemetry_protocol_version = 1\n"
            "                OR s.stream_id LIKE 'artifact-v3-%%'",
            statement,
        )

    def test_finishing_publishers_wait_until_their_retry_deadline(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []

        self.assertEqual(pending_metric_run_ids(conn), [])

        statement = cursor.execute.call_args.args[0]
        self.assertEqual(statement.count("live_publication_next_retry_at <= now()"), 3)
        self.assertIn(
            "live_publication_status NOT IN ('complete', 'disabled', 'failed')",
            statement,
        )
        self.assertIn("t.status = 'finalizing'", statement)
        self.assertIn("checkpoint_eval_backend", statement)
        self.assertIn("<> 'modal'", statement)
        self.assertIn("s.stream_id LIKE 'artifact-v2-%%'", statement)
        self.assertIn(
            "t.telemetry_protocol_version = 1\n"
            "                  OR s.stream_id LIKE 'artifact-v3-%%'",
            statement,
        )
        self.assertIn(
            "t.status IN ('succeeded', 'failed', 'canceled', 'finalization_failed')",
            statement,
        )

    def test_worker_preflight_checks_the_command_poll_procedure(self) -> None:
        mailbox = WorkerMailbox("postgresql://worker/db", "train-7", "token")
        conn = mock.MagicMock()
        conn.cursor.return_value.__enter__.return_value.fetchone.return_value = {"ready": 1}

        with (
            mock.patch("rlab.telemetry_mailbox.mailbox_connect", return_value=conn),
            mock.patch.object(mailbox, "poll_commands", return_value=[]) as poll,
            mock.patch.object(mailbox, "append_event") as append_event,
        ):
            mailbox.preflight()

        poll.assert_called_once_with()
        append_event.assert_called_once()

    def test_remote_confirmation_is_rechecked_after_five_seconds(self) -> None:
        conn = mock.MagicMock()

        mark_submitted_batches(
            conn,
            [{"id": 1, "stream_id": "train-7", "batch_sequence": 3}],
        )
        calls = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
        lease_update = next(call for call in calls if "lease_expires_at" in call.args[0])
        self.assertEqual(lease_update.args[1]["delay"], 5.0)
        self.assertFalse(lease_update.args[1]["refresh_submitted_at"])
        self.assertIn(
            "ELSE COALESCE(submitted_at, clock_timestamp()) END",
            lease_update.args[0],
        )

    def test_wandb_confirmation_retains_canonical_source(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value

        commit_published_batches(
            conn,
            [{"id": 1, "stream_id": "train-7", "batch_sequence": 3}],
        )

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(any("wandb_confirmed_at" in sql for sql in statements))
        self.assertFalse(any("DELETE FROM metric_batches" in sql for sql in statements))
        self.assertFalse(any("published_sequence = GREATEST" in sql for sql in statements))

    def test_wandb_opt_out_retains_source_for_archive(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 4

        self.assertEqual(discard_disabled_metric_batches(conn), 4)

        statement = cursor.execute.call_args.args[0]
        self.assertIn("UPDATE metric_batches", statement)
        self.assertIn("retained for canonical archive", statement)
        self.assertNotIn("DELETE FROM metric_batches", statement)

    def test_stalled_actor_claim_release_is_scoped_by_run_and_owner(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 4

        released = release_metric_batch_claims_by_owner(
            conn,
            train_job_id=7,
            owner="actor-7-session",
            error="watchdog timeout",
        )

        self.assertEqual(released, 4)
        statement, params = cursor.execute.call_args.args
        self.assertIn("a.train_job_id = %(train_job_id)s", statement)
        self.assertIn("b.lease_owner = %(owner)s", statement)
        self.assertEqual(params["train_job_id"], 7)
        self.assertEqual(params["owner"], "actor-7-session")

    def test_cursor_reassertion_refreshes_confirmation_window(self) -> None:
        conn = mock.MagicMock()

        mark_submitted_batches(
            conn,
            [{"id": 1, "stream_id": "train-7", "batch_sequence": 39}],
            refresh_submitted_at=True,
        )

        calls = conn.cursor.return_value.__enter__.return_value.execute.call_args_list
        lease_update = next(call for call in calls if "lease_expires_at" in call.args[0])
        self.assertTrue(lease_update.args[1]["refresh_submitted_at"])
        self.assertIn("CASE WHEN %(refresh_submitted_at)s", lease_update.args[0])

    def test_batch_is_deterministic_gzip_and_preserves_explicit_global_steps(self) -> None:
        batch = encode_metric_batch([frame(1, step=300), frame(2, step=100), frame(3, step=200)])
        again = encode_metric_batch([frame(1, step=300), frame(2, step=100), frame(3, step=200)])

        self.assertEqual(batch.payload, again.payload)
        self.assertEqual(
            [item["payload"]["global_step"] for item in decode_metric_batch(batch.payload)],
            [300, 100, 200],
        )

    def test_batch_splits_at_frame_limit(self) -> None:
        batch = encode_metric_batch([frame(index) for index in range(1, 1002)])
        self.assertEqual(batch.frame_count, 1000)
        self.assertEqual(batch.frame_ids[0], 1)
        self.assertEqual(batch.frame_ids[-1], 1000)

    def test_oversized_single_frame_is_rejected(self) -> None:
        oversized = frame(
            1,
            payload={
                "global_step": 1,
                "train/throughput/loop_fps": 1.0,
                "padding": "x" * 1_100_000,
            },
        )
        with self.assertRaises((MailboxProtocolError, ValueError)):
            encode_metric_batch([oversized])

    def test_training_completion_immediately_flushes_last_frame_and_watermark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            stop_file = Path(tmp) / "publisher.stop"
            stop_file.write_text("stop\n", encoding="utf-8")
            store = MetricStore(metric_store_path(run_dir))
            store.init()
            store.append_metrics(
                {"train/throughput/loop_fps": 12.0},
                step=300,
                source="learner",
            )
            mailbox = FakeMailbox()

            with mock.patch("rlab.telemetry_relay.WorkerMailbox.from_env", return_value=mailbox):
                result = relay_main(
                    [
                        "--run-dir",
                        str(run_dir),
                        "--stop-file",
                        str(stop_file),
                        "--final-flush-seconds",
                        "1",
                    ]
                )

            self.assertEqual(result, 0)
            self.assertTrue(mailbox.preflighted)
            self.assertEqual(len(mailbox.batches), 1)
            self.assertTrue(mailbox.batches[0]["final"])
            self.assertEqual(mailbox.batches[0]["frames"][0]["payload"]["global_step"], 300)
            self.assertEqual(mailbox.events[0][0], "metric_stream_closed")
            self.assertEqual(mailbox.events[0][1]["final_sequence"], 1)
            self.assertEqual(store.pending_mailbox_frames(), [])

    def test_empty_run_still_sends_final_watermark_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            stop_file = Path(tmp) / "publisher.stop"
            stop_file.write_text("stop\n", encoding="utf-8")
            MetricStore(metric_store_path(run_dir)).init()
            mailbox = FakeMailbox()

            with mock.patch("rlab.telemetry_relay.WorkerMailbox.from_env", return_value=mailbox):
                relay_main(["--run-dir", str(run_dir), "--stop-file", str(stop_file)])

            self.assertEqual(mailbox.batches[0]["frames"], [])
            self.assertTrue(mailbox.batches[0]["final"])

    def test_command_relay_uses_per_command_files_and_acks_only_signal_receipts(self) -> None:
        class CommandMailbox(FakeMailbox):
            def __init__(self) -> None:
                super().__init__()
                self.delivered: list[str] = []
                self.acknowledged: list[tuple[str, str | None]] = []

            def poll_commands(self):
                return [
                    {
                        "command_id": "stop-1",
                        "command_type": "stop",
                        "payload": {},
                    },
                    {
                        "command_id": "stop-2",
                        "command_type": "stop",
                        "payload": {},
                    },
                ]

            def mark_command_delivered(self, command_id):
                self.delivered.append(command_id)
                return True

            def acknowledge_command(self, command_id, *, acknowledged_at=None):
                self.acknowledged.append((command_id, acknowledged_at))
                return True

        with tempfile.TemporaryDirectory() as tmp:
            inbox = Path(tmp) / "inbox"
            receipts = Path(tmp) / "receipts"
            mailbox = CommandMailbox()
            relay = CommandRelay(
                mailbox,
                inbox_dir=inbox,
                receipt_dir=receipts,
                poll_seconds=1.0,
            )

            relay._deliver()

            inbox_entries = sorted(inbox.glob("*.json"))
            self.assertEqual(len(inbox_entries), 2)
            self.assertNotEqual(inbox_entries[0].name, inbox_entries[1].name)
            self.assertEqual(mailbox.delivered, ["stop-1", "stop-2"])
            self.assertEqual(mailbox.acknowledged, [])

            receipts.mkdir(parents=True)
            receipt = receipts / inbox_entries[0].name
            command_id = json.loads(inbox_entries[0].read_text())["command_id"]
            receipt.write_text(
                json.dumps(
                    {
                        "command_id": command_id,
                        "signal_sent_at": "2026-07-16T14:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            relay._ack_receipts()

            self.assertEqual(
                mailbox.acknowledged,
                [(command_id, "2026-07-16T14:00:00+00:00")],
            )
            self.assertFalse(receipt.exists())

    def test_persistent_command_poll_failure_stops_the_relay(self) -> None:
        class BrokenCommandMailbox(FakeMailbox):
            def poll_commands(self):
                raise RuntimeError("command procedure unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            MetricStore(metric_store_path(run_dir)).init()
            mailbox = BrokenCommandMailbox()

            with mock.patch("rlab.telemetry_relay.WorkerMailbox.from_env", return_value=mailbox):
                with self.assertRaisesRegex(RuntimeError, "dedicated command relay failed"):
                    relay_main(
                        [
                            "--run-dir",
                            str(run_dir),
                            "--stop-file",
                            str(Path(tmp) / "publisher.stop"),
                            "--command-poll-seconds",
                            "0.01",
                        ]
                    )

    def test_final_flush_retries_without_deleting_sqlite_before_ack(self) -> None:
        class FlakyMailbox(FakeMailbox):
            def __init__(self) -> None:
                super().__init__()
                self.calls = 0

            def submit_batch(self, *, batch_sequence, batch, final):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary Neon outage")
                return super().submit_batch(
                    batch_sequence=batch_sequence,
                    batch=batch,
                    final=final,
                )

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            stop_file = Path(tmp) / "publisher.stop"
            stop_file.write_text("stop\n", encoding="utf-8")
            store = MetricStore(metric_store_path(run_dir))
            store.init()
            store.append_metrics(
                {"train/throughput/loop_fps": 9.0},
                step=9,
                source="learner",
            )
            mailbox = FlakyMailbox()

            with mock.patch("rlab.telemetry_relay.WorkerMailbox.from_env", return_value=mailbox):
                relay_main(
                    [
                        "--run-dir",
                        str(run_dir),
                        "--stop-file",
                        str(stop_file),
                        "--final-flush-seconds",
                        "2",
                    ]
                )

            self.assertEqual(mailbox.calls, 2)
            self.assertEqual(store.pending_mailbox_frames(), [])


if __name__ == "__main__":
    unittest.main()
