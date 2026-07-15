from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlab.metric_store import MetricStore, metric_store_path
from rlab.telemetry_mailbox import (
    MailboxProtocolError,
    decode_metric_batch,
    encode_metric_batch,
)
from rlab.telemetry_relay import _handle_commands, main as relay_main


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
    def test_batch_is_deterministic_gzip_and_preserves_explicit_global_steps(self) -> None:
        batch = encode_metric_batch(
            [frame(1, step=300), frame(2, step=100), frame(3, step=200)]
        )
        again = encode_metric_batch(
            [frame(1, step=300), frame(2, step=100), frame(3, step=200)]
        )

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

            with mock.patch(
                "rlab.telemetry_relay.WorkerMailbox.from_env", return_value=mailbox
            ):
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

            with mock.patch(
                "rlab.telemetry_relay.WorkerMailbox.from_env", return_value=mailbox
            ):
                relay_main(["--run-dir", str(run_dir), "--stop-file", str(stop_file)])

            self.assertEqual(mailbox.batches[0]["frames"], [])
            self.assertTrue(mailbox.batches[0]["final"])

    def test_stop_command_is_written_and_acknowledged_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            command_file = Path(tmp) / "command.json"
            mailbox = FakeMailbox()
            receipt = {
                "commands": [
                    {
                        "command_id": "stop-1",
                        "command_type": "stop",
                        "payload": {"reason": "candidate passed"},
                    }
                ]
            }

            _handle_commands(mailbox, receipt, command_file=command_file)
            _handle_commands(mailbox, receipt, command_file=command_file)

            command = json.loads(command_file.read_text(encoding="utf-8"))
            self.assertEqual(command["command_id"], "stop-1")
            self.assertEqual(mailbox.acknowledged_commands, ["stop-1", "stop-1"])

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

            with mock.patch(
                "rlab.telemetry_relay.WorkerMailbox.from_env", return_value=mailbox
            ):
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
