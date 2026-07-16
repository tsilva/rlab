from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

from rlab.metric_store import MetricStore, metric_store_path
from rlab.telemetry_mailbox import (
    DEFAULT_FINAL_FLUSH_SECONDS,
    DEFAULT_FLUSH_SECONDS,
    DEFAULT_LOCAL_OUTBOX_LIMIT_BYTES,
    MAX_BATCH_FRAMES,
    MAX_BATCH_UNCOMPRESSED_BYTES,
    WorkerMailbox,
    encode_metric_batch,
)


MAX_COMMAND_RELAY_FAILURES = 5


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command_filename(command_id: str) -> str:
    return hashlib.sha256(command_id.encode("utf-8")).hexdigest() + ".json"


def _handle_commands(
    mailbox: WorkerMailbox,
    receipt: dict[str, object],
    *,
    command_file: Path,
) -> None:
    """Legacy receipt adapter retained for old workers; new workers use CommandRelay."""

    raw_commands = receipt.get("commands") or []
    if not isinstance(raw_commands, list):
        return
    for raw in raw_commands:
        if not isinstance(raw, dict):
            continue
        command_id = str(raw.get("command_id") or "")
        command_type = str(raw.get("command_type") or "")
        if not command_id or command_type not in {"stop", "cancel"}:
            continue
        _write_atomic(
            command_file,
            {
                "command_id": command_id,
                "command_type": command_type,
                "payload": raw.get("payload") or {},
            },
        )
        mailbox.acknowledge_command(command_id)


class CommandRelay(threading.Thread):
    def __init__(
        self,
        mailbox: WorkerMailbox,
        *,
        inbox_dir: Path,
        receipt_dir: Path,
        poll_seconds: float,
    ) -> None:
        super().__init__(name="rlab-command-relay", daemon=True)
        self.mailbox = mailbox
        self.inbox_dir = inbox_dir
        self.receipt_dir = receipt_dir
        self.poll_seconds = max(0.1, float(poll_seconds))
        self.stop_event = threading.Event()
        self.error: BaseException | None = None

    def _ack_receipts(self) -> None:
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(self.receipt_dir.glob("*.json")):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                command_id = str(receipt.get("command_id") or "")
                signal_sent_at = str(receipt.get("signal_sent_at") or "").strip() or None
                if command_id and self.mailbox.acknowledge_command(
                    command_id, acknowledged_at=signal_sent_at
                ):
                    path.unlink(missing_ok=True)
            except Exception:
                continue

    def _deliver(self) -> None:
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        for raw in self.mailbox.poll_commands():
            command_id = str(raw.get("command_id") or "")
            command_type = str(raw.get("command_type") or "")
            if not command_id or command_type not in {"stop", "cancel"}:
                continue
            path = self.inbox_dir / _command_filename(command_id)
            if not path.is_file():
                _write_atomic(
                    path,
                    {
                        "command_id": command_id,
                        "command_type": command_type,
                        "payload": raw.get("payload") or {},
                        "created_at": raw.get("created_at"),
                    },
                )
            self.mailbox.mark_command_delivered(command_id)

    def run(self) -> None:
        consecutive_failures = 0
        while not self.stop_event.is_set():
            try:
                self._ack_receipts()
                self._deliver()
            except BaseException as exc:
                self.error = exc
                consecutive_failures += 1
                if consecutive_failures >= MAX_COMMAND_RELAY_FAILURES:
                    return
            else:
                self.error = None
                consecutive_failures = 0
            self.stop_event.wait(self.poll_seconds)
        try:
            self._ack_receipts()
        except BaseException as exc:
            self.error = exc

    def stop(self) -> None:
        self.stop_event.set()


def submit_next_batch(
    store: MetricStore,
    mailbox: WorkerMailbox,
    *,
    final: bool,
    heartbeat: bool = False,
) -> tuple[bool, bool]:
    rows = store.pending_mailbox_frames(limit=MAX_BATCH_FRAMES + 1)
    batch = encode_metric_batch(rows)
    is_last = len(rows) <= len(batch.frame_ids)
    send_final = bool(final and is_last)
    if not batch.frame_ids and not send_final and not heartbeat:
        return False, False
    sequence = store.next_mailbox_batch_sequence()
    mailbox.submit_batch(
        batch_sequence=sequence,
        batch=batch,
        final=send_final,
    )
    store.mark_metric_frames_delivered(batch.frame_ids, batch_sequence=sequence)
    return bool(batch.frame_ids), send_final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relay local telemetry to the Fleet mailbox.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-config-json", type=Path)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--command-inbox-dir", type=Path)
    parser.add_argument("--command-receipt-dir", type=Path)
    parser.add_argument("--learner-stop-marker", type=Path)
    parser.add_argument("--command-poll-seconds", type=float, default=1.0)
    parser.add_argument("--flush-seconds", type=float, default=DEFAULT_FLUSH_SECONDS)
    parser.add_argument(
        "--final-flush-seconds", type=float, default=DEFAULT_FINAL_FLUSH_SECONDS
    )
    parser.add_argument(
        "--outbox-limit-bytes", type=int, default=DEFAULT_LOCAL_OUTBOX_LIMIT_BYTES
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ready_file = args.ready_file or (args.run_dir / "mailbox_relay_ready.json")
    command_inbox_dir = args.command_inbox_dir or (args.run_dir / "mailbox_commands" / "inbox")
    command_receipt_dir = args.command_receipt_dir or (
        args.run_dir / "mailbox_commands" / "receipts"
    )
    learner_stop_marker = args.learner_stop_marker or (
        args.run_dir / "learner_stop_observed.json"
    )
    store = MetricStore(metric_store_path(args.run_dir))
    store.init()
    mailbox = WorkerMailbox.from_env()
    mailbox.preflight()
    _write_atomic(
        ready_file,
        {
            "ready": True,
            "attempt_id": mailbox.attempt_id,
            "protocol_version": 1,
        },
    )
    command_relay = (
        CommandRelay(
            mailbox,
            inbox_dir=command_inbox_dir,
            receipt_dir=command_receipt_dir,
            poll_seconds=args.command_poll_seconds,
        )
        if callable(getattr(mailbox, "poll_commands", None))
        else None
    )
    if command_relay is not None:
        command_relay.start()
    next_flush = time.monotonic()
    retry_delay = 1.0
    while not args.stop_file.exists():
        if (
            command_relay is not None
            and command_relay.error is not None
            and not command_relay.is_alive()
        ):
            raise RuntimeError("dedicated command relay failed") from command_relay.error
        outbox = store.metric_outbox_stats()
        if outbox["bytes"] > max(1, int(args.outbox_limit_bytes)):
            raise RuntimeError("local metric outbox exceeded its safety limit")
        now = time.monotonic()
        batch_full = (
            outbox["frames"] >= MAX_BATCH_FRAMES
            or outbox["bytes"] >= MAX_BATCH_UNCOMPRESSED_BYTES
        )
        if now >= next_flush or batch_full:
            try:
                activity, _ = submit_next_batch(
                    store,
                    mailbox,
                    final=False,
                    heartbeat=True,
                )
                retry_delay = 1.0
                next_flush = now + max(0.1, float(args.flush_seconds))
                if activity:
                    continue
            except Exception as exc:
                print(f"telemetry mailbox flush failed; retaining SQLite rows: {exc}", flush=True)
                next_flush = now + retry_delay
                retry_delay = min(retry_delay * 2.0, 30.0)
        time.sleep(min(max(next_flush - time.monotonic(), 0.05), 0.25))

    store.close_metric_outbox()
    deadline = time.monotonic() + max(0.1, float(args.final_flush_seconds))
    retry_delay = 0.25
    final_sent = False
    while not final_sent:
        try:
            _, final_sent = submit_next_batch(
                store,
                mailbox,
                final=True,
            )
            retry_delay = 0.25
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "final telemetry batch was not acknowledged by Neon"
                ) from exc
            print(f"final telemetry flush failed; retrying: {exc}", flush=True)
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2.0, 5.0)
    if learner_stop_marker.is_file():
        marker = json.loads(learner_stop_marker.read_text(encoding="utf-8"))
        mailbox.append_event(
            "learner_stop_observed",
            marker if isinstance(marker, dict) else {},
            event_id=f"{mailbox.attempt_id}:learner-stop-observed",
        )
    while True:
        try:
            mailbox.append_event(
                "metric_stream_closed",
                {"final_sequence": store.next_mailbox_batch_sequence() - 1},
                event_id=f"{mailbox.attempt_id}:metric-stream-closed",
            )
            break
        except Exception as exc:
            if time.monotonic() >= deadline:
                raise RuntimeError("metric stream closure was not acknowledged by Neon") from exc
            time.sleep(0.25)
    if command_relay is not None:
        command_relay.stop()
        command_relay.join(timeout=2.0)
        if command_relay.error is not None:
            raise RuntimeError("dedicated command relay failed") from command_relay.error
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
