from __future__ import annotations

import argparse
import json
import sys
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


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _handle_commands(
    mailbox: WorkerMailbox,
    receipt: dict[str, object],
    *,
    command_file: Path,
) -> None:
    raw_commands = receipt.get("commands") or []
    if not isinstance(raw_commands, list):
        return
    for raw in raw_commands:
        if not isinstance(raw, dict):
            continue
        command_id = str(raw.get("command_id") or "")
        command_type = str(raw.get("command_type") or "")
        if command_type not in {"stop", "cancel"} or not command_id:
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


def submit_next_batch(
    store: MetricStore,
    mailbox: WorkerMailbox,
    *,
    final: bool,
    heartbeat: bool = False,
    command_file: Path,
) -> tuple[bool, bool]:
    rows = store.pending_mailbox_frames(limit=MAX_BATCH_FRAMES + 1)
    batch = encode_metric_batch(rows)
    is_last = len(rows) <= len(batch.frame_ids)
    send_final = bool(final and is_last)
    if not batch.frame_ids and not send_final and not heartbeat:
        return False, False
    sequence = store.next_mailbox_batch_sequence()
    receipt = mailbox.submit_batch(
        batch_sequence=sequence,
        batch=batch,
        final=send_final,
    )
    store.mark_metric_frames_delivered(batch.frame_ids, batch_sequence=sequence)
    _handle_commands(mailbox, receipt, command_file=command_file)
    return bool(batch.frame_ids), send_final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Relay local telemetry to the Fleet mailbox.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-config-json", type=Path)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--command-file", type=Path)
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
    command_file = args.command_file or (args.run_dir / "mailbox_command.json")
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
    next_flush = time.monotonic()
    retry_delay = 1.0
    while not args.stop_file.exists():
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
                    command_file=command_file,
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
                command_file=command_file,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
