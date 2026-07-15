from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import secrets
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg2
import psycopg2.extras

from rlab.metric_names import validate_metric_payload


PROTOCOL_VERSION = 1
TRANSPORT_NAME = "neon_mailbox_v1"
MAILBOX_DATABASE_ENV = "WORKER_MAILBOX_DATABASE_URL"
ATTEMPT_ID_ENV = "RLAB_WORKER_ATTEMPT_ID"
ATTEMPT_TOKEN_ENV = "RLAB_WORKER_ATTEMPT_TOKEN"
MAX_BATCH_FRAMES = 1_000
MAX_BATCH_UNCOMPRESSED_BYTES = 1_048_576
MAX_BATCH_COMPRESSED_BYTES = 2_097_152
DEFAULT_FLUSH_SECONDS = 30.0
DEFAULT_FINAL_FLUSH_SECONDS = 120.0
DEFAULT_LOCAL_OUTBOX_LIMIT_BYTES = 256 * 1024 * 1024
LEASE_SECONDS = 180


class MailboxProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class EncodedBatch:
    payload: bytes
    frame_count: int
    first_event_sequence: int | None
    last_event_sequence: int | None
    frame_ids: tuple[int, ...]


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def metric_frame(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(str(row["payload_json"]))
    if not isinstance(payload, dict):
        raise MailboxProtocolError("metric frame payload must be an object")
    kind = str(row.get("kind") or "history")
    if kind == "history":
        validate_metric_payload(payload)
    return {
        "event_id": str(row["event_id"]),
        "sequence": int(row["id"]),
        "observed_at": float(row.get("created_at") or time.time()),
        "global_step": payload.get("global_step", row.get("step")),
        "kind": kind,
        "payload": payload,
    }


def encode_metric_batch(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_frames: int = MAX_BATCH_FRAMES,
    max_uncompressed_bytes: int = MAX_BATCH_UNCOMPRESSED_BYTES,
) -> EncodedBatch:
    frames: list[dict[str, Any]] = []
    frame_ids: list[int] = []
    encoded_size = 2
    for row in rows:
        if len(frames) >= max_frames:
            break
        frame = metric_frame(row)
        candidate_size = len(_json_bytes(frame)) + (1 if frames else 0)
        if frames and encoded_size + candidate_size > max_uncompressed_bytes:
            break
        if not frames and encoded_size + candidate_size > max_uncompressed_bytes:
            raise MailboxProtocolError("one metric frame exceeds the mailbox batch limit")
        frames.append(frame)
        frame_ids.append(int(row["id"]))
        encoded_size += candidate_size
    raw = _json_bytes(frames)
    payload = gzip.compress(raw, compresslevel=6, mtime=0)
    if len(payload) > MAX_BATCH_COMPRESSED_BYTES:
        raise MailboxProtocolError("compressed metric batch exceeds the mailbox limit")
    return EncodedBatch(
        payload=payload,
        frame_count=len(frames),
        first_event_sequence=frame_ids[0] if frame_ids else None,
        last_event_sequence=frame_ids[-1] if frame_ids else None,
        frame_ids=tuple(frame_ids),
    )


def decode_metric_batch(payload: bytes) -> list[dict[str, Any]]:
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
            raw = compressed.read(MAX_BATCH_UNCOMPRESSED_BYTES + 1)
        if len(raw) > MAX_BATCH_UNCOMPRESSED_BYTES:
            raise MailboxProtocolError("uncompressed metric batch exceeds the mailbox limit")
        value = json.loads(raw)
    except (OSError, EOFError, json.JSONDecodeError) as exc:
        raise MailboxProtocolError("invalid gzip-json-v1 metric batch") from exc
    if not isinstance(value, list):
        raise MailboxProtocolError("metric batch payload must contain a list")
    frames: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            raise MailboxProtocolError("metric batch frame must be an object")
        if not str(raw.get("event_id") or "").strip():
            raise MailboxProtocolError("metric batch frame is missing event_id")
        try:
            sequence = int(raw.get("sequence"))
            float(raw.get("observed_at"))
        except (TypeError, ValueError) as exc:
            raise MailboxProtocolError(
                "metric batch frame sequence or timestamp is invalid"
            ) from exc
        if sequence < 1:
            raise MailboxProtocolError("metric batch frame sequence must be positive")
        payload_value = raw.get("payload")
        if not isinstance(payload_value, dict):
            raise MailboxProtocolError("metric batch frame payload must be an object")
        if str(raw.get("kind") or "history") == "history":
            validate_metric_payload(payload_value)
        frames.append(dict(raw))
    return frames


def mailbox_connect(database_url: str):
    return psycopg2.connect(
        database_url,
        connect_timeout=10,
        cursor_factory=psycopg2.extras.RealDictCursor,
        sslmode="require",
    )


class WorkerMailbox:
    def __init__(self, database_url: str, attempt_id: str, token: str) -> None:
        self.database_url = str(database_url).strip()
        self.attempt_id = str(attempt_id).strip()
        self.token = str(token).strip()
        if not self.database_url or not self.attempt_id or not self.token:
            raise ValueError("worker mailbox URL, attempt id, and token are required")

    @classmethod
    def from_env(cls) -> WorkerMailbox:
        return cls(
            os.environ.get(MAILBOX_DATABASE_ENV, ""),
            os.environ.get(ATTEMPT_ID_ENV, ""),
            os.environ.get(ATTEMPT_TOKEN_ENV, ""),
        )

    def preflight(self) -> None:
        conn = mailbox_connect(self.database_url)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS ready")
                    row = cur.fetchone()
            if not row or int(row["ready"]) != 1:
                raise MailboxProtocolError("worker mailbox preflight failed")
        finally:
            conn.close()
        self.append_event(
            "mailbox_preflight",
            {"protocol_version": PROTOCOL_VERSION},
            event_id=f"{self.attempt_id}:mailbox-preflight",
        )

    def submit_batch(
        self,
        *,
        batch_sequence: int,
        batch: EncodedBatch,
        final: bool,
    ) -> dict[str, Any]:
        conn = mailbox_connect(self.database_url)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT worker_submit_metric_batch(
                          %(attempt_id)s, %(token)s, %(protocol_version)s,
                          %(batch_sequence)s, %(first_event_sequence)s,
                          %(last_event_sequence)s, %(frame_count)s, %(payload)s, %(final)s
                        ) AS receipt
                        """,
                        {
                            "attempt_id": self.attempt_id,
                            "token": self.token,
                            "protocol_version": PROTOCOL_VERSION,
                            "batch_sequence": int(batch_sequence),
                            "first_event_sequence": batch.first_event_sequence,
                            "last_event_sequence": batch.last_event_sequence,
                            "frame_count": batch.frame_count,
                            "payload": psycopg2.Binary(batch.payload),
                            "final": bool(final),
                        },
                    )
                    row = cur.fetchone()
            receipt = dict((row or {}).get("receipt") or {})
            if int(receipt.get("accepted_sequence") or 0) < int(batch_sequence):
                raise MailboxProtocolError("mailbox did not acknowledge the submitted batch")
            return receipt
        finally:
            conn.close()

    def append_event(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        event_id: str | None = None,
    ) -> str:
        stable_id = event_id or str(uuid.uuid4())
        conn = mailbox_connect(self.database_url)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT worker_append_attempt_event(%s, %s, %s, %s, %s)",
                        (
                            self.attempt_id,
                            self.token,
                            stable_id,
                            str(event_type),
                            psycopg2.extras.Json(dict(payload)),
                        ),
                    )
            return stable_id
        finally:
            conn.close()

    def acknowledge_command(self, command_id: str) -> bool:
        conn = mailbox_connect(self.database_url)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT worker_ack_attempt_command(%s, %s, %s) AS acknowledged",
                        (self.attempt_id, self.token, str(command_id)),
                    )
                    row = cur.fetchone()
            return bool(row and row.get("acknowledged"))
        finally:
            conn.close()


def issue_worker_attempt_token(
    conn,
    *,
    attempt_id: str,
    lifetime: timedelta = timedelta(days=30),
) -> str:
    token = secrets.token_urlsafe(32)
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    expires_at = datetime.now(UTC) + lifetime
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE worker_attempts
                SET token_sha256 = %(digest)s, token_expires_at = %(expires_at)s
                WHERE attempt_id = %(attempt_id)s
                  AND status IN ('launching', 'running')
                RETURNING attempt_id
                """,
                {
                    "attempt_id": str(attempt_id),
                    "digest": digest,
                    "expires_at": expires_at,
                },
            )
            if not cur.fetchone():
                raise RuntimeError(f"worker attempt cannot receive a token: {attempt_id}")
    return token


def wandb_run_lock_key(train_job_id: int) -> str:
    return f"rlab-wandb-run:{int(train_job_id)}"


def claim_run_metric_batches(
    conn,
    *,
    owner: str,
    limit: int = 20,
    exclude_train_job_ids: Sequence[int] = (),
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.*
            FROM metric_batches b
            JOIN metric_streams s ON s.stream_id = b.stream_id
            JOIN worker_attempts a ON a.attempt_id = s.attempt_id
            JOIN train_jobs t ON t.id = a.train_job_id
            WHERE (b.lease_expires_at IS NULL OR b.lease_expires_at <= now())
              AND t.telemetry_transport = 'neon_mailbox_v1'
              AND COALESCE((t.train_config->>'wandb')::boolean, FALSE)
              AND t.id <> ALL(%(excluded_ids)s)
            ORDER BY b.created_at, b.id
            LIMIT 1
            """,
            {"excluded_ids": list(exclude_train_job_ids) or [0]},
        )
        run = cur.fetchone()
    if not run:
        return None
    run = dict(run)
    lock_key = wandb_run_lock_key(int(run["id"]))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0)) AS acquired",
            {"key": lock_key},
        )
        if not bool(cur.fetchone()["acquired"]):
            return None
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT b.*, s.submitted_sequence, s.published_sequence
                    FROM metric_batches b
                    JOIN metric_streams s ON s.stream_id = b.stream_id
                    JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                    WHERE a.train_job_id = %(train_job_id)s
                      AND (b.lease_expires_at IS NULL OR b.lease_expires_at <= now())
                    ORDER BY b.created_at, b.id
                    FOR UPDATE OF b SKIP LOCKED
                    LIMIT %(limit)s
                    """,
                    {"train_job_id": int(run["id"]), "limit": max(1, int(limit))},
                )
                batches = [dict(row) for row in cur.fetchall()]
                if not batches:
                    release_wandb_run_lock(conn, int(run["id"]))
                    return None
                ids = [int(row["id"]) for row in batches]
                cur.execute(
                    """
                    UPDATE metric_batches
                    SET lease_owner = %(owner)s,
                        lease_expires_at = now() + (%(lease_seconds)s * interval '1 second'),
                        attempts = attempts + 1,
                        last_error = NULL
                    WHERE id = ANY(%(ids)s)
                    """,
                    {"owner": owner, "lease_seconds": LEASE_SECONDS, "ids": ids},
                )
        return run, batches
    except Exception:
        release_wandb_run_lock(conn, int(run["id"]))
        raise


def release_wandb_run_lock(conn, train_job_id: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0))",
            {"key": wandb_run_lock_key(train_job_id)},
        )


def release_metric_batch_claims(
    conn,
    batches: Sequence[Mapping[str, Any]],
    *,
    error: str,
) -> None:
    ids = [int(row["id"]) for row in batches]
    if not ids:
        return
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE metric_batches
                SET lease_owner = NULL, lease_expires_at = NULL, last_error = %(error)s
                WHERE id = ANY(%(ids)s)
                """,
                {"ids": ids, "error": str(error)[:4000]},
            )


def commit_published_batches(conn, batches: Sequence[Mapping[str, Any]]) -> None:
    if not batches:
        return
    grouped: dict[str, int] = {}
    ids: list[int] = []
    for row in batches:
        stream_id = str(row["stream_id"])
        grouped[stream_id] = max(grouped.get(stream_id, 0), int(row["batch_sequence"]))
        ids.append(int(row["id"]))
    with conn:
        with conn.cursor() as cur:
            for stream_id, sequence in grouped.items():
                cur.execute(
                    """
                    UPDATE metric_streams
                    SET published_sequence = GREATEST(published_sequence, %(sequence)s),
                        updated_at = now()
                    WHERE stream_id = %(stream_id)s
                    """,
                    {"stream_id": stream_id, "sequence": sequence},
                )
            cur.execute("DELETE FROM metric_batches WHERE id = ANY(%(ids)s)", {"ids": ids})


def mark_submitted_batches(
    conn,
    batches: Sequence[Mapping[str, Any]],
    *,
    confirm_after_seconds: float = 30.0,
) -> None:
    """Record W&B submission without treating it as remote confirmation."""

    if not batches:
        return
    grouped: dict[str, int] = {}
    ids: list[int] = []
    for row in batches:
        stream_id = str(row["stream_id"])
        grouped[stream_id] = max(grouped.get(stream_id, 0), int(row["batch_sequence"]))
        ids.append(int(row["id"]))
    with conn:
        with conn.cursor() as cur:
            for stream_id, sequence in grouped.items():
                cur.execute(
                    """
                    UPDATE metric_streams
                    SET submitted_sequence = GREATEST(submitted_sequence, %(sequence)s),
                        updated_at = now()
                    WHERE stream_id = %(stream_id)s
                    """,
                    {"stream_id": stream_id, "sequence": sequence},
                )
            cur.execute(
                """
                UPDATE metric_batches
                SET lease_owner = NULL,
                    lease_expires_at = now() + (%(delay)s * interval '1 second'),
                    last_error = NULL
                WHERE id = ANY(%(ids)s)
                """,
                {
                    "ids": ids,
                    "delay": max(float(confirm_after_seconds), 0.0),
                },
            )


def mailbox_storage_bytes(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT pg_total_relation_size('metric_batches')
                 + pg_total_relation_size('attempt_events')
                 + pg_total_relation_size('attempt_commands') AS bytes
            """
        )
        row = cur.fetchone()
    return int((row or {}).get("bytes") or 0)


def discard_disabled_metric_batches(conn) -> int:
    """Acknowledge transient batches for runs that explicitly opted out of W&B."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH discarded AS (
                  DELETE FROM metric_batches b
                  USING metric_streams s, worker_attempts a, train_jobs t
                  WHERE s.stream_id = b.stream_id
                    AND a.attempt_id = s.attempt_id
                    AND t.id = a.train_job_id
                    AND NOT COALESCE((t.train_config->>'wandb')::boolean, FALSE)
                  RETURNING b.stream_id, b.batch_sequence
                ), advanced AS (
                  SELECT stream_id, max(batch_sequence) AS sequence
                  FROM discarded GROUP BY stream_id
                )
                UPDATE metric_streams s
                SET published_sequence = GREATEST(s.published_sequence, advanced.sequence),
                    updated_at = now()
                FROM advanced
                WHERE s.stream_id = advanced.stream_id
                RETURNING s.stream_id
                """
            )
            return cur.rowcount


def finalize_mailbox_runs_without_eval(conn) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs t
                SET status = 'succeeded', finished_at = now(), error = NULL,
                    live_publication_error = NULL,
                    live_publication_next_retry_at = NULL
                WHERE t.status = 'finalizing'
                  AND t.telemetry_transport = 'neon_mailbox_v1'
                  AND COALESCE(t.train_config->>'checkpoint_eval_backend', 'local') <> 'modal'
                  AND t.live_publication_status IN ('complete', 'disabled')
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_batches b
                    JOIN metric_streams s ON s.stream_id = b.stream_id
                    JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                    WHERE a.train_job_id = t.id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_streams s
                    JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                    WHERE a.train_job_id = t.id
                      AND (s.final_sequence IS NULL
                           OR s.published_sequence < s.final_sequence)
                  )
                """
            )
            return cur.rowcount


def enqueue_projection_payload(
    conn,
    *,
    eval_job_id: int,
    payload: Mapping[str, Any],
) -> bool:
    stable_payload = dict(payload)
    stable_payload["eval_job_id"] = int(eval_job_id)
    event_id = hashlib.sha256(_json_bytes(stable_payload)).hexdigest()
    row = {
        "id": 1,
        "event_id": event_id,
        "created_at": time.time(),
        "kind": "projection",
        "step": int(stable_payload.get("checkpoint_step") or 0),
        "payload_json": json.dumps(stable_payload, sort_keys=True, default=str),
    }
    batch = encode_metric_batch([row])
    stream_id = f"eval-projection-{int(eval_job_id)}"
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.train_job_id, COALESCE(a.attempt_id, '') AS modal_attempt_id
                FROM eval_jobs j
                LEFT JOIN eval_attempts a ON a.id = j.accepted_attempt_id
                WHERE j.id = %(eval_job_id)s
                FOR UPDATE OF j
                """,
                {"eval_job_id": int(eval_job_id)},
            )
            eval_row = cur.fetchone()
            if not eval_row:
                raise RuntimeError(f"evaluation job not found: {eval_job_id}")
            attempt_id = str(eval_row.get("modal_attempt_id") or "") or f"fleet-{stream_id}"
            cur.execute(
                """
                INSERT INTO worker_attempts (
                  attempt_id, train_job_id, eval_job_id, task_kind, provider,
                  provider_run_id, status, started_at, finished_at
                ) VALUES (
                  %(attempt_id)s, %(train_job_id)s, %(eval_job_id)s, 'eval', 'modal',
                  NULLIF(%(provider_run_id)s, ''), 'succeeded', now(), now()
                )
                ON CONFLICT (attempt_id) DO NOTHING
                """,
                {
                    "attempt_id": attempt_id,
                    "train_job_id": int(eval_row["train_job_id"]),
                    "eval_job_id": int(eval_job_id),
                    "provider_run_id": str(eval_row.get("modal_attempt_id") or ""),
                },
            )
            cur.execute(
                """
                INSERT INTO metric_streams (
                  stream_id, attempt_id, accepted_sequence, final_sequence
                ) VALUES (%(stream_id)s, %(attempt_id)s, 1, 1)
                ON CONFLICT (stream_id) DO NOTHING
                """,
                {"stream_id": stream_id, "attempt_id": attempt_id},
            )
            cur.execute(
                """
                INSERT INTO metric_batches (
                  stream_id, batch_sequence, first_event_sequence, last_event_sequence,
                  frame_count, payload, final
                ) VALUES (%(stream_id)s, 1, 1, 1, 1, %(payload)s, TRUE)
                ON CONFLICT (stream_id, batch_sequence) DO NOTHING
                RETURNING id
                """,
                {"stream_id": stream_id, "payload": psycopg2.Binary(batch.payload)},
            )
            inserted = cur.fetchone() is not None
            cur.execute(
                """
                UPDATE eval_jobs
                SET projection_enqueued_at = COALESCE(projection_enqueued_at, now()),
                    projection_error = NULL, updated_at = now()
                WHERE id = %(eval_job_id)s
                """,
                {"eval_job_id": int(eval_job_id)},
            )
            cur.execute(
                """
                UPDATE train_jobs
                SET live_publication_status = 'pending',
                    live_publication_error = NULL,
                    live_publication_next_retry_at = NULL
                WHERE id = %(train_job_id)s
                """,
                {"train_job_id": int(eval_row["train_job_id"])},
            )
    return inserted


def consume_attempt_events(conn, *, limit: int = 100) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM attempt_events
                WHERE event_type IN ('mailbox_preflight', 'metric_stream_closed')
                ORDER BY id
                FOR UPDATE SKIP LOCKED
                LIMIT %(limit)s
                """,
                {"limit": max(1, int(limit))},
            )
            rows = [dict(row) for row in cur.fetchall()]
            for row in rows:
                if str(row["event_type"]) == "metric_stream_closed":
                    final_sequence = int((row.get("payload_json") or {}).get("final_sequence") or 0)
                    cur.execute(
                        """
                        UPDATE metric_streams
                        SET final_sequence = COALESCE(final_sequence, %(final_sequence)s),
                            updated_at = now()
                        WHERE attempt_id = %(attempt_id)s
                        """,
                        {
                            "attempt_id": str(row["attempt_id"]),
                            "final_sequence": final_sequence,
                        },
                    )
            if rows:
                cur.execute(
                    "DELETE FROM attempt_events WHERE id = ANY(%(ids)s)",
                    {"ids": [int(row["id"]) for row in rows]},
                )
    return len(rows)
