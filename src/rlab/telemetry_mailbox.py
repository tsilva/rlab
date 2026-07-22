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
REMOTE_CONFIRM_POLL_SECONDS = 5.0
REMOTE_CONFIRM_TIMEOUT_SECONDS = 120.0


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


def decode_metric_batch(payload: bytes, *, schema_version: int = 5) -> list[dict[str, Any]]:
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
            validate_metric_payload(payload_value, schema_version=schema_version)
        frames.append(dict(raw))
    return frames


def mailbox_connect(database_url: str):
    return psycopg2.connect(
        database_url,
        connect_timeout=10,
        keepalives=1,
        keepalives_idle=10,
        keepalives_interval=5,
        keepalives_count=3,
        tcp_user_timeout=30000,
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
        # Readiness must cover the independent stop-critical command path, not
        # only generic connectivity and metric/event procedures.
        self.poll_commands()
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

    def acknowledge_command(self, command_id: str, *, acknowledged_at: str | None = None) -> bool:
        conn = mailbox_connect(self.database_url)
        try:
            with conn:
                with conn.cursor() as cur:
                    if acknowledged_at is None:
                        cur.execute(
                            "SELECT worker_ack_attempt_command(%s, %s, %s) AS acknowledged",
                            (self.attempt_id, self.token, str(command_id)),
                        )
                    else:
                        cur.execute(
                            "SELECT worker_ack_attempt_command(%s, %s, %s, %s::timestamptz) "
                            "AS acknowledged",
                            (
                                self.attempt_id,
                                self.token,
                                str(command_id),
                                str(acknowledged_at),
                            ),
                        )
                    row = cur.fetchone()
            return bool(row and row.get("acknowledged"))
        finally:
            conn.close()

    def poll_commands(self) -> list[dict[str, Any]]:
        conn = mailbox_connect(self.database_url)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT worker_poll_attempt_commands(%s, %s) AS commands",
                        (self.attempt_id, self.token),
                    )
                    row = cur.fetchone()
            commands = (row or {}).get("commands") or []
            if not isinstance(commands, list):
                raise MailboxProtocolError("mailbox command poll returned an invalid payload")
            return [dict(value) for value in commands if isinstance(value, Mapping)]
        finally:
            conn.close()

    def mark_command_delivered(self, command_id: str) -> bool:
        conn = mailbox_connect(self.database_url)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT worker_mark_attempt_command_delivered(%s, %s, %s) AS delivered",
                        (self.attempt_id, self.token, str(command_id)),
                    )
                    row = cur.fetchone()
            return bool(row and row.get("delivered"))
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
    train_job_id: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    from rlab.job_queue import acquire_fleet_admission_xact_lock

    acquire_fleet_admission_xact_lock(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.*
            FROM metric_batches b
            JOIN metric_streams s ON s.stream_id = b.stream_id
            JOIN worker_attempts a ON a.attempt_id = s.attempt_id
            JOIN train_jobs t ON t.id = a.train_job_id
            WHERE (b.lease_expires_at IS NULL OR b.lease_expires_at <= now())
              AND (
                t.telemetry_transport = 'neon_mailbox_v1'
                OR (
                  t.status IN (
                    'finalizing', 'succeeded', 'failed', 'canceled', 'finalization_failed'
                  )
                  AND s.stream_id LIKE 'artifact-v2-%%'
                )
              )
              AND COALESCE((t.train_config->>'wandb')::boolean, FALSE)
              AND (
                t.live_publication_status NOT IN ('complete', 'disabled', 'failed')
                OR (
                  t.status IN ('succeeded', 'failed', 'canceled', 'finalization_failed')
                  AND t.live_publication_status = 'complete'
                  AND s.stream_id LIKE 'artifact-v2-%%'
                )
              )
              AND (t.live_publication_next_retry_at IS NULL
                   OR t.live_publication_next_retry_at <= now())
              AND t.id <> ALL(%(excluded_ids)s)
              AND (%(train_job_id)s IS NULL OR t.id = %(train_job_id)s)
            ORDER BY b.created_at, b.id
            LIMIT 1
            """,
            {
                "excluded_ids": list(exclude_train_job_ids) or [0],
                "train_job_id": None if train_job_id is None else int(train_job_id),
            },
        )
        run = cur.fetchone()
    if not run:
        conn.rollback()
        return None
    run = dict(run)
    lock_key = wandb_run_lock_key(int(run["id"]))
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0)) AS acquired",
            {"key": lock_key},
        )
        if not bool(cur.fetchone()["acquired"]):
            conn.rollback()
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
                    JOIN train_jobs t ON t.id = a.train_job_id
                    WHERE a.train_job_id = %(train_job_id)s
                      AND (
                        t.telemetry_transport = 'neon_mailbox_v1'
                        OR (
                          t.status IN (
                            'finalizing', 'succeeded', 'failed', 'canceled',
                            'finalization_failed'
                          )
                          AND s.stream_id LIKE 'artifact-v2-%%'
                        )
                      )
                      AND (
                        t.live_publication_status NOT IN ('complete', 'disabled', 'failed')
                        OR (
                          t.status IN (
                            'succeeded', 'failed', 'canceled', 'finalization_failed'
                          )
                          AND t.live_publication_status = 'complete'
                          AND s.stream_id LIKE 'artifact-v2-%%'
                        )
                      )
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


def pending_metric_run_ids(conn, *, limit: int = 100) -> list[int]:
    """Return independent W&B runs ready for parallel publication."""

    with conn.cursor() as cur:
        cur.execute(
            """
            WITH candidates AS (
              SELECT t.id, min(b.created_at) AS ready_at
              FROM metric_batches b
              JOIN metric_streams s ON s.stream_id = b.stream_id
              JOIN worker_attempts a ON a.attempt_id = s.attempt_id
              JOIN train_jobs t ON t.id = a.train_job_id
              WHERE (b.lease_expires_at IS NULL OR b.lease_expires_at <= now())
                AND (
                  t.telemetry_transport = 'neon_mailbox_v1'
                  OR (
                    t.status IN (
                      'finalizing', 'succeeded', 'failed', 'canceled', 'finalization_failed'
                    )
                    AND s.stream_id LIKE 'artifact-v2-%%'
                  )
                )
                AND COALESCE((t.train_config->>'wandb')::boolean, FALSE)
                AND (
                  t.live_publication_status NOT IN ('complete', 'disabled', 'failed')
                  OR (
                    t.status IN ('succeeded', 'failed', 'canceled', 'finalization_failed')
                    AND t.live_publication_status = 'complete'
                    AND s.stream_id LIKE 'artifact-v2-%%'
                  )
                )
                AND (t.live_publication_next_retry_at IS NULL
                     OR t.live_publication_next_retry_at <= now())
              GROUP BY t.id
              UNION ALL
              SELECT t.id, t.created_at AS ready_at
              FROM train_jobs t
              WHERE t.live_publication_status = 'finishing'
                AND (
                  t.telemetry_transport = 'neon_mailbox_v1'
                  OR EXISTS (
                    SELECT 1 FROM metric_streams s
                    JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                    WHERE a.train_job_id = t.id
                      AND s.stream_id LIKE 'artifact-v2-%%'
                  )
                )
                AND (t.live_publication_next_retry_at IS NULL
                     OR t.live_publication_next_retry_at <= now())
              UNION ALL
              SELECT t.id, t.created_at AS ready_at
              FROM train_jobs t
              WHERE t.status IN ('succeeded', 'failed', 'canceled', 'finalization_failed')
                AND t.live_publication_status IN ('pending', 'live')
                AND (t.live_publication_next_retry_at IS NULL
                     OR t.live_publication_next_retry_at <= now())
                AND EXISTS (
                  SELECT 1 FROM metric_streams s
                  JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                  WHERE a.train_job_id = t.id
                    AND s.stream_id LIKE 'artifact-v2-%%'
                )
              UNION ALL
              SELECT t.id, COALESCE(t.process_exited_at, t.created_at) AS ready_at
              FROM train_jobs t
              WHERE t.status = 'finalizing'
                AND t.process_exited_at IS NOT NULL
                AND t.telemetry_transport = 'neon_mailbox_v1'
                AND COALESCE((t.train_config->>'wandb')::boolean, FALSE)
                AND COALESCE(t.train_config->>'checkpoint_eval_backend', 'local')
                    <> 'modal'
                AND t.live_publication_status IN ('pending', 'live')
            )
            SELECT id, min(ready_at) AS ready_at
            FROM candidates
            GROUP BY id
            ORDER BY ready_at, id
            LIMIT %(limit)s
            """,
            {"limit": max(1, int(limit))},
        )
        return [int(row["id"]) for row in cur.fetchall()]


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
    confirm_after_seconds: float = REMOTE_CONFIRM_POLL_SECONDS,
    refresh_submitted_at: bool = False,
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
                    last_error = NULL,
                    submitted_at = CASE WHEN %(refresh_submitted_at)s
                      THEN now() ELSE COALESCE(submitted_at, now()) END
                WHERE id = ANY(%(ids)s)
                """,
                {
                    "ids": ids,
                    "delay": max(float(confirm_after_seconds), 0.0),
                    "refresh_submitted_at": bool(refresh_submitted_at),
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
                SET status = CASE
                      WHEN l.state = 'canceled' THEN 'canceled'
                      WHEN l.state = 'failed' THEN 'failed'
                      ELSE 'succeeded'
                    END,
                    finished_at = now(),
                    error = CASE
                      WHEN l.state = 'failed' THEN COALESCE(l.error, t.error)
                      ELSE NULL
                    END,
                    live_publication_error = NULL,
                    live_publication_next_retry_at = NULL
                FROM job_launches l
                WHERE t.status = 'finalizing'
                  AND l.job_id = t.id
                  AND l.state IN ('succeeded', 'failed', 'canceled')
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
    stream_kind: str = "evaluation",
) -> bool:
    if stream_kind not in {"evaluation", "artifact"}:
        raise ValueError("projection stream kind must be evaluation or artifact")
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
    stream_id = (
        f"eval-projection-{int(eval_job_id)}"
        if stream_kind == "evaluation"
        else f"artifact-projection-{int(eval_job_id)}"
    )
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
            if stream_kind == "evaluation":
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


def _artifact_publication_aliases(kind: str, step: int, role: str) -> list[str]:
    aliases = [f"step-{int(step)}"]
    if kind in {"final", "interrupted"}:
        aliases.append(kind)
    if role == "promotion":
        aliases.append("promoted")
    return sorted(set(aliases))


def schedule_artifact_publications(conn, *, limit: int = 10) -> dict[str, int]:
    """Create durable W&B projection streams from the verified artifact ledger."""

    limit = max(1, int(limit))
    scheduled = 0
    opted_out = 0
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_xact_lock(hashtextextended(%(key)s, 0)) AS acquired",
                {"key": "rlab-artifact-publication-scheduler-v2"},
            )
            if not bool(cur.fetchone()["acquired"]):
                return {"scheduled": 0, "opted_out": 0}
            cur.execute(
                """
                SELECT l.*, t.train_config, t.wandb_run_id, t.run_name,
                  t.wandb_group, t.wandb_tags,
                  'availability'::text AS publication_role,
                  0::bigint AS publication_revision
                FROM artifact_announcement_ledger l
                JOIN train_jobs t ON t.id = l.train_job_id
                WHERE l.disposition = 'ready'
                  AND (
                    t.status IN ('launching', 'starting', 'running', 'finalizing')
                    OR (
                      t.status IN ('succeeded', 'failed', 'canceled', 'finalization_failed')
                      AND t.live_publication_status IN ('pending', 'live')
                    )
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM artifact_publication_receipts receipt
                    WHERE receipt.train_job_id = l.train_job_id
                      AND receipt.ledger_id = l.ledger_id
                      AND receipt.role = 'availability'
                      AND receipt.promotion_revision = 0
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_streams stream
                    WHERE stream.stream_id =
                      'artifact-v2-' || l.train_job_id::text || '-' || l.ledger_id::text ||
                      '-availability-r0'
                  )
                ORDER BY l.verified_at, l.train_job_id, l.ledger_id
                LIMIT %(limit)s
                """,
                {"limit": limit},
            )
            candidates = [dict(row) for row in cur.fetchall()]
            cur.execute(
                """
                SELECT l.*, t.train_config, t.wandb_run_id, t.run_name,
                  t.wandb_group, t.wandb_tags,
                  'promotion'::text AS publication_role,
                  r.promotion_revision AS publication_revision
                FROM eval_runs r
                JOIN train_jobs t ON t.id = r.train_job_id
                JOIN eval_jobs promoted ON promoted.id = r.promoted_eval_job_id
                JOIN artifact_announcement_ledger l
                  ON l.train_job_id = promoted.train_job_id
                 AND l.ledger_id = promoted.ledger_id
                WHERE l.disposition = 'ready'
                  AND r.promotion_revision >= 1
                  AND (
                    t.status IN ('running', 'finalizing')
                    OR (
                      t.status IN ('succeeded', 'failed', 'canceled', 'finalization_failed')
                      AND t.live_publication_status IN ('pending', 'live')
                    )
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM artifact_publication_receipts receipt
                    WHERE receipt.train_job_id = l.train_job_id
                      AND receipt.ledger_id = l.ledger_id
                      AND receipt.role = 'promotion'
                      AND receipt.promotion_revision = r.promotion_revision
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_streams stream
                    WHERE stream.stream_id =
                      'artifact-v2-' || l.train_job_id::text || '-' || l.ledger_id::text ||
                      '-promotion-r' || r.promotion_revision::text
                  )
                ORDER BY r.updated_at, l.train_job_id, l.ledger_id
                LIMIT %(limit)s
                """,
                {"limit": limit},
            )
            candidates.extend(dict(row) for row in cur.fetchall())
            candidates.sort(
                key=lambda row: (
                    row.get("verified_at") or datetime.min.replace(tzinfo=UTC),
                    0 if row["publication_role"] == "availability" else 1,
                    int(row["train_job_id"]),
                    int(row["ledger_id"]),
                    int(row["publication_revision"]),
                )
            )
            for candidate in candidates[:limit]:
                train_job_id = int(candidate["train_job_id"])
                ledger_id = int(candidate["ledger_id"])
                role = str(candidate["publication_role"])
                revision = int(candidate["publication_revision"])
                kind = str(candidate["artifact_kind"])
                step = int(candidate["checkpoint_step"])
                aliases = _artifact_publication_aliases(kind, step, role)
                train_config = dict(candidate.get("train_config") or {})
                train_config.update(
                    {
                        "wandb_run_id": str(
                            candidate.get("wandb_run_id") or train_config.get("wandb_run_id") or ""
                        ),
                        "run_name": str(candidate.get("run_name") or ""),
                        "wandb_group": str(candidate.get("wandb_group") or ""),
                        "wandb_tags": list(candidate.get("wandb_tags") or ()),
                    }
                )
                disabled = not bool(train_config.get("wandb", False)) or bool(
                    train_config.get("no_wandb_artifacts", False)
                )
                stream_id = f"artifact-v2-{train_job_id}-{ledger_id}-{role}-r{revision}"
                if disabled:
                    cur.execute(
                        """
                        INSERT INTO artifact_publication_receipts (
                          train_job_id, ledger_id, role, promotion_revision,
                          disposition, artifact_kind, checkpoint_step,
                          checkpoint_sha256, checkpoint_uri, metadata_uri,
                          metadata_sha256, recipe_uri, recipe_sha256,
                          announcement_sha256, expected_aliases
                        ) VALUES (
                          %(train_job_id)s, %(ledger_id)s, %(role)s, %(revision)s,
                          'opted_out', %(kind)s, %(step)s,
                          %(checkpoint_sha256)s, %(checkpoint_uri)s, %(metadata_uri)s,
                          %(metadata_sha256)s, %(recipe_uri)s, %(recipe_sha256)s,
                          %(announcement_sha256)s, %(aliases)s
                        )
                        ON CONFLICT (train_job_id, ledger_id, role, promotion_revision)
                        DO NOTHING
                        """,
                        {
                            "train_job_id": train_job_id,
                            "ledger_id": ledger_id,
                            "role": role,
                            "revision": revision,
                            "kind": kind,
                            "step": step,
                            "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                            "checkpoint_uri": str(candidate["checkpoint_uri"]),
                            "metadata_uri": str(candidate["metadata_uri"]),
                            "metadata_sha256": str(candidate["metadata_sha256"]),
                            "recipe_uri": candidate.get("recipe_uri"),
                            "recipe_sha256": candidate.get("recipe_sha256"),
                            "announcement_sha256": str(candidate["announcement_sha256"]),
                            "aliases": aliases,
                        },
                    )
                    opted_out += int(cur.rowcount == 1)
                    continue
                payload = {
                    "projection_kind": "artifact_reference",
                    "artifact_publication_schema": "v2",
                    "train_config": train_config,
                    "train_job_id": train_job_id,
                    "ledger_id": ledger_id,
                    "publication_role": role,
                    "promotion_revision": revision,
                    "publication_stream_id": stream_id,
                    "announcement_sha256": str(candidate["announcement_sha256"]),
                    "artifact_kind": kind,
                    "checkpoint_step": step,
                    "checkpoint_uri": str(candidate["checkpoint_uri"]),
                    "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                    "metadata_uri": str(candidate["metadata_uri"]),
                    "metadata_sha256": str(candidate["metadata_sha256"]),
                    "recipe_uri": candidate.get("recipe_uri"),
                    "recipe_sha256": candidate.get("recipe_sha256"),
                    "artifact_aliases": aliases,
                }
                event_id = hashlib.sha256(_json_bytes(payload)).hexdigest()
                batch = encode_metric_batch(
                    [
                        {
                            "id": 1,
                            "event_id": event_id,
                            "created_at": time.time(),
                            "kind": "projection",
                            "step": step,
                            "payload_json": json.dumps(payload, sort_keys=True, default=str),
                        }
                    ]
                )
                attempt_id = f"fleet-artifact-v2-{train_job_id}"
                cur.execute(
                    """
                    INSERT INTO worker_attempts (
                      attempt_id, train_job_id, task_kind, provider, status,
                      started_at, finished_at
                    ) VALUES (
                      %(attempt_id)s, %(train_job_id)s, 'train', 'fleet-wandb',
                      'succeeded', now(), now()
                    ) ON CONFLICT (attempt_id) DO NOTHING
                    """,
                    {"attempt_id": attempt_id, "train_job_id": train_job_id},
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
                if cur.fetchone() is None:
                    continue
                scheduled += 1
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET live_publication_status = 'pending',
                        live_publication_error = NULL,
                        live_publication_next_retry_at = NULL
                    WHERE id = %(train_job_id)s
                      AND live_publication_status NOT IN ('disabled', 'failed')
                    """,
                    {"train_job_id": train_job_id},
                )
    return {"scheduled": scheduled, "opted_out": opted_out}


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
                        INSERT INTO metric_streams (
                          stream_id, attempt_id, accepted_sequence, final_sequence,
                          submitted_sequence, published_sequence
                        ) VALUES (
                          %(attempt_id)s, %(attempt_id)s, %(final_sequence)s,
                          %(final_sequence)s, %(final_sequence)s, %(final_sequence)s
                        )
                        ON CONFLICT (stream_id) DO NOTHING
                        """,
                        {
                            "attempt_id": str(row["attempt_id"]),
                            "final_sequence": final_sequence,
                        },
                    )
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
