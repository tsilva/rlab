from __future__ import annotations

import json
import hashlib
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from rlab.file_utils import file_sha256 as _file_sha256
from rlab.metric_names import METRICS_SCHEMA_VERSION, validate_metric_payload


file_sha256 = _file_sha256


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checkpoints (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_name TEXT NOT NULL,
  kind TEXT NOT NULL,
  step INTEGER,
  path TEXT NOT NULL UNIQUE,
  metadata_path TEXT,
  sha256 TEXT,
  status TEXT NOT NULL DEFAULT 'ready',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS checkpoints_status_idx
  ON checkpoints (status, kind, step);

CREATE TABLE IF NOT EXISTS artifact_uploads (
  checkpoint_id INTEGER PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'pending',
  artifact_ref TEXT,
  storage_uri TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  FOREIGN KEY(checkpoint_id) REFERENCES checkpoints(id)
);

CREATE INDEX IF NOT EXISTS artifact_uploads_status_idx
  ON artifact_uploads (status, updated_at);

CREATE TABLE IF NOT EXISTS eval_results (
  checkpoint_id INTEGER PRIMARY KEY,
  status TEXT NOT NULL DEFAULT 'pending',
  episodes INTEGER,
  metrics_json TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  FOREIGN KEY(checkpoint_id) REFERENCES checkpoints(id)
);

CREATE INDEX IF NOT EXISTS eval_results_status_idx
  ON eval_results (status, updated_at);

CREATE TABLE IF NOT EXISTS checkpoint_eval_stages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  checkpoint_id INTEGER NOT NULL,
  stage_name TEXT NOT NULL,
  stage_index INTEGER NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  episodes INTEGER,
  n_envs INTEGER,
  metrics_json TEXT,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(checkpoint_id, stage_name),
  FOREIGN KEY(checkpoint_id) REFERENCES checkpoints(id)
);

CREATE INDEX IF NOT EXISTS checkpoint_eval_stages_status_idx
  ON checkpoint_eval_stages (status, stage_index, updated_at);

CREATE TABLE IF NOT EXISTS metric_frames (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  step INTEGER,
  source TEXT NOT NULL,
  kind TEXT NOT NULL DEFAULT 'history',
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  published_at REAL,
  relayed_at REAL,
  mailbox_batch_sequence INTEGER,
  archived_at REAL,
  archive_root_sha256 TEXT
);

CREATE INDEX IF NOT EXISTS metric_frames_status_idx
  ON metric_frames (status, id);

CREATE TABLE IF NOT EXISTS metric_latest (
  name TEXT PRIMARY KEY,
  value REAL NOT NULL,
  step INTEGER,
  source TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_state (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  local_latest_step INTEGER,
  published_step INTEGER,
  mailbox_accepted_sequence INTEGER NOT NULL DEFAULT 0,
  outbox_closed_at REAL,
  producer_terminal_sequence INTEGER,
  producer_terminal_sha256 TEXT,
  publisher_heartbeat REAL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS telemetry_recovery_manifest (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  manifest_sha256 TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending',
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
"""


FINAL_LIKE_KINDS = {"final", "interrupted"}


class MetricStore:
    def __init__(self, path: Path | str, *, timeout: float = 5.0) -> None:
        self.path = Path(path)
        self.timeout = timeout

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=self.timeout)
        conn.row_factory = sqlite3.Row
        conn.execute(f"PRAGMA busy_timeout={max(0, int(self.timeout * 1000))}")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init(self) -> None:
        with self.connection() as conn:
            # Journal mode is persistent database state. Reapplying it on every
            # short-lived connection requires an exclusive lock and can make a
            # healthy WAL workload fail spuriously under concurrent writers.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA_SQL)
            telemetry_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(telemetry_state)")
            }
            if "mailbox_accepted_sequence" not in telemetry_columns:
                conn.execute(
                    "ALTER TABLE telemetry_state ADD COLUMN "
                    "mailbox_accepted_sequence INTEGER NOT NULL DEFAULT 0"
                )
            if "outbox_closed_at" not in telemetry_columns:
                conn.execute("ALTER TABLE telemetry_state ADD COLUMN outbox_closed_at REAL")
            if "producer_terminal_sequence" not in telemetry_columns:
                conn.execute(
                    "ALTER TABLE telemetry_state ADD COLUMN producer_terminal_sequence INTEGER"
                )
            if "producer_terminal_sha256" not in telemetry_columns:
                conn.execute(
                    "ALTER TABLE telemetry_state ADD COLUMN producer_terminal_sha256 TEXT"
                )
            frame_columns = {
                str(row[1]) for row in conn.execute("PRAGMA table_info(metric_frames)")
            }
            for name, declaration in (
                ("relayed_at", "REAL"),
                ("mailbox_batch_sequence", "INTEGER"),
                ("archived_at", "REAL"),
                ("archive_root_sha256", "TEXT"),
            ):
                if name not in frame_columns:
                    conn.execute(f"ALTER TABLE metric_frames ADD COLUMN {name} {declaration}")
            conn.execute("BEGIN IMMEDIATE")
            legacy_metrics = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'metric_observations'"
            ).fetchone()
            if legacy_metrics is not None:
                conn.execute(
                    """
                    INSERT INTO metric_latest (name, value, step, source, updated_at)
                    SELECT observations.name, observations.value, observations.step,
                           observations.source, observations.created_at
                    FROM metric_observations AS observations
                    JOIN (
                      SELECT name, MAX(id) AS id
                      FROM metric_observations
                      GROUP BY name
                    ) AS latest
                      ON latest.id = observations.id
                    ON CONFLICT(name) DO NOTHING
                    """
                )
                conn.execute("DROP TABLE metric_observations")

    def append_metrics(
        self,
        metrics: Mapping[str, object],
        *,
        step: int | None,
        source: str,
        created_at: float | None = None,
        publish: bool = True,
        schema_version: int = METRICS_SCHEMA_VERSION,
    ) -> int:
        payload: dict[str, float] = {}
        now = time.time() if created_at is None else created_at
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            payload[str(name)] = numeric
        if not payload:
            return 0
        validate_metric_payload(payload, schema_version=schema_version)
        payload.setdefault("global_step", float(step) if step is not None else 0.0)
        event_id = self._metric_event_id(source=source, step=step, payload=payload)
        with self.connection() as conn:
            self._require_open_outbox(conn)
            if publish:
                conn.execute(
                    """
                    INSERT INTO metric_frames
                      (event_id, step, source, kind, payload_json, status, created_at, updated_at)
                    VALUES (?, ?, ?, 'history', ?, 'pending', ?, ?)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    (event_id, step, source, json.dumps(payload, sort_keys=True), now, now),
                )
            conn.executemany(
                """
                INSERT INTO metric_latest (name, value, step, source, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                  value = excluded.value,
                  step = excluded.step,
                  source = excluded.source,
                  updated_at = excluded.updated_at
                """,
                [(name, value, step, source, now) for name, value in payload.items()],
            )
            conn.execute(
                """
                INSERT INTO telemetry_state (singleton, local_latest_step, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  local_latest_step = MAX(COALESCE(local_latest_step, 0), COALESCE(excluded.local_latest_step, 0)),
                  updated_at = excluded.updated_at
                """,
                (step, now),
            )
        return len(payload) - (1 if "global_step" not in metrics else 0)

    @staticmethod
    def _metric_event_id(*, source: str, step: int | None, payload: Mapping[str, object]) -> str:
        encoded = json.dumps(
            {"source": source, "step": step, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def enqueue_event(
        self,
        *,
        kind: str,
        payload: Mapping[str, object],
        step: int | None,
        source: str,
        event_id: str | None = None,
        created_at: float | None = None,
    ) -> str:
        now = time.time() if created_at is None else created_at
        normalized = dict(payload)
        if step is not None:
            normalized.setdefault("global_step", step)
        event_id = event_id or self._metric_event_id(
            source=source, step=step, payload={"kind": kind, **normalized}
        )
        with self.connection() as conn:
            self._require_open_outbox(conn)
            conn.execute(
                """
                INSERT INTO metric_frames
                  (event_id, step, source, kind, payload_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    event_id,
                    step,
                    source,
                    kind,
                    json.dumps(normalized, sort_keys=True, default=str),
                    now,
                    now,
                ),
            )
        return event_id

    @staticmethod
    def _require_open_outbox(conn: sqlite3.Connection) -> None:
        row = conn.execute(
            "SELECT outbox_closed_at FROM telemetry_state WHERE singleton = 1"
        ).fetchone()
        if row is not None and row[0] is not None:
            raise RuntimeError("metric outbox is closed")

    def close_metric_outbox(self) -> int:
        now = time.time()
        with self.connection() as conn:
            existing = conn.execute(
                "SELECT outbox_closed_at FROM telemetry_state WHERE singleton = 1"
            ).fetchone()
            if existing is None or existing[0] is None:
                rows = conn.execute(
                    """
                    SELECT id, event_id, kind, payload_json
                    FROM metric_frames ORDER BY id
                    """
                ).fetchall()
                chain = hashlib.sha256()
                for row in rows:
                    chain.update(
                        json.dumps(
                            {
                                "id": int(row["id"]),
                                "event_id": str(row["event_id"]),
                                "kind": str(row["kind"]),
                                "payload_json": str(row["payload_json"]),
                            },
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                terminal_sequence = int(rows[-1]["id"]) if rows else 0
                terminal_sha256 = chain.hexdigest()
            else:
                terminal = conn.execute(
                    """
                    SELECT producer_terminal_sequence, producer_terminal_sha256
                    FROM telemetry_state WHERE singleton = 1
                    """
                ).fetchone()
                terminal_sequence = int((terminal or [0])[0] or 0)
                terminal_sha256 = str((terminal or [None, ""])[1] or "")
            conn.execute(
                """
                INSERT INTO telemetry_state (
                  singleton, outbox_closed_at, producer_terminal_sequence,
                  producer_terminal_sha256, updated_at
                )
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  outbox_closed_at = COALESCE(outbox_closed_at, excluded.outbox_closed_at),
                  producer_terminal_sequence = COALESCE(
                    producer_terminal_sequence, excluded.producer_terminal_sequence
                  ),
                  producer_terminal_sha256 = COALESCE(
                    producer_terminal_sha256, excluded.producer_terminal_sha256
                  ),
                  updated_at = excluded.updated_at
                """,
                (now, terminal_sequence, terminal_sha256, now),
            )
            row = conn.execute(
                "SELECT COUNT(*) FROM metric_frames "
                "WHERE status IN ('pending', 'failed_retryable')"
            ).fetchone()
        return int(row[0] if row else 0)

    def next_mailbox_batch_sequence(self) -> int:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT mailbox_accepted_sequence FROM telemetry_state WHERE singleton = 1"
            ).fetchone()
        return int((row[0] if row else 0) or 0) + 1

    def mark_metric_frames_delivered(
        self,
        frame_ids: Sequence[int],
        *,
        batch_sequence: int,
    ) -> None:
        now = time.time()
        ids = [int(value) for value in frame_ids]
        with self.connection() as conn:
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"""
                    UPDATE metric_frames
                    SET status = 'relayed', relayed_at = ?,
                        mailbox_batch_sequence = ?, last_error = NULL, updated_at = ?
                    WHERE id IN ({placeholders})
                    """,
                    [now, int(batch_sequence), now, *ids],
                )
            conn.execute(
                """
                INSERT INTO telemetry_state
                  (singleton, mailbox_accepted_sequence, publisher_heartbeat, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  mailbox_accepted_sequence = MAX(
                    mailbox_accepted_sequence, excluded.mailbox_accepted_sequence
                  ),
                  publisher_heartbeat = excluded.publisher_heartbeat,
                  last_error = NULL,
                  updated_at = excluded.updated_at
                """,
                (int(batch_sequence), now, now),
            )

    def metric_outbox_bytes(self) -> int:
        return self.metric_outbox_stats()["bytes"]

    def metric_outbox_stats(self) -> dict[str, int]:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS frame_count,
                       COALESCE(SUM(
                         length(payload_json) + length(event_id) + length(source) + 128
                       ), 0) AS byte_count
                FROM metric_frames
                WHERE status IN ('pending', 'failed_retryable')
                """
            ).fetchone()
        return {
            "frames": int(row[0] if row else 0),
            "bytes": int(row[1] if row else 0),
        }

    def register_recovery_manifest(self, manifest: Mapping[str, object]) -> str:
        """Persist a secret-free recovery contract before external credentials are used."""

        forbidden = {
            key
            for key in manifest
            if any(token in str(key).lower() for token in ("token", "secret", "password", "api_key"))
        }
        if forbidden:
            raise ValueError(f"recovery manifest contains secret-like fields: {sorted(forbidden)}")
        normalized = dict(manifest)
        payload = json.dumps(
            normalized, sort_keys=True, separators=(",", ":"), default=str
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO telemetry_recovery_manifest (
                  singleton, manifest_sha256, manifest_json, state, created_at, updated_at
                ) VALUES (1, ?, ?, 'pending', ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  manifest_sha256 = CASE
                    WHEN telemetry_recovery_manifest.manifest_sha256 = excluded.manifest_sha256
                    THEN excluded.manifest_sha256
                    ELSE telemetry_recovery_manifest.manifest_sha256
                  END,
                  updated_at = excluded.updated_at
                """,
                (digest, payload, now, now),
            )
            row = conn.execute(
                "SELECT manifest_sha256 FROM telemetry_recovery_manifest WHERE singleton = 1"
            ).fetchone()
            if not row or str(row[0]) != digest:
                raise RuntimeError("recovery manifest conflicts with the registered run")
        return digest

    def authorize_archived_frame_cleanup(self, *, archive_root_sha256: str) -> int:
        digest = str(archive_root_sha256).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("archive root must be a lowercase SHA-256")
        now = time.time()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE metric_frames
                SET archived_at = COALESCE(archived_at, ?),
                    archive_root_sha256 = ?
                WHERE status IN ('relayed', 'published', 'failed_terminal')
                """,
                (now, digest),
            )
        return int(cursor.rowcount)

    def delete_archived_frames(self, *, archive_root_sha256: str) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM metric_frames
                WHERE archived_at IS NOT NULL AND archive_root_sha256 = ?
                """,
                (str(archive_root_sha256),),
            )
        return int(cursor.rowcount)

    def pending_mailbox_frames(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM metric_frames
                WHERE status IN ('pending', 'failed_retryable')
                ORDER BY id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def pending_metric_frames(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM metric_frames
                WHERE status IN ('pending', 'failed_retryable')
                ORDER BY CASE WHEN kind = 'checkpoint_preview' THEN 1 ELSE 0 END, id
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_metric_frame(self, frame_id: int) -> bool:
        now = time.time()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE metric_frames
                SET status = 'publishing', attempts = attempts + 1, updated_at = ?
                WHERE id = ? AND status IN ('pending', 'failed_retryable')
                """,
                (now, frame_id),
            )
        return cursor.rowcount == 1

    def mark_metric_frame_published(self, frame_id: int, *, step: int | None) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE metric_frames
                SET status = 'published', last_error = NULL, published_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, now, frame_id),
            )
            conn.execute(
                """
                INSERT INTO telemetry_state
                  (singleton, published_step, publisher_heartbeat, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  published_step = MAX(COALESCE(published_step, 0), COALESCE(excluded.published_step, 0)),
                  publisher_heartbeat = excluded.publisher_heartbeat,
                  last_error = NULL,
                  updated_at = excluded.updated_at
                """,
                (step, now, now),
            )

    def mark_metric_frame_failed(self, frame_id: int, error: str) -> None:
        now = time.time()
        message = error[:4000]
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE metric_frames
                SET status = 'failed_retryable', last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (message, now, frame_id),
            )
            conn.execute(
                """
                INSERT INTO telemetry_state
                  (singleton, publisher_heartbeat, retry_count, last_error, updated_at)
                VALUES (1, ?, 1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  publisher_heartbeat = excluded.publisher_heartbeat,
                  retry_count = retry_count + 1,
                  last_error = excluded.last_error,
                  updated_at = excluded.updated_at
                """,
                (now, message, now),
            )

    def mark_metric_frame_terminal_failure(self, frame_id: int, error: str) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE metric_frames
                SET status = 'failed_terminal', last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (error[:4000], now, frame_id),
            )

    def reset_interrupted_metric_frames(self) -> int:
        now = time.time()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE metric_frames
                SET status = 'failed_retryable', last_error = 'publisher interrupted', updated_at = ?
                WHERE status = 'publishing'
                """,
                (now,),
            )
        return cursor.rowcount

    def telemetry_health(self) -> dict[str, object]:
        with self.connection() as conn:
            state = conn.execute("SELECT * FROM telemetry_state WHERE singleton = 1").fetchone()
            backlog = conn.execute(
                """
                SELECT count(*) AS pending_frames, min(created_at) AS oldest_created_at
                FROM metric_frames
                WHERE status IN ('pending', 'failed_retryable', 'publishing')
                """
            ).fetchone()
        result = dict(state) if state is not None else {}
        result.update(dict(backlog))
        return result

    def touch_publisher(self) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO telemetry_state (singleton, publisher_heartbeat, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  publisher_heartbeat = excluded.publisher_heartbeat,
                  updated_at = excluded.updated_at
                """,
                (now, now),
            )

    def record_publisher_error(self, error: str) -> None:
        now = time.time()
        message = error[:4000]
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO telemetry_state
                  (singleton, publisher_heartbeat, retry_count, last_error, updated_at)
                VALUES (1, ?, 1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  publisher_heartbeat = excluded.publisher_heartbeat,
                  retry_count = retry_count + 1,
                  last_error = excluded.last_error,
                  updated_at = excluded.updated_at
                """,
                (now, message, now),
            )

    def latest_metric(self, name: str) -> float | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT value FROM metric_latest WHERE name = ?
                """,
                (name,),
            ).fetchone()
        return None if row is None else float(row["value"])

    def result_projection(self) -> dict[str, Any]:
        """Project durable run-result evidence from the structured ledger."""

        with self.connection() as conn:
            metric_rows = conn.execute(
                "SELECT name, value FROM metric_latest ORDER BY name"
            ).fetchall()
            artifact_rows = conn.execute(
                """
                SELECT c.kind, c.path, u.artifact_ref, u.storage_uri
                FROM artifact_uploads AS u
                JOIN checkpoints AS c ON c.id = u.checkpoint_id
                WHERE u.status = 'uploaded'
                ORDER BY c.id
                """
            ).fetchall()
        metrics: dict[str, int | float] = {}
        for row in metric_rows:
            value = float(row["value"])
            metrics[str(row["name"])] = int(value) if value.is_integer() else value
        artifact_refs = []
        for row in artifact_rows:
            artifact_ref = str(row["artifact_ref"] or "").strip()
            collection = artifact_ref.rsplit("/", 1)[-1].rsplit(":", 1)[0]
            artifact_refs.append(
                {
                    "name": collection or str(row["kind"]),
                    "location": str(row["storage_uri"] or row["path"]),
                    "artifact_ref": artifact_ref or None,
                }
            )
        return {
            "artifact_refs": artifact_refs,
            "metrics_json": metrics,
            "phase_counts": self.phase_counts(),
            "telemetry_health": self.telemetry_health(),
        }

    def reset_interrupted_artifact_uploads(self) -> int:
        now = time.time()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE artifact_uploads
                SET status = 'failed_retryable', last_error = 'publisher interrupted', updated_at = ?
                WHERE status = 'running'
                """,
                (now,),
            )
        return cursor.rowcount

    def requeue_uploaded_artifacts_for_recovery(self) -> int:
        """Recheck staged objects during explicit recovery without changing the ledger."""

        now = time.time()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE artifact_uploads
                SET status = 'failed_retryable',
                    last_error = 'explicit recovery remote recheck',
                    updated_at = ?
                WHERE status = 'uploaded'
                """,
                (now,),
            )
        return cursor.rowcount

    def record_checkpoint(
        self,
        *,
        run_name: str,
        kind: str,
        step: int | None,
        path: Path | str,
        metadata_path: Path | str | None,
        sha256: str | None = None,
        status: str = "ready",
        created_at: float | None = None,
        eval_required: bool = True,
    ) -> int:
        now = time.time() if created_at is None else created_at
        path_text = str(path)
        metadata_text = str(metadata_path) if metadata_path is not None else None
        with self.connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE path = ?",
                (path_text,),
            ).fetchone()
            if row is None:
                cursor = conn.execute(
                    """
                    INSERT INTO checkpoints
                      (run_name, kind, step, path, metadata_path, sha256, status,
                       created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_name,
                        kind,
                        step,
                        path_text,
                        metadata_text,
                        sha256,
                        status,
                        now,
                        now,
                    ),
                )
                checkpoint_id = int(cursor.lastrowid)
            else:
                expected = {
                    "run_name": str(run_name),
                    "kind": str(kind),
                    "step": step,
                    "metadata_path": metadata_text,
                    "status": str(status),
                }
                mismatches = [
                    key
                    for key, value in expected.items()
                    if row[key] != value
                ]
                if sha256 is not None and row["sha256"] not in (None, sha256):
                    mismatches.append("sha256")
                if mismatches:
                    raise ValueError(
                        "checkpoint ledger replay conflicts for "
                        f"{path_text}: {', '.join(sorted(set(mismatches)))}"
                    )
                checkpoint_id = int(row["id"])
            conn.execute(
                """
                INSERT INTO artifact_uploads (checkpoint_id, status, created_at, updated_at)
                VALUES (?, 'pending', ?, ?)
                ON CONFLICT(checkpoint_id) DO NOTHING
                """,
                (checkpoint_id, now, now),
            )
            if eval_required and kind not in FINAL_LIKE_KINDS:
                conn.execute(
                    """
                    INSERT INTO eval_results (checkpoint_id, status, created_at, updated_at)
                    VALUES (?, 'pending', ?, ?)
                    ON CONFLICT(checkpoint_id) DO NOTHING
                    """,
                    (checkpoint_id, now, now),
                )
        return checkpoint_id

    def pending_artifact_uploads(self, *, limit: int = 10) -> list[dict[str, Any]]:
        return self._pending_rows(
            table="artifact_uploads",
            statuses=("pending", "failed_retryable"),
            limit=limit,
        )

    def checkpoint(self, checkpoint_id: int) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM checkpoints WHERE id = ?", (checkpoint_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def checkpoints(self) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM checkpoints ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def set_checkpoint_sha256(self, checkpoint_id: int, sha256: str) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                "UPDATE checkpoints SET sha256 = ?, updated_at = ? WHERE id = ?",
                (str(sha256), now, checkpoint_id),
            )

    def checkpoint_eval_stage_status(self, checkpoint_id: int, stage_name: str) -> str | None:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT status FROM checkpoint_eval_stages WHERE checkpoint_id = ? AND stage_name = ?",
                (checkpoint_id, stage_name),
            ).fetchone()
        return str(row["status"]) if row is not None else None

    def mark_artifact_terminal_failure(
        self, checkpoint_id: int, *, error: str, storage_uri: str | None = None
    ) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE artifact_uploads
                SET status = 'failed_terminal', storage_uri = ?, last_error = ?, updated_at = ?
                WHERE checkpoint_id = ?
                """,
                (storage_uri, error[:4000], now, checkpoint_id),
            )

    def apply_modal_eval_decision(
        self,
        checkpoint_id: int,
        *,
        stage_name: str,
        stage_index: int,
        episodes: int,
        n_envs: int,
        metrics: Mapping[str, object],
        passed: bool,
        candidate_stop: bool,
        publish: bool,
    ) -> None:
        now = time.time()
        stage_status = "succeeded" if passed else "failed_gate"
        with self.connection() as conn:
            row = conn.execute(
                "SELECT id FROM checkpoint_eval_stages WHERE checkpoint_id = ? AND stage_name = ?",
                (checkpoint_id, stage_name),
            ).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO checkpoint_eval_stages
                      (checkpoint_id, stage_name, stage_index, status, episodes, n_envs,
                       metrics_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        checkpoint_id,
                        stage_name,
                        stage_index,
                        stage_status,
                        episodes,
                        n_envs,
                        json.dumps(metrics, sort_keys=True, default=str),
                        now,
                        now,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE checkpoint_eval_stages
                    SET status = ?, episodes = ?, n_envs = ?, metrics_json = ?,
                        last_error = NULL, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        stage_status,
                        episodes,
                        n_envs,
                        json.dumps(metrics, sort_keys=True, default=str),
                        now,
                        int(row["id"]),
                    ),
                )
            eval_status = "candidate" if passed and candidate_stop else "non_candidate"
            conn.execute(
                """
                UPDATE eval_results
                SET status = ?, episodes = ?, metrics_json = ?, last_error = NULL, updated_at = ?
                WHERE checkpoint_id = ?
                """,
                (
                    eval_status,
                    episodes,
                    json.dumps(metrics, sort_keys=True, default=str),
                    now,
                    checkpoint_id,
                ),
            )
        checkpoint = self.checkpoint(checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"checkpoint ledger row disappeared: {checkpoint_id}")
        checkpoint_step = int(checkpoint.get("step") or 0)
        self.append_metrics(
            metrics,
            step=checkpoint_step,
            source="modal_checkpoint_eval",
            publish=bool(publish),
            schema_version=4,
        )

    def apply_modal_eval_skip(
        self,
        checkpoint_id: int,
        *,
        stage_name: str,
        stage_index: int,
        episodes: int,
        n_envs: int,
    ) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoint_eval_stages
                  (checkpoint_id, stage_name, stage_index, status, episodes, n_envs,
                   created_at, updated_at)
                VALUES (?, ?, ?, 'skipped_stale', ?, ?, ?, ?)
                ON CONFLICT(checkpoint_id, stage_name) DO UPDATE SET
                  status = 'skipped_stale', updated_at = excluded.updated_at
                """,
                (checkpoint_id, stage_name, stage_index, episodes, n_envs, now, now),
            )
            conn.execute(
                """
                UPDATE eval_results SET status = 'skipped_stale', last_error = NULL, updated_at = ?
                WHERE checkpoint_id = ?
                """,
                (now, checkpoint_id),
            )

    def pending_evals(self, *, limit: int = 1) -> list[dict[str, Any]]:
        return self._pending_rows(
            table="eval_results",
            statuses=("pending", "failed_retryable"),
            limit=limit,
        )

    def ensure_checkpoint_eval_stages(
        self,
        stages: Sequence[Mapping[str, Any]],
    ) -> None:
        if not stages:
            return
        now = time.time()
        first_stage = stages[0]
        with self.connection() as conn:
            checkpoint_rows = conn.execute(
                """
                SELECT c.id
                FROM eval_results r
                JOIN checkpoints c ON c.id = r.checkpoint_id
                LEFT JOIN checkpoint_eval_stages s ON s.checkpoint_id = c.id
                WHERE c.status = 'ready'
                  AND r.status IN ('pending', 'failed_retryable')
                  AND s.id IS NULL
                """
            ).fetchall()
            conn.executemany(
                """
                INSERT INTO checkpoint_eval_stages
                  (checkpoint_id, stage_name, stage_index, status, episodes, n_envs, created_at, updated_at)
                VALUES (?, ?, 0, 'pending', ?, ?, ?, ?)
                ON CONFLICT(checkpoint_id, stage_name) DO NOTHING
                """,
                [
                    (
                        int(row["id"]),
                        str(first_stage["name"]),
                        int(first_stage["episodes"]),
                        first_stage.get("n_envs"),
                        now,
                        now,
                    )
                    for row in checkpoint_rows
                ],
            )

    def pending_checkpoint_eval_stages(self, *, limit: int = 1) -> list[dict[str, Any]]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                  c.*,
                  s.id AS eval_stage_id,
                  s.stage_name,
                  s.stage_index,
                  s.status AS stage_status,
                  s.episodes AS stage_episodes,
                  s.n_envs AS stage_n_envs,
                  s.attempts AS stage_attempts,
                  s.last_error AS stage_last_error,
                  r.status AS worker_status,
                  r.attempts,
                  r.last_error
                FROM checkpoint_eval_stages s
                JOIN checkpoints c ON c.id = s.checkpoint_id
                JOIN eval_results r ON r.checkpoint_id = c.id
                WHERE c.status = 'ready'
                  AND r.status IN ('pending', 'failed_retryable')
                  AND s.status IN ('pending', 'failed_retryable')
                ORDER BY
                  CASE WHEN s.stage_index > 0 THEN 0 ELSE 1 END,
                  c.step IS NULL,
                  c.step DESC,
                  c.id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def skip_stale_initial_checkpoint_eval_stages(self, *, keep_checkpoint_id: int) -> int:
        now = time.time()
        with self.connection() as conn:
            keep_row = conn.execute(
                "SELECT step FROM checkpoints WHERE id = ?",
                (keep_checkpoint_id,),
            ).fetchone()
            keep_step = None if keep_row is None else keep_row["step"]
            if keep_step is None:
                return 0
            rows = conn.execute(
                """
                SELECT s.checkpoint_id
                FROM checkpoint_eval_stages s
                JOIN checkpoints c ON c.id = s.checkpoint_id
                JOIN eval_results r ON r.checkpoint_id = c.id
                WHERE s.stage_index = 0
                  AND s.status = 'pending'
                  AND s.attempts = 0
                  AND r.status = 'pending'
                  AND c.status = 'ready'
                  AND c.step IS NOT NULL
                  AND c.step < ?
                """,
                (int(keep_step),),
            ).fetchall()
            checkpoint_ids = [int(row["checkpoint_id"]) for row in rows]
            if not checkpoint_ids:
                return 0
            placeholders = ",".join("?" for _ in checkpoint_ids)
            conn.execute(
                f"""
                UPDATE checkpoint_eval_stages
                SET status = 'skipped_stale', updated_at = ?
                WHERE checkpoint_id IN ({placeholders})
                  AND stage_index = 0
                  AND status = 'pending'
                  AND attempts = 0
                """,
                (now, *checkpoint_ids),
            )
            conn.execute(
                f"""
                UPDATE eval_results
                SET status = 'skipped_stale', updated_at = ?
                WHERE checkpoint_id IN ({placeholders})
                  AND status = 'pending'
                """,
                (now, *checkpoint_ids),
            )
        return len(checkpoint_ids)

    def _pending_rows(
        self,
        *,
        table: str,
        statuses: tuple[str, ...],
        limit: int,
    ) -> list[dict[str, Any]]:
        placeholders = ",".join("?" for _ in statuses)
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*, r.status AS worker_status, r.attempts, r.last_error
                FROM {table} r
                JOIN checkpoints c ON c.id = r.checkpoint_id
                WHERE c.status = 'ready' AND r.status IN ({placeholders})
                ORDER BY c.step IS NULL, c.step, c.id
                LIMIT ?
                """,
                (*statuses, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_artifact_upload(self, checkpoint_id: int) -> bool:
        return self._claim("artifact_uploads", checkpoint_id)

    def claim_eval(self, checkpoint_id: int) -> bool:
        return self._claim("eval_results", checkpoint_id)

    def claim_checkpoint_eval_stage(self, stage_id: int) -> bool:
        now = time.time()
        with self.connection() as conn:
            cursor = conn.execute(
                """
                UPDATE checkpoint_eval_stages
                SET status = 'running', attempts = attempts + 1, updated_at = ?
                WHERE id = ? AND status IN ('pending', 'failed_retryable')
                """,
                (now, stage_id),
            )
            changed = cursor.rowcount
        return changed == 1

    def _claim(self, table: str, checkpoint_id: int) -> bool:
        now = time.time()
        with self.connection() as conn:
            cursor = conn.execute(
                f"""
                UPDATE {table}
                SET status = 'running', attempts = attempts + 1, updated_at = ?
                WHERE checkpoint_id = ? AND status IN ('pending', 'failed_retryable')
                """,
                (now, checkpoint_id),
            )
            changed = cursor.rowcount
        return changed == 1

    def mark_artifact_uploaded(
        self,
        checkpoint_id: int,
        *,
        artifact_ref: str | None,
        storage_uri: str | None,
    ) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE artifact_uploads
                SET status = 'uploaded',
                    artifact_ref = ?,
                    storage_uri = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE checkpoint_id = ?
                """,
                (artifact_ref, storage_uri, now, checkpoint_id),
            )

    def mark_artifact_failed(self, checkpoint_id: int, error: str) -> None:
        self._mark_failed("artifact_uploads", checkpoint_id, error)

    def mark_eval_succeeded(
        self,
        checkpoint_id: int,
        *,
        episodes: int,
        metrics: Mapping[str, object],
    ) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE eval_results
                SET status = 'succeeded',
                    episodes = ?,
                    metrics_json = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE checkpoint_id = ?
                """,
                (
                    int(episodes),
                    json.dumps(metrics, sort_keys=True, default=str),
                    now,
                    checkpoint_id,
                ),
            )

    def mark_eval_failed(self, checkpoint_id: int, error: str) -> None:
        self._mark_failed("eval_results", checkpoint_id, error)

    def mark_checkpoint_eval_stage_succeeded(
        self,
        stage_id: int,
        *,
        episodes: int,
        n_envs: int,
        metrics: Mapping[str, object],
    ) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE checkpoint_eval_stages
                SET status = 'succeeded',
                    episodes = ?,
                    n_envs = ?,
                    metrics_json = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    int(episodes),
                    int(n_envs),
                    json.dumps(metrics, sort_keys=True, default=str),
                    now,
                    stage_id,
                ),
            )

    def mark_checkpoint_eval_stage_failed(self, stage_id: int, error: str) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE checkpoint_eval_stages
                SET status = 'failed_retryable',
                    last_error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (error[:4000], now, stage_id),
            )
            row = conn.execute(
                "SELECT checkpoint_id FROM checkpoint_eval_stages WHERE id = ?",
                (stage_id,),
            ).fetchone()
            if row is not None:
                conn.execute(
                    """
                    UPDATE eval_results
                    SET status = 'failed_retryable',
                        last_error = ?,
                        updated_at = ?
                    WHERE checkpoint_id = ?
                    """,
                    (error[:4000], now, int(row["checkpoint_id"])),
                )

    def enqueue_checkpoint_eval_stage(
        self,
        checkpoint_id: int,
        stage: Mapping[str, Any],
        *,
        stage_index: int,
    ) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoint_eval_stages
                  (checkpoint_id, stage_name, stage_index, status, episodes, n_envs, created_at, updated_at)
                VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)
                ON CONFLICT(checkpoint_id, stage_name) DO NOTHING
                """,
                (
                    checkpoint_id,
                    str(stage["name"]),
                    int(stage_index),
                    int(stage["episodes"]),
                    stage.get("n_envs"),
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                UPDATE eval_results
                SET status = 'pending', last_error = NULL, updated_at = ?
                WHERE checkpoint_id = ?
                """,
                (now, checkpoint_id),
            )

    def mark_checkpoint_eval_candidate(
        self,
        checkpoint_id: int,
        *,
        episodes: int,
        metrics: Mapping[str, object],
    ) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE eval_results
                SET status = 'candidate',
                    episodes = ?,
                    metrics_json = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE checkpoint_id = ?
                """,
                (
                    int(episodes),
                    json.dumps(metrics, sort_keys=True, default=str),
                    now,
                    checkpoint_id,
                ),
            )

    def mark_checkpoint_eval_non_candidate(
        self,
        checkpoint_id: int,
        *,
        episodes: int,
        metrics: Mapping[str, object],
    ) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE eval_results
                SET status = 'non_candidate',
                    episodes = ?,
                    metrics_json = ?,
                    last_error = NULL,
                    updated_at = ?
                WHERE checkpoint_id = ?
                """,
                (
                    int(episodes),
                    json.dumps(metrics, sort_keys=True, default=str),
                    now,
                    checkpoint_id,
                ),
            )

    def _mark_failed(self, table: str, checkpoint_id: int, error: str) -> None:
        now = time.time()
        with self.connection() as conn:
            conn.execute(
                f"""
                UPDATE {table}
                SET status = 'failed_retryable',
                    last_error = ?,
                    updated_at = ?
                WHERE checkpoint_id = ?
                """,
                (error[:4000], now, checkpoint_id),
            )

    def phase_counts(self) -> dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT 'checkpoints:' || status AS key, count(*) AS count
                FROM checkpoints
                GROUP BY status
                UNION ALL
                SELECT 'artifacts:' || status AS key, count(*) AS count
                FROM artifact_uploads
                GROUP BY status
                UNION ALL
                SELECT 'evals:' || status AS key, count(*) AS count
                FROM eval_results
                GROUP BY status
                UNION ALL
                SELECT 'eval_stages:' || status AS key, count(*) AS count
                FROM checkpoint_eval_stages
                GROUP BY status
                UNION ALL
                SELECT 'telemetry:' || status AS key, count(*) AS count
                FROM metric_frames
                GROUP BY status
                """
            ).fetchall()
        return {str(row["key"]): int(row["count"]) for row in rows}


def metric_store_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "rlab.sqlite"
