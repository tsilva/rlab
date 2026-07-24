from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from collections.abc import Iterator, Mapping
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
  eval_required INTEGER NOT NULL,
  upload_status TEXT NOT NULL DEFAULT 'pending',
  public_url TEXT,
  last_error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS checkpoints_upload_idx
  ON checkpoints (upload_status, id);

CREATE TABLE IF NOT EXISTS metric_frames (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id TEXT NOT NULL UNIQUE,
  step INTEGER,
  source TEXT NOT NULL,
  kind TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  published_at REAL
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

CREATE TABLE IF NOT EXISTS outbox_state (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  local_latest_step INTEGER,
  published_step INTEGER,
  publisher_heartbeat REAL,
  retry_count INTEGER NOT NULL DEFAULT 0,
  last_error TEXT,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS recovery_manifest (
  singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
  manifest_sha256 TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);
"""


class MetricStore:
    """Container-local SQLite WAL outbox shared only by learner and supervisor."""

    def __init__(self, path: Path | str, *, timeout: float = 5.0) -> None:
        self.path = Path(path)
        self.timeout = timeout

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={max(0, int(self.timeout * 1000))}")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def init(self) -> None:
        with self.connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.executescript(SCHEMA_SQL)

    @staticmethod
    def _event_id(
        *,
        source: str,
        step: int | None,
        kind: str,
        payload: Mapping[str, object],
    ) -> str:
        encoded = json.dumps(
            {
                "source": source,
                "step": step,
                "kind": kind,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

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
        payload = {
            str(name): float(value)
            for name, value in metrics.items()
            if not isinstance(value, bool)
            and isinstance(value, int | float)
            and math.isfinite(float(value))
        }
        if not payload:
            return 0
        validate_metric_payload(payload, schema_version=schema_version)
        payload.setdefault("global_step", float(step or 0))
        now = time.time() if created_at is None else float(created_at)
        event_id = self._event_id(
            source=source,
            step=step,
            kind="history",
            payload=payload,
        )
        with self.connection() as connection:
            if publish:
                connection.execute(
                    """
                    INSERT INTO metric_frames
                      (event_id, step, source, kind, payload_json, status,
                       created_at, updated_at)
                    VALUES (?, ?, ?, 'history', ?, 'pending', ?, ?)
                    ON CONFLICT(event_id) DO NOTHING
                    """,
                    (
                        event_id,
                        step,
                        source,
                        json.dumps(payload, sort_keys=True),
                        now,
                        now,
                    ),
                )
            connection.executemany(
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
            connection.execute(
                """
                INSERT INTO outbox_state (singleton, local_latest_step, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  local_latest_step = MAX(
                    COALESCE(local_latest_step, 0),
                    COALESCE(excluded.local_latest_step, 0)
                  ),
                  updated_at = excluded.updated_at
                """,
                (step, now),
            )
        return len(payload) - (0 if "global_step" in metrics else 1)

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
        normalized = dict(payload)
        if step is not None:
            normalized.setdefault("global_step", step)
        identity = event_id or self._event_id(
            source=source,
            step=step,
            kind=kind,
            payload=normalized,
        )
        now = time.time() if created_at is None else float(created_at)
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO metric_frames
                  (event_id, step, source, kind, payload_json, status,
                   created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(event_id) DO NOTHING
                """,
                (
                    identity,
                    step,
                    source,
                    kind,
                    json.dumps(normalized, sort_keys=True, default=str),
                    now,
                    now,
                ),
            )
        return identity

    def pending_metric_frames(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                """
                SELECT * FROM metric_frames
                WHERE status IN ('pending', 'failed_retryable')
                ORDER BY id
                LIMIT ?
                """,
                (int(limit),),
            ).fetchall()
        return [dict(row) for row in rows]

    def claim_metric_frame(self, frame_id: int) -> bool:
        now = time.time()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE metric_frames
                SET status = 'publishing', attempts = attempts + 1, updated_at = ?
                WHERE id = ? AND status IN ('pending', 'failed_retryable')
                """,
                (now, int(frame_id)),
            )
        return cursor.rowcount == 1

    def mark_metric_frame_published(
        self,
        frame_id: int,
        *,
        step: int | None,
    ) -> None:
        now = time.time()
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE metric_frames
                SET status = 'published', last_error = NULL,
                    published_at = ?, updated_at = ?
                WHERE id = ? AND status = 'publishing'
                """,
                (now, now, int(frame_id)),
            )
            connection.execute(
                """
                INSERT INTO outbox_state
                  (singleton, published_step, publisher_heartbeat, updated_at)
                VALUES (1, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  published_step = MAX(
                    COALESCE(published_step, 0),
                    COALESCE(excluded.published_step, 0)
                  ),
                  publisher_heartbeat = excluded.publisher_heartbeat,
                  last_error = NULL,
                  updated_at = excluded.updated_at
                """,
                (step, now, now),
            )

    def mark_metric_frame_failed(self, frame_id: int, error: str) -> None:
        now = time.time()
        message = str(error)[:4000]
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE metric_frames
                SET status = 'failed_retryable', last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (message, now, int(frame_id)),
            )
            connection.execute(
                """
                INSERT INTO outbox_state
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

    def reset_interrupted_metric_frames(self) -> int:
        now = time.time()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE metric_frames
                SET status = 'failed_retryable',
                    last_error = 'publisher interrupted',
                    updated_at = ?
                WHERE status = 'publishing'
                """,
                (now,),
            )
        return int(cursor.rowcount)

    def metric_outbox_stats(self) -> dict[str, int]:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*),
                       COALESCE(SUM(length(payload_json) + length(event_id) + 128), 0)
                FROM metric_frames
                WHERE status IN ('pending', 'failed_retryable', 'publishing')
                """
            ).fetchone()
        return {
            "frames": int(row[0] if row else 0),
            "bytes": int(row[1] if row else 0),
        }

    def outbox_health(self) -> dict[str, object]:
        with self.connection() as connection:
            state = connection.execute(
                "SELECT * FROM outbox_state WHERE singleton = 1"
            ).fetchone()
            backlog = connection.execute(
                """
                SELECT COUNT(*) AS pending_frames,
                       MIN(created_at) AS oldest_created_at
                FROM metric_frames
                WHERE status IN ('pending', 'failed_retryable', 'publishing')
                """
            ).fetchone()
        result = dict(state) if state is not None else {}
        result.update(dict(backlog))
        return result

    def latest_metric(self, name: str) -> float | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT value FROM metric_latest WHERE name = ?",
                (str(name),),
            ).fetchone()
        return None if row is None else float(row["value"])

    def register_recovery_manifest(self, manifest: Mapping[str, object]) -> str:
        forbidden = {
            key
            for key in manifest
            if any(
                token in str(key).lower()
                for token in ("token", "secret", "password", "api_key")
            )
        }
        if forbidden:
            raise ValueError(
                f"recovery manifest contains secret-like fields: {sorted(forbidden)}"
            )
        payload = json.dumps(
            dict(manifest),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        digest = hashlib.sha256(payload.encode()).hexdigest()
        now = time.time()
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO recovery_manifest
                  (singleton, manifest_sha256, manifest_json, created_at, updated_at)
                VALUES (1, ?, ?, ?, ?)
                ON CONFLICT(singleton) DO UPDATE SET
                  updated_at = excluded.updated_at
                """,
                (digest, payload, now, now),
            )
            row = connection.execute(
                "SELECT manifest_sha256 FROM recovery_manifest WHERE singleton = 1"
            ).fetchone()
            if row is None or str(row[0]) != digest:
                raise RuntimeError("recovery manifest conflicts with the registered run")
        return digest

    def record_checkpoint(
        self,
        *,
        run_name: str,
        kind: str,
        step: int | None,
        path: Path | str,
        metadata_path: Path | str | None,
        sha256: str | None = None,
        created_at: float | None = None,
        eval_required: bool = True,
    ) -> int:
        now = time.time() if created_at is None else float(created_at)
        values = {
            "run_name": str(run_name),
            "kind": str(kind),
            "step": step,
            "path": str(path),
            "metadata_path": (
                str(metadata_path) if metadata_path is not None else None
            ),
            "sha256": sha256,
            "eval_required": int(bool(eval_required)),
        }
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM checkpoints WHERE path = ?",
                (values["path"],),
            ).fetchone()
            if row is None:
                cursor = connection.execute(
                    """
                    INSERT INTO checkpoints
                      (run_name, kind, step, path, metadata_path, sha256,
                       eval_required, upload_status, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                    """,
                    (
                        values["run_name"],
                        values["kind"],
                        values["step"],
                        values["path"],
                        values["metadata_path"],
                        values["sha256"],
                        values["eval_required"],
                        now,
                        now,
                    ),
                )
                return int(cursor.lastrowid)
            for name in (
                "run_name",
                "kind",
                "step",
                "metadata_path",
                "eval_required",
            ):
                if row[name] != values[name]:
                    raise ValueError(
                        f"checkpoint ledger replay conflicts for {values['path']}: {name}"
                    )
            if sha256 is not None and row["sha256"] not in (None, sha256):
                raise ValueError(
                    f"checkpoint ledger replay conflicts for {values['path']}: sha256"
                )
            return int(row["id"])

    def checkpoints(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT * FROM checkpoints ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_checkpoint_uploaded(self, checkpoint_id: int, public_url: str) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE checkpoints
                SET upload_status = 'uploaded', public_url = ?,
                    last_error = NULL, updated_at = ?
                WHERE id = ?
                """,
                (str(public_url), time.time(), int(checkpoint_id)),
            )

    def mark_checkpoint_upload_failed(self, checkpoint_id: int, error: str) -> None:
        with self.connection() as connection:
            connection.execute(
                """
                UPDATE checkpoints
                SET upload_status = 'failed_retryable', last_error = ?, updated_at = ?
                WHERE id = ?
                """,
                (str(error)[:4000], time.time(), int(checkpoint_id)),
            )


def metric_store_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "rlab.sqlite"
