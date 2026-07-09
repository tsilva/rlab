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

CREATE TABLE IF NOT EXISTS metric_observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  value REAL NOT NULL,
  step INTEGER,
  source TEXT NOT NULL,
  checkpoint_step INTEGER,
  created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS metric_observations_name_id_idx
  ON metric_observations (name, id DESC);

CREATE INDEX IF NOT EXISTS metric_observations_source_idx
  ON metric_observations (source, id DESC);
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
        conn.execute("PRAGMA journal_mode=WAL")
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
            conn.executescript(SCHEMA_SQL)

    def append_metrics(
        self,
        metrics: Mapping[str, object],
        *,
        step: int | None,
        source: str,
        checkpoint_step: int | None = None,
        created_at: float | None = None,
    ) -> int:
        rows: list[tuple[str, float, int | None, str, int | None, float]] = []
        now = time.time() if created_at is None else created_at
        for name, value in metrics.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            rows.append((str(name), numeric, step, source, checkpoint_step, now))
        if not rows:
            return 0
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO metric_observations
                  (name, value, step, source, checkpoint_step, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def latest_metric(self, name: str) -> float | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT value FROM metric_observations
                WHERE name = ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (name,),
            ).fetchone()
        return None if row is None else float(row["value"])

    def latest_metrics(self, names: Sequence[str]) -> dict[str, float]:
        return {
            name: value
            for name in names
            if (value := self.latest_metric(name)) is not None
        }

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
    ) -> int:
        now = time.time() if created_at is None else created_at
        path_text = str(path)
        metadata_text = str(metadata_path) if metadata_path is not None else None
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO checkpoints
                  (run_name, kind, step, path, metadata_path, sha256, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                  run_name = excluded.run_name,
                  kind = excluded.kind,
                  step = excluded.step,
                  metadata_path = excluded.metadata_path,
                  sha256 = excluded.sha256,
                  status = excluded.status,
                  updated_at = excluded.updated_at
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
            row = conn.execute("SELECT id FROM checkpoints WHERE path = ?", (path_text,)).fetchone()
            checkpoint_id = int(row["id"])
            conn.execute(
                """
                INSERT INTO artifact_uploads (checkpoint_id, status, created_at, updated_at)
                VALUES (?, 'pending', ?, ?)
                ON CONFLICT(checkpoint_id) DO NOTHING
                """,
                (checkpoint_id, now, now),
            )
            if kind not in FINAL_LIKE_KINDS:
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

    def pending_evals(self, *, limit: int = 1) -> list[dict[str, Any]]:
        return self._pending_rows(
            table="eval_results",
            statuses=("pending", "failed_retryable"),
            limit=limit,
        )

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
                """
            ).fetchall()
        return {str(row["key"]): int(row["count"]) for row in rows}


def metric_store_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "rlab.sqlite"


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
