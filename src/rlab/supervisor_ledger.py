from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS supervisor_state (
  key TEXT PRIMARY KEY,
  value_json TEXT NOT NULL,
  updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS metric_segments (
  first_event_seq INTEGER NOT NULL,
  last_event_seq INTEGER NOT NULL UNIQUE,
  object_key TEXT NOT NULL UNIQUE,
  sha256 TEXT NOT NULL,
  event_count INTEGER NOT NULL,
  created_at REAL NOT NULL,
  PRIMARY KEY (first_event_seq, last_event_seq)
);

CREATE TABLE IF NOT EXISTS checkpoint_publications (
  checkpoint_ledger_id INTEGER PRIMARY KEY,
  checkpoint_id TEXT NOT NULL UNIQUE,
  step INTEGER NOT NULL,
  purpose TEXT NOT NULL,
  manifest_json TEXT NOT NULL,
  published_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_dispatches (
  idempotency_key TEXT PRIMARY KEY,
  checkpoint_ledger_id INTEGER NOT NULL UNIQUE,
  checkpoint_id TEXT NOT NULL,
  checkpoint_step INTEGER NOT NULL,
  intent_json TEXT NOT NULL,
  attempt INTEGER NOT NULL DEFAULT 0,
  modal_call_id TEXT,
  attempt_expires_at REAL,
  status TEXT NOT NULL DEFAULT 'pending',
  result_json TEXT,
  result_observed_at REAL,
  stop_requested_at REAL,
  last_error TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS eval_dispatches_status_idx
  ON eval_dispatches (status, checkpoint_step);
"""


class SupervisorLedger:
    def __init__(self, path: Path | str, *, timeout: float = 5.0):
        self.path = Path(path)
        self.timeout = timeout

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout={max(0, int(self.timeout * 1000))}")
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
            connection.executescript(SCHEMA_SQL)
            eval_columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(eval_dispatches)")
            }
            if "attempt_expires_at" not in eval_columns:
                connection.execute(
                    "ALTER TABLE eval_dispatches ADD COLUMN attempt_expires_at REAL"
                )

    def state(self, key: str, default: Any = None) -> Any:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT value_json FROM supervisor_state WHERE key = ?",
                (key,),
            ).fetchone()
        return default if row is None else json.loads(str(row["value_json"]))

    def set_state(self, key: str, value: Any) -> None:
        now = time.time()
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO supervisor_state (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                  value_json = excluded.value_json,
                  updated_at = excluded.updated_at
                """,
                (key, payload, now),
            )

    def next_metric_events(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(last_event_seq), 0) FROM metric_segments"
            ).fetchone()
            high_water = int(row[0] if row else 0)
            rows = connection.execute(
                """
                SELECT id, event_id, step, source, kind, payload_json, created_at
                FROM metric_frames
                WHERE id > ?
                ORDER BY id
                LIMIT ?
                """,
                (high_water, max(1, int(limit))),
            ).fetchall()
        return [
            {
                "event_seq": int(row["id"]),
                "event_id": str(row["event_id"]),
                "step": row["step"],
                "source": str(row["source"]),
                "kind": str(row["kind"]),
                "payload": json.loads(str(row["payload_json"])),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def record_metric_segment(
        self,
        *,
        events: Sequence[Mapping[str, Any]],
        object_key: str,
        sha256: str,
    ) -> None:
        if not events:
            raise ValueError("metric segment cannot be empty")
        first = int(events[0]["event_seq"])
        last = int(events[-1]["event_seq"])
        now = time.time()
        with self.connection() as connection:
            previous = connection.execute(
                "SELECT COALESCE(MAX(last_event_seq), 0) FROM metric_segments"
            ).fetchone()
            if int(previous[0] if previous else 0) >= first:
                raise ValueError("metric segment overlaps the durable high-water mark")
            connection.execute(
                """
                INSERT INTO metric_segments (
                  first_event_seq, last_event_seq, object_key, sha256, event_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (first, last, object_key, sha256, len(events), now),
            )

    def metric_segment_high_water(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT COALESCE(MAX(last_event_seq), 0) FROM metric_segments"
            ).fetchone()
        return int(row[0] if row else 0)

    def record_checkpoint_publication(
        self,
        *,
        checkpoint_ledger_id: int,
        manifest: Mapping[str, Any],
    ) -> None:
        now = time.time()
        payload = json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"))
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO checkpoint_publications (
                  checkpoint_ledger_id, checkpoint_id, step, purpose, manifest_json, published_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(checkpoint_ledger_id) DO NOTHING
                """,
                (
                    int(checkpoint_ledger_id),
                    str(manifest["checkpoint_id"]),
                    int(manifest["step"]),
                    str(manifest["purpose"]),
                    payload,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT manifest_json FROM checkpoint_publications "
                "WHERE checkpoint_ledger_id = ?",
                (int(checkpoint_ledger_id),),
            ).fetchone()
            if row is None or json.loads(str(row["manifest_json"])) != dict(manifest):
                raise RuntimeError("checkpoint publication conflicts with durable local state")

    def checkpoint_publication(self, checkpoint_ledger_id: int) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT manifest_json FROM checkpoint_publications "
                "WHERE checkpoint_ledger_id = ?",
                (int(checkpoint_ledger_id),),
            ).fetchone()
        return None if row is None else json.loads(str(row["manifest_json"]))

    def checkpoint_publication_by_id(
        self,
        checkpoint_id: str,
    ) -> dict[str, Any] | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT checkpoint_ledger_id, manifest_json "
                "FROM checkpoint_publications WHERE checkpoint_id = ?",
                (str(checkpoint_id),),
            ).fetchone()
        if row is None:
            return None
        return {
            "checkpoint_ledger_id": int(row["checkpoint_ledger_id"]),
            **json.loads(str(row["manifest_json"])),
        }

    def checkpoint_publications(self) -> list[dict[str, Any]]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT checkpoint_ledger_id, manifest_json FROM checkpoint_publications "
                "ORDER BY step, checkpoint_ledger_id"
            ).fetchall()
        return [
            {
                "checkpoint_ledger_id": int(row["checkpoint_ledger_id"]),
                **json.loads(str(row["manifest_json"])),
            }
            for row in rows
        ]

    def ensure_eval(
        self,
        *,
        checkpoint_ledger_id: int,
        intent: Mapping[str, Any],
    ) -> None:
        now = time.time()
        payload = json.dumps(dict(intent), sort_keys=True, separators=(",", ":"))
        with self.connection() as connection:
            connection.execute(
                """
                INSERT INTO eval_dispatches (
                  idempotency_key, checkpoint_ledger_id, checkpoint_id,
                  checkpoint_step, intent_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO NOTHING
                """,
                (
                    str(intent["idempotency_key"]),
                    int(checkpoint_ledger_id),
                    str(intent["checkpoint_id"]),
                    int(intent["checkpoint_step"]),
                    payload,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT intent_json FROM eval_dispatches WHERE idempotency_key = ?",
                (str(intent["idempotency_key"]),),
            ).fetchone()
            if row is None or json.loads(str(row["intent_json"])) != dict(intent):
                raise RuntimeError("eval intent conflicts with durable local state")

    def evals(self, *, statuses: Sequence[str] | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM eval_dispatches"
        parameters: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            query += f" WHERE status IN ({placeholders})"
            parameters.extend(str(status) for status in statuses)
        query += " ORDER BY checkpoint_step, checkpoint_ledger_id"
        with self.connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            item["intent"] = json.loads(str(item.pop("intent_json")))
            if item.get("result_json") is not None:
                item["result"] = json.loads(str(item.pop("result_json")))
            result.append(item)
        return result

    def eval(self, idempotency_key: str) -> dict[str, Any] | None:
        rows = [
            row
            for row in self.evals()
            if str(row["idempotency_key"]) == str(idempotency_key)
        ]
        return rows[0] if rows else None

    def mark_eval_submitted(
        self,
        *,
        idempotency_key: str,
        attempt: int,
        modal_call_id: str,
        attempt_expires_at: float,
    ) -> None:
        now = time.time()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE eval_dispatches
                SET status = 'submitted', attempt = ?, modal_call_id = ?,
                    attempt_expires_at = ?, last_error = NULL, updated_at = ?
                WHERE idempotency_key = ? AND status = 'pending'
                """,
                (
                    int(attempt),
                    modal_call_id,
                    float(attempt_expires_at),
                    now,
                    idempotency_key,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"eval intent is not pending: {idempotency_key}")

    def reset_expired_eval(self, *, idempotency_key: str, error: str) -> None:
        now = time.time()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE eval_dispatches
                SET status = 'pending', modal_call_id = NULL, attempt_expires_at = NULL,
                    last_error = ?, updated_at = ?
                WHERE idempotency_key = ? AND status = 'submitted' AND attempt < 2
                """,
                (error[:4000], now, idempotency_key),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"eval cannot be retried: {idempotency_key}")

    def record_eval_error(self, *, idempotency_key: str, error: str) -> None:
        now = time.time()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE eval_dispatches
                SET last_error = ?, updated_at = ?
                WHERE idempotency_key = ?
                  AND status IN ('pending', 'submitted')
                """,
                (error[:4000], now, idempotency_key),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"active eval not found: {idempotency_key}")

    def mark_eval_terminal(
        self,
        *,
        idempotency_key: str,
        status: str,
        result: Mapping[str, Any],
    ) -> None:
        if status not in {"accepted", "rejected", "failed", "expired", "canceled"}:
            raise ValueError(f"invalid terminal eval status: {status}")
        now = time.time()
        payload = json.dumps(dict(result), sort_keys=True, separators=(",", ":"), default=str)
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE eval_dispatches
                SET status = ?, result_json = ?, result_observed_at = ?,
                    last_error = NULL, updated_at = ?
                WHERE idempotency_key = ?
                  AND status IN ('pending', 'submitted')
                """,
                (status, payload, now, now, idempotency_key),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT status, result_json FROM eval_dispatches WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if (
                    row is None
                    or str(row["status"]) != status
                    or json.loads(str(row["result_json"])) != dict(result)
                ):
                    raise RuntimeError(f"eval terminal result conflicts: {idempotency_key}")

    def mark_stop_requested(self, *, idempotency_key: str) -> float:
        now = time.time()
        with self.connection() as connection:
            cursor = connection.execute(
                """
                UPDATE eval_dispatches
                SET stop_requested_at = COALESCE(stop_requested_at, ?), updated_at = ?
                WHERE idempotency_key = ? AND status = 'accepted'
                """,
                (now, now, idempotency_key),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"accepted eval not found: {idempotency_key}")
            row = connection.execute(
                "SELECT stop_requested_at FROM eval_dispatches WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        return float(row["stop_requested_at"])

    def all_evals_terminal(self) -> bool:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM eval_dispatches
                WHERE status NOT IN ('accepted', 'rejected', 'failed', 'expired', 'canceled')
                """
            ).fetchone()
        return int(row[0] if row else 0) == 0

    def terminal_eval_count(self) -> int:
        with self.connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) FROM eval_dispatches
                WHERE status IN ('accepted', 'rejected', 'failed', 'expired', 'canceled')
                """
            ).fetchone()
        return int(row[0] if row else 0)
