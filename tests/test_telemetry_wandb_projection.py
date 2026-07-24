from __future__ import annotations

import unittest
import inspect
from datetime import UTC, datetime
from unittest import mock

from rlab.telemetry_integrity import sha256_bytes
from rlab.telemetry_mailbox import encode_metric_batch
from rlab.telemetry_v2_controller import ambiguous_recovery_action, project_once
from rlab.telemetry_wandb_projection import (
    ProjectionRow,
    activate_verified_generation,
    claim_projection_row,
    materialize_projection_rows,
    publication_aggregate,
    projection_verification_window,
    projection_fingerprint,
    publish_projection_row,
    terminal_completion_eligible,
    verify_exact_prefix,
)


class _Cursor:
    def __init__(self):
        self.executions = []

    def execute(self, sql, params=None):
        self.executions.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Connection:
    def __init__(self):
        self.value = _Cursor()

    def cursor(self):
        return self.value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _MaterializeCursor(_Cursor):
    def __init__(self, source_rows):
        super().__init__()
        self.source_rows = source_rows
        self.result = None
        self.rowcount = 0

    def execute(self, sql, params=None):
        super().execute(sql, params)
        self.rowcount = 0
        if "SELECT *" in sql and "wandb_projection_generations" in sql:
            self.result = {
                "state": "pending",
                "next_ordinal": 0,
                "projection_chain_sha256": None,
                "last_source_at": None,
            }
        elif "SELECT count(*) AS count" in sql:
            self.result = {"count": 0}
        elif "SELECT e.*, p.producer_identity" in sql:
            self.result = list(self.source_rows)
        elif "UPDATE wandb_projection_source_cursors" in sql:
            self.rowcount = 1
            self.result = None
        else:
            self.result = None

    def fetchone(self):
        return self.result

    def fetchall(self):
        return list(self.result or [])


class _MaterializeConnection(_Connection):
    def __init__(self, source_rows):
        self.value = _MaterializeCursor(source_rows)


class _Run:
    def __init__(self):
        self.calls = []

    def log(self, payload, *, step, commit):
        self.calls.append((payload, step, commit))


class _AmbiguousRun(_Run):
    def log(self, payload, *, step, commit):
        super().log(payload, step=step, commit=commit)
        raise TimeoutError("write may have reached W&B")


class _ActivationCursor(_Cursor):
    def __init__(self, rows):
        super().__init__()
        self.rows = rows
        self.rowcount = 0

    def execute(self, sql, params=None):
        super().execute(sql, params)
        self.rowcount = 1 if "SET state = 'active'" in sql else 0

    def fetchall(self):
        return list(self.rows)


class _ActivationConnection(_Connection):
    def __init__(self, rows):
        self.value = _ActivationCursor(rows)


class _ClaimCursor(_Cursor):
    def __init__(self):
        super().__init__()
        self._row = {
            "train_job_id": 7,
            "projection_generation": 2,
            "output_ordinal": 3,
            "step_offset": 100,
            "stable_key": "stable",
            "payload_sha256": "a" * 64,
            "payload_json": {"metric": 1.0},
        }

    def fetchall(self):
        row, self._row = self._row, None
        return [] if row is None else [row]


class _ClaimConnection(_Connection):
    def __init__(self):
        self.value = _ClaimCursor()


class WandbProjectionTests(unittest.TestCase):
    def expected(self):
        return [
            {
                "output_ordinal": 0,
                "stable_key": "a",
                "payload_sha256": "a" * 64,
                "predecessor_sha256": None,
                "output_kind": "history",
                "payload_json": {
                    "_rlab_event_id": "event-a",
                    "_rlab_payload_sha256": "a" * 64,
                },
            },
            {
                "output_ordinal": 1,
                "stable_key": "b",
                "payload_sha256": "b" * 64,
                "predecessor_sha256": "a" * 64,
                "output_kind": "history",
                "payload_json": {
                    "_rlab_event_id": "event-b",
                    "_rlab_payload_sha256": "b" * 64,
                },
            },
        ]

    def test_exact_prefix_accepts_only_stable_identity_digest_and_step(self):
        observed = [
            {
                "_step": 10,
                "_rlab_event_id": "event-a",
                "_rlab_payload_sha256": "a" * 64,
            }
        ]
        result = verify_exact_prefix(self.expected(), observed, step_offset=10)
        self.assertTrue(result.exact)
        self.assertEqual(0, result.verified_through_ordinal)

        conflict = verify_exact_prefix(
            self.expected(),
            [{**observed[0], "_step": 11}],
            step_offset=10,
        )
        self.assertFalse(conflict.exact)
        self.assertEqual("past_step_hole_or_foreign_step", conflict.reason)
        digest_conflict = verify_exact_prefix(
            self.expected(),
            [{**observed[0], "_rlab_payload_sha256": "f" * 64}],
            step_offset=10,
        )
        self.assertFalse(digest_conflict.exact)
        self.assertEqual("payload_digest_conflict", digest_conflict.reason)
        foreign_suffix = verify_exact_prefix(
            self.expected(),
            [observed[0], {}, {}],
            step_offset=10,
        )
        self.assertFalse(foreign_suffix.exact)
        self.assertEqual("foreign_or_duplicate_suffix", foreign_suffix.reason)

    def test_successful_sdk_return_becomes_submitted(self):
        conn = _Connection()
        run = _Run()
        row = ProjectionRow(
            7,
            2,
            3,
            100,
            "stable",
            "a" * 64,
            {"_rlab_event_id": "event", "_rlab_payload_sha256": "a" * 64},
        )
        publish_projection_row(conn, run, row, owner="worker")
        self.assertEqual(1, len(run.calls))
        self.assertEqual(103, run.calls[0][1])
        self.assertTrue(run.calls[0][2])
        self.assertTrue(any("state = 'submitted'" in sql for sql, _params in conn.value.executions))

    def test_write_then_raise_becomes_ambiguous(self):
        conn = _Connection()
        run = _AmbiguousRun()
        row = ProjectionRow(
            7,
            2,
            3,
            100,
            "stable",
            "a" * 64,
            {"_rlab_event_id": "event", "_rlab_payload_sha256": "a" * 64},
        )

        with self.assertRaisesRegex(TimeoutError, "may have reached"):
            publish_projection_row(conn, run, row, owner="worker")

        self.assertEqual(1, len(run.calls))
        self.assertTrue(any("state = 'ambiguous'" in sql for sql, _params in conn.value.executions))

    def test_claim_has_no_verified_predecessor_gate(self):
        conn = _ClaimConnection()

        row = claim_projection_row(conn, owner="worker", train_job_id=7)

        self.assertIsNotNone(row)
        claim_sql = next(
            sql for sql, _params in conn.value.executions if "SELECT r.*, g.step_offset" in sql
        )
        self.assertNotIn("prior.state", claim_sql)
        self.assertIn("LIMIT %(limit)s", claim_sql)
        self.assertTrue(
            any(
                "SET state='pending'" in sql and "claim_expires_at <= now()" in sql
                for sql, _params in conn.value.executions
            )
        )

    def test_empty_event_advances_cursor_before_real_metrics(self):
        empty = encode_metric_batch([])
        real = encode_metric_batch(
            [
                {
                    "id": 1,
                    "event_id": "history-1",
                    "created_at": 1.0,
                    "step": 200,
                    "kind": "history",
                    "payload_json": ('{"global_step":200,"train/throughput/loop_fps":0.5}'),
                }
            ]
        )
        now = datetime.now(UTC)

        def source(sequence, event_id, batch):
            return {
                "train_job_id": 7,
                "telemetry_generation": 1,
                "producer_ordinal": 0,
                "producer_identity": "learner",
                "source_sequence": sequence,
                "event_identity": event_id,
                "event_kind": "metric_batch",
                "payload_encoding": "metric_batch_zlib_json_v1",
                "payload": batch.payload,
                "payload_sha256": sha256_bytes(batch.payload),
                "event_sha256": "f" * 64,
                "predecessor_sha256": None,
                "terminal": False,
                "metrics_schema_version": 5,
                "created_at": now,
            }

        conn = _MaterializeConnection([source(1, "empty", empty), source(2, "real", real)])

        inserted = materialize_projection_rows(
            conn,
            train_job_id=7,
            projection_generation=1,
        )

        self.assertEqual(1, inserted)
        cursor_sequences = [
            params["sequence"]
            for sql, params in conn.value.executions
            if "UPDATE wandb_projection_source_cursors" in sql
        ]
        self.assertEqual([1, 2], cursor_sequences)
        row_inserts = [
            sql
            for sql, _params in conn.value.executions
            if "INSERT INTO wandb_projection_rows" in sql
        ]
        self.assertEqual(1, len(row_inserts))

    def test_verification_window_is_incremental_and_bounded(self):
        window = projection_verification_window(
            {
                "verified_through_ordinal": 10_000,
                "submitted_through_ordinal": 20_000,
            }
        )

        self.assertEqual((10_001, 10_256), window)

    def test_verified_history_cannot_clear_artifact_failure(self):
        aggregate = publication_aggregate(
            {
                "history": "complete",
                "artifacts": "failed",
                "terminal": "pending",
            },
            telemetry_no_more_producers=True,
        )

        self.assertEqual("failed", aggregate)

    def test_manager_materializer_never_opens_wandb_sdk(self):
        source = inspect.getsource(project_once)

        self.assertNotIn("wandb.init", source)
        self.assertNotIn("WandbProjector", source)

    def test_ambiguous_write_restart_adopts_or_republishes_exactly(self):
        self.assertEqual(
            "adopt",
            ambiguous_recovery_action(
                exact=True,
                verified_through_ordinal=7,
                last_ambiguous_ordinal=7,
                remote_last_step=7,
                step_offset=0,
            ),
        )
        self.assertEqual(
            "republish",
            ambiguous_recovery_action(
                exact=True,
                verified_through_ordinal=6,
                last_ambiguous_ordinal=7,
                remote_last_step=6,
                step_offset=0,
            ),
        )
        self.assertEqual(
            "quarantine",
            ambiguous_recovery_action(
                exact=False,
                verified_through_ordinal=6,
                last_ambiguous_ordinal=7,
                remote_last_step=8,
                step_offset=0,
            ),
        )

    def test_terminal_restart_requires_verified_sentinel_and_no_later_step(self):
        self.assertTrue(
            terminal_completion_eligible(
                close_ordinal=12,
                verified_through_ordinal=12,
                step_offset=100,
                remote_state="finished",
                remote_last_step=112,
            )
        )
        self.assertFalse(
            terminal_completion_eligible(
                close_ordinal=12,
                verified_through_ordinal=12,
                step_offset=100,
                remote_state="finished",
                remote_last_step=113,
            )
        )
        self.assertFalse(
            terminal_completion_eligible(
                close_ordinal=12,
                verified_through_ordinal=11,
                step_offset=100,
                remote_state="running",
                remote_last_step=112,
            )
        )

    def test_fingerprint_binds_predecessor_and_presentation_kind(self):
        first = projection_fingerprint(self.expected())
        changed = [dict(row) for row in self.expected()]
        changed[1]["predecessor_sha256"] = "c" * 64
        self.assertNotEqual(first, projection_fingerprint(changed))

    def test_activation_projects_verified_v2_generation_into_run_readiness(self):
        rows = [{**row, "state": "verified"} for row in self.expected()]
        conn = _ActivationConnection(rows)
        fingerprint = projection_fingerprint(rows)

        with mock.patch("rlab.telemetry_wandb_projection.set_publication_component"):
            result = activate_verified_generation(
                conn,
                train_job_id=194,
                projection_generation=1,
                remote_fingerprint=fingerprint,
                wandb_run_id="rlab-run",
                wandb_url="https://wandb.ai/entity/project/runs/rlab-run",
            )

        self.assertEqual(result, fingerprint)
        readiness = [
            (sql, params)
            for sql, params in conn.value.executions
            if "SET active_wandb_projection_generation" in sql
        ]
        self.assertEqual(len(readiness), 1)
        sql, params = readiness[0]
        self.assertIn("wandb_ready_at", sql)
        self.assertNotIn("live_publication_status", sql)
        self.assertEqual(params["wandb_run_id"], "rlab-run")
        self.assertEqual(params["wandb_url"], "https://wandb.ai/entity/project/runs/rlab-run")


if __name__ == "__main__":
    unittest.main()
