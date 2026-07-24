from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from rlab.modal_eval_storage import ObjectStore
from rlab.telemetry_archive import (
    ArchiveClaim,
    ArchiveCopy,
    archive_policy_sha256,
    build_archive_root_document,
    canonical_event_from_ledger_row,
    finalize_exact_archive_root,
    write_claimed_segment,
)
from rlab.telemetry_reducer import reduce_run_integrity
from rlab.telemetry_v2_controller import finalize_roots_once
from rlab.telemetry_integrity import CanonicalEvent, ProducerKey, build_canonical_segment


class _Cursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, object]] = []
        self.rowcount = 1

    def execute(self, sql, params=None):
        self.statements.append((sql, params))

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _Connection:
    def __init__(self) -> None:
        self.cursor_value = _Cursor()

    def cursor(self):
        return self.cursor_value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class TelemetryArchiveTests(unittest.TestCase):
    def _claim(self) -> ArchiveClaim:
        producer = ProducerKey(17, 1, 0, "train")
        segment = build_canonical_segment(
            [
                CanonicalEvent(producer, 1, "one", "history", {"score": 1.25}),
                CanonicalEvent(producer, 2, "close", "close", {"ok": True}),
            ]
        )
        return ArchiveClaim(17, 1, 0, "train", 1, 2, segment, "owner")

    def test_dual_policy_requires_independent_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ObjectStore(Path(directory).as_uri())
            with self.assertRaisesRegex(ValueError, "independently rooted"):
                archive_policy_sha256(
                    "queued_dual_r2_v1",
                    [ArchiveCopy("primary", store), ArchiveCopy("backup", store)],
                )

    def test_writes_and_fully_verifies_both_immutable_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copies = [
                ArchiveCopy("primary", ObjectStore((root / "primary").as_uri())),
                ArchiveCopy("backup", ObjectStore((root / "backup").as_uri())),
            ]
            conn = _Connection()
            receipts = write_claimed_segment(
                conn,
                self._claim(),
                policy_name="queued_dual_r2_v1",
                copies=copies,
            )
            self.assertEqual(["primary", "backup"], [row["copy_role"] for row in receipts])
            self.assertTrue(
                all(Path(row["object_uri"].removeprefix("file://")).is_file() for row in receipts)
            )
            self.assertTrue(
                any("telemetry_archive_receipts" in sql for sql, _ in conn.cursor_value.statements)
            )

    def test_root_binds_obligations_claims_segments_and_evidence(self):
        root = build_archive_root_document(
            train_job_id=17,
            generation=1,
            policy_name="queued_dual_r2_v1",
            expected_obligations=[
                {
                    "obligation_key": "training-terminal",
                    "obligation_kind": "training",
                    "producer_ordinal": 0,
                    "expected_disposition": "complete",
                    "realized_disposition": "complete",
                    "evidence_sha256": "a" * 64,
                }
            ],
            producers=[
                {
                    "producer_ordinal": 0,
                    "producer_identity": "train",
                    "final_sequence": 2,
                    "final_sha256": "b" * 64,
                }
            ],
            segments=[
                {
                    "producer_ordinal": 0,
                    "first_sequence": 1,
                    "last_sequence": 2,
                    "compressed_sha256": "c" * 64,
                }
            ],
            evidence_roots=[
                {
                    "scope_sha256": "d" * 64,
                    "scope_kind": "eval_scope_exact",
                }
            ],
        )
        self.assertEqual("telemetry-run-root-v1", root["version"])
        self.assertEqual(64, len(root["expected_set_sha256"]))
        self.assertEqual(64, len(root["coverage_sha256"]))
        self.assertEqual("d" * 64, root["evidence_roots"][0]["scope_sha256"])

    def test_ledger_decode_uses_the_run_metrics_schema_version(self):
        frames = [
            {
                "event_id": "metric-1",
                "sequence": 1,
                "observed_at": 1.0,
                "global_step": 10,
                "kind": "history",
                "payload": {
                    "global_step": 10,
                    "train/episode/return/shaped/from/target/mean": 1.25,
                },
            }
        ]
        payload = gzip.compress(
            json.dumps(frames, sort_keys=True, separators=(",", ":")).encode(),
            mtime=0,
        )
        event = canonical_event_from_ledger_row(
            {
                "payload_encoding": "metric_batch_zlib_json_v1",
                "payload": payload,
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "event_sha256": "a" * 64,
                "predecessor_sha256": None,
                "terminal": False,
                "train_job_id": 194,
                "telemetry_generation": 1,
                "producer_ordinal": 0,
                "producer_identity": "train-194",
                "source_sequence": 1,
                "event_identity": "metric-batch-1",
                "event_kind": "metric_batch",
                "metrics_schema_version": 6,
            }
        )

        self.assertEqual(
            1.25,
            event.payload["frames"][0]["payload"][
                "train/episode/return/shaped/from/target/mean"
            ],
        )

    def test_finalizing_runs_are_eligible_for_exact_archive_freeze(self):
        root_source = inspect.getsource(finalize_exact_archive_root)
        controller_source = inspect.getsource(finalize_roots_once)

        self.assertIn("'finalizing'", root_source)
        self.assertIn("'finalizing'", controller_source)

    def test_integrity_counts_distinct_segments_before_receipt_join(self):
        source = inspect.getsource(reduce_run_integrity)

        self.assertIn("count(DISTINCT (", source)
        self.assertIn("count(r.*) AS receipts", source)


if __name__ == "__main__":
    unittest.main()
