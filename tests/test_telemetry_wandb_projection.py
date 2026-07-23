from __future__ import annotations

import unittest

from rlab.telemetry_wandb_projection import (
    ProjectionRow,
    projection_fingerprint,
    publish_projection_row,
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


class _Run:
    def __init__(self):
        self.calls = []

    def log(self, payload, *, step, commit):
        self.calls.append((payload, step, commit))


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

    def test_publish_is_one_explicit_committed_sdk_call_then_ambiguous(self):
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
        self.assertTrue(
            any(
                "state = 'ambiguous'" in sql for sql, _params in conn.value.executions
            )
        )

    def test_fingerprint_binds_predecessor_and_presentation_kind(self):
        first = projection_fingerprint(self.expected())
        changed = [dict(row) for row in self.expected()]
        changed[1]["predecessor_sha256"] = "c" * 64
        self.assertNotEqual(first, projection_fingerprint(changed))


if __name__ == "__main__":
    unittest.main()
