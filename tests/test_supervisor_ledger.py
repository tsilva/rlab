from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from rlab.metric_store import MetricStore
from rlab.supervisor_ledger import SupervisorLedger


class SupervisorLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "rlab.sqlite"
        self.metrics = MetricStore(self.path)
        self.metrics.init()
        self.ledger = SupervisorLedger(self.path)
        self.ledger.init()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_segments_advance_contiguously(self) -> None:
        self.metrics.append_metrics(
            {"train/throughput/loop_fps": 1.0},
            step=10,
            source="train",
        )
        self.metrics.append_metrics(
            {"train/throughput/loop_fps": 2.0},
            step=20,
            source="train",
        )
        events = self.ledger.next_metric_events()
        self.assertEqual([row["event_seq"] for row in events], [1, 2])
        self.ledger.record_metric_segment(
            events=events,
            object_key="runs/x/segment.jsonl",
            sha256="a" * 64,
        )
        self.assertEqual(self.ledger.metric_segment_high_water(), 2)
        self.assertEqual(self.ledger.next_metric_events(), [])

    def test_checkpoint_publication_is_idempotent_but_conflict_is_rejected(self) -> None:
        manifest = {
            "checkpoint_id": "checkpoint-10-aaaaaaaaaaaaaaaa",
            "step": 10,
            "purpose": "periodic",
        }
        self.ledger.record_checkpoint_publication(
            checkpoint_ledger_id=1,
            manifest=manifest,
        )
        self.ledger.record_checkpoint_publication(
            checkpoint_ledger_id=1,
            manifest=manifest,
        )
        with self.assertRaisesRegex(RuntimeError, "conflicts"):
            self.ledger.record_checkpoint_publication(
                checkpoint_ledger_id=1,
                manifest={**manifest, "step": 11},
            )

    def test_eval_lifecycle_is_exactly_once(self) -> None:
        intent = {
            "idempotency_key": "a" * 64,
            "checkpoint_id": "checkpoint-10-aaaaaaaaaaaaaaaa",
            "checkpoint_step": 10,
        }
        self.ledger.ensure_eval(checkpoint_ledger_id=1, intent=intent)
        self.ledger.mark_eval_submitted(
            idempotency_key="a" * 64,
            attempt=1,
            modal_call_id="fc-one",
            attempt_expires_at=1_000.0,
        )
        result = {"status": "accepted", "episode_results": list(range(100))}
        self.ledger.mark_eval_terminal(
            idempotency_key="a" * 64,
            status="accepted",
            result=result,
        )
        self.ledger.mark_eval_terminal(
            idempotency_key="a" * 64,
            status="accepted",
            result=result,
        )
        self.assertTrue(self.ledger.all_evals_terminal())
        row = self.ledger.evals()[0]
        self.assertEqual(row["result"], result)
        self.assertIsInstance(self.ledger.mark_stop_requested(idempotency_key="a" * 64), float)

    def test_state_round_trips_json(self) -> None:
        self.ledger.set_state("freeze", {"checkpoint_ids": ["one"]})
        self.assertEqual(self.ledger.state("freeze"), {"checkpoint_ids": ["one"]})
        self.assertIsNone(self.ledger.state("missing"))


if __name__ == "__main__":
    unittest.main()
