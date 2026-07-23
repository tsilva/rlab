from __future__ import annotations

import unittest

from rlab.telemetry_recovery import FairRunQueue, RecoveryLimits, TelemetryRecoveryExecutors


class TelemetryRecoveryTests(unittest.TestCase):
    def test_fair_queue_round_robins_large_and_small_runs(self):
        queue = FairRunQueue(per_run_limit=2)
        queue.extend(1, range(1000))
        queue.extend(2, ["only"])
        self.assertEqual((1, 0), queue.pop())
        self.assertEqual((2, "only"), queue.pop())
        self.assertEqual((1, 1), queue.pop())

    def test_executor_lanes_and_poison_isolation_are_independent(self):
        executors = TelemetryRecoveryExecutors(
            RecoveryLimits(
                archive_workers=1,
                wandb_workers=1,
                artifact_workers=1,
                poison_failures=1,
            )
        )
        try:
            failed = executors.submit(
                "archive",
                run_id=1,
                poison_key="bad",
                work=lambda: (_ for _ in ()).throw(RuntimeError("bad")),
            )
            with self.assertRaises(RuntimeError):
                failed.result()
            with self.assertRaisesRegex(RuntimeError, "poison-isolated"):
                executors.submit(
                    "archive", run_id=1, poison_key="bad", work=lambda: None
                )
            healthy = executors.submit(
                "wandb", run_id=2, poison_key="good", work=lambda: 7
            )
            self.assertEqual(7, healthy.result())
        finally:
            executors.close()


if __name__ == "__main__":
    unittest.main()
