from __future__ import annotations

import unittest

from rlab.telemetry_projection_benchmark import benchmark_pair


class TelemetryProjectionBenchmarkTests(unittest.TestCase):
    def test_sustained_twenty_thousand_steps_per_second_is_bounded(self) -> None:
        result = benchmark_pair()
        legacy = result["legacy"]
        bounded = result["bounded_v2"]

        self.assertEqual(20_000.0, bounded.producer_steps_per_second)
        self.assertEqual(100.0 / 30.0, bounded.raw_normalized_frames_per_second)
        self.assertEqual(1.0 / 30.0, bounded.selected_rows_per_second)
        self.assertGreater(legacy.maximum_wandb_freshness_seconds, 60.0)
        self.assertLess(bounded.maximum_wandb_freshness_seconds, 1.0)
        self.assertLess(bounded.terminal_drain_seconds, 300.0)
        self.assertLess(bounded.maximum_backlog_rows, legacy.maximum_backlog_rows)

    def test_two_runs_have_independent_bounded_actors(self) -> None:
        first = benchmark_pair()["bounded_v2"]
        second = benchmark_pair()["bounded_v2"]

        self.assertEqual(first, second)
        self.assertLess(first.maximum_wandb_freshness_seconds, 60.0)
        self.assertLess(second.maximum_wandb_freshness_seconds, 60.0)


if __name__ == "__main__":
    unittest.main()
