from __future__ import annotations

import unittest

from rlab.run_job import WorkerModules, publication_attempt_count, worker_modules


class RunJobWorkerTests(unittest.TestCase):
    def test_backend_worker_matrix(self) -> None:
        self.assertEqual(
            worker_modules("modal", wandb_enabled=True),
            WorkerModules(
                artifact_uploader="rlab.checkpoint_coordinator",
                evaluator=None,
                publisher="rlab.wandb_publisher",
            ),
        )
        self.assertEqual(
            worker_modules("local", wandb_enabled=True),
            WorkerModules(
                artifact_uploader=None,
                evaluator="rlab.checkpoint_eval_worker",
                publisher="rlab.wandb_publisher",
            ),
        )
        self.assertEqual(
            worker_modules("none", wandb_enabled=True),
            WorkerModules(
                artifact_uploader=None,
                evaluator=None,
                publisher="rlab.wandb_publisher",
            ),
        )
        self.assertEqual(
            worker_modules("none", wandb_enabled=False),
            WorkerModules(
                artifact_uploader=None,
                evaluator=None,
                publisher="rlab.wandb_publisher",
            ),
        )
        neon_expected = {
            "modal": WorkerModules(
                artifact_uploader="rlab.checkpoint_coordinator",
                evaluator=None,
                publisher="rlab.telemetry_relay",
            ),
            "local": WorkerModules(
                artifact_uploader="rlab.checkpoint_coordinator",
                evaluator="rlab.checkpoint_eval_worker",
                publisher="rlab.telemetry_relay",
            ),
            "none": WorkerModules(
                artifact_uploader="rlab.checkpoint_coordinator",
                evaluator=None,
                publisher="rlab.telemetry_relay",
            ),
        }
        for backend, expected in neon_expected.items():
            with self.subTest(backend=backend):
                self.assertEqual(
                    worker_modules(
                        backend,
                        wandb_enabled=True,
                        telemetry_transport="neon_mailbox_v1",
                    ),
                    expected,
                )

    def test_unknown_backend_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported checkpoint evaluation backend"):
            worker_modules("typo", wandb_enabled=True)

    def test_mailbox_relay_is_not_counted_as_wandb_attempt(self) -> None:
        self.assertEqual(
            publication_attempt_count(
                telemetry_transport="neon_mailbox_v1",
                publisher_results=[{"returncode": 0}],
            ),
            0,
        )

    def test_legacy_publisher_execution_retains_attempt_count(self) -> None:
        self.assertEqual(
            publication_attempt_count(
                telemetry_transport="legacy_local",
                publisher_results=[{"returncode": 0}],
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
