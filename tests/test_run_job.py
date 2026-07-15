from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from rlab.run_job import run_training_process, write_result


class RunJobResultTests(unittest.TestCase):
    def test_write_result_atomically_replaces_existing_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            path = output_dir / "result.json"
            path.write_text('{"status": "old"}\n', encoding="utf-8")

            written = write_result(
                output_dir,
                {
                    "schema_version": 1,
                    "job_id": 7,
                    "launch_id": "train-7-stable",
                    "status": "succeeded",
                },
            )

            self.assertEqual(written, path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["job_id"], 7)
            self.assertEqual(path.stat().st_mode & 0o777, 0o644)
            self.assertEqual(list(output_dir.glob(".result-*.tmp")), [])

    def test_training_readiness_requires_learner_and_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            run_dir = output_dir / "runs" / "run"
            run_dir.mkdir(parents=True)
            script = (
                "import pathlib,time; "
                f"p=pathlib.Path({str(run_dir)!r}); "
                "(p/'learner_ready.json').write_text('{}'); "
                "time.sleep(0.1); "
                "(p/'wandb_run_id.txt').write_text('abc123\\n'); "
                "(p/'wandb_url.txt').write_text('https://wandb.ai/e/p/runs/abc123\\n'); "
                "time.sleep(0.1)"
            )
            with (output_dir / "train.log").open("w", encoding="utf-8") as log:
                returncode = run_training_process(
                    [sys.executable, "-c", script],
                    log_file=log,
                    env=os.environ,
                    output_dir=output_dir,
                    run_dir=run_dir,
                    readiness_workers=[],
                    startup_timeout=2,
                )

            self.assertEqual(returncode, 0)
            readiness = json.loads((output_dir / "readiness.json").read_text(encoding="utf-8"))
            self.assertEqual(readiness["wandb_run_id"], "abc123")
            self.assertLessEqual(readiness["learner_ready_at"], readiness["wandb_ready_at"])

    def test_training_readiness_timeout_fails_without_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            run_dir = output_dir / "runs" / "run"
            run_dir.mkdir(parents=True)
            with (output_dir / "train.log").open("w", encoding="utf-8") as log:
                with self.assertRaisesRegex(RuntimeError, "did not reach learner/W&B readiness"):
                    run_training_process(
                        [sys.executable, "-c", "import time; time.sleep(10)"],
                        log_file=log,
                        env=os.environ,
                        output_dir=output_dir,
                        run_dir=run_dir,
                        readiness_workers=[],
                        startup_timeout=0.1,
                    )

            self.assertFalse((output_dir / "readiness.json").exists())

    def test_wandb_disabled_readiness_requires_only_learner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            run_dir = output_dir / "runs" / "run"
            run_dir.mkdir(parents=True)
            script = (
                "import pathlib,time; "
                f"p=pathlib.Path({str(run_dir)!r}); "
                "(p/'learner_ready.json').write_text('{}'); time.sleep(0.1)"
            )
            with (output_dir / "train.log").open("w", encoding="utf-8") as log:
                returncode = run_training_process(
                    [sys.executable, "-c", script],
                    log_file=log,
                    env=os.environ,
                    output_dir=output_dir,
                    run_dir=run_dir,
                    readiness_workers=[],
                    wandb_enabled=False,
                    startup_timeout=2,
                )

            self.assertEqual(returncode, 0)
            readiness = json.loads((output_dir / "readiness.json").read_text())
            self.assertIs(readiness["wandb_enabled"], False)
            self.assertNotIn("wandb_run_id", readiness)


if __name__ == "__main__":
    unittest.main()
