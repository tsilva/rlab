from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from rlab.run_job import (
    RequiredWorkerExited,
    run_train_payload,
    run_training_process,
    write_result,
)


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

    def test_required_worker_exit_reports_startup_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            run_dir = output_dir / "runs" / "run"
            run_dir.mkdir(parents=True)
            worker_log_path = output_dir / "worker.log"
            with worker_log_path.open("w", encoding="utf-8") as worker_log:
                worker = subprocess.Popen(
                    [sys.executable, "-c", "raise SystemExit(7)"],
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                worker._rlab_name = "checkpoint_coordinator"  # type: ignore[attr-defined]
                worker._rlab_log_path = worker_log_path  # type: ignore[attr-defined]
                with (output_dir / "train.log").open("w", encoding="utf-8") as log:
                    with self.assertRaises(RequiredWorkerExited) as caught:
                        run_training_process(
                            [sys.executable, "-c", "import time; time.sleep(10)"],
                            log_file=log,
                            env=os.environ,
                            output_dir=output_dir,
                            run_dir=run_dir,
                            readiness_workers=[worker],
                            wandb_enabled=False,
                            startup_timeout=2,
                        )
            worker.wait(timeout=2)
            self.assertEqual(caught.exception.phase, "startup")
            self.assertEqual(caught.exception.component, "checkpoint_coordinator")
            self.assertEqual(caught.exception.returncode, 7)

    def test_required_worker_exit_reports_running_phase_after_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            run_dir = output_dir / "runs" / "run"
            run_dir.mkdir(parents=True)
            learner = (
                "import pathlib,time; "
                f"p=pathlib.Path({str(run_dir)!r}); "
                "(p/'learner_ready.json').write_text('{}'); "
                "time.sleep(10)"
            )
            worker_log_path = output_dir / "worker.log"
            with worker_log_path.open("w", encoding="utf-8") as worker_log:
                worker = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(0.3); raise SystemExit(9)"],
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                worker._rlab_name = "checkpoint_coordinator"  # type: ignore[attr-defined]
                worker._rlab_log_path = worker_log_path  # type: ignore[attr-defined]
                with (output_dir / "train.log").open("w", encoding="utf-8") as log:
                    with self.assertRaises(RequiredWorkerExited) as caught:
                        run_training_process(
                            [sys.executable, "-c", learner],
                            log_file=log,
                            env=os.environ,
                            output_dir=output_dir,
                            run_dir=run_dir,
                            readiness_workers=[worker],
                            wandb_enabled=False,
                            startup_timeout=2,
                        )
            worker.wait(timeout=2)
            self.assertTrue((output_dir / "readiness.json").is_file())
            self.assertEqual(caught.exception.phase, "running")
            self.assertEqual(caught.exception.returncode, 9)

    def test_train_payload_preserves_structured_failure_and_durability_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            run_dir = output_dir / "runs" / "run"
            config_path = output_dir / "train_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "run_name": "run",
                        "runs_dir": str(output_dir / "runs"),
                        "wandb": True,
                        "wandb_run_id": "rlab-run",
                        "checkpoint_eval_backend": "none",
                        "telemetry_transport": "neon_mailbox_v1",
                    }
                ),
                encoding="utf-8",
            )
            publisher = mock.Mock(pid=11)
            coordinator = mock.Mock(pid=12)
            failure = RequiredWorkerExited(
                component="checkpoint_coordinator",
                phase="running",
                returncode=1,
                log_path="/output/checkpoint.log",
                log_tail="ledger replay conflict",
            )
            payload = {
                "job_kind": "train",
                "launch_id": "train-7",
                "machine": "beast-3",
                "runtime_image_ref": "docker:example@sha256:" + "a" * 64,
                "job": {
                    "id": 7,
                    "runtime_image_ref": "docker:example@sha256:" + "a" * 64,
                    "train_config": {},
                },
            }

            with (
                mock.patch(
                    "rlab.run_job.write_train_config_file",
                    return_value=config_path,
                ),
                mock.patch("rlab.run_job.train_command_for_job", return_value=["train"]),
                mock.patch("rlab.run_job.run_dir_from_config", return_value=run_dir),
                mock.patch(
                    "rlab.run_job.start_worker",
                    side_effect=[publisher, coordinator],
                ),
                mock.patch("rlab.run_job.wait_for_mailbox_preflight"),
                mock.patch("rlab.run_job.run_training_process", side_effect=failure),
                mock.patch(
                    "rlab.run_job.stop_workers",
                    side_effect=[
                        [
                            {
                                "pid": 12,
                                "returncode": 1,
                                "log_path": "/output/checkpoint_coordinator.log",
                            }
                        ],
                        [
                            {
                                "pid": 11,
                                "returncode": 0,
                                "log_path": "/output/wandb_publisher.log",
                            }
                        ],
                    ],
                ),
                mock.patch(
                    "rlab.run_job.collect_result_metadata",
                    return_value={"phase_counts": {"artifacts:pending": 1}},
                ),
            ):
                result = run_train_payload(payload, output_dir)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 1)
        self.assertEqual(result["failure"]["phase"], "running")
        self.assertEqual(result["failure"]["component"], "checkpoint_coordinator")
        self.assertEqual(result["telemetry_handoff"]["status"], "complete")
        self.assertEqual(
            result["checkpoint_coordinator_status"],
            "awaiting_artifact_recovery",
        )
        self.assertTrue(result["durability_finalization_required"])
        self.assertEqual(len(result["workers"]), 2)

    def test_command_inbox_signals_learner_writes_receipt_and_removes_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            run_dir = output_dir / "runs" / "run"
            inbox_dir = run_dir / "mailbox_commands" / "inbox"
            receipt_dir = run_dir / "mailbox_commands" / "receipts"
            inbox_dir.mkdir(parents=True)
            command_path = inbox_dir / "command.json"
            observed = run_dir / "child_signal_observed.json"
            script = "\n".join(
                [
                    "import pathlib, signal, sys, time",
                    f"run_dir = pathlib.Path({str(run_dir)!r})",
                    f"observed = pathlib.Path({str(observed)!r})",
                    "def stop(_signum, _frame):",
                    "    observed.write_text('observed\\n')",
                    "    raise SystemExit(0)",
                    "signal.signal(signal.SIGUSR1, stop)",
                    "(run_dir / 'learner_ready.json').write_text('{}')",
                    "while True:",
                    "    time.sleep(0.05)",
                ]
            )
            learner_ready = run_dir / "learner_ready.json"

            def deliver_after_learner_ready() -> None:
                deadline = time.monotonic() + 2
                while not learner_ready.is_file() and time.monotonic() < deadline:
                    time.sleep(0.01)
                command_path.write_text(
                    json.dumps(
                        {
                            "command_id": "acceptance-stop:17",
                            "command_type": "stop",
                            "payload": {"checkpoint_step": 500000},
                        }
                    ),
                    encoding="utf-8",
                )

            delivery = threading.Thread(target=deliver_after_learner_ready)
            delivery.start()
            with (output_dir / "train.log").open("w", encoding="utf-8") as log:
                returncode = run_training_process(
                    [sys.executable, "-c", script],
                    log_file=log,
                    env=os.environ,
                    output_dir=output_dir,
                    run_dir=run_dir,
                    readiness_workers=[],
                    wandb_enabled=False,
                    command_inbox_dir=inbox_dir,
                    command_receipt_dir=receipt_dir,
                    startup_timeout=2,
                )
            delivery.join(timeout=2)

            self.assertEqual(returncode, 0)
            self.assertTrue(observed.is_file())
            self.assertFalse(command_path.exists())
            receipt = json.loads(
                (receipt_dir / "command.json").read_text(encoding="utf-8")
            )
            self.assertEqual(receipt["command_id"], "acceptance-stop:17")
            self.assertEqual(receipt["command_type"], "stop")
            self.assertTrue(receipt["signal_sent_at"])


if __name__ == "__main__":
    unittest.main()
