from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from rlab.job_queue import connect, count_nonterminal_jobs, database_url


POLL_SECONDS = 2.0
REMOTE_PASS_BUDGET_SECONDS = 55.0


class SleepAssertion:
    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None

    def update(self, needed: bool) -> None:
        if needed and self.process is None and Path("/usr/bin/caffeinate").is_file():
            self.process = subprocess.Popen(
                ["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
        elif not needed and self.process is not None:
            self.process.terminate()
            self.process.wait(timeout=5)
            self.process = None

    def close(self) -> None:
        self.update(False)


def _has_work() -> bool:
    conn = connect(database_url(use_direct=True))
    try:
        return count_nonterminal_jobs(conn) > 0
    finally:
        conn.close()


def run_machine_controller(repo_root: Path, *, once: bool = False) -> int:
    from rlab.fleet import run_service_machine_pass
    from rlab.job_queue import machines_with_service_work

    assertion = SleepAssertion()
    try:
        while True:
            conn = connect(database_url(use_direct=True))
            try:
                machines = machines_with_service_work(conn)
            finally:
                conn.close()
            assertion.update(bool(machines))
            deadline = time.monotonic() + REMOTE_PASS_BUDGET_SECONDS
            for machine in machines:
                run_service_machine_pass(
                    machine_name=machine,
                    machines_path=repo_root / "experiments" / "machines.yaml",
                    repo_root=repo_root,
                    deadline_monotonic=deadline,
                )
            if once:
                return 0
            time.sleep(POLL_SECONDS)
    finally:
        assertion.close()


def run_evaluation_controller(repo_root: Path, *, once: bool = False) -> int:
    from rlab.job_queue import admit_pending_eval_load
    from rlab.modal_eval_orchestrator import run_service_eval_pass
    from rlab.telemetry_mailbox import (
        consume_attempt_events,
        discard_disabled_metric_batches,
        finalize_mailbox_runs_without_eval,
    )

    assertion = SleepAssertion()
    try:
        while True:
            assertion.update(_has_work())
            conn = connect(database_url(use_direct=True))
            try:
                admit_pending_eval_load(conn)
            finally:
                conn.close()
            deadline = time.monotonic() + REMOTE_PASS_BUDGET_SECONDS
            run_service_eval_pass(repo_root=repo_root, deadline_monotonic=deadline)
            conn = connect(database_url(use_direct=True))
            try:
                consume_attempt_events(conn)
                discard_disabled_metric_batches(conn)
                finalize_mailbox_runs_without_eval(conn)
            finally:
                conn.close()
            if once:
                return 0
            time.sleep(POLL_SECONDS)
    finally:
        assertion.close()


def run_wandb_manager(repo_root: Path, *, once: bool = False) -> int:
    from rlab.telemetry_mailbox import pending_metric_run_ids

    actors: dict[int, subprocess.Popen] = {}
    assertion = SleepAssertion()
    try:
        while True:
            actors = {
                run_id: process
                for run_id, process in actors.items()
                if process.poll() is None
            }
            conn = connect(database_url(use_direct=True))
            try:
                run_ids = pending_metric_run_ids(conn, limit=10_000)
            finally:
                conn.close()
            assertion.update(bool(run_ids or actors))
            for run_id in run_ids:
                if run_id in actors:
                    continue
                actors[run_id] = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "rlab.fleet_wandb_publisher",
                        "--train-job-id",
                        str(run_id),
                    ],
                    cwd=repo_root,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                    close_fds=True,
                )
            if once:
                for process in actors.values():
                    process.wait(timeout=60)
                return 0
            time.sleep(POLL_SECONDS)
    finally:
        assertion.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated rlab fleet controller.")
    parser.add_argument("controller", choices=("machine", "evaluation", "wandb"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    from rlab.fleet_service import _load_repo_environment

    _load_repo_environment(repo_root)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: raise_system_exit())
    if args.controller == "machine":
        return run_machine_controller(repo_root, once=args.once)
    if args.controller == "evaluation":
        return run_evaluation_controller(repo_root, once=args.once)
    return run_wandb_manager(repo_root, once=args.once)


def raise_system_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    raise SystemExit(main())
