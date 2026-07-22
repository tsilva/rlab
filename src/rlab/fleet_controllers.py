from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from psycopg2 import Error as DatabaseError

from rlab.job_queue import connect, count_nonterminal_jobs, database_url
from rlab.fleet_service import (
    CONTROL_PLANE_PROTOCOL_VERSION,
    controller_service_paths,
    controller_source_fingerprint,
    default_service_paths,
)


POLL_SECONDS = 2.0
REMOTE_PASS_BUDGET_SECONDS = 55.0
MAX_WANDB_ACTORS = 16
WANDB_ACTOR_SHUTDOWN_SECONDS = 5.0
WANDB_ACTOR_SESSION_TIMEOUT_SECONDS = 120.0
WANDB_ACTOR_CONTENTION_BACKOFF_SECONDS = 30.0
WANDB_ACTOR_LOCK_BUSY_EXIT_CODE = 75
CONTROLLER_MAX_BACKOFF_SECONDS = 60.0


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_controller_readiness(
    path: Path,
    *,
    source_fingerprint: str,
    started_at: float,
    phase: str,
) -> None:
    _atomic_write_json(
        path,
        {
            "pid": os.getpid(),
            "started_at": started_at,
            "source_fingerprint": source_fingerprint,
            "protocol_version": CONTROL_PLANE_PROTOCOL_VERSION,
            "phase": phase,
            "last_reconciled_at": time.time(),
        },
    )


def _write_controller_heartbeat(
    path: Path,
    *,
    controller: str,
    source_fingerprint: str,
    started_at: float,
    phase: str,
    last_success_at: float | None,
    last_error: str | None,
) -> None:
    _atomic_write_json(
        path,
        {
            "pid": os.getpid(),
            "controller": controller,
            "started_at": started_at,
            "updated_at": time.time(),
            "source_fingerprint": source_fingerprint,
            "protocol_version": CONTROL_PLANE_PROTOCOL_VERSION,
            "phase": phase,
            "last_success_at": last_success_at,
            "last_error": last_error,
        },
    )


def _controller_state_paths(repo_root: Path, controller: str) -> tuple[Path, Path]:
    state = controller_service_paths(
        default_service_paths(repo_root=repo_root), controller
    ).state_dir
    return state / "heartbeat.json", state / "readiness.json"


def _controller_machines(repo_root: Path) -> tuple[str, ...]:
    from rlab.job_queue import machines_with_service_work
    from rlab.machines import load_machine_registry

    conn = connect(database_url(use_direct=True))
    try:
        selected = set(machines_with_service_work(conn))
    finally:
        conn.close()
    registry = load_machine_registry(repo_root / "experiments" / "machines.yaml")
    now = time.time()
    for name in registry.machines:
        marker = repo_root / "logs" / "fleet" / f"maintenance-{name}.stamp"
        try:
            maintenance_due = now - marker.stat().st_mtime >= 3600.0
        except FileNotFoundError:
            maintenance_due = True
        if maintenance_due:
            selected.add(name)
    return tuple(sorted(selected))


def _shutdown_wandb_actors(actors: dict[int, subprocess.Popen]) -> None:
    live = {run_id: process for run_id, process in actors.items() if process.poll() is None}
    for process in live.values():
        process.terminate()
    deadline = time.monotonic() + WANDB_ACTOR_SHUTDOWN_SECONDS
    while live and time.monotonic() < deadline:
        live = {run_id: process for run_id, process in live.items() if process.poll() is None}
        if live:
            time.sleep(0.05)
    if live:
        print(
            "force-killing W&B publisher actors: "
            + ", ".join(f"train/{run_id}=pid/{process.pid}" for run_id, process in live.items()),
            file=sys.stderr,
            flush=True,
        )
        for process in live.values():
            process.kill()
    for process in actors.values():
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)


def _wandb_actor_session_timed_out(
    repo_root: Path,
    *,
    run_id: int,
    process: subprocess.Popen,
    now: float | None = None,
) -> bool:
    path = repo_root / "logs" / "fleet" / "wandb-actors" / f"train-{int(run_id)}.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError, OSError, ValueError, TypeError:
        return False
    if int(state.get("pid") or 0) != int(process.pid):
        return False
    if str(state.get("phase") or "") != "publishing":
        return False
    try:
        started_at = float(state["session_started_at"])
    except KeyError, TypeError, ValueError:
        return False
    return (time.time() if now is None else float(now)) - started_at > (
        WANDB_ACTOR_SESSION_TIMEOUT_SECONDS
    )


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

    assertion = SleepAssertion()
    heartbeat, _readiness = _controller_state_paths(repo_root, "machine")
    source_fingerprint = controller_source_fingerprint(repo_root)
    started_at = time.time()
    last_success_at: float | None = None
    last_error: str | None = None
    backoff = POLL_SECONDS
    try:
        _write_controller_heartbeat(
            heartbeat,
            controller="machine",
            source_fingerprint=source_fingerprint,
            started_at=started_at,
            phase="starting",
            last_success_at=None,
            last_error=None,
        )
        while True:
            try:
                machines = _controller_machines(repo_root)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:4000]
                _write_controller_heartbeat(
                    heartbeat,
                    controller="machine",
                    source_fingerprint=source_fingerprint,
                    started_at=started_at,
                    phase="error",
                    last_success_at=last_success_at,
                    last_error=last_error,
                )
                if once:
                    raise
                time.sleep(backoff)
                backoff = min(CONTROLLER_MAX_BACKOFF_SECONDS, backoff * 2)
                continue
            assertion.update(bool(machines))
            _write_controller_heartbeat(
                heartbeat,
                controller="machine",
                source_fingerprint=source_fingerprint,
                started_at=started_at,
                phase="reconciling" if machines else "idle",
                last_success_at=last_success_at,
                last_error=last_error,
            )
            errors: list[str] = []
            database_error: DatabaseError | None = None
            for machine in machines:
                try:
                    run_service_machine_pass(
                        machine_name=machine,
                        machines_path=repo_root / "experiments" / "machines.yaml",
                        repo_root=repo_root,
                        deadline_monotonic=time.monotonic() + REMOTE_PASS_BUDGET_SECONDS,
                    )
                except DatabaseError as exc:
                    database_error = exc
                    break
                except Exception as exc:
                    errors.append(f"{machine}:{type(exc).__name__}:{exc}")
                    last_error = "; ".join(errors)[:4000]
                    _write_controller_heartbeat(
                        heartbeat,
                        controller="machine",
                        source_fingerprint=source_fingerprint,
                        started_at=started_at,
                        phase="degraded",
                        last_success_at=last_success_at,
                        last_error=last_error,
                    )
                else:
                    last_success_at = time.time()
                    _write_controller_heartbeat(
                        heartbeat,
                        controller="machine",
                        source_fingerprint=source_fingerprint,
                        started_at=started_at,
                        phase="reconciling",
                        last_success_at=last_success_at,
                        last_error="; ".join(errors)[:4000] or last_error,
                    )
            if database_error is not None:
                last_error = f"{type(database_error).__name__}: {database_error}"[:4000]
                _write_controller_heartbeat(
                    heartbeat,
                    controller="machine",
                    source_fingerprint=source_fingerprint,
                    started_at=started_at,
                    phase="error",
                    last_success_at=last_success_at,
                    last_error=last_error,
                )
                if once:
                    raise database_error
                time.sleep(backoff)
                backoff = min(CONTROLLER_MAX_BACKOFF_SECONDS, backoff * 2)
                continue
            if not machines:
                last_success_at = time.time()
            last_error = "; ".join(errors)[:4000] or None
            _write_controller_heartbeat(
                heartbeat,
                controller="machine",
                source_fingerprint=source_fingerprint,
                started_at=started_at,
                phase="degraded" if errors else "idle",
                last_success_at=last_success_at,
                last_error=last_error,
            )
            if once:
                return 0
            if errors:
                time.sleep(backoff)
                backoff = min(CONTROLLER_MAX_BACKOFF_SECONDS, backoff * 2)
            else:
                backoff = POLL_SECONDS
                time.sleep(POLL_SECONDS)
    finally:
        assertion.close()


def run_evaluation_controller(repo_root: Path, *, once: bool = False) -> int:
    from rlab.modal_eval_orchestrator import run_service_eval_pass
    from rlab.telemetry_mailbox import (
        consume_attempt_events,
        discard_disabled_metric_batches,
        finalize_mailbox_runs_without_eval,
    )

    assertion = SleepAssertion()
    heartbeat, _readiness = _controller_state_paths(repo_root, "evaluation")
    source_fingerprint = controller_source_fingerprint(repo_root)
    started_at = time.time()
    last_success_at: float | None = None
    last_error: str | None = None
    backoff = POLL_SECONDS
    try:
        _write_controller_heartbeat(
            heartbeat,
            controller="evaluation",
            source_fingerprint=source_fingerprint,
            started_at=started_at,
            phase="starting",
            last_success_at=None,
            last_error=None,
        )
        while True:
            try:
                assertion.update(_has_work())
                _write_controller_heartbeat(
                    heartbeat,
                    controller="evaluation",
                    source_fingerprint=source_fingerprint,
                    started_at=started_at,
                    phase="reconciling",
                    last_success_at=last_success_at,
                    last_error=last_error,
                )
                deadline = time.monotonic() + REMOTE_PASS_BUDGET_SECONDS
                run_service_eval_pass(repo_root=repo_root, deadline_monotonic=deadline)
                conn = connect(database_url(use_direct=True))
                try:
                    consume_attempt_events(conn)
                    discard_disabled_metric_batches(conn)
                    finalize_mailbox_runs_without_eval(conn)
                finally:
                    conn.close()
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:4000]
                _write_controller_heartbeat(
                    heartbeat,
                    controller="evaluation",
                    source_fingerprint=source_fingerprint,
                    started_at=started_at,
                    phase="error",
                    last_success_at=last_success_at,
                    last_error=last_error,
                )
                if once:
                    raise
                time.sleep(backoff)
                backoff = min(CONTROLLER_MAX_BACKOFF_SECONDS, backoff * 2)
                continue
            last_success_at = time.time()
            last_error = None
            _write_controller_heartbeat(
                heartbeat,
                controller="evaluation",
                source_fingerprint=source_fingerprint,
                started_at=started_at,
                phase="idle",
                last_success_at=last_success_at,
                last_error=None,
            )
            backoff = POLL_SECONDS
            if once:
                return 0
            time.sleep(POLL_SECONDS)
    finally:
        assertion.close()


def run_wandb_manager(repo_root: Path, *, once: bool = False) -> int:
    from rlab.telemetry_mailbox import (
        pending_metric_run_ids,
        schedule_artifact_publications,
    )

    actors: dict[int, subprocess.Popen] = {}
    ownership_backoff: dict[int, float] = {}
    assertion = SleepAssertion()
    heartbeat, readiness = _controller_state_paths(repo_root, "wandb")
    source_fingerprint = controller_source_fingerprint(repo_root)
    started_at = time.time()
    last_success_at: float | None = None
    backoff = POLL_SECONDS
    try:
        _write_controller_heartbeat(
            heartbeat,
            controller="wandb",
            source_fingerprint=source_fingerprint,
            started_at=started_at,
            phase="starting",
            last_success_at=None,
            last_error=None,
        )
        _write_controller_readiness(
            readiness,
            source_fingerprint=source_fingerprint,
            started_at=started_at,
            phase="starting",
        )
        while True:
            now = time.monotonic()
            for run_id, process in list(actors.items()):
                returncode = process.poll()
                if returncode is None and _wandb_actor_session_timed_out(
                    repo_root,
                    run_id=run_id,
                    process=process,
                ):
                    process.terminate()
                    try:
                        process.wait(timeout=WANDB_ACTOR_SHUTDOWN_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
                    actors.pop(run_id, None)
                    continue
                if returncode is None:
                    continue
                actors.pop(run_id, None)
                if returncode == WANDB_ACTOR_LOCK_BUSY_EXIT_CODE:
                    ownership_backoff[run_id] = now + WANDB_ACTOR_CONTENTION_BACKOFF_SECONDS
            try:
                conn = connect(database_url(use_direct=True))
                try:
                    schedule_artifact_publications(conn, limit=10)
                    run_ids = pending_metric_run_ids(conn, limit=10_000)
                finally:
                    conn.close()
            except Exception as exc:
                _write_controller_heartbeat(
                    heartbeat,
                    controller="wandb",
                    source_fingerprint=source_fingerprint,
                    started_at=started_at,
                    phase="error",
                    last_success_at=last_success_at,
                    last_error=f"{type(exc).__name__}: {exc}"[:4000],
                )
                if once:
                    raise
                time.sleep(backoff)
                backoff = min(CONTROLLER_MAX_BACKOFF_SECONDS, backoff * 2)
                continue
            assertion.update(bool(run_ids or actors))
            for run_id in run_ids:
                if len(actors) >= MAX_WANDB_ACTORS:
                    break
                if run_id in actors:
                    continue
                if ownership_backoff.get(run_id, 0.0) > now:
                    continue
                ownership_backoff.pop(run_id, None)
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
                    close_fds=True,
                )
            _write_controller_readiness(
                readiness,
                source_fingerprint=source_fingerprint,
                started_at=started_at,
                phase="reconciling" if run_ids or actors else "idle",
            )
            last_success_at = time.time()
            _write_controller_heartbeat(
                heartbeat,
                controller="wandb",
                source_fingerprint=source_fingerprint,
                started_at=started_at,
                phase="reconciling" if run_ids or actors else "idle",
                last_success_at=last_success_at,
                last_error=None,
            )
            backoff = POLL_SECONDS
            if once:
                for process in actors.values():
                    process.wait(timeout=60)
                return 0
            time.sleep(POLL_SECONDS)
    finally:
        try:
            _shutdown_wandb_actors(actors)
        finally:
            assertion.close()


def run_workspace_controller(repo_root: Path, *, once: bool = False) -> int:
    from rlab.artifact_durability import (
        ArtifactDurabilityPolicy,
        verify_due_artifact_receipts,
    )
    from rlab.workspace_gc import (
        WorkspaceNotReady,
        close_due_telemetry_obligations,
        due_proof_reduction_launches,
        finalize_host_deleted,
        reduce_cleanup_proof,
        workspace_protocol_mode,
    )

    assertion = SleepAssertion()
    heartbeat, readiness = _controller_state_paths(repo_root, "workspace")
    source_fingerprint = controller_source_fingerprint(repo_root)
    started_at = time.time()
    last_success_at: float | None = None
    backoff = POLL_SECONDS
    try:
        while True:
            reduced = 0
            pending = 0
            finalized = 0
            obligations_closed = 0
            artifacts_verified = 0
            error = None
            try:
                conn = connect(database_url(use_direct=True))
                try:
                    protocol_mode = workspace_protocol_mode(conn)
                    policy_path = str(
                        os.environ.get("RLAB_ARTIFACT_DURABILITY_POLICY_FILE") or ""
                    ).strip()
                    if protocol_mode != "dormant":
                        if not policy_path:
                            raise RuntimeError(
                                "RLAB_ARTIFACT_DURABILITY_POLICY_FILE is required outside "
                                "dormant workspace mode"
                            )
                        policy = ArtifactDurabilityPolicy.load(Path(policy_path))
                        artifacts_verified = verify_due_artifact_receipts(
                            conn, policy=policy, limit=8
                        )
                    obligations_closed = close_due_telemetry_obligations(conn, limit=100)
                    launch_ids = due_proof_reduction_launches(conn, limit=100)
                    for launch_id in launch_ids:
                        try:
                            reduce_cleanup_proof(conn, launch_id=launch_id)
                            reduced += 1
                        except WorkspaceNotReady:
                            pending += 1
                    finalized = finalize_host_deleted(conn, limit=100)
                finally:
                    conn.close()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"[:4000]
            phase = (
                "error"
                if error
                else "reconciling"
                if reduced or finalized or obligations_closed or artifacts_verified
                else "idle"
            )
            if not error:
                last_success_at = time.time()
            assertion.update(
                bool(pending or reduced or finalized or obligations_closed or artifacts_verified)
            )
            _write_controller_heartbeat(
                heartbeat,
                controller="workspace",
                source_fingerprint=source_fingerprint,
                started_at=started_at,
                phase=phase,
                last_success_at=last_success_at,
                last_error=error,
            )
            _write_controller_readiness(
                readiness,
                source_fingerprint=source_fingerprint,
                started_at=started_at,
                phase=phase,
            )
            if once:
                if error:
                    raise RuntimeError(error)
                return 0
            if error:
                time.sleep(backoff)
                backoff = min(CONTROLLER_MAX_BACKOFF_SECONDS, backoff * 2)
            else:
                backoff = POLL_SECONDS
                time.sleep(POLL_SECONDS)
    finally:
        assertion.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one isolated rlab fleet controller.")
    parser.add_argument("controller", choices=("machine", "evaluation", "wandb", "workspace"))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol-version", type=int, required=True)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.protocol_version != CONTROL_PLANE_PROTOCOL_VERSION:
        raise SystemExit(
            "fleet controller protocol mismatch: "
            f"installed={args.protocol_version} expected={CONTROL_PLANE_PROTOCOL_VERSION}"
        )
    repo_root = args.repo_root.expanduser().resolve()
    from rlab.fleet_service import _load_repo_environment

    _load_repo_environment(repo_root)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: raise_system_exit())
    if args.controller == "machine":
        return run_machine_controller(repo_root, once=args.once)
    if args.controller == "evaluation":
        return run_evaluation_controller(repo_root, once=args.once)
    if args.controller == "workspace":
        return run_workspace_controller(repo_root, once=args.once)
    return run_wandb_manager(repo_root, once=args.once)


def raise_system_exit() -> None:
    raise SystemExit(0)


if __name__ == "__main__":
    raise SystemExit(main())
