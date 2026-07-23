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
WANDB_ACTOR_CLOSE_TIMEOUT_SECONDS = 300.0
WANDB_ACTOR_CONTENTION_BACKOFF_SECONDS = 30.0
WANDB_ACTOR_FAILURE_BACKOFF_SECONDS = 5.0
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


def _wandb_actor_state(
    repo_root: Path,
    *,
    run_id: int,
    process: subprocess.Popen,
) -> dict[str, object] | None:
    path = repo_root / "logs" / "fleet" / "wandb-actors" / f"train-{int(run_id)}.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError, OSError, ValueError, TypeError:
        return None
    if int(state.get("pid") or 0) != int(process.pid):
        return None
    return dict(state)


def _wandb_actors_starting(
    repo_root: Path,
    actors: dict[int, subprocess.Popen],
    *,
    source_fingerprint: str,
) -> bool:
    for run_id, process in actors.items():
        if process.poll() is not None:
            continue
        state = _wandb_actor_state(
            repo_root,
            run_id=run_id,
            process=process,
        )
        if (
            state is None
            or str(state.get("source_fingerprint") or "") != source_fingerprint
            or str(state.get("phase") or "") not in {"idle", "publishing"}
        ):
            return True
    return False


def _wandb_actor_watchdog_reason(
    repo_root: Path,
    *,
    run_id: int,
    process: subprocess.Popen,
    now: float | None = None,
) -> str | None:
    state = _wandb_actor_state(repo_root, run_id=run_id, process=process)
    if state is None:
        return None
    if str(state.get("phase") or "") != "publishing":
        return None
    current_time = time.time() if now is None else float(now)
    if str(state.get("stage") or "") == "session_closing":
        try:
            close_started_at = float(state["close_started_at"])
        except KeyError, TypeError, ValueError:
            close_started_at = 0.0
        if close_started_at and current_time - close_started_at > (
            WANDB_ACTOR_CLOSE_TIMEOUT_SECONDS
        ):
            return "close_timeout"
        return None
    try:
        progress_at = float(
            state.get("progress_at")
            or state.get("updated_at")
            or state["session_started_at"]
        )
    except KeyError, TypeError, ValueError:
        return None
    if current_time - progress_at > WANDB_ACTOR_SESSION_TIMEOUT_SECONDS:
        return "no_progress"
    return None


def _wandb_actor_session_timed_out(
    repo_root: Path,
    *,
    run_id: int,
    process: subprocess.Popen,
    now: float | None = None,
) -> bool:
    return (
        _wandb_actor_watchdog_reason(
            repo_root,
            run_id=run_id,
            process=process,
            now=now,
        )
        is not None
    )


def _recover_stalled_wandb_actor(
    repo_root: Path,
    *,
    run_id: int,
    process: subprocess.Popen,
    state: dict[str, object] | None,
    watchdog: str,
) -> int:
    lease_owner = str((state or {}).get("lease_owner") or "").strip()
    if not lease_owner:
        return 0
    from rlab.fleet_wandb_publisher import recover_stalled_actor_claim

    stage = str((state or {}).get("stage") or "unknown")
    error = (
        (f"W&B publisher session close exceeded {WANDB_ACTOR_CLOSE_TIMEOUT_SECONDS:g}s")
        if watchdog == "close_timeout"
        else (f"W&B publisher actor made no progress for {WANDB_ACTOR_SESSION_TIMEOUT_SECONDS:g}s")
    ) + f": stage={stage}"
    conn = connect(database_url(use_direct=True))
    try:
        return recover_stalled_actor_claim(
            conn,
            train_job_id=int(run_id),
            lease_owner=lease_owner,
            error=error,
            watchdog=watchdog,
        )
    finally:
        conn.close()


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


def _record_wandb_actor_failure(
    conn,
    *,
    run_id: int,
    error: str,
    returncode: int,
    retry_delay_seconds: float,
    state: dict[str, object] | None,
) -> None:
    from rlab.job_queue import record_job_event

    message = str(error)[:4000]
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs
                SET live_publication_status = CASE
                      WHEN live_publication_status IN ('complete', 'disabled', 'failed')
                      THEN live_publication_status ELSE 'pending' END,
                    live_publication_error = %(error)s,
                    live_publication_next_retry_at =
                      clock_timestamp() + (%(retry_delay)s * interval '1 second')
                WHERE id = %(run_id)s
                  AND live_publication_status NOT IN ('complete', 'disabled', 'failed')
                """,
                {
                    "run_id": int(run_id),
                    "error": message,
                    "retry_delay": max(float(retry_delay_seconds), 0.0),
                },
            )
            record_job_event(
                conn,
                job_id=int(run_id),
                event_type=(
                    "publisher_actor_start_failed"
                    if str((state or {}).get("phase") or "") == "startup_failed"
                    else "publisher_actor_exited"
                ),
                message=message,
                metadata={
                    "returncode": int(returncode),
                    "retry_delay_seconds": max(float(retry_delay_seconds), 0.0),
                    "actor_phase": str((state or {}).get("phase") or "unknown"),
                    "actor_stage": str((state or {}).get("stage") or "unknown"),
                    "actor_source_fingerprint": str((state or {}).get("source_fingerprint") or ""),
                    "expected_source_fingerprint": str(
                        (state or {}).get("expected_source_fingerprint") or ""
                    ),
                },
            )


def _record_wandb_actor_recovered(conn, *, run_id: int) -> None:
    from rlab.job_queue import record_job_event

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs
                SET live_publication_error = NULL,
                    live_publication_next_retry_at = NULL
                WHERE id = %(run_id)s
                  AND live_publication_status NOT IN ('complete', 'disabled', 'failed')
                  AND live_publication_error LIKE 'publisher actor%%'
                """,
                {"run_id": int(run_id)},
            )
            record_job_event(
                conn,
                job_id=int(run_id),
                event_type="publisher_actor_recovered",
                message="W&B publisher actor passed startup and source-fingerprint checks",
                metadata={},
            )


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
    actor_failures: dict[int, dict[str, object]] = {}
    pending_failure_records: list[dict[str, object]] = []
    pending_recoveries: set[int] = set()
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
            observed_source_fingerprint = controller_source_fingerprint(repo_root)
            if observed_source_fingerprint != source_fingerprint:
                _shutdown_wandb_actors(actors)
                actors.clear()
                error = (
                    "W&B controller source changed while running; reload required: "
                    f"started={source_fingerprint}, observed={observed_source_fingerprint}"
                )
                _write_controller_readiness(
                    readiness,
                    source_fingerprint=source_fingerprint,
                    started_at=started_at,
                    phase="error",
                )
                _write_controller_heartbeat(
                    heartbeat,
                    controller="wandb",
                    source_fingerprint=source_fingerprint,
                    started_at=started_at,
                    phase="error",
                    last_success_at=last_success_at,
                    last_error=error,
                )
                if once:
                    raise RuntimeError(error)
                time.sleep(backoff)
                backoff = min(CONTROLLER_MAX_BACKOFF_SECONDS, backoff * 2)
                continue
            for run_id, process in list(actors.items()):
                returncode = process.poll()
                state = _wandb_actor_state(
                    repo_root,
                    run_id=run_id,
                    process=process,
                )
                if (
                    returncode is None
                    and state is not None
                    and str(state.get("source_fingerprint") or "") != source_fingerprint
                ):
                    process.terminate()
                    try:
                        process.wait(timeout=WANDB_ACTOR_SHUTDOWN_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
                    returncode = process.returncode or 1
                    state["phase"] = "startup_failed"
                    state["stage"] = "source_fingerprint_mismatch"
                    state["error"] = (
                        "publisher actor source fingerprint mismatch: "
                        f"manager={source_fingerprint}, "
                        f"actor={state.get('source_fingerprint') or 'unknown'}"
                    )
                watchdog = (
                    None
                    if returncode is not None
                    else _wandb_actor_watchdog_reason(
                        repo_root,
                        run_id=run_id,
                        process=process,
                    )
                )
                if returncode is None and watchdog is not None:
                    state = _wandb_actor_state(
                        repo_root,
                        run_id=run_id,
                        process=process,
                    )
                    process.terminate()
                    try:
                        process.wait(timeout=WANDB_ACTOR_SHUTDOWN_SECONDS)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
                    actors.pop(run_id, None)
                    try:
                        _recover_stalled_wandb_actor(
                            repo_root,
                            run_id=run_id,
                            process=process,
                            state=state,
                            watchdog=watchdog,
                        )
                    except Exception as exc:
                        print(
                            f"failed to release stalled W&B actor claim for train/{run_id}: "
                            f"{type(exc).__name__}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                    continue
                if returncode is None:
                    if (
                        run_id in actor_failures
                        and state is not None
                        and str(state.get("source_fingerprint") or "") == source_fingerprint
                        and str(state.get("phase") or "") in {"idle", "publishing"}
                    ):
                        actor_failures.pop(run_id, None)
                        pending_recoveries.add(run_id)
                    continue
                actors.pop(run_id, None)
                if returncode == WANDB_ACTOR_LOCK_BUSY_EXIT_CODE:
                    ownership_backoff[run_id] = now + WANDB_ACTOR_CONTENTION_BACKOFF_SECONDS
                    continue
                if returncode == 0:
                    if run_id in actor_failures:
                        actor_failures.pop(run_id, None)
                        pending_recoveries.add(run_id)
                    continue
                previous_delay = float((actor_failures.get(run_id) or {}).get("delay") or 0.0)
                delay = min(
                    CONTROLLER_MAX_BACKOFF_SECONDS,
                    max(
                        WANDB_ACTOR_FAILURE_BACKOFF_SECONDS,
                        previous_delay * 2,
                    ),
                )
                error = str(
                    (state or {}).get("error")
                    or f"publisher actor exited unexpectedly: returncode={returncode}"
                )[:4000]
                actor_failures[run_id] = {
                    "next_at": now + delay,
                    "delay": delay,
                    "error": error,
                    "returncode": int(returncode),
                }
                pending_failure_records.append(
                    {
                        "run_id": int(run_id),
                        "error": error,
                        "returncode": int(returncode),
                        "retry_delay_seconds": delay,
                        "state": state,
                    }
                )
            try:
                conn = connect(database_url(use_direct=True))
                try:
                    for failure in pending_failure_records:
                        _record_wandb_actor_failure(conn, **failure)
                    pending_failure_records.clear()
                    for run_id in sorted(pending_recoveries):
                        _record_wandb_actor_recovered(conn, run_id=run_id)
                    pending_recoveries.clear()
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
                if float((actor_failures.get(run_id) or {}).get("next_at") or 0.0) > now:
                    continue
                ownership_backoff.pop(run_id, None)
                try:
                    actors[run_id] = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "rlab.fleet_wandb_publisher",
                            "--train-job-id",
                            str(run_id),
                            "--expected-source-fingerprint",
                            source_fingerprint,
                        ],
                        cwd=repo_root,
                        stdin=subprocess.DEVNULL,
                        close_fds=True,
                    )
                except OSError as exc:
                    previous_delay = float((actor_failures.get(run_id) or {}).get("delay") or 0.0)
                    delay = min(
                        CONTROLLER_MAX_BACKOFF_SECONDS,
                        max(
                            WANDB_ACTOR_FAILURE_BACKOFF_SECONDS,
                            previous_delay * 2,
                        ),
                    )
                    error = (f"publisher actor process start failed: {type(exc).__name__}: {exc}")[
                        :4000
                    ]
                    actor_failures[run_id] = {
                        "next_at": now + delay,
                        "delay": delay,
                        "error": error,
                        "returncode": 1,
                    }
                    pending_failure_records.append(
                        {
                            "run_id": int(run_id),
                            "error": error,
                            "returncode": 1,
                            "retry_delay_seconds": delay,
                            "state": None,
                        }
                    )
            actor_starting = _wandb_actors_starting(
                repo_root,
                actors,
                source_fingerprint=source_fingerprint,
            )
            phase = (
                "error"
                if actor_failures
                else (
                    "starting"
                    if actor_starting
                    else ("reconciling" if run_ids or actors else "idle")
                )
            )
            actor_error = (
                "; ".join(
                    f"train/{run_id}: {failure.get('error')}"
                    for run_id, failure in sorted(actor_failures.items())
                )[:4000]
                if actor_failures
                else None
            )
            _write_controller_readiness(
                readiness,
                source_fingerprint=source_fingerprint,
                started_at=started_at,
                phase=phase,
            )
            if not actor_failures:
                last_success_at = time.time()
            _write_controller_heartbeat(
                heartbeat,
                controller="wandb",
                source_fingerprint=source_fingerprint,
                started_at=started_at,
                phase=phase,
                last_success_at=last_success_at,
                last_error=actor_error,
            )
            backoff = POLL_SECONDS
            if once:
                for run_id, process in actors.items():
                    returncode = process.wait(timeout=60)
                    if returncode not in {0, WANDB_ACTOR_LOCK_BUSY_EXIT_CODE}:
                        raise RuntimeError(
                            f"W&B publisher actor train/{run_id} exited {returncode}"
                        )
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
        quarantine_expired_host_operation_leases,
        record_proof_reducer_result,
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
            expired_operations = 0
            error = None
            try:
                conn = connect(database_url(use_direct=True))
                try:
                    expired_operations = quarantine_expired_host_operation_leases(
                        conn, limit=25
                    )
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
                            record_proof_reducer_result(
                                conn, launch_id=launch_id, ready=True
                            )
                            reduced += 1
                        except WorkspaceNotReady as exc:
                            record_proof_reducer_result(
                                conn, launch_id=launch_id, ready=False, error=str(exc)
                            )
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
                if reduced
                or finalized
                or obligations_closed
                or artifacts_verified
                or expired_operations
                else "idle"
            )
            if not error:
                last_success_at = time.time()
            assertion.update(
                bool(
                    pending
                    or reduced
                    or finalized
                    or obligations_closed
                    or artifacts_verified
                    or expired_operations
                )
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
