from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rlab.json_utils import json_safe
from rlab.job_execution import (
    collect_result_metadata,
    train_command_for_job,
    write_train_config_file,
)
from rlab.env import default_run_dir
from rlab.metric_store import MetricStore, metric_store_path


RESULT_SCHEMA_VERSION = 1
TRAIN_STARTUP_TIMEOUT_SECONDS = 300.0


def worker_modules(
    eval_backend: str,
    *,
    wandb_enabled: bool,
    telemetry_transport: str = "legacy_local",
) -> tuple[str | None, str | None]:
    if eval_backend not in {"local", "modal", "none"}:
        raise ValueError(f"unsupported checkpoint evaluation backend: {eval_backend}")
    producer = (
        "rlab.checkpoint_coordinator"
        if eval_backend == "modal"
        else "rlab.checkpoint_eval_worker"
        if eval_backend == "local"
        else None
    )
    publisher = (
        "rlab.telemetry_relay"
        if telemetry_transport == "neon_mailbox_v1"
        else "rlab.wandb_publisher"
        if wandb_enabled or eval_backend != "modal"
        else None
    )
    return producer, publisher


def publication_attempt_count(
    *,
    telemetry_transport: str,
    publisher_results: list[Mapping[str, Any]],
) -> int:
    """Count W&B attempts without treating mailbox relay execution as publication."""

    if telemetry_transport == "neon_mailbox_v1":
        return 0
    return int(bool(publisher_results))


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("job payload must be a JSON object")
    if payload.get("job_kind") != "train":
        raise ValueError("job payload must define job_kind=train")
    if not isinstance(payload.get("job"), dict):
        raise ValueError("job payload must define a job object")
    return payload


def write_atomic_json(path: Path, result: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(json_safe(dict(result)), indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".json.tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.chmod(0o644)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def write_result(output_dir: Path, result: Mapping[str, Any]) -> Path:
    return write_atomic_json(output_dir / "result.json", result)


def run_training_process(
    command: list[str],
    *,
    log_file,
    env: Mapping[str, str],
    output_dir: Path,
    run_dir: Path,
    readiness_workers: list[subprocess.Popen],
    wandb_enabled: bool = True,
    telemetry_transport: str = "legacy_local",
    configured_wandb_run_id: str | None = None,
    command_inbox_dir: Path | None = None,
    command_receipt_dir: Path | None = None,
    startup_timeout: float = TRAIN_STARTUP_TIMEOUT_SECONDS,
) -> int:
    """Run training while forwarding container termination as a graceful stop."""

    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=dict(env),
    )
    previous_handler = signal.getsignal(signal.SIGTERM)

    def request_graceful_stop(_signum, _frame) -> None:
        if process.poll() is None:
            graceful_signal = getattr(signal, "SIGUSR1", signal.SIGTERM)
            process.send_signal(graceful_signal)

    signal.signal(signal.SIGTERM, request_graceful_stop)

    def worker_log_tail(workers: list[subprocess.Popen]) -> str:
        excerpts = []
        for worker in workers:
            path = Path(str(getattr(worker, "_rlab_log_path", "")))
            if not path.is_file():
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                excerpts.append(f"{path.name}: {' | '.join(lines[-5:])}")
        return "; ".join(excerpts)

    try:
        deadline = time.monotonic() + startup_timeout
        readiness_written = False
        while True:
            learner_ready = run_dir / "learner_ready.json"
            wandb_run_id_path = run_dir / "wandb_run_id.txt"
            wandb_url_path = run_dir / "wandb_url.txt"
            relay_ready = run_dir / "mailbox_relay_ready.json"
            if learner_ready.is_file() and not readiness_written:
                receipt: dict[str, Any] = {
                    "schema_version": 1,
                    "ready": True,
                    "wandb_enabled": bool(wandb_enabled),
                    "learner_ready_at": datetime.fromtimestamp(
                        learner_ready.stat().st_mtime, UTC
                    ).isoformat(),
                }
                if telemetry_transport == "neon_mailbox_v1" and relay_ready.is_file():
                    receipt.update(
                        {
                            "telemetry_transport": telemetry_transport,
                            "mailbox_ready_at": datetime.fromtimestamp(
                                relay_ready.stat().st_mtime, UTC
                            ).isoformat(),
                            "wandb_run_id": configured_wandb_run_id,
                        }
                    )
                    write_atomic_json(output_dir / "readiness.json", receipt)
                    readiness_written = True
                elif not wandb_enabled:
                    write_atomic_json(output_dir / "readiness.json", receipt)
                    readiness_written = True
                if wandb_run_id_path.is_file() and wandb_url_path.is_file():
                    wandb_run_id = wandb_run_id_path.read_text(encoding="utf-8").strip()
                    wandb_url = wandb_url_path.read_text(encoding="utf-8").strip()
                    if wandb_run_id and wandb_url.startswith("https://wandb.ai/"):
                        receipt.update(
                            {
                                "wandb_ready_at": datetime.fromtimestamp(
                                    max(
                                        wandb_run_id_path.stat().st_mtime,
                                        wandb_url_path.stat().st_mtime,
                                    ),
                                    UTC,
                                ).isoformat(),
                                "wandb_run_id": wandb_run_id,
                                "wandb_url": wandb_url,
                            }
                        )
                        write_atomic_json(output_dir / "readiness.json", receipt)
                        readiness_written = True
            returncode = process.poll()
            if returncode is not None:
                return int(returncode)
            failed_worker = next(
                (worker for worker in readiness_workers if worker.poll() is not None),
                None,
            )
            if failed_worker is not None:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise RuntimeError(
                    "training worker exited before required readiness: "
                    f"{getattr(failed_worker, '_rlab_log_path', '')} "
                    f"returncode={failed_worker.returncode}; "
                    f"{worker_log_tail([failed_worker])}"
                )
            if command_inbox_dir is not None and process.poll() is None:
                for command_path in sorted(command_inbox_dir.glob("*.json")):
                    command = json.loads(command_path.read_text(encoding="utf-8"))
                    command_id = str(command.get("command_id") or "")
                    if not command_id:
                        raise RuntimeError(f"invalid command inbox entry: {command_path}")
                    graceful_signal = getattr(signal, "SIGUSR1", signal.SIGTERM)
                    process.send_signal(graceful_signal)
                    receipt_dir = command_receipt_dir or command_inbox_dir.parent / "receipts"
                    write_atomic_json(
                        receipt_dir / command_path.name,
                        {
                            "command_id": command_id,
                            "command_type": str(command.get("command_type") or ""),
                            "signal": int(graceful_signal),
                            "signal_sent_at": datetime.now(UTC).isoformat(),
                            "learner_pid": process.pid,
                        },
                    )
                    command_path.unlink(missing_ok=True)
            if not readiness_written and time.monotonic() >= deadline:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
                raise RuntimeError(
                    "training did not reach learner/W&B readiness requirements within "
                    f"{startup_timeout:g}s; "
                    f"{worker_log_tail(readiness_workers)}"
                )
            time.sleep(0.25)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def run_dir_from_config(config_path: Path) -> Path:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"train config must be a JSON object: {config_path}")
    run_name = str(config.get("run_name") or "run")
    runs_dir = str(config.get("runs_dir") or "runs")
    return Path(default_run_dir(run_name, runs_dir))


def start_worker(
    *,
    module: str,
    name: str,
    output_dir: Path,
    run_dir: Path,
    config_path: Path,
    stop_file: Path,
) -> subprocess.Popen:
    log_path = output_dir / "logs" / f"{name}_{uuid.uuid4().hex[:8]}.log"
    log_file = log_path.open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        module,
        "--run-dir",
        str(run_dir),
        "--train-config-json",
        str(config_path),
        "--stop-file",
        str(stop_file),
    ]
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    process._rlab_log_file = log_file  # type: ignore[attr-defined]
    process._rlab_log_path = log_path  # type: ignore[attr-defined]
    return process


def stop_workers(
    processes: list[subprocess.Popen], stop_file: Path, *, timeout: float = 30.0
) -> list[dict[str, Any]]:
    stop_file.parent.mkdir(parents=True, exist_ok=True)
    stop_file.write_text("stop\n", encoding="utf-8")
    results: list[dict[str, Any]] = []
    deadline = time.monotonic() + timeout
    for process in processes:
        remaining = max(deadline - time.monotonic(), 0.0)
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=5)
        log_file = getattr(process, "_rlab_log_file", None)
        if log_file is not None:
            log_file.close()
        results.append(
            {
                "pid": process.pid,
                "returncode": returncode,
                "log_path": str(getattr(process, "_rlab_log_path", "")),
            }
        )
    return results


def wait_for_mailbox_preflight(
    process: subprocess.Popen,
    ready_file: Path,
    *,
    timeout: float = 30.0,
) -> None:
    deadline = time.monotonic() + max(0.1, float(timeout))
    while time.monotonic() < deadline:
        if ready_file.is_file():
            return
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"telemetry mailbox preflight failed before training: exit={returncode}"
            )
        time.sleep(0.1)
    raise RuntimeError("telemetry mailbox preflight timed out before training")


def base_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    job = dict(payload["job"])
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "job_kind": payload.get("job_kind"),
        "job_id": int(job["id"]),
        "launch_id": payload.get("launch_id"),
        "machine": payload.get("machine"),
        "runtime_image_ref": payload.get("runtime_image_ref") or job.get("runtime_image_ref"),
    }


def run_train_payload(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    job = dict(payload["job"])
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_train_config_file(
        job,
        output_dir / "train_config.json",
        runs_dir=output_dir / "runs",
    )
    command = train_command_for_job(config_path)
    run_dir = run_dir_from_config(config_path)
    # Establish the ledger schema and persistent WAL mode before independent
    # publisher/coordinator processes begin opening it concurrently.
    MetricStore(metric_store_path(run_dir)).init()
    producer_stop_file = output_dir / "producers.stop"
    publisher_stop_file = output_dir / "publisher.stop"
    producer_workers: list[subprocess.Popen] = []
    publisher_workers: list[subprocess.Popen] = []
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    wandb_enabled = bool(config_document.get("wandb", False))
    eval_backend = str(config_document.get("checkpoint_eval_backend") or "local")
    telemetry_transport = str(
        config_document.get("telemetry_transport") or "legacy_local"
    )
    modal_eval = eval_backend == "modal"
    eval_disabled = eval_backend == "none"
    producer_module, publisher_module = worker_modules(
        eval_backend,
        wandb_enabled=wandb_enabled,
        telemetry_transport=telemetry_transport,
    )
    try:
        if publisher_module:
            publisher_workers.append(
                start_worker(
                    module=publisher_module,
                    name="wandb_publisher",
                    output_dir=output_dir,
                    run_dir=run_dir,
                    config_path=config_path,
                    stop_file=publisher_stop_file,
                )
            )
        if producer_module:
            producer_workers.append(
                start_worker(
                    module=producer_module,
                    name="checkpoint_coordinator" if modal_eval else "checkpoint_eval_worker",
                    output_dir=output_dir,
                    run_dir=run_dir,
                    config_path=config_path,
                    stop_file=producer_stop_file,
                )
            )
    except Exception:
        stop_workers(producer_workers, producer_stop_file)
        stop_workers(publisher_workers, publisher_stop_file)
        raise
    if telemetry_transport == "neon_mailbox_v1":
        try:
            wait_for_mailbox_preflight(
                publisher_workers[0],
                run_dir / "mailbox_relay_ready.json",
            )
        except Exception:
            stop_workers(producer_workers, producer_stop_file)
            stop_workers(publisher_workers, publisher_stop_file)
            raise
    log_path = log_dir / f"train_job_{job['id']}_{uuid.uuid4().hex[:8]}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            train_env = os.environ.copy()
            train_env["RLAB_INTERNAL_LEARNER"] = "1"
            returncode = run_training_process(
                command,
                log_file=log_file,
                env=train_env,
                output_dir=output_dir,
                run_dir=run_dir,
                readiness_workers=[*producer_workers, *publisher_workers],
                wandb_enabled=wandb_enabled,
                telemetry_transport=telemetry_transport,
                configured_wandb_run_id=str(config_document.get("wandb_run_id") or "") or None,
                command_inbox_dir=run_dir / "mailbox_commands" / "inbox",
                command_receipt_dir=run_dir / "mailbox_commands" / "receipts",
            )
    finally:
        producer_results = stop_workers(
            producer_workers, producer_stop_file, timeout=120.0 if modal_eval else 30.0
        )
        publisher_results = stop_workers(
            publisher_workers, publisher_stop_file, timeout=135.0
        )
        worker_results = [*producer_results, *publisher_results]
    metadata_job = dict(job)
    metadata_job["train_config"] = {
        **dict(job.get("train_config") or {}),
        "runs_dir": str(output_dir / "runs"),
    }
    metadata = collect_result_metadata(metadata_job)
    producer_failed = any(int(worker.get("returncode") or 0) != 0 for worker in producer_results)
    publisher_failed = any(
        int(worker.get("returncode") or 0) != 0 for worker in publisher_results
    )
    critical_worker_failure = (
        producer_failed or publisher_failed
        if telemetry_transport == "neon_mailbox_v1"
        else not modal_eval and (producer_failed or (not wandb_enabled and publisher_failed))
    )
    status = "succeeded" if returncode == 0 and not critical_worker_failure else "failed"
    publisher_drained = bool(publisher_results) and all(
        int(worker.get("returncode") or 0) == 0 for worker in publisher_results
    )
    publication_error = None
    if (
        wandb_enabled
        and telemetry_transport != "neon_mailbox_v1"
        and not publisher_drained
    ):
        publication_error = "live W&B publisher did not drain cleanly"
    result = {
        **base_result(payload),
        "status": status,
        "exit_code": returncode,
        "train": {
            "status": status,
            "result": metadata,
            "log_path": str(log_path),
        },
        "workers": worker_results,
        "live_publication": {
            "status": (
                "disabled"
                if not wandb_enabled
                else "pending"
                if telemetry_transport == "neon_mailbox_v1"
                else "complete"
                if publisher_drained
                else "pending"
            ),
            "attempts": publication_attempt_count(
                telemetry_transport=telemetry_transport,
                publisher_results=publisher_results,
            ),
            "error": publication_error,
        },
        "evaluation": {
            "backend": eval_backend,
            "status": "disabled" if eval_disabled else "enabled",
        },
    }
    if telemetry_transport == "neon_mailbox_v1":
        result["telemetry_handoff"] = {
            "transport": telemetry_transport,
            "status": "complete" if publisher_drained else "failed",
        }
    if modal_eval:
        artifact_phases = dict(metadata.get("phase_counts") or {})
        incomplete_artifacts = any(
            key.startswith("artifacts:")
            and key not in {"artifacts:uploaded", "artifacts:failed_terminal"}
            and int(count) > 0
            for key, count in artifact_phases.items()
        )
        coordinator_failed = any(
            "checkpoint_coordinator" in str(worker.get("log_path") or "")
            and int(worker.get("returncode") or 0) != 0
            for worker in worker_results
        )
        result["checkpoint_coordinator_status"] = (
            "awaiting_artifact_recovery"
            if incomplete_artifacts or coordinator_failed
            else "complete"
        )
    if returncode != 0:
        result["error"] = f"train process exited {returncode}"
    elif critical_worker_failure:
        result["error"] = "required local evaluation/artifact worker did not drain cleanly"
    if telemetry_transport == "neon_mailbox_v1" and status == "succeeded":
        store_path = metric_store_path(run_dir)
        for path in (
            store_path,
            store_path.with_name(store_path.name + "-wal"),
            store_path.with_name(store_path.name + "-shm"),
        ):
            path.unlink(missing_ok=True)
    return result


def run_payload(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if payload.get("job_kind") == "train":
        return run_train_payload(payload, output_dir)
    raise ValueError(f"unsupported job_kind: {payload.get('job_kind')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab run-job", description="Run one claimed rlab job payload and write result.json."
    )
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = load_payload(args.payload)
    try:
        result = run_payload(payload, args.output_dir)
    except Exception as exc:
        result = {
            **base_result(payload),
            "status": "failed",
            "exit_code": 1,
            "error": repr(exc),
        }
        write_result(args.output_dir, result)
        raise
    write_result(args.output_dir, result)
    return int(result.get("exit_code") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
