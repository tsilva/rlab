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
from pathlib import Path
from typing import Any

from rlab.json_utils import json_safe
from rlab.job_execution import (
    collect_result_metadata,
    train_command_for_job,
    write_train_config_file,
)
from rlab.env import default_run_dir


RESULT_SCHEMA_VERSION = 1


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("job payload must be a JSON object")
    if payload.get("job_kind") != "train":
        raise ValueError("job payload must define job_kind=train")
    if not isinstance(payload.get("job"), dict):
        raise ValueError("job payload must define a job object")
    return payload


def write_result(output_dir: Path, result: Mapping[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "result.json"
    payload = json.dumps(json_safe(dict(result)), indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=".result-",
        suffix=".json.tmp",
        dir=output_dir,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def run_training_process(
    command: list[str],
    *,
    log_file,
    env: Mapping[str, str],
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
    try:
        return int(process.wait())
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
    config_path = write_train_config_file(job, output_dir / "train_config.json")
    command = train_command_for_job(config_path)
    run_dir = run_dir_from_config(config_path)
    producer_stop_file = output_dir / "producers.stop"
    publisher_stop_file = output_dir / "publisher.stop"
    producer_workers: list[subprocess.Popen] = []
    publisher_workers: list[subprocess.Popen] = []
    config_document = json.loads(config_path.read_text(encoding="utf-8"))
    wandb_enabled = bool(config_document.get("wandb", False))
    try:
        publisher_workers.append(
            start_worker(
                module="rlab.wandb_publisher" if wandb_enabled else "rlab.artifact_worker",
                name="wandb_publisher" if wandb_enabled else "artifact_worker",
                output_dir=output_dir,
                run_dir=run_dir,
                config_path=config_path,
                stop_file=publisher_stop_file,
            )
        )
        producer_workers.append(
            start_worker(
                module="rlab.checkpoint_eval_worker",
                name="checkpoint_eval_worker",
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
    log_path = log_dir / f"train_job_{job['id']}_{uuid.uuid4().hex[:8]}.log"
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            train_env = os.environ.copy()
            if wandb_enabled:
                train_env["RLAB_EXTERNAL_WANDB_PUBLISHER"] = "1"
            returncode = run_training_process(
                command,
                log_file=log_file,
                env=train_env,
            )
    finally:
        worker_results = stop_workers(producer_workers, producer_stop_file)
        worker_results.extend(stop_workers(publisher_workers, publisher_stop_file, timeout=120.0))
    metadata = collect_result_metadata(job, log_path)
    status = "succeeded" if returncode == 0 else "failed"
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
    }
    if returncode != 0:
        result["error"] = f"train process exited {returncode}"
    return result


def run_payload(payload: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    if payload.get("job_kind") == "train":
        return run_train_payload(payload, output_dir)
    raise ValueError(f"unsupported job_kind: {payload.get('job_kind')}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one claimed rlab job payload and write result.json."
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
