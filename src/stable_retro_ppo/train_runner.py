from __future__ import annotations

import argparse
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from stable_retro_ppo.campaign import (
    campaign_status,
    claim_train_job,
    connect,
    database_url,
    finish_train_job,
    heartbeat_train_job,
    new_worker_id,
    print_status,
)
from stable_retro_ppo.cli import build_train_command


ARTIFACT_RE = re.compile(r"wandb artifact logged: (?P<name>[^ ]+) \((?P<location>[^)]+)\)")
METRIC_ROW_RE = re.compile(r"\|\s+(?P<key>[A-Za-z0-9_./-]+)\s+\|\s+(?P<value>[^|]+?)\s+\|")
WANDB_RUN_URL_RE = re.compile(r"https://wandb\.ai/\S+/runs/[A-Za-z0-9_-]+")


def normalize_train_config(job: dict[str, Any]) -> dict[str, Any]:
    config = dict(job.get("train_config") or {})
    run_name = job.get("run_name") or config.get("run_name") or f"train_job_{job['id']}"
    config["run_name"] = run_name
    if job.get("run_description"):
        config["run_description"] = job["run_description"]
    if job.get("wandb_group"):
        config["wandb_group"] = job["wandb_group"]
    tags = job.get("wandb_tags") or []
    if tags and not config.get("wandb_tags"):
        config["wandb_tags"] = ",".join(str(tag) for tag in tags)
    if isinstance(config.get("wandb_tags"), list):
        config["wandb_tags"] = ",".join(str(tag) for tag in config["wandb_tags"])
    return config


def train_command_for_job(job: dict[str, Any]) -> list[str]:
    cmd = build_train_command(normalize_train_config(job))
    if cmd[:3] != ["python", "-m", "stable_retro_ppo.train"]:
        raise ValueError("unexpected train command prefix")
    return [sys.executable, *cmd[1:]]


def read_text_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def parse_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def parse_metric_value(value: str) -> int | float | str:
    text = value.strip().replace(",", "")
    try:
        number = float(text)
    except ValueError:
        return value.strip()
    if number.is_integer() and not any(marker in text.lower() for marker in (".", "e")):
        return int(number)
    return number


def parse_log_metrics(log_text: str) -> dict[str, int | float | str]:
    metrics: dict[str, int | float | str] = {}
    section = ""
    for line in log_text.splitlines():
        match = METRIC_ROW_RE.search(line)
        if not match:
            continue
        key = match.group("key").strip()
        value = match.group("value").strip()
        if key.endswith("/") and not value:
            section = key.rstrip("/")
            continue
        metric_key = key if "/" in key or not section else f"{section}/{key}"
        parsed = parse_metric_value(value)
        metrics[metric_key] = parsed
        if key in {"total_timesteps", "completion_episode_rate", "completion_episodes_total"}:
            metrics[key] = parsed
    return metrics


def parse_wandb_run_url(log_text: str) -> str | None:
    matches = WANDB_RUN_URL_RE.findall(log_text)
    return matches[-1] if matches else None


def collect_result_metadata(job: dict[str, Any], log_path: Path) -> dict[str, Any]:
    config = normalize_train_config(job)
    run_name = str(config["run_name"])
    runs_dir = str(config.get("runs_dir") or "runs")
    run_dir = Path(runs_dir) / run_name
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    artifact_refs = [
        {"name": match.group("name"), "location": match.group("location")}
        for match in ARTIFACT_RE.finditer(log_text)
    ]
    metrics = parse_log_metrics(log_text)
    metrics.update(parse_key_value_file(run_dir / "early_stop.txt"))
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "final_model_path": str(run_dir / "final_model.zip")
        if (run_dir / "final_model.zip").is_file()
        else None,
        "wandb_run_id": read_text_file(run_dir / "wandb_run_id.txt"),
        "wandb_url": read_text_file(run_dir / "wandb_url.txt") or parse_wandb_run_url(log_text),
        "artifact_refs": artifact_refs,
        "metrics_json": metrics,
    }


def terminate_process(process: subprocess.Popen[str], *, grace_seconds: float = 20.0) -> None:
    if process.poll() is not None:
        return
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.5)
    if process.poll() is None:
        process.kill()


def run_training_job(
    conn,
    *,
    job: dict[str, Any],
    worker_id: str,
    lease_seconds: int,
    heartbeat_interval: float,
    log_dir: Path,
    dry_run: bool = False,
) -> str:
    command = train_command_for_job(job)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"train_job_{job['id']}_{uuid.uuid4().hex[:8]}.log"
    print(f"train_job={job['id']} command={' '.join(command)}", flush=True)

    if dry_run:
        finish_train_job(
            conn,
            job=job,
            worker_id=worker_id,
            status="succeeded",
            exit_code=0,
            result={
                "run_name": normalize_train_config(job)["run_name"],
                "metrics_json": {"dry_run": True},
            },
        )
        return "succeeded"

    canceled = False
    drain_after_job = bool(job.get("drain_requested"))
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        output_queue: queue.Queue[str | None] = queue.Queue()

        def read_output() -> None:
            assert process.stdout is not None
            for output_line in process.stdout:
                output_queue.put(output_line)
            output_queue.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        last_heartbeat = 0.0
        saw_eof = False
        try:
            while True:
                try:
                    line = output_queue.get(timeout=0.5)
                except queue.Empty:
                    line = ""
                if line is None:
                    saw_eof = True
                    line = ""
                elif line:
                    log_file.write(line)
                    log_file.flush()
                    if (
                        "wandb.ai/" in line
                        or "wandb artifact logged:" in line
                        or "early stop" in line.lower()
                    ):
                        print(line.rstrip(), flush=True)
                if process.poll() is not None and saw_eof:
                    break
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_interval:
                    heartbeat = heartbeat_train_job(
                        conn,
                        job_id=int(job["id"]),
                        worker_id=worker_id,
                        lease_seconds=lease_seconds,
                    )
                    last_heartbeat = now
                    if heartbeat is None:
                        terminate_process(process)
                        canceled = True
                        break
                    if heartbeat.get("cancel_requested"):
                        terminate_process(process)
                        canceled = True
                        break
                    if heartbeat.get("drain_requested"):
                        drain_after_job = True
        finally:
            if process.poll() is None:
                terminate_process(process)
            reader.join(timeout=5.0)

    exit_code = process.returncode
    result = collect_result_metadata(job, log_path)
    if canceled:
        finish_train_job(
            conn,
            job=job,
            worker_id=worker_id,
            status="canceled",
            exit_code=exit_code,
            result=result,
            error="cancel requested or lease lost",
        )
        return "canceled"
    if exit_code == 0:
        finish_train_job(
            conn,
            job=job,
            worker_id=worker_id,
            status="succeeded",
            exit_code=exit_code,
            result=result,
        )
        if drain_after_job:
            return "succeeded_drained"
        return "succeeded"
    finish_train_job(
        conn,
        job=job,
        worker_id=worker_id,
        status="failed",
        exit_code=exit_code,
        result=result,
        error=f"train process exited {exit_code}",
    )
    return "failed"


def worker_loop(args: argparse.Namespace, *, worker_id: str) -> None:
    conn = connect(database_url(args.direct))
    completed = 0
    try:
        while args.max_jobs <= 0 or completed < args.max_jobs:
            job = claim_train_job(
                conn,
                profile_id=args.profile,
                worker_id=worker_id,
                lease_seconds=args.lease_seconds,
            )
            if job is None:
                if args.once:
                    return
                time.sleep(args.poll_seconds)
                continue
            status = run_training_job(
                conn,
                job=job,
                worker_id=worker_id,
                lease_seconds=args.lease_seconds,
                heartbeat_interval=args.heartbeat_seconds,
                log_dir=Path(args.log_dir),
                dry_run=args.dry_run,
            )
            completed += 1
            print(f"worker={worker_id} train_job={job['id']} status={status}", flush=True)
            if job.get("drain_requested") or status.endswith("_drained"):
                return
    finally:
        conn.close()


def run_pool(args: argparse.Namespace) -> int:
    workers = max(args.workers, 1)
    if workers == 1:
        worker_loop(args, worker_id=args.worker_id or new_worker_id("train-runner"))
        return 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                worker_loop,
                args,
                worker_id=f"{args.worker_id or 'train-runner'}-{index}-{uuid.uuid4().hex[:8]}",
            )
            for index in range(workers)
        ]
        for future in futures:
            future.result()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drain Codex-authored PPO train jobs.")
    parser.add_argument("--profile", required=True, help="Exact train_jobs.profile_id to claim.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-id")
    parser.add_argument("--lease-seconds", type=int, default=1800)
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--max-jobs", type=int, default=0, help="0 means unlimited.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Exit when no matching job is available.",
    )
    parser.add_argument("--log-dir", default="logs/train_runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Claim and complete jobs without training.",
    )
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    parser.add_argument(
        "--status-goal",
        help="Print compact campaign status for this goal before starting workers.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.status_goal:
        conn = connect(database_url(args.direct))
        try:
            print_status(campaign_status(conn, goal_slug_or_id=args.status_goal))
        finally:
            conn.close()
    raise SystemExit(run_pool(args))


if __name__ == "__main__":
    main()
