#!/usr/bin/env python3
"""Launch and compactly monitor queue-backed rlab training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlparse

TERMINAL_STATUSES = {"succeeded", "failed", "finalization_failed", "canceled"}
ERROR_STATUSES = {"failed", "finalization_failed"}
SUMMARY_METRICS = (
    "global_step",
    "eval/acceptance/pass",
    "eval/acceptance/episodes/planned",
    "eval/acceptance/episodes/completed",
    "eval/acceptance/failure/count",
    "eval/acceptance/duration/seconds",
    "eval/full/outcome/success/rate/min",
    "eval/full/outcome/success/rate/mean",
    "eval/full/checkpoint/step",
    "eval/full/episode/return/mean",
    "train/outcome/success/window_100/rate/mean",
    "train/throughput/loop_fps",
)


def emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, **payload},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ),
        flush=True,
    )


def safe_error(exc: BaseException) -> str:
    from rlab.fleet_service import redact

    return str(redact(str(exc)))[:2000]


def command(
    argv: Sequence[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(argv), cwd=cwd, text=True, capture_output=True, check=False
    )
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{shlex.join(argv)} failed: {detail[-2000:]}")
    return result


def repository_root() -> Path:
    result = command(
        ["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd(), check=True
    )
    return Path(result.stdout.strip()).resolve()


def tracked_path(root: Path, value: str, *, label: str) -> Path:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside the repository: {path}") from exc
    if not path.is_file():
        raise ValueError(f"{label} does not exist: {relative}")
    result = command(
        ["git", "ls-files", "--error-unmatch", str(relative)], cwd=root, check=False
    )
    if result.returncode:
        raise ValueError(f"{label} must be checked in: {relative}")
    changed = command(
        ["git", "diff", "--quiet", "HEAD", "--", str(relative)],
        cwd=root,
        check=False,
    )
    if changed.returncode:
        raise ValueError(
            f"{label} has uncommitted changes; commit it before launching: {relative}"
        )
    return path


def source_sha(root: Path) -> str:
    return command(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()


def source_branch(root: Path) -> str:
    from rlab.runtime_refs import current_git_branch

    return current_git_branch(root)


def dirty(root: Path) -> bool:
    result = command(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
    )
    return bool(result.stdout.strip())


def link_if_present(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)


@contextmanager
def isolated_worktree(root: Path, *, branch: str) -> Iterator[Path]:
    revision = source_sha(root)
    path = Path(tempfile.gettempdir()) / (
        f"rlab-training-{revision[:12]}-{uuid.uuid4().hex[:8]}"
    )
    command(["git", "worktree", "add", "--detach", "--quiet", str(path), revision], cwd=root)
    try:
        link_if_present(root / ".env", path / ".env")
        link_if_present(
            root / "logs" / "fleet" / "modal-eval-assets.json",
            path / "logs" / "fleet" / "modal-eval-assets.json",
        )
        emit(
            "workspace_ready",
            source_sha=revision,
            source_branch=branch,
            caller_dirty=dirty(root),
            isolated=True,
        )
        yield path
    finally:
        command(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=root,
            check=False,
        )
        command(["git", "worktree", "prune"], cwd=root, check=False)
        emit("workspace_cleaned", source_sha=revision)


def build_launch_command(
    args: argparse.Namespace,
    root: Path,
    *,
    branch: str,
) -> list[str]:
    goal = tracked_path(root, args.goal, label="goal")
    recipe = tracked_path(root, args.recipe, label="recipe")
    argv = [
        "uv",
        "run",
        "--frozen",
        "rlab",
        "train",
        "--goal-file",
        str(goal.relative_to(root)),
        "--recipe-file",
        str(recipe.relative_to(root)),
        "--machine",
        args.machine,
        "--image-branch",
        branch,
        "--wait",
        "running",
        "--timeout",
        str(args.launch_timeout),
        "--json",
    ]
    if args.seed is not None:
        argv.extend(["--seed", str(args.seed)])
    if args.checkpoint_eval_backend:
        argv.extend(["--checkpoint-eval-backend", args.checkpoint_eval_backend])
    if args.runtime_image_ref_file:
        receipt = Path(args.runtime_image_ref_file).expanduser().resolve()
        if not receipt.is_file():
            raise ValueError(f"runtime image receipt does not exist: {receipt}")
        argv.extend(["--runtime-image-ref-file", str(receipt)])
    for value in args.set_values:
        if "=" not in value:
            raise ValueError(f"--set must be key=value, got {value!r}")
        argv.extend(["--set", value])
    return argv


def parse_json_output(output: str) -> dict[str, Any]:
    text = output.strip()
    if text:
        try:
            value = json.loads(text)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("rlab train did not return a JSON object")


def launch(argv: Sequence[str], *, cwd: Path) -> dict[str, Any]:
    from rlab.fleet_service import redact

    emit("launch_started", machine=argv[argv.index("--machine") + 1])
    with tempfile.TemporaryFile(mode="w+") as stdout, tempfile.TemporaryFile(
        mode="w+"
    ) as stderr:
        process = subprocess.Popen(list(argv), cwd=cwd, text=True, stdout=stdout, stderr=stderr)
        started = time.monotonic()
        next_heartbeat = started + 60
        while process.poll() is None:
            time.sleep(2)
            now = time.monotonic()
            if now >= next_heartbeat:
                emit("launch_waiting", elapsed_seconds=round(now - started, 1))
                next_heartbeat = now + 60
        stdout.seek(0)
        stderr.seek(0)
        output = stdout.read()
        error = stderr.read()
    if process.returncode:
        evidence = str(redact(error.strip() or output.strip()))[-2000:]
        emit(
            "potential_bug",
            job_id=None,
            fingerprint=hashlib.sha256(evidence.encode()).hexdigest()[:12],
            reasons=["training launch failed before reaching running state"],
            evidence=evidence,
        )
        raise RuntimeError("training launch failed; see potential_bug event")
    return parse_json_output(output)


def open_queue(root: Path):
    from rlab.dotenv import load_env_file
    from rlab.job_queue import connect, database_url

    load_env_file(root / ".env")
    conn = connect(database_url())
    conn.autocommit = True
    return conn


def queue_row(conn, job_id: int) -> dict[str, Any]:
    from rlab.job_queue import queue_status

    report = queue_status(conn, job_id=job_id)
    jobs = report.get("jobs") or []
    if len(jobs) != 1:
        raise RuntimeError(f"expected one queue row for job {job_id}, found {len(jobs)}")
    return dict(jobs[0])


def diagnostics(conn, job_id: int) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              COALESCE((SELECT MAX(checkpoint_step) FROM eval_jobs WHERE train_job_id=%(job)s), 0) AS max_checkpoint_step,
              (SELECT COUNT(*) FROM eval_jobs WHERE train_job_id=%(job)s AND status='failed') AS failed_eval_jobs,
              (SELECT COUNT(*) FROM eval_jobs WHERE train_job_id=%(job)s AND status IN ('pending','submitted','running')) AS active_eval_jobs,
              (SELECT COUNT(*) FROM eval_attempts a JOIN eval_jobs j ON j.id=a.eval_job_id WHERE j.train_job_id=%(job)s) AS eval_attempts,
              (SELECT COUNT(*) FROM eval_attempts a JOIN eval_jobs j ON j.id=a.eval_job_id WHERE j.train_job_id=%(job)s AND a.attempt_number>1) AS eval_retries,
              (SELECT COUNT(*) FROM eval_attempts a JOIN eval_jobs j ON j.id=a.eval_job_id WHERE j.train_job_id=%(job)s AND a.status='failed') AS failed_eval_attempts,
              (SELECT COUNT(*) FROM metric_batches b JOIN metric_streams s ON s.stream_id=b.stream_id JOIN worker_attempts w ON w.attempt_id=s.attempt_id WHERE w.train_job_id=%(job)s) AS pending_metric_batches,
              (SELECT COUNT(*) FROM attempt_events e JOIN worker_attempts w ON w.attempt_id=e.attempt_id WHERE w.train_job_id=%(job)s AND (e.last_error IS NOT NULL OR e.attempts>0)) AS errored_mailbox_events,
              (SELECT COUNT(*) FROM attempt_commands c JOIN worker_attempts w ON w.attempt_id=c.attempt_id WHERE w.train_job_id=%(job)s AND (c.delivered_at IS NULL OR c.acknowledged_at IS NULL)) AS unhandled_commands,
              (SELECT COUNT(*) FROM metric_streams s JOIN worker_attempts w ON w.attempt_id=s.attempt_id WHERE w.train_job_id=%(job)s AND NOT (s.accepted_sequence=s.submitted_sequence AND s.submitted_sequence=s.published_sequence AND s.published_sequence=s.final_sequence)) AS incomplete_streams,
              (SELECT COUNT(*) FROM worker_attempts WHERE train_job_id=%(job)s AND status IN ('launching','running')) AS active_worker_attempts,
              (SELECT outcome FROM eval_runs WHERE train_job_id=%(job)s) AS eval_outcome,
              (SELECT acceptance_committed_at FROM eval_runs WHERE train_job_id=%(job)s) AS acceptance_committed_at,
              (SELECT stop_delivery_slo_met FROM eval_runs WHERE train_job_id=%(job)s) AS stop_delivery_slo_met,
              (SELECT promoted_eval_job_id FROM eval_runs WHERE train_job_id=%(job)s) AS promoted_eval_job_id
            """,
            {"job": job_id},
        )
        return dict(cur.fetchone())


def seconds_between(start: Any, end: Any) -> float | None:
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    return round((end - start).total_seconds(), 3)


def snapshot(row: Mapping[str, Any], diag: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": row.get("status"),
        "launch_state": row.get("launch_state"),
        "eval_status": row.get("eval_status"),
        "eval_outcome": diag.get("eval_outcome"),
        "publication_status": row.get("live_publication_status"),
        "artifact_status": row.get("artifact_status"),
        "max_checkpoint_step": int(diag.get("max_checkpoint_step") or 0),
        "active_eval_jobs": int(diag.get("active_eval_jobs") or 0),
        "eval_retries": int(diag.get("eval_retries") or 0),
        "failed_eval_jobs": int(diag.get("failed_eval_jobs") or 0),
        "pending_metric_batches": int(diag.get("pending_metric_batches") or 0),
    }


def potential_bug_incidents(
    row: Mapping[str, Any],
    diag: Mapping[str, Any],
    *,
    now: float,
    progress_changed_at: float,
    backlog_started_at: float | None,
    stale_seconds: float,
) -> dict[str, str]:
    incidents: dict[str, str] = {}
    status = str(row.get("status") or "")
    if status in ERROR_STATUSES:
        incidents["terminal_job_error"] = f"terminal job status is {status}"
    if status == "canceled" and not row.get("cancel_requested"):
        incidents["unexpected_cancellation"] = (
            "job became canceled without a recorded cancel request"
        )
    for key in (
        "error",
        "launch_error",
        "eval_error",
        "live_publication_error",
        "promoted_projection_error",
    ):
        if row.get(key):
            incidents[f"queue_error:{key}"] = f"{key} is set"
    eval_counters = {
        "failed eval jobs": int(diag.get("failed_eval_jobs") or 0),
        "eval retries": int(diag.get("eval_retries") or 0),
        "failed eval attempts": int(diag.get("failed_eval_attempts") or 0),
    }
    if any(eval_counters.values()):
        incidents["eval_execution_failure"] = ", ".join(
            f"{label}={value}" for label, value in eval_counters.items() if value
        )
    if int(diag.get("errored_mailbox_events") or 0) > 0:
        incidents["mailbox_delivery_error"] = (
            f"errored mailbox events={int(diag['errored_mailbox_events'])}"
        )
    if int(row.get("live_publication_attempts") or 0) > 0:
        incidents["wandb_publication_retry"] = (
            f"W&B publication retries={int(row.get('live_publication_attempts') or 0)}"
        )
    if int(row.get("artifact_projection_attempts") or 0) > 0:
        incidents["artifact_projection_retry"] = (
            f"artifact projection retries={int(row.get('artifact_projection_attempts') or 0)}"
        )
    if status == "running" and row.get("ready_at") and now - progress_changed_at >= stale_seconds:
        incidents["checkpoint_progress_stalled"] = (
            f"checkpoint progress has not advanced for {int(stale_seconds)} seconds"
        )
    ready_at = row.get("ready_at")
    if (
        status == "running"
        and isinstance(ready_at, datetime)
        and not row.get("wandb_url")
        and (datetime.now(ready_at.tzinfo) - ready_at).total_seconds() >= 180
    ):
        incidents["wandb_url_missing"] = (
            "W&B URL is still missing 180 seconds after readiness"
        )
    if backlog_started_at is not None and now - backlog_started_at >= 120:
        incidents["eval_backlog"] = (
            "more than three eval jobs have remained active for 120 seconds"
        )
    return incidents


def wandb_audit(wandb_url: str) -> dict[str, Any]:
    import wandb

    parsed = urlparse(wandb_url)
    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 4 or parts[-2] != "runs":
        raise ValueError(f"unrecognized W&B run URL: {wandb_url}")
    entity, project, run_id = parts[0], parts[1], parts[-1]
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            run = wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}")
            summary = run.summary
            artifacts = []
            chosen = None
            for artifact in run.logged_artifacts():
                aliases = list(artifact.aliases)
                record = {
                    "name": artifact.name,
                    "type": artifact.type,
                    "aliases": aliases,
                    "state": artifact.state,
                    "metadata": dict(artifact.metadata or {}),
                }
                artifacts.append(record)
                if artifact.type == "model" and "promoted" in aliases:
                    chosen = record
                elif chosen is None and artifact.type == "model" and "latest" in aliases:
                    chosen = record
            artifact_ref = None
            if chosen:
                collection = str(chosen["name"]).split(":", 1)[0]
                alias = "promoted" if "promoted" in chosen["aliases"] else "latest"
                artifact_ref = f"{entity}/{project}/{collection}:{alias}"

            def summary_value(key: str) -> Any:
                value = summary.get(key)
                if not isinstance(value, dict) and hasattr(value, "keys"):
                    value = dict(value)
                if isinstance(value, dict) and len(value) == 1:
                    return next(iter(value.values()))
                return value

            return {
                "state": run.state,
                "url": run.url,
                "metrics": {key: summary_value(key) for key in SUMMARY_METRICS},
                "best_artifact": artifact_ref,
                "artifacts": [
                    {key: item[key] for key in ("name", "type", "aliases", "state")}
                    for item in artifacts
                ],
            }
        except Exception as exc:  # network/API failures are reported after bounded retry
            last_error = exc
            if attempt < 2:
                time.sleep(5)
    assert last_error is not None
    raise last_error


def evaluation_evidence(conn, job_id: int) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.outcome, r.acceptance_committed_at, r.stop_delivery_slo_met,
                   r.promoted_eval_job_id, j.checkpoint_step, j.checkpoint_uri,
                   j.checkpoint_sha256, a.attempt_id, a.attempt_number, a.result_uri,
                   a.result_json, j.contract_json
            FROM eval_runs r
            LEFT JOIN LATERAL (
              SELECT candidate.*
              FROM eval_jobs candidate
              WHERE candidate.train_job_id=r.train_job_id
                AND (
                  candidate.id=r.promoted_eval_job_id
                  OR (
                    r.promoted_eval_job_id IS NULL
                    AND candidate.status='succeeded'
                    AND candidate.purpose='acceptance'
                    AND candidate.source_announcement_json->>'kind'='final'
                  )
                )
              ORDER BY (candidate.id=r.promoted_eval_job_id) DESC, candidate.id DESC
              LIMIT 1
            ) j ON TRUE
            LEFT JOIN eval_attempts a ON a.id=j.accepted_attempt_id
            WHERE r.train_job_id=%(job)s
            """,
            {"job": job_id},
        )
        row = cur.fetchone()
    if not row:
        return None
    value = dict(row)
    result = value.pop("result_json", None) or {}
    contract = value.pop("contract_json", None) or {}
    validation_error = None
    if result and contract and value.get("attempt_id"):
        from rlab.modal_eval_protocol import validate_attempt_result

        try:
            result = validate_attempt_result(
                result,
                contract=contract,
                attempt_id=str(value["attempt_id"]),
            )
        except Exception as exc:
            validation_error = str(exc)
    else:
        validation_error = "terminal evaluation evidence is incomplete"
    episodes = result.get("episode_results") or []
    planned = (contract.get("manifest") or {}).get("episodes") or []
    actual_ids = [str(item.get("episode_id")) for item in episodes]
    planned_ids = [str(item.get("episode_id")) for item in planned]
    value.update(
        {
            "verdict": result.get("verdict"),
            "evidence_valid": validation_error is None,
            "validation_error": validation_error,
            "episodes_completed": len(episodes),
            "unique_episode_ids": len(set(actual_ids)),
            "episodes_planned": len(planned),
            "exact_manifest": bool(planned_ids)
            and len(actual_ids) == len(planned_ids)
            and set(actual_ids) == set(planned_ids),
            "successes": sum(item.get("outcome") == "success" for item in episodes),
            "failures": sum(item.get("outcome") != "success" for item in episodes),
            "claimed_aggregates": result.get("claimed_aggregates"),
        }
    )
    return value


def classify_terminal(
    row: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any] | None,
    metrics: Mapping[str, Any],
    durable_objects: bool,
    best_artifact: Any,
    operational_clean: bool,
    wandb_state: Any,
    audit_errors: Sequence[str],
) -> str:
    base_valid = bool(
        row.get("status") == "succeeded"
        and not row.get("error")
        and row.get("live_publication_status") == "complete"
        and wandb_state == "finished"
        and evidence
        and evidence.get("evidence_valid")
        and durable_objects
        and best_artifact
        and operational_clean
        and not audit_errors
    )
    accepted = bool(
        base_valid
        and evidence
        and evidence.get("outcome") == "accepted"
        and evidence.get("verdict") == "accepted"
        and evidence.get("exact_manifest")
        and int(evidence.get("episodes_planned") or 0) > 0
        and int(evidence.get("episodes_completed") or 0)
        == int(evidence.get("episodes_planned") or 0)
        and int(evidence.get("successes") or 0)
        == int(evidence.get("episodes_planned") or 0)
        and int(evidence.get("failures") or 0) == 0
        and metrics.get("eval/acceptance/pass") == 1
        and metrics.get("eval/full/outcome/success/rate/min") == 1
        and metrics.get("eval/acceptance/episodes/completed")
        == metrics.get("eval/acceptance/episodes/planned")
    )
    if accepted:
        return "accepted"
    rejected = bool(
        base_valid
        and evidence
        and evidence.get("outcome") == "not_accepted"
        and evidence.get("verdict") == "rejected"
        and int(evidence.get("episodes_planned") or 0) > 0
        and 0 < int(evidence.get("episodes_completed") or 0)
        < int(evidence.get("episodes_planned") or 0)
        and int(evidence.get("failures") or 0) >= 1
        and metrics.get("eval/acceptance/pass") == 0
        and float(metrics.get("eval/acceptance/failure/count") or 0) >= 1
        and float(metrics.get("eval/acceptance/episodes/completed") or 0)
        < float(metrics.get("eval/acceptance/episodes/planned") or 0)
    )
    return "goal_rejected" if rejected else "operational_failure"


def object_audit(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    if not evidence or not evidence.get("checkpoint_uri"):
        return {}
    from rlab.modal_eval_storage import ObjectStore, object_store_base_uri

    store = ObjectStore(object_store_base_uri())
    result = {"checkpoint": store.head(str(evidence["checkpoint_uri"]))}
    if evidence.get("result_uri"):
        result["result"] = store.head(str(evidence["result_uri"]))
    return result


def audit_terminal(conn, job_id: int) -> dict[str, Any]:
    row = queue_row(conn, job_id)
    diag = diagnostics(conn, job_id)
    evidence = evaluation_evidence(conn, job_id)
    wandb_result: dict[str, Any] = {}
    audit_errors: list[str] = []
    if row.get("wandb_url"):
        try:
            wandb_result = wandb_audit(str(row["wandb_url"]))
        except Exception as exc:
            audit_errors.append(f"W&B verification failed: {exc}")
    else:
        audit_errors.append("terminal job has no W&B URL")
    objects: dict[str, Any] = {}
    try:
        objects = object_audit(evidence)
    except Exception as exc:
        audit_errors.append(f"object-store verification failed: {exc}")
    operational_clean = all(
        int(diag.get(key) or 0) == 0
        for key in (
            "failed_eval_jobs",
            "eval_retries",
            "failed_eval_attempts",
            "pending_metric_batches",
            "errored_mailbox_events",
            "unhandled_commands",
            "incomplete_streams",
            "active_worker_attempts",
        )
    ) and int(row.get("active_reservations") or 0) == 0
    best_artifact = wandb_result.get("best_artifact") or row.get("artifact_ref")
    play_target = best_artifact or row.get("wandb_url")
    play_command = (
        shlex.join(["uv", "run", "rlab", "play", str(play_target), "--episodes", "1"])
        if play_target
        else None
    )
    metrics = wandb_result.get("metrics") or {}
    durable_objects = bool(
        objects.get("checkpoint", {}).get("size")
        and objects.get("result", {}).get("size")
    )
    terminal_classification = classify_terminal(
        row,
        evidence=evidence,
        metrics=metrics,
        durable_objects=durable_objects,
        best_artifact=best_artifact,
        operational_clean=operational_clean,
        wandb_state=wandb_result.get("state"),
        audit_errors=audit_errors,
    )
    verified_success = terminal_classification == "accepted"
    return {
        "job_id": job_id,
        "verified_success": verified_success,
        "terminal_classification": terminal_classification,
        "operationally_valid": terminal_classification != "operational_failure",
        "status": row.get("status"),
        "batch_id": row.get("batch_id"),
        "run_name": row.get("run_name"),
        "machine": row.get("machine"),
        "source_sha": row.get("repo_git_commit"),
        "runtime_image_ref": row.get("runtime_image_ref"),
        "timing": {
            "created_at": row.get("created_at"),
            "started_at": row.get("started_at"),
            "ready_at": row.get("ready_at"),
            "finished_at": row.get("finished_at"),
            "submission_to_finish_seconds": seconds_between(
                row.get("created_at"), row.get("finished_at")
            ),
            "start_to_finish_seconds": seconds_between(
                row.get("started_at"), row.get("finished_at")
            ),
        },
        "evaluation": evidence,
        "operations": {
            **diag,
            "active_reservations": int(row.get("active_reservations") or 0),
            "publication_status": row.get("live_publication_status"),
            "artifact_status": row.get("artifact_status"),
            "learner_stop_observed_at": row.get("learner_stop_observed_at"),
            "process_exited_at": row.get("process_exited_at"),
        },
        "objects": objects,
        "wandb": wandb_result,
        "wandb_url": row.get("wandb_url"),
        "best_artifact": best_artifact,
        "play_command": play_command,
        "audit_errors": audit_errors,
    }


def monitor_jobs(
    root: Path,
    job_ids: Sequence[int],
    *,
    poll_seconds: float,
    stale_seconds: float,
) -> tuple[list[dict[str, Any]], bool]:
    conn = open_queue(root)
    state: dict[int, dict[str, Any]] = {
        job_id: {
            "status": None,
            "wandb_url": None,
            "max_step": 0,
            "last_progress_report": 0,
            "progress_changed_at": time.monotonic(),
            "backlog_started_at": None,
            "last_heartbeat": 0.0,
            "bug_fingerprints": set(),
            "terminal": False,
        }
        for job_id in job_ids
    }
    summaries: list[dict[str, Any]] = []
    try:
        while not all(item["terminal"] for item in state.values()):
            loop_started = time.monotonic()
            for job_id, item in state.items():
                if item["terminal"]:
                    continue
                row = queue_row(conn, job_id)
                diag = diagnostics(conn, job_id)
                status = str(row.get("status") or "unknown")
                max_step = int(diag.get("max_checkpoint_step") or 0)
                if max_step > int(item["max_step"]):
                    item["max_step"] = max_step
                    item["progress_changed_at"] = loop_started
                if int(diag.get("active_eval_jobs") or 0) > 3:
                    item["backlog_started_at"] = item["backlog_started_at"] or loop_started
                else:
                    item["backlog_started_at"] = None
                current_snapshot = snapshot(row, diag)
                if row.get("wandb_url") and row.get("wandb_url") != item["wandb_url"]:
                    item["wandb_url"] = row.get("wandb_url")
                    emit(
                        "wandb_url",
                        job_id=job_id,
                        url=row.get("wandb_url"),
                        run_id=row.get("wandb_run_id"),
                    )
                should_report_progress = (
                    status != item["status"]
                    or max_step - int(item["last_progress_report"]) >= 500_000
                    or loop_started - float(item["last_heartbeat"]) >= 120
                )
                if should_report_progress:
                    emit("progress", job_id=job_id, **current_snapshot)
                    item["status"] = status
                    item["last_progress_report"] = max_step
                    item["last_heartbeat"] = loop_started
                incidents = potential_bug_incidents(
                    row,
                    diag,
                    now=loop_started,
                    progress_changed_at=float(item["progress_changed_at"]),
                    backlog_started_at=item["backlog_started_at"],
                    stale_seconds=stale_seconds,
                )
                for category, reason in incidents.items():
                    fingerprint = hashlib.sha256(
                        f"{job_id}:{category}".encode()
                    ).hexdigest()[:12]
                    if fingerprint not in item["bug_fingerprints"]:
                        item["bug_fingerprints"].add(fingerprint)
                        emit(
                            "potential_bug",
                            job_id=job_id,
                            fingerprint=fingerprint,
                            reasons=[reason],
                            category=category,
                            snapshot=current_snapshot,
                        )
                if status in TERMINAL_STATUSES:
                    try:
                        summary = audit_terminal(conn, job_id)
                    except Exception as exc:
                        fingerprint = hashlib.sha256(str(exc).encode()).hexdigest()[:12]
                        emit(
                            "potential_bug",
                            job_id=job_id,
                            fingerprint=fingerprint,
                            reasons=["terminal audit failed"],
                            evidence=str(exc),
                        )
                        summary = {
                            "job_id": job_id,
                            "verified_success": False,
                            "status": status,
                            "audit_errors": [str(exc)],
                        }
                    if summary.get("terminal_classification") == "operational_failure":
                        reason_text = summary.get("audit_errors") or [
                            "terminal run has inconsistent or incomplete operational evidence"
                        ]
                        fingerprint = hashlib.sha256(
                            f"{job_id}:terminal_operational_failure".encode()
                        ).hexdigest()[:12]
                        if fingerprint not in item["bug_fingerprints"]:
                            item["bug_fingerprints"].add(fingerprint)
                            emit(
                                "potential_bug",
                                job_id=job_id,
                                fingerprint=fingerprint,
                                reasons=reason_text,
                                category="terminal_operational_failure",
                                snapshot=current_snapshot,
                            )
                    emit("terminal", **summary)
                    summaries.append(summary)
                    item["terminal"] = True
            if not all(item["terminal"] for item in state.values()):
                elapsed = time.monotonic() - loop_started
                time.sleep(max(0.5, poll_seconds - elapsed))
    finally:
        conn.close()
    return summaries, bool(summaries) and all(
        summary.get("verified_success") for summary in summaries
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Launch and compactly monitor queue-backed rlab training."
    )
    value.add_argument("--goal", help="Checked-in goal YAML path.")
    value.add_argument("--recipe", help="Checked-in recipe YAML path.")
    value.add_argument("--machine", default="beast-3")
    value.add_argument("--seed", type=int)
    value.add_argument("--set", dest="set_values", action="append", default=[])
    value.add_argument(
        "--checkpoint-eval-backend", choices=("local", "modal", "none")
    )
    value.add_argument("--runtime-image-ref-file")
    value.add_argument("--launch-timeout", type=float, default=1200)
    value.add_argument("--poll-seconds", type=float, default=5)
    value.add_argument("--stale-seconds", type=float, default=300)
    value.add_argument("--monitor-job", type=int)
    value.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and print the launch shape without mutation.",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = repository_root()
    if args.monitor_job is not None:
        if args.goal or args.recipe or args.validate_only:
            raise SystemExit("--monitor-job cannot be combined with launch inputs")
        try:
            summaries, success = monitor_jobs(
                root,
                [args.monitor_job],
                poll_seconds=max(1, args.poll_seconds),
                stale_seconds=max(60, args.stale_seconds),
            )
        except Exception as exc:
            evidence = safe_error(exc)
            emit(
                "potential_bug",
                job_id=args.monitor_job,
                fingerprint=hashlib.sha256(evidence.encode()).hexdigest()[:12],
                reasons=["authoritative run monitoring failed"],
                evidence=evidence,
            )
            emit("workflow_error", error=evidence)
            return 1
        emit(
            "complete",
            job_ids=[args.monitor_job],
            verified_success=success,
            terminal_jobs=len(summaries),
        )
        return 0 if success else 1
    if not args.goal or not args.recipe:
        raise SystemExit("--goal and --recipe are required when launching")
    goal = tracked_path(root, args.goal, label="goal")
    recipe = tracked_path(root, args.recipe, label="recipe")
    branch = source_branch(root)
    if args.validate_only:
        emit(
            "validated",
            goal=str(goal.relative_to(root)),
            recipe=str(recipe.relative_to(root)),
            machine=args.machine,
            seed=args.seed,
            source_sha=source_sha(root),
            source_branch=branch,
            isolated_worktree=True,
        )
        return 0
    summaries: list[dict[str, Any]] = []
    success = False
    try:
        with isolated_worktree(root, branch=branch) as worktree:
            launch_argv = build_launch_command(args, root, branch=branch)
            report = launch(launch_argv, cwd=worktree)
            job_ids = [int(value) for value in report.get("job_ids") or []]
            if not job_ids:
                raise RuntimeError("rlab train returned no job IDs")
            emit(
                "submitted",
                job_ids=job_ids,
                batch_id=report.get("batch_id"),
                machine=report.get("machine") or args.machine,
                source_sha=report.get("source_sha") or source_sha(root),
                runtime_image_ref=report.get("runtime_image_ref"),
            )
            summaries, success = monitor_jobs(
                root,
                job_ids,
                poll_seconds=max(1, args.poll_seconds),
                stale_seconds=max(60, args.stale_seconds),
            )
    except Exception as exc:
        emit("workflow_error", error=safe_error(exc))
        return 1
    emit(
        "complete",
        job_ids=[summary["job_id"] for summary in summaries],
        verified_success=success,
        terminal_jobs=len(summaries),
    )
    return 0 if success else 1


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        emit("workflow_error", error=safe_error(exc))
        exit_code = 1
    raise SystemExit(exit_code)
