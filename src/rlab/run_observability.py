from __future__ import annotations

import hashlib
import json
import shlex
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse

from rlab.json_utils import json_safe


SCHEMA_VERSION = 1
TERMINAL_STATUSES = frozenset({"succeeded", "failed", "finalization_failed", "canceled"})
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


def _iso_utc(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).isoformat().replace("+00:00", "Z")


def _seconds_between(start: Any, end: Any) -> float | None:
    if isinstance(start, str):
        try:
            start = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except ValueError:
            return None
    if isinstance(end, str):
        try:
            end = datetime.fromisoformat(end.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return None
    return round((end - start).total_seconds(), 3)


def _acceptance_required(row: Mapping[str, Any]) -> bool:
    config = row.get("train_config")
    if not isinstance(config, Mapping):
        return row.get("eval_status") is not None
    return str(config.get("checkpoint_eval_backend") or "local") != "none"


def _wandb_artifact_ref(row: Mapping[str, Any]) -> str | None:
    url = str(row.get("wandb_url") or "")
    run_id = str(row.get("wandb_run_id") or "")
    step = row.get("promoted_step")
    if not url or not run_id or step is None:
        return None
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) < 2:
        return None
    return f"{parts[0]}/{parts[1]}/{run_id}-checkpoint:step-{int(step)}"


def _fingerprint(run_id: int, category: str, subject: str) -> str:
    value = f"{int(run_id)}:{category}:{subject}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _incident(
    run_id: int,
    category: str,
    subject: str,
    detail: str,
) -> dict[str, Any]:
    from rlab.fleet_service import redact

    return {
        "fingerprint": _fingerprint(run_id, category, subject),
        "category": category,
        "subject": subject,
        "detail": str(redact(detail)),
    }


def _diagnostics(conn, run_id: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              COALESCE((SELECT MAX(checkpoint_step) FROM eval_jobs WHERE train_job_id=%(run)s), 0) AS max_checkpoint_step,
              (SELECT COUNT(*) FROM eval_jobs WHERE train_job_id=%(run)s AND status='failed') AS failed_eval_jobs,
              (SELECT COUNT(*) FROM eval_jobs WHERE train_job_id=%(run)s AND status IN ('pending','dispatching','submitted','blocked_budget')) AS active_eval_jobs,
              (SELECT COUNT(*) FROM eval_attempts a JOIN eval_jobs j ON j.id=a.eval_job_id WHERE j.train_job_id=%(run)s) AS eval_attempts,
              (SELECT COUNT(*) FROM eval_attempts a JOIN eval_jobs j ON j.id=a.eval_job_id WHERE j.train_job_id=%(run)s AND (a.attempt_number>1 OR a.retry_round>0)) AS eval_retries,
              (SELECT COUNT(*) FROM eval_attempts a JOIN eval_jobs j ON j.id=a.eval_job_id WHERE j.train_job_id=%(run)s AND a.status IN ('failed','expired')) AS failed_eval_attempts,
              (SELECT COUNT(*) FROM metric_batches b JOIN metric_streams s ON s.stream_id=b.stream_id JOIN worker_attempts w ON w.attempt_id=s.attempt_id WHERE w.train_job_id=%(run)s) AS pending_metric_batches,
              (SELECT COUNT(*) FROM attempt_events e JOIN worker_attempts w ON w.attempt_id=e.attempt_id WHERE w.train_job_id=%(run)s AND (e.last_error IS NOT NULL OR e.attempts>0)) AS errored_mailbox_events,
              (SELECT COUNT(*) FROM attempt_commands c JOIN worker_attempts w ON w.attempt_id=c.attempt_id WHERE w.train_job_id=%(run)s AND (c.delivered_at IS NULL OR c.acknowledged_at IS NULL)) AS unhandled_commands,
              (SELECT COUNT(*) FROM metric_streams s JOIN worker_attempts w ON w.attempt_id=s.attempt_id WHERE w.train_job_id=%(run)s AND NOT (s.accepted_sequence=s.submitted_sequence AND s.submitted_sequence=s.published_sequence AND s.final_sequence IS NOT NULL AND s.published_sequence=s.final_sequence)) AS incomplete_streams,
              (SELECT COUNT(*) FROM worker_attempts WHERE train_job_id=%(run)s AND status IN ('launching','running')) AS active_worker_attempts,
              (SELECT outcome FROM eval_runs WHERE train_job_id=%(run)s) AS eval_outcome,
              (SELECT acceptance_committed_at FROM eval_runs WHERE train_job_id=%(run)s) AS acceptance_committed_at,
              (SELECT stop_delivery_slo_met FROM eval_runs WHERE train_job_id=%(run)s) AS stop_delivery_slo_met,
              (SELECT promoted_eval_job_id FROM eval_runs WHERE train_job_id=%(run)s) AS promoted_eval_job_id
            """,
            {"run": int(run_id)},
        )
        diagnostics = dict(cur.fetchone() or {})
        cur.execute(
            """
            SELECT event_type, message, metadata_json, created_at
            FROM job_events
            WHERE job_kind='train' AND job_id=%(run)s
            ORDER BY id DESC
            LIMIT 50
            """,
            {"run": int(run_id)},
        )
        history = [dict(row) for row in cur.fetchall()]
    return diagnostics, history


def current_incidents(
    row: Mapping[str, Any], diagnostics: Mapping[str, Any]
) -> list[dict[str, Any]]:
    run_id = int(row["id"])
    incidents: list[dict[str, Any]] = []
    status = str(row.get("status") or "")
    eval_status = str(row.get("eval_status") or "")
    if status in {"failed", "finalization_failed"}:
        incidents.append(
            _incident(
                run_id,
                "terminal_operational_failure",
                status,
                str(row.get("error") or f"run terminalized as {status}"),
            )
        )
    if status == "canceled" and not row.get("cancel_requested"):
        incidents.append(
            _incident(
                run_id,
                "unexpected_cancellation",
                "cancel_requested=false",
                "run was canceled without a recorded cancellation request",
            )
        )
    for field in (
        "launch_error",
        "eval_error",
        "live_publication_error",
        "promoted_projection_error",
    ):
        if row.get(field):
            incidents.append(
                _incident(run_id, "state_error", field, f"{field} is set: {row[field]}")
            )
    failures = {
        "failed_eval_jobs": int(diagnostics.get("failed_eval_jobs") or 0),
        "eval_retries": int(diagnostics.get("eval_retries") or 0),
        "failed_eval_attempts": int(diagnostics.get("failed_eval_attempts") or 0),
    }
    if any(failures.values()):
        detail = ", ".join(f"{key}={value}" for key, value in failures.items() if value)
        incidents.append(_incident(run_id, "eval_execution_failure", "acceptance", detail))
    if int(diagnostics.get("errored_mailbox_events") or 0):
        incidents.append(
            _incident(
                run_id,
                "mailbox_delivery_error",
                "worker_mailbox",
                f"errored_mailbox_events={int(diagnostics['errored_mailbox_events'])}",
            )
        )
    if int(row.get("live_publication_attempts") or 0):
        incidents.append(
            _incident(
                run_id,
                "wandb_publication_retry",
                "wandb",
                f"attempts={int(row['live_publication_attempts'])}",
            )
        )
    if int(row.get("artifact_projection_attempts") or 0):
        incidents.append(
            _incident(
                run_id,
                "artifact_projection_retry",
                "promoted_checkpoint",
                f"attempts={int(row['artifact_projection_attempts'])}",
            )
        )
    if status in TERMINAL_STATUSES and eval_status in {"active", "finalizing"}:
        incidents.append(
            _incident(
                run_id,
                "inconsistent_terminal_state",
                eval_status,
                f"training is {status} while evaluation remains {eval_status}",
            )
        )
    ready_at = row.get("ready_at")
    if (
        status == "running"
        and isinstance(ready_at, datetime)
        and not row.get("wandb_url")
        and (datetime.now(ready_at.tzinfo or UTC) - ready_at).total_seconds() >= 180
    ):
        incidents.append(
            _incident(
                run_id,
                "wandb_url_missing",
                "wandb",
                "W&B URL is missing more than 180 seconds after learner readiness",
            )
        )
    if status in TERMINAL_STATUSES:
        terminal_counters = {
            "pending_metric_batches": int(diagnostics.get("pending_metric_batches") or 0),
            "unhandled_commands": int(diagnostics.get("unhandled_commands") or 0),
            "incomplete_streams": int(diagnostics.get("incomplete_streams") or 0),
            "active_worker_attempts": int(diagnostics.get("active_worker_attempts") or 0),
            "active_reservations": int(row.get("active_reservations") or 0),
        }
        if any(terminal_counters.values()):
            detail = ", ".join(
                f"{key}={value}" for key, value in terminal_counters.items() if value
            )
            incidents.append(_incident(run_id, "terminal_resources_open", "run", detail))
    unique = {item["fingerprint"]: item for item in incidents}
    return [unique[key] for key in sorted(unique)]


def terminal_classification(projection: Mapping[str, Any]) -> str | None:
    status = str(projection.get("status") or "")
    if status not in TERMINAL_STATUSES:
        return None
    if status == "canceled":
        return "canceled"
    if status != "succeeded":
        return "operational_failure"
    outcome = str((projection.get("evaluation") or {}).get("outcome") or "")
    if outcome == "accepted":
        return "accepted"
    if outcome == "not_accepted":
        return "goal_rejected"
    if not bool((projection.get("evaluation") or {}).get("acceptance_required")):
        return "completed"
    return "operational_failure"


def run_projection(conn, run_id: int) -> dict[str, Any]:
    from rlab.job_queue import queue_status

    report = queue_status(conn, job_id=int(run_id))
    rows = list(report.get("runs") or report.get("jobs") or [])
    if len(rows) != 1:
        raise ValueError(f"run {int(run_id)} does not exist")
    row = dict(rows[0])
    diagnostics, history = _diagnostics(conn, int(run_id))
    wandb_artifact = _wandb_artifact_ref(row)
    incidents = current_incidents(row, diagnostics)
    projection: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "observed_at": _iso_utc(),
        "run_id": int(row["id"]),
        "batch_id": row.get("batch_id"),
        "run_name": row.get("run_name"),
        "goal": row.get("goal_slug"),
        "recipe": row.get("recipe_slug"),
        "machine": row.get("machine"),
        "source_sha": row.get("repo_git_commit"),
        "runtime_image_ref": row.get("runtime_image_ref"),
        "status": row.get("status"),
        "error": row.get("error"),
        "launch": {
            "state": row.get("launch_state"),
            "launch_id": row.get("launch_id"),
            "container_name": row.get("container_name"),
            "error": row.get("launch_error"),
        },
        "evaluation": {
            "status": row.get("eval_status"),
            "outcome": diagnostics.get("eval_outcome"),
            "acceptance_required": _acceptance_required(row),
            "promoted_eval_job_id": diagnostics.get("promoted_eval_job_id"),
            "promoted_step": row.get("promoted_step"),
            "max_checkpoint_step": int(diagnostics.get("max_checkpoint_step") or 0),
            "acceptance_committed_at": diagnostics.get("acceptance_committed_at"),
            "stop_delivery_slo_met": diagnostics.get("stop_delivery_slo_met"),
            "learner_stop_observed_at": row.get("learner_stop_observed_at"),
            "process_exited_at": row.get("process_exited_at"),
        },
        "publication": {
            "status": row.get("live_publication_status"),
            "attempts": int(row.get("live_publication_attempts") or 0),
            "error": row.get("live_publication_error"),
            "artifact_status": row.get("artifact_status"),
            "artifact_projection_attempts": int(row.get("artifact_projection_attempts") or 0),
        },
        "wandb": {
            "run_id": row.get("wandb_run_id"),
            "url": row.get("wandb_url"),
            "remote_verified": row.get("live_publication_status") == "complete",
        },
        "artifacts": {
            "r2_checkpoint_uri": row.get("promoted_checkpoint_uri"),
            "wandb_artifact": wandb_artifact,
        },
        "operations": {
            key: int(diagnostics.get(key) or 0)
            for key in (
                "active_eval_jobs",
                "eval_attempts",
                "eval_retries",
                "failed_eval_jobs",
                "failed_eval_attempts",
                "pending_metric_batches",
                "errored_mailbox_events",
                "unhandled_commands",
                "incomplete_streams",
                "active_worker_attempts",
            )
        }
        | {"active_reservations": int(row.get("active_reservations") or 0)},
        "timestamps": {
            key: row.get(key)
            for key in (
                "created_at",
                "started_at",
                "learner_ready_at",
                "wandb_ready_at",
                "ready_at",
                "finished_at",
            )
        },
        "incidents": {
            "current": incidents,
            "history": [
                {
                    "event": item.get("event_type"),
                    "message": item.get("message"),
                    "metadata": item.get("metadata_json") or {},
                    "created_at": item.get("created_at"),
                }
                for item in history
            ],
        },
    }
    projection["terminal_classification"] = terminal_classification(projection)
    return json_safe(projection)


def projections_for_selector(
    conn,
    *,
    run_id: int | None = None,
    batch_id: str | None = None,
    machine: str | None = None,
    goal: str | None = None,
) -> dict[str, Any]:
    from rlab.job_queue import queue_status

    selectors = sum(value is not None for value in (run_id, batch_id, machine, goal))
    if selectors != 1:
        raise ValueError("exactly one --run, --batch, --machine, or --goal is required")
    if run_id is not None:
        ids = [int(run_id)]
        selector = {"run_id": int(run_id)}
    else:
        report = queue_status(
            conn,
            batch_id=batch_id,
            machine=machine,
            goal_slug=goal,
        )
        ids = [int(row["id"]) for row in report.get("runs") or report.get("jobs") or []]
        selector = {
            key: value
            for key, value in (("batch_id", batch_id), ("machine", machine), ("goal", goal))
            if value is not None
        }
    runs = [run_projection(conn, value) for value in ids]
    counts: dict[str, int] = {}
    for item in runs:
        status = str(item.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "observed_at": _iso_utc(),
        "selector": selector,
        "counts": counts,
        "runs": runs,
    }


def terminal_wandb_summary(url: str) -> dict[str, Any]:
    """Perform one bounded terminal API read for presentation only."""

    import wandb

    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) < 4 or parts[-2] != "runs":
        raise ValueError(f"unrecognized W&B run URL: {url}")
    run = wandb.Api(timeout=30).run(f"{parts[0]}/{parts[1]}/{parts[-1]}")
    summary = run.summary

    def value(key: str) -> Any:
        item = summary.get(key)
        if not isinstance(item, dict) and hasattr(item, "keys"):
            item = dict(item)
        if isinstance(item, dict) and len(item) == 1:
            return next(iter(item.values()))
        return item

    return {
        "state": run.state,
        "url": run.url,
        "metrics": {key: value(key) for key in SUMMARY_METRICS},
    }


def terminal_payload(
    projection: Mapping[str, Any],
    *,
    wandb_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    classification = str(projection.get("terminal_classification") or "")
    artifacts = dict(projection.get("artifacts") or {})
    artifact = artifacts.get("wandb_artifact")
    play_command = (
        shlex.join(["uv", "run", "rlab", "play", str(artifact), "--episodes", "1"])
        if artifact
        else None
    )
    timestamps = dict(projection.get("timestamps") or {})
    evaluation = dict(projection.get("evaluation") or {})
    return {
        **dict(projection),
        "verified_success": classification in {"accepted", "completed"},
        "early_stop_detected": bool(evaluation.get("learner_stop_observed_at")),
        "submission_to_finish_seconds": _seconds_between(
            timestamps.get("created_at"), timestamps.get("finished_at")
        ),
        "start_to_finish_seconds": _seconds_between(
            timestamps.get("started_at"), timestamps.get("finished_at")
        ),
        "wandb_terminal": dict(wandb_summary or {}),
        "play_command": play_command,
    }


def _semantic_progress(projection: Mapping[str, Any]) -> str:
    value = {
        "status": projection.get("status"),
        "launch": projection.get("launch"),
        "evaluation": projection.get("evaluation"),
        "publication": projection.get("publication"),
        "operations": projection.get("operations"),
    }
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def follow_run(
    conn,
    run_id: int,
    *,
    emit: Callable[[Mapping[str, Any]], None],
    poll_seconds: float = 2.0,
    heartbeat_seconds: float = 60.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wandb_reader: Callable[[str], Mapping[str, Any]] = terminal_wandb_summary,
) -> int:
    seen_url = False
    seen_incidents: set[str] = set()
    previous_progress = ""
    last_progress = float("-inf")
    while True:
        started = clock()
        projection = run_projection(conn, int(run_id))

        def send(event: str, **payload: Any) -> None:
            emit(
                json_safe(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "event": event,
                        "run_id": int(run_id),
                        "observed_at": _iso_utc(),
                        **payload,
                    }
                )
            )

        url = str((projection.get("wandb") or {}).get("url") or "")
        if url and not seen_url:
            seen_url = True
            send("wandb_url", url=url, wandb_run_id=(projection.get("wandb") or {}).get("run_id"))
        semantic = _semantic_progress(projection)
        now = clock()
        if semantic != previous_progress or now - last_progress >= heartbeat_seconds:
            send("progress", run=projection)
            previous_progress = semantic
            last_progress = now
        for incident in (projection.get("incidents") or {}).get("current") or []:
            fingerprint = str(incident["fingerprint"])
            if fingerprint in seen_incidents:
                continue
            seen_incidents.add(fingerprint)
            send("potential_bug", incident=incident, run=projection)
        classification = projection.get("terminal_classification")
        if classification:
            wandb_summary: Mapping[str, Any] = {}
            if url:
                try:
                    wandb_summary = wandb_reader(url)
                except Exception as exc:
                    send(
                        "reporting_warning",
                        category="wandb_terminal_read_failed",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
            send("terminal", **terminal_payload(projection, wandb_summary=wandb_summary))
            return {
                "accepted": 0,
                "completed": 0,
                "goal_rejected": 2,
                "canceled": 3,
                "operational_failure": 1,
            }[str(classification)]
        elapsed = clock() - started
        sleep(max(0.0, float(poll_seconds) - elapsed))
