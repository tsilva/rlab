from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rlab.checkpoint_eval_worker import update_best_checkpoint_summary
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.telemetry_mailbox import (
    REMOTE_CONFIRM_TIMEOUT_SECONDS,
    claim_run_metric_batches,
    commit_published_batches,
    decode_metric_batch,
    mark_submitted_batches,
    pending_metric_run_ids,
    release_metric_batch_claims,
    release_wandb_run_lock,
)
from rlab.job_queue import record_job_event
from rlab.wandb_publisher import (
    WandbProjector,
    _publish_frame,
    project_payload_to_run,
)
from rlab.wandb_utils import load_wandb_env, resolve_wandb_namespace
from rlab.metric_names import (
    EVAL_ACCEPTANCE_DURATION_SECONDS,
    EVAL_ACCEPTANCE_EPISODES_COMPLETED,
    EVAL_ACCEPTANCE_EPISODES_PLANNED,
    EVAL_ACCEPTANCE_FAILURE_COUNT,
    EVAL_ACCEPTANCE_PASS,
    EVAL_FULL_CHECKPOINT_STEP,
    LEADER_CHECKPOINT_ACCEPTANCE_PASS,
    LEADER_CHECKPOINT_STEP,
)


SUMMARY_CURSOR_KEY = "_rlab_telemetry_cursors"
DEFAULT_BATCH_LIMIT = 100
ACTOR_POLL_SECONDS = 2.0
MAX_FINALIZATION_ATTEMPTS = 3
FINALIZATION_RETRY_DELAYS_SECONDS = (15, 30, 60)
TERMINAL_WANDB_STATES = frozenset({"finished", "crashed", "failed", "killed"})
SUPPORTED_FRAME_KINDS = {
    "history",
    "histogram",
    "checkpoint_eval",
    "checkpoint_preview",
    "projection",
}


class InvalidTelemetryBatchError(RuntimeError):
    pass


class WandbFinalizationVerificationError(RuntimeError):
    def __init__(self, predicates: list[str]) -> None:
        self.predicates = tuple(predicates)
        super().__init__("W&B finalization verification failed: " + ", ".join(self.predicates))


@dataclass(frozen=True)
class WandbPublicationState:
    state: str
    cursors: dict[str, int]
    step_max: float | None


class WandbCursorConfirmationError(RuntimeError):
    pass


def _cursor_mapping(raw: object) -> dict[str, int]:
    items = getattr(raw, "items", None)
    if not callable(items):
        return {}
    return {str(key): int(value) for key, value in items()}


def _summary_step_max(raw: object) -> float | None:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    items = getattr(raw, "items", None)
    if not callable(items):
        return None
    try:
        value = dict(items()).get("max")
        return None if value is None else float(value)
    except TypeError, ValueError:
        return None


def _train_config(run: dict[str, Any]) -> dict[str, Any]:
    config = dict(run.get("train_config") or {})
    for key in (
        "run_name",
        "wandb_group",
        "wandb_tags",
        "wandb_run_id",
    ):
        if run.get(key) is not None:
            config[key] = run[key]
    return config


def _remote_publication_state(
    train_config: dict[str, Any],
) -> WandbPublicationState:
    load_wandb_env()
    import wandb

    entity, project = resolve_wandb_namespace(
        train_config.get("wandb_entity"),
        train_config.get("wandb_project"),
        str(train_config.get("game") or ""),
        env_provider=train_config.get("env_provider"),
    )
    run_id = str(train_config["wandb_run_id"])
    remote = wandb.Api().run(f"{entity}/{project}/{run_id}")
    summary = dict(remote.summary)
    raw = summary.get(SUMMARY_CURSOR_KEY) or {}
    return WandbPublicationState(
        state=str(getattr(remote, "state", "") or "").lower(),
        cursors=_cursor_mapping(raw),
        step_max=_summary_step_max(summary.get("global_step")),
    )


def _wandb_api_run(train_config: dict[str, Any]):
    load_wandb_env()
    import wandb

    entity, project = resolve_wandb_namespace(
        train_config.get("wandb_entity"),
        train_config.get("wandb_project"),
        str(train_config.get("game") or ""),
        env_provider=train_config.get("env_provider"),
    )
    return wandb.Api().run(f"{entity}/{project}/{train_config['wandb_run_id']}")


def _remote_has_promoted_artifact(remote, checkpoint_sha256: str) -> bool:
    logged = getattr(remote, "logged_artifacts", None)
    if not callable(logged):
        return False
    for artifact in logged():
        aliases = {str(value) for value in (getattr(artifact, "aliases", ()) or ())}
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        if "promoted" in aliases and str(metadata.get("checkpoint_sha256") or "") == str(
            checkpoint_sha256
        ):
            return True
    return False


def _canonical_goal_summary(run: dict[str, Any]) -> dict[str, Any]:
    """Return the terminal goal evidence that must win over later eval projections."""

    outcome = str(run.get("outcome") or "unknown")
    summary: dict[str, Any] = {
        "rlab/goal/outcome": outcome,
        "rlab/operational/status": "finished",
    }
    if outcome != "accepted":
        return summary
    promotion = dict(run.get("promotion_json") or {})
    raw_metrics = dict(promotion.get("raw_metrics") or {})
    aggregates = dict(raw_metrics.get("_acceptance_aggregates") or {})
    checkpoint_step = int(
        promotion.get("checkpoint_step") or raw_metrics.get("checkpoint_step") or 0
    )
    raw_metrics.setdefault("checkpoint_step", checkpoint_step)
    summary.update(
        {
            EVAL_ACCEPTANCE_PASS: 1.0,
            EVAL_ACCEPTANCE_EPISODES_PLANNED: int(
                aggregates.get("episodes_planned") or raw_metrics.get("episodes") or 0
            ),
            EVAL_ACCEPTANCE_EPISODES_COMPLETED: int(
                aggregates.get("episodes_completed") or raw_metrics.get("episodes") or 0
            ),
            EVAL_ACCEPTANCE_FAILURE_COUNT: int(aggregates.get("failure_count") or 0),
            EVAL_ACCEPTANCE_DURATION_SECONDS: float(
                raw_metrics.get("_acceptance_duration_seconds")
                or raw_metrics.get("eval/full/duration/seconds")
                or 0.0
            ),
            EVAL_FULL_CHECKPOINT_STEP: checkpoint_step,
        }
    )
    for key, value in raw_metrics.items():
        if (
            str(key).startswith("eval/full/")
            and isinstance(value, int | float)
            and not isinstance(value, bool)
        ):
            summary[str(key)] = value
    leader = SimpleNamespace(summary={})
    update_best_checkpoint_summary(
        leader,
        metrics=raw_metrics,
        checkpoint_path=str(promotion.get("checkpoint_uri") or ""),
        checkpoint_step_value=checkpoint_step,
        artifact_ref=str(promotion.get("checkpoint_uri") or ""),
        eval_source="modal:acceptance",
        selection_rank=_train_config(run).get("selection_rank") or (),
        force=True,
    )
    summary.update(leader.summary)
    summary[LEADER_CHECKPOINT_ACCEPTANCE_PASS] = 1.0
    return summary


def _wandb_finalization_failures(
    run: dict[str, Any],
    candidate,
    expected_cursors: dict[str, int],
) -> list[str]:
    summary = dict(candidate.summary)
    cursors = _cursor_mapping(summary.get(SUMMARY_CURSOR_KEY) or {})
    failures: list[str] = []
    if str(getattr(candidate, "state", "")).lower() != "finished":
        failures.append("remote_state")
    for name, sequence in expected_cursors.items():
        if int(cursors.get(name, -1)) != sequence:
            failures.append(f"cursor:{name}")
    outcome = str(run.get("outcome") or "unknown")
    if str(summary.get("rlab/goal/outcome") or "") != outcome:
        failures.append("outcome")
    if outcome == "accepted":
        promotion = dict(run.get("promotion_json") or {})
        expected_step = int(promotion.get("checkpoint_step") or 0)
        try:
            leader_step = int(summary.get(LEADER_CHECKPOINT_STEP))
        except TypeError, ValueError:
            leader_step = -1
        if leader_step != expected_step:
            failures.append("leader_step")
        try:
            acceptance_pass = float(summary.get(LEADER_CHECKPOINT_ACCEPTANCE_PASS, 0.0) or 0.0)
        except TypeError, ValueError:
            acceptance_pass = 0.0
        if acceptance_pass != 1.0:
            failures.append("acceptance")
        if not _remote_has_promoted_artifact(
            candidate, str(promotion.get("checkpoint_sha256") or "")
        ):
            failures.append("artifact")
    return failures


def finalize_finishing_run(conn, train_job_id: int) -> bool:
    lock_key = f"rlab-wandb-run:{int(train_job_id)}"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0)) AS acquired",
            {"key": lock_key},
        )
        if not bool(cur.fetchone()["acquired"]):
            return False
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.*, r.outcome, r.promotion_json
                FROM train_jobs t
                JOIN eval_runs r ON r.train_job_id = t.id
                WHERE t.id = %(id)s AND t.live_publication_status = 'finishing'
                  AND (t.live_publication_next_retry_at IS NULL
                       OR t.live_publication_next_retry_at <= now())
                """,
                {"id": int(train_job_id)},
            )
            run = cur.fetchone()
            if not run:
                return False
            run = dict(run)
            cur.execute(
                """
                SELECT s.stream_id, s.final_sequence, s.published_sequence
                FROM metric_streams s
                JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                WHERE a.train_job_id = %(id)s
                """,
                {"id": int(train_job_id)},
            )
            streams = [dict(row) for row in cur.fetchall()]
        if any(
            row.get("final_sequence") is None
            or int(row["published_sequence"]) < int(row["final_sequence"])
            for row in streams
        ):
            return False
        expected = {str(row["stream_id"]): int(row["final_sequence"]) for row in streams}
        train_config = _train_config(run)
        remote = _wandb_api_run(train_config)
        remote_summary = dict(remote.summary)
        remote_cursors = _cursor_mapping(remote_summary.get(SUMMARY_CURSOR_KEY) or {})
        failures = _wandb_finalization_failures(run, remote, expected)
        if failures:
            projector = WandbProjector.resume(
                train_config,
                allow_create=False,
                update_finish_state=True,
            )
            projector.run.summary[SUMMARY_CURSOR_KEY] = {
                **remote_cursors,
                **expected,
            }
            for key, value in _canonical_goal_summary(run).items():
                projector.run.summary[key] = value
            projector.close()
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                remote = _wandb_api_run(train_config)
                failures = _wandb_finalization_failures(run, remote, expected)
                if not failures:
                    break
                time.sleep(1.0)
            else:
                raise WandbFinalizationVerificationError(failures)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET live_publication_status = 'complete',
                        live_publication_error = NULL,
                        live_publication_next_retry_at = NULL
                    WHERE id = %(id)s AND live_publication_status = 'finishing'
                    """,
                    {"id": int(train_job_id)},
                )
                return cur.rowcount == 1
    except Exception as exc:
        attempts = int((locals().get("run") or {}).get("live_publication_attempts") or 0) + 1
        terminal = attempts >= MAX_FINALIZATION_ATTEMPTS
        retry_delay = FINALIZATION_RETRY_DELAYS_SECONDS[
            min(attempts - 1, len(FINALIZATION_RETRY_DELAYS_SECONDS) - 1)
        ]
        if isinstance(exc, WandbFinalizationVerificationError):
            error = str(exc)
        else:
            error = f"W&B finalization remote_api failed: {type(exc).__name__}: {exc}"
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET live_publication_error = %(error)s,
                        live_publication_attempts = %(attempts)s,
                        live_publication_status = CASE WHEN %(terminal)s
                          THEN 'failed' ELSE 'finishing' END,
                        live_publication_next_retry_at = CASE WHEN %(terminal)s
                          THEN NULL
                          ELSE now() + (%(retry_delay)s * interval '1 second') END,
                        status = CASE WHEN %(terminal)s
                          AND status IN ('finalizing', 'succeeded')
                          THEN 'finalization_failed' ELSE status END,
                        finished_at = CASE WHEN %(terminal)s
                          AND status IN ('finalizing', 'succeeded')
                          THEN now() ELSE finished_at END,
                        error = CASE WHEN %(terminal)s
                          AND status IN ('finalizing', 'succeeded')
                          THEN %(error)s ELSE error END
                    WHERE id = %(id)s AND live_publication_status = 'finishing'
                    """,
                    {
                        "id": int(train_job_id),
                        "error": error[:4000],
                        "attempts": attempts,
                        "terminal": terminal,
                        "retry_delay": retry_delay,
                    },
                )
        raise
    finally:
        release_wandb_run_lock(conn, int(train_job_id))


def _partition_batches(
    batches: list[dict[str, Any]],
    remote: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    confirmed: list[dict[str, Any]] = []
    awaiting_confirmation: list[dict[str, Any]] = []
    unpublished: list[dict[str, Any]] = []
    for row in batches:
        stream_id = str(row["stream_id"])
        sequence = int(row["batch_sequence"])
        if int(remote.get(stream_id, 0)) >= sequence:
            confirmed.append(row)
        elif int(row.get("submitted_sequence") or 0) >= sequence:
            awaiting_confirmation.append(row)
        else:
            unpublished.append(row)
    return confirmed, awaiting_confirmation, unpublished


def _submitted_at(value: object) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _raise_for_stalled_confirmations(
    batches: list[dict[str, Any]],
    remote: WandbPublicationState,
    *,
    now: datetime | None = None,
) -> None:
    if not batches:
        return
    now = now or datetime.now(UTC)
    stalled: list[tuple[dict[str, Any], float | None]] = []
    terminal_remote = remote.state in TERMINAL_WANDB_STATES
    for batch in batches:
        submitted_at = _submitted_at(batch.get("submitted_at"))
        age = None if submitted_at is None else max(0.0, (now - submitted_at).total_seconds())
        if terminal_remote or (age is not None and age >= REMOTE_CONFIRM_TIMEOUT_SECONDS):
            stalled.append((batch, age))
    if not stalled:
        return
    reason = "terminal_remote_state" if terminal_remote else "confirmation_timeout"
    details = []
    for batch, age in stalled:
        stream_id = str(batch["stream_id"])
        expected = int(batch["batch_sequence"])
        observed = int(remote.cursors.get(stream_id, 0))
        age_text = "unknown" if age is None else f"{age:.1f}"
        details.append(
            f"{stream_id}:expected={expected}:observed={observed}:age_seconds={age_text}"
        )
    raise WandbCursorConfirmationError(
        "W&B cursor confirmation failed: "
        f"reason={reason}, remote_state={remote.state or 'unknown'}, "
        f"streams=[{', '.join(details)}]"
    )


def _record_committed_effects(
    conn,
    *,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
    decoded: dict[int, list[dict[str, Any]]],
    wandb_url: str | None,
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs t
                SET live_publication_status = CASE
                      WHEN NOT EXISTS (
                        SELECT 1 FROM metric_batches b
                        JOIN metric_streams s ON s.stream_id = b.stream_id
                        JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                        WHERE a.train_job_id = t.id
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM metric_streams s
                        JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                        WHERE a.train_job_id = t.id
                          AND (s.final_sequence IS NULL
                               OR s.published_sequence < s.final_sequence)
                      ) THEN 'pending'
                      ELSE 'live'
                    END,
                    live_publication_error = NULL,
                    live_publication_attempts = 0,
                    live_publication_next_retry_at = NULL,
                    wandb_url = COALESCE(wandb_url, %(wandb_url)s)
                WHERE t.id = %(train_job_id)s
                """,
                {
                    "train_job_id": int(run["id"]),
                    "wandb_url": wandb_url,
                },
            )
            projection_ids: list[int] = []
            for batch in batches:
                for frame in decoded[int(batch["id"])]:
                    if str(frame.get("kind") or "") != "projection":
                        continue
                    eval_job_id = frame.get("payload", {}).get("eval_job_id")
                    if eval_job_id is not None:
                        projection_ids.append(int(eval_job_id))
            if projection_ids:
                cur.execute(
                    """
                    UPDATE eval_jobs
                    SET projected_at = now(), projection_error = NULL,
                        projection_next_retry_at = NULL, updated_at = now()
                    WHERE id = ANY(%(ids)s)
                    """,
                    {"ids": sorted(set(projection_ids))},
                )


def publish_claimed_run(
    conn,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
) -> int:
    train_config = _train_config(run)
    decoded: dict[int, list[dict[str, Any]]] = {}
    for batch in batches:
        try:
            frames = decode_metric_batch(bytes(batch["payload"]))
        except Exception as exc:
            raise InvalidTelemetryBatchError(str(exc)) from exc
        unsupported = {
            str(frame.get("kind") or "history") for frame in frames
        } - SUPPORTED_FRAME_KINDS
        if unsupported:
            raise InvalidTelemetryBatchError(
                f"unsupported telemetry frame kinds: {sorted(unsupported)}"
            )
        decoded[int(batch["id"])] = frames
    remote_state = _remote_publication_state(train_config)
    remote = remote_state.cursors
    confirmed, awaiting_confirmation, unpublished = _partition_batches(batches, remote)
    if confirmed:
        commit_published_batches(conn, confirmed)
        _record_committed_effects(
            conn,
            run=run,
            batches=confirmed,
            decoded=decoded,
            wandb_url=None,
        )
        run["live_publication_attempts"] = 0
    if awaiting_confirmation:
        _raise_for_stalled_confirmations(awaiting_confirmation, remote_state)
        mark_submitted_batches(conn, awaiting_confirmation)
    wandb_url: str | None = None
    if unpublished:
        session_step_max = remote_state.step_max
        expected: dict[str, int] = {}
        for row in unpublished:
            stream_id = str(row["stream_id"])
            expected[stream_id] = max(expected.get(stream_id, 0), int(row["batch_sequence"]))
        projector = WandbProjector.resume(
            train_config,
            allow_create=True,
            update_finish_state=str(run.get("status") or "")
            in {"succeeded", "failed", "finalization_failed", "canceled"},
        )
        try:
            wandb_url = str(getattr(projector.run, "url", "") or "") or None
            args = SimpleNamespace(**train_config)
            config = resolve_env_config(env_config_from_args(args, include_states=True))
            for batch in unpublished:
                for frame in decoded[int(batch["id"])]:
                    try:
                        frame_step = float(frame["global_step"])
                    except KeyError, TypeError, ValueError:
                        frame_step = None
                    if frame_step is not None:
                        session_step_max = max(session_step_max or frame_step, frame_step)
                    kind = str(frame.get("kind") or "history")
                    payload = dict(frame["payload"])
                    if kind == "projection":
                        project_payload_to_run(
                            projector.run,
                            payload,
                            allow_artifact_references=True,
                        )
                    else:
                        _publish_frame(
                            projector.run,
                            {
                                "kind": kind,
                                "payload_json": __import__("json").dumps(payload),
                            },
                            args=args,
                            config=config,
                        )
            merged = dict(remote)
            for stream_id, sequence in expected.items():
                merged[stream_id] = max(int(merged.get(stream_id, 0)), int(sequence))
            projector.run.summary[SUMMARY_CURSOR_KEY] = merged
            if session_step_max is not None:
                summary_step: int | float = session_step_max
                if session_step_max.is_integer():
                    summary_step = int(session_step_max)
                projector.run.summary["global_step"] = {"max": summary_step}
        finally:
            projector.close()
        mark_submitted_batches(conn, unpublished)
    if awaiting_confirmation or unpublished:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET live_publication_status = 'pending',
                        live_publication_error = NULL,
                        live_publication_next_retry_at = NULL,
                        wandb_url = COALESCE(wandb_url, %(wandb_url)s)
                    WHERE id = %(train_job_id)s
                    """,
                    {
                        "train_job_id": int(run["id"]),
                        "wandb_url": wandb_url,
                    },
                )
    return len(confirmed)


def _drain_claim(conn, run: dict[str, Any], batches: list[dict[str, Any]]) -> int:
    try:
        return publish_claimed_run(conn, run, batches)
    except Exception as exc:
        release_metric_batch_claims(conn, batches, error=repr(exc))
        attempts = int(run.get("live_publication_attempts") or 0) + 1
        invalid_batch = isinstance(exc, InvalidTelemetryBatchError)
        finalizing = str(run.get("status") or "") == "finalizing"
        terminal = finalizing and (invalid_batch or attempts >= MAX_FINALIZATION_ATTEMPTS)
        retry_delay = (
            FINALIZATION_RETRY_DELAYS_SECONDS[
                min(attempts - 1, len(FINALIZATION_RETRY_DELAYS_SECONDS) - 1)
            ]
            if finalizing
            else 30
        )
        error = (str(exc) if isinstance(exc, WandbCursorConfirmationError) else repr(exc))[:4000]
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET live_publication_status = CASE WHEN %(terminal)s
                          THEN 'failed' ELSE 'pending' END,
                        live_publication_error = %(error)s,
                        live_publication_attempts = %(attempts)s,
                        live_publication_next_retry_at = CASE WHEN %(terminal)s
                          THEN NULL
                          ELSE now() + (%(retry_delay)s * interval '1 second') END,
                        status = CASE WHEN %(terminal)s AND status = 'finalizing'
                          THEN 'finalization_failed' ELSE status END,
                        finished_at = CASE WHEN %(terminal)s AND status = 'finalizing'
                          THEN now() ELSE finished_at END,
                        error = CASE WHEN %(terminal)s AND status = 'finalizing'
                          THEN %(error)s ELSE error END
                    WHERE id = %(train_job_id)s
                    """,
                    {
                        "train_job_id": int(run["id"]),
                        "error": error,
                        "attempts": attempts,
                        "terminal": terminal,
                        "retry_delay": retry_delay,
                    },
                )
                record_job_event(
                    conn,
                    job_id=int(run["id"]),
                    event_type=(
                        "live_publication_failed" if terminal else "live_publication_retry"
                    ),
                    message=error,
                    metadata={
                        "attempts": attempts,
                        "terminal": terminal,
                        "run_status": str(run.get("status") or ""),
                    },
                )
        raise
    finally:
        release_wandb_run_lock(conn, int(run["id"]))


def drain_once(
    conn,
    *,
    owner: str | None = None,
    limit: int = DEFAULT_BATCH_LIMIT,
    exclude_train_job_ids: tuple[int, ...] = (),
    train_job_id: int | None = None,
) -> int:
    owner = owner or f"fleet-publisher-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    claim = claim_run_metric_batches(
        conn,
        owner=owner,
        limit=limit,
        exclude_train_job_ids=exclude_train_job_ids,
        train_job_id=train_job_id,
    )
    if claim is None:
        if train_job_id is not None:
            return int(finalize_finishing_run(conn, int(train_job_id)))
        return 0
    run, batches = claim
    return _drain_claim(conn, run, batches)


def _publisher_actor_done(conn, train_job_id: int) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT live_publication_status
            FROM train_jobs
            WHERE id = %(id)s
            """,
            {"id": int(train_job_id)},
        )
        row = cur.fetchone()
    return not row or str(row["live_publication_status"]) in {
        "complete",
        "disabled",
        "failed",
    }


def run_publisher_actor(
    conn,
    train_job_id: int,
    *,
    limit: int = DEFAULT_BATCH_LIMIT,
    poll_seconds: float = ACTOR_POLL_SECONDS,
    once: bool = False,
) -> int:
    """Own one run until publication completes, surviving idle producer gaps.

    The lifetime advisory lock makes a launchd manager restart safe: a replacement actor waits
    behind the existing owner instead of opening a second W&B SDK writer. Each drain still uses
    the narrower run lock so recovery and tests keep the same transactional claim boundary.
    """

    actor_key = f"rlab-wandb-actor:{int(train_job_id)}"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_lock(hashtextextended(%(key)s, 0))",
            {"key": actor_key},
        )
    published = 0
    try:
        while True:
            published += drain_once(
                conn,
                limit=max(1, int(limit)),
                train_job_id=int(train_job_id),
            )
            done = _publisher_actor_done(conn, int(train_job_id))
            # Do not leave a read transaction open while an active producer is idle.
            conn.rollback()
            if done or once:
                return published
            time.sleep(max(float(poll_seconds), 0.01))
    finally:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0))",
                {"key": actor_key},
            )
        conn.rollback()


def drain_cycle(conn, *, max_runs: int = 10, limit: int = 20) -> dict[str, int]:
    attempted: list[int] = []
    published = 0
    failed = 0
    for _ in range(max(1, int(max_runs))):
        owner = f"fleet-publisher-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        claim = claim_run_metric_batches(
            conn,
            owner=owner,
            limit=limit,
            exclude_train_job_ids=attempted,
        )
        if claim is None:
            break
        run, batches = claim
        train_job_id = int(run["id"])
        attempted.append(train_job_id)
        try:
            published += _drain_claim(conn, run, batches)
        except Exception:
            failed += 1
    return {"runs_attempted": len(attempted), "batches_published": published, "runs_failed": failed}


def drain_cycle_parallel(
    conn,
    *,
    repo_root: Path,
    max_runs: int = 100,
    limit: int = 20,
    deadline_monotonic: float | None = None,
) -> dict[str, int]:
    """Launch independent W&B publishers without blocking eval reconciliation.

    W&B owns process-global SDK state, so process isolation preserves the per-run
    advisory-lock contract while allowing active runs to drain concurrently. The
    child records publication failure durably; the service observes it next pass.
    """

    run_ids = pending_metric_run_ids(conn, limit=max_runs)
    if not run_ids:
        return {
            "runs_attempted": 0,
            "runs_started": 0,
            "batches_published": 0,
            "runs_failed": 0,
        }
    if deadline_monotonic is not None and deadline_monotonic <= time.monotonic():
        return {
            "runs_attempted": len(run_ids),
            "runs_started": 0,
            "batches_published": 0,
            "runs_failed": len(run_ids),
        }
    started = 0
    failed = 0
    for train_job_id in run_ids:
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "rlab.fleet_wandb_publisher",
                    "--limit",
                    str(max(1, int(limit))),
                    "--train-job-id",
                    str(train_job_id),
                ],
                cwd=repo_root,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
        except OSError:
            failed += 1
        else:
            started += 1
    return {
        "runs_attempted": len(run_ids),
        "runs_started": started,
        "batches_published": 0,
        "runs_failed": failed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drain Fleet telemetry mailboxes to W&B.")
    parser.add_argument("--limit", type=int, default=DEFAULT_BATCH_LIMIT)
    parser.add_argument("--train-job-id", type=int)
    parser.add_argument("--once", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from rlab.job_queue import connect, database_url

    # Publishing holds a session-scoped per-run lock for the whole W&B
    # session, so it must not use a PgBouncer-backed connection.
    conn = connect(database_url(use_direct=True))
    try:
        if args.train_job_id is not None:
            published = run_publisher_actor(
                conn,
                int(args.train_job_id),
                limit=max(1, args.limit),
                once=bool(args.once),
            )
        else:
            published = drain_once(conn, limit=max(1, args.limit))
        print(f"published_batches={published}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
