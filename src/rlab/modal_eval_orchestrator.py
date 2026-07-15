from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import tempfile
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

from rlab.job_queue import connect, database_url, json_arg
from rlab.checkpoint_eval_worker import evaluation_metric_payload
from rlab.eval_metrics import eval_by_start_rows
from rlab.metric_names import checkpoint_eval_stage_metric, validate_metric_payload
from rlab.ranking import parse_persisted_objective_rank, rank_score
from rlab.modal_eval_config import ModalEvalConfig, load_modal_eval_config, modal_app_name
from rlab.modal_app_cleanup import ModalAppClient, run_modal_app_cleanup
from rlab.modal_eval_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    apply_decision_rules,
    promotion_job_descriptor,
    stage_job_descriptor,
    validate_attempt_result,
    validate_announcement,
)
from rlab.modal_eval_storage import (
    ObjectNotFound,
    ObjectStore,
    object_store_base_uri,
    preview_public_base_url,
    preview_storage_base_uri,
)


EVAL_RECONCILE_LOCK = "rlab-fleet-reconciler:eval:modal-cpu"
ACTIVE_ATTEMPT_STATES = ("dispatching", "submitted")
MAX_PROJECTION_ATTEMPTS = 3
PROJECTION_RETRY_DELAYS_SECONDS = (30, 120, 300)


def deterministic_eval_failure(error: object) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "terminal:",
            "contract",
            "schema",
            "hash mismatch",
            "seed protocol",
            "environment contract is invalid",
        )
    )


class ModalInvoker(Protocol):
    def spawn(self, app_name: str, function_name: str, payload: Mapping[str, Any]) -> str: ...
    def poll(self, call_id: str) -> tuple[str, object | None]: ...
    def cancel(self, call_id: str) -> None: ...


class DefaultModalInvoker:
    def spawn(self, app_name: str, function_name: str, payload: Mapping[str, Any]) -> str:
        import modal

        function = modal.Function.from_name(app_name, function_name)
        call = function.spawn(dict(payload))
        return str(call.object_id)

    def poll(self, call_id: str) -> tuple[str, object | None]:
        import modal

        call = modal.FunctionCall.from_id(call_id)
        try:
            return "finished", call.get(timeout=0)
        except TimeoutError:
            return "pending", None
        except (
            modal.exception.FunctionTimeoutError,
            modal.exception.OutputExpiredError,
            modal.exception.RemoteError,
            modal.exception.UserCodeException,
            RuntimeError,
        ) as exc:
            return "failed", repr(exc)
        except Exception as exc:
            return "unknown", type(exc).__name__

    def cancel(self, call_id: str) -> None:
        import modal

        modal.FunctionCall.from_id(call_id).cancel()


def _try_lock(conn) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0)) AS acquired",
            {"key": EVAL_RECONCILE_LOCK},
        )
        row = cur.fetchone()
    return bool(row and row["acquired"])


def _unlock(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0))",
            {"key": EVAL_RECONCILE_LOCK},
        )


def ensure_eval_runs(conn) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval_runs (train_job_id, contract_json)
                SELECT id, train_config || jsonb_build_object('runtime_image_ref', runtime_image_ref)
                FROM train_jobs
                WHERE train_config->>'checkpoint_eval_backend' = 'modal'
                  AND status NOT IN ('canceled')
                ON CONFLICT (train_job_id) DO NOTHING
                """
            )
            return cur.rowcount


def _mark_eval_run_failed(conn, train_job_id: int, error: object) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_runs SET status = 'failed', error = %(error)s,
                  updated_at = now() WHERE train_job_id = %(job_id)s
                """,
                {"error": str(error)[:4000], "job_id": int(train_job_id)},
            )


def _insert_eval_job(
    conn, announcement: Mapping[str, Any], descriptor: Mapping[str, Any]
) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO eval_jobs (
              train_job_id, ledger_id, checkpoint_step, checkpoint_sha256,
              checkpoint_uri, metadata_uri, stage_name, stage_index, purpose,
              execution_key, job_key, contract_json, source_announcement_json,
              decision_rules_json, candidate_stop
            ) VALUES (
              %(train_job_id)s, %(ledger_id)s, %(checkpoint_step)s, %(checkpoint_sha256)s,
              %(checkpoint_uri)s, %(metadata_uri)s, %(stage_name)s, %(stage_index)s,
              %(purpose)s, %(execution_key)s, %(job_key)s, %(contract_json)s,
              %(announcement)s, %(rules)s, %(candidate_stop)s
            )
            ON CONFLICT (job_key) DO NOTHING
            RETURNING id
            """,
            {
                "train_job_id": int(announcement["train_job_id"]),
                "ledger_id": int(announcement["ledger_id"]),
                "checkpoint_step": int(announcement["step"]),
                "checkpoint_sha256": str(announcement["sha256"]),
                "checkpoint_uri": str(announcement["model_uri"]),
                "metadata_uri": str(announcement["metadata_uri"]),
                "stage_name": str(descriptor["stage_name"]),
                "stage_index": int(descriptor["stage_index"]),
                "purpose": str(descriptor["purpose"]),
                "execution_key": str(descriptor["execution_key"]),
                "job_key": str(descriptor["job_key"]),
                "contract_json": json_arg(descriptor["contract"]),
                "announcement": json_arg(announcement),
                "rules": json_arg(descriptor["decision_rules"]),
                "candidate_stop": bool(descriptor["candidate_stop"]),
            },
        )
        row = cur.fetchone()
    return int(row["id"]) if row else None


def ingest_announcements(
    conn, store: ObjectStore, *, deadline_monotonic: float, limit: int = 50
) -> int:
    ingested = 0
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.*, t.status AS train_status
            FROM eval_runs r
            JOIN train_jobs t ON t.id = r.train_job_id
            WHERE r.status IN ('active', 'awaiting_artifact_recovery', 'finalizing')
            ORDER BY r.updated_at, r.train_job_id
            """
        )
        runs = [dict(row) for row in cur.fetchall()]
    for run in runs:
        while ingested < limit and time.monotonic() < deadline_monotonic:
            ordinal = int(run["next_announcement_id"])
            key = f"artifact-announcements/{int(run['train_job_id'])}/{ordinal:08d}.json"
            announcement = store.get_json_optional(key)
            if announcement is None:
                complete = store.get_json_optional(
                    f"artifact-announcements/{int(run['train_job_id'])}/complete.json"
                )
                if complete is not None:
                    last_id = int(complete.get("last_ledger_id") or 0)
                    seen = ordinal > last_id
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                UPDATE eval_runs
                                SET complete_announcement_seen = %(seen)s,
                                    status = CASE WHEN %(seen)s THEN 'finalizing' ELSE status END,
                                    updated_at = now(),
                                    error = CASE WHEN %(seen)s THEN NULL ELSE 'announcement gap' END
                                WHERE train_job_id = %(job_id)s
                                """,
                                {
                                    "seen": seen,
                                    "job_id": int(run["train_job_id"]),
                                },
                            )
                    if not seen and str(run["train_status"]) in {
                        "finalizing",
                        "succeeded",
                        "failed",
                        "finalization_failed",
                    }:
                        with conn:
                            with conn.cursor() as cur:
                                cur.execute(
                                    """
                                    UPDATE eval_runs SET status = 'awaiting_artifact_recovery',
                                      error = 'complete marker precedes missing announcement', updated_at = now()
                                    WHERE train_job_id = %(job_id)s
                                    """,
                                    {"job_id": int(run["train_job_id"])},
                                )
                break
            if int(announcement.get("train_job_id") or 0) != int(run["train_job_id"]):
                _mark_eval_run_failed(
                    conn, int(run["train_job_id"]), "checkpoint announcement train job id mismatch"
                )
                break
            if int(announcement.get("ledger_id") or 0) != ordinal:
                _mark_eval_run_failed(
                    conn, int(run["train_job_id"]), "checkpoint announcement ledger id mismatch"
                )
                break
            if str(announcement.get("kind")) != "tombstone":
                try:
                    announcement = validate_announcement(
                        announcement,
                        materialized_train_config=dict(run["contract_json"]),
                    )
                    for uri, expected_sha in (
                        (str(announcement["model_uri"]), str(announcement["sha256"])),
                        (
                            str(announcement["metadata_uri"]),
                            str(announcement["metadata_sha256"]),
                        ),
                        (
                            str(announcement["eval"]["asset"]["object_uri"]),
                            str(announcement["eval"]["asset"]["sha256"]),
                        ),
                    ):
                        head = store.head(uri)
                        if int(head["size"]) < 1:
                            raise ValueError("checkpoint artifact is empty")
                        remote_sha = str(head.get("metadata", {}).get("sha256") or "")
                        if store.scheme == "s3" and remote_sha != expected_sha:
                            raise ValueError("checkpoint artifact hash metadata mismatch")
                    metadata = store.get_json(str(announcement["metadata_uri"]))
                    if (
                        int(metadata.get("queue_train_job_id") or 0)
                        != int(announcement["train_job_id"])
                        or int(metadata.get("checkpoint_step") or 0) != int(announcement["step"])
                        or str(metadata.get("runtime_image_ref") or "")
                        != str(announcement["runtime_image_ref"])
                    ):
                        raise ValueError("checkpoint metadata does not match the announcement")
                except ObjectNotFound:
                    break
                except ValueError as exc:
                    _mark_eval_run_failed(conn, int(run["train_job_id"]), exc)
                    break
            with conn:
                if str(announcement.get("kind")) == "checkpoint":
                    stages = announcement.get("eval", {}).get("stages") or []
                    if stages:
                        descriptor = stage_job_descriptor(announcement, stage_index=0)
                    else:
                        descriptor = promotion_job_descriptor(announcement)
                    _insert_eval_job(conn, announcement, descriptor)
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE eval_jobs
                            SET status = 'skipped_stale', finished_at = now(), updated_at = now(),
                                error = 'coalesced by newer undispatched screen'
                            WHERE train_job_id = %(job_id)s AND stage_index = 0
                              AND status IN ('pending', 'blocked_budget')
                              AND checkpoint_step < %(step)s
                            """,
                            {
                                "job_id": int(run["train_job_id"]),
                                "step": int(announcement["step"]),
                            },
                        )
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE eval_runs
                        SET next_announcement_id = next_announcement_id + 1,
                            status = 'active', error = NULL, updated_at = now()
                        WHERE train_job_id = %(job_id)s
                          AND next_announcement_id = %(ordinal)s
                        """,
                        {"job_id": int(run["train_job_id"]), "ordinal": ordinal},
                    )
            run["next_announcement_id"] = ordinal + 1
            ingested += 1
    return ingested


def ingest_mailbox_announcements(
    conn,
    store: ObjectStore,
    *,
    limit: int = 50,
) -> int:
    ingested = 0
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.*, r.contract_json, r.next_announcement_id
            FROM attempt_events e
            JOIN worker_attempts a ON a.attempt_id = e.attempt_id
            JOIN eval_runs r ON r.train_job_id = a.train_job_id
            JOIN train_jobs t ON t.id = r.train_job_id
            WHERE e.event_type IN (
                'checkpoint_ready', 'checkpoint_tombstone', 'checkpoint_stream_closed'
              )
              AND t.telemetry_transport = 'neon_mailbox_v1'
              AND (e.next_retry_at IS NULL OR e.next_retry_at <= now())
            ORDER BY e.id
            LIMIT %(limit)s
            """,
            {"limit": max(1, int(limit))},
        )
        events = [dict(row) for row in cur.fetchall()]
    for event in events:
        payload = dict(event.get("payload_json") or {})
        event_type = str(event["event_type"])
        train_job_id = int(payload.get("train_job_id") or 0)
        if not train_job_id:
            raise ValueError("checkpoint mailbox event is missing train_job_id")
        if event_type == "checkpoint_stream_closed":
            last_id = int(payload.get("last_ledger_id") or 0)
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT next_announcement_id FROM eval_runs "
                        "WHERE train_job_id = %(train_job_id)s FOR UPDATE",
                        {"train_job_id": train_job_id},
                    )
                    current = cur.fetchone()
                    if not current or int(current["next_announcement_id"]) <= last_id:
                        continue
                    cur.execute(
                        """
                        UPDATE eval_runs
                        SET complete_announcement_seen = TRUE,
                            status = 'finalizing',
                            error = NULL,
                            updated_at = now()
                        WHERE train_job_id = %(train_job_id)s
                        """,
                        {"train_job_id": train_job_id, "last_id": last_id},
                    )
                    if cur.rowcount:
                        cur.execute(
                            "DELETE FROM attempt_events WHERE id = %(id)s",
                            {"id": int(event["id"])},
                        )
            ingested += 1
            continue
        ledger_id = int(payload.get("ledger_id") or 0)
        if ledger_id != int(event["next_announcement_id"]):
            continue
        announcement: Mapping[str, Any] = payload
        if event_type == "checkpoint_ready":
            try:
                announcement = validate_announcement(
                    payload,
                    materialized_train_config=dict(event["contract_json"]),
                )
                for uri, expected_sha in (
                    (str(announcement["model_uri"]), str(announcement["sha256"])),
                    (
                        str(announcement["metadata_uri"]),
                        str(announcement["metadata_sha256"]),
                    ),
                    (
                        str(announcement["eval"]["asset"]["object_uri"]),
                        str(announcement["eval"]["asset"]["sha256"]),
                    ),
                ):
                    head = store.head(uri)
                    if int(head["size"]) < 1:
                        raise ValueError("checkpoint artifact is empty")
                    remote_sha = str(head.get("metadata", {}).get("sha256") or "")
                    if store.scheme == "s3" and remote_sha != expected_sha:
                        raise ValueError("checkpoint artifact hash metadata mismatch")
            except Exception as exc:
                attempts = int(event.get("attempts") or 0) + 1
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE attempt_events
                            SET attempts = %(attempts)s, last_error = %(error)s,
                                next_retry_at = now() + (
                                  LEAST(300, 30 * %(attempts)s) * interval '1 second'
                                )
                            WHERE id = %(id)s
                            """,
                            {
                                "id": int(event["id"]),
                                "attempts": attempts,
                                "error": repr(exc)[:4000],
                            },
                        )
                        if attempts >= 3:
                            cur.execute(
                                """
                                UPDATE eval_runs
                                SET status = 'failed', error = %(error)s, updated_at = now()
                                WHERE train_job_id = %(train_job_id)s
                                """,
                                {
                                    "train_job_id": train_job_id,
                                    "error": (
                                        "checkpoint artifact verification failed: " + repr(exc)
                                    )[:4000],
                                },
                            )
                continue
        with conn:
            if event_type == "checkpoint_ready" and str(announcement.get("kind")) == "checkpoint":
                stages = announcement.get("eval", {}).get("stages") or []
                descriptor = (
                    stage_job_descriptor(announcement, stage_index=0)
                    if stages
                    else promotion_job_descriptor(announcement)
                )
                _insert_eval_job(conn, announcement, descriptor)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_runs
                    SET next_announcement_id = next_announcement_id + 1,
                        status = 'active', error = NULL, updated_at = now()
                    WHERE train_job_id = %(train_job_id)s
                      AND next_announcement_id = %(ledger_id)s
                    """,
                    {"train_job_id": train_job_id, "ledger_id": ledger_id},
                )
                if cur.rowcount:
                    cur.execute(
                        "DELETE FROM attempt_events WHERE id = %(id)s",
                        {"id": int(event["id"])},
                    )
        ingested += 1
    return ingested


def publish_skipped_decisions(conn, store: ObjectStore, *, limit: int = 100) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.*, t.telemetry_transport FROM eval_jobs j
            JOIN train_jobs t ON t.id = j.train_job_id
            WHERE j.status = 'skipped_stale' AND j.decision_json IS NULL
            ORDER BY j.id LIMIT %(limit)s
            """,
            {"limit": max(1, int(limit))},
        )
        jobs = [dict(row) for row in cur.fetchall()]
    published = 0
    for job in jobs:
        decision = {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "status": "skipped_stale",
            "job_key": str(job["job_key"]),
            "execution_key": str(job["execution_key"]),
            "train_job_id": int(job["train_job_id"]),
            "ledger_id": int(job["ledger_id"]),
            "stage_name": str(job["stage_name"]),
            "stage_index": int(job["stage_index"]),
            "purpose": str(job["purpose"]),
            "passed": False,
            "candidate_stop": bool(job["candidate_stop"]),
            "metrics": {},
            "raw_metrics": {},
            "reason": "coalesced by newer undispatched screen",
        }
        if str(job.get("telemetry_transport") or "legacy_local") != "neon_mailbox_v1":
            store.put_json(
                f"eval-decisions/{int(job['train_job_id'])}/{job['job_key']}.json",
                decision,
                create_only=True,
            )
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_jobs SET decision_json = %(decision)s, updated_at = now()
                    WHERE id = %(id)s AND decision_json IS NULL
                    """,
                    {"decision": json_arg(decision), "id": int(job["id"])},
                )
                published += cur.rowcount
    return published


def _stage_metrics(
    job: Mapping[str, Any], raw_metrics: Mapping[str, Any], passed: bool
) -> dict[str, object]:
    if str(job["purpose"]) == "promotion":
        return evaluation_metric_payload(
            protocol="full",
            metrics=raw_metrics,
            checkpoint_step=int(job["checkpoint_step"]),
            checkpoint_artifact=str(job["checkpoint_uri"]),
            eval_source="modal",
        )
    stage_name = str(job["stage_name"])
    metrics = evaluation_metric_payload(
        protocol=stage_name,
        metrics=raw_metrics,
        checkpoint_step=int(job["checkpoint_step"]),
        checkpoint_artifact=str(job["checkpoint_uri"]),
        eval_source="modal",
    )
    metrics[checkpoint_eval_stage_metric(stage_name, "candidate/pass")] = 1.0 if passed else 0.0
    metrics[checkpoint_eval_stage_metric(stage_name, "candidate/stage_index")] = float(
        job["stage_index"]
    )
    metrics[checkpoint_eval_stage_metric(stage_name, "source")] = "modal"
    validate_metric_payload(metrics)
    return metrics


def _mark_attempt_failure(
    conn, *, attempt: Mapping[str, Any], error: str, config: ModalEvalConfig, terminal: bool = False
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_attempts SET status = 'failed', error = %(error)s,
                  finished_at = now() WHERE id = %(id)s
                """,
                {"id": int(attempt["id"]), "error": error[:4000]},
            )
            cur.execute(
                """
                UPDATE worker_attempts
                SET status = 'failed', error = %(error)s, finished_at = now()
                WHERE attempt_id = %(attempt_id)s
                """,
                {
                    "attempt_id": str(attempt["attempt_id"]),
                    "error": error[:4000],
                },
            )
            cur.execute(
                """
                SELECT count(*) AS count FROM eval_attempts
                WHERE eval_job_id = %(job_id)s AND retry_round = %(retry_round)s
                """,
                {
                    "job_id": int(attempt["eval_job_id"]),
                    "retry_round": int(attempt.get("retry_round") or 0),
                },
            )
            attempts = int(cur.fetchone()["count"])
            retry = not terminal and attempts < config.max_attempts
            cur.execute(
                """
                UPDATE eval_jobs
                SET status = %(status)s, error = %(error)s, updated_at = now(),
                    finished_at = CASE WHEN %(retry)s THEN NULL ELSE now() END
                WHERE id = %(job_id)s
                """,
                {
                    "status": "pending" if retry else "failed",
                    "error": error[:4000],
                    "retry": retry,
                    "job_id": int(attempt["eval_job_id"]),
                },
            )


def _next_stage_or_promotion(conn, job: Mapping[str, Any], passed: bool) -> None:
    if not passed or str(job["purpose"]) == "promotion":
        return
    announcement = job["source_announcement_json"]
    if isinstance(announcement, str):
        announcement = json.loads(announcement)
    next_index = int(job["stage_index"]) + 1
    stages = announcement.get("eval", {}).get("stages", [])
    if next_index < len(stages):
        _insert_eval_job(
            conn, announcement, stage_job_descriptor(announcement, stage_index=next_index)
        )


def accept_attempt_result(
    conn,
    store: ObjectStore,
    *,
    attempt: Mapping[str, Any],
    result: Mapping[str, Any],
    timeout_seconds: int | None = None,
) -> None:
    contract = attempt["contract_json"]
    if isinstance(contract, str):
        contract = json.loads(contract)
    validated = validate_attempt_result(
        result, contract=contract, attempt_id=str(attempt["attempt_id"])
    )
    rules = attempt["decision_rules_json"]
    if isinstance(rules, str):
        rules = json.loads(rules)
    raw_metrics = dict(validated["metrics"])
    if str(attempt["purpose"]) == "promotion":
        raw_metrics.update(
            {
                "checkpoint_step": int(attempt["checkpoint_step"]),
                "checkpoint_artifact": str(attempt["checkpoint_uri"]),
                "_eval_by_start_rows": eval_by_start_rows(
                    [dict(episode) for episode in validated["episode_results"]]
                ),
            }
        )
    passed, observed = apply_decision_rules(raw_metrics, rules) if rules else (True, [])
    metrics = _stage_metrics(attempt, raw_metrics, passed)
    decision = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "job_key": str(attempt["job_key"]),
        "execution_key": str(attempt["execution_key"]),
        "attempt_id": str(attempt["attempt_id"]),
        "train_job_id": int(attempt["train_job_id"]),
        "ledger_id": int(attempt["ledger_id"]),
        "stage_name": str(attempt["stage_name"]),
        "stage_index": int(attempt["stage_index"]),
        "purpose": str(attempt["purpose"]),
        "passed": passed,
        "candidate_stop": bool(attempt["candidate_stop"]),
        "observed_rules": observed,
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "result_uri": str(attempt["result_uri"]),
    }
    preview = validated.get("preview")
    if str(attempt["purpose"]) == "screen" and isinstance(preview, Mapping):
        decision["preview"] = dict(preview)
    train_config = dict(attempt.get("train_config") or {})
    if str(train_config.get("telemetry_transport") or "legacy_local") != ("neon_mailbox_v1"):
        store.put_json(
            f"eval-decisions/{int(attempt['train_job_id'])}/{attempt['job_key']}.json",
            decision,
            create_only=True,
        )
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_attempts SET status = 'succeeded', result_json = %(result)s,
                  actual_cost_usd = CASE
                    WHEN %(timeout)s IS NULL THEN COALESCE(actual_cost_usd, reserved_cost_usd)
                    ELSE reserved_cost_usd * LEAST(
                      1.0,
                      GREATEST(0.0, EXTRACT(EPOCH FROM (now() - created_at))) /
                        GREATEST(1.0, %(timeout)s)
                    )
                  END,
                  finished_at = now(), error = NULL
                WHERE id = %(id)s
                """,
                {
                    "id": int(attempt["id"]),
                    "result": json_arg(validated),
                    "timeout": timeout_seconds,
                },
            )
            cur.execute(
                """
                UPDATE worker_attempts
                SET status = 'succeeded', finished_at = now(), error = NULL
                WHERE attempt_id = %(attempt_id)s
                """,
                {"attempt_id": str(attempt["attempt_id"])},
            )
            cur.execute(
                """
                UPDATE eval_jobs SET status = 'succeeded', accepted_attempt_id = %(attempt_id)s,
                  decision_json = %(decision)s, updated_at = now(), finished_at = now(), error = NULL
                WHERE id = %(job_id)s
                """,
                {
                    "attempt_id": int(attempt["id"]),
                    "decision": json_arg(decision),
                    "job_id": int(attempt["eval_job_id"]),
                },
            )
            if bool(attempt.get("candidate_stop")) and bool(decision.get("passed")):
                cur.execute(
                    """
                    INSERT INTO attempt_commands (
                      command_id, attempt_id, command_type, payload_json
                    )
                    SELECT
                      'candidate-stop:' || %(eval_job_id)s::text,
                      worker.attempt_id,
                      'stop',
                      jsonb_build_object(
                        'eval_job_id', %(eval_job_id)s,
                        'checkpoint_step', %(checkpoint_step)s
                      )
                    FROM worker_attempts worker
                    WHERE worker.train_job_id = %(train_job_id)s
                      AND worker.task_kind = 'train'
                      AND worker.status = 'running'
                    ON CONFLICT (command_id) DO NOTHING
                    """,
                    {
                        "eval_job_id": int(attempt["eval_job_id"]),
                        "checkpoint_step": int(attempt["checkpoint_step"]),
                        "train_job_id": int(attempt["train_job_id"]),
                    },
                )
    if (
        str(train_config.get("telemetry_transport") or "legacy_local") == "neon_mailbox_v1"
        and str(attempt["purpose"]) != "promotion"
    ):
        from rlab.telemetry_mailbox import enqueue_projection_payload

        enqueue_projection_payload(
            conn,
            eval_job_id=int(attempt["eval_job_id"]),
            payload={
                "projection_kind": "evaluation",
                "train_config": train_config,
                "decision": decision,
                "purpose": str(attempt["purpose"]),
                "checkpoint_uri": str(attempt["checkpoint_uri"]),
                "checkpoint_step": int(attempt["checkpoint_step"]),
                "canonical_promotion": False,
            },
        )
    _next_stage_or_promotion(conn, attempt, passed)


def poll_attempts(
    conn,
    store: ObjectStore,
    invoker: ModalInvoker,
    config: ModalEvalConfig,
    *,
    deadline_monotonic: float,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.*, j.train_job_id, j.ledger_id, j.checkpoint_step, j.checkpoint_uri,
              j.stage_name,
              j.stage_index, j.purpose, j.execution_key, j.job_key, j.contract_json,
              j.source_announcement_json, j.decision_rules_json, j.candidate_stop,
              t.train_config
            FROM eval_attempts a JOIN eval_jobs j ON j.id = a.eval_job_id
            JOIN train_jobs t ON t.id = j.train_job_id
            WHERE a.status IN ('dispatching', 'submitted')
            ORDER BY a.created_at
            """
        )
        attempts = [dict(row) for row in cur.fetchall()]
    changed = 0
    now = datetime.now(UTC)
    for attempt in attempts:
        if time.monotonic() >= deadline_monotonic:
            break
        result = store.get_json_optional(str(attempt["result_uri"]))
        if result is not None:
            try:
                receipt = attempt.get("receipt_json")
                if isinstance(receipt, Mapping):
                    if str(receipt.get("result_uri") or "") != str(attempt["result_uri"]):
                        raise ValueError("Modal receipt result URI mismatch")
                    expected_result_sha = str(receipt.get("result_sha256") or "")
                    if (
                        expected_result_sha
                        and hashlib.sha256(store.get_bytes(str(attempt["result_uri"]))).hexdigest()
                        != expected_result_sha
                    ):
                        raise ValueError("Modal receipt result hash mismatch")
                if str(result.get("status") or "") != "succeeded":
                    detail = str(result.get("error") or result.get("status") or "unknown failure")
                    terminal = deterministic_eval_failure(detail)
                    raise RuntimeError(f"{'terminal: ' if terminal else ''}{detail}")
                accept_attempt_result(
                    conn,
                    store,
                    attempt=attempt,
                    result=result,
                    timeout_seconds=config.timeout_for(
                        str(attempt["purpose"]), int(attempt["stage_index"])
                    ),
                )
            except Exception as exc:
                terminal = deterministic_eval_failure(exc)
                _mark_attempt_failure(
                    conn, attempt=attempt, error=repr(exc), config=config, terminal=terminal
                )
            changed += 1
            continue
        expires_at = attempt["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if now >= expires_at:
            if attempt.get("modal_call_id"):
                try:
                    invoker.cancel(str(attempt["modal_call_id"]))
                except Exception:
                    pass
            _mark_attempt_failure(
                conn, attempt=attempt, error="Modal eval attempt expired", config=config
            )
            changed += 1
            continue
        if attempt.get("modal_call_id"):
            state, detail = invoker.poll(str(attempt["modal_call_id"]))
            if state == "failed":
                _mark_attempt_failure(conn, attempt=attempt, error=str(detail), config=config)
                changed += 1
            elif state == "finished":
                if not isinstance(detail, Mapping):
                    _mark_attempt_failure(
                        conn,
                        attempt=attempt,
                        error="Modal eval receipt is not a mapping",
                        config=config,
                    )
                    changed += 1
                elif (
                    str(detail.get("result_uri") or "") != str(attempt["result_uri"])
                    or len(str(detail.get("result_sha256") or "")) != 64
                ):
                    _mark_attempt_failure(
                        conn,
                        attempt=attempt,
                        error="Modal eval receipt identity is invalid",
                        config=config,
                        terminal=True,
                    )
                    changed += 1
                else:
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                "UPDATE eval_attempts SET receipt_json = %(receipt)s WHERE id = %(id)s",
                                {"receipt": json_arg(detail), "id": int(attempt["id"])},
                            )
    return changed


def cancel_requested_attempts(
    conn,
    invoker: ModalInvoker,
    *,
    deadline_monotonic: float,
) -> int:
    """Best-effort cancel Modal calls whose logical train job was canceled."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.id, a.modal_call_id
            FROM eval_attempts a
            JOIN eval_jobs j ON j.id = a.eval_job_id
            JOIN train_jobs t ON t.id = j.train_job_id
            WHERE t.status = 'canceled'
              AND a.status IN ('dispatching', 'submitted')
            ORDER BY a.created_at
            """
        )
        attempts = [dict(row) for row in cur.fetchall()]
    changed = 0
    for attempt in attempts:
        if time.monotonic() >= deadline_monotonic:
            break
        call_id = str(attempt.get("modal_call_id") or "")
        error = None
        if call_id:
            try:
                invoker.cancel(call_id)
            except Exception as exc:
                error = repr(exc)
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_attempts
                    SET status = 'canceled', finished_at = now(), error = %(error)s
                    WHERE id = %(id)s AND status IN ('dispatching', 'submitted')
                    """,
                    {"id": int(attempt["id"]), "error": error},
                )
                changed += cur.rowcount
    return changed


def _reuse_result(conn, store: ObjectStore, job: Mapping[str, Any]) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.*
            FROM eval_attempts a
            JOIN eval_jobs source ON source.id = a.eval_job_id
            WHERE source.execution_key = %(execution_key)s AND a.status = 'succeeded'
            ORDER BY a.finished_at DESC LIMIT 1
            """,
            {
                "execution_key": str(job["execution_key"]),
            },
        )
        source = cur.fetchone()
    if not source:
        return False
    result = source.get("result_json")
    if not isinstance(result, Mapping):
        return False
    synthetic = {**dict(source), **dict(job)}
    synthetic["id"] = int(source["id"])
    synthetic["eval_job_id"] = int(job["id"])
    accept_attempt_result(conn, store, attempt=synthetic, result=result)
    return True


def round_robin_jobs(
    jobs: list[dict[str, Any]], *, slots: int, after_train_job_id: int = 0
) -> list[dict[str, Any]]:
    if slots < 1:
        return []
    promotions = [job for job in jobs if str(job["purpose"]) == "promotion"]
    promotions.sort(
        key=lambda job: (
            0 if int(job["train_job_id"]) > after_train_job_id else 1,
            int(job["train_job_id"]),
            int(job["id"]),
        )
    )
    promotion = promotions[0] if promotions else None
    selected: list[dict[str, Any]] = [promotion] if promotion is not None else []
    remaining = [job for job in jobs if job is not promotion]
    remaining.sort(
        key=lambda job: (
            0 if int(job["stage_index"]) > 0 and str(job["purpose"]) != "promotion" else 1,
            int(job["id"]),
        )
    )
    by_run: dict[int, deque[dict[str, Any]]] = defaultdict(deque)
    for job in remaining:
        by_run[int(job["train_job_id"])].append(job)
    sorted_run_ids = sorted(by_run)
    run_ids = deque(
        [run_id for run_id in sorted_run_ids if run_id > after_train_job_id]
        + [run_id for run_id in sorted_run_ids if run_id <= after_train_job_id]
    )
    while run_ids and len(selected) < max(slots * 4, slots):
        run_id = run_ids.popleft()
        selected.append(by_run[run_id].popleft())
        if by_run[run_id]:
            run_ids.append(run_id)
    return selected


def available_eval_slots(*, effective_capacity: int, active_calls: int, hard_cap: int) -> int:
    return max(0, min(int(effective_capacity), int(hard_cap)) - max(0, int(active_calls)))


def budget_allows(
    *,
    run_cost_usd: float,
    rolling_cost_usd: float,
    reserved_cost_usd: float,
    config: ModalEvalConfig,
) -> bool:
    return (
        run_cost_usd + reserved_cost_usd <= config.per_run_budget_usd
        and rolling_cost_usd + reserved_cost_usd <= config.rolling_24h_budget_usd
    )


def dispatch_pending(
    conn,
    store: ObjectStore,
    invoker: ModalInvoker,
    config: ModalEvalConfig,
    *,
    deadline_monotonic: float,
) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM eval_backend_state WHERE backend = 'modal'")
        state = dict(cur.fetchone())
        if state["drained"]:
            return 0
        effective = min(int(state["effective_capacity"]), config.hard_max_active)
        cur.execute(
            "SELECT count(*) AS count FROM eval_attempts WHERE status IN ('dispatching', 'submitted')"
        )
        active = int(cur.fetchone()["count"])
    slots = available_eval_slots(
        effective_capacity=effective,
        active_calls=active,
        hard_cap=config.hard_max_active,
    )
    if not slots:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.* FROM eval_jobs j
            JOIN train_jobs t ON t.id = j.train_job_id
            WHERE j.status IN ('pending', 'blocked_budget')
              AND (j.purpose <> 'promotion' OR t.status = 'finalizing')
            ORDER BY CASE WHEN stage_index > 0 AND purpose <> 'promotion' THEN 0
                          WHEN purpose = 'promotion' THEN 1 ELSE 2 END,
                     train_job_id, created_at
            LIMIT 200
            """,
        )
        jobs = [dict(row) for row in cur.fetchall()]
    dispatched = 0
    ordered = round_robin_jobs(
        jobs,
        slots=slots,
        after_train_job_id=int(state["round_robin_after_train_job_id"]),
    )
    for job in ordered:
        if dispatched >= slots or time.monotonic() >= deadline_monotonic:
            break
        if _reuse_result(conn, store, job):
            continue
        timeout = config.timeout_for(str(job["purpose"]), int(job["stage_index"]))
        reserved = config.reserved_cost(timeout)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(sum(CASE
                  WHEN a.status IN ('dispatching', 'submitted') THEN reserved_cost_usd
                  ELSE COALESCE(actual_cost_usd, reserved_cost_usd)
                END), 0) AS total
                FROM eval_attempts a JOIN eval_jobs j ON j.id = a.eval_job_id
                WHERE j.train_job_id = %(job_id)s
                """,
                {"job_id": int(job["train_job_id"])},
            )
            run_cost = float(cur.fetchone()["total"])
            cur.execute(
                """
                SELECT COALESCE(sum(CASE
                  WHEN status IN ('dispatching', 'submitted') THEN reserved_cost_usd
                  ELSE COALESCE(actual_cost_usd, reserved_cost_usd)
                END), 0) AS total FROM eval_attempts
                WHERE created_at >= now() - interval '24 hours'
                """
            )
            rolling_cost = float(cur.fetchone()["total"])
        if not budget_allows(
            run_cost_usd=run_cost,
            rolling_cost_usd=rolling_cost,
            reserved_cost_usd=reserved,
            config=config,
        ):
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE eval_jobs SET status = 'blocked_budget', updated_at = now() WHERE id = %(id)s",
                        {"id": int(job["id"])},
                    )
            continue
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) AS count FROM eval_attempts
                WHERE eval_job_id = %(job_id)s AND retry_round = %(retry_round)s
                """,
                {
                    "job_id": int(job["id"]),
                    "retry_round": int(job.get("retry_round") or 0),
                },
            )
            attempt_number = int(cur.fetchone()["count"]) + 1
        if attempt_number > config.max_attempts:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE eval_jobs SET status = 'failed', error = 'attempt limit exhausted', finished_at = now() WHERE id = %(id)s",
                        {"id": int(job["id"])},
                    )
            continue
        attempt_id = uuid.uuid4().hex
        app_name = modal_app_name(
            config.app_name_prefix, str(job["contract_json"]["runtime_image_ref"])
        )
        result_uri = store.uri(f"eval-attempts/{job['execution_key']}/{attempt_id}.json")
        expires_at = datetime.now(UTC) + timedelta(
            seconds=config.startup_timeout_seconds + timeout + config.expiry_margin_seconds
        )
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eval_attempts (
                      attempt_id, eval_job_id, attempt_number, retry_round, modal_app_name,
                      modal_function_name, result_uri, reserved_cost_usd, expires_at
                    ) VALUES (
                      %(attempt_id)s, %(job_id)s, %(number)s, %(retry_round)s,
                      %(app)s, %(function)s,
                      %(result_uri)s, %(reserved)s, %(expires_at)s
                    ) RETURNING id
                    """,
                    {
                        "attempt_id": attempt_id,
                        "job_id": int(job["id"]),
                        "number": attempt_number,
                        "retry_round": int(job.get("retry_round") or 0),
                        "app": app_name,
                        "function": config.function_name,
                        "result_uri": result_uri,
                        "reserved": reserved,
                        "expires_at": expires_at,
                    },
                )
                attempt_row_id = int(cur.fetchone()["id"])
                cur.execute(
                    """
                    INSERT INTO worker_attempts (
                      attempt_id, train_job_id, eval_job_id, task_kind, provider,
                      status, started_at
                    ) VALUES (
                      %(attempt_id)s, %(train_job_id)s, %(job_id)s, 'eval', 'modal',
                      'launching', now()
                    )
                    ON CONFLICT (attempt_id) DO NOTHING
                    """,
                    {
                        "attempt_id": attempt_id,
                        "train_job_id": int(job["train_job_id"]),
                        "job_id": int(job["id"]),
                    },
                )
                cur.execute(
                    "UPDATE eval_jobs SET status = 'dispatching', updated_at = now(), error = NULL WHERE id = %(id)s",
                    {"id": int(job["id"])},
                )
        seconds_to_expiry = max(1, int((expires_at - datetime.now(UTC)).total_seconds()))
        asset = job["source_announcement_json"]["eval"].get("asset")
        payload = {
            "attempt_id": attempt_id,
            "contract": job["contract_json"],
            "expires_at": expires_at.timestamp(),
            "child_timeout_seconds": max(1, timeout - config.child_margin_seconds),
            "model_get_url": store.presign_get(
                str(job["checkpoint_uri"]), expires_seconds=seconds_to_expiry
            ),
            "metadata_get_url": store.presign_get(
                str(job["metadata_uri"]), expires_seconds=seconds_to_expiry
            ),
            "metadata_sha256": str(job["source_announcement_json"]["metadata_sha256"]),
            "result_uri": result_uri,
            "result_put_url": store.presign_put(result_uri, expires_seconds=seconds_to_expiry),
        }
        if isinstance(asset, Mapping):
            payload["rom_get_url"] = store.presign_get(
                str(asset["object_uri"]), expires_seconds=seconds_to_expiry
            )
        if config.preview_enabled and str(job["purpose"]) == "screen":
            try:
                preview_store = ObjectStore(preview_storage_base_uri())
                preview_key = (
                    f"eval-previews/{int(job['train_job_id'])}/"
                    f"{job['checkpoint_sha256']}/{attempt_id}.mp4"
                )
                cache_control = "public, max-age=31536000, immutable"
                payload["preview"] = {
                    "object_uri": preview_store.uri(preview_key),
                    "put_url": preview_store.presign_put(
                        preview_key,
                        expires_seconds=seconds_to_expiry,
                        content_type="video/mp4",
                        cache_control=cache_control,
                    ),
                    "public_url": (
                        f"{preview_public_base_url()}/{quote(preview_key, safe='/._-')}"
                    ),
                    "content_type": "video/mp4",
                    "cache_control": cache_control,
                    "max_frames": config.preview_max_frames,
                    "fps": config.preview_fps,
                    "max_lanes": config.preview_max_lanes,
                    "scale": config.preview_scale,
                    "max_bytes": config.preview_max_bytes,
                    "encode_timeout_seconds": config.preview_encode_timeout_seconds,
                    "upload_timeout_seconds": config.preview_upload_timeout_seconds,
                }
            except Exception as exc:
                print(f"Modal eval preview disabled for attempt {attempt_id}: {exc}", flush=True)
        try:
            call_id = invoker.spawn(app_name, config.function_name, payload)
        except Exception as exc:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT a.*, j.train_job_id FROM eval_attempts a
                    JOIN eval_jobs j ON j.id = a.eval_job_id WHERE a.id = %(id)s
                    """,
                    {"id": attempt_row_id},
                )
                attempt = dict(cur.fetchone())
            _mark_attempt_failure(conn, attempt=attempt, error=repr(exc), config=config)
            continue
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_attempts SET status = 'submitted', modal_call_id = %(call_id)s,
                      started_at = now() WHERE id = %(id)s
                    """,
                    {"call_id": call_id, "id": attempt_row_id},
                )
                cur.execute(
                    """
                    UPDATE worker_attempts
                    SET status = 'running', provider_run_id = %(call_id)s,
                        started_at = COALESCE(started_at, now()), last_heartbeat_at = now()
                    WHERE attempt_id = %(attempt_id)s
                    """,
                    {"attempt_id": attempt_id, "call_id": call_id},
                )
                cur.execute(
                    "UPDATE eval_jobs SET status = 'submitted', updated_at = now() WHERE id = %(id)s",
                    {"id": int(job["id"])},
                )
                cur.execute(
                    "UPDATE eval_runs SET last_scheduled_at = now(), updated_at = now() WHERE train_job_id = %(id)s",
                    {"id": int(job["train_job_id"])},
                )
                cur.execute(
                    """
                    UPDATE eval_backend_state SET round_robin_after_train_job_id = %(job_id)s,
                      updated_at = now() WHERE backend = 'modal'
                    """,
                    {"job_id": int(job["train_job_id"])},
                )
        dispatched += 1
    return dispatched


def enqueue_post_train_promotions(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (j.train_job_id, j.ledger_id) j.* FROM eval_jobs j
            JOIN eval_runs r ON r.train_job_id = j.train_job_id
            JOIN train_jobs t ON t.id = j.train_job_id
            WHERE j.status = 'succeeded' AND j.purpose <> 'promotion'
              AND j.stage_index = jsonb_array_length(
                j.source_announcement_json->'eval'->'stages'
              ) - 1
              AND (j.decision_json->>'passed')::boolean = TRUE
              AND r.complete_announcement_seen = TRUE
              AND t.status = 'finalizing'
            ORDER BY j.train_job_id, j.ledger_id, j.checkpoint_step
            """
        )
        candidates = [dict(row) for row in cur.fetchall()]
    created = 0
    for candidate in candidates:
        announcement = candidate["source_announcement_json"]
        with conn:
            created += int(
                _insert_eval_job(conn, announcement, promotion_job_descriptor(announcement))
                is not None
            )
    return created


def finalize_runs(conn) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH completed AS (
                  UPDATE eval_runs r
                  SET status = 'complete', updated_at = now(), error = NULL
                WHERE r.complete_announcement_seen = TRUE
                  AND r.artifacts_projected_at IS NOT NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM eval_jobs promoted
                    WHERE promoted.train_job_id = r.train_job_id
                      AND promoted.purpose = 'promotion'
                      AND promoted.status = 'succeeded'
                      AND promoted.projected_at IS NULL
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM eval_jobs unprojected
                    WHERE unprojected.train_job_id = r.train_job_id
                      AND unprojected.status = 'succeeded'
                      AND unprojected.projected_at IS NULL
                  )
                  AND (
                    NOT EXISTS (
                      SELECT 1 FROM eval_jobs candidate
                      WHERE candidate.train_job_id = r.train_job_id
                        AND candidate.purpose = 'promotion'
                        AND candidate.status = 'succeeded'
                    )
                    OR r.promoted_eval_job_id IS NOT NULL
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM eval_jobs j WHERE j.train_job_id = r.train_job_id
                      AND j.status IN (
                        'pending', 'dispatching', 'submitted', 'blocked_budget', 'failed'
                      )
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_batches b
                    JOIN metric_streams s ON s.stream_id = b.stream_id
                    JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                    WHERE a.train_job_id = r.train_job_id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_streams s
                    JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                    WHERE a.train_job_id = r.train_job_id
                      AND (s.final_sequence IS NULL
                           OR s.published_sequence < s.final_sequence)
                  )
                  AND r.status = 'finalizing'
                  AND EXISTS (
                    SELECT 1 FROM train_jobs ready
                    WHERE ready.id = r.train_job_id
                      AND ready.status = 'finalizing'
                      AND ready.live_publication_status IN ('complete', 'disabled')
                  )
                  RETURNING r.train_job_id
                )
                UPDATE train_jobs t
                SET status = 'succeeded', finished_at = now(), error = NULL,
                  live_publication_next_retry_at = NULL,
                  live_publication_error = NULL
                FROM completed
                WHERE t.id = completed.train_job_id
                  AND t.status = 'finalizing'
                  AND t.live_publication_status IN ('complete', 'disabled')
                """
            )
            return cur.rowcount


def reconcile_eval_run_failures(conn) -> int:
    """Promote exhausted required eval jobs to a run-level finalization failure."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_runs r
                SET status = 'failed', updated_at = now(),
                  error = COALESCE(
                    (SELECT j.error FROM eval_jobs j
                     WHERE j.train_job_id = r.train_job_id AND j.status = 'failed'
                     ORDER BY j.updated_at DESC LIMIT 1),
                    'required evaluation exhausted retries'
                  )
                WHERE r.status IN ('active', 'awaiting_artifact_recovery', 'finalizing')
                  AND EXISTS (
                    SELECT 1 FROM eval_jobs j
                    WHERE j.train_job_id = r.train_job_id AND j.status = 'failed'
                  )
                """
            )
            return cur.rowcount


def reconcile_train_finalization_failures(conn) -> int:
    """Expose exhausted post-train work without changing the successful launch."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs t
                SET status = 'finalization_failed', finished_at = now(),
                  error = COALESCE(r.error, t.live_publication_error, 'finalization failed')
                FROM eval_runs r
                WHERE r.train_job_id = t.id
                  AND t.status = 'finalizing'
                  AND (
                    r.status = 'failed'
                    OR t.live_publication_status = 'failed'
                  )
                """
            )
            changed = cur.rowcount
            cur.execute(
                """
                UPDATE train_jobs
                SET status = 'finalization_failed', finished_at = now(),
                  error = COALESCE(live_publication_error, 'live publication failed')
                WHERE status = 'finalizing'
                  AND live_publication_status = 'failed'
                """
            )
            return changed + cur.rowcount


def promotion_candidate_key(job: Mapping[str, Any]) -> tuple[Any, int, int]:
    decision = job["decision_json"]
    raw_metrics = dict(decision["raw_metrics"])
    selection_rank = job["train_config"].get("selection_rank") or ()
    criteria = parse_persisted_objective_rank(selection_rank)
    if not criteria:
        raise ValueError("persisted objective.rank contains unsupported metric criteria")
    return (
        rank_score(raw_metrics, criteria),
        -int(job["checkpoint_step"]),
        -int(job["id"]),
    )


def reconcile_promotions(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.*, t.train_config
            FROM eval_jobs j JOIN train_jobs t ON t.id = j.train_job_id
            JOIN eval_runs r ON r.train_job_id = j.train_job_id
            WHERE j.purpose = 'promotion' AND j.status = 'succeeded'
              AND r.status = 'finalizing'
            ORDER BY j.train_job_id, j.id
            """
        )
        jobs = [dict(row) for row in cur.fetchall()]
    by_run: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        by_run[int(job["train_job_id"])].append(job)
    updated = 0
    for train_job_id, candidates in by_run.items():
        try:
            winner = max(candidates, key=promotion_candidate_key)
        except (KeyError, TypeError, ValueError) as exc:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE eval_runs SET status = 'failed', error = %(error)s,
                          updated_at = now() WHERE train_job_id = %(train_job_id)s
                        """,
                        {
                            "error": f"promotion reconciliation failed: {exc}"[:4000],
                            "train_job_id": train_job_id,
                        },
                    )
            continue
        promotion = {
            "eval_job_id": int(winner["id"]),
            "accepted_attempt_id": int(winner["accepted_attempt_id"]),
            "checkpoint_sha256": str(winner["checkpoint_sha256"]),
            "checkpoint_step": int(winner["checkpoint_step"]),
            "checkpoint_uri": str(winner["checkpoint_uri"]),
            "result_uri": str(winner["decision_json"]["result_uri"]),
            "raw_metrics": dict(winner["decision_json"]["raw_metrics"]),
        }
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_runs SET promoted_eval_job_id = %(job_id)s,
                      promotion_json = %(promotion)s, updated_at = now()
                    WHERE train_job_id = %(train_job_id)s
                      AND promoted_eval_job_id IS DISTINCT FROM %(job_id)s
                    """,
                    {
                        "job_id": int(winner["id"]),
                        "promotion": json_arg(promotion),
                        "train_job_id": train_job_id,
                    },
                )
                updated += cur.rowcount
    return updated


def _execute_projection(
    payload: Mapping[str, Any], *, repo_root: Path, deadline_monotonic: float, label: str
) -> str | None:
    if deadline_monotonic - time.monotonic() < 2.0:
        return "projection deadline is exhausted"
    projection_dir = repo_root / "logs" / "fleet" / "modal-eval-projections"
    projection_dir.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{label}-", suffix=".json", dir=projection_dir)
    payload_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, default=str)
            handle.write("\n")
        timeout = max(1.0, min(60.0, deadline_monotonic - time.monotonic()))
        completed = subprocess.run(
            [sys.executable, "-m", "rlab.modal_eval_projection", "--payload", str(payload_path)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if completed.returncode:
            return (completed.stderr or completed.stdout or "projection failed")[-4000:]
        return None
    except subprocess.TimeoutExpired:
        return "projection timed out"
    finally:
        payload_path.unlink(missing_ok=True)


def project_eval_results(conn, *, repo_root: Path, deadline_monotonic: float) -> int:
    if deadline_monotonic - time.monotonic() < 2.0:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.*, t.train_config,
              (r.promoted_eval_job_id = j.id) AS canonical_promotion
            FROM eval_jobs j JOIN train_jobs t ON t.id = j.train_job_id
            JOIN eval_runs r ON r.train_job_id = j.train_job_id
            WHERE j.status = 'succeeded' AND j.projected_at IS NULL
              AND j.projection_enqueued_at IS NULL
              AND j.purpose = 'promotion'
              AND t.status = 'finalizing'
              AND t.live_publication_status IN ('complete', 'disabled')
              AND r.status = 'finalizing'
              AND j.projection_attempts < %(max_attempts)s
              AND (j.projection_next_retry_at IS NULL OR j.projection_next_retry_at <= now())
            ORDER BY COALESCE(j.projection_next_retry_at, j.created_at),
              j.train_job_id, j.checkpoint_step, j.stage_index
            LIMIT 1
            """,
            {"max_attempts": MAX_PROJECTION_ATTEMPTS},
        )
        job = cur.fetchone()
    if not job:
        return 0
    job = dict(job)
    payload = {
        "projection_kind": "evaluation",
        "train_config": job["train_config"],
        "decision": job["decision_json"],
        "purpose": job["purpose"],
        "checkpoint_uri": job["checkpoint_uri"],
        "checkpoint_sha256": job["checkpoint_sha256"],
        "checkpoint_step": job["checkpoint_step"],
        "canonical_promotion": bool(job["canonical_promotion"]),
    }
    if str(job["train_config"].get("telemetry_transport") or "legacy_local") == ("neon_mailbox_v1"):
        from rlab.telemetry_mailbox import enqueue_projection_payload

        enqueue_projection_payload(conn, eval_job_id=int(job["id"]), payload=payload)
        return 1
    error = _execute_projection(
        payload,
        repo_root=repo_root,
        deadline_monotonic=deadline_monotonic,
        label=f"eval-job-{job['id']}",
    )
    with conn:
        with conn.cursor() as cur:
            if error is None:
                cur.execute(
                    """
                    UPDATE eval_jobs SET projected_at = now(), projection_error = NULL,
                      projection_next_retry_at = NULL, updated_at = now()
                    WHERE id = %(id)s
                    """,
                    {"id": int(job["id"])},
                )
            else:
                attempts = int(job.get("projection_attempts") or 0) + 1
                retry_delay = PROJECTION_RETRY_DELAYS_SECONDS[
                    min(attempts - 1, len(PROJECTION_RETRY_DELAYS_SECONDS) - 1)
                ]
                cur.execute(
                    """
                    UPDATE eval_jobs SET projection_error = %(error)s,
                      projection_attempts = %(attempts)s,
                      projection_next_retry_at = now() + (%(retry_delay)s * interval '1 second'),
                      updated_at = now() WHERE id = %(id)s
                    """,
                    {
                        "error": error[:4000],
                        "attempts": attempts,
                        "retry_delay": retry_delay,
                        "id": int(job["id"]),
                    },
                )
                if attempts >= MAX_PROJECTION_ATTEMPTS:
                    cur.execute(
                        """
                        UPDATE eval_runs SET status = 'failed', error = %(error)s,
                          updated_at = now() WHERE train_job_id = %(train_job_id)s
                        """,
                        {
                            "error": f"evaluation projection exhausted retries: {error}"[:4000],
                            "train_job_id": int(job["train_job_id"]),
                        },
                    )
    return int(error is None)


def enqueue_missing_mailbox_projections(conn, *, limit: int = 100) -> int:
    """Recover a crash after accepting an eval but before creating its W&B frame."""

    from rlab.telemetry_mailbox import enqueue_projection_payload

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.*, t.train_config
            FROM eval_jobs j
            JOIN train_jobs t ON t.id = j.train_job_id
            WHERE j.status = 'succeeded'
              AND j.purpose <> 'promotion'
              AND j.projected_at IS NULL
              AND j.projection_enqueued_at IS NULL
              AND t.telemetry_transport = 'neon_mailbox_v1'
            ORDER BY j.id
            LIMIT %(limit)s
            """,
            {"limit": max(1, int(limit))},
        )
        jobs = [dict(row) for row in cur.fetchall()]
    enqueued = 0
    for job in jobs:
        enqueued += int(
            enqueue_projection_payload(
                conn,
                eval_job_id=int(job["id"]),
                payload={
                    "projection_kind": "evaluation",
                    "train_config": job["train_config"],
                    "decision": job["decision_json"],
                    "purpose": str(job["purpose"]),
                    "checkpoint_uri": str(job["checkpoint_uri"]),
                    "checkpoint_step": int(job["checkpoint_step"]),
                    "canonical_promotion": False,
                },
            )
        )
    return enqueued


def reconcile_published_mailbox_projections(conn) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_jobs j
                SET projected_at = now(), projection_error = NULL,
                    projection_next_retry_at = NULL, updated_at = now()
                FROM metric_streams s
                WHERE s.stream_id = 'eval-projection-' || j.id::text
                  AND j.projection_enqueued_at IS NOT NULL
                  AND j.projected_at IS NULL
                  AND s.final_sequence IS NOT NULL
                  AND s.published_sequence >= s.final_sequence
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_batches b WHERE b.stream_id = s.stream_id
                  )
                """
            )
            return cur.rowcount


def project_artifact_references(
    conn,
    store: ObjectStore,
    *,
    repo_root: Path,
    deadline_monotonic: float,
) -> int:
    if deadline_monotonic - time.monotonic() < 2.0:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.*, t.train_config,
              promoted.ledger_id AS promoted_ledger_id,
              promoted.source_announcement_json AS promoted_announcement
            FROM eval_runs r JOIN train_jobs t ON t.id = r.train_job_id
            LEFT JOIN eval_jobs promoted ON promoted.id = r.promoted_eval_job_id
            WHERE r.complete_announcement_seen = TRUE AND r.artifacts_projected_at IS NULL
              AND t.status = 'finalizing'
              AND t.live_publication_status IN ('complete', 'disabled')
              AND r.status = 'finalizing'
              AND r.artifact_projection_attempts < %(max_attempts)s
              AND (
                r.artifact_projection_next_retry_at IS NULL
                OR r.artifact_projection_next_retry_at <= now()
              )
            ORDER BY COALESCE(r.artifact_projection_next_retry_at, r.created_at),
              r.train_job_id LIMIT 1
            """,
            {"max_attempts": MAX_PROJECTION_ATTEMPTS},
        )
        run = cur.fetchone()
    if not run:
        return 0
    run = dict(run)
    train_job_id = int(run["train_job_id"])
    mailbox_transport = (
        str(run["train_config"].get("telemetry_transport") or "legacy_local") == "neon_mailbox_v1"
    )
    ordinal = int(run["next_artifact_projection_id"])
    promoted_ledger_id = (
        int(run["promoted_ledger_id"]) if run.get("promoted_ledger_id") is not None else None
    )
    if promoted_ledger_id is not None and run.get("promoted_artifact_projected_at") is None:
        if ordinal > promoted_ledger_id:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE eval_runs SET promoted_artifact_projected_at = now(),
                          updated_at = now() WHERE train_job_id = %(id)s
                        """,
                        {"id": train_job_id},
                    )
            return 0
        promoted_announcement = dict(run.get("promoted_announcement") or {})
        error = None
        try:
            model_metadata = store.get_json(str(promoted_announcement["metadata_uri"]))
            if not isinstance(model_metadata.get("training_metadata"), dict):
                raise ValueError("checkpoint metadata is missing training_metadata")
        except Exception as exc:
            error = repr(exc)
        else:
            checkpoint_step = int(promoted_announcement["step"])
            payload = {
                "projection_kind": "artifact_reference",
                "train_config": run["train_config"],
                "artifact_kind": promoted_announcement["kind"],
                "checkpoint_uri": promoted_announcement["model_uri"],
                "metadata_uri": promoted_announcement["metadata_uri"],
                "checkpoint_sha256": promoted_announcement["sha256"],
                "metadata_sha256": promoted_announcement["metadata_sha256"],
                "checkpoint_step": checkpoint_step,
                "model_metadata": model_metadata,
                "artifact_aliases": [
                    "latest",
                    "promoted",
                    f"step-{checkpoint_step}",
                ],
            }
            error = _execute_projection(
                payload,
                repo_root=repo_root,
                deadline_monotonic=deadline_monotonic,
                label=f"promoted-artifact-{train_job_id}-{promoted_ledger_id}",
            )
        with conn:
            with conn.cursor() as cur:
                if error is None:
                    cur.execute(
                        """
                        UPDATE eval_runs SET promoted_artifact_projected_at = now(),
                          artifact_projection_attempts = 0,
                          artifact_projection_next_retry_at = NULL,
                          error = NULL, updated_at = now() WHERE train_job_id = %(id)s
                        """,
                        {"id": train_job_id},
                    )
                else:
                    attempts = int(run.get("artifact_projection_attempts") or 0) + 1
                    retry_delay = PROJECTION_RETRY_DELAYS_SECONDS[
                        min(attempts - 1, len(PROJECTION_RETRY_DELAYS_SECONDS) - 1)
                    ]
                    cur.execute(
                        """
                        UPDATE eval_runs SET error = %(error)s,
                          artifact_projection_attempts = %(attempts)s,
                          artifact_projection_next_retry_at =
                            now() + (%(retry_delay)s * interval '1 second'),
                          status = CASE WHEN %(attempts)s >= %(max_attempts)s
                            THEN 'failed' ELSE status END,
                          updated_at = now() WHERE train_job_id = %(id)s
                        """,
                        {
                            "error": (
                                f"promoted artifact projection exhausted retries: {error}"
                                if attempts >= MAX_PROJECTION_ATTEMPTS
                                else error
                            )[:4000],
                            "attempts": attempts,
                            "max_attempts": MAX_PROJECTION_ATTEMPTS,
                            "retry_delay": retry_delay,
                            "id": train_job_id,
                        },
                    )
        return int(error is None)
    if mailbox_transport:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_runs
                    SET next_artifact_projection_id = next_announcement_id,
                        artifacts_projected_at = now(),
                        artifact_projection_attempts = 0,
                        artifact_projection_next_retry_at = NULL,
                        error = NULL,
                        updated_at = now()
                    WHERE train_job_id = %(train_job_id)s
                    """,
                    {"train_job_id": train_job_id},
                )
        return 1
    complete = store.get_json_optional(f"artifact-announcements/{train_job_id}/complete.json")
    if complete is None:
        return 0
    last_ordinal = int(complete.get("last_ledger_id") or 0)
    if ordinal > last_ordinal:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_runs SET artifacts_projected_at = now(),
                      artifact_projection_attempts = 0,
                      artifact_projection_next_retry_at = NULL, updated_at = now()
                    WHERE train_job_id = %(id)s
                    """,
                    {"id": train_job_id},
                )
        return 0
    announcement = store.get_json_optional(
        f"artifact-announcements/{train_job_id}/{ordinal:08d}.json"
    )
    if announcement is None:
        return 0
    if (
        promoted_ledger_id is not None
        and ordinal == promoted_ledger_id
        and run.get("promoted_artifact_projected_at") is not None
    ):
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_runs SET next_artifact_projection_id =
                      next_artifact_projection_id + 1, updated_at = now()
                    WHERE train_job_id = %(id)s
                    """,
                    {"id": train_job_id},
                )
        return 0
    error = None
    if str(announcement.get("kind")) != "tombstone":
        try:
            model_metadata = store.get_json(str(announcement["metadata_uri"]))
            if not isinstance(model_metadata.get("training_metadata"), dict):
                raise ValueError("checkpoint metadata is missing training_metadata")
        except Exception as exc:
            error = repr(exc)
        else:
            payload = {
                "projection_kind": "artifact_reference",
                "train_config": run["train_config"],
                "artifact_kind": announcement["kind"],
                "checkpoint_uri": announcement["model_uri"],
                "metadata_uri": announcement["metadata_uri"],
                "checkpoint_sha256": announcement["sha256"],
                "metadata_sha256": announcement["metadata_sha256"],
                "checkpoint_step": announcement["step"],
                "model_metadata": model_metadata,
            }
            if promoted_ledger_id is not None and str(announcement["kind"]) == "checkpoint":
                payload["artifact_aliases"] = [f"step-{int(announcement['step'])}"]
            error = _execute_projection(
                payload,
                repo_root=repo_root,
                deadline_monotonic=deadline_monotonic,
                label=f"artifact-{train_job_id}-{ordinal}",
            )
    with conn:
        with conn.cursor() as cur:
            if error is None:
                cur.execute(
                    """
                    UPDATE eval_runs SET next_artifact_projection_id = next_artifact_projection_id + 1,
                      artifact_projection_attempts = 0,
                      artifact_projection_next_retry_at = NULL,
                      error = NULL, updated_at = now() WHERE train_job_id = %(id)s
                    """,
                    {"id": train_job_id},
                )
            else:
                attempts = int(run.get("artifact_projection_attempts") or 0) + 1
                retry_delay = PROJECTION_RETRY_DELAYS_SECONDS[
                    min(attempts - 1, len(PROJECTION_RETRY_DELAYS_SECONDS) - 1)
                ]
                cur.execute(
                    """
                    UPDATE eval_runs SET error = %(error)s,
                      artifact_projection_attempts = %(attempts)s,
                      artifact_projection_next_retry_at =
                        now() + (%(retry_delay)s * interval '1 second'),
                      status = CASE WHEN %(attempts)s >= %(max_attempts)s
                        THEN 'failed' ELSE status END,
                      updated_at = now() WHERE train_job_id = %(id)s
                    """,
                    {
                        "error": (
                            f"artifact projection exhausted retries: {error}"
                            if attempts >= MAX_PROJECTION_ATTEMPTS
                            else error
                        )[:4000],
                        "attempts": attempts,
                        "max_attempts": MAX_PROJECTION_ATTEMPTS,
                        "retry_delay": retry_delay,
                        "id": train_job_id,
                    },
                )
    return int(error is None)


def run_service_eval_pass(
    *,
    repo_root: Path,
    deadline_monotonic: float,
    invoker: ModalInvoker | None = None,
    store: ObjectStore | None = None,
    app_client: ModalAppClient | None = None,
    progress: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    config = load_modal_eval_config(repo_root / "experiments" / "modal_eval.yaml")
    if not config.enabled:
        return {"status": "disabled", "hard_cap": config.hard_max_active}
    conn = connect(database_url())
    invoker = invoker or DefaultModalInvoker()
    store = store or ObjectStore(object_store_base_uri())
    try:
        if progress:
            progress("CHECKING EVALUATION", "Acquiring the Modal evaluation scheduler lock")
        if not _try_lock(conn):
            return {"status": "locked"}
        if progress:
            progress("INGESTING EVALUATION", "Reading checkpoint announcements and mailbox events")
        created_runs = ensure_eval_runs(conn)
        canceled_attempts = cancel_requested_attempts(
            conn, invoker, deadline_monotonic=deadline_monotonic
        )
        mailbox_ingested = ingest_mailbox_announcements(conn, store)
        ingested = ingest_announcements(conn, store, deadline_monotonic=deadline_monotonic)
        skipped_decisions = publish_skipped_decisions(conn, store)
        if progress:
            progress("POLLING EVALUATION", "Observing submitted Modal attempts and durable results")
        polled = poll_attempts(
            conn,
            store,
            invoker,
            config,
            deadline_monotonic=deadline_monotonic,
        )
        recovered_projections = enqueue_missing_mailbox_projections(conn)
        promotions = enqueue_post_train_promotions(conn)
        if progress:
            progress("DISPATCHING EVALUATION", "Filling available evaluation capacity")
        dispatched = dispatch_pending(
            conn,
            store,
            invoker,
            config,
            deadline_monotonic=deadline_monotonic,
        )
        reconciled_promotions = reconcile_promotions(conn)
        if progress:
            progress("PROJECTING RESULTS", "Publishing evaluation decisions and promoted artifacts")
        projected = project_eval_results(
            conn, repo_root=repo_root, deadline_monotonic=deadline_monotonic
        )
        projected_artifacts = project_artifact_references(
            conn,
            store,
            repo_root=repo_root,
            deadline_monotonic=deadline_monotonic,
        )
        reconciled_projections = reconcile_published_mailbox_projections(conn)
        eval_run_failures = reconcile_eval_run_failures(conn)
        finalized = finalize_runs(conn)
        finalization_failures = reconcile_train_finalization_failures(conn)
        try:
            if progress:
                progress("REMOVING STALE APPS", "Cleaning unused owned Modal deployments")
            app_cleanup = run_modal_app_cleanup(
                conn,
                config,
                repo_root=repo_root,
                deadline_monotonic=deadline_monotonic,
                client=app_client,
            )
        except Exception as exc:
            app_cleanup = {
                "status": "error",
                "owned_deployed": 0,
                "protected": 0,
                "candidates": 0,
                "stopped": 0,
                "stopped_apps": [],
                "errors": [type(exc).__name__],
            }
        return {
            "status": "ok",
            "created_runs": created_runs,
            "canceled_attempts": canceled_attempts,
            "ingested": ingested,
            "mailbox_ingested": mailbox_ingested,
            "skipped_decisions": skipped_decisions,
            "polled": polled,
            "recovered_projections": recovered_projections,
            "promotions": promotions,
            "reconciled_promotions": reconciled_promotions,
            "dispatched": dispatched,
            "projected": projected,
            "projected_artifacts": projected_artifacts,
            "reconciled_projections": reconciled_projections,
            "eval_run_failures": eval_run_failures,
            "finalized": finalized,
            "finalization_failures": finalization_failures,
            "app_cleanup": app_cleanup,
        }
    finally:
        try:
            _unlock(conn)
        except Exception:
            pass
        conn.close()
