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

from psycopg2 import Error as DatabaseError

from rlab.job_queue import (
    acquire_fleet_admission_xact_lock,
    connect,
    database_url,
    json_arg,
)
from rlab.checkpoint_eval_worker import evaluation_metric_payload
from rlab.eval_metrics import eval_by_start_rows
from rlab.metric_names import (
    EVAL_ACCEPTANCE_DURATION_SECONDS,
    EVAL_ACCEPTANCE_EPISODES_COMPLETED,
    EVAL_ACCEPTANCE_EPISODES_PLANNED,
    EVAL_ACCEPTANCE_FAILURE_COUNT,
    EVAL_ACCEPTANCE_PASS,
    EVAL_FULL_DURATION_SECONDS,
    GLOBAL_STEP,
    checkpoint_eval_stage_metric,
    validate_metric_payload,
)
from rlab.ranking import parse_persisted_objective_rank, rank_score
from rlab.modal_eval_config import ModalEvalConfig, load_modal_eval_config, modal_app_name
from rlab.modal_app_cleanup import ModalAppClient, run_modal_app_cleanup
from rlab.modal_eval_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    acceptance_job_descriptor,
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
)
from rlab.policy_bundle import (
    evaluation_contract_sha256,
    model_document_as_metadata,
    playback_contract_sha256,
    validate_recipe_document,
)


EVAL_RECONCILE_LOCK = "rlab-fleet-reconciler:eval:modal-cpu"
ACTIVE_ATTEMPT_STATES = ("dispatching", "submitted")
MAX_PROJECTION_ATTEMPTS = 3
PROJECTION_RETRY_DELAYS_SECONDS = (30, 120, 300)
CANCEL_ATTEMPT_BUDGET_SECONDS = 2.0
ATTEMPT_POLL_BUDGET_SECONDS = 5.0
ANNOUNCEMENT_INGEST_BUDGET_SECONDS = 5.0


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


def _verify_checkpoint_artifacts(
    store: ObjectStore,
    announcement: Mapping[str, Any],
    *,
    queued_recipe_document: Mapping[str, Any] | None = None,
) -> None:
    versioned_bundle = "model_document_sha256" in announcement
    artifacts = [
        (str(announcement["model_uri"]), str(announcement["sha256"])),
        (
            str(announcement["metadata_uri"]),
            str(announcement["model_document_sha256" if versioned_bundle else "metadata_sha256"]),
        ),
    ]
    if versioned_bundle:
        artifacts.append((str(announcement["recipe_uri"]), str(announcement["recipe_sha256"])))
    eval_contract = announcement.get("eval")
    asset = eval_contract.get("asset") if isinstance(eval_contract, Mapping) else None
    if isinstance(asset, Mapping):
        artifacts.append((str(asset["object_uri"]), str(asset["sha256"])))

    for uri, expected_sha in artifacts:
        head = store.head(uri)
        if int(head["size"]) < 1:
            raise ValueError("checkpoint artifact is empty")
        remote_sha = str(head.get("metadata", {}).get("sha256") or "")
        if store.scheme == "s3" and remote_sha != expected_sha:
            raise ValueError("checkpoint artifact hash metadata mismatch")

    model_document = store.get_json(str(announcement["metadata_uri"]))
    metadata = model_document_as_metadata(model_document) if versioned_bundle else model_document
    if (
        int(metadata.get("queue_train_job_id") or 0) != int(announcement["train_job_id"])
        or int(metadata.get("checkpoint_step") or 0) != int(announcement["step"])
        or str(metadata.get("runtime_image_ref") or "") != str(announcement["runtime_image_ref"])
    ):
        raise ValueError("checkpoint metadata does not match the announcement")
    if not versioned_bundle:
        return
    recipe = validate_recipe_document(
        store.get_json(str(announcement["recipe_uri"])),
        source=str(announcement["recipe_uri"]),
    )
    if (
        isinstance(queued_recipe_document, Mapping)
        and queued_recipe_document.get("document_type") == "rlab.recipe"
    ):
        queued = validate_recipe_document(
            queued_recipe_document,
            source="train_jobs.recipe_payload_json",
        )
        if recipe != queued:
            raise ValueError("checkpoint recipe does not exactly match the queued recipe")
    if str(model_document["checkpoint"]["sha256"]) != str(announcement["sha256"]):
        raise ValueError("model document checkpoint binding mismatch")
    if str(model_document["recipe"]["sha256"]) != str(announcement["recipe_sha256"]):
        raise ValueError("model document recipe binding mismatch")
    if "evaluation_contract_sha256" in announcement:
        if evaluation_contract_sha256(recipe) != str(announcement["evaluation_contract_sha256"]):
            raise ValueError("recipe evaluation contract binding mismatch")
    elif "playback_contract_sha256" in announcement:
        if playback_contract_sha256(recipe) != str(announcement["playback_contract_sha256"]):
            raise ValueError("recipe playback contract binding mismatch")
    else:
        raise ValueError("checkpoint announcement lacks a portable contract binding")


class ModalInvoker(Protocol):
    def spawn(self, app_name: str, function_name: str, payload: Mapping[str, Any]) -> str: ...
    def poll(self, call_id: str) -> tuple[str, object | None]: ...
    def cancel(self, call_id: str) -> None: ...

    def stage_rom(self, app_name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


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
        except TimeoutError, modal.exception.TimeoutError:
            return "pending", None
        except (
            modal.exception.FunctionTimeoutError,
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

    def stage_rom(self, app_name: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        import modal

        result = modal.Function.from_name(app_name, "stage_rom").remote(dict(payload))
        if not isinstance(result, Mapping):
            raise RuntimeError("Modal ROM stager returned an invalid receipt")
        return dict(result)


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
                WHERE (
                    train_config->>'checkpoint_eval_backend' = 'modal'
                    OR telemetry_transport = 'neon_mailbox_v1'
                  )
                  AND (
                    status <> 'canceled'
                    OR telemetry_transport = 'neon_mailbox_v1'
                  )
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


def _defer_attempt_event(conn, event: Mapping[str, Any], error: object) -> int:
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
                    "error": repr(error)[:4000],
                },
            )
    return attempts


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
        if row is not None:
            cur.execute(
                """
                UPDATE eval_jobs j
                SET status = 'canceled', finished_at = now(), updated_at = now(),
                    error = 'superseded by accepted checkpoint'
                FROM eval_runs r
                WHERE j.id = %(id)s AND r.train_job_id = j.train_job_id
                  AND r.outcome = 'accepted'
                """,
                {"id": int(row["id"])},
            )
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
            try:
                announcement = store.get_json_optional(key)
            except Exception as exc:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            UPDATE eval_runs SET error = %(error)s, updated_at = now()
                            WHERE train_job_id = %(job_id)s
                            """,
                            {
                                "job_id": int(run["train_job_id"]),
                                "error": f"announcement observation failed: {exc!r}"[:4000],
                            },
                        )
                break
            if announcement is None:
                try:
                    complete = store.get_json_optional(
                        f"artifact-announcements/{int(run['train_job_id'])}/complete.json"
                    )
                except Exception as exc:
                    with conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                UPDATE eval_runs SET error = %(error)s, updated_at = now()
                                WHERE train_job_id = %(job_id)s
                                """,
                                {
                                    "job_id": int(run["train_job_id"]),
                                    "error": (f"complete announcement observation failed: {exc!r}")[
                                        :4000
                                    ],
                                },
                            )
                    break
                if complete is not None:
                    if not isinstance(complete, Mapping):
                        _mark_eval_run_failed(
                            conn,
                            int(run["train_job_id"]),
                            "complete announcement is not a mapping",
                        )
                        break
                    try:
                        last_id = int(complete.get("last_ledger_id") or 0)
                    except (TypeError, ValueError) as exc:
                        _mark_eval_run_failed(conn, int(run["train_job_id"]), exc)
                        break
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
            if not isinstance(announcement, Mapping):
                _mark_eval_run_failed(
                    conn,
                    int(run["train_job_id"]),
                    "checkpoint announcement is not a mapping",
                )
                break
            try:
                announced_train_job_id = int(announcement.get("train_job_id") or 0)
                announced_ledger_id = int(announcement.get("ledger_id") or 0)
            except (TypeError, ValueError) as exc:
                _mark_eval_run_failed(conn, int(run["train_job_id"]), exc)
                break
            if announced_train_job_id != int(run["train_job_id"]):
                _mark_eval_run_failed(
                    conn, int(run["train_job_id"]), "checkpoint announcement train job id mismatch"
                )
                break
            if announced_ledger_id != ordinal:
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
                    _verify_checkpoint_artifacts(store, announcement)
                except ObjectNotFound:
                    break
                except ValueError as exc:
                    _mark_eval_run_failed(conn, int(run["train_job_id"]), exc)
                    break
            with conn:
                if str(announcement.get("kind")) in {"checkpoint", "final"}:
                    eval_contract = announcement.get("eval", {})
                    stages = eval_contract.get("stages") or []
                    if "acceptance" in eval_contract:
                        descriptor = acceptance_job_descriptor(announcement)
                    elif stages:
                        descriptor = stage_job_descriptor(announcement, stage_index=0)
                    else:
                        descriptor = promotion_job_descriptor(announcement)
                    _insert_eval_job(conn, announcement, descriptor)
                    if "acceptance" not in eval_contract:
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


def _persist_artifact_announcement(
    conn,
    *,
    train_job_id: int,
    ledger_id: int,
    event_type: str,
    announcement: Mapping[str, Any],
) -> None:
    """Persist an ordered artifact event before its mailbox source is acknowledged."""

    payload = dict(announcement)
    disposition = "tombstone" if event_type == "checkpoint_tombstone" else "ready"
    artifact_kind = str(payload.get("kind") or "")
    if disposition == "tombstone":
        if artifact_kind != "tombstone":
            raise ValueError("checkpoint tombstone event must declare kind=tombstone")
    elif artifact_kind not in {"checkpoint", "final", "interrupted"}:
        raise ValueError(f"unsupported checkpoint artifact kind: {artifact_kind or 'missing'}")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    announcement_sha256 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    metadata_sha256 = str(
        payload.get("model_document_sha256") or payload.get("metadata_sha256") or ""
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO artifact_announcement_ledger (
              train_job_id, ledger_id, disposition, artifact_kind, checkpoint_step,
              checkpoint_sha256, checkpoint_uri, metadata_uri, metadata_sha256,
              recipe_uri, recipe_sha256, evaluation_contract_sha256,
              announcement_sha256, announcement_json
            ) VALUES (
              %(train_job_id)s, %(ledger_id)s, %(disposition)s, %(artifact_kind)s,
              %(checkpoint_step)s, %(checkpoint_sha256)s, %(checkpoint_uri)s,
              %(metadata_uri)s, %(metadata_sha256)s, %(recipe_uri)s,
              %(recipe_sha256)s, %(evaluation_contract_sha256)s,
              %(announcement_sha256)s, %(announcement_json)s
            )
            ON CONFLICT (train_job_id, ledger_id) DO NOTHING
            RETURNING announcement_sha256
            """,
            {
                "train_job_id": int(train_job_id),
                "ledger_id": int(ledger_id),
                "disposition": disposition,
                "artifact_kind": artifact_kind,
                "checkpoint_step": (
                    int(payload.get("step") or 0) if disposition == "ready" else None
                ),
                "checkpoint_sha256": (
                    str(payload.get("sha256") or "") if disposition == "ready" else None
                ),
                "checkpoint_uri": (
                    str(payload.get("model_uri") or "") if disposition == "ready" else None
                ),
                "metadata_uri": (
                    str(payload.get("metadata_uri") or "") if disposition == "ready" else None
                ),
                "metadata_sha256": metadata_sha256 if disposition == "ready" else None,
                "recipe_uri": str(payload.get("recipe_uri") or "") or None,
                "recipe_sha256": str(payload.get("recipe_sha256") or "") or None,
                "evaluation_contract_sha256": (
                    str(payload.get("evaluation_contract_sha256") or "") or None
                ),
                "announcement_sha256": announcement_sha256,
                "announcement_json": json_arg(payload),
            },
        )
        inserted = cur.fetchone()
        if inserted:
            return
        cur.execute(
            """
            SELECT announcement_sha256
            FROM artifact_announcement_ledger
            WHERE train_job_id = %(train_job_id)s AND ledger_id = %(ledger_id)s
            """,
            {"train_job_id": int(train_job_id), "ledger_id": int(ledger_id)},
        )
        existing = cur.fetchone()
    if not existing or str(existing["announcement_sha256"]) != announcement_sha256:
        raise RuntimeError(
            "artifact announcement ledger conflict: "
            f"train_job_id={int(train_job_id)} ledger_id={int(ledger_id)}"
        )


def ingest_mailbox_announcements(
    conn,
    store: ObjectStore,
    *,
    limit: int = 50,
    deadline_monotonic: float | None = None,
) -> int:
    ingested = 0
    next_announcement_ids: dict[int, int] = {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT e.*, r.contract_json, r.next_announcement_id,
              t.recipe_payload_json AS queued_recipe_document,
              r.train_job_id AS authoritative_train_job_id
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
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            break
        try:
            raw_payload = event.get("payload_json") or {}
            if not isinstance(raw_payload, Mapping):
                raise ValueError("checkpoint mailbox event payload is not a mapping")
            payload = dict(raw_payload)
            event_type = str(event["event_type"])
            train_job_id = int(
                event.get("authoritative_train_job_id") or payload.get("train_job_id") or 0
            )
            payload_train_job_id = int(payload.get("train_job_id") or 0)
        except Exception as exc:
            attempts = _defer_attempt_event(conn, event, exc)
            authoritative_train_job_id = event.get("authoritative_train_job_id")
            if attempts >= 3 and authoritative_train_job_id is not None:
                _mark_eval_run_failed(conn, int(authoritative_train_job_id), exc)
            continue
        if payload_train_job_id != train_job_id:
            attempts = _defer_attempt_event(
                conn,
                event,
                "checkpoint mailbox event train_job_id does not match its worker attempt",
            )
            if attempts >= 3:
                _mark_eval_run_failed(
                    conn,
                    train_job_id,
                    "checkpoint mailbox event has inconsistent train_job_id",
                )
            continue
        if event_type == "checkpoint_stream_closed":
            try:
                last_id = int(payload.get("last_ledger_id") or 0)
                checkpoint_count = int(payload.get("checkpoint_count") or 0)
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT pg_advisory_xact_lock(
                              hashtextextended(
                                'rlab-checkpoint-events:' || %(train_job_id)s::text, 0
                              )
                            )
                            """,
                            {"train_job_id": train_job_id},
                        )
                        cur.execute(
                            "SELECT next_announcement_id, complete_announcement_seen, "
                            "contract_json->'checkpoint_close_fence' AS close_fence "
                            "FROM eval_runs "
                            "WHERE train_job_id = %(train_job_id)s FOR UPDATE",
                            {"train_job_id": train_job_id},
                        )
                        current = cur.fetchone()
                        if not current:
                            raise ValueError("checkpoint close has no authoritative eval run")
                        if bool(current["complete_announcement_seen"]):
                            if current.get("close_fence") != payload:
                                raise ValueError(
                                    "checkpoint close fence conflicts with accepted close"
                                )
                        else:
                            cur.execute(
                                """
                                SELECT count(*) AS ledger_count,
                                  COALESCE(min(ledger_id), 0) AS first_id,
                                  COALESCE(max(ledger_id), 0) AS last_id
                                FROM artifact_announcement_ledger
                                WHERE train_job_id = %(train_job_id)s
                                """,
                                {"train_job_id": train_job_id},
                            )
                            closure = cur.fetchone() or {}
                            if (
                                last_id < 0
                                or checkpoint_count != last_id
                                or int(current["next_announcement_id"]) != last_id + 1
                                or int(closure.get("ledger_count") or 0) != checkpoint_count
                                or (
                                    checkpoint_count > 0
                                    and (
                                        int(closure.get("first_id") or 0) != 1
                                        or int(closure.get("last_id") or 0) != last_id
                                    )
                                )
                            ):
                                raise ValueError(
                                    "checkpoint close does not exactly match ledger closure"
                                )
                            cur.execute(
                                """
                                UPDATE eval_runs
                                SET complete_announcement_seen = TRUE,
                                    contract_json = jsonb_set(
                                      contract_json,
                                      '{checkpoint_close_fence}',
                                      %(close_fence)s,
                                      TRUE
                                    ),
                                    status = 'finalizing',
                                    error = NULL,
                                    updated_at = now()
                                WHERE train_job_id = %(train_job_id)s
                                """,
                                {
                                    "train_job_id": train_job_id,
                                    "close_fence": json_arg(payload),
                                },
                            )
                        cur.execute(
                            "DELETE FROM attempt_events WHERE id = %(id)s",
                            {"id": int(event["id"])},
                        )
            except (TypeError, ValueError) as exc:
                _defer_attempt_event(conn, event, exc)
            else:
                ingested += 1
            continue
        try:
            ledger_id = int(payload.get("ledger_id") or 0)
        except (TypeError, ValueError) as exc:
            _defer_attempt_event(conn, event, exc)
            continue
        announcement: Mapping[str, Any] = payload
        contract = event.get("contract_json") or {}
        eval_backend = (
            str(contract.get("checkpoint_eval_backend") or "local")
            if isinstance(contract, Mapping)
            else "local"
        )
        if event_type == "checkpoint_ready":
            try:
                announcement = validate_announcement(
                    payload,
                    materialized_train_config=dict(event["contract_json"]),
                )
                _verify_checkpoint_artifacts(
                    store,
                    announcement,
                    queued_recipe_document=(
                        event.get("queued_recipe_document")
                        if isinstance(event.get("queued_recipe_document"), Mapping)
                        else None
                    ),
                )
            except Exception as exc:
                attempts = _defer_attempt_event(conn, event, exc)
                with conn:
                    with conn.cursor() as cur:
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
        replayed = False
        sequence_error: str | None = None
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pg_advisory_xact_lock(
                      hashtextextended(
                        'rlab-checkpoint-events:' || %(train_job_id)s::text, 0
                      )
                    )
                    """,
                    {"train_job_id": train_job_id},
                )
                cur.execute(
                    """
                    SELECT next_announcement_id
                    FROM eval_runs
                    WHERE train_job_id = %(train_job_id)s
                    FOR UPDATE
                    """,
                    {"train_job_id": train_job_id},
                )
                current = cur.fetchone()
                current_id = int((current or {}).get("next_announcement_id") or 0)
                if current_id < 1:
                    sequence_error = "checkpoint mailbox eval run is missing"
                elif ledger_id < current_id:
                    cur.execute(
                        """
                        SELECT announcement_json
                        FROM artifact_announcement_ledger
                        WHERE train_job_id = %(train_job_id)s
                          AND ledger_id = %(ledger_id)s
                        FOR UPDATE
                        """,
                        {"train_job_id": train_job_id, "ledger_id": ledger_id},
                    )
                    existing = cur.fetchone()
                    if existing and existing.get("announcement_json") == dict(announcement):
                        cur.execute(
                            "DELETE FROM attempt_events WHERE id = %(id)s",
                            {"id": int(event["id"])},
                        )
                        replayed = True
                    else:
                        sequence_error = (
                            "checkpoint mailbox replay conflicts with authoritative ledger: "
                            f"ledger_id={ledger_id}"
                        )
                elif ledger_id > current_id:
                    sequence_error = (
                        f"checkpoint mailbox ledger gap: expected={current_id} got={ledger_id}"
                    )
                else:
                    _persist_artifact_announcement(
                        conn,
                        train_job_id=train_job_id,
                        ledger_id=ledger_id,
                        event_type=event_type,
                        announcement=announcement,
                    )
                    if (
                        eval_backend == "modal"
                        and event_type == "checkpoint_ready"
                        and str(announcement.get("kind"))
                        in {"checkpoint", "final"}
                    ):
                        eval_contract = announcement.get("eval", {})
                        stages = eval_contract.get("stages") or []
                        descriptor = (
                            acceptance_job_descriptor(announcement)
                            if "acceptance" in eval_contract
                            else stage_job_descriptor(announcement, stage_index=0)
                            if stages
                            else promotion_job_descriptor(announcement)
                        )
                        _insert_eval_job(conn, announcement, descriptor)
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
                        next_announcement_ids[train_job_id] = ledger_id + 1
        if sequence_error is not None:
            _defer_attempt_event(conn, event, sequence_error)
            continue
        if replayed:
            next_announcement_ids[train_job_id] = max(
                next_announcement_ids.get(train_job_id, 1),
                ledger_id + 1,
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
            acquire_fleet_admission_xact_lock(conn)
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
    train_config = job.get("train_config")
    schema_version = int(
        train_config.get("metrics_schema_version", 4)
        if isinstance(train_config, Mapping)
        else 4
    )
    if str(job["purpose"]) == "acceptance":
        aggregates = raw_metrics.get("_acceptance_aggregates")
        if not isinstance(aggregates, Mapping):
            raise ValueError("acceptance metrics are missing recomputed aggregates")
        metrics = (
            evaluation_metric_payload(
                protocol="full",
                metrics=raw_metrics,
                checkpoint_step=int(job["checkpoint_step"]),
                checkpoint_artifact=str(job["checkpoint_uri"]),
                eval_source="modal",
                metrics_schema_version=schema_version,
                acceptance_projection=schema_version >= 5,
            )
            if passed
            else {GLOBAL_STEP: int(job["checkpoint_step"])}
        )
        metrics.update(
            {
                EVAL_ACCEPTANCE_PASS: 1.0 if passed else 0.0,
                EVAL_ACCEPTANCE_EPISODES_PLANNED: int(aggregates["episodes_planned"]),
                EVAL_ACCEPTANCE_EPISODES_COMPLETED: int(aggregates["episodes_completed"]),
                EVAL_ACCEPTANCE_DURATION_SECONDS: float(
                    raw_metrics.get("_acceptance_duration_seconds") or 0.0
                ),
            }
        )
        if schema_version == 4:
            metrics[EVAL_ACCEPTANCE_FAILURE_COUNT] = int(aggregates["failure_count"])
        validate_metric_payload(metrics, schema_version=schema_version)
        return metrics
    if str(job["purpose"]) == "promotion":
        return evaluation_metric_payload(
            protocol="full",
            metrics=raw_metrics,
            checkpoint_step=int(job["checkpoint_step"]),
            checkpoint_artifact=str(job["checkpoint_uri"]),
            eval_source="modal",
            metrics_schema_version=schema_version,
        )
    stage_name = str(job["stage_name"])
    metrics = evaluation_metric_payload(
        protocol=stage_name,
        metrics=raw_metrics,
        checkpoint_step=int(job["checkpoint_step"]),
        checkpoint_artifact=str(job["checkpoint_uri"]),
        eval_source="modal",
        metrics_schema_version=schema_version,
    )
    metrics[checkpoint_eval_stage_metric(stage_name, "candidate/pass")] = 1.0 if passed else 0.0
    metrics[checkpoint_eval_stage_metric(stage_name, "candidate/stage_index")] = float(
        job["stage_index"]
    )
    metrics[checkpoint_eval_stage_metric(stage_name, "source")] = "modal"
    validate_metric_payload(metrics, schema_version=schema_version)
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
    acceptance = str(attempt["purpose"]) == "acceptance"
    if acceptance:
        raw_metrics["_acceptance_aggregates"] = dict(validated["claimed_aggregates"])
        raw_metrics["_acceptance_duration_seconds"] = float(
            validated.get("duration_seconds") or 0.0
        )
        passed = str(validated.get("verdict") or "") == "accepted"
        observed = []
        if passed:
            raw_metrics[EVAL_FULL_DURATION_SECONDS] = float(
                validated.get("duration_seconds") or 0.0
            )
            raw_metrics.update(
                {
                    "checkpoint_step": int(attempt["checkpoint_step"]),
                    "checkpoint_artifact": str(attempt["checkpoint_uri"]),
                    "_eval_by_start_rows": eval_by_start_rows(
                        [dict(episode) for episode in validated["episode_results"]]
                    ),
                }
            )
    elif str(attempt["purpose"]) == "promotion":
        raw_metrics.update(
            {
                "checkpoint_step": int(attempt["checkpoint_step"]),
                "checkpoint_artifact": str(attempt["checkpoint_uri"]),
                "_eval_by_start_rows": eval_by_start_rows(
                    [dict(episode) for episode in validated["episode_results"]]
                ),
            }
        )
    if not acceptance:
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
        "verdict": (
            str(validated.get("verdict")) if acceptance else "accepted" if passed else "rejected"
        ),
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
            promotion_won = False
            if acceptance and bool(decision.get("passed")):
                promotion = {
                    "eval_job_id": int(attempt["eval_job_id"]),
                    "accepted_attempt_id": int(attempt["id"]),
                    "checkpoint_sha256": str(attempt["checkpoint_sha256"]),
                    "checkpoint_step": int(attempt["checkpoint_step"]),
                    "checkpoint_uri": str(attempt["checkpoint_uri"]),
                    "result_uri": str(attempt["result_uri"]),
                    "raw_metrics": raw_metrics,
                }
                cur.execute(
                    """
                    UPDATE eval_runs
                    SET promoted_eval_job_id = %(eval_job_id)s,
                        promotion_revision = promotion_revision + 1,
                        promotion_json = %(promotion)s::jsonb || jsonb_build_object(
                          'promotion_revision', promotion_revision + 1
                        ),
                        outcome = 'accepted',
                        acceptance_committed_at = now(),
                        updated_at = now(), error = NULL
                    WHERE train_job_id = %(train_job_id)s
                      AND status IN ('active', 'finalizing')
                      AND promoted_eval_job_id IS NULL
                    """,
                    {
                        "eval_job_id": int(attempt["eval_job_id"]),
                        "promotion": json_arg(promotion),
                        "train_job_id": int(attempt["train_job_id"]),
                    },
                )
                promotion_won = cur.rowcount == 1
                if promotion_won:
                    cur.execute(
                        """
                        UPDATE eval_jobs
                        SET status = 'canceled', finished_at = now(), updated_at = now(),
                            error = 'superseded by accepted checkpoint'
                        WHERE train_job_id = %(train_job_id)s
                          AND id <> %(eval_job_id)s
                          AND status IN ('pending', 'blocked_budget', 'dispatching', 'submitted')
                        """,
                        {
                            "train_job_id": int(attempt["train_job_id"]),
                            "eval_job_id": int(attempt["eval_job_id"]),
                        },
                    )
            if (
                bool(attempt.get("candidate_stop"))
                and bool(decision.get("passed"))
                and (not acceptance or promotion_won)
            ):
                cur.execute(
                    """
                    INSERT INTO attempt_commands (
                      command_id, attempt_id, command_type, payload_json
                    )
                    SELECT
                      'acceptance-stop:' || %(eval_job_id)s::text,
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
                "canonical_promotion": bool(acceptance and passed and promotion_won),
            },
        )
    if not acceptance:
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
            SELECT a.*, j.train_job_id, j.ledger_id, j.checkpoint_step,
              j.checkpoint_sha256, j.checkpoint_uri,
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
        try:
            result = store.get_json_optional(str(attempt["result_uri"]))
        except Exception as exc:
            _mark_attempt_failure(
                conn,
                attempt=attempt,
                error=f"result observation failed: {exc!r}",
                config=config,
            )
            changed += 1
            continue
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
            except DatabaseError:
                raise
            except Exception as exc:
                terminal = deterministic_eval_failure(exc)
                _mark_attempt_failure(
                    conn, attempt=attempt, error=repr(exc), config=config, terminal=terminal
                )
            changed += 1
            continue
        try:
            expires_at = attempt["expires_at"]
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
        except Exception as exc:
            _mark_attempt_failure(
                conn,
                attempt=attempt,
                error=f"terminal: malformed attempt expiry: {exc!r}",
                config=config,
                terminal=True,
            )
            changed += 1
            continue
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
            try:
                state, detail = invoker.poll(str(attempt["modal_call_id"]))
            except Exception as exc:
                _mark_attempt_failure(
                    conn,
                    attempt=attempt,
                    error=f"Modal poll failed: {exc!r}",
                    config=config,
                )
                changed += 1
                continue
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
            SELECT a.id, a.attempt_id, a.modal_call_id
            FROM eval_attempts a
            JOIN eval_jobs j ON j.id = a.eval_job_id
            JOIN train_jobs t ON t.id = j.train_job_id
            WHERE (t.status = 'canceled' OR j.status = 'canceled')
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
                attempt_changed = cur.rowcount
                cur.execute(
                    """
                    UPDATE worker_attempts
                    SET status = 'canceled', finished_at = now(), error = %(error)s
                    WHERE attempt_id = %(attempt_id)s
                      AND status IN ('launching', 'running')
                    """,
                    {
                        "attempt_id": str(attempt["attempt_id"]),
                        "error": error,
                    },
                )
                changed += attempt_changed
    return changed


def reconcile_canceled_eval_state(conn) -> dict[str, int]:
    """Close logical eval work that raced with train-job cancellation.

    The operator cancellation transaction closes eval rows that already exist,
    but checkpoint announcements can be ingested after that transaction while
    the training container is shutting down. Those late rows must not remain
    pending/submitted forever or keep appearing as queued work. Also repair worker rows left active
    by controllers that terminalized the paired eval attempt before worker synchronization existed.
    """

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_jobs j
                SET status = 'canceled', finished_at = now(), updated_at = now(),
                  error = 'training finalization canceled'
                FROM train_jobs t
                JOIN eval_runs r ON r.train_job_id = t.id
                WHERE t.id = j.train_job_id
                  AND (t.status = 'canceled' OR r.outcome = 'canceled')
                  AND j.status IN ('pending', 'dispatching', 'submitted', 'blocked_budget')
                """
            )
            jobs = int(cur.rowcount)
            cur.execute(
                """
                UPDATE eval_runs r
                SET status = 'canceled', outcome = 'canceled', updated_at = now(),
                  error = 'training finalization canceled'
                FROM train_jobs t
                WHERE t.id = r.train_job_id
                  AND t.status = 'canceled'
                  AND r.status NOT IN ('complete', 'failed', 'canceled')
                """
            )
            runs = int(cur.rowcount)
            cur.execute(
                """
                UPDATE worker_attempts w
                SET status = 'canceled',
                    finished_at = COALESCE(w.finished_at, a.finished_at, now()),
                    error = COALESCE(w.error, a.error, 'evaluation attempt canceled')
                FROM eval_attempts a
                WHERE w.attempt_id = a.attempt_id
                  AND w.task_kind = 'eval'
                  AND a.status = 'canceled'
                  AND w.status IN ('launching', 'running')
                """
            )
            workers = int(cur.rowcount)
    return {"jobs": jobs, "runs": runs, "workers": workers}


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


def available_eval_slots(*, active_calls: int, hard_cap: int) -> int:
    return max(0, int(hard_cap) - max(0, int(active_calls)))


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


def _record_eval_job_record_error(conn, *, job_id: int, error: object, terminal: bool) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_jobs
                SET status = CASE WHEN %(terminal)s THEN 'failed' ELSE 'pending' END,
                    error = %(error)s,
                    finished_at = CASE WHEN %(terminal)s THEN now() ELSE NULL END,
                    updated_at = now()
                WHERE id = %(job_id)s
                  AND status IN ('pending', 'blocked_budget', 'dispatching')
                """,
                {
                    "job_id": int(job_id),
                    "error": str(error)[:4000],
                    "terminal": bool(terminal),
                },
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
        cur.execute(
            "SELECT count(*) AS count FROM eval_attempts WHERE status IN ('dispatching', 'submitted')"
        )
        active = int(cur.fetchone()["count"])
    slots = available_eval_slots(
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
            JOIN eval_runs r ON r.train_job_id = j.train_job_id
            WHERE j.status IN ('pending', 'blocked_budget')
              AND r.outcome IS NULL
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
        try:
            reused = _reuse_result(conn, store, job)
        except DatabaseError:
            raise
        except Exception as exc:
            _record_eval_job_record_error(
                conn,
                job_id=int(job["id"]),
                error=f"result reuse observation failed: {exc!r}",
                terminal=deterministic_eval_failure(exc),
            )
            continue
        if reused:
            continue
        try:
            timeout = config.timeout_for(str(job["purpose"]), int(job["stage_index"]))
            reserved = config.reserved_cost(timeout)
            app_name = modal_app_name(
                config.app_name_prefix,
                str(job["contract_json"]["runtime_image_ref"]),
            )
        except Exception as exc:
            _record_eval_job_record_error(
                conn,
                job_id=int(job["id"]),
                error=f"malformed evaluation record: {exc!r}",
                terminal=True,
            )
            continue
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
        try:
            result_uri = store.uri(f"eval-attempts/{job['execution_key']}/{attempt_id}.json")
            expires_at = datetime.now(UTC) + timedelta(
                seconds=config.startup_timeout_seconds + timeout + config.expiry_margin_seconds
            )
        except Exception as exc:
            _record_eval_job_record_error(
                conn,
                job_id=int(job["id"]),
                error=f"evaluation dispatch preparation failed: {exc!r}",
                terminal=deterministic_eval_failure(exc),
            )
            continue
        with conn:
            acquire_fleet_admission_xact_lock(conn)
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
        try:
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
                "model_document_get_url": store.presign_get(
                    str(job["metadata_uri"]), expires_seconds=seconds_to_expiry
                ),
                "model_document_sha256": str(
                    job["source_announcement_json"]["model_document_sha256"]
                ),
                "recipe_get_url": store.presign_get(
                    str(job["source_announcement_json"]["recipe_uri"]),
                    expires_seconds=seconds_to_expiry,
                ),
                "result_uri": result_uri,
                "result_put_url": store.presign_put(result_uri, expires_seconds=seconds_to_expiry),
            }
            if isinstance(asset, Mapping):
                payload["rom_get_url"] = store.presign_get(
                    str(asset["object_uri"]), expires_seconds=seconds_to_expiry
                )
        except Exception as exc:
            _mark_attempt_failure(
                conn,
                attempt={
                    "id": attempt_row_id,
                    "attempt_id": attempt_id,
                    "eval_job_id": int(job["id"]),
                    "retry_round": int(job.get("retry_round") or 0),
                },
                error=f"dispatch payload preparation failed: {exc!r}",
                config=config,
                terminal=deterministic_eval_failure(exc),
            )
            continue
        if isinstance(asset, Mapping):
            stager = getattr(invoker, "stage_rom", None)
            if callable(stager):
                try:
                    receipt = stager(
                        app_name,
                        {
                            "manifest": dict(asset),
                            "rom_get_url": payload["rom_get_url"],
                        },
                    )
                    if str(receipt.get("sha256") or "") != str(asset["sha256"]):
                        raise ValueError("Modal ROM cache staging receipt hash mismatch")
                except Exception as exc:
                    payload["rom_cache_degraded"] = type(exc).__name__
        try:
            call_id = invoker.spawn(app_name, config.function_name, payload)
        except Exception as exc:
            # A transport failure can happen after Modal accepted the call but before its call id
            # reached us. Keep the immutable attempt in dispatching state and reconcile its
            # create-only R2 result until expiry; immediately retrying could execute the same
            # logical checkpoint twice.
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE eval_attempts
                        SET error = %(error)s
                        WHERE id = %(id)s AND status = 'dispatching'
                        """,
                        {
                            "id": attempt_row_id,
                            "error": f"spawn outcome uncertain: {exc!r}"[:4000],
                        },
                    )
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
            WHERE j.status = 'succeeded' AND j.purpose NOT IN ('promotion', 'acceptance')
              AND j.stage_index = jsonb_array_length(
                j.source_announcement_json->'eval'->'stages'
              ) - 1
              AND (j.decision_json->>'passed')::boolean = TRUE
              AND r.complete_announcement_seen = TRUE
              AND r.outcome IS NULL
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


def terminalize_artifact_only_runs(conn) -> int:
    """Terminalize non-Modal runs only after their artifact stream is durable."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH ready AS (
                  SELECT
                    r.train_job_id,
                    CASE
                      WHEN t.cancel_requested OR r.outcome = 'canceled' THEN 'canceled'
                      WHEN EXISTS (
                        SELECT 1 FROM job_launches launch
                        WHERE launch.job_id = r.train_job_id
                          AND launch.job_kind = 'train'
                          AND launch.state = 'failed'
                      ) THEN 'failed'
                      WHEN EXISTS (
                        SELECT 1 FROM artifact_announcement_ledger tombstone
                        WHERE tombstone.train_job_id = r.train_job_id
                          AND tombstone.disposition = 'tombstone'
                      ) THEN 'finalization_failed'
                      ELSE 'succeeded'
                    END AS terminal_status
                  FROM eval_runs r
                  JOIN train_jobs t ON t.id = r.train_job_id
                  WHERE t.status = 'finalizing'
                    AND t.process_exited_at IS NOT NULL
                    AND COALESCE(r.contract_json->>'checkpoint_eval_backend', 'local') <> 'modal'
                    AND r.complete_announcement_seen = TRUE
                    AND (
                      SELECT count(*)
                      FROM artifact_announcement_ledger ledger
                      WHERE ledger.train_job_id = r.train_job_id
                    ) = GREATEST(r.next_announcement_id - 1, 0)
                    AND NOT EXISTS (
                      SELECT 1 FROM artifact_announcement_ledger ledger
                      WHERE ledger.train_job_id = r.train_job_id
                        AND ledger.disposition = 'ready'
                        AND NOT EXISTS (
                          SELECT 1 FROM artifact_publication_receipts receipt
                          WHERE receipt.train_job_id = ledger.train_job_id
                            AND receipt.ledger_id = ledger.ledger_id
                          AND receipt.role = 'availability'
                          AND receipt.promotion_revision = 0
                          AND (
                            (
                              COALESCE((t.train_config->>'wandb')::boolean, FALSE)
                              AND NOT COALESCE(
                                (t.train_config->>'no_wandb_artifacts')::boolean,
                                FALSE
                              )
                              AND receipt.disposition = 'confirmed'
                              AND (
                                receipt.stream_id LIKE 'artifact-v3-%%'
                                OR receipt.stream_id LIKE 'artifact-v2-%%'
                              )
                            )
                            OR (
                              (
                                NOT COALESCE(
                                  (t.train_config->>'wandb')::boolean,
                                  FALSE
                                )
                                OR COALESCE(
                                  (t.train_config->>'no_wandb_artifacts')::boolean,
                                  FALSE
                                )
                              )
                              AND receipt.disposition = 'opted_out'
                            )
                          )
                        )
                    )
                    AND (
                      EXISTS (
                        SELECT 1 FROM job_launches launch
                        WHERE launch.job_id = r.train_job_id
                          AND launch.job_kind = 'train'
                          AND launch.state IN ('failed', 'canceled')
                      )
                      OR EXISTS (
                        SELECT 1 FROM artifact_announcement_ledger tombstone
                        WHERE tombstone.train_job_id = r.train_job_id
                          AND tombstone.disposition = 'tombstone'
                      )
                      OR EXISTS (
                        SELECT 1 FROM artifact_announcement_ledger final_artifact
                        WHERE final_artifact.train_job_id = r.train_job_id
                          AND final_artifact.disposition = 'ready'
                          AND final_artifact.artifact_kind = 'final'
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
                        AND (
                          s.final_sequence IS NULL
                          OR s.published_sequence < s.final_sequence
                        )
                    )
                    AND t.live_publication_status IN ('complete', 'disabled')
                  FOR UPDATE OF r, t
                ), completed AS (
                  UPDATE eval_runs r
                  SET status = CASE
                        WHEN ready.terminal_status = 'canceled' THEN 'canceled'
                        WHEN ready.terminal_status IN ('failed', 'finalization_failed')
                          THEN 'failed'
                        ELSE 'complete'
                      END,
                      updated_at = now(),
                      error = CASE
                        WHEN ready.terminal_status = 'canceled'
                          THEN COALESCE(r.error, 'training finalization canceled')
                        WHEN ready.terminal_status = 'failed'
                          THEN COALESCE(r.error, 'training failed')
                        WHEN ready.terminal_status = 'finalization_failed'
                          THEN COALESCE(r.error, 'checkpoint artifact upload exhausted retries')
                        ELSE NULL
                      END
                  FROM ready
                  WHERE r.train_job_id = ready.train_job_id
                  RETURNING r.train_job_id, ready.terminal_status
                )
                UPDATE train_jobs t
                SET status = completed.terminal_status,
                  finished_at = now(),
                  error = CASE
                    WHEN completed.terminal_status = 'succeeded' THEN NULL
                    WHEN completed.terminal_status = 'canceled'
                      THEN COALESCE(t.error, 'training finalization canceled')
                    WHEN completed.terminal_status = 'failed'
                      THEN COALESCE(t.error, 'training failed')
                    ELSE COALESCE(t.error, 'checkpoint artifact upload exhausted retries')
                  END,
                  live_publication_next_retry_at = NULL,
                  live_publication_error = CASE
                    WHEN completed.terminal_status = 'succeeded' THEN NULL
                    ELSE live_publication_error
                  END
                FROM completed
                WHERE t.id = completed.train_job_id
                  AND t.status = 'finalizing'
                """
            )
            return cur.rowcount


def terminalize_runs(conn) -> int:
    """Apply the one authoritative, idempotent terminal state reduction.

    Goal outcome and operational state remain separate: a valid not-accepted run is
    operationally successful, while unknown evidence or a required but unobserved acceptance stop
    is a finalization failure. Acceptance committed after natural process exit needs no stop
    callback. Nothing becomes terminal before the process exit, durable evidence, closed metric
    streams, projections, and remote W&B completion are all confirmed.
    """

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH ready AS (
                  SELECT
                    r.train_job_id,
                    r.outcome,
                    CASE
                      WHEN r.outcome = 'canceled' THEN 'canceled'
                      WHEN r.outcome = 'unknown' THEN 'finalization_failed'
                      WHEN r.outcome = 'accepted'
                        AND r.acceptance_committed_at IS NULL
                        THEN 'finalization_failed'
                      WHEN r.outcome = 'accepted'
                        AND r.acceptance_committed_at < t.process_exited_at
                        AND (
                          t.learner_stop_observed_at IS NULL
                          OR r.stop_delivery_slo_met IS NOT TRUE
                        ) THEN 'finalization_failed'
                      ELSE 'succeeded'
                    END AS terminal_status,
                    CASE
                      WHEN r.outcome = 'unknown' THEN
                        COALESCE(r.error, 'acceptance outcome is unknown')
                      WHEN r.outcome = 'accepted'
                        AND r.acceptance_committed_at IS NULL THEN
                        'accepted outcome is missing its commit timestamp'
                      WHEN r.outcome = 'accepted'
                        AND r.acceptance_committed_at < t.process_exited_at
                        AND t.learner_stop_observed_at IS NULL THEN
                        'accepted checkpoint was not observed by the learner stop callback'
                      WHEN r.outcome = 'accepted'
                        AND r.acceptance_committed_at < t.process_exited_at
                        AND r.stop_delivery_slo_met IS NOT TRUE THEN
                        'acceptance stop delivery exceeded five seconds'
                      ELSE NULL
                    END AS terminal_error
                  FROM eval_runs r
                  JOIN train_jobs t ON t.id = r.train_job_id
                  WHERE t.status = 'finalizing'
                    AND t.process_exited_at IS NOT NULL
                    AND r.complete_announcement_seen = TRUE
                    AND (
                      SELECT count(*)
                      FROM artifact_announcement_ledger ledger
                      WHERE ledger.train_job_id = r.train_job_id
                    ) = GREATEST(r.next_announcement_id - 1, 0)
                    AND r.outcome IN ('accepted', 'not_accepted', 'unknown', 'canceled')
                    AND NOT EXISTS (
                      SELECT 1 FROM artifact_announcement_ledger ledger
                      WHERE ledger.train_job_id = r.train_job_id
                        AND ledger.disposition = 'ready'
                        AND NOT EXISTS (
                          SELECT 1 FROM artifact_publication_receipts receipt
                          WHERE receipt.train_job_id = ledger.train_job_id
                            AND receipt.ledger_id = ledger.ledger_id
                            AND receipt.role = 'availability'
                            AND receipt.promotion_revision = 0
                            AND (
                              (
                                COALESCE((t.train_config->>'wandb')::boolean, FALSE)
                                AND NOT COALESCE(
                                  (t.train_config->>'no_wandb_artifacts')::boolean,
                                  FALSE
                                )
                                AND receipt.disposition = 'confirmed'
                                AND (
                                  receipt.stream_id LIKE 'artifact-v3-%%'
                                  OR receipt.stream_id LIKE 'artifact-v2-%%'
                                )
                              )
                              OR (
                                (
                                  NOT COALESCE(
                                    (t.train_config->>'wandb')::boolean,
                                    FALSE
                                  )
                                  OR COALESCE(
                                    (t.train_config->>'no_wandb_artifacts')::boolean,
                                    FALSE
                                  )
                                )
                                AND receipt.disposition = 'opted_out'
                              )
                            )
                        )
                    )
                    AND (
                      r.outcome = 'canceled'
                      OR EXISTS (
                        SELECT 1 FROM artifact_announcement_ledger final_artifact
                        WHERE final_artifact.train_job_id = r.train_job_id
                          AND final_artifact.disposition = 'ready'
                          AND final_artifact.artifact_kind = 'final'
                      )
                    )
                    AND (
                      r.promoted_eval_job_id IS NULL
                      OR EXISTS (
                        SELECT 1 FROM eval_jobs promoted
                        JOIN artifact_publication_receipts receipt
                          ON receipt.train_job_id = promoted.train_job_id
                         AND receipt.ledger_id = promoted.ledger_id
                         AND receipt.role = 'promotion'
                         AND receipt.promotion_revision = r.promotion_revision
                         AND (
                           (
                             COALESCE((t.train_config->>'wandb')::boolean, FALSE)
                             AND NOT COALESCE(
                               (t.train_config->>'no_wandb_artifacts')::boolean,
                               FALSE
                             )
                             AND receipt.disposition = 'confirmed'
                           )
                           OR (
                             (
                               NOT COALESCE(
                                 (t.train_config->>'wandb')::boolean,
                                 FALSE
                               )
                               OR COALESCE(
                                 (t.train_config->>'no_wandb_artifacts')::boolean,
                                 FALSE
                               )
                             )
                             AND receipt.disposition = 'opted_out'
                           )
                         )
                        WHERE promoted.id = r.promoted_eval_job_id
                      )
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM eval_jobs j
                      WHERE j.train_job_id = r.train_job_id
                        AND j.status IN (
                          'pending', 'dispatching', 'submitted', 'blocked_budget'
                        )
                    )
                    AND NOT EXISTS (
                      SELECT 1 FROM eval_jobs j
                      WHERE j.train_job_id = r.train_job_id
                        AND j.status = 'succeeded'
                        AND j.projected_at IS NULL
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
                        AND (
                          s.final_sequence IS NULL
                          OR s.published_sequence < s.final_sequence
                        )
                    )
                    AND t.live_publication_status IN ('complete', 'disabled')
                  FOR UPDATE OF r, t
                ), completed AS (
                  UPDATE eval_runs r
                  SET status = CASE
                        WHEN ready.terminal_status = 'canceled' THEN 'canceled'
                        WHEN ready.terminal_status = 'finalization_failed' THEN 'failed'
                        ELSE 'complete'
                      END,
                      updated_at = now(),
                      error = ready.terminal_error
                  FROM ready
                  WHERE r.train_job_id = ready.train_job_id
                  RETURNING r.train_job_id, ready.terminal_status, ready.terminal_error
                )
                UPDATE train_jobs t
                SET status = completed.terminal_status,
                  finished_at = now(), error = completed.terminal_error,
                  live_publication_next_retry_at = NULL,
                  live_publication_error = CASE
                    WHEN completed.terminal_status = 'succeeded' THEN NULL
                    ELSE live_publication_error
                  END
                FROM completed
                WHERE t.id = completed.train_job_id
                  AND t.status = 'finalizing'
                """
            )
            return cur.rowcount


def reconcile_publication_finishing(conn) -> int:
    """Close producer side of W&B publication only after every projection is durable."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs t
                SET live_publication_status = 'finishing',
                    live_publication_attempts = 0,
                    live_publication_error = NULL,
                    live_publication_next_retry_at = NULL
                FROM eval_runs r
                WHERE r.train_job_id = t.id
                  AND t.status = 'finalizing'
                  AND (
                    t.telemetry_transport = 'neon_mailbox_v1'
                    OR EXISTS (
                      SELECT 1 FROM metric_streams artifact_stream
                      JOIN worker_attempts artifact_attempt
                        ON artifact_attempt.attempt_id = artifact_stream.attempt_id
                      WHERE artifact_attempt.train_job_id = t.id
                        AND (
                          artifact_stream.stream_id LIKE 'artifact-v2-%%'
                          OR artifact_stream.stream_id LIKE 'artifact-v3-%%'
                        )
                    )
                  )
                  AND COALESCE((t.train_config->>'wandb')::boolean, FALSE)
                  AND t.live_publication_status IN ('pending', 'live')
                  AND r.complete_announcement_seen = TRUE
                  AND (
                    SELECT count(*)
                    FROM artifact_announcement_ledger ledger
                    WHERE ledger.train_job_id = r.train_job_id
                  ) = GREATEST(r.next_announcement_id - 1, 0)
                  AND (
                    COALESCE(r.contract_json->>'checkpoint_eval_backend', 'local') <> 'modal'
                    OR r.outcome IN ('accepted', 'not_accepted', 'unknown', 'canceled')
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM artifact_announcement_ledger ledger
                    WHERE ledger.train_job_id = r.train_job_id
                      AND ledger.disposition = 'ready'
                      AND NOT EXISTS (
                        SELECT 1 FROM artifact_publication_receipts receipt
                        WHERE receipt.train_job_id = ledger.train_job_id
                          AND receipt.ledger_id = ledger.ledger_id
                          AND receipt.role = 'availability'
                          AND receipt.promotion_revision = 0
                          AND receipt.disposition = 'confirmed'
                          AND (
                            receipt.stream_id LIKE 'artifact-v3-%%'
                            OR receipt.stream_id LIKE 'artifact-v2-%%'
                          )
                      )
                  )
                  AND (
                    EXISTS (
                      SELECT 1 FROM job_launches launch
                      WHERE launch.job_id = r.train_job_id
                        AND launch.job_kind = 'train'
                        AND launch.state IN ('failed', 'canceled')
                    )
                    OR EXISTS (
                      SELECT 1 FROM artifact_announcement_ledger tombstone
                      WHERE tombstone.train_job_id = r.train_job_id
                        AND tombstone.disposition = 'tombstone'
                    )
                    OR EXISTS (
                      SELECT 1 FROM artifact_announcement_ledger final_artifact
                      WHERE final_artifact.train_job_id = r.train_job_id
                        AND final_artifact.disposition = 'ready'
                        AND final_artifact.artifact_kind = 'final'
                    )
                  )
                  AND (
                    r.promoted_eval_job_id IS NULL
                    OR EXISTS (
                      SELECT 1 FROM eval_jobs promoted
                      JOIN artifact_publication_receipts receipt
                        ON receipt.train_job_id = promoted.train_job_id
                       AND receipt.ledger_id = promoted.ledger_id
                       AND receipt.role = 'promotion'
                       AND receipt.promotion_revision = r.promotion_revision
                       AND receipt.disposition = 'confirmed'
                      WHERE promoted.id = r.promoted_eval_job_id
                    )
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM eval_jobs j
                    WHERE j.train_job_id = t.id
                      AND j.status IN ('pending', 'dispatching', 'submitted', 'blocked_budget')
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM eval_jobs j
                    WHERE j.train_job_id = t.id
                      AND j.status = 'succeeded'
                      AND j.projected_at IS NULL
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_batches b
                    JOIN metric_streams s ON s.stream_id = b.stream_id
                    JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                    WHERE a.train_job_id = t.id
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_streams s
                    JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                    WHERE a.train_job_id = t.id
                      AND (
                        s.final_sequence IS NULL
                        OR s.published_sequence < s.final_sequence
                      )
                  )
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
                SET status = 'failed', outcome = 'unknown', updated_at = now(),
                  error = COALESCE(
                    (SELECT j.error FROM eval_jobs j
                     WHERE j.train_job_id = r.train_job_id AND j.status = 'failed'
                     ORDER BY j.updated_at DESC LIMIT 1),
                    CASE
                      WHEN NOT EXISTS (
                        SELECT 1 FROM eval_jobs j
                        WHERE j.train_job_id = r.train_job_id
                      ) THEN 'checkpoint stream closed without evaluable evidence'
                      ELSE 'required evaluation exhausted retries'
                    END
                  )
                WHERE r.status IN ('active', 'awaiting_artifact_recovery', 'finalizing')
                  AND COALESCE(r.contract_json->>'checkpoint_eval_backend', 'local') = 'modal'
                  AND r.outcome IS NULL
                    AND r.complete_announcement_seen = TRUE
                    AND (
                      SELECT count(*)
                      FROM artifact_announcement_ledger ledger
                      WHERE ledger.train_job_id = r.train_job_id
                    ) = GREATEST(r.next_announcement_id - 1, 0)
                  AND EXISTS (
                    SELECT 1 FROM train_jobs t
                    WHERE t.id = r.train_job_id AND t.status = 'finalizing'
                  )
                  AND (
                    EXISTS (
                      SELECT 1 FROM eval_jobs j
                      WHERE j.train_job_id = r.train_job_id AND j.status = 'failed'
                    )
                    OR NOT EXISTS (
                      SELECT 1 FROM eval_jobs j
                      WHERE j.train_job_id = r.train_job_id
                    )
                  )
                """
            )
            return cur.rowcount


def terminalize_failed_eval_runs(conn) -> int:
    """Bound failed evaluation state once no evaluation execution remains active."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs t
                SET status = 'finalization_failed', finished_at = now(),
                  error = COALESCE(r.error, 'evaluation finalization failed')
                FROM eval_runs r
                WHERE r.train_job_id = t.id
                  AND t.status = 'finalizing'
                  AND t.process_exited_at IS NOT NULL
                  AND r.status = 'failed'
                  AND r.complete_announcement_seen = TRUE
                  AND (
                    SELECT count(*)
                    FROM artifact_announcement_ledger ledger
                    WHERE ledger.train_job_id = r.train_job_id
                  ) = GREATEST(r.next_announcement_id - 1, 0)
                  AND NOT EXISTS (
                    SELECT 1 FROM artifact_announcement_ledger ledger
                    WHERE ledger.train_job_id = r.train_job_id
                      AND ledger.disposition = 'ready'
                      AND NOT EXISTS (
                        SELECT 1 FROM artifact_publication_receipts receipt
                        WHERE receipt.train_job_id = ledger.train_job_id
                          AND receipt.ledger_id = ledger.ledger_id
                          AND receipt.role = 'availability'
                          AND receipt.promotion_revision = 0
                      )
                  )
                  AND (
                    r.promoted_eval_job_id IS NULL
                    OR EXISTS (
                      SELECT 1 FROM eval_jobs promoted
                      JOIN artifact_publication_receipts receipt
                        ON receipt.train_job_id = promoted.train_job_id
                       AND receipt.ledger_id = promoted.ledger_id
                       AND receipt.role = 'promotion'
                       AND receipt.promotion_revision = r.promotion_revision
                      WHERE promoted.id = r.promoted_eval_job_id
                    )
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM eval_jobs j
                    WHERE j.train_job_id = t.id
                      AND j.status IN (
                        'pending', 'dispatching', 'submitted', 'blocked_budget'
                      )
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM eval_attempts a
                    JOIN eval_jobs j ON j.id = a.eval_job_id
                    WHERE j.train_job_id = t.id
                      AND a.status IN ('dispatching', 'submitted')
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM worker_attempts w
                    WHERE w.train_job_id = t.id
                      AND w.task_kind = 'eval'
                      AND w.status IN ('launching', 'running')
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_batches b
                    JOIN metric_streams s ON s.stream_id = b.stream_id
                    JOIN worker_attempts w ON w.attempt_id = s.attempt_id
                    WHERE w.train_job_id = t.id
                  )
                  AND t.live_publication_status IN ('complete', 'disabled')
                """
            )
            return cur.rowcount


def reconcile_definitive_non_acceptance(conn) -> int:
    """Close a finished stream that cannot establish goal acceptance.

    Current contracts reach this state after every acceptance attempt is rejected.
    Historical promotion-only contracts also need a bounded outcome once all of their
    durable evaluation evidence has been projected; otherwise they remain finalizing
    forever despite having no work left to perform.
    """

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_runs r
                SET outcome = 'not_accepted', status = 'finalizing',
                    updated_at = now(), error = NULL
                WHERE r.outcome IS NULL
                  AND COALESCE(r.contract_json->>'checkpoint_eval_backend', 'local') = 'modal'
                  AND r.complete_announcement_seen = TRUE
                  AND EXISTS (
                    SELECT 1 FROM train_jobs t
                    WHERE t.id = r.train_job_id AND t.status = 'finalizing'
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM eval_jobs j
                    WHERE j.train_job_id = r.train_job_id
                      AND j.status IN (
                        'pending', 'dispatching', 'submitted', 'blocked_budget', 'failed'
                      )
                  )
                  AND (
                    (
                      EXISTS (
                        SELECT 1 FROM eval_jobs j
                        WHERE j.train_job_id = r.train_job_id
                          AND j.purpose = 'acceptance'
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM eval_jobs j
                        WHERE j.train_job_id = r.train_job_id
                          AND j.purpose = 'acceptance'
                          AND COALESCE(j.decision_json->>'verdict', '') <> 'rejected'
                      )
                    )
                    OR (
                      NOT EXISTS (
                        SELECT 1 FROM eval_jobs j
                        WHERE j.train_job_id = r.train_job_id
                          AND j.purpose = 'acceptance'
                      )
                      AND EXISTS (
                        SELECT 1 FROM eval_jobs j
                        WHERE j.train_job_id = r.train_job_id
                          AND j.status = 'succeeded'
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM eval_jobs j
                        WHERE j.train_job_id = r.train_job_id
                          AND j.status = 'succeeded'
                          AND j.projected_at IS NULL
                      )
                    )
                  )
                """
            )
            return cur.rowcount


def reconcile_learner_stop_observation(conn) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs t
                SET learner_stop_observed_at = COALESCE(
                      t.learner_stop_observed_at,
                      (e.payload_json->>'observed_at')::timestamptz
                    )
                FROM attempt_events e
                JOIN worker_attempts a ON a.attempt_id = e.attempt_id
                WHERE a.train_job_id = t.id
                  AND e.event_type = 'learner_stop_observed'
                  AND t.learner_stop_observed_at IS NULL
                """
            )
            return cur.rowcount


def reconcile_stop_delivery_slo(conn) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_runs r
                SET stop_delivery_slo_met = CASE
                      WHEN c.acknowledged_at IS NOT NULL
                        THEN c.acknowledged_at <= r.acceptance_committed_at
                          + interval '5 seconds'
                      WHEN now() > r.acceptance_committed_at + interval '5 seconds'
                        THEN FALSE
                      ELSE NULL
                    END,
                    updated_at = now()
                FROM attempt_commands c
                WHERE r.outcome = 'accepted'
                  AND c.command_id =
                    'acceptance-stop:' || r.promoted_eval_job_id::text
                  AND r.stop_delivery_slo_met IS DISTINCT FROM CASE
                    WHEN c.acknowledged_at IS NOT NULL
                      THEN c.acknowledged_at <= r.acceptance_committed_at
                        + interval '5 seconds'
                    WHEN now() > r.acceptance_committed_at + interval '5 seconds'
                      THEN FALSE
                    ELSE NULL
                  END
                """
            )
            return cur.rowcount


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
            SELECT DISTINCT j.train_job_id
            FROM eval_jobs j
            JOIN eval_runs r ON r.train_job_id = j.train_job_id
            WHERE j.purpose = 'promotion' AND j.status = 'succeeded'
              AND r.status = 'finalizing'
            ORDER BY j.train_job_id
            """
        )
        train_job_ids = [int(row["train_job_id"]) for row in cur.fetchall()]
    updated = 0
    for train_job_id in train_job_ids:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT status, promotion_revision
                    FROM eval_runs
                    WHERE train_job_id = %(train_job_id)s
                    FOR UPDATE
                    """,
                    {"train_job_id": train_job_id},
                )
                eval_run = cur.fetchone()
                if not eval_run or str(eval_run["status"]) != "finalizing":
                    continue
                expected_revision = int(eval_run.get("promotion_revision") or 0)
                cur.execute(
                    """
                    SELECT j.*, t.train_config
                    FROM eval_jobs j JOIN train_jobs t ON t.id = j.train_job_id
                    WHERE j.train_job_id = %(train_job_id)s
                      AND j.purpose = 'promotion' AND j.status = 'succeeded'
                    ORDER BY j.id
                    """,
                    {"train_job_id": train_job_id},
                )
                candidates = [dict(row) for row in cur.fetchall()]
                try:
                    winner = max(candidates, key=promotion_candidate_key)
                except (KeyError, TypeError, ValueError) as exc:
                    cur.execute(
                        """
                        UPDATE eval_runs SET status = 'failed', error = %(error)s,
                          updated_at = now()
                        WHERE train_job_id = %(train_job_id)s
                          AND status = 'finalizing'
                          AND promotion_revision = %(expected_revision)s
                        """,
                        {
                            "error": f"promotion reconciliation failed: {exc}"[:4000],
                            "train_job_id": train_job_id,
                            "expected_revision": expected_revision,
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
                    "promotion_revision": expected_revision + 1,
                }
                cur.execute(
                    """
                    UPDATE eval_runs SET promoted_eval_job_id = %(job_id)s,
                      promotion_revision = promotion_revision + 1,
                      promotion_json = %(promotion)s, updated_at = now()
                    WHERE train_job_id = %(train_job_id)s
                      AND status = 'finalizing'
                      AND promotion_revision = %(expected_revision)s
                      AND promoted_eval_job_id IS DISTINCT FROM %(job_id)s
                    """,
                    {
                        "job_id": int(winner["id"]),
                        "promotion": json_arg(promotion),
                        "train_job_id": train_job_id,
                        "expected_revision": expected_revision,
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


def project_eval_results(
    conn, *, repo_root: Path, deadline_monotonic: float, limit: int = 100
) -> int:
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
              AND j.purpose IN ('promotion', 'acceptance')
              AND t.status = 'finalizing'
              AND t.live_publication_status <> 'failed'
              AND r.status = 'finalizing'
              AND j.projection_attempts < %(max_attempts)s
              AND (j.projection_next_retry_at IS NULL OR j.projection_next_retry_at <= now())
            ORDER BY COALESCE(j.projection_next_retry_at, j.created_at),
              j.train_job_id, j.checkpoint_step, j.stage_index
            LIMIT %(limit)s
            """,
            {
                "max_attempts": MAX_PROJECTION_ATTEMPTS,
                "limit": max(1, int(limit)),
            },
        )
        jobs = [dict(row) for row in cur.fetchall()]
    projected = 0
    for job in jobs:
        if deadline_monotonic - time.monotonic() < 2.0:
            break
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
        if str(job["train_config"].get("telemetry_transport") or "legacy_local") == (
            "neon_mailbox_v1"
        ):
            from rlab.telemetry_mailbox import enqueue_projection_payload

            projected += int(
                enqueue_projection_payload(conn, eval_job_id=int(job["id"]), payload=payload)
            )
            continue
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
        projected += int(error is None)
    return projected


def enqueue_missing_mailbox_projections(conn, *, limit: int = 100) -> int:
    """Recover a crash after accepting an eval but before creating its W&B frame."""

    from rlab.telemetry_mailbox import enqueue_projection_payload

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.*, t.train_config,
              (r.promoted_eval_job_id = j.id) AS canonical_promotion
            FROM eval_jobs j
            JOIN train_jobs t ON t.id = j.train_job_id
            JOIN eval_runs r ON r.train_job_id = j.train_job_id
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
                    "canonical_promotion": bool(job.get("canonical_promotion")),
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
            changed = cur.rowcount
            cur.execute(
                """
                UPDATE eval_runs r
                SET promoted_artifact_projected_at = now(), updated_at = now()
                FROM metric_streams s
                WHERE r.promoted_eval_job_id IS NOT NULL
                  AND s.stream_id =
                    'artifact-projection-' || r.promoted_eval_job_id::text
                  AND r.promoted_artifact_projection_enqueued_at IS NOT NULL
                  AND r.promoted_artifact_projected_at IS NULL
                  AND s.final_sequence IS NOT NULL
                  AND s.published_sequence >= s.final_sequence
                  AND NOT EXISTS (
                    SELECT 1 FROM metric_batches b WHERE b.stream_id = s.stream_id
                  )
                """
            )
            return changed + cur.rowcount


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
            WHERE r.complete_announcement_seen = TRUE
              AND t.telemetry_transport <> 'neon_mailbox_v1'
              AND (
                r.artifacts_projected_at IS NULL
                OR (
                  r.promoted_eval_job_id IS NOT NULL
                  AND r.promoted_artifact_projected_at IS NULL
                  AND r.promoted_artifact_projection_enqueued_at IS NULL
                )
              )
              AND t.status = 'finalizing'
              AND t.live_publication_status <> 'failed'
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
    if (
        promoted_ledger_id is not None
        and run.get("promoted_artifact_projected_at") is None
        and run.get("promoted_artifact_projection_enqueued_at") is None
    ):
        promoted_announcement = dict(run.get("promoted_announcement") or {})
        error = None
        mailbox_enqueued = False
        try:
            model_document = store.get_json(str(promoted_announcement["metadata_uri"]))
            model_metadata = (
                model_document_as_metadata(model_document)
                if promoted_announcement.get("recipe_uri")
                else model_document
            )
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
                **(
                    {
                        "recipe_uri": promoted_announcement["recipe_uri"],
                        "recipe_sha256": promoted_announcement["recipe_sha256"],
                    }
                    if promoted_announcement.get("recipe_uri")
                    else {}
                ),
                "checkpoint_step": checkpoint_step,
                "model_metadata": model_metadata,
                "artifact_aliases": [
                    "latest",
                    "promoted",
                    f"step-{checkpoint_step}",
                ],
            }
            if mailbox_transport and run.get("promoted_eval_job_id") is not None:
                from rlab.telemetry_mailbox import enqueue_projection_payload

                enqueue_projection_payload(
                    conn,
                    eval_job_id=int(run["promoted_eval_job_id"]),
                    payload=payload,
                    stream_kind="artifact",
                )
                mailbox_enqueued = True
            else:
                error = _execute_projection(
                    payload,
                    repo_root=repo_root,
                    deadline_monotonic=deadline_monotonic,
                    label=f"promoted-artifact-{train_job_id}-{promoted_ledger_id}",
                )
        with conn:
            with conn.cursor() as cur:
                if error is None and mailbox_enqueued:
                    cur.execute(
                        """
                        UPDATE eval_runs
                        SET promoted_artifact_projection_enqueued_at = now(),
                          artifact_projection_attempts = 0,
                          artifact_projection_next_retry_at = NULL,
                          error = NULL, updated_at = now()
                        WHERE train_job_id = %(id)s
                        """,
                        {"id": train_job_id},
                    )
                elif error is None:
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
            model_document = store.get_json(str(announcement["metadata_uri"]))
            model_metadata = (
                model_document_as_metadata(model_document)
                if announcement.get("recipe_uri")
                else model_document
            )
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
                **(
                    {
                        "recipe_uri": announcement["recipe_uri"],
                        "recipe_sha256": announcement["recipe_sha256"],
                    }
                    if announcement.get("recipe_uri")
                    else {}
                ),
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
    # The eval reconciler lock is session-scoped and must live on a direct
    # connection so process death releases it instead of poisoning a pool.
    conn = connect(database_url(use_direct=True))
    invoker = invoker or DefaultModalInvoker()
    store = store or ObjectStore(object_store_base_uri())
    try:
        if progress:
            progress("CHECKING EVALUATION", "Acquiring the Modal evaluation scheduler lock")
        if not _try_lock(conn):
            return {"status": "locked"}
        if progress:
            progress("POLLING EVALUATION", "Observing submitted Modal attempts and durable results")
        created_runs = ensure_eval_runs(conn)
        cancel_deadline = min(
            deadline_monotonic,
            time.monotonic() + CANCEL_ATTEMPT_BUDGET_SECONDS,
        )
        canceled_attempts = cancel_requested_attempts(
            conn, invoker, deadline_monotonic=cancel_deadline
        )
        poll_deadline = min(
            deadline_monotonic,
            time.monotonic() + ATTEMPT_POLL_BUDGET_SECONDS,
        )
        polled = poll_attempts(
            conn,
            store,
            invoker,
            config,
            deadline_monotonic=poll_deadline,
        )
        if progress:
            progress("INGESTING EVALUATION", "Reading checkpoint announcements and mailbox events")
        ingestion_deadline = min(
            deadline_monotonic,
            time.monotonic() + ANNOUNCEMENT_INGEST_BUDGET_SECONDS,
        )
        ingestion_errors: list[str] = []
        try:
            mailbox_ingested = ingest_mailbox_announcements(
                conn,
                store,
                deadline_monotonic=ingestion_deadline,
            )
        except DatabaseError:
            raise
        except Exception as exc:
            conn.rollback()
            mailbox_ingested = 0
            ingestion_errors.append(f"mailbox:{type(exc).__name__}:{exc}")
        try:
            ingested = ingest_announcements(
                conn,
                store,
                deadline_monotonic=ingestion_deadline,
            )
        except DatabaseError:
            raise
        except Exception as exc:
            conn.rollback()
            ingested = 0
            ingestion_errors.append(f"object_store:{type(exc).__name__}:{exc}")
        canceled_eval_state = reconcile_canceled_eval_state(conn)
        skipped_decisions = publish_skipped_decisions(conn, store)
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
        learner_stop_observations = reconcile_learner_stop_observation(conn)
        stop_delivery_slo = reconcile_stop_delivery_slo(conn)
        non_acceptance = reconcile_definitive_non_acceptance(conn)
        eval_run_failures = reconcile_eval_run_failures(conn)
        failed_terminalizations = terminalize_failed_eval_runs(conn)
        publication_finishing = reconcile_publication_finishing(conn)
        artifact_only_finalized = terminalize_artifact_only_runs(conn)
        finalized = terminalize_runs(conn)
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
        except DatabaseError:
            raise
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
            "canceled_eval_jobs": canceled_eval_state["jobs"],
            "canceled_eval_runs": canceled_eval_state["runs"],
            "canceled_eval_workers": canceled_eval_state["workers"],
            "ingested": ingested,
            "mailbox_ingested": mailbox_ingested,
            "ingestion_errors": ingestion_errors,
            "skipped_decisions": skipped_decisions,
            "polled": polled,
            "recovered_projections": recovered_projections,
            "promotions": promotions,
            "reconciled_promotions": reconciled_promotions,
            "dispatched": dispatched,
            "projected": projected,
            "projected_artifacts": projected_artifacts,
            "reconciled_projections": reconciled_projections,
            "learner_stop_observations": learner_stop_observations,
            "stop_delivery_slo": stop_delivery_slo,
            "non_acceptance": non_acceptance,
            "eval_run_failures": eval_run_failures,
            "failed_terminalizations": failed_terminalizations,
            "publication_finishing": publication_finishing,
            "artifact_only_finalized": artifact_only_finalized,
            "finalized": finalized,
            "finalization_failures": 0,
            "app_cleanup": app_cleanup,
        }
    finally:
        try:
            _unlock(conn)
        except Exception:
            pass
        conn.close()
