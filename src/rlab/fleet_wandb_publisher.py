from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from rlab.checkpoint_eval_worker import update_best_checkpoint_summary
from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.telemetry_mailbox import (
    REMOTE_CONFIRM_TIMEOUT_SECONDS,
    claim_run_metric_batches,
    decode_metric_batch,
    mark_submitted_batches,
    pending_metric_run_ids,
    release_metric_batch_claims,
    release_metric_batch_claims_by_owner,
    release_wandb_run_lock,
)
from rlab.job_queue import record_job_event
from rlab.modal_eval_storage import ObjectStore, object_store_base_uri
from rlab.policy_bundle import (
    CHECKPOINT_FILENAME,
    MODEL_FILENAME,
    RECIPE_FILENAME,
    load_policy_bundle,
    model_document_as_metadata,
)
from rlab.wandb_artifacts import artifact_collection_name
from rlab.wandb_publisher import (
    WandbArtifactAcknowledgmentTimeout,
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
    EVAL_FULL_DURATION_SECONDS,
    LEADER_CHECKPOINT_ACCEPTANCE_PASS,
    LEADER_CHECKPOINT_STEP,
)


SUMMARY_CURSOR_KEY = "_rlab_telemetry_cursors"
DEFAULT_BATCH_LIMIT = 20
ACTOR_POLL_SECONDS = 2.0
MAX_FINALIZATION_ATTEMPTS = 3
FINALIZATION_RETRY_DELAYS_SECONDS = (15, 30, 60)
ARTIFACT_VISIBILITY_RETRY_SECONDS = 5
WANDB_API_TIMEOUT_SECONDS = 30
ACTOR_LOCK_BUSY_EXIT_CODE = 75
ACTOR_SOURCE_MISMATCH_EXIT_CODE = 76
ACTOR_START_FAILED_EXIT_CODE = 77
TERMINAL_WANDB_STATES = frozenset({"finished", "crashed", "failed", "killed"})
SUPPORTED_FRAME_KINDS = {
    "history",
    "histogram",
    "checkpoint_eval",
    "checkpoint_preview",
    "projection",
}
_ACTOR_SOURCE_FINGERPRINT: str | None = None


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


class WandbArtifactVisibilityError(RuntimeError):
    pass


class WandbArtifactConflictError(InvalidTelemetryBatchError):
    pass


class WandbPublisherActorLockBusy(RuntimeError):
    pass


def _actor_state_path(train_job_id: int) -> Path:
    return Path.cwd() / "logs" / "fleet" / "wandb-actors" / f"train-{int(train_job_id)}.json"


def _write_actor_state(
    train_job_id: int,
    *,
    phase: str,
    session_started_at: float | None,
    lease_owner: str | None = None,
    progress_at: float | None = None,
    stage: str | None = None,
    claimed_batches: int = 0,
    completed_batches: int = 0,
    completed_frames: int = 0,
    close_started_at: float | None = None,
    expected_source_fingerprint: str | None = None,
    error: str | None = None,
) -> None:
    global _ACTOR_SOURCE_FINGERPRINT
    from rlab.fleet_service import controller_source_fingerprint

    if _ACTOR_SOURCE_FINGERPRINT is None:
        _ACTOR_SOURCE_FINGERPRINT = controller_source_fingerprint(Path.cwd())
    path = _actor_state_path(train_job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    now = time.time()
    temporary.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "pid": os.getpid(),
                "train_job_id": int(train_job_id),
                "phase": phase,
                "session_started_at": session_started_at,
                "lease_owner": lease_owner,
                "progress_at": now if progress_at is None else float(progress_at),
                "stage": stage or phase,
                "claimed_batches": max(0, int(claimed_batches)),
                "completed_batches": max(0, int(completed_batches)),
                "completed_frames": max(0, int(completed_frames)),
                "close_started_at": close_started_at,
                "updated_at": now,
                "source_fingerprint": _ACTOR_SOURCE_FINGERPRINT,
                "expected_source_fingerprint": expected_source_fingerprint,
                "error": error,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verified_json_object(store: ObjectStore, uri: str, expected_sha256: str) -> dict[str, Any]:
    payload = store.get_bytes(uri)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != str(expected_sha256):
        raise ValueError(f"artifact sidecar hash mismatch: {uri}")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"artifact sidecar must contain a JSON object: {uri}")
    return value


def _hydrate_artifact_projection_payload(payload: dict[str, Any]) -> dict[str, Any]:
    schema = str(payload.get("artifact_publication_schema") or "")
    if schema not in {"v2", "v3"}:
        return payload
    store = ObjectStore(object_store_base_uri())
    document = _verified_json_object(
        store,
        str(payload["metadata_uri"]),
        str(payload["metadata_sha256"]),
    )
    members = None
    if payload.get("recipe_uri"):
        recipe_document = _verified_json_object(
            store,
            str(payload["recipe_uri"]),
            str(payload["recipe_sha256"]),
        )
        metadata = model_document_as_metadata(document)
        if schema == "v3":
            if payload.get("content_mode") != "wandb_native_v1":
                raise ValueError("v3 artifact publication content mode is invalid")
            model_bytes = store.get_bytes(str(payload["checkpoint_uri"]))
            if hashlib.sha256(model_bytes).hexdigest() != str(payload["checkpoint_sha256"]):
                raise ValueError("artifact checkpoint hash mismatch")
            with tempfile.TemporaryDirectory(prefix="rlab-artifact-verify-") as temporary:
                root = Path(temporary)
                encoded_model = (
                    json.dumps(
                        document,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                encoded_recipe = (
                    json.dumps(
                        recipe_document,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                (root / CHECKPOINT_FILENAME).write_bytes(model_bytes)
                (root / MODEL_FILENAME).write_bytes(encoded_model)
                (root / RECIPE_FILENAME).write_bytes(encoded_recipe)
                load_policy_bundle(root)
                members = {
                    CHECKPOINT_FILENAME: {
                        "sha256": str(payload["checkpoint_sha256"]),
                        "size_bytes": len(model_bytes),
                    },
                    MODEL_FILENAME: {
                        "sha256": str(payload["metadata_sha256"]),
                        "size_bytes": len(encoded_model),
                    },
                    RECIPE_FILENAME: {
                        "sha256": str(payload["recipe_sha256"]),
                        "size_bytes": len(encoded_recipe),
                    },
                }
    else:
        metadata = document
    if not isinstance(metadata.get("training_metadata"), dict):
        raise ValueError("artifact sidecar is missing training_metadata")
    return {**payload, "model_metadata": metadata, "artifact_members": members}


def _repair_artifact_projection_identity(
    payload: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Any]:
    if str(payload.get("artifact_publication_schema") or "") not in {"v2", "v3"}:
        return payload
    payload_config = dict(payload.get("train_config") or {})
    for key in ("wandb_run_id", "run_name", "wandb_group", "wandb_tags"):
        value = train_config.get(key)
        if value not in (None, "", []):
            payload_config[key] = value
    return {**payload, "train_config": payload_config}


def _hydrate_receipt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if str(payload.get("artifact_publication_schema") or "") == "v3":
        return _hydrate_artifact_projection_payload(payload)
    return payload


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
    remote = wandb.Api(timeout=WANDB_API_TIMEOUT_SECONDS).run(f"{entity}/{project}/{run_id}")
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
    return wandb.Api(timeout=WANDB_API_TIMEOUT_SECONDS).run(
        f"{entity}/{project}/{train_config['wandb_run_id']}"
    )


def _artifact_aliases(artifact: Any) -> set[str]:
    return {
        str(getattr(alias, "alias", alias)) for alias in (getattr(artifact, "aliases", ()) or ())
    }


def _artifact_version_number(artifact: Any) -> int:
    raw = str(getattr(artifact, "version", "") or "")
    if not raw:
        qualified = str(getattr(artifact, "qualified_name", "") or "")
        raw = qualified.rsplit(":", 1)[-1] if ":" in qualified else ""
    return int(raw[1:]) if raw.startswith("v") and raw[1:].isdigit() else -1


def _artifact_qualified_ref(artifact: Any) -> str:
    qualified = str(getattr(artifact, "qualified_name", "") or "")
    if qualified and ":" in qualified:
        return qualified
    version = str(getattr(artifact, "version", "") or "")
    name = str(getattr(artifact, "name", "") or "")
    return f"{name}:{version}" if name and version else ""


def _artifact_collection(artifact: Any) -> str:
    qualified = _artifact_qualified_ref(artifact)
    if qualified:
        return qualified.rsplit("/", 1)[-1].rsplit(":", 1)[0]
    return str(getattr(artifact, "name", "") or "").rsplit("/", 1)[-1]


def _artifact_receipts_from_remote(
    remote: Any,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not payloads:
        return []
    logged = getattr(remote, "logged_artifacts", None)
    if not callable(logged):
        raise WandbArtifactVisibilityError("W&B run does not expose logged artifacts")
    artifacts = list(logged())
    receipts: list[dict[str, Any]] = []
    for payload in payloads:
        train_config = dict(payload["train_config"])
        kind = str(payload["artifact_kind"])
        collection = artifact_collection_name(
            kind,
            run_id=str(train_config["wandb_run_id"]),
        )
        required_aliases = {str(alias) for alias in payload.get("artifact_aliases") or ()}
        matches: list[Any] = []
        alias_conflicts: list[Any] = []
        for artifact in artifacts:
            if str(getattr(artifact, "type", "") or "") != "model":
                continue
            if _artifact_collection(artifact) != collection:
                continue
            metadata = dict(getattr(artifact, "metadata", {}) or {})
            publication_schema = str(payload.get("artifact_publication_schema") or "v2")
            aliases_match = required_aliases.issubset(_artifact_aliases(artifact))
            expected = {
                "artifact_publication_schema": publication_schema,
                "train_job_id": int(payload["train_job_id"]),
                "ledger_id": int(payload["ledger_id"]),
                "artifact_kind": kind,
                "announcement_sha256": str(payload["announcement_sha256"]),
                "checkpoint_step": int(payload["checkpoint_step"]),
                "checkpoint_sha256": str(payload["checkpoint_sha256"]),
                "artifact_storage_uri": str(payload["checkpoint_uri"]),
                "metadata_uri": str(payload["metadata_uri"]),
                "metadata_sha256": str(payload["metadata_sha256"]),
            }
            if any(str(metadata.get(key)) != str(value) for key, value in expected.items()):
                if aliases_match:
                    alias_conflicts.append(artifact)
                continue
            if publication_schema == "v3" and (
                metadata.get("content_mode") != "wandb_native_v1"
                or metadata.get("artifact_members") != payload.get("artifact_members")
            ):
                if aliases_match:
                    alias_conflicts.append(artifact)
                continue
            # W&B content-deduplicates availability and promotion publications of
            # the same immutable checkpoint into one artifact version. The remote
            # version can therefore carry either stream's role metadata. Ledger
            # identity, hashes, sidecars, and the role-specific aliases are the
            # durable proof; the receipt records the distinct local stream role.
            if payload.get("recipe_uri") and (
                str(metadata.get("recipe_uri")) != str(payload["recipe_uri"])
                or str(metadata.get("recipe_sha256")) != str(payload["recipe_sha256"])
            ):
                continue
            if not aliases_match:
                continue
            if _artifact_version_number(artifact) < 0 or not _artifact_qualified_ref(artifact):
                continue
            matches.append(artifact)
        if not matches:
            if alias_conflicts:
                raise WandbArtifactConflictError(
                    "W&B artifact alias is already bound to conflicting immutable content: "
                    f"train/{int(payload['train_job_id'])} ledger/{int(payload['ledger_id'])}"
                )
            raise WandbArtifactVisibilityError(
                "W&B cursor is visible but artifact API membership is pending: "
                f"train/{int(payload['train_job_id'])} ledger/{int(payload['ledger_id'])} "
                f"role/{payload['publication_role']} revision/{int(payload.get('promotion_revision') or 0)}"
            )
        selected = max(matches, key=_artifact_version_number)
        version_number = _artifact_version_number(selected)
        receipts.append(
            {
                "train_job_id": int(payload["train_job_id"]),
                "ledger_id": int(payload["ledger_id"]),
                "role": str(payload["publication_role"]),
                "promotion_revision": int(payload.get("promotion_revision") or 0),
                "artifact_kind": kind,
                "checkpoint_step": int(payload["checkpoint_step"]),
                "checkpoint_sha256": str(payload["checkpoint_sha256"]),
                "checkpoint_uri": str(payload["checkpoint_uri"]),
                "metadata_uri": str(payload["metadata_uri"]),
                "metadata_sha256": str(payload["metadata_sha256"]),
                "recipe_uri": payload.get("recipe_uri"),
                "recipe_sha256": payload.get("recipe_sha256"),
                "announcement_sha256": str(payload["announcement_sha256"]),
                "collection_name": collection,
                "artifact_version": f"v{version_number}",
                "artifact_ref": _artifact_qualified_ref(selected),
                "stream_id": str(payload["publication_stream_id"]),
                "expected_aliases": sorted(required_aliases),
            }
        )
    return receipts


def _publication_is_pristine(conn, run: dict[str, Any]) -> bool:
    """Return whether durable state permits first creation of the assigned W&B id."""

    if not bool((run.get("train_config") or {}).get("wandb", False)):
        return False
    if str(run.get("wandb_url") or "").strip():
        return False
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
              NOT EXISTS (
                SELECT 1
                FROM metric_streams s
                JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                WHERE a.train_job_id = %(train_job_id)s
                  AND (s.submitted_sequence <> 0 OR s.published_sequence <> 0)
              ) AS streams_unpublished,
              NOT EXISTS (
                SELECT 1
                FROM metric_batches b
                JOIN metric_streams s ON s.stream_id = b.stream_id
                JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                WHERE a.train_job_id = %(train_job_id)s
                  AND b.submitted_at IS NOT NULL
              ) AS batches_unsubmitted
            """,
            {"train_job_id": int(run["id"])},
        )
        evidence = cur.fetchone() or {}
    return bool(evidence.get("streams_unpublished")) and bool(evidence.get("batches_unsubmitted"))


def _remote_has_promoted_artifact(
    remote,
    checkpoint_sha256: str,
    promotion_revision: int,
) -> bool:
    logged = getattr(remote, "logged_artifacts", None)
    if not callable(logged):
        return False
    for artifact in logged():
        aliases = {str(value) for value in (getattr(artifact, "aliases", ()) or ())}
        metadata = dict(getattr(artifact, "metadata", {}) or {})
        if (
            "promoted" in aliases
            and str(metadata.get("checkpoint_sha256") or "") == str(checkpoint_sha256)
            and str(metadata.get("publication_role") or "") == "promotion"
            and int(metadata.get("promotion_revision") or 0) == int(promotion_revision)
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
    schema_version = int(_train_config(run).get("metrics_schema_version", 4) or 4)
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
            EVAL_ACCEPTANCE_DURATION_SECONDS: float(
                raw_metrics.get("_acceptance_duration_seconds")
                or raw_metrics.get("eval/full/duration/seconds")
                or 0.0
            ),
        }
    )
    if schema_version == 4:
        summary[EVAL_ACCEPTANCE_FAILURE_COUNT] = int(aggregates.get("failure_count") or 0)
        summary[EVAL_FULL_CHECKPOINT_STEP] = checkpoint_step
    for key, value in raw_metrics.items():
        if (
            str(key).startswith("eval/full/")
            and isinstance(value, int | float)
            and not isinstance(value, bool)
            and not (
                schema_version >= 5
                and (
                    str(key).startswith("eval/full/outcome/")
                    or str(key) == EVAL_FULL_DURATION_SECONDS
                )
            )
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
        metrics_schema_version=schema_version,
        include_completion=schema_version == 4,
    )
    summary.update(leader.summary)
    if schema_version == 4:
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
            candidate,
            str(promotion.get("checkpoint_sha256") or ""),
            int(run.get("promotion_revision") or 0),
        ):
            failures.append("artifact")
    return failures


def finalize_finishing_run(
    conn,
    train_job_id: int,
    *,
    progress: Callable[..., None] | None = None,
) -> bool:
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
                SELECT t.*, COALESCE(r.outcome, 'unknown') AS outcome,
                  r.promotion_json, COALESCE(r.promotion_revision, 0) AS promotion_revision
                FROM train_jobs t
                LEFT JOIN eval_runs r ON r.train_job_id = t.id
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
        pristine = _publication_is_pristine(conn, run)
        if pristine:
            remote_cursors: dict[str, int] = {}
            failures = ["remote_run"]
        else:
            remote = _wandb_api_run(train_config)
            remote_summary = dict(remote.summary)
            remote_cursors = _cursor_mapping(remote_summary.get(SUMMARY_CURSOR_KEY) or {})
            failures = _wandb_finalization_failures(run, remote, expected)
        wandb_url: str | None = None
        if pristine or failures:
            projector = WandbProjector.resume(
                train_config,
                allow_create=pristine,
                update_finish_state=True,
            )
            wandb_url = str(getattr(projector.run, "url", "") or "") or None
            projector.run.summary[SUMMARY_CURSOR_KEY] = {
                **remote_cursors,
                **expected,
            }
            for key, value in _canonical_goal_summary(run).items():
                projector.run.summary[key] = value
            _close_projector_with_heartbeat(
                projector,
                progress=progress,
                claimed_batches=0,
                completed_batches=0,
                completed_frames=0,
            )
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                remote = _wandb_api_run(train_config)
                failures = _wandb_finalization_failures(run, remote, expected)
                _report_actor_progress(
                    progress,
                    "finalization_verified",
                    claimed_batches=0,
                )
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
                        live_publication_next_retry_at = NULL,
                        wandb_url = COALESCE(wandb_url, %(wandb_url)s)
                    WHERE id = %(id)s AND live_publication_status = 'finishing'
                    """,
                    {"id": int(train_job_id), "wandb_url": wandb_url},
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
                          ELSE now() + (%(retry_delay)s * interval '1 second') END
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


def _durable_cursor_floor(conn, train_job_id: int) -> dict[str, int]:
    """Return cursors whose W&B sessions closed successfully for one run."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT s.stream_id,
              GREATEST(s.submitted_sequence, s.published_sequence) AS sequence
            FROM metric_streams s
            JOIN worker_attempts a ON a.attempt_id = s.attempt_id
            WHERE a.train_job_id = %(train_job_id)s
              AND GREATEST(s.submitted_sequence, s.published_sequence) > 0
            """,
            {"train_job_id": int(train_job_id)},
        )
        rows = cur.fetchall()
    return {str(row["stream_id"]): int(row["sequence"]) for row in rows}


def _merge_cursor_mappings(*mappings: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for mapping in mappings:
        for stream_id, sequence in mapping.items():
            merged[str(stream_id)] = max(
                int(merged.get(str(stream_id), 0)),
                int(sequence),
            )
    return merged


def _confirmation_is_stalled(
    batches: list[dict[str, Any]],
    remote: WandbPublicationState,
) -> bool:
    try:
        _raise_for_stalled_confirmations(batches, remote)
    except WandbCursorConfirmationError:
        return True
    return False


def _reconcile_terminal_artifact_publication(conn, train_job_id: int) -> bool:
    """Complete publication-only recovery without changing a terminal run outcome."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs t
                SET live_publication_status = 'complete',
                    live_publication_attempts = 0,
                    live_publication_error = NULL,
                    live_publication_next_retry_at = NULL
                WHERE t.id = %(train_job_id)s
                  AND t.status IN ('succeeded', 'failed', 'canceled', 'finalization_failed')
                  AND t.live_publication_status IN ('pending', 'live')
                  AND EXISTS (
                    SELECT 1 FROM metric_streams artifact_stream
                    JOIN worker_attempts artifact_attempt
                      ON artifact_attempt.attempt_id = artifact_stream.attempt_id
                    WHERE artifact_attempt.train_job_id = t.id
                      AND (
                        artifact_stream.stream_id LIKE 'artifact-v2-%%'
                        OR artifact_stream.stream_id LIKE 'artifact-v3-%%'
                      )
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
                """,
                {"train_job_id": int(train_job_id)},
            )
            return cur.rowcount == 1


def _submitted_at(value: object) -> datetime | None:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def _artifact_visibility_is_propagating(
    batches: list[dict[str, Any]],
    *,
    now: datetime | None = None,
) -> bool:
    submitted = [
        timestamp
        for batch in batches
        if (timestamp := _submitted_at(batch.get("submitted_at"))) is not None
    ]
    if not submitted:
        return False
    age = max(0.0, ((now or datetime.now(UTC)) - min(submitted)).total_seconds())
    return age < REMOTE_CONFIRM_TIMEOUT_SECONDS


def _artifact_identity(payload: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(payload["train_job_id"]),
        int(payload["ledger_id"]),
        str(payload["publication_role"]),
        int(payload.get("promotion_revision") or 0),
    )


def _mark_artifact_acknowledgment_pending(conn, batch_id: int) -> None:
    """Persist the start of an indeterminate artifact-acknowledgment window."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE metric_batches
                SET submitted_at = COALESCE(submitted_at, clock_timestamp()),
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    last_error = 'W&B artifact acknowledgment pending'
                WHERE id = %(batch_id)s
                """,
                {"batch_id": int(batch_id)},
            )


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
    artifact_receipts: list[dict[str, Any]] | None = None,
) -> None:
    grouped: dict[str, int] = {}
    batch_ids: list[int] = []
    for batch in batches:
        stream_id = str(batch["stream_id"])
        grouped[stream_id] = max(grouped.get(stream_id, 0), int(batch["batch_sequence"]))
        batch_ids.append(int(batch["id"]))
    with conn:
        with conn.cursor() as cur:
            for stream_id, sequence in grouped.items():
                cur.execute(
                    """
                    UPDATE metric_streams
                    SET submitted_sequence = GREATEST(submitted_sequence, %(sequence)s),
                        updated_at = now()
                    WHERE stream_id = %(stream_id)s
                    """,
                    {"stream_id": stream_id, "sequence": sequence},
                )
            for receipt in artifact_receipts or ():
                cur.execute(
                    """
                    INSERT INTO artifact_publication_receipts (
                      train_job_id, ledger_id, role, promotion_revision,
                      disposition, artifact_kind, checkpoint_step,
                      checkpoint_sha256, checkpoint_uri, metadata_uri,
                      metadata_sha256, recipe_uri, recipe_sha256,
                      announcement_sha256, collection_name, artifact_version,
                      artifact_ref, stream_id, expected_aliases
                    ) VALUES (
                      %(train_job_id)s, %(ledger_id)s, %(role)s, %(promotion_revision)s,
                      'confirmed', %(artifact_kind)s, %(checkpoint_step)s,
                      %(checkpoint_sha256)s, %(checkpoint_uri)s, %(metadata_uri)s,
                      %(metadata_sha256)s, %(recipe_uri)s, %(recipe_sha256)s,
                      %(announcement_sha256)s, %(collection_name)s, %(artifact_version)s,
                      %(artifact_ref)s, %(stream_id)s, %(expected_aliases)s
                    )
                    ON CONFLICT (train_job_id, ledger_id, role, promotion_revision)
                    DO NOTHING
                    """,
                    receipt,
                )
            if batch_ids:
                cur.execute(
                    """
                    UPDATE metric_batches
                    SET wandb_confirmed_at = COALESCE(wandb_confirmed_at, now()),
                        lease_owner = NULL,
                        lease_expires_at = NULL,
                        last_error = NULL
                    WHERE id = ANY(%(ids)s)
                    """,
                    {"ids": batch_ids},
                )
            cur.execute(
                """
                UPDATE train_jobs t
                SET live_publication_status = CASE
                      WHEN NOT EXISTS (
                        SELECT 1 FROM metric_batches b
                        JOIN metric_streams s ON s.stream_id = b.stream_id
                        JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                        WHERE a.train_job_id = t.id
                          AND b.wandb_confirmed_at IS NULL
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM metric_streams s
                        JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                        WHERE a.train_job_id = t.id
                          AND (s.final_sequence IS NULL
                               OR s.submitted_sequence < s.final_sequence)
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
                    payload = frame.get("payload", {})
                    if str(payload.get("projection_kind") or "evaluation") != "evaluation":
                        continue
                    eval_job_id = payload.get("eval_job_id")
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


def _report_actor_progress(
    progress: Callable[..., None] | None,
    stage: str,
    *,
    claimed_batches: int,
    completed_batches: int = 0,
    completed_frames: int = 0,
    close_started_at: float | None = None,
) -> None:
    if progress is not None:
        progress(
            stage=stage,
            claimed_batches=max(0, int(claimed_batches)),
            completed_batches=max(0, int(completed_batches)),
            completed_frames=max(0, int(completed_frames)),
            close_started_at=close_started_at,
        )


def _close_projector_with_heartbeat(
    projector: WandbProjector,
    *,
    progress: Callable[..., None] | None,
    claimed_batches: int,
    completed_batches: int,
    completed_frames: int,
    heartbeat_seconds: float = 10.0,
) -> None:
    close_started_at = time.time()
    stopped = threading.Event()

    def report_close_progress() -> None:
        _report_actor_progress(
            progress,
            "session_closing",
            claimed_batches=claimed_batches,
            completed_batches=completed_batches,
            completed_frames=completed_frames,
            close_started_at=close_started_at,
        )

    def heartbeat() -> None:
        while not stopped.wait(max(float(heartbeat_seconds), 0.1)):
            report_close_progress()

    report_close_progress()
    thread = threading.Thread(
        target=heartbeat,
        name="rlab-wandb-close-heartbeat",
        daemon=True,
    )
    thread.start()
    try:
        projector.close()
    finally:
        stopped.set()
        thread.join(timeout=1.0)


def publish_claimed_run(
    conn,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
    *,
    progress: Callable[..., None] | None = None,
) -> int:
    train_config = _train_config(run)
    decoded: dict[int, list[dict[str, Any]]] = {}
    decoded_frames = 0
    for batch in batches:
        try:
            frames = decode_metric_batch(
                bytes(batch["payload"]),
                schema_version=int(train_config.get("metrics_schema_version", 4) or 4),
            )
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
        decoded_frames += len(frames)
        _report_actor_progress(
            progress,
            "decoded",
            claimed_batches=len(batches),
            completed_batches=len(decoded),
            completed_frames=decoded_frames,
        )
    pristine = _publication_is_pristine(conn, run)
    remote_state = (
        WandbPublicationState(state="", cursors={}, step_max=None)
        if pristine
        else _remote_publication_state(train_config)
    )
    remote = remote_state.cursors
    _report_actor_progress(
        progress,
        "remote_inspected",
        claimed_batches=len(batches),
        completed_frames=decoded_frames,
    )
    durable_floor = _durable_cursor_floor(conn, int(run["id"]))
    confirmed, awaiting_confirmation, unpublished = _partition_batches(batches, remote)
    if confirmed:
        artifact_batches: list[dict[str, Any]] = []
        cursor_only_batches: list[dict[str, Any]] = []
        for batch in confirmed:
            if any(
                str(frame.get("kind") or "") == "projection"
                and str(frame.get("payload", {}).get("artifact_publication_schema") or "")
                in {"v2", "v3"}
                for frame in decoded[int(batch["id"])]
            ):
                artifact_batches.append(batch)
            else:
                cursor_only_batches.append(batch)
        if cursor_only_batches:
            _record_committed_effects(
                conn,
                run=run,
                batches=cursor_only_batches,
                decoded=decoded,
                wandb_url=None,
            )
            _report_actor_progress(
                progress,
                "cursor_confirmations_committed",
                claimed_batches=len(batches),
                completed_batches=len(cursor_only_batches),
                completed_frames=decoded_frames,
            )
        artifact_payloads = [
            _hydrate_receipt_payload(
                _repair_artifact_projection_identity(dict(frame["payload"]), train_config)
            )
            for batch in artifact_batches
            for frame in decoded[int(batch["id"])]
            if str(frame.get("kind") or "") == "projection"
            and str(frame.get("payload", {}).get("artifact_publication_schema") or "")
            in {"v2", "v3"}
        ]
        artifact_receipts = []
        if artifact_payloads:
            artifact_receipts = _artifact_receipts_from_remote(
                _wandb_api_run(train_config),
                artifact_payloads,
            )
            _report_actor_progress(
                progress,
                "artifact_membership_verified",
                claimed_batches=len(batches),
                completed_batches=len(cursor_only_batches),
                completed_frames=decoded_frames,
            )
        _record_committed_effects(
            conn,
            run=run,
            batches=artifact_batches,
            decoded=decoded,
            wandb_url=None,
            artifact_receipts=artifact_receipts,
        )
        _report_actor_progress(
            progress,
            "confirmed_committed",
            claimed_batches=len(batches),
            completed_batches=len(confirmed),
            completed_frames=decoded_frames,
        )
        run["live_publication_attempts"] = 0
    regressed_floor = {
        stream_id: sequence
        for stream_id, sequence in durable_floor.items()
        if int(remote.get(stream_id, 0)) < int(sequence)
    }
    reassert_cursor_floor = bool(regressed_floor) and _confirmation_is_stalled(
        awaiting_confirmation,
        remote_state,
    )
    if awaiting_confirmation and not reassert_cursor_floor:
        _raise_for_stalled_confirmations(awaiting_confirmation, remote_state)
        mark_submitted_batches(conn, awaiting_confirmation)
    wandb_url: str | None = None
    adopted_artifacts: set[tuple[int, int, str, int]] = set()
    artifact_api_run = None
    for batch in unpublished:
        for frame in decoded[int(batch["id"])]:
            if str(frame.get("kind") or "") != "projection":
                continue
            raw_payload = dict(frame["payload"])
            publication_schema = str(raw_payload.get("artifact_publication_schema") or "")
            if publication_schema not in {"v2", "v3"} or pristine:
                continue
            payload = _repair_artifact_projection_identity(raw_payload, train_config)
            payload = _hydrate_artifact_projection_payload(payload)
            if artifact_api_run is None:
                artifact_api_run = _wandb_api_run(train_config)
            try:
                _artifact_receipts_from_remote(artifact_api_run, [payload])
            except WandbArtifactVisibilityError:
                if _artifact_visibility_is_propagating([batch]):
                    raise
            else:
                adopted_artifacts.add(_artifact_identity(payload))
            _report_actor_progress(
                progress,
                "artifact_membership_inspected",
                claimed_batches=len(batches),
                completed_frames=decoded_frames,
            )
    if unpublished or reassert_cursor_floor:
        session_step_max = remote_state.step_max
        expected: dict[str, int] = {}
        for row in unpublished:
            stream_id = str(row["stream_id"])
            expected[stream_id] = max(expected.get(stream_id, 0), int(row["batch_sequence"]))
        projector = WandbProjector.resume(
            train_config,
            allow_create=pristine,
            update_finish_state=str(run.get("status") or "")
            in {"succeeded", "failed", "finalization_failed", "canceled"},
        )
        _report_actor_progress(
            progress,
            "session_open",
            claimed_batches=len(batches),
            completed_frames=decoded_frames,
        )
        try:
            wandb_url = str(getattr(projector.run, "url", "") or "") or None
            args = SimpleNamespace(**train_config) if unpublished else None
            config = (
                resolve_env_config(env_config_from_args(args, include_states=True))
                if args is not None
                else None
            )
            projected_batches = 0
            projected_frames = 0
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
                        payload = _repair_artifact_projection_identity(payload, train_config)
                        payload = _hydrate_artifact_projection_payload(payload)
                        publication_schema = str(payload.get("artifact_publication_schema") or "")
                        adopted = (
                            publication_schema in {"v2", "v3"}
                            and _artifact_identity(payload) in adopted_artifacts
                        )
                        if not adopted:
                            try:
                                project_payload_to_run(
                                    projector.run,
                                    payload,
                                    allow_artifact_references=True,
                                    artifact_wait_timeout_seconds=WANDB_API_TIMEOUT_SECONDS,
                                )
                            except WandbArtifactAcknowledgmentTimeout as exc:
                                exc.batch_id = int(batch["id"])
                                exc.artifact_identity = _artifact_identity(payload)
                                raise
                    else:
                        assert args is not None and config is not None
                        _publish_frame(
                            projector.run,
                            {
                                "kind": kind,
                                "payload_json": __import__("json").dumps(payload),
                            },
                            args=args,
                            config=config,
                        )
                    projected_frames += 1
                    _report_actor_progress(
                        progress,
                        "frame_projected",
                        claimed_batches=len(batches),
                        completed_batches=projected_batches,
                        completed_frames=projected_frames,
                    )
                projected_batches += 1
                _report_actor_progress(
                    progress,
                    "batch_projected",
                    claimed_batches=len(batches),
                    completed_batches=projected_batches,
                    completed_frames=projected_frames,
                )
            merged = _merge_cursor_mappings(remote, durable_floor, expected)
            projector.run.summary[SUMMARY_CURSOR_KEY] = merged
            if session_step_max is not None:
                summary_step: int | float = session_step_max
                if session_step_max.is_integer():
                    summary_step = int(session_step_max)
                projector.run.summary["global_step"] = {"max": summary_step}
        finally:
            _close_projector_with_heartbeat(
                projector,
                progress=progress,
                claimed_batches=len(batches),
                completed_batches=len(unpublished),
                completed_frames=decoded_frames,
            )
        _report_actor_progress(
            progress,
            "session_closed",
            claimed_batches=len(batches),
            completed_batches=len(unpublished),
            completed_frames=decoded_frames,
        )
        if unpublished:
            mark_submitted_batches(conn, unpublished)
        if reassert_cursor_floor:
            mark_submitted_batches(
                conn,
                awaiting_confirmation,
                refresh_submitted_at=True,
            )
        _report_actor_progress(
            progress,
            "submission_committed",
            claimed_batches=len(batches),
            completed_batches=len(unpublished) + len(confirmed),
            completed_frames=decoded_frames,
        )
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


def _drain_claim(
    conn,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
    *,
    progress: Callable[..., None] | None = None,
) -> int:
    try:
        return publish_claimed_run(conn, run, batches, progress=progress)
    except Exception as exc:
        acknowledgment_pending = isinstance(exc, WandbArtifactAcknowledgmentTimeout)
        acknowledgment_batch = next(
            (batch for batch in batches if int(batch["id"]) == int(getattr(exc, "batch_id", -1))),
            None,
        )
        acknowledgment_propagating = bool(
            acknowledgment_pending
            and acknowledgment_batch is not None
            and (
                _submitted_at(acknowledgment_batch.get("submitted_at")) is None
                or _artifact_visibility_is_propagating([acknowledgment_batch])
            )
        )
        if acknowledgment_batch is not None:
            _mark_artifact_acknowledgment_pending(
                conn,
                int(acknowledgment_batch["id"]),
            )
        release_metric_batch_claims(conn, batches, error=repr(exc))
        current_attempts = int(run.get("live_publication_attempts") or 0)
        visibility_propagating = acknowledgment_propagating or (
            isinstance(exc, WandbArtifactVisibilityError)
            and _artifact_visibility_is_propagating(batches)
        )
        attempts = current_attempts if visibility_propagating else current_attempts + 1
        invalid_batch = isinstance(exc, InvalidTelemetryBatchError)
        run_status = str(run.get("status") or "")
        bounded_publication = run_status in {
            "finalizing",
            "succeeded",
            "failed",
            "canceled",
            "finalization_failed",
        }
        terminal = (
            bounded_publication
            and not visibility_propagating
            and (invalid_batch or attempts >= MAX_FINALIZATION_ATTEMPTS)
        )
        retry_delay = (
            ARTIFACT_VISIBILITY_RETRY_SECONDS
            if visibility_propagating
            else (
                FINALIZATION_RETRY_DELAYS_SECONDS[
                    min(attempts - 1, len(FINALIZATION_RETRY_DELAYS_SECONDS) - 1)
                ]
                if bounded_publication
                else 30
            )
        )
        error = (
            str(exc)
            if isinstance(
                exc,
                (
                    WandbCursorConfirmationError,
                    WandbArtifactVisibilityError,
                    WandbArtifactAcknowledgmentTimeout,
                ),
            )
            else repr(exc)
        )[:4000]
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
                          ELSE now() + (%(retry_delay)s * interval '1 second') END
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
                        "visibility_propagating": visibility_propagating,
                        "artifact_acknowledgment_pending": acknowledgment_pending,
                    },
                )
        raise
    finally:
        release_wandb_run_lock(conn, int(run["id"]))


def recover_stalled_actor_claim(
    conn,
    *,
    train_job_id: int,
    lease_owner: str,
    error: str,
    watchdog: str = "no_progress",
) -> int:
    released = release_metric_batch_claims_by_owner(
        conn,
        train_job_id=int(train_job_id),
        owner=lease_owner,
        error=error,
    )
    if not released:
        return 0
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, live_publication_attempts
                FROM train_jobs
                WHERE id = %(train_job_id)s
                FOR UPDATE
                """,
                {"train_job_id": int(train_job_id)},
            )
            row = cur.fetchone()
            if not row:
                return released
            attempts = int(row.get("live_publication_attempts") or 0) + 1
            bounded = str(row.get("status") or "") in {
                "finalizing",
                "succeeded",
                "failed",
                "canceled",
                "finalization_failed",
            }
            exhausted = bounded and attempts >= MAX_FINALIZATION_ATTEMPTS
            cur.execute(
                """
                UPDATE train_jobs
                SET live_publication_status = %(publication_status)s,
                    live_publication_attempts = %(attempts)s,
                    live_publication_error = %(error)s,
                    live_publication_next_retry_at = CASE
                      WHEN %(exhausted)s THEN NULL ELSE now() END
                WHERE id = %(train_job_id)s
                """,
                {
                    "train_job_id": int(train_job_id),
                    "publication_status": "failed" if exhausted else "pending",
                    "attempts": attempts,
                    "error": str(error)[:4000],
                    "exhausted": exhausted,
                },
            )
            record_job_event(
                conn,
                job_id=int(train_job_id),
                event_type=(
                    "live_publication_failed"
                    if exhausted
                    else (
                        "wandb_session_close_timeout"
                        if watchdog == "close_timeout"
                        else "live_publication_retry"
                    )
                ),
                message=str(error)[:4000],
                metadata={
                    "attempts": attempts,
                    "terminal": False,
                    "watchdog": watchdog,
                    "released_batches": released,
                    "lease_owner": lease_owner,
                },
            )
    return released


def drain_once(
    conn,
    *,
    owner: str | None = None,
    limit: int = DEFAULT_BATCH_LIMIT,
    exclude_train_job_ids: tuple[int, ...] = (),
    train_job_id: int | None = None,
    progress: Callable[..., None] | None = None,
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
            _reconcile_terminal_artifact_publication(conn, int(train_job_id))
            return int(
                finalize_finishing_run(
                    conn,
                    int(train_job_id),
                    progress=progress,
                )
            )
        return 0
    run, batches = claim
    _report_actor_progress(
        progress,
        "claimed",
        claimed_batches=len(batches),
    )
    return _drain_claim(conn, run, batches, progress=progress)


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
    stop_requested: Any | None = None,
    expected_source_fingerprint: str | None = None,
) -> int:
    """Own one run until publication completes, surviving idle producer gaps.

    The lifetime advisory lock makes a launchd manager restart safe: a replacement actor waits
    behind the existing owner instead of opening a second W&B SDK writer. Each drain still uses
    the narrower run lock so recovery and tests keep the same transactional claim boundary.
    """

    actor_key = f"rlab-wandb-actor:{int(train_job_id)}"
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtextextended(%(key)s, 0)) AS acquired",
            {"key": actor_key},
        )
        if not bool(cur.fetchone()["acquired"]):
            conn.rollback()
            raise WandbPublisherActorLockBusy(
                f"publisher actor already owns train/{int(train_job_id)}"
            )
    published = 0
    try:
        _write_actor_state(
            int(train_job_id),
            phase="idle",
            session_started_at=None,
            expected_source_fingerprint=expected_source_fingerprint,
        )
        while True:
            if callable(stop_requested) and stop_requested():
                return published
            session_started_at = time.time()
            lease_owner = (
                f"fleet-publisher-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            )

            def report_progress(
                *,
                stage: str,
                claimed_batches: int,
                completed_batches: int = 0,
                completed_frames: int = 0,
                close_started_at: float | None = None,
            ) -> None:
                _write_actor_state(
                    int(train_job_id),
                    phase="publishing",
                    session_started_at=session_started_at,
                    lease_owner=lease_owner,
                    progress_at=time.time(),
                    stage=stage,
                    claimed_batches=claimed_batches,
                    completed_batches=completed_batches,
                    completed_frames=completed_frames,
                    close_started_at=close_started_at,
                    expected_source_fingerprint=expected_source_fingerprint,
                )

            _write_actor_state(
                int(train_job_id),
                phase="publishing",
                session_started_at=session_started_at,
                lease_owner=lease_owner,
                progress_at=session_started_at,
                stage="claiming",
                expected_source_fingerprint=expected_source_fingerprint,
            )
            try:
                published += drain_once(
                    conn,
                    owner=lease_owner,
                    limit=max(1, int(limit)),
                    train_job_id=int(train_job_id),
                    progress=report_progress,
                )
            finally:
                _write_actor_state(
                    int(train_job_id),
                    phase="idle",
                    session_started_at=None,
                    expected_source_fingerprint=expected_source_fingerprint,
                )
            done = _publisher_actor_done(conn, int(train_job_id))
            # Do not leave a read transaction open while an active producer is idle.
            conn.rollback()
            if done or once:
                return published
            if callable(stop_requested) and stop_requested():
                return published
            time.sleep(max(float(poll_seconds), 0.01))
    finally:
        # A publisher failure may leave the connection in an aborted transaction.
        # Advisory locks are session scoped, so rollback first and then release the
        # lock explicitly without masking the original publication exception.
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_unlock(hashtextextended(%(key)s, 0))",
                {"key": actor_key},
            )
        conn.rollback()


def _validate_publisher_schema(conn) -> None:
    required_columns = {
        ("metric_batches", "wandb_confirmed_at"),
    }
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND (table_name, column_name) IN (
                SELECT * FROM unnest(%(tables)s::text[], %(columns)s::text[])
              )
            """,
            {
                "tables": [table for table, _column in sorted(required_columns)],
                "columns": [column for _table, column in sorted(required_columns)],
            },
        )
        present = {
            (str(row["table_name"]), str(row["column_name"]))
            for row in cur.fetchall()
        }
    conn.rollback()
    missing = sorted(required_columns - present)
    if missing:
        rendered = ", ".join(f"{table}.{column}" for table, column in missing)
        raise RuntimeError(
            "publisher database schema is incompatible; missing "
            f"{rendered}; run `rlab fleet queue setup` while the fleet is quiescent"
        )


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
    from rlab.fleet_service import controller_source_fingerprint

    source_fingerprint = controller_source_fingerprint(repo_root)
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
                    "--expected-source-fingerprint",
                    source_fingerprint,
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
    parser.add_argument("--expected-source-fingerprint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from rlab.fleet_service import controller_source_fingerprint, redact
    from rlab.job_queue import connect, database_url

    expected_source_fingerprint = str(args.expected_source_fingerprint or "").strip()
    observed_source_fingerprint = controller_source_fingerprint(Path.cwd())
    if expected_source_fingerprint and observed_source_fingerprint != expected_source_fingerprint:
        error = (
            "publisher actor source fingerprint mismatch: "
            f"expected={expected_source_fingerprint}, "
            f"observed={observed_source_fingerprint}"
        )
        if args.train_job_id is not None:
            _write_actor_state(
                int(args.train_job_id),
                phase="startup_failed",
                session_started_at=None,
                stage="source_fingerprint_mismatch",
                expected_source_fingerprint=expected_source_fingerprint,
                error=error,
            )
        print(error, file=sys.stderr)
        return ACTOR_SOURCE_MISMATCH_EXIT_CODE

    # Publishing holds a session-scoped per-run lock for the whole W&B
    # session, so it must not use a PgBouncer-backed connection.
    stop_event = threading.Event()
    if args.train_job_id is not None and not args.once:
        signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_event.set())
    try:
        conn = connect(database_url(use_direct=True))
    except Exception as exc:
        error = str(redact(f"publisher actor startup failed: {type(exc).__name__}: {exc}"))[:4000]
        if args.train_job_id is not None:
            _write_actor_state(
                int(args.train_job_id),
                phase="startup_failed",
                session_started_at=None,
                stage="database_connect",
                expected_source_fingerprint=expected_source_fingerprint or None,
                error=error,
            )
        print(error, file=sys.stderr)
        return ACTOR_START_FAILED_EXIT_CODE
    try:
        try:
            _validate_publisher_schema(conn)
        except Exception as exc:
            error = str(
                redact(
                    f"publisher actor startup failed: {type(exc).__name__}: {exc}"
                )
            )[:4000]
            if args.train_job_id is not None:
                _write_actor_state(
                    int(args.train_job_id),
                    phase="startup_failed",
                    session_started_at=None,
                    stage="database_schema",
                    expected_source_fingerprint=expected_source_fingerprint or None,
                    error=error,
                )
            print(error, file=sys.stderr)
            return ACTOR_START_FAILED_EXIT_CODE
        try:
            if args.train_job_id is not None:
                published = run_publisher_actor(
                    conn,
                    int(args.train_job_id),
                    limit=max(1, args.limit),
                    once=bool(args.once),
                    stop_requested=stop_event.is_set,
                    expected_source_fingerprint=expected_source_fingerprint or None,
                )
            else:
                published = drain_once(conn, limit=max(1, args.limit))
        except WandbPublisherActorLockBusy as exc:
            print(str(exc), file=sys.stderr)
            return ACTOR_LOCK_BUSY_EXIT_CODE
        print(f"published_batches={published}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
