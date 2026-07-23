from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from rlab.telemetry_integrity import canonical_json_bytes, sha256_bytes, write_fsync


CUTOVER_MARKER_VERSION = "telemetry-cutover-marker-v1"
DEFAULT_CUTOVER_MARKER = Path("/var/lib/rlab/telemetry-cutover.json")


@dataclass(frozen=True)
class CutoverVerification:
    cutover_generation: int
    admission_fenced: bool
    destructive_hold: bool
    service_credential_rotated: bool
    pre_fence_legacy_active: int
    post_fence_legacy_rows: int
    legacy_sessions: int

    @property
    def safe_to_stop_legacy_publishers(self) -> bool:
        return (
            self.admission_fenced
            and self.destructive_hold
            and self.service_credential_rotated
            and self.pre_fence_legacy_active == 0
            and self.post_fence_legacy_rows == 0
            and self.legacy_sessions == 0
        )


def marker_document(*, generation: int, reason: str) -> dict[str, Any]:
    body = {
        "version": CUTOVER_MARKER_VERSION,
        "cutover_generation": int(generation),
        "admission_fenced": True,
        "reason": str(reason),
    }
    return {**body, "sha256": sha256_bytes(canonical_json_bytes(body))}


def write_cutover_marker(path: Path, *, generation: int, reason: str) -> dict[str, Any]:
    document = marker_document(generation=generation, reason=reason)
    write_fsync(Path(path), canonical_json_bytes(document))
    return document


def read_cutover_marker(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("telemetry cutover marker is not a JSON mapping")
    digest = str(value.pop("sha256", ""))
    if digest != sha256_bytes(canonical_json_bytes(value)):
        raise RuntimeError("telemetry cutover marker digest is invalid")
    value["sha256"] = digest
    if value.get("version") != CUTOVER_MARKER_VERSION:
        raise RuntimeError("telemetry cutover marker version is unsupported")
    return value


def require_runtime_admission(
    config: Mapping[str, Any],
    *,
    environment: Mapping[str, str] | None = None,
    marker_path: Path | None = None,
) -> None:
    """Fence a learner before environment resources or W&B can be created."""

    values = os.environ if environment is None else environment
    path = Path(
        values.get("RLAB_TELEMETRY_CUTOVER_MARKER_PATH")
        or marker_path
        or DEFAULT_CUTOVER_MARKER
    )
    marker = read_cutover_marker(path)
    fenced = str(values.get("RLAB_TELEMETRY_ADMISSION_FENCED") or "") == "1"
    if marker is None and not fenced:
        return
    expected = int(
        (marker or {}).get("cutover_generation")
        or values.get("RLAB_TELEMETRY_CUTOVER_GENERATION")
        or 0
    )
    observed = int(values.get("RLAB_TELEMETRY_CUTOVER_GENERATION") or 0)
    configured = int(config.get("telemetry_generation") or 0)
    protocol = int(config.get("telemetry_protocol_version") or 1)
    if protocol != 2 or expected < 1 or observed != expected or configured != expected:
        raise RuntimeError(
            "training admission rejected by the telemetry cutover generation fence"
        )


def install_cutover_fences(
    conn,
    *,
    marker_path: Path,
    reason: str,
    migration_principal: str,
    service_wandb_credential_generation: int,
) -> dict[str, Any]:
    """Make the host marker first, then atomically fence queue admission/deletion."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT cutover_generation, legacy_wandb_credential_generation
            FROM telemetry_rollout_controls
            WHERE singleton = TRUE
            """
        )
        current = cur.fetchone()
    generation = int(current["cutover_generation"]) + 1
    if int(service_wandb_credential_generation) <= int(
        current["legacy_wandb_credential_generation"]
    ):
        raise ValueError("service W&B credential generation must rotate past legacy")
    marker = write_cutover_marker(marker_path, generation=generation, reason=reason)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pg_advisory_xact_lock(
                  hashtextextended('rlab-fleet-admission', 0)
                )
                """
            )
            cur.execute(
                """
                UPDATE telemetry_rollout_controls
                SET cutover_generation = %(generation)s,
                    admission_fenced = TRUE,
                    destructive_hold = TRUE,
                    service_wandb_credential_generation = %(credential_generation)s,
                    migration_principal = %(principal)s,
                    reason = %(reason)s,
                    updated_at = now()
                WHERE singleton = TRUE
                RETURNING *
                """,
                {
                    "generation": generation,
                    "credential_generation": int(service_wandb_credential_generation),
                    "principal": str(migration_principal),
                    "reason": str(reason),
                },
            )
            state = dict(cur.fetchone())
    return {"marker": marker, "rollout": state}


def terminate_legacy_sessions(
    conn,
    *,
    application_names: Sequence[str],
) -> list[int]:
    names = sorted({str(name).strip() for name in application_names if str(name).strip()})
    if not names:
        return []
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT pid
                FROM pg_stat_activity
                WHERE pid <> pg_backend_pid()
                  AND application_name = ANY(%(names)s)
                FOR UPDATE
                """,
                {"names": names},
            )
            pids = [int(row["pid"]) for row in cur.fetchall()]
            for pid in pids:
                cur.execute("SELECT pg_terminate_backend(%(pid)s)", {"pid": pid})
    return pids


def verify_cutover_fences(
    conn,
    *,
    legacy_application_names: Sequence[str] = (),
) -> CutoverVerification:
    names = sorted(
        {str(name).strip() for name in legacy_application_names if str(name).strip()}
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM telemetry_rollout_controls WHERE singleton = TRUE"
        )
        state = cur.fetchone()
        cur.execute(
            """
            SELECT
              count(*) FILTER (
                WHERE telemetry_protocol_version = 1
                  AND status NOT IN (
                    'succeeded', 'failed', 'canceled', 'finalization_failed'
                  )
              ) AS active,
              count(*) FILTER (
                WHERE telemetry_protocol_version = 1
                  AND telemetry_generation >= %(generation)s
              ) AS post_fence
            FROM train_jobs
            """,
            {"generation": int(state["cutover_generation"])},
        )
        legacy = cur.fetchone()
        if names:
            cur.execute(
                """
                SELECT count(*) AS count
                FROM pg_stat_activity
                WHERE pid <> pg_backend_pid()
                  AND application_name = ANY(%(names)s)
                """,
                {"names": names},
            )
            sessions = int(cur.fetchone()["count"])
        else:
            sessions = 0
    service_generation = state["service_wandb_credential_generation"]
    return CutoverVerification(
        cutover_generation=int(state["cutover_generation"]),
        admission_fenced=bool(state["admission_fenced"]),
        destructive_hold=bool(state["destructive_hold"]),
        service_credential_rotated=(
            service_generation is not None
            and int(service_generation) > int(state["legacy_wandb_credential_generation"])
        ),
        pre_fence_legacy_active=int(legacy["active"]),
        post_fence_legacy_rows=int(legacy["post_fence"]),
        legacy_sessions=sessions,
    )


def classify_vulnerable_v1_runs(conn) -> list[dict[str, Any]]:
    """Classify all protocol-v1 rows from retained evidence; never infer intactness."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH stream_evidence AS (
                  SELECT
                    w.train_job_id,
                    count(*) AS stream_count,
                    count(*) FILTER (WHERE s.final_sequence IS NOT NULL) AS terminal_streams,
                    bool_and(
                      s.accepted_sequence = COALESCE((
                        SELECT count(*) FROM metric_batches b
                        WHERE b.stream_id = s.stream_id
                          AND b.batch_sequence BETWEEN 1 AND s.accepted_sequence
                      ), 0)
                    ) AS retained_without_gap
                  FROM worker_attempts w
                  JOIN metric_streams s ON s.attempt_id = w.attempt_id
                  GROUP BY w.train_job_id
                )
                SELECT
                  j.id,
                  j.status,
                  COALESCE(e.stream_count, 0) AS stream_count,
                  COALESCE(e.terminal_streams, 0) AS terminal_streams,
                  COALESCE(e.retained_without_gap, FALSE) AS retained_without_gap
                FROM train_jobs j
                LEFT JOIN stream_evidence e ON e.train_job_id = j.id
                WHERE j.telemetry_protocol_version = 1
                ORDER BY j.id
                FOR UPDATE OF j
                """
            )
            rows = [dict(row) for row in cur.fetchall()]
            results: list[dict[str, Any]] = []
            for row in rows:
                terminal = str(row["status"]) in {
                    "succeeded",
                    "failed",
                    "canceled",
                    "finalization_failed",
                }
                complete_streams = int(row["stream_count"]) > 0 and int(
                    row["terminal_streams"]
                ) == int(row["stream_count"])
                # Retained v1 bytes alone prove only that the surviving population is
                # internally continuous. They cannot prove that deleted batches never existed.
                if not terminal or not complete_streams:
                    classification = "legacy_unknown"
                    reason = "legacy terminal or producer evidence is incomplete"
                elif not bool(row["retained_without_gap"]):
                    classification = "degraded"
                    reason = "known retained v1 source gap"
                else:
                    classification = "legacy_unknown"
                    reason = "surviving v1 bytes are continuous but deletion absence is unprovable"
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET telemetry_integrity_classification = %(classification)s
                    WHERE id = %(run)s
                    """,
                    {"run": int(row["id"]), "classification": classification},
                )
                cur.execute(
                    """
                    INSERT INTO telemetry_incidents (
                      train_job_id, telemetry_generation, incident_key,
                      severity, category, state, details_json
                    ) VALUES (
                      %(run)s, 1, 'legacy-v1-classification',
                      %(severity)s, 'legacy_telemetry_integrity', 'permanent',
                      jsonb_build_object(
                        'classification', %(classification)s,
                        'reason', %(reason)s
                      )
                    )
                    ON CONFLICT (
                      train_job_id, telemetry_generation, incident_key
                    ) DO UPDATE
                    SET details_json = EXCLUDED.details_json, last_seen_at = now()
                    """,
                    {
                        "run": int(row["id"]),
                        "severity": "critical" if classification == "degraded" else "warning",
                        "classification": classification,
                        "reason": reason,
                    },
                )
                results.append(
                    {
                        "train_job_id": int(row["id"]),
                        "classification": classification,
                        "reason": reason,
                    }
                )
    return results
