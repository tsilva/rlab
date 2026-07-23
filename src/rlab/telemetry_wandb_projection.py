from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from rlab.job_queue import json_arg
from rlab.telemetry_archive import canonical_event_from_ledger_row
from rlab.telemetry_integrity import (
    canonical_json_bytes,
    normalize_wandb_rows,
    sha256_bytes,
    sha256_json,
)


PROJECTION_PROTOCOL_VERSION = "wandb-generational-projection-v1"


class WandbRun(Protocol):
    def log(self, payload: Mapping[str, object], *, step: int, commit: bool) -> object: ...


@dataclass(frozen=True)
class ProjectionRow:
    train_job_id: int
    projection_generation: int
    output_ordinal: int
    step_offset: int
    stable_key: str
    payload_sha256: str
    payload: Mapping[str, object]

    @property
    def wandb_step(self) -> int:
        return self.step_offset + self.output_ordinal


@dataclass(frozen=True)
class PrefixVerification:
    verified_through_ordinal: int
    exact: bool
    reason: str | None = None


def new_projection_generation(
    conn,
    *,
    train_job_id: int,
    service_credential_generation: int,
    wandb_run_id: str,
    step_offset: int = 0,
) -> dict[str, Any]:
    token = secrets.token_urlsafe(32)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(max(projection_generation), 0) + 1 AS generation
                FROM wandb_projection_generations
                WHERE train_job_id = %(run)s
                FOR UPDATE
                """,
                {"run": int(train_job_id)},
            )
            generation = int(cur.fetchone()["generation"])
            cur.execute(
                """
                INSERT INTO wandb_projection_generations (
                  train_job_id, projection_generation,
                  service_credential_generation, wandb_run_id,
                  generation_token_sha256, state, step_offset
                ) VALUES (
                  %(run)s, %(generation)s, %(credential_generation)s,
                  %(wandb_run_id)s, %(token_sha256)s, 'pending', %(step_offset)s
                )
                RETURNING *
                """,
                {
                    "run": int(train_job_id),
                    "generation": generation,
                    "credential_generation": int(service_credential_generation),
                    "wandb_run_id": str(wandb_run_id),
                    "token_sha256": sha256_bytes(token.encode("utf-8")),
                    "step_offset": int(step_offset),
                },
            )
            result = dict(cur.fetchone())
    return result


def materialize_projection_rows(
    conn,
    *,
    train_job_id: int,
    projection_generation: int,
    max_events: int = 1000,
) -> int:
    """Atomically reserve a contiguous ordinal range for complete source events."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM wandb_projection_generations
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                FOR UPDATE
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            generation = cur.fetchone()
            if not generation:
                raise ValueError("unknown W&B projection generation")
            if str(generation["state"]) in {"sealed", "quarantined", "failed", "disabled"}:
                raise RuntimeError("W&B projection generation does not accept new rows")
            cur.execute(
                """
                SELECT e.*, p.producer_identity
                FROM telemetry_events e
                JOIN telemetry_producers p
                  ON p.train_job_id = e.train_job_id
                 AND p.telemetry_generation = e.telemetry_generation
                 AND p.producer_ordinal = e.producer_ordinal
                WHERE e.train_job_id = %(run)s
                  AND NOT EXISTS (
                    SELECT 1 FROM wandb_projection_rows r
                    WHERE r.train_job_id = e.train_job_id
                      AND r.projection_generation = %(generation)s
                      AND r.source_event_id = (
                        e.train_job_id::text || ':' ||
                        e.telemetry_generation::text || ':' ||
                        e.producer_ordinal::text || ':' ||
                        e.source_sequence::text || ':' ||
                        e.event_identity
                      )
                  )
                ORDER BY e.producer_ordinal, e.source_sequence
                LIMIT %(limit)s
                FOR SHARE OF e
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "limit": max(1, min(int(max_events), 1000)),
                },
            )
            events = [canonical_event_from_ledger_row(dict(row)) for row in cur.fetchall()]
            next_ordinal = int(generation["next_ordinal"])
            predecessor = None
            if next_ordinal:
                cur.execute(
                    """
                    SELECT payload_sha256 FROM wandb_projection_rows
                    WHERE train_job_id = %(run)s
                      AND projection_generation = %(generation)s
                      AND output_ordinal = %(ordinal)s
                    """,
                    {
                        "run": int(train_job_id),
                        "generation": int(projection_generation),
                        "ordinal": next_ordinal - 1,
                    },
                )
                previous = cur.fetchone()
                if not previous:
                    raise RuntimeError("W&B projection predecessor row is missing")
                predecessor = str(previous["payload_sha256"])
            inserted = 0
            for event in events:
                rows = normalize_wandb_rows(event, first_ordinal=next_ordinal)
                for normalized in rows:
                    ordinal = int(normalized["ordinal"])
                    core_payload = dict(normalized["payload"])
                    digest = str(normalized["payload_sha256"])
                    payload = {
                        **core_payload,
                        "_rlab_payload_sha256": digest,
                        "_rlab_predecessor_sha256": predecessor,
                        "_rlab_projection_generation": int(projection_generation),
                        "_rlab_projection_protocol": PROJECTION_PROTOCOL_VERSION,
                    }
                    cur.execute(
                        """
                        INSERT INTO wandb_projection_rows (
                          train_job_id, projection_generation, output_ordinal,
                          stable_key, source_event_id, adapter_version,
                          normalization_version, output_kind, output_index,
                          predecessor_sha256, payload_sha256, payload_json
                        ) VALUES (
                          %(run)s, %(generation)s, %(ordinal)s, %(stable_key)s,
                          %(source_event_id)s, %(adapter_version)s,
                          %(normalization_version)s, %(output_kind)s,
                          %(output_index)s, %(predecessor)s, %(digest)s, %(payload)s
                        )
                        """,
                        {
                            "run": int(train_job_id),
                            "generation": int(projection_generation),
                            "ordinal": ordinal,
                            "stable_key": normalized["stable_key"],
                            "source_event_id": normalized["source_event_id"],
                            "adapter_version": normalized["adapter_version"],
                            "normalization_version": normalized["normalization_version"],
                            "output_kind": normalized["output_kind"],
                            "output_index": int(normalized["output_index"]),
                            "predecessor": predecessor,
                            "digest": digest,
                            "payload": json_arg(payload),
                        },
                    )
                    predecessor = digest
                    next_ordinal += 1
                    inserted += 1
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET next_ordinal = %(next_ordinal)s,
                    state = CASE WHEN state = 'pending' THEN 'publishing' ELSE state END
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "next_ordinal": next_ordinal,
                },
            )
            return inserted


def claim_projection_row(
    conn,
    *,
    owner: str,
    train_job_id: int | None = None,
) -> ProjectionRow | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.*, g.step_offset
                FROM wandb_projection_rows r
                JOIN wandb_projection_generations g
                  ON g.train_job_id = r.train_job_id
                 AND g.projection_generation = r.projection_generation
                WHERE (%(run)s IS NULL OR r.train_job_id = %(run)s)
                  AND g.state IN ('publishing', 'verifying')
                  AND r.state IN ('pending', 'ambiguous')
                  AND (r.claim_expires_at IS NULL OR r.claim_expires_at <= now())
                  AND NOT EXISTS (
                    SELECT 1 FROM wandb_projection_rows prior
                    WHERE prior.train_job_id = r.train_job_id
                      AND prior.projection_generation = r.projection_generation
                      AND prior.output_ordinal < r.output_ordinal
                      AND prior.state <> 'verified'
                  )
                ORDER BY r.train_job_id, r.projection_generation, r.output_ordinal
                FOR UPDATE OF r SKIP LOCKED
                LIMIT 1
                """
                ,
                {"run": train_job_id},
            )
            row = cur.fetchone()
            if not row:
                return None
            cur.execute(
                """
                UPDATE wandb_projection_rows
                SET state = 'claimed', claimed_by = %(owner)s,
                    claim_expires_at = now() + interval '2 minutes',
                    attempts = attempts + 1, last_error = NULL
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                  AND output_ordinal = %(ordinal)s
                """,
                {
                    "owner": str(owner),
                    "run": int(row["train_job_id"]),
                    "generation": int(row["projection_generation"]),
                    "ordinal": int(row["output_ordinal"]),
                },
            )
    return ProjectionRow(
        train_job_id=int(row["train_job_id"]),
        projection_generation=int(row["projection_generation"]),
        output_ordinal=int(row["output_ordinal"]),
        step_offset=int(row["step_offset"]),
        stable_key=str(row["stable_key"]),
        payload_sha256=str(row["payload_sha256"]),
        payload=dict(row["payload_json"]),
    )


def publish_projection_row(
    conn,
    run: WandbRun,
    row: ProjectionRow,
    *,
    owner: str,
) -> None:
    """Perform exactly one committed SDK call; ambiguity remains recoverable."""

    try:
        run.log(dict(row.payload), step=row.wandb_step, commit=True)
    except Exception as exc:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE wandb_projection_rows
                    SET state = 'ambiguous', claimed_by = NULL,
                        claim_expires_at = NULL, last_error = %(error)s
                    WHERE train_job_id = %(run)s
                      AND projection_generation = %(generation)s
                      AND output_ordinal = %(ordinal)s
                      AND claimed_by = %(owner)s
                    """,
                    {
                        "run": row.train_job_id,
                        "generation": row.projection_generation,
                        "ordinal": row.output_ordinal,
                        "owner": str(owner),
                        "error": f"{type(exc).__name__}: {exc}"[:4000],
                    },
                )
        raise
    # A successful return is still not authoritative. Verification owns the
    # state transition to verified.
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE wandb_projection_rows
                SET state = 'ambiguous', claimed_by = NULL, claim_expires_at = NULL
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                  AND output_ordinal = %(ordinal)s
                  AND claimed_by = %(owner)s
                """,
                {
                    "run": row.train_job_id,
                    "generation": row.projection_generation,
                    "ordinal": row.output_ordinal,
                    "owner": str(owner),
                },
            )


def verify_exact_prefix(
    expected: Sequence[Mapping[str, Any]],
    observed: Iterable[Mapping[str, Any]],
    *,
    step_offset: int,
) -> PrefixVerification:
    observed_rows = list(observed)
    seen: set[tuple[int, str]] = set()
    if len(observed_rows) > len(expected):
        return PrefixVerification(-1, False, "foreign_or_duplicate_suffix")
    for index, remote in enumerate(observed_rows):
        expected_row = expected[index]
        ordinal = int(expected_row["output_ordinal"])
        remote_step = int(remote.get("_step", remote.get("step", -1)))
        stable = str(remote.get("_rlab_event_id") or "")
        digest = str(remote.get("_rlab_payload_sha256") or "")
        key = (remote_step, stable)
        if key in seen:
            return PrefixVerification(index - 1, False, "duplicate_remote_identity")
        seen.add(key)
        if remote_step != int(step_offset) + ordinal:
            return PrefixVerification(index - 1, False, "past_step_hole_or_foreign_step")
        payload = expected_row.get("payload_json") or expected_row.get("payload") or {}
        if stable != str(payload.get("_rlab_event_id") or ""):
            return PrefixVerification(index - 1, False, "stable_identity_conflict")
        if digest != str(expected_row["payload_sha256"]):
            return PrefixVerification(index - 1, False, "payload_digest_conflict")
    return PrefixVerification(len(observed_rows) - 1, True)


def persist_prefix_verification(
    conn,
    *,
    train_job_id: int,
    projection_generation: int,
    observed_rows: Sequence[Mapping[str, Any]],
) -> PrefixVerification:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM wandb_projection_generations
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                FOR UPDATE
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            generation = cur.fetchone()
            if not generation:
                raise ValueError("unknown W&B projection generation")
            cur.execute(
                """
                SELECT output_ordinal, payload_sha256, payload_json
                FROM wandb_projection_rows
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                ORDER BY output_ordinal
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            expected = [dict(row) for row in cur.fetchall()]
            verification = verify_exact_prefix(
                expected, observed_rows, step_offset=int(generation["step_offset"])
            )
            if not verification.exact:
                cur.execute(
                    """
                    INSERT INTO telemetry_incidents (
                      train_job_id, telemetry_generation, incident_key,
                      severity, category, details_json
                    )
                    SELECT
                      %(run)s, j.telemetry_generation,
                      'wandb-generation-' || %(generation)s::text,
                      'critical', 'wandb_projection_conflict',
                      jsonb_build_object('reason', %(reason)s)
                    FROM train_jobs j WHERE j.id = %(run)s
                    ON CONFLICT (
                      train_job_id, telemetry_generation, incident_key
                    ) DO UPDATE
                    SET last_seen_at = now(), details_json = EXCLUDED.details_json
                    RETURNING id
                    """,
                    {
                        "run": int(train_job_id),
                        "generation": int(projection_generation),
                        "reason": verification.reason,
                    },
                )
                incident = cur.fetchone()
                cur.execute(
                    """
                    UPDATE wandb_projection_generations
                    SET state = 'quarantined', incident_id = %(incident)s,
                        sealed_at = now()
                    WHERE train_job_id = %(run)s
                      AND projection_generation = %(generation)s
                    """,
                    {
                        "run": int(train_job_id),
                        "generation": int(projection_generation),
                        "incident": int(incident["id"]),
                    },
                )
                return verification
            through = verification.verified_through_ordinal
            if through >= 0:
                cur.execute(
                    """
                    UPDATE wandb_projection_rows
                    SET state = 'verified', verified_at = COALESCE(verified_at, now()),
                        claimed_by = NULL, claim_expires_at = NULL, last_error = NULL
                    WHERE train_job_id = %(run)s
                      AND projection_generation = %(generation)s
                      AND output_ordinal <= %(through)s
                    """,
                    {
                        "run": int(train_job_id),
                        "generation": int(projection_generation),
                        "through": through,
                    },
                )
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET verified_through_ordinal = %(through)s,
                    state = 'verifying'
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "through": through,
                },
            )
            return verification


def projection_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_json(
        [
            {
                "ordinal": int(row["output_ordinal"]),
                "stable_key": str(row["stable_key"]),
                "payload_sha256": str(row["payload_sha256"]),
                "predecessor_sha256": row.get("predecessor_sha256"),
                "output_kind": str(row["output_kind"]),
            }
            for row in rows
        ]
    )


def activate_verified_generation(
    conn,
    *,
    train_job_id: int,
    projection_generation: int,
    remote_fingerprint: str,
) -> str:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM wandb_projection_rows
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                ORDER BY output_ordinal
                FOR UPDATE
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            rows = [dict(row) for row in cur.fetchall()]
            if not rows or any(str(row["state"]) != "verified" for row in rows):
                raise RuntimeError("W&B generation is not an exact verified prefix")
            local_fingerprint = projection_fingerprint(rows)
            if local_fingerprint != str(remote_fingerprint):
                raise RuntimeError("W&B scalar/rich/artifact presentation fingerprint mismatch")
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET state = 'sealed', sealed_at = COALESCE(sealed_at, now())
                WHERE train_job_id = %(run)s
                  AND state = 'active'
                  AND projection_generation <> %(generation)s
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET state = 'active', activated_at = COALESCE(activated_at, now())
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                  AND state = 'verifying'
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            if cur.rowcount != 1:
                raise RuntimeError("W&B generation cannot be activated from its current state")
            cur.execute(
                """
                UPDATE train_jobs
                SET active_wandb_projection_generation = %(generation)s
                WHERE id = %(run)s
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
    from rlab.telemetry_reducer import reduce_run_integrity

    reduce_run_integrity(conn, train_job_id=int(train_job_id))
    return local_fingerprint
