from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol, Sequence

from rlab.job_queue import json_arg
from rlab.telemetry_archive import canonical_event_from_ledger_row
from rlab.telemetry_integrity import (
    normalize_wandb_rows,
    sha256_bytes,
    sha256_json,
)


PROJECTION_PROTOCOL_VERSION = "wandb-generational-projection-v1"
BOUNDED_PROJECTION_PROTOCOL_VERSION = "wandb-generational-projection-v2"
MAX_MATERIALIZED_EVENTS_PER_CYCLE = 64
MAX_UNVERIFIED_ROWS_PER_RUN = 256
MAX_CLAIMED_ROWS_PER_CYCLE = 32
MAX_VERIFICATION_WINDOW = 256


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
    source_created_at: object | None = None

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
                "SELECT id FROM train_jobs WHERE id = %(run)s FOR UPDATE",
                {"run": int(train_job_id)},
            )
            if not cur.fetchone():
                raise ValueError("unknown train job")
            cur.execute(
                """
                SELECT COALESCE(max(projection_generation), 0) + 1 AS generation
                FROM wandb_projection_generations
                WHERE train_job_id = %(run)s
                """,
                {"run": int(train_job_id)},
            )
            generation = int(cur.fetchone()["generation"])
            cur.execute(
                """
                INSERT INTO wandb_projection_generations (
                  train_job_id, projection_generation,
                  service_credential_generation, wandb_run_id,
                  generation_token_sha256, state, step_offset, protocol_version
                ) VALUES (
                  %(run)s, %(generation)s, %(credential_generation)s,
                  %(wandb_run_id)s, %(token_sha256)s, 'pending', %(step_offset)s,
                  %(protocol_version)s
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
                    "protocol_version": BOUNDED_PROJECTION_PROTOCOL_VERSION,
                },
            )
            result = dict(cur.fetchone())
    return result


def repair_quarantined_generation(
    conn,
    *,
    train_job_id: int,
    quarantined_generation: int,
    service_credential_generation: int,
    fresh_wandb_run_id: str,
) -> dict[str, Any]:
    """Seal incident evidence and materialize a full ordinal-zero replay generation."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET sealed_at = COALESCE(sealed_at, now())
                WHERE train_job_id=%(run)s
                  AND projection_generation=%(generation)s
                  AND state='quarantined'
                RETURNING step_offset
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(quarantined_generation),
                },
            )
            old = cur.fetchone()
            if not old:
                raise RuntimeError("W&B repair requires a quarantined source generation")
    fresh = new_projection_generation(
        conn,
        train_job_id=int(train_job_id),
        service_credential_generation=int(service_credential_generation),
        wandb_run_id=str(fresh_wandb_run_id),
        step_offset=0,
    )
    total = 0
    while True:
        count = materialize_projection_rows(
            conn,
            train_job_id=int(train_job_id),
            projection_generation=int(fresh["projection_generation"]),
            max_events=1000,
        )
        total += count
        if count == 0:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                      SELECT 1 FROM telemetry_events e
                      JOIN wandb_projection_source_cursors c
                        ON c.train_job_id=e.train_job_id
                       AND c.projection_generation=%(generation)s
                       AND c.telemetry_generation=e.telemetry_generation
                       AND c.producer_ordinal=e.producer_ordinal
                      WHERE e.train_job_id=%(run)s
                        AND e.source_sequence > c.consumed_through_sequence
                    ) AS pending
                    """,
                    {
                        "run": int(train_job_id),
                        "generation": int(fresh["projection_generation"]),
                    },
                )
                if not bool(cur.fetchone()["pending"]):
                    break
    return {**fresh, "replayed_rows": total}


def materialize_projection_rows(
    conn,
    *,
    train_job_id: int,
    projection_generation: int,
    max_events: int = MAX_MATERIALIZED_EVENTS_PER_CYCLE,
) -> int:
    """Consume bounded canonical source events and materialize at most one history row each."""

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
                SELECT count(*) AS count
                FROM wandb_projection_rows
                WHERE train_job_id=%(run)s
                  AND projection_generation=%(generation)s
                  AND state <> 'verified'
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            unverified = int(cur.fetchone()["count"])
            remaining_capacity = MAX_UNVERIFIED_ROWS_PER_RUN - unverified
            if remaining_capacity <= 0:
                return 0
            cur.execute(
                """
                INSERT INTO wandb_projection_source_cursors (
                  train_job_id, projection_generation, telemetry_generation,
                  producer_ordinal
                )
                SELECT p.train_job_id, %(generation)s, p.telemetry_generation,
                       p.producer_ordinal
                FROM telemetry_producers p
                WHERE p.train_job_id=%(run)s
                ON CONFLICT DO NOTHING
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            cur.execute(
                """
                SELECT e.*, p.producer_identity,
                       COALESCE(
                         (j.train_config->>'metrics_schema_version')::integer,
                         5
                       ) AS metrics_schema_version
                FROM telemetry_events e
                JOIN telemetry_producers p
                 ON p.train_job_id = e.train_job_id
                 AND p.telemetry_generation = e.telemetry_generation
                 AND p.producer_ordinal = e.producer_ordinal
                JOIN train_jobs j ON j.id = e.train_job_id
                JOIN wandb_projection_source_cursors c
                  ON c.train_job_id=e.train_job_id
                 AND c.projection_generation=%(generation)s
                 AND c.telemetry_generation=e.telemetry_generation
                 AND c.producer_ordinal=e.producer_ordinal
                WHERE e.train_job_id = %(run)s
                  AND e.source_sequence > c.consumed_through_sequence
                ORDER BY e.created_at, e.producer_ordinal, e.source_sequence
                LIMIT %(limit)s
                FOR SHARE OF e
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "limit": max(
                        1,
                        min(
                            int(max_events),
                            MAX_MATERIALIZED_EVENTS_PER_CYCLE,
                            remaining_capacity,
                        ),
                    ),
                },
            )
            source_rows = [dict(row) for row in cur.fetchall()]
            events = [(source, canonical_event_from_ledger_row(source)) for source in source_rows]
            next_ordinal = int(generation["next_ordinal"])
            chain = (
                None
                if generation.get("projection_chain_sha256") is None
                else str(generation["projection_chain_sha256"])
            )
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
            consumed = 0
            raw_frames = 0
            last_source_at = generation.get("last_source_at")
            for source, event in events:
                rows = normalize_wandb_rows(event, first_ordinal=next_ordinal)
                frames = event.payload.get("frames") if isinstance(event.payload, Mapping) else None
                event_raw_frames = len(frames) if isinstance(frames, Sequence) else 0
                for normalized in rows:
                    ordinal = int(normalized["ordinal"])
                    core_payload = dict(normalized["payload"])
                    digest = str(normalized["payload_sha256"])
                    payload = {
                        **core_payload,
                        "_rlab_payload_sha256": digest,
                        "_rlab_predecessor_sha256": predecessor,
                        "_rlab_projection_generation": int(projection_generation),
                        "_rlab_projection_protocol": BOUNDED_PROJECTION_PROTOCOL_VERSION,
                    }
                    cur.execute(
                        """
                        INSERT INTO wandb_projection_rows (
                          train_job_id, projection_generation, output_ordinal,
                          stable_key, source_event_id, adapter_version,
                          normalization_version, output_kind, output_index,
                          predecessor_sha256, payload_sha256, payload_json,
                          source_created_at
                        ) VALUES (
                          %(run)s, %(generation)s, %(ordinal)s, %(stable_key)s,
                          %(source_event_id)s, %(adapter_version)s,
                          %(normalization_version)s, %(output_kind)s,
                          %(output_index)s, %(predecessor)s, %(digest)s, %(payload)s,
                          %(source_created_at)s
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
                            "source_created_at": source["created_at"],
                        },
                    )
                    predecessor = digest
                    chain = sha256_bytes(f"{chain or ''}:{digest}".encode("utf-8"))
                    next_ordinal += 1
                    inserted += 1
                cur.execute(
                    """
                    UPDATE wandb_projection_source_cursors
                    SET consumed_through_sequence=%(sequence)s,
                        source_events_consumed=source_events_consumed + 1,
                        raw_frames_consumed=raw_frames_consumed + %(raw_frames)s,
                        updated_at=now()
                    WHERE train_job_id=%(run)s
                      AND projection_generation=%(generation)s
                      AND telemetry_generation=%(telemetry_generation)s
                      AND producer_ordinal=%(producer_ordinal)s
                      AND consumed_through_sequence < %(sequence)s
                    """,
                    {
                        "run": int(train_job_id),
                        "generation": int(projection_generation),
                        "telemetry_generation": int(source["telemetry_generation"]),
                        "producer_ordinal": int(source["producer_ordinal"]),
                        "sequence": int(source["source_sequence"]),
                        "raw_frames": event_raw_frames,
                    },
                )
                consumed += int(cur.rowcount == 1)
                raw_frames += event_raw_frames
                last_source_at = source["created_at"]
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET next_ordinal = %(next_ordinal)s,
                    source_events_consumed=source_events_consumed + %(consumed)s,
                    raw_frames_consumed=raw_frames_consumed + %(raw_frames)s,
                    selected_rows=selected_rows + %(inserted)s,
                    projection_chain_sha256=%(chain)s,
                    last_source_at=COALESCE(%(last_source_at)s, last_source_at),
                    state = CASE WHEN state = 'pending' THEN 'publishing' ELSE state END
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "next_ordinal": next_ordinal,
                    "consumed": consumed,
                    "raw_frames": raw_frames,
                    "inserted": inserted,
                    "chain": chain,
                    "last_source_at": last_source_at,
                },
            )
            return inserted


def claim_projection_rows(
    conn,
    *,
    owner: str,
    train_job_id: int | None = None,
    limit: int = MAX_CLAIMED_ROWS_PER_CYCLE,
) -> list[ProjectionRow]:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE wandb_projection_rows
                SET state='pending', claimed_by=NULL, claim_expires_at=NULL
                WHERE state='claimed'
                  AND claim_expires_at <= now()
                  AND (%(run)s IS NULL OR train_job_id=%(run)s)
                """,
                {"run": train_job_id},
            )
            cur.execute(
                """
                SELECT r.*, g.step_offset
                FROM wandb_projection_rows r
                JOIN wandb_projection_generations g
                  ON g.train_job_id = r.train_job_id
                 AND g.projection_generation = r.projection_generation
                WHERE (%(run)s IS NULL OR r.train_job_id = %(run)s)
                  AND g.state IN ('publishing', 'verifying', 'active', 'draining')
                  AND r.state = 'pending'
                  AND (r.claim_expires_at IS NULL OR r.claim_expires_at <= now())
                ORDER BY r.train_job_id, r.projection_generation, r.output_ordinal
                FOR UPDATE OF r SKIP LOCKED
                LIMIT %(limit)s
                """,
                {
                    "run": train_job_id,
                    "limit": max(1, min(int(limit), MAX_CLAIMED_ROWS_PER_CYCLE)),
                },
            )
            rows = [dict(row) for row in cur.fetchall()]
            if not rows:
                return []
            cur.execute(
                """
                UPDATE wandb_projection_rows
                SET state = 'claimed', claimed_by = %(owner)s,
                    claim_expires_at = now() + interval '30 seconds',
                    attempts = attempts + 1, last_error = NULL
                WHERE (train_job_id, projection_generation, output_ordinal)
                  IN (
                    SELECT * FROM unnest(
                      %(runs)s::bigint[], %(generations)s::bigint[],
                      %(ordinals)s::bigint[]
                    )
                  )
                """,
                {
                    "owner": str(owner),
                    "runs": [int(row["train_job_id"]) for row in rows],
                    "generations": [int(row["projection_generation"]) for row in rows],
                    "ordinals": [int(row["output_ordinal"]) for row in rows],
                },
            )
    return [
        ProjectionRow(
            train_job_id=int(row["train_job_id"]),
            projection_generation=int(row["projection_generation"]),
            output_ordinal=int(row["output_ordinal"]),
            step_offset=int(row["step_offset"]),
            stable_key=str(row["stable_key"]),
            payload_sha256=str(row["payload_sha256"]),
            payload=dict(row["payload_json"]),
            source_created_at=row.get("source_created_at"),
        )
        for row in rows
    ]


def claim_projection_row(
    conn,
    *,
    owner: str,
    train_job_id: int | None = None,
) -> ProjectionRow | None:
    rows = claim_projection_rows(
        conn,
        owner=owner,
        train_job_id=train_job_id,
        limit=1,
    )
    return rows[0] if rows else None


def publish_projection_rows(
    conn,
    run: WandbRun,
    rows: Sequence[ProjectionRow],
    *,
    owner: str,
    max_rows_per_second: float = 5.0,
) -> int:
    """Enqueue an ordered suffix; successful returns remain submitted until verified."""

    submitted = 0
    last_submit_at: float | None = None
    for row in rows:
        if max_rows_per_second > 0 and last_submit_at is not None:
            remaining = (1.0 / max_rows_per_second) - (time.monotonic() - last_submit_at)
            if remaining > 0:
                time.sleep(remaining)
        try:
            run.log(dict(row.payload), step=row.wandb_step, commit=True)
            last_submit_at = time.monotonic()
        except Exception as exc:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE wandb_projection_rows
                        SET state = 'ambiguous', ambiguous_at=now(), claimed_by = NULL,
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
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE wandb_projection_rows
                    SET state = 'submitted', submitted_at = COALESCE(submitted_at, now()),
                        claimed_by = NULL, claim_expires_at = NULL, last_error = NULL
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
                cur.execute(
                    """
                    UPDATE wandb_projection_generations
                    SET submitted_through_ordinal=GREATEST(
                          submitted_through_ordinal, %(ordinal)s
                        ),
                        last_submitted_at=now(), state=CASE
                          WHEN state='publishing' THEN 'verifying' ELSE state END
                    WHERE train_job_id=%(run)s
                      AND projection_generation=%(generation)s
                    """,
                    {
                        "run": row.train_job_id,
                        "generation": row.projection_generation,
                        "ordinal": row.output_ordinal,
                    },
                )
        submitted += 1
    return submitted


def publish_projection_row(
    conn,
    run: WandbRun,
    row: ProjectionRow,
    *,
    owner: str,
) -> None:
    publish_projection_rows(conn, run, [row], owner=owner)


def verify_exact_prefix(
    expected: Sequence[Mapping[str, Any]],
    observed: Iterable[Mapping[str, Any]],
    *,
    step_offset: int,
) -> PrefixVerification:
    observed_rows = list(observed)
    seen: set[tuple[int, str]] = set()
    if len(observed_rows) > len(expected):
        through = int(expected[-1]["output_ordinal"]) if expected else -1
        return PrefixVerification(through, False, "foreign_or_duplicate_suffix")
    for index, remote in enumerate(observed_rows):
        expected_row = expected[index]
        ordinal = int(expected_row["output_ordinal"])
        remote_step = int(remote.get("_step", remote.get("step", -1)))
        stable = str(remote.get("_rlab_event_id") or "")
        digest = str(remote.get("_rlab_payload_sha256") or "")
        key = (remote_step, stable)
        if key in seen:
            return PrefixVerification(ordinal - 1, False, "duplicate_remote_identity")
        seen.add(key)
        if remote_step != int(step_offset) + ordinal:
            return PrefixVerification(ordinal - 1, False, "past_step_hole_or_foreign_step")
        payload = expected_row.get("payload_json") or expected_row.get("payload") or {}
        if stable != str(payload.get("_rlab_event_id") or ""):
            return PrefixVerification(ordinal - 1, False, "stable_identity_conflict")
        if digest != str(expected_row["payload_sha256"]):
            return PrefixVerification(ordinal - 1, False, "payload_digest_conflict")
    through = (
        int(expected[len(observed_rows) - 1]["output_ordinal"])
        if observed_rows
        else (int(expected[0]["output_ordinal"]) - 1 if expected else -1)
    )
    return PrefixVerification(through, True)


def projection_verification_window(
    generation: Mapping[str, Any],
    *,
    limit: int = MAX_VERIFICATION_WINDOW,
) -> tuple[int, int] | None:
    start = int(generation["verified_through_ordinal"]) + 1
    submitted = int(generation.get("submitted_through_ordinal") or -1)
    if submitted < start:
        return None
    end = min(submitted, start + max(1, min(int(limit), MAX_VERIFICATION_WINDOW)) - 1)
    return start, end


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
            window = projection_verification_window(generation)
            if window is None:
                return PrefixVerification(
                    int(generation["verified_through_ordinal"]),
                    True,
                )
            start, end = window
            cur.execute(
                """
                SELECT output_ordinal, payload_sha256, payload_json, source_created_at
                FROM wandb_projection_rows
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                  AND output_ordinal BETWEEN %(start)s AND %(end)s
                ORDER BY output_ordinal
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "start": start,
                    "end": end,
                },
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
                    last_verified_at=CASE
                      WHEN %(through)s >= %(start)s THEN now() ELSE last_verified_at END,
                    latest_verified_source_at=CASE
                      WHEN %(through)s >= %(start)s THEN (
                        SELECT source_created_at
                        FROM wandb_projection_rows r
                        WHERE r.train_job_id=%(run)s
                          AND r.projection_generation=%(generation)s
                          AND r.output_ordinal=%(through)s
                      )
                      ELSE latest_verified_source_at END,
                    state = CASE
                      WHEN state IN ('publishing','verifying') THEN 'active'
                      ELSE state END,
                    activated_at=CASE
                      WHEN %(through)s >= %(start)s
                      THEN COALESCE(activated_at, now()) ELSE activated_at END
                WHERE train_job_id = %(run)s
                  AND projection_generation = %(generation)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "through": through,
                    "start": start,
                },
            )
            if through >= start:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET active_wandb_projection_generation=%(generation)s
                    WHERE id=%(run)s
                    """,
                    {
                        "run": int(train_job_id),
                        "generation": int(projection_generation),
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
    wandb_run_id: str | None = None,
    wandb_url: str | None = None,
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
                SET active_wandb_projection_generation = %(generation)s,
                    wandb_run_id = COALESCE(%(wandb_run_id)s, wandb_run_id),
                    wandb_url = COALESCE(NULLIF(%(wandb_url)s, ''), wandb_url),
                    wandb_ready_at = COALESCE(wandb_ready_at, now())
                WHERE id = %(run)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "wandb_run_id": (None if wandb_run_id is None else str(wandb_run_id)),
                    "wandb_url": None if wandb_url is None else str(wandb_url),
                },
            )
    set_publication_component(
        conn,
        train_job_id=int(train_job_id),
        component="history",
        state="live",
    )
    return local_fingerprint


def set_publication_component(
    conn,
    *,
    train_job_id: int,
    component: str,
    state: str,
    error: str | None = None,
) -> None:
    """Persist one lane without allowing it to erase another lane's failure."""

    if component not in {"history", "artifacts", "terminal"}:
        raise ValueError(f"unknown W&B publication component: {component}")
    if state not in {"pending", "live", "retrying", "complete", "failed", "disabled"}:
        raise ValueError(f"unknown W&B publication component state: {state}")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO wandb_publication_components (
                  train_job_id, component, state, attempts, error, completed_at
                ) VALUES (
                  %(run)s, %(component)s, %(state)s,
                  CASE WHEN %(state)s IN ('retrying','failed') THEN 1 ELSE 0 END,
                  %(error)s,
                  CASE WHEN %(state)s IN ('complete','disabled') THEN now() ELSE NULL END
                )
                ON CONFLICT (train_job_id, component) DO UPDATE
                SET state=EXCLUDED.state,
                    attempts=wandb_publication_components.attempts
                      + CASE WHEN EXCLUDED.state IN ('retrying','failed') THEN 1 ELSE 0 END,
                    error=EXCLUDED.error,
                    completed_at=CASE
                      WHEN EXCLUDED.state IN ('complete','disabled')
                      THEN COALESCE(wandb_publication_components.completed_at, now())
                      ELSE NULL END,
                    updated_at=now()
                """,
                {
                    "run": int(train_job_id),
                    "component": component,
                    "state": state,
                    "error": None if error is None else str(error)[:4000],
                },
            )
    reduce_publication_state(conn, train_job_id=int(train_job_id))


def publication_aggregate(
    states: Mapping[str, str],
    *,
    telemetry_no_more_producers: bool,
) -> str:
    required = {"history", "artifacts", "terminal"}
    if any(states.get(name) == "failed" for name in required):
        return "failed"
    if all(states.get(name) in {"complete", "disabled"} for name in required):
        return "complete"
    if telemetry_no_more_producers:
        return "finishing"
    if states:
        return "live"
    return "pending"


def reduce_publication_state(conn, *, train_job_id: int) -> str:
    """Reduce the independent history, artifact, and terminal facts."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT j.telemetry_no_more_producers,
                       c.component, c.state, c.error
                FROM train_jobs j
                LEFT JOIN wandb_publication_components c
                  ON c.train_job_id=j.id
                WHERE j.id=%(run)s
                ORDER BY c.component
                FOR UPDATE OF j
                """,
                {"run": int(train_job_id)},
            )
            rows = [dict(row) for row in cur.fetchall()]
            if not rows:
                raise ValueError("unknown train job")
            states = {
                str(row["component"]): str(row["state"])
                for row in rows
                if row.get("component") is not None
            }
            errors = [
                str(row["error"])
                for row in rows
                if row.get("error") and str(row.get("state")) in {"retrying", "failed"}
            ]
            aggregate = publication_aggregate(
                states,
                telemetry_no_more_producers=bool(rows[0]["telemetry_no_more_producers"]),
            )
            cur.execute(
                """
                UPDATE train_jobs
                SET live_publication_status=%(status)s,
                    live_publication_error=%(error)s,
                    live_publication_next_retry_at=CASE
                      WHEN %(status)s='failed' THEN live_publication_next_retry_at
                      ELSE NULL END
                WHERE id=%(run)s
                """,
                {
                    "run": int(train_job_id),
                    "status": aggregate,
                    "error": errors[0] if errors else None,
                },
            )
            if aggregate == "complete":
                cur.execute(
                    """
                    SELECT
                      EXISTS (
                        SELECT 1 FROM metric_batches
                        JOIN metric_streams s
                          ON s.stream_id=metric_batches.stream_id
                        JOIN worker_attempts w
                          ON w.attempt_id=s.attempt_id
                        WHERE w.train_job_id=%(run)s
                          AND metric_batches.wandb_confirmed_at IS NULL
                      ) AS batches_incomplete,
                      EXISTS (
                        SELECT 1 FROM telemetry_producers
                        WHERE train_job_id=%(run)s
                          AND final_sequence IS NULL
                      ) AS streams_open,
                      EXISTS (
                        SELECT 1 FROM artifact_announcement_ledger a
                        WHERE a.train_job_id=%(run)s
                          AND a.disposition='ready'
                          AND NOT EXISTS (
                            SELECT 1 FROM artifact_publication_receipts r
                            WHERE r.train_job_id=a.train_job_id
                              AND r.ledger_id=a.ledger_id
                              AND r.role='availability'
                              AND r.promotion_revision=0
                          )
                      ) AS receipts_missing
                    """,
                    {"run": int(train_job_id)},
                )
                invariant = dict(cur.fetchone())
                if any(bool(value) for value in invariant.values()):
                    cur.execute(
                        """
                        INSERT INTO telemetry_incidents (
                          train_job_id, telemetry_generation, incident_key,
                          severity, category, details_json
                        )
                        SELECT id, telemetry_generation,
                               'wandb-complete-with-incomplete-evidence',
                               'critical', 'wandb_publication_invariant',
                               %(details)s
                        FROM train_jobs WHERE id=%(run)s
                        ON CONFLICT (
                          train_job_id, telemetry_generation, incident_key
                        ) DO UPDATE SET last_seen_at=now(),
                                        details_json=EXCLUDED.details_json
                        """,
                        {
                            "run": int(train_job_id),
                            "details": json_arg(invariant),
                        },
                    )
                    raise RuntimeError(
                        "W&B publication cannot be complete while durable evidence is incomplete"
                    )
    from rlab.telemetry_reducer import reduce_run_integrity

    reduce_run_integrity(conn, train_job_id=int(train_job_id))
    return aggregate


def pending_projection_run_ids(conn, *, limit: int = 10_000) -> list[int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.id
            FROM train_jobs j
            WHERE j.telemetry_protocol_version=2
              AND COALESCE((j.train_config->>'wandb')::boolean, FALSE)
              AND j.status IN (
                'pending','launching','starting','running',
                'finalizing','finalization_failed'
              )
              AND (
                NOT j.telemetry_no_more_producers
                OR NOT EXISTS (
                  SELECT 1 FROM wandb_projection_generations g
                  WHERE g.train_job_id=j.id AND g.state='complete'
                )
              )
            ORDER BY j.created_at, j.id
            LIMIT %(limit)s
            """,
            {"limit": max(1, int(limit))},
        )
        return [int(row["id"]) for row in cur.fetchall()]


def ensure_projection_close_row(
    conn,
    *,
    train_job_id: int,
    projection_generation: int,
) -> int | None:
    """Append the one deterministic close sentinel once all exact inputs are ready."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT g.*, j.telemetry_generation, j.telemetry_no_more_producers,
                       j.telemetry_frozen_at,
                       root.root_sha256
                FROM wandb_projection_generations g
                JOIN train_jobs j ON j.id=g.train_job_id
                LEFT JOIN telemetry_archive_roots root
                  ON root.train_job_id=j.id
                 AND root.telemetry_generation=j.telemetry_generation
                WHERE g.train_job_id=%(run)s
                  AND g.projection_generation=%(generation)s
                FOR UPDATE OF g
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            generation = cur.fetchone()
            if not generation:
                raise ValueError("unknown W&B projection generation")
            if generation["close_ordinal"] is not None:
                return int(generation["close_ordinal"])
            if not bool(generation["telemetry_no_more_producers"]) or not generation["root_sha256"]:
                return None
            cur.execute(
                """
                SELECT c.telemetry_generation, c.producer_ordinal,
                       c.consumed_through_sequence, p.final_sequence
                FROM wandb_projection_source_cursors c
                JOIN telemetry_producers p
                  ON p.train_job_id=c.train_job_id
                 AND p.telemetry_generation=c.telemetry_generation
                 AND p.producer_ordinal=c.producer_ordinal
                WHERE c.train_job_id=%(run)s
                  AND c.projection_generation=%(generation)s
                ORDER BY c.telemetry_generation, c.producer_ordinal
                """,
                {"run": int(train_job_id), "generation": int(projection_generation)},
            )
            cursors = [dict(row) for row in cur.fetchall()]
            if not cursors or any(
                row["final_sequence"] is None
                or int(row["consumed_through_sequence"]) < int(row["final_sequence"])
                for row in cursors
            ):
                return None
            cur.execute(
                """
                SELECT a.ledger_id, r.role, r.promotion_revision,
                       r.disposition, r.artifact_kind, r.checkpoint_step,
                       r.checkpoint_sha256, r.announcement_sha256,
                       r.artifact_version, r.artifact_ref, r.expected_aliases
                FROM artifact_announcement_ledger a
                LEFT JOIN artifact_publication_receipts r
                  ON r.train_job_id=a.train_job_id
                 AND r.ledger_id=a.ledger_id
                 AND r.role='availability'
                 AND r.promotion_revision=0
                WHERE a.train_job_id=%(run)s
                  AND a.disposition='ready'
                ORDER BY a.ledger_id
                """,
                {"run": int(train_job_id)},
            )
            receipts = [dict(row) for row in cur.fetchall()]
            if any(row["role"] is None for row in receipts):
                return None
            ordinal = int(generation["next_ordinal"])
            cur.execute(
                """
                SELECT payload_sha256
                FROM wandb_projection_rows
                WHERE train_job_id=%(run)s
                  AND projection_generation=%(generation)s
                  AND output_ordinal=%(ordinal)s - 1
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "ordinal": ordinal,
                },
            )
            predecessor_row = cur.fetchone()
            predecessor = (
                None if predecessor_row is None else str(predecessor_row["payload_sha256"])
            )
            bindings = {
                "archive_root_sha256": str(generation["root_sha256"]),
                "cursor_set_sha256": sha256_json(cursors),
                "selected_output_chain_sha256": generation["projection_chain_sha256"],
                "artifact_receipt_set_sha256": sha256_json(receipts),
            }
            stable_key = f"train/{int(train_job_id)}/projection/{int(projection_generation)}/close"
            core_payload = {
                "_rlab_event_id": stable_key,
                "_rlab_output_ordinal": ordinal,
                "_rlab_projection_close": True,
                **bindings,
            }
            digest = sha256_json(core_payload)
            payload = {
                **core_payload,
                "_rlab_payload_sha256": digest,
                "_rlab_predecessor_sha256": predecessor,
                "_rlab_projection_generation": int(projection_generation),
                "_rlab_projection_protocol": BOUNDED_PROJECTION_PROTOCOL_VERSION,
            }
            cur.execute(
                """
                INSERT INTO wandb_projection_rows (
                  train_job_id, projection_generation, output_ordinal,
                  stable_key, source_event_id, adapter_version,
                  normalization_version, output_kind, output_index,
                  predecessor_sha256, payload_sha256, payload_json, source_created_at
                ) VALUES (
                  %(run)s, %(generation)s, %(ordinal)s, %(stable_key)s,
                  %(stable_key)s, 'projection-close-v1', 'projection-close-v1',
                  'projection_close', 0, %(predecessor)s, %(digest)s,
                  %(payload)s, now()
                )
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "ordinal": ordinal,
                    "stable_key": stable_key,
                    "predecessor": predecessor,
                    "digest": digest,
                    "payload": json_arg(payload),
                },
            )
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET next_ordinal=%(next)s, close_ordinal=%(ordinal)s,
                    state='draining', terminal_started_at=COALESCE(
                      terminal_started_at, %(frozen_at)s, now()
                    )
                WHERE train_job_id=%(run)s
                  AND projection_generation=%(generation)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "ordinal": ordinal,
                    "next": ordinal + 1,
                    "frozen_at": generation["telemetry_frozen_at"],
                },
            )
    return ordinal


def mark_projection_terminal_complete(
    conn,
    *,
    train_job_id: int,
    projection_generation: int,
    remote_state: str,
    remote_last_step: int,
) -> bool:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET state='complete', completed_at=COALESCE(completed_at, now())
                WHERE train_job_id=%(run)s
                  AND projection_generation=%(generation)s
                  AND close_ordinal IS NOT NULL
                  AND verified_through_ordinal=close_ordinal
                  AND %(remote_state)s='finished'
                  AND %(remote_last_step)s=step_offset + close_ordinal
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(projection_generation),
                    "remote_state": str(remote_state),
                    "remote_last_step": int(remote_last_step),
                },
            )
            complete = cur.rowcount == 1
            if complete:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET active_wandb_projection_generation=%(generation)s
                    WHERE id=%(run)s
                    """,
                    {
                        "run": int(train_job_id),
                        "generation": int(projection_generation),
                    },
                )
    if complete:
        set_publication_component(
            conn,
            train_job_id=int(train_job_id),
            component="history",
            state="complete",
        )
        set_publication_component(
            conn,
            train_job_id=int(train_job_id),
            component="terminal",
            state="complete",
        )
    return complete


def terminal_completion_eligible(
    *,
    close_ordinal: int | None,
    verified_through_ordinal: int,
    step_offset: int,
    remote_state: str,
    remote_last_step: int,
) -> bool:
    return bool(
        close_ordinal is not None
        and verified_through_ordinal == close_ordinal
        and remote_state == "finished"
        and remote_last_step == step_offset + close_ordinal
    )


def projection_observability(conn, *, train_job_id: int) -> dict[str, Any]:
    """Return bounded backlog/freshness counters for one active generation."""

    with conn.cursor() as cur:
        cur.execute(
            """
            WITH g AS (
              SELECT * FROM wandb_projection_generations
              WHERE train_job_id=%(run)s
                AND state NOT IN ('sealed','quarantined','failed','disabled')
              ORDER BY projection_generation DESC LIMIT 1
            )
            SELECT
              g.projection_generation,
              g.source_events_consumed,
              g.raw_frames_consumed,
              g.selected_rows,
              g.source_events_consumed / GREATEST(
                EXTRACT(EPOCH FROM now() - g.created_at), 1
              ) AS source_events_per_second,
              g.raw_frames_consumed / GREATEST(
                EXTRACT(EPOCH FROM now() - g.created_at), 1
              ) AS raw_frames_per_second,
              g.selected_rows / GREATEST(
                EXTRACT(EPOCH FROM now() - g.created_at), 1
              ) AS selected_rows_per_second,
              GREATEST(g.submitted_through_ordinal + 1, 0) / GREATEST(
                EXTRACT(EPOCH FROM now() - g.created_at), 1
              ) AS submitted_rows_per_second,
              GREATEST(g.verified_through_ordinal + 1, 0) / GREATEST(
                EXTRACT(EPOCH FROM now() - g.created_at), 1
              ) AS verified_rows_per_second,
              g.submitted_through_ordinal,
              g.verified_through_ordinal,
              GREATEST(g.next_ordinal - g.submitted_through_ordinal - 1, 0)
                AS row_backlog,
              (
                SELECT count(*) FROM telemetry_events e
                JOIN wandb_projection_source_cursors c
                  ON c.train_job_id=e.train_job_id
                 AND c.projection_generation=g.projection_generation
                 AND c.telemetry_generation=e.telemetry_generation
                 AND c.producer_ordinal=e.producer_ordinal
                WHERE e.train_job_id=%(run)s
                  AND e.source_sequence > c.consumed_through_sequence
              ) AS unconsumed_events,
              EXTRACT(EPOCH FROM now() - (
                SELECT min(r.source_created_at)
                FROM wandb_projection_rows r
                WHERE r.train_job_id=%(run)s
                  AND r.projection_generation=g.projection_generation
                  AND r.state <> 'verified'
              )) AS oldest_backlog_age_seconds,
              EXTRACT(EPOCH FROM now() - COALESCE(
                g.latest_verified_source_at, g.created_at
              ))
                AS wandb_freshness_seconds,
              EXTRACT(EPOCH FROM g.completed_at - g.terminal_started_at)
                AS terminal_drain_seconds
            FROM g
            """,
            {"run": int(train_job_id)},
        )
        row = cur.fetchone()
    return {} if row is None else dict(row)
