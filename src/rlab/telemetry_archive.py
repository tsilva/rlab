from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence

from rlab.job_queue import json_arg
from rlab.modal_eval_storage import ObjectStore
from rlab.telemetry_integrity import (
    CANONICAL_FORMAT_VERSION,
    CanonicalEvent,
    CanonicalSegment,
    ProducerKey,
    build_canonical_segment,
    canonical_json_bytes,
    decode_canonical_segment,
    sha256_bytes,
    sha256_json,
)
from rlab.telemetry_mailbox import decode_metric_batch


ARCHIVER_VERSION = "telemetry-archiver-v1"
POLICY_RECEIPT_ROLES = {
    "queued_dual_r2_v1": ("primary", "backup"),
    "local_mirrored_v1": ("local", "mirror"),
    "local_singlecopy_optout_v1": ("local",),
}


@dataclass(frozen=True)
class ArchiveCopy:
    role: str
    store: ObjectStore


@dataclass(frozen=True)
class ArchiveClaim:
    train_job_id: int
    telemetry_generation: int
    producer_ordinal: int
    producer_identity: str
    first_sequence: int
    last_sequence: int
    segment: CanonicalSegment
    claim_owner: str


def archive_policy_sha256(policy_name: str, copies: Sequence[ArchiveCopy]) -> str:
    required = POLICY_RECEIPT_ROLES.get(policy_name)
    roles = tuple(copy.role for copy in copies)
    if required is None:
        raise ValueError(f"unknown telemetry archive policy: {policy_name}")
    if roles != required:
        raise ValueError(f"{policy_name} requires archive roles {required}, observed {roles}")
    roots = [copy.store.base_uri for copy in copies]
    if len(set(roots)) != len(roots):
        raise ValueError("telemetry archive copies require independently rooted stores")
    return sha256_json(
        {
            "version": "telemetry-archive-policy-v1",
            "name": policy_name,
            "copies": [
                {"role": copy.role, "base_uri": copy.store.base_uri} for copy in copies
            ],
        }
    )


def canonical_event_from_ledger_row(row: Mapping[str, Any]) -> CanonicalEvent:
    encoding = str(row["payload_encoding"])
    raw = bytes(row["payload"])
    if sha256_bytes(raw) != str(row["payload_sha256"]):
        raise RuntimeError("canonical telemetry event payload digest mismatch")
    if encoding == "metric_batch_zlib_json_v1":
        payload: Mapping[str, object] = {
            "encoding": encoding,
            "frames": decode_metric_batch(raw),
            "source_payload_sha256": str(row["payload_sha256"]),
            "ledger_event_sha256": str(row["event_sha256"]),
            "predecessor_sha256": row.get("predecessor_sha256"),
            "terminal": bool(row.get("terminal")),
        }
        frames = payload["frames"]
        global_step = None
        if isinstance(frames, list) and len(frames) == 1:
            candidate = frames[0].get("global_step")
            global_step = None if candidate is None else int(candidate)
    elif encoding == "canonical_json_v1":
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise RuntimeError("canonical JSON telemetry event must be a mapping")
        payload = {
            "encoding": encoding,
            "value": value,
            "source_payload_sha256": str(row["payload_sha256"]),
            "ledger_event_sha256": str(row["event_sha256"]),
            "predecessor_sha256": row.get("predecessor_sha256"),
            "terminal": bool(row.get("terminal")),
        }
        global_step = None
    else:
        raise RuntimeError(f"unsupported canonical event encoding: {encoding}")
    return CanonicalEvent(
        producer=ProducerKey(
            run_id=int(row["train_job_id"]),
            generation=int(row["telemetry_generation"]),
            producer_ordinal=int(row["producer_ordinal"]),
            producer_identity=str(row["producer_identity"]),
        ),
        source_sequence=int(row["source_sequence"]),
        event_id=str(row["event_identity"]),
        kind=str(row["event_kind"]),
        payload=payload,
        global_step=global_step,
    )


def claim_next_segment(
    conn,
    *,
    owner: str,
    max_events: int = 1000,
    train_job_id: int | None = None,
) -> ArchiveClaim | None:
    """Persist an exact byte claim and release the transaction before remote I/O."""

    owner = str(owner).strip()
    if not owner:
        raise ValueError("archive claim owner is required")
    params: dict[str, Any] = {
        "owner": owner,
        "limit": max(1, min(int(max_events), 1000)),
        "train_job_id": train_job_id,
    }
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  p.train_job_id, p.telemetry_generation, p.producer_ordinal,
                  p.producer_identity,
                  COALESCE((
                    SELECT max(s.last_sequence)
                    FROM telemetry_archive_segments s
                    WHERE s.train_job_id = p.train_job_id
                      AND s.telemetry_generation = p.telemetry_generation
                      AND s.producer_ordinal = p.producer_ordinal
                      AND s.claim_state = 'verified'
                  ), 0) + 1 AS first_unarchived
                FROM telemetry_producers p
                JOIN train_jobs j ON j.id = p.train_job_id
                WHERE (%(train_job_id)s IS NULL OR p.train_job_id = %(train_job_id)s)
                  AND EXISTS (
                    SELECT 1 FROM telemetry_events e
                    WHERE e.train_job_id = p.train_job_id
                      AND e.telemetry_generation = p.telemetry_generation
                      AND e.producer_ordinal = p.producer_ordinal
                      AND e.source_sequence > COALESCE((
                        SELECT max(s.last_sequence)
                        FROM telemetry_archive_segments s
                        WHERE s.train_job_id = p.train_job_id
                          AND s.telemetry_generation = p.telemetry_generation
                          AND s.producer_ordinal = p.producer_ordinal
                          AND s.claim_state = 'verified'
                      ), 0)
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM telemetry_archive_segments active
                    WHERE active.train_job_id = p.train_job_id
                      AND active.telemetry_generation = p.telemetry_generation
                      AND active.producer_ordinal = p.producer_ordinal
                      AND active.claim_state IN ('claimed', 'writing')
                      AND active.claim_expires_at > now()
                  )
                ORDER BY j.created_at, p.train_job_id, p.producer_ordinal
                FOR UPDATE OF p SKIP LOCKED
                LIMIT 1
                """,
                params,
            )
            producer = cur.fetchone()
            if not producer:
                return None
            cur.execute(
                """
                SELECT e.*, p.producer_identity
                FROM telemetry_events e
                JOIN telemetry_producers p
                  ON p.train_job_id = e.train_job_id
                 AND p.telemetry_generation = e.telemetry_generation
                 AND p.producer_ordinal = e.producer_ordinal
                WHERE e.train_job_id = %(train_job_id)s
                  AND e.telemetry_generation = %(generation)s
                  AND e.producer_ordinal = %(producer_ordinal)s
                  AND e.source_sequence >= %(first_sequence)s
                ORDER BY e.source_sequence
                LIMIT %(limit)s
                """,
                {
                    "train_job_id": int(producer["train_job_id"]),
                    "generation": int(producer["telemetry_generation"]),
                    "producer_ordinal": int(producer["producer_ordinal"]),
                    "first_sequence": int(producer["first_unarchived"]),
                    "limit": params["limit"],
                },
            )
            rows = [dict(row) for row in cur.fetchall()]
            if not rows or int(rows[0]["source_sequence"]) != int(
                producer["first_unarchived"]
            ):
                raise RuntimeError("canonical telemetry archive coverage contains a gap")
            segment = build_canonical_segment(
                canonical_event_from_ledger_row(row) for row in rows
            )
            first_sequence = int(rows[0]["source_sequence"])
            last_sequence = int(rows[-1]["source_sequence"])
            cur.execute(
                """
                INSERT INTO telemetry_archive_segments (
                  train_job_id, telemetry_generation, producer_ordinal,
                  first_sequence, last_sequence, event_count, format_version,
                  uncompressed_sha256, compressed_sha256, byte_count,
                  claim_state, claim_owner, claim_expires_at
                ) VALUES (
                  %(train_job_id)s, %(generation)s, %(producer_ordinal)s,
                  %(first_sequence)s, %(last_sequence)s, %(event_count)s,
                  %(format_version)s, %(uncompressed_sha256)s,
                  %(compressed_sha256)s, %(byte_count)s,
                  'claimed', %(owner)s, now() + interval '5 minutes'
                )
                ON CONFLICT (
                  train_job_id, telemetry_generation, producer_ordinal,
                  first_sequence, last_sequence
                ) DO UPDATE
                SET claim_owner = EXCLUDED.claim_owner,
                    claim_expires_at = EXCLUDED.claim_expires_at,
                    claim_state = CASE
                      WHEN telemetry_archive_segments.compressed_sha256 =
                           EXCLUDED.compressed_sha256
                      THEN 'claimed'
                      ELSE 'conflict'
                    END
                RETURNING claim_state
                """,
                {
                    "train_job_id": int(producer["train_job_id"]),
                    "generation": int(producer["telemetry_generation"]),
                    "producer_ordinal": int(producer["producer_ordinal"]),
                    "first_sequence": first_sequence,
                    "last_sequence": last_sequence,
                    "event_count": segment.event_count,
                    "format_version": CANONICAL_FORMAT_VERSION,
                    "uncompressed_sha256": segment.uncompressed_sha256,
                    "compressed_sha256": segment.compressed_sha256,
                    "byte_count": len(segment.payload),
                    "owner": owner,
                },
            )
            if str(cur.fetchone()["claim_state"]) == "conflict":
                raise RuntimeError("canonical archive segment claim conflicts with prior bytes")
    return ArchiveClaim(
        train_job_id=int(producer["train_job_id"]),
        telemetry_generation=int(producer["telemetry_generation"]),
        producer_ordinal=int(producer["producer_ordinal"]),
        producer_identity=str(producer["producer_identity"]),
        first_sequence=first_sequence,
        last_sequence=last_sequence,
        segment=segment,
        claim_owner=owner,
    )


def _segment_key(claim: ArchiveClaim) -> str:
    return (
        f"telemetry/v2/run-{claim.train_job_id}/generation-"
        f"{claim.telemetry_generation}/producer-{claim.producer_ordinal}/"
        f"{claim.first_sequence:020d}-{claim.last_sequence:020d}-"
        f"{claim.segment.compressed_sha256}.jsonl.gz"
    )


def write_claimed_segment(
    conn,
    claim: ArchiveClaim,
    *,
    policy_name: str,
    copies: Sequence[ArchiveCopy],
) -> list[dict[str, Any]]:
    """Write and fully read-verify every required copy without an open DB transaction."""

    policy_sha256 = archive_policy_sha256(policy_name, copies)
    key = _segment_key(claim)
    receipts: list[dict[str, Any]] = []
    for copy in copies:
        uri = copy.store.put_bytes(
            key,
            claim.segment.payload,
            content_type="application/gzip",
            create_only=True,
            metadata={
                "sha256": claim.segment.compressed_sha256,
                "uncompressed-sha256": claim.segment.uncompressed_sha256,
                "format-version": CANONICAL_FORMAT_VERSION,
            },
        )
        observed = copy.store.get_bytes(uri)
        if observed != claim.segment.payload:
            raise RuntimeError(f"telemetry archive full read verification failed: {uri}")
        documents = decode_canonical_segment(observed)
        if len(documents) != claim.segment.event_count:
            raise RuntimeError(f"telemetry archive event coverage mismatch: {uri}")
        receipts.append(
            {
                "version": "telemetry-archive-receipt-v1",
                "copy_role": copy.role,
                "policy_name": policy_name,
                "policy_sha256": policy_sha256,
                "object_uri": uri,
                "object_version": claim.segment.compressed_sha256,
                "compressed_sha256": sha256_bytes(observed),
                "uncompressed_sha256": claim.segment.uncompressed_sha256,
                "event_count": claim.segment.event_count,
                "first_sequence": claim.first_sequence,
                "last_sequence": claim.last_sequence,
                "verified_at": datetime.now(UTC).isoformat(),
            }
        )
    with conn:
        with conn.cursor() as cur:
            for receipt in receipts:
                cur.execute(
                    """
                    INSERT INTO telemetry_archive_receipts (
                      train_job_id, telemetry_generation, producer_ordinal,
                      first_sequence, last_sequence, copy_role, policy_name,
                      policy_sha256, object_uri, object_version,
                      compressed_sha256, full_read_verified_at, receipt_json
                    ) VALUES (
                      %(train_job_id)s, %(generation)s, %(producer_ordinal)s,
                      %(first_sequence)s, %(last_sequence)s, %(copy_role)s,
                      %(policy_name)s, %(policy_sha256)s, %(object_uri)s,
                      %(object_version)s, %(compressed_sha256)s, now(),
                      %(receipt_json)s
                    )
                    ON CONFLICT (
                      train_job_id, telemetry_generation, producer_ordinal,
                      first_sequence, last_sequence, copy_role
                    ) DO UPDATE
                    SET receipt_json = CASE
                      WHEN telemetry_archive_receipts.compressed_sha256 =
                           EXCLUDED.compressed_sha256
                       AND telemetry_archive_receipts.object_uri = EXCLUDED.object_uri
                      THEN EXCLUDED.receipt_json
                      ELSE telemetry_archive_receipts.receipt_json
                    END
                    """,
                    {
                        "train_job_id": claim.train_job_id,
                        "generation": claim.telemetry_generation,
                        "producer_ordinal": claim.producer_ordinal,
                        "first_sequence": claim.first_sequence,
                        "last_sequence": claim.last_sequence,
                        "copy_role": receipt["copy_role"],
                        "policy_name": policy_name,
                        "policy_sha256": policy_sha256,
                        "object_uri": receipt["object_uri"],
                        "object_version": receipt["object_version"],
                        "compressed_sha256": receipt["compressed_sha256"],
                        "receipt_json": json_arg(receipt),
                    },
                )
            cur.execute(
                """
                UPDATE telemetry_archive_segments
                SET claim_state = 'verified', verified_at = now(),
                    claim_owner = NULL, claim_expires_at = NULL
                WHERE train_job_id = %(train_job_id)s
                  AND telemetry_generation = %(generation)s
                  AND producer_ordinal = %(producer_ordinal)s
                  AND first_sequence = %(first_sequence)s
                  AND last_sequence = %(last_sequence)s
                  AND compressed_sha256 = %(compressed_sha256)s
                  AND claim_owner = %(owner)s
                """,
                {
                    "train_job_id": claim.train_job_id,
                    "generation": claim.telemetry_generation,
                    "producer_ordinal": claim.producer_ordinal,
                    "first_sequence": claim.first_sequence,
                    "last_sequence": claim.last_sequence,
                    "compressed_sha256": claim.segment.compressed_sha256,
                    "owner": claim.claim_owner,
                },
            )
            if cur.rowcount != 1:
                raise RuntimeError("telemetry archive claim expired or changed before commit")
    return receipts


def build_archive_root_document(
    *,
    train_job_id: int,
    generation: int,
    policy_name: str,
    expected_obligations: Sequence[Mapping[str, Any]],
    producers: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    evidence_roots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_set_sha256 = sha256_json(
        [
            {
                "key": row["obligation_key"],
                "kind": row["obligation_kind"],
                "producer_ordinal": int(row["producer_ordinal"]),
                "expected": row["expected_disposition"],
                "realized": row["realized_disposition"],
                "evidence_sha256": row["evidence_sha256"],
            }
            for row in sorted(expected_obligations, key=lambda item: item["obligation_key"])
        ]
    )
    coverage = [
        {
            "producer_ordinal": int(row["producer_ordinal"]),
            "producer_identity": row["producer_identity"],
            "final_sequence": int(row["final_sequence"]),
            "final_sha256": row["final_sha256"],
        }
        for row in sorted(producers, key=lambda item: int(item["producer_ordinal"]))
    ]
    return {
        "version": "telemetry-run-root-v1",
        "train_job_id": int(train_job_id),
        "telemetry_generation": int(generation),
        "policy_name": policy_name,
        "expected_set_sha256": expected_set_sha256,
        "coverage_sha256": sha256_json(coverage),
        "producer_claims": coverage,
        "segments": [
            dict(row)
            for row in sorted(
                segments,
                key=lambda item: (
                    int(item["producer_ordinal"]),
                    int(item["first_sequence"]),
                ),
            )
        ],
        "evidence_roots": [
            dict(row)
            for row in sorted(evidence_roots, key=lambda item: str(item["scope_sha256"]))
        ],
    }


def finalize_exact_archive_root(conn, *, train_job_id: int) -> dict[str, Any]:
    """Atomically freeze and root a run only after exact closure and receipt proof."""

    finalized: dict[str, Any]
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM train_jobs WHERE id = %(run)s FOR UPDATE
                """,
                {"run": int(train_job_id)},
            )
            job = cur.fetchone()
            if not job:
                raise ValueError(f"unknown train job: {train_job_id}")
            if str(job["telemetry_durability_policy"]) == "local_singlecopy_optout_v1":
                raise RuntimeError("durability-opted-out telemetry cannot have an exact root")
            cur.execute(
                """
                UPDATE train_jobs
                SET telemetry_no_more_producers = TRUE
                WHERE id = %(run)s
                  AND telemetry_protocol_version = 2
                  AND telemetry_frozen_at IS NULL
                  AND status IN ('succeeded', 'failed', 'canceled', 'finalization_failed')
                """,
                {"run": int(train_job_id)},
            )
            cur.execute(
                """
                UPDATE telemetry_recovery_manifests
                SET state = 'complete', owner = NULL, last_error = NULL,
                    updated_at = now()
                WHERE train_job_id = %(run)s
                  AND telemetry_generation = %(generation)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(job["telemetry_generation"]),
                },
            )
            cur.execute(
                """
                SELECT * FROM telemetry_producers
                WHERE train_job_id = %(run)s
                  AND telemetry_generation = %(generation)s
                ORDER BY producer_ordinal
                """,
                {"run": int(train_job_id), "generation": int(job["telemetry_generation"])},
            )
            producers = [dict(row) for row in cur.fetchall()]
            if not producers or any(row["final_sequence"] is None for row in producers):
                raise RuntimeError("not every expected telemetry producer has a final claim")
            cur.execute(
                """
                SELECT * FROM telemetry_expected_obligations
                WHERE train_job_id = %(run)s
                  AND telemetry_generation = %(generation)s
                ORDER BY obligation_key
                """,
                {"run": int(train_job_id), "generation": int(job["telemetry_generation"])},
            )
            obligations = [dict(row) for row in cur.fetchall()]
            if not obligations or any(
                row["realized_disposition"] != row["expected_disposition"]
                for row in obligations
            ):
                raise RuntimeError("telemetry obligation set is not exactly realized")
            cur.execute(
                """
                SELECT
                  s.producer_ordinal, s.first_sequence, s.last_sequence,
                  s.event_count, s.format_version, s.uncompressed_sha256,
                  s.compressed_sha256,
                  jsonb_agg(r.receipt_json ORDER BY r.copy_role) AS receipts
                FROM telemetry_archive_segments s
                JOIN telemetry_archive_receipts r
                  ON r.train_job_id = s.train_job_id
                 AND r.telemetry_generation = s.telemetry_generation
                 AND r.producer_ordinal = s.producer_ordinal
                 AND r.first_sequence = s.first_sequence
                 AND r.last_sequence = s.last_sequence
                WHERE s.train_job_id = %(run)s
                  AND s.telemetry_generation = %(generation)s
                  AND s.claim_state = 'verified'
                GROUP BY
                  s.producer_ordinal, s.first_sequence, s.last_sequence,
                  s.event_count, s.format_version, s.uncompressed_sha256,
                  s.compressed_sha256
                ORDER BY s.producer_ordinal, s.first_sequence
                """,
                {"run": int(train_job_id), "generation": int(job["telemetry_generation"])},
            )
            segments = [dict(row) for row in cur.fetchall()]
            policy_name = str(job["telemetry_durability_policy"])
            required_roles = set(POLICY_RECEIPT_ROLES[policy_name])
            coverage: dict[int, int] = {}
            for segment in segments:
                roles = {receipt["copy_role"] for receipt in segment["receipts"]}
                if roles != required_roles:
                    raise RuntimeError("archive segment is missing a policy-required receipt")
                ordinal = int(segment["producer_ordinal"])
                expected_first = coverage.get(ordinal, 0) + 1
                if int(segment["first_sequence"]) != expected_first:
                    raise RuntimeError("archive segment coverage contains a gap")
                coverage[ordinal] = int(segment["last_sequence"])
            for producer in producers:
                if coverage.get(int(producer["producer_ordinal"])) != int(
                    producer["final_sequence"]
                ):
                    raise RuntimeError("archive coverage does not match producer final claim")
            cur.execute(
                """
                SELECT scope_kind, scope_key, scope_sha256, root_sha256
                FROM telemetry_evidence_scopes
                WHERE train_job_id = %(run)s AND state = 'exact'
                ORDER BY scope_kind, scope_key
                """,
                {"run": int(train_job_id)},
            )
            evidence = [dict(row) for row in cur.fetchall()]
            root = build_archive_root_document(
                train_job_id=int(train_job_id),
                generation=int(job["telemetry_generation"]),
                policy_name=policy_name,
                expected_obligations=obligations,
                producers=producers,
                segments=segments,
                evidence_roots=evidence,
            )
            root_sha256 = sha256_bytes(canonical_json_bytes(root))
            cur.execute(
                """
                INSERT INTO telemetry_archive_roots (
                  train_job_id, telemetry_generation, root_kind, root_sha256,
                  expected_set_sha256, coverage_sha256, policy_name, root_json
                ) VALUES (
                  %(run)s, %(generation)s, 'exact', %(root_sha256)s,
                  %(expected_set_sha256)s, %(coverage_sha256)s,
                  %(policy_name)s, %(root_json)s
                )
                ON CONFLICT (train_job_id, telemetry_generation) DO UPDATE
                SET root_json = CASE
                  WHEN telemetry_archive_roots.root_sha256 = EXCLUDED.root_sha256
                  THEN EXCLUDED.root_json
                  ELSE telemetry_archive_roots.root_json
                END
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(job["telemetry_generation"]),
                    "root_sha256": root_sha256,
                    "expected_set_sha256": root["expected_set_sha256"],
                    "coverage_sha256": root["coverage_sha256"],
                    "policy_name": policy_name,
                    "root_json": json_arg(root),
                },
            )
            cur.execute(
                """
                UPDATE train_jobs
                SET telemetry_frozen_at = COALESCE(telemetry_frozen_at, now()),
                    telemetry_integrity_classification = 'intact_with_proof'
                WHERE id = %(run)s
                """,
                {"run": int(train_job_id)},
            )
            cur.execute(
                """
                UPDATE metric_batches b
                SET archived_at = COALESCE(b.archived_at, now()),
                    archive_root_sha256 = %(root_sha256)s
                FROM metric_streams s, worker_attempts w
                WHERE b.stream_id = s.stream_id
                  AND s.attempt_id = w.attempt_id
                  AND w.train_job_id = %(run)s
                """,
                {"run": int(train_job_id), "root_sha256": root_sha256},
            )
            finalized = {**root, "root_sha256": root_sha256}
    from rlab.telemetry_reducer import reduce_run_integrity

    reduce_run_integrity(conn, train_job_id=int(train_job_id))
    return finalized


def delete_rooted_operational_batches(
    conn, *, train_job_id: int, cutover_generation: int
) -> int:
    """Delete only redundant buffers already bound to a verified root."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('rlab.telemetry_delete_generation', %(value)s, TRUE)",
                {"value": str(int(cutover_generation))},
            )
            cur.execute(
                """
                DELETE FROM metric_batches b
                USING metric_streams s, worker_attempts w, telemetry_integrity i
                WHERE b.stream_id = s.stream_id
                  AND s.attempt_id = w.attempt_id
                  AND w.train_job_id = %(run)s
                  AND i.train_job_id = w.train_job_id
                  AND i.cleanup_eligible = TRUE
                  AND b.archived_at IS NOT NULL
                  AND b.archive_root_sha256 IS NOT NULL
                """,
                {"run": int(train_job_id)},
            )
            return int(cur.rowcount)


def prepare_legacy_forensic_ledger(conn, *, train_job_id: int) -> int:
    """Ledger every surviving v1 byte without claiming that deleted bytes are recoverable."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM train_jobs
                WHERE id = %(run)s AND telemetry_protocol_version = 1
                FOR UPDATE
                """,
                {"run": int(train_job_id)},
            )
            job = cur.fetchone()
            if not job:
                raise ValueError("legacy forensic preparation requires a protocol-v1 run")
            if str(job["telemetry_integrity_classification"]) == "intact_with_proof":
                raise RuntimeError("legacy forensic preparation cannot manufacture intact proof")
            cur.execute(
                """
                SELECT attempt_id,
                       row_number() OVER (ORDER BY created_at, attempt_id) - 1 AS ordinal
                FROM worker_attempts
                WHERE train_job_id = %(run)s
                ORDER BY created_at, attempt_id
                """,
                {"run": int(train_job_id)},
            )
            attempts = [dict(row) for row in cur.fetchall()]
            for attempt in attempts:
                ordinal = int(attempt["ordinal"])
                identity = f"legacy-forensic:{attempt['attempt_id']}"
                cur.execute(
                    """
                    INSERT INTO telemetry_producers (
                      train_job_id, telemetry_generation, producer_ordinal,
                      producer_identity, attempt_id, producer_kind, state
                    ) VALUES (
                      %(run)s, 1, %(ordinal)s, %(identity)s, %(attempt_id)s,
                      'legacy_forensic', 'registered'
                    )
                    ON CONFLICT (
                      train_job_id, telemetry_generation, producer_ordinal
                    ) DO NOTHING
                    """,
                    {
                        "run": int(train_job_id),
                        "ordinal": ordinal,
                        "identity": identity,
                        "attempt_id": str(attempt["attempt_id"]),
                    },
                )
                cur.execute(
                    """
                    INSERT INTO telemetry_expected_obligations (
                      train_job_id, telemetry_generation, obligation_key,
                      obligation_kind, producer_ordinal
                    ) VALUES (
                      %(run)s, 1, %(key)s, 'legacy_forensic', %(ordinal)s
                    )
                    ON CONFLICT DO NOTHING
                    """,
                    {
                        "run": int(train_job_id),
                        "key": f"legacy-forensic:{attempt['attempt_id']}",
                        "ordinal": ordinal,
                    },
                )
                cur.execute(
                    """
                    SELECT b.batch_sequence, b.payload
                    FROM metric_batches b
                    JOIN metric_streams s ON s.stream_id = b.stream_id
                    WHERE s.attempt_id = %(attempt_id)s
                    ORDER BY b.batch_sequence
                    """,
                    {"attempt_id": str(attempt["attempt_id"])},
                )
                for batch in cur.fetchall():
                    cur.execute(
                        """
                        SELECT rlab_append_canonical_telemetry_event(
                          %(attempt_id)s, %(identity)s, 'legacy_metric_batch',
                          'metric_batch_zlib_json_v1', %(payload)s, FALSE
                        )
                        """,
                        {
                            "attempt_id": str(attempt["attempt_id"]),
                            "identity": f"legacy-batch:{int(batch['batch_sequence'])}",
                            "payload": bytes(batch["payload"]),
                        },
                    )
                cur.execute(
                    """
                    SELECT rlab_append_canonical_telemetry_event(
                      %(attempt_id)s, 'legacy-forensic-close',
                      'legacy_forensic_terminal', 'canonical_json_v1',
                      convert_to(
                        jsonb_build_object(
                          'classification', %(classification)s,
                          'claim', 'surviving-bytes-only'
                        )::text,
                        'UTF8'
                      ),
                      TRUE
                    )
                    """,
                    {
                        "attempt_id": str(attempt["attempt_id"]),
                        "classification": str(
                            job["telemetry_integrity_classification"]
                        ),
                    },
                )
            cur.execute(
                """
                UPDATE train_jobs
                SET telemetry_no_more_producers = TRUE,
                    telemetry_durability_policy = COALESCE(
                      telemetry_durability_policy, 'queued_dual_r2_v1'
                    )
                WHERE id = %(run)s
                """,
                {"run": int(train_job_id)},
            )
    return len(attempts)


def finalize_legacy_loss_adjudicated_root(
    conn, *, train_job_id: int
) -> dict[str, Any]:
    """Root surviving bytes and permanent incident evidence without intactness."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT * FROM train_jobs WHERE id = %(run)s FOR UPDATE
                """,
                {"run": int(train_job_id)},
            )
            job = cur.fetchone()
            if not job or int(job["telemetry_protocol_version"]) != 1:
                raise ValueError("legacy adjudication requires a protocol-v1 run")
            if str(job["telemetry_integrity_classification"]) not in {
                "degraded",
                "legacy_unknown",
            }:
                raise RuntimeError("legacy adjudication requires a permanent loss classification")
            cur.execute(
                """
                SELECT producer_ordinal, producer_identity, final_sequence, final_sha256
                FROM telemetry_producers
                WHERE train_job_id = %(run)s AND telemetry_generation = 1
                ORDER BY producer_ordinal
                """,
                {"run": int(train_job_id)},
            )
            producers = [dict(row) for row in cur.fetchall()]
            if not producers or any(row["final_sequence"] is None for row in producers):
                raise RuntimeError("legacy forensic producers are not terminal")
            cur.execute(
                """
                SELECT
                  s.producer_ordinal, s.first_sequence, s.last_sequence,
                  s.compressed_sha256,
                  jsonb_agg(r.receipt_json ORDER BY r.copy_role) AS receipts
                FROM telemetry_archive_segments s
                JOIN telemetry_archive_receipts r
                  ON r.train_job_id=s.train_job_id
                 AND r.telemetry_generation=s.telemetry_generation
                 AND r.producer_ordinal=s.producer_ordinal
                 AND r.first_sequence=s.first_sequence
                 AND r.last_sequence=s.last_sequence
                WHERE s.train_job_id=%(run)s AND s.telemetry_generation=1
                  AND s.claim_state='verified'
                GROUP BY s.producer_ordinal, s.first_sequence, s.last_sequence,
                         s.compressed_sha256
                ORDER BY s.producer_ordinal, s.first_sequence
                """,
                {"run": int(train_job_id)},
            )
            segments = [dict(row) for row in cur.fetchall()]
            coverage: dict[int, int] = {}
            for segment in segments:
                roles = {row["copy_role"] for row in segment["receipts"]}
                if roles != {"primary", "backup"}:
                    raise RuntimeError("legacy adjudication requires dual archive receipts")
                ordinal = int(segment["producer_ordinal"])
                if int(segment["first_sequence"]) != coverage.get(ordinal, 0) + 1:
                    raise RuntimeError("legacy forensic archive coverage contains a gap")
                coverage[ordinal] = int(segment["last_sequence"])
            for producer in producers:
                if coverage.get(int(producer["producer_ordinal"])) != int(
                    producer["final_sequence"]
                ):
                    raise RuntimeError("legacy forensic archive coverage is incomplete")
            cur.execute(
                """
                SELECT incident_key, severity, category, state, details_json
                FROM telemetry_incidents
                WHERE train_job_id=%(run)s AND telemetry_generation=1
                ORDER BY incident_key
                """,
                {"run": int(train_job_id)},
            )
            incidents = [dict(row) for row in cur.fetchall()]
            root = {
                "version": "telemetry-legacy-loss-root-v1",
                "train_job_id": int(train_job_id),
                "telemetry_generation": 1,
                "classification": str(job["telemetry_integrity_classification"]),
                "claim": "all-surviving-bytes-dual-archived-no-reconstruction",
                "producer_claims": producers,
                "segments": segments,
                "incidents": incidents,
            }
            root_sha256 = sha256_bytes(canonical_json_bytes(root))
            coverage_sha256 = sha256_json(producers)
            cur.execute(
                """
                INSERT INTO telemetry_archive_roots (
                  train_job_id, telemetry_generation, root_kind, root_sha256,
                  coverage_sha256, policy_name, root_json
                ) VALUES (
                  %(run)s, 1, 'legacy_loss_adjudicated', %(root)s,
                  %(coverage)s, 'queued_dual_r2_v1', %(document)s
                )
                """,
                {
                    "run": int(train_job_id),
                    "root": root_sha256,
                    "coverage": coverage_sha256,
                    "document": json_arg(root),
                },
            )
            cur.execute(
                """
                UPDATE metric_batches b
                SET archived_at=COALESCE(archived_at, now()),
                    archive_root_sha256=%(root)s
                FROM metric_streams s, worker_attempts w
                WHERE b.stream_id=s.stream_id AND s.attempt_id=w.attempt_id
                  AND w.train_job_id=%(run)s
                """,
                {"run": int(train_job_id), "root": root_sha256},
            )
            cur.execute(
                """
                UPDATE telemetry_recovery_manifests
                SET state='complete', updated_at=now(), last_error=NULL
                WHERE train_job_id=%(run)s AND telemetry_generation=1
                """,
                {"run": int(train_job_id)},
            )
            finalized = {**root, "root_sha256": root_sha256}
    from rlab.telemetry_reducer import reduce_run_integrity

    reduce_run_integrity(conn, train_job_id=int(train_job_id))
    return finalized


def new_archiver_owner() -> str:
    return f"telemetry-archiver-{uuid.uuid4().hex}"
