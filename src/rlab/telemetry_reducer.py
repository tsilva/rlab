from __future__ import annotations

from typing import Any

from rlab.job_queue import json_arg
from rlab.telemetry_archive import POLICY_RECEIPT_ROLES
from rlab.telemetry_integrity import IntegrityInputs, reduce_integrity


def reduce_run_integrity(conn, *, train_job_id: int) -> dict[str, Any]:
    """Materialize the single fail-closed integrity record used by all consumers."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM train_jobs WHERE id = %(run)s FOR UPDATE",
                {"run": int(train_job_id)},
            )
            job = cur.fetchone()
            if not job:
                raise ValueError(f"unknown train job: {train_job_id}")
            generation = int(job["telemetry_generation"])
            policy = str(
                job["telemetry_durability_policy"] or "local_singlecopy_optout_v1"
            )
            cur.execute(
                """
                SELECT obligation_key, expected_disposition, realized_disposition
                FROM telemetry_expected_obligations
                WHERE train_job_id = %(run)s
                  AND telemetry_generation = %(generation)s
                """,
                {"run": int(train_job_id), "generation": generation},
            )
            obligations = [dict(row) for row in cur.fetchall()]
            expected = {
                str(row["obligation_key"]): str(row["expected_disposition"])
                for row in obligations
            }
            realized = {
                str(row["obligation_key"]): str(row["realized_disposition"])
                for row in obligations
                if row["realized_disposition"] is not None
            }
            cur.execute(
                """
                SELECT producer_ordinal, final_sequence, final_sha256, state
                FROM telemetry_producers
                WHERE train_job_id = %(run)s
                  AND telemetry_generation = %(generation)s
                ORDER BY producer_ordinal
                """,
                {"run": int(train_job_id), "generation": generation},
            )
            producer_rows = [dict(row) for row in cur.fetchall()]
            claims = {
                int(row["producer_ordinal"]): (
                    int(row["final_sequence"]),
                    str(row["final_sha256"]),
                )
                for row in producer_rows
                if row["final_sequence"] is not None and row["final_sha256"] is not None
            }
            cur.execute(
                """
                SELECT producer_ordinal, max(last_sequence) AS last_sequence
                FROM telemetry_archive_segments
                WHERE train_job_id = %(run)s
                  AND telemetry_generation = %(generation)s
                  AND claim_state = 'verified'
                GROUP BY producer_ordinal
                """,
                {"run": int(train_job_id), "generation": generation},
            )
            coverage = {
                int(row["producer_ordinal"]): int(row["last_sequence"])
                for row in cur.fetchall()
            }
            cur.execute(
                """
                SELECT count(*) AS segments,
                       count(r.*) AS receipts
                FROM telemetry_archive_segments s
                LEFT JOIN telemetry_archive_receipts r
                  ON r.train_job_id = s.train_job_id
                 AND r.telemetry_generation = s.telemetry_generation
                 AND r.producer_ordinal = s.producer_ordinal
                 AND r.first_sequence = s.first_sequence
                 AND r.last_sequence = s.last_sequence
                WHERE s.train_job_id = %(run)s
                  AND s.telemetry_generation = %(generation)s
                  AND s.claim_state = 'verified'
                """,
                {"run": int(train_job_id), "generation": generation},
            )
            receipt_counts = cur.fetchone()
            cur.execute(
                """
                SELECT category || ':' || incident_key AS incident
                FROM telemetry_incidents
                WHERE train_job_id = %(run)s
                  AND telemetry_generation = %(generation)s
                  AND state = 'open'
                  AND category <> 'wandb_projection_conflict'
                ORDER BY incident_key
                """,
                {"run": int(train_job_id), "generation": generation},
            )
            incidents = [str(row["incident"]) for row in cur.fetchall()]
            cur.execute(
                """
                SELECT EXISTS (
                  SELECT 1 FROM telemetry_recovery_manifests
                  WHERE train_job_id = %(run)s
                    AND telemetry_generation = %(generation)s
                    AND state <> 'complete'
                ) AS pending
                """,
                {"run": int(train_job_id), "generation": generation},
            )
            recovery_pending = bool(cur.fetchone()["pending"])
            cur.execute(
                """
                SELECT * FROM telemetry_archive_roots
                WHERE train_job_id = %(run)s
                  AND telemetry_generation = %(generation)s
                """,
                {"run": int(train_job_id), "generation": generation},
            )
            root = cur.fetchone()
            classification = str(job["telemetry_integrity_classification"])
            if (
                int(job["telemetry_protocol_version"]) == 2
                and root
                and str(root["root_kind"]) == "exact"
            ):
                classification = "intact_with_proof"
            required_per_segment = len(POLICY_RECEIPT_ROLES.get(policy, ()))
            result = reduce_integrity(
                IntegrityInputs(
                    classification=classification,
                    expected_obligations=expected,
                    realized_obligations=realized,
                    producer_final_claims=claims,
                    archived_coverage=coverage,
                    no_more_producers=bool(job["telemetry_no_more_producers"]),
                    durability_policy=policy,
                    required_archive_receipts=(
                        int(receipt_counts["segments"]) * required_per_segment
                    ),
                    observed_archive_receipts=int(receipt_counts["receipts"]),
                    recovery_pending=recovery_pending,
                    incidents=incidents,
                )
            )
            cur.execute(
                """
                SELECT state, projection_generation
                FROM wandb_projection_generations
                WHERE train_job_id = %(run)s
                  AND projection_generation =
                      %(active_generation)s
                """,
                {
                    "run": int(train_job_id),
                    "active_generation": job["active_wandb_projection_generation"],
                },
            )
            projection = cur.fetchone()
            wandb_enabled = bool((job["train_config"] or {}).get("wandb"))
            wandb_terminal = (not wandb_enabled) or (
                projection is not None and str(projection["state"]) == "active"
            )
            cur.execute(
                """
                SELECT scope_sha256 FROM telemetry_run_facts
                WHERE train_job_id = %(run)s
                """,
                {"run": int(train_job_id)},
            )
            facts = cur.fetchone()
            cleanup_eligible = bool(result.cleanup_eligible and facts and wandb_terminal)
            reasons = list(result.reasons)
            if result.exact and not facts:
                reasons.append("run_final_facts_pending")
            if result.exact and not wandb_terminal:
                reasons.append("wandb_terminal_disposition_pending")
            state = {
                "version": "telemetry-integrity-v1",
                "fence": {
                    "no_more_producers": bool(job["telemetry_no_more_producers"]),
                    "frozen_at": (
                        None
                        if job["telemetry_frozen_at"] is None
                        else str(job["telemetry_frozen_at"])
                    ),
                },
                "vulnerable_window_classification": classification,
                "expected_set_closed": set(expected) == set(realized),
                "archive": {
                    "coverage_sha256": result.coverage_sha256,
                    "root_sha256": None if not root else str(root["root_sha256"]),
                    "durability_policy": policy,
                    "receipts": int(receipt_counts["receipts"]),
                    "required_receipts": (
                        int(receipt_counts["segments"]) * required_per_segment
                    ),
                },
                "recovery_backlog": recovery_pending,
                "executor_health": "healthy" if not recovery_pending else "backlogged",
                "evidence_freshness": "final" if facts else "pending",
                "cohort_comparability": "exact" if facts else "unavailable",
                "wandb_generation_health": (
                    "disabled"
                    if not wandb_enabled
                    else (str(projection["state"]) if projection else "pending")
                ),
                "cleanup_eligible": cleanup_eligible,
                "reasons": reasons,
            }
            disposition = (
                "exact"
                if result.exact
                else (
                    "durability_opted_out"
                    if policy == "local_singlecopy_optout_v1"
                    else (
                        "legacy_loss_adjudicated"
                        if root and str(root["root_kind"]) == "legacy_loss_adjudicated"
                        else "pending"
                    )
                )
            )
            cur.execute(
                """
                INSERT INTO telemetry_integrity (
                  train_job_id, telemetry_generation, classification,
                  disposition, exact, cleanup_eligible, expected_set_sha256,
                  coverage_sha256, archive_root_sha256, facts_sha256,
                  active_wandb_generation, reasons, state_json, updated_at
                ) VALUES (
                  %(run)s, %(generation)s, %(classification)s, %(disposition)s,
                  %(exact)s, %(cleanup)s, %(expected_sha)s, %(coverage_sha)s,
                  %(root_sha)s, %(facts_sha)s, %(wandb_generation)s,
                  %(reasons)s, %(state)s, now()
                )
                ON CONFLICT (train_job_id, telemetry_generation) DO UPDATE
                SET classification = EXCLUDED.classification,
                    disposition = EXCLUDED.disposition,
                    exact = EXCLUDED.exact,
                    cleanup_eligible = EXCLUDED.cleanup_eligible,
                    expected_set_sha256 = EXCLUDED.expected_set_sha256,
                    coverage_sha256 = EXCLUDED.coverage_sha256,
                    archive_root_sha256 = EXCLUDED.archive_root_sha256,
                    facts_sha256 = EXCLUDED.facts_sha256,
                    active_wandb_generation = EXCLUDED.active_wandb_generation,
                    reasons = EXCLUDED.reasons,
                    state_json = EXCLUDED.state_json,
                    updated_at = now()
                """,
                {
                    "run": int(train_job_id),
                    "generation": generation,
                    "classification": classification,
                    "disposition": disposition,
                    "exact": bool(result.exact),
                    "cleanup": cleanup_eligible,
                    "expected_sha": result.expected_set_sha256,
                    "coverage_sha": result.coverage_sha256,
                    "root_sha": None if not root else str(root["root_sha256"]),
                    "facts_sha": None if not facts else str(facts["scope_sha256"]),
                    "wandb_generation": (
                        None if not projection else int(projection["projection_generation"])
                    ),
                    "reasons": reasons,
                    "state": json_arg(state),
                },
            )
            cur.execute(
                """
                UPDATE train_jobs
                SET telemetry_integrity_classification = %(classification)s
                WHERE id = %(run)s
                """,
                {"run": int(train_job_id), "classification": classification},
            )
    return {
        "train_job_id": int(train_job_id),
        "telemetry_generation": generation,
        "classification": classification,
        "disposition": disposition,
        "exact": bool(result.exact),
        "cleanup_eligible": cleanup_eligible,
        "reasons": reasons,
        "state": state,
    }
