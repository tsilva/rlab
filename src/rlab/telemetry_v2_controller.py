from __future__ import annotations

import argparse
import os
from typing import Any

from rlab.job_queue import connect, database_url
from rlab.modal_eval_storage import ObjectStore
from rlab.telemetry_archive import (
    ArchiveCopy,
    claim_next_segment,
    finalize_exact_archive_root,
    new_archiver_owner,
    write_claimed_segment,
)
from rlab.telemetry_wandb_projection import (
    MAX_CLAIMED_ROWS_PER_CYCLE,
    claim_projection_rows,
    ensure_projection_close_row,
    mark_projection_terminal_complete,
    materialize_projection_rows,
    new_projection_generation,
    persist_prefix_verification,
    projection_verification_window,
    publish_projection_rows,
    set_publication_component,
)
from rlab.wandb_publisher import WandbProjector
from rlab.wandb_utils import load_wandb_env, resolve_wandb_namespace


def ambiguous_recovery_action(
    *,
    exact: bool,
    verified_through_ordinal: int,
    last_ambiguous_ordinal: int,
    remote_last_step: int,
    step_offset: int,
) -> str:
    if not exact:
        return "quarantine"
    if verified_through_ordinal >= last_ambiguous_ordinal:
        return "adopt"
    next_step = step_offset + verified_through_ordinal + 1
    return "republish" if remote_last_step < next_step else "wait"


def archive_once(conn, *, limit: int) -> int:
    primary = str(os.environ.get("TELEMETRY_ARCHIVE_PRIMARY_URI") or "").strip()
    backup = str(os.environ.get("TELEMETRY_ARCHIVE_BACKUP_URI") or "").strip()
    if not primary or not backup:
        raise RuntimeError(
            "TELEMETRY_ARCHIVE_PRIMARY_URI and TELEMETRY_ARCHIVE_BACKUP_URI are required"
        )
    copies = [
        ArchiveCopy("primary", ObjectStore(primary)),
        ArchiveCopy("backup", ObjectStore(backup)),
    ]
    owner = new_archiver_owner()
    completed = 0
    for _ in range(max(1, int(limit))):
        claim = claim_next_segment(conn, owner=owner)
        if claim is None:
            break
        write_claimed_segment(
            conn,
            claim,
            policy_name="queued_dual_r2_v1",
            copies=copies,
        )
        completed += 1
    return completed


def finalize_roots_once(conn, *, limit: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.id
            FROM train_jobs j
            WHERE j.telemetry_protocol_version=2
              AND j.status IN (
                'finalizing','succeeded','failed','canceled','finalization_failed'
              )
              AND NOT EXISTS (
                SELECT 1 FROM telemetry_archive_roots root
                WHERE root.train_job_id=j.id
                  AND root.telemetry_generation=j.telemetry_generation
              )
            ORDER BY j.finished_at, j.id
            LIMIT %(limit)s
            """,
            {"limit": max(1, int(limit))},
        )
        run_ids = [int(row["id"]) for row in cur.fetchall()]
    completed = 0
    for run_id in run_ids:
        try:
            finalize_exact_archive_root(conn, train_job_id=run_id)
        except RuntimeError:
            continue
        completed += 1
    return completed


def _projection_generation(conn, run: dict[str, Any]) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT *,
                   (
                     last_verification_at IS NULL
                     OR last_verification_at <= now() - interval '5 seconds'
                   ) AS verification_due
            FROM wandb_projection_generations
            WHERE train_job_id=%(run)s
              AND state NOT IN ('sealed','quarantined','failed','disabled')
            ORDER BY projection_generation DESC LIMIT 1
            """,
            {"run": int(run["id"])},
        )
        generation = cur.fetchone()
        if generation:
            return dict(generation)
        cur.execute(
            """
            SELECT service_wandb_credential_generation
            FROM telemetry_rollout_controls WHERE singleton=TRUE
            """
        )
        credential = cur.fetchone()
        if not credential or credential["service_wandb_credential_generation"] is None:
            raise RuntimeError("service W&B credential generation is not installed")
        cur.execute(
            """
            SELECT COALESCE(max(projection_generation),0)+1 AS generation,
                   count(*) AS generation_count
            FROM wandb_projection_generations WHERE train_job_id=%(run)s
            """,
            {"run": int(run["id"])},
        )
        generation_state = cur.fetchone()
        next_generation = int(generation_state["generation"])
    base = str((run.get("train_config") or {}).get("wandb_run_id") or f"rlab-{run['id']}")
    wandb_run_id = (
        base
        if int(generation_state["generation_count"]) == 0
        else f"{base}-projection-g{next_generation}"
    )
    return new_projection_generation(
        conn,
        train_job_id=int(run["id"]),
        service_credential_generation=int(credential["service_wandb_credential_generation"]),
        wandb_run_id=wandb_run_id,
    )


def project_once(conn, *, limit: int) -> int:
    """Materialize canonical events only; the advisory-locked actor owns W&B."""

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT j.*
            FROM train_jobs j
            WHERE j.telemetry_protocol_version=2
              AND COALESCE((j.train_config->>'wandb')::boolean,FALSE)
              AND j.status IN (
                'pending','launching','starting','running',
                'finalizing','finalization_failed'
              )
            ORDER BY j.created_at
            LIMIT %(limit)s
            """,
            {"limit": max(1, int(limit))},
        )
        runs = [dict(row) for row in cur.fetchall()]
    selected = 0
    for run in runs:
        generation = _projection_generation(conn, run)
        selected += materialize_projection_rows(
            conn,
            train_job_id=int(run["id"]),
            projection_generation=int(generation["projection_generation"]),
            max_events=max(1, min(64, int(limit))),
        )
    return selected


def _recover_ambiguous_suffix(
    conn,
    *,
    train_job_id: int,
    generation: dict[str, Any],
    remote,
) -> None:
    """Adopt or safely release only the stabilized unresolved suffix."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT min(output_ordinal) AS first_ordinal,
                       max(output_ordinal) AS last_ordinal
                FROM wandb_projection_rows
                WHERE train_job_id=%(run)s
                  AND projection_generation=%(generation)s
                  AND state='ambiguous'
                  AND ambiguous_at <= now() - interval '30 seconds'
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(generation["projection_generation"]),
                },
            )
            ambiguous = cur.fetchone()
            if not ambiguous or ambiguous["first_ordinal"] is None:
                return
            last = int(ambiguous["last_ordinal"])
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET submitted_through_ordinal=GREATEST(
                  submitted_through_ordinal, %(last)s
                )
                WHERE train_job_id=%(run)s
                  AND projection_generation=%(generation)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(generation["projection_generation"]),
                    "last": last,
                },
            )
    start = int(generation["verified_through_ordinal"]) + 1
    step_offset = int(generation["step_offset"])
    observed = list(
        remote.scan_history(
            keys=[
                "_step",
                "_rlab_event_id",
                "_rlab_payload_sha256",
                "_rlab_output_ordinal",
            ],
            min_step=step_offset + start,
            max_step=step_offset + last + 1,
            page_size=256,
        )
    )
    verification = persist_prefix_verification(
        conn,
        train_job_id=int(train_job_id),
        projection_generation=int(generation["projection_generation"]),
        observed_rows=observed,
    )
    action = ambiguous_recovery_action(
        exact=verification.exact,
        verified_through_ordinal=verification.verified_through_ordinal,
        last_ambiguous_ordinal=last,
        remote_last_step=int(getattr(remote, "lastHistoryStep", -1) or -1),
        step_offset=step_offset,
    )
    if action != "republish":
        return
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE wandb_projection_rows
                SET state='pending', ambiguous_at=NULL, last_error=NULL
                WHERE train_job_id=%(run)s
                  AND projection_generation=%(generation)s
                  AND state='ambiguous'
                  AND output_ordinal > %(through)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(generation["projection_generation"]),
                    "through": int(verification.verified_through_ordinal),
                },
            )
            cur.execute(
                """
                UPDATE wandb_projection_generations
                SET submitted_through_ordinal=%(through)s
                WHERE train_job_id=%(run)s
                  AND projection_generation=%(generation)s
                """,
                {
                    "run": int(train_job_id),
                    "generation": int(generation["projection_generation"]),
                    "through": int(verification.verified_through_ordinal),
                },
            )


def publish_run_once(
    conn,
    *,
    train_job_id: int,
    owner: str,
) -> int:
    """Publish one bounded v2 suffix from inside the run's lifetime actor lock."""

    load_wandb_env()
    import wandb

    with conn.cursor() as cur:
        cur.execute("SELECT * FROM train_jobs WHERE id=%(run)s", {"run": train_job_id})
        row = cur.fetchone()
        if not row or int(row["telemetry_protocol_version"] or 1) != 2:
            return 0
        run = dict(row)
    generation = _projection_generation(conn, run)
    materialize_projection_rows(
        conn,
        train_job_id=train_job_id,
        projection_generation=int(generation["projection_generation"]),
        max_events=64,
    )
    close_ordinal = ensure_projection_close_row(
        conn,
        train_job_id=train_job_id,
        projection_generation=int(generation["projection_generation"]),
    )
    config = {
        **dict(run["train_config"]),
        "wandb_run_id": str(generation["wandb_run_id"]),
        "run_name": str(run.get("run_name") or generation["wandb_run_id"]),
        "wandb_group": str(run.get("batch_id") or ""),
        "telemetry_projection_only": True,
        "train_job_id": train_job_id,
    }
    entity, project = resolve_wandb_namespace(
        config.get("wandb_entity"),
        config.get("wandb_project"),
        config.get("game"),
        env_provider=config.get("env_provider"),
    )
    remote_path = f"{entity}/{project}/{generation['wandb_run_id']}"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
              SELECT 1 FROM wandb_projection_rows
              WHERE train_job_id=%(run)s
                AND projection_generation=%(generation)s
                AND state='ambiguous'
                AND ambiguous_at <= now() - interval '30 seconds'
            ) AS ready
            """,
            {
                "run": train_job_id,
                "generation": int(generation["projection_generation"]),
            },
        )
        ambiguous_ready = bool(cur.fetchone()["ready"])
    if ambiguous_ready:
        remote_for_recovery = wandb.Api(timeout=30).run(remote_path)
        _recover_ambiguous_suffix(
            conn,
            train_job_id=train_job_id,
            generation=generation,
            remote=remote_for_recovery,
        )
    rows = claim_projection_rows(
        conn,
        owner=owner,
        train_job_id=train_job_id,
        limit=MAX_CLAIMED_ROWS_PER_CYCLE,
    )
    submitted = 0
    if rows:
        submitting_close = any(row.output_ordinal == close_ordinal for row in rows)
        projector = WandbProjector.resume(
            config,
            allow_create=True,
            update_finish_state=submitting_close,
        )
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE train_jobs
                        SET wandb_run_id=%(wandb_run_id)s,
                            wandb_url=COALESCE(NULLIF(%(url)s,''), wandb_url),
                            wandb_ready_at=COALESCE(wandb_ready_at, now())
                        WHERE id=%(run)s
                        """,
                        {
                            "run": train_job_id,
                            "wandb_run_id": str(generation["wandb_run_id"]),
                            "url": str(getattr(projector.run, "url", "") or ""),
                        },
                    )
            submitted = publish_projection_rows(conn, projector.run, rows, owner=owner)
        except Exception as exc:
            set_publication_component(
                conn,
                train_job_id=train_job_id,
                component="history",
                state="retrying",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        finally:
            projector.close()

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM wandb_projection_generations
            WHERE train_job_id=%(run)s AND projection_generation=%(generation)s
            """,
            {
                "run": train_job_id,
                "generation": int(generation["projection_generation"]),
            },
        )
        current = dict(cur.fetchone())
    window = projection_verification_window(current)
    verification_due = bool(current["verification_due"])
    if window is not None and verification_due:
        start, end = window
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE wandb_projection_generations
                    SET last_verification_at=now()
                    WHERE train_job_id=%(run)s
                      AND projection_generation=%(generation)s
                    """,
                    {
                        "run": train_job_id,
                        "generation": int(generation["projection_generation"]),
                    },
                )
        remote = wandb.Api(timeout=30).run(remote_path)
        observed = list(
            remote.scan_history(
                keys=[
                    "_step",
                    "_rlab_event_id",
                    "_rlab_payload_sha256",
                    "_rlab_output_ordinal",
                ],
                min_step=int(current["step_offset"]) + start,
                max_step=int(current["step_offset"]) + end + 1,
                page_size=256,
            )
        )
        verification = persist_prefix_verification(
            conn,
            train_job_id=train_job_id,
            projection_generation=int(generation["projection_generation"]),
            observed_rows=observed,
        )
        if verification.exact:
            set_publication_component(
                conn,
                train_job_id=train_job_id,
                component="history",
                state="live",
            )
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT close_ordinal, verified_through_ordinal
            FROM wandb_projection_generations
            WHERE train_job_id=%(run)s
              AND projection_generation=%(generation)s
            """,
            {
                "run": train_job_id,
                "generation": int(generation["projection_generation"]),
            },
        )
        terminal = dict(cur.fetchone())
    if (
        verification_due
        and terminal["close_ordinal"] is not None
        and int(terminal["verified_through_ordinal"]) == int(terminal["close_ordinal"])
    ):
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE wandb_projection_generations
                    SET last_verification_at=now()
                    WHERE train_job_id=%(run)s
                      AND projection_generation=%(generation)s
                    """,
                    {
                        "run": train_job_id,
                        "generation": int(generation["projection_generation"]),
                    },
                )
        remote = wandb.Api(timeout=30).run(remote_path)
        mark_projection_terminal_complete(
            conn,
            train_job_id=train_job_id,
            projection_generation=int(generation["projection_generation"]),
            remote_state=str(getattr(remote, "state", "") or ""),
            remote_last_step=int(getattr(remote, "lastHistoryStep", -1) or -1),
        )
    return submitted


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run bounded canonical telemetry v2 archive and W&B projection work."
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--skip-wandb", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    conn = connect(database_url(use_direct=True))
    try:
        archived = archive_once(conn, limit=args.limit)
        finalized = finalize_roots_once(conn, limit=args.limit)
        projected = 0 if args.skip_wandb else project_once(conn, limit=args.limit)
        print(
            f"archived_segments={archived} finalized_roots={finalized} projected_rows={projected}"
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
