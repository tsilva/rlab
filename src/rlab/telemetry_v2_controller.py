from __future__ import annotations

import argparse
import os
import uuid
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
    activate_verified_generation,
    claim_projection_row,
    materialize_projection_rows,
    new_projection_generation,
    persist_prefix_verification,
    projection_fingerprint,
    publish_projection_row,
)
from rlab.wandb_utils import load_wandb_env, resolve_wandb_namespace


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
              AND j.status IN ('succeeded','failed','canceled','finalization_failed')
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
            SELECT * FROM wandb_projection_generations
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
            SELECT COALESCE(max(projection_generation),0)+1 AS generation
            FROM wandb_projection_generations WHERE train_job_id=%(run)s
            """,
            {"run": int(run["id"])},
        )
        next_generation = int(cur.fetchone()["generation"])
    base = str((run.get("train_config") or {}).get("wandb_run_id") or f"rlab-{run['id']}")
    return new_projection_generation(
        conn,
        train_job_id=int(run["id"]),
        service_credential_generation=int(
            credential["service_wandb_credential_generation"]
        ),
        wandb_run_id=f"{base}-projection-g{next_generation}",
    )


def project_once(conn, *, limit: int) -> int:
    load_wandb_env()
    import wandb

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT service_wandb_credential_generation
            FROM telemetry_rollout_controls WHERE singleton=TRUE
            """
        )
        control = cur.fetchone()
        configured_generation = int(
            os.environ.get("RLAB_WANDB_SERVICE_CREDENTIAL_GENERATION") or 0
        )
        configured_identity = str(
            os.environ.get("RLAB_WANDB_SERVICE_IDENTITY") or ""
        ).strip()
        if (
            not control
            or control["service_wandb_credential_generation"] is None
            or configured_generation
            != int(control["service_wandb_credential_generation"])
            or not configured_identity
        ):
            raise RuntimeError(
                "W&B projection requires the fenced service credential identity/generation"
            )
        cur.execute(
            """
            SELECT *
            FROM train_jobs
            WHERE telemetry_protocol_version=2
              AND COALESCE((train_config->>'wandb')::boolean,FALSE)
            ORDER BY created_at
            LIMIT %(limit)s
            """,
            {"limit": max(1, int(limit))},
        )
        runs = [dict(row) for row in cur.fetchall()]
    published = 0
    owner = f"wandb-v2-{uuid.uuid4().hex}"
    for run in runs:
        generation = _projection_generation(conn, run)
        materialize_projection_rows(
            conn,
            train_job_id=int(run["id"]),
            projection_generation=int(generation["projection_generation"]),
        )
        config = dict(run["train_config"])
        entity, project = resolve_wandb_namespace(
            config.get("wandb_entity"),
            config.get("wandb_project"),
            config.get("game"),
            env_provider=config.get("env_provider"),
        )
        sdk_run = wandb.init(
            project=project,
            entity=entity,
            id=str(generation["wandb_run_id"]),
            name=str(run.get("run_name") or generation["wandb_run_id"]),
            resume="allow",
            reinit="finish_previous",
            config={"telemetry_projection_only": True, "train_job_id": int(run["id"])},
        )
        try:
            while published < max(1, int(limit)):
                row = claim_projection_row(
                    conn, owner=owner, train_job_id=int(run["id"])
                )
                if row is None:
                    break
                publish_projection_row(conn, sdk_run, row, owner=owner)
                published += 1
            remote = wandb.Api(timeout=30).run(
                f"{entity}/{project}/{generation['wandb_run_id']}"
            )
            observed = list(
                remote.scan_history(
                    keys=[
                        "_step",
                        "_rlab_event_id",
                        "_rlab_payload_sha256",
                        "_rlab_output_ordinal",
                    ],
                    page_size=1000,
                )
            )
            verification = persist_prefix_verification(
                conn,
                train_job_id=int(run["id"]),
                projection_generation=int(generation["projection_generation"]),
                observed_rows=observed,
            )
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT * FROM wandb_projection_rows
                    WHERE train_job_id=%(run)s
                      AND projection_generation=%(generation)s
                    ORDER BY output_ordinal
                    """,
                    {
                        "run": int(run["id"]),
                        "generation": int(generation["projection_generation"]),
                    },
                )
                rows = [dict(value) for value in cur.fetchall()]
            if verification.exact and len(observed) == len(rows) and rows:
                activate_verified_generation(
                    conn,
                    train_job_id=int(run["id"]),
                    projection_generation=int(generation["projection_generation"]),
                    remote_fingerprint=projection_fingerprint(rows),
                )
        finally:
            sdk_run.finish(quiet=True)
    return published


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
            f"archived_segments={archived} finalized_roots={finalized} "
            f"projected_rows={projected}"
        )
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
