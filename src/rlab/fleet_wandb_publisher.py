from __future__ import annotations

import argparse
import os
import sys
import uuid
from types import SimpleNamespace
from typing import Any

from rlab.env import resolve_env_config
from rlab.env_config import env_config_from_args
from rlab.telemetry_mailbox import (
    claim_run_metric_batches,
    commit_published_batches,
    decode_metric_batch,
    mark_submitted_batches,
    release_metric_batch_claims,
    release_wandb_run_lock,
)
from rlab.wandb_publisher import (
    WandbProjector,
    _publish_frame,
    project_payload_to_run,
)
from rlab.wandb_utils import load_wandb_env, resolve_wandb_namespace


SUMMARY_CURSOR_KEY = "_rlab_telemetry_cursors"
SUPPORTED_FRAME_KINDS = {
    "history",
    "histogram",
    "checkpoint_eval",
    "checkpoint_preview",
    "projection",
}


class InvalidTelemetryBatchError(RuntimeError):
    pass


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
) -> tuple[dict[str, int], float | None]:
    load_wandb_env()
    import wandb

    entity, project = resolve_wandb_namespace(
        train_config.get("wandb_entity"),
        train_config.get("wandb_project"),
        str(train_config.get("game") or ""),
        env_provider=train_config.get("env_provider"),
    )
    run_id = str(train_config["wandb_run_id"])
    try:
        remote = wandb.Api().run(f"{entity}/{project}/{run_id}")
    except Exception:
        return {}, None
    summary = dict(remote.summary)
    raw = summary.get(SUMMARY_CURSOR_KEY) or {}
    return _cursor_mapping(raw), _summary_step_max(summary.get("global_step"))


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


def _record_committed_effects(
    conn,
    *,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
    decoded: dict[int, list[dict[str, Any]]],
    wandb_url: str | None,
) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs t
                SET live_publication_status = CASE
                      WHEN NOT EXISTS (
                        SELECT 1 FROM metric_batches b
                        JOIN metric_streams s ON s.stream_id = b.stream_id
                        JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                        WHERE a.train_job_id = t.id
                      )
                      AND NOT EXISTS (
                        SELECT 1 FROM metric_streams s
                        JOIN worker_attempts a ON a.attempt_id = s.attempt_id
                        WHERE a.train_job_id = t.id
                          AND (s.final_sequence IS NULL
                               OR s.published_sequence < s.final_sequence)
                      ) THEN 'complete'
                      ELSE 'pending'
                    END,
                    live_publication_error = NULL,
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
                    eval_job_id = frame.get("payload", {}).get("eval_job_id")
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


def publish_claimed_run(
    conn,
    run: dict[str, Any],
    batches: list[dict[str, Any]],
) -> int:
    train_config = _train_config(run)
    decoded: dict[int, list[dict[str, Any]]] = {}
    for batch in batches:
        try:
            frames = decode_metric_batch(bytes(batch["payload"]))
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
    remote, remote_step_max = _remote_publication_state(train_config)
    confirmed, awaiting_confirmation, unpublished = _partition_batches(batches, remote)
    if confirmed:
        commit_published_batches(conn, confirmed)
        _record_committed_effects(
            conn,
            run=run,
            batches=confirmed,
            decoded=decoded,
            wandb_url=None,
        )
    if awaiting_confirmation:
        mark_submitted_batches(conn, awaiting_confirmation)
    wandb_url: str | None = None
    if unpublished:
        session_step_max = remote_step_max
        expected: dict[str, int] = {}
        for row in unpublished:
            stream_id = str(row["stream_id"])
            expected[stream_id] = max(expected.get(stream_id, 0), int(row["batch_sequence"]))
        projector = WandbProjector.resume(
            train_config,
            allow_create=True,
            update_finish_state=str(run.get("status") or "")
            in {"succeeded", "failed", "finalization_failed", "canceled"},
        )
        try:
            wandb_url = str(getattr(projector.run, "url", "") or "") or None
            args = SimpleNamespace(**train_config)
            config = resolve_env_config(env_config_from_args(args, include_states=True))
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
                        project_payload_to_run(
                            projector.run,
                            payload,
                            allow_artifact_references=False,
                        )
                    else:
                        _publish_frame(
                            projector.run,
                            {
                                "kind": kind,
                                "payload_json": __import__("json").dumps(payload),
                            },
                            args=args,
                            config=config,
                        )
            merged = dict(remote)
            for stream_id, sequence in expected.items():
                merged[stream_id] = max(int(merged.get(stream_id, 0)), int(sequence))
            projector.run.summary[SUMMARY_CURSOR_KEY] = merged
            if session_step_max is not None:
                summary_step: int | float = session_step_max
                if session_step_max.is_integer():
                    summary_step = int(session_step_max)
                projector.run.summary["global_step"] = {"max": summary_step}
        finally:
            projector.close()
        mark_submitted_batches(conn, unpublished)
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


def _drain_claim(conn, run: dict[str, Any], batches: list[dict[str, Any]]) -> int:
    try:
        return publish_claimed_run(conn, run, batches)
    except Exception as exc:
        release_metric_batch_claims(conn, batches, error=repr(exc))
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET live_publication_status = %(publication_status)s,
                        live_publication_error = %(error)s,
                        live_publication_attempts = live_publication_attempts + 1,
                        live_publication_next_retry_at = now() + interval '30 seconds'
                    WHERE id = %(train_job_id)s
                    """,
                    {
                        "train_job_id": int(run["id"]),
                        "error": repr(exc)[:4000],
                        "publication_status": (
                            "failed" if isinstance(exc, InvalidTelemetryBatchError) else "pending"
                        ),
                    },
                )
                if isinstance(exc, InvalidTelemetryBatchError):
                    cur.execute(
                        """
                        UPDATE train_jobs
                        SET status = 'finalization_failed', finished_at = now(),
                            error = %(error)s
                        WHERE id = %(train_job_id)s AND status = 'finalizing'
                        """,
                        {
                            "train_job_id": int(run["id"]),
                            "error": repr(exc)[:4000],
                        },
                    )
        raise
    finally:
        release_wandb_run_lock(conn, int(run["id"]))


def drain_once(
    conn,
    *,
    owner: str | None = None,
    limit: int = 20,
    exclude_train_job_ids: tuple[int, ...] = (),
) -> int:
    owner = owner or f"fleet-publisher-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    claim = claim_run_metric_batches(
        conn,
        owner=owner,
        limit=limit,
        exclude_train_job_ids=exclude_train_job_ids,
    )
    if claim is None:
        return 0
    run, batches = claim
    return _drain_claim(conn, run, batches)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drain Fleet telemetry mailboxes to W&B.")
    parser.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from rlab.job_queue import connect, database_url

    # Publishing holds a session-scoped per-run lock for the whole W&B
    # session, so it must not use a PgBouncer-backed connection.
    conn = connect(database_url(use_direct=True))
    try:
        print(f"published_batches={drain_once(conn, limit=max(1, args.limit))}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
