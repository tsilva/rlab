from __future__ import annotations

import argparse
from datetime import UTC, datetime
from dataclasses import dataclass
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from rlab.cli_args import add_direct_database_arg, add_dry_run_arg
from rlab.dotenv import load_env_file
from rlab.json_utils import json_safe
from rlab.runtime_refs import (
    DEFAULT_IMAGE_ARTIFACT,
    DEFAULT_IMAGE_BRANCH,
    DEFAULT_IMAGE_WORKFLOW,
    normalize_runtime_image_ref,
    runtime_image_ref_from_args,
)
from rlab.recipe_documents import (
    assert_no_secrets,
    compiled_recipe_payload,
    load_recipe_document,
    recipe_goal_slug,
    recipe_metadata,
    recipe_slug,
    recipe_tags,
    validate_launch_event_config,
    validate_launch_seed_config,
)
from rlab.seeds import validate_training_seed
from rlab.recipe_schema import require_explicit_queue_train_config, validate_materialized_train_recipe
from rlab.provider_config import provider_num_envs
from rlab.train_config import validate_and_normalize_train_config


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS train_jobs (
  id BIGSERIAL PRIMARY KEY,
  goal_slug TEXT NOT NULL,
  recipe_slug TEXT,
  recipe_path TEXT,
  recipe_sha256 TEXT,
  repo_git_commit TEXT,
  repo_dirty BOOLEAN NOT NULL DEFAULT FALSE,
  recipe_payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  runtime_image_ref TEXT NOT NULL,
  run_target TEXT,
  train_config JSONB NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 1,
  lease_owner TEXT,
  lease_expires_at TIMESTAMPTZ,
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  drain_requested BOOLEAN NOT NULL DEFAULT FALSE,
  run_name TEXT,
  run_description TEXT,
  seed INTEGER,
  wandb_group TEXT,
  wandb_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  heartbeat_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  error TEXT
);

CREATE TABLE IF NOT EXISTS job_launches (
  id BIGSERIAL PRIMARY KEY,
  launch_id TEXT NOT NULL UNIQUE,
  job_kind TEXT NOT NULL CHECK (job_kind IN ('train')),
  job_id BIGINT NOT NULL,
  backend TEXT NOT NULL,
  machine TEXT NOT NULL,
  runtime_image_ref TEXT NOT NULL,
  container_name TEXT,
  provider_run_id TEXT,
  output_uri TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'launching',
  exit_code INTEGER,
  error TEXT,
  result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  last_observed_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS job_events (
  id BIGSERIAL PRIMARY KEY,
  job_kind TEXT NOT NULL CHECK (job_kind IN ('train')),
  job_id BIGINT NOT NULL,
  event_type TEXT NOT NULL,
  message TEXT,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DROP INDEX IF EXISTS train_jobs_runtime_claim_idx;
DROP INDEX IF EXISTS train_jobs_claim_idx;
DROP INDEX IF EXISTS train_jobs_spec_status_idx;

ALTER TABLE train_jobs DROP COLUMN IF EXISTS priority;

CREATE INDEX IF NOT EXISTS train_jobs_claim_idx
  ON train_jobs (status, id)
  WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS train_jobs_runtime_claim_idx
  ON train_jobs (runtime_image_ref, status, id)
  WHERE status IN ('pending', 'running') AND runtime_image_ref IS NOT NULL;

CREATE INDEX IF NOT EXISTS train_jobs_goal_status_idx
  ON train_jobs (goal_slug, status);

CREATE INDEX IF NOT EXISTS train_jobs_recipe_status_idx
  ON train_jobs (goal_slug, recipe_slug, status);

CREATE INDEX IF NOT EXISTS job_launches_machine_state_idx
  ON job_launches (machine, state, created_at);

CREATE INDEX IF NOT EXISTS job_launches_job_idx
  ON job_launches (job_kind, job_id, created_at DESC);

CREATE INDEX IF NOT EXISTS job_events_job_idx
  ON job_events (job_kind, job_id, created_at DESC);
"""

RESET_TABLES = (
    "job_events",
    "job_launches",
    "train_jobs",
)
TRAIN_JOB_KIND = "train"


@dataclass(frozen=True)
class QueueDemand:
    runtime_image_ref: str
    run_target: str | None
    pending_count: int
    running_count: int
    oldest_job_id: int

    @property
    def total(self) -> int:
        return self.pending_count + self.running_count


QUEUE_DEMAND_SQL = """
SELECT
  runtime_image_ref,
  run_target,
  COUNT(*) FILTER (WHERE status = 'pending') AS pending_count,
  COUNT(*) FILTER (WHERE status = 'running') AS running_count,
  MIN(id) AS oldest_job_id
FROM train_jobs
WHERE runtime_image_ref IS NOT NULL
  AND cancel_requested = FALSE
  AND status IN ('pending', 'running')
GROUP BY runtime_image_ref, run_target
ORDER BY oldest_job_id ASC
"""


def queue_demands(conn) -> list[QueueDemand]:
    with conn.cursor() as cur:
        cur.execute(QUEUE_DEMAND_SQL)
        rows = cur.fetchall()
    return [
        QueueDemand(
            runtime_image_ref=normalize_runtime_image_ref(row["runtime_image_ref"]),
            run_target=row.get("run_target"),
            pending_count=int(row["pending_count"]),
            running_count=int(row["running_count"]),
            oldest_job_id=int(row["oldest_job_id"]),
        )
        for row in rows
    ]


def machine_queue_counts(conn) -> dict[str, dict[str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM train_jobs
            GROUP BY status
            ORDER BY status
            """
        )
        train_counts = {str(row["status"]): int(row["count"]) for row in cur.fetchall()}
    return {"train": train_counts}


def json_arg(value: Any) -> psycopg2.extras.Json:
    return psycopg2.extras.Json(value)


def database_url(use_direct: bool = False) -> str:
    load_env_file()
    if use_direct:
        value = os.environ.get("DIRECT_DATABASE_URL") or os.environ.get("DATABASE_URL")
    else:
        value = (
            os.environ.get("TRAIN_QUEUE_DATABASE_URL")
            or os.environ.get("DATABASE_URL")
            or os.environ.get("DIRECT_DATABASE_URL")
        )
    if not value:
        raise SystemExit(
            "TRAIN_QUEUE_DATABASE_URL, DATABASE_URL, or DIRECT_DATABASE_URL must be set"
        )
    return value


def normalize_run_target(value: str | None) -> str | None:
    text = str(value or "").strip()
    return text or None


def connect(url: str):
    return psycopg2.connect(url, cursor_factory=psycopg2.extras.RealDictCursor)


def apply_schema(conn) -> None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def _table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%(table_name)s) AS table_name", {"table_name": table_name})
        row = cur.fetchone()
    return bool(row and row.get("table_name"))


def export_existing_tables(conn, export_dir: Path) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "tables": [],
    }
    for table_name in RESET_TABLES:
        if not _table_exists(conn, table_name):
            continue
        path = export_dir / f"{table_name}.jsonl"
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name} ORDER BY id")
            rows = [dict(row) for row in cur.fetchall()]
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        manifest["tables"].append({"table": table_name, "rows": len(rows), "path": str(path)})
    (export_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return export_dir


def reset_schema(conn, *, export_dir: Path) -> Path:
    exported = export_existing_tables(conn, export_dir)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                DROP TABLE IF EXISTS
                  job_events,
                  job_launches,
                  train_jobs
                CASCADE
                """
            )
            cur.execute(SCHEMA_SQL)
    return exported


def record_job_event(
    conn,
    *,
    job_id: int,
    event_type: str,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    metadata = dict(metadata or {})
    assert_no_secrets(metadata, label="event metadata")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO job_events (job_kind, job_id, event_type, message, metadata_json)
            VALUES (%(job_kind)s, %(job_id)s, %(event_type)s, %(message)s, %(metadata_json)s)
            """,
            {
                "job_kind": TRAIN_JOB_KIND,
                "job_id": job_id,
                "event_type": event_type,
                "message": message,
                "metadata_json": json_arg(metadata),
            },
        )


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _format_queue_template(
    template: str | None,
    *,
    seed: int | None,
    recipe_id: str,
    utc: str,
    group_id: str = "",
) -> str | None:
    if not template:
        return None
    return str(template).format(
        seed="" if seed is None else seed,
        recipe_id=recipe_id,
        timestamp=utc,
        utc=utc,
        group_id=group_id,
    )


def _run_name_slug(value: str, *, limit: int = 32) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    slug = "".join(chars).strip("-") or "run"
    return slug[:limit].strip("-") or "run"


def _run_name_batch_id(group_id: str) -> str:
    group = _run_name_slug(group_id)
    match = re.match(r"^(b\d+)(?:-|$)", group)
    return match.group(1) if match else group


def _format_default_run_name(
    group_id: str,
    *,
    label: str,
    seed: int | None,
    utc: str,
) -> str:
    batch_id = _run_name_batch_id(group_id)
    description = _run_name_slug(label, limit=24)
    seed_label = f"s{seed}" if seed is not None else "s"
    return f"{batch_id}-{description}-{seed_label}-{utc}"


def _document_seeds(
    document: Mapping[str, Any], override_seeds: Sequence[int] = ()
) -> list[int | None]:
    if override_seeds:
        return [int(seed) for seed in override_seeds]
    seeds = document.get("seeds")
    if isinstance(seeds, Sequence) and not isinstance(seeds, str):
        return [int(seed) for seed in seeds]
    train_config = document.get("train_config")
    if isinstance(train_config, Mapping) and train_config.get("seed") is not None:
        return [int(train_config["seed"])]
    return [None]


def enqueue_train_jobs_from_recipe_document(
    conn,
    *,
    document: Mapping[str, Any],
    runtime_image_ref: str,
    run_target: str | None = None,
    recipe_path: str | None = None,
    recipe_sha256: str | None = None,
    repo_git_commit: str | None = None,
    repo_dirty: bool = False,
    seeds: Sequence[int] = (),
) -> list[dict[str, Any]]:
    validate_materialized_train_recipe(document)
    goal_slug = recipe_goal_slug(document)
    document_slug = recipe_slug(document)
    utc = _utc_stamp()
    rows = []
    for seed in _document_seeds(document, seeds):
        train_config = dict(document["train_config"])
        recipe_overrides = document.get("recipe_overrides")
        if isinstance(recipe_overrides, Sequence) and not isinstance(recipe_overrides, str | bytes):
            train_config["recipe_overrides"] = [str(item) for item in recipe_overrides]
        if seed is not None:
            validate_training_seed(
                seed,
                label="recipe seed",
                seed_span=provider_num_envs(train_config, explicit_n_envs=train_config.get("n_envs")),
            )
        row_run_target = normalize_run_target(run_target or train_config.get("run_target"))
        for row_owned_key in ("seed", "recipe_slug", "recipe_path", "run_target"):
            train_config.pop(row_owned_key, None)
        group_id = str(document["group_id"])
        payload = compiled_recipe_payload(document)
        row = enqueue_train_job(
            conn,
            goal_slug=goal_slug,
            recipe_slug=document_slug,
            recipe_path=recipe_path,
            recipe_sha256=recipe_sha256,
            repo_git_commit=repo_git_commit,
            repo_dirty=repo_dirty,
            recipe_payload=payload,
            runtime_image_ref=runtime_image_ref,
            run_target=row_run_target,
            train_config=train_config,
            max_attempts=int(document.get("max_attempts") or 1),
            run_name=_format_default_run_name(
                str(document.get("batch_id") or group_id),
                label=document_slug,
                seed=seed,
                utc=utc,
            ),
            run_description=_format_queue_template(
                document.get("description"),
                seed=seed,
                recipe_id=document_slug,
                utc=utc,
                group_id=group_id,
            ),
            seed=seed,
            wandb_group=group_id,
            wandb_tags=recipe_tags(document),
        )
        rows.append(row)
    return rows


def enqueue_train_jobs_from_recipe_file(
    conn,
    *,
    path: Path,
    runtime_image_ref: str,
    run_target: str | None = None,
    seeds: Sequence[int] = (),
    recipe_overrides: Sequence[str] = (),
) -> list[dict[str, Any]]:
    document = load_recipe_document(path, recipe_overrides=recipe_overrides)
    metadata = recipe_metadata(path, document)
    return enqueue_train_jobs_from_recipe_document(
        conn,
        document=document,
        runtime_image_ref=runtime_image_ref,
        run_target=run_target,
        recipe_path=metadata["recipe_path"],
        recipe_sha256=metadata["recipe_sha256"],
        repo_git_commit=metadata["repo_git_commit"],
        repo_dirty=metadata["repo_dirty"],
        seeds=seeds,
    )


def enqueue_train_job(
    conn,
    *,
    goal_slug: str,
    recipe_slug: str | None = None,
    recipe_path: str | None = None,
    recipe_sha256: str | None = None,
    repo_git_commit: str | None = None,
    repo_dirty: bool = False,
    recipe_payload: Mapping[str, Any] | None = None,
    runtime_image_ref: str,
    train_config: Mapping[str, Any],
    run_target: str | None = None,
    max_attempts: int = 1,
    run_name: str | None = None,
    run_description: str | None = None,
    seed: int | None = None,
    wandb_group: str | None = None,
    wandb_tags: Sequence[str] = (),
) -> dict[str, Any]:
    goal_slug = str(goal_slug).strip()
    if not goal_slug:
        raise ValueError("goal_slug is required")
    config = validate_and_normalize_train_config(train_config)
    assert_no_secrets(config, label="train_config")
    assert_no_secrets(recipe_payload or {}, label="recipe_payload")
    require_explicit_queue_train_config(config)
    validate_launch_seed_config(config, seed=seed)
    validate_launch_event_config(config)
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    run_target = normalize_run_target(run_target)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO train_jobs (
                  goal_slug, recipe_slug, recipe_path, recipe_sha256, repo_git_commit,
                  repo_dirty, recipe_payload_json, runtime_image_ref,
                  run_target, train_config, max_attempts, run_name,
                  run_description, seed, wandb_group, wandb_tags
                )
                VALUES (
                  %(goal_slug)s, %(recipe_slug)s, %(recipe_path)s, %(recipe_sha256)s,
                  %(repo_git_commit)s, %(repo_dirty)s, %(recipe_payload_json)s,
                  %(runtime_image_ref)s, %(run_target)s,
                  %(train_config)s, %(max_attempts)s, %(run_name)s,
                  %(run_description)s, %(seed)s, %(wandb_group)s, %(wandb_tags)s
                )
                RETURNING *
                """,
                {
                    "goal_slug": goal_slug,
                    "recipe_slug": recipe_slug,
                    "recipe_path": recipe_path,
                    "recipe_sha256": recipe_sha256,
                    "repo_git_commit": repo_git_commit,
                    "repo_dirty": bool(repo_dirty),
                    "recipe_payload_json": json_arg(dict(recipe_payload or {})),
                    "runtime_image_ref": runtime_image_ref,
                    "run_target": run_target,
                    "train_config": json_arg(config),
                    "max_attempts": max_attempts,
                    "run_name": run_name,
                    "run_description": run_description,
                    "seed": seed,
                    "wandb_group": wandb_group,
                    "wandb_tags": list(wandb_tags),
                },
            )
            row = dict(cur.fetchone())
            record_job_event(
                conn,
                job_id=int(row["id"]),
                event_type="enqueued",
                message="train job enqueued",
                metadata={"goal_slug": goal_slug, "recipe_slug": recipe_slug},
            )
            return row


def new_train_launch_id(job_id: int | None = None) -> str:
    suffix = uuid.uuid4().hex[:12]
    if job_id is None:
        return f"{TRAIN_JOB_KIND}-{suffix}"
    return f"{TRAIN_JOB_KIND}-{int(job_id)}-{suffix}"


def claim_job_launch(
    conn,
    *,
    machine: str,
    backend: str,
    runtime_image_ref: str | None = None,
    run_target: str | None = None,
    job_id: int | None = None,
    launch_id: str | None = None,
    container_name: str | None = None,
    output_uri: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    runtime_image_ref = (
        normalize_runtime_image_ref(runtime_image_ref) if runtime_image_ref else None
    )
    filters = ["cancel_requested = FALSE", "status = 'pending'"]
    params: dict[str, Any] = {
        "machine": str(machine),
        "backend": str(backend),
        "output_uri": str(output_uri),
        "launch_id": launch_id or new_train_launch_id(job_id),
        "container_name": container_name,
        "job_kind": TRAIN_JOB_KIND,
    }
    if job_id is not None:
        filters.append("id = %(job_id)s")
        params["job_id"] = int(job_id)
    if runtime_image_ref is not None:
        filters.append("runtime_image_ref = %(runtime_image_ref)s")
        params["runtime_image_ref"] = runtime_image_ref
    if run_target is not None:
        filters.append("run_target = %(run_target)s")
        params["run_target"] = normalize_run_target(run_target)
    where = "\n    AND ".join(filters)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH next_job AS (
                  SELECT *
                  FROM train_jobs
                  WHERE {where}
                  ORDER BY id ASC
                  LIMIT 1
                  FOR UPDATE SKIP LOCKED
                ),
                updated AS (
                  UPDATE train_jobs AS job
                  SET status = 'launching',
                      lease_owner = %(launch_id)s,
                      lease_expires_at = NULL,
                      heartbeat_at = now(),
                      error = NULL
                  FROM next_job
                  WHERE job.id = next_job.id
                  RETURNING job.*
                ),
                inserted_launch AS (
                  INSERT INTO job_launches (
                    launch_id, job_kind, job_id, backend, machine, runtime_image_ref,
                    container_name, output_uri, state, last_observed_at
                  )
                  SELECT
                    %(launch_id)s, %(job_kind)s, updated.id, %(backend)s, %(machine)s,
                    updated.runtime_image_ref, %(container_name)s, %(output_uri)s,
                    'launching', now()
                  FROM updated
                  RETURNING *
                )
                SELECT
                  row_to_json(updated) AS job_json,
                  row_to_json(inserted_launch) AS launch_json
                FROM updated, inserted_launch
                """,
                params,
            )
            row = cur.fetchone()
            if not row:
                return None
            job = dict(row["job_json"])
            launch = dict(row["launch_json"])
            record_job_event(
                conn,
                job_id=int(job["id"]),
                event_type="launching",
                message=f"job launch claimed on {machine}",
                metadata={"launch_id": launch["launch_id"], "machine": machine, "backend": backend},
            )
            return job, launch


def mark_job_launch_running(
    conn,
    *,
    launch_id: str,
    container_name: str | None = None,
    provider_run_id: str | None = None,
) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_launches AS launch
                SET state = 'running',
                    container_name = COALESCE(%(container_name)s, container_name),
                    provider_run_id = COALESCE(%(provider_run_id)s, provider_run_id),
                    started_at = COALESCE(started_at, now()),
                    last_observed_at = now()
                WHERE launch_id = %(launch_id)s
                RETURNING *
                """,
                {
                    "launch_id": launch_id,
                    "container_name": container_name,
                    "provider_run_id": provider_run_id,
                },
            )
            launch = cur.fetchone()
            if not launch:
                return None
            if launch["job_kind"] != TRAIN_JOB_KIND:
                raise RuntimeError(f"launch {launch_id} is not a train launch")
            cur.execute(
                """
                UPDATE train_jobs
                SET status = 'running',
                    started_at = COALESCE(started_at, now()),
                    heartbeat_at = now(),
                    attempts = attempts + 1
                WHERE id = %(job_id)s
                  AND lease_owner = %(launch_id)s
                  AND status = 'launching'
                """,
                {"job_id": launch["job_id"], "launch_id": launch_id},
            )
            record_job_event(
                conn,
                job_id=int(launch["job_id"]),
                event_type="running",
                message="job container started",
                metadata={"launch_id": launch_id},
            )
            return dict(launch)


def release_job_launch(
    conn,
    *,
    launch_id: str,
    error: str | None = None,
) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_launches
                SET state = 'released',
                    error = %(error)s,
                    last_observed_at = now(),
                    finished_at = now()
                WHERE launch_id = %(launch_id)s
                RETURNING *
                """,
                {"launch_id": launch_id, "error": error},
            )
            launch = cur.fetchone()
            if not launch:
                return None
            if launch["job_kind"] != TRAIN_JOB_KIND:
                raise RuntimeError(f"launch {launch_id} is not a train launch")
            cur.execute(
                """
                UPDATE train_jobs
                SET status = 'pending',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    heartbeat_at = NULL,
                    error = %(error)s
                WHERE id = %(job_id)s
                  AND lease_owner = %(launch_id)s
                  AND status = 'launching'
                """,
                {
                    "job_id": launch["job_id"],
                    "launch_id": launch_id,
                    "error": error,
                },
            )
            record_job_event(
                conn,
                job_id=int(launch["job_id"]),
                event_type="released",
                message=error,
                metadata={"launch_id": launch_id},
            )
            return dict(launch)


def adopt_successful_train_launch(
    conn,
    *,
    launch_id: str,
    superseded_launch_ids: Sequence[str],
) -> tuple[str, ...]:
    """Make a successful launch finalizable after inactive duplicate launches."""
    superseded = tuple(str(value) for value in superseded_launch_ids if str(value) != launch_id)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM job_launches
                WHERE launch_id = %(launch_id)s
                  AND job_kind = 'train'
                  AND state IN ('launching', 'running')
                FOR UPDATE
                """,
                {"launch_id": launch_id},
            )
            launch = cur.fetchone()
            if not launch:
                raise RuntimeError(f"active train launch not found: {launch_id}")
            job_id = int(launch["job_id"])
            if superseded:
                cur.execute(
                    """
                    UPDATE job_launches
                    SET state = 'released',
                        error = %(error)s,
                        last_observed_at = now(),
                        finished_at = now()
                    WHERE job_id = %(job_id)s
                      AND launch_id = ANY(%(launch_ids)s)
                      AND state IN ('launching', 'running')
                    RETURNING launch_id
                    """,
                    {
                        "job_id": job_id,
                        "launch_ids": list(superseded),
                        "error": f"superseded by successful launch {launch_id}",
                    },
                )
                released = tuple(str(row["launch_id"]) for row in cur.fetchall())
            else:
                released = ()
            cur.execute(
                """
                UPDATE train_jobs
                SET lease_owner = %(launch_id)s,
                    heartbeat_at = now()
                WHERE id = %(job_id)s
                  AND status IN ('launching', 'running')
                RETURNING id
                """,
                {"job_id": job_id, "launch_id": launch_id},
            )
            if not cur.fetchone():
                raise RuntimeError(f"active train job not found for launch {launch_id}")
            return released


def active_job_launches(
    conn,
    *,
    machine: str | None = None,
    states: Sequence[str] = ("launching", "running"),
) -> list[dict[str, Any]]:
    filters = ["state = ANY(%(states)s)"]
    params: dict[str, Any] = {"states": list(states)}
    if machine:
        filters.append("machine = %(machine)s")
        params["machine"] = machine
    where = "\n    AND ".join(filters)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT *
            FROM job_launches
            WHERE {where}
            ORDER BY created_at ASC, id ASC
            """,
            params,
        )
        return [dict(row) for row in cur.fetchall()]


def job_payload_for_launch(job: Mapping[str, Any], launch: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "job_kind": launch["job_kind"],
        "job": dict(job),
        "launch_id": launch["launch_id"],
        "machine": launch["machine"],
        "backend": launch["backend"],
        "runtime_image_ref": launch["runtime_image_ref"],
        "output_uri": launch["output_uri"],
    }
    assert_no_secrets(payload, label="job payload")
    return json_safe(payload)


def request_cancel_train_job(conn, *, job_id: int) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs
                SET cancel_requested = TRUE,
                    status = CASE WHEN status IN ('pending', 'launching') THEN 'canceled' ELSE status END,
                    finished_at = CASE WHEN status IN ('pending', 'launching') THEN now() ELSE finished_at END
                WHERE id = %(job_id)s
                  AND status IN ('pending', 'launching', 'running')
                """,
                {"job_id": job_id},
            )
            return int(cur.rowcount)


def _terminal_status_from_result(result: Mapping[str, Any]) -> str:
    status = str(result.get("status") or "").strip()
    if status in {"succeeded", "failed", "canceled"}:
        return status
    exit_code = result.get("exit_code")
    return "succeeded" if exit_code == 0 else "failed"


def _strip_metric_payloads(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _strip_metric_payloads(nested)
            for key, nested in value.items()
            if key != "metrics_json"
        }
    if isinstance(value, list):
        return [_strip_metric_payloads(item) for item in value]
    return value


def launch_result_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    """Keep launch bookkeeping useful without mirroring W&B metrics in Postgres."""

    return json_safe(_strip_metric_payloads(dict(result)))


def finish_train_launch_from_result(
    conn,
    *,
    launch_id: str,
    result: Mapping[str, Any],
) -> None:
    status = _terminal_status_from_result(result)
    exit_code = result.get("exit_code")
    error = str(result.get("error") or "") or None
    train_result = result.get("train")
    train_payload = train_result.get("result") if isinstance(train_result, Mapping) else {}
    train_payload = dict(train_payload or {})
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_launches
                SET state = %(state)s,
                    exit_code = %(exit_code)s,
                    error = %(error)s,
                    result_json = %(result_json)s,
                    last_observed_at = now(),
                    finished_at = now()
                WHERE launch_id = %(launch_id)s
                RETURNING *
                """,
                {
                    "state": status,
                    "exit_code": exit_code,
                    "error": error,
                    "result_json": json_arg(launch_result_metadata(result)),
                    "launch_id": launch_id,
                },
            )
            launch = cur.fetchone()
            if not launch:
                raise RuntimeError(f"unknown launch_id {launch_id}")
            if launch["job_kind"] != "train":
                raise RuntimeError(f"launch {launch_id} is not a train launch")
            cur.execute(
                """
                UPDATE train_jobs
                SET status = %(status)s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    heartbeat_at = now(),
                    finished_at = now(),
                    error = %(error)s
                WHERE id = %(job_id)s
                  AND lease_owner = %(launch_id)s
                  AND status IN ('launching', 'running')
                RETURNING *
                """,
                {
                    "status": status,
                    "error": error,
                    "job_id": launch["job_id"],
                    "launch_id": launch_id,
                },
            )
            job = cur.fetchone()
            if not job:
                raise RuntimeError(f"could not finish train job for launch {launch_id}")
            record_job_event(
                conn,
                job_id=int(job["id"]),
                event_type=status,
                message=error,
                metadata={
                    "launch_id": launch_id,
                    "exit_code": exit_code,
                    "run_name": train_payload.get("run_name") or job.get("run_name"),
                    "wandb_run_id": train_payload.get("wandb_run_id"),
                    "wandb_url": train_payload.get("wandb_url"),
                },
            )


def finish_job_launch_from_result(
    conn,
    *,
    launch_id: str,
    result: Mapping[str, Any],
) -> None:
    job_kind = str(result.get("job_kind") or "")
    if job_kind == "train":
        finish_train_launch_from_result(conn, launch_id=launch_id, result=result)
    else:
        raise ValueError(f"result does not identify train job kind: {job_kind!r}")


def queue_status(conn, *, goal_slug: str) -> dict[str, Any]:
    goal_slug = str(goal_slug).strip()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM train_jobs
            WHERE goal_slug = %(goal_slug)s
            GROUP BY status
            ORDER BY status
            """,
            {"goal_slug": goal_slug},
        )
        train_jobs = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute(
            """
            SELECT id, goal_slug, recipe_slug, status, run_name,
                   run_target, lease_owner, heartbeat_at, created_at
            FROM train_jobs
            WHERE goal_slug = %(goal_slug)s
              AND status IN ('pending', 'launching', 'running')
            ORDER BY
              CASE status WHEN 'running' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
              id ASC
            LIMIT 10
            """,
            {"goal_slug": goal_slug},
        )
        active_train_jobs = [dict(row) for row in cur.fetchall()]
    return {
        "goal_slug": goal_slug,
        "train_jobs": train_jobs,
        "active_train_jobs": active_train_jobs,
    }


def print_status(report: Mapping[str, Any]) -> None:
    print(f"goal: {report['goal_slug']}")
    print(f"train_jobs: {json.dumps(report['train_jobs'], sort_keys=True)}")
    print("active_train_jobs:")
    for row in report.get("active_train_jobs", []):
        print(
            "  "
            f"job={row['id']} status={row['status']} image={row.get('runtime_image_ref') or ''} "
            f"run={row.get('run_name') or ''}"
        )


def build_train_enqueue_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab train",
        description="Create queue-backed train jobs from a checked-in recipe file.",
    )
    add_direct_database_arg(parser)
    parser.add_argument("--recipe-file", dest="recipe_file", type=Path, required=True)
    parser.add_argument("--runtime-image-ref")
    parser.add_argument(
        "--run-target",
        help=(
            "Queue target to claim from a matching fleet machine, for example "
            "rtx4090, rtx2060, or local-macbook."
        ),
    )
    parser.add_argument(
        "--runtime-image-ref-file",
        type=Path,
        help=(
            "JSON artifact or plain-text file containing the immutable runtime image ref; "
            "defaults to latest."
        ),
    )
    parser.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
    parser.add_argument("--image-branch", default=DEFAULT_IMAGE_BRANCH)
    parser.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument(
        "--set",
        dest="recipe_overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Hydra/OmegaConf dotlist recipe override. Repeat for sweeps, for example "
            "--set recipe_id=lr2e4 --set train.policy.learning_rate=2e-4."
        ),
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage rlab train job queues.")
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Create queue tables")
    setup.set_defaults(func=cmd_setup)

    reset = subparsers.add_parser(
        "reset-schema",
        help="Export old queue tables, then drop and recreate the queue schema.",
    )
    reset.add_argument(
        "--export-dir",
        type=Path,
        help="Directory for JSONL exports; defaults to logs/campaign-db-export-<utc>.",
    )
    add_dry_run_arg(reset)
    reset.set_defaults(func=cmd_reset_schema)

    cancel = subparsers.add_parser("cancel-train", help="Request cancellation for a train job")
    cancel.add_argument("job_id", type=int)
    cancel.set_defaults(func=cmd_cancel_train)

    status = subparsers.add_parser("status", help="Print compact queue status")
    status.add_argument("--goal", required=True, dest="goal_slug")
    status.set_defaults(func=cmd_status)
    return parser


def _connect_from_args(args: argparse.Namespace):
    return connect(database_url(args.direct))


def cmd_setup(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        apply_schema(conn)
    finally:
        conn.close()
    print("queue_schema=ok")
    return 0


def default_export_dir() -> Path:
    return Path("logs") / f"campaign-db-export-{_utc_stamp()}"


def cmd_reset_schema(args: argparse.Namespace) -> int:
    export_dir = args.export_dir or default_export_dir()
    if not args.execute:
        print(f"dry_run: would export queue tables to {export_dir} and reset schema")
        print("dry_run: rerun without --dry-run to apply")
        return 0
    conn = _connect_from_args(args)
    try:
        exported = reset_schema(conn, export_dir=export_dir)
    finally:
        conn.close()
    print(f"queue_schema_reset=ok export_dir={exported}")
    return 0


def cmd_enqueue_train(args: argparse.Namespace) -> int:
    runtime_image_ref = runtime_image_ref_from_args(args, default_latest=True)
    if not runtime_image_ref:
        raise SystemExit(
            "--runtime-image-ref, --runtime-image-ref-file, or latest image resolution is required"
        )
    conn = _connect_from_args(args)
    try:
        rows = enqueue_train_jobs_from_recipe_file(
            conn,
            path=args.recipe_file,
            runtime_image_ref=runtime_image_ref,
            run_target=args.run_target,
            seeds=args.seed,
            recipe_overrides=args.recipe_overrides,
        )
    finally:
        conn.close()
    for row in rows:
        print(
            f"train_job_id={row['id']} image={row.get('runtime_image_ref') or ''} "
            f"run_name={row.get('run_name') or ''}"
        )
    return 0


def cmd_cancel_train(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        count = request_cancel_train_job(conn, job_id=args.job_id)
    finally:
        conn.close()
    print(f"cancel_requested={count}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        report = queue_status(
            conn,
            goal_slug=args.goal_slug,
        )
    finally:
        conn.close()
    print_status(report)
    return 0


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
