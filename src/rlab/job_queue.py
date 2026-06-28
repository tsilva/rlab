from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import os
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from rlab.compute_targets import instance_defaults, load_json_file
from rlab.dotenv import load_env_file
from rlab.runtime_refs import (
    DEFAULT_IMAGE_ARTIFACT,
    DEFAULT_IMAGE_BRANCH,
    DEFAULT_IMAGE_WORKFLOW,
    latest_runtime_image_ref,
    normalize_runtime_image_ref,
    runtime_image_ref_from_file,
)
from rlab.seeds import validate_training_seed
from rlab.spec_schema import validate_train_spec_schema


SECRET_KEY_FRAGMENTS = (
    "api_key",
    "access_key",
    "secret",
    "token",
    "password",
    "credential",
    "database_url",
)
LEGACY_EVENT_TRAIN_CONFIG_KEYS = ("done_on_info_json", "done_on_info")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS train_jobs (
  id BIGSERIAL PRIMARY KEY,
  goal_slug TEXT NOT NULL,
  spec_slug TEXT,
  spec_path TEXT,
  spec_sha256 TEXT,
  repo_git_commit TEXT,
  repo_dirty BOOLEAN NOT NULL DEFAULT FALSE,
  spec_payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  profile_id TEXT,
  runtime_image_ref TEXT NOT NULL,
  run_target TEXT,
  train_config JSONB NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS train_results (
  id BIGSERIAL PRIMARY KEY,
  train_job_id BIGINT NOT NULL UNIQUE REFERENCES train_jobs(id) ON DELETE CASCADE,
  goal_slug TEXT NOT NULL,
  spec_slug TEXT,
  profile_id TEXT,
  runtime_image_ref TEXT,
  run_target TEXT,
  status TEXT NOT NULL,
  exit_code INTEGER,
  run_name TEXT,
  run_dir TEXT,
  final_model_path TEXT,
  wandb_run_id TEXT,
  wandb_url TEXT,
  artifact_refs JSONB NOT NULL DEFAULT '[]'::jsonb,
  metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_jobs (
  id BIGSERIAL PRIMARY KEY,
  goal_slug TEXT NOT NULL,
  spec_slug TEXT,
  spec_path TEXT,
  train_job_id BIGINT REFERENCES train_jobs(id) ON DELETE SET NULL,
  profile_id TEXT NOT NULL,
  eval_config JSONB NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'pending',
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 1,
  lease_owner TEXT,
  lease_expires_at TIMESTAMPTZ,
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  drain_requested BOOLEAN NOT NULL DEFAULT FALSE,
  candidate_label TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  heartbeat_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  error TEXT
);

CREATE TABLE IF NOT EXISTS eval_results (
  id BIGSERIAL PRIMARY KEY,
  eval_job_id BIGINT NOT NULL UNIQUE REFERENCES eval_jobs(id) ON DELETE CASCADE,
  goal_slug TEXT NOT NULL,
  spec_slug TEXT,
  train_job_id BIGINT REFERENCES train_jobs(id) ON DELETE SET NULL,
  profile_id TEXT NOT NULL,
  status TEXT NOT NULL,
  candidate_label TEXT,
  model_ref TEXT,
  output_path TEXT,
  video_path TEXT,
  metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS job_events (
  id BIGSERIAL PRIMARY KEY,
  job_kind TEXT NOT NULL CHECK (job_kind IN ('train', 'eval')),
  job_id BIGINT NOT NULL,
  event_type TEXT NOT NULL,
  message TEXT,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS train_jobs_claim_idx
  ON train_jobs (profile_id, status, priority DESC, id)
  WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS train_jobs_runtime_claim_idx
  ON train_jobs (profile_id, runtime_image_ref, run_target, status, priority DESC, id)
  WHERE status IN ('pending', 'running') AND runtime_image_ref IS NOT NULL;

CREATE INDEX IF NOT EXISTS train_jobs_goal_status_idx
  ON train_jobs (goal_slug, status);

CREATE INDEX IF NOT EXISTS eval_jobs_claim_idx
  ON eval_jobs (profile_id, status, priority DESC, id)
  WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS train_jobs_spec_status_idx
  ON train_jobs (goal_slug, spec_slug, status);

CREATE INDEX IF NOT EXISTS eval_jobs_goal_status_idx
  ON eval_jobs (goal_slug, status);

CREATE INDEX IF NOT EXISTS job_events_job_idx
  ON job_events (job_kind, job_id, created_at DESC);
"""

RESET_TABLES = (
    "job_events",
    "eval_results",
    "eval_jobs",
    "train_results",
    "train_jobs",
)


CLAIM_TRAIN_JOB_SQL = """
WITH next_job AS (
  SELECT id
  FROM train_jobs
  WHERE
    (%(profile_id)s IS NULL OR profile_id = %(profile_id)s)
    AND runtime_image_ref = %(runtime_image_ref)s
    AND (run_target IS NULL OR run_target = %(run_target)s)
    AND cancel_requested = FALSE
    AND status = 'pending'
  ORDER BY priority DESC, id ASC
  LIMIT 1
  FOR UPDATE SKIP LOCKED
)
UPDATE train_jobs AS job
SET
  status = 'running',
  lease_owner = %(worker_id)s,
  lease_expires_at = now() + (%(lease_seconds)s || ' seconds')::interval,
  attempts = attempts + 1,
  started_at = COALESCE(started_at, now()),
  heartbeat_at = now(),
  error = NULL
FROM next_job
WHERE job.id = next_job.id
RETURNING job.*;
"""


CLAIM_EVAL_JOB_SQL = """
WITH next_job AS (
  SELECT id
  FROM eval_jobs
  WHERE
    profile_id = %(profile_id)s
    AND cancel_requested = FALSE
    AND status = 'pending'
  ORDER BY priority DESC, id ASC
  LIMIT 1
  FOR UPDATE SKIP LOCKED
)
UPDATE eval_jobs AS job
SET
  status = 'running',
  lease_owner = %(worker_id)s,
  lease_expires_at = now() + (%(lease_seconds)s || ' seconds')::interval,
  attempts = attempts + 1,
  started_at = COALESCE(started_at, now()),
  heartbeat_at = now(),
  error = NULL
FROM next_job
WHERE job.id = next_job.id
RETURNING job.*;
"""


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


def canonicalize_run_target(
    value: str | None,
    *,
    instances_path: Path | None = None,
) -> str | None:
    target = normalize_run_target(value)
    if target is None:
        return None
    path = instances_path or Path("experiments/instances.json")
    if not path.is_file():
        return target
    return str(instance_defaults(load_json_file(path), target).get("name", target))


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
                  eval_results,
                  eval_jobs,
                  train_results,
                  train_jobs
                CASCADE
                """
            )
            cur.execute(SCHEMA_SQL)
    return exported


def _contains_secret_key(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key).lower()
            nested_path = f"{path}.{key}" if path else str(key)
            if any(fragment in key_text for fragment in SECRET_KEY_FRAGMENTS):
                return nested_path
            found = _contains_secret_key(nested, nested_path)
            if found:
                return found
    elif isinstance(value, list | tuple):
        for index, nested in enumerate(value):
            found = _contains_secret_key(nested, f"{path}[{index}]")
            if found:
                return found
    return None


def assert_no_secrets(value: Any, *, label: str) -> None:
    found = _contains_secret_key(value)
    if found:
        raise ValueError(f"{label} appears to contain a secret-like key: {found}")


def load_json_arg(value: str | None, *, default: Any) -> Any:
    if value is None or value == "":
        return default
    path = Path(value)
    text = path.read_text(encoding="utf-8") if path.is_file() else value
    return json.loads(text)


def load_spec_document(path: Path) -> dict[str, Any]:
    document = load_json_arg(str(path), default={})
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain a JSON object")
    validate_train_spec_schema(document, label=f"spec file {path}")
    assert_no_secrets(document, label=f"spec file {path}")
    validate_launch_event_config(
        document["train_config"],
        label=f"spec file {path} train_config",
    )
    return document


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_text(args: Sequence[str], *, cwd: Path = Path(".")) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def repo_git_commit(cwd: Path = Path(".")) -> str | None:
    return _git_text(("rev-parse", "HEAD"), cwd=cwd)


def repo_is_dirty(cwd: Path = Path(".")) -> bool:
    text = _git_text(("status", "--porcelain"), cwd=cwd)
    return bool(text)


def spec_slug(document: Mapping[str, Any]) -> str:
    return str(document.get("slug") or "").strip()


def spec_metadata(path: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "goal_slug": spec_goal_slug(document),
        "spec_slug": spec_slug(document),
        "spec_path": str(path),
        "spec_sha256": file_sha256(path),
        "repo_git_commit": repo_git_commit(),
        "repo_dirty": repo_is_dirty(),
        "spec_payload": dict(document),
    }


def record_job_event(
    conn,
    *,
    job_kind: str,
    job_id: int,
    event_type: str,
    message: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    if job_kind not in {"train", "eval"}:
        raise ValueError(f"invalid job_kind: {job_kind}")
    metadata = dict(metadata or {})
    assert_no_secrets(metadata, label="event metadata")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO job_events (job_kind, job_id, event_type, message, metadata_json)
            VALUES (%(job_kind)s, %(job_id)s, %(event_type)s, %(message)s, %(metadata_json)s)
            """,
            {
                "job_kind": job_kind,
                "job_id": job_id,
                "event_type": event_type,
                "message": message,
                "metadata_json": json_arg(metadata),
            },
        )


def _non_empty_config_value(value: Any) -> bool:
    return value not in (None, "", (), [], {})


def _configured_event_names(value: Any, *, label: str) -> tuple[str, ...]:
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        names = tuple(name.strip() for name in value.split(",") if name.strip())
    elif isinstance(value, Sequence):
        names = tuple(str(name).strip() for name in value if str(name).strip())
    else:
        raise ValueError(f"{label} must be a comma-separated string or list")
    return tuple(dict.fromkeys(names))


def _configured_info_event_map(value: Any, *, label: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, Mapping):
            return parsed
    raise ValueError(f"{label} must define info_events_json as an object")


def validate_launch_event_config(train_config: Mapping[str, Any], *, label: str = "train_config") -> None:
    legacy_keys = [
        key
        for key in LEGACY_EVENT_TRAIN_CONFIG_KEYS
        if _non_empty_config_value(train_config.get(key))
    ]
    if legacy_keys:
        raise ValueError(
            f"{label} uses legacy event key(s) {', '.join(legacy_keys)}; "
            "use info_events_json plus done_on_events for new launches"
        )
    done_event_names = _configured_event_names(
        train_config.get("done_on_events"),
        label=f"{label}.done_on_events",
    )
    if not done_event_names:
        return
    info_events = _configured_info_event_map(
        train_config.get("info_events_json"),
        label=label,
    )
    missing = [name for name in done_event_names if name not in info_events]
    if missing:
        raise ValueError(
            f"{label}.done_on_events references unconfigured info event(s): "
            f"{', '.join(missing)}"
        )


def validate_launch_seed_config(
    train_config: Mapping[str, Any],
    *,
    seed: int | None = None,
    label: str = "train_config",
) -> None:
    config_seed = train_config.get("seed")
    seed_span = train_config.get("n_envs", 1)
    if _non_empty_config_value(config_seed):
        validate_training_seed(config_seed, label=f"{label}.seed", seed_span=seed_span)
    if seed is not None:
        validate_training_seed(seed, label="seed", seed_span=seed_span)


def spec_goal_slug(document: Mapping[str, Any]) -> str:
    return str(document.get("goal") or document.get("goal_slug") or "").strip()


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _format_seed_template(template: str | None, *, seed: int | None, slug: str, utc: str) -> str | None:
    if not template:
        return None
    return str(template).format(seed="" if seed is None else seed, slug=slug, utc=utc)


def _document_seeds(document: Mapping[str, Any], override_seeds: Sequence[int] = ()) -> list[int | None]:
    if override_seeds:
        return [int(seed) for seed in override_seeds]
    seeds = document.get("seeds")
    if isinstance(seeds, Sequence) and not isinstance(seeds, str):
        return [int(seed) for seed in seeds]
    train_config = document.get("train_config")
    if isinstance(train_config, Mapping) and train_config.get("seed") is not None:
        return [int(train_config["seed"])]
    return [None]


def enqueue_train_jobs_from_spec_document(
    conn,
    *,
    document: Mapping[str, Any],
    runtime_image_ref: str,
    spec_path: str | None = None,
    spec_sha256: str | None = None,
    repo_git_commit: str | None = None,
    repo_dirty: bool = False,
    profile_id: str | None = None,
    run_target: str | None = None,
    instances_path: Path | None = None,
    seeds: Sequence[int] = (),
    priority_override: int | None = None,
) -> list[dict[str, Any]]:
    validate_train_spec_schema(document)
    profile = str(profile_id).strip() if profile_id else None
    canonical_target = canonicalize_run_target(
        run_target if run_target is not None else document.get("run_target"),
        instances_path=instances_path,
    )
    goal_slug = spec_goal_slug(document)
    document_slug = spec_slug(document)
    utc = _utc_stamp()
    rows = []
    for seed in _document_seeds(document, seeds):
        train_config = dict(document["train_config"])
        if seed is not None:
            validate_training_seed(
                seed,
                label="spec seed",
                seed_span=train_config.get("n_envs", 1),
            )
            train_config["seed"] = seed
        row = enqueue_train_job(
            conn,
            goal_slug=goal_slug,
            spec_slug=document_slug,
            spec_path=spec_path,
            spec_sha256=spec_sha256,
            repo_git_commit=repo_git_commit,
            repo_dirty=repo_dirty,
            spec_payload=document,
            profile_id=profile,
            runtime_image_ref=runtime_image_ref,
            run_target=canonical_target,
            train_config=train_config,
            priority=int(priority_override if priority_override is not None else document.get("priority") or 0),
            max_attempts=int(document.get("max_attempts") or 1),
            run_name=_format_seed_template(
                document.get("run_name_template"),
                seed=seed,
                slug=document_slug,
                utc=utc,
            ),
            run_description=_format_seed_template(
                document.get("run_description_template"),
                seed=seed,
                slug=document_slug,
                utc=utc,
            ),
            seed=seed,
            wandb_group=document.get("wandb_group"),
            wandb_tags=[str(tag) for tag in document.get("wandb_tags") or []],
        )
        rows.append(row)
    return rows


def enqueue_train_jobs_from_spec_file(
    conn,
    *,
    path: Path,
    runtime_image_ref: str,
    profile_id: str | None = None,
    run_target: str | None = None,
    instances_path: Path | None = None,
    seeds: Sequence[int] = (),
    priority_override: int | None = None,
) -> list[dict[str, Any]]:
    document = load_spec_document(path)
    metadata = spec_metadata(path, document)
    return enqueue_train_jobs_from_spec_document(
        conn,
        document=document,
        runtime_image_ref=runtime_image_ref,
        spec_path=metadata["spec_path"],
        spec_sha256=metadata["spec_sha256"],
        repo_git_commit=metadata["repo_git_commit"],
        repo_dirty=metadata["repo_dirty"],
        profile_id=profile_id,
        run_target=run_target,
        instances_path=instances_path,
        seeds=seeds,
        priority_override=priority_override,
    )


def enqueue_train_job(
    conn,
    *,
    goal_slug: str,
    spec_slug: str | None = None,
    spec_path: str | None = None,
    spec_sha256: str | None = None,
    repo_git_commit: str | None = None,
    repo_dirty: bool = False,
    spec_payload: Mapping[str, Any] | None = None,
    profile_id: str | None,
    runtime_image_ref: str,
    train_config: Mapping[str, Any],
    run_target: str | None = None,
    priority: int = 0,
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
    config = dict(train_config)
    assert_no_secrets(config, label="train_config")
    assert_no_secrets(spec_payload or {}, label="spec_payload")
    validate_launch_seed_config(config, seed=seed)
    validate_launch_event_config(config)
    profile_id = str(profile_id).strip() if profile_id else None
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    run_target = normalize_run_target(run_target)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO train_jobs (
                  goal_slug, spec_slug, spec_path, spec_sha256, repo_git_commit,
                  repo_dirty, spec_payload_json, profile_id, runtime_image_ref,
                  run_target, train_config, priority, max_attempts, run_name,
                  run_description, seed, wandb_group, wandb_tags
                )
                VALUES (
                  %(goal_slug)s, %(spec_slug)s, %(spec_path)s, %(spec_sha256)s,
                  %(repo_git_commit)s, %(repo_dirty)s, %(spec_payload_json)s,
                  %(profile_id)s, %(runtime_image_ref)s, %(run_target)s,
                  %(train_config)s, %(priority)s, %(max_attempts)s, %(run_name)s,
                  %(run_description)s, %(seed)s, %(wandb_group)s, %(wandb_tags)s
                )
                RETURNING *
                """,
                {
                    "goal_slug": goal_slug,
                    "spec_slug": spec_slug,
                    "spec_path": spec_path,
                    "spec_sha256": spec_sha256,
                    "repo_git_commit": repo_git_commit,
                    "repo_dirty": bool(repo_dirty),
                    "spec_payload_json": json_arg(dict(spec_payload or {})),
                    "profile_id": profile_id,
                    "runtime_image_ref": runtime_image_ref,
                    "run_target": run_target,
                    "train_config": json_arg(config),
                    "priority": priority,
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
                job_kind="train",
                job_id=int(row["id"]),
                event_type="enqueued",
                message="train job enqueued",
                metadata={"goal_slug": goal_slug, "spec_slug": spec_slug},
            )
            return row


def enqueue_eval_job(
    conn,
    *,
    goal_slug: str,
    profile_id: str,
    eval_config: Mapping[str, Any],
    spec_slug: str | None = None,
    spec_path: str | None = None,
    train_job_id: int | None = None,
    priority: int = 0,
    max_attempts: int = 1,
    candidate_label: str | None = None,
) -> dict[str, Any]:
    goal_slug = str(goal_slug).strip()
    if not goal_slug:
        raise ValueError("goal_slug is required")
    config = dict(eval_config)
    assert_no_secrets(config, label="eval_config")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval_jobs (
                  goal_slug, spec_slug, spec_path, train_job_id, profile_id,
                  eval_config, priority, max_attempts, candidate_label
                )
                VALUES (
                  %(goal_slug)s, %(spec_slug)s, %(spec_path)s, %(train_job_id)s,
                  %(profile_id)s, %(eval_config)s, %(priority)s, %(max_attempts)s,
                  %(candidate_label)s
                )
                RETURNING *
                """,
                {
                    "goal_slug": goal_slug,
                    "spec_slug": spec_slug,
                    "spec_path": spec_path,
                    "train_job_id": train_job_id,
                    "profile_id": profile_id,
                    "eval_config": json_arg(config),
                    "priority": priority,
                    "max_attempts": max_attempts,
                    "candidate_label": candidate_label,
                },
            )
            row = dict(cur.fetchone())
            record_job_event(
                conn,
                job_kind="eval",
                job_id=int(row["id"]),
                event_type="enqueued",
                message="eval job enqueued",
                metadata={"goal_slug": goal_slug, "spec_slug": spec_slug},
            )
            return row


def claim_train_job(
    conn,
    *,
    profile_id: str | None,
    runtime_image_ref: str,
    run_target: str | None,
    worker_id: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    profile_id = str(profile_id).strip() if profile_id else None
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    run_target = normalize_run_target(run_target)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                CLAIM_TRAIN_JOB_SQL,
                {
                    "profile_id": profile_id,
                    "runtime_image_ref": runtime_image_ref,
                    "run_target": run_target,
                    "worker_id": worker_id,
                    "lease_seconds": lease_seconds,
                },
            )
            row = cur.fetchone()
            return dict(row) if row else None


def claim_eval_job(
    conn,
    *,
    profile_id: str,
    worker_id: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                CLAIM_EVAL_JOB_SQL,
                {
                    "profile_id": profile_id,
                    "worker_id": worker_id,
                    "lease_seconds": lease_seconds,
                },
            )
            row = cur.fetchone()
            return dict(row) if row else None


def heartbeat_train_job(
    conn,
    *,
    job_id: int,
    worker_id: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs
                SET heartbeat_at = now(),
                    lease_expires_at = now() + (%(lease_seconds)s || ' seconds')::interval
                WHERE id = %(job_id)s
                  AND lease_owner = %(worker_id)s
                  AND status = 'running'
                RETURNING id, cancel_requested, drain_requested
                """,
                {"job_id": job_id, "worker_id": worker_id, "lease_seconds": lease_seconds},
            )
            row = cur.fetchone()
            return dict(row) if row else None


def heartbeat_eval_job(
    conn,
    *,
    job_id: int,
    worker_id: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_jobs
                SET heartbeat_at = now(),
                    lease_expires_at = now() + (%(lease_seconds)s || ' seconds')::interval
                WHERE id = %(job_id)s
                  AND lease_owner = %(worker_id)s
                  AND status = 'running'
                RETURNING id, cancel_requested, drain_requested
                """,
                {"job_id": job_id, "worker_id": worker_id, "lease_seconds": lease_seconds},
            )
            row = cur.fetchone()
            return dict(row) if row else None


def request_cancel_train_job(conn, *, job_id: int) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs
                SET cancel_requested = TRUE,
                    status = CASE WHEN status = 'pending' THEN 'canceled' ELSE status END,
                    finished_at = CASE WHEN status = 'pending' THEN now() ELSE finished_at END
                WHERE id = %(job_id)s
                  AND status IN ('pending', 'running')
                """,
                {"job_id": job_id},
            )
            return int(cur.rowcount)


def request_cancel_eval_job(conn, *, job_id: int) -> int:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_jobs
                SET cancel_requested = TRUE,
                    status = CASE WHEN status = 'pending' THEN 'canceled' ELSE status END,
                    finished_at = CASE WHEN status = 'pending' THEN now() ELSE finished_at END
                WHERE id = %(job_id)s
                  AND status IN ('pending', 'running')
                """,
                {"job_id": job_id},
            )
            return int(cur.rowcount)


def _normalize_positive_ids(values: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(int(value) for value in values)
    invalid = [value for value in ids if value <= 0]
    if invalid:
        raise ValueError(f"job ids must be positive integers: {invalid}")
    return ids


def _normalize_stale_limit(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _like_prefix(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"{escaped}%"


def _stale_job_filters(
    *,
    job_ids: Sequence[int] = (),
    profile_id: str | None = None,
    lease_owner_prefix: str | None = None,
    older_than_seconds: int = 300,
    limit: int | None = 50,
) -> tuple[list[str], dict[str, Any]]:
    if older_than_seconds < 0:
        raise ValueError("older_than_seconds must be non-negative")
    normalized_ids = _normalize_positive_ids(job_ids)
    profile_id = str(profile_id).strip() if profile_id else None
    lease_owner_prefix = str(lease_owner_prefix).strip() if lease_owner_prefix else None
    filters = [
        "status = 'running'",
        (
            "COALESCE(heartbeat_at, started_at, created_at) <= "
            "now() - (%(older_than_seconds)s || ' seconds')::interval"
        ),
    ]
    params: dict[str, Any] = {
        "older_than_seconds": int(older_than_seconds),
        "limit": _normalize_stale_limit(limit),
    }
    if normalized_ids:
        filters.append("id = ANY(%(job_ids)s)")
        params["job_ids"] = list(normalized_ids)
    if profile_id is not None:
        filters.append("profile_id = %(profile_id)s")
        params["profile_id"] = profile_id
    if lease_owner_prefix is not None:
        filters.append("lease_owner LIKE %(lease_owner_like)s ESCAPE '\\'")
        params["lease_owner_like"] = _like_prefix(lease_owner_prefix)
    return filters, params


def _stale_train_candidate_sql(
    *,
    job_ids: Sequence[int] = (),
    profile_id: str | None = None,
    run_target: str | None = None,
    lease_owner_prefix: str | None = None,
    older_than_seconds: int = 300,
    limit: int | None = 50,
    lock: bool = False,
) -> tuple[str, dict[str, Any]]:
    filters, params = _stale_job_filters(
        job_ids=job_ids,
        profile_id=profile_id,
        lease_owner_prefix=lease_owner_prefix,
        older_than_seconds=older_than_seconds,
        limit=limit,
    )
    run_target = normalize_run_target(run_target)
    if run_target is not None:
        filters.append("run_target = %(run_target)s")
        params["run_target"] = run_target
    lock_clause = "FOR UPDATE SKIP LOCKED" if lock else ""
    where = "\n    AND ".join(filters)
    return (
        f"""
        SELECT
          id, goal_slug, spec_slug, profile_id, runtime_image_ref,
          run_target, run_name, lease_owner AS stale_lease_owner,
          lease_expires_at AS stale_lease_expires_at,
          started_at AS stale_started_at, heartbeat_at AS stale_heartbeat_at
        FROM train_jobs
        WHERE {where}
        ORDER BY id ASC
        LIMIT %(limit)s
        {lock_clause}
        """,
        params,
    )


def list_stale_train_jobs(
    conn,
    *,
    job_ids: Sequence[int] = (),
    profile_id: str | None = None,
    run_target: str | None = None,
    lease_owner_prefix: str | None = None,
    older_than_seconds: int = 300,
    limit: int | None = 50,
) -> list[dict[str, Any]]:
    sql, params = _stale_train_candidate_sql(
        job_ids=job_ids,
        profile_id=profile_id,
        run_target=run_target,
        lease_owner_prefix=lease_owner_prefix,
        older_than_seconds=older_than_seconds,
        limit=limit,
        lock=False,
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def mark_stale_train_jobs_failed(
    conn,
    *,
    job_ids: Sequence[int] = (),
    profile_id: str | None = None,
    run_target: str | None = None,
    lease_owner_prefix: str | None = None,
    older_than_seconds: int = 300,
    limit: int | None = 50,
    error: str | None = None,
) -> list[dict[str, Any]]:
    candidate_sql, params = _stale_train_candidate_sql(
        job_ids=job_ids,
        profile_id=profile_id,
        run_target=run_target,
        lease_owner_prefix=lease_owner_prefix,
        older_than_seconds=older_than_seconds,
        limit=limit,
        lock=True,
    )
    error = error or "worker_lost: stale train job marked failed by rlab-queue"
    params["error"] = error
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH candidates AS (
                  {candidate_sql}
                ),
                updated AS (
                  UPDATE train_jobs AS job
                  SET status = 'failed',
                      lease_owner = NULL,
                      lease_expires_at = NULL,
                      finished_at = now(),
                      error = %(error)s
                  FROM candidates
                  WHERE job.id = candidates.id
                  RETURNING job.*
                ),
                upserted AS (
                  INSERT INTO train_results (
                    train_job_id, goal_slug, spec_slug, profile_id,
                    runtime_image_ref, run_target, status, exit_code, run_name,
                    run_dir, final_model_path, wandb_run_id, wandb_url,
                    artifact_refs, metrics_json, error
                  )
                  SELECT
                    updated.id, updated.goal_slug, updated.spec_slug,
                    updated.profile_id, updated.runtime_image_ref, updated.run_target,
                    'failed', NULL, updated.run_name, existing.run_dir,
                    existing.final_model_path, existing.wandb_run_id, existing.wandb_url,
                    COALESCE(existing.artifact_refs, '[]'::jsonb),
                    COALESCE(existing.metrics_json, '{{}}'::jsonb),
                    %(error)s
                  FROM updated
                  LEFT JOIN train_results AS existing
                    ON existing.train_job_id = updated.id
                  ON CONFLICT (train_job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    exit_code = EXCLUDED.exit_code,
                    run_name = COALESCE(train_results.run_name, EXCLUDED.run_name),
                    run_dir = COALESCE(train_results.run_dir, EXCLUDED.run_dir),
                    final_model_path = COALESCE(
                      train_results.final_model_path,
                      EXCLUDED.final_model_path
                    ),
                    wandb_run_id = COALESCE(train_results.wandb_run_id, EXCLUDED.wandb_run_id),
                    wandb_url = COALESCE(train_results.wandb_url, EXCLUDED.wandb_url),
                    artifact_refs = train_results.artifact_refs,
                    metrics_json = train_results.metrics_json,
                    error = EXCLUDED.error,
                    created_at = now()
                  RETURNING train_job_id
                )
                SELECT
                  updated.id, updated.profile_id, updated.runtime_image_ref,
                  updated.run_target, updated.run_name,
                  candidates.stale_lease_owner, candidates.stale_lease_expires_at,
                  candidates.stale_started_at, candidates.stale_heartbeat_at,
                  updated.finished_at, updated.error
                FROM updated
                JOIN candidates ON candidates.id = updated.id
                ORDER BY updated.id ASC
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]


def _stale_eval_candidate_sql(
    *,
    job_ids: Sequence[int] = (),
    profile_id: str | None = None,
    lease_owner_prefix: str | None = None,
    older_than_seconds: int = 300,
    limit: int | None = 50,
    lock: bool = False,
) -> tuple[str, dict[str, Any]]:
    filters, params = _stale_job_filters(
        job_ids=job_ids,
        profile_id=profile_id,
        lease_owner_prefix=lease_owner_prefix,
        older_than_seconds=older_than_seconds,
        limit=limit,
    )
    lock_clause = "FOR UPDATE SKIP LOCKED" if lock else ""
    where = "\n    AND ".join(filters)
    return (
        f"""
        SELECT
          id, goal_slug, spec_slug, train_job_id, profile_id,
          candidate_label, lease_owner AS stale_lease_owner,
          lease_expires_at AS stale_lease_expires_at,
          started_at AS stale_started_at, heartbeat_at AS stale_heartbeat_at
        FROM eval_jobs
        WHERE {where}
        ORDER BY id ASC
        LIMIT %(limit)s
        {lock_clause}
        """,
        params,
    )


def list_stale_eval_jobs(
    conn,
    *,
    job_ids: Sequence[int] = (),
    profile_id: str | None = None,
    lease_owner_prefix: str | None = None,
    older_than_seconds: int = 300,
    limit: int | None = 50,
) -> list[dict[str, Any]]:
    sql, params = _stale_eval_candidate_sql(
        job_ids=job_ids,
        profile_id=profile_id,
        lease_owner_prefix=lease_owner_prefix,
        older_than_seconds=older_than_seconds,
        limit=limit,
        lock=False,
    )
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def mark_stale_eval_jobs_failed(
    conn,
    *,
    job_ids: Sequence[int] = (),
    profile_id: str | None = None,
    lease_owner_prefix: str | None = None,
    older_than_seconds: int = 300,
    limit: int | None = 50,
    error: str | None = None,
) -> list[dict[str, Any]]:
    candidate_sql, params = _stale_eval_candidate_sql(
        job_ids=job_ids,
        profile_id=profile_id,
        lease_owner_prefix=lease_owner_prefix,
        older_than_seconds=older_than_seconds,
        limit=limit,
        lock=True,
    )
    error = error or "worker_lost: stale eval job marked failed by rlab-queue"
    params["error"] = error
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH candidates AS (
                  {candidate_sql}
                ),
                updated AS (
                  UPDATE eval_jobs AS job
                  SET status = 'failed',
                      lease_owner = NULL,
                      lease_expires_at = NULL,
                      finished_at = now(),
                      error = %(error)s
                  FROM candidates
                  WHERE job.id = candidates.id
                  RETURNING job.*
                ),
                upserted AS (
                  INSERT INTO eval_results (
                    eval_job_id, goal_slug, spec_slug, train_job_id, profile_id,
                    status, candidate_label, model_ref, output_path, video_path,
                    metrics_json, error
                  )
                  SELECT
                    updated.id, updated.goal_slug, updated.spec_slug,
                    updated.train_job_id, updated.profile_id, 'failed',
                    updated.candidate_label, existing.model_ref, existing.output_path,
                    existing.video_path, COALESCE(existing.metrics_json, '{{}}'::jsonb),
                    %(error)s
                  FROM updated
                  LEFT JOIN eval_results AS existing
                    ON existing.eval_job_id = updated.id
                  ON CONFLICT (eval_job_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    candidate_label = COALESCE(
                      eval_results.candidate_label,
                      EXCLUDED.candidate_label
                    ),
                    model_ref = COALESCE(eval_results.model_ref, EXCLUDED.model_ref),
                    output_path = COALESCE(eval_results.output_path, EXCLUDED.output_path),
                    video_path = COALESCE(eval_results.video_path, EXCLUDED.video_path),
                    metrics_json = eval_results.metrics_json,
                    error = EXCLUDED.error,
                    created_at = now()
                  RETURNING eval_job_id
                )
                SELECT
                  updated.id, updated.profile_id, updated.candidate_label,
                  candidates.stale_lease_owner, candidates.stale_lease_expires_at,
                  candidates.stale_started_at, candidates.stale_heartbeat_at,
                  updated.finished_at, updated.error
                FROM updated
                JOIN candidates ON candidates.id = updated.id
                ORDER BY updated.id ASC
                """,
                params,
            )
            return [dict(row) for row in cur.fetchall()]


def finish_train_job(
    conn,
    *,
    job: Mapping[str, Any],
    worker_id: str,
    status: str,
    exit_code: int | None,
    result: Mapping[str, Any],
    error: str | None = None,
) -> None:
    if status not in {"succeeded", "failed", "canceled"}:
        raise ValueError(f"invalid train job terminal status: {status}")
    metrics_json = dict(result.get("metrics_json") or {})
    artifact_refs = list(result.get("artifact_refs") or [])
    assert_no_secrets(metrics_json, label="metrics_json")
    assert_no_secrets(artifact_refs, label="artifact_refs")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs
                SET status = %(status)s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    finished_at = now(),
                    error = %(error)s
                WHERE id = %(job_id)s
                  AND lease_owner = %(worker_id)s
                  AND status = 'running'
                """,
                {
                    "status": status,
                    "error": error,
                    "job_id": job["id"],
                    "worker_id": worker_id,
                },
            )
            if cur.rowcount != 1:
                raise RuntimeError(f"could not finish train job {job['id']} for worker {worker_id}")
            cur.execute(
                """
                INSERT INTO train_results (
                  train_job_id, goal_slug, spec_slug, profile_id,
                  runtime_image_ref, run_target, status, exit_code, run_name,
                  run_dir, final_model_path, wandb_run_id, wandb_url,
                  artifact_refs, metrics_json, error
                )
                VALUES (
                  %(train_job_id)s, %(goal_slug)s, %(spec_slug)s, %(profile_id)s,
                  %(runtime_image_ref)s, %(run_target)s, %(status)s, %(exit_code)s,
                  %(run_name)s, %(run_dir)s, %(final_model_path)s,
                  %(wandb_run_id)s, %(wandb_url)s, %(artifact_refs)s,
                  %(metrics_json)s, %(error)s
                )
                ON CONFLICT (train_job_id) DO UPDATE SET
                  runtime_image_ref = EXCLUDED.runtime_image_ref,
                  run_target = EXCLUDED.run_target,
                  status = EXCLUDED.status,
                  exit_code = EXCLUDED.exit_code,
                  run_name = EXCLUDED.run_name,
                  run_dir = EXCLUDED.run_dir,
                  final_model_path = EXCLUDED.final_model_path,
                  wandb_run_id = EXCLUDED.wandb_run_id,
                  wandb_url = EXCLUDED.wandb_url,
                  artifact_refs = EXCLUDED.artifact_refs,
                  metrics_json = EXCLUDED.metrics_json,
                  error = EXCLUDED.error,
                  created_at = now()
                """,
                {
                    "train_job_id": job["id"],
                    "goal_slug": job["goal_slug"],
                    "spec_slug": job.get("spec_slug"),
                    "profile_id": job["profile_id"],
                    "runtime_image_ref": job.get("runtime_image_ref"),
                    "run_target": job.get("run_target"),
                    "status": status,
                    "exit_code": exit_code,
                    "run_name": result.get("run_name") or job.get("run_name"),
                    "run_dir": result.get("run_dir"),
                    "final_model_path": result.get("final_model_path"),
                    "wandb_run_id": result.get("wandb_run_id"),
                    "wandb_url": result.get("wandb_url"),
                    "artifact_refs": json_arg(artifact_refs),
                    "metrics_json": json_arg(metrics_json),
                    "error": error,
                },
            )
            record_job_event(
                conn,
                job_kind="train",
                job_id=int(job["id"]),
                event_type=status,
                message=error,
                metadata={"exit_code": exit_code},
            )


def record_running_train_result(
    conn,
    *,
    job: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    metrics_json = dict(result.get("metrics_json") or {})
    artifact_refs = list(result.get("artifact_refs") or [])
    assert_no_secrets(metrics_json, label="metrics_json")
    assert_no_secrets(artifact_refs, label="artifact_refs")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO train_results (
                  train_job_id, goal_slug, spec_slug, profile_id,
                  runtime_image_ref, run_target, status, exit_code, run_name,
                  run_dir, final_model_path, wandb_run_id, wandb_url,
                  artifact_refs, metrics_json, error
                )
                VALUES (
                  %(train_job_id)s, %(goal_slug)s, %(spec_slug)s, %(profile_id)s,
                  %(runtime_image_ref)s, %(run_target)s, 'running', NULL,
                  %(run_name)s, %(run_dir)s, %(final_model_path)s,
                  %(wandb_run_id)s, %(wandb_url)s, %(artifact_refs)s,
                  %(metrics_json)s, NULL
                )
                ON CONFLICT (train_job_id) DO UPDATE SET
                  runtime_image_ref = EXCLUDED.runtime_image_ref,
                  run_target = EXCLUDED.run_target,
                  status = CASE
                    WHEN train_results.status = 'running' THEN EXCLUDED.status
                    ELSE train_results.status
                  END,
                  run_name = COALESCE(EXCLUDED.run_name, train_results.run_name),
                  run_dir = COALESCE(EXCLUDED.run_dir, train_results.run_dir),
                  final_model_path = COALESCE(EXCLUDED.final_model_path, train_results.final_model_path),
                  wandb_run_id = COALESCE(EXCLUDED.wandb_run_id, train_results.wandb_run_id),
                  wandb_url = COALESCE(EXCLUDED.wandb_url, train_results.wandb_url),
                  artifact_refs = EXCLUDED.artifact_refs,
                  metrics_json = EXCLUDED.metrics_json
                """,
                {
                    "train_job_id": job["id"],
                    "goal_slug": job["goal_slug"],
                    "spec_slug": job.get("spec_slug"),
                    "profile_id": job["profile_id"],
                    "runtime_image_ref": job.get("runtime_image_ref"),
                    "run_target": job.get("run_target"),
                    "run_name": result.get("run_name") or job.get("run_name"),
                    "run_dir": result.get("run_dir"),
                    "final_model_path": result.get("final_model_path"),
                    "wandb_run_id": result.get("wandb_run_id"),
                    "wandb_url": result.get("wandb_url"),
                    "artifact_refs": json_arg(artifact_refs),
                    "metrics_json": json_arg(metrics_json),
                },
            )


def finish_eval_job(
    conn,
    *,
    job: Mapping[str, Any],
    worker_id: str,
    status: str,
    result: Mapping[str, Any],
    error: str | None = None,
) -> None:
    if status not in {"succeeded", "failed", "canceled"}:
        raise ValueError(f"invalid eval job terminal status: {status}")
    metrics_json = dict(result.get("metrics_json") or {})
    assert_no_secrets(metrics_json, label="metrics_json")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE eval_jobs
                SET status = %(status)s,
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    finished_at = now(),
                    error = %(error)s
                WHERE id = %(job_id)s
                  AND lease_owner = %(worker_id)s
                  AND status = 'running'
                """,
                {
                    "status": status,
                    "error": error,
                    "job_id": job["id"],
                    "worker_id": worker_id,
                },
            )
            if cur.rowcount != 1:
                raise RuntimeError(f"could not finish eval job {job['id']} for worker {worker_id}")
            cur.execute(
                """
                INSERT INTO eval_results (
                  eval_job_id, goal_slug, spec_slug, train_job_id, profile_id,
                  status, candidate_label, model_ref, output_path, video_path,
                  metrics_json, error
                )
                VALUES (
                  %(eval_job_id)s, %(goal_slug)s, %(spec_slug)s,
                  %(train_job_id)s, %(profile_id)s, %(status)s, %(candidate_label)s,
                  %(model_ref)s, %(output_path)s, %(video_path)s, %(metrics_json)s,
                  %(error)s
                )
                ON CONFLICT (eval_job_id) DO UPDATE SET
                  status = EXCLUDED.status,
                  candidate_label = EXCLUDED.candidate_label,
                  model_ref = EXCLUDED.model_ref,
                  output_path = EXCLUDED.output_path,
                  video_path = EXCLUDED.video_path,
                  metrics_json = EXCLUDED.metrics_json,
                  error = EXCLUDED.error,
                  created_at = now()
                """,
                {
                    "eval_job_id": job["id"],
                    "goal_slug": job["goal_slug"],
                    "spec_slug": job.get("spec_slug"),
                    "train_job_id": job.get("train_job_id"),
                    "profile_id": job["profile_id"],
                    "status": status,
                    "candidate_label": result.get("candidate_label") or job.get("candidate_label"),
                    "model_ref": result.get("model_ref"),
                    "output_path": result.get("output_path"),
                    "video_path": result.get("video_path"),
                    "metrics_json": json_arg(metrics_json),
                    "error": error,
                },
            )
            record_job_event(
                conn,
                job_kind="eval",
                job_id=int(job["id"]),
                event_type=status,
                message=error,
            )


def _one_line(value: Any, *, limit: int = 140) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def _metric_summary(metrics: Mapping[str, Any], keys: Sequence[str]) -> str:
    parts = []
    for key in keys:
        if key in metrics and metrics[key] is not None:
            value = metrics[key]
            if isinstance(value, float):
                parts.append(f"{key}={value:.3g}")
            else:
                parts.append(f"{key}={value}")
    return " ".join(parts)


def _metric_float(metrics: Mapping[str, Any], key: str, default: float = float("-inf")) -> float:
    value = metrics.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def eval_selection_score(metrics: Mapping[str, Any]) -> tuple[float, float, float]:
    """Eval-first policy ranking: completion, then progress, then mean reward."""

    completion = _metric_float(
        metrics,
        "eval/done/level_change/from_rate/min",
        default=_metric_float(
            metrics,
            "eval/done/level_change/rate",
            default=_metric_float(metrics, "completion_rate"),
        ),
    )
    return (
        completion,
        _metric_float(metrics, "max_x_max"),
        _metric_float(metrics, "reward_mean"),
    )


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
            SELECT status, COUNT(*) AS count
            FROM eval_jobs
            WHERE goal_slug = %(goal_slug)s
            GROUP BY status
            ORDER BY status
            """,
            {"goal_slug": goal_slug},
        )
        eval_jobs = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute(
            """
            SELECT id, goal_slug, spec_slug, profile_id, status, run_name,
                   run_target, lease_owner, heartbeat_at, created_at
            FROM train_jobs
            WHERE goal_slug = %(goal_slug)s
              AND status IN ('pending', 'running')
            ORDER BY
              CASE status WHEN 'running' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
              priority DESC,
              id ASC
            LIMIT 10
            """,
            {"goal_slug": goal_slug},
        )
        active_train_jobs = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT id, goal_slug, spec_slug, train_job_id, profile_id, status,
                   candidate_label, lease_owner, heartbeat_at, created_at
            FROM eval_jobs
            WHERE goal_slug = %(goal_slug)s
              AND status IN ('pending', 'running')
            ORDER BY
              CASE status WHEN 'running' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END,
              priority DESC,
              id ASC
            LIMIT 10
            """,
            {"goal_slug": goal_slug},
        )
        active_eval_jobs = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT id, goal_slug, spec_slug, profile_id, status, run_name, wandb_url,
                   final_model_path, created_at
            FROM train_results
            WHERE goal_slug = %(goal_slug)s
            ORDER BY created_at DESC
            LIMIT 10
            """,
            {"goal_slug": goal_slug},
        )
        results = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT id, goal_slug, spec_slug, profile_id, status, candidate_label, model_ref,
                   metrics_json, created_at
            FROM eval_results
            WHERE goal_slug = %(goal_slug)s
            ORDER BY created_at DESC
            LIMIT 10
            """,
            {"goal_slug": goal_slug},
        )
        eval_results = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT id, goal_slug, spec_slug, profile_id, status, candidate_label, model_ref,
                   metrics_json, created_at
            FROM eval_results
            WHERE goal_slug = %(goal_slug)s
              AND status = 'succeeded'
            ORDER BY created_at DESC
            LIMIT 100
            """,
            {"goal_slug": goal_slug},
        )
        eval_selection_leaders = sorted(
            [dict(row) for row in cur.fetchall()],
            key=lambda row: eval_selection_score(row.get("metrics_json") or {}),
            reverse=True,
        )[:5]
    return {
        "goal_slug": goal_slug,
        "train_jobs": train_jobs,
        "eval_jobs": eval_jobs,
        "active_train_jobs": active_train_jobs,
        "active_eval_jobs": active_eval_jobs,
        "recent_results": results,
        "recent_eval_results": eval_results,
        "eval_selection_leaders": eval_selection_leaders,
    }


def print_status(report: Mapping[str, Any]) -> None:
    print(f"goal: {report['goal_slug']}")
    print(f"train_jobs: {json.dumps(report['train_jobs'], sort_keys=True)}")
    print(f"eval_jobs: {json.dumps(report['eval_jobs'], sort_keys=True)}")
    print("active_train_jobs:")
    for row in report.get("active_train_jobs", []):
        print(
            "  "
            f"job={row['id']} status={row['status']} profile={row.get('profile_id') or 'any'} "
            f"target={row.get('run_target') or 'any'} run={row.get('run_name') or ''}"
        )
    print("active_eval_jobs:")
    for row in report.get("active_eval_jobs", []):
        print(
            "  "
            f"job={row['id']} status={row['status']} profile={row['profile_id']} "
            f"candidate={row.get('candidate_label') or ''}"
        )
    print("recent_results:")
    for row in report["recent_results"]:
        print(
            "  "
            f"result={row['id']} status={row['status']} profile={row['profile_id']} "
            f"run={row.get('run_name') or ''} wandb={row.get('wandb_url') or ''}"
        )
    print("recent_eval_results:")
    for row in report["recent_eval_results"]:
        metrics = row.get("metrics_json") or {}
        summary = _metric_summary(metrics, ("completion_rate", "max_x_max", "reward_mean"))
        print(
            "  "
            f"result={row['id']} status={row['status']} profile={row['profile_id']} "
            f"candidate={row.get('candidate_label') or ''} model={row.get('model_ref') or ''} "
            f"{summary}"
        )
    print("eval_selection_leaders:")
    for row in report.get("eval_selection_leaders", []):
        metrics = row.get("metrics_json") or {}
        score = eval_selection_score(metrics)
        summary = _metric_summary(
            metrics,
            (
                "eval/done/level_change/from_rate/min",
                "eval/done/level_change/rate",
                "completion_rate",
                "max_x_max",
                "reward_mean",
                "episodes",
            ),
        )
        print(
            "  "
            f"result={row['id']} score=({score[0]:.3g},{score[1]:.3g},{score[2]:.3g}) "
            f"candidate={row.get('candidate_label') or ''} {summary}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage rlab train/eval job queues.")
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

    enqueue = subparsers.add_parser("enqueue-train", help="Create train jobs from a spec file")
    enqueue.add_argument("--spec-file", type=Path, required=True)
    enqueue.add_argument("--profile", help="Optional exact train_jobs.profile_id to require.")
    enqueue.add_argument("--runtime-image-ref")
    enqueue.add_argument(
        "--runtime-image-ref-file",
        type=Path,
        help="JSON artifact or plain-text file containing the immutable runtime image ref; defaults to latest.",
    )
    enqueue.add_argument("--latest-image", action="store_true", help="Resolve the latest successful train image digest.")
    enqueue.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
    enqueue.add_argument("--image-branch", default=DEFAULT_IMAGE_BRANCH)
    enqueue.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)
    enqueue.add_argument("--target", dest="run_target", help="Optional compute target required by this job")
    enqueue.add_argument(
        "--instances",
        type=Path,
        default=Path("experiments/instances.json"),
        help="Target config used to canonicalize --target.",
    )
    enqueue.add_argument("--priority", type=int, help="Override the priority stored in the spec file.")
    enqueue.add_argument("--seed", type=int, action="append", default=[])
    enqueue.set_defaults(func=cmd_enqueue_train)

    enqueue_eval = subparsers.add_parser("enqueue-eval", help="Create a concrete eval job")
    enqueue_eval.add_argument("--goal", required=True, help="Research goal slug")
    enqueue_eval.add_argument("--spec-slug")
    enqueue_eval.add_argument("--spec-path")
    enqueue_eval.add_argument("--train-job-id", type=int)
    enqueue_eval.add_argument("--profile", required=True)
    enqueue_eval.add_argument("--eval-config-json", required=True)
    enqueue_eval.add_argument("--priority", type=int, default=0)
    enqueue_eval.add_argument("--max-attempts", type=int, default=1)
    enqueue_eval.add_argument("--candidate-label")
    enqueue_eval.set_defaults(func=cmd_enqueue_eval)

    cancel = subparsers.add_parser("cancel-train", help="Request cancellation for a train job")
    cancel.add_argument("job_id", type=int)
    cancel.set_defaults(func=cmd_cancel_train)

    cancel_eval = subparsers.add_parser("cancel-eval", help="Request cancellation for an eval job")
    cancel_eval.add_argument("job_id", type=int)
    cancel_eval.set_defaults(func=cmd_cancel_eval)

    stale = subparsers.add_parser(
        "mark-stale-failed",
        help="Mark stale running queue jobs failed after their worker is known lost.",
    )
    stale.add_argument("--job-kind", choices=("train", "eval"), default="train")
    stale.add_argument("--job-id", type=int, action="append", default=[])
    stale.add_argument("--profile", help="Restrict to one profile_id.")
    stale.add_argument("--target", dest="run_target", help="Restrict train jobs to one run_target.")
    stale.add_argument(
        "--instances",
        type=Path,
        default=Path("experiments/instances.json"),
        help="Target config used to canonicalize --target.",
    )
    stale.add_argument(
        "--lease-owner-prefix",
        help="Restrict to running jobs whose lease_owner starts with this prefix.",
    )
    stale.add_argument("--older-than-seconds", type=int, default=300)
    stale.add_argument("--limit", type=int, default=50, help="Maximum rows to affect; 0 means no limit.")
    stale.add_argument("--error", help="Failure message to store on job/result rows.")
    stale.add_argument("--all", action="store_true", help="Allow an unscoped apply.")
    add_dry_run_arg(stale)
    stale.set_defaults(func=cmd_mark_stale_failed)

    status = subparsers.add_parser("status", help="Print compact queue status")
    status.add_argument("--goal", required=True, dest="goal_slug")
    status.set_defaults(func=cmd_status)
    return parser


def add_dry_run_arg(parser: argparse.ArgumentParser) -> None:
    parser.set_defaults(execute=True)
    parser.add_argument(
        "--dry-run",
        dest="execute",
        action="store_false",
        help="Preview planned changes without applying them.",
    )


def _connect_from_args(args: argparse.Namespace):
    return connect(database_url(args.direct))


def runtime_image_ref_from_args(args: argparse.Namespace, *, default_latest: bool = False) -> str | None:
    if getattr(args, "runtime_image_ref_file", None):
        return runtime_image_ref_from_file(args.runtime_image_ref_file)
    if getattr(args, "runtime_image_ref", None):
        return normalize_runtime_image_ref(args.runtime_image_ref)
    if default_latest or getattr(args, "latest_image", False):
        return latest_runtime_image_ref(
            workflow=getattr(args, "image_workflow", DEFAULT_IMAGE_WORKFLOW),
            branch=getattr(args, "image_branch", DEFAULT_IMAGE_BRANCH),
            artifact_name=getattr(args, "image_artifact", DEFAULT_IMAGE_ARTIFACT),
        )
    return None


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
        raise SystemExit("--runtime-image-ref, --runtime-image-ref-file, or latest image resolution is required")
    conn = _connect_from_args(args)
    try:
        rows = enqueue_train_jobs_from_spec_file(
            conn,
            path=args.spec_file,
            profile_id=args.profile,
            runtime_image_ref=runtime_image_ref,
            run_target=args.run_target,
            instances_path=args.instances,
            seeds=args.seed,
            priority_override=args.priority,
        )
    finally:
        conn.close()
    for row in rows:
        target = row.get("run_target") or "any"
        print(
            f"train_job_id={row['id']} profile={row['profile_id'] or 'any'} "
            f"run_name={row.get('run_name') or ''} target={target}"
        )
    return 0


def cmd_enqueue_eval(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        row = enqueue_eval_job(
            conn,
            goal_slug=args.goal,
            spec_slug=args.spec_slug,
            spec_path=args.spec_path,
            train_job_id=args.train_job_id,
            profile_id=args.profile,
            eval_config=load_json_arg(args.eval_config_json, default={}),
            priority=args.priority,
            max_attempts=args.max_attempts,
            candidate_label=args.candidate_label,
        )
    finally:
        conn.close()
    print(f"eval_job_id={row['id']} profile={row['profile_id']}")
    return 0


def cmd_cancel_train(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        count = request_cancel_train_job(conn, job_id=args.job_id)
    finally:
        conn.close()
    print(f"cancel_requested={count}")
    return 0


def cmd_cancel_eval(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        count = request_cancel_eval_job(conn, job_id=args.job_id)
    finally:
        conn.close()
    print(f"cancel_requested={count}")
    return 0


def _stale_scope_selected(args: argparse.Namespace) -> bool:
    return bool(args.job_id or args.profile or args.lease_owner_prefix or args.run_target)


def _print_stale_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    job_kind: str,
    execute: bool,
) -> None:
    action = "failed" if execute else "would_fail"
    print(f"stale_{job_kind}_jobs_{action}={len(rows)}")
    for row in rows:
        name = row.get("run_name") or row.get("candidate_label") or ""
        target = row.get("run_target") or ""
        print(
            "  "
            f"{job_kind}_job_id={row['id']} "
            f"profile={row.get('profile_id') or 'any'} "
            f"target={target or 'any'} "
            f"owner={row.get('stale_lease_owner') or 'unknown'} "
            f"heartbeat={row.get('stale_heartbeat_at') or 'unknown'} "
            f"name={name}"
        )


def cmd_mark_stale_failed(args: argparse.Namespace) -> int:
    if args.job_kind == "eval" and args.run_target:
        raise SystemExit("--target is only valid for train jobs")
    if args.execute and not args.all and not _stale_scope_selected(args):
        raise SystemExit(
            "refusing unscoped apply; pass --job-id, --profile, "
            "--target, --lease-owner-prefix, or --all"
        )
    run_target = (
        canonicalize_run_target(args.run_target, instances_path=args.instances)
        if args.job_kind == "train"
        else None
    )
    conn = _connect_from_args(args)
    try:
        if args.job_kind == "train":
            common = {
                "job_ids": args.job_id,
                "profile_id": args.profile,
                "run_target": run_target,
                "lease_owner_prefix": args.lease_owner_prefix,
                "older_than_seconds": args.older_than_seconds,
                "limit": args.limit,
            }
            if args.execute:
                rows = mark_stale_train_jobs_failed(conn, **common, error=args.error)
            else:
                rows = list_stale_train_jobs(conn, **common)
        else:
            common = {
                "job_ids": args.job_id,
                "profile_id": args.profile,
                "lease_owner_prefix": args.lease_owner_prefix,
                "older_than_seconds": args.older_than_seconds,
                "limit": args.limit,
            }
            if args.execute:
                rows = mark_stale_eval_jobs_failed(conn, **common, error=args.error)
            else:
                rows = list_stale_eval_jobs(conn, **common)
    finally:
        conn.close()
    _print_stale_rows(rows, job_kind=args.job_kind, execute=args.execute)
    if not args.execute:
        print("dry_run: rerun without --dry-run to mark these stale jobs failed")
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


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


def new_worker_id(prefix: str = "train-runner") -> str:
    return f"{prefix}-{uuid.uuid4()}"


if __name__ == "__main__":
    main()
