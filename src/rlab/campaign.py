from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from rlab.compute_targets import instance_defaults, load_json_file
from rlab.runtime_refs import (
    DEFAULT_IMAGE_ARTIFACT,
    DEFAULT_IMAGE_BRANCH,
    DEFAULT_IMAGE_WORKFLOW,
    latest_runtime_image_ref,
    normalize_runtime_image_ref,
    runtime_image_ref_from_file,
)


SECRET_KEY_FRAGMENTS = (
    "api_key",
    "access_key",
    "secret",
    "token",
    "password",
    "credential",
    "database_url",
)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_goals (
  id BIGSERIAL PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE,
  title TEXT NOT NULL,
  objective_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  constraints_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  allowed_train_profiles TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  allowed_eval_profiles TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  status TEXT NOT NULL DEFAULT 'active',
  active_notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment_specs (
  id BIGSERIAL PRIMARY KEY,
  goal_id BIGINT NOT NULL REFERENCES research_goals(id) ON DELETE CASCADE,
  slug TEXT NOT NULL,
  hypothesis TEXT NOT NULL,
  expected_signal TEXT,
  parent_spec_id BIGINT REFERENCES experiment_specs(id) ON DELETE SET NULL,
  train_config JSONB NOT NULL,
  priority INTEGER NOT NULL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'active',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (goal_id, slug)
);

CREATE TABLE IF NOT EXISTS train_jobs (
  id BIGSERIAL PRIMARY KEY,
  goal_id BIGINT NOT NULL REFERENCES research_goals(id) ON DELETE CASCADE,
  experiment_spec_id BIGINT NOT NULL REFERENCES experiment_specs(id) ON DELETE CASCADE,
  profile_id TEXT,
  runtime_image_ref TEXT,
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
  goal_id BIGINT NOT NULL REFERENCES research_goals(id) ON DELETE CASCADE,
  experiment_spec_id BIGINT NOT NULL REFERENCES experiment_specs(id) ON DELETE CASCADE,
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
  goal_id BIGINT NOT NULL REFERENCES research_goals(id) ON DELETE CASCADE,
  experiment_spec_id BIGINT REFERENCES experiment_specs(id) ON DELETE SET NULL,
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
  goal_id BIGINT NOT NULL REFERENCES research_goals(id) ON DELETE CASCADE,
  experiment_spec_id BIGINT REFERENCES experiment_specs(id) ON DELETE SET NULL,
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

ALTER TABLE IF EXISTS train_jobs
  ADD COLUMN IF NOT EXISTS runtime_image_ref TEXT,
  ADD COLUMN IF NOT EXISTS run_target TEXT;

ALTER TABLE IF EXISTS train_results
  ADD COLUMN IF NOT EXISTS runtime_image_ref TEXT,
  ADD COLUMN IF NOT EXISTS run_target TEXT;

ALTER TABLE IF EXISTS train_jobs
  ALTER COLUMN profile_id DROP NOT NULL;

ALTER TABLE IF EXISTS train_results
  ALTER COLUMN profile_id DROP NOT NULL;

ALTER TABLE IF EXISTS eval_jobs
  ADD COLUMN IF NOT EXISTS goal_id BIGINT,
  ADD COLUMN IF NOT EXISTS experiment_spec_id BIGINT,
  ADD COLUMN IF NOT EXISTS train_job_id BIGINT,
  ADD COLUMN IF NOT EXISTS profile_id TEXT,
  ADD COLUMN IF NOT EXISTS eval_config JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS max_attempts INTEGER NOT NULL DEFAULT 1,
  ADD COLUMN IF NOT EXISTS lease_owner TEXT,
  ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS drain_requested BOOLEAN NOT NULL DEFAULT FALSE,
  ADD COLUMN IF NOT EXISTS candidate_label TEXT,
  ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ,
  ADD COLUMN IF NOT EXISTS error TEXT;

DO $$
BEGIN
  IF to_regclass('eval_jobs') IS NOT NULL THEN
    IF EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_name = 'eval_jobs' AND column_name = 'candidate_id'
    ) THEN
      ALTER TABLE eval_jobs ALTER COLUMN candidate_id DROP NOT NULL;
    END IF;
    IF EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_name = 'eval_jobs' AND column_name = 'stage'
    ) THEN
      ALTER TABLE eval_jobs ALTER COLUMN stage DROP NOT NULL;
    END IF;
    IF EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_name = 'eval_jobs' AND column_name = 'episodes'
    ) THEN
      ALTER TABLE eval_jobs ALTER COLUMN episodes DROP NOT NULL;
    END IF;
    IF EXISTS (
      SELECT 1 FROM information_schema.columns
      WHERE table_name = 'eval_jobs' AND column_name = 'seed_start'
    ) THEN
      ALTER TABLE eval_jobs ALTER COLUMN seed_start DROP NOT NULL;
    END IF;
  END IF;
END $$;

ALTER TABLE IF EXISTS eval_results
  ADD COLUMN IF NOT EXISTS eval_job_id BIGINT,
  ADD COLUMN IF NOT EXISTS goal_id BIGINT,
  ADD COLUMN IF NOT EXISTS experiment_spec_id BIGINT,
  ADD COLUMN IF NOT EXISTS train_job_id BIGINT,
  ADD COLUMN IF NOT EXISTS profile_id TEXT,
  ADD COLUMN IF NOT EXISTS status TEXT,
  ADD COLUMN IF NOT EXISTS candidate_label TEXT,
  ADD COLUMN IF NOT EXISTS model_ref TEXT,
  ADD COLUMN IF NOT EXISTS output_path TEXT,
  ADD COLUMN IF NOT EXISTS video_path TEXT,
  ADD COLUMN IF NOT EXISTS metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS error TEXT;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conname = 'eval_results_eval_job_id_unique'
  ) THEN
    ALTER TABLE eval_results
      ADD CONSTRAINT eval_results_eval_job_id_unique UNIQUE (eval_job_id);
  END IF;
END $$;

CREATE TABLE IF NOT EXISTS campaign_decisions (
  id BIGSERIAL PRIMARY KEY,
  goal_id BIGINT NOT NULL REFERENCES research_goals(id) ON DELETE CASCADE,
  decision_type TEXT NOT NULL,
  summary TEXT NOT NULL,
  rationale TEXT NOT NULL,
  affected_spec_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[],
  affected_train_job_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[],
  affected_eval_job_ids BIGINT[] NOT NULL DEFAULT ARRAY[]::BIGINT[],
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
  ON train_jobs (goal_id, status);

CREATE INDEX IF NOT EXISTS eval_jobs_claim_idx
  ON eval_jobs (profile_id, status, priority DESC, id)
  WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS eval_jobs_goal_status_idx
  ON eval_jobs (goal_id, status);

CREATE INDEX IF NOT EXISTS campaign_decisions_goal_created_idx
  ON campaign_decisions (goal_id, created_at DESC);

ALTER TABLE IF EXISTS experiment_specs
  ADD COLUMN IF NOT EXISTS origin_decision_id BIGINT REFERENCES campaign_decisions(id) ON DELETE SET NULL;

ALTER TABLE IF EXISTS train_jobs
  ADD COLUMN IF NOT EXISTS origin_decision_id BIGINT REFERENCES campaign_decisions(id) ON DELETE SET NULL;

ALTER TABLE IF EXISTS eval_jobs
  ADD COLUMN IF NOT EXISTS origin_decision_id BIGINT REFERENCES campaign_decisions(id) ON DELETE SET NULL;
"""


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


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def database_url(use_direct: bool = False) -> str:
    load_dotenv()
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
    if not str(document.get("goal") or document.get("goal_slug") or "").strip():
        raise ValueError(f"{path} must define goal or goal_slug")
    if not str(document.get("slug") or "").strip():
        raise ValueError(f"{path} must define slug")
    if not isinstance(document.get("train_config"), dict):
        raise ValueError(f"{path} must define train_config as an object")
    assert_no_secrets(document, label=f"spec file {path}")
    return document


def spec_goal_slug(document: Mapping[str, Any]) -> str:
    return str(document.get("goal") or document.get("goal_slug") or "").strip()


def experiment_spec_id_from_slug(conn, *, goal_id: int, slug: str) -> int | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id
            FROM experiment_specs
            WHERE goal_id = %(goal_id)s AND slug = %(slug)s
            """,
            {"goal_id": goal_id, "slug": slug},
        )
        row = cur.fetchone()
    return int(row["id"]) if row else None


def create_experiment_spec_from_document(conn, document: Mapping[str, Any]) -> dict[str, Any]:
    goal_id = goal_id_from_slug(conn, spec_goal_slug(document))
    parent_spec_id = document.get("parent_spec_id")
    parent_slug = str(document.get("parent_spec_slug") or "").strip()
    if parent_spec_id is None and parent_slug:
        parent_spec_id = experiment_spec_id_from_slug(
            conn,
            goal_id=goal_id,
            slug=parent_slug,
        )
        if parent_spec_id is None:
            raise ValueError(f"unknown parent_spec_slug for goal {goal_id}: {parent_slug}")
    return create_experiment_spec(
        conn,
        goal_id=goal_id,
        slug=str(document["slug"]),
        hypothesis=str(document.get("hypothesis") or ""),
        expected_signal=document.get("expected_signal"),
        parent_spec_id=int(parent_spec_id) if parent_spec_id is not None else None,
        train_config=document["train_config"],
        priority=int(document.get("priority") or 0),
    )


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
    profile_id: str | None = None,
    run_target: str | None = None,
    instances_path: Path | None = None,
    seeds: Sequence[int] = (),
) -> list[dict[str, Any]]:
    spec = create_experiment_spec_from_document(conn, document)
    goal_id = int(spec["goal_id"])
    spec_id = int(spec["id"])
    profile = str(profile_id).strip() if profile_id else None
    canonical_target = canonicalize_run_target(
        run_target if run_target is not None else document.get("run_target"),
        instances_path=instances_path,
    )
    utc = _utc_stamp()
    rows = []
    for seed in _document_seeds(document, seeds):
        train_config = dict(document["train_config"])
        if seed is not None:
            train_config["seed"] = seed
        row = enqueue_train_job(
            conn,
            goal_id=goal_id,
            experiment_spec_id=spec_id,
            profile_id=profile,
            runtime_image_ref=runtime_image_ref,
            run_target=canonical_target,
            train_config=train_config,
            priority=int(document.get("priority") or 0),
            max_attempts=int(document.get("max_attempts") or 1),
            run_name=_format_seed_template(
                document.get("run_name_template"),
                seed=seed,
                slug=str(document["slug"]),
                utc=utc,
            ),
            run_description=_format_seed_template(
                document.get("run_description_template"),
                seed=seed,
                slug=str(document["slug"]),
                utc=utc,
            ),
            wandb_group=document.get("wandb_group"),
            wandb_tags=[str(tag) for tag in document.get("wandb_tags") or []],
        )
        rows.append(row)
    return rows


def create_goal(
    conn,
    *,
    slug: str,
    title: str,
    objective: Mapping[str, Any] | None = None,
    constraints: Mapping[str, Any] | None = None,
    allowed_train_profiles: Sequence[str] = (),
    allowed_eval_profiles: Sequence[str] = (),
    active_notes: str | None = None,
) -> dict[str, Any]:
    objective = dict(objective or {})
    constraints = dict(constraints or {})
    assert_no_secrets(objective, label="objective")
    assert_no_secrets(constraints, label="constraints")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO research_goals (
                  slug, title, objective_json, constraints_json, allowed_train_profiles,
                  allowed_eval_profiles, active_notes
                )
                VALUES (
                  %(slug)s, %(title)s, %(objective_json)s, %(constraints_json)s,
                  %(allowed_train_profiles)s, %(allowed_eval_profiles)s, %(active_notes)s
                )
                ON CONFLICT (slug) DO UPDATE SET
                  title = EXCLUDED.title,
                  objective_json = EXCLUDED.objective_json,
                  constraints_json = EXCLUDED.constraints_json,
                  allowed_train_profiles = EXCLUDED.allowed_train_profiles,
                  allowed_eval_profiles = EXCLUDED.allowed_eval_profiles,
                  active_notes = EXCLUDED.active_notes,
                  updated_at = now()
                RETURNING *
                """,
                {
                    "slug": slug,
                    "title": title,
                    "objective_json": json_arg(objective),
                    "constraints_json": json_arg(constraints),
                    "allowed_train_profiles": list(allowed_train_profiles),
                    "allowed_eval_profiles": list(allowed_eval_profiles),
                    "active_notes": active_notes,
                },
            )
            return dict(cur.fetchone())


def goal_id_from_slug(conn, slug: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM research_goals WHERE slug = %(slug)s", {"slug": slug})
        row = cur.fetchone()
    if not row:
        raise ValueError(f"unknown research goal slug: {slug}")
    return int(row["id"])


def create_experiment_spec(
    conn,
    *,
    goal_id: int,
    slug: str,
    hypothesis: str,
    train_config: Mapping[str, Any],
    expected_signal: str | None = None,
    parent_spec_id: int | None = None,
    origin_decision_id: int | None = None,
    priority: int = 0,
) -> dict[str, Any]:
    config = dict(train_config)
    assert_no_secrets(config, label="train_config")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO experiment_specs (
                  goal_id, slug, hypothesis, expected_signal, parent_spec_id,
                  origin_decision_id, train_config, priority
                )
                VALUES (
                  %(goal_id)s, %(slug)s, %(hypothesis)s, %(expected_signal)s,
                  %(parent_spec_id)s, %(origin_decision_id)s, %(train_config)s,
                  %(priority)s
                )
                ON CONFLICT (goal_id, slug) DO UPDATE SET
                  hypothesis = EXCLUDED.hypothesis,
                  expected_signal = EXCLUDED.expected_signal,
                  parent_spec_id = EXCLUDED.parent_spec_id,
                  origin_decision_id = EXCLUDED.origin_decision_id,
                  train_config = EXCLUDED.train_config,
                  priority = EXCLUDED.priority,
                  updated_at = now()
                RETURNING *
                """,
                {
                    "goal_id": goal_id,
                    "slug": slug,
                    "hypothesis": hypothesis,
                    "expected_signal": expected_signal,
                    "parent_spec_id": parent_spec_id,
                    "origin_decision_id": origin_decision_id,
                    "train_config": json_arg(config),
                    "priority": priority,
                },
            )
            return dict(cur.fetchone())


def record_decision(
    conn,
    *,
    goal_id: int,
    decision_type: str,
    summary: str,
    rationale: str,
    affected_spec_ids: Sequence[int] = (),
    affected_train_job_ids: Sequence[int] = (),
    affected_eval_job_ids: Sequence[int] = (),
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    assert_no_secrets(metadata, label="decision metadata")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO campaign_decisions (
                  goal_id, decision_type, summary, rationale, affected_spec_ids,
                  affected_train_job_ids, affected_eval_job_ids, metadata_json
                )
                VALUES (
                  %(goal_id)s, %(decision_type)s, %(summary)s, %(rationale)s,
                  %(affected_spec_ids)s, %(affected_train_job_ids)s,
                  %(affected_eval_job_ids)s, %(metadata_json)s
                )
                RETURNING *
                """,
                {
                    "goal_id": goal_id,
                    "decision_type": decision_type,
                    "summary": summary,
                    "rationale": rationale,
                    "affected_spec_ids": list(affected_spec_ids),
                    "affected_train_job_ids": list(affected_train_job_ids),
                    "affected_eval_job_ids": list(affected_eval_job_ids),
                    "metadata_json": json_arg(metadata),
                },
            )
            return dict(cur.fetchone())


def enqueue_train_job(
    conn,
    *,
    goal_id: int,
    experiment_spec_id: int,
    profile_id: str | None,
    runtime_image_ref: str,
    train_config: Mapping[str, Any],
    run_target: str | None = None,
    priority: int = 0,
    max_attempts: int = 1,
    run_name: str | None = None,
    run_description: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: Sequence[str] = (),
    origin_decision_id: int | None = None,
) -> dict[str, Any]:
    config = dict(train_config)
    assert_no_secrets(config, label="train_config")
    profile_id = str(profile_id).strip() if profile_id else None
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    run_target = normalize_run_target(run_target)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO train_jobs (
                  goal_id, experiment_spec_id, profile_id, runtime_image_ref,
                  run_target, train_config, priority, max_attempts, run_name,
                  run_description, wandb_group, wandb_tags, origin_decision_id
                )
                VALUES (
                  %(goal_id)s, %(experiment_spec_id)s, %(profile_id)s,
                  %(runtime_image_ref)s, %(run_target)s, %(train_config)s,
                  %(priority)s, %(max_attempts)s, %(run_name)s,
                  %(run_description)s, %(wandb_group)s, %(wandb_tags)s,
                  %(origin_decision_id)s
                )
                RETURNING *
                """,
                {
                    "goal_id": goal_id,
                    "experiment_spec_id": experiment_spec_id,
                    "profile_id": profile_id,
                    "runtime_image_ref": runtime_image_ref,
                    "run_target": run_target,
                    "train_config": json_arg(config),
                    "priority": priority,
                    "max_attempts": max_attempts,
                    "run_name": run_name,
                    "run_description": run_description,
                    "wandb_group": wandb_group,
                    "wandb_tags": list(wandb_tags),
                    "origin_decision_id": origin_decision_id,
                },
            )
            return dict(cur.fetchone())


def enqueue_eval_job(
    conn,
    *,
    goal_id: int,
    profile_id: str,
    eval_config: Mapping[str, Any],
    experiment_spec_id: int | None = None,
    train_job_id: int | None = None,
    priority: int = 0,
    max_attempts: int = 1,
    candidate_label: str | None = None,
    origin_decision_id: int | None = None,
) -> dict[str, Any]:
    config = dict(eval_config)
    assert_no_secrets(config, label="eval_config")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO eval_jobs (
                  goal_id, experiment_spec_id, train_job_id, profile_id,
                  eval_config, priority, max_attempts, candidate_label,
                  origin_decision_id
                )
                VALUES (
                  %(goal_id)s, %(experiment_spec_id)s, %(train_job_id)s,
                  %(profile_id)s, %(eval_config)s, %(priority)s, %(max_attempts)s,
                  %(candidate_label)s, %(origin_decision_id)s
                )
                RETURNING *
                """,
                {
                    "goal_id": goal_id,
                    "experiment_spec_id": experiment_spec_id,
                    "train_job_id": train_job_id,
                    "profile_id": profile_id,
                    "eval_config": json_arg(config),
                    "priority": priority,
                    "max_attempts": max_attempts,
                    "candidate_label": candidate_label,
                    "origin_decision_id": origin_decision_id,
                },
            )
            return dict(cur.fetchone())


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
                  train_job_id, goal_id, experiment_spec_id, profile_id,
                  runtime_image_ref, run_target, status, exit_code, run_name,
                  run_dir, final_model_path, wandb_run_id, wandb_url,
                  artifact_refs, metrics_json, error
                )
                VALUES (
                  %(train_job_id)s, %(goal_id)s, %(experiment_spec_id)s, %(profile_id)s,
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
                    "goal_id": job["goal_id"],
                    "experiment_spec_id": job["experiment_spec_id"],
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
                  eval_job_id, goal_id, experiment_spec_id, train_job_id, profile_id,
                  status, candidate_label, model_ref, output_path, video_path,
                  metrics_json, error
                )
                VALUES (
                  %(eval_job_id)s, %(goal_id)s, %(experiment_spec_id)s,
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
                    "goal_id": job["goal_id"],
                    "experiment_spec_id": job.get("experiment_spec_id"),
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


def _goal_filter(goal_slug_or_id: str) -> tuple[str, dict[str, Any]]:
    if goal_slug_or_id.isdigit():
        return "id = %(goal_id)s", {"goal_id": int(goal_slug_or_id)}
    return "slug = %(goal_slug)s", {"goal_slug": goal_slug_or_id}


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


def _decision_label(decision: Mapping[str, Any]) -> str:
    return (
        f"decision {decision['id']} [{decision['decision_type']}] "
        f"{_one_line(decision['summary'])}"
    )


def _related_decisions(
    row: Mapping[str, Any],
    *,
    decisions_by_id: Mapping[int, Mapping[str, Any]],
    affected_index: Mapping[int, list[Mapping[str, Any]]],
) -> list[Mapping[str, Any]]:
    related = []
    seen = set()
    origin_id = row.get("origin_decision_id")
    if origin_id is not None:
        decision = decisions_by_id.get(int(origin_id))
        if decision is not None:
            related.append(decision)
            seen.add(int(decision["id"]))
    for decision in affected_index.get(int(row["id"]), []):
        decision_id = int(decision["id"])
        if decision_id not in seen:
            related.append(decision)
            seen.add(decision_id)
    return related


def _append_decision_lines(
    lines: list[str],
    prefix: str,
    row: Mapping[str, Any],
    *,
    decisions_by_id: Mapping[int, Mapping[str, Any]],
    affected_index: Mapping[int, list[Mapping[str, Any]]],
) -> None:
    for decision in _related_decisions(
        row,
        decisions_by_id=decisions_by_id,
        affected_index=affected_index,
    ):
        lines.append(f"{prefix}cause: {_decision_label(decision)}")
        rationale = _one_line(decision.get("rationale"), limit=180)
        if rationale:
            lines.append(f"{prefix}why: {rationale}")


def _index_decisions(
    decisions: Sequence[Mapping[str, Any]],
    affected_key: str,
) -> dict[int, list[Mapping[str, Any]]]:
    index: dict[int, list[Mapping[str, Any]]] = {}
    for decision in decisions:
        for row_id in decision.get(affected_key) or []:
            index.setdefault(int(row_id), []).append(decision)
    return index


def _append_eval_lines(
    lines: list[str],
    *,
    eval_jobs: Sequence[Mapping[str, Any]],
    prefix: str,
    decisions_by_id: Mapping[int, Mapping[str, Any]],
    decisions_by_eval: Mapping[int, list[Mapping[str, Any]]],
) -> None:
    for eval_job in sorted(
        eval_jobs,
        key=lambda row: eval_selection_score(row.get("metrics_json") or {}),
        reverse=True,
    ):
        label = eval_job.get("candidate_label") or ""
        model_ref = eval_job.get("model_ref") or ""
        parts = [
            f"eval {eval_job['id']} [{eval_job['status']}]",
            f"profile={eval_job['profile_id']}",
        ]
        if label:
            parts.append(f"candidate={label}")
        if model_ref:
            parts.append(f"model={model_ref}")
        lines.append(f"{prefix}- {' '.join(parts)}")
        _append_decision_lines(
            lines,
            f"{prefix}  ",
            eval_job,
            decisions_by_id=decisions_by_id,
            affected_index=decisions_by_eval,
        )
        metrics = eval_job.get("metrics_json") or {}
        if metrics:
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
            if summary:
                lines.append(f"{prefix}  result: {summary}")
        if eval_job.get("error") or eval_job.get("result_error"):
            lines.append(f"{prefix}  error: {_one_line(eval_job.get('error') or eval_job['result_error'])}")


def _append_train_lines(
    lines: list[str],
    *,
    train_jobs: Sequence[Mapping[str, Any]],
    evals_by_train: Mapping[int, list[Mapping[str, Any]]],
    prefix: str,
    decisions_by_id: Mapping[int, Mapping[str, Any]],
    decisions_by_train: Mapping[int, list[Mapping[str, Any]]],
    decisions_by_eval: Mapping[int, list[Mapping[str, Any]]],
) -> None:
    for train_job in train_jobs:
        run_name = train_job.get("run_name") or ""
        parts = [
            f"run {train_job['id']} [{train_job['status']}]",
            f"profile={train_job['profile_id']}",
        ]
        if run_name:
            parts.append(f"name={run_name}")
        lines.append(f"{prefix}- {' '.join(parts)}")
        description = _one_line(train_job.get("run_description"))
        if description:
            lines.append(f"{prefix}  desc: {description}")
        _append_decision_lines(
            lines,
            f"{prefix}  ",
            train_job,
            decisions_by_id=decisions_by_id,
            affected_index=decisions_by_train,
        )
        metrics = train_job.get("metrics_json") or {}
        if metrics:
            summary = _metric_summary(
                metrics,
                (
                    "train/done/all",
                    "train/done/level_change",
                    "train/done/level_change/from/0-0",
                    "train/done/level_change/from/0-1",
                    "train/done/life_loss",
                    "total_timesteps",
                    "time/fps",
                ),
            )
            if summary:
                lines.append(f"{prefix}  result: {summary}")
        if train_job.get("wandb_url"):
            lines.append(f"{prefix}  wandb: {train_job['wandb_url']}")
        if train_job.get("error") or train_job.get("result_error"):
            lines.append(
                f"{prefix}  error: {_one_line(train_job.get('error') or train_job['result_error'])}"
            )
        _append_eval_lines(
            lines,
            eval_jobs=evals_by_train.get(int(train_job["id"]), []),
            prefix=f"{prefix}  ",
            decisions_by_id=decisions_by_id,
            decisions_by_eval=decisions_by_eval,
        )


def render_lineage_tree(report: Mapping[str, Any]) -> str:
    goal = report["goal"]
    specs = list(report["specs"])
    train_jobs = list(report["train_jobs"])
    eval_jobs = list(report["eval_jobs"])
    decisions = list(report["decisions"])

    decisions_by_id = {int(decision["id"]): decision for decision in decisions}
    decisions_by_spec = _index_decisions(decisions, "affected_spec_ids")
    decisions_by_train = _index_decisions(decisions, "affected_train_job_ids")
    decisions_by_eval = _index_decisions(decisions, "affected_eval_job_ids")

    specs_by_id = {int(spec["id"]): spec for spec in specs}
    child_specs: dict[int | None, list[Mapping[str, Any]]] = {}
    for spec in specs:
        parent_id = spec.get("parent_spec_id")
        parent_key = int(parent_id) if parent_id is not None and int(parent_id) in specs_by_id else None
        child_specs.setdefault(parent_key, []).append(spec)
    for children in child_specs.values():
        children.sort(key=lambda row: (-int(row.get("priority") or 0), int(row["id"])))

    trains_by_spec: dict[int, list[Mapping[str, Any]]] = {}
    for train_job in train_jobs:
        trains_by_spec.setdefault(int(train_job["experiment_spec_id"]), []).append(train_job)
    for children in trains_by_spec.values():
        children.sort(key=lambda row: (-int(row.get("priority") or 0), int(row["id"])))

    evals_by_train: dict[int, list[Mapping[str, Any]]] = {}
    evals_by_spec: dict[int, list[Mapping[str, Any]]] = {}
    for eval_job in eval_jobs:
        train_job_id = eval_job.get("train_job_id")
        if train_job_id is not None:
            evals_by_train.setdefault(int(train_job_id), []).append(eval_job)
        elif eval_job.get("experiment_spec_id") is not None:
            evals_by_spec.setdefault(int(eval_job["experiment_spec_id"]), []).append(eval_job)
    for children in list(evals_by_train.values()) + list(evals_by_spec.values()):
        children.sort(key=lambda row: (-int(row.get("priority") or 0), int(row["id"])))

    lines = [f"goal {goal['id']}: {goal['slug']} [{goal['status']}]", f"title: {goal['title']}"]
    objective = goal.get("objective_json") or {}
    if objective:
        lines.append(f"objective: {json.dumps(objective, sort_keys=True, default=str)}")
    constraints = goal.get("constraints_json") or {}
    if constraints:
        lines.append(f"constraints: {json.dumps(constraints, sort_keys=True, default=str)}")

    visited: set[int] = set()

    def append_spec(spec: Mapping[str, Any], prefix: str = "") -> None:
        spec_id = int(spec["id"])
        if spec_id in visited:
            lines.append(f"{prefix}- spec {spec_id} {spec['slug']} [cycle]")
            return
        visited.add(spec_id)
        lines.append(f"{prefix}- spec {spec_id} {spec['slug']} [{spec['status']}]")
        parent_id = spec.get("parent_spec_id")
        if parent_id is not None:
            parent = specs_by_id.get(int(parent_id))
            parent_label = parent["slug"] if parent is not None else f"missing:{parent_id}"
            lines.append(f"{prefix}  parent: spec {parent_id} {parent_label}")
        lines.append(f"{prefix}  hypothesis: {_one_line(spec['hypothesis'], limit=180)}")
        expected_signal = _one_line(spec.get("expected_signal"), limit=180)
        if expected_signal:
            lines.append(f"{prefix}  expected: {expected_signal}")
        _append_decision_lines(
            lines,
            f"{prefix}  ",
            spec,
            decisions_by_id=decisions_by_id,
            affected_index=decisions_by_spec,
        )
        _append_train_lines(
            lines,
            train_jobs=trains_by_spec.get(spec_id, []),
            evals_by_train=evals_by_train,
            prefix=f"{prefix}  ",
            decisions_by_id=decisions_by_id,
            decisions_by_train=decisions_by_train,
            decisions_by_eval=decisions_by_eval,
        )
        _append_eval_lines(
            lines,
            eval_jobs=evals_by_spec.get(spec_id, []),
            prefix=f"{prefix}  ",
            decisions_by_id=decisions_by_id,
            decisions_by_eval=decisions_by_eval,
        )
        for child in child_specs.get(spec_id, []):
            append_spec(child, f"{prefix}  ")

    for root in child_specs.get(None, []):
        append_spec(root)
    for spec in specs:
        if int(spec["id"]) not in visited:
            lines.append("unreached_or_cycle:")
            append_spec(spec, "  ")

    return "\n".join(lines)


def campaign_lineage(conn, *, goal_slug_or_id: str) -> dict[str, Any]:
    goal_filter, params = _goal_filter(goal_slug_or_id)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT *
            FROM research_goals
            WHERE {goal_filter}
            """,
            params,
        )
        goal = cur.fetchone()
        if goal is None:
            raise ValueError(f"unknown research goal: {goal_slug_or_id}")
        goal_id = int(goal["id"])
        cur.execute(
            """
            SELECT id, slug, hypothesis, expected_signal, parent_spec_id,
                   origin_decision_id, priority, status, created_at, updated_at
            FROM experiment_specs
            WHERE goal_id = %(goal_id)s
            ORDER BY parent_spec_id NULLS FIRST, priority DESC, id ASC
            """,
            {"goal_id": goal_id},
        )
        specs = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT j.id, j.experiment_spec_id, j.profile_id, j.status, j.priority,
                   j.run_name, j.run_description, j.origin_decision_id,
                   j.created_at, j.started_at, j.finished_at, j.error,
                   r.status AS result_status, r.wandb_url, r.artifact_refs,
                   r.metrics_json, r.error AS result_error
            FROM train_jobs j
            LEFT JOIN train_results r ON r.train_job_id = j.id
            WHERE j.goal_id = %(goal_id)s
            ORDER BY j.experiment_spec_id, j.priority DESC, j.id ASC
            """,
            {"goal_id": goal_id},
        )
        train_jobs = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT j.id, j.experiment_spec_id, j.train_job_id, j.profile_id,
                   j.status, j.priority, j.candidate_label, j.origin_decision_id,
                   j.created_at, j.started_at, j.finished_at, j.error,
                   r.status AS result_status, r.model_ref, r.metrics_json,
                   r.error AS result_error
            FROM eval_jobs j
            LEFT JOIN eval_results r ON r.eval_job_id = j.id
            WHERE j.goal_id = %(goal_id)s
            ORDER BY j.experiment_spec_id NULLS LAST, j.train_job_id NULLS LAST,
                     j.priority DESC, j.id ASC
            """,
            {"goal_id": goal_id},
        )
        eval_jobs = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT id, decision_type, summary, rationale, affected_spec_ids,
                   affected_train_job_ids, affected_eval_job_ids, metadata_json,
                   created_at
            FROM campaign_decisions
            WHERE goal_id = %(goal_id)s
            ORDER BY created_at ASC, id ASC
            """,
            {"goal_id": goal_id},
        )
        decisions = [dict(row) for row in cur.fetchall()]
    return {
        "goal": dict(goal),
        "specs": specs,
        "train_jobs": train_jobs,
        "eval_jobs": eval_jobs,
        "decisions": decisions,
    }


def campaign_status(conn, *, goal_slug_or_id: str, recent_decisions: int = 5) -> dict[str, Any]:
    goal_filter = _goal_filter(goal_slug_or_id)
    where, params = goal_filter
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM research_goals WHERE {where}", params)
        goal = cur.fetchone()
        if not goal:
            raise ValueError(f"unknown research goal: {goal_slug_or_id}")
        goal = dict(goal)
        goal_id = int(goal["id"])
        cur.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM train_jobs
            WHERE goal_id = %(goal_id)s
            GROUP BY status
            ORDER BY status
            """,
            {"goal_id": goal_id},
        )
        train_jobs = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM eval_jobs
            WHERE goal_id = %(goal_id)s
            GROUP BY status
            ORDER BY status
            """,
            {"goal_id": goal_id},
        )
        eval_jobs = {row["status"]: int(row["count"]) for row in cur.fetchall()}
        cur.execute(
            """
            SELECT id, profile_id, status, run_name, wandb_url, final_model_path, created_at
            FROM train_results
            WHERE goal_id = %(goal_id)s
            ORDER BY created_at DESC
            LIMIT 10
            """,
            {"goal_id": goal_id},
        )
        results = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT id, profile_id, status, candidate_label, model_ref, metrics_json, created_at
            FROM eval_results
            WHERE goal_id = %(goal_id)s
            ORDER BY created_at DESC
            LIMIT 10
            """,
            {"goal_id": goal_id},
        )
        eval_results = [dict(row) for row in cur.fetchall()]
        cur.execute(
            """
            SELECT id, profile_id, status, candidate_label, model_ref, metrics_json, created_at
            FROM eval_results
            WHERE goal_id = %(goal_id)s
              AND status = 'succeeded'
            ORDER BY created_at DESC
            LIMIT 100
            """,
            {"goal_id": goal_id},
        )
        eval_selection_leaders = sorted(
            [dict(row) for row in cur.fetchall()],
            key=lambda row: eval_selection_score(row.get("metrics_json") or {}),
            reverse=True,
        )[:5]
        cur.execute(
            """
            SELECT decision_type, summary, rationale, created_at
            FROM campaign_decisions
            WHERE goal_id = %(goal_id)s
            ORDER BY created_at DESC
            LIMIT %(limit)s
            """,
            {"goal_id": goal_id, "limit": recent_decisions},
        )
        decisions = [dict(row) for row in cur.fetchall()]
    return {
        "goal": goal,
        "train_jobs": train_jobs,
        "eval_jobs": eval_jobs,
        "recent_results": results,
        "recent_eval_results": eval_results,
        "eval_selection_leaders": eval_selection_leaders,
        "decisions": decisions,
    }


def print_status(report: Mapping[str, Any]) -> None:
    goal = report["goal"]
    print(f"goal {goal['id']}: {goal['slug']} [{goal['status']}]")
    print(f"title: {goal['title']}")
    print(f"objective: {json.dumps(goal['objective_json'], sort_keys=True, default=str)}")
    print(f"constraints: {json.dumps(goal['constraints_json'], sort_keys=True, default=str)}")
    print(f"train_jobs: {json.dumps(report['train_jobs'], sort_keys=True)}")
    print(f"eval_jobs: {json.dumps(report['eval_jobs'], sort_keys=True)}")
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
    print("recent_decisions:")
    for row in report["decisions"]:
        print(f"  [{row['decision_type']}] {row['summary']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Codex-led PPO research campaigns.")
    parser.add_argument("--direct", action="store_true", help="Use DIRECT_DATABASE_URL.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    setup = subparsers.add_parser("setup", help="Create or update campaign tables")
    setup.set_defaults(func=cmd_setup)

    goal = subparsers.add_parser("create-goal", help="Create or update a research goal")
    goal.add_argument("--slug", required=True)
    goal.add_argument("--title", required=True)
    goal.add_argument("--objective-json", default="{}")
    goal.add_argument("--constraints-json", default="{}")
    goal.add_argument("--train-profile", action="append", default=[])
    goal.add_argument("--eval-profile", action="append", default=[])
    goal.add_argument("--notes")
    goal.set_defaults(func=cmd_create_goal)

    spec = subparsers.add_parser("add-spec", help="Create or update an experiment spec")
    spec.add_argument("--goal", required=True, help="Research goal slug")
    spec.add_argument("--slug", required=True)
    spec.add_argument("--hypothesis", required=True)
    spec.add_argument("--train-config-json", required=True)
    spec.add_argument("--expected-signal")
    spec.add_argument("--parent-spec-id", type=int)
    spec.add_argument("--origin-decision-id", type=int)
    spec.add_argument("--priority", type=int, default=0)
    spec.set_defaults(func=cmd_add_spec)

    spec_file = subparsers.add_parser(
        "add-spec-file",
        help="Create or update an experiment spec from a checked-in JSON spec file.",
    )
    spec_file.add_argument("path", type=Path)
    spec_file.set_defaults(func=cmd_add_spec_file)

    enqueue = subparsers.add_parser("enqueue-train", help="Create a concrete train job")
    enqueue.add_argument("--goal", required=True, help="Research goal slug")
    enqueue.add_argument("--spec-id", type=int, required=True)
    enqueue.add_argument("--profile", required=True)
    enqueue.add_argument("--runtime-image-ref")
    enqueue.add_argument(
        "--runtime-image-ref-file",
        type=Path,
        help="JSON artifact or plain-text file containing the immutable runtime image ref.",
    )
    enqueue.add_argument("--target", dest="run_target", help="Optional compute target required by this job")
    enqueue.add_argument(
        "--instances",
        type=Path,
        default=Path("experiments/instances.json"),
        help="Target config used to canonicalize --target.",
    )
    enqueue.add_argument("--train-config-json", required=True)
    enqueue.add_argument("--priority", type=int, default=0)
    enqueue.add_argument("--max-attempts", type=int, default=1)
    enqueue.add_argument("--run-name")
    enqueue.add_argument("--run-description")
    enqueue.add_argument("--wandb-group")
    enqueue.add_argument("--wandb-tag", action="append", default=[])
    enqueue.add_argument("--origin-decision-id", type=int)
    enqueue.set_defaults(func=cmd_enqueue_train)

    enqueue_spec = subparsers.add_parser(
        "enqueue-train-from-spec",
        help="Create/update a spec file and enqueue one train job per configured seed.",
    )
    enqueue_spec.add_argument("path", type=Path)
    enqueue_spec.add_argument("--profile", help="Override spec profile_id.")
    enqueue_spec.add_argument("--runtime-image-ref")
    enqueue_spec.add_argument(
        "--runtime-image-ref-file",
        type=Path,
        help="JSON artifact or plain-text file containing the immutable runtime image ref.",
    )
    enqueue_spec.add_argument("--target", dest="run_target", help="Override spec run_target.")
    enqueue_spec.add_argument(
        "--instances",
        type=Path,
        default=Path("experiments/instances.json"),
        help="Target config used to canonicalize the spec or override target.",
    )
    enqueue_spec.add_argument("--seed", type=int, action="append", default=[])
    enqueue_spec.set_defaults(func=cmd_enqueue_train_from_spec)

    enqueue_eval = subparsers.add_parser("enqueue-eval", help="Create a concrete eval job")
    enqueue_eval.add_argument("--goal", required=True, help="Research goal slug")
    enqueue_eval.add_argument("--spec-id", type=int)
    enqueue_eval.add_argument("--train-job-id", type=int)
    enqueue_eval.add_argument("--profile", required=True)
    enqueue_eval.add_argument("--eval-config-json", required=True)
    enqueue_eval.add_argument("--priority", type=int, default=0)
    enqueue_eval.add_argument("--max-attempts", type=int, default=1)
    enqueue_eval.add_argument("--candidate-label")
    enqueue_eval.add_argument("--origin-decision-id", type=int)
    enqueue_eval.set_defaults(func=cmd_enqueue_eval)

    decision = subparsers.add_parser("decision", help="Append a Codex campaign decision")
    decision.add_argument("--goal", required=True, help="Research goal slug")
    decision.add_argument("--type", required=True, dest="decision_type")
    decision.add_argument("--summary", required=True)
    decision.add_argument("--rationale", required=True)
    decision.add_argument("--spec-id", type=int, action="append", default=[])
    decision.add_argument("--train-job-id", type=int, action="append", default=[])
    decision.add_argument("--eval-job-id", type=int, action="append", default=[])
    decision.add_argument("--metadata-json", default="{}")
    decision.set_defaults(func=cmd_decision)

    cancel = subparsers.add_parser("cancel-train", help="Request cancellation for a train job")
    cancel.add_argument("job_id", type=int)
    cancel.set_defaults(func=cmd_cancel_train)

    cancel_eval = subparsers.add_parser("cancel-eval", help="Request cancellation for an eval job")
    cancel_eval.add_argument("job_id", type=int)
    cancel_eval.set_defaults(func=cmd_cancel_eval)

    status = subparsers.add_parser("status", help="Print compact campaign status")
    status.add_argument("goal")
    status.add_argument("--recent-decisions", type=int, default=5)
    status.set_defaults(func=cmd_status)

    lineage = subparsers.add_parser("lineage", help="Render a goal's experiment lineage tree")
    lineage.add_argument("goal")
    lineage.add_argument("--format", choices=("text", "json"), default="text")
    lineage.set_defaults(func=cmd_lineage)
    return parser


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
    print("campaign_schema=ok")
    return 0


def cmd_create_goal(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        row = create_goal(
            conn,
            slug=args.slug,
            title=args.title,
            objective=load_json_arg(args.objective_json, default={}),
            constraints=load_json_arg(args.constraints_json, default={}),
            allowed_train_profiles=args.train_profile,
            allowed_eval_profiles=args.eval_profile,
            active_notes=args.notes,
        )
    finally:
        conn.close()
    print(f"goal_id={row['id']} slug={row['slug']}")
    return 0


def cmd_add_spec(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        goal_id = goal_id_from_slug(conn, args.goal)
        row = create_experiment_spec(
            conn,
            goal_id=goal_id,
            slug=args.slug,
            hypothesis=args.hypothesis,
            expected_signal=args.expected_signal,
            parent_spec_id=args.parent_spec_id,
            origin_decision_id=args.origin_decision_id,
            train_config=load_json_arg(args.train_config_json, default={}),
            priority=args.priority,
        )
    finally:
        conn.close()
    print(f"spec_id={row['id']} slug={row['slug']}")
    return 0


def cmd_add_spec_file(args: argparse.Namespace) -> int:
    document = load_spec_document(args.path)
    conn = _connect_from_args(args)
    try:
        row = create_experiment_spec_from_document(conn, document)
    finally:
        conn.close()
    print(f"spec_id={row['id']} slug={row['slug']}")
    return 0


def cmd_enqueue_train(args: argparse.Namespace) -> int:
    runtime_image_ref = (
        runtime_image_ref_from_file(args.runtime_image_ref_file)
        if getattr(args, "runtime_image_ref_file", None)
        else args.runtime_image_ref
    )
    if not runtime_image_ref:
        raise SystemExit("--runtime-image-ref or --runtime-image-ref-file is required")
    conn = _connect_from_args(args)
    try:
        goal_id = goal_id_from_slug(conn, args.goal)
        row = enqueue_train_job(
            conn,
            goal_id=goal_id,
            experiment_spec_id=args.spec_id,
            profile_id=args.profile,
            runtime_image_ref=runtime_image_ref,
            run_target=canonicalize_run_target(args.run_target, instances_path=args.instances),
            train_config=load_json_arg(args.train_config_json, default={}),
            priority=args.priority,
            max_attempts=args.max_attempts,
            run_name=args.run_name,
            run_description=args.run_description,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tag,
            origin_decision_id=args.origin_decision_id,
        )
    finally:
        conn.close()
    target = row.get("run_target") or "any"
    print(
        f"train_job_id={row['id']} profile={row['profile_id']} "
        f"runtime_image_ref={row['runtime_image_ref']} target={target}"
    )
    return 0


def cmd_enqueue_train_from_spec(args: argparse.Namespace) -> int:
    runtime_image_ref = (
        runtime_image_ref_from_file(args.runtime_image_ref_file)
        if getattr(args, "runtime_image_ref_file", None)
        else args.runtime_image_ref
    )
    if not runtime_image_ref:
        raise SystemExit("--runtime-image-ref or --runtime-image-ref-file is required")
    document = load_spec_document(args.path)
    conn = _connect_from_args(args)
    try:
        rows = enqueue_train_jobs_from_spec_document(
            conn,
            document=document,
            runtime_image_ref=runtime_image_ref,
            profile_id=args.profile,
            run_target=args.run_target,
            instances_path=args.instances,
            seeds=args.seed,
        )
    finally:
        conn.close()
    for row in rows:
        target = row.get("run_target") or "any"
        print(
            f"train_job_id={row['id']} profile={row['profile_id']} "
            f"run_name={row.get('run_name') or ''} target={target}"
        )
    return 0


def cmd_enqueue_eval(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        goal_id = goal_id_from_slug(conn, args.goal)
        row = enqueue_eval_job(
            conn,
            goal_id=goal_id,
            experiment_spec_id=args.spec_id,
            train_job_id=args.train_job_id,
            profile_id=args.profile,
            eval_config=load_json_arg(args.eval_config_json, default={}),
            priority=args.priority,
            max_attempts=args.max_attempts,
            candidate_label=args.candidate_label,
            origin_decision_id=args.origin_decision_id,
        )
    finally:
        conn.close()
    print(f"eval_job_id={row['id']} profile={row['profile_id']}")
    return 0


def cmd_decision(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        goal_id = goal_id_from_slug(conn, args.goal)
        row = record_decision(
            conn,
            goal_id=goal_id,
            decision_type=args.decision_type,
            summary=args.summary,
            rationale=args.rationale,
            affected_spec_ids=args.spec_id,
            affected_train_job_ids=args.train_job_id,
            affected_eval_job_ids=args.eval_job_id,
            metadata=load_json_arg(args.metadata_json, default={}),
        )
    finally:
        conn.close()
    print(f"decision_id={row['id']} type={row['decision_type']}")
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


def cmd_status(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        report = campaign_status(
            conn,
            goal_slug_or_id=args.goal,
            recent_decisions=args.recent_decisions,
        )
    finally:
        conn.close()
    print_status(report)
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        report = campaign_lineage(conn, goal_slug_or_id=args.goal)
    finally:
        conn.close()
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(render_lineage_tree(report))
    return 0


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


def new_worker_id(prefix: str = "train-runner") -> str:
    return f"{prefix}-{uuid.uuid4()}"


if __name__ == "__main__":
    main()
