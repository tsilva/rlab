from __future__ import annotations

import argparse
import copy
from datetime import UTC, datetime, timedelta
from dataclasses import dataclass
import hashlib
import json
import os
import re
import shlex
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras

from rlab.cli_args import add_direct_database_arg, add_dry_run_arg
from rlab.dotenv import load_env_file
from rlab.json_utils import json_safe
from rlab.machines import DEFAULT_MACHINE_REGISTRY, load_machine_registry, resolve_machine
from rlab.runtime_refs import (
    DEFAULT_IMAGE_ARTIFACT,
    DEFAULT_IMAGE_BRANCH,
    DEFAULT_IMAGE_WORKFLOW,
    normalize_runtime_image_ref,
    runtime_release_from_args,
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
from rlab.seeds import DEFAULT_TRAIN_SEED, validate_training_seed
from rlab.recipe_schema import (
    require_explicit_queue_train_config,
    validate_materialized_train_recipe,
)
from rlab.provider_config import provider_num_envs
from rlab.train_config import validate_and_normalize_train_config
from rlab.modal_eval_assets import asset_manifest_for_game
from rlab.modal_eval_config import load_modal_eval_config


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
  machine TEXT NOT NULL,
  train_config JSONB NOT NULL,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending', 'launching', 'starting', 'running', 'succeeded', 'failed', 'canceled')),
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  batch_id TEXT NOT NULL,
  campaign_id TEXT,
  submission_key TEXT NOT NULL,
  submission_ordinal INTEGER NOT NULL CHECK (submission_ordinal >= 0),
  request_hash TEXT NOT NULL,
  retry_of_job_id BIGINT REFERENCES train_jobs(id),
  retried_from_job_id BIGINT REFERENCES train_jobs(id),
  run_name TEXT,
  run_description TEXT,
  seed INTEGER,
  wandb_group TEXT,
  wandb_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  ready_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  wandb_run_id TEXT,
  wandb_url TEXT,
  error TEXT,
  UNIQUE (submission_key, submission_ordinal)
);

CREATE TABLE IF NOT EXISTS job_launches (
  id BIGSERIAL PRIMARY KEY,
  launch_id TEXT NOT NULL UNIQUE,
  job_kind TEXT NOT NULL CHECK (job_kind IN ('train')),
  job_id BIGINT NOT NULL UNIQUE REFERENCES train_jobs(id) ON DELETE CASCADE,
  backend TEXT NOT NULL,
  machine TEXT NOT NULL,
  runtime_image_ref TEXT NOT NULL,
  container_name TEXT NOT NULL,
  provider_run_id TEXT,
  output_uri TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'launching'
    CHECK (state IN ('launching', 'running', 'succeeded', 'failed', 'canceled')),
  exit_code INTEGER,
  error TEXT,
  next_retry_at TIMESTAMPTZ,
  result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  last_observed_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS machine_controls (
  machine TEXT PRIMARY KEY,
  drained BOOLEAN NOT NULL DEFAULT FALSE,
  effective_capacity INTEGER CHECK (effective_capacity IS NULL OR effective_capacity >= 1),
  reason TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS job_events (
  id BIGSERIAL PRIMARY KEY,
  job_kind TEXT NOT NULL CHECK (job_kind IN ('train')),
  job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  message TEXT,
  metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS eval_runs (
  train_job_id BIGINT PRIMARY KEY REFERENCES train_jobs(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'awaiting_artifact_recovery', 'finalizing', 'complete', 'failed')),
  contract_json JSONB NOT NULL,
  next_announcement_id BIGINT NOT NULL DEFAULT 1 CHECK (next_announcement_id >= 1),
  next_artifact_projection_id BIGINT NOT NULL DEFAULT 1 CHECK (next_artifact_projection_id >= 1),
  complete_announcement_seen BOOLEAN NOT NULL DEFAULT FALSE,
  last_scheduled_at TIMESTAMPTZ,
  promoted_eval_job_id BIGINT,
  promotion_json JSONB,
  artifacts_projected_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  error TEXT
);

CREATE TABLE IF NOT EXISTS eval_jobs (
  id BIGSERIAL PRIMARY KEY,
  train_job_id BIGINT NOT NULL REFERENCES eval_runs(train_job_id) ON DELETE CASCADE,
  ledger_id BIGINT NOT NULL,
  checkpoint_step BIGINT NOT NULL,
  checkpoint_sha256 TEXT NOT NULL,
  checkpoint_uri TEXT NOT NULL,
  metadata_uri TEXT NOT NULL,
  stage_name TEXT NOT NULL,
  stage_index INTEGER NOT NULL CHECK (stage_index >= 0),
  purpose TEXT NOT NULL CHECK (purpose IN ('screen', 'confirm', 'promotion')),
  execution_key TEXT NOT NULL,
  job_key TEXT NOT NULL UNIQUE,
  contract_json JSONB NOT NULL,
  source_announcement_json JSONB NOT NULL,
  decision_rules_json JSONB NOT NULL DEFAULT '[]'::jsonb,
  candidate_stop BOOLEAN NOT NULL DEFAULT FALSE,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN (
      'pending', 'dispatching', 'submitted', 'succeeded', 'failed', 'skipped_stale',
      'blocked_budget', 'canceled'
    )),
  accepted_attempt_id BIGINT,
  decision_json JSONB,
  projected_at TIMESTAMPTZ,
  projection_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finished_at TIMESTAMPTZ,
  error TEXT,
  UNIQUE (train_job_id, ledger_id, stage_name)
);

CREATE TABLE IF NOT EXISTS eval_attempts (
  id BIGSERIAL PRIMARY KEY,
  attempt_id TEXT NOT NULL UNIQUE,
  eval_job_id BIGINT NOT NULL REFERENCES eval_jobs(id) ON DELETE CASCADE,
  attempt_number INTEGER NOT NULL CHECK (attempt_number BETWEEN 1 AND 2),
  status TEXT NOT NULL DEFAULT 'dispatching'
    CHECK (status IN ('dispatching', 'submitted', 'succeeded', 'failed', 'expired', 'canceled')),
  modal_app_name TEXT NOT NULL,
  modal_function_name TEXT NOT NULL,
  modal_call_id TEXT,
  result_uri TEXT NOT NULL,
  reserved_cost_usd DOUBLE PRECISION NOT NULL CHECK (reserved_cost_usd >= 0),
  actual_cost_usd DOUBLE PRECISION CHECK (actual_cost_usd IS NULL OR actual_cost_usd >= 0),
  expires_at TIMESTAMPTZ NOT NULL,
  result_json JSONB,
  receipt_json JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  error TEXT,
  UNIQUE (eval_job_id, attempt_number)
);

CREATE TABLE IF NOT EXISTS eval_backend_state (
  backend TEXT PRIMARY KEY CHECK (backend = 'modal'),
  drained BOOLEAN NOT NULL DEFAULT FALSE,
  effective_capacity INTEGER NOT NULL CHECK (effective_capacity >= 1),
  round_robin_after_train_job_id BIGINT NOT NULL DEFAULT 0,
  reason TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO eval_backend_state (backend, effective_capacity)
VALUES ('modal', 1)
ON CONFLICT (backend) DO NOTHING;

DROP INDEX IF EXISTS train_jobs_runtime_claim_idx;
DROP INDEX IF EXISTS train_jobs_claim_idx;
DROP INDEX IF EXISTS train_jobs_spec_status_idx;

ALTER TABLE train_jobs DROP COLUMN IF EXISTS priority;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS campaign_id TEXT;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS retry_of_job_id BIGINT REFERENCES train_jobs(id);
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS ready_at TIMESTAMPTZ;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS wandb_run_id TEXT;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS wandb_url TEXT;
ALTER TABLE train_jobs DROP CONSTRAINT IF EXISTS train_jobs_status_check;
ALTER TABLE train_jobs ADD CONSTRAINT train_jobs_status_check
  CHECK (status IN ('pending', 'launching', 'starting', 'running', 'succeeded', 'failed', 'canceled'));

CREATE INDEX IF NOT EXISTS train_jobs_claim_idx
  ON train_jobs (machine, status, id)
  WHERE status = 'pending' AND cancel_requested = FALSE;

CREATE INDEX IF NOT EXISTS train_jobs_runtime_claim_idx
  ON train_jobs (machine, runtime_image_ref, status, id)
  WHERE status IN ('pending', 'launching', 'starting', 'running');

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

CREATE INDEX IF NOT EXISTS eval_jobs_status_idx
  ON eval_jobs (status, stage_index DESC, train_job_id, created_at);

CREATE INDEX IF NOT EXISTS eval_attempts_status_idx
  ON eval_attempts (status, expires_at, created_at);

CREATE INDEX IF NOT EXISTS eval_jobs_execution_idx
  ON eval_jobs (execution_key, status);
"""

RESET_TABLES = (
    "eval_attempts",
    "eval_jobs",
    "eval_runs",
    "eval_backend_state",
    "job_events",
    "job_launches",
    "train_jobs",
    "machine_controls",
)
TRAIN_JOB_KIND = "train"
SCHEMA_MAINTENANCE_LOCK = "rlab-fleet-schema-maintenance"


@dataclass(frozen=True)
class QueueDemand:
    machine: str
    runtime_image_ref: str
    pending_count: int
    active_count: int
    oldest_job_id: int

    @property
    def total(self) -> int:
        return self.pending_count + self.active_count


QUEUE_DEMAND_SQL = """
SELECT
  machine,
  runtime_image_ref,
  COUNT(*) FILTER (WHERE status = 'pending') AS pending_count,
  COUNT(*) FILTER (WHERE status IN ('launching', 'starting', 'running')) AS active_count,
  MIN(id) AS oldest_job_id
FROM train_jobs
WHERE runtime_image_ref IS NOT NULL
  AND cancel_requested = FALSE
  AND status IN ('pending', 'launching', 'starting', 'running')
GROUP BY machine, runtime_image_ref
ORDER BY oldest_job_id ASC
"""


def queue_demands(conn) -> list[QueueDemand]:
    with conn.cursor() as cur:
        cur.execute(QUEUE_DEMAND_SQL)
        rows = cur.fetchall()
    return [
        QueueDemand(
            machine=str(row["machine"]),
            runtime_image_ref=normalize_runtime_image_ref(row["runtime_image_ref"]),
            pending_count=int(row["pending_count"]),
            active_count=int(row["active_count"]),
            oldest_job_id=int(row["oldest_job_id"]),
        )
        for row in rows
    ]


def machine_queue_counts(conn, *, machine: str) -> dict[str, dict[str, int]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM train_jobs
            WHERE machine = %(machine)s
            GROUP BY status
            ORDER BY status
            """,
            {"machine": normalize_machine(machine)},
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


def normalize_machine(value: str | None) -> str:
    machine = str(value or "").strip()
    if not machine:
        raise ValueError("machine is required")
    return machine


def connect(url: str):
    return psycopg2.connect(
        url,
        connect_timeout=10,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def prepare_schema_upgrade(conn) -> None:
    """Preserve the retired eval queue before creating Modal eval tables.

    Older installations used ``eval_jobs`` for a leased machine queue with a
    completely different contract. Rename that table transactionally instead
    of attempting an unsafe in-place reinterpretation or dropping its rows.
    """

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = current_schema() AND table_name = 'eval_jobs'
            """
        )
        columns = {str(row["column_name"]) for row in cur.fetchall()}
        if not columns or "job_key" in columns:
            return
        legacy_table = "legacy_eval_jobs_pre_modal"
        cur.execute("SELECT to_regclass(%(table)s) AS table_name", {"table": legacy_table})
        if cur.fetchone()["table_name"]:
            raise RuntimeError(
                "legacy eval_jobs schema is present but its preservation table already exists"
            )
        cur.execute(f"ALTER TABLE eval_jobs RENAME TO {legacy_table}")


def apply_schema(conn) -> None:
    with conn:
        prepare_schema_upgrade(conn)
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
        order_column = (
            "machine"
            if table_name == "machine_controls"
            else "backend"
            if table_name == "eval_backend_state"
            else "train_job_id"
            if table_name == "eval_runs"
            else "id"
        )
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table_name} ORDER BY {order_column}")
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
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%(key)s, 0))",
                {"key": SCHEMA_MAINTENANCE_LOCK},
            )
        exported = export_existing_tables(conn, export_dir)
        with conn.cursor() as cur:
            cur.execute(
                """
                DROP TABLE IF EXISTS
                  eval_attempts,
                  eval_jobs,
                  eval_runs,
                  eval_backend_state,
                  job_events,
                  job_launches,
                  train_jobs,
                  machine_controls
                CASCADE
                """
            )
            cur.execute(SCHEMA_SQL)
            cur.execute(
                """
                DO $$
                BEGIN
                  IF to_regclass('train_jobs') IS NULL
                     OR to_regclass('job_launches') IS NULL
                     OR to_regclass('machine_controls') IS NULL
                     OR to_regclass('job_events') IS NULL
                     OR to_regclass('eval_runs') IS NULL
                     OR to_regclass('eval_jobs') IS NULL
                     OR to_regclass('eval_attempts') IS NULL
                     OR to_regclass('eval_backend_state') IS NULL THEN
                    RAISE EXCEPTION 'queue schema validation failed';
                  END IF;
                END $$
                """
            )
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
    seed: int,
    recipe_id: str,
    utc: str,
    batch_id: str,
    campaign_id: str = "",
) -> str | None:
    if not template:
        return None
    return str(template).format(
        seed=seed,
        recipe_id=recipe_id,
        timestamp=utc,
        utc=utc,
        batch_id=batch_id,
        campaign_id=campaign_id,
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


def _format_default_run_name(
    batch_id: str,
    *,
    label: str,
    seed: int,
    utc: str,
) -> str:
    batch_id = _run_name_slug(batch_id)
    description = _run_name_slug(label, limit=24)
    return f"{batch_id}-{description}-s{seed}-{utc}"


def _validate_queue_run_name(value: str, *, batch_id: str, label: str, seed: int) -> str:
    prefix = _format_default_run_name(batch_id, label=label, seed=seed, utc="")
    if not re.fullmatch(rf"{re.escape(prefix)}\d{{8}}T\d{{6}}Z", value):
        raise ValueError(
            "queue run_name must use <batch_id>-<recipe>-s<seed>-<utc> with the effective seed"
        )
    return value


def _document_seeds(
    document: Mapping[str, Any], override_seeds: Sequence[int] = ()
) -> list[int]:
    if override_seeds:
        return [int(seed) for seed in override_seeds]
    seeds = document.get("seeds")
    if isinstance(seeds, Sequence) and not isinstance(seeds, str):
        return [int(seed) for seed in seeds]
    train_config = document.get("train_config")
    if isinstance(train_config, Mapping) and train_config.get("seed") is not None:
        return [int(train_config["seed"])]
    return [DEFAULT_TRAIN_SEED]


def _hash_json(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _submission_batch_id(submission_key: str) -> str:
    return "bx" + _hash_json({"submission_key": submission_key})[:16]


def _submission_request_hash(
    *,
    document: Mapping[str, Any],
    machine: str,
    runtime_image_ref: str,
    seeds: Sequence[int],
) -> str:
    return _hash_json(
        {
            "document": document,
            "machine": machine,
            "runtime_image_ref": runtime_image_ref,
            "seeds": list(seeds),
        }
    )


def _existing_submission(conn, *, submission_key: str) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM train_jobs
            WHERE submission_key = %(submission_key)s
            ORDER BY submission_ordinal
            """,
            {"submission_key": submission_key},
        )
        return [dict(row) for row in cur.fetchall()]


def modal_eval_readiness_report(*, runtime_image_ref: str, game: str) -> dict[str, Any]:
    # Keep this import local: modal_eval_cli uses the queue connection helpers.
    from rlab.modal_eval_cli import modal_preflight

    return modal_preflight(runtime_image_ref=runtime_image_ref, game=game)


def require_modal_eval_ready(*, runtime_image_ref: str, game: str) -> dict[str, Any]:
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    game = str(game or "").strip()
    report = modal_eval_readiness_report(runtime_image_ref=runtime_image_ref, game=game)
    if bool(report.get("ready")):
        return report
    failed = [check for check in report.get("checks", []) if not bool(check.get("ok"))]
    detail = (
        ", ".join(
            f"{check.get('name', 'unknown')} ({check.get('detail', 'failed')})"
            for check in failed
        )
        or "unknown readiness failure"
    )
    remediation = shlex.join(
        [
            "rlab",
            "eval",
            "modal",
            "preflight",
            "--runtime-image-ref",
            runtime_image_ref,
            "--game",
            game,
        ]
    )
    raise RuntimeError(f"Modal eval preflight failed: {detail}. Remediation: run {remediation}")


def enqueue_train_jobs_from_recipe_document(
    conn,
    *,
    document: Mapping[str, Any],
    runtime_image_ref: str,
    machine: str,
    submission_key: str | None = None,
    recipe_path: str | None = None,
    recipe_sha256: str | None = None,
    repo_git_commit: str | None = None,
    repo_dirty: bool = False,
    seeds: Sequence[int] = (),
    checkpoint_eval_backend: str | None = None,
    runtime_config_validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    document = copy.deepcopy(dict(document))
    train_config = dict(document.get("train_config") or {})
    backend = str(checkpoint_eval_backend or train_config.get("checkpoint_eval_backend") or "")
    if not backend:
        modal_config_path = Path(__file__).resolve().parents[2] / "experiments" / "modal_eval.yaml"
        backend = (
            "modal"
            if modal_config_path.is_file() and load_modal_eval_config(modal_config_path).enabled
            else "local"
        )
    if backend not in {"local", "modal"}:
        raise ValueError("checkpoint_eval_backend must be local or modal")
    train_config["checkpoint_eval_backend"] = backend
    document["train_config"] = train_config
    validate_materialized_train_recipe(document)
    goal_slug = recipe_goal_slug(document)
    document_slug = recipe_slug(document)
    machine = normalize_machine(machine)
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    utc = _utc_stamp()
    campaign_id = str(document.get("campaign_id") or "").strip() or None
    explicit_submission_key = submission_key is not None
    submission_key = str(submission_key or f"submission-{uuid.uuid4().hex}").strip()
    if not submission_key:
        raise ValueError("submission_key is required")
    batch_id = _submission_batch_id(submission_key)
    document_seeds = _document_seeds(document, seeds)
    request_hash = _submission_request_hash(
        document=document,
        machine=machine,
        runtime_image_ref=runtime_image_ref,
        seeds=document_seeds,
    )
    if explicit_submission_key:
        with conn:
            existing = _existing_submission(conn, submission_key=submission_key)
        if existing:
            if any(str(row["request_hash"]) != request_hash for row in existing):
                raise ValueError(
                    f"submission_key {submission_key!r} was reused with different content"
                )
            if len(existing) != len(document_seeds):
                raise RuntimeError(
                    f"submission_key {submission_key!r} is incomplete: "
                    f"expected {len(document_seeds)} jobs, found {len(existing)}"
                )
            return existing
    modal_readiness_validated = backend != "modal"
    if not modal_readiness_validated:
        require_modal_eval_ready(
            runtime_image_ref=runtime_image_ref,
            game=str(train_config.get("game") or ""),
        )
        modal_readiness_validated = True
    rows = []
    with conn:
        if explicit_submission_key:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%(key)s, 0))",
                    {"key": submission_key},
                )
                cur.execute(
                    "SELECT * FROM train_jobs WHERE submission_key = %(key)s ORDER BY submission_ordinal",
                    {"key": submission_key},
                )
                concurrent = [dict(row) for row in cur.fetchall()]
            if concurrent:
                if any(str(row["request_hash"]) != request_hash for row in concurrent):
                    raise ValueError(
                        f"submission_key {submission_key!r} was reused with different content"
                    )
                if len(concurrent) != len(document_seeds):
                    raise RuntimeError(f"submission_key {submission_key!r} is incomplete")
                return concurrent
        for ordinal, seed in enumerate(document_seeds):
            train_config = dict(document["train_config"])
            recipe_overrides = document.get("recipe_overrides")
            if isinstance(recipe_overrides, Sequence) and not isinstance(
                recipe_overrides, str | bytes
            ):
                train_config["recipe_overrides"] = [str(item) for item in recipe_overrides]
            if seed is not None:
                validate_training_seed(
                    seed,
                    label="recipe seed",
                    seed_span=provider_num_envs(
                        train_config, explicit_n_envs=train_config.get("n_envs")
                    ),
                )
            for row_owned_key in ("seed", "recipe_slug", "recipe_path", "machine"):
                train_config.pop(row_owned_key, None)
            row = enqueue_train_job(
                conn,
                goal_slug=goal_slug,
                recipe_slug=document_slug,
                recipe_path=recipe_path,
                recipe_sha256=recipe_sha256,
                repo_git_commit=repo_git_commit,
                repo_dirty=repo_dirty,
                recipe_payload=compiled_recipe_payload(document),
                runtime_image_ref=runtime_image_ref,
                machine=machine,
                train_config=train_config,
                batch_id=batch_id,
                campaign_id=campaign_id,
                submission_key=submission_key,
                submission_ordinal=ordinal,
                request_hash=request_hash,
                run_name=_format_default_run_name(
                    batch_id, label=document_slug, seed=seed, utc=utc
                ),
                run_description=_format_queue_template(
                    document.get("description"),
                    seed=seed,
                    recipe_id=document_slug,
                    utc=utc,
                    batch_id=batch_id,
                    campaign_id=campaign_id or "",
                ),
                seed=seed,
                wandb_group=batch_id,
                wandb_tags=recipe_tags(document),
                manage_transaction=False,
                _modal_readiness_validated=modal_readiness_validated,
                runtime_config_validator=runtime_config_validator,
            )
            rows.append(row)
    return rows


def enqueue_train_jobs_from_recipe_file(
    conn,
    *,
    path: Path,
    runtime_image_ref: str,
    machine: str,
    submission_key: str | None = None,
    seeds: Sequence[int] = (),
    recipe_overrides: Sequence[str] = (),
    checkpoint_eval_backend: str | None = None,
    runtime_config_validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> list[dict[str, Any]]:
    document = load_recipe_document(path, recipe_overrides=recipe_overrides)
    metadata = recipe_metadata(path, document)
    return enqueue_train_jobs_from_recipe_document(
        conn,
        document=document,
        runtime_image_ref=runtime_image_ref,
        machine=machine,
        submission_key=submission_key,
        recipe_path=metadata["recipe_path"],
        recipe_sha256=metadata["recipe_sha256"],
        repo_git_commit=metadata["repo_git_commit"],
        repo_dirty=metadata["repo_dirty"],
        seeds=seeds,
        checkpoint_eval_backend=checkpoint_eval_backend,
        runtime_config_validator=runtime_config_validator,
    )


def enqueue_train_job(
    conn,
    *,
    goal_slug: str,
    runtime_image_ref: str,
    machine: str,
    train_config: Mapping[str, Any],
    recipe_slug: str | None = None,
    recipe_path: str | None = None,
    recipe_sha256: str | None = None,
    repo_git_commit: str | None = None,
    repo_dirty: bool = False,
    recipe_payload: Mapping[str, Any] | None = None,
    batch_id: str | None = None,
    campaign_id: str | None = None,
    submission_key: str | None = None,
    submission_ordinal: int = 0,
    request_hash: str | None = None,
    retry_of_job_id: int | None = None,
    run_name: str | None = None,
    run_description: str | None = None,
    seed: int | None = None,
    wandb_group: str | None = None,
    wandb_tags: Sequence[str] = (),
    manage_transaction: bool = True,
    _modal_readiness_validated: bool = False,
    runtime_config_validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    goal_slug = str(goal_slug).strip()
    if not goal_slug:
        raise ValueError("goal_slug is required")
    submission_key = str(submission_key or f"submission-{uuid.uuid4().hex}").strip()
    batch_id = str(batch_id or _submission_batch_id(submission_key)).strip()
    campaign_id = str(campaign_id or "").strip() or None
    if submission_ordinal < 0:
        raise ValueError("submission_ordinal must be at least zero")
    requested_config = dict(train_config)
    seed = int(seed if seed is not None else requested_config.get("seed", DEFAULT_TRAIN_SEED))
    if "checkpoint_eval_backend" not in requested_config:
        modal_config_path = Path(__file__).resolve().parents[2] / "experiments" / "modal_eval.yaml"
        if modal_config_path.is_file() and load_modal_eval_config(modal_config_path).enabled:
            requested_config["checkpoint_eval_backend"] = "modal"
    config = validate_and_normalize_train_config(requested_config)
    config["batch_id"] = batch_id
    if campaign_id:
        config["campaign_id"] = campaign_id
    if retry_of_job_id is not None:
        config["retry_of_job_id"] = int(retry_of_job_id)
    config["wandb_run_id"] = (
        "rlab-"
        + _hash_json({"submission_key": submission_key, "submission_ordinal": submission_ordinal})[
            :24
        ]
    )
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    if str(config.get("checkpoint_eval_backend") or "local") == "modal":
        if not _modal_readiness_validated:
            require_modal_eval_ready(
                runtime_image_ref=runtime_image_ref,
                game=str(config.get("game") or ""),
            )
        if not config.get("checkpoint_eval_asset_manifest"):
            config["checkpoint_eval_asset_manifest"] = asset_manifest_for_game(
                str(config.get("game") or "")
            )
        config.setdefault("checkpoint_eval_seed_protocol", "vector-lane-v1")
    assert_no_secrets(config, label="train_config")
    assert_no_secrets(recipe_payload or {}, label="recipe_payload")
    require_explicit_queue_train_config(config)
    validate_launch_seed_config(config, seed=seed)
    validate_launch_event_config(config)
    machine = normalize_machine(machine)
    run_name = _validate_queue_run_name(
        str(
            run_name
            or _format_default_run_name(
                batch_id,
                label=str(recipe_slug or goal_slug),
                seed=seed,
                utc=_utc_stamp(),
            )
        ),
        batch_id=batch_id,
        label=str(recipe_slug or goal_slug),
        seed=seed,
    )
    wandb_group = batch_id
    normalized_tags = [str(tag).strip() for tag in wandb_tags if str(tag).strip()]
    if campaign_id and f"campaign_id:{campaign_id}" not in normalized_tags:
        normalized_tags.append(f"campaign_id:{campaign_id}")
    if retry_of_job_id is not None:
        normalized_tags = [
            tag for tag in normalized_tags if not tag.startswith("retry_of_job_id:")
        ]
        retry_tag = f"retry_of_job_id:{int(retry_of_job_id)}"
        if retry_tag not in normalized_tags:
            normalized_tags.append(retry_tag)
    if runtime_config_validator is not None:
        from rlab.wandb_utils import game_family_for_environment

        preflight_config = {
            **config,
            "game_family": game_family_for_environment(
                config.get("env_provider"), config.get("game")
            ),
            "goal_slug": goal_slug,
            "machine": machine,
            "queue_train_job_id": 1,
            "recipe_path": str(recipe_path or ""),
            "recipe_slug": str(recipe_slug or ""),
            "run_description": str(run_description or ""),
            "run_name": run_name,
            "runtime_image_ref": runtime_image_ref,
            "seed": seed,
            "wandb_group": wandb_group,
            "wandb_tags": ",".join(normalized_tags),
        }
        runtime_config_validator(preflight_config)
    request_hash = str(
        request_hash
        or _hash_json(
            {
                "goal_slug": goal_slug,
                "recipe_slug": recipe_slug,
                "runtime_image_ref": runtime_image_ref,
                "machine": machine,
                "train_config": config,
                "seed": seed,
            }
        )
    )

    def insert() -> dict[str, Any]:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO train_jobs (
                  goal_slug, recipe_slug, recipe_path, recipe_sha256, repo_git_commit,
                  repo_dirty, recipe_payload_json, runtime_image_ref, machine, train_config,
                  batch_id, campaign_id, submission_key, submission_ordinal, request_hash,
                  retry_of_job_id, retried_from_job_id, run_name, run_description, seed,
                  wandb_group, wandb_tags
                )
                VALUES (
                  %(goal_slug)s, %(recipe_slug)s, %(recipe_path)s, %(recipe_sha256)s,
                  %(repo_git_commit)s, %(repo_dirty)s, %(recipe_payload_json)s,
                  %(runtime_image_ref)s, %(machine)s, %(train_config)s,
                  %(batch_id)s, %(campaign_id)s, %(submission_key)s,
                  %(submission_ordinal)s, %(request_hash)s,
                  %(retry_of_job_id)s, %(retried_from_job_id)s, %(run_name)s,
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
                    "machine": machine,
                    "train_config": json_arg(config),
                    "batch_id": batch_id,
                    "campaign_id": campaign_id,
                    "submission_key": submission_key,
                    "submission_ordinal": submission_ordinal,
                    "request_hash": request_hash,
                    "retry_of_job_id": retry_of_job_id,
                    # Retain the legacy column for compatibility with older queue readers.
                    "retried_from_job_id": retry_of_job_id,
                    "run_name": run_name,
                    "run_description": run_description,
                    "seed": seed,
                    "wandb_group": wandb_group,
                    "wandb_tags": normalized_tags,
                },
            )
            row = dict(cur.fetchone())
            record_job_event(
                conn,
                job_id=int(row["id"]),
                event_type="enqueued",
                message="train job enqueued",
                metadata={
                    "goal_slug": goal_slug,
                    "recipe_slug": recipe_slug,
                    "machine": machine,
                    "batch_id": batch_id,
                    "campaign_id": campaign_id,
                },
            )
            return row

    if manage_transaction:
        with conn:
            return insert()
    return insert()


def new_train_launch_id(job_id: int | None = None) -> str:
    if job_id is None:
        raise ValueError("job_id is required for a stable launch identity")
    return f"{TRAIN_JOB_KIND}-{int(job_id)}"


def _machine_control_lock_key(machine: str) -> str:
    return f"rlab-fleet-machine-control:{normalize_machine(machine)}"


def acquire_machine_control_xact_lock(conn, *, machine: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_advisory_xact_lock(hashtextextended(%(key)s, 0))",
            {"key": _machine_control_lock_key(machine)},
        )


def claim_job_launch(
    conn,
    *,
    machine: str,
    backend: str,
    runtime_image_ref: str | None = None,
    job_id: int | None = None,
    launch_id: str | None = None,
    container_name: str | None = None,
    output_uri: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    runtime_image_ref = (
        normalize_runtime_image_ref(runtime_image_ref) if runtime_image_ref else None
    )
    machine = normalize_machine(machine)
    filters = [
        "job.cancel_requested = FALSE",
        "job.status = 'pending'",
        "job.machine = %(machine)s",
        "NOT EXISTS (SELECT 1 FROM job_launches existing WHERE existing.job_id = job.id)",
        "NOT EXISTS (SELECT 1 FROM machine_controls control "
        "WHERE control.machine = job.machine AND control.drained)",
    ]
    if job_id is None and launch_id is not None:
        raise ValueError("job_id is required when launch_id is provided")
    stable_launch_id = launch_id or (new_train_launch_id(job_id) if job_id is not None else None)
    params: dict[str, Any] = {
        "machine": machine,
        "backend": str(backend),
        "output_uri": str(output_uri),
        "launch_id": stable_launch_id,
        "container_name": container_name,
        "job_kind": TRAIN_JOB_KIND,
    }
    if job_id is not None:
        filters.append("job.id = %(job_id)s")
        params["job_id"] = int(job_id)
    if runtime_image_ref is not None:
        filters.append("job.runtime_image_ref = %(runtime_image_ref)s")
        params["runtime_image_ref"] = runtime_image_ref
    where = "\n    AND ".join(filters)
    with conn:
        acquire_machine_control_xact_lock(conn, machine=machine)
        with conn.cursor() as cur:
            cur.execute(
                f"""
                WITH next_job AS (
                  SELECT job.*
                  FROM train_jobs AS job
                  WHERE {where}
                  ORDER BY id ASC
                  LIMIT 1
                  FOR UPDATE SKIP LOCKED
                ),
                updated AS (
                  UPDATE train_jobs AS job
                  SET status = 'launching',
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
                    COALESCE(%(launch_id)s, 'train-' || updated.id::text),
                    %(job_kind)s, updated.id, %(backend)s, %(machine)s,
                    updated.runtime_image_ref,
                    COALESCE(%(container_name)s, 'rlab-train-' || updated.id::text),
                    %(output_uri)s,
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
                    last_observed_at = now(),
                    error = NULL,
                    next_retry_at = NULL
                WHERE launch_id = %(launch_id)s
                  AND state IN ('launching', 'running')
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
                SET status = 'starting',
                    started_at = COALESCE(started_at, now())
                WHERE id = %(job_id)s
                  AND status = 'launching'
                RETURNING id
                """,
                {"job_id": launch["job_id"], "launch_id": launch_id},
            )
            job = cur.fetchone()
            if not job:
                raise RuntimeError(f"job for launch {launch_id} is not launching")
            record_job_event(
                conn,
                job_id=int(launch["job_id"]),
                event_type="starting",
                message="job container started",
                metadata={"launch_id": launch_id},
            )
            return dict(launch)


def mark_train_job_ready(
    conn,
    *,
    launch_id: str,
    readiness: Mapping[str, Any],
) -> dict[str, Any] | None:
    wandb_run_id = str(readiness.get("wandb_run_id") or "").strip()
    wandb_url = str(readiness.get("wandb_url") or "").strip()
    if not wandb_run_id or not wandb_url.startswith("https://wandb.ai/"):
        raise ValueError("training readiness requires a W&B run id and URL")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs AS job
                SET status = 'running',
                    ready_at = COALESCE(ready_at, now()),
                    wandb_run_id = %(wandb_run_id)s,
                    wandb_url = %(wandb_url)s,
                    error = NULL
                FROM job_launches AS launch
                WHERE launch.launch_id = %(launch_id)s
                  AND launch.job_id = job.id
                  AND launch.state = 'running'
                  AND job.status = 'starting'
                RETURNING job.*
                """,
                {
                    "launch_id": launch_id,
                    "wandb_run_id": wandb_run_id,
                    "wandb_url": wandb_url,
                },
            )
            row = cur.fetchone()
            if not row:
                return None
            job = dict(row)
            record_job_event(
                conn,
                job_id=int(job["id"]),
                event_type="running",
                message="learner and W&B publisher ready",
                metadata={
                    "launch_id": launch_id,
                    "wandb_run_id": wandb_run_id,
                    "wandb_url": wandb_url,
                },
            )
            return job


def record_job_launch_error(
    conn,
    *,
    launch_id: str,
    error: str,
    retry_after_seconds: float = 30.0,
) -> dict[str, Any] | None:
    """Record an observation/control error without releasing the stable launch."""
    retry_at = datetime.now(UTC) + timedelta(seconds=max(0.0, retry_after_seconds))
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE job_launches AS launch
                SET error = %(error)s,
                    next_retry_at = %(next_retry_at)s,
                    last_observed_at = now()
                WHERE launch_id = %(launch_id)s
                  AND state IN ('launching', 'running')
                RETURNING *
                """,
                {"launch_id": launch_id, "error": error, "next_retry_at": retry_at},
            )
            launch = cur.fetchone()
            if not launch:
                return None
            cur.execute(
                """
                UPDATE train_jobs
                SET error = %(error)s
                WHERE id = %(job_id)s
                  AND status IN ('launching', 'starting', 'running')
                """,
                {"job_id": launch["job_id"], "error": error},
            )
            record_job_event(
                conn,
                job_id=int(launch["job_id"]),
                event_type="control_error",
                message=error,
                metadata={"launch_id": launch_id, "next_retry_at": retry_at.isoformat()},
            )
            return dict(launch)


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


def machine_control(conn, *, machine: str) -> dict[str, Any]:
    machine = normalize_machine(machine)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT machine, drained, effective_capacity, reason, updated_at
            FROM machine_controls
            WHERE machine = %(machine)s
            """,
            {"machine": machine},
        )
        row = cur.fetchone()
    return (
        dict(row)
        if row
        else {
            "machine": machine,
            "drained": False,
            "effective_capacity": None,
            "reason": None,
        }
    )


def set_machine_control(
    conn,
    *,
    machine: str,
    drained: bool | None = None,
    effective_capacity: int | None = None,
    reset_capacity: bool = False,
    reason: str | None = None,
    manage_transaction: bool = True,
) -> dict[str, Any]:
    machine = normalize_machine(machine)
    if effective_capacity is not None and effective_capacity < 1:
        raise ValueError("effective_capacity must be at least one")

    def update() -> dict[str, Any]:
        acquire_machine_control_xact_lock(conn, machine=machine)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO machine_controls (machine, drained, effective_capacity, reason)
                VALUES (
                  %(machine)s,
                  COALESCE(%(drained)s, FALSE),
                  %(effective_capacity)s,
                  %(reason)s
                )
                ON CONFLICT (machine) DO UPDATE
                SET drained = COALESCE(%(drained)s, machine_controls.drained),
                    effective_capacity = CASE
                      WHEN %(reset_capacity)s THEN NULL
                      WHEN %(effective_capacity)s IS NOT NULL THEN %(effective_capacity)s
                      ELSE machine_controls.effective_capacity
                    END,
                    reason = COALESCE(%(reason)s, machine_controls.reason),
                    updated_at = now()
                RETURNING *
                """,
                {
                    "machine": machine,
                    "drained": drained,
                    "effective_capacity": effective_capacity,
                    "reset_capacity": reset_capacity,
                    "reason": reason,
                },
            )
            return dict(cur.fetchone())

    if manage_transaction:
        with conn:
            return update()
    return update()


def machines_with_service_work(conn=None) -> tuple[str, ...]:
    owned_connection = conn is None
    if owned_connection:
        conn = connect(database_url())
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT machine
                FROM train_jobs
                WHERE status IN ('pending', 'launching', 'starting', 'running')
                   OR cancel_requested = TRUE
                ORDER BY machine
                """
            )
            return tuple(str(row["machine"]) for row in cur.fetchall())
    finally:
        if owned_connection:
            conn.close()


def count_nonterminal_jobs(conn=None) -> int:
    owned_connection = conn is None
    if owned_connection:
        conn = connect(database_url())
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM train_jobs
                   WHERE status IN ('pending', 'launching', 'starting', 'running'))
                  +
                  (SELECT COUNT(*) FROM eval_jobs
                   WHERE status IN ('pending', 'dispatching', 'submitted', 'blocked_budget'))
                  AS count
                """
            )
            row = cur.fetchone()
        return int(row["count"] if row else 0)
    finally:
        if owned_connection:
            conn.close()


def next_pending_train_job(conn, *, machine: str) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT job.*
            FROM train_jobs AS job
            LEFT JOIN machine_controls AS control ON control.machine = job.machine
            WHERE job.machine = %(machine)s
              AND job.status = 'pending'
              AND job.cancel_requested = FALSE
              AND COALESCE(control.drained, FALSE) = FALSE
              AND NOT EXISTS (
                SELECT 1 FROM job_launches AS launch WHERE launch.job_id = job.id
              )
            ORDER BY job.id
            LIMIT 1
            """,
            {"machine": normalize_machine(machine)},
        )
        row = cur.fetchone()
    return dict(row) if row else None


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
                    status = CASE WHEN status = 'pending' THEN 'canceled' ELSE status END,
                    finished_at = CASE WHEN status = 'pending' THEN now() ELSE finished_at END
                WHERE id = %(job_id)s
                  AND status IN ('pending', 'launching', 'starting', 'running')
                """,
                {"job_id": job_id},
            )
            changed = int(cur.rowcount)
        if changed:
            record_job_event(
                conn,
                job_id=job_id,
                event_type="cancel_requested",
                message="operator requested cancellation",
            )
        return changed


def request_cancel_train_jobs(
    conn,
    *,
    job_id: int | None = None,
    batch_id: str | None = None,
    machine: str | None = None,
    drain: bool = False,
) -> list[int]:
    selectors = sum(value is not None for value in (job_id, batch_id, machine))
    if selectors != 1:
        raise ValueError("exactly one of job_id, batch_id, or machine is required")
    filters = ["status IN ('pending', 'launching', 'starting', 'running')"]
    params: dict[str, Any] = {}
    if job_id is not None:
        filters.append("id = %(job_id)s")
        params["job_id"] = int(job_id)
    elif batch_id is not None:
        filters.append("batch_id = %(batch_id)s")
        params["batch_id"] = str(batch_id)
    else:
        params["machine"] = normalize_machine(machine)
        filters.append("machine = %(machine)s")
    with conn:
        if machine is not None:
            acquire_machine_control_xact_lock(conn, machine=machine)
        if drain:
            if machine is None:
                raise ValueError("drain is only valid with a machine selector")
            set_machine_control(
                conn,
                machine=machine,
                drained=True,
                reason="cancel all active",
                manage_transaction=False,
            )
        with conn.cursor() as cur:
            cur.execute(
                f"""
                UPDATE train_jobs
                SET cancel_requested = TRUE,
                    status = CASE WHEN status = 'pending' THEN 'canceled' ELSE status END,
                    finished_at = CASE WHEN status = 'pending' THEN now() ELSE finished_at END
                WHERE {" AND ".join(filters)}
                RETURNING id
                """,
                params,
            )
            job_ids = [int(row["id"]) for row in cur.fetchall()]
        for canceled_job_id in job_ids:
            record_job_event(
                conn,
                job_id=canceled_job_id,
                event_type="cancel_requested",
                message="operator requested cancellation",
                metadata={"drain": bool(drain)},
            )
        return job_ids


def retry_train_job(
    conn,
    *,
    job_id: int,
    submission_key: str | None = None,
    runtime_image_ref: str | None = None,
    repo_git_commit: str | None = None,
    runtime_config_validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Create a new pending job from a terminal job; execution is never retried in place."""
    submission_key = str(submission_key or f"retry-{job_id}-{uuid.uuid4().hex}")
    with conn:
        existing = _existing_submission(conn, submission_key=submission_key)
        if existing:
            retry_source = existing[0].get("retry_of_job_id") or existing[0].get(
                "retried_from_job_id"
            )
            if len(existing) != 1 or retry_source != int(job_id):
                raise ValueError(
                    f"submission_key {submission_key!r} was reused for a different retry"
                )
            return existing[0]
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM train_jobs
                WHERE id = %(job_id)s
                  AND status IN ('succeeded', 'failed', 'canceled')
                """,
                {"job_id": int(job_id)},
            )
            preview = cur.fetchone()
    if not preview:
        raise ValueError(f"job {job_id} is not terminal or does not exist")
    preview = dict(preview)
    preview_config = dict(preview.get("train_config") or {})
    effective_runtime_image_ref = str(
        runtime_image_ref or preview["runtime_image_ref"]
    )
    modal_readiness_validated = (
        str(preview_config.get("checkpoint_eval_backend") or "local") != "modal"
    )
    if not modal_readiness_validated:
        require_modal_eval_ready(
            runtime_image_ref=effective_runtime_image_ref,
            game=str(preview_config.get("game") or ""),
        )
        modal_readiness_validated = True
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%(key)s, 0))",
                {"key": submission_key},
            )
            cur.execute(
                "SELECT * FROM train_jobs WHERE submission_key = %(key)s ORDER BY submission_ordinal",
                {"key": submission_key},
            )
            existing = [dict(row) for row in cur.fetchall()]
            if existing:
                retry_source = existing[0].get("retry_of_job_id") or existing[0].get(
                    "retried_from_job_id"
                )
                if len(existing) != 1 or retry_source != int(job_id):
                    raise ValueError(
                        f"submission_key {submission_key!r} was reused for a different retry"
                    )
                return existing[0]
            cur.execute(
                """
                SELECT *
                FROM train_jobs
                WHERE id = %(job_id)s
                  AND status IN ('succeeded', 'failed', 'canceled')
                FOR UPDATE
                """,
                {"job_id": int(job_id)},
            )
            source = cur.fetchone()
            if not source:
                raise ValueError(f"job {job_id} is not terminal or does not exist")
        source = dict(source)
        result = enqueue_train_job(
            conn,
            goal_slug=source["goal_slug"],
            recipe_slug=source.get("recipe_slug"),
            recipe_path=source.get("recipe_path"),
            recipe_sha256=source.get("recipe_sha256"),
            repo_git_commit=repo_git_commit or source.get("repo_git_commit"),
            repo_dirty=False if repo_git_commit else bool(source.get("repo_dirty")),
            recipe_payload=source.get("recipe_payload_json") or {},
            runtime_image_ref=runtime_image_ref or source["runtime_image_ref"],
            machine=source["machine"],
            train_config=source["train_config"],
            batch_id=source["batch_id"],
            campaign_id=source.get("campaign_id"),
            submission_key=submission_key,
            request_hash=_hash_json({"retry_of": int(job_id), "submission_key": submission_key}),
            retry_of_job_id=int(job_id),
            run_description=source.get("run_description"),
            seed=source.get("seed"),
            wandb_tags=source.get("wandb_tags") or (),
            manage_transaction=False,
            _modal_readiness_validated=modal_readiness_validated,
            runtime_config_validator=runtime_config_validator,
        )
        record_job_event(
            conn,
            job_id=int(result["id"]),
            event_type="retried",
            message=f"explicit retry of job {job_id}",
            metadata={"retry_of_job_id": int(job_id)},
        )
        return result


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
    exit_code = result.get("exit_code")
    error = str(result.get("error") or "") or None
    train_result = result.get("train")
    train_payload = train_result.get("result") if isinstance(train_result, Mapping) else {}
    train_payload = dict(train_payload or {})
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  launch.*,
                  job.cancel_requested,
                  job.status AS job_status,
                  job.machine AS job_machine
                FROM job_launches AS launch
                JOIN train_jobs AS job ON job.id = launch.job_id
                WHERE launch.launch_id = %(launch_id)s
                FOR UPDATE OF launch, job
                """,
                {"launch_id": launch_id},
            )
            launch = cur.fetchone()
            if not launch:
                raise RuntimeError(f"unknown launch_id {launch_id}")
            expected = {
                "schema_version": 1,
                "job_id": int(launch["job_id"]),
                "launch_id": launch_id,
                "machine": str(launch["machine"]),
                "runtime_image_ref": str(launch["runtime_image_ref"]),
            }
            for field, expected_value in expected.items():
                actual = result.get(field)
                if actual != expected_value:
                    raise ValueError(
                        f"result {field} mismatch for {launch_id}: "
                        f"expected {expected_value!r}, got {actual!r}"
                    )
            status = (
                "canceled"
                if bool(launch["cancel_requested"])
                else _terminal_status_from_result(result)
            )
            if launch["state"] in {"succeeded", "failed", "canceled"}:
                if launch["state"] == status:
                    return
                raise RuntimeError(f"launch {launch_id} is already terminal as {launch['state']}")
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
                  AND state IN ('launching', 'running')
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
            updated_launch = cur.fetchone()
            if not updated_launch:
                raise RuntimeError(f"launch {launch_id} is already terminal")
            if updated_launch["job_kind"] != "train":
                raise RuntimeError(f"launch {launch_id} is not a train launch")
            cur.execute(
                """
                UPDATE train_jobs
                SET status = %(status)s,
                    finished_at = now(),
                    wandb_run_id = COALESCE(%(wandb_run_id)s, wandb_run_id),
                    wandb_url = COALESCE(%(wandb_url)s, wandb_url),
                    error = %(error)s
                WHERE id = %(job_id)s
                  AND status IN ('launching', 'starting', 'running')
                RETURNING *
                """,
                {
                    "status": status,
                    "error": error,
                    "job_id": updated_launch["job_id"],
                    "launch_id": launch_id,
                    "wandb_run_id": train_payload.get("wandb_run_id"),
                    "wandb_url": train_payload.get("wandb_url"),
                },
            )
            job = cur.fetchone()
            if not job:
                raise RuntimeError(f"could not finish train job for launch {launch_id}")
            if (
                str(result.get("checkpoint_coordinator_status") or "")
                == "awaiting_artifact_recovery"
            ):
                cur.execute(
                    """
                    UPDATE eval_runs SET status = 'awaiting_artifact_recovery',
                      error = 'checkpoint coordinator drain ended with incomplete uploads',
                      updated_at = now()
                    WHERE train_job_id = %(job_id)s AND status <> 'complete'
                    """,
                    {"job_id": updated_launch["job_id"]},
                )
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


def queue_status(
    conn,
    *,
    job_id: int | None = None,
    batch_id: str | None = None,
    machine: str | None = None,
    goal_slug: str | None = None,
    machine_capacities: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    selectors = sum(value is not None for value in (job_id, batch_id, machine, goal_slug))
    if selectors != 1:
        raise ValueError("exactly one job, batch, machine, or goal selector is required")
    params: dict[str, Any] = {}
    if job_id is not None:
        where = "job.id = %(job_id)s"
        params["job_id"] = int(job_id)
        selector = {"job_id": int(job_id)}
    elif batch_id is not None:
        where = "job.batch_id = %(batch_id)s"
        params["batch_id"] = str(batch_id)
        selector = {"batch_id": str(batch_id)}
    elif machine is not None:
        where = "job.machine = %(machine)s"
        params["machine"] = normalize_machine(machine)
        selector = {"machine": params["machine"]}
    else:
        where = "job.goal_slug = %(goal_slug)s"
        params["goal_slug"] = str(goal_slug).strip()
        selector = {"goal_slug": params["goal_slug"]}
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT
              job.*,
              launch.launch_id,
              launch.container_name,
              launch.state AS launch_state,
              launch.error AS launch_error,
              launch.next_retry_at,
              launch.output_uri,
              launch.last_observed_at,
              COALESCE(control.drained, FALSE) AS machine_drained,
              control.effective_capacity,
              (
                SELECT COUNT(*)
                FROM job_launches AS active_launch
                WHERE active_launch.machine = job.machine
                  AND active_launch.state IN ('launching', 'running')
              ) AS active_reservations
            FROM train_jobs AS job
            LEFT JOIN job_launches AS launch ON launch.job_id = job.id
            LEFT JOIN machine_controls AS control ON control.machine = job.machine
            WHERE {where}
            ORDER BY job.id
            """,
            params,
        )
        jobs = [dict(row) for row in cur.fetchall()]
    capacities = dict(machine_capacities or {})
    unreachable_terms = ("unreachable", "timed out", "permission denied", "connection")
    for row in jobs:
        blocked_reason = None
        if row.get("cancel_requested") and row.get("status") in {"launching", "starting", "running"}:
            blocked_reason = "canceling"
        elif row.get("status") == "pending" and row.get("machine_drained"):
            blocked_reason = "drained"
        elif row.get("status") == "pending":
            hard_capacity = capacities.get(str(row["machine"]))
            override = row.get("effective_capacity")
            effective = (
                min(hard_capacity, int(override))
                if hard_capacity and override
                else (int(override) if override else hard_capacity)
            )
            if effective is not None and int(row.get("active_reservations") or 0) >= effective:
                blocked_reason = "at_capacity"
        launch_error = str(row.get("launch_error") or "").lower()
        if (
            blocked_reason is None
            and row.get("status") in {"launching", "starting", "running"}
            and any(term in launch_error for term in unreachable_terms)
        ):
            blocked_reason = "unreachable"
        row["blocked_reason"] = blocked_reason
    counts: dict[str, int] = {}
    for row in jobs:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    return {"selector": selector, "counts": counts, "jobs": jobs}


def print_status(report: Mapping[str, Any]) -> None:
    print(f"selector: {json.dumps(report['selector'], sort_keys=True)}")
    print(f"counts: {json.dumps(report['counts'], sort_keys=True)}")
    print("jobs:")
    for row in report.get("jobs", []):
        print(
            "  "
            f"job={row['id']} machine={row['machine']} status={row['status']} "
            f"image={row.get('runtime_image_ref') or ''} "
            f"run={row.get('run_name') or ''}"
        )


def build_train_enqueue_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab train",
        description="Create queue-backed train jobs from a checked-in recipe file.",
    )
    add_direct_database_arg(parser)
    parser.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    parser.add_argument("--recipe-file", dest="recipe_file", type=Path, required=True)
    parser.add_argument("--machine", required=True, help="Exact registered machine name.")
    parser.add_argument("--request-id", dest="submission_key")
    parser.add_argument("--runtime-image-ref")
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
        "--checkpoint-eval-backend",
        choices=("local", "modal"),
        default=None,
        help=(
            "Materialize the checkpoint evaluation backend for this submission; "
            "defaults to the checked-in Modal rollout policy."
        ),
    )
    parser.add_argument("--wait", choices=("running", "terminal"))
    parser.add_argument("--timeout", type=parse_duration_seconds, default=12 * 60 * 60)
    parser.add_argument("--json", action="store_true")
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

    status = subparsers.add_parser("status", help="Inspect jobs by job, batch, machine, or goal.")
    add_job_selector(status, include_machine=True, include_goal=True)
    status.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    wait = subparsers.add_parser("wait", help="Wait for jobs to run or become terminal.")
    add_job_selector(wait, include_machine=False, include_goal=False)
    wait.add_argument("--until", choices=("running", "terminal"), required=True)
    wait.add_argument("--timeout", type=parse_duration_seconds, default=12 * 60 * 60)
    wait.add_argument("--json", action="store_true")
    wait.set_defaults(func=cmd_wait)

    cancel = subparsers.add_parser("cancel", help="Request idempotent job cancellation.")
    add_job_selector(cancel, include_machine=True, include_goal=False)
    cancel.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    cancel.add_argument("--all-active", action="store_true")
    cancel.add_argument("--drain", action="store_true")
    cancel.add_argument("--wait", action="store_true")
    cancel.add_argument("--timeout", type=parse_duration_seconds, default=10 * 60)
    cancel.add_argument("--json", action="store_true")
    cancel.set_defaults(func=cmd_cancel)

    retry = subparsers.add_parser("retry", help="Create a new job from a terminal job.")
    retry.add_argument("--job", dest="job_id", type=int, required=True)
    retry.add_argument("--request-id", dest="submission_key")
    retry.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    retry.add_argument("--runtime-image-ref")
    retry.add_argument("--runtime-image-ref-file", type=Path)
    retry.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
    retry.add_argument("--image-branch", default=DEFAULT_IMAGE_BRANCH)
    retry.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)
    retry.add_argument("--wait", choices=("running", "terminal"))
    retry.add_argument("--timeout", type=parse_duration_seconds, default=12 * 60 * 60)
    retry.add_argument("--json", action="store_true")
    retry.set_defaults(func=cmd_retry)

    logs = subparsers.add_parser("logs", help="Read durable output logs for one job.")
    logs.add_argument("--job", dest="job_id", type=int, required=True)
    logs.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    logs.add_argument("--tail", type=int, default=100)
    logs.add_argument("--follow", action="store_true")
    logs.set_defaults(func=cmd_logs)
    return parser


def parse_duration_seconds(value: str | int | float) -> float:
    if isinstance(value, int | float):
        return float(value)
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([smh]?)\s*", str(value))
    if not match:
        raise argparse.ArgumentTypeError(
            "duration must use seconds, m, or h (for example 30s or 12h)"
        )
    amount = float(match.group(1))
    multiplier = {"": 1.0, "s": 1.0, "m": 60.0, "h": 3600.0}[match.group(2)]
    return amount * multiplier


def add_job_selector(
    parser: argparse.ArgumentParser,
    *,
    include_machine: bool,
    include_goal: bool,
) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--job", dest="job_id", type=int)
    group.add_argument("--batch", dest="batch_id")
    if include_machine:
        group.add_argument("--machine")
    if include_goal:
        group.add_argument("--goal", dest="goal_slug")


def selector_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: getattr(args, key, None)
        for key in ("job_id", "batch_id", "machine", "goal_slug")
        if getattr(args, key, None) is not None
    }


def _connect_from_args(args: argparse.Namespace):
    return connect(database_url(args.direct))


def _machine_capacities(path: Path = DEFAULT_MACHINE_REGISTRY) -> dict[str, int]:
    registry = load_machine_registry(path)
    return {
        name: machine.limits.max_parallel_containers for name, machine in registry.machines.items()
    }


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
    from rlab.fleet_service import default_service_paths, schema_change_service_guard

    with schema_change_service_guard(default_service_paths()):
        conn = _connect_from_args(args)
        try:
            exported = reset_schema(conn, export_dir=export_dir)
        finally:
            conn.close()
    print(f"queue_schema_reset=ok export_dir={exported}")
    return 0


def cmd_enqueue_train(args: argparse.Namespace) -> int:
    registry = load_machine_registry(args.machines)
    machine_config = resolve_machine(registry, args.machine)
    release = runtime_release_from_args(args)
    runtime_image_ref = release.runtime_image_ref
    from rlab.docker_host import DockerRunnerHost

    host = DockerRunnerHost(machine_config)

    def validate_runtime_config(train_config: Mapping[str, Any]) -> dict[str, Any]:
        return host.validate_runtime_train_config(
            runtime_image_ref=runtime_image_ref,
            train_config=train_config,
            expected_source_sha=release.source_sha,
            expected_contract_sha256=release.train_config_contract_sha256,
        )

    conn = _connect_from_args(args)
    try:
        rows = enqueue_train_jobs_from_recipe_file(
            conn,
            path=args.recipe_file,
            runtime_image_ref=runtime_image_ref,
            machine=args.machine,
            submission_key=args.submission_key,
            seeds=args.seed,
            recipe_overrides=args.recipe_overrides,
            checkpoint_eval_backend=args.checkpoint_eval_backend,
            runtime_config_validator=validate_runtime_config,
        )
        dispatch = dispatch_fleet_service()
        wait_result = None
        if args.wait:
            wait_result = wait_for_job_ids(
                conn,
                [int(row["id"]) for row in rows],
                until=args.wait,
                timeout=float(args.timeout),
            )
        report = queue_status(
            conn,
            batch_id=str(rows[0]["batch_id"]),
            machine_capacities={
                name: machine.limits.max_parallel_containers
                for name, machine in registry.machines.items()
            },
        )
    finally:
        conn.close()
    payload = {
        "batch_id": rows[0]["batch_id"] if rows else None,
        "job_ids": [int(row["id"]) for row in rows],
        "machine": args.machine,
        "runtime_image_ref": runtime_image_ref,
        "dispatch": dispatch,
        "jobs": report["jobs"],
        "wait": wait_result,
    }
    if args.json:
        print(json.dumps(json_safe(payload), sort_keys=True))
    else:
        print(
            f"batch={payload['batch_id']} jobs={','.join(map(str, payload['job_ids']))} "
            f"machine={args.machine} dispatch={dispatch}"
        )
        if wait_result:
            print(json.dumps(json_safe(wait_result), sort_keys=True))
    return 0 if not wait_result or wait_result["reached"] else 1


def cmd_status(args: argparse.Namespace) -> int:
    capacities = _machine_capacities(args.machines)
    if args.machine:
        resolve_machine(load_machine_registry(args.machines), args.machine)
    conn = _connect_from_args(args)
    try:
        report = queue_status(
            conn,
            **selector_from_args(args),
            machine_capacities=capacities,
        )
    finally:
        conn.close()
    if args.json:
        print(json.dumps(json_safe(report), sort_keys=True))
    else:
        print_status(report)
    return 0


def dispatch_fleet_service() -> str:
    try:
        from rlab.fleet_service import kick_service

        return "kicked" if kick_service() else "degraded"
    except Exception:
        return "degraded"


def _job_ids_for_selector(
    conn, *, job_id: int | None = None, batch_id: str | None = None
) -> list[int]:
    if (job_id is None) == (batch_id is None):
        raise ValueError("exactly one job or batch selector is required")
    with conn.cursor() as cur:
        if job_id is not None:
            cur.execute("SELECT id FROM train_jobs WHERE id = %(job_id)s", {"job_id": job_id})
        else:
            cur.execute(
                "SELECT id FROM train_jobs WHERE batch_id = %(batch_id)s ORDER BY id",
                {"batch_id": batch_id},
            )
        return [int(row["id"]) for row in cur.fetchall()]


def wait_for_job_ids(
    conn,
    job_ids: Sequence[int],
    *,
    until: str,
    timeout: float,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    ids = tuple(sorted({int(job_id) for job_id in job_ids}))
    if not ids:
        raise ValueError("no jobs matched")
    deadline = time.monotonic() + max(0.0, timeout)
    terminal = {"succeeded", "failed", "canceled"}
    while True:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, status, started_at, finished_at, error
                FROM train_jobs
                WHERE id = ANY(%(job_ids)s)
                ORDER BY id
                """,
                {"job_ids": list(ids)},
            )
            rows = [dict(row) for row in cur.fetchall()]
        if len(rows) != len(ids):
            raise ValueError("one or more jobs no longer exist")
        statuses = {str(row["status"]) for row in rows}
        if until == "terminal":
            reached = all(status in terminal for status in statuses)
            terminal_before_target = False
        elif until == "running":
            reached = all(row["status"] == "running" for row in rows)
            terminal_before_target = any(row["status"] in terminal for row in rows)
        else:
            raise ValueError(f"unsupported wait target: {until}")
        if reached or terminal_before_target:
            return {
                "until": until,
                "reached": reached,
                "timed_out": False,
                "terminal_before_target": terminal_before_target,
                "jobs": rows,
            }
        if time.monotonic() >= deadline:
            return {
                "until": until,
                "reached": False,
                "timed_out": True,
                "terminal_before_target": False,
                "jobs": rows,
            }
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))


def cmd_wait(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        ids = _job_ids_for_selector(conn, job_id=args.job_id, batch_id=args.batch_id)
        result = wait_for_job_ids(conn, ids, until=args.until, timeout=float(args.timeout))
    finally:
        conn.close()
    if args.json:
        print(json.dumps(json_safe(result), sort_keys=True))
    else:
        print(json.dumps(json_safe(result), sort_keys=True))
    return 0 if result["reached"] else 1


def cmd_cancel(args: argparse.Namespace) -> int:
    if args.machine and not args.all_active:
        raise SystemExit("--machine cancellation requires --all-active")
    if args.all_active and not args.machine:
        raise SystemExit("--all-active requires --machine")
    if args.drain and not args.machine:
        raise SystemExit("--drain requires --machine")
    capacities = _machine_capacities(args.machines)
    if args.machine:
        resolve_machine(load_machine_registry(args.machines), args.machine)
    conn = _connect_from_args(args)
    try:
        ids = request_cancel_train_jobs(
            conn,
            job_id=args.job_id,
            batch_id=args.batch_id,
            machine=args.machine,
            drain=bool(args.drain),
        )
        dispatch = dispatch_fleet_service()
        wait_result = (
            wait_for_job_ids(conn, ids, until="terminal", timeout=float(args.timeout))
            if args.wait and ids
            else None
        )
        reports = [
            queue_status(conn, job_id=job_id, machine_capacities=capacities)["jobs"][0]
            for job_id in ids
        ]
    finally:
        conn.close()
    payload = {"job_ids": ids, "dispatch": dispatch, "jobs": reports, "wait": wait_result}
    print(json.dumps(json_safe(payload), sort_keys=True))
    return 0 if not wait_result or wait_result["reached"] else 1


def cmd_retry(args: argparse.Namespace) -> int:
    release = runtime_release_from_args(args)
    registry = load_machine_registry(args.machines)
    capacities = {
        name: machine.limits.max_parallel_containers
        for name, machine in registry.machines.items()
    }
    conn = _connect_from_args(args)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT machine FROM train_jobs WHERE id = %(job_id)s",
                {"job_id": args.job_id},
            )
            source = cur.fetchone()
        if not source:
            raise ValueError(f"job {args.job_id} does not exist")
        machine_config = resolve_machine(registry, str(source["machine"]))
        from rlab.docker_host import DockerRunnerHost

        host = DockerRunnerHost(machine_config)

        def validate_runtime_config(train_config: Mapping[str, Any]) -> dict[str, Any]:
            return host.validate_runtime_train_config(
                runtime_image_ref=release.runtime_image_ref,
                train_config=train_config,
                expected_source_sha=release.source_sha,
                expected_contract_sha256=release.train_config_contract_sha256,
            )

        row = retry_train_job(
            conn,
            job_id=args.job_id,
            submission_key=args.submission_key,
            runtime_image_ref=release.runtime_image_ref,
            repo_git_commit=release.source_sha,
            runtime_config_validator=validate_runtime_config,
        )
        dispatch = dispatch_fleet_service()
        wait_result = (
            wait_for_job_ids(
                conn,
                [int(row["id"])],
                until=args.wait,
                timeout=float(args.timeout),
            )
            if args.wait
            else None
        )
        report = queue_status(
            conn,
            job_id=int(row["id"]),
            machine_capacities=capacities,
        )
    finally:
        conn.close()
    payload = {
        "job_id": int(row["id"]),
        "retried_from": args.job_id,
        "batch_id": row["batch_id"],
        "machine": row["machine"],
        "runtime_image_ref": row["runtime_image_ref"],
        "dispatch": dispatch,
        "jobs": report["jobs"],
        "wait": wait_result,
    }
    print(json.dumps(json_safe(payload), sort_keys=True))
    return 0 if not wait_result or wait_result["reached"] else 1


def cmd_logs(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job.machine, launch.output_uri
                FROM train_jobs AS job
                JOIN job_launches AS launch ON launch.job_id = job.id
                WHERE job.id = %(job_id)s
                """,
                {"job_id": args.job_id},
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if not row:
        raise SystemExit(f"job {args.job_id} has no launch output")
    from rlab.docker_host import DockerRunnerHost

    machine = resolve_machine(load_machine_registry(args.machines), str(row["machine"]))
    return DockerRunnerHost(machine).stream_logs(
        str(row["output_uri"]),
        tail=int(args.tail),
        follow=bool(args.follow),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
