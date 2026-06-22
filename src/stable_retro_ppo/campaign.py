from __future__ import annotations

import argparse
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psycopg2
import psycopg2.extras


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
  profile_id TEXT NOT NULL,
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
  profile_id TEXT NOT NULL,
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

CREATE INDEX IF NOT EXISTS train_jobs_goal_status_idx
  ON train_jobs (goal_id, status);

CREATE INDEX IF NOT EXISTS campaign_decisions_goal_created_idx
  ON campaign_decisions (goal_id, created_at DESC);
"""


CLAIM_TRAIN_JOB_SQL = """
WITH next_job AS (
  SELECT id
  FROM train_jobs
  WHERE
    profile_id = %(profile_id)s
    AND cancel_requested = FALSE
    AND (
      status = 'pending'
      OR (
        status = 'running'
        AND lease_expires_at < now()
        AND attempts < max_attempts
      )
    )
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
                  train_config, priority
                )
                VALUES (
                  %(goal_id)s, %(slug)s, %(hypothesis)s, %(expected_signal)s,
                  %(parent_spec_id)s, %(train_config)s, %(priority)s
                )
                ON CONFLICT (goal_id, slug) DO UPDATE SET
                  hypothesis = EXCLUDED.hypothesis,
                  expected_signal = EXCLUDED.expected_signal,
                  parent_spec_id = EXCLUDED.parent_spec_id,
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
    profile_id: str,
    train_config: Mapping[str, Any],
    priority: int = 0,
    max_attempts: int = 1,
    run_name: str | None = None,
    run_description: str | None = None,
    wandb_group: str | None = None,
    wandb_tags: Sequence[str] = (),
) -> dict[str, Any]:
    config = dict(train_config)
    assert_no_secrets(config, label="train_config")
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO train_jobs (
                  goal_id, experiment_spec_id, profile_id, train_config, priority,
                  max_attempts, run_name, run_description, wandb_group, wandb_tags
                )
                VALUES (
                  %(goal_id)s, %(experiment_spec_id)s, %(profile_id)s,
                  %(train_config)s, %(priority)s, %(max_attempts)s, %(run_name)s,
                  %(run_description)s, %(wandb_group)s, %(wandb_tags)s
                )
                RETURNING *
                """,
                {
                    "goal_id": goal_id,
                    "experiment_spec_id": experiment_spec_id,
                    "profile_id": profile_id,
                    "train_config": json_arg(config),
                    "priority": priority,
                    "max_attempts": max_attempts,
                    "run_name": run_name,
                    "run_description": run_description,
                    "wandb_group": wandb_group,
                    "wandb_tags": list(wandb_tags),
                },
            )
            return dict(cur.fetchone())


def claim_train_job(
    conn,
    *,
    profile_id: str,
    worker_id: str,
    lease_seconds: int,
) -> dict[str, Any] | None:
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                CLAIM_TRAIN_JOB_SQL,
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
                  train_job_id, goal_id, experiment_spec_id, profile_id, status,
                  exit_code, run_name, run_dir, final_model_path, wandb_run_id,
                  wandb_url, artifact_refs, metrics_json, error
                )
                VALUES (
                  %(train_job_id)s, %(goal_id)s, %(experiment_spec_id)s, %(profile_id)s,
                  %(status)s, %(exit_code)s, %(run_name)s, %(run_dir)s,
                  %(final_model_path)s, %(wandb_run_id)s, %(wandb_url)s,
                  %(artifact_refs)s, %(metrics_json)s, %(error)s
                )
                ON CONFLICT (train_job_id) DO UPDATE SET
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


def campaign_status(conn, *, goal_slug_or_id: str, recent_decisions: int = 5) -> dict[str, Any]:
    goal_filter = (
        ("id = %(goal_id)s", {"goal_id": int(goal_slug_or_id)})
        if goal_slug_or_id.isdigit()
        else ("slug = %(goal_slug)s", {"goal_slug": goal_slug_or_id})
    )
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
        "recent_results": results,
        "decisions": decisions,
    }


def print_status(report: Mapping[str, Any]) -> None:
    goal = report["goal"]
    print(f"goal {goal['id']}: {goal['slug']} [{goal['status']}]")
    print(f"title: {goal['title']}")
    print(f"objective: {json.dumps(goal['objective_json'], sort_keys=True, default=str)}")
    print(f"constraints: {json.dumps(goal['constraints_json'], sort_keys=True, default=str)}")
    print(f"train_jobs: {json.dumps(report['train_jobs'], sort_keys=True)}")
    print("recent_results:")
    for row in report["recent_results"]:
        print(
            "  "
            f"result={row['id']} status={row['status']} profile={row['profile_id']} "
            f"run={row.get('run_name') or ''} wandb={row.get('wandb_url') or ''}"
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
    spec.add_argument("--priority", type=int, default=0)
    spec.set_defaults(func=cmd_add_spec)

    enqueue = subparsers.add_parser("enqueue-train", help="Create a concrete train job")
    enqueue.add_argument("--goal", required=True, help="Research goal slug")
    enqueue.add_argument("--spec-id", type=int, required=True)
    enqueue.add_argument("--profile", required=True)
    enqueue.add_argument("--train-config-json", required=True)
    enqueue.add_argument("--priority", type=int, default=0)
    enqueue.add_argument("--max-attempts", type=int, default=1)
    enqueue.add_argument("--run-name")
    enqueue.add_argument("--run-description")
    enqueue.add_argument("--wandb-group")
    enqueue.add_argument("--wandb-tag", action="append", default=[])
    enqueue.set_defaults(func=cmd_enqueue_train)

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

    status = subparsers.add_parser("status", help="Print compact campaign status")
    status.add_argument("goal")
    status.add_argument("--recent-decisions", type=int, default=5)
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
            train_config=load_json_arg(args.train_config_json, default={}),
            priority=args.priority,
        )
    finally:
        conn.close()
    print(f"spec_id={row['id']} slug={row['slug']}")
    return 0


def cmd_enqueue_train(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        goal_id = goal_id_from_slug(conn, args.goal)
        row = enqueue_train_job(
            conn,
            goal_id=goal_id,
            experiment_spec_id=args.spec_id,
            profile_id=args.profile,
            train_config=load_json_arg(args.train_config_json, default={}),
            priority=args.priority,
            max_attempts=args.max_attempts,
            run_name=args.run_name,
            run_description=args.run_description,
            wandb_group=args.wandb_group,
            wandb_tags=args.wandb_tag,
        )
    finally:
        conn.close()
    print(f"train_job_id={row['id']} profile={row['profile_id']}")
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


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


def new_worker_id(prefix: str = "train-runner") -> str:
    return f"{prefix}-{uuid.uuid4()}"


if __name__ == "__main__":
    main()
