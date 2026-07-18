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
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psycopg2
import psycopg2.extras

from rlab.cli_args import add_direct_database_arg
from rlab.dotenv import load_env_file
from rlab.env_registry import resolve_env_provider
from rlab.json_utils import json_safe
from rlab.machines import DEFAULT_MACHINE_REGISTRY, load_machine_registry, resolve_machine
from rlab.runtime_refs import (
    DEFAULT_IMAGE_ARTIFACT,
    DEFAULT_IMAGE_WORKFLOW,
    DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
    clean_git_source_sha,
    normalize_runtime_image_ref,
    runtime_release_from_args,
    wait_for_modal_readiness,
)
from rlab.recipe_documents import (
    assert_no_secrets,
    compose_train_document,
    compiled_recipe_payload,
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
from rlab.policy_bundle import build_recipe_document
from rlab.train_config import validate_and_normalize_train_config
from rlab.training_backend import accepts_first_training_success
from rlab.modal_eval_assets import asset_manifest_for_game
from rlab.checkpoint_acceptance import (
    EVAL_SEED_START,
    checkpoint_eval_contract_from_train_config,
)
from rlab.modal_eval_config import load_modal_eval_config
from rlab.modal_eval_protocol import SEED_PROTOCOL


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS train_jobs (
  id BIGSERIAL PRIMARY KEY,
  goal_slug TEXT NOT NULL,
  goal_path TEXT,
  goal_sha256 TEXT,
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
    CHECK (status IN (
      'pending', 'launching', 'starting', 'running', 'finalizing',
      'succeeded', 'failed', 'finalization_failed', 'canceled'
    )),
  cancel_requested BOOLEAN NOT NULL DEFAULT FALSE,
  batch_id TEXT NOT NULL,
  campaign_id TEXT,
  submission_key TEXT NOT NULL,
  submission_ordinal INTEGER NOT NULL CHECK (submission_ordinal >= 0),
  request_hash TEXT NOT NULL,
  retry_of_job_id BIGINT REFERENCES train_jobs(id),
  run_name TEXT,
  run_description TEXT,
  seed INTEGER,
  wandb_group TEXT,
  wandb_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  learner_ready_at TIMESTAMPTZ,
  wandb_ready_at TIMESTAMPTZ,
  ready_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  wandb_run_id TEXT,
  wandb_url TEXT,
  live_publication_status TEXT NOT NULL DEFAULT 'pending'
    CHECK (live_publication_status IN (
      'pending', 'live', 'finishing', 'complete', 'disabled', 'failed'
    )),
  live_publication_attempts INTEGER NOT NULL DEFAULT 0
    CHECK (live_publication_attempts >= 0),
  live_publication_next_retry_at TIMESTAMPTZ,
  live_publication_error TEXT,
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

CREATE TABLE IF NOT EXISTS runtime_image_states (
  machine TEXT NOT NULL,
  runtime_image_ref TEXT NOT NULL,
  retry_count INTEGER NOT NULL DEFAULT 0 CHECK (retry_count >= 0),
  next_retry_at TIMESTAMPTZ,
  last_error TEXT,
  last_attempt_at TIMESTAMPTZ,
  last_ready_at TIMESTAMPTZ,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (machine, runtime_image_ref)
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
    CHECK (status IN (
      'active', 'awaiting_artifact_recovery', 'finalizing', 'complete', 'failed', 'canceled'
    )),
  contract_json JSONB NOT NULL,
  next_announcement_id BIGINT NOT NULL DEFAULT 1 CHECK (next_announcement_id >= 1),
  next_artifact_projection_id BIGINT NOT NULL DEFAULT 1 CHECK (next_artifact_projection_id >= 1),
  complete_announcement_seen BOOLEAN NOT NULL DEFAULT FALSE,
  last_scheduled_at TIMESTAMPTZ,
  promoted_eval_job_id BIGINT,
  promotion_revision BIGINT NOT NULL DEFAULT 0 CHECK (promotion_revision >= 0),
  promotion_json JSONB,
  outcome TEXT CHECK (
    outcome IS NULL OR outcome IN ('accepted', 'not_accepted', 'unknown', 'canceled')
  ),
  acceptance_committed_at TIMESTAMPTZ,
  stop_delivery_slo_met BOOLEAN,
  promoted_artifact_projection_enqueued_at TIMESTAMPTZ,
  promoted_artifact_projected_at TIMESTAMPTZ,
  artifacts_projected_at TIMESTAMPTZ,
  artifact_projection_attempts INTEGER NOT NULL DEFAULT 0
    CHECK (artifact_projection_attempts >= 0),
  artifact_projection_next_retry_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  error TEXT
);

CREATE TABLE IF NOT EXISTS artifact_announcement_ledger (
  train_job_id BIGINT NOT NULL REFERENCES eval_runs(train_job_id) ON DELETE CASCADE,
  ledger_id BIGINT NOT NULL CHECK (ledger_id >= 1),
  disposition TEXT NOT NULL CHECK (disposition IN ('ready', 'tombstone')),
  artifact_kind TEXT NOT NULL,
  checkpoint_step BIGINT,
  checkpoint_sha256 TEXT,
  checkpoint_uri TEXT,
  metadata_uri TEXT,
  metadata_sha256 TEXT,
  recipe_uri TEXT,
  recipe_sha256 TEXT,
  evaluation_contract_sha256 TEXT,
  announcement_sha256 TEXT NOT NULL,
  announcement_json JSONB NOT NULL,
  verified_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (train_job_id, ledger_id),
  CHECK (
    (disposition = 'tombstone' AND artifact_kind = 'tombstone')
    OR (
      disposition = 'ready'
      AND artifact_kind IN ('checkpoint', 'final', 'interrupted')
      AND checkpoint_step IS NOT NULL
      AND checkpoint_sha256 IS NOT NULL
      AND checkpoint_uri IS NOT NULL
      AND metadata_uri IS NOT NULL
      AND metadata_sha256 IS NOT NULL
    )
  )
);

CREATE TABLE IF NOT EXISTS artifact_publication_receipts (
  train_job_id BIGINT NOT NULL,
  ledger_id BIGINT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('availability', 'promotion')),
  promotion_revision BIGINT NOT NULL DEFAULT 0 CHECK (promotion_revision >= 0),
  disposition TEXT NOT NULL CHECK (disposition IN ('confirmed', 'opted_out')),
  artifact_kind TEXT NOT NULL,
  checkpoint_step BIGINT NOT NULL,
  checkpoint_sha256 TEXT NOT NULL,
  checkpoint_uri TEXT NOT NULL,
  metadata_uri TEXT NOT NULL,
  metadata_sha256 TEXT NOT NULL,
  recipe_uri TEXT,
  recipe_sha256 TEXT,
  announcement_sha256 TEXT NOT NULL,
  collection_name TEXT,
  artifact_version TEXT,
  artifact_ref TEXT,
  stream_id TEXT,
  expected_aliases TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  confirmed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (train_job_id, ledger_id, role, promotion_revision),
  FOREIGN KEY (train_job_id, ledger_id)
    REFERENCES artifact_announcement_ledger(train_job_id, ledger_id) ON DELETE CASCADE,
  CHECK (
    (role = 'availability' AND promotion_revision = 0)
    OR (role = 'promotion' AND promotion_revision >= 1)
  ),
  CHECK (
    (disposition = 'opted_out'
      AND collection_name IS NULL AND artifact_version IS NULL AND artifact_ref IS NULL)
    OR (disposition = 'confirmed'
      AND collection_name IS NOT NULL AND artifact_version IS NOT NULL
      AND artifact_ref IS NOT NULL AND stream_id IS NOT NULL)
  )
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
  purpose TEXT NOT NULL CHECK (purpose IN ('screen', 'confirm', 'promotion', 'acceptance')),
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
  projection_enqueued_at TIMESTAMPTZ,
  projected_at TIMESTAMPTZ,
  projection_error TEXT,
  projection_attempts INTEGER NOT NULL DEFAULT 0 CHECK (projection_attempts >= 0),
  projection_next_retry_at TIMESTAMPTZ,
  retry_round INTEGER NOT NULL DEFAULT 0 CHECK (retry_round >= 0),
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
  retry_round INTEGER NOT NULL DEFAULT 0 CHECK (retry_round >= 0),
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
  UNIQUE (eval_job_id, retry_round, attempt_number)
);

CREATE TABLE IF NOT EXISTS worker_attempts (
  id BIGSERIAL PRIMARY KEY,
  attempt_id TEXT NOT NULL UNIQUE,
  train_job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  eval_job_id BIGINT REFERENCES eval_jobs(id) ON DELETE CASCADE,
  task_kind TEXT NOT NULL CHECK (task_kind IN ('train', 'eval')),
  provider TEXT NOT NULL,
  provider_run_id TEXT,
  status TEXT NOT NULL DEFAULT 'launching'
    CHECK (status IN ('launching', 'running', 'succeeded', 'failed', 'canceled')),
  protocol_version INTEGER NOT NULL DEFAULT 1 CHECK (protocol_version = 1),
  token_sha256 TEXT,
  token_expires_at TIMESTAMPTZ,
  last_heartbeat_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ,
  error TEXT,
  CHECK (
    (task_kind = 'train' AND eval_job_id IS NULL)
    OR (task_kind = 'eval' AND eval_job_id IS NOT NULL)
  )
);

CREATE TABLE IF NOT EXISTS metric_streams (
  id BIGSERIAL PRIMARY KEY,
  stream_id TEXT NOT NULL UNIQUE,
  attempt_id TEXT NOT NULL REFERENCES worker_attempts(attempt_id) ON DELETE CASCADE,
  accepted_sequence BIGINT NOT NULL DEFAULT 0 CHECK (accepted_sequence >= 0),
  final_sequence BIGINT CHECK (final_sequence IS NULL OR final_sequence >= 0),
  submitted_sequence BIGINT NOT NULL DEFAULT 0 CHECK (submitted_sequence >= 0),
  published_sequence BIGINT NOT NULL DEFAULT 0 CHECK (published_sequence >= 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (final_sequence IS NULL OR accepted_sequence >= final_sequence),
  CHECK (accepted_sequence >= submitted_sequence),
  CHECK (submitted_sequence >= published_sequence)
);

CREATE TABLE IF NOT EXISTS metric_batches (
  id BIGSERIAL PRIMARY KEY,
  stream_id TEXT NOT NULL REFERENCES metric_streams(stream_id) ON DELETE CASCADE,
  batch_sequence BIGINT NOT NULL CHECK (batch_sequence >= 1),
  first_event_sequence BIGINT CHECK (
    first_event_sequence IS NULL OR first_event_sequence >= 1
  ),
  last_event_sequence BIGINT CHECK (
    last_event_sequence IS NULL OR last_event_sequence >= first_event_sequence
  ),
  frame_count INTEGER NOT NULL CHECK (frame_count BETWEEN 0 AND 1000),
  codec TEXT NOT NULL DEFAULT 'gzip-json-v1' CHECK (codec = 'gzip-json-v1'),
  payload BYTEA NOT NULL CHECK (octet_length(payload) <= 2097152),
  final BOOLEAN NOT NULL DEFAULT FALSE,
  lease_owner TEXT,
  lease_expires_at TIMESTAMPTZ,
  attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  last_error TEXT,
  submitted_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (stream_id, batch_sequence)
);

CREATE TABLE IF NOT EXISTS attempt_events (
  id BIGSERIAL PRIMARY KEY,
  event_id TEXT NOT NULL UNIQUE,
  attempt_id TEXT NOT NULL REFERENCES worker_attempts(attempt_id) ON DELETE CASCADE,
  event_type TEXT NOT NULL,
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  next_retry_at TIMESTAMPTZ,
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS attempt_commands (
  id BIGSERIAL PRIMARY KEY,
  command_id TEXT NOT NULL UNIQUE,
  attempt_id TEXT NOT NULL REFERENCES worker_attempts(attempt_id) ON DELETE CASCADE,
  command_type TEXT NOT NULL CHECK (command_type IN ('stop', 'cancel')),
  payload_json JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  delivered_at TIMESTAMPTZ,
  acknowledged_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS eval_backend_state (
  backend TEXT PRIMARY KEY CHECK (backend = 'modal'),
  drained BOOLEAN NOT NULL DEFAULT FALSE,
  round_robin_after_train_job_id BIGINT NOT NULL DEFAULT 0,
  reason TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

ALTER TABLE eval_backend_state DROP COLUMN IF EXISTS effective_capacity;

INSERT INTO eval_backend_state (backend)
VALUES ('modal')
ON CONFLICT (backend) DO NOTHING;

DROP INDEX IF EXISTS train_jobs_runtime_claim_idx;
DROP INDEX IF EXISTS train_jobs_claim_idx;
DROP INDEX IF EXISTS train_jobs_spec_status_idx;

ALTER TABLE train_jobs DROP COLUMN IF EXISTS priority;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS campaign_id TEXT;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS goal_path TEXT;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS goal_sha256 TEXT;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS retry_of_job_id BIGINT REFERENCES train_jobs(id);
DO $$
BEGIN
  IF EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = current_schema()
      AND table_name = 'train_jobs'
      AND column_name = 'retried_from_job_id'
  ) THEN
    UPDATE train_jobs
    SET retry_of_job_id = COALESCE(retry_of_job_id, retried_from_job_id)
    WHERE retried_from_job_id IS NOT NULL;
    ALTER TABLE train_jobs DROP COLUMN retried_from_job_id;
  END IF;
END $$;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS learner_ready_at TIMESTAMPTZ;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS wandb_ready_at TIMESTAMPTZ;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS ready_at TIMESTAMPTZ;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS wandb_run_id TEXT;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS wandb_url TEXT;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS live_publication_status TEXT NOT NULL DEFAULT 'pending';
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS live_publication_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS live_publication_next_retry_at TIMESTAMPTZ;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS live_publication_error TEXT;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS learner_stop_observed_at TIMESTAMPTZ;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS process_exited_at TIMESTAMPTZ;
UPDATE train_jobs t
SET process_exited_at = l.finished_at
FROM job_launches l
WHERE l.job_kind = 'train'
  AND l.job_id = t.id
  AND l.state IN ('succeeded', 'failed', 'canceled')
  AND l.finished_at IS NOT NULL
  AND t.process_exited_at IS NULL;
ALTER TABLE train_jobs DROP CONSTRAINT IF EXISTS train_jobs_eval_load_check;
ALTER TABLE train_jobs DROP COLUMN IF EXISTS eval_load;
ALTER TABLE train_jobs DROP COLUMN IF EXISTS eval_capacity_policy_sha256;
ALTER TABLE train_jobs DROP COLUMN IF EXISTS eval_load_admitted_at;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS artifact_projection_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS artifact_projection_next_retry_at TIMESTAMPTZ;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS promoted_artifact_projected_at TIMESTAMPTZ;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS outcome TEXT;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS acceptance_committed_at TIMESTAMPTZ;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS stop_delivery_slo_met BOOLEAN;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS promoted_artifact_projection_enqueued_at TIMESTAMPTZ;
ALTER TABLE eval_runs ADD COLUMN IF NOT EXISTS promotion_revision BIGINT NOT NULL DEFAULT 0;
ALTER TABLE eval_runs DROP CONSTRAINT IF EXISTS eval_runs_promotion_revision_check;
ALTER TABLE eval_runs ADD CONSTRAINT eval_runs_promotion_revision_check
  CHECK (promotion_revision >= 0);
ALTER TABLE eval_jobs ADD COLUMN IF NOT EXISTS projection_attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE eval_jobs ADD COLUMN IF NOT EXISTS projection_enqueued_at TIMESTAMPTZ;
ALTER TABLE eval_jobs ADD COLUMN IF NOT EXISTS projection_next_retry_at TIMESTAMPTZ;
ALTER TABLE eval_jobs ADD COLUMN IF NOT EXISTS retry_round INTEGER NOT NULL DEFAULT 0;
ALTER TABLE attempt_events ADD COLUMN IF NOT EXISTS attempts INTEGER NOT NULL DEFAULT 0;
ALTER TABLE attempt_events ADD COLUMN IF NOT EXISTS next_retry_at TIMESTAMPTZ;
ALTER TABLE attempt_events ADD COLUMN IF NOT EXISTS last_error TEXT;
ALTER TABLE attempt_commands ADD COLUMN IF NOT EXISTS delivered_at TIMESTAMPTZ;
ALTER TABLE attempt_events DROP CONSTRAINT IF EXISTS attempt_events_attempts_check;
ALTER TABLE attempt_events ADD CONSTRAINT attempt_events_attempts_check CHECK (attempts >= 0);
ALTER TABLE metric_streams ADD COLUMN IF NOT EXISTS submitted_sequence BIGINT NOT NULL DEFAULT 0;
ALTER TABLE metric_batches ADD COLUMN IF NOT EXISTS submitted_at TIMESTAMPTZ;
UPDATE metric_streams
SET submitted_sequence = GREATEST(submitted_sequence, published_sequence);
ALTER TABLE metric_streams DROP CONSTRAINT IF EXISTS metric_streams_sequence_order_check;
ALTER TABLE metric_streams ADD CONSTRAINT metric_streams_sequence_order_check CHECK (
  accepted_sequence >= submitted_sequence
  AND submitted_sequence >= published_sequence
);
ALTER TABLE eval_attempts ADD COLUMN IF NOT EXISTS retry_round INTEGER NOT NULL DEFAULT 0;
ALTER TABLE train_jobs ADD COLUMN IF NOT EXISTS telemetry_transport TEXT NOT NULL
  DEFAULT 'legacy_local';
ALTER TABLE train_jobs DROP CONSTRAINT IF EXISTS train_jobs_telemetry_transport_check;
ALTER TABLE train_jobs ADD CONSTRAINT train_jobs_telemetry_transport_check
  CHECK (telemetry_transport IN ('legacy_local', 'neon_mailbox_v1'));
ALTER TABLE eval_attempts DROP CONSTRAINT IF EXISTS eval_attempts_eval_job_id_attempt_number_key;
ALTER TABLE eval_attempts DROP CONSTRAINT IF EXISTS eval_attempts_eval_job_id_retry_round_attempt_number_key;
ALTER TABLE eval_attempts ADD CONSTRAINT eval_attempts_eval_job_id_retry_round_attempt_number_key
  UNIQUE (eval_job_id, retry_round, attempt_number);
ALTER TABLE eval_jobs DROP CONSTRAINT IF EXISTS eval_jobs_retry_round_check;
ALTER TABLE eval_jobs ADD CONSTRAINT eval_jobs_retry_round_check CHECK (retry_round >= 0);
ALTER TABLE eval_attempts DROP CONSTRAINT IF EXISTS eval_attempts_retry_round_check;
ALTER TABLE eval_attempts ADD CONSTRAINT eval_attempts_retry_round_check CHECK (retry_round >= 0);
ALTER TABLE train_jobs DROP CONSTRAINT IF EXISTS train_jobs_status_check;
ALTER TABLE train_jobs ADD CONSTRAINT train_jobs_status_check
  CHECK (status IN (
    'pending', 'launching', 'starting', 'running', 'finalizing',
    'succeeded', 'failed', 'finalization_failed', 'canceled'
  ));
ALTER TABLE train_jobs DROP CONSTRAINT IF EXISTS train_jobs_live_publication_status_check;
ALTER TABLE train_jobs ADD CONSTRAINT train_jobs_live_publication_status_check
  CHECK (live_publication_status IN (
    'pending', 'live', 'finishing', 'complete', 'disabled', 'failed'
  ));
ALTER TABLE train_jobs DROP CONSTRAINT IF EXISTS train_jobs_live_publication_attempts_check;
ALTER TABLE train_jobs ADD CONSTRAINT train_jobs_live_publication_attempts_check
  CHECK (live_publication_attempts >= 0);
ALTER TABLE eval_runs DROP CONSTRAINT IF EXISTS eval_runs_status_check;
ALTER TABLE eval_runs ADD CONSTRAINT eval_runs_status_check
  CHECK (status IN (
    'active', 'awaiting_artifact_recovery', 'finalizing', 'complete', 'failed', 'canceled'
  ));
ALTER TABLE eval_runs DROP CONSTRAINT IF EXISTS eval_runs_outcome_check;
ALTER TABLE eval_runs ADD CONSTRAINT eval_runs_outcome_check
  CHECK (outcome IS NULL OR outcome IN ('accepted', 'not_accepted', 'unknown', 'canceled'));
ALTER TABLE eval_jobs DROP CONSTRAINT IF EXISTS eval_jobs_purpose_check;
ALTER TABLE eval_jobs ADD CONSTRAINT eval_jobs_purpose_check
  CHECK (purpose IN ('screen', 'confirm', 'promotion', 'acceptance'));

UPDATE train_jobs
SET live_publication_status = CASE
  WHEN COALESCE((train_config->>'wandb')::boolean, FALSE) THEN 'complete'
  ELSE 'disabled'
END
WHERE live_publication_status = 'pending'
  AND status IN ('succeeded', 'failed', 'finalization_failed', 'canceled');

UPDATE train_jobs t
SET status = 'finalizing', finished_at = NULL
WHERE t.status = 'succeeded'
  AND COALESCE(t.train_config->>'checkpoint_eval_backend', 'local') = 'modal'
  AND EXISTS (
    SELECT 1 FROM eval_runs r
    WHERE r.train_job_id = t.id AND r.status <> 'complete'
  );

CREATE INDEX IF NOT EXISTS train_jobs_claim_idx
  ON train_jobs (machine, status, id)
  WHERE status = 'pending' AND cancel_requested = FALSE;

CREATE INDEX IF NOT EXISTS train_jobs_runtime_claim_idx
  ON train_jobs (machine, runtime_image_ref, status, id)
  WHERE status IN ('pending', 'launching', 'starting', 'running');

CREATE INDEX IF NOT EXISTS runtime_image_states_retry_idx
  ON runtime_image_states (machine, next_retry_at);

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

CREATE INDEX IF NOT EXISTS artifact_announcement_ledger_publication_idx
  ON artifact_announcement_ledger (verified_at, train_job_id, ledger_id)
  WHERE disposition = 'ready';

CREATE INDEX IF NOT EXISTS artifact_publication_receipts_run_idx
  ON artifact_publication_receipts (train_job_id, role, ledger_id, promotion_revision);

DROP INDEX IF EXISTS artifact_publication_receipts_ref_idx;
CREATE UNIQUE INDEX artifact_publication_receipts_ref_idx
  ON artifact_publication_receipts (artifact_ref, role, promotion_revision)
  WHERE artifact_ref IS NOT NULL;

CREATE INDEX IF NOT EXISTS eval_attempts_status_idx
  ON eval_attempts (status, expires_at, created_at);

CREATE INDEX IF NOT EXISTS worker_attempts_run_status_idx
  ON worker_attempts (train_job_id, status, created_at);

CREATE INDEX IF NOT EXISTS metric_batches_claim_idx
  ON metric_batches (lease_expires_at, created_at, stream_id, batch_sequence);

CREATE INDEX IF NOT EXISTS attempt_events_attempt_idx
  ON attempt_events (attempt_id, created_at);
CREATE INDEX IF NOT EXISTS attempt_events_retry_idx
  ON attempt_events (next_retry_at, created_at);

CREATE INDEX IF NOT EXISTS attempt_commands_pending_idx
  ON attempt_commands (attempt_id, created_at)
  WHERE acknowledged_at IS NULL;

ALTER TABLE metric_batches SET (
  autovacuum_vacuum_scale_factor = 0.02,
  autovacuum_vacuum_threshold = 1000,
  autovacuum_analyze_scale_factor = 0.05
);
ALTER TABLE attempt_events SET (
  autovacuum_vacuum_scale_factor = 0.02,
  autovacuum_vacuum_threshold = 1000
);
ALTER TABLE attempt_commands SET (
  autovacuum_vacuum_scale_factor = 0.02,
  autovacuum_vacuum_threshold = 1000
);

CREATE INDEX IF NOT EXISTS eval_jobs_execution_idx
  ON eval_jobs (execution_key, status);

CREATE EXTENSION IF NOT EXISTS pgcrypto;

UPDATE eval_runs
SET promotion_revision = 1,
    promotion_json = COALESCE(promotion_json, '{}'::jsonb)
      || jsonb_build_object('promotion_revision', 1)
WHERE promoted_eval_job_id IS NOT NULL AND promotion_revision = 0;

INSERT INTO artifact_announcement_ledger (
  train_job_id, ledger_id, disposition, artifact_kind, checkpoint_step,
  checkpoint_sha256, checkpoint_uri, metadata_uri, metadata_sha256,
  recipe_uri, recipe_sha256, evaluation_contract_sha256,
  announcement_sha256, announcement_json, verified_at
)
SELECT DISTINCT ON (j.train_job_id, j.ledger_id)
  j.train_job_id,
  j.ledger_id,
  'ready',
  j.source_announcement_json->>'kind',
  (j.source_announcement_json->>'step')::bigint,
  j.source_announcement_json->>'sha256',
  j.source_announcement_json->>'model_uri',
  j.source_announcement_json->>'metadata_uri',
  COALESCE(
    j.source_announcement_json->>'model_document_sha256',
    j.source_announcement_json->>'metadata_sha256'
  ),
  NULLIF(j.source_announcement_json->>'recipe_uri', ''),
  NULLIF(j.source_announcement_json->>'recipe_sha256', ''),
  NULLIF(j.source_announcement_json->>'evaluation_contract_sha256', ''),
  encode(digest(j.source_announcement_json::text, 'sha256'), 'hex'),
  j.source_announcement_json,
  COALESCE(j.created_at, now())
FROM eval_jobs j
WHERE j.source_announcement_json->>'kind' IN ('checkpoint', 'final', 'interrupted')
ON CONFLICT (train_job_id, ledger_id) DO NOTHING;

INSERT INTO artifact_announcement_ledger (
  train_job_id, ledger_id, disposition, artifact_kind,
  announcement_sha256, announcement_json, verified_at
)
SELECT
  r.train_job_id,
  missing.ledger_id,
  'tombstone',
  'tombstone',
  encode(digest(document.payload::text, 'sha256'), 'hex'),
  document.payload,
  now()
FROM eval_runs r
CROSS JOIN LATERAL generate_series(
  1, GREATEST(r.next_announcement_id - 1, 0)
) AS missing(ledger_id)
CROSS JOIN LATERAL (
  SELECT jsonb_build_object(
    'kind', 'tombstone',
    'ledger_id', missing.ledger_id,
    'reason', 'historical announcement unavailable during ledger migration'
  ) AS payload
) AS document
WHERE r.complete_announcement_seen = TRUE
  AND NOT EXISTS (
    SELECT 1 FROM artifact_announcement_ledger existing
    WHERE existing.train_job_id = r.train_job_id
      AND existing.ledger_id = missing.ledger_id
  )
ON CONFLICT (train_job_id, ledger_id) DO NOTHING;

CREATE OR REPLACE FUNCTION worker_submit_metric_batch(
  p_attempt_id TEXT,
  p_token TEXT,
  p_protocol_version INTEGER,
  p_batch_sequence BIGINT,
  p_first_event_sequence BIGINT,
  p_last_event_sequence BIGINT,
  p_frame_count INTEGER,
  p_payload BYTEA,
  p_final BOOLEAN
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  attempt_row worker_attempts%ROWTYPE;
  stream_row metric_streams%ROWTYPE;
  commands JSONB;
BEGIN
  SELECT * INTO attempt_row FROM worker_attempts
  WHERE attempt_id = p_attempt_id FOR UPDATE;
  IF NOT FOUND
     OR attempt_row.protocol_version <> p_protocol_version
     OR attempt_row.status NOT IN ('launching', 'running')
     OR attempt_row.token_expires_at IS NULL
     OR attempt_row.token_expires_at <= now()
     OR attempt_row.token_sha256 IS DISTINCT FROM encode(digest(p_token, 'sha256'), 'hex') THEN
    RAISE EXCEPTION 'worker mailbox authentication failed';
  END IF;
  IF p_batch_sequence < 1
     OR p_frame_count < 0 OR p_frame_count > 1000
     OR octet_length(p_payload) > 2097152 THEN
    RAISE EXCEPTION 'worker metric batch exceeds protocol limits';
  END IF;

  INSERT INTO metric_streams (stream_id, attempt_id)
  VALUES (p_attempt_id, p_attempt_id)
  ON CONFLICT (stream_id) DO NOTHING;
  SELECT * INTO stream_row FROM metric_streams
  WHERE stream_id = p_attempt_id FOR UPDATE;

  IF stream_row.final_sequence IS NOT NULL
     AND p_batch_sequence > stream_row.final_sequence THEN
    RAISE EXCEPTION 'worker metric stream is already closed';
  END IF;
  IF p_batch_sequence > stream_row.accepted_sequence + 1 THEN
    RAISE EXCEPTION 'worker metric batch sequence gap';
  END IF;
  IF p_batch_sequence = stream_row.accepted_sequence + 1 THEN
    INSERT INTO metric_batches (
      stream_id, batch_sequence, first_event_sequence, last_event_sequence,
      frame_count, payload, final
    ) VALUES (
      p_attempt_id, p_batch_sequence, p_first_event_sequence, p_last_event_sequence,
      p_frame_count, p_payload, p_final
    );
    UPDATE metric_streams
    SET accepted_sequence = p_batch_sequence,
        final_sequence = CASE WHEN p_final THEN p_batch_sequence ELSE final_sequence END,
        updated_at = now()
    WHERE stream_id = p_attempt_id;
  ELSE
    IF p_batch_sequence > stream_row.published_sequence AND NOT EXISTS (
      SELECT 1 FROM metric_batches
      WHERE stream_id = p_attempt_id
        AND batch_sequence = p_batch_sequence
        AND first_event_sequence IS NOT DISTINCT FROM p_first_event_sequence
        AND last_event_sequence IS NOT DISTINCT FROM p_last_event_sequence
        AND frame_count = p_frame_count
        AND payload = p_payload
    ) THEN
      RAISE EXCEPTION 'duplicate worker metric batch does not match accepted payload';
    END IF;
  END IF;
  IF p_final THEN
    UPDATE metric_streams
    SET final_sequence = COALESCE(final_sequence, p_batch_sequence), updated_at = now()
    WHERE stream_id = p_attempt_id;
  END IF;

  UPDATE worker_attempts SET last_heartbeat_at = now()
  WHERE attempt_id = p_attempt_id;
  UPDATE train_jobs
  SET live_publication_status = CASE
        WHEN COALESCE((train_config->>'wandb')::boolean, FALSE)
          THEN 'pending'
        ELSE 'disabled'
      END,
      live_publication_error = NULL,
      live_publication_next_retry_at = NULL
  WHERE id = attempt_row.train_job_id;
  SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'command_id', command_id,
      'command_type', command_type,
      'payload', payload_json
    ) ORDER BY id), '[]'::jsonb)
  INTO commands
  FROM attempt_commands
  WHERE attempt_id = p_attempt_id AND acknowledged_at IS NULL;
  RETURN jsonb_build_object(
    'accepted_sequence', GREATEST(stream_row.accepted_sequence, p_batch_sequence),
    'commands', commands
  );
END;
$$;

CREATE OR REPLACE FUNCTION worker_append_attempt_event(
  p_attempt_id TEXT,
  p_token TEXT,
  p_event_id TEXT,
  p_event_type TEXT,
  p_payload JSONB
) RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  IF p_event_type NOT IN (
       'mailbox_preflight', 'metric_stream_closed',
       'checkpoint_ready', 'checkpoint_tombstone', 'checkpoint_stream_closed',
       'learner_stop_observed'
     )
     OR p_payload IS NULL
     OR octet_length(p_payload::text) > 1048576 THEN
    RAISE EXCEPTION 'worker attempt event exceeds protocol limits';
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM worker_attempts
    WHERE attempt_id = p_attempt_id
      AND status IN ('launching', 'running')
      AND token_expires_at > now()
      AND token_sha256 = encode(digest(p_token, 'sha256'), 'hex')
  ) THEN
    RAISE EXCEPTION 'worker mailbox authentication failed';
  END IF;
  INSERT INTO attempt_events (event_id, attempt_id, event_type, payload_json)
  VALUES (p_event_id, p_attempt_id, p_event_type, COALESCE(p_payload, '{}'::jsonb))
  ON CONFLICT (event_id) DO NOTHING;
  UPDATE worker_attempts SET last_heartbeat_at = now()
  WHERE attempt_id = p_attempt_id;
  RETURN TRUE;
END;
$$;

CREATE OR REPLACE FUNCTION worker_poll_attempt_commands(
  p_attempt_id TEXT,
  p_token TEXT
) RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
  commands JSONB;
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM worker_attempts
    WHERE attempt_id = p_attempt_id
      AND status IN ('launching', 'running')
      AND token_expires_at > now()
      AND token_sha256 = encode(digest(p_token, 'sha256'), 'hex')
  ) THEN
    RAISE EXCEPTION 'worker mailbox authentication failed';
  END IF;
  SELECT COALESCE(jsonb_agg(jsonb_build_object(
      'command_id', command_id,
      'command_type', command_type,
      'payload', payload_json,
      'created_at', created_at
    ) ORDER BY id), '[]'::jsonb)
  INTO commands
  FROM attempt_commands
  WHERE attempt_id = p_attempt_id AND acknowledged_at IS NULL;
  UPDATE worker_attempts SET last_heartbeat_at = now()
  WHERE attempt_id = p_attempt_id;
  RETURN commands;
END;
$$;

CREATE OR REPLACE FUNCTION worker_mark_attempt_command_delivered(
  p_attempt_id TEXT,
  p_token TEXT,
  p_command_id TEXT
) RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM worker_attempts
    WHERE attempt_id = p_attempt_id
      AND status IN ('launching', 'running')
      AND token_expires_at > now()
      AND token_sha256 = encode(digest(p_token, 'sha256'), 'hex')
  ) THEN
    RAISE EXCEPTION 'worker mailbox authentication failed';
  END IF;
  UPDATE attempt_commands
  SET delivered_at = COALESCE(delivered_at, now())
  WHERE attempt_id = p_attempt_id AND command_id = p_command_id
    AND acknowledged_at IS NULL;
  RETURN FOUND;
END;
$$;

CREATE OR REPLACE FUNCTION worker_ack_attempt_command(
  p_attempt_id TEXT,
  p_token TEXT,
  p_command_id TEXT,
  p_acknowledged_at TIMESTAMPTZ
) RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM worker_attempts
    WHERE attempt_id = p_attempt_id
      AND status IN ('launching', 'running')
      AND token_expires_at > now()
      AND token_sha256 = encode(digest(p_token, 'sha256'), 'hex')
  ) THEN
    RAISE EXCEPTION 'worker mailbox authentication failed';
  END IF;
  UPDATE attempt_commands
  SET delivered_at = COALESCE(delivered_at, p_acknowledged_at),
      acknowledged_at = p_acknowledged_at
  WHERE attempt_id = p_attempt_id AND command_id = p_command_id
    AND acknowledged_at IS NULL;
  RETURN FOUND;
END;
$$;

CREATE OR REPLACE FUNCTION worker_ack_attempt_command(
  p_attempt_id TEXT,
  p_token TEXT,
  p_command_id TEXT
) RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
  SELECT worker_ack_attempt_command(p_attempt_id, p_token, p_command_id, now());
$$;

REVOKE ALL ON worker_attempts, metric_streams, metric_batches,
  attempt_events, attempt_commands FROM PUBLIC;
REVOKE ALL ON FUNCTION worker_submit_metric_batch(
  TEXT, TEXT, INTEGER, BIGINT, BIGINT, BIGINT, INTEGER, BYTEA, BOOLEAN
) FROM PUBLIC;
REVOKE ALL ON FUNCTION worker_append_attempt_event(
  TEXT, TEXT, TEXT, TEXT, JSONB
) FROM PUBLIC;
REVOKE ALL ON FUNCTION worker_poll_attempt_commands(TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION worker_mark_attempt_command_delivered(
  TEXT, TEXT, TEXT
) FROM PUBLIC;
REVOKE ALL ON FUNCTION worker_ack_attempt_command(TEXT, TEXT, TEXT) FROM PUBLIC;
REVOKE ALL ON FUNCTION worker_ack_attempt_command(
  TEXT, TEXT, TEXT, TIMESTAMPTZ
) FROM PUBLIC;
"""

RESET_TABLES = (
    "attempt_commands",
    "attempt_events",
    "metric_batches",
    "metric_streams",
    "worker_attempts",
    "eval_attempts",
    "eval_jobs",
    "artifact_publication_receipts",
    "artifact_announcement_ledger",
    "eval_runs",
    "eval_backend_state",
    "job_events",
    "job_launches",
    "train_jobs",
    "runtime_image_states",
    "machine_controls",
)
TRAIN_JOB_KIND = "train"
SCHEMA_MAINTENANCE_LOCK = "rlab-fleet-schema-maintenance"
FLEET_ADMISSION_LOCK = "rlab-fleet-admission-v1"


def acquire_fleet_admission_xact_lock(conn, *, exclusive: bool = False) -> None:
    function = "pg_advisory_xact_lock" if exclusive else "pg_advisory_xact_lock_shared"
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT {function}(hashtextextended(%(key)s, 0))",
            {"key": FLEET_ADMISSION_LOCK},
        )


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
        keepalives=1,
        keepalives_idle=10,
        keepalives_interval=5,
        keepalives_count=3,
        tcp_user_timeout=30000,
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
        acquire_fleet_admission_xact_lock(conn, exclusive=True)
        prepare_schema_upgrade(conn)
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)


def grant_worker_mailbox_role(conn, role: str) -> None:
    role = str(role).strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", role):
        raise ValueError("worker mailbox role must be a plain PostgreSQL identifier")
    identifier = psycopg2.extensions.quote_ident(role, conn)
    with conn:
        acquire_fleet_admission_xact_lock(conn)
        with conn.cursor() as cur:
            cur.execute(f"GRANT USAGE ON SCHEMA public TO {identifier}")
            cur.execute(
                "GRANT EXECUTE ON FUNCTION worker_submit_metric_batch("
                f"TEXT, TEXT, INTEGER, BIGINT, BIGINT, BIGINT, INTEGER, BYTEA, BOOLEAN) "
                f"TO {identifier}"
            )
            cur.execute(
                "GRANT EXECUTE ON FUNCTION worker_append_attempt_event("
                f"TEXT, TEXT, TEXT, TEXT, JSONB) TO {identifier}"
            )
            cur.execute(
                "GRANT EXECUTE ON FUNCTION worker_poll_attempt_commands("
                f"TEXT, TEXT) TO {identifier}"
            )
            cur.execute(
                "GRANT EXECUTE ON FUNCTION worker_mark_attempt_command_delivered("
                f"TEXT, TEXT, TEXT) TO {identifier}"
            )
            cur.execute(
                "GRANT EXECUTE ON FUNCTION worker_ack_attempt_command("
                f"TEXT, TEXT, TEXT) TO {identifier}"
            )
            cur.execute(
                "GRANT EXECUTE ON FUNCTION worker_ack_attempt_command("
                f"TEXT, TEXT, TEXT, TIMESTAMPTZ) TO {identifier}"
            )


def configured_worker_mailbox_role() -> str | None:
    """Resolve the actual restricted mailbox role from its configured connection."""

    load_env_file()
    worker_database_url = str(os.environ.get("WORKER_MAILBOX_DATABASE_URL") or "").strip()
    if not worker_database_url:
        return None
    worker_conn = connect(worker_database_url)
    try:
        with worker_conn.cursor() as cur:
            cur.execute("SELECT current_user AS role")
            row = cur.fetchone()
    finally:
        worker_conn.close()
    role = str((row or {}).get("role") or "").strip()
    if not role:
        raise RuntimeError("worker mailbox database did not report its PostgreSQL role")
    return role


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
            if table_name in {"machine_controls", "runtime_image_states"}
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
        acquire_fleet_admission_xact_lock(conn, exclusive=True)
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
                  attempt_commands,
                  attempt_events,
                  metric_batches,
                  metric_streams,
                  worker_attempts,
                  eval_attempts,
                  eval_jobs,
                  artifact_publication_receipts,
                  artifact_announcement_ledger,
                  eval_runs,
                  eval_backend_state,
                  job_events,
                  job_launches,
                  train_jobs,
                  runtime_image_states,
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
                     OR to_regclass('runtime_image_states') IS NULL
                     OR to_regclass('job_events') IS NULL
                     OR to_regclass('eval_runs') IS NULL
                     OR to_regclass('artifact_announcement_ledger') IS NULL
                     OR to_regclass('artifact_publication_receipts') IS NULL
                     OR to_regclass('eval_jobs') IS NULL
                     OR to_regclass('eval_attempts') IS NULL
                     OR to_regclass('worker_attempts') IS NULL
                     OR to_regclass('metric_streams') IS NULL
                     OR to_regclass('metric_batches') IS NULL
                     OR to_regclass('attempt_events') IS NULL
                     OR to_regclass('attempt_commands') IS NULL
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


def _document_seeds(document: Mapping[str, Any], override_seeds: Sequence[int] = ()) -> list[int]:
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
    goal_path: str | None = None,
    goal_sha256: str | None = None,
    recipe_path: str | None = None,
    recipe_sha256: str | None = None,
) -> str:
    return _hash_json(
        {
            "document": document,
            "goal_path": str(goal_path or ""),
            "goal_sha256": str(goal_sha256 or ""),
            "machine": machine,
            "recipe_path": str(recipe_path or ""),
            "recipe_sha256": str(recipe_sha256 or ""),
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


def modal_eval_readiness_report(
    *,
    runtime_image_ref: str,
    game: str,
    env_provider: str = "",
    runtime_input_sha256: str = "",
    runtime_build_source_sha: str = "",
) -> dict[str, Any]:
    # Keep this import local: modal_eval_cli uses the queue connection helpers.
    from rlab.modal_eval_cli import modal_preflight

    return modal_preflight(
        runtime_image_ref=runtime_image_ref,
        game=game,
        env_provider=env_provider,
        runtime_input_sha256=runtime_input_sha256,
        runtime_build_source_sha=runtime_build_source_sha,
    )


def require_modal_eval_ready(
    *,
    runtime_image_ref: str,
    game: str,
    env_provider: str = "",
    runtime_input_sha256: str = "",
    runtime_build_source_sha: str = "",
) -> dict[str, Any]:
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    game = str(game or "").strip()
    env_provider = str(env_provider or "").strip()
    readiness_options: dict[str, Any] = {
        "runtime_image_ref": runtime_image_ref,
        "game": game,
    }
    if env_provider:
        readiness_options["env_provider"] = env_provider
    if runtime_input_sha256:
        readiness_options["runtime_input_sha256"] = runtime_input_sha256
    if runtime_build_source_sha:
        readiness_options["runtime_build_source_sha"] = runtime_build_source_sha
    report = modal_eval_readiness_report(**readiness_options)
    if bool(report.get("ready")):
        return report
    failed = [check for check in report.get("checks", []) if not bool(check.get("ok"))]
    detail = (
        ", ".join(
            f"{check.get('name', 'unknown')} ({check.get('detail', 'failed')})" for check in failed
        )
        or "unknown readiness failure"
    )
    remediation_parts = [
        "rlab",
        "eval",
        "modal",
        "preflight",
        "--runtime-image-ref",
        runtime_image_ref,
        "--game",
        game,
    ]
    if env_provider:
        remediation_parts.extend(["--env-provider", env_provider])
    remediation = shlex.join(remediation_parts)
    raise RuntimeError(f"Modal eval preflight failed: {detail}. Remediation: run {remediation}")


def resolve_checkpoint_eval_backend(
    train_config: Mapping[str, Any],
    *,
    checkpoint_eval_backend: str | None,
) -> str:
    configured = str(train_config.get("checkpoint_eval_backend") or "")
    declared_training_acceptance = accepts_first_training_success(train_config)
    if (
        configured == "none"
        and checkpoint_eval_backend != "none"
        and not (checkpoint_eval_backend is None and declared_training_acceptance)
    ):
        raise ValueError(
            "checkpoint_eval_backend=none is a per-submission smoke/debug override and "
            "cannot be a checked-in recipe default unless the training backend declares "
            "first-training-success acceptance"
        )
    backend = str(checkpoint_eval_backend or configured or "")
    if not backend:
        modal_config_path = Path(__file__).resolve().parents[2] / "experiments" / "modal_eval.yaml"
        backend = (
            "modal"
            if modal_config_path.is_file() and load_modal_eval_config(modal_config_path).enabled
            else "local"
        )
    if backend not in {"local", "modal", "none"}:
        raise ValueError("checkpoint_eval_backend must be local, modal, or none")
    return backend


def enqueue_train_jobs_from_recipe_document(
    conn,
    *,
    document: Mapping[str, Any],
    runtime_image_ref: str,
    machine: str,
    runtime_input_sha256: str = "",
    runtime_build_source_sha: str = "",
    submission_key: str | None = None,
    goal_path: str | None = None,
    goal_sha256: str | None = None,
    recipe_path: str | None = None,
    recipe_sha256: str | None = None,
    repo_git_commit: str | None = None,
    repo_dirty: bool = False,
    seeds: Sequence[int] = (),
    checkpoint_eval_backend: str | None = None,
    runtime_config_validator: Callable[[Mapping[str, Any]], Any] | None = None,
    _modal_readiness_validated: bool = False,
    readiness_barrier: Callable[[], Any] | None = None,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    document = copy.deepcopy(dict(document))
    train_config = dict(document.get("train_config") or {})
    declared_training_acceptance = accepts_first_training_success(train_config)
    backend = resolve_checkpoint_eval_backend(
        train_config,
        checkpoint_eval_backend=checkpoint_eval_backend,
    )
    train_config["checkpoint_eval_backend"] = backend
    train_config["runtime_input_sha256"] = str(runtime_input_sha256).strip()
    train_config["runtime_build_source_sha"] = str(runtime_build_source_sha).strip()
    train_config["source_sha"] = str(repo_git_commit or "").strip()
    if backend == "none":
        train_config["early_stop"] = None
        train_config["checkpoint_eval_stages"] = []
        train_config["stop_on_acceptance"] = False
        train_config.pop("checkpoint_eval_acceptance", None)
        train_config.pop("checkpoint_eval_contract", None)
        train_config.pop("checkpoint_eval_asset_manifest", None)
        tags = [str(tag) for tag in document.get("tags", [])]
        if "checkpoint_eval_backend:none" not in tags:
            tags.append("checkpoint_eval_backend:none")
        document["tags"] = tags
    document["train_config"] = train_config
    validate_materialized_train_recipe(
        document,
        allow_no_eval_backend=backend == "none"
        and (checkpoint_eval_backend == "none" or declared_training_acceptance),
    )
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
        goal_path=goal_path,
        goal_sha256=goal_sha256,
        recipe_path=recipe_path,
        recipe_sha256=recipe_sha256,
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
    modal_readiness_validated = backend != "modal" or _modal_readiness_validated
    if not modal_readiness_validated:
        require_modal_eval_ready(
            runtime_image_ref=runtime_image_ref,
            game=str(train_config.get("game") or ""),
            env_provider=str(train_config.get("env_provider") or ""),
        )
        modal_readiness_validated = True
    rows = []
    with conn:
        acquire_fleet_admission_xact_lock(conn)
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
            run_name = _format_default_run_name(batch_id, label=document_slug, seed=seed, utc=utc)
            run_description = _format_queue_template(
                document.get("description"),
                seed=seed,
                recipe_id=document_slug,
                utc=utc,
                batch_id=batch_id,
                campaign_id=campaign_id or "",
            )
            if isinstance(document.get("_composition"), Mapping):
                source_commit = str(repo_git_commit or "").strip()
                if not source_commit:
                    raise ValueError(
                        "repo_git_commit is required when enqueuing a composed policy recipe"
                    )
                recipe_payload = build_recipe_document(
                    document,
                    repo_root=repo_root or Path(__file__).resolve().parents[2],
                    source_commit=source_commit,
                    run_description=run_description,
                    seed=seed,
                    runtime_image_ref=runtime_image_ref,
                )
            else:
                # Compatibility for pre-composition queue rows and focused unit fixtures.
                recipe_payload = compiled_recipe_payload(document)
            row = enqueue_train_job(
                conn,
                goal_slug=goal_slug,
                goal_path=goal_path,
                goal_sha256=goal_sha256,
                recipe_slug=document_slug,
                recipe_path=recipe_path,
                recipe_sha256=recipe_sha256,
                repo_git_commit=repo_git_commit,
                repo_dirty=repo_dirty,
                recipe_payload=recipe_payload,
                runtime_image_ref=runtime_image_ref,
                runtime_input_sha256=runtime_input_sha256,
                runtime_build_source_sha=runtime_build_source_sha,
                machine=machine,
                train_config=train_config,
                batch_id=batch_id,
                campaign_id=campaign_id,
                submission_key=submission_key,
                submission_ordinal=ordinal,
                request_hash=request_hash,
                run_name=run_name,
                run_description=run_description,
                seed=seed,
                wandb_group=batch_id,
                wandb_tags=recipe_tags(document),
                manage_transaction=False,
                _modal_readiness_validated=modal_readiness_validated,
                runtime_config_validator=runtime_config_validator,
            )
            rows.append(row)
        if readiness_barrier is not None:
            readiness_barrier()
    return rows


def enqueue_train_job(
    conn,
    *,
    goal_slug: str,
    runtime_image_ref: str,
    machine: str,
    train_config: Mapping[str, Any],
    runtime_input_sha256: str = "",
    runtime_build_source_sha: str = "",
    goal_path: str | None = None,
    goal_sha256: str | None = None,
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
    requested_config["runtime_input_sha256"] = str(runtime_input_sha256).strip()
    requested_config["runtime_build_source_sha"] = str(runtime_build_source_sha).strip()
    requested_config["source_sha"] = str(repo_git_commit or "").strip()
    requested_config["goal_path"] = str(goal_path or "")
    requested_config["goal_sha256"] = str(goal_sha256 or "").strip()
    requested_config["recipe_path"] = str(recipe_path or "")
    requested_config["recipe_sha256"] = str(recipe_sha256 or "").strip()
    composition = (recipe_payload or {}).get("_composition")
    requested_config["recipe_composition"] = (
        copy.deepcopy(dict(composition)) if isinstance(composition, Mapping) else {}
    )
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
    config["telemetry_transport"] = "neon_mailbox_v1"
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    if str(config.get("checkpoint_eval_backend") or "local") == "modal":
        if not _modal_readiness_validated:
            require_modal_eval_ready(
                runtime_image_ref=runtime_image_ref,
                game=str(config.get("game") or ""),
                env_provider=str(config.get("env_provider") or ""),
            )
        provider = str(config.get("env_provider") or "").strip()
        requires_rom_asset = (
            resolve_env_provider(provider).uses_stable_retro_roms if provider else True
        )
        if requires_rom_asset and not config.get("checkpoint_eval_asset_manifest"):
            config["checkpoint_eval_asset_manifest"] = asset_manifest_for_game(
                str(config.get("game") or "")
            )
        elif not requires_rom_asset:
            config.pop("checkpoint_eval_asset_manifest", None)
        config.setdefault("checkpoint_eval_seed", EVAL_SEED_START)
        config.setdefault("checkpoint_eval_seed_protocol", SEED_PROTOCOL)
        if config.get("stop_on_acceptance"):
            config["checkpoint_eval_contract"] = checkpoint_eval_contract_from_train_config(config)
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
        normalized_tags = [tag for tag in normalized_tags if not tag.startswith("retry_of_job_id:")]
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
            "goal_path": str(goal_path or ""),
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
        acquire_fleet_admission_xact_lock(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO train_jobs (
                  goal_slug, goal_path, goal_sha256, recipe_slug, recipe_path, recipe_sha256,
                  repo_git_commit,
                  repo_dirty, recipe_payload_json, runtime_image_ref, machine, train_config,
                  telemetry_transport,
                  batch_id, campaign_id, submission_key, submission_ordinal, request_hash,
                  retry_of_job_id, run_name, run_description, seed,
                  wandb_group, wandb_tags
                )
                VALUES (
                  %(goal_slug)s, %(goal_path)s, %(goal_sha256)s, %(recipe_slug)s,
                  %(recipe_path)s, %(recipe_sha256)s,
                  %(repo_git_commit)s, %(repo_dirty)s, %(recipe_payload_json)s,
                  %(runtime_image_ref)s, %(machine)s, %(train_config)s,
                  %(telemetry_transport)s,
                  %(batch_id)s, %(campaign_id)s, %(submission_key)s,
                  %(submission_ordinal)s, %(request_hash)s,
                  %(retry_of_job_id)s, %(run_name)s,
                  %(run_description)s, %(seed)s, %(wandb_group)s, %(wandb_tags)s
                )
                RETURNING *
                """,
                {
                    "goal_slug": goal_slug,
                    "goal_path": goal_path,
                    "goal_sha256": goal_sha256,
                    "recipe_slug": recipe_slug,
                    "recipe_path": recipe_path,
                    "recipe_sha256": recipe_sha256,
                    "repo_git_commit": repo_git_commit,
                    "repo_dirty": bool(repo_dirty),
                    "recipe_payload_json": json_arg(dict(recipe_payload or {})),
                    "runtime_image_ref": runtime_image_ref,
                    "machine": machine,
                    "train_config": json_arg(config),
                    "telemetry_transport": str(config["telemetry_transport"]),
                    "batch_id": batch_id,
                    "campaign_id": campaign_id,
                    "submission_key": submission_key,
                    "submission_ordinal": submission_ordinal,
                    "request_hash": request_hash,
                    "retry_of_job_id": retry_of_job_id,
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
        acquire_fleet_admission_xact_lock(conn)
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
                ),
                inserted_attempt AS (
                  INSERT INTO worker_attempts (
                    attempt_id, train_job_id, task_kind, provider, status
                  )
                  SELECT launch_id, job_id, 'train', backend, 'launching'
                  FROM inserted_launch
                  ON CONFLICT (attempt_id) DO UPDATE
                  SET provider = EXCLUDED.provider
                  RETURNING *
                )
                SELECT
                  row_to_json(updated) AS job_json,
                  row_to_json(inserted_launch) AS launch_json,
                  row_to_json(inserted_attempt) AS attempt_json
                FROM updated, inserted_launch, inserted_attempt
                """,
                params,
            )
            row = cur.fetchone()
            if not row:
                return None
            job = dict(row["job_json"])
            launch = dict(row["launch_json"])
            attempt_json = row.get("attempt_json") or {}
            launch["worker_attempt_id"] = str(attempt_json.get("attempt_id") or launch["launch_id"])
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
        acquire_fleet_admission_xact_lock(conn)
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
                UPDATE worker_attempts
                SET status = 'running',
                    provider_run_id = COALESCE(%(provider_run_id)s, provider_run_id),
                    started_at = COALESCE(started_at, now()),
                    last_heartbeat_at = now(),
                    error = NULL
                WHERE attempt_id = %(launch_id)s
                  AND status IN ('launching', 'running')
                """,
                {
                    "launch_id": launch_id,
                    "provider_run_id": provider_run_id,
                },
            )
            cur.execute(
                """
                UPDATE train_jobs
                SET status = 'starting',
                    started_at = COALESCE(started_at, now()),
                    live_publication_status = CASE
                      WHEN NOT COALESCE((train_config->>'wandb')::boolean, FALSE)
                        THEN 'disabled'
                      WHEN telemetry_transport = 'neon_mailbox_v1'
                        THEN 'pending'
                      ELSE 'live'
                    END,
                    live_publication_error = NULL,
                    live_publication_next_retry_at = NULL
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
    wandb_enabled = bool(readiness.get("wandb_enabled", True))
    telemetry_transport = str(readiness.get("telemetry_transport") or "legacy_local")
    wandb_run_id = str(readiness.get("wandb_run_id") or "").strip()
    wandb_url = str(readiness.get("wandb_url") or "").strip()
    if (
        wandb_enabled
        and telemetry_transport != "neon_mailbox_v1"
        and (not wandb_run_id or not wandb_url.startswith("https://wandb.ai/"))
    ):
        raise ValueError("training readiness requires a W&B run id and URL")
    learner_ready_at = str(readiness.get("learner_ready_at") or "").strip() or None
    wandb_ready_at = str(readiness.get("wandb_ready_at") or "").strip() or None
    for label, value in (
        ("learner_ready_at", learner_ready_at),
        ("wandb_ready_at", wandb_ready_at),
    ):
        if value is not None:
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"training readiness {label} must be ISO-8601") from exc
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE train_jobs AS job
                SET status = 'running',
                    learner_ready_at = COALESCE(
                        learner_ready_at, %(learner_ready_at)s::timestamptz, now()
                    ),
                    wandb_ready_at = COALESCE(
                        wandb_ready_at,
                        CASE WHEN %(wandb_enabled)s AND NOT %(mailbox_transport)s
                          THEN COALESCE(%(wandb_ready_at)s::timestamptz, now())
                          ELSE NULL
                        END
                    ),
                    ready_at = COALESCE(ready_at, now()),
                    wandb_run_id = COALESCE(%(wandb_run_id)s, wandb_run_id),
                    wandb_url = COALESCE(%(wandb_url)s, wandb_url),
                    live_publication_status = CASE
                      WHEN NOT %(wandb_enabled)s THEN 'disabled'
                      WHEN %(mailbox_transport)s THEN 'pending'
                      ELSE 'live'
                    END,
                    error = NULL
                FROM job_launches AS launch
                WHERE launch.launch_id = %(launch_id)s
                  AND launch.job_id = job.id
                  AND launch.state = 'running'
                  AND job.status = 'starting'
                  AND COALESCE((job.train_config->>'wandb')::boolean, FALSE)
                    = %(wandb_enabled)s
                RETURNING job.*
                """,
                {
                    "launch_id": launch_id,
                    "wandb_run_id": wandb_run_id or None,
                    "wandb_url": wandb_url or None,
                    "wandb_enabled": wandb_enabled,
                    "mailbox_transport": telemetry_transport == "neon_mailbox_v1",
                    "learner_ready_at": learner_ready_at,
                    "wandb_ready_at": wandb_ready_at,
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
                message=(
                    "learner and Neon telemetry relay ready"
                    if telemetry_transport == "neon_mailbox_v1"
                    else "learner and W&B publisher ready"
                    if wandb_enabled
                    else "learner ready; W&B publication disabled"
                ),
                metadata={
                    "launch_id": launch_id,
                    "wandb_enabled": wandb_enabled,
                    "telemetry_transport": telemetry_transport,
                    "wandb_run_id": wandb_run_id,
                    "wandb_url": wandb_url,
                    "learner_ready_at": learner_ready_at,
                    "wandb_ready_at": wandb_ready_at,
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
                   OR (
                     status = 'finalizing'
                     AND telemetry_transport <> 'neon_mailbox_v1'
                     AND live_publication_status IN ('pending', 'live')
                   )
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
            cur.execute("SELECT to_regclass('train_jobs') AS train_jobs")
            schema_row = cur.fetchone()
            if not schema_row or schema_row["train_jobs"] is None:
                return 0
            cur.execute(
                """
                SELECT
                  (SELECT COUNT(*) FROM train_jobs
                   WHERE status IN ('pending', 'launching', 'starting', 'running', 'finalizing'))
                  +
                  (SELECT COUNT(*) FROM eval_jobs
                   WHERE status IN ('pending', 'dispatching', 'submitted', 'blocked_budget'))
                  +
                  (SELECT COUNT(*) FROM job_launches
                   WHERE state IN ('launching', 'running'))
                  +
                  (SELECT COUNT(*) FROM eval_attempts
                   WHERE status IN ('dispatching', 'submitted'))
                  +
                  (SELECT COUNT(*) FROM worker_attempts
                   WHERE status IN ('launching', 'running'))
                  AS count
                """
            )
            row = cur.fetchone()
        return int(row["count"] if row else 0)
    finally:
        if owned_connection:
            conn.close()


def claim_live_publication_recovery(conn, *, machine: str) -> dict[str, Any] | None:
    """Claim one CPU-only publisher recovery after its Docker launch is terminal."""

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.id
                FROM train_jobs t
                JOIN job_launches l ON l.job_id = t.id
                WHERE t.machine = %(machine)s
                  AND t.status = 'finalizing'
                  AND t.telemetry_transport <> 'neon_mailbox_v1'
                  AND l.state = 'succeeded'
                  AND t.live_publication_status IN ('pending', 'live')
                  AND (
                    t.live_publication_next_retry_at IS NULL
                    OR t.live_publication_next_retry_at <= now()
                  )
                ORDER BY t.id
                FOR UPDATE OF t SKIP LOCKED
                LIMIT 1
                """,
                {"machine": normalize_machine(machine)},
            )
            candidate = cur.fetchone()
            if not candidate:
                return None
            cur.execute(
                """
                UPDATE train_jobs t
                SET live_publication_status = 'live',
                  live_publication_attempts = live_publication_attempts + 1,
                  live_publication_next_retry_at = now() + interval '20 minutes',
                  live_publication_error = NULL
                FROM job_launches l
                WHERE t.id = %(job_id)s AND l.job_id = t.id
                RETURNING t.*, l.launch_id, l.output_uri, l.state AS launch_state
                """,
                {"job_id": int(candidate["id"])},
            )
            return dict(cur.fetchone())


def finish_live_publication_recovery(
    conn,
    *,
    job_id: int,
    error: str | None,
    max_attempts: int = 3,
) -> dict[str, Any]:
    with conn:
        with conn.cursor() as cur:
            if error is None:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET status = CASE
                        WHEN COALESCE(train_config->>'checkpoint_eval_backend', 'local') <> 'modal'
                          THEN 'succeeded' ELSE status END,
                      finished_at = CASE
                        WHEN COALESCE(train_config->>'checkpoint_eval_backend', 'local') <> 'modal'
                          THEN now() ELSE finished_at END,
                      live_publication_status = 'complete',
                      live_publication_next_retry_at = NULL,
                      live_publication_error = NULL
                    WHERE id = %(job_id)s AND status = 'finalizing'
                    RETURNING *
                    """,
                    {"job_id": int(job_id)},
                )
            else:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET status = CASE
                        WHEN live_publication_attempts >= %(max_attempts)s
                          THEN 'finalization_failed' ELSE status END,
                      finished_at = CASE
                        WHEN live_publication_attempts >= %(max_attempts)s
                          THEN now() ELSE finished_at END,
                      error = CASE
                        WHEN live_publication_attempts >= %(max_attempts)s
                          THEN %(error)s ELSE error END,
                      live_publication_status = CASE
                        WHEN live_publication_attempts >= %(max_attempts)s
                          THEN 'failed' ELSE 'pending' END,
                      live_publication_next_retry_at = CASE
                        WHEN live_publication_attempts >= %(max_attempts)s THEN NULL
                        ELSE now() + (LEAST(300, 120 * live_publication_attempts)
                          * interval '1 second') END,
                      live_publication_error = %(error)s
                    WHERE id = %(job_id)s AND status = 'finalizing'
                    RETURNING *
                    """,
                    {
                        "job_id": int(job_id),
                        "max_attempts": int(max_attempts),
                        "error": str(error)[:4000],
                    },
                )
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"publication recovery job {job_id} is no longer finalizing")
            result = dict(row)
            record_job_event(
                conn,
                job_id=int(job_id),
                event_type="live_publication_recovery",
                message=("publisher recovery complete" if error is None else str(error)[:4000]),
                metadata={
                    "status": result.get("live_publication_status"),
                    "attempts": result.get("live_publication_attempts"),
                },
            )
            return result


def runtime_image_retry_state(
    conn, *, machine: str, runtime_image_ref: str
) -> dict[str, Any] | None:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM runtime_image_states
            WHERE machine = %(machine)s AND runtime_image_ref = %(runtime_image_ref)s
            """,
            {
                "machine": normalize_machine(machine),
                "runtime_image_ref": normalize_runtime_image_ref(runtime_image_ref),
            },
        )
        row = cur.fetchone()
    return dict(row) if row else None


def record_runtime_image_failure(
    conn,
    *,
    machine: str,
    runtime_image_ref: str,
    error: str,
    base_retry_seconds: int = 30,
    max_retry_seconds: int = 900,
) -> dict[str, Any]:
    machine = normalize_machine(machine)
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runtime_image_states (
                  machine, runtime_image_ref, retry_count, next_retry_at,
                  last_error, last_attempt_at
                ) VALUES (
                  %(machine)s, %(runtime_image_ref)s, 1,
                  now() + (%(base_retry_seconds)s * interval '1 second'),
                  %(error)s, now()
                )
                ON CONFLICT (machine, runtime_image_ref) DO UPDATE
                SET retry_count = runtime_image_states.retry_count + 1,
                    next_retry_at = now() + (
                      LEAST(
                        %(max_retry_seconds)s,
                        %(base_retry_seconds)s * power(
                          2, LEAST(runtime_image_states.retry_count, 5)
                        )
                      ) * interval '1 second'
                    ),
                    last_error = EXCLUDED.last_error,
                    last_attempt_at = now(),
                    updated_at = now()
                RETURNING *
                """,
                {
                    "machine": machine,
                    "runtime_image_ref": runtime_image_ref,
                    "error": str(error)[:4000],
                    "base_retry_seconds": max(1, int(base_retry_seconds)),
                    "max_retry_seconds": max(1, int(max_retry_seconds)),
                },
            )
            return dict(cur.fetchone() or {})


def reset_runtime_image_retry(conn, *, machine: str, runtime_image_ref: str) -> dict[str, Any]:
    machine = normalize_machine(machine)
    runtime_image_ref = normalize_runtime_image_ref(runtime_image_ref)
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runtime_image_states (
                  machine, runtime_image_ref, retry_count, next_retry_at,
                  last_error, last_attempt_at, last_ready_at
                ) VALUES (
                  %(machine)s, %(runtime_image_ref)s, 0, NULL, NULL, now(), now()
                )
                ON CONFLICT (machine, runtime_image_ref) DO UPDATE
                SET retry_count = 0, next_retry_at = NULL, last_error = NULL,
                    last_attempt_at = now(), last_ready_at = now(), updated_at = now()
                RETURNING *
                """,
                {"machine": machine, "runtime_image_ref": runtime_image_ref},
            )
            return dict(cur.fetchone() or {})


def next_pending_train_job(
    conn,
    *,
    machine: str,
    exclude_runtime_image_refs: Sequence[str] = (),
) -> dict[str, Any] | None:
    excluded = [normalize_runtime_image_ref(value) for value in exclude_runtime_image_refs]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT job.*
            FROM train_jobs AS job
            LEFT JOIN machine_controls AS control ON control.machine = job.machine
            LEFT JOIN runtime_image_states AS image_state
              ON image_state.machine = job.machine
             AND image_state.runtime_image_ref = job.runtime_image_ref
            WHERE job.machine = %(machine)s
              AND job.status = 'pending'
              AND job.cancel_requested = FALSE
              AND NOT (job.runtime_image_ref = ANY(%(excluded_runtime_images)s))
              AND (
                SELECT pg_total_relation_size('metric_batches')
                     + pg_total_relation_size('attempt_events')
                     + pg_total_relation_size('attempt_commands')
              ) < 5368709120
              AND COALESCE(control.drained, FALSE) = FALSE
              AND (
                image_state.next_retry_at IS NULL
                OR image_state.next_retry_at <= now()
              )
              AND NOT EXISTS (
                SELECT 1 FROM job_launches AS launch WHERE launch.job_id = job.id
              )
            ORDER BY job.id
            LIMIT 1
            """,
            {
                "machine": normalize_machine(machine),
                "excluded_runtime_images": excluded,
            },
        )
        row = cur.fetchone()
    return dict(row) if row else None


def job_payload_for_launch(job: Mapping[str, Any], launch: Mapping[str, Any]) -> dict[str, Any]:
    telemetry_transport = str(
        job.get("telemetry_transport")
        or (job.get("train_config") or {}).get("telemetry_transport")
        or "legacy_local"
    )
    payload = {
        "schema_version": 1,
        "job_kind": launch["job_kind"],
        "job": dict(job),
        "launch_id": launch["launch_id"],
        "machine": launch["machine"],
        "backend": launch["backend"],
        "runtime_image_ref": launch["runtime_image_ref"],
        "output_uri": launch["output_uri"],
        "telemetry": {
            "transport": telemetry_transport,
            "protocol_version": 1,
            "attempt_id": str(launch.get("worker_attempt_id") or launch["launch_id"]),
        },
    }
    assert_no_secrets(payload, label="job payload")
    return json_safe(payload)


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
    filters = ["status IN ('pending', 'launching', 'starting', 'running', 'finalizing')"]
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
                    status = CASE
                      WHEN status = 'pending' THEN 'canceled'
                      ELSE status
                    END,
                    finished_at = CASE
                      WHEN status = 'pending' THEN now()
                      ELSE finished_at
                    END,
                    live_publication_status = CASE
                      WHEN status = 'pending' THEN 'disabled'
                      ELSE live_publication_status
                    END
                WHERE {" AND ".join(filters)}
                RETURNING id
                """,
                params,
            )
            job_ids = [int(row["id"]) for row in cur.fetchall()]
            if job_ids:
                cur.execute(
                    """
                    UPDATE eval_jobs SET status = 'canceled', finished_at = now(),
                      updated_at = now(), error = 'training finalization canceled'
                    WHERE train_job_id = ANY(%(job_ids)s)
                      AND status IN ('pending', 'dispatching', 'submitted', 'blocked_budget')
                    """,
                    {"job_ids": job_ids},
                )
                cur.execute(
                    """
                    UPDATE eval_runs r
                    SET status = CASE
                          WHEN t.status = 'canceled' THEN 'canceled'
                          WHEN t.status = 'finalizing' THEN 'finalizing'
                          ELSE r.status
                        END,
                      outcome = 'canceled', updated_at = now(),
                      error = 'training cancellation requested'
                    FROM train_jobs t
                    WHERE t.id = r.train_job_id
                      AND r.train_job_id = ANY(%(job_ids)s)
                      AND r.status NOT IN ('complete', 'failed')
                    """,
                    {"job_ids": job_ids},
                )
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
    runtime_input_sha256: str | None = None,
    runtime_build_source_sha: str | None = None,
    repo_git_commit: str | None = None,
    runtime_config_validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Create a new pending job from a terminal job; execution is never retried in place."""
    submission_key = str(submission_key or f"retry-{job_id}-{uuid.uuid4().hex}")
    with conn:
        existing = _existing_submission(conn, submission_key=submission_key)
        if existing:
            retry_source = existing[0].get("retry_of_job_id")
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
                  AND status IN ('succeeded', 'failed', 'finalization_failed', 'canceled')
                """,
                {"job_id": int(job_id)},
            )
            preview = cur.fetchone()
    if not preview:
        raise ValueError(f"job {job_id} is not terminal or does not exist")
    preview = dict(preview)
    preview_config = dict(preview.get("train_config") or {})
    effective_runtime_image_ref = str(runtime_image_ref or preview["runtime_image_ref"])
    modal_readiness_validated = (
        str(preview_config.get("checkpoint_eval_backend") or "local") != "modal"
    )
    if not modal_readiness_validated:
        require_modal_eval_ready(
            runtime_image_ref=effective_runtime_image_ref,
            game=str(preview_config.get("game") or ""),
            env_provider=str(preview_config.get("env_provider") or ""),
        )
        modal_readiness_validated = True
    with conn:
        acquire_fleet_admission_xact_lock(conn)
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
                retry_source = existing[0].get("retry_of_job_id")
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
                  AND status IN ('succeeded', 'failed', 'finalization_failed', 'canceled')
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
            goal_path=source.get("goal_path"),
            goal_sha256=source.get("goal_sha256"),
            recipe_slug=source.get("recipe_slug"),
            recipe_path=source.get("recipe_path"),
            recipe_sha256=source.get("recipe_sha256"),
            repo_git_commit=repo_git_commit or source.get("repo_git_commit"),
            repo_dirty=False if repo_git_commit else bool(source.get("repo_dirty")),
            recipe_payload=source.get("recipe_payload_json") or {},
            runtime_image_ref=runtime_image_ref or source["runtime_image_ref"],
            runtime_input_sha256=(
                runtime_input_sha256
                if runtime_input_sha256 is not None
                else str(source.get("train_config", {}).get("runtime_input_sha256") or "")
            ),
            runtime_build_source_sha=(
                runtime_build_source_sha
                if runtime_build_source_sha is not None
                else str(source.get("train_config", {}).get("runtime_build_source_sha") or "")
            ),
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


def retry_train_job_finalization(conn, *, job_id: int) -> dict[str, Any]:
    """Reopen post-train work or restamp a completed publication without retraining."""

    with conn:
        acquire_fleet_admission_xact_lock(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT t.*, l.state AS launch_state,
                  r.status AS eval_status, r.outcome AS eval_outcome,
                  r.complete_announcement_seen,
                  r.artifacts_projected_at, r.promoted_eval_job_id,
                  r.promoted_artifact_projected_at,
                  (SELECT COUNT(*) FROM artifact_announcement_ledger ledger
                    WHERE ledger.train_job_id=t.id
                      AND ledger.disposition='ready'
                      AND NOT EXISTS (
                        SELECT 1 FROM artifact_publication_receipts receipt
                        WHERE receipt.train_job_id=ledger.train_job_id
                          AND receipt.ledger_id=ledger.ledger_id
                          AND receipt.role='availability'
                          AND receipt.promotion_revision=0
                      )) AS missing_artifact_receipts,
                  (SELECT COUNT(*) FROM eval_jobs j
                    WHERE j.train_job_id=t.id
                      AND j.status IN ('pending','dispatching','submitted','blocked_budget'))
                    AS active_eval_jobs,
                  (SELECT COUNT(*) FROM eval_attempts a
                    JOIN eval_jobs j ON j.id=a.eval_job_id
                    WHERE j.train_job_id=t.id AND a.status IN ('dispatching','submitted'))
                    AS active_eval_attempts,
                  (SELECT COUNT(*) FROM worker_attempts w
                    WHERE w.train_job_id=t.id AND w.task_kind='eval'
                      AND w.status IN ('launching','running')) AS active_eval_workers,
                  (SELECT COUNT(*) FROM worker_attempts w
                    WHERE w.train_job_id=t.id AND w.task_kind='train'
                      AND w.status IN ('launching','running')) AS active_train_workers
                FROM train_jobs t
                JOIN job_launches l ON l.job_id = t.id
                LEFT JOIN eval_runs r ON r.train_job_id = t.id
                WHERE t.id = %(job_id)s
                FOR UPDATE OF t, l
                """,
                {"job_id": int(job_id)},
            )
            source = cur.fetchone()
            if not source:
                raise ValueError(f"job {job_id} does not exist")
            if str(source["status"]) == "canceled":
                active_counts = {
                    key: int(source.get(key) or 0)
                    for key in (
                        "active_eval_jobs",
                        "active_eval_attempts",
                        "active_eval_workers",
                        "active_train_workers",
                    )
                }
                if (
                    str(source.get("launch_state") or "") != "canceled"
                    or not bool(source.get("cancel_requested"))
                    or source.get("process_exited_at") is None
                    or str(source.get("eval_outcome") or "") != "canceled"
                    or not bool(source.get("complete_announcement_seen"))
                    or str(source.get("eval_status") or "") != "canceled"
                    or any(active_counts.values())
                ):
                    raise ValueError(
                        f"canceled job {job_id} lacks publication-only recovery evidence"
                    )
                if str(source.get("live_publication_status") or "") != "complete":
                    raise ValueError(
                        f"canceled job {job_id} does not have a complete publication state"
                    )
                if int(source.get("missing_artifact_receipts") or 0) == 0:
                    raise ValueError(f"canceled job {job_id} has no missing artifact receipts")
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET live_publication_status = 'pending',
                      live_publication_attempts = 0,
                      live_publication_next_retry_at = now(),
                      live_publication_error = NULL
                    WHERE id = %(job_id)s AND status = 'canceled'
                      AND live_publication_status = 'complete'
                    RETURNING *
                    """,
                    {"job_id": int(job_id)},
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError(f"canceled job {job_id} changed while reopening publication")
                record_job_event(
                    conn,
                    job_id=int(job_id),
                    event_type="finalization_retried",
                    message="operator requested canceled-run artifact publication recovery",
                    metadata={"launch_state": "canceled", "publication_only": True},
                )
                return dict(row)
            if str(source["status"]) == "succeeded":
                if str(source["launch_state"]) != "succeeded":
                    raise ValueError(f"job {job_id} does not have a successful training launch")
                if str(source.get("live_publication_status") or "") != "complete":
                    raise ValueError(
                        f"succeeded job {job_id} does not have a complete W&B publication"
                    )
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET live_publication_status = 'finishing',
                      live_publication_attempts = 0,
                      live_publication_next_retry_at = now(),
                      live_publication_error = NULL
                    WHERE id = %(job_id)s AND status = 'succeeded'
                      AND live_publication_status = 'complete'
                    RETURNING *
                    """,
                    {"job_id": int(job_id)},
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError(f"job {job_id} changed while restamping publication")
                record_job_event(
                    conn,
                    job_id=int(job_id),
                    event_type="finalization_retried",
                    message="operator requested canonical W&B publication restamp",
                    metadata={"launch_state": "succeeded", "restamp_only": True},
                )
                return dict(row)
            if str(source["status"]) != "finalization_failed":
                raise ValueError(f"job {job_id} is not finalization_failed or succeeded")
            launch_state = str(source.get("launch_state") or "")
            publication_failed = str(source.get("live_publication_status") or "") == "failed"
            active_counts = {
                key: int(source.get(key) or 0)
                for key in (
                    "active_eval_jobs",
                    "active_eval_attempts",
                    "active_eval_workers",
                    "active_train_workers",
                )
            }
            no_active_work = not any(active_counts.values())
            canceled_publication_only = (
                launch_state == "canceled"
                and publication_failed
                and bool(source.get("cancel_requested"))
                and source.get("process_exited_at") is not None
                and str(source.get("eval_outcome") or "") == "canceled"
                and bool(source.get("complete_announcement_seen"))
                and str(source.get("eval_status") or "") in {"finalizing", "canceled"}
                and no_active_work
            )
            successful_publication_only = (
                launch_state == "succeeded"
                and publication_failed
                and source.get("process_exited_at") is not None
                and no_active_work
                and (
                    (
                        str(source.get("eval_outcome") or "") in {"accepted", "not_accepted"}
                        and bool(source.get("complete_announcement_seen"))
                        and source.get("artifacts_projected_at") is not None
                        and (
                            source.get("promoted_eval_job_id") is None
                            or source.get("promoted_artifact_projected_at") is not None
                        )
                    )
                    or str(
                        (source.get("train_config") or {}).get("checkpoint_eval_backend") or "local"
                    )
                    != "modal"
                )
            )
            publication_only = canceled_publication_only or successful_publication_only
            if launch_state == "canceled" and not canceled_publication_only:
                missing = [key for key, value in active_counts.items() if value]
                raise ValueError(
                    f"canceled job {job_id} lacks publication-only recovery evidence"
                    + (f"; active={','.join(missing)}" if missing else "")
                )
            if launch_state not in {"succeeded", "canceled"}:
                raise ValueError(
                    f"job {job_id} launch state {launch_state!r} cannot retry finalization"
                )
            cur.execute(
                """
                WITH residual_streams AS MATERIALIZED (
                  SELECT DISTINCT s.stream_id
                  FROM metric_batches b
                  JOIN metric_streams s ON s.stream_id = b.stream_id
                  JOIN worker_attempts w ON w.attempt_id = s.attempt_id
                  WHERE w.train_job_id = %(job_id)s
                ), reset_streams AS (
                  UPDATE metric_streams s
                  SET submitted_sequence = published_sequence, updated_at = now()
                  FROM residual_streams r
                  WHERE s.stream_id = r.stream_id
                  RETURNING s.stream_id
                ), reset_batches AS (
                  UPDATE metric_batches b
                  SET lease_owner = NULL, lease_expires_at = NULL,
                    attempts = 0, last_error = NULL, submitted_at = NULL
                  FROM metric_streams s
                  JOIN worker_attempts w ON w.attempt_id = s.attempt_id
                  WHERE b.stream_id = s.stream_id
                    AND w.train_job_id = %(job_id)s
                  RETURNING b.id
                )
                SELECT
                  (SELECT COUNT(*) FROM reset_streams) AS residual_streams,
                  (SELECT COUNT(*) FROM reset_batches) AS residual_batches,
                  (SELECT COUNT(*) FROM metric_streams s
                    JOIN worker_attempts w ON w.attempt_id=s.attempt_id
                    WHERE w.train_job_id=%(job_id)s
                      AND (s.final_sequence IS NULL OR s.published_sequence < s.final_sequence)
                  ) AS incomplete_streams
                """,
                {"job_id": int(job_id)},
            )
            reset = dict(cur.fetchone() or {})
            residual_batches = int(reset.get("residual_batches") or 0)
            incomplete_streams = int(reset.get("incomplete_streams") or 0)
            if residual_batches == 0 and incomplete_streams:
                raise RuntimeError(
                    f"job {job_id} has {incomplete_streams} incomplete stream(s) without retained batches"
                )
            if publication_only:
                cur.execute(
                    """
                    UPDATE train_jobs
                    SET status = 'finalizing', finished_at = NULL, error = NULL,
                      live_publication_status = %(publication_status)s,
                      live_publication_attempts = 0,
                      live_publication_next_retry_at = now(),
                      live_publication_error = NULL
                    WHERE id = %(job_id)s AND status = 'finalization_failed'
                    RETURNING *
                    """,
                    {
                        "job_id": int(job_id),
                        "publication_status": "pending" if residual_batches else "finishing",
                    },
                )
                row = cur.fetchone()
                if not row:
                    raise RuntimeError(f"job {job_id} changed while retrying publication")
                record_job_event(
                    conn,
                    job_id=int(job_id),
                    event_type="finalization_retried",
                    message="operator reopened publication-only finalization work",
                    metadata={
                        "launch_state": launch_state,
                        "recovery_mode": "publication_only",
                        "residual_streams": int(reset.get("residual_streams") or 0),
                        "residual_batches": residual_batches,
                        "publication_delivery": "at_least_once",
                    },
                )
                return dict(row)
            cur.execute(
                """
                UPDATE eval_runs
                SET status = CASE WHEN complete_announcement_seen
                    THEN 'finalizing' ELSE 'active' END,
                  contract_json = contract_json || jsonb_build_object(
                    'checkpoint_eval_seed', COALESCE(
                      NULLIF(contract_json->>'checkpoint_eval_seed', '')::integer,
                      %(checkpoint_eval_seed)s
                    ),
                    'checkpoint_eval_seed_protocol', COALESCE(
                      NULLIF(contract_json->>'checkpoint_eval_seed_protocol', ''),
                      %(checkpoint_eval_seed_protocol)s
                    )
                  ),
                  artifact_projection_attempts = 0,
                  artifact_projection_next_retry_at = NULL,
                  error = NULL,
                  updated_at = now()
                WHERE train_job_id = %(job_id)s AND status = 'failed'
                """,
                {
                    "job_id": int(job_id),
                    "checkpoint_eval_seed": EVAL_SEED_START,
                    "checkpoint_eval_seed_protocol": SEED_PROTOCOL,
                },
            )
            cur.execute(
                """
                UPDATE eval_jobs
                SET status = CASE WHEN status = 'failed' THEN 'pending' ELSE status END,
                  retry_round = CASE WHEN status = 'failed' THEN retry_round + 1 ELSE retry_round END,
                  finished_at = CASE WHEN status = 'failed' THEN NULL ELSE finished_at END,
                  error = CASE WHEN status = 'failed' THEN NULL ELSE error END,
                  projection_attempts = 0,
                  projection_next_retry_at = NULL,
                  projection_error = NULL,
                  updated_at = now()
                WHERE train_job_id = %(job_id)s AND projected_at IS NULL
                """,
                {"job_id": int(job_id)},
            )
            cur.execute(
                """
                UPDATE attempt_events e
                SET attempts = 0, next_retry_at = NULL, last_error = NULL
                FROM worker_attempts w
                WHERE w.attempt_id = e.attempt_id
                  AND w.train_job_id = %(job_id)s
                  AND e.event_type IN (
                    'checkpoint_ready', 'checkpoint_tombstone',
                    'checkpoint_stream_closed'
                  )
                  AND (
                    e.attempts <> 0
                    OR e.next_retry_at IS NOT NULL
                    OR e.last_error IS NOT NULL
                  )
                """,
                {"job_id": int(job_id)},
            )
            reset_attempt_events = cur.rowcount
            cur.execute(
                """
                UPDATE train_jobs
                SET status = 'finalizing', finished_at = NULL, error = NULL,
                  cancel_requested = FALSE,
                  live_publication_status = CASE
                    WHEN live_publication_status = 'failed' THEN 'pending'
                    ELSE live_publication_status
                  END,
                  live_publication_attempts = CASE
                    WHEN live_publication_status = 'failed' THEN 0
                    ELSE live_publication_attempts
                  END,
                  live_publication_next_retry_at = CASE
                    WHEN live_publication_status IN ('failed', 'pending') THEN now()
                    ELSE NULL
                  END,
                  live_publication_error = NULL
                WHERE id = %(job_id)s AND status = 'finalization_failed'
                RETURNING *
                """,
                {"job_id": int(job_id)},
            )
            row = cur.fetchone()
            if not row:
                raise RuntimeError(f"job {job_id} changed while retrying finalization")
            record_job_event(
                conn,
                job_id=int(job_id),
                event_type="finalization_retried",
                message="operator reopened failed finalization work",
                metadata={
                    "launch_state": "succeeded",
                    "recovery_mode": "full_post_training",
                    "residual_streams": int(reset.get("residual_streams") or 0),
                    "residual_batches": int(reset.get("residual_batches") or 0),
                    "reset_attempt_events": reset_attempt_events,
                    "publication_delivery": "at_least_once",
                },
            )
            return dict(row)


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
    live_publication = result.get("live_publication")
    live_publication = dict(live_publication) if isinstance(live_publication, Mapping) else {}
    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                  launch.*,
                  job.cancel_requested,
                  job.status AS job_status,
                  job.machine AS job_machine,
                  job.train_config AS job_train_config
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
            launch_status = (
                "canceled"
                if bool(launch["cancel_requested"])
                else _terminal_status_from_result(result)
            )
            if launch["state"] in {"succeeded", "failed", "canceled"}:
                if launch["state"] == launch_status:
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
                    "state": launch_status,
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
            train_config = dict(launch.get("job_train_config") or {})
            wandb_enabled = bool(train_config.get("wandb", False))
            publication_status = str(live_publication.get("status") or "").strip()
            if publication_status not in {"pending", "complete", "disabled", "failed"}:
                publication_status = "pending" if wandb_enabled else "disabled"
            if wandb_enabled and publication_status == "disabled":
                publication_status = "pending"
            if not wandb_enabled:
                publication_status = "disabled"
            publication_error = str(live_publication.get("error") or "").strip() or None
            publication_attempts = max(0, int(live_publication.get("attempts") or 0))
            eval_backend = str(train_config.get("checkpoint_eval_backend") or "local")
            telemetry_transport = str(train_config.get("telemetry_transport") or "legacy_local")
            requires_finalization = (
                telemetry_transport == "neon_mailbox_v1"
                or eval_backend == "modal"
                or publication_status not in {"complete", "disabled"}
            )
            if launch_status == "canceled":
                job_status = "finalizing" if requires_finalization else "canceled"
            elif launch_status != "succeeded":
                job_status = "failed"
            elif (
                telemetry_transport == "neon_mailbox_v1"
                or eval_backend == "modal"
                or publication_status not in {"complete", "disabled"}
            ):
                job_status = "finalizing"
            else:
                job_status = "succeeded"
            cur.execute(
                """
                UPDATE train_jobs
                SET status = %(status)s,
                    finished_at = CASE WHEN %(status)s = 'finalizing' THEN NULL ELSE now() END,
                    process_exited_at = now(),
                    wandb_run_id = COALESCE(%(wandb_run_id)s, wandb_run_id),
                    wandb_url = COALESCE(%(wandb_url)s, wandb_url),
                    live_publication_status = %(publication_status)s,
                    live_publication_attempts = GREATEST(
                      live_publication_attempts, %(publication_attempts)s
                    ),
                    live_publication_next_retry_at = CASE
                      WHEN %(publication_status)s IN ('pending', 'failed') THEN now()
                      ELSE NULL
                    END,
                    live_publication_error = %(publication_error)s,
                    error = %(error)s
                WHERE id = %(job_id)s
                  AND status IN ('launching', 'starting', 'running')
                RETURNING *
                """,
                {
                    "status": job_status,
                    "error": error,
                    "job_id": updated_launch["job_id"],
                    "launch_id": launch_id,
                    "wandb_run_id": train_payload.get("wandb_run_id"),
                    "wandb_url": train_payload.get("wandb_url"),
                    "publication_status": publication_status,
                    "publication_attempts": publication_attempts,
                    "publication_error": publication_error,
                },
            )
            job = cur.fetchone()
            if not job:
                raise RuntimeError(f"could not finish train job for launch {launch_id}")
            if launch_status == "canceled" and job_status == "finalizing":
                cur.execute(
                    """
                    UPDATE eval_runs
                    SET status = 'finalizing', outcome = 'canceled',
                      updated_at = now(), error = 'training canceled; finalization continues'
                    WHERE train_job_id = %(job_id)s
                      AND status NOT IN ('complete', 'failed')
                    """,
                    {"job_id": updated_launch["job_id"]},
                )
            cur.execute(
                """
                UPDATE worker_attempts
                SET status = %(status)s,
                    finished_at = now(),
                    token_expires_at = now(),
                    error = %(error)s
                WHERE attempt_id = %(launch_id)s
                  AND status IN ('launching', 'running')
                """,
                {
                    "launch_id": launch_id,
                    "status": launch_status,
                    "error": error,
                },
            )
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
                event_type=job_status,
                message=(
                    "training launch complete; finalization continues"
                    if job_status == "finalizing"
                    else error
                ),
                metadata={
                    "launch_id": launch_id,
                    "exit_code": exit_code,
                    "launch_status": launch_status,
                    "live_publication_status": publication_status,
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
              launch.finished_at AS training_finished_at,
              launch.error AS launch_error,
              launch.next_retry_at,
              launch.output_uri,
              launch.last_observed_at,
              eval_run.status AS eval_status,
              eval_run.error AS eval_error,
              eval_run.artifacts_projected_at AS published_at,
              eval_run.promoted_artifact_projected_at AS playable_at,
              eval_run.artifact_projection_attempts,
              eval_run.artifact_projection_next_retry_at,
              (SELECT COUNT(*) FROM artifact_announcement_ledger ledger
                WHERE ledger.train_job_id = job.id
                  AND ledger.disposition = 'ready') AS ready_artifact_count,
              (SELECT COUNT(*) FROM artifact_publication_receipts receipt
                WHERE receipt.train_job_id = job.id
                  AND receipt.role = 'availability') AS availability_receipt_count,
              (SELECT receipt.artifact_ref
                FROM artifact_publication_receipts receipt
                WHERE receipt.train_job_id = job.id
                  AND receipt.role = 'promotion'
                  AND receipt.promotion_revision = eval_run.promotion_revision
                  AND receipt.disposition = 'confirmed'
                ORDER BY receipt.confirmed_at DESC LIMIT 1) AS promoted_receipt_ref,
              (SELECT receipt.artifact_ref
                FROM artifact_publication_receipts receipt
                WHERE receipt.train_job_id = job.id
                  AND receipt.role = 'availability'
                  AND receipt.disposition = 'confirmed'
                ORDER BY receipt.checkpoint_step DESC,
                  CASE receipt.artifact_kind
                    WHEN 'final' THEN 2 WHEN 'interrupted' THEN 1 ELSE 0 END DESC,
                  receipt.confirmed_at DESC
                LIMIT 1) AS latest_receipt_ref,
              promoted.checkpoint_step AS promoted_step,
              promoted.checkpoint_uri AS promoted_checkpoint_uri,
              promoted.projected_at AS promoted_projection_at,
              promoted.projection_error AS promoted_projection_error,
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
            LEFT JOIN eval_runs AS eval_run ON eval_run.train_job_id = job.id
            LEFT JOIN eval_jobs AS promoted ON promoted.id = eval_run.promoted_eval_job_id
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
        if row.get("cancel_requested") and row.get("status") in {
            "launching",
            "starting",
            "running",
        }:
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
        selected_receipt_ref = row.get("promoted_receipt_ref") or row.get("latest_receipt_ref")
        all_availability_receipted = int(row.get("ready_artifact_count") or 0) > 0 and int(
            row.get("availability_receipt_count") or 0
        ) >= int(row.get("ready_artifact_count") or 0)
        if row.get("eval_status") is None:
            artifact_status = "not_applicable"
        elif all_availability_receipted and row.get("eval_status") == "complete":
            artifact_status = "published"
        elif selected_receipt_ref:
            artifact_status = "playable"
        elif row.get("published_at") is not None:
            artifact_status = "published"
        elif row.get("playable_at") is not None:
            artifact_status = "playable"
        elif row.get("eval_status") == "failed":
            artifact_status = "failed"
        else:
            artifact_status = "pending"
        row["artifact_status"] = artifact_status
        row["r2_checkpoint_uri"] = row.get("promoted_checkpoint_uri")
        row["wandb_artifact_ref"] = selected_receipt_ref
        wandb_url = str(row.get("wandb_url") or "")
        wandb_run_id = str(row.get("wandb_run_id") or "")
        if (
            row["wandb_artifact_ref"] is None
            and artifact_status in {"playable", "published"}
            and wandb_url
            and wandb_run_id
            and row.get("promoted_step") is not None
        ):
            parts = [part for part in urlparse(wandb_url).path.split("/") if part]
            if len(parts) >= 2:
                alias = f"step-{int(row['promoted_step'])}"
                row["wandb_artifact_ref"] = (
                    f"{parts[0]}/{parts[1]}/{wandb_run_id}-checkpoint:{alias}"
                )
        # Internal compatibility only. Public experiment JSON exposes the two stores separately.
        row["artifact_ref"] = row["wandb_artifact_ref"]
    counts: dict[str, int] = {}
    for row in jobs:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    return {"selector": selector, "counts": counts, "runs": jobs, "jobs": jobs}


def print_status(report: Mapping[str, Any]) -> None:
    print(f"selector: {json.dumps(report['selector'], sort_keys=True)}")
    print(f"counts: {json.dumps(report['counts'], sort_keys=True)}")
    print("runs:")
    for row in report.get("runs", report.get("jobs", [])):
        print(
            "  "
            f"run={row['id']} machine={row['machine']} status={row['status']} "
            f"image={row.get('runtime_image_ref') or ''} "
            f"name={row.get('run_name') or ''} "
            f"launch={row.get('launch_state') or 'not_started'} "
            f"publication={row.get('live_publication_status') or 'pending'} "
            f"eval={row.get('eval_status') or 'not_applicable'} "
            f"artifact={row.get('artifact_status') or 'unknown'}"
        )


def build_train_enqueue_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlab experiment launch",
        description=(
            "Create queue-backed train jobs from one checked-in goal contract and recipe."
        ),
        allow_abbrev=False,
    )
    add_direct_database_arg(parser)
    parser.add_argument("--machines", type=Path, default=DEFAULT_MACHINE_REGISTRY)
    parser.add_argument("--goal-file", dest="goal_file", type=Path, required=True)
    parser.add_argument("--recipe-file", dest="recipe_file", type=Path, required=True)
    parser.add_argument("--machine", required=True, help="Exact registered machine name.")
    parser.add_argument("--request-id", dest="submission_key")
    parser.add_argument(
        "--runtime-image-ref-file",
        type=Path,
        help=(
            "Optional JSON artifact or plain-text descriptor containing the immutable runtime "
            "image ref for the current clean source revision. When omitted, rlab resolves or "
            "builds that exact-source image and never falls back to an older image."
        ),
    )
    parser.add_argument("--image-workflow", default=DEFAULT_IMAGE_WORKFLOW)
    parser.add_argument(
        "--image-branch",
        help="Pushed branch used for automatic workflow dispatch; defaults to the current branch.",
    )
    parser.add_argument("--image-artifact", default=DEFAULT_IMAGE_ARTIFACT)
    parser.add_argument(
        "--runtime-readiness-timeout",
        type=parse_duration_seconds,
        default=DEFAULT_RUNTIME_READINESS_TIMEOUT_SECONDS,
        help="Maximum wait for exact-source image and required backend readiness (default: 20m).",
    )
    parser.add_argument("--seed", type=int, action="append", default=[])
    parser.add_argument(
        "--checkpoint-eval-backend",
        choices=("local", "modal", "none"),
        default=None,
        help=(
            "Materialize the checkpoint evaluation backend for this submission; "
            "none is limited to non-promotable smoke/debug runs."
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
            "--set recipe_id=lr2e4 "
            "--set train.backend.config.learning_rate=2e-4."
        ),
    )
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


def _connect_from_args(args: argparse.Namespace):
    return connect(database_url(args.direct))


def _machine_capacities(path: Path = DEFAULT_MACHINE_REGISTRY) -> dict[str, int]:
    registry = load_machine_registry(path)
    return {
        name: machine.limits.max_parallel_containers for name, machine in registry.machines.items()
    }


def cmd_setup(args: argparse.Namespace) -> int:
    from rlab.fleet_service import default_service_paths, schema_change_service_guard

    with schema_change_service_guard(default_service_paths()):
        conn = _connect_from_args(args)
        try:
            apply_schema(conn)
            worker_mailbox_role = args.worker_mailbox_role or configured_worker_mailbox_role()
            if worker_mailbox_role:
                grant_worker_mailbox_role(conn, worker_mailbox_role)
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
    from rlab.fleet import default_repo_root, load_mailbox_runner_env

    # New submissions use the mailbox transport. Fail before image readiness or
    # queue mutation when the restricted worker DSN/R2 handoff credentials are absent.
    load_mailbox_runner_env(default_repo_root() / ".env")
    document = compose_train_document(
        args.goal_file,
        args.recipe_file,
        recipe_overrides=args.recipe_overrides,
    )
    backend = resolve_checkpoint_eval_backend(
        dict(document.get("train_config") or {}),
        checkpoint_eval_backend=args.checkpoint_eval_backend,
    )
    registry = load_machine_registry(args.machines)
    machine_config = resolve_machine(registry, args.machine)
    timings: dict[str, float] = {}
    readiness_started = time.perf_counter()
    release = runtime_release_from_args(
        args,
        checkpoint_eval_backend=backend,
        wait_for_modal=False,
    )
    timings["image_resolution_seconds"] = time.perf_counter() - readiness_started
    runtime_image_ref = release.runtime_image_ref
    if clean_git_source_sha() != release.source_sha:
        raise RuntimeError(
            "Git source changed while waiting for runtime readiness; rerun rlab experiment launch"
        )
    metadata = recipe_metadata(args.goal_file, args.recipe_file, document)
    from rlab.docker_host import DockerRunnerHost

    host = DockerRunnerHost(machine_config)
    executor: ThreadPoolExecutor | None = None
    modal_future: Future[Any] | None = None

    if backend == "modal":
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rlab-modal-ready")

        def validate_modal_readiness() -> Any:
            modal_started = time.perf_counter()
            remaining = max(
                float(args.runtime_readiness_timeout) - (time.perf_counter() - readiness_started),
                0.0,
            )
            ready_release = wait_for_modal_readiness(
                release,
                timeout=remaining,
                image_workflow=args.image_workflow,
            )
            timings["modal_readiness_seconds"] = time.perf_counter() - modal_started
            live_started = time.perf_counter()
            report = require_modal_eval_ready(
                runtime_image_ref=runtime_image_ref,
                game=str(document.get("train_config", {}).get("game") or ""),
                env_provider=str(document.get("train_config", {}).get("env_provider") or ""),
                runtime_input_sha256=release.runtime_input_sha256,
                runtime_build_source_sha=release.runtime_build_source_sha,
            )
            timings["modal_live_preflight_seconds"] = time.perf_counter() - live_started
            return ready_release, report

        modal_future = executor.submit(validate_modal_readiness)

    def readiness_barrier() -> None:
        if modal_future is not None:
            modal_future.result()
        timings["runtime_readiness_seconds"] = time.perf_counter() - readiness_started

    def validate_runtime_config(train_config: Mapping[str, Any]) -> dict[str, Any]:
        if clean_git_source_sha() != release.source_sha:
            raise RuntimeError(
                "Git source changed before target-machine preflight; rerun rlab experiment launch"
            )
        started = time.perf_counter()
        try:
            receipt = host.validate_runtime_train_config(
                runtime_image_ref=runtime_image_ref,
                train_config=train_config,
                expected_source_sha=release.runtime_build_source_sha or release.source_sha,
                expected_contract_sha256=release.train_config_contract_sha256,
                expected_runtime_input_sha256=release.runtime_input_sha256,
            )
            for key, value in dict(receipt.get("preflight_timings") or {}).items():
                timings[key] = timings.get(key, 0.0) + float(value)
            return receipt
        finally:
            timings["target_preflight_seconds"] = (
                timings.get("target_preflight_seconds", 0.0) + time.perf_counter() - started
            )

    conn = None
    try:
        conn = _connect_from_args(args)
        enqueue_started = time.perf_counter()
        rows = enqueue_train_jobs_from_recipe_document(
            conn,
            document=document,
            runtime_image_ref=runtime_image_ref,
            machine=args.machine,
            runtime_input_sha256=release.runtime_input_sha256,
            runtime_build_source_sha=release.runtime_build_source_sha,
            submission_key=args.submission_key,
            goal_path=metadata["goal_path"],
            goal_sha256=metadata["goal_sha256"],
            recipe_path=metadata["recipe_path"],
            recipe_sha256=metadata["recipe_sha256"],
            repo_git_commit=metadata["repo_git_commit"],
            repo_dirty=metadata["repo_dirty"],
            seeds=args.seed,
            checkpoint_eval_backend=args.checkpoint_eval_backend,
            runtime_config_validator=validate_runtime_config,
            _modal_readiness_validated=backend == "modal",
            readiness_barrier=readiness_barrier,
            repo_root=default_repo_root(),
        )
        timings["enqueue_preflight_seconds"] = time.perf_counter() - enqueue_started
        dispatch_started = time.perf_counter()
        dispatch = dispatch_fleet_service(
            "train_enqueue",
            entity_kind="batch",
            entity_id=str(rows[0]["batch_id"]) if rows else "",
        )
        timings["dispatch_seconds"] = time.perf_counter() - dispatch_started
        wait_result = None
        if args.wait:
            wait_started = time.perf_counter()
            wait_result = wait_for_job_ids(
                conn,
                [int(row["id"]) for row in rows],
                until=args.wait,
                timeout=float(args.timeout),
            )
            timings["job_readiness_wait_seconds"] = time.perf_counter() - wait_started
        report = queue_status(
            conn,
            batch_id=str(rows[0]["batch_id"]),
            machine_capacities={
                name: machine.limits.max_parallel_containers
                for name, machine in registry.machines.items()
            },
        )
    finally:
        if conn is not None:
            conn.close()
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
    if "runtime_readiness_seconds" not in timings:
        timings["runtime_readiness_seconds"] = time.perf_counter() - readiness_started
    if report.get("jobs"):
        job = report["jobs"][0]

        def timestamp(value: object) -> datetime | None:
            text = str(value or "").strip()
            return datetime.fromisoformat(text) if text else None

        created_at = timestamp(job.get("created_at"))
        started_at = timestamp(job.get("started_at"))
        learner_ready_at = timestamp(job.get("learner_ready_at"))
        wandb_ready_at = timestamp(job.get("wandb_ready_at"))
        ready_at = timestamp(job.get("ready_at"))
        if created_at and started_at:
            timings["queue_to_container_start_seconds"] = (started_at - created_at).total_seconds()
        if started_at and ready_at:
            timings["container_to_learner_wandb_ready_seconds"] = (
                ready_at - started_at
            ).total_seconds()
        if started_at and learner_ready_at:
            timings["container_to_learner_ready_seconds"] = (
                learner_ready_at - started_at
            ).total_seconds()
        if started_at and wandb_ready_at:
            timings["container_to_wandb_ready_seconds"] = (
                wandb_ready_at - started_at
            ).total_seconds()
    payload = {
        "batch_id": rows[0]["batch_id"] if rows else None,
        "job_ids": [int(row["id"]) for row in rows],
        "machine": args.machine,
        "runtime_image_ref": runtime_image_ref,
        "runtime_input_sha256": release.runtime_input_sha256,
        "runtime_build_source_sha": release.runtime_build_source_sha,
        "source_sha": release.source_sha,
        "dispatch": dispatch,
        "jobs": report["jobs"],
        "wait": wait_result,
        "readiness_timings": {key: round(value, 3) for key, value in timings.items()},
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


def dispatch_fleet_service(
    reason: str = "unknown",
    *,
    entity_kind: str = "",
    entity_id: str | int = "",
) -> str:
    try:
        from rlab.fleet_service import kick_service

        return (
            "kicked"
            if kick_service(
                reason=reason,
                entity_kind=entity_kind,
                entity_id=entity_id,
            )
            else "degraded"
        )
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
    terminal = {"succeeded", "failed", "finalization_failed", "canceled"}
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
        dispatch = dispatch_fleet_service(
            "train_cancel",
            entity_kind="train",
            entity_id=",".join(str(job_id) for job_id in ids),
        )
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
    registry = load_machine_registry(args.machines)
    capacities = {
        name: machine.limits.max_parallel_containers for name, machine in registry.machines.items()
    }
    conn = _connect_from_args(args)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT machine, train_config FROM train_jobs WHERE id = %(job_id)s",
                {"job_id": args.job_id},
            )
            source = cur.fetchone()
        if not source:
            raise ValueError(f"job {args.job_id} does not exist")
        source_config = dict(source.get("train_config") or {})
        backend = str(source_config.get("checkpoint_eval_backend") or "local")
        release = runtime_release_from_args(args, checkpoint_eval_backend=backend)
        if clean_git_source_sha() != release.source_sha:
            raise RuntimeError(
                "Git source changed while waiting for runtime readiness; rerun the retry"
            )
        machine_config = resolve_machine(registry, str(source["machine"]))
        from rlab.docker_host import DockerRunnerHost

        host = DockerRunnerHost(machine_config)

        def validate_runtime_config(train_config: Mapping[str, Any]) -> dict[str, Any]:
            if clean_git_source_sha() != release.source_sha:
                raise RuntimeError(
                    "Git source changed before target-machine preflight; rerun the retry"
                )
            return host.validate_runtime_train_config(
                runtime_image_ref=release.runtime_image_ref,
                train_config=train_config,
                expected_source_sha=release.runtime_build_source_sha or release.source_sha,
                expected_contract_sha256=release.train_config_contract_sha256,
                expected_runtime_input_sha256=release.runtime_input_sha256,
            )

        row = retry_train_job(
            conn,
            job_id=args.job_id,
            submission_key=args.submission_key,
            runtime_image_ref=release.runtime_image_ref,
            runtime_input_sha256=release.runtime_input_sha256,
            runtime_build_source_sha=release.runtime_build_source_sha,
            repo_git_commit=release.source_sha,
            runtime_config_validator=validate_runtime_config,
        )
        dispatch = dispatch_fleet_service(
            "train_retry",
            entity_kind="train",
            entity_id=int(row["id"]),
        )
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


def cmd_retry_finalization(args: argparse.Namespace) -> int:
    conn = _connect_from_args(args)
    try:
        row = retry_train_job_finalization(conn, job_id=int(args.job_id))
        dispatch = dispatch_fleet_service(
            "train_finalization_retry",
            entity_kind="train",
            entity_id=int(args.job_id),
        )
    finally:
        conn.close()
    payload = {"job": row, "dispatch": dispatch}
    print(json.dumps(json_safe(payload), sort_keys=True))
    return 0


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
