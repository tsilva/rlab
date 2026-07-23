from __future__ import annotations


TELEMETRY_V2_TABLES = (
    "telemetry_run_facts",
    "telemetry_evidence_scopes",
    "wandb_projection_rows",
    "wandb_projection_generations",
    "telemetry_recovery_manifests",
    "telemetry_integrity",
    "telemetry_incidents",
    "telemetry_archive_roots",
    "telemetry_archive_receipts",
    "telemetry_archive_segments",
    "telemetry_events",
    "telemetry_expected_obligations",
    "telemetry_producers",
    "telemetry_rollout_controls",
)


TELEMETRY_V2_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS telemetry_rollout_controls (
  singleton BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (singleton),
  cutover_generation BIGINT NOT NULL DEFAULT 1 CHECK (cutover_generation >= 1),
  admission_fenced BOOLEAN NOT NULL DEFAULT FALSE,
  destructive_hold BOOLEAN NOT NULL DEFAULT FALSE,
  legacy_wandb_credential_generation BIGINT NOT NULL DEFAULT 1
    CHECK (legacy_wandb_credential_generation >= 1),
  service_wandb_credential_generation BIGINT,
  migration_principal TEXT,
  reason TEXT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO telemetry_rollout_controls (singleton)
VALUES (TRUE)
ON CONFLICT (singleton) DO NOTHING;

ALTER TABLE train_jobs
  ADD COLUMN IF NOT EXISTS telemetry_protocol_version INTEGER NOT NULL DEFAULT 1;
ALTER TABLE train_jobs
  ADD COLUMN IF NOT EXISTS telemetry_generation BIGINT NOT NULL DEFAULT 1;
ALTER TABLE train_jobs
  ADD COLUMN IF NOT EXISTS telemetry_integrity_classification TEXT NOT NULL
    DEFAULT 'pending';
ALTER TABLE train_jobs
  ADD COLUMN IF NOT EXISTS telemetry_durability_policy TEXT;
ALTER TABLE train_jobs
  ADD COLUMN IF NOT EXISTS telemetry_no_more_producers BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE train_jobs
  ADD COLUMN IF NOT EXISTS telemetry_frozen_at TIMESTAMPTZ;
ALTER TABLE train_jobs
  ADD COLUMN IF NOT EXISTS active_wandb_projection_generation BIGINT;
ALTER TABLE train_jobs DROP CONSTRAINT IF EXISTS train_jobs_telemetry_protocol_version_check;
ALTER TABLE train_jobs ADD CONSTRAINT train_jobs_telemetry_protocol_version_check
  CHECK (telemetry_protocol_version IN (1, 2));
ALTER TABLE train_jobs DROP CONSTRAINT IF EXISTS train_jobs_telemetry_generation_check;
ALTER TABLE train_jobs ADD CONSTRAINT train_jobs_telemetry_generation_check
  CHECK (telemetry_generation >= 1);
ALTER TABLE train_jobs
  DROP CONSTRAINT IF EXISTS train_jobs_telemetry_integrity_classification_check;
ALTER TABLE train_jobs ADD CONSTRAINT train_jobs_telemetry_integrity_classification_check
  CHECK (telemetry_integrity_classification IN (
    'pending', 'intact_with_proof', 'degraded', 'legacy_unknown'
  ));
ALTER TABLE train_jobs DROP CONSTRAINT IF EXISTS train_jobs_telemetry_durability_policy_check;
ALTER TABLE train_jobs ADD CONSTRAINT train_jobs_telemetry_durability_policy_check
  CHECK (
    telemetry_durability_policy IS NULL OR telemetry_durability_policy IN (
      'queued_dual_r2_v1', 'local_mirrored_v1', 'local_singlecopy_optout_v1'
    )
  );

ALTER TABLE worker_attempts
  ADD COLUMN IF NOT EXISTS telemetry_generation BIGINT NOT NULL DEFAULT 1;
ALTER TABLE worker_attempts ADD COLUMN IF NOT EXISTS producer_ordinal INTEGER;
ALTER TABLE worker_attempts ADD COLUMN IF NOT EXISTS producer_identity TEXT;
ALTER TABLE worker_attempts DROP CONSTRAINT IF EXISTS worker_attempts_protocol_version_check;
ALTER TABLE worker_attempts ADD CONSTRAINT worker_attempts_protocol_version_check
  CHECK (protocol_version IN (1, 2));
ALTER TABLE worker_attempts DROP CONSTRAINT IF EXISTS worker_attempts_producer_ordinal_check;
ALTER TABLE worker_attempts ADD CONSTRAINT worker_attempts_producer_ordinal_check
  CHECK (producer_ordinal IS NULL OR producer_ordinal >= 0);

ALTER TABLE metric_batches ADD COLUMN IF NOT EXISTS payload_sha256 TEXT;
ALTER TABLE metric_batches ADD COLUMN IF NOT EXISTS wandb_confirmed_at TIMESTAMPTZ;
ALTER TABLE metric_batches ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;
ALTER TABLE metric_batches ADD COLUMN IF NOT EXISTS archive_root_sha256 TEXT;
ALTER TABLE metric_batches ADD COLUMN IF NOT EXISTS archive_lease_owner TEXT;
ALTER TABLE metric_batches ADD COLUMN IF NOT EXISTS archive_lease_expires_at TIMESTAMPTZ;
ALTER TABLE metric_batches DROP CONSTRAINT IF EXISTS metric_batches_payload_sha256_check;
ALTER TABLE metric_batches ADD CONSTRAINT metric_batches_payload_sha256_check
  CHECK (payload_sha256 IS NULL OR payload_sha256 ~ '^[0-9a-f]{64}$');
ALTER TABLE metric_batches DROP CONSTRAINT IF EXISTS metric_batches_archive_root_sha256_check;
ALTER TABLE metric_batches ADD CONSTRAINT metric_batches_archive_root_sha256_check
  CHECK (archive_root_sha256 IS NULL OR archive_root_sha256 ~ '^[0-9a-f]{64}$');

CREATE OR REPLACE FUNCTION rlab_metric_batch_digest()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE actual TEXT;
BEGIN
  actual := encode(digest(NEW.payload, 'sha256'), 'hex');
  IF NEW.payload_sha256 IS NOT NULL AND NEW.payload_sha256 <> actual THEN
    RAISE EXCEPTION 'metric batch payload digest conflicts with payload bytes';
  END IF;
  NEW.payload_sha256 := actual;
  RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS metric_batch_digest_guard ON metric_batches;
CREATE TRIGGER metric_batch_digest_guard
BEFORE INSERT OR UPDATE OF payload, payload_sha256 ON metric_batches
FOR EACH ROW EXECUTE FUNCTION rlab_metric_batch_digest();

CREATE TABLE IF NOT EXISTS telemetry_producers (
  train_job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  telemetry_generation BIGINT NOT NULL CHECK (telemetry_generation >= 1),
  producer_ordinal INTEGER NOT NULL CHECK (producer_ordinal >= 0),
  producer_identity TEXT NOT NULL,
  attempt_id TEXT REFERENCES worker_attempts(attempt_id) ON DELETE SET NULL,
  producer_kind TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'registered'
    CHECK (state IN ('registered', 'active', 'closed', 'canceled', 'aborted', 'failed')),
  next_source_sequence BIGINT NOT NULL DEFAULT 1 CHECK (next_source_sequence >= 1),
  chain_sha256 TEXT CHECK (chain_sha256 IS NULL OR chain_sha256 ~ '^[0-9a-f]{64}$'),
  final_sequence BIGINT CHECK (final_sequence IS NULL OR final_sequence >= 0),
  final_sha256 TEXT CHECK (final_sha256 IS NULL OR final_sha256 ~ '^[0-9a-f]{64}$'),
  registered_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  closed_at TIMESTAMPTZ,
  PRIMARY KEY (train_job_id, telemetry_generation, producer_ordinal),
  UNIQUE (train_job_id, telemetry_generation, producer_identity),
  CHECK (
    (state IN ('closed', 'canceled', 'aborted', 'failed')
      AND final_sequence IS NOT NULL AND final_sha256 IS NOT NULL AND closed_at IS NOT NULL)
    OR (state IN ('registered', 'active') AND closed_at IS NULL)
  )
);

CREATE TABLE IF NOT EXISTS telemetry_expected_obligations (
  train_job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  telemetry_generation BIGINT NOT NULL CHECK (telemetry_generation >= 1),
  obligation_key TEXT NOT NULL,
  obligation_kind TEXT NOT NULL,
  producer_ordinal INTEGER NOT NULL CHECK (producer_ordinal >= 0),
  expected_disposition TEXT NOT NULL DEFAULT 'complete'
    CHECK (expected_disposition IN (
      'complete', 'canceled', 'aborted_before_release', 'disabled', 'failed'
    )),
  realized_disposition TEXT CHECK (realized_disposition IS NULL OR realized_disposition IN (
    'complete', 'canceled', 'aborted_before_release', 'disabled', 'failed'
  )),
  evidence_sha256 TEXT CHECK (
    evidence_sha256 IS NULL OR evidence_sha256 ~ '^[0-9a-f]{64}$'
  ),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  realized_at TIMESTAMPTZ,
  PRIMARY KEY (train_job_id, telemetry_generation, obligation_key),
  FOREIGN KEY (train_job_id, telemetry_generation, producer_ordinal)
    REFERENCES telemetry_producers(train_job_id, telemetry_generation, producer_ordinal)
    DEFERRABLE INITIALLY DEFERRED
);

CREATE TABLE IF NOT EXISTS telemetry_events (
  train_job_id BIGINT NOT NULL,
  telemetry_generation BIGINT NOT NULL CHECK (telemetry_generation >= 1),
  producer_ordinal INTEGER NOT NULL CHECK (producer_ordinal >= 0),
  source_sequence BIGINT NOT NULL CHECK (source_sequence >= 1),
  event_identity TEXT NOT NULL,
  event_kind TEXT NOT NULL,
  payload_encoding TEXT NOT NULL CHECK (
    payload_encoding IN ('metric_batch_zlib_json_v1', 'canonical_json_v1')
  ),
  payload BYTEA NOT NULL,
  payload_sha256 TEXT NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
  event_sha256 TEXT NOT NULL CHECK (event_sha256 ~ '^[0-9a-f]{64}$'),
  predecessor_sha256 TEXT CHECK (
    predecessor_sha256 IS NULL OR predecessor_sha256 ~ '^[0-9a-f]{64}$'
  ),
  terminal BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (
    train_job_id, telemetry_generation, producer_ordinal, source_sequence
  ),
  UNIQUE (
    train_job_id, telemetry_generation, producer_ordinal, event_identity
  ),
  FOREIGN KEY (train_job_id, telemetry_generation, producer_ordinal)
    REFERENCES telemetry_producers(train_job_id, telemetry_generation, producer_ordinal)
);

CREATE OR REPLACE FUNCTION rlab_append_canonical_telemetry_event(
  p_attempt_id TEXT,
  p_event_identity TEXT,
  p_event_kind TEXT,
  p_payload_encoding TEXT,
  p_payload BYTEA,
  p_terminal BOOLEAN
) RETURNS BIGINT
LANGUAGE plpgsql
SET search_path = public, pg_temp
AS $$
DECLARE
  producer telemetry_producers%ROWTYPE;
  job train_jobs%ROWTYPE;
  existing telemetry_events%ROWTYPE;
  payload_digest TEXT;
  event_digest TEXT;
  next_digest TEXT;
BEGIN
  SELECT p.* INTO producer
  FROM telemetry_producers p
  WHERE p.attempt_id = p_attempt_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RETURN NULL;
  END IF;
  SELECT * INTO job FROM train_jobs WHERE id = producer.train_job_id FOR UPDATE;
  IF job.telemetry_frozen_at IS NOT NULL OR job.telemetry_no_more_producers THEN
    RAISE EXCEPTION 'canonical telemetry generation is fenced';
  END IF;
  SELECT * INTO existing
  FROM telemetry_events e
  WHERE e.train_job_id = producer.train_job_id
    AND e.telemetry_generation = producer.telemetry_generation
    AND e.producer_ordinal = producer.producer_ordinal
    AND e.event_identity = p_event_identity;
  payload_digest := encode(digest(p_payload, 'sha256'), 'hex');
  IF FOUND THEN
    IF existing.event_kind <> p_event_kind
       OR existing.payload_encoding <> p_payload_encoding
       OR existing.payload_sha256 <> payload_digest
       OR existing.terminal <> p_terminal THEN
      RAISE EXCEPTION 'canonical telemetry event identity conflict';
    END IF;
    RETURN existing.source_sequence;
  END IF;
  IF producer.state NOT IN ('registered', 'active') THEN
    RAISE EXCEPTION 'canonical telemetry producer is terminal';
  END IF;
  event_digest := encode(digest(convert_to(
    concat_ws(E'\\n',
      'telemetry-event-v2',
      producer.train_job_id::text,
      producer.telemetry_generation::text,
      producer.producer_ordinal::text,
      producer.next_source_sequence::text,
      p_event_identity,
      p_event_kind,
      p_payload_encoding,
      payload_digest,
      CASE WHEN p_terminal THEN 'true' ELSE 'false' END
    ), 'UTF8'
  ), 'sha256'), 'hex');
  next_digest := encode(digest(convert_to(
    COALESCE(producer.chain_sha256, '') || event_digest, 'UTF8'
  ), 'sha256'), 'hex');
  INSERT INTO telemetry_events (
    train_job_id, telemetry_generation, producer_ordinal, source_sequence,
    event_identity, event_kind, payload_encoding, payload, payload_sha256,
    event_sha256, predecessor_sha256, terminal
  ) VALUES (
    producer.train_job_id, producer.telemetry_generation, producer.producer_ordinal,
    producer.next_source_sequence, p_event_identity, p_event_kind,
    p_payload_encoding, p_payload, payload_digest, event_digest,
    producer.chain_sha256, p_terminal
  );
  UPDATE telemetry_producers
  SET state = CASE WHEN p_terminal THEN 'closed' ELSE 'active' END,
      next_source_sequence = next_source_sequence + 1,
      chain_sha256 = next_digest,
      final_sequence = CASE WHEN p_terminal THEN producer.next_source_sequence ELSE NULL END,
      final_sha256 = CASE WHEN p_terminal THEN next_digest ELSE NULL END,
      closed_at = CASE WHEN p_terminal THEN now() ELSE NULL END
  WHERE train_job_id = producer.train_job_id
    AND telemetry_generation = producer.telemetry_generation
    AND producer_ordinal = producer.producer_ordinal;
  IF p_terminal THEN
    UPDATE telemetry_expected_obligations
    SET realized_disposition = expected_disposition,
        evidence_sha256 = next_digest,
        realized_at = now()
    WHERE train_job_id = producer.train_job_id
      AND telemetry_generation = producer.telemetry_generation
      AND producer_ordinal = producer.producer_ordinal
      AND realized_disposition IS NULL;
  END IF;
  RETURN producer.next_source_sequence;
END;
$$;

CREATE OR REPLACE FUNCTION rlab_capture_metric_batch_event()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  PERFORM rlab_append_canonical_telemetry_event(
    NEW.stream_id,
    'metric-batch:' || NEW.batch_sequence::text,
    'metric_batch',
    'metric_batch_zlib_json_v1',
    NEW.payload,
    FALSE
  );
  RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS canonical_metric_batch_event ON metric_batches;
CREATE TRIGGER canonical_metric_batch_event
AFTER INSERT ON metric_batches
FOR EACH ROW EXECUTE FUNCTION rlab_capture_metric_batch_event();

CREATE OR REPLACE FUNCTION rlab_capture_attempt_event()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  PERFORM rlab_append_canonical_telemetry_event(
    NEW.attempt_id,
    'attempt-event:' || NEW.event_id,
    'control:' || NEW.event_type,
    'canonical_json_v1',
    convert_to(NEW.payload_json::text, 'UTF8'),
    NEW.event_type = 'metric_stream_closed'
  );
  RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS canonical_attempt_event ON attempt_events;
CREATE TRIGGER canonical_attempt_event
AFTER INSERT ON attempt_events
FOR EACH ROW EXECUTE FUNCTION rlab_capture_attempt_event();

CREATE OR REPLACE FUNCTION rlab_capture_command_ack()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF OLD.acknowledged_at IS NULL AND NEW.acknowledged_at IS NOT NULL THEN
    PERFORM rlab_append_canonical_telemetry_event(
      NEW.attempt_id,
      'command-ack:' || NEW.command_id,
      'command_ack',
      'canonical_json_v1',
      convert_to(jsonb_build_object(
        'command_id', NEW.command_id,
        'command_type', NEW.command_type,
        'acknowledged_at', NEW.acknowledged_at
      )::text, 'UTF8'),
      FALSE
    );
  END IF;
  RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS canonical_command_ack ON attempt_commands;
CREATE TRIGGER canonical_command_ack
AFTER UPDATE OF acknowledged_at ON attempt_commands
FOR EACH ROW EXECUTE FUNCTION rlab_capture_command_ack();

CREATE OR REPLACE FUNCTION rlab_guard_late_telemetry_registration()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
  IF EXISTS (
    SELECT 1 FROM train_jobs
    WHERE id = NEW.train_job_id
      AND (telemetry_no_more_producers OR telemetry_frozen_at IS NOT NULL)
  ) THEN
    RAISE EXCEPTION 'late telemetry producer or obligation registration is fenced';
  END IF;
  RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS telemetry_producer_registration_guard ON telemetry_producers;
CREATE TRIGGER telemetry_producer_registration_guard
BEFORE INSERT ON telemetry_producers
FOR EACH ROW EXECUTE FUNCTION rlab_guard_late_telemetry_registration();
DROP TRIGGER IF EXISTS telemetry_obligation_registration_guard
  ON telemetry_expected_obligations;
CREATE TRIGGER telemetry_obligation_registration_guard
BEFORE INSERT ON telemetry_expected_obligations
FOR EACH ROW EXECUTE FUNCTION rlab_guard_late_telemetry_registration();

CREATE TABLE IF NOT EXISTS telemetry_archive_segments (
  train_job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  telemetry_generation BIGINT NOT NULL CHECK (telemetry_generation >= 1),
  producer_ordinal INTEGER NOT NULL CHECK (producer_ordinal >= 0),
  first_sequence BIGINT NOT NULL CHECK (first_sequence >= 1),
  last_sequence BIGINT NOT NULL CHECK (last_sequence >= first_sequence),
  event_count INTEGER NOT NULL CHECK (event_count >= 1),
  format_version TEXT NOT NULL,
  uncompressed_sha256 TEXT NOT NULL CHECK (uncompressed_sha256 ~ '^[0-9a-f]{64}$'),
  compressed_sha256 TEXT NOT NULL CHECK (compressed_sha256 ~ '^[0-9a-f]{64}$'),
  byte_count BIGINT NOT NULL CHECK (byte_count >= 1),
  claim_state TEXT NOT NULL DEFAULT 'claimed'
    CHECK (claim_state IN ('claimed', 'writing', 'verified', 'conflict', 'failed')),
  claim_owner TEXT,
  claim_expires_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  verified_at TIMESTAMPTZ,
  PRIMARY KEY (
    train_job_id, telemetry_generation, producer_ordinal, first_sequence, last_sequence
  ),
  FOREIGN KEY (train_job_id, telemetry_generation, producer_ordinal)
    REFERENCES telemetry_producers(train_job_id, telemetry_generation, producer_ordinal)
);

CREATE TABLE IF NOT EXISTS telemetry_archive_receipts (
  train_job_id BIGINT NOT NULL,
  telemetry_generation BIGINT NOT NULL,
  producer_ordinal INTEGER NOT NULL,
  first_sequence BIGINT NOT NULL,
  last_sequence BIGINT NOT NULL,
  copy_role TEXT NOT NULL CHECK (copy_role IN ('primary', 'backup', 'local', 'mirror')),
  policy_name TEXT NOT NULL CHECK (policy_name IN (
    'queued_dual_r2_v1', 'local_mirrored_v1', 'local_singlecopy_optout_v1'
  )),
  policy_sha256 TEXT NOT NULL CHECK (policy_sha256 ~ '^[0-9a-f]{64}$'),
  object_uri TEXT NOT NULL,
  object_version TEXT NOT NULL,
  compressed_sha256 TEXT NOT NULL CHECK (compressed_sha256 ~ '^[0-9a-f]{64}$'),
  full_read_verified_at TIMESTAMPTZ NOT NULL,
  receipt_json JSONB NOT NULL,
  PRIMARY KEY (
    train_job_id, telemetry_generation, producer_ordinal,
    first_sequence, last_sequence, copy_role
  ),
  FOREIGN KEY (
    train_job_id, telemetry_generation, producer_ordinal, first_sequence, last_sequence
  ) REFERENCES telemetry_archive_segments(
    train_job_id, telemetry_generation, producer_ordinal, first_sequence, last_sequence
  ) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS telemetry_archive_roots (
  train_job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  telemetry_generation BIGINT NOT NULL CHECK (telemetry_generation >= 1),
  root_kind TEXT NOT NULL CHECK (root_kind IN ('exact', 'legacy_loss_adjudicated')),
  root_sha256 TEXT NOT NULL CHECK (root_sha256 ~ '^[0-9a-f]{64}$'),
  expected_set_sha256 TEXT CHECK (
    expected_set_sha256 IS NULL OR expected_set_sha256 ~ '^[0-9a-f]{64}$'
  ),
  coverage_sha256 TEXT NOT NULL CHECK (coverage_sha256 ~ '^[0-9a-f]{64}$'),
  policy_name TEXT NOT NULL,
  root_json JSONB NOT NULL,
  finalized_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (train_job_id, telemetry_generation),
  UNIQUE (root_sha256),
  CHECK (
    (root_kind = 'exact' AND expected_set_sha256 IS NOT NULL)
    OR root_kind = 'legacy_loss_adjudicated'
  )
);

CREATE TABLE IF NOT EXISTS telemetry_incidents (
  id BIGSERIAL PRIMARY KEY,
  train_job_id BIGINT REFERENCES train_jobs(id) ON DELETE CASCADE,
  telemetry_generation BIGINT,
  incident_key TEXT NOT NULL,
  severity TEXT NOT NULL CHECK (severity IN ('warning', 'error', 'critical')),
  category TEXT NOT NULL,
  state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open', 'resolved', 'permanent')),
  details_json JSONB NOT NULL,
  first_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  resolved_at TIMESTAMPTZ,
  UNIQUE (train_job_id, telemetry_generation, incident_key)
);

CREATE TABLE IF NOT EXISTS telemetry_integrity (
  train_job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  telemetry_generation BIGINT NOT NULL CHECK (telemetry_generation >= 1),
  classification TEXT NOT NULL CHECK (
    classification IN ('pending', 'intact_with_proof', 'degraded', 'legacy_unknown')
  ),
  disposition TEXT NOT NULL CHECK (
    disposition IN ('pending', 'exact', 'legacy_loss_adjudicated', 'durability_opted_out')
  ),
  exact BOOLEAN NOT NULL DEFAULT FALSE,
  cleanup_eligible BOOLEAN NOT NULL DEFAULT FALSE,
  expected_set_sha256 TEXT,
  coverage_sha256 TEXT,
  archive_root_sha256 TEXT,
  facts_sha256 TEXT,
  active_wandb_generation BIGINT,
  reasons TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
  state_json JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (train_job_id, telemetry_generation)
);

CREATE TABLE IF NOT EXISTS telemetry_recovery_manifests (
  train_job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  telemetry_generation BIGINT NOT NULL CHECK (telemetry_generation >= 1),
  manifest_sha256 TEXT NOT NULL CHECK (manifest_sha256 ~ '^[0-9a-f]{64}$'),
  parser_version TEXT NOT NULL,
  archive_policy TEXT NOT NULL,
  absolute_source_paths TEXT[] NOT NULL,
  manifest_json JSONB NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending'
    CHECK (state IN ('pending', 'claimed', 'complete', 'blocked_credentials', 'failed')),
  owner TEXT,
  next_due_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (train_job_id, telemetry_generation)
);

CREATE TABLE IF NOT EXISTS wandb_projection_generations (
  train_job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  projection_generation BIGINT NOT NULL CHECK (projection_generation >= 1),
  service_credential_generation BIGINT NOT NULL CHECK (service_credential_generation >= 1),
  wandb_run_id TEXT NOT NULL,
  generation_token_sha256 TEXT NOT NULL CHECK (
    generation_token_sha256 ~ '^[0-9a-f]{64}$'
  ),
  state TEXT NOT NULL DEFAULT 'pending'
    CHECK (state IN (
      'pending', 'publishing', 'verifying', 'active', 'sealed',
      'quarantined', 'failed', 'disabled'
    )),
  step_offset BIGINT NOT NULL DEFAULT 0 CHECK (step_offset >= 0),
  next_ordinal BIGINT NOT NULL DEFAULT 0 CHECK (next_ordinal >= 0),
  verified_through_ordinal BIGINT NOT NULL DEFAULT -1 CHECK (verified_through_ordinal >= -1),
  incident_id BIGINT REFERENCES telemetry_incidents(id),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  activated_at TIMESTAMPTZ,
  sealed_at TIMESTAMPTZ,
  PRIMARY KEY (train_job_id, projection_generation),
  UNIQUE (wandb_run_id)
);

CREATE TABLE IF NOT EXISTS wandb_projection_rows (
  train_job_id BIGINT NOT NULL,
  projection_generation BIGINT NOT NULL,
  output_ordinal BIGINT NOT NULL CHECK (output_ordinal >= 0),
  stable_key TEXT NOT NULL,
  source_event_id TEXT NOT NULL,
  adapter_version TEXT NOT NULL,
  normalization_version TEXT NOT NULL,
  output_kind TEXT NOT NULL,
  output_index INTEGER NOT NULL CHECK (output_index >= 0),
  predecessor_sha256 TEXT,
  payload_sha256 TEXT NOT NULL CHECK (payload_sha256 ~ '^[0-9a-f]{64}$'),
  payload_json JSONB NOT NULL,
  state TEXT NOT NULL DEFAULT 'pending'
    CHECK (state IN ('pending', 'claimed', 'ambiguous', 'verified', 'conflict')),
  attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  claimed_by TEXT,
  claim_expires_at TIMESTAMPTZ,
  verified_at TIMESTAMPTZ,
  last_error TEXT,
  PRIMARY KEY (train_job_id, projection_generation, output_ordinal),
  UNIQUE (train_job_id, projection_generation, stable_key),
  FOREIGN KEY (train_job_id, projection_generation)
    REFERENCES wandb_projection_generations(train_job_id, projection_generation)
    ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS telemetry_evidence_scopes (
  train_job_id BIGINT NOT NULL REFERENCES train_jobs(id) ON DELETE CASCADE,
  scope_kind TEXT NOT NULL CHECK (
    scope_kind IN ('eval_scope_exact', 'training_success_scope_exact')
  ),
  scope_key TEXT NOT NULL,
  scope_sha256 TEXT NOT NULL CHECK (scope_sha256 ~ '^[0-9a-f]{64}$'),
  root_sha256 TEXT CHECK (root_sha256 IS NULL OR root_sha256 ~ '^[0-9a-f]{64}$'),
  evidence_json JSONB NOT NULL,
  state TEXT NOT NULL CHECK (state IN ('pending', 'exact', 'rejected')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  finalized_at TIMESTAMPTZ,
  PRIMARY KEY (train_job_id, scope_kind, scope_key),
  UNIQUE (scope_sha256)
);

CREATE TABLE IF NOT EXISTS telemetry_run_facts (
  train_job_id BIGINT PRIMARY KEY REFERENCES train_jobs(id) ON DELETE CASCADE,
  scope_sha256 TEXT NOT NULL CHECK (scope_sha256 ~ '^[0-9a-f]{64}$'),
  archive_root_sha256 TEXT NOT NULL CHECK (archive_root_sha256 ~ '^[0-9a-f]{64}$'),
  comparability_sha256 TEXT NOT NULL CHECK (comparability_sha256 ~ '^[0-9a-f]{64}$'),
  cohort_manifest_sha256 TEXT NOT NULL CHECK (cohort_manifest_sha256 ~ '^[0-9a-f]{64}$'),
  seed INTEGER NOT NULL,
  rank_metric TEXT NOT NULL,
  rank_direction TEXT NOT NULL CHECK (rank_direction IN ('min', 'max')),
  facts_json JSONB NOT NULL,
  finalized_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS telemetry_archive_segments_due_idx
  ON telemetry_archive_segments (claim_state, claim_expires_at, created_at);
CREATE INDEX IF NOT EXISTS telemetry_incidents_open_idx
  ON telemetry_incidents (state, severity, last_seen_at);
CREATE INDEX IF NOT EXISTS wandb_projection_rows_due_idx
  ON wandb_projection_rows (
    state, claim_expires_at, train_job_id, projection_generation, output_ordinal
  );
CREATE INDEX IF NOT EXISTS telemetry_run_facts_comparison_idx
  ON telemetry_run_facts (
    comparability_sha256, cohort_manifest_sha256, rank_metric, seed
  );
CREATE INDEX IF NOT EXISTS metric_batches_archive_due_idx
  ON metric_batches (archived_at, archive_lease_expires_at, created_at);
CREATE INDEX IF NOT EXISTS telemetry_events_archive_idx
  ON telemetry_events (
    train_job_id, telemetry_generation, producer_ordinal, source_sequence
  );

CREATE OR REPLACE FUNCTION rlab_guard_metric_batch_delete()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
  rollout telemetry_rollout_controls%ROWTYPE;
  authorization TEXT;
BEGIN
  SELECT * INTO rollout FROM telemetry_rollout_controls WHERE singleton = TRUE;
  authorization := current_setting('rlab.telemetry_delete_generation', TRUE);
  IF OLD.archived_at IS NULL OR OLD.archive_root_sha256 IS NULL THEN
    RAISE EXCEPTION 'metric batch deletion requires a verified canonical archive root';
  END IF;
  IF rollout.destructive_hold
     AND authorization IS DISTINCT FROM rollout.cutover_generation::text THEN
    RAISE EXCEPTION 'telemetry destructive hold denies metric batch deletion';
  END IF;
  RETURN OLD;
END;
$$;
DROP TRIGGER IF EXISTS metric_batch_delete_guard ON metric_batches;
CREATE TRIGGER metric_batch_delete_guard
BEFORE DELETE ON metric_batches
FOR EACH ROW EXECUTE FUNCTION rlab_guard_metric_batch_delete();

CREATE OR REPLACE FUNCTION rlab_guard_legacy_train_admission()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
  rollout telemetry_rollout_controls%ROWTYPE;
  marker TEXT;
BEGIN
  SELECT * INTO rollout FROM telemetry_rollout_controls WHERE singleton = TRUE;
  IF NOT rollout.admission_fenced THEN
    RETURN NEW;
  END IF;
  marker := current_setting('rlab.telemetry_cutover_generation', TRUE);
  IF NEW.telemetry_protocol_version <> 2
     OR marker IS DISTINCT FROM rollout.cutover_generation::text THEN
    RAISE EXCEPTION 'legacy telemetry admission is fenced at generation %',
      rollout.cutover_generation;
  END IF;
  RETURN NEW;
END;
$$;
DROP TRIGGER IF EXISTS telemetry_train_admission_guard ON train_jobs;
CREATE TRIGGER telemetry_train_admission_guard
BEFORE INSERT ON train_jobs
FOR EACH ROW EXECUTE FUNCTION rlab_guard_legacy_train_admission();
"""
