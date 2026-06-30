from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rlab import job_queue
from rlab import main as rlab_main
from rlab.artifacts import wandb_artifact_storage_uri
from rlab.dotenv import load_env_file
from rlab.eval_job_runner import log_eval_to_wandb, normalize_eval_config
from rlab.json_utils import json_safe
from rlab.metric_names import TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST
from rlab.seeds import DEFAULT_EVAL_SEED
from rlab.train_runner import (
    AutoscaleConfig,
    AutoscaleController,
    GRACEFUL_STOP_SIGNAL,
    ResourceSample,
    WORKER_IDLE,
    WORKER_RUNNING,
    WORKER_RETIRING,
    WorkerSlot,
    build_parser as build_train_runner_parser,
    collect_result_metadata,
    enqueue_eval_jobs_for_checkpoints,
    mark_surplus_workers_for_retirement,
    matching_pending_train_job_exists,
    normalize_train_config,
    parse_log_metrics,
    purge_successful_run_data,
    request_graceful_stop,
    resolve_worker_bounds,
    should_purge_successful_run_data,
    train_command_for_job,
    write_train_config_file,
)


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)


class FakeProcess:
    def __init__(self, poll_result=None) -> None:
        self.poll_result = poll_result
        self.sent_signals = []

    def poll(self):
        return self.poll_result

    def send_signal(self, signum) -> None:
        self.sent_signals.append(signum)


class FakeCursor:
    def __init__(self, row=None, rows=None) -> None:
        self.row = row
        self.rows = rows if rows is not None else []
        self.executed_sql = ""
        self.executed_params = {}
        self.executed_sqls = []
        self.executed_params_list = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql, params=None) -> None:
        self.executed_sql = sql
        self.executed_params = params or {}
        self.executed_sqls.append(sql)
        self.executed_params_list.append(params or {})

    def fetchone(self):
        return self.row

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, row=None, rows=None) -> None:
        self.cursor_obj = FakeCursor(row=row, rows=rows)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self):
        return self.cursor_obj


def valid_train_spec() -> dict:
    return {
        "schema_version": 1,
        "goal": "Level1-1",
        "slug": "candidate",
        "stage": "confirm",
        "hypothesis": "Candidate should reproduce the expected completion signal.",
        "expected_signal": "Rank by completion rate, then reward.",
        "parent_spec_slug": None,
        "priority": 7,
        "seeds": [23, 24],
        "wandb_group": "b-test",
        "wandb_tags": ["mario", "confirm"],
        "run_name_template": "btest_s{seed}_{utc}",
        "run_description_template": "candidate seed {seed}",
        "selection_metrics": ["train/completion_episode_rate", "train/reward/mean"],
        "train_config": {
            "game": "SuperMarioBros-Nes-v0",
            "state": "Level1-1",
            "timesteps": 1024,
            "wandb": True,
            "wandb_mode": "online",
        },
    }


class TrainRunnerSignalTests(unittest.TestCase):
    @unittest.skipIf(GRACEFUL_STOP_SIGNAL is None, "SIGUSR1 is unavailable on this platform")
    def test_request_graceful_stop_sends_sigusr1_to_running_process(self) -> None:
        process = FakeProcess()

        self.assertTrue(request_graceful_stop(process))

        self.assertEqual(process.sent_signals, [GRACEFUL_STOP_SIGNAL])

    def test_request_graceful_stop_skips_exited_process(self) -> None:
        process = FakeProcess(poll_result=0)

        self.assertFalse(request_graceful_stop(process))
        self.assertEqual(process.sent_signals, [])


class JobQueueTests(unittest.TestCase):
    def test_claim_train_job_filters_exact_runtime_image_only(self) -> None:
        conn = FakeConnection(row={"id": 7, "profile_id": "mario-ppo/post16/rtx4090-screening"})

        row = job_queue.claim_train_job(
            conn,
            profile_id="mario-ppo/post16/rtx4090-screening",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            worker_id="worker-a",
            lease_seconds=60,
        )

        self.assertEqual(row["id"], 7)
        self.assertIn("runtime_image_ref = %(runtime_image_ref)s", conn.cursor_obj.executed_sql)
        self.assertNotIn("profile_id = %(profile_id)s", conn.cursor_obj.executed_sql)
        self.assertNotIn("run_target = %(run_target)s", conn.cursor_obj.executed_sql)
        self.assertEqual(conn.cursor_obj.executed_params["runtime_image_ref"], RUNTIME_IMAGE_REF)

    def test_claim_train_job_does_not_reclaim_expired_running_leases(self) -> None:
        conn = FakeConnection(row=None)

        row = job_queue.claim_train_job(
            conn,
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            worker_id="worker-any",
            lease_seconds=60,
        )

        self.assertIsNone(row)
        self.assertIn("AND status = 'pending'", conn.cursor_obj.executed_sql)
        self.assertNotIn("lease_expires_at < now()", conn.cursor_obj.executed_sql)
        self.assertNotIn("attempts < max_attempts", conn.cursor_obj.executed_sql)

    def test_secret_like_keys_are_rejected_from_persisted_json(self) -> None:
        with self.assertRaisesRegex(ValueError, "secret-like key"):
            job_queue.assert_no_secrets(
                {"learning_rate": 0.0001, "WANDB_API_KEY": "do-not-store"},
                label="train_config",
            )

    def test_schema_defines_queue_tables(self) -> None:
        self.assertNotIn("CREATE TABLE IF NOT EXISTS research_goals", job_queue.SCHEMA_SQL)
        self.assertNotIn("CREATE TABLE IF NOT EXISTS experiment_specs", job_queue.SCHEMA_SQL)
        self.assertNotIn("CREATE TABLE IF NOT EXISTS campaign_decisions", job_queue.SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS train_jobs", job_queue.SCHEMA_SQL)
        self.assertIn("goal_slug TEXT NOT NULL", job_queue.SCHEMA_SQL)
        self.assertIn("spec_payload_json JSONB", job_queue.SCHEMA_SQL)
        self.assertIn("spec_sha256 TEXT", job_queue.SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS eval_jobs", job_queue.SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS eval_results", job_queue.SCHEMA_SQL)

    def test_reset_schema_drops_only_current_queue_tables(self) -> None:
        conn = FakeConnection()
        with tempfile.TemporaryDirectory() as tmp:
            job_queue.reset_schema(conn, export_dir=Path(tmp))

        drop_sql = next(sql for sql in conn.cursor_obj.executed_sqls if "DROP TABLE" in sql)
        self.assertIn("train_jobs", drop_sql)
        self.assertIn("eval_jobs", drop_sql)
        self.assertNotIn("research_goals", drop_sql)
        self.assertNotIn("experiment_specs", drop_sql)
        self.assertNotIn("campaign_decisions", drop_sql)
        self.assertIn("CREATE TABLE IF NOT EXISTS job_events", job_queue.SCHEMA_SQL)
        self.assertNotIn("origin_decision_id", job_queue.SCHEMA_SQL)
        self.assertIn("runtime_image_ref TEXT", job_queue.SCHEMA_SQL)
        self.assertIn("run_target TEXT", job_queue.SCHEMA_SQL)
        self.assertIn("train_jobs_runtime_claim_idx", job_queue.SCHEMA_SQL)

    def test_load_spec_document_validates_schema_and_preserves_extra_fields(self) -> None:
        spec = valid_train_spec()
        spec["operator_note"] = {"why": "kept outside the formal schema for now"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps(spec), encoding="utf-8")

            loaded = job_queue.load_spec_document(path)

        self.assertEqual(loaded["operator_note"], {"why": "kept outside the formal schema for now"})

    def test_load_spec_document_resolves_hydra_defaults_and_materializes_train_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            recipe = root / "recipes" / "base.yaml"
            recipe.parent.mkdir(parents=True)
            recipe.write_text(
                """
schema_version: 1
kind: train_recipe
env:
  game: SuperMarioBros-Nes-v0
  n_envs: 16
  info_events_json:
    life_loss: [lives, decrease]
    level_change: [[levelHi, levelLo], change]
  done_on_events: [life_loss, level_change]
train:
  timesteps: 1024
  learning_rate: 0.00015
reward:
  death_penalty: 25
logging:
  wandb: true
  wandb_mode: online
""",
                encoding="utf-8",
            )
            spec = root / "goals" / "candidate.yaml"
            spec.parent.mkdir(parents=True)
            spec.write_text(
                """
schema_version: 1
kind: train_experiment
defaults:
- ../recipes/base@_global_
- _self_
goal: Level1-1
slug: candidate
stage: screen
hypothesis: Candidate should reproduce the expected completion signal.
expected_signal: Rank by completion rate, then reward.
parent_spec_slug: null
priority: 7
seeds: [23, 24]
run_target: rtx4090
state: Level1-1
wandb_group: b-test
wandb_tags: [mario, confirm]
run_name_template: btest_s{seed}_{utc}
run_description_template: candidate seed {seed}
selection_gate:
  primary: train/completion_episode_rate
  tie_breakers: [train/reward/mean]
overrides:
  train:
    learning_rate: 0.0001
  reward:
    death_penalty: 0
""",
                encoding="utf-8",
            )

            loaded = job_queue.load_spec_document(spec)

        self.assertEqual(loaded["train_config"]["game"], "SuperMarioBros-Nes-v0")
        self.assertEqual(loaded["train_config"]["state"], "Level1-1")
        self.assertEqual(loaded["train_config"]["learning_rate"], 0.0001)
        self.assertEqual(loaded["train_config"]["death_penalty"], 0)
        self.assertEqual(loaded["train_config"]["done_on_events"], ["life_loss", "level_change"])
        self.assertEqual(loaded["environment"]["env_id"], "stable-retro-turbo:SuperMarioBros-Nes-v0")
        self.assertTrue(loaded["environment_hash"].startswith("sha256:"))
        self.assertEqual(len(loaded["_composition"]["source_files"]), 2)

    def test_load_spec_document_materializes_first_class_environment_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.yaml"
            path.write_text(
                """
schema_version: 1
kind: train_experiment
goal: Level1-1
slug: candidate
stage: screen
hypothesis: Candidate should reproduce the expected completion signal.
expected_signal: Rank by completion rate, then reward.
parent_spec_slug: null
priority: 7
seeds: [23]
run_target: rtx4090
environment:
  env_id: stable-retro-turbo:SuperMarioBros-Nes-v0
  state: Level1-1
  action:
    action_set: simple
  preprocessing:
    frame_skip: 4
    max_pool_frames: false
    obs_resize: [84, 84]
    obs_crop: [32, 0, 0, 0]
  termination:
    max_episode_steps: 4500
    info_events_json:
      life_loss: [lives, decrease]
    done_on_events: [life_loss]
  reward:
    reward_mode: score
    death_penalty: 25
wandb_group: b-test
wandb_tags: [mario, env-hash]
run_name_template: btest_s{seed}_{utc}
run_description_template: candidate seed {seed}
selection_gate:
  primary: train/completion_episode_rate
train:
  timesteps: 1024
logging:
  wandb: true
  wandb_mode: online
""",
                encoding="utf-8",
            )

            loaded = job_queue.load_spec_document(path)

        self.assertEqual(loaded["train_config"]["game"], "SuperMarioBros-Nes-v0")
        self.assertEqual(loaded["train_config"]["state"], "Level1-1")
        self.assertEqual(loaded["train_config"]["frame_skip"], 4)
        self.assertEqual(loaded["train_config"]["hud_crop_top"], 32)
        self.assertNotIn("obs_crop", loaded["train_config"])
        self.assertEqual(loaded["train_config"]["observation_size"], 84)
        self.assertNotIn("obs_resize", loaded["train_config"])
        self.assertEqual(loaded["train_config"]["death_penalty"], 25)
        self.assertEqual(loaded["environment"]["env_id"], "stable-retro-turbo:SuperMarioBros-Nes-v0")
        self.assertEqual(loaded["environment"]["state"], "Level1-1")
        self.assertNotIn("hud_crop_top", loaded["environment"]["preprocessing"])
        self.assertEqual(loaded["environment"]["preprocessing"]["obs_crop"], [32, 0, 0, 0])
        self.assertNotIn("observation_size", loaded["environment"]["preprocessing"])
        self.assertEqual(loaded["environment"]["preprocessing"]["obs_resize"], [84, 84])
        self.assertTrue(loaded["environment_hash"].startswith("sha256:"))

    def test_load_spec_document_rejects_missing_mandatory_schema_field(self) -> None:
        spec = valid_train_spec()
        del spec["run_description_template"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps(spec), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "run_description_template"):
                job_queue.load_spec_document(path)

    def test_load_spec_document_rejects_non_compliant_schema_field(self) -> None:
        spec = valid_train_spec()
        spec["run_name_template"] = "candidate_{utc}"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps(spec), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "run_name_template.*seed"):
                job_queue.load_spec_document(path)

    def test_checked_in_goal_yaml_specs_match_train_spec_schema(self) -> None:
        spec_paths = sorted(Path("experiments/goals").glob("*/specs/*.y*ml"))
        self.assertGreater(len(spec_paths), 0)
        for path in spec_paths:
            with self.subTest(path=str(path)):
                job_queue.load_spec_document(path)

    def test_level1_3_specs_configure_goal_metric_early_stop(self) -> None:
        spec_paths = sorted(Path("experiments/goals/Level1-3/specs").glob("*.yaml"))
        self.assertGreater(len(spec_paths), 0)
        for path in spec_paths:
            with self.subTest(path=str(path)):
                spec = job_queue.load_spec_document(path)
                train_config = spec["train_config"]
                self.assertEqual(
                    train_config["early_stop_metric"],
                    TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                )
                self.assertEqual(train_config["early_stop_threshold"], 0.99)
                self.assertEqual(train_config["early_stop_operator"], ">")

    def test_record_running_train_result_upserts_wandb_url(self) -> None:
        conn = FakeConnection()

        job_queue.record_running_train_result(
            conn,
            job={
                "id": 12,
                "goal_slug": "goal",
                "spec_slug": "spec",
                "profile_id": None,
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "run_target": "rtx4090",
                "run_name": "candidate",
            },
            result={
                "run_name": "candidate",
                "run_dir": "runs/candidate",
                "wandb_run_id": "abc123",
                "wandb_url": "https://wandb.ai/tsilva/SuperMarioBros-NES/runs/abc123",
                "artifact_refs": [],
                "metrics_json": {"train/done/all": 20},
            },
        )

        self.assertIn("INSERT INTO train_results", conn.cursor_obj.executed_sql)
        self.assertIn("ON CONFLICT (train_job_id) DO UPDATE", conn.cursor_obj.executed_sql)
        self.assertEqual(conn.cursor_obj.executed_params["train_job_id"], 12)
        self.assertEqual(
            conn.cursor_obj.executed_params["wandb_url"],
            "https://wandb.ai/tsilva/SuperMarioBros-NES/runs/abc123",
        )
        self.assertEqual(conn.cursor_obj.executed_params["run_target"], "rtx4090")

    def test_list_stale_train_jobs_filters_target_prefix_and_age(self) -> None:
        conn = FakeConnection(
            rows=[
                {
                    "id": 12,
                    "profile_id": None,
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "run_target": "rtx2060",
                    "run_name": "candidate",
                    "stale_lease_owner": "rlab-beast-2-rtx2060-any-profile-cccc-0-deadbeef",
                    "stale_heartbeat_at": None,
                }
            ]
        )

        rows = job_queue.list_stale_train_jobs(
            conn,
            run_target="rtx2060",
            lease_owner_prefix="rlab-beast-2-",
            older_than_seconds=600,
            limit=25,
        )

        self.assertEqual(rows[0]["id"], 12)
        self.assertIn("FROM train_jobs", conn.cursor_obj.executed_sql)
        self.assertIn("status = 'running'", conn.cursor_obj.executed_sql)
        self.assertIn("run_target = %(run_target)s", conn.cursor_obj.executed_sql)
        self.assertIn("lease_owner LIKE %(lease_owner_like)s", conn.cursor_obj.executed_sql)
        self.assertNotIn("UPDATE train_jobs", conn.cursor_obj.executed_sql)
        self.assertEqual(conn.cursor_obj.executed_params["run_target"], "rtx2060")
        self.assertEqual(conn.cursor_obj.executed_params["lease_owner_like"], "rlab-beast-2-%")
        self.assertEqual(conn.cursor_obj.executed_params["older_than_seconds"], 600)
        self.assertEqual(conn.cursor_obj.executed_params["limit"], 25)

    def test_mark_stale_train_jobs_failed_updates_job_and_result(self) -> None:
        conn = FakeConnection(rows=[{"id": 12, "stale_lease_owner": "rlab-beast-2-x"}])

        rows = job_queue.mark_stale_train_jobs_failed(
            conn,
            job_ids=[12],
            run_target="rtx2060",
            lease_owner_prefix="rlab-beast-2-",
            older_than_seconds=1,
            error="worker_lost: beast-2 powered off",
        )

        self.assertEqual(rows[0]["id"], 12)
        self.assertIn("WITH candidates AS", conn.cursor_obj.executed_sql)
        self.assertIn("FOR UPDATE SKIP LOCKED", conn.cursor_obj.executed_sql)
        self.assertIn("UPDATE train_jobs AS job", conn.cursor_obj.executed_sql)
        self.assertIn("INSERT INTO train_results", conn.cursor_obj.executed_sql)
        self.assertIn("status = 'failed'", conn.cursor_obj.executed_sql)
        self.assertEqual(conn.cursor_obj.executed_params["job_ids"], [12])
        self.assertEqual(
            conn.cursor_obj.executed_params["error"],
            "worker_lost: beast-2 powered off",
        )

    def test_mark_stale_failed_default_apply_requires_scope_or_all(self) -> None:
        args = job_queue.build_parser().parse_args(["mark-stale-failed"])

        with self.assertRaisesRegex(SystemExit, "refusing unscoped"):
            job_queue.cmd_mark_stale_failed(args)

    def test_dry_run_replaces_execute_flag(self) -> None:
        args = job_queue.build_parser().parse_args(["mark-stale-failed", "--dry-run"])

        self.assertFalse(args.execute)
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            job_queue.build_parser().parse_args(["mark-stale-failed", "--" + "execute"])

    def test_enqueue_train_job_persists_runtime_image_only(self) -> None:
        conn = FakeConnection(
            row={
                "id": 9,
                "profile_id": None,
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "run_target": None,
            }
        )

        row = job_queue.enqueue_train_job(
            conn,
            goal_slug="goal",
            spec_slug="spec",
            profile_id="mario-ppo/post21/rtx4090",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            train_config={"timesteps": 1024},
        )

        self.assertEqual(row["runtime_image_ref"], RUNTIME_IMAGE_REF)
        all_sql = "\n".join(conn.cursor_obj.executed_sqls)
        self.assertIn("runtime_image_ref", all_sql)
        insert_params = conn.cursor_obj.executed_params_list[0]
        self.assertEqual(insert_params["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertIsNone(insert_params["profile_id"])
        self.assertIsNone(insert_params["run_target"])
        self.assertEqual(insert_params["goal_slug"], "goal")
        self.assertEqual(insert_params["spec_slug"], "spec")

    def test_enqueue_train_job_allows_profileless_digest_locked_jobs(self) -> None:
        conn = FakeConnection(
            row={
                "id": 9,
                "profile_id": None,
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "run_target": None,
            }
        )

        row = job_queue.enqueue_train_job(
            conn,
            goal_slug="goal",
            spec_slug="spec",
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            train_config={"timesteps": 1024},
        )

        self.assertIsNone(row["profile_id"])
        insert_params = conn.cursor_obj.executed_params_list[0]
        self.assertIsNone(insert_params["profile_id"])
        self.assertEqual(insert_params["runtime_image_ref"], RUNTIME_IMAGE_REF)

    def test_enqueue_train_job_rejects_legacy_event_launch_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy event key.*done_on_info_json"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                spec_slug="spec",
                profile_id=None,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                run_target="rtx4090",
                train_config={
                    "timesteps": 1024,
                    "done_on_info_json": {
                        "level_change": [["levelHi", "levelLo"], "change"],
                    },
                },
            )

    def test_enqueue_train_job_requires_done_events_to_be_info_events(self) -> None:
        with self.assertRaisesRegex(ValueError, "references unconfigured info event"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                spec_slug="spec",
                profile_id=None,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                run_target="rtx4090",
                train_config={
                    "timesteps": 1024,
                    "info_events_json": {"life_loss": ["lives", "decrease"]},
                    "done_on_events": "life_loss,level_change",
                },
            )

    def test_enqueue_train_job_rejects_eval_reserved_seed_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved for eval"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                spec_slug="spec",
                profile_id=None,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                run_target="rtx4090",
                train_config={"timesteps": 1024, "seed": DEFAULT_EVAL_SEED},
            )

        with self.assertRaisesRegex(ValueError, "training env slot"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                spec_slug="spec",
                profile_id=None,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                run_target="rtx4090",
                train_config={"timesteps": 1024, "seed": 9999, "n_envs": 2},
            )

    def test_enqueue_train_job_rejects_mutable_runtime_tag(self) -> None:
        conn = FakeConnection(row={"id": 9})

        with self.assertRaisesRegex(ValueError, "immutable docker digest ref"):
            job_queue.enqueue_train_job(
                conn,
                goal_slug="goal",
                spec_slug="spec",
                profile_id="mario-ppo/post21/rtx4090",
                runtime_image_ref="docker:ghcr.io/tsilva/rlab/rlab-train:latest",
                train_config={"timesteps": 1024},
            )

    def test_runtime_image_ref_from_args_defaults_to_latest_digest(self) -> None:
        args = SimpleNamespace(
            runtime_image_ref=None,
            runtime_image_ref_file=None,
            latest_image=False,
            image_workflow="workflow",
            image_branch="main",
            image_artifact="artifact",
        )
        original = job_queue.latest_runtime_image_ref
        calls = []

        def fake_latest_runtime_image_ref(**kwargs):
            calls.append(kwargs)
            return RUNTIME_IMAGE_REF

        job_queue.latest_runtime_image_ref = fake_latest_runtime_image_ref
        try:
            self.assertEqual(
                job_queue.runtime_image_ref_from_args(args, default_latest=True),
                RUNTIME_IMAGE_REF,
            )
        finally:
            job_queue.latest_runtime_image_ref = original
        self.assertEqual(
            calls,
            [{"workflow": "workflow", "branch": "main", "artifact_name": "artifact"}],
        )

    def test_claim_eval_job_filters_exact_runtime_image(self) -> None:
        conn = FakeConnection(row={"id": 8, "profile_id": "mario-ppo/post16/rtx4090-eval"})

        row = job_queue.claim_eval_job(
            conn,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            worker_id="worker-a",
            lease_seconds=60,
        )

        self.assertEqual(row["id"], 8)
        self.assertIn("runtime_image_ref = %(runtime_image_ref)s", conn.cursor_obj.executed_sql)
        self.assertNotIn("profile_id = %(profile_id)s", conn.cursor_obj.executed_sql)
        self.assertEqual(conn.cursor_obj.executed_params["runtime_image_ref"], RUNTIME_IMAGE_REF)

    def test_claim_eval_job_does_not_reclaim_expired_running_leases(self) -> None:
        conn = FakeConnection(row=None)

        row = job_queue.claim_eval_job(
            conn,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            worker_id="worker-a",
            lease_seconds=60,
        )

        self.assertIsNone(row)
        self.assertIn("AND status = 'pending'", conn.cursor_obj.executed_sql)
        self.assertNotIn("lease_expires_at < now()", conn.cursor_obj.executed_sql)
        self.assertNotIn("attempts < max_attempts", conn.cursor_obj.executed_sql)

    def test_enqueue_eval_job_persists_runtime_artifact_and_protocol(self) -> None:
        conn = FakeConnection(
            row={
                "id": 10,
                "profile_id": None,
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "artifact_ref": "entity/project/model:step-100",
                "eval_protocol_hash": "abc123",
            }
        )

        row = job_queue.enqueue_eval_job(
            conn,
            goal_slug="Level1-1",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            artifact_ref="entity/project/model:step-100",
            checkpoint_step=100,
            eval_protocol_hash="abc123",
            eval_config={"episodes": 100, "seed": DEFAULT_EVAL_SEED},
        )

        self.assertEqual(row["runtime_image_ref"], RUNTIME_IMAGE_REF)
        sql = conn.cursor_obj.executed_sqls[0]
        params = conn.cursor_obj.executed_params_list[0]
        self.assertIn("ON CONFLICT (artifact_ref, eval_protocol_hash)", sql)
        self.assertIsNone(params["profile_id"])
        self.assertEqual(params["artifact_ref"], "entity/project/model:step-100")
        self.assertEqual(params["checkpoint_step"], 100)
        self.assertEqual(params["eval_protocol_hash"], "abc123")

    def test_record_eval_wandb_status_updates_projection_fields(self) -> None:
        conn = FakeConnection()

        job_queue.record_eval_wandb_status(
            conn,
            eval_job_id=12,
            wandb_run_id="run-id",
            logged=True,
            log_step=500000,
        )

        self.assertIn("UPDATE eval_results", conn.cursor_obj.executed_sql)
        self.assertEqual(conn.cursor_obj.executed_params["eval_job_id"], 12)
        self.assertEqual(conn.cursor_obj.executed_params["wandb_run_id"], "run-id")
        self.assertTrue(conn.cursor_obj.executed_params["wandb_logged"])
        self.assertEqual(conn.cursor_obj.executed_params["wandb_log_step"], 500000)

    def test_checkpoint_artifact_upload_enqueues_goal_eval_job(self) -> None:
        calls: list[dict] = []

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {"id": 77, **kwargs}

        job = {
            "id": 12,
            "goal_slug": "Level1-1",
            "spec_slug": "candidate",
            "spec_path": "experiments/goals/Level1-1/specs/candidate.yaml",
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "priority": 7,
            "run_name": "candidate_run",
            "train_config": {
                "run_name": "candidate_run",
                "wandb_entity": "entity",
                "wandb_project": "project",
            },
        }
        result = {
            "artifact_refs": [
                {
                    "name": "candidate_run-checkpoint",
                    "location": "s3://bucket/path/ppo_test_500000_steps.zip",
                },
                {"name": "candidate_run-final", "location": "s3://bucket/path/final_model.zip"},
            ]
        }
        eval_spec = {
            "schema_version": 1,
            "eval_config": {
                "episodes": 100,
                "seed": DEFAULT_EVAL_SEED,
                "n_envs": 20,
                "max_steps": 4500,
                "stochastic": True,
            },
        }

        with (
            patch("rlab.train_runner.load_goal_eval_spec", return_value=eval_spec),
            patch("rlab.train_runner.enqueue_eval_job", side_effect=fake_enqueue),
            patch("rlab.train_runner.eval_protocol_hash", return_value="protocol-hash"),
        ):
            count = enqueue_eval_jobs_for_checkpoints(FakeConnection(), job, result)

        self.assertEqual(count, 1)
        self.assertEqual(calls[0]["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertEqual(calls[0]["artifact_ref"], "entity/project/candidate_run-checkpoint:step-500000")
        self.assertEqual(calls[0]["checkpoint_step"], 500000)
        self.assertEqual(calls[0]["eval_protocol_hash"], "protocol-hash")
        self.assertEqual(calls[0]["eval_config"]["artifact_ref"], calls[0]["artifact_ref"])

    def test_log_eval_to_wandb_uses_checkpoint_global_step(self) -> None:
        class FakeRun:
            def __init__(self) -> None:
                self.logged: list[tuple[dict, int | None]] = []
                self.finished = False

            def log(self, payload: dict, step: int | None = None) -> None:
                self.logged.append((payload, step))

            def finish(self) -> None:
                self.finished = True

        class FakeWandb:
            def __init__(self) -> None:
                self.run = FakeRun()

            def init(self, **kwargs):
                self.init_kwargs = kwargs
                return self.run

        fake_wandb = FakeWandb()
        metrics = {
            "reward_mean": 12.5,
            "reward_std": 1.0,
            "reward_max": 15.0,
            "max_x_mean": 300.0,
            "max_x_max": 500,
            "max_level_x_mean": 300.0,
            "max_level_x_max": 500,
            "death_count": 1,
            "death_rate": 0.01,
            "best_episode": {"reward": 15.0, "max_x_pos": 500},
            "eval/done/level_change/rate": 1.0,
        }

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.zip"
            model_path.write_bytes(b"zip")
            model_path.with_suffix(".metadata.json").write_text(
                json.dumps(
                    {
                        "wandb_run_id": "run-id",
                        "wandb_run_path": "entity/project/runs/run-id",
                        "checkpoint_step": 500000,
                    }
                ),
                encoding="utf-8",
            )
            with (
                patch.dict(sys.modules, {"wandb": fake_wandb}),
                patch("rlab.eval_job_runner.load_wandb_env"),
            ):
                result = log_eval_to_wandb(
                    job={"id": 9, "eval_protocol_hash": "protocol"},
                    config={"artifact_ref": "entity/project/model:step-500000"},
                    model_path=model_path,
                    metrics=metrics,
                    video_path=None,
                )

        payload, step = fake_wandb.run.logged[0]
        self.assertTrue(result["wandb_logged"])
        self.assertEqual(result["wandb_run_id"], "run-id")
        self.assertEqual(step, 500000)
        self.assertEqual(payload["global_step"], 500000)
        self.assertEqual(payload["eval/reward/mean"], 12.5)
        self.assertEqual(payload["eval/checkpoint/step"], 500000)
        self.assertEqual(payload["eval/done/level_change/rate"], 1.0)
        self.assertTrue(fake_wandb.run.finished)

    def test_parser_removed_research_db_commands(self) -> None:
        parser = job_queue.build_parser()
        for command in (
            "create-goal",
            "add-spec",
            "add-spec-file",
            "enqueue-train-from-spec",
            "decision",
            "lineage",
        ):
            with self.subTest(command=command):
                with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                    parser.parse_args([command])

    def test_train_parser_uses_spec_file_for_train_enqueue(self) -> None:
        args = rlab_main.build_train_enqueue_parser().parse_args(
            [
                "--spec-file",
                "experiments/goals/example/specs/candidate.yaml",
                "--runtime-image-ref-file",
                "rlab-train-image.json",
            ]
        )

        self.assertEqual(args.spec_file, Path("experiments/goals/example/specs/candidate.yaml"))

    def test_jobs_parser_no_longer_owns_train_enqueue(self) -> None:
        with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
            job_queue.build_parser().parse_args(
                [
                    "enqueue-train",
                    "--spec-file",
                    "experiments/goals/example/specs/candidate.yaml",
                ]
            )

    def test_eval_selection_score_prefers_eval_min_completion_then_progress(self) -> None:
        weak_pooled = {
            "completion_rate": 1.0,
            "eval/done/level_change/from_rate/min": 0.25,
            "max_x_max": 4000,
            "reward_mean": 900.0,
        }
        balanced = {
            "completion_rate": 0.8,
            "eval/done/level_change/from_rate/min": 0.75,
            "max_x_max": 3200,
            "reward_mean": 600.0,
        }

        self.assertGreater(
            job_queue.eval_selection_score(balanced),
            job_queue.eval_selection_score(weak_pooled),
        )

    def test_enqueue_train_jobs_from_spec_expands_seed_templates(self) -> None:
        calls = []
        old_enqueue = job_queue.enqueue_train_job
        old_utc = job_queue._utc_stamp

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {
                "id": 100 + len(calls),
                "profile_id": kwargs["profile_id"],
                "run_name": kwargs["run_name"],
                "run_target": kwargs["run_target"],
            }

        job_queue.enqueue_train_job = fake_enqueue
        job_queue._utc_stamp = lambda: "20260626T120000Z"
        try:
            document = valid_train_spec()
            document["profile_id"] = "mario-ppo/post21/rtx4090"
            document["operator_note"] = "non-schema metadata persists"
            document["train_config"] = {
                **document["train_config"],
                "info_events_json": {
                    "life_loss": ["lives", "decrease"],
                    "level_change": [["levelHi", "levelLo"], "change"],
                },
                "done_on_events": "life_loss,level_change",
            }
            rows = job_queue.enqueue_train_jobs_from_spec_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                spec_path="experiments/goals/mario/specs/candidate.yaml",
                spec_sha256="abc123",
                repo_git_commit="deadbeef",
                repo_dirty=True,
                instances_path=Path("/tmp/does-not-exist.json"),
            )
        finally:
            job_queue.enqueue_train_job = old_enqueue
            job_queue._utc_stamp = old_utc

        self.assertEqual([row["run_name"] for row in rows], ["btest_s23_20260626T120000Z", "btest_s24_20260626T120000Z"])
        self.assertEqual([call["train_config"]["seed"] for call in calls], [23, 24])
        self.assertEqual(
            calls[0]["train_config"]["info_events_json"],
            {
                "life_loss": ["lives", "decrease"],
                "level_change": [["levelHi", "levelLo"], "change"],
            },
        )
        self.assertEqual(calls[0]["train_config"]["done_on_events"], "life_loss,level_change")
        self.assertNotIn("done_on_info_json", calls[0]["train_config"])
        self.assertEqual(calls[0]["priority"], 7)
        self.assertEqual(calls[0]["wandb_tags"], ["mario", "confirm"])
        self.assertEqual(calls[0]["goal_slug"], "Level1-1")
        self.assertEqual(calls[0]["spec_slug"], "candidate")
        self.assertIsNone(calls[0]["profile_id"])
        self.assertIsNone(calls[0]["run_target"])
        self.assertEqual(calls[0]["spec_path"], "experiments/goals/mario/specs/candidate.yaml")
        self.assertEqual(calls[0]["spec_sha256"], "abc123")
        self.assertEqual(calls[0]["repo_git_commit"], "deadbeef")
        self.assertTrue(calls[0]["repo_dirty"])
        self.assertEqual(calls[0]["spec_payload"]["operator_note"], "non-schema metadata persists")

    def test_enqueue_train_jobs_from_spec_document_enforces_schema(self) -> None:
        document = copy.deepcopy(valid_train_spec())
        del document["schema_version"]

        with self.assertRaisesRegex(ValueError, "schema_version"):
            job_queue.enqueue_train_jobs_from_spec_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                instances_path=Path("/tmp/does-not-exist.json"),
            )

    def test_enqueue_train_jobs_from_spec_document_rejects_eval_reserved_seed(self) -> None:
        document = copy.deepcopy(valid_train_spec())
        document["seeds"] = [DEFAULT_EVAL_SEED]

        with self.assertRaisesRegex(ValueError, "reserved for eval"):
            job_queue.enqueue_train_jobs_from_spec_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                instances_path=Path("/tmp/does-not-exist.json"),
            )


class TrainRunnerAutoscaleTests(unittest.TestCase):
    def train_runner_args(self, *extra: str):
        return build_train_runner_parser().parse_args(
            ["--runtime-image-ref", RUNTIME_IMAGE_REF, *extra]
        )

    def test_fixed_mode_worker_bounds_preserve_workers(self) -> None:
        args = self.train_runner_args("--workers", "3")

        bounds = resolve_worker_bounds(args)

        self.assertEqual(bounds.starter_workers, 3)
        self.assertEqual(bounds.min_workers, 3)
        self.assertEqual(bounds.max_workers, 3)

    def test_fixed_mode_defaults_to_four_workers(self) -> None:
        args = self.train_runner_args()

        bounds = resolve_worker_bounds(args)

        self.assertEqual(bounds.starter_workers, 4)
        self.assertEqual(bounds.min_workers, 4)
        self.assertEqual(bounds.max_workers, 4)

    def test_autoscale_defaults_to_min_one_start_four_max_sixteen(self) -> None:
        args = self.train_runner_args("--autoscale")

        bounds = resolve_worker_bounds(args)

        self.assertEqual(bounds.starter_workers, 4)
        self.assertEqual(bounds.min_workers, 1)
        self.assertEqual(bounds.max_workers, 16)

    def test_autoscale_rejects_invalid_worker_range(self) -> None:
        args = self.train_runner_args(
            "--workers",
            "1",
            "--autoscale",
            "--min-workers",
            "2",
            "--max-workers",
            "5",
        )

        with self.assertRaisesRegex(SystemExit, "--min-workers <= --workers <= --max-workers"):
            resolve_worker_bounds(args)

    def test_autoscale_scales_up_with_headroom_and_pending_jobs(self) -> None:
        controller = AutoscaleController(
            AutoscaleConfig(
                starter_workers=2,
                min_workers=1,
                max_workers=5,
                window_size=2,
                cooldown_seconds=0,
            )
        )
        controller.observe(ResourceSample(cpu_percent=50, memory_percent=50, gpu_percent=50, vram_percent=50))
        controller.observe(ResourceSample(cpu_percent=55, memory_percent=55, gpu_percent=55, vram_percent=55))

        decision = controller.decide(pending_jobs=True, active_workers=2, now=10)

        self.assertEqual(decision.action, "scale_up")
        self.assertEqual(decision.target_workers, 3)

    def test_autoscale_does_not_scale_up_without_pending_jobs(self) -> None:
        controller = AutoscaleController(
            AutoscaleConfig(
                starter_workers=2,
                min_workers=1,
                max_workers=5,
                window_size=1,
                cooldown_seconds=0,
            )
        )
        controller.observe(ResourceSample(cpu_percent=50, memory_percent=50, gpu_percent=50, vram_percent=50))

        decision = controller.decide(pending_jobs=False, active_workers=2, now=10)

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.target_workers, 2)
        self.assertIn("no pending", decision.reason)

    def test_autoscale_scales_down_on_resource_saturation(self) -> None:
        controller = AutoscaleController(
            AutoscaleConfig(
                starter_workers=3,
                min_workers=1,
                max_workers=5,
                window_size=2,
                cooldown_seconds=0,
            )
        )
        controller.observe(ResourceSample(cpu_percent=91, memory_percent=50, gpu_percent=50, vram_percent=50))
        controller.observe(ResourceSample(cpu_percent=92, memory_percent=50, gpu_percent=50, vram_percent=50))

        decision = controller.decide(pending_jobs=True, active_workers=3, now=10)

        self.assertEqual(decision.action, "scale_down")
        self.assertEqual(decision.target_workers, 2)
        self.assertIn("cpu_percent", decision.reason)

    def test_autoscale_respects_min_and_max_bounds(self) -> None:
        controller = AutoscaleController(
            AutoscaleConfig(
                starter_workers=1,
                min_workers=1,
                max_workers=1,
                window_size=1,
                cooldown_seconds=0,
            )
        )
        controller.observe(ResourceSample(cpu_percent=10, memory_percent=10, gpu_percent=10, vram_percent=10))

        decision = controller.decide(pending_jobs=True, active_workers=1, now=10)

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.target_workers, 1)
        self.assertIn("max workers", decision.reason)

    def test_autoscale_holds_when_probe_fails(self) -> None:
        controller = AutoscaleController(
            AutoscaleConfig(
                starter_workers=2,
                min_workers=1,
                max_workers=5,
                window_size=1,
                cooldown_seconds=0,
            )
        )
        controller.observe(ResourceSample(error="nvidia-smi timed out"))

        decision = controller.decide(pending_jobs=True, active_workers=2, now=10)

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.target_workers, 2)
        self.assertIn("resource sample failed", decision.reason)

    def test_surplus_workers_retire_idle_slots_before_busy_slots(self) -> None:
        idle = WorkerSlot(index=0, worker_id="worker-0", state=WORKER_IDLE)
        busy_a = WorkerSlot(index=1, worker_id="worker-1", state=WORKER_RUNNING)
        busy_b = WorkerSlot(index=2, worker_id="worker-2", state=WORKER_RUNNING)

        retired = mark_surplus_workers_for_retirement(
            [idle, busy_a, busy_b],
            target_workers=1,
        )

        self.assertEqual(retired, ("worker-0", "worker-1"))
        self.assertEqual(idle.snapshot()["state"], WORKER_RETIRING)
        self.assertTrue(busy_a.snapshot()["retire_requested"])
        self.assertFalse(busy_b.snapshot()["retire_requested"])

    def test_pending_train_probe_matches_runner_claim_scope(self) -> None:
        conn = FakeConnection(row={"has_pending": True})
        args = self.train_runner_args("--run-target", "rtx4090")

        self.assertTrue(matching_pending_train_job_exists(conn, args))

        self.assertIn("status = 'pending'", conn.cursor_obj.executed_sql)
        self.assertIn("cancel_requested = FALSE", conn.cursor_obj.executed_sql)
        self.assertIn("runtime_image_ref = %(runtime_image_ref)s", conn.cursor_obj.executed_sql)
        self.assertNotIn("run_target", conn.cursor_obj.executed_sql)
        self.assertNotIn("profile_id", conn.cursor_obj.executed_sql)
        self.assertEqual(conn.cursor_obj.executed_params["runtime_image_ref"], RUNTIME_IMAGE_REF)


class TrainRunnerTests(unittest.TestCase):
    def test_checkpoint_bucket_placeholder_resolves_before_command_build(self) -> None:
        old_value = os.environ.get("CHECKPOINT_BUCKET_URI")
        os.environ["CHECKPOINT_BUCKET_URI"] = '"s3://bucket/checkpoints"'
        try:
            job = {
                "id": 13,
                "train_config": {
                    "game": "SuperMarioBros-Nes-v0",
                    "timesteps": 1024,
                    "state": "Level1-2",
                    "wandb_artifact_storage_uri": "${CHECKPOINT_BUCKET_URI}",
                },
                "run_name": "placeholder_candidate",
            }

            config = normalize_train_config(job)
            with tempfile.TemporaryDirectory() as tmp:
                config_path = write_train_config_file(job, Path(tmp) / "train_config.json")
                command = train_command_for_job(config_path)
                written_config = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(config["wandb_artifact_storage_uri"], "s3://bucket/checkpoints")
            self.assertEqual(
                written_config["wandb_artifact_storage_uri"],
                "s3://bucket/checkpoints",
            )
            self.assertIn("--train-config-json", command)
            self.assertIn("train_config.json", command[-1])
            self.assertNotIn('"s3://bucket/checkpoints"', command)
            self.assertNotIn("${CHECKPOINT_BUCKET_URI}", command)
        finally:
            if old_value is None:
                os.environ.pop("CHECKPOINT_BUCKET_URI", None)
            else:
                os.environ["CHECKPOINT_BUCKET_URI"] = old_value

    def test_resume_artifact_resolves_to_local_resume_path(self) -> None:
        import rlab.train_runner as train_runner

        calls = []
        old_download = train_runner.download_model_artifact

        def fake_download(ref, root):
            calls.append((ref, root))
            return Path("/tmp/downloaded/model.zip")

        train_runner.download_model_artifact = fake_download
        try:
            job = {
                "id": 14,
                "train_config": {
                    "game": "SuperMarioBros-Nes-v0",
                    "timesteps": 1024,
                    "resume_artifact": "entity/project/run-checkpoint:step-5000000",
                },
                "run_name": "resume_candidate",
            }

            config = normalize_train_config(job)

            self.assertEqual(
                calls,
                [
                    (
                        "entity/project/run-checkpoint:step-5000000",
                        train_runner.RESUME_ARTIFACT_ROOT
                        / "entity_project_run-checkpoint_step-5000000",
                    )
                ],
            )
            self.assertEqual(config["resume"], "/tmp/downloaded/model.zip")
            self.assertNotIn("resume_artifact", config)
            calls.clear()

            with tempfile.TemporaryDirectory() as tmp:
                config_path = write_train_config_file(job, Path(tmp) / "train_config.json")
                command = train_command_for_job(config_path)
                written_config = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(
                calls,
                [
                    (
                        "entity/project/run-checkpoint:step-5000000",
                        train_runner.RESUME_ARTIFACT_ROOT
                        / "entity_project_run-checkpoint_step-5000000",
                    )
                ],
            )
            self.assertEqual(written_config["resume"], "/tmp/downloaded/model.zip")
            self.assertIn("--train-config-json", command)
        finally:
            train_runner.download_model_artifact = old_download

    def test_collect_result_metadata_does_not_resolve_resume_artifact(self) -> None:
        import rlab.train_runner as train_runner

        old_download = train_runner.download_model_artifact

        def fake_download(ref, root):
            raise AssertionError("result collection should not download resume artifacts")

        train_runner.download_model_artifact = fake_download
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                run_dir = root / "runs" / "resume_candidate"
                log_path = root / "train.log"
                run_dir.mkdir(parents=True)
                log_path.write_text("done\n", encoding="utf-8")
                job = {
                    "id": 16,
                    "run_name": "resume_candidate",
                    "train_config": {
                        "runs_dir": str(root / "runs"),
                        "resume_artifact": "entity/project/run-checkpoint:latest",
                    },
                }

                result = collect_result_metadata(job, log_path)

            self.assertEqual(result["run_name"], "resume_candidate")
        finally:
            train_runner.download_model_artifact = old_download

    def test_resume_and_resume_artifact_conflict_is_rejected(self) -> None:
        job = {
            "id": 15,
            "train_config": {
                "resume": "/tmp/local.zip",
                "resume_artifact": "entity/project/run-final:latest",
            },
            "run_name": "bad_resume_candidate",
        }

        with self.assertRaisesRegex(ValueError, "Use only one of resume or resume_artifact"):
            normalize_train_config(job)

    def test_normalize_train_config_rejects_eval_reserved_seed_range(self) -> None:
        job = {
            "id": 16,
            "train_config": {"seed": DEFAULT_EVAL_SEED},
            "run_name": "bad_seed_candidate",
        }

        with self.assertRaisesRegex(ValueError, "reserved for eval"):
            normalize_train_config(job)

    def test_train_command_uses_job_profile_config_without_secrets(self) -> None:
        job = {
            "id": 12,
            "train_config": {
                "game": "SuperMarioBros-Nes-v0",
                "timesteps": 1024,
                "states": ["Level1-1", "Level1-2"],
                "wandb": True,
                "wandb_tags": ["screen", "post16"],
            },
            "goal_slug": "Levels_2_d25102",
            "run_name": "b52_seed23",
            "run_description": "Codex-authored smoke job.",
            "wandb_group": "b52",
            "wandb_tags": ["fallback"],
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "run_target": "rtx4090",
        }

        config = normalize_train_config(job)
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_train_config_file(job, Path(tmp) / "train_config.json")
            command = train_command_for_job(config_path)
            written_config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["wandb_tags"],
            "screen,post16,goal:Levels_2_d25102,level:Level1-1,level:Level1-2",
        )
        self.assertEqual(written_config["run_name"], "b52_seed23")
        self.assertEqual(written_config["states"], ["Level1-1", "Level1-2"])
        self.assertEqual(written_config["wandb_group"], "b52")
        self.assertEqual(written_config["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertEqual(written_config["run_target"], "rtx4090")
        self.assertTrue(written_config["wandb"])
        self.assertEqual(command[1:4], ["-m", "rlab.train", "--train-config-json"])
        self.assertNotIn("--run-name", command)
        self.assertNotIn("--states", command)

    def test_collect_result_metadata_reads_run_markers_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "candidate"
            log_path = root / "train.log"
            run_dir.mkdir(parents=True)
            (run_dir / "final_model.zip").write_bytes(b"model")
            (run_dir / "wandb_url.txt").write_text(
                "https://wandb.ai/e/p/runs/abc\n",
                encoding="utf-8",
            )
            (run_dir / "wandb_run_id.txt").write_text("abc\n", encoding="utf-8")
            (run_dir / "early_stop.txt").write_text(
                "completion_rate=1.000000\n"
                "timesteps=3881520\n",
                encoding="utf-8",
            )
            log_path.write_text(
                "wandb artifact logged: candidate-final "
                "(s3://bucket/SuperMarioBros-Nes-v0/candidate/final_model.zip)\n",
                encoding="utf-8",
            )
            job = {
                "id": 3,
                "run_name": "candidate",
                "train_config": {"runs_dir": str(root / "runs")},
            }

            result = collect_result_metadata(job, log_path)

        self.assertEqual(result["wandb_run_id"], "abc")
        self.assertEqual(result["metrics_json"]["completion_rate"], "1.000000")
        self.assertEqual(result["artifact_refs"][0]["name"], "candidate-final")
        self.assertTrue(result["final_model_path"].endswith("final_model.zip"))

    def test_collect_result_metadata_parses_normal_completion_log_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "candidate"
            log_path = root / "train.log"
            run_dir.mkdir(parents=True)
            (run_dir / "final_model.zip").write_bytes(b"model")
            log_path.write_text(
                "\n".join(
                    [
                        "wandb: 🚀 View run at "
                        "https://wandb.ai/tsilva/SuperMarioBros-NES/runs/abc123",
                        "|    total_timesteps                | 256         |",
                        "| train/done/                       |             |",
                        "|    all                            | 10          |",
                        "|    total_timesteps                | 512         |",
                        "| time/                             |             |",
                        "|    fps                            | 240         |",
                        "| train/                            |             |",
                        "|    loss                           | 1.5         |",
                        "|    rollout/ep_rew_mean            | 3.02e+03    |",
                        "| train/done/                       |             |",
                        "|    all                            | 20          |",
                        "wandb artifact logged: candidate-final "
                        "(s3://bucket/SuperMarioBros-Nes-v0/candidate/final_model.zip)",
                    ]
                ),
                encoding="utf-8",
            )
            job = {
                "id": 3,
                "run_name": "candidate",
                "train_config": {"runs_dir": str(root / "runs")},
            }

            result = collect_result_metadata(job, log_path)

        self.assertEqual(
            result["wandb_url"],
            "https://wandb.ai/tsilva/SuperMarioBros-NES/runs/abc123",
        )
        self.assertEqual(result["metrics_json"]["total_timesteps"], 512)
        self.assertEqual(result["metrics_json"]["train/done/all"], 20)
        self.assertEqual(result["metrics_json"]["rollout/ep_rew_mean"], 3020.0)
        self.assertEqual(result["metrics_json"]["time/fps"], 240)
        self.assertEqual(result["metrics_json"]["train/loss"], 1.5)

    def test_parse_log_metrics_keeps_last_seen_values(self) -> None:
        metrics = parse_log_metrics(
            "\n".join(
                [
                    "|    total_timesteps                | 256         |",
                    "| train/done/                       |             |",
                    "|    all                            | 10          |",
                    "|    total_timesteps                | 512         |",
                    "| train/done/                       |             |",
                    "|    all                            | 20          |",
                ]
            )
        )

        self.assertEqual(metrics["total_timesteps"], 512)
        self.assertEqual(metrics["train/done/all"], 20)

    def test_parse_log_metrics_prefixes_sb3_sections(self) -> None:
        metrics = parse_log_metrics(
            "\n".join(
                [
                    "| rollout/                          |             |",
                    "|    ep_rew_mean                    | 3.02e+03    |",
                    "| time/                             |             |",
                    "|    fps                            | 240         |",
                    "| train/                            |             |",
                    "|    loss                           | 1.5         |",
                    "|    total_timesteps                | 1024        |",
                ]
            )
        )

        self.assertEqual(metrics["rollout/ep_rew_mean"], 3020.0)
        self.assertEqual(metrics["time/fps"], 240)
        self.assertEqual(metrics["train/loss"], 1.5)
        self.assertEqual(metrics["total_timesteps"], 1024)

    def test_successful_online_artifact_run_data_is_purged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "runs" / "candidate"
            run_dir.mkdir(parents=True)
            (run_dir / "final_model.zip").write_bytes(b"model")
            (run_dir / "wandb" / "cache").mkdir(parents=True)
            (run_dir / "wandb" / "cache" / "data").write_bytes(b"cache")
            job = {
                "id": 3,
                "run_name": "candidate",
                "train_config": {
                    "runs_dir": str(root / "runs"),
                    "wandb": True,
                    "wandb_mode": "online",
                },
            }
            result = {
                "run_dir": str(run_dir),
                "artifact_refs": [{"name": "candidate-final", "location": "s3://bucket/model.zip"}],
            }

            self.assertTrue(should_purge_successful_run_data(job, result))
            self.assertTrue(purge_successful_run_data(job, result))

            self.assertFalse(run_dir.exists())

    def test_successful_run_data_purge_refuses_paths_outside_runs_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs_dir = root / "runs"
            escaped_dir = root / "escaped"
            runs_dir.mkdir()
            escaped_dir.mkdir()
            (escaped_dir / "final_model.zip").write_bytes(b"model")
            job = {
                "id": 3,
                "run_name": "../escaped",
                "train_config": {
                    "runs_dir": str(runs_dir),
                    "wandb": True,
                    "wandb_mode": "online",
                },
            }
            result = {
                "run_dir": str(escaped_dir),
                "artifact_refs": [{"name": "candidate-final", "location": "s3://bucket/model.zip"}],
            }

            self.assertFalse(purge_successful_run_data(job, result))
            self.assertTrue(escaped_dir.exists())


class ArtifactConfigTests(unittest.TestCase):
    def test_load_env_file_strips_quotes_and_respects_filter(self) -> None:
        old_allowed = os.environ.get("RLAB_TEST_ALLOWED")
        old_blocked = os.environ.get("RLAB_TEST_BLOCKED")
        os.environ.pop("RLAB_TEST_ALLOWED", None)
        os.environ.pop("RLAB_TEST_BLOCKED", None)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / ".env"
                path.write_text(
                    "RLAB_TEST_ALLOWED='kept value'\nRLAB_TEST_BLOCKED=ignored\n",
                    encoding="utf-8",
                )
                load_env_file(path, key_filter=lambda key: key == "RLAB_TEST_ALLOWED")

            self.assertEqual(os.environ.get("RLAB_TEST_ALLOWED"), "kept value")
            self.assertIsNone(os.environ.get("RLAB_TEST_BLOCKED"))
        finally:
            if old_allowed is None:
                os.environ.pop("RLAB_TEST_ALLOWED", None)
            else:
                os.environ["RLAB_TEST_ALLOWED"] = old_allowed
            if old_blocked is None:
                os.environ.pop("RLAB_TEST_BLOCKED", None)
            else:
                os.environ["RLAB_TEST_BLOCKED"] = old_blocked

    def test_checkpoint_bucket_placeholder_uses_environment(self) -> None:
        old_value = os.environ.get("CHECKPOINT_BUCKET_URI")
        os.environ["CHECKPOINT_BUCKET_URI"] = '"s3://bucket/from-env"'
        try:
            args = SimpleNamespace(wandb_artifact_storage_uri="${CHECKPOINT_BUCKET_URI}")

            self.assertEqual(wandb_artifact_storage_uri(args), "s3://bucket/from-env")
        finally:
            if old_value is None:
                os.environ.pop("CHECKPOINT_BUCKET_URI", None)
            else:
                os.environ["CHECKPOINT_BUCKET_URI"] = old_value

    def test_configured_storage_uri_strips_env_file_quotes(self) -> None:
        args = SimpleNamespace(wandb_artifact_storage_uri='"s3://bucket/from-arg"')

        self.assertEqual(wandb_artifact_storage_uri(args), "s3://bucket/from-arg")


class EvalJobRunnerTests(unittest.TestCase):
    def test_normalize_eval_config_defaults_to_100_episode_stochastic_vector_eval(self) -> None:
        config = normalize_eval_config(
            {
                "id": 4,
                "eval_config": {"artifact_ref": "tsilva/SuperMarioBros-NES/model:v1"},
            }
        )

        self.assertEqual(config["episodes"], 100)
        self.assertEqual(config["n_envs"], 20)
        self.assertEqual(config["seed"], DEFAULT_EVAL_SEED)
        self.assertTrue(config["stochastic"])
        self.assertFalse(config["capture_best_video"])

    def test_normalize_eval_config_rejects_training_seed_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved for training"):
            normalize_eval_config({"id": 4, "eval_config": {"seed": 9999}})

    def test_json_safe_converts_nested_non_json_values(self) -> None:
        class Scalar:
            def item(self):
                return 7

        self.assertEqual(json_safe({"a": (Scalar(), Path("x"))}), {"a": [7, "x"]})


if __name__ == "__main__":
    unittest.main()
