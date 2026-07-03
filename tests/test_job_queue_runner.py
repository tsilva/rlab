from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import yaml

from rlab import job_queue
from rlab import main as rlab_main
from rlab import wandb_leaders
from rlab.artifacts import wandb_artifact_storage_uri
from rlab.dotenv import load_env_file
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


class FakeWandbRun:
    def __init__(
        self,
        *,
        run_id: str,
        name: str,
        config: dict,
        summary: dict,
        group: str = "",
        tags: tuple[str, ...] = (),
    ) -> None:
        self.id = run_id
        self.name = name
        self.config = config
        self.summary = summary
        self.group = group
        self.tags = tags
        self.url = f"https://wandb.ai/entity/project/runs/{run_id}"


def contains_key(value, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(contains_key(nested, key) for nested in value.values())
    if isinstance(value, list):
        return any(contains_key(item, key) for item in value)
    return False


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
        "goal": {"goal_id": "Level1-1"},
        "spec_id": "candidate",
        "description": (
            "Candidate seed {seed} reproduces the expected completion signal and was created "
            "to rank by completion rate, then reward."
        ),
        "seeds": [23, 24],
        "group_id": "b-test",
        "run_name_template": "{group_id}_{spec_id}_s{seed}_{timestamp}",
        "tags": ["b55", "confirm"],
        "selection_metrics": ["train/completion_episode_rate", "train/reward/mean"],
        "train_config": {
            "game": "SuperMarioBros-Nes-v0",
            "state": "Level1-1",
            "timesteps": 1024,
            "wandb": True,
            "wandb_mode": "online",
            "wandb_artifact_storage_uri": "s3://bucket/checkpoints",
        },
    }


def explicit_train_config(**overrides) -> dict:
    config = {
        "game": "SuperMarioBros-Nes-v0",
        "state": "Level1-1",
        "timesteps": 1024,
        "wandb": True,
        "wandb_mode": "online",
        "wandb_artifact_storage_uri": "s3://bucket/checkpoints",
    }
    config.update(overrides)
    return config


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
    def test_wandb_run_leaders_rank_by_worst_seed_then_mean(self) -> None:
        runs = [
            FakeWandbRun(
                run_id="a1",
                name="a-s1",
                config={"goal_slug": "Level1-1", "spec_slug": "a", "seed": 1},
                summary={"info/level_complete/rate/min/last": 1.0},
            ),
            FakeWandbRun(
                run_id="a2",
                name="a-s2",
                config={"goal_slug": "Level1-1", "spec_slug": "a", "seed": 2},
                summary={"info/level_complete/rate/min/last": 0.8},
            ),
            FakeWandbRun(
                run_id="b1",
                name="b-s1",
                config={"goal_slug": "Level1-1", "spec_slug": "b", "seed": 1},
                summary={"info/level_complete/rate/min/last": 0.9},
            ),
            FakeWandbRun(
                run_id="b2",
                name="b-s2",
                config={"goal_slug": "Level1-1", "spec_slug": "b", "seed": 2},
                summary={"info/level_complete/rate/min/last": 0.9},
            ),
        ]

        scores = [
            score
            for score in (
                wandb_leaders.run_score(
                    run,
                    objective_keys=wandb_leaders.RUN_OBJECTIVE_KEYS,
                )
                for run in runs
            )
            if score is not None
        ]
        leaders = wandb_leaders.rank_run_leaders(scores, min_seeds=2)

        self.assertEqual(leaders[0].spec_slug, "b")
        self.assertEqual(leaders[0].worst_seed, 0.9)
        self.assertEqual(leaders[1].spec_slug, "a")
        self.assertEqual(leaders[1].worst_seed, 0.8)

    def test_wandb_checkpoint_leaders_include_source_run(self) -> None:
        run = FakeWandbRun(
            run_id="run-1",
            name="candidate",
            config={"goal_slug": "Level1-4", "spec_slug": "b257"},
            summary={
                "leader/checkpoint/completion_rate": 1.0,
                "leader/checkpoint/completion_rate_mean": 0.95,
                "leader/checkpoint/max_x_max": 4610,
                "leader/checkpoint/reward_mean": 4200.0,
                "leader/checkpoint/step": 4500000,
                "leader/checkpoint/steps_to_completion_goal": 4500000,
                "leader/checkpoint/artifact_ref": "entity/project/candidate:step-4500000",
                "leader/checkpoint/eval_source": "post_train_inline",
            },
        )

        leader = wandb_leaders.checkpoint_leader(run)

        self.assertIsNotNone(leader)
        assert leader is not None
        self.assertEqual(leader.run_id, "run-1")
        self.assertEqual(leader.run_name, "candidate")
        self.assertEqual(leader.completion_rate, 1.0)
        self.assertEqual(leader.completion_rate_mean, 0.95)
        self.assertEqual(leader.checkpoint_step, 4500000)
        self.assertEqual(leader.steps_to_completion_goal, 4500000)
        self.assertEqual(leader.artifact_ref, "entity/project/candidate:step-4500000")
        self.assertEqual(leader.eval_source, "post_train_inline")

    def test_wandb_checkpoint_leaders_infer_solved_step_for_legacy_rows(self) -> None:
        run = FakeWandbRun(
            run_id="run-1",
            name="candidate",
            config={"goal_slug": "Level1-4", "spec_slug": "b257"},
            summary={
                "leader/checkpoint/completion_rate": 1.0,
                "leader/checkpoint/completion_rate_mean": 1.0,
                "leader/checkpoint/max_x_max": 4610,
                "leader/checkpoint/reward_mean": 4200.0,
                "leader/checkpoint/step": 3500000,
                "leader/checkpoint/artifact_ref": "entity/project/candidate:step-3500000",
            },
        )

        leader = wandb_leaders.checkpoint_leader(run)

        self.assertIsNotNone(leader)
        assert leader is not None
        self.assertEqual(leader.steps_to_completion_goal, 3500000)

    def test_wandb_checkpoint_leaders_rank_solved_runs_by_timesteps_before_reward(self) -> None:
        slower_higher_reward = FakeWandbRun(
            run_id="slow",
            name="slow",
            config={"goal_slug": "Level1-1", "spec_slug": "slow"},
            summary={
                "leader/checkpoint/completion_rate": 1.0,
                "leader/checkpoint/completion_rate_mean": 1.0,
                "leader/checkpoint/max_x_max": 4610,
                "leader/checkpoint/reward_mean": 4200.0,
                "leader/checkpoint/step": 5000000,
                "leader/checkpoint/steps_to_completion_goal": 5000000,
                "leader/checkpoint/artifact_ref": "entity/project/slow:step-5000000",
            },
        )
        faster_lower_reward = FakeWandbRun(
            run_id="fast",
            name="fast",
            config={"goal_slug": "Level1-1", "spec_slug": "fast"},
            summary={
                "leader/checkpoint/completion_rate": 1.0,
                "leader/checkpoint/completion_rate_mean": 1.0,
                "leader/checkpoint/max_x_max": 4610,
                "leader/checkpoint/reward_mean": 100.0,
                "leader/checkpoint/step": 3500000,
                "leader/checkpoint/steps_to_completion_goal": 3500000,
                "leader/checkpoint/artifact_ref": "entity/project/fast:step-3500000",
            },
        )
        leaders = [
            leader
            for leader in (
                wandb_leaders.checkpoint_leader(run)
                for run in (slower_higher_reward, faster_lower_reward)
            )
            if leader is not None
        ]

        ranked = wandb_leaders.rank_checkpoint_leaders(leaders)

        self.assertEqual(ranked[0].run_id, "fast")

    def test_wandb_checkpoint_leaders_accept_current_eval_artifact_key(self) -> None:
        run = FakeWandbRun(
            run_id="run-1",
            name="candidate",
            config={"goal_slug": "Level1-4", "spec_slug": "b257"},
            summary={
                "eval/done/level_change/rate": 0.8,
                "eval/progress/x/max": 4610,
                "eval/reward/mean": 4200.0,
                "eval/checkpoint/artifact": "entity/project/candidate:step-4500000",
            },
        )

        leader = wandb_leaders.checkpoint_leader(run)

        self.assertIsNotNone(leader)
        assert leader is not None
        self.assertEqual(leader.artifact_ref, "entity/project/candidate:step-4500000")

    def test_wandb_checkpoint_filter_requires_evaluated_checkpoint_summary(self) -> None:
        expected_current = {
            "$and": [
                {"summary_metrics.leader/checkpoint/completion_rate": {"$exists": True}},
                {"summary_metrics.leader/checkpoint/completion_rate_mean": {"$exists": True}},
                {"summary_metrics.leader/checkpoint/max_x_max": {"$exists": True}},
                {"summary_metrics.leader/checkpoint/reward_mean": {"$exists": True}},
                {"summary_metrics.leader/checkpoint/artifact_ref": {"$exists": True}},
            ]
        }
        expected_legacy = {
            "$and": [
                {"summary_metrics.eval/done/level_change/rate": {"$exists": True}},
                {"summary_metrics.eval/progress/x/max": {"$exists": True}},
                {"summary_metrics.eval/reward/mean": {"$exists": True}},
                {
                    "$or": [
                        {"summary_metrics.eval/checkpoint/artifact": {"$exists": True}},
                        {"summary_metrics.eval/checkpoint_artifact": {"$exists": True}},
                    ]
                },
            ]
        }

        self.assertEqual(
            wandb_leaders.checkpoint_summary_filter(),
            {"$or": [expected_current, expected_legacy]},
        )

    def test_wandb_goal_filter_accepts_config_or_tag_partition(self) -> None:
        self.assertEqual(
            wandb_leaders.goal_run_filter("Level1-1"),
            {
                "$or": [
                    {"config.goal_slug": "Level1-1"},
                    {"tags": "goal_id:Level1-1"},
                    {"tags": "goal:Level1-1"},
                ]
            },
        )
        self.assertEqual(wandb_leaders.goal_run_filter(None), {})

    def test_wandb_run_objective_filter_requires_any_objective_summary(self) -> None:
        self.assertEqual(
            wandb_leaders.run_objective_filter(("metric/a", "metric/b")),
            {
                "$or": [
                    {"summary_metrics.metric/a": {"$exists": True}},
                    {"summary_metrics.metric/b": {"$exists": True}},
                ]
            },
        )

    def test_wandb_run_query_uses_current_metric_by_default(self) -> None:
        args = SimpleNamespace(objective_key=[], include_legacy_objectives=False)

        self.assertEqual(
            wandb_leaders.run_query_objective_keys(args),
            ("train/info/level_complete/rate/min/last",),
        )

    def test_wandb_run_query_can_include_legacy_objective_aliases(self) -> None:
        args = SimpleNamespace(objective_key=[], include_legacy_objectives=True)

        self.assertEqual(
            wandb_leaders.run_query_objective_keys(args), wandb_leaders.RUN_OBJECTIVE_KEYS
        )

    def test_wandb_run_query_explicit_objective_keys_override_defaults(self) -> None:
        args = SimpleNamespace(
            objective_key=["metric/a", "metric/b"],
            include_legacy_objectives=True,
        )

        self.assertEqual(wandb_leaders.run_query_objective_keys(args), ("metric/a", "metric/b"))

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
        obsolete_eval_table = "eval" + "_jobs"
        self.assertNotIn("CREATE TABLE IF NOT EXISTS research_goals", job_queue.SCHEMA_SQL)
        self.assertNotIn("CREATE TABLE IF NOT EXISTS experiment_specs", job_queue.SCHEMA_SQL)
        self.assertNotIn("CREATE TABLE IF NOT EXISTS campaign_decisions", job_queue.SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS train_jobs", job_queue.SCHEMA_SQL)
        self.assertIn("goal_slug TEXT NOT NULL", job_queue.SCHEMA_SQL)
        self.assertIn("spec_payload_json JSONB", job_queue.SCHEMA_SQL)
        self.assertIn("spec_sha256 TEXT", job_queue.SCHEMA_SQL)
        self.assertNotIn(f"CREATE TABLE IF NOT EXISTS {obsolete_eval_table}", job_queue.SCHEMA_SQL)
        self.assertNotIn("CREATE TABLE IF NOT EXISTS train_results", job_queue.SCHEMA_SQL)
        self.assertNotIn("CREATE TABLE IF NOT EXISTS eval_results", job_queue.SCHEMA_SQL)

    def test_reset_schema_drops_only_current_queue_tables(self) -> None:
        conn = FakeConnection()
        with tempfile.TemporaryDirectory() as tmp:
            job_queue.reset_schema(conn, export_dir=Path(tmp))

        drop_sql = next(sql for sql in conn.cursor_obj.executed_sqls if "DROP TABLE" in sql)
        self.assertIn("train_jobs", drop_sql)
        self.assertNotIn("eval" + "_jobs", drop_sql)
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
  wandb_artifact_storage_uri: s3://bucket/checkpoints
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
goal:
  goal_id: Level1-1
spec_id: candidate
description: Candidate seed {seed} reproduces the expected completion signal and was created to rank by completion rate, then reward.
seeds: [23, 24]
run_target: rtx4090
state: Level1-1
group_id: b-test
tags: [mario, confirm]
selection_metrics: [train/completion_episode_rate, train/reward/mean]
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
        self.assertEqual(
            loaded["environment"]["env_id"], "stable-retro-turbo:SuperMarioBros-Nes-v0"
        )
        self.assertTrue(loaded["environment_hash"].startswith("sha256:"))
        self.assertEqual(len(loaded["_composition"]["source_files"]), 2)

    def test_load_spec_document_renders_template_vars_before_queue_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "SuperMarioBros-Nes-v0" / "Level1-2"
            spec_dir = goal_dir / "specs"
            spec_dir.mkdir(parents=True)
            (goal_dir.parent / "_base.yaml").write_text(
                """
objective:
  rank:
  - max(eval/reward/mean)
template_vars:
  state: "{goal_id}"
train:
  early_stop:
  - metric: train/info/level_complete/rate/min/last
    operator: '>'
    threshold: 0.99
  environment:
    env_provider: stable-retro-turbo
    env_config:
      game: SuperMarioBros-Nes-v0
      state: "{state}"
      action_set: simple
      frame_skip: 4
      max_pool_frames: false
      sticky_action_prob: 0.0
      observation_size: 84
      hud_crop_top: 32
      obs_resize_algorithm: area
      info_events:
        life_loss: [lives, decrease]
        level_change: [[levelHi, levelLo], change]
      done_on_events: [life_loss, level_change]
eval:
  environment:
    env_provider: stable-retro-turbo
    env_config:
      game: SuperMarioBros-Nes-v0
      state: "{state}"
      action_set: simple
      frame_skip: 4
      max_pool_frames: false
      sticky_action_prob: 0.0
      observation_size: 84
      hud_crop_top: 32
      obs_resize_algorithm: area
      info_events:
        level_change: [[levelHi, levelLo], change]
      done_on_events: [level_change]
release:
  huggingface:
    repo: "{env_id}_{goal_id}"
    card_template: stable-retro-sb3
    checkpoint_filename: model.zip
    preview_filename: replay.mp4
    include_youtube_preview: true
""",
                encoding="utf-8",
            )
            (goal_dir / "_goal.yaml").write_text(
                """
defaults:
- ../_base@_global_
- _self_
goal_id: "{goal_id}"
title: "{goal_id} completion"
""",
                encoding="utf-8",
            )
            spec = spec_dir / "candidate.yaml"
            spec.write_text(
                """
defaults:
- ../_goal@goal
- _self_
spec_id: candidate
template_vars:
  batch_id: b272
  recipe: b55
description: "{goal_id} {recipe} seed {seed} transfer run created to validate recipe transfer."
group_id: "{batch_id}-{level_short}-{recipe}"
run_name_template: "{group_id}_{spec_id}_s{seed}_{timestamp}"
tags: ["goal_id:{goal_id}", "spec_id:{spec_id}", "env_id:{env_id}"]
selection_metrics: [eval/reward/mean]
train:
  policy:
    timesteps: 1024
logging:
  wandb: true
  wandb_mode: online
  wandb_artifact_storage_uri: s3://bucket/checkpoints
""",
                encoding="utf-8",
            )

            loaded = job_queue.load_spec_document(spec)

        self.assertFalse(contains_key(loaded, "template_vars"))
        self.assertEqual(loaded["goal"]["goal_id"], "Level1-2")
        self.assertEqual(loaded["group_id"], "b272-l12-b55")
        self.assertEqual(
            loaded["tags"],
            [
                "goal_id:Level1-2",
                "spec_id:candidate",
                "env_id:SuperMarioBros-Nes-v0",
            ],
        )
        self.assertEqual(
            loaded["description"],
            "Level1-2 b55 seed {seed} transfer run created to validate recipe transfer.",
        )
        self.assertEqual(
            loaded["goal"]["release"]["huggingface"]["checkpoint_filename"],
            "model.zip",
        )

    def test_load_spec_document_materializes_first_class_environment_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.yaml"
            path.write_text(
                """
schema_version: 1
kind: train_experiment
goal:
  goal_id: Level1-1
spec_id: candidate
description: Candidate seed {seed} reproduces the expected completion signal and was created to rank by completion rate, then reward.
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
group_id: b-test
tags: [mario, env-hash]
selection_metrics: [train/completion_episode_rate]
train:
  timesteps: 1024
logging:
  wandb: true
  wandb_mode: online
  wandb_artifact_storage_uri: s3://bucket/checkpoints
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
        self.assertEqual(
            loaded["environment"]["env_id"], "stable-retro-turbo:SuperMarioBros-Nes-v0"
        )
        self.assertEqual(loaded["environment"]["state"], "Level1-1")
        self.assertNotIn("hud_crop_top", loaded["environment"]["preprocessing"])
        self.assertEqual(loaded["environment"]["preprocessing"]["obs_crop"], [32, 0, 0, 0])
        self.assertNotIn("observation_size", loaded["environment"]["preprocessing"])
        self.assertEqual(loaded["environment"]["preprocessing"]["obs_resize"], [84, 84])
        self.assertTrue(loaded["environment_hash"].startswith("sha256:"))

    def test_load_spec_document_materializes_env_config_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.yaml"
            path.write_text(
                """
schema_version: 1
kind: train_experiment
goal:
  goal_id: Level1-1
spec_id: candidate
description: Candidate seed {seed} reproduces the expected completion signal and was created to rank by completion rate, then reward.
seeds: [23]
run_target: rtx4090
group_id: b-test
tags: [mario, env-config]
selection_metrics: [train/completion_episode_rate]
train:
  environment:
    env_config:
      env_provider: stable-retro-turbo
      game: SuperMarioBros-Nes-v0
      n_envs: 16
      env_threads: 4
      state: Level1-1
      action_set: simple
      frame_skip: 4
      max_pool_frames: false
      observation_size: 84
      hud_crop_top: 32
      max_episode_steps: 4500
      info_events:
        life_loss: [lives, decrease]
      done_on_events: [life_loss]
  timesteps: 1024
logging:
  wandb: true
  wandb_mode: online
  wandb_artifact_storage_uri: s3://bucket/checkpoints
""",
                encoding="utf-8",
            )

            loaded = job_queue.load_spec_document(path)

        self.assertEqual(loaded["train_config"]["game"], "SuperMarioBros-Nes-v0")
        self.assertEqual(loaded["train_config"]["env_provider"], "stable-retro-turbo")
        self.assertEqual(loaded["train_config"]["state"], "Level1-1")
        self.assertNotIn("environment", loaded["train_config"])
        self.assertEqual(
            loaded["train_config"]["info_events_json"],
            {"life_loss": ["lives", "decrease"]},
        )
        self.assertEqual(loaded["train_config"]["done_on_events"], ["life_loss"])
        self.assertEqual(
            loaded["environment"]["env_id"], "stable-retro-turbo:SuperMarioBros-Nes-v0"
        )
        self.assertNotIn("env_config", loaded["environment"])
        self.assertEqual(loaded["environment"]["state"], "Level1-1")
        self.assertEqual(loaded["environment"]["preprocessing"]["obs_crop"], [32, 0, 0, 0])
        self.assertTrue(loaded["environment_hash"].startswith("sha256:"))

    def test_load_spec_document_inherits_goal_owned_contract_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "Level1-1"
            specs_dir = goal_dir / "specs"
            specs_dir.mkdir(parents=True)
            goal_dir.joinpath("_goal.yaml").write_text(
                """
goal_id: Level1-1
title: Level 1-1
objective:
  rank:
  - metric: train/info/level_complete/rate/min/last
    aggregation: max
    direction: maximize
train:
  policy:
    timesteps: 1024
  checkpoint_freq: 500000
  early_stop:
  - metric: train/info/level_complete/rate/min/last
    operator: '>'
    threshold: 0.99
  environment:
    env_config:
      env_provider: stable-retro-turbo
      game: SuperMarioBros-Nes-v0
      n_envs: 16
      env_threads: 4
      state: Level1-1
      action_set: simple
      frame_skip: 4
      max_pool_frames: false
      observation_size: 84
      hud_crop_top: 32
      max_episode_steps: 4500
      info_events:
        life_loss: [lives, decrease]
      done_on_events: [life_loss]
logging:
  wandb: true
  wandb_mode: online
  wandb_artifact_storage_uri: s3://bucket/checkpoints
""",
                encoding="utf-8",
            )
            spec = specs_dir / "candidate.yaml"
            spec.write_text(
                """
schema_version: 1
defaults:
- ../_goal@goal
- _self_
spec_id: candidate
description: Candidate seed {seed} inherits the goal contract and was created to verify queue materialization of env identity and training policy.
seeds: [23]
group_id: b-test
tags: [mario, env-config]
state: WrongState
train:
  policy:
    timesteps: 1024
logging:
  wandb: true
  wandb_mode: online
  wandb_artifact_storage_uri: s3://bucket/checkpoints
train_config:
  game: WrongGame
  state: WrongState
  action_set: complex
  early_stop_metric: wrong/metric
  early_stop_threshold: 0.1
""",
                encoding="utf-8",
            )

            source_spec = yaml.safe_load(spec.read_text(encoding="utf-8"))
            loaded = job_queue.load_spec_document(spec)

        self.assertEqual(loaded["goal"]["goal_id"], "Level1-1")
        self.assertNotIn("goal", source_spec)
        self.assertIn("train", source_spec)
        self.assertNotIn("n_envs", loaded["train"]["environment"])
        self.assertNotIn("env_threads", loaded["train"]["environment"])
        self.assertEqual(loaded["train"]["environment"]["env_config"]["n_envs"], 16)
        self.assertEqual(loaded["train"]["environment"]["env_config"]["env_threads"], 4)
        self.assertEqual(loaded["train"]["environment"]["env_config"]["state"], "Level1-1")
        self.assertEqual(loaded["train_config"]["game"], "SuperMarioBros-Nes-v0")
        self.assertEqual(loaded["train_config"]["state"], "Level1-1")
        self.assertEqual(loaded["train_config"]["action_set"], "simple")
        self.assertEqual(loaded["train_config"]["n_envs"], 16)
        self.assertEqual(loaded["train_config"]["env_threads"], 4)
        self.assertEqual(
            loaded["train_config"]["early_stop"],
            [
                {
                    "metric": TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                    "operator": ">",
                    "threshold": 0.99,
                }
            ],
        )
        self.assertEqual(loaded["train_config"]["timesteps"], 1024)
        self.assertEqual(loaded["train_config"]["checkpoint_freq"], 500000)
        self.assertTrue(loaded["train_config"]["wandb"])
        self.assertEqual(loaded["train_config"]["wandb_mode"], "online")
        self.assertEqual(
            loaded["train_config"]["wandb_artifact_storage_uri"], "s3://bucket/checkpoints"
        )
        self.assertIn("selection_policy", loaded)
        source_paths = [source["path"] for source in loaded["_composition"]["source_files"]]
        self.assertTrue(any(path.endswith("_goal.yaml") for path in source_paths))

    def test_explicit_train_environment_overrides_goal_done_on_rules(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            goal_dir = root / "experiments" / "goals" / "Level1-1"
            specs_dir = goal_dir / "specs"
            specs_dir.mkdir(parents=True)
            goal_dir.joinpath("_goal.yaml").write_text(
                """
goal_id: Level1-1
title: Level 1-1
objective:
  rank:
  - max(eval/reward/mean)
train:
  policy:
    timesteps: 1024
  environment:
    env_config:
      env_provider: stable-retro-turbo
      game: SuperMarioBros-Nes-v0
      state: Level1-1
      n_envs: 16
      info_events:
        life_loss: [lives, decrease]
        level_change: [[levelHi, levelLo], change]
      done_on_events: [life_loss, level_change]
logging:
  wandb: true
  wandb_mode: online
  wandb_artifact_storage_uri: s3://bucket/checkpoints
""",
                encoding="utf-8",
            )
            spec = specs_dir / "candidate.yaml"
            spec.write_text(
                """
schema_version: 1
defaults:
- ../_goal@goal
- _self_
spec_id: candidate
description: Candidate seed {seed} disables native terminal boundaries and was created to verify spec overrides on the goal contract.
seeds: [23]
group_id: b-test
tags: [mario, no-terminal]
train:
  policy:
    timesteps: 1024
  environment:
    env_config:
      done_on: []
logging:
  wandb: true
  wandb_mode: online
  wandb_artifact_storage_uri: s3://bucket/checkpoints
""",
                encoding="utf-8",
            )

            loaded = job_queue.load_spec_document(spec)

        self.assertEqual(loaded["train"]["environment"]["env_config"]["done_on"], [])
        self.assertEqual(loaded["train_config"]["done_on_events"], [])

    def test_load_spec_document_rejects_missing_mandatory_schema_field(self) -> None:
        spec = valid_train_spec()
        del spec["description"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps(spec), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "description"):
                job_queue.load_spec_document(path)

    def test_load_spec_document_rejects_missing_explicit_train_timestep(self) -> None:
        spec = valid_train_spec()
        del spec["train_config"]["timesteps"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps(spec), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "train_config.timesteps"):
                job_queue.load_spec_document(path)

    def test_load_spec_document_rejects_removed_spec_fields(self) -> None:
        for field in (
            "hypothesis",
            "parent_spec_slug",
            "parent_spec_id",
            "run_description_template",
            "slug",
            "wandb_tags",
            "wandb_group",
        ):
            with self.subTest(field=field):
                spec = valid_train_spec()
                spec[field] = "removed"
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / "candidate.json"
                    path.write_text(json.dumps(spec), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, "removed train spec field"):
                        job_queue.load_spec_document(path)

    def test_enqueue_train_jobs_from_spec_document_derives_short_run_name(self) -> None:
        calls = []

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {"run_name": kwargs["run_name"]}

        spec = valid_train_spec()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            document = job_queue.load_spec_document(path)

        old_enqueue = job_queue.enqueue_train_job
        old_utc = job_queue._utc_stamp
        job_queue.enqueue_train_job = fake_enqueue
        job_queue._utc_stamp = lambda: "20260626T120000Z"
        try:
            rows = job_queue.enqueue_train_jobs_from_spec_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
            )
        finally:
            job_queue.enqueue_train_job = old_enqueue
            job_queue._utc_stamp = old_utc

        self.assertEqual(
            [row["run_name"] for row in rows],
            ["b-test_candidate_s23_20260626T120000Z", "b-test_candidate_s24_20260626T120000Z"],
        )
        self.assertEqual(
            [call["run_name"] for call in calls],
            ["b-test_candidate_s23_20260626T120000Z", "b-test_candidate_s24_20260626T120000Z"],
        )

    def test_enqueue_train_jobs_from_spec_document_uses_run_name_template(self) -> None:
        calls = []

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {"run_name": kwargs["run_name"]}

        document = valid_train_spec()
        document["group_id"] = "b82-l11-b55-post21-revalidate"

        old_enqueue = job_queue.enqueue_train_job
        old_utc = job_queue._utc_stamp
        job_queue.enqueue_train_job = fake_enqueue
        job_queue._utc_stamp = lambda: "20260702T150934Z"
        try:
            rows = job_queue.enqueue_train_jobs_from_spec_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                seeds=[6],
            )
        finally:
            job_queue.enqueue_train_job = old_enqueue
            job_queue._utc_stamp = old_utc

        self.assertEqual(
            [row["run_name"] for row in rows],
            ["b82-l11-b55-post21-revalidate_candidate_s6_20260702T150934Z"],
        )
        self.assertEqual(
            [call["run_name"] for call in calls],
            ["b82-l11-b55-post21-revalidate_candidate_s6_20260702T150934Z"],
        )

    def test_checked_in_goal_yaml_specs_match_train_spec_schema(self) -> None:
        spec_paths = sorted(Path("experiments/goals").rglob("specs/*.y*ml"))
        self.assertGreater(len(spec_paths), 0)
        for path in spec_paths:
            with self.subTest(path=str(path)):
                job_queue.load_spec_document(path)

    def test_active_level1_1_specs_configure_goal_metric_early_stop(self) -> None:
        spec_paths = sorted(
            Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/specs").glob("*.yaml")
        )
        self.assertGreater(len(spec_paths), 0)
        for path in spec_paths:
            with self.subTest(path=str(path)):
                spec = job_queue.load_spec_document(path)
                train_config = spec["train_config"]
                self.assertEqual(
                    train_config["early_stop"],
                    [
                        {
                            "metric": TRAIN_INFO_LEVEL_COMPLETE_RATE_MIN_LAST,
                            "operator": ">",
                            "threshold": 0.99,
                        }
                    ],
                )

    def test_transfer_specs_inherit_level1_1_policy_recipe(self) -> None:
        level1_1 = job_queue.load_spec_document(
            Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/specs/base.yaml")
        )
        self.assertEqual(
            level1_1["group_id"],
            "SuperMarioBros-Nes-v0_Level1-1",
        )
        for level in ("Level1-2", "Level1-3", "Level2-2"):
            with self.subTest(level=level):
                transfer = job_queue.load_spec_document(
                    Path(f"experiments/goals/SuperMarioBros-Nes-v0/{level}/specs/base.yaml")
                )

                self.assertEqual(transfer["train"]["policy"], level1_1["train"]["policy"])
                self.assertEqual(transfer["goal"]["goal_id"], level)
                self.assertEqual(transfer["train"]["environment"]["env_config"]["state"], level)
                self.assertEqual(
                    transfer["group_id"],
                    f"SuperMarioBros-Nes-v0_{level}",
                )
                self.assertFalse(contains_key(transfer, "template_vars"))
                self.assertEqual(
                    transfer["goal"]["release"]["huggingface"]["repo"],
                    f"SuperMarioBros-Nes-v0_{level}",
                )
                self.assertEqual(
                    transfer["goal"]["release"]["huggingface"]["checkpoint_filename"],
                    "model.zip",
                )
                self.assertEqual(
                    transfer["tags"],
                    [
                        f"goal_id:{level}",
                        "spec_id:base",
                        "env_id:SuperMarioBros-Nes-v0",
                    ],
                )
                self.assertIn(level, transfer["description"])

    def test_launch_result_metadata_strips_metrics_json(self) -> None:
        result = job_queue.launch_result_metadata(
            {
                "job_kind": "train",
                "train": {
                    "result": {
                        "run_name": "candidate",
                        "metrics_json": {"train/done/all": 20},
                    }
                },
            }
        )

        self.assertEqual(result["train"]["result"]["run_name"], "candidate")
        self.assertNotIn("metrics_json", result["train"]["result"])

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

    def test_mark_stale_train_jobs_failed_updates_job_only(self) -> None:
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
        self.assertNotIn("INSERT INTO train_results", conn.cursor_obj.executed_sql)
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
            train_config=explicit_train_config(),
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
            train_config=explicit_train_config(),
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
                train_config=explicit_train_config(
                    done_on_info_json={
                        "level_change": [["levelHi", "levelLo"], "change"],
                    },
                ),
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
                train_config=explicit_train_config(
                    info_events_json={"life_loss": ["lives", "decrease"]},
                    done_on_events="life_loss,level_change",
                ),
            )

    def test_enqueue_train_job_allows_stable_retro_turbo_provider_owned_done_events(self) -> None:
        row = job_queue.enqueue_train_job(
            FakeConnection(row={"id": 7}),
            goal_slug="goal",
            spec_slug="spec",
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            train_config=explicit_train_config(
                env_provider="stable-retro-turbo",
                done_on_events="life_loss,level_change",
            ),
        )

        self.assertEqual(row["id"], 7)

    def test_enqueue_train_job_rejects_eval_reserved_seed_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved for eval"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                spec_slug="spec",
                profile_id=None,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                run_target="rtx4090",
                train_config=explicit_train_config(seed=DEFAULT_EVAL_SEED),
            )

        with self.assertRaisesRegex(ValueError, "training env slot"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                spec_slug="spec",
                profile_id=None,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                run_target="rtx4090",
                train_config=explicit_train_config(seed=9999, n_envs=2),
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
                train_config=explicit_train_config(),
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

    def test_eval_selection_score_prefers_min_completion_then_mean_then_solved_step(self) -> None:
        weak_bottleneck = {
            "completion_rate": 1.0,
            "eval/done/level_change/from_rate/min": 0.25,
            "eval/done/level_change/from_rate/mean": 1.0,
            "max_x_max": 4000,
            "reward_mean": 900.0,
        }
        balanced = {
            "completion_rate": 0.8,
            "eval/done/level_change/from_rate/min": 0.75,
            "eval/done/level_change/from_rate/mean": 0.8,
            "max_x_max": 3200,
            "reward_mean": 600.0,
        }
        same_min_better_mean = {
            "completion_rate": 0.9,
            "eval/done/level_change/from_rate/min": 0.75,
            "eval/done/level_change/from_rate/mean": 0.9,
            "max_x_max": 3000,
            "reward_mean": 10.0,
        }

        self.assertGreater(
            job_queue.eval_selection_score(balanced),
            job_queue.eval_selection_score(weak_bottleneck),
        )
        self.assertGreater(
            job_queue.eval_selection_score(same_min_better_mean),
            job_queue.eval_selection_score(balanced),
        )
        slower_higher_reward = {
            "completion_rate": 1.0,
            "eval/done/level_change/from_rate/min": 1.0,
            "eval/done/level_change/from_rate/mean": 1.0,
            "checkpoint_step": 5000000,
            "reward_mean": 900.0,
        }
        faster_lower_reward = {
            "completion_rate": 1.0,
            "eval/done/level_change/from_rate/min": 1.0,
            "eval/done/level_change/from_rate/mean": 1.0,
            "checkpoint_step": 3500000,
            "reward_mean": 10.0,
        }
        self.assertGreater(
            job_queue.eval_selection_score(faster_lower_reward),
            job_queue.eval_selection_score(slower_higher_reward),
        )

    def test_enqueue_train_jobs_from_spec_derives_group_run_names(self) -> None:
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

        self.assertEqual(
            [row["run_name"] for row in rows],
            ["b-test_candidate_s23_20260626T120000Z", "b-test_candidate_s24_20260626T120000Z"],
        )
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
        self.assertEqual(calls[0]["wandb_tags"], ["b55", "confirm"])
        self.assertEqual(calls[0]["goal_slug"], "Level1-1")
        self.assertEqual(calls[0]["spec_slug"], "candidate")
        self.assertIsNone(calls[0]["profile_id"])
        self.assertIsNone(calls[0]["run_target"])
        self.assertEqual(calls[0]["spec_path"], "experiments/goals/mario/specs/candidate.yaml")
        self.assertEqual(calls[0]["spec_sha256"], "abc123")
        self.assertEqual(calls[0]["repo_git_commit"], "deadbeef")
        self.assertTrue(calls[0]["repo_dirty"])
        self.assertEqual(calls[0]["spec_payload"]["operator_note"], "non-schema metadata persists")

    def test_enqueue_train_jobs_from_spec_document_rejects_wrong_schema_version(self) -> None:
        document = copy.deepcopy(valid_train_spec())
        document["schema_version"] = 2

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
        controller.observe(
            ResourceSample(cpu_percent=50, memory_percent=50, gpu_percent=50, vram_percent=50)
        )
        controller.observe(
            ResourceSample(cpu_percent=55, memory_percent=55, gpu_percent=55, vram_percent=55)
        )

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
        controller.observe(
            ResourceSample(cpu_percent=50, memory_percent=50, gpu_percent=50, vram_percent=50)
        )

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
        controller.observe(
            ResourceSample(cpu_percent=91, memory_percent=50, gpu_percent=50, vram_percent=50)
        )
        controller.observe(
            ResourceSample(cpu_percent=92, memory_percent=50, gpu_percent=50, vram_percent=50)
        )

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
        controller.observe(
            ResourceSample(cpu_percent=10, memory_percent=10, gpu_percent=10, vram_percent=10)
        )

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
    def test_checkpoint_bucket_default_resolves_before_command_build(self) -> None:
        old_value = os.environ.get("CHECKPOINT_BUCKET_URI")
        os.environ["CHECKPOINT_BUCKET_URI"] = '"s3://bucket/checkpoints"'
        try:
            job = {
                "id": 13,
                "train_config": {
                    "game": "SuperMarioBros-Nes-v0",
                    "timesteps": 1024,
                    "state": "Level1-2",
                    "wandb": True,
                    "wandb_mode": "online",
                    "wandb_artifact_storage_uri": "CHECKPOINT_BUCKET_URI",
                },
                "run_name": "placeholder_candidate",
            }

            config = normalize_train_config(job)
            with tempfile.TemporaryDirectory() as tmp:
                config_path = write_train_config_file(job, Path(tmp) / "train_config.json")
                command = train_command_for_job(config_path)
                written_config = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(config["wandb_artifact_storage_uri"], "s3://bucket/checkpoints")
            self.assertTrue(config["wandb"])
            self.assertEqual(config["wandb_mode"], "online")
            self.assertEqual(
                written_config["wandb_artifact_storage_uri"],
                "s3://bucket/checkpoints",
            )
            self.assertTrue(written_config["wandb"])
            self.assertEqual(written_config["wandb_mode"], "online")
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
                    "wandb": True,
                    "wandb_mode": "online",
                    "wandb_artifact_storage_uri": "s3://bucket/checkpoints",
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
                "state": "Level1-1",
                "wandb": True,
                "wandb_mode": "online",
                "wandb_artifact_storage_uri": "s3://bucket/checkpoints",
                "wandb_tags": ["screen"],
            },
            "goal_slug": "Level1-1",
            "spec_slug": "base",
            "spec_path": "experiments/goals/SuperMarioBros-Nes-v0/Level1-1/specs/base.yaml",
            "run_name": "lowkl_seed23",
            "run_description": "Codex-authored smoke job.",
            "wandb_group": "level1-1-lowkl-lrdecay",
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
            "screen,goal_id:Level1-1,spec_id:base,level_id:Level1-1",
        )
        self.assertEqual(written_config["goal_slug"], "Level1-1")
        self.assertEqual(written_config["spec_slug"], "base")
        self.assertEqual(
            written_config["spec_path"],
            "experiments/goals/SuperMarioBros-Nes-v0/Level1-1/specs/base.yaml",
        )
        self.assertEqual(written_config["queue_train_job_id"], 12)
        self.assertEqual(written_config["run_name"], "lowkl_seed23")
        self.assertEqual(written_config["state"], "Level1-1")
        self.assertEqual(written_config["wandb_group"], "level1-1-lowkl-lrdecay")
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
                "completion_rate=1.000000\ntimesteps=3881520\n",
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
                "train_config": explicit_train_config(runs_dir=str(root / "runs")),
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
                "train_config": explicit_train_config(runs_dir=str(runs_dir)),
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
            for placeholder in (
                "${CHECKPOINT_BUCKET_URI}",
                "$CHECKPOINT_BUCKET_URI",
                "CHECKPOINT_BUCKET_URI",
            ):
                with self.subTest(placeholder=placeholder):
                    args = SimpleNamespace(wandb_artifact_storage_uri=placeholder)

                    self.assertEqual(wandb_artifact_storage_uri(args), "s3://bucket/from-env")
        finally:
            if old_value is None:
                os.environ.pop("CHECKPOINT_BUCKET_URI", None)
            else:
                os.environ["CHECKPOINT_BUCKET_URI"] = old_value

    def test_configured_storage_uri_strips_env_file_quotes(self) -> None:
        args = SimpleNamespace(wandb_artifact_storage_uri='"s3://bucket/from-arg"')

        self.assertEqual(wandb_artifact_storage_uri(args), "s3://bucket/from-arg")


class JsonSafeTests(unittest.TestCase):
    def test_json_safe_converts_nested_non_json_values(self) -> None:
        class Scalar:
            def item(self):
                return 7

        self.assertEqual(json_safe({"a": (Scalar(), Path("x"))}), {"a": [7, "x"]})


if __name__ == "__main__":
    unittest.main()
