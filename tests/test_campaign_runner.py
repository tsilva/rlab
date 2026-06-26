from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from rlab import campaign
from rlab.artifacts import wandb_artifact_storage_uri
from rlab.eval_job_runner import normalize_eval_config
from rlab.json_utils import json_safe
from rlab.train_runner import (
    collect_result_metadata,
    normalize_train_config,
    parse_log_metrics,
    train_command_for_job,
)


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)


class FakeCursor:
    def __init__(self, row=None) -> None:
        self.row = row
        self.executed_sql = ""
        self.executed_params = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def execute(self, sql, params=None) -> None:
        self.executed_sql = sql
        self.executed_params = params or {}

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, row=None) -> None:
        self.cursor_obj = FakeCursor(row=row)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def cursor(self):
        return self.cursor_obj


class CampaignQueueTests(unittest.TestCase):
    def test_claim_train_job_filters_exact_profile(self) -> None:
        conn = FakeConnection(row={"id": 7, "profile_id": "mario-ppo/post16/rtx4090-screening"})

        row = campaign.claim_train_job(
            conn,
            profile_id="mario-ppo/post16/rtx4090-screening",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            worker_id="worker-a",
            lease_seconds=60,
        )

        self.assertEqual(row["id"], 7)
        self.assertIn("%(profile_id)s IS NULL OR profile_id = %(profile_id)s", conn.cursor_obj.executed_sql)
        self.assertIn("runtime_image_ref = %(runtime_image_ref)s", conn.cursor_obj.executed_sql)
        self.assertIn("run_target IS NULL OR run_target = %(run_target)s", conn.cursor_obj.executed_sql)
        self.assertEqual(
            conn.cursor_obj.executed_params["profile_id"],
            "mario-ppo/post16/rtx4090-screening",
        )
        self.assertEqual(conn.cursor_obj.executed_params["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertEqual(conn.cursor_obj.executed_params["run_target"], "rtx4090")

    def test_claim_train_job_allows_any_profile_when_unspecified(self) -> None:
        conn = FakeConnection(row={"id": 9, "profile_id": "mario-ppo/post21/any-lane"})

        row = campaign.claim_train_job(
            conn,
            profile_id=None,
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            worker_id="worker-any",
            lease_seconds=60,
        )

        self.assertEqual(row["id"], 9)
        self.assertIsNone(conn.cursor_obj.executed_params["profile_id"])
        self.assertEqual(conn.cursor_obj.executed_params["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertEqual(conn.cursor_obj.executed_params["run_target"], "rtx4090")

    def test_claim_train_job_does_not_reclaim_expired_running_leases(self) -> None:
        conn = FakeConnection(row=None)

        row = campaign.claim_train_job(
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
            campaign.assert_no_secrets(
                {"learning_rate": 0.0001, "WANDB_API_KEY": "do-not-store"},
                label="train_config",
            )

    def test_schema_defines_research_campaign_tables(self) -> None:
        self.assertIn("CREATE TABLE IF NOT EXISTS research_goals", campaign.SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS experiment_specs", campaign.SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS train_jobs", campaign.SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS eval_jobs", campaign.SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS eval_results", campaign.SCHEMA_SQL)
        self.assertIn("CREATE TABLE IF NOT EXISTS campaign_decisions", campaign.SCHEMA_SQL)
        self.assertIn("origin_decision_id", campaign.SCHEMA_SQL)
        self.assertIn("runtime_image_ref TEXT", campaign.SCHEMA_SQL)
        self.assertIn("run_target TEXT", campaign.SCHEMA_SQL)
        self.assertIn("train_jobs_runtime_claim_idx", campaign.SCHEMA_SQL)

    def test_enqueue_train_job_persists_runtime_and_target(self) -> None:
        conn = FakeConnection(
            row={
                "id": 9,
                "profile_id": "mario-ppo/post21/rtx4090",
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "run_target": "rtx4090",
            }
        )

        row = campaign.enqueue_train_job(
            conn,
            goal_id=1,
            experiment_spec_id=2,
            profile_id="mario-ppo/post21/rtx4090",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            train_config={"timesteps": 1024},
        )

        self.assertEqual(row["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertIn("runtime_image_ref", conn.cursor_obj.executed_sql)
        self.assertEqual(conn.cursor_obj.executed_params["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertEqual(conn.cursor_obj.executed_params["run_target"], "rtx4090")

    def test_enqueue_train_job_rejects_mutable_runtime_tag(self) -> None:
        conn = FakeConnection(row={"id": 9})

        with self.assertRaisesRegex(ValueError, "immutable docker digest ref"):
            campaign.enqueue_train_job(
                conn,
                goal_id=1,
                experiment_spec_id=2,
                profile_id="mario-ppo/post21/rtx4090",
                runtime_image_ref="docker:ghcr.io/tsilva/rlab/rlab-train:latest",
                train_config={"timesteps": 1024},
            )

    def test_claim_eval_job_filters_exact_profile(self) -> None:
        conn = FakeConnection(row={"id": 8, "profile_id": "mario-ppo/post16/rtx4090-eval"})

        row = campaign.claim_eval_job(
            conn,
            profile_id="mario-ppo/post16/rtx4090-eval",
            worker_id="worker-a",
            lease_seconds=60,
        )

        self.assertEqual(row["id"], 8)
        self.assertIn("profile_id = %(profile_id)s", conn.cursor_obj.executed_sql)
        self.assertEqual(
            conn.cursor_obj.executed_params["profile_id"],
            "mario-ppo/post16/rtx4090-eval",
        )

    def test_claim_eval_job_does_not_reclaim_expired_running_leases(self) -> None:
        conn = FakeConnection(row=None)

        row = campaign.claim_eval_job(
            conn,
            profile_id="mario-ppo/post16/rtx4090-eval",
            worker_id="worker-a",
            lease_seconds=60,
        )

        self.assertIsNone(row)
        self.assertIn("AND status = 'pending'", conn.cursor_obj.executed_sql)
        self.assertNotIn("lease_expires_at < now()", conn.cursor_obj.executed_sql)
        self.assertNotIn("attempts < max_attempts", conn.cursor_obj.executed_sql)

    def test_render_lineage_tree_shows_decision_spec_run_and_eval_causality(self) -> None:
        report = {
            "goal": {
                "id": 1,
                "slug": "mario-level1",
                "title": "Solve Mario Level 1",
                "status": "active",
                "objective_json": {"target": "completion"},
                "constraints_json": {},
            },
            "specs": [
                {
                    "id": 10,
                    "slug": "baseline",
                    "hypothesis": "Known reward shape is a viable baseline.",
                    "expected_signal": "Some completions by 5M steps.",
                    "parent_spec_id": None,
                    "origin_decision_id": 100,
                    "priority": 0,
                    "status": "active",
                },
                {
                    "id": 11,
                    "slug": "lower-kl",
                    "hypothesis": "Lower KL should stabilize late policy updates.",
                    "expected_signal": "Higher completion rate than baseline.",
                    "parent_spec_id": 10,
                    "origin_decision_id": None,
                    "priority": 1,
                    "status": "active",
                },
            ],
            "train_jobs": [
                {
                    "id": 20,
                    "experiment_spec_id": 10,
                    "profile_id": "rtx4090-screening",
                    "status": "succeeded",
                    "priority": 0,
                    "run_name": "baseline_s1",
                    "run_description": "Baseline seed.",
                    "origin_decision_id": 100,
                    "metrics_json": {"completion_rate": "0.20", "total_timesteps": 5000000},
                    "wandb_url": "https://wandb.ai/e/p/runs/abc",
                    "error": None,
                    "result_error": None,
                }
            ],
            "eval_jobs": [
                {
                    "id": 30,
                    "experiment_spec_id": 10,
                    "train_job_id": 20,
                    "profile_id": "level1-eval",
                    "status": "succeeded",
                    "priority": 0,
                    "candidate_label": "baseline-step-5m",
                    "origin_decision_id": None,
                    "model_ref": "entity/project/baseline-checkpoint:v50",
                    "metrics_json": {"completion_rate": 0.2, "episodes": 100},
                    "error": None,
                    "result_error": None,
                }
            ],
            "decisions": [
                {
                    "id": 100,
                    "decision_type": "launch",
                    "summary": "Start from the known baseline.",
                    "rationale": "Previous evals showed enough signal to justify a control.",
                    "affected_spec_ids": [10],
                    "affected_train_job_ids": [20],
                    "affected_eval_job_ids": [],
                    "metadata_json": {},
                },
                {
                    "id": 101,
                    "decision_type": "branch",
                    "summary": "Try lower KL after baseline plateaued.",
                    "rationale": "Baseline eval completed some episodes but late updates were unstable.",
                    "affected_spec_ids": [11],
                    "affected_train_job_ids": [],
                    "affected_eval_job_ids": [30],
                    "metadata_json": {},
                },
            ],
        }

        tree = campaign.render_lineage_tree(report)

        self.assertIn("goal 1: mario-level1 [active]", tree)
        self.assertIn("- spec 10 baseline [active]", tree)
        self.assertIn("cause: decision 100 [launch] Start from the known baseline.", tree)
        self.assertIn("run 20 [succeeded] profile=rtx4090-screening name=baseline_s1", tree)
        self.assertIn("eval 30 [succeeded] profile=level1-eval", tree)
        self.assertIn("result: completion_rate=0.2 episodes=100", tree)
        self.assertIn("- spec 11 lower-kl [active]", tree)
        self.assertIn("parent: spec 10 baseline", tree)
        self.assertIn("cause: decision 101 [branch] Try lower KL after baseline plateaued.", tree)

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
            campaign.eval_selection_score(balanced),
            campaign.eval_selection_score(weak_pooled),
        )

    def test_enqueue_train_jobs_from_spec_expands_seed_templates(self) -> None:
        calls = []
        old_create = campaign.create_experiment_spec_from_document
        old_enqueue = campaign.enqueue_train_job
        old_utc = campaign._utc_stamp

        def fake_create(conn, document):
            self.assertEqual(document["slug"], "candidate")
            return {"id": 22, "goal_id": 11}

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {
                "id": 100 + len(calls),
                "profile_id": kwargs["profile_id"],
                "run_name": kwargs["run_name"],
                "run_target": kwargs["run_target"],
            }

        campaign.create_experiment_spec_from_document = fake_create
        campaign.enqueue_train_job = fake_enqueue
        campaign._utc_stamp = lambda: "20260626T120000Z"
        try:
            rows = campaign.enqueue_train_jobs_from_spec_document(
                object(),
                document={
                    "goal": "mario-level1-100of100",
                    "slug": "candidate",
                    "profile_id": "mario-ppo/post21/rtx4090",
                    "run_target": "rtx4090",
                    "priority": 7,
                    "seeds": [23, 24],
                    "wandb_group": "b-test",
                    "wandb_tags": ["mario", "confirm"],
                    "run_name_template": "btest_s{seed}_{utc}",
                    "run_description_template": "candidate seed {seed}",
                    "train_config": {"game": "SuperMarioBros-Nes-v0", "timesteps": 1024},
                },
                runtime_image_ref=RUNTIME_IMAGE_REF,
                instances_path=Path("/tmp/does-not-exist.json"),
            )
        finally:
            campaign.create_experiment_spec_from_document = old_create
            campaign.enqueue_train_job = old_enqueue
            campaign._utc_stamp = old_utc

        self.assertEqual([row["run_name"] for row in rows], ["btest_s23_20260626T120000Z", "btest_s24_20260626T120000Z"])
        self.assertEqual([call["train_config"]["seed"] for call in calls], [23, 24])
        self.assertEqual(calls[0]["priority"], 7)
        self.assertEqual(calls[0]["wandb_tags"], ["mario", "confirm"])


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
            command = train_command_for_job(job)

            self.assertEqual(config["wandb_artifact_storage_uri"], "s3://bucket/checkpoints")
            self.assertIn("--wandb-artifact-storage-uri", command)
            self.assertIn("s3://bucket/checkpoints", command)
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

            command = train_command_for_job(job)

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
            self.assertIn("--resume", command)
            self.assertIn("/tmp/downloaded/model.zip", command)
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
            "run_name": "b52_seed23",
            "run_description": "Codex-authored smoke job.",
            "wandb_group": "b52",
            "wandb_tags": ["fallback"],
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "run_target": "rtx4090",
        }

        config = normalize_train_config(job)
        command = train_command_for_job(job)

        self.assertEqual(config["wandb_tags"], "screen,post16")
        self.assertIn("--run-name", command)
        self.assertIn("b52_seed23", command)
        self.assertIn("--states", command)
        self.assertIn("Level1-1,Level1-2", command)
        self.assertIn("--wandb-group", command)
        self.assertIn("b52", command)
        self.assertIn("--runtime-image-ref", command)
        self.assertIn(RUNTIME_IMAGE_REF, command)
        self.assertIn("--run-target", command)
        self.assertIn("rtx4090", command)
        self.assertIn("--wandb", command)

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


class ArtifactConfigTests(unittest.TestCase):
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
        self.assertTrue(config["stochastic"])
        self.assertFalse(config["capture_best_video"])

    def test_json_safe_converts_nested_non_json_values(self) -> None:
        class Scalar:
            def item(self):
                return 7

        self.assertEqual(json_safe({"a": (Scalar(), Path("x"))}), {"a": [7, "x"]})


if __name__ == "__main__":
    unittest.main()
