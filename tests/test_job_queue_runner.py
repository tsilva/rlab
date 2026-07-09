from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rlab import job_queue, wandb_leaders
from rlab.job_execution import normalize_train_config, train_command_for_job, write_train_config_file
from rlab.seeds import DEFAULT_EVAL_SEED


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)


class FakeWandbRun:
    def __init__(self, *, run_id: str, name: str, config: dict, summary: dict, group: str = "") -> None:
        self.id = run_id
        self.name = name
        self.config = config
        self.summary = summary
        self.group = group
        self.tags = ()
        self.url = f"https://wandb.ai/entity/project/runs/{run_id}"


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


def valid_train_recipe() -> dict:
    return {
        "schema_version": 1,
        "goal": {"goal_id": "Level1-1"},
        "recipe_id": "candidate",
        "description": "Candidate seed {seed} for {recipe_id}.",
        "seeds": [23, 24],
        "group_id": "b-test",
        "run_name_template": "{group_id}_{recipe_id}_s{seed}_{timestamp}",
        "tags": ["b55", "confirm"],
        "selection_metrics": ["train/completion_episode_rate", "train/reward/mean"],
        "train_config": explicit_train_config(),
    }


class JobQueueTests(unittest.TestCase):
    def test_schema_uses_recipe_columns_without_profile_or_spec_aliases(self) -> None:
        self.assertIn("CREATE TABLE IF NOT EXISTS train_jobs", job_queue.SCHEMA_SQL)
        self.assertIn("recipe_slug TEXT", job_queue.SCHEMA_SQL)
        self.assertIn("recipe_payload_json JSONB", job_queue.SCHEMA_SQL)

    def test_wandb_run_leaders_rank_by_recipe_slug(self) -> None:
        runs = [
            FakeWandbRun(
                run_id="a1",
                name="a-s1",
                config={"goal_slug": "Level1-1", "recipe_slug": "a", "seed": 1},
                summary={"train/info/level_complete/rate/min/last": 1.0},
            ),
            FakeWandbRun(
                run_id="a2",
                name="a-s2",
                config={"goal_slug": "Level1-1", "recipe_slug": "a", "seed": 2},
                summary={"train/info/level_complete/rate/min/last": 0.8},
            ),
            FakeWandbRun(
                run_id="b1",
                name="b-s1",
                config={"goal_slug": "Level1-1", "recipe_slug": "b", "seed": 1},
                summary={"train/info/level_complete/rate/min/last": 0.9},
            ),
            FakeWandbRun(
                run_id="b2",
                name="b-s2",
                config={"goal_slug": "Level1-1", "recipe_slug": "b", "seed": 2},
                summary={"train/info/level_complete/rate/min/last": 0.9},
            ),
        ]

        scores = [
            score
            for score in (
                wandb_leaders.run_score(run, objective_keys=wandb_leaders.RUN_OBJECTIVE_KEYS)
                for run in runs
            )
            if score is not None
        ]
        leaders = wandb_leaders.rank_run_leaders(scores, min_seeds=2)

        self.assertEqual(leaders[0].recipe_slug, "b")
        self.assertEqual(leaders[0].worst_seed, 0.9)
        self.assertEqual(leaders[1].recipe_slug, "a")

    def test_enqueue_train_job_persists_recipe_runtime_and_target(self) -> None:
        conn = FakeConnection(
            row={
                "id": 9,
                "recipe_slug": "candidate",
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "run_target": "rtx4090",
            }
        )

        row = job_queue.enqueue_train_job(
            conn,
            goal_slug="Level1-1",
            recipe_slug="candidate",
            recipe_path="experiments/goals/mario/recipes/candidate.yaml",
            recipe_sha256="abc123",
            recipe_payload={"recipe_id": "candidate"},
            runtime_image_ref=RUNTIME_IMAGE_REF,
            run_target="rtx4090",
            train_config=explicit_train_config(),
        )

        insert_sql = conn.cursor_obj.executed_sqls[0]
        insert_params = conn.cursor_obj.executed_params_list[0]
        self.assertEqual(row["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertIn("recipe_slug", insert_sql)
        self.assertEqual(insert_params["recipe_slug"], "candidate")
        self.assertEqual(insert_params["run_target"], "rtx4090")
        self.assertEqual(insert_params["runtime_image_ref"], RUNTIME_IMAGE_REF)

    def test_enqueue_train_job_rejects_mutable_runtime_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable docker digest ref"):
            job_queue.enqueue_train_job(
                FakeConnection(row={"id": 9}),
                goal_slug="goal",
                recipe_slug="candidate",
                runtime_image_ref="docker:ghcr.io/tsilva/rlab/rlab-train:latest",
                train_config=explicit_train_config(),
            )

    def test_enqueue_train_job_rejects_legacy_event_launch_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy event key.*done_on_info_json"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                recipe_slug="candidate",
                runtime_image_ref=RUNTIME_IMAGE_REF,
                train_config=explicit_train_config(done_on_info_json={"level_change": ["bad"]}),
            )

    def test_train_recipe_rejects_unknown_top_level_fields(self) -> None:
        document = valid_train_recipe()
        document["hypotesis"] = "typo"

        with self.assertRaisesRegex(ValueError, "unknown train recipe field.*hypotesis"):
            job_queue.enqueue_train_jobs_from_recipe_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
            )

    def test_train_recipe_rejects_unknown_train_config_fields(self) -> None:
        document = valid_train_recipe()
        document["train_config"]["learnnig_rate"] = 1e-4

        with self.assertRaisesRegex(ValueError, "learnnig_rate.*known train config field"):
            job_queue.enqueue_train_jobs_from_recipe_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
            )

    def test_enqueue_train_job_rejects_eval_reserved_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved for eval"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                recipe_slug="candidate",
                runtime_image_ref=RUNTIME_IMAGE_REF,
                train_config=explicit_train_config(seed=DEFAULT_EVAL_SEED),
            )

    def test_enqueue_train_jobs_from_recipe_document_derives_seeded_run_names(self) -> None:
        calls = []
        old_enqueue = job_queue.enqueue_train_job
        old_utc = job_queue._utc_stamp

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {"id": 100 + len(calls), "run_name": kwargs["run_name"]}

        job_queue.enqueue_train_job = fake_enqueue
        job_queue._utc_stamp = lambda: "20260626T120000Z"
        try:
            rows = job_queue.enqueue_train_jobs_from_recipe_document(
                object(),
                document=valid_train_recipe(),
                runtime_image_ref=RUNTIME_IMAGE_REF,
                recipe_path="experiments/goals/mario/recipes/candidate.yaml",
                recipe_sha256="abc123",
                repo_git_commit="deadbeef",
                repo_dirty=True,
            )
        finally:
            job_queue.enqueue_train_job = old_enqueue
            job_queue._utc_stamp = old_utc

        self.assertEqual(
            [row["run_name"] for row in rows],
            ["b-test_candidate_s23_20260626T120000Z", "b-test_candidate_s24_20260626T120000Z"],
        )
        self.assertEqual([call["train_config"]["seed"] for call in calls], [23, 24])
        self.assertEqual(calls[0]["recipe_slug"], "candidate")
        self.assertEqual(calls[0]["recipe_path"], "experiments/goals/mario/recipes/candidate.yaml")
        self.assertEqual(calls[0]["recipe_sha256"], "abc123")

    def test_enqueue_train_jobs_records_recipe_overrides_in_worker_config(self) -> None:
        calls = []
        old_enqueue = job_queue.enqueue_train_job
        overrides = ["train.policy.learning_rate=2e-4", "recipe_id=lr2e4"]
        document = valid_train_recipe()
        document["recipe_overrides"] = overrides

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {"id": 100 + len(calls), "run_name": kwargs["run_name"]}

        job_queue.enqueue_train_job = fake_enqueue
        try:
            job_queue.enqueue_train_jobs_from_recipe_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
            )
        finally:
            job_queue.enqueue_train_job = old_enqueue

        self.assertEqual(calls[0]["train_config"]["recipe_overrides"], overrides)
        self.assertEqual(calls[0]["recipe_payload"]["recipe_overrides"], overrides)

    def test_load_recipe_document_applies_hydra_dotlist_overrides(self) -> None:
        overrides = [
            "recipe_id=lr2e4",
            "group_id=Level1-1-lr2e4",
            "train.policy.learning_rate=2e-4",
            "train.policy.normalize_advantage=true",
            "train.environment.env_config.done_on_events=[]",
        ]

        document = job_queue.load_recipe_document(
            Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/base.yaml"),
            recipe_overrides=overrides,
        )

        self.assertEqual(document["recipe_id"], "lr2e4")
        self.assertEqual(document["group_id"], "Level1-1-lr2e4")
        self.assertEqual(document["tags"][1], "recipe_id:lr2e4")
        self.assertEqual(document["train_config"]["learning_rate"], 2e-4)
        self.assertIs(document["train_config"]["normalize_advantage"], True)
        self.assertEqual(document["train_config"]["done_on_events"], [])
        self.assertEqual(document["recipe_overrides"], overrides)

    def test_load_recipe_document_allows_ale_episodic_life_override(self) -> None:
        overrides = [
            "recipe_id=episodic-life",
            "train.environment.env_config.episodic_life=true",
        ]

        document = job_queue.load_recipe_document(
            Path("experiments/goals/alepy__mspacman/recipes/base.yaml"),
            recipe_overrides=overrides,
        )

        self.assertEqual(document["recipe_id"], "episodic-life")
        self.assertEqual(document["train_config"]["timesteps"], 100000000)
        self.assertIs(document["train_config"]["episodic_life"], True)
        self.assertEqual(document["train_config"]["obs_crop"], [0, 0, 37, 0])
        self.assertEqual(document["recipe_overrides"], overrides)

    def test_recipe_schema_rejects_removed_template_alias(self) -> None:
        document = valid_train_recipe()
        document["description"] = "Candidate {" + "spec" + "_id}"

        with self.assertRaisesRegex(ValueError, "unsupported template field"):
            job_queue.enqueue_train_jobs_from_recipe_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
            )

    def test_load_recipe_document_rejects_active_specs_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "removed active specs/ layout"):
            job_queue.load_recipe_document(Path("experiments/goals/demo/specs/base.yaml"))

    def test_queue_status_selects_recipe_without_profile(self) -> None:
        conn = FakeConnection(rows=[])

        report = job_queue.queue_status(conn, goal_slug="Level1-1")

        self.assertEqual(report["goal_slug"], "Level1-1")
        status_sql = conn.cursor_obj.executed_sqls[-1]
        self.assertIn("recipe_slug", status_sql)


class JobExecutionTests(unittest.TestCase):
    def test_train_command_uses_recipe_metadata_without_secrets(self) -> None:
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
            "recipe_slug": "base",
            "recipe_path": "experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/base.yaml",
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
            "screen,goal_id:Level1-1,recipe_id:base,level_id:Level1-1",
        )
        self.assertEqual(written_config["recipe_slug"], "base")
        self.assertEqual(written_config["queue_train_job_id"], 12)
        self.assertEqual(command[-3:], ["rlab.train", "--train-config-json", str(config_path)])
        self.assertNotIn("WANDB_API_KEY", json.dumps(written_config))


if __name__ == "__main__":
    unittest.main()
