from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlab import job_queue, wandb_leaders
from rlab.job_execution import (
    normalize_train_config,
    train_command_for_job,
    write_train_config_file,
)
from rlab.recipe_documents import materialize_train_recipe_document, validate_source_recipe_shape
from rlab.recipe_schema import validate_materialized_train_recipe
from rlab.seeds import DEFAULT_EVAL_SEED, DEFAULT_TRAIN_SEED
from tests.db_fakes import FakeConnection


RUNTIME_IMAGE_REF = (
    "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:"
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
)


class FakeWandbRun:
    def __init__(
        self, *, run_id: str, name: str, config: dict, summary: dict, group: str = ""
    ) -> None:
        self.id = run_id
        self.name = name
        self.config = config
        self.summary = summary
        self.group = group
        self.tags = ()
        self.url = f"https://wandb.ai/entity/project/runs/{run_id}"


def explicit_train_config(**overrides) -> dict:
    config = {
        "game": "SuperMarioBros-Nes-v0",
        "state": "Level1-1",
        "timesteps": 1024,
        "wandb": True,
        "wandb_mode": "online",
        "wandb_artifact_storage_uri": "s3://bucket/checkpoints",
        "checkpoint_eval_backend": "local",
    }
    config.update(overrides)
    return config


def valid_train_recipe() -> dict:
    return {
        "schema_version": 2,
        "goal": {"goal_id": "Level1-1"},
        "recipe_id": "candidate",
        "description": "Candidate seed {seed} for {recipe_id}.",
        "seeds": [23, 24],
        "campaign_id": "b-test",
        "tags": ["b55", "confirm"],
        "train_config": explicit_train_config(),
    }


class JobQueueTests(unittest.TestCase):
    def test_schema_upgrade_preserves_retired_eval_queue(self) -> None:
        conn = FakeConnection(
            results=[
                {
                    "rows": [
                        {"column_name": "id"},
                        {"column_name": "lease_owner"},
                        {"column_name": "eval_config"},
                    ]
                },
                {"row": {"table_name": None}},
            ]
        )

        job_queue.prepare_schema_upgrade(conn)

        self.assertIn(
            "ALTER TABLE eval_jobs RENAME TO legacy_eval_jobs_pre_modal",
            conn.cursor_obj.executed_sqls[-1],
        )

    def test_train_enqueue_parser_accepts_explicit_modal_backend(self) -> None:
        args = job_queue.build_train_enqueue_parser().parse_args(
            [
                "--recipe-file",
                "recipe.yaml",
                "--machine",
                "beast-3",
                "--runtime-image-ref",
                RUNTIME_IMAGE_REF,
                "--checkpoint-eval-backend",
                "modal",
            ]
        )

        self.assertEqual(args.checkpoint_eval_backend, "modal")

    def test_train_enqueue_parser_accepts_explicit_no_eval_backend(self) -> None:
        args = job_queue.build_train_enqueue_parser().parse_args(
            [
                "--recipe-file",
                "recipe.yaml",
                "--machine",
                "beast-3",
                "--checkpoint-eval-backend",
                "none",
            ]
        )

        self.assertEqual(args.checkpoint_eval_backend, "none")
        self.assertEqual(args.runtime_readiness_timeout, 20 * 60)

    def test_implicit_submission_defaults_to_modal_backend(self) -> None:
        calls = []
        document = valid_train_recipe()
        document["train_config"].pop("checkpoint_eval_backend")
        old_enqueue = job_queue.enqueue_train_job

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {"id": 100 + len(calls), "run_name": kwargs["run_name"]}

        job_queue.enqueue_train_job = fake_enqueue
        try:
            with patch.object(
                job_queue,
                "modal_eval_readiness_report",
                return_value={"ready": True, "checks": []},
            ) as preflight:
                job_queue.enqueue_train_jobs_from_recipe_document(
                    FakeConnection(),
                    document=document,
                    runtime_image_ref=RUNTIME_IMAGE_REF,
                    machine="beast-3",
                )
        finally:
            job_queue.enqueue_train_job = old_enqueue

        self.assertTrue(calls)
        self.assertTrue(
            all(call["train_config"]["checkpoint_eval_backend"] == "modal" for call in calls)
        )
        preflight.assert_called_once_with(
            runtime_image_ref=RUNTIME_IMAGE_REF,
            game="SuperMarioBros-Nes-v0",
        )

    def test_all_checked_in_recipes_materialize_modal_backend(self) -> None:
        recipe_paths = sorted(Path("experiments/goals").glob("**/recipes/*.yaml"))

        self.assertTrue(recipe_paths)
        for path in recipe_paths:
            with self.subTest(path=path):
                document = job_queue.load_recipe_document(path)
                self.assertEqual(document["train_config"]["checkpoint_eval_backend"], "modal")

    def test_submission_backend_override_is_materialized_before_enqueue(self) -> None:
        calls = []
        old_enqueue = job_queue.enqueue_train_job

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {"id": 100 + len(calls), "run_name": kwargs["run_name"]}

        job_queue.enqueue_train_job = fake_enqueue
        try:
            with patch.object(
                job_queue,
                "modal_eval_readiness_report",
                return_value={"ready": True, "checks": []},
            ) as preflight:
                job_queue.enqueue_train_jobs_from_recipe_document(
                    FakeConnection(),
                    document=valid_train_recipe(),
                    runtime_image_ref=RUNTIME_IMAGE_REF,
                    machine="beast-3",
                    checkpoint_eval_backend="modal",
                )
        finally:
            job_queue.enqueue_train_job = old_enqueue

        self.assertTrue(calls)
        preflight.assert_called_once_with(
            runtime_image_ref=RUNTIME_IMAGE_REF,
            game="SuperMarioBros-Nes-v0",
        )
        self.assertTrue(
            all(call["train_config"]["checkpoint_eval_backend"] == "modal" for call in calls)
        )
        self.assertTrue(all(call["_modal_readiness_validated"] for call in calls))

    def test_modal_readiness_reports_each_failed_check_and_remediation(self) -> None:
        check_names = (
            "config_guards",
            "fleet_eval_service",
            "postgres_schema",
            "backend_state",
            "rom_asset",
            "modal_deployment",
        )
        for check_name in check_names:
            with (
                self.subTest(check_name=check_name),
                patch.object(
                    job_queue,
                    "modal_eval_readiness_report",
                    return_value={
                        "ready": False,
                        "checks": [{"name": check_name, "ok": False, "detail": "not ready"}],
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"{check_name}.*rlab eval modal preflight",
                ):
                    job_queue.require_modal_eval_ready(
                        runtime_image_ref=RUNTIME_IMAGE_REF,
                        game="SuperMarioBros-Nes-v0",
                    )

    def test_failed_modal_preflight_inserts_no_jobs(self) -> None:
        conn = FakeConnection()
        document = valid_train_recipe()
        document["train_config"]["checkpoint_eval_backend"] = "modal"
        with patch.object(
            job_queue,
            "modal_eval_readiness_report",
            return_value={
                "ready": False,
                "checks": [{"name": "modal_deployment", "ok": False, "detail": "missing"}],
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "modal_deployment"):
                job_queue.enqueue_train_jobs_from_recipe_document(
                    conn,
                    document=document,
                    runtime_image_ref=RUNTIME_IMAGE_REF,
                    machine="beast-3",
                )

        self.assertEqual(conn.cursor_obj.executed_sqls, [])

    def test_local_submission_skips_modal_preflight(self) -> None:
        with patch.object(
            job_queue,
            "modal_eval_readiness_report",
            side_effect=AssertionError("preflight should not run"),
        ):
            job_queue.enqueue_train_job(
                FakeConnection(row={"id": 10}),
                goal_slug="Level1-1",
                train_config=explicit_train_config(),
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
            )

    def test_explicit_no_eval_submission_disables_eval_owned_behavior(self) -> None:
        document = valid_train_recipe()
        document["train_config"]["early_stop"] = [
            {
                "metric": "eval/confirm/candidate/pass",
                "operator": ">=",
                "threshold": 1.0,
            }
        ]
        document["train_config"]["checkpoint_eval_stages"] = [
            {
                "name": "screen",
                "episodes": 1,
                "n_envs": 1,
                "pass": [
                    {
                        "metric": "eval/full/outcome/success/rate/min",
                        "operator": ">=",
                        "threshold": 1.0,
                    }
                ],
            }
        ]
        calls = []

        def fake_enqueue(conn, **kwargs):
            calls.append(kwargs)
            return {"id": 100 + len(calls), "run_name": kwargs["run_name"]}

        with (
            patch.object(job_queue, "enqueue_train_job", side_effect=fake_enqueue),
            patch.object(
                job_queue,
                "modal_eval_readiness_report",
                side_effect=AssertionError("Modal preflight must not run"),
            ),
        ):
            job_queue.enqueue_train_jobs_from_recipe_document(
                FakeConnection(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
                checkpoint_eval_backend="none",
            )

        self.assertTrue(calls)
        self.assertTrue(
            all(call["train_config"]["checkpoint_eval_backend"] == "none" for call in calls)
        )
        self.assertTrue(all(call["train_config"]["early_stop"] is None for call in calls))
        self.assertTrue(all(call["train_config"]["checkpoint_eval_stages"] == [] for call in calls))
        self.assertTrue(
            all("checkpoint_eval_asset_manifest" not in call["train_config"] for call in calls)
        )
        self.assertTrue(all("checkpoint_eval_backend:none" in call["wandb_tags"] for call in calls))

    def test_checked_in_no_eval_default_is_rejected(self) -> None:
        document = valid_train_recipe()
        document["train_config"]["checkpoint_eval_backend"] = "none"
        with self.assertRaisesRegex(ValueError, "per-submission smoke/debug override"):
            job_queue.enqueue_train_jobs_from_recipe_document(
                FakeConnection(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
            )

    def test_queue_demands_groups_by_machine_and_runtime_digest(self) -> None:
        conn = FakeConnection(
            rows=[
                {
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "machine": "beast-3",
                    "pending_count": 2,
                    "active_count": 1,
                    "oldest_job_id": 7,
                }
            ]
        )

        rows = job_queue.queue_demands(conn)

        self.assertEqual(rows[0].runtime_image_ref, RUNTIME_IMAGE_REF)
        self.assertEqual(rows[0].machine, "beast-3")
        self.assertEqual(rows[0].pending_count, 2)
        self.assertEqual(rows[0].active_count, 1)
        self.assertIn("GROUP BY machine, runtime_image_ref", conn.cursor_obj.executed_sql)

    def test_schema_uses_recipe_columns_without_profile_or_spec_aliases(self) -> None:
        self.assertIn("CREATE TABLE IF NOT EXISTS train_jobs", job_queue.SCHEMA_SQL)
        self.assertIn("recipe_slug TEXT", job_queue.SCHEMA_SQL)
        self.assertIn("recipe_payload_json JSONB", job_queue.SCHEMA_SQL)
        self.assertIn("campaign_id TEXT", job_queue.SCHEMA_SQL)
        self.assertIn("ADD COLUMN IF NOT EXISTS campaign_id", job_queue.SCHEMA_SQL)
        self.assertIn("retry_of_job_id BIGINT", job_queue.SCHEMA_SQL)
        self.assertIn("ADD COLUMN IF NOT EXISTS retry_of_job_id", job_queue.SCHEMA_SQL)
        self.assertIn("ready_at TIMESTAMPTZ", job_queue.SCHEMA_SQL)
        self.assertIn("wandb_run_id TEXT", job_queue.SCHEMA_SQL)
        self.assertIn("'starting'", job_queue.SCHEMA_SQL)
        self.assertIn("machine TEXT NOT NULL", job_queue.SCHEMA_SQL)
        self.assertIn("job_id BIGINT NOT NULL UNIQUE REFERENCES train_jobs", job_queue.SCHEMA_SQL)
        self.assertNotIn("max_attempts", job_queue.SCHEMA_SQL)

    def test_runtime_validator_receives_execution_complete_config_before_insert(self) -> None:
        validated = []
        conn = FakeConnection(row={"id": 9})

        job_queue.enqueue_train_job(
            conn,
            goal_slug="Level1-1",
            recipe_slug="candidate",
            recipe_path="experiments/goals/mario/recipes/candidate.yaml",
            runtime_image_ref=RUNTIME_IMAGE_REF,
            machine="beast-3",
            train_config=explicit_train_config(),
            runtime_config_validator=lambda config: validated.append(dict(config)),
        )

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0]["machine"], "beast-3")
        self.assertIn("batch_id", validated[0])
        self.assertIn("queue_train_job_id", validated[0])
        self.assertIn("wandb_run_id", validated[0])

    def test_failed_runtime_validator_inserts_no_job(self) -> None:
        conn = FakeConnection(row={"id": 9})

        def reject(_config):
            raise RuntimeError("old runtime rejected batch_id")

        with self.assertRaisesRegex(RuntimeError, "old runtime rejected batch_id"):
            job_queue.enqueue_train_job(
                conn,
                goal_slug="Level1-1",
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
                train_config=explicit_train_config(),
                runtime_config_validator=reject,
            )

        self.assertEqual(conn.cursor_obj.executed_sqls, [])

    def test_wait_running_rejects_terminal_job_that_had_started(self) -> None:
        conn = FakeConnection(
            rows=[
                {
                    "id": 15,
                    "status": "failed",
                    "started_at": "2026-07-14T15:17:27Z",
                    "finished_at": "2026-07-14T15:18:19Z",
                    "error": "train process exited 1",
                }
            ]
        )

        result = job_queue.wait_for_job_ids(conn, [15], until="running", timeout=0)

        self.assertFalse(result["reached"])
        self.assertTrue(result["terminal_before_target"])

    def test_wandb_run_leaders_rank_by_recipe_slug(self) -> None:
        runs = [
            FakeWandbRun(
                run_id="a1",
                name="a-s1",
                config={"goal_slug": "Level1-1", "recipe_slug": "a", "seed": 1},
                summary={"train/outcome/success/window_100/rate/min": 1.0},
            ),
            FakeWandbRun(
                run_id="a2",
                name="a-s2",
                config={"goal_slug": "Level1-1", "recipe_slug": "a", "seed": 2},
                summary={"train/outcome/success/window_100/rate/min": 0.8},
            ),
            FakeWandbRun(
                run_id="b1",
                name="b-s1",
                config={"goal_slug": "Level1-1", "recipe_slug": "b", "seed": 1},
                summary={"train/outcome/success/rate/window_100/min": 0.9},
            ),
            FakeWandbRun(
                run_id="b2",
                name="b-s2",
                config={"goal_slug": "Level1-1", "recipe_slug": "b", "seed": 2},
                summary={"train/outcome/success/window_100/rate/min": 0.9},
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

    def test_enqueue_train_job_persists_recipe_runtime_and_machine(self) -> None:
        conn = FakeConnection(
            row={
                "id": 9,
                "recipe_slug": "candidate",
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "machine": "beast-3",
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
            machine="beast-3",
            train_config=explicit_train_config(),
        )

        insert_sql = conn.cursor_obj.executed_sqls[0]
        insert_params = conn.cursor_obj.executed_params_list[0]
        self.assertEqual(row["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertIn("recipe_slug", insert_sql)
        self.assertEqual(insert_params["recipe_slug"], "candidate")
        self.assertEqual(insert_params["machine"], "beast-3")
        self.assertEqual(insert_params["runtime_image_ref"], RUNTIME_IMAGE_REF)

    def test_enqueue_train_jobs_keeps_machine_in_queue_column(self) -> None:
        conn = FakeConnection(
            row={
                "id": 10,
                "recipe_slug": "candidate",
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "machine": "local-macbook",
            }
        )

        job_queue.enqueue_train_jobs_from_recipe_document(
            conn,
            document=valid_train_recipe(),
            runtime_image_ref=RUNTIME_IMAGE_REF,
            machine="local-macbook",
            seeds=[23],
        )

        insert_params = conn.cursor_obj.executed_params_list[0]
        train_config = insert_params["train_config"].adapted
        self.assertEqual(insert_params["machine"], "local-macbook")
        self.assertNotIn("machine", train_config)

    def test_enqueue_train_job_rejects_mutable_runtime_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "immutable docker digest ref"):
            job_queue.enqueue_train_job(
                FakeConnection(row={"id": 9}),
                goal_slug="goal",
                recipe_slug="candidate",
                runtime_image_ref="docker:ghcr.io/tsilva/rlab/rlab-train:latest",
                machine="beast-3",
                train_config=explicit_train_config(),
            )

    def test_train_recipe_rejects_unknown_top_level_fields(self) -> None:
        document = valid_train_recipe()
        document["hypotesis"] = "typo"

        with self.assertRaisesRegex(ValueError, "unknown train recipe field.*hypotesis"):
            job_queue.enqueue_train_jobs_from_recipe_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
            )

    def test_materialization_does_not_normalize_rejected_recipe_fields(self) -> None:
        rejected_fields = {
            "env": {"action_set": "right"},
            "state": "Level9-9",
            "states": ["Level9-8", "Level9-9"],
            "state_probs": [0.25, 0.75],
            "resume": "checkpoint.zip",
            "overrides": {"train": {"policy": {"learning_rate": 9e-4}}},
        }

        for field, value in rejected_fields.items():
            with self.subTest(field=field):
                document = valid_train_recipe()
                expected_train_config = dict(document["train_config"])
                document[field] = value

                materialized = materialize_train_recipe_document(document)

                self.assertEqual(materialized["train_config"], expected_train_config)
                with self.assertRaisesRegex(
                    ValueError,
                    rf"unknown train recipe field.*{field}",
                ):
                    validate_materialized_train_recipe(materialized)

    def test_materialization_rejects_mixed_source_and_compiled_shapes(self) -> None:
        document = valid_train_recipe()
        document["train_config"].pop("wandb_mode")
        document["train"] = {
            "policy": {"learning_rate": 2e-4},
            "environment": {
                "task": {
                    "id": "identity",
                    "action": {"set": "native"},
                    "signals": {},
                    "events": {},
                    "termination": {},
                    "reward": {"reward_mode": "native"},
                }
            },
        }
        document["logging"] = {"wandb_mode": "offline"}

        with self.assertRaisesRegex(ValueError, "cannot mix compiled train_config"):
            materialize_train_recipe_document(document)

    def test_source_recipe_shape_rejects_compiled_and_flat_policy_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "compiled or retired source field.*train_config"):
            validate_source_recipe_shape(
                {"train_config": explicit_train_config()},
                label="recipe",
            )
        for field in ("group_id", "batch_id"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, rf"compiled or retired source field.*{field}"),
            ):
                validate_source_recipe_shape({field: "legacy"}, label="recipe")
        with self.assertRaisesRegex(ValueError, "unsupported flat field.*learning_rate"):
            validate_source_recipe_shape(
                {"train": {"learning_rate": 2e-4}},
                label="recipe",
            )

    def test_train_recipe_rejects_unknown_train_config_fields(self) -> None:
        document = valid_train_recipe()
        document["train_config"]["learnnig_rate"] = 1e-4

        with self.assertRaisesRegex(ValueError, "learnnig_rate.*known train config field"):
            job_queue.enqueue_train_jobs_from_recipe_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
            )

    def test_enqueue_train_job_rejects_eval_reserved_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "reserved for eval"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                recipe_slug="candidate",
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
                train_config=explicit_train_config(seed=DEFAULT_EVAL_SEED),
            )

    def test_enqueue_train_job_rejects_noncanonical_run_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "queue run_name must use"):
            job_queue.enqueue_train_job(
                FakeConnection(),
                goal_slug="goal",
                recipe_slug="candidate",
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
                train_config=explicit_train_config(seed=23),
                submission_key="strict-name",
                run_name="candidate-s23",
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
                FakeConnection(),
                document=valid_train_recipe(),
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
                submission_key="naming-test",
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
            [
                f"{job_queue._submission_batch_id('naming-test')}-candidate-s23-20260626T120000Z",
                f"{job_queue._submission_batch_id('naming-test')}-candidate-s24-20260626T120000Z",
            ],
        )
        self.assertTrue(all(call["wandb_group"] == calls[0]["batch_id"] for call in calls))
        self.assertTrue(all(call["campaign_id"] == "b-test" for call in calls))
        self.assertEqual([call["seed"] for call in calls], [23, 24])
        self.assertTrue(all("seed" not in call["train_config"] for call in calls))
        self.assertEqual(calls[0]["recipe_slug"], "candidate")
        self.assertEqual(calls[0]["recipe_path"], "experiments/goals/mario/recipes/candidate.yaml")
        self.assertEqual(calls[0]["recipe_sha256"], "abc123")

    def test_enqueue_uses_effective_default_seed_in_row_and_run_name(self) -> None:
        calls = []
        document = valid_train_recipe()
        document.pop("seeds")

        with (
            patch.object(
                job_queue,
                "enqueue_train_job",
                side_effect=lambda _conn, **kwargs: calls.append(kwargs) or kwargs,
            ),
            patch.object(job_queue, "_utc_stamp", return_value="20260626T120000Z"),
        ):
            rows = job_queue.enqueue_train_jobs_from_recipe_document(
                FakeConnection(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
                submission_key="default-seed",
            )

        batch_id = job_queue._submission_batch_id("default-seed")
        self.assertEqual(rows[0]["seed"], DEFAULT_TRAIN_SEED)
        self.assertEqual(
            rows[0]["run_name"],
            f"{batch_id}-candidate-s{DEFAULT_TRAIN_SEED}-20260626T120000Z",
        )
        self.assertNotIn("-s-", rows[0]["run_name"])

    def test_submission_batch_is_stable_per_key_and_distinct_across_keys(self) -> None:
        first = job_queue._submission_batch_id("submission-one")
        self.assertEqual(first, job_queue._submission_batch_id("submission-one"))
        self.assertNotEqual(first, job_queue._submission_batch_id("submission-two"))
        self.assertRegex(first, r"^bx[0-9a-f]{16}$")

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
                FakeConnection(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
            )
        finally:
            job_queue.enqueue_train_job = old_enqueue

        self.assertEqual(calls[0]["train_config"]["recipe_overrides"], overrides)
        self.assertEqual(calls[0]["recipe_payload"]["recipe_overrides"], overrides)

    def test_load_recipe_document_applies_hydra_dotlist_overrides(self) -> None:
        overrides = [
            "recipe_id=lr2e4",
            "campaign_id=Level1-1-lr2e4",
            "train.policy.learning_rate=2e-4",
            "train.policy.normalize_advantage=true",
            "train.environment.task.termination.failure=[]",
        ]

        document = job_queue.load_recipe_document(
            Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/recipes/base.yaml"),
            recipe_overrides=overrides,
        )

        self.assertEqual(document["recipe_id"], "lr2e4")
        self.assertEqual(document["campaign_id"], "Level1-1-lr2e4")
        self.assertEqual(document["tags"][1], "recipe_id:lr2e4")
        self.assertEqual(document["train_config"]["learning_rate"], 2e-4)
        self.assertIs(document["train_config"]["normalize_advantage"], True)
        self.assertEqual(document["train_config"]["task"]["termination"]["failure"], [])
        self.assertEqual(document["recipe_overrides"], overrides)
        source_paths = [row["path"] for row in document["_composition"]["source_files"]]
        self.assertEqual(source_paths, list(dict.fromkeys(source_paths)))
        self.assertEqual(
            sum(path.endswith("/Level1-1/_goal.yaml") for path in source_paths),
            1,
        )

    def test_load_recipe_document_rejects_ale_only_arg_for_stable_retro(self) -> None:
        overrides = [
            "recipe_id=episodic-life",
            "train.environment.env_config.env_args.episodic_life=true",
        ]

        with self.assertRaisesRegex(
            ValueError,
            "unexpected or canonically-owned stable-retro-turbo constructor argument",
        ):
            job_queue.load_recipe_document(
                Path("experiments/goals/alepy__mspacman/recipes/base.yaml"),
                recipe_overrides=overrides,
            )

    def test_recipe_schema_rejects_removed_template_alias(self) -> None:
        document = valid_train_recipe()
        document["description"] = "Candidate {" + "spec" + "_id}"

        with self.assertRaisesRegex(ValueError, "unsupported template field"):
            job_queue.enqueue_train_jobs_from_recipe_document(
                object(),
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
            )

    def test_load_recipe_document_rejects_active_specs_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "removed active specs/ layout"):
            job_queue.load_recipe_document(Path("experiments/goals/demo/specs/base.yaml"))

    def test_queue_status_selects_recipe_without_profile(self) -> None:
        conn = FakeConnection(rows=[])

        report = job_queue.queue_status(conn, goal_slug="Level1-1")

        self.assertEqual(report["selector"], {"goal_slug": "Level1-1"})
        status_sql = conn.cursor_obj.executed_sqls[-1]
        self.assertIn("job.goal_slug", status_sql)
        self.assertNotIn("profile_id", status_sql)
        self.assertIn("eval_run.status AS eval_status", status_sql)

    def test_queue_status_exposes_published_artifact(self) -> None:
        conn = FakeConnection(
            rows=[
                {
                    "id": 17,
                    "machine": "beast-3",
                    "status": "succeeded",
                    "eval_status": "complete",
                    "published_at": "2026-07-14T16:00:00Z",
                    "promoted_step": 500,
                    "wandb_run_id": "rlab-run-id",
                    "wandb_url": "https://wandb.ai/entity/project/runs/rlab-run-id",
                    "cancel_requested": False,
                    "machine_drained": False,
                    "active_reservations": 0,
                }
            ]
        )

        report = job_queue.queue_status(conn, job_id=17)
        job = report["jobs"][0]

        self.assertEqual(job["artifact_status"], "published")
        self.assertEqual(
            job["artifact_ref"],
            "entity/project/rlab-run-id-checkpoint:step-500",
        )

    def test_claim_uses_exact_machine_and_one_stable_launch(self) -> None:
        conn = FakeConnection(
            row={
                "job_json": {"id": 7, "machine": "beast-3", "status": "launching"},
                "launch_json": {
                    "launch_id": "train-7",
                    "job_kind": "train",
                    "job_id": 7,
                    "machine": "beast-3",
                    "backend": "docker_ssh",
                },
            }
        )

        claimed = job_queue.claim_job_launch(
            conn,
            machine="beast-3",
            backend="docker_ssh",
            job_id=7,
            launch_id=job_queue.new_train_launch_id(7),
            output_uri="/output/train-7",
        )

        self.assertIsNotNone(claimed)
        sql = conn.cursor_obj.executed_sqls[1]
        self.assertIn("job.machine = %(machine)s", sql)
        self.assertIn("NOT EXISTS (SELECT 1 FROM job_launches", sql)
        self.assertEqual(conn.cursor_obj.executed_params_list[1]["launch_id"], "train-7")

    def test_cancel_only_terminalizes_pending_jobs(self) -> None:
        conn = FakeConnection(results=[{"rowcount": 1}])

        changed = job_queue.request_cancel_train_job(conn, job_id=9)

        self.assertEqual(changed, 1)
        sql = conn.cursor_obj.executed_sqls[0]
        self.assertIn("CASE WHEN status = 'pending' THEN 'canceled'", sql)
        self.assertNotIn("status IN ('pending', 'launching') THEN 'canceled'", sql)

    def test_result_identity_is_validated_before_mutation(self) -> None:
        launch = {
            "launch_id": "train-7",
            "job_kind": "train",
            "job_id": 7,
            "machine": "beast-3",
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "state": "running",
            "cancel_requested": False,
        }
        conn = FakeConnection(results=[{"row": launch}])

        with self.assertRaisesRegex(ValueError, "result machine mismatch"):
            job_queue.finish_train_launch_from_result(
                conn,
                launch_id="train-7",
                result={
                    "schema_version": 1,
                    "job_kind": "train",
                    "job_id": 7,
                    "launch_id": "train-7",
                    "machine": "beast-2",
                    "runtime_image_ref": RUNTIME_IMAGE_REF,
                    "status": "succeeded",
                    "exit_code": 0,
                },
            )

        self.assertEqual(len(conn.cursor_obj.executed_sqls), 1)

    def test_cancel_intent_dominates_success_result(self) -> None:
        launch = {
            "launch_id": "train-7",
            "job_kind": "train",
            "job_id": 7,
            "machine": "beast-3",
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "state": "running",
            "cancel_requested": True,
        }
        terminal_launch = {**launch, "state": "canceled"}
        job = {"id": 7, "run_name": "run"}
        conn = FakeConnection(
            results=[
                {"row": launch},
                {"row": terminal_launch},
                {"row": job},
                {},
            ]
        )

        job_queue.finish_train_launch_from_result(
            conn,
            launch_id="train-7",
            result={
                "schema_version": 1,
                "job_kind": "train",
                "job_id": 7,
                "launch_id": "train-7",
                "machine": "beast-3",
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "status": "succeeded",
                "exit_code": 0,
            },
        )

        self.assertEqual(conn.cursor_obj.executed_params_list[1]["state"], "canceled")
        self.assertEqual(conn.cursor_obj.executed_params_list[2]["status"], "canceled")

    def test_idempotent_batch_returns_existing_jobs(self) -> None:
        existing = [
            {"id": 10, "request_hash": "same", "submission_ordinal": 0},
            {"id": 11, "request_hash": "same", "submission_ordinal": 1},
        ]
        conn = FakeConnection(rows=existing)

        document = valid_train_recipe()
        document["train_config"]["checkpoint_eval_backend"] = "modal"
        with (
            patch.object(job_queue, "_submission_request_hash", return_value="same"),
            patch.object(
                job_queue,
                "modal_eval_readiness_report",
                side_effect=AssertionError("existing submissions do not need preflight"),
            ),
        ):
            rows = job_queue.enqueue_train_jobs_from_recipe_document(
                conn,
                document=document,
                runtime_image_ref=RUNTIME_IMAGE_REF,
                machine="beast-3",
                submission_key="request-123",
            )

        self.assertEqual([row["id"] for row in rows], [10, 11])
        self.assertEqual(len(conn.cursor_obj.executed_sqls), 1)

    def test_retry_preflights_modal_once_before_inserting(self) -> None:
        source = {
            "id": 7,
            "status": "succeeded",
            "goal_slug": "Level1-1",
            "recipe_slug": "candidate",
            "recipe_path": "recipe.yaml",
            "recipe_sha256": "abc123",
            "repo_git_commit": "deadbeef",
            "repo_dirty": False,
            "recipe_payload_json": {},
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "machine": "beast-3",
            "train_config": explicit_train_config(checkpoint_eval_backend="modal"),
            "batch_id": "b-test",
            "campaign_id": "b93",
            "run_name": "candidate-s23",
            "run_description": "retry",
            "seed": 23,
            "wandb_group": "b-test",
            "wandb_tags": [],
        }
        inserted = {**source, "id": 8, "retry_of_job_id": 7, "retried_from_job_id": 7}
        conn = FakeConnection(
            results=[
                {"rows": []},
                {"row": source},
                {},
                {"rows": []},
                {"row": source},
                {"row": inserted},
                {},
                {},
            ]
        )
        with (
            patch.object(
                job_queue,
                "modal_eval_readiness_report",
                return_value={"ready": True, "checks": []},
            ) as preflight,
            patch.object(
                job_queue,
                "asset_manifest_for_game",
                return_value={
                    "game": "SuperMarioBros-Nes-v0",
                    "sha256": "a" * 64,
                    "object_uri": "s3://bucket/rom.nes",
                    "filename": "rom.nes",
                    "provider_rom_identity": "b" * 40,
                },
            ),
        ):
            result = job_queue.retry_train_job(
                conn,
                job_id=7,
                submission_key="retry-7",
            )

        self.assertEqual(result["id"], 8)
        preflight.assert_called_once_with(
            runtime_image_ref=RUNTIME_IMAGE_REF,
            game="SuperMarioBros-Nes-v0",
        )
        insert_params = next(
            params
            for params in conn.cursor_obj.executed_params_list
            if params.get("retry_of_job_id") == 7
        )
        self.assertEqual(insert_params["batch_id"], "b-test")
        self.assertEqual(insert_params["campaign_id"], "b93")
        self.assertEqual(insert_params["wandb_group"], "b-test")
        self.assertIn("campaign_id:b93", insert_params["wandb_tags"])
        self.assertIn("retry_of_job_id:7", insert_params["wandb_tags"])

    def test_machine_controls_persist_drain_and_capacity(self) -> None:
        row = {
            "machine": "beast-3",
            "drained": True,
            "effective_capacity": 4,
        }
        conn = FakeConnection(row=row)

        result = job_queue.set_machine_control(
            conn,
            machine="beast-3",
            drained=True,
            effective_capacity=4,
            reason="maintenance",
        )

        self.assertEqual(result, row)
        params = conn.cursor_obj.executed_params_list[1]
        self.assertIs(params["drained"], True)
        self.assertEqual(params["effective_capacity"], 4)


class JobExecutionTests(unittest.TestCase):
    def test_artifact_storage_placeholder_uses_one_canonical_resolver(self) -> None:
        job = {
            "id": 12,
            "train_config": {"wandb_artifact_storage_uri": "CHECKPOINT_BUCKET_URI"},
        }
        with patch.dict("os.environ", {"CHECKPOINT_BUCKET_URI": '"s3://bucket/checkpoints"'}):
            config = normalize_train_config(job, require_explicit_train_fields=False)

        self.assertEqual(config["wandb_artifact_storage_uri"], "s3://bucket/checkpoints")

    def test_empty_artifact_storage_stays_deferred_at_queue_boundary(self) -> None:
        job = {"id": 12, "train_config": {"wandb_artifact_storage_uri": ""}}
        with patch.dict("os.environ", {"CHECKPOINT_BUCKET_URI": "s3://bucket/checkpoints"}):
            config = normalize_train_config(job, require_explicit_train_fields=False)

        self.assertEqual(config["wandb_artifact_storage_uri"], "")

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
            "seed": 23,
            "wandb_group": "level1-1-lowkl-lrdecay",
            "wandb_tags": ["fallback"],
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "machine": "beast-3",
        }

        config = normalize_train_config(job)
        with tempfile.TemporaryDirectory() as tmp:
            config_path = write_train_config_file(job, Path(tmp) / "train_config.json")
            command = train_command_for_job(config_path)
            written_config = json.loads(config_path.read_text(encoding="utf-8"))

        self.assertEqual(
            config["wandb_tags"],
            "screen,game_family:NES-SuperMarioBros,goal_id:Level1-1,"
            "recipe_id:base,level_id:Level1-1",
        )
        self.assertEqual(written_config["recipe_slug"], "base")
        self.assertEqual(written_config["seed"], 23)
        self.assertEqual(written_config["machine"], "beast-3")
        self.assertEqual(written_config["queue_train_job_id"], 12)
        self.assertEqual(command[-3:], ["rlab.train", "--train-config-json", str(config_path)])
        self.assertNotIn("WANDB_API_KEY", json.dumps(written_config))


if __name__ == "__main__":
    unittest.main()
