from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import time
import unittest
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab.modal_eval_config import load_modal_eval_config, modal_app_name
from rlab.modal_eval_orchestrator import (
    DefaultModalInvoker,
    _stage_metrics,
    _verify_checkpoint_artifacts,
    accept_attempt_result,
    available_eval_slots,
    budget_allows,
    cancel_requested_attempts,
    deterministic_eval_failure,
    dispatch_pending,
    ensure_eval_runs,
    enqueue_post_train_promotions,
    project_eval_results,
    ingest_mailbox_announcements,
    poll_attempts,
    project_artifact_references,
    promotion_candidate_key,
    reconcile_canceled_eval_state,
    reconcile_definitive_non_acceptance,
    reconcile_eval_run_failures,
    reconcile_publication_finishing,
    reconcile_promotions,
    reconcile_stop_delivery_slo,
    round_robin_jobs,
    run_service_eval_pass,
    terminalize_failed_eval_runs,
    terminalize_artifact_only_runs,
    terminalize_runs,
)
from rlab.checkpoint_eval_worker import evaluation_metric_payload
from rlab.modal_eval_projection import project_payload
from rlab.modal_eval_protocol import (
    PROTOCOL_SCHEMA_VERSION,
    SEED_PROTOCOL,
    build_execution_contract,
    execution_key,
    job_key,
    validate_announcement,
    validate_attempt_result,
)
from rlab.modal_eval_storage import ObjectStore, file_sha256
from rlab.modal_eval_worker import execute_attempt
from rlab.policy_bundle import (
    build_model_document,
    build_recipe_document,
    playback_contract_sha256,
    write_canonical_json,
)
from rlab.recipe_documents import compose_train_document
from rlab.rom_assets import install_rom_file
from rlab.checkpoint_coordinator import (
    checkpoint_event_path,
    prepare_checkpoint_event,
    process_upload,
    reconcile_orphan_models,
)
from rlab.metric_store import MetricStore
from rlab.training_backend import training_backend_config_hash
from rlab import checkpoint_coordinator, modal_eval_cli
from tests.db_fakes import FakeConnection


def contract(root: Path, *, episodes: int = 2, n_envs: int = 2) -> dict:
    rom = root / "game.nes"
    rom.write_bytes(b"NES\x1a" + bytes(12) + b"rom")
    return build_execution_contract(
        checkpoint_sha256="a" * 64,
        runtime_image_ref="docker:example.invalid/rlab@sha256:" + "b" * 64,
        eval_environment={"game": "Game-Nes-v0", "task": {}},
        episodes=episodes,
        n_envs=n_envs,
        max_steps=100,
        seed=10_000,
        seed_protocol=SEED_PROTOCOL,
        asset_manifest={
            "schema_version": 2,
            "game": "Game-Nes-v0",
            "filename": rom.name,
            "size_bytes": rom.stat().st_size,
            "sha256": file_sha256(rom),
            "object_uri": rom.resolve().as_uri(),
            "provider_rom_identity": "c" * 40,
            "provider_rom_identity_algorithm": "sha1-provider-body-v1",
        },
    )


def successful_result(eval_contract: dict, *, attempt_id: str = "attempt") -> dict:
    asset = eval_contract.get("asset")
    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "contract_schema_version": eval_contract["schema_version"],
        "attempt_id": attempt_id,
        "execution_key": execution_key(eval_contract),
        "checkpoint_sha256": eval_contract["checkpoint_sha256"],
        "runtime_image_ref": eval_contract["runtime_image_ref"],
        "rom_sha256": asset["sha256"] if isinstance(asset, dict) else "",
        "seed_protocol": eval_contract["seed_protocol"],
        "n_envs": eval_contract["n_envs"],
        "episodes": eval_contract["episodes"],
        "status": "succeeded",
        "duration_seconds": 1.0,
        "metrics": {
            "eval/full/episode/return/mean": 1.0,
        },
        "episode_results": [
            {
                "seed": eval_contract["seed"],
                "seed_protocol": eval_contract["seed_protocol"],
                "seed_lane": index,
                "seed_episode_ordinal": 0,
                "start_state": "Start",
            }
            for index in range(eval_contract["episodes"])
        ],
    }


class ModalEvalContractTests(unittest.TestCase):
    def test_poll_attempt_provider_failure_does_not_block_later_record(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            {
                "id": 1,
                "attempt_id": "attempt-1",
                "eval_job_id": 11,
                "retry_round": 0,
                "result_uri": "s3://bucket/one.json",
                "expires_at": datetime.now(UTC) + timedelta(minutes=5),
            },
            {
                "id": 2,
                "attempt_id": "attempt-2",
                "eval_job_id": 12,
                "retry_round": 0,
                "result_uri": "s3://bucket/two.json",
                "expires_at": datetime.now(UTC) + timedelta(minutes=5),
            },
        ]
        store = mock.MagicMock()
        store.get_json_optional.side_effect = [RuntimeError("provider down"), None]

        with mock.patch("rlab.modal_eval_orchestrator._mark_attempt_failure") as mark_failure:
            changed = poll_attempts(
                conn,
                store,
                mock.MagicMock(),
                mock.MagicMock(),
                deadline_monotonic=time.monotonic() + 1,
            )

        self.assertEqual(changed, 1)
        self.assertEqual(store.get_json_optional.call_count, 2)
        self.assertIn("result observation failed", mark_failure.call_args.kwargs["error"])

    def test_poll_attempts_loads_checkpoint_hash_for_accepted_commit(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            {
                "id": 101,
                "attempt_id": "attempt-101",
                "result_uri": "s3://bucket/result.json",
                "receipt_json": None,
                "purpose": "acceptance",
                "stage_index": 0,
                "checkpoint_sha256": "a" * 64,
            }
        ]
        store = mock.MagicMock()
        store.get_json_optional.return_value = {"status": "succeeded"}
        config = mock.MagicMock()
        config.timeout_for.return_value = 1200

        with mock.patch("rlab.modal_eval_orchestrator.accept_attempt_result") as accept_result:
            changed = poll_attempts(
                conn,
                store,
                mock.MagicMock(),
                config,
                deadline_monotonic=time.monotonic() + 1,
            )

        self.assertEqual(changed, 1)
        self.assertIn("j.checkpoint_sha256", cursor.execute.call_args_list[0].args[0])
        self.assertEqual(
            accept_result.call_args.kwargs["attempt"]["checkpoint_sha256"],
            "a" * 64,
        )

    def test_canceled_train_reconciliation_closes_late_eval_rows(self) -> None:
        conn = FakeConnection(results=[{"rowcount": 3}, {"rowcount": 1}, {"rowcount": 2}])

        self.assertEqual(
            reconcile_canceled_eval_state(conn),
            {"jobs": 3, "runs": 1, "workers": 2},
        )

        jobs_sql, runs_sql, workers_sql = conn.cursor_obj.executed_sqls
        self.assertIn("t.status = 'canceled'", jobs_sql)
        self.assertIn("r.outcome = 'canceled'", jobs_sql)
        self.assertIn("'pending', 'dispatching', 'submitted', 'blocked_budget'", jobs_sql)
        self.assertIn("t.status = 'canceled'", runs_sql)
        self.assertIn("outcome = 'canceled'", runs_sql)
        self.assertIn("r.status NOT IN ('complete', 'failed', 'canceled')", runs_sql)
        self.assertIn("FROM eval_attempts a", workers_sql)
        self.assertIn("a.status = 'canceled'", workers_sql)
        self.assertIn("w.status IN ('launching', 'running')", workers_sql)

    def test_canceling_eval_attempt_also_terminalizes_worker(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [
            {
                "id": 101,
                "attempt_id": "attempt-101",
                "modal_call_id": "fc-101",
            }
        ]
        cursor.rowcount = 1
        invoker = mock.MagicMock()

        changed = cancel_requested_attempts(
            conn,
            invoker,
            deadline_monotonic=time.monotonic() + 1,
        )

        self.assertEqual(changed, 1)
        invoker.cancel.assert_called_once_with("fc-101")
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertIn("a.attempt_id", statements[0])
        self.assertTrue(
            any(
                "UPDATE worker_attempts" in statement and "status = 'canceled'" in statement
                for statement in statements
            )
        )

    def test_operator_retry_starts_a_fresh_attempt_round(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {
            "id": 10254,
            "train_job_id": 38,
            "status": "pending",
            "retry_round": 1,
        }
        with (
            mock.patch.object(modal_eval_cli, "_conn", return_value=conn),
            mock.patch.object(modal_eval_cli, "_kick") as kick,
        ):
            self.assertEqual(modal_eval_cli.cmd_retry(SimpleNamespace(eval_job_id=10254)), 0)

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        statement = statements[0]
        self.assertIn("retry_round = retry_round + 1", statement)
        self.assertNotIn("count(*)", statement)
        self.assertTrue(any("CASE WHEN complete_announcement_seen" in sql for sql in statements))
        self.assertTrue(
            any("UPDATE train_jobs SET status = 'finalizing'" in sql for sql in statements)
        )
        kick.assert_called_once()

    def test_rom_free_announcement_verifies_only_model_and_metadata(self) -> None:
        announcement = {
            "train_job_id": 38,
            "step": 256,
            "runtime_image_ref": "docker:example.invalid/rlab@sha256:" + "b" * 64,
            "model_uri": "s3://bucket/model.zip",
            "sha256": "a" * 64,
            "metadata_uri": "s3://bucket/metadata.json",
            "metadata_sha256": "c" * 64,
            "eval": {"asset": None},
        }
        store = mock.MagicMock(spec=ObjectStore)
        store.scheme = "s3"
        store.head.side_effect = [
            {"size": 10, "metadata": {"sha256": "a" * 64}},
            {"size": 20, "metadata": {"sha256": "c" * 64}},
        ]
        store.get_json.return_value = {
            "queue_train_job_id": 38,
            "checkpoint_step": 256,
            "runtime_image_ref": announcement["runtime_image_ref"],
        }

        _verify_checkpoint_artifacts(store, announcement)

        self.assertEqual(store.head.call_count, 2)

    def test_rom_free_execution_contract_accepts_empty_rom_identity(self) -> None:
        eval_contract = build_execution_contract(
            checkpoint_sha256="a" * 64,
            runtime_image_ref="docker:example.invalid/rlab@sha256:" + "b" * 64,
            eval_environment={"env_provider": "rlab", "game": "Bandit-v0"},
            episodes=1,
            n_envs=1,
            max_steps=1,
            seed=10_000,
            seed_protocol=SEED_PROTOCOL,
            asset_manifest=None,
        )

        self.assertIsNone(eval_contract["asset"])
        self.assertEqual(
            validate_attempt_result(
                successful_result(eval_contract),
                contract=eval_contract,
                attempt_id="attempt",
            )["rom_sha256"],
            "",
        )

        result = successful_result(eval_contract)
        result["episode_results"][0]["start_state"] = None
        self.assertEqual(
            validate_attempt_result(
                result,
                contract=eval_contract,
                attempt_id="attempt",
            )["status"],
            "succeeded",
        )

    def test_modal_and_local_full_eval_publish_the_same_metric_projection(self) -> None:
        raw_metrics = {
            "return_mean": 2.0,
            "eval/full/episode/return/mean": 2.0,
            "eval/full/episode/return/std": 0.5,
            "eval/full/episode/return/median": 2.0,
            "eval/full/episode/return/best": 3.0,
            "eval/full/episode/count": 10,
            "eval/full/outcome/success/rate/min": 0.8,
            "eval/full/progress/x/max": 400,
            "eval/full/duration/seconds": 3.0,
        }
        expected = evaluation_metric_payload(
            protocol="full",
            metrics=raw_metrics,
            checkpoint_step=123,
            checkpoint_artifact="s3://bucket/checkpoint.zip",
            eval_source="modal",
        )

        actual = _stage_metrics(
            {
                "purpose": "promotion",
                "checkpoint_step": 123,
                "checkpoint_uri": "s3://bucket/checkpoint.zip",
                "train_config": {"metrics_schema_version": 5},
            },
            raw_metrics,
            True,
        )

        self.assertEqual(actual, expected)

    def test_v5_acceptance_projection_suppresses_constant_outcomes(self) -> None:
        raw_metrics = {
            "eval/full/episode/return/mean": 2.0,
            "eval/full/episode/length/mean": 10.0,
            "eval/full/episode/count": 100,
            "eval/full/progress/x/max": 3160.0,
            "eval/full/outcome/success/rate/min": 1.0,
            "eval/full/outcome/success/from/Level1-1/rate": 1.0,
            "eval/full/outcome/reason/stalled/rate": 0.0,
            "eval/full/duration/seconds": 12.0,
            "_acceptance_duration_seconds": 13.0,
            "_acceptance_aggregates": {
                "episodes_planned": 100,
                "episodes_completed": 100,
                "failure_count": 0,
            },
        }
        job = {
            "purpose": "acceptance",
            "checkpoint_step": 123,
            "checkpoint_uri": "s3://bucket/checkpoint.zip",
            "train_config": {"metrics_schema_version": 5},
        }

        projected = _stage_metrics(job, raw_metrics, True)

        self.assertEqual(projected["global_step"], 123)
        self.assertEqual(projected["eval/acceptance/pass"], 1.0)
        self.assertEqual(projected["eval/acceptance/episodes/planned"], 100)
        self.assertEqual(projected["eval/acceptance/episodes/completed"], 100)
        self.assertEqual(projected["eval/acceptance/duration/seconds"], 13.0)
        self.assertEqual(projected["eval/full/episode/return/mean"], 2.0)
        self.assertEqual(projected["eval/full/episode/length/mean"], 10.0)
        self.assertEqual(projected["eval/full/episode/count"], 100)
        self.assertEqual(projected["eval/full/progress/x/max"], 3160.0)
        self.assertEqual(
            projected["eval/full/checkpoint/artifact"],
            "s3://bucket/checkpoint.zip",
        )
        self.assertEqual(projected["eval/full/source"], "modal")
        self.assertNotIn("eval/acceptance/failure/count", projected)
        self.assertNotIn("eval/full/checkpoint/step", projected)
        self.assertNotIn("eval/full/duration/seconds", projected)
        self.assertFalse(any(name.startswith("eval/full/outcome/") for name in projected))

    def test_v5_rejected_acceptance_has_no_partial_full_projection(self) -> None:
        projected = _stage_metrics(
            {
                "purpose": "acceptance",
                "checkpoint_step": 123,
                "checkpoint_uri": "s3://bucket/checkpoint.zip",
                "train_config": {"metrics_schema_version": 5},
            },
            {
                "eval/full/episode/return/mean": 2.0,
                "_acceptance_duration_seconds": 3.0,
                "_acceptance_aggregates": {
                    "episodes_planned": 100,
                    "episodes_completed": 1,
                    "failure_count": 1,
                },
            },
            False,
        )

        self.assertEqual(projected["eval/acceptance/pass"], 0.0)
        self.assertFalse(any(name.startswith("eval/full/") for name in projected))

    def test_preflight_checks_schema_asset_backend_and_exact_deployment(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {"drained": False}
        function = mock.MagicMock()
        image_ref = "docker:example.invalid/rlab@sha256:" + "b" * 64
        manifest = {
            "game": "SuperMarioBros-Nes-v0",
            "sha256": "c" * 64,
            "object_uri": "s3://bucket/rom.nes",
            "filename": "rom.nes",
            "provider_rom_identity": "d" * 40,
        }
        with (
            mock.patch.object(modal_eval_cli, "_conn", return_value=conn),
            mock.patch.object(modal_eval_cli, "_missing_schema_tables", return_value=[]),
            mock.patch.object(modal_eval_cli, "asset_manifest_for_game", return_value=manifest),
            mock.patch.object(modal_eval_cli, "object_store_base_uri", return_value="s3://bucket"),
            mock.patch.object(modal_eval_cli, "ObjectStore") as object_store,
            mock.patch(
                "rlab.fleet_service.eval_service_health",
                return_value={"ready": True},
            ),
            mock.patch("modal.Function.from_name", return_value=function),
        ):
            from rlab.runtime_contract import train_config_contract_sha256

            object_store.return_value.head.return_value = {
                "size": 1024,
                "metadata": {"sha256": manifest["sha256"]},
            }
            function.remote.return_value = {
                "schema_version": 1,
                "app_name": "rlab-eval-" + "b" * 12,
                "runtime_image_ref": image_ref,
                "source_sha": "source",
                "train_config_contract_sha256": train_config_contract_sha256(),
                "object_store_configured": True,
            }
            report = modal_eval_cli.modal_preflight(
                runtime_image_ref=image_ref,
                game="SuperMarioBros-Nes-v0",
                env_provider="supermariobrosnes-turbo",
            )

        self.assertTrue(report["ready"])
        self.assertEqual(
            {check["name"] for check in report["checks"]},
            {
                "config_guards",
                "fleet_eval_service",
                "postgres_schema",
                "backend_state",
                "rom_asset",
                "modal_deployment",
                "modal_startup_probe",
            },
        )
        function.hydrate.assert_called_once_with()

    def test_checked_in_config_has_ten_call_hard_cap(self) -> None:
        config = load_modal_eval_config(Path("experiments/modal_eval.yaml"))
        self.assertTrue(config.enabled)
        self.assertTrue(config.cleanup_enabled)
        self.assertEqual(config.cleanup_interval_seconds, 3600)
        self.assertEqual(config.cleanup_grace_seconds, 86400)
        self.assertEqual(config.cleanup_max_stops_per_pass, 10)
        self.assertEqual(config.hard_max_active, 10)
        self.assertEqual(config.max_containers, 10)
        self.assertFalse(config.single_use_containers)
        self.assertEqual(config.max_attempts, 2)

    def test_resume_only_clears_the_drain(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {"backend": "modal", "drained": False}
        with (
            mock.patch.object(modal_eval_cli, "_conn", return_value=conn),
            mock.patch.object(modal_eval_cli, "_kick"),
        ):
            self.assertEqual(
                modal_eval_cli._set_backend(drained=False, reason="operator resume"),
                0,
            )

    def test_train_image_packages_modal_contract_at_expected_path(self) -> None:
        dockerfile = Path("containers/train/Dockerfile").read_text(encoding="utf-8")
        self.assertIn(
            "COPY experiments/modal_eval.yaml /root/rlab/experiments/modal_eval.yaml",
            dockerfile,
        )

    def test_config_rejects_unknown_fields_and_excess_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path("experiments/modal_eval.yaml").read_text(encoding="utf-8")
            unknown = Path(temporary) / "unknown.yaml"
            unknown.write_text(source.replace("  cpu: 8.0", "  cpu: 8.0\n  surprise: true"))
            with self.assertRaisesRegex(ValueError, "resources has unknown"):
                load_modal_eval_config(unknown)
            mismatched = Path(temporary) / "mismatched.yaml"
            mismatched.write_text(source.replace("  max_containers: 10", "  max_containers: 11"))
            with self.assertRaisesRegex(ValueError, "must equal"):
                load_modal_eval_config(mismatched)
            retired_preview = Path(temporary) / "retired-preview.yaml"
            retired_preview.write_text(source + "preview:\n  enabled: false\n")
            with self.assertRaisesRegex(ValueError, "unknown field.*preview"):
                load_modal_eval_config(retired_preview)
            invalid_cleanup = Path(temporary) / "invalid-cleanup.yaml"
            invalid_cleanup.write_text(
                source.replace("  grace_seconds: 86400", "  grace_seconds: 0")
            )
            with self.assertRaisesRegex(ValueError, "cleanup.grace_seconds must be at least 1"):
                load_modal_eval_config(invalid_cleanup)

    def test_execution_identity_excludes_job_purpose_and_job_identity_includes_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value = contract(Path(temporary))
            first = execution_key(value)
            self.assertEqual(first, execution_key(json.loads(json.dumps(value))))
            screen = job_key(
                train_job_id=1,
                ledger_id=2,
                stage_name="screen",
                purpose="screen",
                candidate_stop=False,
                execution_key_value=first,
                decision_rules=[],
            )
            confirm = job_key(
                train_job_id=1,
                ledger_id=2,
                stage_name="confirm",
                purpose="confirm",
                candidate_stop=True,
                execution_key_value=first,
                decision_rules=[{"metric": "x", "operator": ">=", "threshold": 1}],
            )
            self.assertNotEqual(screen, confirm)

    def test_result_validation_enforces_episode_seed_and_start_state_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value = contract(Path(temporary))
            result = successful_result(value)
            self.assertEqual(
                validate_attempt_result(result, contract=value, attempt_id="attempt")["status"],
                "succeeded",
            )
            result["episode_results"][1]["seed_lane"] = 0
            with self.assertRaisesRegex(ValueError, "duplicated"):
                validate_attempt_result(result, contract=value, attempt_id="attempt")
            result["schema_version"] = PROTOCOL_SCHEMA_VERSION + 1
            with self.assertRaisesRegex(ValueError, "schema version"):
                validate_attempt_result(result, contract=value, attempt_id="attempt")

    def test_result_validation_rejects_runtime_and_rom_identity_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value = contract(Path(temporary))
            result = successful_result(value)
            result["runtime_image_ref"] = "docker:other@sha256:" + "d" * 64
            with self.assertRaisesRegex(ValueError, "runtime identity"):
                validate_attempt_result(result, contract=value, attempt_id="attempt")
            result = successful_result(value)
            result["rom_sha256"] = "e" * 64
            with self.assertRaisesRegex(ValueError, "ROM hash"):
                validate_attempt_result(result, contract=value, attempt_id="attempt")

    def test_result_validation_preserves_optional_preview_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            value = contract(Path(temporary))
            result = successful_result(value)
            result["preview"] = {
                "status": "succeeded",
                "public_url": "https://preview.example/eval.mp4",
                "lane_count": 2,
            }

            validated = validate_attempt_result(result, contract=value, attempt_id="attempt")

        self.assertEqual(validated["preview"], result["preview"])

    def test_first_accepted_result_wins_promotion_and_is_the_only_stop_creator(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        promotion_available = True

        def execute(statement, _params=None):
            nonlocal promotion_available
            if "UPDATE eval_runs" in statement and "promoted_eval_job_id IS NULL" in statement:
                cursor.rowcount = int(promotion_available)
                promotion_available = False
            else:
                cursor.rowcount = 1

        cursor.execute.side_effect = execute
        base_attempt = {
            "id": 101,
            "attempt_id": "attempt-101",
            "eval_job_id": 201,
            "train_job_id": 7,
            "ledger_id": 11,
            "job_key": "job-201",
            "execution_key": "execution",
            "contract_json": {},
            "decision_rules_json": [],
            "purpose": "acceptance",
            "stage_name": "acceptance",
            "stage_index": 0,
            "candidate_stop": True,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_step": 500000,
            "checkpoint_uri": "s3://bucket/model.zip",
            "result_uri": "s3://bucket/result.json",
            "train_config": {"telemetry_transport": "neon_mailbox_v1"},
        }
        validated = {
            "metrics": {"eval/full/outcome/success/rate/min": 1.0},
            "claimed_aggregates": {
                "episodes_planned": 100,
                "episodes_completed": 100,
                "failure_count": 0,
            },
            "duration_seconds": 30.0,
            "verdict": "accepted",
            "episode_results": [],
        }
        second_attempt = {
            **base_attempt,
            "id": 102,
            "attempt_id": "attempt-102",
            "eval_job_id": 202,
            "ledger_id": 12,
            "job_key": "job-202",
            "checkpoint_sha256": "b" * 64,
            "checkpoint_step": 750000,
            "checkpoint_uri": "s3://bucket/model-2.zip",
            "result_uri": "s3://bucket/result-2.json",
        }

        with (
            mock.patch(
                "rlab.modal_eval_orchestrator.validate_attempt_result",
                return_value=validated,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator._stage_metrics",
                return_value={"eval/acceptance/pass": 1.0},
            ),
            mock.patch(
                "rlab.telemetry_mailbox.enqueue_projection_payload",
                return_value=1,
            ) as enqueue,
        ):
            accept_attempt_result(
                conn,
                mock.MagicMock(),
                attempt=base_attempt,
                result={},
            )
            accept_attempt_result(
                conn,
                mock.MagicMock(),
                attempt=second_attempt,
                result={},
            )

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        stop_inserts = [
            statement for statement in statements if "INSERT INTO attempt_commands" in statement
        ]
        superseding_updates = [
            statement
            for statement in statements
            if "superseded by accepted checkpoint" in statement
        ]
        self.assertEqual(len(stop_inserts), 1)
        self.assertEqual(len(superseding_updates), 1)
        self.assertIn("ON CONFLICT (command_id) DO NOTHING", stop_inserts[0])
        self.assertEqual(enqueue.call_count, 2)
        self.assertTrue(enqueue.call_args_list[0].kwargs["payload"]["canonical_promotion"])
        self.assertFalse(enqueue.call_args_list[1].kwargs["payload"]["canonical_promotion"])

    def test_accepted_screen_decision_propagates_preview_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = contract(root)
            result = successful_result(value)
            result["preview"] = {
                "status": "succeeded",
                "public_url": "https://preview.example/eval.mp4",
                "lane_count": 2,
            }
            attempt = {
                "id": 1,
                "eval_job_id": 2,
                "attempt_id": "attempt",
                "contract_json": value,
                "decision_rules_json": [
                    {
                        "metric": "eval/full/episode/return/mean",
                        "operator": ">=",
                        "threshold": 2.0,
                    }
                ],
                "purpose": "screen",
                "checkpoint_step": 500000,
                "checkpoint_uri": "s3://bucket/model.zip",
                "job_key": "job-key",
                "execution_key": execution_key(value),
                "train_job_id": 9,
                "ledger_id": 3,
                "stage_name": "screen",
                "stage_index": 0,
                "candidate_stop": False,
                "result_uri": "s3://bucket/result.json",
            }
            store = ObjectStore((root / "objects").resolve().as_uri())

            accept_attempt_result(FakeConnection(), store, attempt=attempt, result=result)
            decision = store.get_json("eval-decisions/9/job-key.json")

        self.assertEqual(decision["preview"], result["preview"])
        self.assertFalse(decision["passed"])

    def test_modal_app_name_requires_immutable_digest(self) -> None:
        self.assertEqual(
            modal_app_name("rlab-eval", "docker:repo/image@sha256:" + "a" * 64),
            "rlab-eval-aaaaaaaaaaaa",
        )
        with self.assertRaisesRegex(ValueError, "immutable"):
            modal_app_name("rlab-eval", "docker:repo/image:latest")


class ModalEvalRecoveryTests(unittest.TestCase):
    def test_drain_and_exit_polls_until_completion_marker_is_written(self) -> None:
        store = mock.MagicMock()
        store.pending_artifact_uploads.return_value = []
        with (
            mock.patch.object(
                checkpoint_coordinator,
                "materialized_train_args",
                return_value=SimpleNamespace(
                    checkpoint_bucket_uri="file:///objects",
                    wandb_artifact_storage_uri="file:///objects",
                ),
            ),
            mock.patch.object(checkpoint_coordinator, "MetricStore", return_value=store),
            mock.patch.object(checkpoint_coordinator, "ObjectStore"),
            mock.patch.object(checkpoint_coordinator, "reconcile_orphan_models", return_value=0),
            mock.patch.object(checkpoint_coordinator, "import_decisions", return_value=0),
            mock.patch.object(
                checkpoint_coordinator,
                "write_complete_marker",
                side_effect=[False, True],
            ) as marker,
            mock.patch.object(checkpoint_coordinator.time, "sleep") as sleep,
        ):
            result = checkpoint_coordinator.main(
                [
                    "--run-dir",
                    "/run",
                    "--train-config-json",
                    "/run/train.json",
                    "--drain-and-exit",
                ]
            )

        self.assertEqual(result, 0)
        store.reset_interrupted_artifact_uploads.assert_called_once_with()
        store.requeue_uploaded_artifacts_for_recovery.assert_not_called()
        self.assertEqual(marker.call_count, 2)
        sleep.assert_called_once()

    def test_recover_rejects_nonterminal_or_nonrecovery_state_without_remote_calls(self) -> None:
        cases = (
            (
                {"status": "running", "eval_run_status": "awaiting_artifact_recovery"},
                "not terminal",
            ),
            (
                {
                    "status": "succeeded",
                    "eval_run_status": "active",
                    "missing_artifact_receipts": 0,
                },
                "not awaiting",
            ),
        )
        for state, error in cases:
            with (
                self.subTest(state=state),
                mock.patch.object(
                    modal_eval_cli,
                    "_conn",
                    return_value=FakeConnection(
                        row={
                            **state,
                            "id": 13,
                            "launch_id": "train-13",
                            "machine": "beast-3",
                            "runtime_image_ref": "docker:example.invalid/rlab@sha256:" + "b" * 64,
                        }
                    ),
                ),
                mock.patch("rlab.docker_host.run_checkpoint_coordinator_container") as recover,
            ):
                with self.assertRaisesRegex(ValueError, error):
                    modal_eval_cli.cmd_recover(SimpleNamespace(train_job_id=13))
                recover.assert_not_called()

    def test_recover_uses_fresh_checkpoint_only_attempt(self) -> None:
        runtime_ref = "docker:example.invalid/rlab@sha256:" + "b" * 64
        conn = FakeConnection(
            results=[
                {
                    "row": {
                        "id": 13,
                        "status": "succeeded",
                        "eval_run_status": "awaiting_artifact_recovery",
                        "launch_id": "train-13",
                        "output_uri": "/host/outputs/train-13",
                        "machine": "beast-3",
                        "runtime_image_ref": runtime_ref,
                        "run_name": "run-13",
                    }
                },
                {"row": None},
                {"row": None},
                {"row": None},
                {"row": None},
                {"row": None},
                {"row": {"train_job_id": 13}},
            ]
        )
        machine = SimpleNamespace(
            paths=SimpleNamespace(
                container_outputs_dir="/output",
                env_file="/host/.env.runner",
                outputs_dir="/host/outputs",
            )
        )
        recovery_host = mock.MagicMock()
        recovery_host.write_attempt_env.return_value = "/host/recovery.env"
        with (
            mock.patch.object(modal_eval_cli, "_conn", return_value=conn),
            mock.patch("rlab.machines.load_machine_registry", return_value=object()),
            mock.patch("rlab.machines.resolve_machine", return_value=machine),
            mock.patch("rlab.docker_host.DockerRunnerHost", return_value=recovery_host),
            mock.patch(
                "rlab.telemetry_mailbox.issue_worker_attempt_token",
                return_value="recovery-token",
            ),
            mock.patch(
                "rlab.fleet.load_mailbox_runner_env",
                return_value={
                    "WORKER_MAILBOX_DATABASE_URL": "postgresql://mailbox.invalid/rlab",
                    "AWS_ACCESS_KEY_ID": "access-key",
                    "AWS_SECRET_ACCESS_KEY": "secret-key",
                    "AWS_S3_ENDPOINT_URL": "https://objects.invalid",
                    "AWS_REGION": "auto",
                    "CHECKPOINT_BUCKET_URI": "s3://checkpoints",
                },
            ),
            mock.patch("rlab.workspace_gc.workspace_protocol_mode", return_value="dormant"),
            mock.patch("rlab.docker_host.run_checkpoint_coordinator_container") as recover,
            mock.patch.object(modal_eval_cli, "_kick"),
        ):
            result = modal_eval_cli.cmd_recover(SimpleNamespace(train_job_id=13))

        self.assertEqual(result, 0)
        self.assertEqual(recover.call_args.kwargs["launch_id"], "train-13")
        self.assertEqual(recover.call_args.kwargs["run_name"], "run-13")
        self.assertEqual(recover.call_args.kwargs["runtime_image_ref"], runtime_ref)
        self.assertEqual(recover.call_args.kwargs["attempt_env_path"], "/host/recovery.env")
        recovery_env = recovery_host.write_attempt_env.call_args.args[1]
        self.assertEqual(
            recovery_env["WORKER_MAILBOX_DATABASE_URL"],
            "postgresql://mailbox.invalid/rlab",
        )
        self.assertEqual(recovery_env["RLAB_WORKER_ATTEMPT_TOKEN"], "recovery-token")
        self.assertTrue(recovery_env["RLAB_WORKER_ATTEMPT_ID"].startswith("checkpoint-recovery-13-"))
        recovery_host.remove_attempt_env.assert_called_once()
        self.assertTrue(any("UPDATE eval_runs" in sql for sql in conn.cursor_obj.executed_sqls))

    def test_failed_recovery_preserves_awaiting_state(self) -> None:
        conn = FakeConnection(
            results=[
                {
                    "row": {
                        "id": 13,
                        "status": "failed",
                        "eval_run_status": "awaiting_artifact_recovery",
                        "launch_id": "train-13",
                        "output_uri": "/host/outputs/train-13",
                        "machine": "beast-3",
                        "runtime_image_ref": (
                            "docker:example.invalid/rlab@sha256:" + "b" * 64
                        ),
                        "run_name": "run-13",
                    }
                },
                {"row": None},
                {"row": None},
                {"row": None},
                {"row": None},
                {"row": None},
            ]
        )
        machine = SimpleNamespace(
            paths=SimpleNamespace(
                container_outputs_dir="/output",
                env_file="/host/.env.runner",
                outputs_dir="/host/outputs",
            )
        )
        recovery_host = mock.MagicMock()
        recovery_host.write_attempt_env.return_value = "/host/recovery.env"
        with (
            mock.patch.object(modal_eval_cli, "_conn", return_value=conn),
            mock.patch("rlab.machines.load_machine_registry", return_value=object()),
            mock.patch("rlab.machines.resolve_machine", return_value=machine),
            mock.patch("rlab.docker_host.DockerRunnerHost", return_value=recovery_host),
            mock.patch(
                "rlab.telemetry_mailbox.issue_worker_attempt_token",
                return_value="recovery-token",
            ),
            mock.patch("rlab.workspace_gc.workspace_protocol_mode", return_value="dormant"),
            mock.patch(
                "rlab.docker_host.run_checkpoint_coordinator_container",
                side_effect=RuntimeError("recovery failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "recovery failed"):
                modal_eval_cli.cmd_recover(SimpleNamespace(train_job_id=13))

        self.assertFalse(any("UPDATE eval_runs" in sql for sql in conn.cursor_obj.executed_sqls))

    def test_abandon_closes_nonterminal_eval_after_failed_training(self) -> None:
        conn = FakeConnection(
            results=[
                {
                    "row": {
                        "train_status": "failed",
                        "eval_run_status": "active",
                        "active_attempts": 0,
                    }
                },
                {"rowcount": 0},
                {"rowcount": 1},
                {"rowcount": 1},
            ]
        )
        with (
            mock.patch.object(modal_eval_cli, "_conn", return_value=conn),
            mock.patch.object(modal_eval_cli, "_kick") as kick,
        ):
            self.assertEqual(modal_eval_cli.cmd_abandon(SimpleNamespace(train_job_id=27)), 0)

        statements = conn.cursor_obj.executed_sqls
        self.assertTrue(any("UPDATE eval_jobs" in sql for sql in statements))
        self.assertTrue(any("UPDATE eval_runs" in sql for sql in statements))
        kick.assert_called_once_with("modal_eval_abandon", entity_kind="train", entity_id=27)

    def test_abandon_rejects_active_modal_attempts(self) -> None:
        conn = FakeConnection(
            row={
                "train_status": "failed",
                "eval_run_status": "active",
                "active_attempts": 1,
            }
        )
        with (
            mock.patch.object(modal_eval_cli, "_conn", return_value=conn),
            mock.patch.object(modal_eval_cli, "_kick") as kick,
            self.assertRaisesRegex(ValueError, "active Modal attempts"),
        ):
            modal_eval_cli.cmd_abandon(SimpleNamespace(train_job_id=27))
        kick.assert_not_called()


class ModalEvalSchedulingTests(unittest.TestCase):
    def test_post_train_promotion_does_not_reopen_decided_runs(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []

        self.assertEqual(enqueue_post_train_promotions(conn), 0)

        statement = cursor.execute.call_args.args[0]
        self.assertIn("r.outcome IS NULL", statement)

    def test_completed_attempt_polling_precedes_and_survives_ingestion_failure(self) -> None:
        order: list[str] = []
        conn = mock.MagicMock()
        config = SimpleNamespace(enabled=True, hard_max_active=20)

        def poll(*_args, **_kwargs):
            order.append("poll")
            return 1

        def mailbox(*_args, **_kwargs):
            order.append("mailbox")
            raise RuntimeError("announcement backlog unavailable")

        def legacy(*_args, **_kwargs):
            order.append("legacy")
            return 0

        patchers = (
            mock.patch(
                "rlab.modal_eval_orchestrator.load_modal_eval_config",
                return_value=config,
            ),
            mock.patch("rlab.modal_eval_orchestrator.database_url", return_value="postgres://db"),
            mock.patch("rlab.modal_eval_orchestrator.connect", return_value=conn),
            mock.patch("rlab.modal_eval_orchestrator._try_lock", return_value=True),
            mock.patch("rlab.modal_eval_orchestrator._unlock"),
            mock.patch("rlab.modal_eval_orchestrator.ensure_eval_runs", return_value=0),
            mock.patch(
                "rlab.modal_eval_orchestrator.cancel_requested_attempts",
                return_value=0,
            ),
            mock.patch("rlab.modal_eval_orchestrator.poll_attempts", side_effect=poll),
            mock.patch(
                "rlab.modal_eval_orchestrator.ingest_mailbox_announcements",
                side_effect=mailbox,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.ingest_announcements",
                side_effect=legacy,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.reconcile_canceled_eval_state",
                return_value={"jobs": 0, "runs": 0, "workers": 0},
            ),
            mock.patch("rlab.modal_eval_orchestrator.publish_skipped_decisions", return_value=0),
            mock.patch(
                "rlab.modal_eval_orchestrator.enqueue_missing_mailbox_projections",
                return_value=0,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.enqueue_post_train_promotions",
                return_value=0,
            ),
            mock.patch("rlab.modal_eval_orchestrator.dispatch_pending", return_value=0),
            mock.patch("rlab.modal_eval_orchestrator.reconcile_promotions", return_value=0),
            mock.patch("rlab.modal_eval_orchestrator.project_eval_results", return_value=0),
            mock.patch(
                "rlab.modal_eval_orchestrator.project_artifact_references",
                return_value=0,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.reconcile_published_mailbox_projections",
                return_value=0,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.reconcile_learner_stop_observation",
                return_value=0,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.reconcile_stop_delivery_slo",
                return_value=0,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.reconcile_definitive_non_acceptance",
                return_value=0,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.reconcile_eval_run_failures",
                return_value=0,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.terminalize_failed_eval_runs",
                return_value=0,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.reconcile_publication_finishing",
                return_value=0,
            ),
            mock.patch(
                "rlab.modal_eval_orchestrator.terminalize_artifact_only_runs",
                return_value=0,
            ),
            mock.patch("rlab.modal_eval_orchestrator.terminalize_runs", return_value=0),
            mock.patch(
                "rlab.modal_eval_orchestrator.run_modal_app_cleanup",
                return_value={"status": "ok"},
            ),
        )
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            result = run_service_eval_pass(
                repo_root=Path.cwd(),
                deadline_monotonic=time.monotonic() + 60,
                invoker=mock.MagicMock(),
                store=mock.MagicMock(),
            )

        self.assertEqual(order, ["poll", "mailbox", "legacy"])
        self.assertEqual(result["polled"], 1)
        self.assertEqual(result["mailbox_ingested"], 0)
        self.assertEqual(
            result["ingestion_errors"],
            ["mailbox:RuntimeError:announcement backlog unavailable"],
        )

    def test_mailbox_ingestion_advances_ordering_cursor_within_batch(self) -> None:
        events = [
            {
                "id": ledger_id,
                "event_type": "checkpoint_tombstone",
                "payload_json": {
                    "train_job_id": 44,
                    "ledger_id": ledger_id,
                    "kind": "tombstone",
                },
                "next_announcement_id": 1,
                "contract_json": {},
            }
            for ledger_id in (1, 2, 3)
        ]
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = events
        cursor.fetchone.side_effect = [
            {"next_announcement_id": 1},
            {"announcement_sha256": "a" * 64},
            {"next_announcement_id": 2},
            {"announcement_sha256": "b" * 64},
            {"next_announcement_id": 3},
            {"announcement_sha256": "c" * 64},
        ]
        cursor.rowcount = 1

        count = ingest_mailbox_announcements(conn, mock.MagicMock())

        self.assertEqual(count, 3)
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        ledger_inserts = [
            index
            for index, statement in enumerate(statements)
            if "INSERT INTO artifact_announcement_ledger" in statement
        ]
        delete_calls = [
            call
            for call in cursor.execute.call_args_list
            if "DELETE FROM attempt_events" in call.args[0]
        ]
        delete_indexes = [
            index
            for index, statement in enumerate(statements)
            if "DELETE FROM attempt_events" in statement
        ]
        self.assertEqual(len(ledger_inserts), 3)
        self.assertEqual(len(delete_calls), 3)
        self.assertTrue(
            all(insert < delete for insert, delete in zip(ledger_inserts, delete_indexes))
        )

    def test_mailbox_ingestion_interleaves_runs_to_prevent_checkpoint_starvation(
        self,
    ) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []

        self.assertEqual(ingest_mailbox_announcements(conn, mock.MagicMock()), 0)

        selection = cursor.execute.call_args_list[0].args[0]
        self.assertIn("row_number() OVER", selection)
        self.assertIn("PARTITION BY r.train_job_id", selection)
        self.assertIn("ORDER BY run_event_ordinal, id", selection)

    def test_none_backend_ingests_checkpoint_without_creating_eval_job(self) -> None:
        event = {
            "id": 1,
            "event_type": "checkpoint_ready",
            "payload_json": {
                "train_job_id": 44,
                "ledger_id": 1,
                "kind": "checkpoint",
            },
            "authoritative_train_job_id": 44,
            "next_announcement_id": 1,
            "contract_json": {"checkpoint_eval_backend": "none"},
        }
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = [event]
        cursor.rowcount = 1

        with (
            mock.patch(
                "rlab.modal_eval_orchestrator.validate_announcement",
                return_value=event["payload_json"],
            ),
            mock.patch("rlab.modal_eval_orchestrator._verify_checkpoint_artifacts"),
            mock.patch("rlab.modal_eval_orchestrator._persist_artifact_announcement") as persist,
            mock.patch("rlab.modal_eval_orchestrator._insert_eval_job") as insert_eval,
        ):
            count = ingest_mailbox_announcements(conn, mock.MagicMock())

        self.assertEqual(count, 1)
        persist.assert_called_once()
        insert_eval.assert_not_called()

    def test_mailbox_ingestion_defers_malformed_record_and_continues(self) -> None:
        events = [
            {
                "id": 1,
                "event_type": "checkpoint_ready",
                "payload_json": "malformed",
                "authoritative_train_job_id": 44,
                "next_announcement_id": 1,
                "contract_json": {},
            },
            {
                "id": 2,
                "event_type": "checkpoint_tombstone",
                "payload_json": {
                    "train_job_id": 44,
                    "ledger_id": 1,
                    "kind": "tombstone",
                },
                "authoritative_train_job_id": 44,
                "next_announcement_id": 1,
                "contract_json": {},
            },
        ]
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = events
        cursor.rowcount = 1

        with mock.patch(
            "rlab.modal_eval_orchestrator._defer_attempt_event", return_value=1
        ) as defer:
            count = ingest_mailbox_announcements(conn, mock.MagicMock())

        self.assertEqual(count, 1)
        defer.assert_called_once()

    def test_mailbox_ingestion_time_slices_a_large_backlog(self) -> None:
        events = [
            {
                "id": ledger_id,
                "event_type": "checkpoint_tombstone",
                "payload_json": {
                    "train_job_id": 44,
                    "ledger_id": ledger_id,
                    "kind": "tombstone",
                },
                "next_announcement_id": 1,
                "contract_json": {},
            }
            for ledger_id in range(1, 51)
        ]
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = events
        cursor.rowcount = 1

        with mock.patch(
            "rlab.modal_eval_orchestrator.time.monotonic",
            side_effect=[0.0, 2.0],
        ):
            count = ingest_mailbox_announcements(
                conn,
                mock.MagicMock(),
                deadline_monotonic=1.0,
            )

        self.assertEqual(count, 1)

    def test_plain_runtime_error_from_modal_is_a_terminal_call_failure(self) -> None:
        call = mock.MagicMock()
        call.get.side_effect = RuntimeError("remote worker failed")
        with mock.patch("modal.FunctionCall.from_id", return_value=call):
            state, detail = DefaultModalInvoker().poll("fc-test")
        self.assertEqual(state, "failed")
        self.assertIn("remote worker failed", str(detail))

    def test_output_expired_is_ambiguous_until_durable_result_arrives(self) -> None:
        import modal

        call = mock.MagicMock()
        call.get.side_effect = modal.exception.OutputExpiredError()
        with mock.patch("modal.FunctionCall.from_id", return_value=call):
            state, detail = DefaultModalInvoker().poll("fc-test")

        self.assertEqual(state, "pending")
        self.assertIsNone(detail)

    def test_only_transient_failures_are_retryable(self) -> None:
        self.assertTrue(deterministic_eval_failure("checkpoint hash mismatch"))
        self.assertTrue(deterministic_eval_failure("environment contract is invalid"))
        self.assertFalse(deterministic_eval_failure("provider connection reset"))

    def test_available_slots_never_exceed_hard_cap_and_count_unknown_calls(self) -> None:
        self.assertEqual(
            available_eval_slots(active_calls=9, hard_cap=10),
            1,
        )
        self.assertEqual(
            available_eval_slots(active_calls=10, hard_cap=10),
            0,
        )

    def test_modal_spawn_before_call_id_persistence_does_not_dispatch_twice(self) -> None:
        job = {
            "id": 1,
            "train_job_id": 7,
            "stage_index": 0,
            "purpose": "acceptance",
            "retry_round": 0,
            "execution_key": "execution-key",
            "checkpoint_uri": "s3://bucket/model.zip",
            "metadata_uri": "s3://bucket/metadata.json",
            "contract_json": {
                "runtime_image_ref": "docker:example.invalid/rlab@sha256:" + "a" * 64,
            },
            "source_announcement_json": {
                "eval": {"asset": None},
                "model_document_sha256": "b" * 64,
                "recipe_uri": "s3://bucket/recipe.json",
            },
        }

        class Cursor:
            def __init__(self) -> None:
                self.row = None
                self.rows = []
                self.rowcount = 0
                self.attempt_persisted = False
                self.job_status = "pending"

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def execute(self, statement, _params=None):
                compact = " ".join(statement.split())
                self.row = None
                self.rows = []
                if compact.startswith("SELECT * FROM eval_backend_state"):
                    self.row = {
                        "drained": False,
                        "round_robin_after_train_job_id": 0,
                    }
                elif "FROM eval_attempts WHERE status IN" in compact:
                    self.row = {"count": int(self.attempt_persisted)}
                elif compact.startswith("SELECT j.* FROM eval_jobs"):
                    self.rows = [job] if self.job_status == "pending" else []
                elif "SELECT COALESCE(sum(CASE" in compact:
                    self.row = {"total": 0.0}
                elif compact.startswith("SELECT count(*) AS count FROM eval_attempts"):
                    self.row = {"count": 0}
                elif compact.startswith("INSERT INTO eval_attempts"):
                    self.attempt_persisted = True
                    self.row = {"id": 101}
                    self.rowcount = 1
                elif "UPDATE eval_jobs SET status = 'dispatching'" in compact:
                    self.job_status = "dispatching"
                    self.rowcount = 1
                elif "UPDATE eval_attempts SET status = 'submitted'" in compact:
                    raise RuntimeError("database disconnected after Modal spawn")
                else:
                    self.rowcount = 1

            def fetchone(self):
                return self.row

            def fetchall(self):
                return self.rows

        class Connection:
            def __init__(self) -> None:
                self.cursor_obj = Cursor()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def cursor(self):
                return self.cursor_obj

        class Invoker:
            def __init__(self) -> None:
                self.calls = 0

            def spawn(self, *_args):
                self.calls += 1
                return "modal-call-1"

        conn = Connection()
        invoker = Invoker()
        store = mock.MagicMock()
        store.uri.return_value = "s3://bucket/result.json"
        store.presign_get.return_value = "https://get.invalid/object"
        store.presign_put.return_value = "https://put.invalid/object"

        with mock.patch("rlab.modal_eval_orchestrator._reuse_result", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "after Modal spawn"):
                dispatch_pending(
                    conn,
                    store,
                    invoker,
                    load_modal_eval_config(),
                    deadline_monotonic=time.monotonic() + 60,
                )
            self.assertEqual(
                dispatch_pending(
                    conn,
                    store,
                    invoker,
                    load_modal_eval_config(),
                    deadline_monotonic=time.monotonic() + 60,
                ),
                0,
            )

        self.assertEqual(invoker.calls, 1)
        self.assertTrue(conn.cursor_obj.attempt_persisted)
        self.assertEqual(conn.cursor_obj.job_status, "dispatching")

    def test_budget_reservation_is_fail_closed_at_both_limits(self) -> None:
        config = load_modal_eval_config(Path("experiments/modal_eval.yaml"))
        self.assertTrue(
            budget_allows(
                run_cost_usd=4.9,
                rolling_cost_usd=19.9,
                reserved_cost_usd=0.1,
                config=config,
            )
        )
        self.assertFalse(
            budget_allows(
                run_cost_usd=4.91,
                rolling_cost_usd=0.0,
                reserved_cost_usd=0.1,
                config=config,
            )
        )

    def test_round_robin_reserves_a_slot_for_promotion(self) -> None:
        jobs = [
            {"id": 1, "train_job_id": 1, "stage_index": 1, "purpose": "confirm"},
            {"id": 2, "train_job_id": 1, "stage_index": 1, "purpose": "confirm"},
            {"id": 3, "train_job_id": 2, "stage_index": 0, "purpose": "promotion"},
            {"id": 4, "train_job_id": 3, "stage_index": 0, "purpose": "screen"},
        ]
        ordered = round_robin_jobs(jobs, slots=2)
        self.assertEqual(ordered[0]["purpose"], "promotion")
        self.assertEqual([job["train_job_id"] for job in ordered[:3]], [2, 1, 3])
        rotated = round_robin_jobs(
            [job for job in jobs if job["purpose"] != "promotion"],
            slots=2,
            after_train_job_id=1,
        )
        self.assertEqual(rotated[0]["train_job_id"], 3)

    def test_promotion_ranking_uses_accepted_evidence_not_wandb(self) -> None:
        rank = [
            "max(eval/full/outcome/success/rate/min)",
            "max(eval/full/episode/return/mean)",
        ]
        weaker = {
            "id": 1,
            "checkpoint_step": 100,
            "train_config": {"selection_rank": rank},
            "decision_json": {
                "raw_metrics": {
                    "eval/full/outcome/success/rate/min": 0.9,
                    "eval/full/episode/return/mean": 100.0,
                }
            },
        }
        stronger = {
            "id": 2,
            "checkpoint_step": 200,
            "train_config": {"selection_rank": rank},
            "decision_json": {
                "raw_metrics": {
                    "eval/full/outcome/success/rate/min": 1.0,
                    "eval/full/episode/return/mean": 1.0,
                }
            },
        }
        self.assertGreater(promotion_candidate_key(stronger), promotion_candidate_key(weaker))

    def test_promotion_ranking_reads_known_historical_metric_contract(self) -> None:
        historical = {
            "id": 1,
            "checkpoint_step": 100,
            "train_config": {
                "selection_rank": [
                    "max(eval/full/info/level_complete/rate/min)",
                    "max(eval/full/reward/mean)",
                ]
            },
            "decision_json": {
                "raw_metrics": {
                    "eval/full/info/level_complete/rate/min": 1.0,
                    "eval/full/reward/mean": 20.0,
                }
            },
        }

        self.assertEqual(promotion_candidate_key(historical)[0], (1.0, 20.0))

    def test_promotion_reconciliation_isolates_invalid_run_and_continues(self) -> None:
        invalid = {
            "id": 1,
            "train_job_id": 10,
            "accepted_attempt_id": 11,
            "checkpoint_step": 100,
            "checkpoint_sha256": "a" * 64,
            "checkpoint_uri": "s3://bucket/old.zip",
            "train_config": {"selection_rank": ["max(unknown/metric)"]},
            "decision_json": {"result_uri": "s3://bucket/old.json", "raw_metrics": {}},
        }
        valid = {
            "id": 2,
            "train_job_id": 17,
            "accepted_attempt_id": 12,
            "checkpoint_step": 500,
            "checkpoint_sha256": "b" * 64,
            "checkpoint_uri": "s3://bucket/new.zip",
            "train_config": {"selection_rank": ["max(eval/full/outcome/success/rate/min)"]},
            "decision_json": {
                "result_uri": "s3://bucket/new.json",
                "raw_metrics": {"eval/full/outcome/success/rate/min": 1.0},
            },
        }
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.side_effect = [
            [{"train_job_id": 10}, {"train_job_id": 17}],
            [invalid],
            [valid],
        ]
        cursor.fetchone.side_effect = [
            {"status": "finalizing", "promotion_revision": 0},
            {"status": "finalizing", "promotion_revision": 0},
        ]
        cursor.rowcount = 1

        self.assertEqual(reconcile_promotions(conn), 1)

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertIn("r.status = 'finalizing'", statements[0])
        self.assertTrue(any("status = 'failed'" in statement for statement in statements))
        self.assertTrue(any("promoted_eval_job_id" in statement for statement in statements))

    def test_projection_queries_skip_backoff_and_exhausted_rows(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = []

        self.assertEqual(
            project_eval_results(
                conn,
                repo_root=Path.cwd(),
                deadline_monotonic=time.monotonic() + 60,
            ),
            0,
        )

        statement = cursor.execute.call_args.args[0]
        self.assertIn("projection_attempts <", statement)
        self.assertIn("projection_next_retry_at <= now()", statement)
        self.assertIn("r.status = 'finalizing'", statement)

    def test_projection_enqueues_all_ready_mailbox_promotions(self) -> None:
        jobs = [
            {
                "id": job_id,
                "train_job_id": 17,
                "train_config": {"telemetry_transport": "neon_mailbox_v1"},
                "decision_json": {"metrics": {}},
                "purpose": "promotion",
                "checkpoint_uri": f"s3://bucket/{job_id}.zip",
                "checkpoint_sha256": str(job_id) * 64,
                "checkpoint_step": job_id * 100,
                "canonical_promotion": job_id == 2,
            }
            for job_id in (1, 2, 3)
        ]
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchall.return_value = jobs

        with mock.patch(
            "rlab.telemetry_mailbox.enqueue_projection_payload", return_value=1
        ) as enqueue:
            count = project_eval_results(
                conn,
                repo_root=Path.cwd(),
                deadline_monotonic=time.monotonic() + 60,
            )

        self.assertEqual(count, 3)
        self.assertEqual(enqueue.call_count, 3)
        self.assertEqual(cursor.execute.call_args.kwargs.get("limit"), None)
        self.assertEqual(cursor.execute.call_args.args[1]["limit"], 100)

    def test_finalization_waits_for_wandb_projection(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 0

        terminalize_runs(conn)

        statement = cursor.execute.call_args.args[0]
        self.assertIn("artifact_announcement_ledger", statement)
        self.assertIn("artifact_publication_receipts", statement)
        self.assertIn("receipt.role = 'availability'", statement)
        self.assertIn("receipt.role = 'promotion'", statement)
        self.assertIn("r.promotion_revision", statement)
        self.assertIn("GREATEST(r.next_announcement_id - 1, 0)", statement)
        self.assertIn("j.projected_at IS NULL", statement)
        self.assertIn("t.process_exited_at IS NOT NULL", statement)
        self.assertIn("t.live_publication_status IN ('complete', 'disabled')", statement)
        self.assertIn("r.outcome = 'unknown'", statement)
        self.assertIn("t.learner_stop_observed_at IS NULL", statement)
        self.assertIn("r.stop_delivery_slo_met IS NOT TRUE", statement)
        self.assertIn("r.acceptance_committed_at IS NULL", statement)
        self.assertIn(
            "r.acceptance_committed_at < t.process_exited_at",
            statement,
        )
        self.assertIn("THEN 'finalization_failed'", statement)

    def test_legacy_finalization_does_not_require_v2_schema_during_rollout(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {"telemetry_evidence_scopes": None}
        cursor.rowcount = 1

        self.assertEqual(terminalize_runs(conn), 1)

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(any("to_regclass('telemetry_evidence_scopes')" in sql for sql in statements))
        self.assertFalse(any("JOIN telemetry_evidence_scopes" in sql for sql in statements))
        self.assertTrue(any("t.telemetry_protocol_version = 1" in sql for sql in statements))

    def test_v2_training_only_finalization_keeps_outcome_null_and_requires_exact_publication(
        self,
    ) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {"telemetry_evidence_scopes": "telemetry_evidence_scopes"}
        cursor.rowcount = 0

        terminalize_runs(conn)

        statements = [call.args[0] for call in cursor.execute.call_args_list]
        statement = next(sql for sql in statements if "t.telemetry_protocol_version=2" in sql)
        self.assertIn("checkpoint_eval_backend' = 'none'", statement)
        self.assertIn("r.outcome IS NULL", statement)
        self.assertIn("t.live_publication_status IN ('complete','disabled')", statement)
        self.assertIn("FROM telemetry_integrity integrity", statement)
        self.assertIn("integrity.exact=TRUE", statement)
        self.assertIn("integrity.classification='intact_with_proof'", statement)
        self.assertIn("launch.workspace_layout_version IS NOT NULL", statement)
        self.assertIn("FROM artifact_durability_receipts receipt", statement)
        self.assertIn("FROM artifact_publication_receipts receipt", statement)
        self.assertIn("receipt.role='availability'", statement)
        self.assertIn("r.outcome IS DISTINCT FROM 'accepted'", statement)

    def test_artifact_only_finalization_waits_for_complete_durable_stream(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1

        self.assertEqual(terminalize_artifact_only_runs(conn), 1)

        statement = cursor.execute.call_args.args[0]
        self.assertIn("checkpoint_eval_backend', 'local') <> 'modal'", statement)
        self.assertIn("t.process_exited_at IS NOT NULL", statement)
        self.assertIn("t.cancel_requested", statement)
        self.assertIn("disposition = 'tombstone'", statement)
        self.assertIn("'finalization_failed'", statement)
        self.assertIn("r.complete_announcement_seen = TRUE", statement)
        self.assertIn("GREATEST(r.next_announcement_id - 1, 0)", statement)
        self.assertIn("receipt.role = 'availability'", statement)
        self.assertIn("s.final_sequence IS NULL", statement)
        self.assertIn("t.live_publication_status IN ('complete', 'disabled')", statement)

    def test_neon_jobs_get_ordered_checkpoint_stream_state_for_every_backend(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 3

        self.assertEqual(ensure_eval_runs(conn), 3)

        statement = cursor.execute.call_args.args[0]
        self.assertIn("telemetry_transport = 'neon_mailbox_v1'", statement)
        self.assertIn("checkpoint_eval_backend' = 'modal'", statement)
        self.assertIn("status <> 'canceled'", statement)

    def test_terminal_reducers_distinguish_rejection_unknown_and_publication_finishing(
        self,
    ) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.rowcount = 1

        self.assertEqual(reconcile_definitive_non_acceptance(conn), 1)
        non_acceptance = cursor.execute.call_args.args[0]
        self.assertIn("outcome = 'not_accepted'", non_acceptance)
        self.assertIn("t.status = 'finalizing'", non_acceptance)
        self.assertIn("'blocked_budget', 'failed'", non_acceptance)
        self.assertIn("decision_json->>'verdict'", non_acceptance)
        self.assertIn("j.projected_at IS NULL", non_acceptance)
        self.assertIn("j.status = 'succeeded'", non_acceptance)

        self.assertEqual(reconcile_eval_run_failures(conn), 1)
        unknown = cursor.execute.call_args.args[0]
        self.assertIn("outcome = 'unknown'", unknown)
        self.assertIn("checkpoint_eval_backend', 'local') = 'modal'", unknown)
        self.assertIn("t.status = 'finalizing'", unknown)
        self.assertIn("required evaluation exhausted retries", unknown)
        self.assertIn("checkpoint stream closed without evaluable evidence", unknown)

        self.assertEqual(terminalize_failed_eval_runs(conn), 1)
        failed = cursor.execute.call_args.args[0]
        self.assertIn("status = 'finalization_failed'", failed)
        self.assertIn("r.status = 'failed'", failed)
        self.assertIn("a.status IN ('dispatching', 'submitted')", failed)
        self.assertIn("w.status IN ('launching', 'running')", failed)

        self.assertEqual(reconcile_stop_delivery_slo(conn), 1)
        stop_slo = cursor.execute.call_args.args[0]
        self.assertIn("acknowledged_at", stop_slo)
        self.assertIn("interval '5 seconds'", stop_slo)

        self.assertEqual(reconcile_publication_finishing(conn), 1)
        finishing = cursor.execute.call_args.args[0]
        self.assertIn("live_publication_status = 'finishing'", finishing)
        self.assertIn("live_publication_attempts = 0", finishing)
        self.assertIn("artifact_announcement_ledger", finishing)
        self.assertIn("artifact_publication_receipts", finishing)
        self.assertIn("receipt.role = 'promotion'", finishing)
        self.assertIn("s.submitted_sequence < s.final_sequence", finishing)
        self.assertIn("b.wandb_confirmed_at IS NULL", finishing)
        self.assertIn("t.telemetry_protocol_version = 1", finishing)
        self.assertIn("artifact_stream.stream_id LIKE 'artifact-v2-%%'", finishing)


class ModalEvalStorageAndWorkerTests(unittest.TestCase):
    def test_warm_volume_cache_avoids_object_get_and_copies_attempt_locally(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, result_path = self._worker_payload(root)
            asset = payload["contract"]["asset"]
            cache_root = root / "volume"
            source = root / "game.nes"
            cached = install_rom_file(source, asset, cache_root)
            payload["rom_get_url"] = "file:///definitely/missing/game.nes"

            def finish_child(command, **_kwargs):
                child_input = Path(command[command.index("--input") + 1])
                child_output = Path(command[command.index("--output") + 1])
                request = json.loads(child_input.read_text(encoding="utf-8"))
                runtime_rom = Path(request["rom_path"])
                self.assertNotEqual(runtime_rom, cached)
                self.assertEqual(runtime_rom.read_bytes(), cached.read_bytes())
                child_output.write_text(
                    json.dumps(
                        {
                            "metrics": {"eval/full/episode/return/mean": 1.0},
                            "episode_results": [],
                            "preview": None,
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("rlab.modal_eval_worker.subprocess.run", side_effect=finish_child):
                execute_attempt(payload, cache_root=cache_root)

            evidence = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "succeeded")

    def test_rom_free_worker_does_not_require_rom_download(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.zip"
            metadata = root / "metadata.json"
            result_path = root / "result.json"
            model.write_bytes(b"model")
            metadata.write_text("{}\n", encoding="utf-8")
            eval_contract = build_execution_contract(
                checkpoint_sha256=file_sha256(model),
                runtime_image_ref="docker:example.invalid/rlab@sha256:" + "f" * 64,
                eval_environment={"env_provider": "rlab", "game": "Bandit-v0"},
                episodes=1,
                n_envs=1,
                max_steps=1,
                seed=10_000,
                seed_protocol=SEED_PROTOCOL,
                asset_manifest=None,
            )
            payload = {
                "attempt_id": "rom-free-attempt",
                "contract": eval_contract,
                "expires_at": time.time() + 60,
                "child_timeout_seconds": 10,
                "model_get_url": model.resolve().as_uri(),
                "metadata_get_url": metadata.resolve().as_uri(),
                "metadata_sha256": file_sha256(metadata),
                "result_uri": result_path.resolve().as_uri(),
                "result_put_url": result_path.resolve().as_uri(),
            }

            def finish_child(command, **_kwargs):
                child_input = Path(command[command.index("--input") + 1])
                child_output = Path(command[command.index("--output") + 1])
                request = json.loads(child_input.read_text(encoding="utf-8"))
                self.assertIsNone(request["rom_path"])
                child_output.write_text(
                    json.dumps(
                        {
                            "metrics": {"eval/full/episode/return/mean": 1.0},
                            "episode_results": [],
                            "preview": None,
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("rlab.modal_eval_worker.subprocess.run", side_effect=finish_child):
                execute_attempt(payload)

            evidence = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "succeeded")
            self.assertEqual(evidence["rom_sha256"], "")

    def test_artifact_projection_embeds_checkpoint_playback_metadata(self) -> None:
        class FakeArtifact:
            def __init__(self, name, type, metadata):
                self.name = name
                self.type = type
                self.metadata = metadata
                self.references = []

            def add_reference(self, uri):
                self.references.append(uri)

        class FakeRun:
            def __init__(self):
                self.logged = []
                self.finished = False

            def log_artifact(self, artifact, aliases):
                self.logged.append((artifact, aliases))

            def finish(self):
                self.finished = True

        run = FakeRun()
        fake_wandb = SimpleNamespace(
            init=lambda **_kwargs: run,
            Artifact=FakeArtifact,
            Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        model_metadata = {
            "metadata_version": 3,
            "filename": "ppo_game_500_steps.zip",
            "training_metadata": {"env_config": {"game": "Game-Nes-v0"}},
            "training_metadata_hash": "metadata-hash",
        }
        payload = {
            "projection_kind": "artifact_reference",
            "train_config": {
                "wandb": True,
                "wandb_run_id": "run-id",
                "wandb_entity": "entity",
                "wandb_project": "project",
                "run_name": "display-name",
            },
            "artifact_kind": "checkpoint",
            "checkpoint_uri": "s3://bucket/checkpoint/model.zip",
            "metadata_uri": "s3://bucket/checkpoint/metadata.json",
            "checkpoint_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "checkpoint_step": 500,
            "model_metadata": model_metadata,
        }

        with (
            mock.patch.dict("sys.modules", {"wandb": fake_wandb}),
            mock.patch("rlab.wandb_publisher.load_wandb_env"),
            mock.patch(
                "rlab.wandb_publisher.resolve_wandb_namespace",
                return_value=("entity", "project"),
            ),
            mock.patch(
                "rlab.wandb_publisher.configure_wandb_metrics",
                side_effect=lambda value: value,
            ),
        ):
            project_payload(payload)

        artifact, aliases = run.logged[0]
        self.assertEqual(
            artifact.metadata["training_metadata"], model_metadata["training_metadata"]
        )
        self.assertEqual(artifact.metadata["filename"], "model.zip")
        self.assertEqual(artifact.metadata["source_filename"], "ppo_game_500_steps.zip")
        self.assertEqual(artifact.metadata["artifact_storage_uri"], payload["checkpoint_uri"])
        self.assertEqual(artifact.references, [payload["checkpoint_uri"]])
        self.assertEqual(aliases, ["latest", "step-500"])
        self.assertTrue(run.finished)

    def test_artifact_projection_honors_promoted_fast_path_aliases(self) -> None:
        class FakeArtifact:
            def __init__(self, _name, type, metadata):
                self.type = type
                self.metadata = metadata

            def add_reference(self, _uri):
                return None

        run = mock.MagicMock()
        fake_wandb = SimpleNamespace(
            init=lambda **_kwargs: run,
            Artifact=FakeArtifact,
            Settings=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        payload = {
            "projection_kind": "artifact_reference",
            "train_config": {
                "wandb": True,
                "wandb_run_id": "run-id",
                "wandb_entity": "entity",
                "wandb_project": "project",
            },
            "artifact_kind": "checkpoint",
            "checkpoint_uri": "s3://bucket/checkpoint/model.zip",
            "metadata_uri": "s3://bucket/checkpoint/metadata.json",
            "checkpoint_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "checkpoint_step": 500,
            "model_metadata": {"training_metadata": {}},
            "artifact_aliases": ["latest", "promoted", "step-500"],
        }

        with (
            mock.patch.dict("sys.modules", {"wandb": fake_wandb}),
            mock.patch("rlab.wandb_publisher.load_wandb_env"),
            mock.patch(
                "rlab.wandb_publisher.resolve_wandb_namespace",
                return_value=("entity", "project"),
            ),
            mock.patch(
                "rlab.wandb_publisher.configure_wandb_metrics",
                side_effect=lambda value: value,
            ),
        ):
            project_payload(payload)

        self.assertEqual(
            run.log_artifact.call_args.kwargs["aliases"],
            ["latest", "promoted", "step-500"],
        )

    def test_artifact_reference_payload_loads_complete_checkpoint_metadata(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {
            "train_job_id": 10,
            "train_config": {"wandb": True},
            "next_artifact_projection_id": 1,
        }
        complete = {"last_ledger_id": 1}
        announcement = {
            "kind": "checkpoint",
            "model_uri": "s3://bucket/checkpoint/model.zip",
            "metadata_uri": "s3://bucket/checkpoint/metadata.json",
            "sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "step": 500,
        }
        model_metadata = {"training_metadata": {"env_config": {"game": "Game-Nes-v0"}}}
        store = mock.MagicMock()
        store.get_json_optional.side_effect = [complete, announcement]
        store.get_json.return_value = model_metadata
        captured = {}

        def fake_execute(payload, **_kwargs):
            captured.update(payload)
            return None

        with mock.patch(
            "rlab.modal_eval_orchestrator._execute_projection",
            side_effect=fake_execute,
        ):
            projected = project_artifact_references(
                conn,
                store,
                repo_root=Path.cwd(),
                deadline_monotonic=time.monotonic() + 60,
            )

        self.assertEqual(projected, 1)
        self.assertEqual(captured["model_metadata"], model_metadata)
        store.get_json.assert_called_once_with(announcement["metadata_uri"])

    def test_promoted_artifact_projects_before_historical_backlog(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        promoted_announcement = {
            "kind": "checkpoint",
            "model_uri": "s3://bucket/promoted/model.zip",
            "metadata_uri": "s3://bucket/promoted/metadata.json",
            "sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "step": 13500000,
        }
        cursor.fetchone.return_value = {
            "train_job_id": 26,
            "train_config": {"wandb": True},
            "next_artifact_projection_id": 4,
            "promoted_ledger_id": 27,
            "promoted_announcement": promoted_announcement,
            "promoted_artifact_projected_at": None,
        }
        store = mock.MagicMock()
        store.get_json_optional.return_value = {"last_ledger_id": 35}
        store.get_json.return_value = {"training_metadata": {"env_config": {}}}
        captured = {}

        def fake_execute(payload, **_kwargs):
            captured.update(payload)
            return None

        with mock.patch(
            "rlab.modal_eval_orchestrator._execute_projection",
            side_effect=fake_execute,
        ):
            projected = project_artifact_references(
                conn,
                store,
                repo_root=Path.cwd(),
                deadline_monotonic=time.monotonic() + 60,
            )

        self.assertEqual(projected, 1)
        self.assertEqual(captured["checkpoint_step"], 13500000)
        self.assertEqual(
            captured["artifact_aliases"],
            ["latest", "promoted", "step-13500000"],
        )
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertTrue(any("promoted_artifact_projected_at = now()" in sql for sql in statements))

    def test_mailbox_run_projects_promoted_artifact_before_marking_complete(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        promoted_announcement = {
            "kind": "checkpoint",
            "model_uri": "s3://bucket/promoted/model.zip",
            "metadata_uri": "s3://bucket/promoted/metadata.json",
            "sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "step": 2_000_000,
        }
        cursor.fetchone.return_value = {
            "train_job_id": 37,
            "train_config": {"wandb": True, "telemetry_transport": "neon_mailbox_v1"},
            "next_artifact_projection_id": 5,
            "promoted_ledger_id": 4,
            "promoted_announcement": promoted_announcement,
            "promoted_artifact_projected_at": None,
        }
        store = mock.MagicMock()
        store.get_json.return_value = {"training_metadata": {"env_config": {}}}
        captured = {}

        def fake_execute(payload, **_kwargs):
            captured.update(payload)
            return None

        with mock.patch(
            "rlab.modal_eval_orchestrator._execute_projection",
            side_effect=fake_execute,
        ):
            projected = project_artifact_references(
                conn,
                store,
                repo_root=Path.cwd(),
                deadline_monotonic=time.monotonic() + 60,
            )

        self.assertEqual(projected, 1)
        self.assertEqual(captured["checkpoint_uri"], promoted_announcement["model_uri"])
        self.assertEqual(captured["artifact_aliases"], ["latest", "promoted", "step-2000000"])
        statements = [call.args[0] for call in cursor.execute.call_args_list]
        self.assertIn("r.promoted_artifact_projected_at IS NULL", statements[0])
        self.assertTrue(any("promoted_artifact_projected_at = now()" in sql for sql in statements))

    def test_identical_model_bytes_allow_distinct_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            objects = ObjectStore((root / "objects").resolve().as_uri())
            store = MetricStore(root / "rlab.sqlite")
            store.init()
            args = SimpleNamespace(
                queue_train_job_id=9,
                env_provider="stable-retro-turbo",
                checkpoint_eval_environment={
                    "env_provider": "stable-retro-turbo",
                    "game": "Game-Nes-v0",
                },
                checkpoint_eval_stages=[],
                checkpoint_eval_asset_manifest={
                    "schema_version": 2,
                    "game": "Game-Nes-v0",
                    "filename": "Game-Nes-v0.rom",
                    "size_bytes": 1,
                    "sha256": "a" * 64,
                    "provider_rom_identity": "b" * 40,
                    "object_uri": "s3://bucket/Game-Nes-v0.rom",
                },
                checkpoint_eval_n_envs=1,
                checkpoint_eval_seed=10_000,
                checkpoint_eval_seed_protocol="vector-lane-v1",
                post_train_eval_episodes=1,
                post_train_eval_max_steps=10,
            )
            for index, kind in enumerate(("interrupted", "final"), start=1):
                model = root / f"{kind}.zip"
                metadata = root / f"{kind}.metadata.json"
                model.write_bytes(b"same-model")
                metadata.write_text(json.dumps({"kind": kind}) + "\n", encoding="utf-8")
                store.record_checkpoint(
                    run_name="smoke",
                    kind=kind,
                    step=index,
                    path=model,
                    metadata_path=metadata,
                )
                row = store.pending_artifact_uploads(limit=1)[0]
                self.assertTrue(process_upload(store, objects, args, row))

            first = objects.get_json("artifact-announcements/9/00000001.json")
            second = objects.get_json("artifact-announcements/9/00000002.json")
            self.assertEqual(first["model_uri"], second["model_uri"])
            self.assertNotEqual(first["metadata_uri"], second["metadata_uri"])

    def test_none_backend_uploads_checkpoint_to_neon_without_local_eval_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = MetricStore(root / "rlab.sqlite")
            store.init()
            checkpoint_ids = []
            for step in (100, 200):
                model = root / f"model_{step}_steps.zip"
                metadata = root / f"model_{step}_steps.metadata.json"
                model.write_bytes(f"checkpoint-{step}".encode())
                metadata.write_text("{}\n", encoding="utf-8")
                checkpoint_ids.append(
                    store.record_checkpoint(
                        run_name="smoke",
                        kind="checkpoint",
                        step=step,
                        path=model,
                        metadata_path=metadata,
                        eval_required=False,
                    )
                )
            objects = ObjectStore((root / "objects").resolve().as_uri())
            args = SimpleNamespace(
                queue_train_job_id=9,
                telemetry_transport="neon_mailbox_v1",
                runtime_image_ref="docker:example.invalid/rlab@sha256:" + "f" * 64,
                wandb_run_id="rlab-smoke",
                wandb=False,
            )
            mailbox = mock.MagicMock()

            with (
                mock.patch("rlab.checkpoint_coordinator._eval_payload", return_value={}),
                mock.patch(
                    "rlab.telemetry_mailbox.WorkerMailbox.from_env",
                    return_value=mailbox,
                ),
            ):
                for row in store.pending_artifact_uploads(limit=10):
                    self.assertTrue(process_upload(store, objects, args, row))

            with store.connection() as conn:
                eval_count = conn.execute("SELECT COUNT(*) AS count FROM eval_results").fetchone()[
                    "count"
                ]
                uploads = conn.execute(
                    "SELECT checkpoint_id, status, storage_uri FROM artifact_uploads "
                    "ORDER BY checkpoint_id"
                ).fetchall()
            self.assertEqual(eval_count, 0)
            self.assertEqual(
                [upload["checkpoint_id"] for upload in uploads],
                checkpoint_ids,
            )
            self.assertTrue(all(upload["status"] == "uploaded" for upload in uploads))
            self.assertTrue(
                all(str(upload["storage_uri"]).endswith("/model.zip") for upload in uploads)
            )
            self.assertEqual(mailbox.append_event.call_count, 2)
            self.assertEqual(
                [call.args[0] for call in mailbox.append_event.call_args_list],
                ["checkpoint_ready", "checkpoint_ready"],
            )
            self.assertEqual(
                [call.args[1]["step"] for call in mailbox.append_event.call_args_list],
                [100, 200],
            )

    def test_versioned_training_only_bundle_is_ready_and_playback_bound(self) -> None:
        goal = Path("experiments/goals/Breakout-Atari2600-v0/_goal.yaml")
        recipe = goal.parent / "recipes/ppo-snapshot-curriculum.yaml"
        runtime_ref = "docker:example.invalid/rlab@sha256:" + "f" * 64
        materialized = compose_train_document(goal, recipe)
        recipe_document = build_recipe_document(
            materialized,
            repo_root=Path.cwd(),
            source_commit="a" * 40,
            run_description="training-only checkpoint publication regression",
            seed=123,
            runtime_image_ref=runtime_ref,
        )
        train_config = dict(recipe_document["recipe"]["train_config"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model.zip"
            model.write_bytes(b"checkpoint")
            recipe_path = write_canonical_json(root / "model.recipe.json", recipe_document)
            model_document = build_model_document(
                model,
                recipe_path,
                {
                    "kind": "checkpoint",
                    "checkpoint_step": 500_000,
                    "queue_train_job_id": 9,
                    "runtime_image_ref": runtime_ref,
                    "algorithm_id": "ppo",
                    "model_class": "stable_baselines3.ppo.ppo.PPO",
                    "training_backend_id": "sb3.ppo",
                    "training_backend_config_hash": training_backend_config_hash(train_config),
                },
            )
            model_document_path = write_canonical_json(root / "model.model.json", model_document)
            metric_store = MetricStore(root / "rlab.sqlite")
            metric_store.init()
            metric_store.record_checkpoint(
                run_name="smoke",
                kind="checkpoint",
                step=500_000,
                path=model,
                metadata_path=model_document_path,
                eval_required=False,
            )
            objects = ObjectStore((root / "objects").resolve().as_uri())
            mailbox = mock.MagicMock()
            args = SimpleNamespace(
                **{
                    **train_config,
                    "queue_train_job_id": 9,
                    "telemetry_transport": "neon_mailbox_v1",
                    "runtime_image_ref": runtime_ref,
                    "wandb_run_id": "rlab-smoke",
                    "wandb": True,
                }
            )

            with mock.patch(
                "rlab.telemetry_mailbox.WorkerMailbox.from_env",
                return_value=mailbox,
            ):
                row = metric_store.pending_artifact_uploads(limit=1)[0]
                self.assertTrue(process_upload(metric_store, objects, args, row))

            announcement = mailbox.append_event.call_args.args[1]
            self.assertEqual(
                announcement["playback_contract_sha256"],
                playback_contract_sha256(recipe_document),
            )
            self.assertNotIn("evaluation_contract_sha256", announcement)
            self.assertEqual(
                validate_announcement(
                    announcement,
                    materialized_train_config={
                        **train_config,
                        "runtime_image_ref": runtime_ref,
                    },
                ),
                announcement,
            )
            _verify_checkpoint_artifacts(objects, announcement)

    def test_s3_client_forces_sigv4_for_r2_presigned_urls(self) -> None:
        store = ObjectStore("s3://bucket/prefix")
        with (
            mock.patch.dict(
                "os.environ",
                {"AWS_S3_ENDPOINT_URL": "https://account.r2.cloudflarestorage.com"},
            ),
            mock.patch("boto3.client") as client,
        ):
            store._s3_client()
        config = client.call_args.kwargs["config"]
        self.assertEqual(config.signature_version, "s3v4")
        self.assertEqual(config.s3["addressing_style"], "path")

    def test_coordinator_ignores_legacy_orphan_without_complete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            model = run_dir / "model_500_steps.zip"
            model.write_bytes(b"model")
            model.with_suffix(".metadata.json").write_text(
                json.dumps({"kind": "checkpoint", "checkpoint_step": 500}) + "\n",
                encoding="utf-8",
            )
            store = MetricStore(run_dir / "rlab.sqlite")
            store.init()
            recovered = reconcile_orphan_models(
                store,
                SimpleNamespace(run_name="smoke"),
                run_dir,
            )
            self.assertEqual(recovered, 0)
            self.assertEqual(store.checkpoints(), [])

    def test_checkpoint_event_outbox_freezes_exact_replay_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            event_id = "checkpoint-ready:9:1"
            first = prepare_checkpoint_event(
                run_dir,
                event_id=event_id,
                event_type="checkpoint_ready",
                payload={"train_job_id": 9, "ledger_id": 1, "step": 100},
            )
            replay = prepare_checkpoint_event(
                run_dir,
                event_id=event_id,
                event_type="checkpoint_ready",
                payload={"train_job_id": 9, "ledger_id": 1, "step": 999},
            )

            self.assertEqual(replay, first)
            self.assertEqual(replay["payload"]["step"], 100)
            self.assertEqual(replay["payload"]["_mailbox_event_id"], event_id)
            self.assertEqual(len(replay["payload"]["_outbox_sha256"]), 64)

            path = checkpoint_event_path(run_dir, event_id)
            corrupted = json.loads(path.read_text(encoding="utf-8"))
            corrupted["payload"]["step"] = 101
            path.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "payload hash mismatch"):
                prepare_checkpoint_event(
                    run_dir,
                    event_id=event_id,
                    event_type="checkpoint_ready",
                    payload={"train_job_id": 9, "ledger_id": 1, "step": 100},
                )

    def test_permanent_upload_failure_emits_ordered_tombstone(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            store = MetricStore(root / "rlab.sqlite")
            store.init()
            store.record_checkpoint(
                run_name="smoke",
                kind="checkpoint",
                step=10,
                path=root / "missing.zip",
                metadata_path=root / "missing.metadata.json",
            )
            objects = ObjectStore((root / "objects").resolve().as_uri())
            args = SimpleNamespace(queue_train_job_id=9)
            for _ in range(3):
                row = store.pending_artifact_uploads(limit=1)[0]
                self.assertFalse(process_upload(store, objects, args, row))
            tombstone = objects.get_json("artifact-announcements/9/00000001.json")
            self.assertEqual(tombstone["kind"], "tombstone")
            self.assertEqual(store.phase_counts()["artifacts:failed_terminal"], 1)

    def test_filesystem_objects_are_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            store = ObjectStore(Path(temporary).resolve().as_uri())
            store.put_bytes("evidence.json", b"one")
            store.put_bytes("evidence.json", b"one")
            with self.assertRaisesRegex(RuntimeError, "different content"):
                store.put_bytes("evidence.json", b"two")

    def _worker_payload(self, root: Path) -> tuple[dict, Path]:
        model = root / "model.zip"
        metadata = root / "metadata.json"
        rom = root / "game.nes"
        result = root / "result.json"
        model.write_bytes(b"model")
        metadata.write_text("{}\n", encoding="utf-8")
        rom.write_bytes(b"NES\x1a" + bytes(12) + b"rom")
        eval_contract = build_execution_contract(
            checkpoint_sha256=file_sha256(model),
            runtime_image_ref="docker:example.invalid/rlab@sha256:" + "f" * 64,
            eval_environment={"game": "Game-Nes-v0", "task": {}},
            episodes=1,
            n_envs=1,
            max_steps=10,
            seed=10_000,
            seed_protocol=SEED_PROTOCOL,
            asset_manifest={
                "schema_version": 2,
                "game": "Game-Nes-v0",
                "filename": rom.name,
                "size_bytes": rom.stat().st_size,
                "sha256": file_sha256(rom),
                "object_uri": rom.resolve().as_uri(),
                "provider_rom_identity": hashlib.sha1(b"rom").hexdigest(),
                "provider_rom_identity_algorithm": "sha1-provider-body-v1",
            },
        )
        return (
            {
                "attempt_id": "worker-attempt",
                "contract": eval_contract,
                "expires_at": time.time() + 60,
                "child_timeout_seconds": 1,
                "model_get_url": model.resolve().as_uri(),
                "metadata_get_url": metadata.resolve().as_uri(),
                "metadata_sha256": file_sha256(metadata),
                "rom_get_url": rom.resolve().as_uri(),
                "result_uri": result.resolve().as_uri(),
                "result_put_url": result.resolve().as_uri(),
            },
            result,
        )

    def test_child_native_crash_becomes_structured_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload, result_path = self._worker_payload(Path(temporary))
            metadata_path = Path(temporary) / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "training_backend_id": "sb3.a2c",
                        "algorithm_id": "a2c",
                        "model_class": "stable_baselines3.a2c.a2c.A2C",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            payload["metadata_sha256"] = file_sha256(metadata_path)
            child_request = {}

            def crash_child(command, **_kwargs):
                child_input = Path(command[command.index("--input") + 1])
                child_request.update(json.loads(child_input.read_text(encoding="utf-8")))
                return subprocess.CompletedProcess(command, -11, "", "segmentation fault")

            with mock.patch(
                "rlab.modal_eval_worker.subprocess.run",
                side_effect=crash_child,
            ) as run:
                receipt = execute_attempt(payload)
            evidence = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed")
            self.assertIn("-11", evidence["error"])
            self.assertEqual(child_request["model_metadata"]["algorithm_id"], "a2c")
            self.assertEqual(receipt["result_uri"], payload["result_uri"])
            self.assertIsNone(run.call_args.kwargs["stdout"])
            self.assertIsNone(run.call_args.kwargs["stderr"])
            self.assertNotIn("capture_output", run.call_args.kwargs)

    def test_child_timeout_becomes_structured_failure_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload, result_path = self._worker_payload(Path(temporary))
            with mock.patch(
                "rlab.modal_eval_worker.subprocess.run",
                side_effect=subprocess.TimeoutExpired(["child"], 1),
            ):
                execute_attempt(payload)
            evidence = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed")
            self.assertEqual(evidence["error"], "eval child timeout")

    def test_expired_attempt_uploads_evidence_before_heavy_work(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            payload, result_path = self._worker_payload(Path(temporary))
            payload["expires_at"] = time.time() - 1
            with mock.patch("rlab.modal_eval_worker.write_downloaded_file") as download:
                execute_attempt(payload)
            download.assert_not_called()
            self.assertEqual(json.loads(result_path.read_text())["status"], "expired")

    def test_successful_eval_uploads_preview_without_changing_eval_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload, result_path = self._worker_payload(root)
            uploaded_path = root / "public" / "preview.mp4"
            payload["preview"] = {
                "put_url": uploaded_path.resolve().as_uri(),
                "public_url": "https://preview.example/preview.mp4",
                "object_uri": "s3://preview-bucket/preview.mp4",
                "content_type": "video/mp4",
                "cache_control": "public, max-age=31536000, immutable",
                "max_frames": 450,
                "fps": 15,
                "max_lanes": 4,
                "scale": 2,
                "max_bytes": 2 * 1024 * 1024,
                "encode_timeout_seconds": 2,
                "upload_timeout_seconds": 3,
            }

            def complete_child(command, **_kwargs):
                output = Path(command[command.index("--output") + 1])
                preview = output.with_suffix(".mp4")
                preview.write_bytes(b"mp4")
                output.write_text(
                    json.dumps(
                        {
                            "metrics": {"eval/full/episode/return/mean": 1.0},
                            "episode_results": [{"seed": 10_000}],
                            "preview": {
                                "status": "ready",
                                "path": str(preview),
                                "lane_count": 2,
                                "frames": 450,
                                "fps": 15,
                                "width": 336,
                                "height": 168,
                                "duration_seconds": 30.0,
                                "size_bytes": 3,
                                "observation_source": "preprocessed_policy_observation",
                            },
                        }
                    ),
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            with mock.patch("rlab.modal_eval_worker.subprocess.run", side_effect=complete_child):
                execute_attempt(payload)

            evidence = json.loads(result_path.read_text(encoding="utf-8"))
            uploaded = uploaded_path.read_bytes()

        self.assertEqual(evidence["status"], "succeeded")
        self.assertEqual(evidence["preview"]["status"], "succeeded")
        self.assertEqual(evidence["preview"]["public_url"], payload["preview"]["public_url"])
        self.assertEqual(uploaded, b"mp4")


if __name__ == "__main__":
    unittest.main()
