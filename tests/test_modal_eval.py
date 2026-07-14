from __future__ import annotations

import json
import hashlib
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from rlab.modal_eval_config import load_modal_eval_config, modal_app_name
from rlab.modal_eval_orchestrator import (
    DefaultModalInvoker,
    _stage_metrics,
    accept_attempt_result,
    available_eval_slots,
    budget_allows,
    deterministic_eval_failure,
    project_artifact_references,
    promotion_candidate_key,
    round_robin_jobs,
)
from rlab.checkpoint_eval_worker import evaluation_metric_payload
from rlab.modal_eval_projection import project_payload
from rlab.modal_eval_protocol import (
    build_execution_contract,
    execution_key,
    job_key,
    validate_attempt_result,
)
from rlab.modal_eval_storage import ObjectStore, file_sha256
from rlab.modal_eval_worker import execute_attempt
from rlab.checkpoint_coordinator import process_upload, reconcile_orphan_models
from rlab.metric_store import MetricStore
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
        seed_protocol="vector-lane-v1",
        asset_manifest={
            "filename": rom.name,
            "sha256": file_sha256(rom),
            "object_uri": rom.resolve().as_uri(),
            "provider_rom_identity": "c" * 40,
            "provider_rom_identity_algorithm": "sha1-provider-body-v1",
        },
    )


def successful_result(eval_contract: dict, *, attempt_id: str = "attempt") -> dict:
    return {
        "schema_version": 1,
        "contract_schema_version": eval_contract["schema_version"],
        "attempt_id": attempt_id,
        "execution_key": execution_key(eval_contract),
        "checkpoint_sha256": eval_contract["checkpoint_sha256"],
        "runtime_image_ref": eval_contract["runtime_image_ref"],
        "rom_sha256": eval_contract["asset"]["sha256"],
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
            },
            raw_metrics,
            True,
        )

        self.assertEqual(actual, expected)

    def test_preflight_checks_schema_asset_backend_and_exact_deployment(self) -> None:
        conn = mock.MagicMock()
        cursor = conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = {"drained": False, "effective_capacity": 1}
        function = mock.MagicMock()
        image_ref = "docker:example.invalid/rlab@sha256:" + "b" * 64
        manifest = {
            "game": "Game-Nes-v0",
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
            }
            report = modal_eval_cli.modal_preflight(
                runtime_image_ref=image_ref,
                game="Game-Nes-v0",
            )

        self.assertTrue(report["ready"])
        self.assertEqual(
            {check["name"] for check in report["checks"]},
            {
                "config_guards",
                "postgres_schema",
                "backend_state",
                "rom_asset",
                "modal_deployment",
                "modal_startup_probe",
            },
        )
        function.hydrate.assert_called_once_with()

    def test_checked_in_config_has_independent_twenty_call_guards(self) -> None:
        config = load_modal_eval_config(Path("experiments/modal_eval.yaml"))
        self.assertTrue(config.enabled)
        self.assertEqual(config.hard_max_active, 20)
        self.assertEqual(config.max_containers, 20)
        self.assertEqual(config.initial_effective_capacity, 1)
        self.assertFalse(config.single_use_containers)
        self.assertEqual(config.max_attempts, 2)
        self.assertFalse(config.preview_enabled)
        self.assertEqual(config.preview_max_frames, 450)
        self.assertEqual(config.preview_fps, 15)
        self.assertEqual(config.preview_max_bytes, 2 * 1024 * 1024)

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
            unknown.write_text(source.replace("  cpu: 4.0", "  cpu: 4.0\n  surprise: true"))
            with self.assertRaisesRegex(ValueError, "resources has unknown"):
                load_modal_eval_config(unknown)
            excessive = Path(temporary) / "excessive.yaml"
            excessive.write_text(
                source.replace("initial_effective_capacity: 1", "initial_effective_capacity: 21")
            )
            with self.assertRaisesRegex(ValueError, "exceeds the hard cap"):
                load_modal_eval_config(excessive)
            oversized_preview = Path(temporary) / "oversized-preview.yaml"
            oversized_preview.write_text(source.replace("  max_lanes: 4", "  max_lanes: 5"))
            with self.assertRaisesRegex(ValueError, "must not exceed 4"):
                load_modal_eval_config(oversized_preview)

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
            result["schema_version"] = 2
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
        self.assertEqual(marker.call_count, 2)
        sleep.assert_called_once()

    def test_recover_rejects_nonterminal_or_nonrecovery_state_without_remote_calls(self) -> None:
        cases = (
            (
                {"status": "running", "eval_run_status": "awaiting_artifact_recovery"},
                "not terminal",
            ),
            ({"status": "succeeded", "eval_run_status": "active"}, "not awaiting"),
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
                mock.patch(
                    "rlab.docker_host.run_checkpoint_coordinator_container"
                ) as recover,
            ):
                with self.assertRaisesRegex(ValueError, error):
                    modal_eval_cli.cmd_recover(SimpleNamespace(train_job_id=13))
                recover.assert_not_called()

    def test_recover_uses_drain_mode_without_host_writes(self) -> None:
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
        with (
            mock.patch.object(modal_eval_cli, "_conn", return_value=conn),
            mock.patch("rlab.machines.load_machine_registry", return_value=object()),
            mock.patch("rlab.machines.resolve_machine", return_value=machine),
            mock.patch(
                "rlab.docker_host.run_checkpoint_coordinator_container"
            ) as recover,
            mock.patch.object(modal_eval_cli, "_kick"),
        ):
            result = modal_eval_cli.cmd_recover(SimpleNamespace(train_job_id=13))

        self.assertEqual(result, 0)
        self.assertEqual(recover.call_args.kwargs["launch_id"], "train-13")
        self.assertEqual(recover.call_args.kwargs["run_name"], "run-13")
        self.assertEqual(recover.call_args.kwargs["runtime_image_ref"], runtime_ref)
        self.assertTrue(any("UPDATE eval_runs" in sql for sql in conn.cursor_obj.executed_sqls))

    def test_failed_recovery_preserves_awaiting_state(self) -> None:
        conn = FakeConnection(
            row={
                "id": 13,
                "status": "failed",
                "eval_run_status": "awaiting_artifact_recovery",
                "launch_id": "train-13",
                "output_uri": "/host/outputs/train-13",
                "machine": "beast-3",
                "runtime_image_ref": "docker:example.invalid/rlab@sha256:" + "b" * 64,
                "run_name": "run-13",
            }
        )
        machine = SimpleNamespace(
            paths=SimpleNamespace(
                container_outputs_dir="/output",
                env_file="/host/.env.runner",
                outputs_dir="/host/outputs",
            )
        )
        with (
            mock.patch.object(modal_eval_cli, "_conn", return_value=conn),
            mock.patch("rlab.machines.load_machine_registry", return_value=object()),
            mock.patch("rlab.machines.resolve_machine", return_value=machine),
            mock.patch(
                "rlab.docker_host.run_checkpoint_coordinator_container",
                side_effect=RuntimeError("recovery failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "recovery failed"):
                modal_eval_cli.cmd_recover(SimpleNamespace(train_job_id=13))

        self.assertFalse(any("UPDATE eval_runs" in sql for sql in conn.cursor_obj.executed_sqls))


class ModalEvalSchedulingTests(unittest.TestCase):
    def test_plain_runtime_error_from_modal_is_a_terminal_call_failure(self) -> None:
        call = mock.MagicMock()
        call.get.side_effect = RuntimeError("remote worker failed")
        with mock.patch("modal.FunctionCall.from_id", return_value=call):
            state, detail = DefaultModalInvoker().poll("fc-test")
        self.assertEqual(state, "failed")
        self.assertIn("remote worker failed", str(detail))

    def test_only_transient_failures_are_retryable(self) -> None:
        self.assertTrue(deterministic_eval_failure("checkpoint hash mismatch"))
        self.assertTrue(deterministic_eval_failure("environment contract is invalid"))
        self.assertFalse(deterministic_eval_failure("provider connection reset"))

    def test_available_slots_never_exceed_hard_cap_and_count_unknown_calls(self) -> None:
        self.assertEqual(
            available_eval_slots(effective_capacity=100, active_calls=19, hard_cap=20),
            1,
        )
        self.assertEqual(
            available_eval_slots(effective_capacity=20, active_calls=20, hard_cap=20),
            0,
        )

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


class ModalEvalStorageAndWorkerTests(unittest.TestCase):
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
            mock.patch("rlab.modal_eval_projection.load_wandb_env"),
            mock.patch(
                "rlab.modal_eval_projection.resolve_wandb_namespace",
                return_value=("entity", "project"),
            ),
            mock.patch(
                "rlab.modal_eval_projection.configure_wandb_metrics",
                side_effect=lambda value: value,
            ),
        ):
            project_payload(payload)

        artifact, aliases = run.logged[0]
        self.assertEqual(artifact.metadata["training_metadata"], model_metadata["training_metadata"])
        self.assertEqual(artifact.metadata["filename"], "model.zip")
        self.assertEqual(artifact.metadata["source_filename"], "ppo_game_500_steps.zip")
        self.assertEqual(artifact.metadata["artifact_storage_uri"], payload["checkpoint_uri"])
        self.assertEqual(artifact.references, [payload["checkpoint_uri"]])
        self.assertEqual(aliases, ["latest", "step-500"])
        self.assertTrue(run.finished)

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
        model_metadata = {
            "training_metadata": {"env_config": {"game": "Game-Nes-v0"}}
        }
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

    def test_coordinator_reconciles_an_orphan_model_with_metadata(self) -> None:
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
            self.assertEqual(recovered, 1)
            self.assertEqual(store.checkpoints()[0]["step"], 500)

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
            seed_protocol="vector-lane-v1",
            asset_manifest={
                "filename": rom.name,
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
            with mock.patch(
                "rlab.modal_eval_worker.subprocess.run",
                return_value=subprocess.CompletedProcess([], -11, "", "segmentation fault"),
            ) as run:
                receipt = execute_attempt(payload)
            evidence = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["status"], "failed")
            self.assertIn("-11", evidence["error"])
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
