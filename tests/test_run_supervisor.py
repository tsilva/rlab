from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rlab.eval_backend import EvalHandle, EvalPoll
from rlab.policy_bundle import (
    build_recipe_document,
    canonical_json_sha256,
    evaluation_contract_sha256,
)
from rlab.r2_store import BucketConfig, RunStorageConfig
from rlab.recipe_documents import compose_train_document
from rlab.run_authority import RunAuthority
from rlab.run_contracts import (
    CheckpointManifest,
    RunManifest,
    new_attempt_id,
    new_run_id,
    utc_now,
)
from rlab.run_supervisor import RunSupervisor, _bind_evaluation_contract


SOURCE_SHA = "a" * 40
BUILD_SOURCE_SHA = "f" * 40
RUNTIME_INPUT_SHA256 = "e" * 64
IMAGE = "docker:registry.example/rlab@sha256:" + "b" * 64
GOAL = Path("experiments/goals/SuperMarioBros-Nes-v0/Level1-1/_goal.yaml")
RECIPE = GOAL.parent / "recipes" / "ppo.yaml"


class FailingSpawnBackend:
    def submit(self, intent):
        raise RuntimeError("connection outcome unknown")

    def poll(self, handle: EvalHandle) -> EvalPoll:
        return EvalPoll(status="running")

    def cancel(self, handle: EvalHandle) -> None:
        return None


class RunSupervisorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.storage = RunStorageConfig(
            control=BucketConfig(uri=f"file://{root}/control"),
            evaluation=BucketConfig(uri=f"file://{root}/eval"),
            models=BucketConfig(
                uri=f"file://{root}/models",
                public_base_url="https://models.example",
            ),
        )
        self.authority = RunAuthority(self.storage)
        document = compose_train_document(GOAL, RECIPE)
        self.run_id = new_run_id()
        self.asset = {
            "schema_version": 2,
            "game": "SuperMarioBros-Nes-v0",
            "filename": "mario.nes",
            "size_bytes": 1,
            "sha256": "c" * 64,
            "object_uri": self.authority.evaluation.uri("rom.nes"),
            "provider_rom_identity": "d" * 40,
            "provider_rom_identity_algorithm": "sha1-provider-body-v1",
        }
        contract_document = dict(document)
        contract_config = dict(contract_document["train_config"])
        contract_config["rom_asset_manifest"] = self.asset
        contract_config["checkpoint_eval_backend"] = "modal"
        contract_document["train_config"] = contract_config
        portable_recipe = build_recipe_document(
            contract_document,
            repo_root=Path.cwd(),
            source_commit=SOURCE_SHA,
            run_description="supervisor unit test",
            seed=123,
            runtime_image_ref=IMAGE,
        )
        self.manifest = RunManifest(
            run_id=self.run_id,
            attempt_id=new_attempt_id(),
            created_at=utc_now(),
            source_sha=SOURCE_SHA,
            image_digest=IMAGE,
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            goal_sha256=str(
                document["train_config"]["effective_goal_contract_sha256"]
            ),
            recipe_slug="ppo",
            recipe_sha256=canonical_json_sha256(portable_recipe),
            recipe_overrides=(),
            environment_sha256=str(document["environment_hash"]).removeprefix("sha256:"),
            seed=123,
            run_description="supervisor unit test",
            compute={
                "request": {
                    "kind": "local",
                    "target": "b3",
                    "max_price": None,
                    "max_cost_usd": None,
                    "allow_on_demand": False,
                    "max_duration_seconds": 3600,
                },
                "selected": {
                    "kind": "local",
                    "target": "b3",
                    "max_price": None,
                    "max_cost_usd": None,
                    "allow_on_demand": False,
                    "max_duration_seconds": 3600,
                },
                "dstack_task": self.run_id,
                "runtime_workflow_run_id": "123",
                "runtime_input_sha256": RUNTIME_INPUT_SHA256,
                "runtime_build_source_sha": BUILD_SOURCE_SHA,
            },
            wandb={
                "run_id": self.run_id,
                "entity": "entity",
                "project": "project",
                "url": f"https://wandb.ai/entity/project/runs/{self.run_id}",
            },
            modal={
                "enabled": True,
                "environment_name": "rlab-eval",
                "app_name": f"rlab-eval-v2-{SOURCE_SHA[:12]}",
                "function_name": "evaluate_checkpoint",
                "deployment_source_sha": SOURCE_SHA,
                "rom_asset_manifest": self.asset,
            },
            storage=self.storage.manifest_locations(),
        )
        self.authority.create_manifest(self.manifest)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def supervisor(self) -> RunSupervisor:
        root = Path(self.temporary.name)
        return RunSupervisor(
            manifest_uri=self.authority.control.uri(
                f"runs/{self.run_id}/manifest.json"
            ),
            storage=self.storage,
            eval_backend=FailingSpawnBackend(),
            repo_root=Path.cwd(),
            work_root=root / "work",
        )

    def test_runtime_verification_uses_build_identity_and_runtime_input(self) -> None:
        supervisor = self.supervisor()
        with (
            patch.dict("os.environ", {"RLAB_ORCHESTRATOR": "dstack"}),
            patch(
                "rlab.run_supervisor.runtime_contract",
                return_value={
                    "runtime_build_source_sha": BUILD_SOURCE_SHA,
                    "runtime_input_sha256": RUNTIME_INPUT_SHA256,
                },
            ),
        ):
            supervisor.validate_runtime()

        with (
            patch.dict("os.environ", {"RLAB_ORCHESTRATOR": "dstack"}),
            patch(
                "rlab.run_supervisor.runtime_contract",
                return_value={
                    "runtime_build_source_sha": SOURCE_SHA,
                    "runtime_input_sha256": RUNTIME_INPUT_SHA256,
                },
            ),
            self.assertRaisesRegex(RuntimeError, "runtime build source SHA"),
        ):
            supervisor.validate_runtime()

    def test_training_only_contract_omits_null_eval_contract(self) -> None:
        config = {"checkpoint_eval_contract": None}
        contract = _bind_evaluation_contract(
            config,
            recipe_document={},
            evaluation_required=False,
        )

        self.assertEqual(contract, {})
        self.assertNotIn("checkpoint_eval_contract", config)

    def test_materializes_exact_mario_acceptance_contract(self) -> None:
        supervisor = self.supervisor()
        with patch("rlab.run_supervisor.verify_rom_file"):
            supervisor.materialize()
        self.assertEqual(supervisor.train_config["timesteps"], 50_000_000)
        self.assertEqual(supervisor.train_config["checkpoint_freq"], 250_000)
        self.assertEqual(supervisor.train_config["n_envs"], 16)
        self.assertEqual(supervisor.eval_contract["episodes"], 100)
        self.assertEqual(
            supervisor.eval_contract["acceptance"],
            [
                {
                    "metric": "eval/full/outcome/success/rate/min",
                    "operator": ">=",
                    "threshold": 1.0,
                }
            ],
        )

    def test_ambiguous_modal_spawn_is_not_immediately_repeated(self) -> None:
        supervisor = self.supervisor()
        with patch("rlab.run_supervisor.verify_rom_file"):
            supervisor.materialize()
        supervisor.metric_store.init()
        supervisor.ledger.init()
        checkpoint = CheckpointManifest(
            run_id=self.run_id,
            checkpoint_id="checkpoint-250000-" + "e" * 16,
            step=250_000,
            purpose="periodic",
            sha256="e" * 64,
            size_bytes=10,
            public_url="https://models.example/model.zip",
            model_document_url="https://models.example/model.json",
            model_document_sha256="f" * 64,
            recipe_document_url="https://models.example/recipe.json",
            recipe_document_sha256="1" * 64,
            goal_sha256=self.manifest.goal_sha256,
            recipe_sha256=self.manifest.recipe_sha256,
            environment_sha256=self.manifest.environment_sha256,
            evaluation_contract_sha256=evaluation_contract_sha256(
                supervisor.recipe_document
            ),
            recovery_sidecar_key="recovery.json",
            created_at=utc_now(),
        )
        supervisor._ensure_eval(1, checkpoint)
        self.assertEqual(supervisor._submit_pending_evals(), 0)
        row = supervisor.ledger.evals()[0]
        self.assertEqual(row["status"], "submitted")
        self.assertEqual(row["attempt"], 1)
        self.assertEqual(row["modal_call_id"], "")
        self.assertEqual(supervisor._submit_pending_evals(), 0)
        self.assertEqual(supervisor.ledger.evals()[0]["attempt"], 1)
        with supervisor.ledger.connection() as connection:
            connection.execute(
                "UPDATE eval_dispatches SET attempt_expires_at = 1000"
            )
        with patch("rlab.run_supervisor.time.time", return_value=1001):
            self.assertEqual(supervisor._poll_evals(10.0), 0)
        self.assertEqual(supervisor.ledger.evals()[0]["status"], "pending")

    def test_rejected_final_checkpoint_does_not_displace_earlier_acceptance(
        self,
    ) -> None:
        supervisor = self.supervisor()
        supervisor.ledger.init()
        checkpoints = [
            {
                "checkpoint_id": "checkpoint-250000-" + "e" * 16,
                "step": 250_000,
                "purpose": "periodic",
                "sha256": "e" * 64,
            },
            {
                "checkpoint_id": "checkpoint-500000-" + "f" * 16,
                "step": 500_000,
                "purpose": "final",
                "sha256": "f" * 64,
            },
        ]
        self.authority.models.put_json(
            f"runs/{self.run_id}/index.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "checkpoints": checkpoints,
                "promotion": None,
            },
            create_only=True,
        )
        for ledger_id, checkpoint in enumerate(checkpoints, start=1):
            supervisor.ledger.record_checkpoint_publication(
                checkpoint_ledger_id=ledger_id,
                manifest=checkpoint,
            )
            key = str(ledger_id) * 64
            supervisor.ledger.ensure_eval(
                checkpoint_ledger_id=ledger_id,
                intent={
                    "idempotency_key": key,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_step": checkpoint["step"],
                },
            )
            supervisor.ledger.mark_eval_terminal(
                idempotency_key=key,
                status="accepted" if ledger_id == 1 else "rejected",
                result={
                    "episode_results": [{}] * (100 if ledger_id == 1 else 1),
                    "status": "accepted" if ledger_id == 1 else "rejected",
                },
            )

        receipt = supervisor._create_promotion()

        self.assertIsNotNone(receipt)
        assert receipt is not None
        self.assertEqual(receipt.checkpoint_step, 250_000)
        self.assertEqual(receipt.checkpoint_id, checkpoints[0]["checkpoint_id"])


if __name__ == "__main__":
    unittest.main()
