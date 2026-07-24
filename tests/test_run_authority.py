from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from rlab.r2_store import BucketConfig, ConditionalWriteConflict, RunStorageConfig
from rlab.run_authority import LeaseUnavailable, RunAuthority
from rlab.run_contracts import (
    EvalIntent,
    PromotionReceipt,
    RunManifest,
    TerminalReceipt,
    checkpoint_id,
    eval_idempotency_key,
    new_attempt_id,
    new_run_id,
    utc_now,
)
from rlab.policy_bundle import model_document_path, recipe_document_path


SHA = "a" * 64


class RunAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.storage = RunStorageConfig(
            control=BucketConfig((root / "control").resolve().as_uri()),
            evaluation=BucketConfig((root / "eval").resolve().as_uri()),
            models=BucketConfig(
                (root / "models").resolve().as_uri(),
                public_base_url="https://models.example.test",
            ),
        )
        self.authority = RunAuthority(self.storage)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def manifest(self, run_id: str, attempt_id: str) -> RunManifest:
        return RunManifest(
            run_id=run_id,
            attempt_id=attempt_id,
            created_at=utc_now(),
            source_sha="e" * 40,
            image_digest="docker:registry.example/rlab@sha256:" + SHA,
            goal_slug="SuperMarioBros-Nes-v0/Level1-1",
            goal_sha256=SHA,
            recipe_slug="ppo",
            recipe_sha256=SHA,
            recipe_overrides=(),
            environment_sha256=SHA,
            seed=123,
            run_description="B3 clean-slate dstack acceptance",
            compute={
                "request": {
                    "kind": "local",
                    "target": "b3",
                    "max_duration_seconds": 86_400,
                },
                "selected": {
                    "kind": "local",
                    "target": "b3",
                    "max_duration_seconds": 86_400,
                },
                "dstack_task": run_id,
                "runtime_workflow_run_id": "123",
                "runtime_input_sha256": SHA,
                "runtime_build_source_sha": "e" * 40,
            },
            wandb={
                "run_id": run_id,
                "entity": "tsilva",
                "project": "super-mario-bros",
                "url": f"https://wandb.ai/tsilva/super-mario-bros/runs/{run_id}",
            },
            modal={
                "enabled": True,
                "environment_name": "rlab-eval",
                "app_name": "rlab-eval-v2-" + "e" * 12,
                "function_name": "evaluate_checkpoint",
                "deployment_source_sha": "e" * 40,
                "rom_asset_manifest": {"sha256": SHA},
            },
            storage=self.storage.manifest_locations(),
        )

    def test_identifiers_have_required_shapes(self) -> None:
        self.assertRegex(new_run_id(), r"^rlab-[0-9a-f]{32}$")
        self.assertRegex(new_attempt_id(), r"^attempt-[0-9a-f]{16}$")
        self.assertEqual(checkpoint_id(step=250_000, sha256=SHA), "checkpoint-250000-aaaaaaaaaaaaaaaa")

    def test_manifest_is_create_only_and_idempotent(self) -> None:
        run_id = new_run_id()
        manifest = self.manifest(run_id, new_attempt_id())
        first = self.authority.create_manifest(manifest)
        second = self.authority.create_manifest(manifest)
        self.assertEqual(first, second)
        changed = {**manifest.to_dict(), "seed": 124}
        with self.assertRaises(ConditionalWriteConflict):
            self.authority.control.put_json(
                f"runs/{run_id}/manifest.json",
                changed,
                create_only=True,
            )

    def test_manifest_binds_recipe_overrides(self) -> None:
        run_id = new_run_id()
        manifest = self.manifest(run_id, new_attempt_id())
        overridden = RunManifest(
            **{
                **manifest.to_dict(),
                "recipe_overrides": [
                    "train.backend.config.learning_rate=0.0002"
                ],
            }
        )
        overridden.validate()
        self.assertEqual(
            overridden.to_dict()["recipe_overrides"],
            ["train.backend.config.learning_rate=0.0002"],
        )
        invalid = RunManifest(
            **{**manifest.to_dict(), "recipe_overrides": [""]}
        )
        with self.assertRaisesRegex(ValueError, "non-empty"):
            invalid.validate()

    def test_lease_takeover_requires_expiry_and_old_etag_cannot_renew(self) -> None:
        run_id = new_run_id()
        old_attempt = new_attempt_id()
        instant = datetime(2026, 7, 24, 10, 0, tzinfo=UTC)
        lease = self.authority.acquire_lease(
            run_id=run_id,
            attempt_id=old_attempt,
            holder_id="container-one",
            now=instant,
        )
        with self.assertRaises(LeaseUnavailable):
            self.authority.acquire_lease(
                run_id=run_id,
                attempt_id=new_attempt_id(),
                holder_id="container-two",
                now=instant + timedelta(seconds=59),
            )
        takeover = self.authority.acquire_lease(
            run_id=run_id,
            attempt_id=new_attempt_id(),
            holder_id="container-two",
            now=instant + timedelta(seconds=61),
        )
        self.assertGreater(takeover.generation, lease.generation)
        with self.assertRaises(LeaseUnavailable):
            self.authority.renew_lease(lease, now=instant + timedelta(seconds=10))

    def test_metric_segments_are_ordered_and_immutable(self) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        key, digest = self.authority.seal_metric_segment(
            run_id=run_id,
            attempt_id=attempt_id,
            events=[
                {"event_seq": 1, "event_id": "one", "metrics": {"x": 1}},
                {"event_seq": 2, "event_id": "two", "metrics": {"x": 2}},
            ],
        )
        self.assertEqual(hashlib.sha256(self.authority.control.get_bytes(key)).hexdigest(), digest)
        with self.assertRaises(ValueError):
            self.authority.seal_metric_segment(
                run_id=run_id,
                attempt_id=attempt_id,
                events=[{"event_seq": 1}, {"event_seq": 1}],
            )

    def test_delivered_metric_journals_move_to_expiring_prefix(self) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        source_key, digest = self.authority.seal_metric_segment(
            run_id=run_id,
            attempt_id=attempt_id,
            events=[
                {"event_seq": 1, "event_id": "one"},
                {"event_seq": 2, "event_id": "two"},
            ],
        )
        archived = self.authority.archive_metric_journals(run_id=run_id)
        self.assertEqual(archived["segment_count"], 1)
        self.assertEqual(
            archived["prefix"],
            f"expiring-metric-journals/{run_id}/",
        )
        self.assertEqual(
            list(
                self.authority.control.iter_keys(
                    f"runs/{run_id}/attempts/{attempt_id}/metric-segments"
                )
            ),
            [],
        )
        destination_key = archived["keys"][0]
        self.assertEqual(
            hashlib.sha256(
                self.authority.control.get_bytes(destination_key)
            ).hexdigest(),
            digest,
        )
        self.assertFalse(self.authority.control._file_path(source_key).exists())
        self.assertEqual(
            self.authority.archive_metric_journals(run_id=run_id),
            archived,
        )

    def test_checkpoint_is_verified_and_public_index_is_cas_updated(self) -> None:
        run_id = new_run_id()
        root = Path(self.temporary.name)
        first_model = root / "one.zip"
        second_model = root / "two.zip"
        first_model.write_bytes(b"first-model")
        second_model.write_bytes(b"second-model")
        for path in (first_model, second_model):
            model_document_path(path).write_text('{"model":true}\n', encoding="utf-8")
            recipe_document_path(path).write_text('{"recipe":true}\n', encoding="utf-8")
        hashes = {
            "goal_sha256": SHA,
            "recipe_sha256": SHA,
            "environment_sha256": SHA,
            "evaluation_contract_sha256": SHA,
        }
        second = self.authority.publish_checkpoint(
            run_id=run_id,
            model_path=second_model,
            step=500_000,
            purpose="periodic",
            contract_hashes=hashes,
            recovery_sidecar={"local_path": "checkpoints/two.zip"},
        )
        first = self.authority.publish_checkpoint(
            run_id=run_id,
            model_path=first_model,
            step=250_000,
            purpose="periodic",
            contract_hashes=hashes,
            recovery_sidecar={"local_path": "checkpoints/one.zip"},
        )
        index = self.authority.models.get_json(f"runs/{run_id}/index.json")
        self.assertEqual(
            [row["checkpoint_id"] for row in index["checkpoints"]],
            [first.checkpoint_id, second.checkpoint_id],
        )
        self.assertTrue(first.public_url.startswith("https://models.example.test/"))
        public_manifest = json.loads(
            self.authority.models.get_bytes(
                f"runs/{run_id}/checkpoints/250000-{first.sha256}/manifest.json"
            )
        )
        self.assertEqual(public_manifest["sha256"], first.sha256)
        promotion = PromotionReceipt(
            run_id=run_id,
            checkpoint_id=first.checkpoint_id,
            checkpoint_step=first.step,
            eval_idempotency_key="f" * 64,
            eval_result_sha256="f" * 64,
            accepted_episode_count=100,
            promoted_at=utc_now(),
        )
        self.authority.create_promotion(promotion)
        promoted_index = self.authority.models.get_json(f"runs/{run_id}/index.json")
        self.assertEqual(
            promoted_index["promotion"]["checkpoint_id"],
            first.checkpoint_id,
        )

    def test_eval_intent_is_deterministic_and_private(self) -> None:
        run_id = new_run_id()
        key = eval_idempotency_key(
            run_id=run_id,
            checkpoint_sha256=SHA,
            evaluation_contract_sha256=SHA,
            episode_manifest_sha256=SHA,
            protocol="acceptance-v2",
        )
        intent = EvalIntent(
            run_id=run_id,
            checkpoint_id=checkpoint_id(step=250_000, sha256=SHA),
            idempotency_key=key,
            checkpoint_sha256=SHA,
            goal_sha256=SHA,
            recipe_sha256=SHA,
            environment_sha256=SHA,
            evaluation_contract_sha256=SHA,
            episode_manifest_sha256=SHA,
            protocol="acceptance-v2",
            execution_contract={"episodes": 100},
            result_key=f"runs/{run_id}/evals/{key}/result.json",
            timeout_seconds=1200,
            created_at=utc_now(),
            expires_at="2026-07-24T12:00:00Z",
        )
        self.authority.put_eval_intent(intent)
        stored = self.authority.evaluation.get_json(
            f"runs/{run_id}/evals/{key}/intent.json"
        )
        self.assertEqual(stored["checkpoint_id"], intent.checkpoint_id)
        self.assertEqual(list(self.authority.models.iter_keys(f"runs/{run_id}/evals")), [])

    def test_canonical_terminal_requires_complete_scientific_evidence(self) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        checkpoint = "checkpoint-250000-" + "a" * 16
        promotion = PromotionReceipt(
            run_id=run_id,
            checkpoint_id=checkpoint,
            checkpoint_step=250_000,
            eval_idempotency_key="b" * 64,
            eval_result_sha256="c" * 64,
            accepted_episode_count=100,
            promoted_at=utc_now(),
        )
        self.authority.control.put_json(
            f"runs/{run_id}/promotion.json",
            promotion.to_dict(),
        )
        drain = {
            "complete": True,
            "metric_segment_high_water": 12,
            "wandb_remote_high_water_mark": 12,
            "publication_capacity_ratio": 2.5,
            "journal_archive": {
                "prefix": f"expiring-metric-journals/{run_id}/",
                "segment_count": 1,
                "keys": ["segment.jsonl"],
            },
            "journal_expires_at": utc_now(),
        }
        receipt = TerminalReceipt(
            run_id=run_id,
            attempt_id=attempt_id,
            state="succeeded",
            acceptance_required=True,
            stop_reason="eval_acceptance",
            final_step=250_000,
            checkpoint_inventory=[
                {
                    "checkpoint_id": checkpoint,
                    "step": 250_000,
                    "purpose": "final",
                }
            ],
            eval_inventory=[
                {
                    "checkpoint_id": checkpoint,
                    "checkpoint_step": 250_000,
                    "status": "accepted",
                }
            ],
            wandb_high_water_mark=12,
            drain=drain,
            completed_at=utc_now(),
        )
        self.authority.create_terminal(receipt)
        self.assertEqual(
            self.authority.control.get_json(f"runs/{run_id}/terminal.json")["state"],
            "succeeded",
        )

        missing_remote = TerminalReceipt(
            **{
                **receipt.to_dict(),
                "run_id": new_run_id(),
                "drain": {**drain, "wandb_remote_high_water_mark": 11},
            }
        )
        with self.assertRaisesRegex(ValueError, "remotely visible"):
            self.authority.create_terminal(missing_remote)

    def test_training_only_terminal_is_attempt_scoped_and_not_scientific_success(
        self,
    ) -> None:
        run_id = new_run_id()
        attempt_id = new_attempt_id()
        receipt = TerminalReceipt(
            run_id=run_id,
            attempt_id=attempt_id,
            state="succeeded",
            acceptance_required=False,
            stop_reason="training_cap_complete",
            final_step=1_000_000,
            checkpoint_inventory=[
                {
                    "checkpoint_id": "checkpoint-1000000-" + "a" * 16,
                    "step": 1_000_000,
                    "purpose": "final",
                }
            ],
            eval_inventory=[],
            wandb_high_water_mark=25,
            drain={
                "complete": True,
                "metric_segment_high_water": 25,
                "wandb_remote_high_water_mark": 25,
            },
            completed_at=utc_now(),
        )

        self.authority.create_attempt_terminal(receipt)
        state = self.authority.semantic_state(run_id)
        self.assertEqual(state["attempt_terminals"], [receipt.to_dict()])
        self.assertIsNone(state["terminal"])
        with self.assertRaisesRegex(ValueError, "acceptance-backed"):
            self.authority.create_terminal(receipt)


if __name__ == "__main__":
    unittest.main()
