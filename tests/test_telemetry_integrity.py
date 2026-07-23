from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from rlab.telemetry_integrity import (
    CanonicalEvent,
    IntegrityInputs,
    ProducerKey,
    TelemetryContractError,
    TelemetryIntegrityError,
    build_canonical_segment,
    build_eval_scope_exact,
    build_run_final_exact,
    build_training_success_scope_exact,
    decode_canonical_segment,
    normalize_wandb_rows,
    reduce_integrity,
    require_exact_contract_match,
    require_comparable_run_facts,
    write_fsync,
)


SHA = "a" * 64


def eval_contract(**overrides):
    value = {
        "canonical_goal_sha256": SHA,
        "effective_goal_contract_sha256": SHA,
        "recipe_sha256": SHA,
        "policy_bundle_sha256": SHA,
        "runtime_image_digest": "sha256:" + SHA,
        "dependency_lock_sha256": SHA,
        "evaluator_implementation_sha256": SHA,
        "metrics_schema_version": 5,
        "seed_protocol": "episode-manifest-v1",
        "n_envs": 1,
        "episodes": 1,
        "max_steps": 1000,
        "deterministic": False,
        "action_sampling": "stochastic",
        "environment": {"provider": "test", "arguments": {"game": "Test-v0"}},
        "observation": {"shape": [4]},
        "action": {"kind": "discrete", "n": 2},
        "preprocessing": {"version": 1},
        "reward": {"program_sha256": SHA},
        "events": {"version": 1},
        "starts": {"kind": "manifest"},
        "termination": {"max_steps": 1000},
        "assets": {"rom_sha256": SHA},
    }
    value.update(overrides)
    return value


def run_dimensions(**overrides):
    value = {
        "goal_slug": "goal",
        "canonical_goal_sha256": SHA,
        "effective_goal_contract_sha256": SHA,
        "target_scope": "level",
        "reward_program_name": "default",
        "reward_program_revision": 1,
        "reward_program_sha256": SHA,
        "recipe_slug": "recipe",
        "resolved_config_sha256": SHA,
        "environment_id": "Test-v0",
        "environment_provider": "test",
        "environment_contract_sha256": SHA,
        "training_backend": "sb3",
        "runtime_image_digest": "sha256:" + SHA,
        "dependency_lock_sha256": SHA,
        "source_sha": SHA,
        "metrics_schema_version": 5,
        "rank_metric": "train/episode/return/mean",
        "rank_direction": "max",
    }
    value.update(overrides)
    return value


class CanonicalTelemetryTests(unittest.TestCase):
    def event(self, sequence: int, payload=None) -> CanonicalEvent:
        return CanonicalEvent(
            producer=ProducerKey(7, 1, 0, "learner"),
            source_sequence=sequence,
            event_id=f"event-{sequence}",
            kind="history",
            global_step=sequence * 10,
            payload=payload or {"loss": 0.5, "count": sequence},
        )

    def test_segment_is_byte_deterministic_and_round_trips(self) -> None:
        first = build_canonical_segment([self.event(1), self.event(2)])
        second = build_canonical_segment([self.event(1), self.event(2)])

        self.assertEqual(first.payload, second.payload)
        self.assertEqual(first.uncompressed_sha256, second.uncompressed_sha256)
        self.assertEqual(len(decode_canonical_segment(first.payload)), 2)
        self.assertTrue(gzip.decompress(first.payload).endswith(b"\n"))

    def test_segment_rejects_sequence_gap(self) -> None:
        with self.assertRaisesRegex(TelemetryContractError, "sequence gap"):
            build_canonical_segment([self.event(1), self.event(3)])

    def test_unregistered_or_nonfinite_values_fail_closed(self) -> None:
        with self.assertRaisesRegex(TelemetryContractError, "unregistered"):
            build_canonical_segment([self.event(1, {"bad": object()})])
        with self.assertRaisesRegex(TelemetryContractError, "non-finite"):
            build_canonical_segment([self.event(1, {"bad": float("nan")})])

    def test_wandb_row_has_stable_identity_and_explicit_ordinal(self) -> None:
        rows = normalize_wandb_rows(self.event(1), first_ordinal=9)
        self.assertEqual(rows[0]["ordinal"], 9)
        self.assertEqual(rows[0]["payload"]["_rlab_output_ordinal"], 9)
        self.assertIn("event-1", rows[0]["stable_key"])

    def test_multirow_metric_batch_gets_contiguous_output_ordinals(self) -> None:
        event = self.event(
            1,
            {
                "frames": [
                    {
                        "kind": "history",
                        "global_step": 10,
                        "payload": {"loss": 1.0},
                    },
                    {
                        "kind": "history",
                        "global_step": 20,
                        "payload": {"loss": 0.5},
                    },
                ]
            },
        )
        event = CanonicalEvent(
            producer=event.producer,
            source_sequence=event.source_sequence,
            event_id=event.event_id,
            kind="metric_batch",
            payload=event.payload,
        )
        rows = normalize_wandb_rows(event, first_ordinal=7)
        self.assertEqual([7, 8], [row["ordinal"] for row in rows])
        self.assertEqual([10, 20], [row["payload"]["global_step"] for row in rows])

    def test_evaluation_contract_mismatch_matrix_fails_exactly(self) -> None:
        base = eval_contract()
        variants = {
            "provider": {
                **base,
                "environment": {
                    **base["environment"],
                    "provider": "other-provider",
                },
            },
            "preprocessing": {**base, "preprocessing": {"version": 2}},
            "reward": {**base, "reward": {"program_sha256": "b" * 64}},
            "runtime": {**base, "runtime_image_digest": "sha256:" + "b" * 64},
            "evaluator": {**base, "evaluator_implementation_sha256": "b" * 64},
            "rom": {**base, "assets": {"rom_sha256": "b" * 64}},
            "action": {**base, "action_sampling": "deterministic"},
        }
        for label, changed in variants.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(TelemetryContractError, "mismatch"):
                    require_exact_contract_match(base, changed, label="eval")

    def test_integrity_requires_exact_sets_coverage_and_archive_receipts(self) -> None:
        result = reduce_integrity(
            IntegrityInputs(
                classification="intact_with_proof",
                expected_obligations={"train": "complete"},
                realized_obligations={"train": "complete"},
                producer_final_claims={0: (2, SHA)},
                archived_coverage={0: 2},
                no_more_producers=True,
                durability_policy="queued_dual_r2_v1",
                required_archive_receipts=2,
                observed_archive_receipts=2,
                recovery_pending=False,
            )
        )
        self.assertTrue(result.exact)
        self.assertTrue(result.cleanup_eligible)

        damaged = reduce_integrity(
            IntegrityInputs(
                classification="legacy_unknown",
                expected_obligations={"train": "complete"},
                realized_obligations={},
                producer_final_claims={},
                archived_coverage={},
                no_more_producers=False,
                durability_policy="queued_dual_r2_v1",
                required_archive_receipts=2,
                observed_archive_receipts=0,
                recovery_pending=True,
            )
        )
        self.assertFalse(damaged.exact)
        self.assertIn("expected_obligation_set_mismatch", damaged.reasons)

    def test_eval_scope_rejects_deterministic_or_incomplete_evidence(self) -> None:
        with self.assertRaisesRegex(TelemetryContractError, "stochastic"):
            build_eval_scope_exact(
                checkpoint={"sha256": SHA, "durability_receipts": [{"copy": "primary"}]},
                evaluation_contract=eval_contract(
                    deterministic=True, action_sampling="deterministic"
                ),
                episode_manifest={"episodes": [{"seed": 1}]},
                results=[{"passed": True}],
                acceptance_rule={"all": "passed"},
                execution_key="exec",
                attestation={"service": "eval"},
            )

    def test_training_success_scope_is_restricted_to_declared_exception(self) -> None:
        contract = {
            "acceptance_mode": "first_training_success",
            "checkpoint_eval_backend": "none",
            "deterministic_search_workflow": True,
            "canonical_goal_sha256": SHA,
            "effective_goal_contract_sha256": SHA,
            "recipe_sha256": SHA,
            "environment_contract_sha256": SHA,
            "reward_program_sha256": SHA,
            "runtime_image_digest": "sha256:" + SHA,
        }
        scope = build_training_success_scope_exact(
            contract=contract,
            success_event={
                "event_id": "success",
                "producer_ordinal": 0,
                "source_sequence": 4,
                "episode_id": "episode-1",
                "global_step": 99,
            },
            policy_artifact={"sha256": SHA, "durability_receipts": [{"copy": "primary"}]},
        )
        self.assertEqual(scope["scope_kind"], "training_success_scope_exact")

        with self.assertRaisesRegex(TelemetryContractError, "declared deterministic"):
            build_training_success_scope_exact(
                contract={**contract, "deterministic_search_workflow": False},
                success_event=scope["success_event"],
                policy_artifact=scope["policy_artifact"],
            )

    def test_run_facts_require_exact_comparability_and_complete_cohort(self) -> None:
        integrity = {"classification": "intact_with_proof", "exact": True}
        one = build_run_final_exact(
            archive_root_sha256=SHA,
            dimensions=run_dimensions(),
            metrics={"peak": 10.0},
            seed=1,
            cohort_manifest={"expected_seeds": [1, 2]},
            integrity=integrity,
        )
        two = build_run_final_exact(
            archive_root_sha256=SHA,
            dimensions=run_dimensions(),
            metrics={"peak": 12.0},
            seed=2,
            cohort_manifest={"expected_seeds": [1, 2]},
            integrity=integrity,
        )
        require_comparable_run_facts([one, two], require_complete_cohort=True)
        with self.assertRaises(TelemetryIntegrityError):
            require_comparable_run_facts([one], require_complete_cohort=True)
        other = build_run_final_exact(
            archive_root_sha256=SHA,
            dimensions=run_dimensions(reward_program_sha256="b" * 64),
            metrics={"peak": 9.0},
            seed=2,
            cohort_manifest={"expected_seeds": [1, 2]},
            integrity=integrity,
        )
        with self.assertRaisesRegex(TelemetryContractError, "not contract-comparable"):
            require_comparable_run_facts([one, other], require_complete_cohort=False)

    def test_fsync_writer_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "root.json"
            write_fsync(path, b"payload\n")
            self.assertEqual(path.read_bytes(), b"payload\n")


if __name__ == "__main__":
    unittest.main()
