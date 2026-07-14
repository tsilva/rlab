from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from rlab import runtime_refs
from rlab.runtime_contract import (
    RUNTIME_DESCRIPTOR_SCHEMA_VERSION,
    runtime_contract,
    train_config_contract_sha256,
    validate_config_payload,
)


RUNTIME_IMAGE_REF = "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:" + "a" * 64


def release_payload(*, source_sha: str = "source") -> dict:
    return {
        "schema_version": RUNTIME_DESCRIPTOR_SCHEMA_VERSION,
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "digest": "sha256:" + "a" * 64,
        "source_sha": source_sha,
        "workflow_run_id": "123",
    }


def modal_readiness_payload(*, source_sha: str = "source") -> dict:
    contract_sha = train_config_contract_sha256()
    app_name = "rlab-eval-" + "a" * 12
    return {
        "schema_version": 1,
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "source_sha": source_sha,
        "modal_app_name": app_name,
        "startup_probe": {
            "schema_version": 1,
            "app_name": app_name,
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "source_sha": source_sha,
            "train_config_contract_sha256": contract_sha,
        },
        "workflow_run_id": "123",
    }


class RuntimeContractTests(unittest.TestCase):
    def test_contract_is_stable_and_runtime_reports_source(self) -> None:
        with mock.patch.dict(os.environ, {"RLAB_SOURCE_SHA": "abc123"}):
            receipt = runtime_contract(runtime_image_ref=RUNTIME_IMAGE_REF)

        self.assertEqual(receipt["source_sha"], "abc123")
        self.assertEqual(receipt["runtime_image_ref"], RUNTIME_IMAGE_REF)
        self.assertEqual(len(receipt["train_config_contract_sha256"]), 64)

    def test_runtime_payload_validation_accepts_queue_execution_fields(self) -> None:
        receipt = validate_config_payload(
            {
                "batch_id": "bx0123456789abcdef",
                "machine": "beast-3",
                "queue_train_job_id": 1,
                "run_name": "bx0123456789abcdef-base-s123-20000101T000000Z",
                "runtime_image_ref": RUNTIME_IMAGE_REF,
                "seed": 123,
                "wandb_group": "bx0123456789abcdef",
                "wandb_run_id": "rlab-test",
            }
        )

        self.assertTrue(receipt["validated"])
        self.assertEqual(receipt["validated_field_count"], 8)

    def test_image_receipt_rejects_legacy_and_digest_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "legacy combined runtime artifacts"):
            runtime_refs.runtime_release_from_payload(
                {"runtime_image_ref": RUNTIME_IMAGE_REF, "source_sha": "source"},
                label="release",
                expected_source_sha="source",
            )
        payload = release_payload()
        payload["digest"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            runtime_refs.runtime_release_from_payload(
                payload,
                label="release",
                expected_source_sha="source",
            )

    def test_receipts_require_exact_source_and_modal_probe_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_sha mismatch"):
            runtime_refs.runtime_release_from_payload(
                release_payload(source_sha="old"),
                label="release",
                expected_source_sha="new",
            )
        payload = modal_readiness_payload()
        payload["startup_probe"]["runtime_image_ref"] = (
            "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:" + "c" * 64
        )
        with self.assertRaisesRegex(ValueError, "startup_probe.runtime_image_ref"):
            runtime_refs.modal_readiness_from_payload(
                payload,
                label="Modal readiness",
                expected_source_sha="source",
                expected_runtime_image_ref=RUNTIME_IMAGE_REF,
            )
        payload = modal_readiness_payload()
        payload["startup_probe"]["train_config_contract_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "startup_probe.train_config_contract_sha256"):
            runtime_refs.modal_readiness_from_payload(
                payload,
                label="Modal readiness",
                expected_source_sha="source",
                expected_runtime_image_ref=RUNTIME_IMAGE_REF,
            )

    def test_clean_git_source_rejects_dirty_worktree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with mock.patch("subprocess.run") as run:
                run.return_value = mock.Mock(returncode=0, stdout=" M file.py\n", stderr="")
                with self.assertRaisesRegex(RuntimeError, "clean worktree"):
                    runtime_refs.clean_git_source_sha(root)


if __name__ == "__main__":
    unittest.main()
