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
SOURCE_SHA = "1" * 40
BUILD_SOURCE_SHA = "2" * 40


def release_payload(*, source_sha: str = SOURCE_SHA) -> dict:
    return {
        "schema_version": RUNTIME_DESCRIPTOR_SCHEMA_VERSION,
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "digest": "sha256:" + "a" * 64,
        "source_sha": source_sha,
        "runtime_input_sha256": "d" * 64,
        "runtime_build_source_sha": BUILD_SOURCE_SHA,
        "tags": ["runtime-" + "d" * 64],
        "uv_lock_sha256": "e" * 64,
        "base_images": {
            "dependencies": "docker:ghcr.io/tsilva/rlab/rlab-train-dependencies@sha256:"
            + "f" * 64
        },
        "workflow_run_id": "123",
    }


def modal_readiness_payload(*, source_sha: str = SOURCE_SHA) -> dict:
    contract_sha = train_config_contract_sha256()
    app_name = "rlab-eval-" + "a" * 12
    return {
        "schema_version": 2,
        "runtime_image_ref": RUNTIME_IMAGE_REF,
        "source_sha": source_sha,
        "runtime_input_sha256": "d" * 64,
        "runtime_build_source_sha": BUILD_SOURCE_SHA,
        "modal_app_name": app_name,
        "startup_probe": {
            "schema_version": 1,
            "app_name": app_name,
            "runtime_image_ref": RUNTIME_IMAGE_REF,
            "source_sha": BUILD_SOURCE_SHA,
            "runtime_build_source_sha": BUILD_SOURCE_SHA,
            "runtime_input_sha256": "d" * 64,
            "train_config_contract_sha256": contract_sha,
        },
        "workflow_run_id": "123",
    }


class RuntimeContractTests(unittest.TestCase):
    def test_contract_is_stable_and_runtime_reports_source(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"RLAB_SOURCE_SHA": "abc123", "RLAB_RUNTIME_INPUT_SHA256": "d" * 64},
        ):
            receipt = runtime_contract(runtime_image_ref=RUNTIME_IMAGE_REF)

        self.assertEqual(receipt["source_sha"], "abc123")
        self.assertEqual(receipt["runtime_build_source_sha"], "abc123")
        self.assertEqual(receipt["runtime_input_sha256"], "d" * 64)
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
                "training_backend": {"id": "sb3.ppo", "config": {}},
                "wandb_group": "bx0123456789abcdef",
                "wandb_run_id": "rlab-test",
            }
        )

        self.assertTrue(receipt["validated"])
        self.assertEqual(receipt["validated_field_count"], 9)

    def test_image_receipt_rejects_legacy_and_digest_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema_version must be 3 or 4"):
            runtime_refs.runtime_release_from_payload(
                {"runtime_image_ref": RUNTIME_IMAGE_REF, "source_sha": SOURCE_SHA},
                label="release",
                expected_source_sha=SOURCE_SHA,
            )
        payload = release_payload()
        payload["digest"] = "sha256:" + "b" * 64
        with self.assertRaisesRegex(ValueError, "digest does not match"):
            runtime_refs.runtime_release_from_payload(
                payload,
                label="release",
                expected_source_sha=SOURCE_SHA,
            )

    def test_receipts_require_exact_source_and_modal_probe_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "source_sha mismatch"):
            runtime_refs.runtime_release_from_payload(
                release_payload(source_sha="3" * 40),
                label="release",
                expected_source_sha="4" * 40,
            )
        payload = modal_readiness_payload()
        payload["startup_probe"]["runtime_image_ref"] = (
            "docker:ghcr.io/tsilva/rlab/rlab-train@sha256:" + "c" * 64
        )
        with self.assertRaisesRegex(ValueError, "startup_probe.runtime_image_ref"):
            runtime_refs.modal_readiness_from_payload(
                payload,
                label="Modal readiness",
                expected_source_sha=SOURCE_SHA,
                expected_runtime_image_ref=RUNTIME_IMAGE_REF,
                expected_runtime_input_sha256="d" * 64,
                expected_runtime_build_source_sha=BUILD_SOURCE_SHA,
            )
        payload = modal_readiness_payload()
        payload["startup_probe"]["train_config_contract_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "startup_probe.train_config_contract_sha256"):
            runtime_refs.modal_readiness_from_payload(
                payload,
                label="Modal readiness",
                expected_source_sha=SOURCE_SHA,
                expected_runtime_image_ref=RUNTIME_IMAGE_REF,
                expected_runtime_input_sha256="d" * 64,
                expected_runtime_build_source_sha=BUILD_SOURCE_SHA,
            )

    def test_version_three_receipt_remains_readable(self) -> None:
        payload = release_payload()
        payload["schema_version"] = 3
        payload.pop("runtime_input_sha256")
        payload.pop("runtime_build_source_sha")

        release = runtime_refs.runtime_release_from_payload(
            payload,
            label="legacy release",
            expected_source_sha=SOURCE_SHA,
        )

        self.assertEqual(release.schema_version, 3)
        self.assertEqual(release.runtime_input_sha256, "")
        self.assertEqual(release.runtime_build_source_sha, SOURCE_SHA)

    def test_version_one_modal_readiness_remains_readable_for_version_three_image(self) -> None:
        payload = modal_readiness_payload()
        payload["schema_version"] = 1
        payload.pop("runtime_input_sha256")
        payload.pop("runtime_build_source_sha")
        payload["startup_probe"].pop("runtime_input_sha256")
        payload["startup_probe"].pop("runtime_build_source_sha")
        payload["startup_probe"]["source_sha"] = SOURCE_SHA

        readiness = runtime_refs.modal_readiness_from_payload(
            payload,
            label="legacy Modal readiness",
            expected_source_sha=SOURCE_SHA,
            expected_runtime_image_ref=RUNTIME_IMAGE_REF,
        )

        self.assertEqual(readiness.schema_version, 1)
        self.assertEqual(readiness.runtime_build_source_sha, SOURCE_SHA)

    def test_version_four_receipt_rejects_invalid_runtime_identity_fields(self) -> None:
        cases = {
            "runtime fingerprint": ("runtime_input_sha256", "short", "runtime_input_sha256"),
            "build source": ("runtime_build_source_sha", "not-a-sha", "runtime_build_source_sha"),
            "dependency": (
                "base_images",
                {"dependencies": "docker:mutable"},
                "dependency image identity",
            ),
            "workflow": ("workflow_run_id", "", "workflow_run_id"),
        }
        for label, (field, value, error) in cases.items():
            with self.subTest(label=label):
                payload = release_payload()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, error):
                    runtime_refs.runtime_release_from_payload(
                        payload,
                        label="release",
                        expected_source_sha=SOURCE_SHA,
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
