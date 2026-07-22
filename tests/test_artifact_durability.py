from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from rlab.artifact_durability import (
    ArtifactDurabilityError,
    ArtifactDurabilityPolicy,
    verify_remote_object,
)


class FakeS3:
    def __init__(self, payload: bytes, *, version_id: str = "version-1") -> None:
        self.payload = payload
        self.version_id = version_id

    def get_object(self, **kwargs):
        return {"Body": io.BytesIO(self.payload), "VersionId": self.version_id}


class ArtifactDurabilityTests(unittest.TestCase):
    def policy(self, root: Path) -> ArtifactDurabilityPolicy:
        document = {
            "protocol_version": 1,
            "endpoint_url": "https://objects.example.test",
            "bucket": "durable",
            "prefix": "rlab",
            "policy_scope": "durable/rlab",
            "verifier_identity": "workspace-controller-readonly",
            "non_expiring_write_once": True,
            "runtime_delete_denied": True,
            "runtime_overwrite_denied": True,
            "runtime_policy_admin_denied": True,
            "content_addressed_keys": True,
            "version_identity_required": True,
            "preflight_receipt_sha256": "a" * 64,
        }
        path = root / "policy.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        path.chmod(0o600)
        return ArtifactDurabilityPolicy.load(path)

    def test_full_read_receipt_binds_object_version_and_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = self.policy(Path(temporary))
            payload = b"durable artifact bytes"
            digest = hashlib.sha256(payload).hexdigest()
            receipt = verify_remote_object(
                policy=policy,
                client=FakeS3(payload),
                train_job_id=1,
                ledger_id=2,
                object_kind="model",
                object_uri=f"s3://durable/rlab/run/sha256/{digest}/checkpoint.zip",
                expected_sha256=digest,
                workspace_evidence={"manifest_sha256": "b" * 64},
            )
            self.assertEqual(receipt.sha256, digest)
            self.assertEqual(receipt.object_version, "version-1")
            self.assertEqual(receipt.size_bytes, len(payload))

    def test_non_content_addressed_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = self.policy(Path(temporary))
            payload = b"artifact"
            digest = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(ArtifactDurabilityError, "not bound"):
                verify_remote_object(
                    policy=policy,
                    client=FakeS3(payload),
                    train_job_id=1,
                    ledger_id=1,
                    object_kind="model",
                    object_uri="s3://durable/rlab/run/checkpoint.zip",
                    expected_sha256=digest,
                    workspace_evidence={},
                )

    def test_missing_version_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            policy = self.policy(Path(temporary))
            payload = b"artifact"
            digest = hashlib.sha256(payload).hexdigest()
            with self.assertRaisesRegex(ArtifactDurabilityError, "version identity"):
                verify_remote_object(
                    policy=policy,
                    client=FakeS3(payload, version_id=""),
                    train_job_id=1,
                    ledger_id=1,
                    object_kind="model",
                    object_uri=f"s3://durable/rlab/run/sha256/{digest}/checkpoint.zip",
                    expected_sha256=digest,
                    workspace_evidence={},
                )


if __name__ == "__main__":
    unittest.main()
