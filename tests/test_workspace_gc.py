from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rlab.workspace_gc import DurabilityReceipt


def receipt(**overrides) -> DurabilityReceipt:
    values = {
        "train_job_id": 1,
        "ledger_id": 1,
        "object_kind": "model",
        "object_uri": "s3://bucket/sha256/abc/model.zip",
        "object_version": "version-1",
        "size_bytes": 10,
        "sha256": "a" * 64,
        "full_read_verified_at": datetime.now(UTC),
        "verifier_identity": "workspace-proof-reducer",
        "policy_scope": "s3://bucket/sha256/",
        "policy_sha256": "b" * 64,
        "non_expiring_write_once": True,
        "runtime_delete_denied": True,
        "runtime_overwrite_denied": True,
        "storage_root_nonoverlap_sha256": "c" * 64,
        "receipt_json": {"verified": True},
    }
    values.update(overrides)
    return DurabilityReceipt(**values)


def test_durability_receipt_requires_full_immutable_policy() -> None:
    receipt().validate()
    with pytest.raises(ValueError, match="non-expiring"):
        receipt(non_expiring_write_once=False).validate()
    with pytest.raises(ValueError, match="delete and overwrite"):
        receipt(runtime_delete_denied=False).validate()


def test_durability_receipt_rejects_unbound_hashes() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        receipt(policy_sha256="metadata-only").validate()

