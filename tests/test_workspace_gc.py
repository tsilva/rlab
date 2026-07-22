from __future__ import annotations

from datetime import UTC, datetime

import pytest

from rlab.job_queue import WORKSPACE_SCHEMA_SQL
from rlab.workspace_gc import DurabilityReceipt, record_workspace_qualification_receipt


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


def test_qualification_receipt_fails_before_database_when_evidence_is_incomplete() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        record_workspace_qualification_receipt(
            None,
            schedule_id="schedule",
            machine="beast-3",
            machine_control_revision=1,
            evidence={"paired_blocks": 5},
        )


def test_schema_contains_database_enforced_pause_drain_and_promotion_gates() -> None:
    assert "workspace_train_insert_gate" in WORKSPACE_SCHEMA_SQL
    assert "workspace_train_claim_gate" in WORKSPACE_SCHEMA_SQL
    assert "machine_controls_drain_guard" in WORKSPACE_SCHEMA_SQL
    assert "workspace_promotion_receipts" in WORKSPACE_SCHEMA_SQL
    assert "workspace_qualification_receipts" in WORKSPACE_SCHEMA_SQL
