from __future__ import annotations

import dataclasses

import pytest

from rlab.workspace_contract import (
    CleanupBatchEnvelope,
    CleanupRowEnvelope,
    WorkspaceContractError,
    build_workspace_manifest,
    canonical_json_bytes,
    exact_bind_mount,
)


PAYLOAD_SHA = "a" * 64


def manifest(**overrides):
    values = {
        "machine": "beast-test",
        "launch_id": "train-12",
        "generation": 1,
        "reservation_nonce": "n" * 32,
        "payload_sha256": PAYLOAD_SHA,
        "host_root": "/srv/rlab",
        "payloads_root": "/srv/rlab/payloads",
        "outputs_root": "/srv/rlab/outputs",
        "protected_metadata_root": "/srv/rlab/workspace-meta",
        "logs_root": "/srv/rlab/logs",
        "shared_env_path": "/srv/rlab/.env.runner",
        "rom_cache_root": "/srv/rom-cache",
        "container_payloads_root": "/input/payloads",
        "container_outputs_root": "/output",
    }
    values.update(overrides)
    return build_workspace_manifest(**values)


def test_manifest_names_exact_targets_and_is_deterministic() -> None:
    first = manifest()
    second = manifest()
    assert first == second
    assert first.digest == second.digest
    assert first.deletion_targets == (
        "/srv/rlab/payloads/train-12.json",
        "/srv/rlab/payloads/train-12.env",
        "/srv/rlab/outputs/train-12",
        "/srv/rlab/workspace-meta/train-12.ownership.json",
        "/srv/rlab/workspace-meta/train-12.reservation.json",
    )
    assert canonical_json_bytes(first.as_dict()) == canonical_json_bytes(second.as_dict())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("payloads_root", "/"),
        ("payloads_root", "relative"),
        ("payloads_root", "/srv/rlab/outputs/inside"),
        ("protected_metadata_root", "/srv/rlab/payloads/meta"),
        ("container_outputs_root", "/input/payloads/output"),
        ("shared_env_path", "/srv/rlab/payloads/.env"),
    ],
)
def test_manifest_rejects_dangerous_or_overlapping_roots(field: str, value: str) -> None:
    with pytest.raises(WorkspaceContractError):
        manifest(**{field: value})


def test_manifest_rejects_unsafe_launch_id() -> None:
    with pytest.raises(WorkspaceContractError, match="launch_id"):
        manifest(launch_id="../../etc")


def test_exact_bind_mount_is_nonrecursive_and_exact() -> None:
    assert exact_bind_mount("/srv/a.json", "/input/a.json", readonly=True) == (
        "type=bind,src=/srv/a.json,dst=/input/a.json,bind-recursive=disabled,readonly"
    )


def test_cleanup_batch_digest_binds_order_and_every_row() -> None:
    row = CleanupRowEnvelope(
        row_id=1,
        cleanup_attempt_id="attempt-1",
        generation=1,
        manifest_sha256="a" * 64,
        prepare_receipt_sha256="b" * 64,
        starting_cursor_sha256="c" * 64,
    )
    other = dataclasses.replace(row, row_id=2, cleanup_attempt_id="attempt-2")
    batch = CleanupBatchEnvelope(
        protocol_version=1,
        batch_id="batch-1",
        machine="beast-test",
        helper_revision="helper-v1",
        key_revision="key-v1",
        control_revision=1,
        epoch=1,
        boot_id="boot-1",
        monotonic_deadline_ns=10_000,
        rows=(row, other),
    )
    batch.validate()
    assert batch.digest != dataclasses.replace(batch, rows=(other, row)).digest
    with pytest.raises(WorkspaceContractError, match="duplicate"):
        dataclasses.replace(batch, rows=(row, row)).validate()
