from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from unittest import mock

import pytest

from rlab.workspace_contract import (
    CleanupBatchEnvelope,
    CleanupRowEnvelope,
    build_workspace_manifest,
    canonical_json_bytes,
    sha256_json,
)
from rlab.workspace_helper import (
    WorkspaceHelperError,
    drain_zero_receipt,
    delete_commit,
    delete_prepare,
    install_key_policy,
    recover_reservation_intent,
    release_generation_env_intent,
    release_reservation_intent,
    reserve_workspace,
    reserve_generation_env,
    reservation_intent_path,
    unlink_attempt_env,
)


def manifest(root: Path, payload: bytes = b'{"job": 1}\n'):
    host = root / "host"
    return build_workspace_manifest(
        machine="local-test",
        launch_id="train-1",
        generation=1,
        reservation_nonce="n" * 32,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        host_root=str(host),
        payloads_root=str(host / "payloads"),
        outputs_root=str(host / "outputs"),
        protected_metadata_root=str(host / "workspace-meta"),
        logs_root=str(host / "logs"),
        shared_env_path=str(host / ".env.runner"),
        rom_cache_root=str(root / "rom-cache"),
        container_payloads_root="/input/payloads",
        container_outputs_root="/output",
    )


def test_reserve_is_exact_durable_and_secret_free(tmp_path: Path) -> None:
    payload = b'{"job": 1}\n'
    contract = manifest(tmp_path, payload)
    receipt = reserve_workspace(
        contract,
        payload=payload,
        attempt_env={"ATTEMPT_TOKEN": "super-secret", "ATTEMPT_ID": "train-1"},
    )
    assert Path(contract.payload_path).read_bytes() == payload
    assert Path(contract.env_path).read_text() == (
        "ATTEMPT_ID=train-1\nATTEMPT_TOKEN=super-secret\n"
    )
    assert (os.stat(contract.env_path).st_mode & 0o777) == 0o600
    assert Path(contract.output_path).is_dir()
    assert Path(contract.ownership_marker_path).is_file()
    assert Path(contract.reservation_receipt_path).is_file()
    assert Path(reservation_intent_path(contract)).is_file()
    assert "super-secret" not in json.dumps(receipt)
    assert "super-secret" not in Path(contract.reservation_receipt_path).read_text()

    release_reservation_intent(contract, receipt_sha256=receipt["receipt_sha256"])
    assert not Path(reservation_intent_path(contract)).exists()


def test_env_unlink_refuses_replacement_inode(tmp_path: Path) -> None:
    contract = manifest(tmp_path)
    receipt = reserve_workspace(
        contract,
        payload=b'{"job": 1}\n',
        attempt_env={"ATTEMPT_TOKEN": "secret"},
    )
    expected = receipt["leaves"]["env"]
    Path(contract.env_path).unlink()
    Path(contract.env_path).write_text("ATTEMPT_TOKEN=replaced\n")
    with pytest.raises(WorkspaceHelperError, match="inode"):
        unlink_attempt_env(contract, expected)


def test_env_unlink_is_idempotent(tmp_path: Path) -> None:
    contract = manifest(tmp_path)
    receipt = reserve_workspace(
        contract,
        payload=b'{"job": 1}\n',
        attempt_env={"ATTEMPT_TOKEN": "secret"},
    )
    expected = receipt["leaves"]["env"]
    assert unlink_attempt_env(contract, expected)["status"] == "unlinked"
    assert unlink_attempt_env(contract, expected)["status"] == "absent"


def test_partial_intent_recovery_removes_secret_and_owned_leaves(tmp_path: Path) -> None:
    contract = manifest(tmp_path)
    reserve_workspace(
        contract,
        payload=b'{"job": 1}\n',
        attempt_env={"ATTEMPT_TOKEN": "secret"},
    )
    Path(contract.reservation_receipt_path).unlink()
    result = recover_reservation_intent(contract)
    assert result["status"] == "cleaned"
    assert not Path(contract.env_path).exists()
    assert not Path(contract.payload_path).exists()
    assert not Path(contract.output_path).exists()
    assert not Path(contract.ownership_marker_path).exists()
    assert not Path(reservation_intent_path(contract)).exists()


def test_prepare_requires_credential_env_absence(tmp_path: Path) -> None:
    contract = manifest(tmp_path)
    reserve_workspace(
        contract,
        payload=b'{"job": 1}\n',
        attempt_env={"ATTEMPT_TOKEN": "secret"},
    )
    with pytest.raises(WorkspaceHelperError, match="credential env"):
        delete_prepare(
            contract,
            cleanup_row_id=1,
            cleanup_attempt_id="attempt-1",
            control_revision=1,
            cursor_sha256=sha256_json({}),
        )


def test_signed_commit_deletes_exact_targets_and_rejects_replay(tmp_path: Path) -> None:
    contract = manifest(tmp_path)
    reservation = reserve_workspace(
        contract,
        payload=b'{"job": 1}\n',
        attempt_env={"ATTEMPT_TOKEN": "secret"},
    )
    release_reservation_intent(
        contract, receipt_sha256=reservation["receipt_sha256"]
    )
    unlink_attempt_env(contract, reservation["leaves"]["env"])
    output = Path(contract.output_path)
    (output / "nested").mkdir()
    (output / "nested" / "model.zip").write_bytes(b"model")
    cursor_sha = sha256_json({"phase": "content", "target_index": 0, "stack": []})
    prepare = delete_prepare(
        contract,
        cleanup_row_id=1,
        cleanup_attempt_id="attempt-1",
        control_revision=1,
        cursor_sha256=cursor_sha,
    )

    private_key = tmp_path / "signer-private.pem"
    public_key = tmp_path / "signer-public.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", private_key],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", private_key, "-pubout", "-out", public_key],
        check=True,
        capture_output=True,
    )
    policy = {
        "protocol_version": 1,
        "keys": {
            "key-v1": {
                "state": "active",
                "public_key_path": str(public_key),
                "public_key_sha256": __import__("hashlib").sha256(public_key.read_bytes()).hexdigest(),
            }
        }
    }
    policy["policy_sha256"] = sha256_json(policy)
    (Path(contract.roots["protected_metadata_root"]) / ".workspace-key-policy.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    row = CleanupRowEnvelope(
        row_id=1,
        cleanup_attempt_id="attempt-1",
        generation=1,
        manifest_sha256=contract.digest,
        prepare_receipt_sha256=prepare["prepare_receipt_sha256"],
        starting_cursor_sha256=cursor_sha,
    )
    envelope = CleanupBatchEnvelope(
        protocol_version=1,
        batch_id="batch-1",
        machine=contract.machine,
        helper_revision="workspace-helper-v1",
        key_revision="key-v1",
        control_revision=1,
        epoch=1,
        boot_id=prepare["boot_id"],
        monotonic_deadline_ns=prepare["deadline_monotonic_ns"],
        rows=(row,),
    )
    signature = subprocess.run(
        ["openssl", "dgst", "-sha256", "-sign", private_key],
        input=canonical_json_bytes(envelope.as_dict()),
        check=True,
        capture_output=True,
    ).stdout
    import base64

    encoded_signature = base64.b64encode(signature).decode("ascii")
    with mock.patch("rlab.workspace_helper._assert_no_container_mounts"):
        result = delete_commit(
            envelope=envelope,
            signature_base64=encoded_signature,
            manifests={1: contract},
        )
    assert result["status"] == "complete"
    assert all(not Path(target).exists() for target in contract.deletion_targets)
    assert Path(result["completed"][0]["host_deleted_journal_path"]).is_file()
    with (
        mock.patch("rlab.workspace_helper._assert_no_container_mounts"),
        pytest.raises(WorkspaceHelperError, match="epoch"),
    ):
        delete_commit(
            envelope=envelope,
            signature_base64=encoded_signature,
            manifests={1: contract},
        )


def test_key_policy_is_digest_bound_and_drain_inventory_detects_residue(
    tmp_path: Path,
) -> None:
    contract = manifest(tmp_path)
    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
    subprocess.run(
        [
            "openssl",
            "genpkey",
            "-algorithm",
            "RSA",
            "-pkeyopt",
            "rsa_keygen_bits:2048",
            "-out",
            private_key,
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["openssl", "pkey", "-in", private_key, "-pubout", "-out", public_key],
        check=True,
        capture_output=True,
    )
    receipt = install_key_policy(
        protected_metadata_root=contract.roots["protected_metadata_root"],
        key_revision="key-v1",
        public_key=public_key.read_bytes(),
    )
    policy = json.loads(Path(receipt["policy_path"]).read_text(encoding="utf-8"))
    supplied = policy.pop("policy_sha256")
    assert supplied == sha256_json(policy)

    intent = Path(contract.roots["protected_metadata_root"]) / "orphan.reservation-intent.json"
    intent.write_text("{}", encoding="utf-8")
    with (
        mock.patch("rlab.workspace_helper._container_absent", return_value=True),
        mock.patch("rlab.workspace_helper._assert_no_container_mounts"),
    ):
        inventory = drain_zero_receipt(
            protected_metadata_root=contract.roots["protected_metadata_root"],
            manifests=(),
            exact_container_names=(),
            exact_protected_paths=(),
            receipt_nonce="n" * 32,
        )
    assert inventory["counts"]["discovered_protected_journals"] == 1


def test_recovery_env_generation_is_exclusive_receipted_and_identity_unlinked(
    tmp_path: Path,
) -> None:
    contract = manifest(tmp_path)
    initial = reserve_workspace(
        contract,
        payload=b'{"job": 1}\n',
        attempt_env={"ATTEMPT_TOKEN": "initial-secret"},
    )
    release_reservation_intent(contract, receipt_sha256=initial["receipt_sha256"])
    unlink_attempt_env(contract, initial["leaves"]["env"])

    receipt = reserve_generation_env(
        contract,
        container_generation=2,
        attempt_env={"WANDB_API_KEY": "recovery-secret"},
    )
    assert "recovery-secret" not in json.dumps(receipt)
    assert Path(contract.env_path).read_text(encoding="utf-8") == (
        "WANDB_API_KEY=recovery-secret\n"
    )
    release_generation_env_intent(
        contract,
        container_generation=2,
        receipt_sha256=receipt["receipt_sha256"],
    )
    unlink_attempt_env(contract, receipt["env_identity"])
    assert not Path(contract.env_path).exists()
