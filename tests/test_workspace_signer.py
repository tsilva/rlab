from __future__ import annotations

import base64
import subprocess
from pathlib import Path

from rlab.workspace_contract import (
    CleanupBatchEnvelope,
    CleanupRowEnvelope,
    canonical_json_bytes,
)
from rlab.workspace_signer import sign_envelope


def test_signer_binds_canonical_envelope(tmp_path: Path) -> None:
    private_key = tmp_path / "private.pem"
    public_key = tmp_path / "public.pem"
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
    row = CleanupRowEnvelope(
        row_id=1,
        cleanup_attempt_id="attempt-1",
        generation=1,
        manifest_sha256="a" * 64,
        prepare_receipt_sha256="b" * 64,
        starting_cursor_sha256="c" * 64,
    )
    envelope = CleanupBatchEnvelope(
        protocol_version=1,
        batch_id="batch-1",
        machine="beast-test",
        helper_revision="workspace-helper-v1",
        key_revision="key-v1",
        control_revision=1,
        epoch=1,
        boot_id="boot-1",
        monotonic_deadline_ns=100,
        rows=(row,),
    )
    signature = base64.b64decode(
        sign_envelope(envelope.as_dict(), private_key_path=private_key)
    )
    signature_path = tmp_path / "signature.bin"
    signature_path.write_bytes(signature)
    valid = subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-verify",
            public_key,
            "-signature",
            signature_path,
        ],
        input=canonical_json_bytes(envelope.as_dict()),
        capture_output=True,
        check=False,
    )
    assert valid.returncode == 0
    tampered = envelope.as_dict()
    tampered["epoch"] = 2
    invalid = subprocess.run(
        [
            "openssl",
            "dgst",
            "-sha256",
            "-verify",
            public_key,
            "-signature",
            signature_path,
        ],
        input=canonical_json_bytes(tampered),
        capture_output=True,
        check=False,
    )
    assert invalid.returncode != 0

