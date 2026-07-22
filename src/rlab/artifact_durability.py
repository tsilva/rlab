from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, Mapping
from urllib.parse import urlparse

from rlab.workspace_contract import sha256_json
from rlab.workspace_gc import DurabilityReceipt, record_artifact_durability_receipt


POLICY_PROTOCOL_VERSION = 1
READ_CHUNK_BYTES = 8 * 1024 * 1024


class ArtifactDurabilityError(RuntimeError):
    pass


@dataclass(frozen=True)
class ArtifactDurabilityPolicy:
    endpoint_url: str
    bucket: str
    prefix: str
    policy_scope: str
    verifier_identity: str
    non_expiring_write_once: bool
    runtime_delete_denied: bool
    runtime_overwrite_denied: bool
    runtime_policy_admin_denied: bool
    content_addressed_keys: bool
    version_identity_required: bool
    preflight_receipt_sha256: str
    policy_sha256: str
    document: Mapping[str, Any]

    @classmethod
    def load(cls, path: Path) -> ArtifactDurabilityPolicy:
        try:
            mode = path.stat().st_mode & 0o777
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ArtifactDurabilityError(f"cannot load durability policy {path}: {exc}") from exc
        if mode & 0o022:
            raise ArtifactDurabilityError("durability policy must not be group/world writable")
        if not isinstance(value, dict):
            raise ArtifactDurabilityError("durability policy must be a JSON object")
        if int(value.get("protocol_version") or 0) != POLICY_PROTOCOL_VERSION:
            raise ArtifactDurabilityError("unsupported durability policy protocol")
        prefix = str(value.get("prefix") or "").strip("/")
        document = dict(value)
        digest = sha256_json(document)
        policy = cls(
            endpoint_url=str(value.get("endpoint_url") or ""),
            bucket=str(value.get("bucket") or ""),
            prefix=prefix,
            policy_scope=str(value.get("policy_scope") or ""),
            verifier_identity=str(value.get("verifier_identity") or ""),
            non_expiring_write_once=bool(value.get("non_expiring_write_once")),
            runtime_delete_denied=bool(value.get("runtime_delete_denied")),
            runtime_overwrite_denied=bool(value.get("runtime_overwrite_denied")),
            runtime_policy_admin_denied=bool(value.get("runtime_policy_admin_denied")),
            content_addressed_keys=bool(value.get("content_addressed_keys")),
            version_identity_required=bool(value.get("version_identity_required", True)),
            preflight_receipt_sha256=str(value.get("preflight_receipt_sha256") or ""),
            policy_sha256=digest,
            document=document,
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not self.bucket or not self.policy_scope or not self.verifier_identity:
            raise ArtifactDurabilityError("durability policy lacks bucket, scope, or verifier")
        if not self.prefix:
            raise ArtifactDurabilityError("durability policy must bind a non-root object prefix")
        if not all(
            (
                self.non_expiring_write_once,
                self.runtime_delete_denied,
                self.runtime_overwrite_denied,
                self.runtime_policy_admin_denied,
                self.content_addressed_keys,
            )
        ):
            raise ArtifactDurabilityError("durability policy is not cleanup-safe")
        digest = self.preflight_receipt_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ArtifactDurabilityError("durability policy lacks immutable preflight evidence")


def _s3_client(policy: ArtifactDurabilityPolicy):
    import boto3

    kwargs: dict[str, Any] = {"endpoint_url": policy.endpoint_url or None}
    region = str(os.environ.get("AWS_REGION") or "").strip()
    if region:
        kwargs["region_name"] = region
    return boto3.client("s3", **kwargs)


def _parse_policy_object_uri(
    policy: ArtifactDurabilityPolicy, uri: str, expected_sha256: str
) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or parsed.netloc != policy.bucket:
        raise ArtifactDurabilityError("artifact URI is outside the attested bucket")
    key = parsed.path.lstrip("/")
    prefix = f"{policy.prefix}/"
    if not key.startswith(prefix):
        raise ArtifactDurabilityError("artifact URI is outside the attested prefix")
    if f"/sha256/{expected_sha256}/" not in f"/{key}":
        raise ArtifactDurabilityError("artifact key is not bound to its declared SHA-256")
    return parsed.netloc, key


def _stream_sha256(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = stream.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def verify_remote_object(
    *,
    policy: ArtifactDurabilityPolicy,
    client,
    train_job_id: int,
    ledger_id: int,
    object_kind: str,
    object_uri: str,
    expected_sha256: str,
    workspace_evidence: Mapping[str, Any],
) -> DurabilityReceipt:
    bucket, key = _parse_policy_object_uri(policy, object_uri, expected_sha256)
    response = client.get_object(Bucket=bucket, Key=key)
    body = response.get("Body")
    if body is None or not hasattr(body, "read"):
        raise ArtifactDurabilityError("object store returned no readable body")
    try:
        actual_sha256, size_bytes = _stream_sha256(body)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if actual_sha256 != expected_sha256:
        raise ArtifactDurabilityError(
            f"stored object hash mismatch for {object_kind}: {actual_sha256}"
        )
    version_id = str(response.get("VersionId") or "").strip()
    if policy.version_identity_required and not version_id:
        raise ArtifactDurabilityError("object store did not return an immutable version identity")
    evidence = {
        "protocol_version": POLICY_PROTOCOL_VERSION,
        "train_job_id": int(train_job_id),
        "ledger_id": int(ledger_id),
        "object_kind": object_kind,
        "object_uri": object_uri,
        "object_version": version_id,
        "size_bytes": size_bytes,
        "sha256": actual_sha256,
        "policy_sha256": policy.policy_sha256,
        "policy_scope": policy.policy_scope,
        "preflight_receipt_sha256": policy.preflight_receipt_sha256,
        "workspace_nonoverlap": dict(workspace_evidence),
    }
    return DurabilityReceipt(
        train_job_id=int(train_job_id),
        ledger_id=int(ledger_id),
        object_kind=object_kind,
        object_uri=object_uri,
        object_version=version_id,
        size_bytes=size_bytes,
        sha256=actual_sha256,
        full_read_verified_at=datetime.now(UTC),
        verifier_identity=policy.verifier_identity,
        policy_scope=policy.policy_scope,
        policy_sha256=policy.policy_sha256,
        non_expiring_write_once=True,
        runtime_delete_denied=True,
        runtime_overwrite_denied=True,
        storage_root_nonoverlap_sha256=sha256_json(workspace_evidence),
        receipt_json=evidence,
    )


def verify_due_artifact_receipts(
    conn,
    *,
    policy: ArtifactDurabilityPolicy,
    client=None,
    limit: int = 8,
) -> int:
    policy.validate()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT ledger.train_job_id, ledger.ledger_id,
              ledger.checkpoint_uri, ledger.checkpoint_sha256,
              ledger.metadata_uri, ledger.metadata_sha256,
              ledger.recipe_uri, ledger.recipe_sha256,
              manifest.manifest_sha256, manifest.manifest_json
            FROM artifact_announcement_ledger ledger
            JOIN job_launches launch ON launch.job_id=ledger.train_job_id
              AND launch.job_kind='train'
            JOIN workspace_manifests manifest ON manifest.launch_id=launch.launch_id
              AND manifest.generation=launch.lifecycle_generation
            WHERE ledger.disposition='ready'
              AND EXISTS (
                SELECT 1 FROM workspace_rollout_controls control
                WHERE control.singleton AND control.protocol_mode <> 'dormant'
              )
              AND (
                NOT EXISTS (
                  SELECT 1 FROM artifact_durability_receipts receipt
                  WHERE receipt.train_job_id=ledger.train_job_id
                    AND receipt.ledger_id=ledger.ledger_id
                    AND receipt.object_kind='model'
                )
                OR NOT EXISTS (
                  SELECT 1 FROM artifact_durability_receipts receipt
                  WHERE receipt.train_job_id=ledger.train_job_id
                    AND receipt.ledger_id=ledger.ledger_id
                    AND receipt.object_kind='metadata'
                )
                OR (ledger.recipe_uri IS NOT NULL AND NOT EXISTS (
                  SELECT 1 FROM artifact_durability_receipts receipt
                  WHERE receipt.train_job_id=ledger.train_job_id
                    AND receipt.ledger_id=ledger.ledger_id
                    AND receipt.object_kind='recipe'
                ))
              )
            ORDER BY ledger.verified_at, ledger.train_job_id, ledger.ledger_id
            LIMIT %(limit)s
            """,
            {"limit": max(1, min(int(limit), 64))},
        )
        rows = [dict(row) for row in cur.fetchall()]
    if not rows:
        return 0
    remote = client or _s3_client(policy)
    verified = 0
    for row in rows:
        manifest = dict(row["manifest_json"])
        workspace_evidence = {
            "manifest_sha256": row["manifest_sha256"],
            "workspace_targets": list(manifest.get("deletion_targets") or ()),
            "remote_scheme": "s3",
            "remote_bucket": policy.bucket,
            "remote_prefix": policy.prefix,
        }
        objects = (
            ("model", row["checkpoint_uri"], row["checkpoint_sha256"]),
            ("metadata", row["metadata_uri"], row["metadata_sha256"]),
            ("recipe", row.get("recipe_uri"), row.get("recipe_sha256")),
        )
        for kind, uri, digest in objects:
            if not uri:
                continue
            receipt = verify_remote_object(
                policy=policy,
                client=remote,
                train_job_id=int(row["train_job_id"]),
                ledger_id=int(row["ledger_id"]),
                object_kind=kind,
                object_uri=str(uri),
                expected_sha256=str(digest),
                workspace_evidence=workspace_evidence,
            )
            record_artifact_durability_receipt(conn, receipt)
            verified += 1
    return verified
